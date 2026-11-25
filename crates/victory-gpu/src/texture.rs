//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! In the runtime's memory a change detiles only the subresources whose
//! bytes lie in the pages written, and of those only the ones whose bytes
//! changed (upstream cc5f9612, by Marvis); heap uploads copy only those.
//! shortcut: memory that is not tracked detiles the whole texture.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    /// Per subresource, the bytes of the texture's memory it lies in (from
    /// its start, whole layout blocks); empty where the layout gives none.
    spans: Vec<(usize, usize)>,
    last: Mutex<Last>,
    /// Subresources detiled so far (tests and diagnostics).
    detiled: AtomicU64,
    /// [`crate::resource::now_ms`] and frames presented when it was last
    /// needed: a list copied from it ([`upload`]) or the memory changed
    /// ([`upload_changed`]). Idle ones go ([`let_go_idle_staging`]).
    used: [AtomicU64; 2],
    /// Made again after one was let go: the next refresh detiles whatever
    /// the tracker or hash says (the buffer is new).
    force: std::sync::atomic::AtomicBool,
}

/// A host texture's uploads from title memory ([`HostTexture::staging`]):
/// its staging buffer, or what one let go while idle knew of the memory.
///
/// [`HostTexture::staging`]: crate::resource::HostTexture
#[derive(Default)]
pub(crate) struct Stage {
    live: Option<Arc<Staging>>,
    dormant: Option<Dormant>,
}

/// What a staging buffer let go while idle knew: the memory's tracker,
/// which keeps marking the pages written since, or (memory not tracked)
/// the hash it last detiled. While the memory is unchanged, the host
/// texture holds it and no staging buffer is needed.
struct Dormant {
    tracked: Option<crate::watch::Tracked>,
    hash: Option<blake3::Hash>,
    memory: *const u8,
    size: usize,
}

// SAFETY: the title memory is only read.
unsafe impl Send for Dormant {}

impl Dormant {
    /// Whether the memory is as the host texture last took it.
    fn unchanged(&self) -> bool {
        if let Some(t) = &self.tracked {
            return t.unchanged();
        }
        if self.hash.is_none() || !committed(self.memory, self.size) {
            // Never detiled, or not readable (a refresh would read nothing).
            return self.hash.is_some();
        }
        let started = std::time::Instant::now();
        // SAFETY: committed memory of the texture's layout size.
        let hash = blake3::hash(unsafe { std::slice::from_raw_parts(self.memory, self.size) });
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(self.size as u64, Relaxed);
        self.hash == Some(hash)
    }
}

impl Staging {
    fn touch(&self) {
        self.used[0].store(crate::resource::now_ms(), Relaxed);
        self.used[1].store(crate::queue::presented(), Relaxed);
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        STAGING[0].fetch_sub(1, Relaxed);
        STAGING[1].fetch_sub(self.size, Relaxed);
    }
}

/// Staging buffers alive and their bytes (upload heap: the PC's memory).
static STAGING: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// Staging buffers let go while idle so far, and their bytes.
static STAGING_LET_GO: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Host textures that have a staging buffer (registered when it is made).
static STAGED: Mutex<Vec<std::sync::Weak<crate::resource::HostTexture>>> = Mutex::new(Vec::new());

/// Staging buffers alive and their bytes, and those let go while idle so
/// far and their bytes.
pub(crate) fn staging_counts() -> (u64, u64, u64, u64) {
    (
        STAGING[0].load(Relaxed),
        STAGING[1].load(Relaxed),
        STAGING_LET_GO[0].load(Relaxed),
        STAGING_LET_GO[1].load(Relaxed),
    )
}

/// Lets go of the staging buffers not needed in 5 seconds and 4 frames
/// (no list copied from one, its memory did not change), keeping their
/// host textures. A staging buffer is the size of its host texture, in the
/// upload heap (the PC's memory), and was kept as long as its host
/// texture: with #1962's placed-texture views, which heap slots keep, that
/// was one per texture the title ever placed and a shader read. The host
/// texture keeps what was uploaded, and the memory's tracker (or last
/// hash) stays with it: while the memory is unchanged no buffer is made
/// again; once it changes, or a list copies from it, a new one is made and
/// the whole memory detiled into it.
///
/// Called at present, at most once a second. `VICTORY_STAGING_SWEEP=0`
/// keeps them all, as before.
pub(crate) fn let_go_idle_staging() {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static LAST: AtomicU64 = AtomicU64::new(0);
    let on = *ON.get_or_init(|| std::env::var_os("VICTORY_STAGING_SWEEP").is_none_or(|v| v != "0"));
    let now = crate::resource::now_ms();
    let last = LAST.load(Relaxed);
    if !on || now < last + 1000 || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    let_go_staging(now, crate::queue::presented(), 5_000, 4, |_| true);
}

/// Lets go of the staging buffers of host textures `within` takes that no
/// upload asked for in `ms` milliseconds and `frames` frames at `now` and
/// frame `frame`, and that no list holds (one not yet reset holds those it
/// will refresh at submission): how many went, and their bytes. A list
/// that copies from one keeps its buffer ([`HostList::keep`]) and the
/// buffer is retired besides, so a submission running now still has it.
fn let_go_staging(
    now: u64,
    frame: u64,
    ms: u64,
    frames: u64,
    within: impl Fn(&crate::resource::HostTexture) -> bool,
) -> (u64, u64) {
    // Out of the list while we look (`staging` registers under its
    // texture's lock, which we take below: never both the other way).
    let staged = std::mem::take(&mut *STAGED.lock().unwrap_or_else(|e| e.into_inner()));
    let mut keep = Vec::with_capacity(staged.len());
    let mut gone = Vec::new();
    for weak in staged {
        let Some(copy) = weak.upgrade() else {
            continue;
        };
        if !within(&copy) {
            keep.push(weak);
            continue;
        }
        let mut stage = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
        let stage = &mut *stage;
        let Some(s) = stage.live.as_mut() else {
            continue; // registered again when made again
        };
        let idle = now.saturating_sub(s.used[0].load(Relaxed)) >= ms
            && frame.saturating_sub(s.used[1].load(Relaxed)) >= frames;
        // Clones are made under this lock: one now is no list's.
        let Some(m) = idle.then(|| Arc::get_mut(s)).flatten() else {
            keep.push(weak);
            continue;
        };
        // A detiling still to come (made, never refreshed): the host
        // texture may not hold the memory yet, so the next one starts anew.
        stage.dormant = (!*m.force.get_mut()).then(|| Dormant {
            tracked: m.tracked.take(),
            hash: m.last.get_mut().unwrap_or_else(|e| e.into_inner()).hash,
            memory: m.memory,
            size: m.layout.size as usize,
        });
        gone.extend(stage.live.take());
    }
    STAGED.lock().unwrap_or_else(|e| e.into_inner()).append(&mut keep);
    let went = gone.len() as u64;
    let mut bytes = 0;
    for s in gone {
        bytes += s.size;
        crate::host::retire(s.buffer.clone());
    }
    if went > 0 {
        STAGING_LET_GO[0].fetch_add(went, Relaxed);
        STAGING_LET_GO[1].fetch_add(bytes, Relaxed);
        tracing::debug!(went, kib = bytes >> 10, "idle staging buffers let go");
    }
    (went, bytes)
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not tracked), or of each subresource's bytes (tracked; none
/// where a whole page of its own changed and it was detiled unhashed).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    hashes: Vec<Option<blake3::Hash>>,
}

/// Per subresource of `layout` (`mips` per array slice; a 3D texture's are
/// its mips), the bytes of the texture's memory it lies in: its first and
/// last element's offsets, rounded out to the layout's blocks (every byte
/// detiling it reads lies there); empty if the layout cannot say. From
/// upstream cc5f9612 (Marvis).
fn spans(layout: &victory_formats::texture::Layout, count: u32, mips: u32) -> Vec<(usize, usize)> {
    use victory_formats::texture::Dimension;
    let d = &layout.desc;
    let block = layout.alignment.max(256);
    (0..count)
        .map(|i| {
            let (mip, slice) = (i % mips.max(1), i / mips.max(1));
            let m = layout.mips.get(mip as usize)?;
            let (z0, z1) = if d.dimension == Dimension::Tex3d {
                (0, m.depth.max(1) - 1)
            } else {
                (slice, slice)
            };
            let first = layout.offset(mip, 0, 0, z0).ok()?;
            let last = layout.offset(mip, m.width.max(1) - 1, m.height.max(1) - 1, z1).ok()?
                + u64::from(d.element_bytes);
            let (from, to) = (first / block * block, last.div_ceil(block) * block);
            Some((from as usize, to.min(layout.size) as usize))
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

/// `spans` sorted and joined where they overlap or meet (spans of mips in
/// one tail block, or of a thick 3D texture's mips, overlap): each byte
/// once. Empty spans go.
fn merged(spans: impl Iterator<Item = (usize, usize)>) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = spans.filter(|&(a, b)| a < b).collect();
    spans.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (a, b) in spans {
        match out.last_mut() {
            Some((_, end)) if a <= *end => *end = (*end).max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// Of `touched` subresources (their `span`s of the memory at `start`, read
/// into `read`), those to detile again, given the changed byte `ranges`
/// (absolute): one in which a whole changed page lies that is its own
/// (inside its span) changed; one only sharing changed pages with other
/// data the title writes (small textures, a page's edges) changed if its
/// own bytes did (`hashes`, kept per subresource). Upstream cc5f9612.
fn changed_subresources(
    touched: &[usize],
    span: impl Fn(usize) -> (usize, usize),
    start: usize,
    ranges: &[(usize, usize)],
    read: &[u8],
    hashes: &mut [Option<blake3::Hash>],
) -> Vec<u32> {
    let page = crate::watch::PAGE;
    touched
        .iter()
        .copied()
        .filter(|&i| {
            let (a, b) = span(i);
            let (a0, b0) = (start + a, start + b);
            if ranges
                .iter()
                .any(|&(f, e)| f.max(a0).next_multiple_of(page) + page <= e.min(b0))
            {
                hashes[i] = None;
                return true;
            }
            let hash = blake3::hash(&read[a..b]);
            hashes[i].replace(hash) != Some(hash)
        })
        .map(|i| i as u32)
        .collect()
}

/// Shared scratch buffer for reading tracked title-texture memory before
/// detiling, so each `Staging` does not retain a full-size `Vec<u8>`.
static READ_SCRATCH: Mutex<Vec<u8>> = Mutex::new(Vec::new());

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve writes `resource`: its host texture's contents
/// are the host's, not title memory's, which stays as it was.
pub(crate) fn host_written(resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(b), Some(host)) = (
        crate::resource::host_resource(from),
        crate::resource::host_resource(to),
        host(),
    ) else {
        return;
    };
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    crate::texture::host_written(to);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..n {
            l.CopyTextureRegion(&sub(b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) || !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture, true) else {
        return;
    };
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset. Every subresource: the
    // refresh comes at submission, after this.
    let all: Vec<u32> = (0..staging.footprints.len() as u32).collect();
    record(list, resource, texture, &staging, &all);
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

/// For textures shaders reach through the heap itself, at submission
/// (`crate::bindless::uploads`): brings the staging buffer up to date and
/// records its copy into `list` only if the memory changed since (or was
/// never uploaded). True if it recorded; the texture is then in the copy
/// destination state.
pub(crate) fn upload_changed(list: &HostList, resource: This, texture: &ID3D12Resource) -> bool {
    if is_host_written(resource) {
        return false;
    }
    // One let go while idle: none is made while the memory is unchanged.
    if let Some(copy) = crate::resource::host_copy(resource) {
        let stage = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
        if stage.live.is_none() && stage.dormant.as_ref().is_some_and(Dormant::unchanged) {
            return false;
        }
    }
    let Some(staging) = staging(resource, texture, false) else {
        return false;
    };
    // Only the subresources detiled again: the host texture holds the
    // others as the staging buffer does.
    let Some(fresh) = staging.refresh() else {
        return false;
    };
    staging.touch();
    record(list, resource, texture, &staging, &fresh);
    list.keep(&staging.buffer);
    true
}

/// Subresources detiled so far into `resource`'s live staging buffer
/// (tests).
#[cfg(test)]
pub(crate) fn detiled(resource: This) -> u64 {
    crate::resource::host_copy(resource).map_or(0, |c| {
        let stage = c.staging.lock().unwrap_or_else(|e| e.into_inner());
        stage.live.as_ref().map_or(0, |s| s.detiled.load(Relaxed))
    })
}

/// Records the copy of `subresources` of `staging` into `texture`, moving
/// it to the copy destination state first.
fn record(
    list: &HostList,
    resource: This,
    texture: &ID3D12Resource,
    staging: &Staging,
    subresources: &[u32],
) {
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    for &i in subresources {
        let i = i as usize;
        let Some(footprint) = staging.footprints.get(i) else {
            continue;
        };
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all): the subresources it
    /// detiled (none: nothing changed).
    pub(crate) fn refresh(&self) -> Option<Vec<u32>> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, fresh) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        if let Some(fresh) = &fresh {
            self.detiled.fetch_add(fresh.len() as u64, Relaxed);
        }
        fresh
    }

    /// Every subresource.
    fn all(&self) -> Vec<u32> {
        (0..self.footprints.len() as u32).collect()
    }

    /// Returns the bytes of title memory read, and the subresources it
    /// detiled.
    fn refresh_now(&self) -> (usize, Option<Vec<u32>>) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        let forced = self.force.swap(false, Relaxed);
        if let Some(t) = &self.tracked {
            // One load, no lock: of the thousands of textures the heaps
            // view, nearly all are unchanged at nearly every submission
            // (about 25 a frame).
            if !forced && t.unchanged() {
                return (0, None);
            }
            let ranges = t.take(start, start + size);
            if ranges.is_empty() && !forced {
                return (0, None);
            }
            // Only the subresources in the pages that changed: streamed
            // textures change mip by mip, and each arriving mip read and
            // detiled every other one again (about 190 MB a frame in the
            // world, layer 4 run L4Fa3). A new buffer (one let go, made
            // again) takes them all.
            let n = self.footprints.len();
            let span = |i: usize| self.spans.get(i).copied().unwrap_or((0, size));
            let touched: Vec<usize> = (0..n)
                .filter(|&i| {
                    let (a, b) = span(i);
                    forced || ranges.iter().any(|&(f, e)| f - start < b && a < e - start)
                })
                .collect();
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page. Only the spans
            // touched: detiling a subresource reads only its span.
            let mut scratch = READ_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            if scratch.len() < size {
                scratch.resize(size, 0);
            }
            let tiled = &mut scratch[..size];
            let mut read = 0;
            for (a, b) in merged(touched.iter().map(|&i| span(i))) {
                crate::host::copy_mapped(
                    tiled[a..].as_mut_ptr(),
                    self.memory.wrapping_add(a),
                    b - a,
                    false,
                );
                read += b - a;
            }
            last.hashes.resize(n, None);
            let fresh = if forced {
                last.hashes.iter_mut().for_each(|h| *h = None);
                self.all()
            } else {
                changed_subresources(&touched, span, start, &ranges, tiled, &mut last.hashes)
            };
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, read as u64);
            if fresh.is_empty() {
                return (read, None);
            }
            self.detile(tiled, &fresh);
            return (read, Some(fresh));
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, None);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) && !forced {
            return (size, None);
        }
        last.hash = Some(hash);
        let all = self.all();
        self.detile(tiled, &all);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, Some(all))
    }

    /// Detiles `subresources` of `tiled` (the texture's memory, of which
    /// only their spans need be current) into the staging buffer.
    fn detile(&self, tiled: &[u8], subresources: &[u32]) {
        #[cfg(feature = "diagnostics")]
        {
            let span = |i: u32| self.spans.get(i as usize).copied().unwrap_or((0, tiled.len()));
            let zero = subresources.iter().all(|&i| {
                let (a, b) = span(i);
                tiled[a.min(b)..b].iter().all(|&byte| byte == 0)
            });
            if zero {
                tracing::debug!(
                    memory = ?self.memory,
                    desc = ?self.layout.desc,
                    "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
                );
            }
        }
        for &i in subresources {
            let (Some(footprint), Some(&rows)) =
                (self.footprints.get(i as usize), self.rows.get(i as usize))
            else {
                continue;
            };
            let (mip, slice) = (i % self.mips, i / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
fn staging(resource: This, texture: &ID3D12Resource, used: bool) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.live.as_ref() {
        if used {
            s.touch();
        }
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    // What one let go knew: the tracker has marked every page written since.
    let (tracked, hash, force) = match held.dormant.take() {
        Some(d) => (d.tracked, d.hash, true),
        None => (crate::watch::track(memory as usize, layout_size, true), None, false),
    };
    let s = Arc::new(Staging {
        buffer,
        size: total,
        data,
        footprints,
        rows,
        spans: spans(&layout, count, u32::from(d.MipLevels)),
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked,
        last: Mutex::new(Last { hash, hashes: Vec::new() }),
        detiled: AtomicU64::new(0),
        used: [AtomicU64::new(0), AtomicU64::new(0)],
        force: force.into(),
    });
    s.touch();
    STAGING[0].fetch_add(1, Relaxed);
    STAGING[1].fetch_add(total, Relaxed);
    held.live = Some(s.clone());
    drop(held);
    STAGED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Arc::downgrade(&copy));
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Detiling a subresource reads only its span: from memory whose bytes
    /// outside the span are garbage (the shared read scratch holds other
    /// textures there) it gives what it gives from the whole memory, for
    /// linear and tiled layouts, mips, arrays, block compression, 3D.
    #[test]
    fn a_subresource_detiles_from_its_span_alone() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let desc = |dimension, width, height, depth, mips, element_bytes, block, swizzle| Desc {
            dimension,
            width,
            height,
            depth,
            mips,
            element_bytes,
            block,
            samples: 1,
            swizzle,
        };
        let cases = [
            desc(Dimension::Tex2d, 300, 200, 3, 4, 4, 1, Swizzle::LINEAR),
            desc(Dimension::Tex2d, 512, 512, 1, 10, 4, 1, Swizzle::SW_64KB_S),
            desc(Dimension::Tex2d, 1024, 512, 1, 11, 8, 1, Swizzle::SW_64KB_D),
            desc(Dimension::Tex2d, 256, 256, 2, 7, 16, 4, Swizzle::SW_4KB_S),
            desc(Dimension::Tex3d, 64, 64, 16, 4, 1, 1, Swizzle::SW_64KB_S),
            desc(Dimension::Tex3d, 40, 24, 6, 3, 4, 1, Swizzle::LINEAR),
        ];
        let mut checked = 0;
        let mut packed = 0;
        for d in cases {
            let layout = Layout::new(&d).expect("a valid layout");
            if !layout.addressable() {
                eprintln!("{:?}: not addressable here, skipped", d.swizzle);
                continue;
            }
            let three_d = d.dimension == Dimension::Tex3d;
            let count = if three_d { d.mips } else { d.mips * d.depth };
            let spans = spans(&layout, count, d.mips);
            assert_eq!(spans.len(), count as usize, "{d:?}: a span per subresource");
            // Packed mips share one block: a write there touches each of them.
            let tail = layout.first_tail_mip;
            if tail + 1 < d.mips {
                for m in tail..d.mips {
                    assert_eq!(spans[m as usize], spans[tail as usize], "{d:?}: packed mip {m} shares the tail block");
                }
                packed += 1;
            }
            let size = layout.size as usize;
            let full: Vec<u8> = (0..size).map(|i| (i * 7 + 13 + i / 251) as u8).collect();
            for (i, &(a, b)) in spans.iter().enumerate() {
                assert!(a < b && b <= size, "{d:?} sub {i}: span {a:#x}..{b:#x} of {size:#x}");
                let (mip, slice) = (i as u32 % d.mips, i as u32 / d.mips);
                let m = &layout.mips[mip as usize];
                let row = (m.width * d.element_bytes) as usize;
                let rows = m.height as usize;
                let depth = if three_d { m.depth as usize } else { 1 };
                let detile = |tiled: &[u8]| {
                    let mut linear = vec![0u8; row * rows * depth];
                    layout.detile(tiled, mip, slice, &mut linear, row, row * rows).expect("detiled");
                    linear
                };
                let mut partial = vec![0xa5u8; size];
                partial[a..b].copy_from_slice(&full[a..b]);
                assert!(detile(&partial) == detile(&full), "{d:?} sub {i} (mip {mip}, slice {slice}) reads outside {a:#x}..{b:#x}");
            }
            checked += 1;
        }
        assert!(checked >= 4, "only {checked} layouts checked");
        assert!(packed >= 1, "no layout with packed mips checked");
    }

    #[test]
    fn touched_spans_are_read_once() {
        let spans = [(0x300, 0x400), (0, 0x100), (0x80, 0x200), (0x200, 0x280), (5, 5), (0x380, 0x390)];
        assert_eq!(merged(spans.into_iter()), [(0, 0x280), (0x300, 0x400)]);
        assert!(merged(std::iter::empty()).is_empty());
    }

    /// Of the subresources a change touches, one in which a whole changed
    /// page of its own lies is detiled unhashed; one only sharing a changed
    /// page with other data is detiled only if its own bytes changed.
    #[test]
    fn shared_pages_detile_a_subresource_only_if_its_bytes_changed() {
        let page = crate::watch::PAGE;
        let start = 0x4000_0000usize;
        // Sub 0 holds pages 0-2; sub 1 the first 4 KiB of page 3, whose
        // rest is other data.
        let spans = [(0, 3 * page), (3 * page, 3 * page + 4096)];
        let span = |i: usize| spans[i];
        let mut read = vec![1u8; 4 * page];
        let mut hashes = vec![None; 2];
        let page_at = |p: usize| (start + p * page, start + (p + 1) * page);
        // Page 1 whole: sub 0, unhashed.
        let got = changed_subresources(&[0], span, start, &[page_at(1)], &read, &mut hashes);
        assert_eq!(got, [0]);
        assert_eq!(hashes[0], None);
        // Page 3 (cut to the texture: its first 4 KiB): sub 1, hashed; at
        // first it was never hashed, so it changed.
        let third = [(start + 3 * page, start + 3 * page + 4096)];
        assert_eq!(changed_subresources(&[1], span, start, &third, &read, &mut hashes), [1]);
        assert!(hashes[1].is_some());
        // The other data in the page changed, sub 1's bytes did not.
        assert!(changed_subresources(&[1], span, start, &third, &read, &mut hashes).is_empty());
        // Its bytes changed.
        read[3 * page + 100] = 9;
        assert_eq!(changed_subresources(&[1], span, start, &third, &read, &mut hashes), [1]);
        assert!(changed_subresources(&[1], span, start, &third, &read, &mut hashes).is_empty());
        // A changed page partly in sub 0 (a range starting mid-page, ending
        // past a whole page): the whole page in its span decides.
        let mid = [(start + page / 2, start + 2 * page)];
        assert_eq!(changed_subresources(&[0], span, start, &mid, &read, &mut hashes), [0]);
    }

    /// A small texture in a page with other data: a write to the other
    /// data detiles nothing, a write to the texture detiles it.
    #[test]
    fn a_neighbours_write_in_the_page_detiles_nothing() {
        let p = crate::watch::tests::pool(1);
        // 16x16 RGBA8, linear: 16 rows of 256 bytes, 4 KiB of the page.
        let words: [u32; 14] = [3, 0, 0, 0, 16, 0, 16, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let Some(r) = crate::resource::texture_at(p as u64, &words) else {
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        let copy = crate::resource::host_copy(r).expect("its host copy");
        let texture = copy.texture.clone();
        let (_, desc) = crate::resource::texture_memory(r).expect("its memory");
        let size = crate::resource::texture_layout(&desc).unwrap().planes[0].layout.size as usize;
        assert!(size <= crate::watch::PAGE / 2, "{size} bytes: the page holds other data");
        let s = staging(r, &texture, false).expect("a staging buffer");
        assert_eq!(s.refresh(), Some(vec![0]), "the first refresh detiles");
        assert_eq!(s.refresh(), None, "unchanged");
        // SAFETY: committed pages of ours, past the texture's bytes.
        unsafe { ((p + crate::watch::PAGE / 2 + 8) as *mut u8).write_volatile(7) };
        assert_eq!(s.refresh(), None, "another's bytes changed, not the texture's");
        // SAFETY: the texture's first texel.
        unsafe { (p as *mut u8).write_volatile(0x80) };
        assert_eq!(s.refresh(), Some(vec![0]), "its own bytes changed");
        assert_eq!(s.refresh(), None);
        // The pool stays: the texture and its tracker outlive the test.
        drop((s, copy));
    }

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    /// A staging buffer not needed for long enough (no list copies from
    /// it, its memory unchanged), and that no list holds, goes; its host
    /// texture stays. While the memory stays unchanged, heap uploads make no
    /// new one; once the title writes it, one is made again and the whole
    /// memory detiled into it.
    #[test]
    fn idle_staging_goes_and_its_host_texture_stays() {
        let p = crate::watch::tests::pool(4);
        let words = crate::resource::texture_words(256, 256, 28, 1);
        let Some(r) = crate::resource::texture_at(p as u64, &words) else {
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        let host = host().expect("a host GPU");
        let copy = crate::resource::host_copy(r).expect("its host copy");
        let texture = copy.texture.clone();
        // Only this test's texture (tests run side by side).
        let mine = |h: &crate::resource::HostTexture| std::ptr::eq(h, &raw const *copy);
        let stage = |f: fn(&Stage) -> bool| f(&copy.staging.lock().unwrap_or_else(|e| e.into_inner()));
        let list = host.new_list().expect("a list");
        upload(&list, r, &texture);
        assert!(stage(|s| s.live.is_some()), "an upload makes it");
        let (now, frame) = (crate::resource::now_ms(), crate::queue::presented());
        let later = |ms: u64, frames: u64| let_go_staging(now + ms, frame + frames, 5_000, 4, mine);
        assert_eq!(later(10_000, 10).0, 0, "a list not yet reset holds it");
        // The list's refresh at submission (it is not submitted here).
        assert!(staging(r, &texture, false).unwrap().refresh().is_some(), "the first refresh detiles");
        list.close();
        list.reset();
        assert_eq!(later(0, 0).0, 0, "just used");
        assert_eq!(later(10_000, 3).0, 0, "too few frames since");
        assert_eq!(later(4_000, 10).0, 0, "too short a time since");
        let (went, bytes) = later(10_000, 10);
        assert_eq!(went, 1);
        assert!(bytes >= 256 * 256 * 4, "{bytes} bytes");
        assert!(stage(|s| s.live.is_none() && s.dormant.is_some()), "the buffer went, the tracker stays");
        assert!(
            crate::resource::host_resource(r).is_some_and(|t| t == &texture),
            "the host texture stays"
        );
        assert_eq!(later(20_000, 20).0, 0, "nothing more to let go");
        let heap = host.new_list().expect("a list");
        assert!(!upload_changed(&heap, r, &texture), "unchanged memory: nothing to upload");
        assert!(stage(|s| s.live.is_none()), "and no buffer made for it");
        // SAFETY: committed pages of ours: the title writes its texture.
        unsafe { ((p + 100) as *mut u8).write_volatile(7) };
        assert!(upload_changed(&heap, r, &texture), "changed memory is uploaded");
        assert!(stage(|s| s.live.is_some() && s.dormant.is_none()), "into a new buffer");
        heap.close();
        heap.reset();
        // The whole memory went into it, not only the page written.
        let again = staging(r, &texture, false).unwrap();
        assert!(again.tracked.as_ref().is_some_and(|t| t.unchanged()), "everything taken");
        drop(again);
        let soon = (crate::resource::now_ms(), crate::queue::presented());
        assert_eq!(let_go_staging(soon.0 + 10_000, soon.1 + 10, 5_000, 4, mine).0, 1, "registered again, let go again");
        // One made but never refreshed leaves no tracker: the next starts anew.
        let pending = host.new_list().expect("a list");
        upload(&pending, r, &texture);
        pending.close();
        pending.reset();
        let soon = (crate::resource::now_ms(), crate::queue::presented());
        assert_eq!(let_go_staging(soon.0 + 10_000, soon.1 + 10, 5_000, 4, mine).0, 1);
        assert!(stage(|s| s.live.is_none() && s.dormant.is_none()), "no tracker kept");
        drop(copy);
        crate::watch::tests::free(p);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }
}
