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
//! shortcut: a change detiles the whole texture, not the pages written.
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
    last: Mutex<Last>,
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
/// (memory not tracked).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
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
    // refresh below holds until the list is reset.
    record(list, resource, texture, &staging);
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
    if !staging.refresh() {
        return false;
    }
    staging.touch();
    record(list, resource, texture, &staging);
    list.keep(&staging.buffer);
    true
}

/// Records the copy of every subresource of `staging` into `texture`,
/// moving it to the copy destination state first.
fn record(list: &HostList, resource: This, texture: &ID3D12Resource, staging: &Staging) {
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    for (i, footprint) in staging.footprints.iter().enumerate() {
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
    /// since the last time (and is mapped at all): whether it did.
    pub(crate) fn refresh(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, changed) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        changed
    }

    /// Returns the bytes of title memory read, and whether it detiled.
    fn refresh_now(&self) -> (usize, bool) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        let forced = self.force.swap(false, Relaxed);
        if let Some(t) = &self.tracked {
            if t.take(start, start + size).is_empty() && !forced {
                return (0, false);
            }
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            let mut scratch = READ_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            if scratch.len() < size {
                scratch.resize(size, 0);
            }
            let tiled = &mut scratch[..size];
            crate::host::copy_mapped(tiled.as_mut_ptr(), self.memory, size, false);
            self.detile(tiled);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, true);
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, false);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) && !forced {
            return (size, false);
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
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
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked,
        last: Mutex::new(Last { hash }),
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
        assert!(staging(r, &texture, false).unwrap().refresh(), "the first refresh detiles");
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
