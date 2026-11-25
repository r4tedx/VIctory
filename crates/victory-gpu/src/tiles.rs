//! Levels on demand (`VICTORY_LEVELS_ON_DEMAND=1`, off by default): host
//! textures of title textures that take the host GPU's memory level by
//! level, as the title's memory holds the levels.
//!
//! A streaming title places a texture's whole chain of levels in its
//! address space and maps only some of them: the smallest first, at the
//! start of the texture's memory (the console layout puts the mip tail
//! first, then the levels from the smallest up, level 0 last). The upload
//! (`crate::texture`) reads the unmapped ones as zeros, but the committed
//! host texture holds every level from the start. Grand Theft Auto VI keeps
//! gigabytes of host texture memory that way for levels it never streamed
//! in.
//!
//! Here such a host texture is a reserved one (`CreateReservedResource`):
//! every level exists, so level numbers, footprints and views are as
//! before, but none has memory. A level is given 64 KiB tiles of a heap of
//! ours (`UpdateTileMappings`, on the host queue) when its title memory is
//! mapped, at the refresh of the texture's staging buffer at submission
//! (`texture.rs` `Staging::refresh`): the refresh runs before the lists of
//! the submission are executed, and every list that samples the texture
//! copies every level from the staging buffer, so a level is written whole
//! before a shader reads it. A level without tiles reads as zeros (tiled
//! resources tier 2), as the zeros the upload of unmapped memory left
//! before. Before the host GPU itself writes the texture (a copy, a
//! resolve), every level is given tiles, after an upload of what the title
//! holds is recorded (`texture.rs` `host_written`).
//!
//! shortcut: tiles are not taken back while the host texture lives; a level
//! the title unmaps keeps them (the upload writes it with zeros). They go
//! back with the host texture.
//! shortcut: views are not clamped to the first level with memory: a
//! level the title does not hold reads as zeros, as before.
//!
//! Only for placed title textures the title fills (no render target, depth
//! or unordered access), two-dimensional, one layer, one sample, more than
//! one level, with a console layout the upload reads, and of [`floor`]
//! bytes or more.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

/// The host GPU's tile: 64 KiB.
const TILE: u64 = 64 << 10;

/// Tiles in one heap of the pool (64 MiB). The largest level of a 16384
/// texture of one byte a texel takes 4,096: such levels are not given
/// ([`Tiled::make`] refuses the texture).
const HEAP_TILES: u32 = 1024;

/// Textures alive; the bytes all their levels would take; the bytes of the
/// tiles given to them; the bytes of the pool's heaps; textures made in
/// all; levels with tiles now (the packed levels of a texture count as
/// one); `UpdateTileMappings` calls; their microseconds; heaps given back;
/// refreshes that left a mapped level without tiles.
static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// Whether title textures are made this way: the switch, and a host GPU
/// that reads zeros from levels without memory (tiled resources tier 2).
pub(crate) fn wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os("VICTORY_LEVELS_ON_DEMAND").is_none_or(|v| v != "1") {
            return false;
        }
        let Some(host) = crate::host::host() else {
            return false;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: a plain query into a structure of its own size.
        let asked = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        let tier = options.TiledResourcesTier.0;
        let on = asked.is_ok() && tier >= D3D12_TILED_RESOURCES_TIER_2.0;
        tracing::info!(
            tier,
            on,
            floor_kb = floor() >> 10,
            "host textures of streamed title textures take memory level by level (VICTORY_LEVELS_ON_DEMAND=1)"
        );
        on
    })
}

fn parse_floor(value: Option<&str>) -> u64 {
    value
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(1 << 20, |kb| kb.max(64).saturating_mul(1024))
}

/// The host texture bytes from which a texture is made this way: a
/// megabyte, or `VICTORY_LEVELS_MIN_KB` (64 at least). Smaller ones have a
/// few tiles in all and the mapping calls would buy little.
pub(crate) fn floor() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| parse_floor(std::env::var("VICTORY_LEVELS_MIN_KB").ok().as_deref()))
}

/// Whether a host texture of description `d` may be made this way (its
/// shape; the caller checks the title's side).
pub(crate) fn shape_fits(d: &D3D12_RESOURCE_DESC) -> bool {
    const USES: i32 = 0x7; // render target, depth, unordered access
    d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.DepthOrArraySize == 1
        && d.MipLevels != 1
        && d.SampleDesc.Count <= 1
        && d.Flags.0 & USES == 0
}

/// Takes `count` tiles from the free runs `free` (first tile, tiles; in
/// order, no two touching), lowest first: the runs taken, or `None` (and
/// `free` untouched) if there are not that many.
fn cut(free: &mut Vec<(u32, u32)>, count: u32) -> Option<Vec<(u32, u32)>> {
    if count == 0 || free.iter().map(|r| r.1).sum::<u32>() < count {
        return None;
    }
    let mut runs = Vec::new();
    let mut left = count;
    while left > 0 {
        let (start, tiles) = free[0];
        let part = tiles.min(left);
        runs.push((start, part));
        if part == tiles {
            free.remove(0);
        } else {
            free[0] = (start + part, tiles - part);
        }
        left -= part;
    }
    Some(runs)
}

/// Gives `runs` back to the free runs `free`, joining runs that touch.
fn join(free: &mut Vec<(u32, u32)>, runs: &[(u32, u32)]) {
    free.extend_from_slice(runs);
    free.sort_unstable();
    let mut joined: Vec<(u32, u32)> = Vec::with_capacity(free.len());
    for &(from, tiles) in free.iter() {
        match joined.last_mut() {
            Some(last) if last.0 + last.1 == from => last.1 += tiles,
            _ => joined.push((from, tiles)),
        }
    }
    *free = joined;
}

/// One heap of the pool and its free runs of tiles.
struct Heap {
    heap: ID3D12Heap,
    free: Vec<(u32, u32)>,
}

impl Heap {
    fn free_tiles(&self) -> u32 {
        self.free.iter().map(|run| run.1).sum()
    }
}

/// The pool's heaps by place; a place is empty after its heap went back to
/// the host GPU (textures name their heaps by place).
static POOL: Mutex<Vec<Option<Heap>>> = Mutex::new(Vec::new());

/// Tiles taken: the heap, its place in the pool and the runs.
type Taken = (ID3D12Heap, usize, Vec<(u32, u32)>);

/// `count` tiles of one heap: the heap, its place in the pool and the runs
/// (first tile, tiles). A new heap is made when no heap has that many free.
fn take(device: &ID3D12Device, count: u32) -> Option<Taken> {
    if count == 0 || count > HEAP_TILES {
        return None;
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let index = match pool.iter().position(|h| h.as_ref().is_some_and(|h| h.free_tiles() >= count)) {
        Some(index) => index,
        None => {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: u64::from(HEAP_TILES) * TILE,
                Properties: D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            };
            let mut made: Option<ID3D12Heap> = None;
            // SAFETY: plain creation; the out pointer is ours.
            if let Err(e) = unsafe { device.CreateHeap(&raw const desc, &mut made) } {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(error = %e, "no heap for texture levels on demand: those levels stay empty for now");
                });
                return None;
            }
            let fresh = Heap {
                heap: made?,
                free: vec![(0, HEAP_TILES)],
            };
            STATS[3].fetch_add(u64::from(HEAP_TILES) * TILE, Ordering::Relaxed);
            match pool.iter().position(Option::is_none) {
                Some(index) => {
                    pool[index] = Some(fresh);
                    index
                }
                None => {
                    pool.push(Some(fresh));
                    pool.len() - 1
                }
            }
        }
    };
    let h = pool[index].as_mut()?;
    let runs = cut(&mut h.free, count)?;
    Some((h.heap.clone(), index, runs))
}

/// Gives the runs `runs` of the heap at `index` back. A heap with every
/// tile free goes back to the host GPU, unless it is the pool's only one.
/// (Called when the host texture that held the tiles is dropped: no list
/// that names it is left.)
fn give_back(index: usize, runs: &[(u32, u32)]) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(h)) = pool.get_mut(index) else {
        return;
    };
    join(&mut h.free, runs);
    if h.free_tiles() == HEAP_TILES && pool.iter().flatten().count() > 1 {
        pool[index] = None;
        STATS[3].fetch_sub(u64::from(HEAP_TILES) * TILE, Ordering::Relaxed);
        STATS[8].fetch_add(1, Ordering::Relaxed);
    }
}

/// What a texture has been given so far.
struct Given {
    /// Per level with tiles of its own: whether it has them.
    levels: Vec<bool>,
    /// Whether the small levels the host GPU packs together have theirs.
    packed: bool,
    /// The runs taken: the heap's place, first tile, tiles.
    runs: Vec<(usize, u32, u32)>,
}

/// A host texture made without memory, and the tiles its levels have.
pub(crate) struct Tiled {
    /// Tiles of each level with tiles of its own (level 0 first).
    standard: Vec<u32>,
    /// Tiles of the levels the host GPU packs together (0: none), the
    /// levels from `standard.len()` on.
    packed: u32,
    /// Levels in all.
    levels: u32,
    /// The bytes all its tiles would take.
    full: u64,
    given: Mutex<Given>,
    /// Every level has tiles: [`Tiled::give`] has nothing left to do.
    whole: AtomicBool,
}

// SAFETY: the heaps named are the device's, used only in queue calls any
// thread may make; the rest is plain data behind a lock.
unsafe impl Send for Heap {}

impl Tiled {
    /// For `texture`, just made reserved: how the host GPU cuts its levels
    /// into tiles. `None` if it will not say, or cuts it in a way this does
    /// not handle (the caller then makes the texture committed).
    pub(crate) fn make(device: &ID3D12Device, texture: &ID3D12Resource) -> Option<Tiled> {
        // SAFETY: a live resource.
        let levels = u32::from(unsafe { texture.GetDesc() }.MipLevels);
        if !(2..=16).contains(&levels) {
            return None;
        }
        let mut total = 0u32;
        let mut packed = D3D12_PACKED_MIP_INFO::default();
        let mut shape = D3D12_TILE_SHAPE::default();
        let mut count = levels;
        let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); levels as usize];
        // SAFETY: outputs of ours, `count` tilings of them.
        unsafe {
            device.GetResourceTiling(
                texture,
                Some(&raw mut total),
                Some(&raw mut packed),
                Some(&raw mut shape),
                Some(&raw mut count),
                0,
                tilings.as_mut_ptr(),
            );
        }
        let standard_levels = u32::from(packed.NumStandardMips);
        if total == 0 || standard_levels + u32::from(packed.NumPackedMips) != levels || standard_levels > count {
            return None;
        }
        let mut standard = Vec::with_capacity(standard_levels as usize);
        for tiling in &tilings[..standard_levels as usize] {
            let tiles = tiling
                .WidthInTiles
                .checked_mul(u32::from(tiling.HeightInTiles))
                .and_then(|n| n.checked_mul(u32::from(tiling.DepthInTiles)))
                .unwrap_or(0);
            if tiles == 0 || tiles > HEAP_TILES || tiling.StartTileIndexInOverallResource == D3D12_PACKED_TILE {
                return None;
            }
            standard.push(tiles);
        }
        let packed_tiles = if packed.NumPackedMips > 0 { packed.NumTilesForPackedMips } else { 0 };
        if (packed.NumPackedMips > 0 && packed_tiles == 0) || packed_tiles > HEAP_TILES {
            return None;
        }
        let full = u64::from(total) * TILE;
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(full, Ordering::Relaxed);
        STATS[4].fetch_add(1, Ordering::Relaxed);
        Some(Tiled {
            given: Mutex::new(Given {
                levels: vec![false; standard.len()],
                packed: false,
                runs: Vec::new(),
            }),
            standard,
            packed: packed_tiles,
            levels,
            full,
            whole: AtomicBool::new(false),
        })
    }

    /// A bit for every level of the texture (level 0 the lowest).
    pub(crate) fn every_level(&self) -> u32 {
        if self.levels >= 32 { u32::MAX } else { (1 << self.levels) - 1 }
    }

    /// Whether every level has tiles (then [`Tiled::give`] does nothing).
    pub(crate) fn whole(&self) -> bool {
        self.whole.load(Ordering::Relaxed)
    }

    /// Gives tiles to the levels `levels` names (a bit a level) that have
    /// none; the levels the host GPU packs together get theirs when any of
    /// them is named. To be called only where the levels given are written
    /// whole before a shader reads them (tiles hold what their last owner
    /// left): the mapping goes on the host queue now, ahead of the lists
    /// submitted after. Returns whether every level named has tiles (not
    /// if no heap could be had).
    pub(crate) fn give(&self, texture: &ID3D12Resource, levels: u32) -> bool {
        match crate::host::host() {
            Some(host) => self.give_on(host.queue(), &host.device, texture, levels),
            None => false,
        }
    }

    /// [`Tiled::give`] on a queue and a device.
    pub(crate) fn give_on(
        &self,
        queue: &ID3D12CommandQueue,
        device: &ID3D12Device,
        texture: &ID3D12Resource,
        levels: u32,
    ) -> bool {
        let levels = levels & self.every_level();
        let named = |level: u32| levels & (1 << level) != 0;
        let first_packed = self.standard.len() as u32;
        let mut given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        // (subresource, tiles, the level if it has tiles of its own).
        let mut wanted: Vec<(u32, u32, Option<usize>)> = Vec::new();
        for level in 0..first_packed {
            if !given.levels[level as usize] && named(level) {
                wanted.push((level, self.standard[level as usize], Some(level as usize)));
            }
        }
        if self.packed > 0 && !given.packed && (first_packed..self.levels).any(named) {
            wanted.push((first_packed, self.packed, None));
        }
        let mut all = true;
        for (subresource, tiles, level) in wanted {
            let Some((heap, index, runs)) = take(device, tiles) else {
                all = false;
                continue;
            };
            let at = D3D12_TILED_RESOURCE_COORDINATE {
                X: 0,
                Y: 0,
                Z: 0,
                Subresource: subresource,
            };
            let size = D3D12_TILE_REGION_SIZE {
                NumTiles: tiles,
                ..Default::default()
            };
            // One region, the subresource's tiles in order, on as many
            // ranges of the heap as the runs taken.
            let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; runs.len()];
            let starts: Vec<u32> = runs.iter().map(|run| run.0).collect();
            let counts: Vec<u32> = runs.iter().map(|run| run.1).collect();
            let began = std::time::Instant::now();
            // SAFETY: a live reserved texture and a live heap of ours; one
            // region of `tiles` tiles from the subresource's first tile;
            // `runs.len()` ranges whose counts add up to `tiles`, in three
            // arrays of that length that live through the call.
            unsafe {
                queue.UpdateTileMappings(
                    texture,
                    1,
                    Some(&raw const at),
                    Some(&raw const size),
                    &heap,
                    runs.len() as u32,
                    Some(flags.as_ptr()),
                    Some(starts.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
            STATS[6].fetch_add(1, Ordering::Relaxed);
            STATS[7].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            match level {
                Some(level) => given.levels[level] = true,
                None => given.packed = true,
            }
            for &(start, count) in &runs {
                given.runs.push((index, start, count));
            }
            STATS[2].fetch_add(u64::from(tiles) * TILE, Ordering::Relaxed);
            STATS[5].fetch_add(1, Ordering::Relaxed);
        }
        if !all {
            STATS[9].fetch_add(1, Ordering::Relaxed);
        }
        if given.levels.iter().all(|&has| has) && (self.packed == 0 || given.packed) {
            self.whole.store(true, Ordering::Relaxed);
        }
        all
    }

    /// Of `levels`, the ones that have no tiles now.
    #[cfg(test)]
    fn without(&self, levels: u32) -> u32 {
        let given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        let first_packed = self.standard.len() as u32;
        let mut without = 0;
        for level in 0..self.levels.min(32) {
            let has = if level < first_packed { given.levels[level as usize] } else { given.packed };
            if !has {
                without |= 1 << level;
            }
        }
        levels & without
    }

    /// The bytes of the tiles it has now.
    #[cfg(test)]
    fn given_bytes(&self) -> u64 {
        let given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        given.runs.iter().map(|run| u64::from(run.2) * TILE).sum()
    }
}

impl Drop for Tiled {
    fn drop(&mut self) {
        let given = self.given.get_mut().unwrap_or_else(|e| e.into_inner());
        let mut bytes = 0;
        let mut by_heap: std::collections::BTreeMap<usize, Vec<(u32, u32)>> = std::collections::BTreeMap::new();
        for &(index, start, tiles) in &given.runs {
            by_heap.entry(index).or_default().push((start, tiles));
            bytes += u64::from(tiles) * TILE;
        }
        for (index, runs) in by_heap {
            give_back(index, &runs);
        }
        let levels = given.levels.iter().filter(|&&has| has).count() as u64 + u64::from(given.packed);
        STATS[5].fetch_sub(levels, Ordering::Relaxed);
        STATS[0].fetch_sub(1, Ordering::Relaxed);
        STATS[1].fetch_sub(self.full, Ordering::Relaxed);
        STATS[2].fetch_sub(bytes, Ordering::Relaxed);
    }
}

/// For the frame-rate lines: what this saves and what the mapping calls
/// cost since the line before. Nothing while no texture was made this way.
pub(crate) fn status() {
    static BEFORE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    if s[4] == 0 {
        return;
    }
    let calls = s[6] - BEFORE[0].swap(s[6], Ordering::Relaxed);
    let micros = s[7] - BEFORE[1].swap(s[7], Ordering::Relaxed);
    tracing::info!(
        "textures with levels on demand: {} alive ({} made so far), {} MB given to {} levels alive; \
         {} MB not allocated for levels the title does not hold ({} MB if every level had memory); \
         heaps {} MB ({} given back so far); {calls} UpdateTileMappings calls in {micros} us since the line before ({} in all); \
         {} refreshes left a mapped level without memory (no heap)",
        s[0],
        s[4],
        s[2] >> 20,
        s[5],
        s[1].saturating_sub(s[2]) >> 20,
        s[1] >> 20,
        s[3] >> 20,
        s[8],
        s[6],
        s[9],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_defaults_to_a_megabyte_and_clamps() {
        assert_eq!(parse_floor(None), 1 << 20);
        assert_eq!(parse_floor(Some("junk")), 1 << 20);
        assert_eq!(parse_floor(Some(" 512 ")), 512 << 10);
        assert_eq!(parse_floor(Some("1")), 64 << 10);
        assert_eq!(parse_floor(Some(&u64::MAX.to_string())), u64::MAX);
    }

    /// Tiles are cut lowest first across runs, refused whole when too few
    /// are free, and runs given back join their neighbours.
    #[test]
    fn free_runs_cut_and_join() {
        let mut free = vec![(0, 4), (10, 6)];
        assert_eq!(cut(&mut free, 11), None);
        assert_eq!(free, [(0, 4), (10, 6)]);
        assert_eq!(cut(&mut free, 0), None);
        assert_eq!(cut(&mut free, 6), Some(vec![(0, 4), (10, 2)]));
        assert_eq!(free, [(12, 4)]);
        join(&mut free, &[(10, 2)]);
        assert_eq!(free, [(10, 6)]);
        join(&mut free, &[(0, 4), (4, 6)]);
        assert_eq!(free, [(0, 16)]);
    }

    #[test]
    fn only_sampled_single_layer_2d_chains_fit() {
        let d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 2048,
            Height: 2048,
            DepthOrArraySize: 1,
            MipLevels: 12,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        assert!(shape_fits(&d));
        assert!(shape_fits(&D3D12_RESOURCE_DESC { MipLevels: 0, ..d }), "0: the whole chain");
        assert!(!shape_fits(&D3D12_RESOURCE_DESC { MipLevels: 1, ..d }));
        assert!(!shape_fits(&D3D12_RESOURCE_DESC { DepthOrArraySize: 6, ..d }));
        assert!(!shape_fits(&D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D, ..d }));
        assert!(!shape_fits(&D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..d }));
        assert!(!shape_fits(&D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..d }));
    }

    /// On the host GPU (or WARP), with tier 2: a reserved 2048x2048 BC1
    /// texture takes no tiles until given, a level given takes exactly its
    /// tiles, the packed levels go together, and the tiles go back to the
    /// pool when the texture is dropped.
    #[test]
    fn levels_take_tiles_when_given_and_give_them_back() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: a plain query into a structure of its own size.
        let asked = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        if asked.is_err() || options.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            return;
        }
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 2048,
            Height: 2048,
            DepthOrArraySize: 1,
            MipLevels: 12,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC1_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a plain description and our out pointer.
        unsafe { host.device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture) }
            .expect("a reserved texture");
        let texture = texture.unwrap();
        let tiled = Tiled::make(&host.device, &texture).expect("the host cuts it into tiles");
        assert_eq!(tiled.every_level(), 0xfff);
        assert_eq!(tiled.given_bytes(), 0);
        assert!(!tiled.whole());
        assert_eq!(tiled.without(0xfff), 0xfff);
        // Level 0 of BC1 2048x2048: 2 MiB, 32 tiles.
        assert!(tiled.give(&texture, 1));
        assert_eq!(tiled.without(0xfff), 0xffe);
        assert_eq!(tiled.given_bytes(), 32 * TILE);
        // The last level: packed with the other small ones on most hosts
        // (giving it gives them all), its own standard level where the
        // host packs nothing (WARP).
        assert!(tiled.give(&texture, 1 << 11));
        let packed_from = tiled.standard.len() as u32;
        assert_eq!(tiled.without(0xfff), 0xffe & !(1 << 11) & ((1 << packed_from) - 1));
        // Already given: nothing more is taken.
        let bytes = tiled.given_bytes();
        assert!(tiled.give(&texture, 1 | (1 << 11)));
        assert_eq!(tiled.given_bytes(), bytes);
        assert!(!tiled.whole(), "level 1 and up have none yet");
        assert!(tiled.give(&texture, tiled.every_level()));
        assert!(tiled.whole());
        assert_eq!(tiled.without(0xfff), 0);
        assert_eq!(tiled.given_bytes(), tiled.full);
        // The mappings above are queued, not done: an empty submission waits
        // for them before the heaps go back (in the title, the list that
        // used the texture holds it past them).
        host.execute(&[]);
        let heaps_before = STATS[3].load(Ordering::Relaxed);
        drop(tiled);
        assert!(STATS[3].load(Ordering::Relaxed) <= heaps_before);
    }
}
