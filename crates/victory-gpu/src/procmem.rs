//! Where the process's memory goes (diagnostics): the line logged every 5 s
//! beside the frame line in diagnostics builds, from counters kept as the
//! memory comes and goes.
//!
//! - The process: its private bytes (commit charge) and working set, and
//!   the system's commit in use against its limit (what a long run runs out
//!   of).
//! - Host D3D12 for this process, as DXGI counts it: video memory and
//!   system memory (upload, read-back and system-memory custom heaps, ours
//!   and the driver's).
//! - The title's own memory: committed bytes of its `XMemVirtualAlloc`
//!   reservations (private), its physical pages (pagefile-backed sections:
//!   they take system commit but are not private bytes), and its
//!   `XMemAllocDefault` heap blocks (private). The title's direct host
//!   `VirtualAlloc` and CRT heap calls reach the host unseen and fall into
//!   "other".
//! - Our CPU-visible D3D12 buffers by kind, counted at creation and given
//!   back when the host destroys them (`ID3DDestructionNotifier`): host
//!   copies of title memory and other system-memory buffers shaders use
//!   (write-back, L0), upload buffers (texture staging among them),
//!   read-back buffers, and GPU upload heaps (video memory).
//! - Copy-back snapshots: in use, and kept for reuse.
//! - Host textures (video memory), the pipeline library's bytes (a mapped
//!   file is shareable, not private) and what its last save serialised.
//! - Other: private bytes less what the counters above say is private.
//!
//! The counters are atomics updated where the memory is made and let go;
//! nothing here allocates per event. The line itself costs one
//! `QueryVirtualMemoryInformation` per title reservation, two process
//! queries and two DXGI queries.

use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// Our CPU-visible D3D12 buffers by kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// System memory shaders read and write, mapped write-back (host copies
    /// of title memory, bitmaps, tables): `Host::shared_buffer`.
    Shared = 0,
    /// Upload heap (texture staging, list scratch and uploads, view
    /// tables): `Host::upload_buffer`.
    Upload = 1,
    /// Read-back heap: `Host::readback_buffer`.
    Readback = 2,
    /// GPU upload heap (video memory the CPU writes through the BAR):
    /// `Host::gpu_upload_buffer`.
    GpuUpload = 3,
    /// Host pipeline states of title pipelines (counted, no bytes: what
    /// the driver keeps for each is its own).
    Pipeline = 4,
    /// Writable twins of depth textures (video memory, not counted with
    /// host textures): `crate::twin`.
    Twin = 5,
    /// Tests' own, which nothing else counts under.
    #[cfg(test)]
    Test = 6,
}

const KINDS: usize = if cfg!(test) { 7 } else { 6 };

/// Per kind: buffers alive, and their bytes.
static BUFFERS: [[AtomicU64; 2]; KINDS] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; KINDS];

/// Bits of a destruction callback's context that hold the size; the kind
/// is above them.
const SIZE_BITS: u32 = 58;

/// Counts `buffer` (`size` bytes) under `kind` while it lives: given back
/// when the host destroys it, whoever holds it last (lists and descriptor
/// tables may hold it past its owner). Not counted if the host has no
/// destruction callbacks.
pub(crate) fn count_buffer(buffer: &ID3D12Resource, kind: Kind, size: u64) {
    count(buffer, kind, size);
}

/// [`count_buffer`] for any host D3D12 object (pipeline states: `size` 0).
pub(crate) fn count(object: &impl windows::core::Interface, kind: Kind, size: u64) {
    // Diagnostics builds only: nothing reads the counts otherwise.
    if !cfg!(any(test, feature = "diagnostics")) {
        return;
    }
    let size = size.min((1 << SIZE_BITS) - 1);
    let context = (kind as usize) << SIZE_BITS | size as usize;
    // SAFETY: a live object; the callback's context is a number, not a
    // pointer.
    let watched = unsafe {
        object
            .cast::<windows::Win32::Graphics::Direct3D::ID3DDestructionNotifier>()
            .and_then(|n| n.RegisterDestructionCallback(Some(destroyed), context as *const _))
    };
    if watched.is_ok() {
        BUFFERS[kind as usize][0].fetch_add(1, Relaxed);
        BUFFERS[kind as usize][1].fetch_add(size, Relaxed);
    }
}

unsafe extern "system" fn destroyed(context: *mut std::ffi::c_void) {
    let context = context as usize;
    let (kind, size) = (
        context >> SIZE_BITS,
        (context & ((1 << SIZE_BITS) - 1)) as u64,
    );
    if let Some([n, bytes]) = BUFFERS.get(kind) {
        n.fetch_sub(1, Relaxed);
        bytes.fetch_sub(size, Relaxed);
    }
}

/// Buffers of `kind` alive and their bytes.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn buffers(kind: Kind) -> (u64, u64) {
    let [n, bytes] = &BUFFERS[kind as usize];
    (n.load(Relaxed), bytes.load(Relaxed))
}

/// Copy-back snapshot bytes in use (taken by copies) and kept for reuse
/// (`crate::shadow`'s pool): capacities.
pub(crate) static SNAPSHOTS_IN_USE: AtomicU64 = AtomicU64::new(0);

/// Host copies of title memory alive (`crate::shadow`): count, title bytes.
pub(crate) static COPIES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub(crate) static SNAPSHOTS_KEPT: AtomicU64 = AtomicU64::new(0);

/// The pipeline library's bytes the host reads (mapped or held), whether
/// they are a mapped file, and the size its last save serialised.
pub(crate) static LIBRARY_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static LIBRARY_MAPPED: AtomicU64 = AtomicU64::new(0);
pub(crate) static LIBRARY_SERIALIZED: AtomicU64 = AtomicU64::new(0);

/// The title's physical pages: bytes of the sections alive, and of their
/// pages committed so far.
static SECTIONS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// The title's `XMemAllocDefault` heap blocks: bytes.
static HEAP: AtomicU64 = AtomicU64::new(0);
/// The title's reservations (their base addresses), as `XMemVirtualAlloc`
/// made them.
static RESERVATIONS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// The title's physical pages changed: `reserved` bytes of sections made
/// (negative: let go), `committed` bytes of their pages committed (or let
/// go with them). `victory_hle::modules::xmem` calls it.
pub fn title_sections(reserved: i64, committed: i64) {
    add_signed(&SECTIONS[0], reserved);
    add_signed(&SECTIONS[1], committed);
}

/// The title's default heap grew by `bytes` (negative: shrank).
pub fn title_heap(bytes: i64) {
    add_signed(&HEAP, bytes);
}

/// The title reserved memory at `base` through `XMemVirtualAlloc`: its
/// commit is read from the host at each line, until it is released.
pub fn title_reserved(base: usize) {
    if !cfg!(any(test, feature = "diagnostics")) {
        return;
    }
    RESERVATIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(base);
}

fn add_signed(counter: &AtomicU64, delta: i64) {
    if delta >= 0 {
        counter.fetch_add(delta as u64, Relaxed);
    } else {
        // Never below zero (a section made before counting began).
        let _ = counter.fetch_update(Relaxed, Relaxed, |v| {
            Some(v.saturating_sub(delta.unsigned_abs()))
        });
    }
}

/// The title's reservations still there: their count, bytes reserved and
/// committed. Released ones (no longer an allocation at their base) are
/// forgotten.
#[cfg(any(test, feature = "diagnostics"))]
fn reservations() -> (u64, u64, u64) {
    use windows_sys::Win32::System::Memory::{
        MemoryRegionInfo, QueryVirtualMemoryInformation, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut list = RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner());
    let (mut reserved, mut committed) = (0u64, 0u64);
    list.sort_unstable();
    list.dedup();
    list.retain(|&base| {
        let mut info = WIN32_MEMORY_REGION_INFORMATION::default();
        // SAFETY: our own process and a structure of the size passed.
        let ok = unsafe {
            QueryVirtualMemoryInformation(
                GetCurrentProcess(),
                base as *const _,
                MemoryRegionInfo,
                (&raw mut info).cast(),
                size_of::<WIN32_MEMORY_REGION_INFORMATION>(),
                std::ptr::null_mut(),
            )
        } != 0;
        let live = ok && info.AllocationBase as usize == base;
        if live {
            reserved += info.RegionSize as u64;
            committed += info.CommitSize as u64;
        }
        live
    });
    (list.len() as u64, reserved, committed)
}

/// Process and system memory: private bytes, working set, system commit in
/// use and its limit.
#[cfg(any(test, feature = "diagnostics"))]
fn process() -> (u64, u64, u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
        PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    let mut p = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and sized structures (the _EX one is read
    // through the base type's pointer, as the API documents).
    unsafe {
        K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast(), c.cb);
        K32GetPerformanceInfo(&mut p, p.cb);
    }
    let page = p.PageSize as u64;
    (
        c.PrivateUsage as u64,
        c.WorkingSetSize as u64,
        p.CommitTotal as u64 * page,
        p.CommitLimit as u64 * page,
    )
}

/// What host D3D12 uses for this process on the host's adapter, and the
/// budget DXGI gives it: video memory, then system memory, each (usage,
/// budget) as DXGI counts them (zeros without a host). Usage past the
/// budget is what the host evicts or refuses.
#[cfg(any(test, feature = "diagnostics"))]
fn d3d() -> [(u64, u64); 2] {
    use windows::Win32::Graphics::Dxgi::*;
    static ADAPTER: Mutex<Option<Option<IDXGIAdapter3>>> = Mutex::new(None);
    let mut adapter = ADAPTER.lock().unwrap_or_else(|e| e.into_inner());
    let adapter = adapter.get_or_insert_with(|| {
        let host = crate::host::host()?;
        // SAFETY: plain queries.
        unsafe {
            let luid = host.device.GetAdapterLuid();
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            factory.EnumAdapterByLuid::<IDXGIAdapter3>(luid).ok()
        }
    });
    let Some(adapter) = adapter.as_ref() else {
        return [(0, 0); 2];
    };
    [
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    ]
    .map(|group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a plain query into a structure of ours.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }
            .map_or((0, 0), |()| (info.CurrentUsage, info.Budget))
    })
}

/// Everything the line shows, bytes (counts where named).
#[cfg(any(test, feature = "diagnostics"))]
#[cfg_attr(
    not(feature = "diagnostics"),
    allow(dead_code, reason = "read by the diagnostics line; tests read part")
)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Figures {
    pub(crate) private: u64,
    pub(crate) working_set: u64,
    pub(crate) commit_used: u64,
    pub(crate) commit_limit: u64,
    /// DXGI's (usage, budget) of video and of system memory.
    pub(crate) d3d_video: (u64, u64),
    pub(crate) d3d_system: (u64, u64),
    pub(crate) title_regions: u64,
    pub(crate) title_reserved: u64,
    pub(crate) title_committed: u64,
    pub(crate) sections_reserved: u64,
    pub(crate) sections_committed: u64,
    pub(crate) title_heap: u64,
    pub(crate) buffers: [(u64, u64); 4],
    pub(crate) pipelines: u64,
    pub(crate) twins: (u64, u64),
    /// Host copies alive, of title buffers alive, of loose windows:
    /// (count, title bytes).
    pub(crate) copies: [(u64, u64); 3],
    pub(crate) staging: (u64, u64),
    pub(crate) snapshots_in_use: u64,
    pub(crate) snapshots_kept: u64,
    pub(crate) textures: (u64, u64),
    pub(crate) library: u64,
    pub(crate) library_mapped: bool,
    pub(crate) library_serialized: u64,
}

#[cfg(any(test, feature = "diagnostics"))]
#[cfg_attr(
    not(feature = "diagnostics"),
    allow(dead_code, reason = "read by the diagnostics line; tests read part")
)]
impl Figures {
    /// Private bytes the counters account for: the title's committed
    /// reservations and heap, our system-memory D3D12 buffers (not GPU
    /// upload heaps, which are video memory), snapshots, and the library's
    /// bytes when they are held rather than mapped.
    pub(crate) fn counted_private(&self) -> u64 {
        let system_buffers: u64 = [Kind::Shared, Kind::Upload, Kind::Readback]
            .iter()
            .map(|&k| self.buffers[k as usize].1)
            .sum();
        self.title_committed
            + self.title_heap
            + system_buffers
            + self.snapshots_in_use
            + self.snapshots_kept
            + if self.library_mapped { 0 } else { self.library }
    }

    /// Private bytes nothing above accounts for (negative if the counted
    /// memory is not all private).
    pub(crate) fn other(&self) -> i64 {
        self.private as i64 - self.counted_private() as i64
    }
}

/// Reads every counter (the host queries included).
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn figures() -> Figures {
    let (private, working_set, commit_used, commit_limit) = process();
    let [d3d_video, d3d_system] = d3d();
    let (title_regions, title_reserved, title_committed) = reservations();
    let (textures, texture_bytes, ..) = crate::resource::texture_counts();
    // Main 13f91ca pools texture staging in shared upload buffers (fdf34ab7,
    // 8a3d9759): counted under upload, none apart.
    let (staging, staging_bytes) = (0u64, 0u64);
    Figures {
        private,
        working_set,
        commit_used,
        commit_limit,
        d3d_video,
        d3d_system,
        title_regions,
        title_reserved,
        title_committed,
        sections_reserved: SECTIONS[0].load(Relaxed),
        sections_committed: SECTIONS[1].load(Relaxed),
        title_heap: HEAP.load(Relaxed),
        buffers: [Kind::Shared, Kind::Upload, Kind::Readback, Kind::GpuUpload].map(buffers),
        pipelines: buffers(Kind::Pipeline).0,
        twins: buffers(Kind::Twin),
        copies: {
            let [registered, loose] = crate::shadow::held_copies();
            [
                (COPIES[0].load(Relaxed), COPIES[1].load(Relaxed)),
                registered,
                loose,
            ]
        },
        staging: (staging, staging_bytes),
        snapshots_in_use: SNAPSHOTS_IN_USE.load(Relaxed),
        snapshots_kept: SNAPSHOTS_KEPT.load(Relaxed),
        textures: (textures, texture_bytes),
        library: LIBRARY_BYTES.load(Relaxed),
        library_mapped: LIBRARY_MAPPED.load(Relaxed) != 0,
        library_serialized: LIBRARY_SERIALIZED.load(Relaxed),
    }
}

/// Logs the line (every 5 s, beside the frame line): one group per class
/// of memory, never summed across classes (video memory, the process's
/// private commit, the title's own allocations, CPU-visible heaps), then
/// what the counters leave of private commit. Bytes copied or scanned per
/// frame are the copy statistics' line, not this one.
#[cfg(feature = "diagnostics")]
pub(crate) fn log() {
    let started = std::time::Instant::now();
    let f = figures();
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let mb = |b: u64| b as f64 / 1e6;
    let [shared, upload, readback, gpu_upload] = f.buffers;
    tracing::info!(
        "process memory, by class: [video, DXGI] used {:.0} of budget {:.0} MB, host textures {} ({:.0} MB), depth twins {} ({:.0} MB), GPU upload heaps {} ({:.0} MB); \
         [process] private commit {:.0} MB, working set {:.0} MB, system commit {:.0} of {:.0} MB ({:.0} MB free); \
         [title] reservations {:.0} MB committed of {:.0} MB in {} regions (private), default heap {:.0} MB (private), \
         physical pages {:.0} MB committed of {:.0} MB (system commit, not private); \
         [CPU-visible] DXGI system memory used {:.0} of budget {:.0} MB; shared {} ({:.0} MB), upload {} ({:.0} MB; texture staging {} {:.0} MB), \
         read-back {} ({:.0} MB); host copies of title memory {} ({:.0} MB): of title buffers alive {} ({:.0} MB), loose windows {} ({:.0} MB), others {} ({:.0} MB: views' own, or of buffers the title released that lists or slots still hold); snapshots {:.0} MB in use, {:.0} MB kept for reuse (cap {:.0} MB); \
         [other] host pipelines {}; pipeline library {:.0} MB {}, last save {:.0} MB; \
         private commit not counted above {:.0} MB ({:.0} MB besides DXGI video usage); line {:.2} ms",
        mb(f.d3d_video.0),
        mb(f.d3d_video.1),
        f.textures.0,
        mb(f.textures.1),
        f.twins.0,
        mb(f.twins.1),
        gpu_upload.0,
        mb(gpu_upload.1),
        mb(f.private),
        mb(f.working_set),
        mb(f.commit_used),
        mb(f.commit_limit),
        mb(f.commit_limit.saturating_sub(f.commit_used)),
        mb(f.title_committed),
        mb(f.title_reserved),
        f.title_regions,
        mb(f.title_heap),
        mb(f.sections_committed),
        mb(f.sections_reserved),
        mb(f.d3d_system.0),
        mb(f.d3d_system.1),
        shared.0,
        mb(shared.1),
        upload.0,
        mb(upload.1),
        f.staging.0,
        mb(f.staging.1),
        readback.0,
        mb(readback.1),
        f.copies[0].0,
        mb(f.copies[0].1),
        f.copies[1].0,
        mb(f.copies[1].1),
        f.copies[2].0,
        mb(f.copies[2].1),
        f.copies[0].0.saturating_sub(f.copies[1].0 + f.copies[2].0),
        mb(f.copies[0].1.saturating_sub(f.copies[1].1 + f.copies[2].1)),
        mb(f.snapshots_in_use),
        mb(f.snapshots_kept),
        mb(crate::shadow::POOL_BYTES as u64),
        f.pipelines,
        mb(f.library),
        if f.library_mapped { "mapped" } else { "held" },
        mb(f.library_serialized),
        f.other() as f64 / 1e6,
        (f.other() - f.d3d_video.0 as i64) as f64 / 1e6,
        ms,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test at a time moves the shared counters.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn a_buffer_counts_while_it_lives() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::host() else {
            return;
        };
        const SIZE: u64 = 3 << 20;
        assert_eq!(buffers(Kind::Test), (0, 0));
        // A host copy's kind of buffer: counted as shared too.
        let (buffer, _) = host.shared_buffer(SIZE).unwrap();
        count_buffer(&buffer, Kind::Test, SIZE);
        assert_eq!(buffers(Kind::Test), (1, SIZE));
        // Other tests' buffers come and go: ours is among them.
        assert!(buffers(Kind::Shared).1 >= SIZE);
        // A list or table holding it keeps it counted.
        let held = buffer.clone();
        drop(buffer);
        assert_eq!(buffers(Kind::Test), (1, SIZE));
        drop(held);
        assert_eq!(buffers(Kind::Test), (0, 0), "given back when destroyed");
    }

    #[test]
    fn upload_and_read_back_buffers_count_by_kind() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::host() else {
            return;
        };
        const SIZE: u64 = 5 << 20;
        let (upload, _) = host.upload_buffer(SIZE).unwrap();
        let (back, _) = host.readback_buffer(SIZE).unwrap();
        assert!(buffers(Kind::Upload).1 >= SIZE);
        assert!(buffers(Kind::Readback).1 >= SIZE);
        let figures = figures();
        assert!(figures.buffers[Kind::Upload as usize].1 >= SIZE);
        assert!(figures.counted_private() >= 2 * SIZE);
        drop((upload, back));
    }

    #[test]
    fn a_destroyed_buffer_gives_its_bytes_back() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        // The callback alone, as the host calls it: counted under its kind,
        // given back exactly.
        assert_eq!(buffers(Kind::Test), (0, 0));
        let context = (Kind::Test as usize) << SIZE_BITS | 12345;
        BUFFERS[Kind::Test as usize][0].fetch_add(1, Relaxed);
        BUFFERS[Kind::Test as usize][1].fetch_add(12345, Relaxed);
        // SAFETY: the callback reads its context as a number.
        unsafe { destroyed(context as *mut _) };
        assert_eq!(buffers(Kind::Test), (0, 0));
    }

    #[test]
    fn title_counters_move_and_never_go_below_zero() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let [r, c] = [&SECTIONS[0], &SECTIONS[1]].map(|a| a.load(Relaxed));
        title_sections(1 << 30, 64 << 10);
        assert_eq!(SECTIONS[0].load(Relaxed), r + (1 << 30));
        assert_eq!(SECTIONS[1].load(Relaxed), c + (64 << 10));
        title_sections(-(1 << 30), -(64 << 10));
        assert_eq!(
            [&SECTIONS[0], &SECTIONS[1]].map(|a| a.load(Relaxed)),
            [r, c]
        );
        let heap = HEAP.load(Relaxed);
        title_heap(-(heap as i64) - 100);
        assert_eq!(HEAP.load(Relaxed), 0);
        title_heap(heap as i64);
    }

    #[test]
    fn a_title_reservation_reports_its_commit_until_released() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const RESERVED: usize = 16 << 20;
        const COMMITTED: usize = 3 << 20;
        // SAFETY: a fresh reservation of ours, part committed.
        let base = unsafe { VirtualAlloc(std::ptr::null(), RESERVED, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        // SAFETY: inside the reservation.
        assert!(!unsafe { VirtualAlloc(base, COMMITTED, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let (n0, r0, c0) = reservations();
        title_reserved(base as usize);
        let (n, r, c) = reservations();
        assert_eq!(
            (n, r, c),
            (n0 + 1, r0 + RESERVED as u64, c0 + COMMITTED as u64)
        );
        // SAFETY: our reservation, released whole.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
        assert_eq!(
            reservations(),
            (n0, r0, c0),
            "a released reservation is forgotten"
        );
    }

    #[test]
    fn the_process_figures_are_sane() {
        let (private, working_set, used, limit) = process();
        assert!(private > 0 && working_set > 0, "{private} {working_set}");
        assert!(used > 0 && used <= limit, "{used} {limit}");
    }
}
