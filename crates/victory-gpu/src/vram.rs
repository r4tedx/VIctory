//! Host copies of title buffers in video memory (opt-in,
//! `VICTORY_VRAM_COPIES`: `=1` for buffers the title made without
//! unordered access, `=2` for those shaders may write too; unset or `=0`
//! for none).
//!
//! A title buffer's host copy ([`crate::shadow`]) is a buffer in system
//! memory that the CPU maps and shaders read across the bus. Here a copy
//! also gets a buffer in video memory that the CPU maps through the PCI BAR
//! (a GPU upload heap, `D3D12_OPTIONS16`), and shaders use that one. The
//! system memory buffer stays the CPU's side: every CPU write into the copy
//! goes there and then into video memory, the CPU reads only it, and what
//! shaders wrote comes back into it by a GPU copy before the CPU reads it
//! (`crate::shadow::pull`, `crate::shadow::pull_current`). A CPU read of the BAR runs at ~50 MB/s
//! (#2120's measurement), so the mapping is a [`WriteOnly`], which cannot
//! be read.
//!
//! Admission: copies in video memory hold at most a share of the budget
//! the system gives this process (DXGI): a quarter of it, and never more
//! than what the rest of the process leaves under 85 % of it, so the cap
//! shrinks as textures, render targets and depth buffers take the card.
//! Once use passes 85 %, no copy is admitted until it is back under 75 %.
//! A copy refused stays in system memory, as without the switch. Without
//! that budget, copies took 19 GB of a 10.5 GB budget on an RTX 5070 and
//! the device was removed (#693).
//!
//! Copies are admitted when they are made, and stay. They are made before
//! most of the title's textures (a title run: 1.2 GB of copies at 54 % of
//! the budget, then textures took the card to 95 %), so they are kept at a
//! low residency priority: past the budget, the system moves them to
//! system memory before the title's textures and targets. Shaders then
//! read them across the bus, as they read a copy in system memory.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::Host;

/// Where a host copy's GPU side is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// System memory (its CPU side is its GPU side).
    System = 0,
    /// Video memory, for a buffer the title made without unordered access.
    VramRead = 1,
    /// Video memory, for a buffer shaders may write.
    VramWritten = 2,
}

/// What `VICTORY_VRAM_COPIES` lets go to video memory: 0 nothing, 1
/// copies of read-only buffers, 2 every buffer copy. Read once.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| level_of(std::env::var_os("VICTORY_VRAM_COPIES").as_deref()))
}

/// [`level`] from the switch's value: unset or `0` none, `1` read-only
/// copies, `2` every copy; any other value means 1, as `=1` did before.
fn level_of(value: Option<&std::ffi::OsStr>) -> u8 {
    match value {
        Some(v) if v == "2" => 2,
        v if crate::switch::value_is_on(v) => 1,
        _ => 0,
    }
}

/// The class a new copy of a title buffer asks for (`read_only`: made
/// without unordered access), before admission.
pub(crate) fn wanted(read_only: bool) -> Class {
    match (level(), read_only) {
        (1.., true) => Class::VramRead,
        (2, false) => Class::VramWritten,
        _ => Class::System,
    }
}

/// A CPU mapping of video memory, which the CPU only writes. It has no way
/// to read what it maps.
pub(crate) struct WriteOnly {
    at: *mut u8,
    len: usize,
}

// SAFETY: a persistent mapping, written by one thread at a time (the
// copy's writers hold the submission or the copy's own locks).
unsafe impl Send for WriteOnly {}
// SAFETY: as above.
unsafe impl Sync for WriteOnly {}

impl WriteOnly {
    /// Writes `bytes` at offset `at`, cut to the mapping.
    pub(crate) fn write(&self, at: usize, bytes: &[u8]) {
        let n = bytes.len().min(self.len.saturating_sub(at));
        // SAFETY: inside the mapping (cut above); nothing reads through it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.at.add(at), n) };
    }
}

/// The budget the system gives this process in video memory on the card,
/// and what it uses, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reading {
    pub(crate) budget: u64,
    pub(crate) usage: u64,
}

/// The part of the budget use may reach with copies admitted: 85 %. The
/// rest is left to the title's textures, render targets and depth buffers,
/// which have nowhere else to go, and to what they stream in later.
pub(crate) const MARGIN: (u64, u64) = (85, 100);

/// Once use passed [`MARGIN`], copies wait until it is back under 75 %: the
/// budget is near full then, and the room a texture gave back is soon taken
/// again.
pub(crate) const RESUME: (u64, u64) = (75, 100);

/// `part` of `budget`.
fn of(budget: u64, part: (u64, u64)) -> u64 {
    (u128::from(budget) * u128::from(part.0) / u128::from(part.1)) as u64
}

/// What copies in video memory may hold in all now, with copies there
/// holding `own` bytes (counted in the reading's use): `cap` (by default a
/// quarter of the budget), and never more than what the rest of the
/// process leaves under [`MARGIN`]. Use by others shrinks it; the copies'
/// own bytes do not.
pub(crate) fn limit(r: Reading, own: u64, cap: Option<u64>) -> u64 {
    let cap = cap.unwrap_or(r.budget / 4);
    let others = r.usage.saturating_sub(own);
    cap.min(of(r.budget, MARGIN).saturating_sub(others))
}

/// Whether a copy of `size` bytes may go to video memory, with copies there
/// holding `own` bytes: within [`limit`], and use stays under [`MARGIN`].
/// `full`: use passed the margin before and has not been back under
/// [`RESUME`] since (set and cleared here). Without a reading, never.
pub(crate) fn admits(
    reading: Option<Reading>,
    own: u64,
    size: u64,
    cap: Option<u64>,
    full: &mut bool,
) -> bool {
    let Some(r) = reading else {
        return false;
    };
    if *full && r.usage > of(r.budget, RESUME) {
        return false;
    }
    *full = false;
    if r.usage + size > of(r.budget, MARGIN) {
        *full = true;
        return false;
    }
    own + size <= limit(r, own, cap)
}

/// Bytes of copies in video memory now: charged when one is made, given
/// back when D3D12 destroys its buffer (lists and heap slots may hold it
/// past the copy), and how many there are of each class.
static OWN: AtomicU64 = AtomicU64::new(0);
static LIVE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static LIVE_BYTES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Copies refused for the budget and their bytes; bytes brought back from
/// video memory and the GPU copies that did it (diagnostics).
pub(crate) static REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
pub(crate) static PULLED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Why bytes were brought back from video memory ([`PULLED_BY`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pulled {
    /// Ranges lists declared as written by shaders, unmarked (a snapshot
    /// tells what changed in them, so they come back whole).
    Declared = 0,
    /// Pages marked in a written-pages bitmap.
    Marked = 1,
    /// Pages written at computed addresses (`crate::memory`).
    Computed = 2,
}

/// Bytes brought back by [`Pulled`], before runs of a copy that meet are
/// joined (diagnostics).
pub(crate) static PULLED_BY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Bytes of copies in video memory now.
pub(crate) fn own() -> u64 {
    OWN.load(Relaxed)
}

/// The last reading, when it was taken, and bytes copies took since (the
/// reading's use has them only from the next one).
struct Last {
    at: Option<Instant>,
    reading: Option<Reading>,
    charged: u64,
    /// Use passed [`MARGIN`] and has not been back under [`RESUME`].
    full: bool,
}

static LAST: Mutex<Last> = Mutex::new(Last {
    at: None,
    reading: None,
    charged: 0,
    full: false,
});

/// How old a reading may be.
const MAX_AGE: Duration = Duration::from_millis(250);

/// The budget and use now, with what copies took since DXGI was asked.
fn reading(last: &mut Last) -> Option<Reading> {
    #[cfg(test)]
    if let Some((budget, others)) = *tests::FAKE.lock().unwrap_or_else(|e| e.into_inner()) {
        return Some(Reading {
            budget,
            usage: others + own(),
        });
    }
    if last.at.is_none_or(|at| at.elapsed() >= MAX_AGE) {
        last.reading =
            crate::host::video_memory().map(|(budget, usage)| Reading { budget, usage });
        last.at = Some(Instant::now());
        last.charged = 0;
    }
    last.reading.map(|r| Reading {
        usage: r.usage + last.charged,
        ..r
    })
}

/// `VICTORY_VRAM_COPIES_MB`: the cap in MB, in place of a quarter of the
/// budget.
fn cap() -> Option<u64> {
    static CAP: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_VRAM_COPIES_MB")
            .ok()?
            .parse::<u64>()
            .ok()
            .map(|mb| mb << 20)
    })
}

/// Whether the host has GPU upload heaps. Asked once.
fn gpu_upload_heaps(host: &Host) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
        // SAFETY: a feature query writing the structure of its size.
        let known = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16,
                (&raw mut o).cast(),
                size_of_val(&o) as u32,
            )
        }
        .is_ok();
        let on = known && o.GPUUploadHeapSupported.as_bool();
        if level() > 0 {
            tracing::info!(
                on,
                level = level(),
                "host copies of title buffers in video memory (VICTORY_VRAM_COPIES)"
            );
        }
        on
    })
}

/// The heap copies in video memory go in: a GPU upload heap. In tests on a
/// host with none (or with `VICTORY_TEST_VRAM_STAND_IN=1`), a
/// write-combined heap the CPU maps in its place, so that the same code
/// runs.
fn heap(host: &Host) -> Option<D3D12_HEAP_PROPERTIES> {
    if gpu_upload_heaps(host) && !tests_stand_in() {
        return Some(D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            ..Default::default()
        });
    }
    cfg!(test).then(|| D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    })
}

/// Whether a test asks for the stand-in heap where GPU upload heaps exist.
fn tests_stand_in() -> bool {
    cfg!(test) && crate::switch::on("VICTORY_TEST_VRAM_STAND_IN")
}

/// Called by D3D12 when a copy's buffer in video memory is destroyed: its
/// size and class (the low bits; sizes are whole 64 KiB pages) leave the
/// counts.
unsafe extern "system" fn given_back(data: *mut std::ffi::c_void) {
    let data = data as usize;
    let (size, class) = (data as u64 & !0xFFFF, data & 3);
    OWN.fetch_sub(size, Relaxed);
    LIVE[class].fetch_sub(1, Relaxed);
    LIVE_BYTES[class].fetch_sub(size, Relaxed);
}

/// A buffer in video memory for a copy of class `class` (not `System`) and
/// `size` bytes (whole 64 KiB pages), and its CPU mapping: None when the
/// host has no GPU upload heaps or the budget does not admit it.
pub(crate) fn make(host: &Host, size: u64, class: Class) -> Option<(ID3D12Resource, WriteOnly)> {
    debug_assert!(class != Class::System && size.is_multiple_of(1 << 16));
    let heap = heap(host)?;
    // Admission and charge under one lock: copies made on two threads at
    // once must not both pass on the same room.
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let now = reading(&mut last);
    if !admits(now, own(), size, cap(), &mut last.full) {
        REFUSED[0].fetch_add(1, Relaxed);
        REFUSED[1].fetch_add(size, Relaxed);
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                own_mb = own() >> 20,
                usage_mb = now.map_or(0, |r| r.usage >> 20),
                budget_mb = now.map_or(0, |r| r.budget >> 20),
                "host copies past their share of the video memory budget, or use past 85 % of it: copies made from now on may stay in system memory"
            );
        });
        return None;
    }
    let (buffer, at) = host.mapped_in(size, &heap, true)?;
    // The bytes go back when D3D12 destroys the buffer, not when the copy
    // drops.
    let data = (size | class as u64) as usize as *const std::ffi::c_void;
    // SAFETY: a live buffer; the callback's data is plain numbers.
    let watched = unsafe {
        buffer
            .cast::<windows::Win32::Graphics::Direct3D::ID3DDestructionNotifier>()
            .and_then(|n| n.RegisterDestructionCallback(Some(given_back), data))
    };
    if watched.is_err() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "no destruction callback on this host: host copies stay in system memory"
            )
        });
        return None;
    }
    low_priority(host, &buffer);
    OWN.fetch_add(size, Relaxed);
    LIVE[class as usize].fetch_add(1, Relaxed);
    LIVE_BYTES[class as usize].fetch_add(size, Relaxed);
    last.charged += size;
    Some((
        buffer,
        WriteOnly {
            at,
            len: size as usize,
        },
    ))
}

/// Puts a copy's buffer in video memory at a low residency priority: past
/// the budget, the system moves it to system memory before the title's
/// textures, render targets and depth buffers (normal or high), whose
/// bytes have nowhere else to be read from. A committed buffer, not a
/// descriptor heap or a placed resource (a priority on those removed the
/// device on NVIDIA).
fn low_priority(host: &Host, buffer: &ID3D12Resource) {
    let (Ok(device), Ok(pageable)) = (
        host.device.cast::<ID3D12Device1>(),
        buffer.cast::<ID3D12Pageable>(),
    ) else {
        return;
    };
    // SAFETY: one live pageable object and its priority.
    let set =
        unsafe { device.SetResidencyPriority(1, &Some(pageable), &D3D12_RESIDENCY_PRIORITY_LOW) };
    if let Err(e) = set {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::debug!(error = %e, "no residency priorities on this host"));
    }
}

/// Logs the copies in video memory and what moved, per frame over `frames`
/// (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    if level() == 0 {
        return;
    }
    let mb = |b: u64| b as f64 / 1e6;
    let [refused, refused_bytes] = [0, 1].map(|i| REFUSED[i].swap(0, Relaxed));
    let [pulled, copies] = [0, 1].map(|i| PULLED[i].swap(0, Relaxed));
    let by = [0, 1, 2].map(|i| PULLED_BY[i].swap(0, Relaxed));
    let (reading, full) = {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        (reading(&mut last), last.full)
    };
    tracing::info!(
        "host copies in video memory: {} of read-only buffers ({:.0} MB), {} of written ones ({:.0} MB), {:.0} MB in all \
         (cap now {:.0} MB; card {:.0} of {:.0} MB{}); refused for the budget {refused} ({:.0} MB); \
         shaders' writes brought back {:.2} MB/frame in {:.1} GPU copies/frame \
         (declared ranges {:.2}, marked pages {:.2}, computed-address pages {:.2} MB/frame); \
         system commit they add: about the MB in all (a GPU upload heap is charged about its size; \
         the system memory sides are the copies made without the switch)",
        LIVE[1].load(Relaxed),
        mb(LIVE_BYTES[1].load(Relaxed)),
        LIVE[2].load(Relaxed),
        mb(LIVE_BYTES[2].load(Relaxed)),
        mb(own()),
        mb(reading.map_or(cap().unwrap_or(0), |r| limit(r, own(), cap()))),
        mb(reading.map_or(0, |r| r.usage)),
        mb(reading.map_or(0, |r| r.budget)),
        if full {
            ", past 85 %: admitting none"
        } else {
            ""
        },
        mb(refused_bytes),
        mb(pulled) / frames,
        copies as f64 / frames,
        mb(by[0]) / frames,
        mb(by[1]) / frames,
        mb(by[2]) / frames,
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `VICTORY_VRAM_COPIES` is opt-in: unset or `0` none, `1` read-only
    /// copies, `2` every copy; any other value 1, as before.
    #[test]
    fn vram_copies_are_opt_in() {
        use std::ffi::OsStr;
        assert_eq!(level_of(None), 0, "unset");
        for (value, want) in [("0", 0), ("1", 1), ("2", 2), ("yes", 1), ("", 1)] {
            assert_eq!(level_of(Some(OsStr::new(value))), want, "{value:?}");
        }
    }

    /// A test's budget and use by others than the copies, in bytes, in
    /// place of DXGI's.
    pub(crate) static FAKE: Mutex<Option<(u64, u64)>> = Mutex::new(None);

    /// Whether copies go to a stand-in for GPU upload heaps here.
    pub(crate) fn stand_in(host: &Host) -> bool {
        !gpu_upload_heaps(host) || tests_stand_in()
    }

    /// Runs this binary's test `name` again in a child process with
    /// `VICTORY_VRAM_COPIES=level` (read once per process): true in the
    /// child, where the test does its work; false in the parent once the
    /// child passed, which it must have run. The child runs with the GPU
    /// upload heaps off (`VICTORY_GPU_UPLOAD=0`, `VICTORY_GPU_UPLOAD_WRITES=0`,
    /// stated whatever their defaults): these tests are of video memory copies alone,
    /// with what they refuse in system memory. The order with both on is
    /// `shadow::tests::copies_try_video_memory_then_upload_heaps_then_system_memory`.
    // The pattern is #2120's (0xkn0wh3r3).
    pub(crate) fn in_child(name: &str, level: &str) -> bool {
        const CHILD: &str = "VICTORY_TEST_VRAM_CHILD";
        if std::env::var_os(CHILD).is_some() {
            return true;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([name, "--exact", "--test-threads=1", "--nocapture"])
            .env(CHILD, "1")
            .env("VICTORY_VRAM_COPIES", level)
            .env("VICTORY_GPU_UPLOAD", "0")
            .env("VICTORY_GPU_UPLOAD_WRITES", "0")
            .env_remove("VICTORY_VRAM_COPIES_MB")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "{name} with VICTORY_VRAM_COPIES={level}:\n{text}"
        );
        assert!(text.contains("1 passed"), "{name} did not run:\n{text}");
        eprintln!("{text}");
        false
    }

    /// The admission rule: within the cap and under 85 % of the budget,
    /// nothing without a reading, and #693's numbers refused.
    #[test]
    fn copies_go_to_video_memory_within_the_cap_and_the_budget() {
        const GB: u64 = 1 << 30;
        let r = |budget: u64, usage: u64| Some(Reading { budget, usage });
        let admits = |reading, own, size, cap| admits(reading, own, size, cap, &mut false);
        assert!(
            admits(r(8 * GB, GB), 0, GB, None),
            "room under a quarter of the budget"
        );
        assert!(
            !admits(r(8 * GB, GB), GB + GB / 2, GB, None),
            "past a quarter of the budget"
        );
        assert!(
            admits(r(8 * GB, GB), 2 * GB, GB, Some(3 * GB)),
            "a cap set by hand"
        );
        assert!(
            !admits(r(10 * GB, 8 * GB), 0, GB, None),
            "into the last 15 % of the budget"
        );
        // #693: 19,042 MB of a 10,570 MB budget.
        assert!(
            !admits(r(10_570 << 20, 19_042 << 20), 0, 64 << 10, None),
            "over budget: not even a page"
        );
        assert!(
            !admits(None, 0, 64 << 10, Some(GB)),
            "no reading, no copy in video memory"
        );
    }

    /// The margin left to textures, render targets and depth buffers, with
    /// a title run's numbers (an 11,668 MB budget): the cap shrinks as
    /// others' use grows, even below what copies hold; past 85 % nothing is
    /// admitted, and nothing until use is back under 75 %, however small
    /// the copy.
    #[test]
    fn the_cap_shrinks_as_the_card_fills_and_copies_wait_near_a_full_budget() {
        const MB: u64 = 1 << 20;
        let budget = 11_668 * MB;
        let at = |others: u64, own: u64| Reading {
            budget,
            usage: others + own,
        };
        // Before the title's textures: a quarter of the budget.
        assert_eq!(limit(at(2_000 * MB, 0), 0, None), budget / 4);
        // Textures took 9,800 MB: under 118 MB left for copies.
        let left = limit(at(9_800 * MB, 0), 0, None);
        assert!((117 * MB..119 * MB).contains(&left), "{} MB", left / MB);
        assert!(
            limit(at(9_800 * MB, 1_231 * MB), 1_231 * MB, None) < 1_231 * MB,
            "below what copies hold: no more of them"
        );
        assert!(
            limit(at(9_800 * MB, 0), 0, Some(4_000 * MB)) == left,
            "a cap set by hand shrinks too"
        );
        let mut full = false;
        assert!(!admits(
            Some(at(2_000 * MB, 2_900 * MB)),
            2_900 * MB,
            64 * MB,
            None,
            &mut full
        ));
        assert!(!full, "refused for the cap, not near a full budget");
        assert!(admits(Some(at(9_800 * MB, 0)), 0, 64 * MB, None, &mut full));
        assert!(!admits(
            Some(at(9_800 * MB, 0)),
            0,
            128 * MB,
            None,
            &mut full
        ));
        assert!(full, "past 85 % with it");
        full = false;
        // The title run's end: 11,063 MB in use, 1,231 MB of it copies.
        assert!(!admits(
            Some(at(9_832 * MB, 1_231 * MB)),
            1_231 * MB,
            64 << 10,
            None,
            &mut full
        ));
        assert!(full, "past 85 %");
        // Textures gave back room: 80 % in use, under the margin, but the
        // budget was full a moment ago.
        let own = 100 * MB;
        let eighty = budget / 10 * 8 - own;
        assert!(!admits(
            Some(at(eighty, own)),
            own,
            64 << 10,
            None,
            &mut full
        ));
        assert!(full, "still waiting");
        // Back under 75 %.
        let seventy = budget / 100 * 74 - own;
        assert!(admits(
            Some(at(seventy, own)),
            own,
            64 << 10,
            None,
            &mut full
        ));
        assert!(!full);
    }
}
