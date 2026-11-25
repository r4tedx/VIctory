//! Benchmark: what the submissions of a world frame cost to prepare, with
//! no title and no hardware GPU. Made-up title memory of the kinds the
//! runtime gives a title (write-watched pools; physical pages shown through
//! two views) is copied, checked, guarded and written back by the real
//! submission path ([`super::Host::execute`]) at the volumes a title run on
//! 940d32f published (issue #400, an RX 6700 XT, its late `copies per frame`
//! line): 33 submissions a frame, 21,974 refreshes asking for 20.6 GB,
//! 254,139 change checks over 92.8 GB of which 240 MB had changed, 7,147
//! protect calls, 563 MB of snapshots, 1,996 marked pages, 75 title copies.
//!
//! What that line does not say is made up here and said so where it is
//! used: how many host copies there are and how large, which lists use
//! title memory, how many ranges shaders declare, and whether the title
//! writes the same pages every submission. The lists record no GPU
//! commands, so the time on the GPU is next to nothing.
//!
//! [`world_frame_preparing`] prints the frame's parts next to the published
//! numbers, then what each mechanism costs on its own ([`unit_costs`]).
//!
//! ```text
//! VICTORY_D3D12_WARP=1 cargo test --profile diag -p victory-gpu --features diagnostics \
//!     -- --ignored --nocapture --test-threads 1 world_frame_preparing
//! ```

use std::ptr::null;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use super::{Batch, COPY_BACK, HostList, SUBMISSION_US, copy_mapped, host};
use crate::shadow::Shadow;
use crate::watch::PAGE;

const MIB: usize = 1 << 20;

/// Submissions a frame, and lists in each (the title's device-removed
/// reports name about 12).
const SUBMISSIONS: usize = 33;
const LISTS: usize = 12;
/// The title's big GPU heap, and the other copies by size: (how many,
/// bytes). 4,100 copies of 1.5 GB in all: the published checks cover
/// 365 KB each on average, and `crate::memory` counts ~4,100 copies.
const HEAP: usize = 174 * MIB;
const COPIES: [(usize, usize); 4] = [(6, 32 * MIB), (40, 8 * MIB), (400, MIB), (3653, 2 * PAGE)];
/// Per list: one refresh of a part of the heap this long, and this many of
/// other copies whole (with the declared ranges' refreshes ~57 a list:
/// 22,000 a frame asking for about 20 GB).
const HEAP_REFRESH: usize = 16 * MIB;
const SMALL_REFRESHES: usize = 54;
/// The lists whose shaders use title memory: their page table names every
/// copy. With the title copies below every copy is asked for about 66
/// times a frame (published: 254,139 checks, 62 of each of 4,100 copies).
const MEMORY_LISTS: [usize; 4] = [0, 2, 4, 6];
/// The lists that declare the heap from an address to its end as written
/// through a root view whose writes shaders mark. Every list but those with
/// root UAVs also declares one other copy whole. How many ranges the title
/// declares is not published.
const MARKED_HEAP_LISTS: [usize; 2] = [1, 6];
/// The lists with root UAVs, 4 each (1 MiB, kept as snapshots: 563 MB a
/// frame published). The published line compares none of them when copying
/// back, as when their lists read memory with no copy: so here.
const ROOT_UAV_LISTS: [usize; 4] = [2, 5, 8, 11];
const ROOT_UAVS: usize = 4;
/// The lists with a title buffer copy (the last in one submission of four:
/// 75 a frame).
const TITLE_COPY_LISTS: [usize; 3] = [3, 9, 10];
/// Pages a submission's shaders mark (1,996 a frame published; about half
/// of these fall in ranges that do not go back and are not compared), and
/// of those the ones where they changed bytes.
const MARKED_PAGES: usize = 60;
const MARKED_CHANGED: usize = 36;
/// Pages the title writes before each submission (3,663 a frame, 240 MB
/// if each were another page; some are the same, 160 MB is copied): this
/// many of them the same ones every time, the rest anywhere. The split is
/// not published.
const WRITTEN: usize = 111;
const WRITTEN_SAME: usize = 55;
/// Frames: warm-up, then counted.
const WARM: usize = 2;
const COUNTED: usize = 5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Write-watched memory, as the runtime's pools.
fn pool(len: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let kind = MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH;
    // SAFETY: a fresh allocation, kept for the process's life.
    let p = unsafe { VirtualAlloc(null(), len, kind, PAGE_READWRITE) } as usize;
    assert_ne!(p, 0, "no pool of {len} bytes");
    p
}

/// Physical pages shown twice, as the title's: `len` bytes of a pagefile
/// section in one view (the title's pool view), and again page by page (as
/// its page mappings into the tiled heap are made), each page reported as
/// the runtime reports it. The second mapping's address, then the first's.
fn views(len: usize) -> (usize, usize) {
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
        MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MapViewOfFile, MapViewOfFile3, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualAlloc2, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: a pagefile section mapped read-write; the views are kept for
    // the process's life. The second mapping replaces, page by page, a
    // placeholder of ours split around each page (as the runtime's `map`).
    unsafe {
        let section =
            CreateFileMappingW(-1isize as _, null(), PAGE_READWRITE, (len >> 32) as u32, len as u32, null());
        assert!(!section.is_null(), "no section of {len} bytes");
        let a = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
        assert_ne!(a, 0);
        let kind = MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
        let b = VirtualAlloc2(GetCurrentProcess(), null(), len, kind, PAGE_NOACCESS, std::ptr::null_mut(), 0) as usize;
        assert_ne!(b, 0, "no placeholder of {len} bytes");
        for k in (0..len).step_by(PAGE) {
            VirtualFree((b + k) as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let at = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                (b + k) as *const _,
                k as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(at.Value as usize, b + k, "{}", std::io::Error::last_os_error());
            crate::watch::title_memory_aliased(b + k, &[a + k]);
        }
        (b, a)
    }
}

/// A title buffer and its host copy.
struct Buffer {
    copy: Arc<Shadow>,
    start: usize,
    len: usize,
    /// Where the other view shows the same bytes; 0 in a pool.
    other: usize,
}

impl Buffer {
    fn new(start: usize, len: usize, other: usize) -> Option<Buffer> {
        crate::shadow::register(start, start as u64, len as u64);
        let copy = crate::shadow::find(start as u64, len as u64)?;
        Some(Buffer { copy, start, len, other })
    }

    /// The title writes page `page` of the buffer: through the other view
    /// where there is one and `through` asks, else here.
    fn write(&self, page: usize, value: u64, through: bool) {
        let base = if through && self.other != 0 { self.other } else { self.start };
        // SAFETY: a dword pair inside the buffer's memory (a guarded page
        // faults into the handler).
        unsafe { ((base + page * PAGE + 64) as *mut u64).write_volatile(value) };
    }
}

/// The heap first, then the other copies; every other one of a size in
/// views, the rest in pools. None without a host device.
fn buffers() -> Option<Vec<Buffer>> {
    let (b, a) = views(HEAP);
    let mut all = vec![Buffer::new(b, HEAP, a)?];
    for (n, len) in COPIES {
        let (in_views, in_pool) = (n / 2, n - n / 2);
        let (b, a) = views(in_views * len);
        let p = pool(in_pool * len);
        for k in 0..in_views {
            all.push(Buffer::new(b + k * len, len, a + k * len)?);
        }
        for k in 0..in_pool {
            all.push(Buffer::new(p + k * len, len, 0)?);
        }
    }
    Some(all)
}

/// Our process's page faults so far (soft ones too).
fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// One frame's numbers.
#[derive(Default, Clone)]
struct Frame {
    /// Milliseconds: the title's writes (faults into the guard handler),
    /// preparing, on the GPU, copying back.
    title: f64,
    preparing: f64,
    gpu: f64,
    back: f64,
    /// Page faults while the title wrote and in the submissions.
    faults_title: u64,
    faults_submitting: u64,
    /// The diagnostics counters (`crate::copystat`).
    #[cfg(feature = "diagnostics")]
    counters: Vec<u64>,
}

/// Records and runs one frame.
fn frame(all: &[Buffer], lists: &[HostList], rng: &mut Rng, stamp: &mut u64) -> Frame {
    let host = host().expect("checked by the caller");
    let heap = &all[0];
    let mut f = Frame::default();
    let us = |a: &std::sync::atomic::AtomicU64| a.load(Relaxed);
    let (prepared, on_gpu, back) = (us(&SUBMISSION_US[0]), us(&SUBMISSION_US[1]), us(&COPY_BACK[0]));
    for s in 0..SUBMISSIONS {
        // The title writes: the same pages as before every submission, then
        // pages anywhere; two in three in the heap.
        let faults = page_faults();
        let started = Instant::now();
        let mut same = Rng(0x5eed);
        for k in 0..WRITTEN {
            let r = if k < WRITTEN_SAME { &mut same } else { &mut *rng };
            let b = if k % 3 != 2 { heap } else { &all[1 + r.below(all.len() - 1)] };
            *stamp += 1;
            b.write(r.below(b.len / PAGE), *stamp, k % 2 == 1);
        }
        f.title += started.elapsed().as_secs_f64() * 1e3;
        f.faults_title += page_faults() - faults;
        // The lists.
        let mut marked: Vec<(usize, usize, usize)> = Vec::new();
        for (l, list) in lists.iter().enumerate() {
            if MEMORY_LISTS.contains(&l) {
                crate::memory::bind(list, crate::memory::OURS).expect("a page table");
            }
            let at = heap.start + (rng.below(heap.len - HEAP_REFRESH) & !255);
            heap.copy.refresh(list, at as u64, HEAP_REFRESH as u64);
            for _ in 0..SMALL_REFRESHES {
                let b = &all[1 + rng.below(all.len() - 1)];
                b.copy.refresh(list, b.start as u64, b.len as u64);
            }
            let root_uavs = ROOT_UAV_LISTS.contains(&l);
            if root_uavs {
                // Root UAVs: a range refreshed, then declared (kept as a
                // snapshot), in a list that read memory with no copy.
                for _ in 0..ROOT_UAVS {
                    let at = (heap.start + (rng.below(heap.len - MIB) & !255)) as u64;
                    heap.copy.refresh(list, at, MIB as u64);
                    heap.copy.gpu_writes(list, at, MIB as u64, false);
                }
                list.defer_missed(|| true);
            }
            // Views whose writes shaders mark.
            if MARKED_HEAP_LISTS.contains(&l) {
                let at = heap.start + (rng.below(heap.len - MIB) & !255);
                heap.copy.refresh(list, at as u64, heap.copy.left(at as u64));
                heap.copy.gpu_writes(list, at as u64, heap.copy.left(at as u64), true);
                marked.push((0, at, heap.start + heap.len - at));
            }
            if !root_uavs {
                let k = 1 + rng.below(all.len() - 1);
                let b = &all[k];
                b.copy.refresh(list, b.start as u64, b.len as u64);
                b.copy.gpu_writes(list, b.start as u64, b.len as u64, true);
                marked.push((k, b.start, b.len));
            }
            // The title's buffer copies: 75 a frame, a few KB each.
            if TITLE_COPY_LISTS[..2].contains(&l) || (l == TITLE_COPY_LISTS[2] && s % 4 == 0) {
                let (to, from) = (&all[1 + rng.below(all.len() - 1)], &all[1 + rng.below(all.len() - 1)]);
                list.defer_copy((to.start + 512) as *mut u8, (from.start + 4096) as *const u8, 5333);
            }
            if l == LISTS - 1 {
                // The shaders' part: pages marked, bytes changed in some.
                // After this list's root UAVs, whose declarations do every
                // refresh asked so far: the bytes land on the copies as they
                // are when the GPU runs.
                let writes: Vec<(Arc<Shadow>, u64, bool)> = (0..MARKED_PAGES)
                    .map(|n| {
                        let (k, start, len) = marked[rng.below(marked.len())];
                        let at = (start + rng.below(len)) & !7;
                        (all[k].copy.clone(), at.min(start + len - 8) as u64, n < MARKED_CHANGED)
                    })
                    .collect();
                let value = *stamp;
                // The list's own bitmap, which its marking shaders write.
                list.view_writes().expect("a mark bitmap");
                let bitmap = list.marks.lock().unwrap_or_else(|e| e.into_inner()).clone();
                let bitmap = bitmap.expect("a mark bitmap");
                list.defer_host((0, 0), (0, 0), move || {
                    for (copy, at, changed) in &writes {
                        copy.mark_written(&bitmap, *at);
                        if *changed {
                            copy.write_host(*at, &value.to_le_bytes());
                        }
                    }
                });
            }
            list.close();
        }
        let faults = page_faults();
        let runs: Vec<&HostList> = lists.iter().collect();
        host.execute(&runs);
        f.faults_submitting += page_faults() - faults;
        for list in lists {
            list.reset();
        }
    }
    let ms = |a: &std::sync::atomic::AtomicU64, before: u64| (us(a) - before) as f64 / 1e3;
    f.preparing = ms(&SUBMISSION_US[0], prepared);
    f.gpu = ms(&SUBMISSION_US[1], on_gpu);
    f.back = ms(&COPY_BACK[0], back);
    #[cfg(feature = "diagnostics")]
    {
        crate::watch::frame_done();
        f.counters = crate::copystat::take();
    }
    f
}

/// The median of `value` over `frames`.
fn median(frames: &[Frame], value: impl Fn(&Frame) -> f64) -> f64 {
    let mut v: Vec<f64> = frames.iter().map(value).collect();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A world frame's submissions through [`super::Host::execute`], then the
/// mechanisms on their own. See the module's text for the command.
#[test]
#[ignore = "benchmark"]
fn world_frame_preparing() {
    if host().is_none() {
        println!("no host device: nothing measured");
        return;
    }
    let Some(all) = buffers() else {
        println!("no host buffers for the copies: nothing measured");
        return;
    };
    assert!(all[0].copy.marked(), "the heap's views mark their writes");
    let total: usize = all.iter().map(|b| b.len).sum();
    println!(
        "{} host copies of {:.2} GB of title memory ({} in views shown twice, {} in write-watched pools)",
        all.len(),
        total as f64 / 1e9,
        all.iter().filter(|b| b.other != 0).count(),
        all.iter().filter(|b| b.other == 0).count()
    );
    let host = host().expect("checked above");
    let lists: Vec<HostList> = (0..LISTS).map(|_| host.new_list().expect("a list")).collect();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut stamp = 0u64;
    let mut frames = Vec::new();
    for n in 0..WARM + COUNTED {
        let f = frame(&all, &lists, &mut rng, &mut stamp);
        println!(
            "frame {n}{}: preparing {:.1} ms, on the GPU {:.1} ms, copying back {:.1} ms; the title's writes {:.1} ms; \
             page faults {} submitting, {} in the title's writes",
            if n < WARM { " (warm-up)" } else { "" },
            f.preparing,
            f.gpu,
            f.back,
            f.title,
            f.faults_submitting,
            f.faults_title
        );
        if n >= WARM {
            frames.push(f);
        }
    }
    println!("\nmedian of {COUNTED} frames (synthetic; published: the title on an RX 6700 XT, #400):");
    let row = |what: &str, ours: f64, published: &str| println!("  {what:<44} {ours:>12.1}   {published}");
    row("preparing, ms", median(&frames, |f| f.preparing), "1198");
    row("copying back, ms", median(&frames, |f| f.back), "122");
    row("the title's writes (guard faults), ms", median(&frames, |f| f.title), "not published");
    row(
        "page faults",
        median(&frames, |f| (f.faults_submitting + f.faults_title) as f64),
        "698986",
    );
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::C;
        let c = |c: C| {
            let i = c as usize;
            median(&frames, |f| f.counters[i] as f64)
        };
        let ms = |x: C| c(x) / 1e3;
        let mb = |x: C| c(x) / 1e6;
        row("refreshes", c(C::Refreshes), "21974");
        row("  asking for, MB", mb(C::RefreshBytes), "20600");
        row("change checks", c(C::Watches), "254139");
        row("  checked, MB", mb(C::Checked), "92800");
        row("  changed (copied), MB", mb(C::Checked) - mb(C::Unchanged), "240");
        row("protect calls", c(C::ProtectCalls), "7147");
        row("  pages", c(C::ProtectPages), "9528");
        row("pages opened", c(C::OpenPages), "not quoted");
        row("guard faults", c(C::GuardFaults), "not quoted");
        row("  in the handler, ms", ms(C::HandlerUs), "74.8");
        row("tracked pages written", c(C::WrittenPages), "not quoted");
        row("  the frame before too", c(C::WrittenAgain), "not quoted");
        row("snapshots, MB", mb(C::SnapshotBytes), "563");
        row("  ms", ms(C::SnapshotUs), "31.7");
        row("marked pages compared", c(C::MarkedPages), "1996");
        row("  written back, MB", mb(C::MarkedBytes), "77.6");
        row("  ms", ms(C::MarkedUs), "74.4");
        row("  saved at the title's first write", c(C::SavedPages), "not quoted");
        row("title copies", c(C::Copies), "75");
        println!("  preparing by step (each with the refresh batch it flushes), ms:");
        row("  refresh steps", ms(C::StepRefreshUs), "not quoted");
        row("  title copies", ms(C::StepCopyUs), "142");
        row("  GPU writes", ms(C::StepGpuWriteUs), "563");
        row("  title memory", ms(C::StepMemoryUs), "109");
        row("  work", ms(C::StepWorkUs), "not quoted");
        row("  state preludes", ms(C::PreludeUs), "not quoted");
        println!("  inside those, ms:");
        row("  refresh batches (checks and copies)", ms(C::FlushUs), "not quoted");
        row("  change checks", ms(C::WatchUs), "36.0");
        row("  protecting", ms(C::ProtectUs), "not quoted");
        row("  waiting for the guard lock", ms(C::GuardWaitUs), "not quoted");
        row("  the kernel's copies", ms(C::MoveUs), "not quoted");
        row("  page faults preparing", c(C::FaultsPrep), "not quoted");
        row("  page faults copying back", c(C::FaultsBack), "not quoted");
    }
    #[cfg(not(feature = "diagnostics"))]
    println!("  (no counters: build with --features diagnostics)");
    unit_costs(&all);
}

/// Milliseconds since `started`.
fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

/// What each mechanism of the frame costs on its own, per unit.
fn unit_costs(all: &[Buffer]) {
    println!("\nunit costs:");
    // Everything taken and guarded first.
    for b in all {
        b.copy.update(b.start, b.len);
    }
    // 1. A check of a copy nothing wrote.
    const ROUNDS: usize = 20;
    let started = Instant::now();
    for _ in 0..ROUNDS {
        for b in all {
            assert_eq!(b.copy.update(b.start, b.len), 0);
        }
    }
    let direct = ms(started) * 1e6 / (ROUNDS * all.len()) as f64;
    let started = Instant::now();
    for _ in 0..ROUNDS {
        let mut batch = Batch::default();
        for b in all {
            batch.add(&b.copy, b.start, b.len);
        }
        assert_eq!(batch.flush(), 0);
    }
    let batched = ms(started) * 1e6 / (ROUNDS * all.len()) as f64;
    println!(
        "  a check of an unchanged copy: {direct:.0} ns on its own, {batched:.0} ns through a refresh batch \
         ({:.1} ms for the published 254,139 a frame)",
        batched * 254_139.0 / 1e6
    );
    // The same checks while shaders' ranges are declared on the heap.
    let heap = &all[0];
    for declared in [1usize, 16, 64] {
        let ranges: Vec<(usize, usize)> = (0..declared)
            .map(|k| (heap.start + k * 2 * MIB + 4096, MIB))
            .collect();
        for &(at, len) in &ranges {
            heap.copy.writing(at, len, true);
        }
        let started = Instant::now();
        for _ in 0..ROUNDS * 10 {
            heap.copy.update(heap.start, heap.len);
        }
        let each = ms(started) * 1e3 / (ROUNDS * 10) as f64;
        for &(at, len) in &ranges {
            heap.copy.copy_back(at, len, 1, |_, _| true);
        }
        println!("  a check of the unchanged heap with {declared} declared ranges on it: {each:.2} us");
    }
    // 2. Declaring a range shaders write through marked views: the heap
    // from an address to its end, as a root view's.
    for (what, at) in [("the heap from its start", heap.start + 256), ("the heap's last 1 MiB", heap.start + heap.len - MIB)] {
        const N: usize = 200;
        let len = heap.start + heap.len - at;
        let started = Instant::now();
        for _ in 0..N {
            heap.copy.writing(at, len, true);
        }
        let arm = ms(started) * 1e3 / N as f64;
        let started = Instant::now();
        heap.copy.copy_back(at, len, 1, |_, _| true);
        println!(
            "  declaring {what} to its end as written (marked): {arm:.1} us each; ending it {:.1} us",
            ms(started) * 1e3
        );
    }
    // 3. A guard's round by the kind of memory: the title writes a page
    // (the fault, the handler), the next refresh copies and guards it again.
    for (kind, in_views) in [("a write-watched pool", false), ("views shown twice", true)] {
        let Some(b) = all.iter().find(|b| b.len == 8 * MIB && (b.other != 0) == in_views) else {
            continue;
        };
        let pages = b.len / PAGE;
        for (what, step) in [("every other page", 2usize), ("every page", 1)] {
            let n = pages / step;
            let (mut write_ms, mut refresh_ms, mut write_faults, mut refresh_faults) = (0.0, 0.0, 0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::take();
            const TIMES: usize = 10;
            for _ in 0..TIMES {
                let faults = page_faults();
                let started = Instant::now();
                for k in 0..n {
                    b.write(k * step, k as u64, false);
                }
                write_ms += ms(started);
                let after = page_faults();
                write_faults += after - faults;
                let started = Instant::now();
                assert_eq!(b.copy.update(b.start, b.len), n * PAGE);
                refresh_ms += ms(started);
                refresh_faults += page_faults() - after;
            }
            let each = |total: f64| total * 1e3 / (TIMES * n) as f64;
            let per = |faults: u64| faults as f64 / (TIMES * n) as f64;
            print!(
                "  {kind}, {what} written: the write {:.1} us and {:.1} page faults a page; \
                 the refresh {:.1} us and {:.1} page faults a page",
                each(write_ms),
                per(write_faults),
                each(refresh_ms),
                per(refresh_faults)
            );
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::C;
                let c = crate::copystat::take();
                let us = |x: C| c[x as usize] as f64 / (TIMES * n) as f64;
                print!(
                    " (the handler {:.1} us, protecting {:.1} us in {:.2} calls, the kernel's copy {:.1} us)",
                    us(C::HandlerUs),
                    us(C::ProtectUs),
                    us(C::ProtectCalls),
                    us(C::MoveUs)
                );
            }
            println!();
        }
    }
    // 4. The copy itself: a page through the kernel and in user mode.
    let from = all.iter().find(|b| b.len == 8 * MIB && b.other == 0).expect("a pool copy");
    let mut to = vec![0u8; from.len];
    for (what, piece) in [("64 KiB pages", PAGE), ("8 MiB at once", 8 * MIB)] {
        const TIMES: usize = 20;
        let mut time = |plain: bool| {
            let started = Instant::now();
            for _ in 0..TIMES {
                for at in (0..from.len).step_by(piece) {
                    let (src, dst) = ((from.start + at) as *const u8, to[at..].as_mut_ptr());
                    if plain {
                        // SAFETY: both ranges are `piece` bytes of ours.
                        unsafe { std::ptr::copy_nonoverlapping(src, dst, piece) };
                    } else {
                        copy_mapped(dst, src, piece, false);
                    }
                }
            }
            (TIMES * from.len) as f64 / 1e6 / started.elapsed().as_secs_f64()
        };
        let (kernel, plain) = (time(false), time(true));
        println!("  copying {what}: {kernel:.0} MB/s through the kernel, {plain:.0} MB/s in user mode");
    }
}
