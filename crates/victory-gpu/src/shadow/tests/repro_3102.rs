//! Tests for the review of #3102: copies in video memory with a mirror.
//!
//! Each one is adapted from a test of #2993 (named in its doc). They use
//! #3102's own helpers (`mirrored_buffer`, `gpu_store`, `gpu_bytes`) and run
//! in a child process with the switches #3102's own tests set for the
//! mirrored path (`VICTORY_GPU_UPLOAD=1`, `VICTORY_GPU_UPLOAD_WRITES=1`, the
//! default on NVIDIA). Nothing returns early: each test checks it runs on the
//! software adapter with GPU upload heaps and mirrors, and prints that.
//!
//! The production code is #3102's, with two test-only hooks
//! (`#[cfg(test)]`, compiled out of every other build): [`copy_fails`] in
//! `Host::copy_now` and [`before_taken`] in `Shadow::copy_back_marked`.

use super::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use windows::Win32::Graphics::Dxgi::*;

/// While set, `Host::copy_now` copies nothing and returns (only logged), as
/// it does when a step of the copy fails.
static FAIL_COPIES: AtomicBool = AtomicBool::new(false);

pub(crate) fn copy_fails() -> bool {
    FAIL_COPIES.load(Relaxed)
}

thread_local! {
    /// Run once in `Shadow::copy_back_marked`, after it wrote title memory
    /// and pushed the mirror, before the page is taken (guarded) again.
    static BEFORE_TAKEN: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

pub(crate) fn before_taken() {
    if let Some(f) = BEFORE_TAKEN.with(|h| h.borrow_mut().take()) {
        f();
    }
}

/// The switches #3102's tests set for copies in video memory with mirrors.
const MIRRORS: &[(&str, &str)] = &[
    ("VICTORY_GPU_UPLOAD", "1"),
    ("VICTORY_GPU_UPLOAD_WRITES", "1"),
];
/// Written copies in system memory (main's behaviour, the control).
const SYSTEM: &[(&str, &str)] = &[
    ("VICTORY_GPU_UPLOAD", "1"),
    ("VICTORY_GPU_UPLOAD_WRITES", "0"),
];

/// Runs this binary's test `name` (in this module) again in a child process
/// with `env` (the switches are read once per process): true in the child,
/// where the test does its work; false in the parent once the child passed.
/// A child's failure fails the parent with the child's output.
fn in_child(name: &str, env: &[(&str, &str)]) -> bool {
    const CHILD: &str = "VICTORY_TEST_REPRO_3102_CHILD";
    if std::env::var_os(CHILD).is_some() {
        return true;
    }
    let name = format!("shadow::tests::repro_3102::{name}");
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            &name[..],
            "--exact",
            "--test-threads=1",
            "--nocapture",
            "--include-ignored",
        ])
        .env(CHILD, "1");
    for (k, v) in env {
        command.env(k, v);
    }
    let out = command.output().unwrap();
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    eprintln!("{text}");
    assert!(out.status.success(), "{name} with {env:?} failed:\n{text}");
    assert!(text.contains("1 passed"), "{name} did not run:\n{text}");
    false
}

/// Checks and prints that the host device is the software adapter (WARP)
/// and whether copies go to GPU upload heaps (`mirrors`: written ones too).
fn on_warp(mirrors: bool) {
    let host = host().expect("a host GPU");
    // SAFETY: plain device and DXGI queries.
    let desc = unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
            .and_then(|a| a.GetDesc1())
    }
    .expect("the adapter's description");
    let end = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    let name = String::from_utf16_lossy(&desc.Description[..end]);
    eprintln!(
        "REPRO-3102 adapter \"{name}\" vendor {:#x}: gpu_upload {} gpu_upload_writes {}",
        desc.VendorId,
        host.gpu_upload(),
        host.gpu_upload_writes()
    );
    assert_eq!(desc.VendorId, 0x1414, "the software adapter (WARP)");
    assert!(host.gpu_upload(), "GPU upload heaps here");
    assert_eq!(
        host.gpu_upload_writes(),
        mirrors,
        "written copies in video memory"
    );
}

fn title(at: usize) -> u8 {
    // SAFETY: a byte of a test pool.
    unsafe { (at as *const u8).read_volatile() }
}

fn write_title(at: usize, byte: u8) {
    // SAFETY: a byte of a test pool.
    unsafe { (at as *mut u8).write_volatile(byte) };
}

/// The mirror's byte at title address `at`, read as is (not freshened).
fn mirror(copy: &Shadow, at: usize) -> u8 {
    // SAFETY: inside the copy's mapping.
    unsafe { copy.data.add(at - copy.start as usize).read_volatile() }
}

/// The shaders' mark for the page of title address `at` (a GPU copy sets
/// none), as in #3102's `gpu_writes_reach_title_memory`.
fn mark(copy: &Shadow, at: usize) {
    let (bit, _) = copy.mark_entry(at as u64).expect("marked memory");
    let words = bitmap(false).expect("the marks bitmap").data as *mut u32;
    // SAFETY: a dword of the mapped bitmap; no submission runs.
    unsafe { *words.add(bit as usize / 32) |= 1 << (bit % 32) };
}

/// (e) A submission's shaders write `byte` into a mirrored copy at `at` in a
/// marked page (0) and in a declared unmarked range (page 1); the GPU copies
/// that bring the writes back to the mirror after it do not run. `current`:
/// the mirror was current before (read by the CPU since its refresh).
/// Adapted from #2993's `a_failed_gpu_copy_of_shaders_writes_loses_none_of_them`.
fn failed_bring_back(current: bool) {
    on_warp(true);
    let host = host().unwrap();
    assert!(bitmap(false).is_some(), "write marks here");
    let (p, len, copy) = mirrored_buffer(2);
    assert!(copy.mirrored());
    if current {
        copy.freshen(0, len);
    }
    let (at, byte) = (8, 0x5au8);
    let list = host.new_list().unwrap();
    copy.refresh(&list, p as u64, len as u64);
    copy.gpu_writes(&list, p as u64, PAGE as u64, true);
    copy.gpu_writes(&list, (p + PAGE) as u64, PAGE as u64, false);
    gpu_store(&list, &copy, p + at, &[byte]);
    gpu_store(&list, &copy, p + PAGE + at, &[byte]);
    // From here on (after the submission's snapshots), the GPU copies fail.
    list.defer_host(|| FAIL_COPIES.store(true, Relaxed));
    list.close();
    mark(&copy, p + at);
    host.execute(&[&list]);
    FAIL_COPIES.store(false, Relaxed);
    let sides = |a: usize| (title(a), mirror(&copy, a), gpu_bytes(&copy, a, 1)[0]);
    let (marked, declared) = (sides(p + at), sides(p + PAGE + at));
    let stale = copy.stale_pages();
    // Bytes of each title page that are no longer the pattern (but `at`).
    let changed = |page: usize| {
        (0..PAGE)
            .filter(|&i| i != at && title(p + page * PAGE + i) != ((page * PAGE + i) % 251) as u8)
            .count()
    };
    eprintln!(
        "REPRO-3102 (e) current {current}: (title, mirror, video) marked {marked:#04x?} declared {declared:#04x?}, \
         pattern there {:#04x}, stale pages after {stale:?}, title bytes off the pattern elsewhere: page 0 {} page 1 {}",
        (at % 251) as u8,
        changed(0),
        changed(1)
    );
    assert_eq!(
        (marked, declared),
        ((byte, byte, byte), (byte, byte, byte)),
        "(title, mirror, video memory) at the marked page and the declared range: the shader's byte \
         should survive a failed bring-back (pattern byte there {:#04x}; stale pages after {stale:?})",
        (at % 251) as u8
    );
    unregister(p, p as u64);
}

#[test]
fn e_a_failed_bring_back_keeps_the_shaders_writes() {
    if in_child("e_a_failed_bring_back_keeps_the_shaders_writes", MIRRORS) {
        failed_bring_back(true);
    }
}

#[test]
fn e_a_failed_bring_back_keeps_the_shaders_writes_stale_mirror() {
    if in_child(
        "e_a_failed_bring_back_keeps_the_shaders_writes_stale_mirror",
        MIRRORS,
    ) {
        failed_bring_back(false);
    }
}

/// (h) A shader wrote a byte through a view no list declared or marked (a
/// shader without write marks): it stays in the GPU side. Later the title
/// writes that byte while a submission whose shaders write another byte of
/// the page declares the page unmarked. At its copy-back the title's byte
/// should stay, as with written copies in system memory, where the snapshot
/// holds the earlier shader's byte. The mirror was current before (read by
/// the CPU since its refresh). Adapted from #2993's
/// `an_old_unmarked_write_does_not_reach_title_memory_late_at_2` and `_at_0`.
fn old_unmarked_write(mirrors: bool) {
    on_warp(mirrors);
    let host = host().unwrap();
    let (p, len, copy) = mirrored_buffer(2);
    assert_eq!(copy.mirrored(), mirrors);
    copy.freshen(0, len);
    // The earlier shader's write: on the GPU side only, nothing declared.
    let list = host.new_list().unwrap();
    gpu_store(&list, &copy, p + 10, &[0xaa]);
    list.close();
    host.execute(&[&list]);
    // This submission: the page declared unmarked, a shader writes p + 20,
    // the title writes p + 10 while it runs.
    let list = host.new_list().unwrap();
    copy.refresh(&list, p as u64, len as u64);
    copy.gpu_writes(&list, p as u64, PAGE as u64, false);
    gpu_store(&list, &copy, p + 20, &[0xbb]);
    list.defer_host(move || write_title(p + 10, 0x77));
    list.close();
    host.execute(&[&list]);
    let got = (title(p + 10), title(p + 20));
    eprintln!("REPRO-3102 (h) mirrors {mirrors}: title (p+10, p+20) = {got:#04x?}");
    assert_eq!(
        got,
        (0x77, 0xbb),
        "mirrors {mirrors}: title memory at p+10 keeps the title's byte (0x77, not the old shader's 0xaa), \
         p+20 gets this submission's 0xbb"
    );
    unregister(p, p as u64);
}

#[test]
fn h_an_old_unmarked_write_does_not_reach_title_memory_late_with_mirrors() {
    if in_child(
        "h_an_old_unmarked_write_does_not_reach_title_memory_late_with_mirrors",
        MIRRORS,
    ) {
        old_unmarked_write(true);
    }
}

#[test]
fn h_an_old_unmarked_write_does_not_reach_title_memory_late_in_system_memory() {
    if in_child(
        "h_an_old_unmarked_write_does_not_reach_title_memory_late_in_system_memory",
        SYSTEM,
    ) {
        old_unmarked_write(false);
    }
}

/// (c) The order of a marked copy-back: the title writes the page after the
/// copy-back wrote title memory and pushed the mirror, before the page is
/// taken (guarded) again. Afterwards, and after the next refresh, the GPU
/// side should have the title's byte. Adapted from #2993's
/// `a_title_write_during_a_copy_back_reaches_video_memory`.
fn title_write_during_copy_back(mirrors: bool) {
    on_warp(mirrors);
    let host = host().unwrap();
    assert!(bitmap(false).is_some(), "write marks here");
    let (p, _, copy) = mirrored_buffer(1);
    assert_eq!(copy.mirrored(), mirrors);
    copy.freshen(0, PAGE);
    let list = host.new_list().unwrap();
    copy.refresh(&list, p as u64, PAGE as u64);
    copy.gpu_writes(&list, p as u64, PAGE as u64, true);
    gpu_store(&list, &copy, p + 10, &[0xaa]);
    list.close();
    mark(&copy, p + 10);
    BEFORE_TAKEN.with(|h| *h.borrow_mut() = Some(Box::new(move || write_title(p + 100, 0x77))));
    host.execute(&[&list]);
    let fired = BEFORE_TAKEN.with(|h| h.borrow_mut().take()).is_none();
    assert!(fired, "the copy-back of the marked page ran the hook");
    assert_eq!(
        (title(p + 10), title(p + 100)),
        (0xaa, 0x77),
        "title memory has both"
    );
    let video = gpu_bytes(&copy, p + 100, 1)[0];
    let mirrored = mirror(&copy, p + 100);
    // What the next submission's refresh copies.
    let refreshed = copy.update(p, PAGE);
    let after = gpu_bytes(&copy, p + 100, 1)[0];
    eprintln!(
        "REPRO-3102 (c) mirrors {mirrors}: at p+100 video {video:#04x} mirror {mirrored:#04x}; \
         the next refresh copied {refreshed} bytes; video after it {after:#04x}"
    );
    assert_eq!(
        (video, mirrored, after),
        (0x77, 0x77, 0x77),
        "mirrors {mirrors}: the GPU side (now, the mirror, after the next refresh, which copied \
         {refreshed} bytes) has the title's write at p+100 (pattern {:#04x})",
        100u8
    );
    unregister(p, p as u64);
}

#[test]
fn c_a_title_write_during_a_copy_back_reaches_video_memory_with_mirrors() {
    if in_child(
        "c_a_title_write_during_a_copy_back_reaches_video_memory_with_mirrors",
        MIRRORS,
    ) {
        title_write_during_copy_back(true);
    }
}

#[test]
fn c_a_title_write_during_a_copy_back_reaches_video_memory_in_system_memory() {
    if in_child(
        "c_a_title_write_during_a_copy_back_reaches_video_memory_in_system_memory",
        SYSTEM,
    ) {
        title_write_during_copy_back(false);
    }
}

/// (d) Every refresh of a mirrored copy reaches video memory, read by a GPU
/// copy out (`gpu_bytes`, which does not freshen the mirror, unlike
/// `read_host`): the first refresh, a title write, and under a range a list
/// declared written. Adapted from #2993's
/// `refreshes_of_copies_in_video_memory_reach_video_memory`.
#[test]
fn d_refreshes_of_mirrored_copies_reach_video_memory() {
    if !in_child("d_refreshes_of_mirrored_copies_reach_video_memory", MIRRORS) {
        return;
    }
    on_warp(true);
    let len = 3 * PAGE;
    let p = crate::watch::tests::pool(3);
    for at in 0..len {
        write_title(p + at, (at % 251) as u8);
    }
    register(p, p as u64, len as u64);
    let copy = find(p as u64, len as u64).unwrap();
    assert!(copy.mirrored());
    assert_eq!(copy.update(p, len), len);
    // SAFETY: our pool.
    let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, len) }.to_vec();
    assert!(gpu_bytes(&copy, p, len) == bytes, "the first refresh");
    write_title(p + PAGE + 5, 0xee);
    assert_eq!(copy.update(p, len), PAGE, "the page written");
    assert_eq!(gpu_bytes(&copy, p + PAGE + 5, 1), [0xee], "a title write");
    copy.writing(p, len, false);
    write_title(p + 2 * PAGE + 7, 0xdd);
    assert!(copy.update(p + 2 * PAGE, 16) > 0);
    assert_eq!(
        gpu_bytes(&copy, p + 2 * PAGE + 7, 1),
        [0xdd],
        "under a busy range"
    );
    copy.copy_back(p, len);
    eprintln!("REPRO-3102 (d) every refresh kind reached video memory");
    unregister(p, p as u64);
}

/// (d) What #2051's `copy_queued` does to `self.data` (title bytes copied
/// into it directly, no `stale` bit, no `push`), done by hand here: under
/// #3102 `data` of a written copy is the mirror. `read_host` reports the new
/// bytes; the GPU side, copied out, does not. This stands in for #2051's
/// code, which #3102 does not contain.
#[test]
fn d_a_write_of_the_mirror_alone_is_seen_by_read_host_not_the_gpu() {
    if !in_child(
        "d_a_write_of_the_mirror_alone_is_seen_by_read_host_not_the_gpu",
        MIRRORS,
    ) {
        return;
    }
    on_warp(true);
    let (p, len, copy) = mirrored_buffer(1);
    assert!(copy.mirrored());
    copy.freshen(0, len);
    write_title(p + 40, 0xc1);
    // As a queued refresh writing `self.data`.
    // SAFETY: inside the mirror's mapping.
    unsafe { copy.data.add(40).write_volatile(title(p + 40)) };
    let host_view = copy.read_host((p + 40) as u64, 1)[0].to_le_bytes()[0];
    let gpu = gpu_bytes(&copy, p + 40, 1)[0];
    eprintln!("REPRO-3102 (d) mirror-only write: read_host {host_view:#04x} GPU side {gpu:#04x}");
    assert_eq!(host_view, 0xc1, "read_host sees the mirror");
    assert_eq!(
        gpu, 0xc1,
        "the GPU side has the queued refresh's byte (pattern {:#04x})",
        40u8
    );
    unregister(p, p as u64);
}

/// (f) With the diagnostics switch `VICTORY_COPIES_IN_VIDEO_MEMORY` set as
/// well: the CPU sides (a written copy's mirror, the marks bitmap, and a
/// written copy kept in system memory, the branch a copy the budget refuses
/// takes) are in system memory (CUSTOM, write-back), not in GPU upload heaps
/// outside the budget. Adapted from #2993's
/// `with_both_switches_the_cpu_sides_stay_in_system_memory`.
fn cpu_sides_with_both_switches(mirrors: bool) {
    on_warp(mirrors);
    let heap = |b: &ID3D12Resource| {
        let mut props = D3D12_HEAP_PROPERTIES::default();
        // SAFETY: a live committed buffer; plain out pointer.
        unsafe { b.GetHeapProperties(Some(&mut props), None) }.unwrap();
        (props.Type.0, props.CPUPageProperty.0)
    };
    let system = (
        D3D12_HEAP_TYPE_CUSTOM.0,
        D3D12_CPU_PAGE_PROPERTY_WRITE_BACK.0,
    );
    let (p, _, copy) = mirrored_buffer(1);
    assert_eq!(copy.mirrored(), mirrors);
    let cpu_side = match &copy.vram {
        Some(v) => heap(&v.mirror),
        None => heap(&copy.buffer),
    };
    let marks = heap(&bitmap(false).expect("the marks bitmap")._buffer);
    eprintln!(
        "REPRO-3102 (f) mirrors {mirrors}: (heap type, CPU page property) copy's CPU side {cpu_side:?} \
         marks bitmap {marks:?}; system memory is {system:?}, GPU upload heap type {}",
        D3D12_HEAP_TYPE_GPU_UPLOAD.0
    );
    assert_eq!(
        (cpu_side, marks),
        (system, system),
        "mirrors {mirrors}: the CPU side of the copy and the marks bitmap are in system memory \
         (heap type {} is a GPU upload heap)",
        D3D12_HEAP_TYPE_GPU_UPLOAD.0
    );
    unregister(p, p as u64);
}

#[test]
fn f_with_both_switches_the_cpu_sides_stay_in_system_memory_mirror() {
    let mut env = MIRRORS.to_vec();
    env.push(("VICTORY_COPIES_IN_VIDEO_MEMORY", "1"));
    if in_child(
        "f_with_both_switches_the_cpu_sides_stay_in_system_memory_mirror",
        &env,
    ) {
        cpu_sides_with_both_switches(true);
    }
}

#[test]
fn f_with_both_switches_the_cpu_sides_stay_in_system_memory_kept_copy() {
    let mut env = SYSTEM.to_vec();
    env.push(("VICTORY_COPIES_IN_VIDEO_MEMORY", "1"));
    if in_child(
        "f_with_both_switches_the_cpu_sides_stay_in_system_memory_kept_copy",
        &env,
    ) {
        cpu_sides_with_both_switches(false);
    }
}
