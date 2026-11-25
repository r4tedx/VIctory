//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! Submissions are not waited for: each signals the next value of the host
//! fence and joins the [`Timeline`], and is finished (what its shaders
//! wrote copied back into title memory, then its bottom-of-pipe writes and
//! the title fence signals queued behind it) in submission order once the
//! GPU has run it: by our completion thread, or by whichever thread first
//! needs it (a later submission's step touching memory it writes, a reset
//! of one of its lists, a present).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::timeline::{Timeline, Touches};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    /// Submissions not finished yet, oldest first.
    timeline: Timeline<Pending>,
    /// Whether the completion thread runs.
    finisher: std::sync::Once,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A ring of CPU-only views, for the CPU handles of unordered access
    /// clears, used as `rtvs`.
    cpu_views: ID3D12DescriptorHeap,
    cpu_view_size: u32,
    next_cpu_view: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the timeline's lists are pinned while in it (`Pending`).
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// The host GPU if it was made (never makes it).
pub(crate) fn made() -> Option<&'static Host> {
    HOST.get()?.as_ref()
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let cpu_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let cpu_view_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("host D3D12 device ready");
        // The console fuses its multiply-adds (`v_fma_f32`, `v_fmac_f32`,
        // `v_fma_mix*`); the lowering's DXIL `FMad` is fused only where the
        // host chooses (the RX 6700 XT does, WARP does not).
        match (victory_shader::emit::host::Host {
            device: device.clone(),
        })
        .fuses_fmad()
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "the host GPU does not fuse multiply-adds: results of the console's fused v_fma and v_fmac can differ in kind here"
            ),
            Err(e) => tracing::debug!(error = %e, "multiply-add fusing not checked"),
        }
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            timeline: Timeline::new(),
            finisher: std::sync::Once::new(),
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            cpu_views,
            cpu_view_size,
            next_cpu_view: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(Presentation::default()),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    use super::*;

    #[test]
    fn list_scratch_grows_to_fit_and_reuses_fenced_chunks() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let first = list.scratch(1).unwrap();
        let second = list.scratch(1).unwrap();
        assert_eq!(second.0 - first.0, 256);
        assert_eq!(second.1 as usize - first.1 as usize, 256);
        assert_eq!(list.scratch.lock().unwrap().chunks[0].capacity, FIRST_SCRATCH_CHUNK);
        let large = list.scratch(96 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[1].capacity, 128 << 10);
        assert!(list.scratch(u64::MAX).is_none());
        assert!(list.scratch(SCRATCH_CHUNK + 1).is_some());
        list.close();
        host.execute(&[&list]);
        host.drain();
        list.reset();
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 2);
        list.close();
        host.execute(&[&list]);
        host.drain();
        list.reset();
        assert_eq!(list.scratch(512).unwrap(), first);
        list.close();
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        // The debug layer appends its own note to the name we gave.
        assert!(is(None, Some("victory host list 7 (debug layer indirect)")));
        assert!(is(None, Some("victory host list 7 (x)")));
        assert!(!is(
            None,
            Some("victory host list 70 (debug layer indirect)")
        ));
        assert!(!is(None, Some("victory host list 7 debug layer")));
        assert!(!is(None, Some("victory host list 7(")));
        assert!(is(None, Some("victory host list 7")));
        assert_eq!(list_name(70), "victory host list 70");
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
        // With the debug layer: our name and the layer's note, in either
        // name field; and a list whose number only starts with ours is not it.
        let note = std::ffi::CString::new(format!("{} (debug layer indirect)", list_name(list.id)))
            .unwrap();
        node.pCommandListDebugNameW = windows::core::PCWSTR::null();
        node.pCommandListDebugNameA = note.as_ptr().cast();
        assert!(
            list.is_node(&node),
            "found by its ANSI name with the layer's note"
        );
        let longer =
            std::ffi::CString::new(format!("{}0 (debug layer indirect)", list_name(list.id)))
                .unwrap();
        node.pCommandListDebugNameA = longer.as_ptr().cast();
        assert!(!list.is_node(&node), "a number that only starts with ours is no match");
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    /// Swapped for a fresh one when the list is reset while it runs.
    allocator: Mutex<ID3D12CommandAllocator>,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Depth textures with a writable twin the list uses (`crate::twin`),
    /// by host copy: the side its first use needs and the side it leaves
    /// newer.
    twins: Mutex<std::collections::HashMap<usize, crate::twin::Use>>,
    /// A copy of ours replaced the list's compute root signature and
    /// pipeline: draws and dispatches set theirs again.
    rebind: std::sync::atomic::AtomicBool,
    /// The list that runs those transitions just before this one.
    prelude: Mutex<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: Mutex<Option<(ID3D12Resource, usize)>>,
    /// This recording's pooled buffer-views' written-pages bitmap, cloned into
    /// `Pending` on submission and replaced on reset while running.
    marks: Mutex<Option<std::sync::Arc<crate::shadow::MarkBitmap>>>,
    /// The host fence value of its last submission (0: none): it is not
    /// reset, resubmitted or dropped before that one has finished.
    last: AtomicU64,
    /// Up to which of its submissions the list's memory was handed to the
    /// timeline by a reset: its recording since uses none of theirs.
    fresh_after: AtomicU64,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// In a frame `crate::capture` lists: the textures the list renders to
    /// or writes, and the pipeline that first wrote each (diagnostics).
    #[cfg(feature = "diagnostics")]
    targets: Mutex<Vec<(usize, String)>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

impl Drop for HostList {
    /// Its allocators and upload memory may not go while the GPU runs it,
    /// and a submission in flight still names it.
    fn drop(&mut self) {
        self.settle();
        if let Some(bm) = self
            .marks
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            && let Ok(bm) = std::sync::Arc::try_unwrap(bm)
        {
            crate::shadow::return_mark_bitmap(bm);
        }
    }
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether a submission is left running when [`Host::execute`] returns
/// (`VICTORY_ASYNC_SUBMIT=1`): it is then finished (copied back, its
/// bottom-of-pipe writes and the title's fence signals behind it done) by
/// the completion thread, or by whichever thread first needs it. Off, the
/// default, each submission is waited for and finished on the submitting
/// thread before `execute` returns, as before submissions had a timeline:
/// the title's thread then never runs while what its GPU work wrote goes
/// back. Tests take the asynchronous path, which is the one with ordering
/// to check.
pub(crate) fn async_submit() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = cfg!(test) || crate::switch::on("VICTORY_ASYNC_SUBMIT");
        if on {
            tracing::info!("host submissions run asynchronously (VICTORY_ASYNC_SUBMIT)");
        }
        on
    })
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches, and it appends its own note to our
/// name ("victory host list 158 (debug layer indirect)"). The name matches
/// exactly or followed by " (": never as the start of another number.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p))
        || node_name.is_some_and(|n| {
            n == name
                || n.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(" ("))
        })
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<ScratchChunk>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;
const FIRST_SCRATCH_CHUNK: u64 = 64 << 10;

struct ScratchChunk {
    _buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: u64,
}

/// What a list reset while it ran leaves to its running submission.
struct Retired {
    allocator: ID3D12CommandAllocator,
    prelude: Prelude,
    deferred: Deferred,
    scratch: Scratch,
    states: States,
    twins: std::collections::HashMap<usize, crate::twin::Use>,
    budget: Option<(ID3D12Resource, usize)>,
}

// SAFETY: D3D12 objects are free-threaded; the scratch and budget pointers
// are persistent mappings of their buffers, which go with them.
unsafe impl Send for Retired {}

type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Allocators, preludes and scratch chunks whose submissions finished,
/// for lists reset while they run (dozens per world frame).
static SPARE_ALLOCATORS: Mutex<Vec<ID3D12CommandAllocator>> = Mutex::new(Vec::new());
static SPARE_PRELUDES: Mutex<Vec<Prelude>> = Mutex::new(Vec::new());
static SPARE_SCRATCH: Mutex<Vec<ScratchChunk>> = Mutex::new(Vec::new());
/// Spares kept of each kind at most.
const SPARES: usize = 256;

fn pool<T>(spares: &'static Mutex<Vec<T>>) -> std::sync::MutexGuard<'static, Vec<T>> {
    spares.lock().unwrap_or_else(|e| e.into_inner())
}

impl Retired {
    /// Once the submission that used them has finished.
    fn retire(self) {
        // SAFETY: nothing running uses the allocators any more.
        unsafe {
            if self.allocator.Reset().is_ok() && pool(&SPARE_ALLOCATORS).len() < SPARES {
                pool(&SPARE_ALLOCATORS).push(self.allocator);
            }
            if self.prelude.0.Reset().is_ok() && pool(&SPARE_PRELUDES).len() < SPARES {
                pool(&SPARE_PRELUDES).push(self.prelude);
            }
        }
        let mut spare = pool(&SPARE_SCRATCH);
        let room = SPARES.saturating_sub(spare.len());
        spare.extend(self.scratch.chunks.into_iter().take(room));
        drop((self.deferred, self.states, self.twins, self.budget));
    }
}

/// 64 KiB page-aligned span `[from, to)` covering `[address, address + len)`.
fn page_span(address: usize, len: usize) -> (usize, usize) {
    if len == 0 {
        return (address, address);
    }
    let page = crate::watch::PAGE;
    let from = address & !(page - 1);
    let to = address.saturating_add(len).next_multiple_of(page);
    (from, to)
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory: the range it writes, the range it
    /// reads (address, bytes).
    Run(Box<dyn Fn() + Send>, (usize, usize), (usize, usize)),
    /// Work that writes host memory only: the title memory it reads, and
    /// the list's own memory it writes (pointer, bytes).
    Host(Box<dyn Fn() + Send>, (usize, usize), (usize, usize)),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the 64 KiB pages they write (no snapshot then).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`), recording UAV shader writes when `writes` is true.
    Heap(std::sync::Arc<crate::bindless::Viewed>, bool),
    /// Shaders read (and when `writes` is true, may write) title memory
    /// through a view heap's page table (the heap's key, `crate::memory`):
    /// the table names every copy, each brought up to date.
    Memory(usize, bool),
    /// Indirect argument clamp mutates `[address, address + len)` in its
    /// host copy without copying back: serializes future refreshes there.
    Indirect(usize, usize),
}

/// What a repeat of a list in one submission writes into the list's own
/// memory, which its earlier run has not read yet: the bytes in an upload
/// buffer of the submission's, and the copies its prelude records (list
/// buffer, offset, offset in `staging`, bytes).
struct Repeat {
    staging: ID3D12Resource,
    copies: Vec<(ID3D12Resource, u64, u64, u64)>,
}

/// Dynamic descriptor table slots leased by a recording; released when the
/// recording drops (after its async submission completes if reset while
/// running).
struct DynamicLease(u64, u32, u32);

impl Drop for DynamicLease {
    fn drop(&mut self) {
        match (self.0, crate::memory::heap()) {
            (crate::memory::OURS_ID, Some(ours)) => ours.release(self.1, self.2),
            (crate::memory::OURS_ID, None) => {}
            _ => crate::descriptor::release_dynamic(self.0, self.1, self.2),
        }
    }
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    /// The list's upload buffers: buffer, mapped pointer, bytes.
    uploads: Vec<(ID3D12Resource, usize, usize)>,
    /// Texture staging buffers its commands copy from, not brought up to
    /// date by its steps (`crate::texture::upload_changed`).
    staged: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Dynamic descriptor ranges leased by this recording, retired with it
    /// if the list resets before its submission finishes.
    dynamic: Vec<DynamicLease>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Work done once the list has run, before the bottom-of-pipe writes
    /// (title memory shaders wrote at addresses they computed goes back).
    after: Vec<std::sync::Arc<dyn Fn() + Send + Sync>>,
    /// Asked once the list has run, before anything goes back: whether its
    /// shaders read title memory with no copy (`crate::memory`); then what
    /// they wrote through views does not go back either.
    missed: Vec<std::sync::Arc<dyn Fn() -> bool + Send + Sync>>,
}

/// Submissions in flight at most: a submission past them waits for the
/// oldest first.
const IN_FLIGHT: usize = 16;

/// A list a submission in flight names.
#[derive(Clone, Copy)]
struct ListRef(*const HostList);

// SAFETY: host lists are used from any thread (their state is behind
// locks), and one is not dropped while a submission naming it is in flight
// (`HostList::last`); a reset hands the memory such a submission uses to
// the timeline instead of reusing it.
unsafe impl Send for ListRef {}

impl ListRef {
    fn get<'a>(self) -> &'a HostList {
        // SAFETY: as above, the list outlives the submission.
        unsafe { &*self.0 }
    }
}

/// A range shaders write (host copy, address, bytes), the runs (by
/// index) that declared it, and the number of matching `Step::GpuWrite`
/// declarations in those runs (matching `Shadow::pending` increments).
type Back = (
    std::sync::Arc<crate::shadow::Shadow>,
    usize,
    usize,
    Vec<usize>,
    u32,
);

/// What is left of a submission once the GPU has run it.
struct Pending {
    lists: Vec<ListRef>,
    /// The ranges its shaders write (host copy, address, bytes), each once,
    /// and the runs that declared each.
    back: Vec<Back>,
    /// Pooled mark bitmaps used by this submission's runs and which runs
    /// wrote into each.
    marks: Vec<(std::sync::Arc<crate::shadow::MarkBitmap>, Vec<usize>)>,
    /// Its bottom-of-pipe writes, in list order: address, what, bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// The preludes of lists it names more than once, past their first:
    /// each recording has its own, back to the spares once it has run.
    preludes: Vec<Prelude>,
    submitted: std::time::Instant,
    /// Per run, asked once it has run: whether its shaders read title
    /// memory with no copy (then what they wrote does not go back).
    missed: Vec<Vec<std::sync::Arc<dyn Fn() -> bool + Send + Sync>>>,
    /// Work once it has run, before its bottom-of-pipe writes, in run
    /// order.
    after: Vec<std::sync::Arc<dyn Fn() + Send + Sync>>,
    /// What the repeats of lists it names more than once copy into the
    /// lists' memory, past their first (`Repeat`), kept until it has run.
    repeats: Vec<Repeat>,
    /// The list it ran alone in when the device was removed (diagnostics).
    removed_in: Option<usize>,
    /// Resources retired before it was submitted (`retire`).
    retired: Vec<ID3D12Resource>,
    /// Shadows retired before it was submitted (`retire_shadow`).
    retired_shadows: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// Mark-bit runs of shadows dropped before it was submitted.
    retired_marks: Vec<(u32, u32)>,
}

impl Pending {
    /// A submission's, after its steps ran, with the title memory its GPU
    /// work writes (copied back, written at the bottom of the pipe) and
    /// reads (host copies and textures brought up to date for it).
    fn of(runs: &[&HostList]) -> (Pending, Touches) {
        let mut back: Vec<Back> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> =
            std::collections::HashMap::new();
        let mut marks: Vec<(std::sync::Arc<crate::shadow::MarkBitmap>, Vec<usize>)> = Vec::new();
        let (mut writes, mut touches) = (Vec::new(), Touches::default());
        let (mut missed, mut after) = (Vec::with_capacity(runs.len()), Vec::new());
        // One list's lock at a time: a submission may name a list twice.
        for (run, list) in runs.iter().enumerate() {
            if let Some(bm) = list.marks.lock().unwrap_or_else(|e| e.into_inner()).clone() {
                if let Some((_, by)) = marks
                    .iter_mut()
                    .find(|(existing, _)| std::sync::Arc::ptr_eq(existing, &bm))
                {
                    if by.last() != Some(&run) {
                        by.push(run);
                    }
                } else {
                    marks.push((bm, vec![run]));
                }
            }
            let deferred = list.deferred();
            missed.push(deferred.missed.clone());
            after.extend(deferred.after.iter().cloned());
            for step in &deferred.copies {
                match step {
                    Step::GpuWrite(shadow, address, len, marked) => {
                        let copy = std::sync::Arc::as_ptr(shadow) as usize;
                        let (from, to) = if *marked {
                            page_span(*address, *len)
                        } else {
                            (*address, address.saturating_add(*len))
                        };
                        touches.shaders.push((copy, from, to));
                        match index.entry((copy, *address, *len)) {
                            std::collections::hash_map::Entry::Occupied(o) => {
                                let entry = &mut back[*o.get()];
                                entry.4 += 1;
                                if entry.3.last() != Some(&run) {
                                    entry.3.push(run);
                                }
                            }
                            std::collections::hash_map::Entry::Vacant(v) => {
                                v.insert(back.len());
                                back.push((shadow.clone(), *address, *len, vec![run], 1));
                            }
                        }
                    }
                    Step::Refresh(_, address, len) => touches.reads.push((*address, address + len)),
                    Step::Texture(staging) => {
                        touches.reads.push(staging.range());
                        touches
                            .staging
                            .insert(std::sync::Arc::as_ptr(staging) as usize);
                    }
                    Step::Indirect(address, len) => {
                        touches.labels.push((*address, address.saturating_add(*len)));
                    }
                    _ => {}
                }
            }
            for staging in &deferred.staged {
                touches.reads.push(staging.range());
                touches
                    .staging
                    .insert(std::sync::Arc::as_ptr(staging) as usize);
            }
            for &(at, what, bytes) in &deferred.writes {
                writes.push((at, what, bytes));
                touches.labels.push((at, at + bytes));
            }
        }
        let pending = Pending {
            lists: runs.iter().map(|&l| ListRef(l)).collect(),
            back,
            marks,
            writes,
            preludes: Vec::new(),
            submitted: std::time::Instant::now(),
            missed,
            after,
            repeats: Vec::new(),
            removed_in: None,
            retired: Vec::new(),
            retired_shadows: Vec::new(),
            retired_marks: Vec::new(),
        };
        (pending, touches)
    }
}

/// A thread's event for waiting on the host fence.
struct Event(HANDLE);

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: our own handle, no longer waited on.
        let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.0) };
    }
}

thread_local! {
    /// Null if it could not be made: the fence wait then blocks by itself.
    // SAFETY: plain event creation.
    static EVENT: Event = Event(unsafe { CreateEventW(None, false, false, None) }.unwrap_or_default());
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host runs 64-bit atomics on typed resources in the
    /// descriptor heap (translated 64-bit image atomics); logged once.
    pub(crate) fn heap_atomics64(&self) -> bool {
        static ATOMICS: OnceLock<bool> = OnceLock::new();
        *ATOMICS.get_or_init(|| {
            let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut typed).cast(),
                        size_of_val(&typed) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_D3D12_OPTIONS11,
                            (&raw mut heap).cast(),
                            size_of_val(&heap) as u32,
                        )
                        .is_ok()
            };
            let atomics = ok
                && typed.AtomicInt64OnTypedResourceSupported.as_bool()
                && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
            tracing::info!(atomics, "host 64-bit atomics on typed heap resources");
            atomics
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            Ok(HostList {
                allocator: Mutex::new(allocator),
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                prelude: Mutex::new((prelude_allocator, prelude)),
                pipelines: Mutex::default(),
                id,
                budget: Mutex::default(),
                marks: Mutex::default(),
                last: AtomicU64::new(0),
                fresh_after: AtomicU64::new(0),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// A command allocator nothing runs from: a spare, or a new one.
    fn fresh_allocator(&self) -> Option<ID3D12CommandAllocator> {
        let spare = pool(&SPARE_ALLOCATORS).pop();
        spare.or_else(|| {
            // SAFETY: plain object creation.
            let made = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            };
            made.ok()
        })
    }

    /// A closed prelude list and its allocator, nothing running from them.
    fn fresh_prelude(&self) -> Option<Prelude> {
        let spare = pool(&SPARE_PRELUDES).pop();
        if spare.is_some() {
            return spare;
        }
        let allocator = self.fresh_allocator()?;
        // SAFETY: plain object creation.
        unsafe {
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            Some((allocator, list))
        }
    }

    /// Runs closed lists. Their submission joins the timeline, finished
    /// once the GPU has run it: before this returns unless submissions run
    /// asynchronously ([`async_submit`]).
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        self.execute_inner(
            runs,
            #[cfg(test)]
            None,
            !async_submit(),
        );
    }

    /// Test helper: [`Host::execute`] as with `VICTORY_ASYNC_SUBMIT` off
    /// (tests otherwise take the asynchronous path).
    #[cfg(test)]
    fn execute_waited(&self, runs: &[&HostList]) {
        self.execute_inner(runs, None, true);
    }

    /// Test helper: arms `gate` on the host queue atomically inside the
    /// submission lock right before `ExecuteCommandLists` so other parallel
    /// tests cannot deadlock behind `gate` while holding `submitted`.
    #[cfg(test)]
    fn execute_gated(&self, runs: &[&HostList], gate: &ID3D12Fence) {
        self.execute_inner(runs, Some(gate), false);
    }

    /// With `wait`, the submission is finished before this returns, on this
    /// thread and under the submission lock.
    fn execute_inner(
        &self,
        runs: &[&HostList],
        #[cfg(test)] gate: Option<&ID3D12Fence>,
        wait: bool,
    ) {
        if runs.is_empty() {
            return;
        }
        // Waited for, every submission is finished by its own thread: no
        // completion thread runs (none is needed to move a fence on).
        if async_submit() {
            self.finisher.call_once(|| {
                let spawned = std::thread::Builder::new()
                    .name("victory-gpu-finisher".into())
                    .spawn(|| made().into_iter().for_each(Host::finish_loop));
                if let Err(e) = spawned {
                    tracing::warn!(error = %e, "no completion thread; submissions finish when needed");
                }
            });
        }
        // One submission at a time from preparation to submission: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // What has run is finished first. A list still in flight is waited
        // for: its upload memory, budget word and prelude are rewritten
        // below.
        self.finish();
        for list in runs {
            if list.running() {
                self.settle(list.last.load(Ordering::Acquire));
            }
        }
        if self.timeline.len() >= IN_FLIGHT
            && let Some((oldest, _)) = self.timeline.span()
        {
            self.settle(oldest);
        }
        // Descriptors the title copied into its heaps itself, before the
        // heaps' views are brought up to date below.
        crate::descriptor::sync_heaps();
        // Textures heap-indexing shaders may read, uploaded first by a list
        // of ours where their title memory changed.
        let heaps: Vec<std::sync::Arc<crate::bindless::Viewed>> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Heap(v, _) => Some(v.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        #[cfg(feature = "diagnostics")]
        let uploading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::UploadsUs);
        let uploads = (!heaps.is_empty())
            .then(|| crate::bindless::uploads(&heaps))
            .flatten();
        #[cfg(feature = "diagnostics")]
        drop(uploading);
        let runs: Vec<&HostList> = uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let runs = &runs[..];
        // A frame `crate::capture` lists: every command of every list, in
        // the order the GPU runs them (with `VICTORY_D3D12_DRED`, which
        // notes them).
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(feature = "diagnostics")]
        if traced {
            for (i, r) in runs.iter().enumerate() {
                let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (note, _)) in notes.iter().enumerate() {
                    tracing::info!(
                        target: "victory_gpu::trace",
                        "s{} l{i}/{} #{k}: {note}",
                        *submitted + 1,
                        runs.len()
                    );
                }
            }
        }
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch {
            host: Some(self),
            ..Batch::default()
        };
        // A list named again in this submission: its earlier prelude is
        // not yet run, so the repeat records into a spare one, and starts
        // a new `ExecuteCommandLists` call (the queue runs them in order, so
        // nothing waits). Where each such call starts in `lists`.
        let mut preludes = Vec::new();
        let mut repeats = Vec::new();
        let mut cuts = vec![0];
        let mut since_cut: Vec<*const HostList> = Vec::with_capacity(runs.len());
        for (i, list) in runs.iter().enumerate() {
            let id: *const HostList = *list;
            let repeated = runs[..i].iter().any(|&l| std::ptr::eq(l, id));
            if (repeated || since_cut.contains(&id)) && lists.len() > *cuts.last().unwrap_or(&0) {
                cuts.push(lists.len());
                since_cut.clear();
            }
            since_cut.push(id);
            // A repeat's steps would rewrite the list's own memory (upload
            // memory, scratch, budget word) before its earlier run has read
            // it: they write into memory of the submission instead, which
            // the repeat's prelude copies into the list's on the GPU, in
            // queue order.
            batch.redirect = repeated.then(Vec::new);
            list.run_deferred(&mut batch);
            let repeat = batch
                .redirect
                .take()
                .filter(|r| !r.is_empty())
                .and_then(|r| list.repeat(self, r));
            firsts.push(lists.len());
            #[cfg(feature = "diagnostics")]
            let _timed =
                crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            if let Some((prelude, spare)) = list.prelude(repeated.then_some(self), repeat.as_ref())
            {
                lists.push(Some(prelude));
                preludes.extend(spare);
            }
            repeats.extend(repeat);
            lists.push(list.list.cast().ok());
            // Keep any list that repeats later in this batch at the end of its
            // ExecuteCommandLists group so queuing the repeat's submission does
            // not disturb the command buffer chaining tail of an earlier group.
            if runs[i + 1..].iter().any(|&l| std::ptr::eq(l, id)) {
                cuts.push(lists.len());
                since_cut.clear();
            }
        }
        batch.flush();
        let (mut pending, mut touches) = Pending::of(runs);
        pending.preludes = preludes;
        pending.repeats = repeats;
        // What steps brought up to date through view heaps and page tables.
        touches.reads.append(&mut batch.reads);
        touches.shaders.append(&mut batch.shaders);
        touches.tables.extend(batch.tables.drain());
        let prepared = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsPrep, crate::copystat::page_faults().saturating_sub(faults));
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        cuts.push(lists.len());
        // Diagnostics: each list on its own, timed and waited for
        // (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let timed = timing();
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timed {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            cuts.windows(2).map(|c| &lists[c[0]..c[1]]).collect()
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups: Vec<&[Option<ID3D12CommandList>]> =
            cuts.windows(2).map(|c| &lists[c[0]..c[1]]).collect();
        // Each list ran alone and was waited for: a removal is its.
        #[cfg(feature = "diagnostics")]
        let per_list = timed;
        #[cfg(not(feature = "diagnostics"))]
        let per_list = false;
        let mut removed_in = None;
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::SUBMITTED, *submitted, 0);
            // SAFETY: closed lists, not running (waited for above); our fence.
            let signalled = unsafe {
                #[cfg(test)]
                if i == 0
                    && let Some(gate) = gate
                {
                    let _ = self.queue.Wait(gate, 1);
                }
                if !group.is_empty() {
                    self.queue.ExecuteCommandLists(group);
                }
                self.queue.Signal(&self.fence, *submitted)
            };
            if let Err(e) = signalled {
                // Nothing would finish it.
                fail(format_args!(
                    "a host submission of {} lists was not signalled ({e}){}",
                    runs.len(),
                    self.removal_data(runs)
                ));
            }
            #[cfg(feature = "diagnostics")]
            if timed {
                self.wait_gpu(*submitted, true);
                crate::resource::life::note(0, crate::resource::life::RAN, *submitted, 0);
                if started.elapsed() > std::time::Duration::from_millis(50) {
                    // SAFETY: a plain query.
                    let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                    let commands = runs[i]
                        .notes
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .len();
                    tracing::warn!(
                        ms = started.elapsed().as_millis(),
                        commands,
                        "slow host list"
                    );
                    for (ms, op) in runs[i].slowest(frequency, 6) {
                        tracing::warn!("slow list command, {ms:.1} ms: {op}");
                    }
                }
            }
            if per_list
                && removed_in.is_none()
                // SAFETY: a plain query.
                && unsafe { self.device.GetDeviceRemovedReason() }.is_err()
            {
                removed_in = Some(i);
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        for list in runs {
            list.last.store(*submitted, Ordering::Release);
        }
        pending.removed_in = removed_in;
        // Resources, shadows, and mark-bit runs retired since the last
        // submission go once this one (and everything before it) has run.
        pending.retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
        pending.retired_shadows =
            std::mem::take(&mut *RETIRED_SHADOWS.lock().unwrap_or_else(|e| e.into_inner()));
        pending.retired_marks = crate::shadow::take_retired_marks();
        self.timeline.push(*submitted, touches, pending);
        if wait {
            // As before submissions had a timeline: what its shaders wrote
            // is back, its bottom-of-pipe writes and the title's signals
            // behind it are done, before the title's thread runs on.
            self.settle(*submitted);
        }
        #[cfg(feature = "diagnostics")]
        if timed {
            self.settle(*submitted);
        }
        // A listed frame: what this submission's lists drew, as it is once
        // it has run (the dumps submit lists of their own).
        #[cfg(feature = "diagnostics")]
        if traced {
            let sub = *submitted;
            drop(submitted);
            self.settle(sub);
            let mut targets: Vec<(usize, String)> = Vec::new();
            for r in runs {
                for t in r.targets.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    if !targets.iter().any(|x| x.0 == t.0) {
                        targets.push(t.clone());
                    }
                }
            }
            crate::capture::dump_submission(sub, &targets);
        }
    }

    /// Finishes, in order, the submissions the GPU has run (no waiting).
    fn finish(&self) {
        // SAFETY: a plain query (all ones once the device is removed).
        let reached = unsafe { self.fence.GetCompletedValue() };
        self.timeline.finish(reached, |p| self.complete(p));
    }

    /// Finishes every submission up to the one that signals `value`,
    /// waiting for the GPU to run them.
    pub(crate) fn settle(&self, value: u64) {
        if self
            .timeline
            .span()
            .is_some_and(|(oldest, _)| oldest <= value)
        {
            self.wait_gpu(value, true);
            self.finish();
        }
    }

    /// Finishes every submission made so far.
    pub(crate) fn drain(&self) {
        loop {
            if let Some((_, newest)) = self.timeline.span() {
                self.settle(newest);
            }
            let _submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
            if self.timeline.span().is_none() {
                release_retired();
                return;
            }
        }
    }

    /// Before a submission's step reads title memory `[address, address +
    /// len)` (or, with `writing`, writes it): finishes the submissions in
    /// flight whose GPU work writes it (or reads it), so that the step sees
    /// title memory and host copies as if they had run when submitted.
    pub(crate) fn settle_range(&self, address: usize, len: usize, writing: bool) {
        if let Some(value) = self
            .timeline
            .conflict(address, address.saturating_add(len), writing)
        {
            self.settle(value);
        }
    }

    /// Before a texture staging buffer (by address) is rewritten: finishes
    /// the submissions in flight that copy from it.
    pub(crate) fn settle_staging(&self, staging: usize) {
        if let Some(value) = self.timeline.staging_conflict(staging) {
            self.settle(value);
        }
    }

    /// Before view heap `key`'s page table is rewritten: finishes the
    /// submissions in flight whose shaders read it.
    fn settle_table(&self, key: usize) {
        if let Some(value) = self.timeline.table_conflict(key) {
            self.settle(value);
        }
    }

    /// Runs `signal` once every submission made so far has finished.
    pub(crate) fn after(&self, signal: impl FnOnce() + Send + 'static) {
        self.timeline.after(signal);
    }

    /// Blocks until the GPU has run what signals `value`; the time counts
    /// as waiting for the GPU if `counted`.
    fn wait_gpu(&self, value: u64, counted: bool) {
        // SAFETY: plain calls on our fence and this thread's event.
        unsafe {
            if self.fence.GetCompletedValue() >= value {
                return;
            }
            let started = std::time::Instant::now();
            let waited = EVENT.with(|e| {
                if e.0.is_invalid() {
                    return Err(windows::core::Error::empty());
                }
                self.fence.SetEventOnCompletion(value, e.0)?;
                if WaitForSingleObject(e.0, INFINITE) != WAIT_OBJECT_0 {
                    return Err(windows::core::Error::from_thread());
                }
                Ok(())
            });
            if let Err(e) = waited {
                tracing::warn!(error = %e, "host fence wait failed; polling");
                while self.fence.GetCompletedValue() < value {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            if counted {
                SUBMISSION_US[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }

    /// The completion thread: finishes each submission as soon as the GPU
    /// has run it, so that title fences and bottom-of-pipe writes move on
    /// their own (the title waits for its events and reads its memory).
    fn finish_loop(&self) {
        loop {
            let value = self.timeline.next();
            self.wait_gpu(value, false);
            self.finish();
        }
    }

    /// Once the GPU has run a submission: what its shaders wrote goes back
    /// into title memory, then its bottom-of-pipe writes, which tell the
    /// title the GPU is done.
    fn complete(&self, p: Pending) {
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            // Any submission in flight may have faulted.
            let mut runs: Vec<&HostList> = p.lists.iter().map(|l| l.get()).collect();
            self.timeline
                .each_work(|w| runs.extend(w.lists.iter().map(|l| l.get())));
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                p.lists.len(),
                self.removal_data(&runs),
                last_notes(&runs),
                removed_during(&runs, p.removed_in)
            ));
        }
        // About how long it ran: from its submission or the end of the one
        // before, whichever is later.
        static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let now = std::time::Instant::now();
        let ran = {
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            let from = last.map_or(p.submitted, |l| l.max(p.submitted));
            *last = Some(now);
            now - from
        };
        // The host resets a device whose submission runs for about 2 s.
        if ran > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = p
                .lists
                .iter()
                .flat_map(|l| {
                    l.get()
                        .pipelines
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone()
                })
                .collect();
            tracing::warn!(
                ms = ran.as_millis(),
                lists = p.lists.len(),
                ?pipelines,
                "slow host submission"
            );
        }
        // What shaders wrote goes back before the bottom-of-pipe writes;
        // each range once per submission, however many dispatches declared
        // it (the world's passes declare the same big ranges dozens of
        // times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // Shaders that read memory with no copy computed something else
        // than the console's: what a list's shaders wrote then stays out of
        // title memory, where it broke the title's own data (runs c1, c4).
        let missed: Vec<bool> = p
            .missed
            .iter()
            .map(|checks| checks.iter().any(|check| check()))
            .collect();
        // Pages shaders marked first: the copy-backs below disarm the copies.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = p
            .back
            .iter()
            .filter(|(_, _, _, by, _)| !by.iter().all(|&run| !missed[run]))
            .map(|(s, a, n, _, _)| (s.clone(), *a, *n))
            .collect();
        let bitmaps: Vec<(&crate::shadow::MarkBitmap, bool)> = p
            .marks
            .iter()
            .map(|(bm, by)| (bm.as_ref(), by.iter().any(|&run| missed[run])))
            .collect();
        let can_sync = |from: usize, to: usize| !self.timeline.has_subsequent_reader(from, to);
        crate::shadow::copy_back_marks(&bitmaps, &withheld, can_sync);
        let mut copies: Vec<&std::sync::Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, address, len, by, count) in &p.back {
            if by.iter().all(|&run| !missed[run]) {
                let written = shadow.copy_back(*address, *len, *count, can_sync);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            } else {
                shadow.keep_writes(*address, *len, *count);
                crate::memory::note_withheld(*len as u64);
            }
            if !copies.iter().any(|c| std::sync::Arc::ptr_eq(c, shadow)) {
                copies.push(shadow);
            }
        }
        copies.iter().for_each(|c| c.submission_done());
        for work in &p.after {
            work();
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        for &(at, what, bytes) in &p.writes {
            what.write(at as *mut u8, bytes);
        }
        drop(p.repeats);
        for prelude in p.preludes {
            // SAFETY: the submission that ran from it has finished.
            if unsafe { prelude.0.Reset() }.is_ok() && pool(&SPARE_PRELUDES).len() < SPARES {
                pool(&SPARE_PRELUDES).push(prelude);
            }
        }
        // It has run, and every submission before it: what was retired
        // before it was submitted goes, and pooled mark bitmaps whose list
        // was reset return cleanly after clearing.
        drop(p.retired);
        crate::shadow::release_marks(p.retired_marks);
        drop(p.retired_shadows);
        for (bm, _) in p.marks {
            if let Ok(bm) = std::sync::Arc::try_unwrap(bm) {
                crate::shadow::return_mark_bitmap(bm);
            }
        }
        self.log_messages();
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                    #[cfg(feature = "diagnostics")]
                    for name in names.iter().filter(|_| what == "inside freed") {
                        text.push_str(&crate::resource::life::of(name));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // The nodes, oldest list first, and which run each is.
            let mut nodes = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                nodes.push(p);
                node = p.pNext;
            }
            let index: Vec<Option<usize>> = nodes
                .iter()
                .map(|p| runs.iter().position(|r| r.is_node(p)))
                .collect();
            if let Some(last) = nodes.last()
                && index.iter().all(Option::is_none)
            {
                let name = |w: windows::core::PCWSTR, a: *const u8| {
                    let w = (!w.is_null()).then(|| w.to_string().unwrap_or_default());
                    let a = (!a.is_null()).then(|| {
                        std::ffi::CStr::from_ptr(a.cast())
                            .to_string_lossy()
                            .into_owned()
                    });
                    format!("{w:?}/{a:?}")
                };
                tracing::warn!(
                    nodes = nodes.len(),
                    last_pointer = ?last.pCommandList.as_ref().map(|l| l.as_raw()),
                    last_names = %name(last.pCommandListDebugNameW, last.pCommandListDebugNameA),
                    "no breadcrumb node named one of our lists"
                );
            }
            // Every list's progress (done/count), in the runtime's order.
            let progress: Vec<String> = nodes
                .iter()
                .zip(&index)
                .map(|(p, index)| {
                    let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                    format!("{done}/{} {index:?}", p.BreadcrumbCount)
                })
                .collect();
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            for (p, index) in nodes.iter().zip(&index) {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.map(|i| runs[i]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A slot of CPU-only views, valid until it is recorded.
    pub(crate) fn cpu_view(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_cpu_view.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a live heap.
        let mut handle = unsafe { self.cpu_views.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += (slot * self.cpu_view_size) as usize;
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, wide, ansi) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
                (!node.pCommandListDebugNameA.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(node.pCommandListDebugNameA.cast())
                        .to_string_lossy()
                        .into_owned()
                }),
            )
        };
        let (pointers, name) = (identities(&self.list), list_name(self.id));
        node_is_list(pointer, wide.as_deref(), &pointers, &name)
            || node_is_list(None, ansi.as_deref(), &pointers, &name)
    }

    pub(crate) fn close(&self) {
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Waits for the list's last submission and finishes it (and those
    /// before it).
    pub(crate) fn settle(&self) {
        let last = self.last.load(Ordering::Acquire);
        if last != 0
            && let Some(host) = made()
        {
            host.settle(last);
        }
    }

    /// Whether the list's last submission, with the memory the list
    /// records into now, has not finished yet.
    pub(crate) fn running(&self) -> bool {
        let last = self.last.load(Ordering::Acquire);
        last > self.fresh_after.load(Ordering::Acquire)
            && made().is_some_and(|h| h.timeline.span().is_some_and(|(oldest, _)| oldest <= last))
    }

    /// Opens the list for recording again. D3D12 lets a list be reset as
    /// soon as it is submitted (titles record the next frame into it while
    /// the GPU runs this one): what its running submission uses (allocator,
    /// prelude, upload and scratch memory, the objects and textures it
    /// keeps, its budget word) then goes to the timeline, freed or reused
    /// once that submission has finished, and the list records into fresh
    /// ones.
    pub(crate) fn reset(&self) {
        let running = if self.running() { made() } else { None };
        let allocator = {
            let mut allocator = self.allocator.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(host) = running {
                let (Some(fresh), Some(prelude)) = (host.fresh_allocator(), host.fresh_prelude())
                else {
                    fail(format_args!(
                        "no fresh host command allocator for a running list"
                    ));
                };
                let mut old_prelude = self.prelude.lock().unwrap_or_else(|e| e.into_inner());
                let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
                let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
                let mut budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());
                let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
                let retired = Retired {
                    allocator: std::mem::replace(&mut *allocator, fresh),
                    prelude: std::mem::replace(&mut *old_prelude, prelude),
                    deferred: std::mem::take(&mut *self.deferred()),
                    scratch: std::mem::take(&mut *scratch),
                    states: std::mem::take(&mut *states),
                    twins: std::mem::take(&mut *twins),
                    budget: budget.take(),
                };
                // Running submission's `Pending` already owns an `Arc` clone
                // of this recording's mark bitmap; drop our handle so a new
                // recording allocates an isolated bitmap.
                drop(self.marks.lock().unwrap_or_else(|e| e.into_inner()).take());
                host.after(move || retired.retire());
                // Still named by that submission: dropping waits for it.
                self.fresh_after
                    .store(self.last.load(Ordering::Acquire), Ordering::Release);
            } else {
                *self.deferred() = Deferred::default();
                if let Some(bm) = self.marks.lock().unwrap_or_else(|e| e.into_inner()).take()
                    && let Ok(bm) = std::sync::Arc::try_unwrap(bm)
                {
                    crate::shadow::return_mark_bitmap(bm);
                }
                self.states
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
                self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
                let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
                (scratch.current, scratch.used) = (0, 0);
                // SAFETY: nothing running uses the allocator.
                if let Err(e) = unsafe { allocator.Reset() } {
                    fail(format_args!("a host command allocator did not reset ({e})"));
                }
            }
            allocator.clone()
        };
        self.pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
        // SAFETY: a closed list (D3D12 allows resetting one that still
        // runs) and an allocator nothing running uses.
        if let Err(e) = unsafe { self.list.Reset(&allocator, None) } {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    /// A depth texture with a writable twin (`crate::twin`) has the side
    /// the use needs brought up to date first; unordered access moves the
    /// twin, not the depth texture (nothing returned).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        if let Some(Some(twin)) = copy.twin.get()
            && crate::twin::route(self, &copy, twin, wanted)
        {
            return None;
        }
        let wanted = legal_state(texture, wanted);
        self.track(&copy, wanted).map(|before| (texture, before))
    }

    /// The list's state for host texture `copy` becomes `wanted` (a legal
    /// one): the state it leaves, if that is a change. Nothing for the
    /// list's first use (its prelude moves the texture there).
    fn track(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// [`HostList::transition`] for a host texture of ours (a twin's
    /// plane), without twins: the barrier to record, if one is needed.
    pub(crate) fn move_texture(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<D3D12_RESOURCE_BARRIER> {
        let wanted = legal_state(&copy.texture, wanted);
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// A use of the depth texture `depth` (with a twin) needing `side`
    /// (`crate::twin::DEPTH` or `TWIN`): whether a copy must be recorded
    /// now (true: into the depth texture). The list's first use is the
    /// prelude's to prepare.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
    ) -> Option<bool> {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(crate::twin::Use::new(depth, side, writes));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => o.get_mut().again(side, writes),
        }
    }

    /// A copy of ours replaced the compute root signature and pipeline.
    pub(crate) fn need_rebind(&self) {
        self.rebind.store(true, Ordering::Relaxed);
    }

    /// Whether the list's compute root signature and pipeline must be set
    /// again (once).
    pub(crate) fn take_rebind(&self) -> bool {
        self.rebind.swap(false, Ordering::Relaxed)
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. With `repeated` (the list's
    /// earlier prelude in the same submission has not run yet), it records
    /// into a spare prelude from `repeated`, returned for the submission to
    /// keep until it has run.
    /// A repeat's `repeat` bytes go into the list's memory first.
    fn prelude(
        &self,
        repeated: Option<&Host>,
        repeat: Option<&Repeat>,
    ) -> Option<(ID3D12CommandList, Option<Prelude>)> {
        // Depth textures and their twins first, from their real states and
        // back: the side each first use needs, made newer.
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let copies = crate::twin::at_submission(twins.values());
        drop(twins);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            })
            .collect();
        if barriers.is_empty() && copies.is_empty() && repeat.is_none() {
            return None;
        }
        let record = |(allocator, list): &Prelude| {
            // SAFETY: nothing pending or running uses the prelude (`execute`
            // waits for the list's last submission first, and a repeat in
            // one submission records into a spare); the textures and the
            // repeat's buffers are live.
            let recorded = unsafe {
                allocator.Reset().and_then(|()| {
                    list.Reset(allocator, None)?;
                    if let Some(repeat) = repeat {
                        // The list's buffers start the call in the common
                        // state (buffers decay at the end of each), go to
                        // copy destination for these copies and back.
                        let mut written: Vec<&ID3D12Resource> = Vec::new();
                        for (buffer, offset, from, n) in &repeat.copies {
                            list.CopyBufferRegion(buffer, *offset, &repeat.staging, *from, *n);
                            if !written.iter().any(|b| b.as_raw() == buffer.as_raw()) {
                                written.push(buffer);
                            }
                        }
                        let back: Vec<D3D12_RESOURCE_BARRIER> = written
                            .into_iter()
                            .map(|b| transition(b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON))
                            .collect();
                        list.ResourceBarrier(&back);
                    }
                    for (depth, to_depth) in &copies {
                        let Some(Some(twin)) = depth.twin.get() else { continue };
                        // From each texture's real state to the copy's and back.
                        let mut back = Vec::new();
                        twin.copy(list, depth, *to_depth, &mut |t, wanted| {
                            let real = t.state.load(Ordering::Relaxed);
                            let wanted = legal_state(&t.texture, wanted);
                            (real != wanted).then(|| {
                                back.push(transition(
                                    &t.texture,
                                    D3D12_RESOURCE_STATES(wanted as i32),
                                    D3D12_RESOURCE_STATES(real as i32),
                                ));
                                transition(&t.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(wanted as i32))
                            })
                        });
                        if !back.is_empty() {
                            list.ResourceBarrier(&back);
                        }
                    }
                    if !barriers.is_empty() {
                        list.ResourceBarrier(&barriers);
                    }
                    list.Close()
                })
            };
            if let Err(e) = recorded {
                fail(format_args!("a host state prelude did not record ({e})"));
            }
            list.cast::<ID3D12CommandList>().ok()
        };
        match repeated {
            Some(host) => {
                let Some(spare) = host.fresh_prelude() else {
                    fail(format_args!("no fresh host prelude for a repeated list"));
                };
                Some((record(&spare)?, Some(spare)))
            }
            None => {
                let prelude = self.prelude.lock().unwrap_or_else(|e| e.into_inner());
                Some((record(&prelude)?, None))
            }
        }
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            traced(&what);
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a texture the list renders to or writes, in a listed frame.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, resource: usize, by: &str) {
        let mut targets = self.targets.lock().unwrap_or_else(|e| e.into_inner());
        if !targets.iter().any(|t| t.0 == resource) {
            targets.push((resource, by.to_owned()));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (`marked`: the pages they mark).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len, marked));
    }

    /// Shaders of the list take views from the heap itself
    /// (`crate::bindless`): what the heap's views reach is brought up to
    /// date at submission (once until a title copy intervenes).
    pub(crate) fn defer_heap(
        &self,
        viewed: &std::sync::Arc<crate::bindless::Viewed>,
        writes: bool,
    ) {
        let mut deferred = self.deferred();
        let key = (
            std::sync::Arc::as_ptr(viewed) as usize,
            usize::MAX,
            usize::from(writes),
        );
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Heap(viewed.clone(), writes));
        }
    }

    /// Shaders of the list read (and when `writes` is true, may write) title
    /// memory through the page table of view heap `heap`
    /// (`crate::memory::prepare` at submission).
    pub(crate) fn defer_memory(&self, heap: usize, writes: bool) {
        let mut deferred = self.deferred();
        let key = (heap, usize::MAX - 1, usize::from(writes));
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Memory(heap, writes));
        }
    }

    /// Indirect argument clamp mutates `[address, address + len)` of its host
    /// copy on the GPU: serializes future refreshes over that range until
    /// this submission finishes.
    pub(crate) fn defer_indirect(&self, address: usize, len: usize) {
        self.deferred()
            .copies
            .push(Step::Indirect(address, len));
    }

    /// After the list has run, before its writes go back: `check` says
    /// whether its shaders read title memory with no copy.
    pub(crate) fn defer_missed(&self, check: impl Fn() -> bool + Send + Sync + 'static) {
        self.deferred().missed.push(std::sync::Arc::new(check));
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes.
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + Sync + 'static) {
        self.deferred().after.push(std::sync::Arc::new(work));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.list_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred()
            .uploads
            .push((buffer.clone(), data as usize, size as usize));
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.list_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred()
                .uploads
                .push((buffer, data as usize, len as usize));
            return Some((gpu, data));
        }
        while let Some(chunk) = s.chunks.get(s.current) {
            if len <= chunk.capacity - s.used {
                break;
            }
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let mut spares = pool(&SPARE_SCRATCH);
            let spare = spares
                .iter()
                .rposition(|chunk| chunk.capacity >= len)
                .map(|i| spares.swap_remove(i));
            drop(spares);
            let chunk = match spare {
                Some(chunk) => chunk,
                None => {
                    let next = s.chunks.last().map_or(FIRST_SCRATCH_CHUNK, |c| 2 * c.capacity);
                    let capacity = len.next_multiple_of(FIRST_SCRATCH_CHUNK).max(next).min(SCRATCH_CHUNK);
                    let (buffer, data) = host.list_buffer(capacity)?;
                    name(&buffer, || format!("list scratch {capacity:#x}"));
                    // SAFETY: a live buffer.
                    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                    #[cfg(feature = "diagnostics")]
                    tracing::debug!(target: "victory_gpu::scratch", list = ?std::ptr::from_ref(self),
                        chunk = s.current, requested = len, capacity, "allocated list scratch");
                    ScratchChunk { _buffer: buffer, mapped: data as usize, gpu, capacity }
                }
            };
            s.chunks.push(chunk);
            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used;
        let result = (chunk.gpu + at, (chunk.mapped + at as usize) as *mut u8);
        s.used += len;
        Some(result)
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = {
            let mut budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());
            if budget.is_none() {
                let (b, d) = host()?.shared_buffer(256)?;
                *budget = Some((b, d as usize));
            }
            budget.clone()?
        };
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred.copies.push(Step::Host(
                Box::new(move || unsafe { (at as *mut u32).write_volatile(0) }),
                (0, 0),
                (at, 4),
            ));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// Host GPU virtual address of this list's pooled buffer-views'
    /// written-pages bitmap (`crate::shadow::copy_back_marks`), allocated on
    /// first marked draw after creation or reset.
    pub(crate) fn view_writes(&self) -> Option<u64> {
        let mut marks = self.marks.lock().unwrap_or_else(|e| e.into_inner());
        if marks.is_none() {
            *marks = crate::shadow::take_mark_bitmap().map(std::sync::Arc::new);
        }
        marks.as_ref().map(|b| b.gpu())
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Keeps dynamic descriptor slots `[first, first + n)` on title heap
    /// `heap_id` leased until this recording's submission finishes (or the
    /// list resets while idle).
    pub(crate) fn keep_dynamic(&self, heap_id: u64, first: u32, n: u32) {
        self.deferred()
            .dynamic
            .push(DynamicLease(heap_id, first, n));
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again): the
    /// range `writes` (address, bytes); it reads `reads`.
    pub(crate) fn defer(
        &self,
        writes: (usize, usize),
        reads: (usize, usize),
        work: impl Fn() + Send + 'static,
    ) {
        self.deferred()
            .copies
            .push(Step::Run(Box::new(work), writes, reads));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory: it
    /// reads the title's `reads` and writes `writes` (pointer, bytes) of the
    /// list's own memory (upload or scratch), if any.
    pub(crate) fn defer_host(
        &self,
        reads: (usize, usize),
        writes: (usize, usize),
        work: impl Fn() + Send + 'static,
    ) {
        self.deferred()
            .copies
            .push(Step::Host(Box::new(work), reads, writes));
    }

    /// The list's commands copy from `staging`, brought up to date for them
    /// outside its steps.
    pub(crate) fn read_staging(&self, staging: &std::sync::Arc<crate::texture::Staging>) {
        self.deferred().staged.push(staging.clone());
    }

    /// For a repeat of the list in one submission: the bytes its steps
    /// would have written into the list's memory (pointer, bytes) put in an
    /// upload buffer of its own, with the copies into the list's buffers
    /// the repeat's prelude records. Memory not the list's is written at
    /// once, as before.
    fn repeat(&self, host: &Host, writes: Vec<(usize, Vec<u8>)>) -> Option<Repeat> {
        let total: usize = writes.iter().map(|(_, b)| b.len()).sum();
        let (staging, data) = host.upload_buffer(total as u64)?;
        name(&staging, || "list repeat".into());
        let mut copies = Vec::with_capacity(writes.len());
        let mut at = 0;
        let deferred = self.deferred();
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());
        for (dst, bytes) in writes {
            let n = bytes.len();
            let inside = |(buffer, start, len): (&ID3D12Resource, usize, usize)| {
                (start <= dst && dst + n <= start + len)
                    .then(|| (buffer.clone(), (dst - start) as u64))
            };
            let found = deferred
                .uploads
                .iter()
                .map(|(b, s, l)| (b, *s, *l))
                .chain(
                    scratch
                        .chunks
                        .iter()
                        .map(|c| (&c._buffer, c.mapped, c.capacity as usize)),
                )
                .chain(budget.iter().map(|(b, s)| (b, *s, 256)))
                .find_map(inside);
            match found {
                Some((buffer, offset)) => {
                    // SAFETY: inside our mapped buffer of `total` bytes.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.add(at), n) };
                    copies.push((buffer, offset, at as u64, n as u64));
                    at += n;
                }
                None => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(
                            "a repeated list's step writes memory not its own: written at once"
                        )
                    });
                    // SAFETY: what the step would have written itself.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, n) };
                }
            }
        }
        (!copies.is_empty()).then_some(Repeat { staging, copies })
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(
                started,
                match step {
                    Step::Refresh(..) => C::StepRefreshUs,
                    Step::Copy(..) | Step::Fill(..) => C::StepCopyUs,
                    Step::Run(..) | Step::Host(..) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(_) => C::StepTextureUs,
                    Step::Heap(..) => C::StepHeapUs,
                    Step::Memory(..) => C::StepMemoryUs,
                    Step::Indirect(..) => C::StepRefreshUs,
                },
            );
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.settle(src, len, false);
                    batch.settle(dst, len, true);
                    batch.title_written();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.settle(src, len, false);
                    match &mut batch.redirect {
                        Some(redirect) => {
                            let mut bytes = vec![0; len];
                            copy_mapped(bytes.as_mut_ptr(), src as *const u8, len, false);
                            redirect.push((dst, bytes));
                        }
                        None => copy_mapped(dst as *mut u8, src as *const u8, len, false),
                    }
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    // Title memory an earlier submission's shaders write is
                    // current only once that one is copied back, if the
                    // title wrote it since the copy took it.
                    batch.settle_refresh(shadow, *address, *len);
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work, _, _) | Step::Host(work, _, _) => {
                    let mut own = (0, 0);
                    match *step {
                        Step::Run(_, (to, written), (from, read)) => {
                            batch.settle(from, read, false);
                            batch.settle(to, written, true);
                            batch.title_written();
                        }
                        Step::Host(_, (from, read), writes) => {
                            batch.settle(from, read, false);
                            own = writes;
                        }
                        _ => {}
                    }
                    match &mut batch.redirect {
                        // A repeat: what the work writes into the list's
                        // memory goes to the repeat's, the list's left as
                        // its earlier run reads it.
                        Some(redirect) if own.1 > 0 => {
                            let (at, n) = own;
                            // SAFETY: the list's own mapped memory, which
                            // nothing running uses (its earlier run in this
                            // submission has not started).
                            let mine = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, n) };
                            let kept = mine.to_vec();
                            work();
                            redirect.push((at, mine.to_vec()));
                            mine.copy_from_slice(&kept);
                        }
                        _ => work(),
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked) => {
                    let (from, to) = if *marked {
                        page_span(*address, *len)
                    } else {
                        (*address, address.saturating_add(*len))
                    };
                    batch.settle(from, to - from, false);
                    if *marked {
                        batch.armed.push((shadow.clone(), *address, *len));
                    } else {
                        batch.flush();
                        shadow.writing(*address, *len, false);
                    }
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        let (from, to) = staging.range();
                        batch.settle(from, to - from, false);
                        staging.refresh(|| batch.settle_staging(staging));
                    }
                    continue;
                }
                Step::Heap(viewed, writes) => {
                    crate::bindless::refresh_kinds(viewed, |s, a, l, uav| {
                        batch.settle_refresh(s, a, l);
                        batch.add(s, a, l);
                        batch.reads.push((a, a + l));
                        if *writes && uav {
                            batch
                                .shaders
                                .push((std::sync::Arc::as_ptr(s) as usize, a, a + l));
                        }
                    });
                    continue;
                }
                &Step::Memory(heap, writes) => {
                    // shortcut: shaders writing title memory through the page
                    // table may write any copy it names (the pages they wrote
                    // are known only once they have run): the submission
                    // counts as writing them all when `writes` is true.
                    let host = batch.host;
                    crate::memory::prepare(
                        heap,
                        || {
                            if let Some(host) = host {
                                host.settle_table(heap);
                            }
                        },
                        |s, a, l| {
                            batch.settle_refresh(s, a, l);
                            batch.add(s, a, l);
                            batch.reads.push((a, a + l));
                            if writes {
                                batch
                                    .shaders
                                    .push((std::sync::Arc::as_ptr(s) as usize, a, a + l));
                            }
                        },
                    );
                    batch.tables.insert(heap);
                    continue;
                }
                Step::Indirect(..) => continue,
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
///
/// A copy's range brought up to date is not copied again in the submission
/// until a step writes title memory: every list of a world submission using
/// title memory asks for every copy whole. What the title's threads write
/// meanwhile is for later submissions (on the console it would race the
/// GPU).
#[derive(Default)]
struct Batch<'a> {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy's refresh is in `refreshes`, by copy.
    pending: std::collections::HashMap<usize, usize>,
    /// The range of each copy brought up to date since the last title
    /// write, by copy (held, so that its address names no other).
    current: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
    /// The host, whose earlier submissions may still be in flight.
    host: Option<&'a Host>,
    /// For a repeat of a list in the submission: what its steps write into
    /// the list's own memory (pointer, bytes), for its prelude to copy.
    redirect: Option<Vec<(usize, Vec<u8>)>>,
    /// Ranges brought up to date through view heaps (the GPU reads them),
    /// and copies a page table names (its shaders may write them).
    reads: Vec<(usize, usize)>,
    shaders: Vec<(usize, usize, usize)>,
    /// Marked `GpuWrite` ranges queued during this batch and armed once
    /// pending refreshes finish so sub-page arming does not split pages.
    armed: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Page tables (by view heap key) the submission's shaders read.
    tables: std::collections::HashSet<usize>,
}

impl Batch<'_> {
    /// Before detiling new bytes into a shared texture staging buffer: wait
    /// for any in-flight submission whose GPU copy still reads that staging
    /// buffer.
    fn settle_staging(&self, staging: &crate::texture::Staging) {
        if let Some(host) = self.host
            && let Some(value) = host
                .timeline
                .staging_conflict(std::ptr::from_ref(staging) as usize)
        {
            host.settle(value);
        }
    }

    /// Before a step reads (or, with `writing`, writes) title memory
    /// `[address, address + len)`: earlier submissions whose GPU work
    /// writes (or reads) it are finished first.
    fn settle(&self, address: usize, len: usize, writing: bool) {
        if let Some(host) = self.host {
            host.settle_range(address, len, writing);
        }
    }

    /// Before a refresh of `[address, address + len)` of `shadow`: as
    /// [`Batch::settle`], but a submission whose shaders write the range
    /// through the same copy is waited for only if the title wrote a page
    /// of the overlap since the copy took it (the world's big heap is both
    /// refreshed and written by shaders in most submissions).
    fn settle_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let Some(host) = self.host else { return };
        let copy = std::sync::Arc::as_ptr(shadow) as usize;
        let to = address.saturating_add(len);
        if let Some(value) = host
            .timeline
            .refresh_conflict(copy, address, to, |a, b| shadow.stale(a, b))
        {
            host.settle(value);
        }
    }

    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let (key, end) = (std::sync::Arc::as_ptr(shadow) as usize, address + len);
        if self.current.get(&key).is_some_and(|&(_, a, b)| a <= address && end <= b) {
            return;
        }
        match self.pending.get(&key) {
            Some(&i) => {
                let (_, from, to) = &mut self.refreshes[i];
                (*from, *to) = ((*from).min(address), (*to).max(end));
            }
            None => {
                self.pending.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end));
            }
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        self.current.clear();
    }

    /// Does the refreshes; returns the bytes copied.
    fn flush(&mut self) -> usize {
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        self.pending.clear();
        let mut total = 0;
        for (shadow, from, to) in self.refreshes.drain(..) {
            let mut skip: Vec<(usize, usize)> = Vec::new();
            if let Some(host) = self.host {
                let copy = std::sync::Arc::as_ptr(&shadow) as usize;
                loop {
                    if let Some(value) =
                        host.timeline
                            .refresh_conflict(copy, from, to, |a, b| shadow.stale(a, b))
                    {
                        host.settle(value);
                        continue;
                    }
                    skip = host
                        .timeline
                        .in_flight_writers(copy, from, to)
                        .into_iter()
                        .map(|(_, a, b)| (a, b))
                        .collect();
                    break;
                }
            }
            skip.sort_unstable();
            let started = std::time::Instant::now();
            let mut at = from;
            let mut copied = 0usize;
            for (w_from, w_to) in skip {
                let w_from = w_from.max(from);
                let w_to = w_to.min(to);
                if w_from > at {
                    copied += shadow.update(at, w_from - at);
                }
                at = at.max(w_to);
            }
            if at < to {
                copied += shadow.update(at, to - at);
            }
            let us = started.elapsed().as_micros() as u64;
            #[cfg(feature = "diagnostics")]
            crate::copystat::range(from, to - from, us, 0);
            PREPARING[0].fetch_add(us, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
            total += copied;
            // Kept: the union with what was current if they touch, else
            // the larger.
            let key = std::sync::Arc::as_ptr(&shadow) as usize;
            let (_, a, b) = self.current.entry(key).or_insert((shadow, from, to));
            if from <= *b && *a <= to {
                (*a, *b) = ((*a).min(from), (*b).max(to));
            } else if to - from > *b - *a {
                (*a, *b) = (from, to);
            }
        }
        for (shadow, address, len) in self.armed.drain(..) {
            shadow.writing(address, len, true);
        }
        total
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // The kernel writes the destination: it must not be guarded meanwhile.
    let _writing = title_dst.then(|| crate::watch::writing(dst as usize, len));
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    /// A GPU-only buffer of `size` bytes in the copy-destination state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        made.ok().and(buffer)
    }

    /// A mapped buffer for a list's own memory (uploads, scratch): as an
    /// upload buffer for the CPU, and a copy destination for the GPU (a
    /// repeat of the list in one submission gets its bytes copied in,
    /// `Repeat`).
    pub(crate) fn list_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no list buffer");
                None
            }
        }
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    /// A desktop can become presentable again. Keep retrying while limiting
    /// the DXGI calls and warnings during the outage.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if self.next_attempt.is_some_and(|next| now < next) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        crate::memory::log_frame();
        #[cfg(feature = "diagnostics")]
        crate::watch::frame_done();
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let root_views = crate::draw::take_root_view_slots() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame",
                making = crate::pipeline::making()
            );
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
            );
            #[cfg(feature = "diagnostics")]
            {
                let copy_frames = std::mem::take(&mut self.copy_stats_frames) as f64;
                crate::copystat::log(copy_frames);
                crate::bindless::log(copy_frames);
                crate::shadow::log_views(copy_frames);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        if presenter.is_none() {
            *presenter = match retry
                .attempt(Instant::now(), || self.presenter(width, height, format))?
            {
                Ok(p) => Some(p),
                Err((step, e)) => {
                    tracing::warn!(error = %e, step, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // The back buffers may not be in use.
            self.drain();
            let resized = retry.attempt(Instant::now(), || {
                // SAFETY: no back buffer is held; every submission has finished.
                unsafe {
                    p.swap_chain.ResizeBuffers(
                        BUFFERS,
                        width,
                        height,
                        format,
                        DXGI_SWAP_CHAIN_FLAG(0),
                    )
                }
            })?;
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            self.execute(&[&p.copy]);
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            // shortcut: the frame's submissions are finished before the
            // present returns, as when every submission was waited for (the
            // title may take a present for the GPU being done with the
            // frame). Only work within a frame overlaps; let one frame run
            // on once titles are known to wait for their fences instead.
            self.drain();
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height).map_err(|e| ("window", e))?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| ("DXGI factory", e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)
                .and_then(|chain| chain.cast::<IDXGISwapChain3>())
                .map_err(|e| ("swap chain", e))?;
            let copy = self.new_list().map_err(|e| ("copy list", e))?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

/// Host resources that view heap slots stopped viewing while a submission
/// may have been running ([`retire`]).
/// Host resources and shadow copies left behind when view heap slots are
/// rewritten while earlier submissions may still be running.
static RETIRED: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());
static RETIRED_SHADOWS: Mutex<Vec<std::sync::Arc<crate::shadow::Shadow>>> = Mutex::new(Vec::new());

/// Keeps `resource` alive until the GPU work running now (or the next
/// submission's) has run: title threads rewrite view heap slots while the
/// GPU runs, and a shader that took the old descriptor still reads what it
/// views.
pub(crate) fn retire(resource: ID3D12Resource) {
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(resource);
}

/// Keeps `shadow` alive (and its mark bits unrecycled) until the next
/// submission finishes so in-flight shaders using an overwritten view slot
/// never see a freed buffer or recycled mark bits.
pub(crate) fn retire_shadow(shadow: std::sync::Arc<crate::shadow::Shadow>) {
    RETIRED_SHADOWS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(shadow);
}

/// Releases all retired resources, shadows, and mark bits (called when idle
/// after [`Host::drain`] or directly in tests).
pub(crate) fn release_retired() {
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    drop(retired);
    let shadows = std::mem::take(&mut *RETIRED_SHADOWS.lock().unwrap_or_else(|e| e.into_inner()));
    drop(shadows);
    crate::shadow::release_marks(crate::shadow::take_retired_marks());
}

/// Diagnostics: a noted command running a shader `VICTORY_TRACE_SHADERS`
/// names (hashes, comma-separated) is logged with what it binds, the first
/// 500.
#[cfg(feature = "diagnostics")]
fn traced(what: &str) {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let hashes = HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_SHADERS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    });
    if hashes.iter().any(|h| what.contains(h.as_str())) && LOGGED.fetch_add(1, Ordering::Relaxed) < 500 {
        tracing::info!("traced command: {what}");
    }
}

/// Diagnostics, `VICTORY_GPU_TIMING`: every command of the list during
/// whose run (each list alone) the device was found removed.
fn removed_during(runs: &[&HostList], list: Option<usize>) -> String {
    #[cfg(feature = "diagnostics")]
    if let Some(i) = list {
        let notes = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = format!("; removed while list {i} ran alone ({} commands):", notes.len());
        for (k, (n, _)) in notes.iter().enumerate() {
            text.push_str(&format!(" [{k}] {n}"));
        }
        return text;
    }
    let _ = (runs, list);
    String::new()
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// A host copy of tracked title memory: a submission whose lists all ask
    /// for it whole copies the one page the title wrote, once, and again
    /// only after a step that writes title memory.
    #[test]
    fn a_copy_is_refreshed_once_per_submission_by_the_pages_written() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len, "all of it at first");
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(7) };
        // The next submission: two lists ask for the copy.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        batch.add(&copy, p + 100, 200);
        assert_eq!(batch.flush(), PAGE, "the page written");
        assert_eq!(copy.read_host((p + PAGE + 8) as u64, 1)[0], 7);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(1) };
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        batch.title_written();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "after a title write, what changed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}

#[cfg(test)]
mod submissions {
    use std::sync::Mutex;

    use windows::Win32::Graphics::Direct3D12::{D3D12_FENCE_FLAG_NONE, ID3D12Fence};

    use crate::queue::Written;

    /// Serializes tests that temporarily gate the shared host queue so two
    /// gates never overlap.
    static GATE_LOCK: Mutex<()> = Mutex::new(());

    /// Private fence armed via `Host::execute_gated`; always signalled on drop
    /// so even a test panic cannot leave the shared host queue stalled.
    struct Gate(ID3D12Fence);

    impl Gate {
        fn new(host: &super::Host) -> Self {
            // SAFETY: private fence on the live host device.
            let fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap() };
            Gate(fence)
        }

        fn release(&self) {
            // SAFETY: our private fence.
            unsafe {
                let _ = self.0.Signal(1);
            }
        }
    }

    impl Drop for Gate {
        fn drop(&mut self) {
            self.release();
        }
    }

    /// With `VICTORY_ASYNC_SUBMIT` off (the default): when `execute`
    /// returns, the submission has run and is finished, its bottom-of-pipe
    /// write done and a signal queued behind it set, with no drain.
    #[test]
    fn a_waited_submission_is_finished_when_execute_returns() {
        let Some(host) = super::host() else { return }; // no host GPU here
        // A gated test's submission would hold this one (and the
        // submission lock) until its gate opens.
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let list = host.new_list().unwrap();
        let word = Box::new(0u64);
        list.defer_write((&raw const *word).cast_mut().cast(), Written::Value(5), 8);
        list.close();
        host.execute_waited(&[&list]);
        assert!(!list.running(), "finished before execute returned");
        // SAFETY: our word; its submission is finished.
        assert_eq!(unsafe { std::ptr::read_volatile(&*word) }, 5);
        let signalled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let set = signalled.clone();
        host.after(move || set.store(true, std::sync::atomic::Ordering::Release));
        // Other tests' submissions may be in flight: the signal follows them.
        host.drain();
        assert!(signalled.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn a_list_reset_while_it_runs_records_into_fresh_memory() {
        let Some(host) = super::host() else { return }; // no host GPU here
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let list = host.new_list().unwrap();
        let (first, second) = (Box::new(0u64), Box::new(0u64));
        list.defer_write((&raw const *first).cast_mut().cast(), Written::Value(7), 8);
        list.close();
        let gate = Gate::new(host);
        host.execute_gated(&[&list], &gate.0);
        // Gate guarantees the first submission is still in flight when reset runs.
        assert!(list.running());
        // Recorded again at once, as titles do: the running submission
        // keeps what it uses, including its bottom-of-pipe write.
        list.reset();
        list.defer_write((&raw const *second).cast_mut().cast(), Written::Value(9), 8);
        list.close();
        gate.release();
        host.execute(&[&list]);
        host.drain();
        // SAFETY: our words; the finishing thread is done with them.
        let (a, b) = unsafe {
            (
                std::ptr::read_volatile(&*first),
                std::ptr::read_volatile(&*second),
            )
        };
        assert_eq!((a, b), (7, 9));
        assert!(!list.running());
    }

    /// A submission naming one list twice: the repeat's prelude records into
    /// its own allocator and list while the first is still pending, so both
    /// run with their own barriers, and nothing waits on the CPU.
    #[test]
    fn a_list_named_twice_in_a_submission_has_two_preludes() {
        use std::sync::atomic::Ordering;
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let null = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        // A 16x16 RGBA8 render target, starting in COMMON.
        let desc: [u32; 14] = [3, 0, 0, 0, 16, 0, 16, (1 << 16) | 1, 28, 1, 0, 0, 1, 0];
        let mut tex = null;
        let made = crate::resource::create_at_address(
            null,
            0x5_0000,
            desc.as_ptr().cast(),
            0,
            null.cast(),
            &iid,
            &mut tex,
        );
        assert_eq!(made, crate::object::S_OK);
        let list = host.new_list().unwrap();
        // The list starts at RENDER_TARGET and ends at COPY_SOURCE.
        assert!(list.transition(tex, 0x4).is_none());
        let (texture, before) = list.transition(tex, 0x800).unwrap();
        assert_eq!(before, 0x4);
        // SAFETY: an open list and a live texture.
        unsafe {
            list.list.ResourceBarrier(&[super::transition(
                texture,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES(0x4),
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        list.close();
        let gate = Gate::new(host);
        // COMMON -> RENDER_TARGET before the first run, COPY_SOURCE ->
        // RENDER_TARGET before the second: two different barriers.
        host.execute_gated(&[&list, &list], &gate.0);
        assert!(list.running());
        let mut spares = 0;
        host.timeline.each_work(|w| {
            if w.lists.iter().any(|l| std::ptr::eq(l.0, &list)) {
                spares += w.preludes.len();
            }
        });
        assert_eq!(spares, 1, "the repeat records into its own prelude");
        gate.release();
        host.drain();
        let copy = crate::resource::host_copy(tex).unwrap();
        assert_eq!(copy.state.load(Ordering::Relaxed), 0x800);
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        type Release = extern "system" fn(crate::object::This) -> u32;
        crate::object::slot::<Release>(tex, 0x10)(tex);
    }

    /// A list named twice around one whose step changes the title memory
    /// it uploads: each run reads what its own steps put in the list's
    /// upload memory (the repeat's bytes are copied in on the GPU, once the
    /// first run is done with them), and nothing waits on the CPU.
    #[test]
    fn a_repeated_list_reads_its_own_upload_memory() {
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (word, later) = (Box::new(0x1111_1111u32), Box::new(0x2222_2222u32));
        let (out, seen) = host.shared_buffer(256).unwrap();
        // SAFETY: our mapped buffer of 256 bytes.
        unsafe { std::ptr::write_bytes(seen, 0, 8) };
        let list = host.new_list().unwrap();
        let (_, to) = list.scratch(4).unwrap();
        list.defer_fill(to, (&raw const *word).cast(), 4);
        let (scratch, offset) = {
            let s = list.scratch.lock().unwrap();
            (s.chunks[0]._buffer.clone(), (to as usize - s.chunks[0].mapped) as u64)
        };
        // SAFETY: an open list and live buffers.
        unsafe { list.list.CopyBufferRegion(&out, 4, &scratch, offset, 4) };
        list.close();
        // Between the runs the title word changes (a step), and the GPU
        // reads the list's upload memory as the first run left it.
        let between = host.new_list().unwrap();
        between.defer_copy(
            (&raw const *word).cast_mut().cast(),
            (&raw const *later).cast(),
            4,
        );
        // SAFETY: as above.
        unsafe { between.list.CopyBufferRegion(&out, 0, &scratch, offset, 4) };
        between.close();
        host.execute(&[&list, &between, &list]);
        host.drain();
        // SAFETY: our mapped buffer; the GPU is done with it.
        let read = unsafe {
            [
                (seen as *const u32).read_volatile(),
                (seen as *const u32).add(1).read_volatile(),
            ]
        };
        assert_eq!(
            read,
            [0x1111_1111, 0x2222_2222],
            "between the runs, then after the repeat"
        );
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn signals_follow_the_writes_of_earlier_submissions() {
        let Some(host) = super::host() else { return };
        let list = host.new_list().unwrap();
        let word: &'static mut u64 = Box::leak(Box::new(0));
        let at = (&raw mut *word) as usize;
        list.defer_write(at as *mut u8, Written::Value(3), 8);
        list.close();
        host.execute(&[&list]);
        let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
        let s = seen.clone();
        host.after(move || {
            // SAFETY: the leaked word lives on.
            let value = unsafe { std::ptr::read_volatile(at as *const u64) };
            *s.lock().unwrap() = Some(value);
        });
        host.drain();
        assert_eq!(*seen.lock().unwrap(), Some(3));
    }

    #[test]
    fn a_step_reading_what_an_earlier_submission_writes_waits_for_it() {
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (first, second) = (host.new_list().unwrap(), host.new_list().unwrap());
        let word = Box::new(0u64);
        let at = (&raw const *word).cast_mut().cast::<u8>();
        first.defer_write(at, Written::Value(0x1234_5678_9abc), 8);
        first.close();
        // The second submission copies the word at submission: after the
        // first's write, as the GPU would order them.
        let copy = Box::new(0u64);
        second.defer_fill((&raw const *copy).cast_mut().cast(), at, 8);
        second.close();
        let gate = Gate::new(host);
        host.execute_gated(&[&first], &gate.0);
        // Signal the gate from another thread so `second` genuinely waits in
        // `settle` without ever deadlocking parallel host tests.
        let gate_fence = gate.0.clone();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            // SAFETY: our private fence.
            unsafe {
                let _ = gate_fence.Signal(1);
            }
        });
        host.execute(&[&second]);
        releaser.join().unwrap();
        // SAFETY: our word, copied while the second was submitted.
        assert_eq!(unsafe { std::ptr::read_volatile(&*copy) }, 0x1234_5678_9abc);
        second.reset();
        first.reset();
    }

    #[test]
    fn dynamic_descriptor_leases_stay_held_until_async_submission_finishes() {
        use crate::descriptor::{HeapDesc, create_heap, heap};
        use crate::object::{S_OK, This, guid, slot};

        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        #[repr(C)]
        struct ExtendedHeapDesc {
            words: [u32; 4], // kind=0 (views), count=8, flags=1 (shader visible), nodes=0
            addr: u64,
            size: u64,
        }
        let heap_iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let ext = ExtendedHeapDesc {
            words: [0, 8, 1, 0],
            addr: 0,
            size: 0,
        };
        let mut title_heap: This = std::ptr::null();
        assert_eq!(
            create_heap(
                std::ptr::null(),
                std::ptr::from_ref(&ext).cast::<HeapDesc>(),
                &heap_iid,
                &mut title_heap
            ),
            S_OK
        );
        let heap_ref = heap(title_heap);
        let host_heap = heap_ref.host().unwrap();
        // Lease every dynamic slot on `list`, in 2^16 runs (a heap has 2^16
        // or 2^18 of them, depending on what the host takes).
        let list = host.new_list().unwrap();
        while let Some(first) = host_heap.dynamic(1 << 16) {
            list.keep_dynamic(heap_ref.id(), first, 1 << 16);
        }
        assert_eq!(host_heap.dynamic(1), None);
        list.close();

        // Hold the host queue behind a private fence armed atomically inside
        // `execute_gated` so parallel tests cannot deadlock before `release()`.
        let gate = Gate::new(host);
        host.execute_gated(&[&list], &gate.0);
        list.reset();
        // Reset while in flight must NOT free the leased dynamic range yet.
        assert_eq!(
            host_heap.dynamic(1),
            None,
            "dynamic descriptor slots must remain leased until the async fence completes"
        );
        gate.release();
        host.drain();
        let after = host_heap.dynamic(1);
        assert!(
            after.is_some(),
            "dynamic descriptor slots must be released once the submission finishes"
        );
        host_heap.release(after.unwrap(), 1);
        assert_eq!(
            slot::<extern "system" fn(This) -> u32>(title_heap, 0x10)(title_heap),
            0
        );
    }

    const MARKED_WRITE_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), UAV(u0), UAV(u1)"
cbuffer Params : register(b0) { uint offset; uint value; uint mark_bit; }
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer marks : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    dst.Store(offset, value);
    marks.InterlockedOr((mark_bit >> 5) * 4, 1u << (mark_bit & 31));
}
"#;

    const READ_COPY_HLSL: &str = r#"
#define RS "SRV(t0), UAV(u0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    dst.Store(0, src.Load(0));
    dst.Store(4, src.Load(16));
}
"#;

    fn compile_cs(
        host: &super::Host,
        source: &str,
    ) -> (
        windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    ) {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: static strings and owned out pointers.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("host_test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: compiled DXBC blob containing an embedded root signature.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        }
    }

    fn marked_write_pipeline(
        host: &super::Host,
    ) -> &'static (
        windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    ) {
        static PIPE: std::sync::OnceLock<(
            windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
            windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
        )> = std::sync::OnceLock::new();
        PIPE.get_or_init(|| compile_cs(host, MARKED_WRITE_HLSL))
    }

    fn read_copy_pipeline(
        host: &super::Host,
    ) -> &'static (
        windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    ) {
        static PIPE: std::sync::OnceLock<(
            windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
            windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
        )> = std::sync::OnceLock::new();
        PIPE.get_or_init(|| compile_cs(host, READ_COPY_HLSL))
    }

    fn record_marked_write(
        host: &super::Host,
        list: &super::HostList,
        copy: &std::sync::Arc<crate::shadow::Shadow>,
        uav_base: u64,
        offset: u32,
        value: u32,
        mark_bit: u32,
    ) {
        let (root, pso) = marked_write_pipeline(host);
        let bitmap = list.view_writes().unwrap();
        let params = [offset, value, mark_bit];
        list.keep(root);
        list.keep(pso);
        // SAFETY: open list with live cached root signature, PSO, shadow copy, and pooled bitmap.
        unsafe {
            list.list.SetComputeRootSignature(root);
            list.list.SetPipelineState(pso);
            list.list
                .SetComputeRoot32BitConstants(0, 3, params.as_ptr().cast(), 0);
            list.list
                .SetComputeRootUnorderedAccessView(1, copy.gpu(uav_base));
            list.list.SetComputeRootUnorderedAccessView(2, bitmap);
            list.list.Dispatch(1, 1, 1);
        }
    }

    #[test]
    fn overlapping_marked_submissions_on_disjoint_pages_preserve_marks_and_title_writes() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        host.drain();
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let copy = crate::shadow::find(p as u64, len as u64).unwrap();
        let bit0 = copy.mark_entry(p as u64).unwrap().0;
        let bit1 = copy.mark_entry((p + PAGE) as u64).unwrap().0;

        let list1 = host.new_list().unwrap();
        copy.refresh(&list1, p as u64, PAGE as u64);
        copy.gpu_writes(&list1, p as u64, 64, true);
        record_marked_write(host, &list1, &copy, p as u64, 0, 0x1111_1111, bit0);
        list1.close();

        let list2 = host.new_list().unwrap();
        copy.refresh(&list2, (p + PAGE) as u64, PAGE as u64);
        copy.gpu_writes(&list2, (p + PAGE) as u64, 64, true);
        record_marked_write(
            host,
            &list2,
            &copy,
            (p + PAGE) as u64,
            0,
            0x2222_2222,
            bit1,
        );
        list2.close();

        let gate = Gate::new(host);
        host.execute_gated(&[&list1], &gate.0);
        assert!(list1.running());
        host.execute(&[&list2]);
        assert!(
            list1.running(),
            "disjoint marked submission on page 1 must not settle page 0"
        );

        // Concurrent title writes to both marked pages while both submissions are in flight.
        // SAFETY: bytes inside our mapped watch pool.
        unsafe {
            ((p + 16) as *mut u32).write_volatile(0xaaaa_aaaa);
            ((p + PAGE + 16) as *mut u32).write_volatile(0xbbbb_bbbb);
        }

        gate.release();
        host.drain();

        // SAFETY: reads of our title pool after GPU completion.
        unsafe {
            assert_eq!((p as *const u32).read_volatile(), 0x1111_1111);
            assert_eq!(((p + 16) as *const u32).read_volatile(), 0xaaaa_aaaa);
            assert_eq!(((p + PAGE) as *const u32).read_volatile(), 0x2222_2222);
            assert_eq!(
                ((p + PAGE + 16) as *const u32).read_volatile(),
                0xbbbb_bbbb
            );
        }

        crate::shadow::unregister(p, p as u64);
        drop(copy);
        host.drain();
        crate::watch::tests::free(p);
    }

    #[test]
    fn same_page_marked_submissions_serialize_and_preserve_all_writers() {
        use crate::watch::PAGE;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        host.drain();
        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, PAGE as u64);
        let copy = crate::shadow::find(p as u64, PAGE as u64).unwrap();
        let bit = copy.mark_entry(p as u64).unwrap().0;

        let list1 = host.new_list().unwrap();
        copy.refresh(&list1, p as u64, PAGE as u64);
        copy.gpu_writes(&list1, p as u64, 64, true);
        record_marked_write(host, &list1, &copy, p as u64, 0, 0x3333_3333, bit);
        list1.close();

        let list2 = host.new_list().unwrap();
        copy.refresh(&list2, (p + 128) as u64, 64);
        copy.gpu_writes(&list2, (p + 128) as u64, 64, true);
        record_marked_write(host, &list2, &copy, p as u64, 128, 0x4444_4444, bit);
        list2.close();

        let gate = Gate::new(host);
        host.execute_gated(&[&list1], &gate.0);

        // Concurrent title write on the same page while submission 1 runs.
        // SAFETY: within our 1-page pool.
        unsafe { ((p + 256) as *mut u32).write_volatile(0x5555_5555) };

        let released = Arc::new(AtomicBool::new(false));
        let flag = released.clone();
        let gate_fence = gate.0.clone();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            flag.store(true, Ordering::Release);
            // SAFETY: our private fence.
            unsafe {
                let _ = gate_fence.Signal(1);
            }
        });

        host.execute(&[&list2]);
        assert!(
            released.load(Ordering::Acquire),
            "second marked submission on the same 64 KiB page must wait for the first"
        );
        releaser.join().unwrap();
        host.drain();

        // SAFETY: reads of our pool after completion.
        unsafe {
            assert_eq!((p as *const u32).read_volatile(), 0x3333_3333);
            assert_eq!(((p + 128) as *const u32).read_volatile(), 0x4444_4444);
            assert_eq!(((p + 256) as *const u32).read_volatile(), 0x5555_5555);
        }

        crate::shadow::unregister(p, p as u64);
        drop(copy);
        host.drain();
        crate::watch::tests::free(p);
    }

    #[test]
    fn list_reset_while_running_isolates_mark_bitmaps() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        host.drain();
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let copy = crate::shadow::find(p as u64, len as u64).unwrap();
        let bit0 = copy.mark_entry(p as u64).unwrap().0;
        let bit1 = copy.mark_entry((p + PAGE) as u64).unwrap().0;

        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, PAGE as u64);
        copy.gpu_writes(&list, p as u64, 64, true);
        record_marked_write(host, &list, &copy, p as u64, 0, 0x6666_6666, bit0);
        let bm1 = list.view_writes().unwrap();
        list.close();

        let gate = Gate::new(host);
        host.execute_gated(&[&list], &gate.0);
        assert!(list.running());
        list.reset();

        copy.refresh(&list, (p + PAGE) as u64, PAGE as u64);
        copy.gpu_writes(&list, (p + PAGE) as u64, 64, true);
        record_marked_write(
            host,
            &list,
            &copy,
            (p + PAGE) as u64,
            0,
            0x7777_7777,
            bit1,
        );
        let bm2 = list.view_writes().unwrap();
        assert_ne!(
            bm1, bm2,
            "reset while running must allocate a distinct mark bitmap"
        );
        list.close();

        host.execute(&[&list]);
        gate.release();
        host.drain();

        // SAFETY: reads of our pool after completion.
        unsafe {
            assert_eq!((p as *const u32).read_volatile(), 0x6666_6666);
            assert_eq!(((p + PAGE) as *const u32).read_volatile(), 0x7777_7777);
        }

        crate::shadow::unregister(p, p as u64);
        drop(copy);
        host.drain();
        crate::watch::tests::free(p);
    }

    #[test]
    fn overlapping_reader_and_title_write_preserve_marked_gpu_and_cpu_bytes() {
        use crate::watch::PAGE;
        use std::sync::atomic::Ordering;

        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        host.drain();
        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, PAGE as u64);
        let copy = crate::shadow::find(p as u64, PAGE as u64).unwrap();
        let bit = copy.mark_entry(p as u64).unwrap().0;

        let list1 = host.new_list().unwrap();
        copy.refresh(&list1, p as u64, PAGE as u64);
        copy.gpu_writes(&list1, p as u64, 64, true);
        record_marked_write(host, &list1, &copy, p as u64, 0, 0x1234_5678, bit);
        list1.close();

        let (out, out_ptr) = host.shared_buffer(256).unwrap();
        // SAFETY: our 256-byte mapped shared buffer.
        unsafe { std::ptr::write_bytes(out_ptr, 0, 256) };
        let (r_root, r_pso) = read_copy_pipeline(host);
        let list2 = host.new_list().unwrap();
        copy.refresh(&list2, p as u64, PAGE as u64);
        list2.keep(r_root);
        list2.keep(r_pso);
        list2.keep(&out);
        // SAFETY: open list and live host buffers/pipeline.
        unsafe {
            list2.list.SetComputeRootSignature(r_root);
            list2.list.SetPipelineState(r_pso);
            list2.list.SetComputeRootShaderResourceView(0, copy.gpu(p as u64));
            list2
                .list
                .SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            list2.list.Dispatch(1, 1, 1);
        }
        list2.close();

        let gate1 = Gate::new(host);
        let gate2 = Gate::new(host);
        host.execute_gated(&[&list1], &gate1.0);
        host.execute_gated(&[&list2], &gate2.0);
        assert!(list1.running() && list2.running());

        // Title writes offset +16 while S1 (writer) and S2 (subsequent reader) are both in flight.
        // SAFETY: inside our 1-page pool.
        unsafe { ((p + 16) as *mut u32).write_volatile(0xdead_beef) };

        gate1.release();
        host.settle(list1.last.load(Ordering::Acquire));

        // While S2 is still reading `copy`, S1's copy-back must merge into title
        // memory without clobbering `copy`'s bytes at +16 or clearing the dirty page.
        // SAFETY: title pool read.
        unsafe {
            assert_eq!((p as *const u32).read_volatile(), 0x1234_5678);
            assert_eq!(((p + 16) as *const u32).read_volatile(), 0xdead_beef);
        }
        assert_eq!(copy.read_host((p + 16) as u64, 1)[0], 0);

        gate2.release();
        host.drain();

        // SAFETY: `out_ptr` is mapped and S2 completed.
        let read = unsafe { std::slice::from_raw_parts(out_ptr.cast::<u32>(), 2).to_vec() };
        assert_eq!(read, [0x1234_5678, 0]);
        assert!(
            copy.update(p, PAGE) > 0,
            "deferred title write remains dirty until after S2 drains"
        );
        assert_eq!(copy.read_host((p + 16) as u64, 1)[0], 0xdead_beef);

        crate::shadow::unregister(p, p as u64);
        drop(copy);
        host.drain();
        crate::watch::tests::free(p);
    }

    #[test]
    fn dropped_shadow_mark_bits_wait_for_in_flight_submission() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else { return };
        let _guard = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        host.drain();

        let p1 = crate::watch::tests::pool(1);
        crate::shadow::register(p1, p1 as u64, PAGE as u64);
        let s1 = crate::shadow::find(p1 as u64, PAGE as u64).unwrap();
        let base1 = s1.mark_entry(p1 as u64).unwrap().0;

        let list = host.new_list().unwrap();
        list.close();
        let gate = Gate::new(host);
        host.execute_gated(&[&list], &gate.0);

        crate::shadow::unregister(p1, p1 as u64);
        drop(s1);

        let p2 = crate::watch::tests::pool(1);
        crate::shadow::register(p2, p2 as u64, PAGE as u64);
        let s2 = crate::shadow::find(p2 as u64, PAGE as u64).unwrap();
        let base2 = s2.mark_entry(p2 as u64).unwrap().0;
        assert_ne!(
            base2, base1,
            "mark bits of a dropped shadow stay retired while a submission is in flight"
        );

        gate.release();
        host.drain();

        let p3 = crate::watch::tests::pool(1);
        crate::shadow::register(p3, p3 as u64, PAGE as u64);
        let s3 = crate::shadow::find(p3 as u64, PAGE as u64).unwrap();
        let base3 = s3.mark_entry(p3 as u64).unwrap().0;
        assert_eq!(
            base3, base1,
            "retired mark bits become reusable once the in-flight submission drains"
        );

        crate::shadow::unregister(p2, p2 as u64);
        crate::shadow::unregister(p3, p3 as u64);
        drop(s2);
        drop(s3);
        host.drain();
        crate::watch::tests::free(p1);
        crate::watch::tests::free(p2);
        crate::watch::tests::free(p3);
    }
}
