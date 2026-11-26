//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
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
    pub(crate) depth_bounds_supported: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
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
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host submissions whose end could not be waited for: a list in one may
/// still run, so its dynamic slots are not taken back (`crate::draw::Lease`).
static UNWAITED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Host submissions not waited for so far ([`UNWAITED`]).
pub(crate) fn unwaited() -> u64 {
    UNWAITED.load(Ordering::Relaxed)
}

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

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => {
            // Logs where host copies go, once, at the start.
            host.gpu_upload_writes();
            Some(host)
        }
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
        let mut debug_enabled = false;
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => {
                    if let Some(l) = &layer {
                        l.EnableDebugLayer();
                        debug_enabled = true;
                    }
                }
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
            #[cfg(test)]
            if std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1")
                || std::env::var("VICTORY_ADAPTER").is_ok_and(|v| !v.is_empty())
            {
                device = Some(
                    victory_shader::emit::host::Host::new(false)
                        .map_err(|e| windows::core::Error::new(windows::Win32::Foundation::E_FAIL, e))?
                        .device,
                );
            }
            if device.is_none() {
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
                // Whether a GPU hang ends the run or freezes the PC (#1054).
                crate::tdr::log();
            }
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
        let depth_bounds_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS2,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
            && options.DepthBoundsTestSupported.as_bool();
        if let Some(cap) = crate::descriptor::host::view_heap_cap(&device) {
            victory_shader::emit::set_view_heap_limit(cap);
            tracing::info!(cap, "host view heaps capped: translated shaders keep their view indices below the cap");
        }
        let messages = debug_enabled
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
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
            depth_bounds_supported,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
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
    /// Host copies go to GPU upload heaps only when asked
    /// (`VICTORY_GPU_UPLOAD=1`, and `VICTORY_GPU_UPLOAD_WRITES=1` for the
    /// written ones too) and the host has them: unset or `=0` keeps them
    /// in system memory (or P-0a's video memory, `crate::vram`, if asked).
    #[test]
    fn upload_placement_follows_the_switches_and_the_heaps() {
        use super::placement;
        use crate::switch::value_is_on as on;
        use std::ffi::OsStr;

        // Unset, =0 and =1.
        let (z, o) = (on(Some(OsStr::new("0"))), on(Some(OsStr::new("1"))));
        let u = on(None);
        // (VICTORY_GPU_UPLOAD, VICTORY_GPU_UPLOAD_WRITES, heaps) -> placement.
        let cases = [
            ((u, u, true), (false, false), "both unset"),
            ((o, o, true), (true, true), "both =1"),
            ((z, z, true), (false, false), "both =0"),
            ((o, u, true), (true, false), "UPLOAD=1 alone"),
            ((u, o, true), (false, false), "WRITES=1 alone"),
            ((z, o, true), (false, false), "UPLOAD=0, WRITES=1"),
            ((o, z, true), (true, false), "WRITES=0"),
            ((u, u, false), (false, false), "no heaps, both unset"),
            ((o, o, false), (false, false), "no heaps, both =1"),
            ((z, z, false), (false, false), "no heaps, both =0"),
        ];
        for ((upload, writes, heaps), want, what) in cases {
            assert_eq!(placement(upload, writes, heaps), want, "{what}");
        }
    }

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

    /// A list renders to a depth texture, then a use of its twin (made in
    /// between) follows: the render is copied to the twin in the list, not
    /// before it.
    #[test]
    fn a_twin_made_after_a_depth_use_in_the_list_gets_the_copy() {
        let Some(host) = host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation; our out pointer.
        let made = unsafe {
            host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture)
        };
        let Ok(()) = made else { return };
        let depth = std::sync::Arc::new(crate::resource::HostTexture::new(texture.unwrap(), 0, Vec::new()));
        let list = host.new_list().unwrap();
        // Draw 1 renders to it: the depth side is used, no twin yet.
        assert!(list.move_texture(&depth, 0x10).is_none(), "the list's first use records nothing");
        // Draw 2 uses its (new) twin: Draw 1's depth is copied over now.
        assert_eq!(list.twin_use(&depth, crate::twin::TWIN, true), Some(false));
        list.close();
    }

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
        list.reset();
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 2);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(list.scratch(512).unwrap(), first);
        list.close();
    }

    /// Render targets, depth buffers and textures shaders write are kept
    /// resident before textures uploaded from title memory (the textures
    /// `crate::texture::title_written` names).
    #[test]
    fn textures_the_gpu_writes_are_kept_resident_first() {
        for flags in [0x1, 0x2, 0x4, 0x1 | 0x4, 0x2 | 0x8, 0x1 | 0x20] {
            assert_eq!(Residency::of_texture(flags), Residency::Frame, "{flags:#x}");
        }
        for flags in [0, 0x8, 0x10, 0x20, 0x40] {
            assert_eq!(
                Residency::of_texture(flags),
                Residency::Streamed,
                "{flags:#x}"
            );
        }
        assert_eq!(Residency::Frame.priority(), D3D12_RESIDENCY_PRIORITY_HIGH);
        assert_eq!(
            Residency::Streamed.priority(),
            D3D12_RESIDENCY_PRIORITY_NORMAL
        );
    }

    /// A list's scratch chunks, read by its commands every frame, are kept
    /// resident.
    #[test]
    fn list_scratch_chunks_are_kept_resident() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        list.scratch(1).unwrap();
        let s = list.scratch.lock().unwrap();
        assert_eq!(kept(&s.chunks[0].buffer), Some(Residency::Frame));
        drop(s);
        list.close();
    }

    /// Copies in GPU upload heaps take half of what the rest of the process
    /// leaves of the video memory budget: less as use rises, none past the
    /// budget, at most 2 GiB (more on large budgets), and 2 GiB without a
    /// reading.
    #[test]
    fn upload_copies_take_their_share_of_the_budget() {
        const GIB: u64 = 1 << 30;
        assert!(upload_fits(None, 0, 2 * GIB));
        assert!(!upload_fits(None, GIB, GIB + 1));
        let fake = crate::budget::tests::Fake::default();
        let budget = crate::budget::tests::always_read(&fake);
        let fits = |own, size| upload_fits(Some(&budget), own, size);
        fake.set(8 * GIB, 2 * GIB);
        assert!(fits(0, 2 * GIB), "2 GiB of the 6 left");
        assert!(!fits(0, 2 * GIB + 1), "the ceiling");
        fake.set(8 * GIB, 6 * GIB);
        assert!(fits(0, GIB));
        assert!(!fits(0, GIB + 1), "half of the 2 GiB left");
        // The pool's own gigabyte is in the 6: the rest leave it 3.
        assert!(fits(GIB, GIB / 2));
        assert!(!fits(GIB, GIB / 2 + 1));
        fake.set(8 * GIB, 8 * GIB);
        assert!(!fits(0, 1), "nothing at the budget");
        fake.set(24 * GIB, 0);
        assert!(fits(0, 6 * GIB), "a quarter of a large budget");
        assert!(!fits(0, 6 * GIB + 1));
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
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Read-back memory for what the list's query resolves write, reused
    /// after a reset.
    readback: Mutex<Readback>,
    /// Upload memory for the tables of batched indirect clamps
    /// (`crate::indirect`): one open at a time, its unused room given back
    /// when it closes ([`HostList::give_back_table`]).
    tables: Mutex<Scratch>,
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
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// Queries begun in the list and not ended yet: host heap, type, index,
    /// and whether a draw since was not run on the host (`missed_draw`).
    queries: Mutex<Vec<(ID3D12QueryHeap, D3D12_QUERY_TYPE, u32, bool)>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Indirect clamps batched per window of indirect executions
    /// (`crate::indirect`, `VICTORY_BATCH_CLAMPS`), and whether a window is
    /// open (checked without the lock by every command that may write).
    pub(crate) clamps: Mutex<crate::indirect::Batch>,
    pub(crate) clamps_open: std::sync::atomic::AtomicBool,
    /// Indirect compute records rewritten on the GPU, reused after execution.
    dispatch_args: Mutex<Option<(ID3D12Resource, u64)>>,
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

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
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

/// Whether each list of a submission is timed on its own, and slow ones are
/// logged with their commands (`VICTORY_GPU_TIMING`, diagnostics builds; the
/// commands are noted with `VICTORY_D3D12_DRED` or the timing on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Whether recorded commands are noted at all: with device-removed data (the
/// markers) or with GPU timing (the per-command timestamps) on.
#[cfg(feature = "diagnostics")]
fn noting() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| dred_on() || timing())
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

/// What a host allocation is to a frame, for the system's choice of what to
/// demote to system memory once the process passes its video memory budget
/// (`crate::budget`). Without priorities it picks among all of them alike,
/// render targets used every frame included, and frames then take seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Residency {
    /// Read or written by most frames, and slow from system memory:
    /// render targets, depth buffers and textures shaders write (the GPU
    /// writes them every frame, and they hold the only copy of what it
    /// wrote), view heaps' tables, the views' written-pages bitmap, list
    /// scratch memory, host copies in GPU upload heaps. `HIGH`: demoted
    /// after everything else of ours. Not `MAXIMUM`, which the system keeps
    /// even past the budget at the cost of other processes.
    Frame,
    /// A texture uploaded from title memory (`crate::texture::title_written`):
    /// the title streams most of them, each read by some frames. `NORMAL`,
    /// what every allocation starts with: these go first, and sampling one
    /// from system memory is slower but does not stall the frame the way a
    /// render target there does. Not `LOW`, which would demote them before
    /// allocations that have no priority set (depth twins, upload staging),
    /// which frames need no more than these.
    Streamed,
}

impl Residency {
    /// A host texture's, from its resource flags: render target, depth
    /// stencil or unordered access, or none of them.
    pub(crate) fn of_texture(flags: u32) -> Self {
        const TARGET_DEPTH_UNORDERED: u32 = 0x1 | 0x2 | 0x4;
        if flags & TARGET_DEPTH_UNORDERED != 0 {
            Residency::Frame
        } else {
            Residency::Streamed
        }
    }

    fn priority(self) -> D3D12_RESIDENCY_PRIORITY {
        match self {
            Residency::Frame => D3D12_RESIDENCY_PRIORITY_HIGH,
            Residency::Streamed => D3D12_RESIDENCY_PRIORITY_NORMAL,
        }
    }
}

/// Sets `object`'s (a heap or committed resource) residency priority as
/// `residency` says. Called where such objects are made. Not for
/// descriptor heaps: a priority on a shader-visible one removed the device
/// at the next submission (RTX 5070 Ti, driver 621.24). Not for resources
/// placed in a heap either: their residency is their heap's, which takes
/// the priority instead (asked of a placed texture, the call removed the
/// device at the next submission too).
pub(crate) fn keep_resident(object: &impl Interface, residency: Residency) {
    #[cfg(test)]
    KEPT.lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(object.as_raw() as usize, residency);
    // The priority every allocation starts with: no call.
    if residency.priority() == D3D12_RESIDENCY_PRIORITY_NORMAL {
        return;
    }
    let Some(host) = host() else { return };
    let (Ok(device), Ok(pageable)) = (
        host.device.cast::<ID3D12Device1>(),
        object.cast::<ID3D12Pageable>(),
    ) else {
        return;
    };
    // SAFETY: one live pageable object and its priority.
    let set = unsafe { device.SetResidencyPriority(1, &Some(pageable), &residency.priority()) };
    if let Err(e) = set {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::debug!(error = %e, "no residency priorities on this host"));
    }
}

/// What [`keep_resident`] was asked for each object, by pointer (tests: the
/// device has no getter for priorities).
#[cfg(test)]
static KEPT: Mutex<std::collections::BTreeMap<usize, Residency>> =
    Mutex::new(std::collections::BTreeMap::new());

/// What [`keep_resident`] was last asked for `object` (tests).
#[cfg(test)]
pub(crate) fn kept(object: &impl Interface) -> Option<Residency> {
    KEPT.lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(object.as_raw() as usize))
        .copied()
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
/// the bytes used in it; the chunks each of the last recordings used.
#[derive(Default)]
struct Scratch {
    chunks: Vec<ScratchChunk>,
    current: usize,
    used: u64,
    recent: [usize; SCRATCH_RECORDINGS],
    recording: usize,
}

/// Recordings whose use of a list's scratch the chunks it keeps cover: the
/// chunks past what the busiest of them used are let go at a reset.
const SCRATCH_RECORDINGS: usize = 8;

impl Scratch {
    /// After the list's last submission ran: the chunks the busiest of the
    /// last recordings used are kept, the rest let go, and memory is handed
    /// out from the first chunk again. Without the trim one busy recording
    /// held its memory for the list's life.
    fn rewind(&mut self) {
        let used = if self.chunks.is_empty() {
            0
        } else {
            self.current + usize::from(self.used > 0)
        };
        self.recent[self.recording % SCRATCH_RECORDINGS] = used;
        self.recording += 1;
        let keep = self.recent.iter().copied().max().unwrap_or(0);
        self.chunks.truncate(keep.max(1));
        (self.current, self.used) = (0, 0);
    }

    /// Bytes of upload memory the chunks hold.
    #[cfg(test)]
    fn held(&self) -> u64 {
        self.chunks.iter().map(|c| c.capacity).sum()
    }
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;
const FIRST_SCRATCH_CHUNK: u64 = 64 << 10;

struct ScratchChunk {
    buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: u64,
}

/// A list's read-back memory ([`HostList::readback`]): mapped chunks of
/// [`READBACK_CHUNK`] bytes (buffer, CPU address), the one being filled and
/// the bytes of it handed out.
#[derive(Default)]
struct Readback {
    chunks: Vec<(ID3D12Resource, usize)>,
    current: usize,
    used: u64,
}

/// 8,192 occlusion results, or 744 of pipeline statistics.
const READBACK_CHUNK: u64 = 64 << 10;

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
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the pages they write (no snapshot then).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`).
    Heap(std::sync::Arc<crate::bindless::Viewed>),
    /// Shaders read title memory through a view heap's page table (the
    /// heap's key, `crate::memory`): the table names every copy, each
    /// brought up to date.
    Memory(usize),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
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
    after: Vec<Box<dyn Fn() + Send>>,
    /// Asked once the list has run, before anything goes back: whether its
    /// shaders read title memory with no copy (`crate::memory`); then what
    /// they wrote through views does not go back either.
    missed: Vec<Box<dyn Fn() -> bool + Send>>,
    /// Buffers shaders wrote, copied at the list's end into read-back
    /// buffers (from, to, bytes; [`HostList::read_back_at_close`]).
    read_backs: Vec<(ID3D12Resource, ID3D12Resource, u64)>,
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

    /// The wave sizes the host runs (D3D12 options 1), which translated
    /// compute shaders that declare the console's wave size must fit;
    /// logged once. Unknown (the query fails): any.
    pub(crate) fn target(&self) -> victory_shader::emit::Target {
        static TARGET: OnceLock<victory_shader::emit::Target> = OnceLock::new();
        *TARGET.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes the structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let target = if ok && options.WaveOps.as_bool() && options.WaveLaneCountMin > 0 {
                victory_shader::emit::Target {
                    lanes_min: options.WaveLaneCountMin,
                    lanes_max: options.WaveLaneCountMax,
                }
            } else {
                victory_shader::emit::Target::ANY
            };
            tracing::info!(
                lanes_min = target.lanes_min,
                lanes_max = target.lanes_max,
                "host wave sizes"
            );
            if !target.runs(64) {
                tracing::warn!(
                    "the host has no 64-lane waves: console wave64 compute programs that need \
                     their whole wave run as two 32-lane waves where there is a form for \
                     them; the rest do not translate, and their dispatches are dropped"
                );
            }
            target
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
    /// Ticks per second of the queue's timestamps (tests time GPU work).
    #[cfg(test)]
    pub(crate) fn timestamp_frequency(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1)
    }

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
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                readback: Mutex::default(),
                tables: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                id,
                queries: Mutex::default(),
                budget: OnceLock::new(),
                clamps: Mutex::default(),
                clamps_open: false.into(),
                dispatch_args: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// The queue lists run on: tile mappings (`crate::tiles`) go on it
    /// ahead of the lists submitted after them.
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// Runs closed lists and waits for them.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        self.submit(runs);
    }

    #[cfg(test)]
    pub(crate) fn execute_checked(&self, list: &HostList) {
        let error = self.submit(&[list]);
        if self.messages.is_some() {
            assert_eq!(error, None, "D3D12 debug layer reported an error");
            eprintln!("D3D12 debug layer checked: no errors");
        } else if !cfg!(feature = "diagnostics") {
            eprintln!("D3D12 debug layer check skipped: diagnostics feature disabled");
        } else if !crate::switch::on("VICTORY_D3D12_DEBUG") {
            eprintln!("D3D12 debug layer check skipped: VICTORY_D3D12_DEBUG unset");
        } else {
            eprintln!("D3D12 debug layer check skipped: debug layer or info queue unavailable");
        }
    }

    fn submit(&self, runs: &[&HostList]) -> Option<String> {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
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
                        Step::Heap(v) => Some(v.clone()),
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
        // Snapshots of copies in video memory read their mirrors: brought up
        // to date at once, and kept so until the copy-backs.
        let snapshots: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::GpuWrite(shadow, address, len, false) => {
                            Some((shadow.clone(), *address, *len))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        crate::shadow::snapshots_ahead(&snapshots);
        let mut batch = Batch::default();
        for list in runs {
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        // What the host wrote into lookup tables in video memory since the
        // last submission (`VICTORY_TABLES_VRAM=copy`), before any list
        // reads them; held until the submission has run.
        let table_copies = crate::tables::copies(self);
        if let Some(c) = &table_copies {
            lists.insert(0, Some(c.list.clone()));
            for first in firsts.iter_mut().skip(1) {
                *first += 1;
            }
        }
        let prepared = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsPrep, crate::copystat::page_faults().saturating_sub(faults));
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        let mut removed_in = None;
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::SUBMITTED, *submitted, 0);
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        WaitForSingleObject(self.event, INFINITE);
                    }
                    Err(e) => {
                        UNWAITED.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(error = %e, "host submission not waited for");
                    }
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::RAN, *submitted, 0);
            #[cfg(feature = "diagnostics")]
            if timing() {
                let times = runs[i].timed(self.frequency());
                report_list(&times, started.elapsed().as_secs_f64() * 1e3);
            }
            // SAFETY: a plain query.
            if groups.len() > 1 && removed_in.is_none() && unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                removed_in = Some(i);
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs),
                removed_during(runs, removed_in)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool)> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> = std::collections::HashMap::new();
        for list in runs {
            let deferred = list.deferred();
            // Shaders that read memory with no copy computed something else
            // than the console's: it stays out of title memory, where it broke
            // the title's own data (runs c1, c4).
            let missed = deferred.missed.iter().any(|check| check());
            for step in &deferred.copies {
                if let Step::GpuWrite(shadow, address, len, _) = step {
                    let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                    match index.get(&key) {
                        Some(&i) => back[i].3 &= !missed,
                        None => {
                            index.insert(key, back.len());
                            back.push((shadow.clone(), *address, *len, !missed));
                        }
                    }
                }
            }
        }
        // Pages shaders marked first: the copy-backs below end what the
        // copies keep for them.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (s.clone(), *a, *n))
            .collect();
        crate::shadow::copy_back_marks(&withheld);
        // What shaders wrote into copies in video memory, for the copy-backs
        // below to read from their mirrors: in the ranges with snapshots
        // (marked ones compare nothing; `copy_back_marks` pulled their pages).
        // Withheld ranges need nothing more for copies whose system memory
        // side stays current (`VICTORY_VRAM_COPIES`, `crate::vram`): those
        // declared unmarked have snapshots (`snapshots` takes every list's,
        // withheld or not), and `copy_back_marks` pulled the marked pages of
        // the others, withheld ones too. Brought back whole once more, they
        // cost a GPU copy of every withheld range, marked ones included, per
        // submission.
        crate::shadow::pull(&snapshots, crate::vram::Pulled::Declared);
        for (shadow, address, len, goes_back) in &back {
            let (address, len) = (*address, *len);
            if *goes_back {
                let written = shadow.copy_back(address, len);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            } else {
                shadow.keep_writes(address, len);
                crate::memory::note_withheld(len as u64);
            }
        }
        crate::shadow::snapshots_done(&snapshots);
        for list in runs {
            for work in &list.deferred().after {
                work();
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        // Host timestamps the lists resolved go on the title's GPU clock,
        // read before any write: a stand-in timestamp written first then
        // never reads earlier than a host one written after it.
        let stamped = |list: &&HostList| {
            let writes = &list.deferred().writes;
            writes.iter().any(|w| matches!(w.1, crate::queue::Written::HostTimestamp(_)))
        };
        let clock = runs.iter().any(stamped).then(|| self.clock()).flatten();
        // In recorded order, list by list: the last write to an address
        // stays, whether a query's result or a stand-in.
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                let what = match (what, &clock) {
                    (crate::queue::Written::HostTimestamp(raw), Some(c)) => {
                        // SAFETY: the list's mapped read-back memory, which
                        // the submission that ran resolved into.
                        let raw = unsafe { (raw as *const u64).read_volatile() };
                        crate::queue::Written::Value(c.convert(raw))
                    }
                    (crate::queue::Written::HostTimestamp(_), None) => crate::queue::Written::GpuTimestamp,
                    (what, _) => what,
                };
                what.write(at as *mut u8, bytes);
            }
        }
        // The submission has run and the next waits for our lock.
        release_retired();
        let error = self.log_messages();
        // A listed frame: what this submission's lists drew, as it is now
        // (the dumps submit lists of their own).
        #[cfg(feature = "diagnostics")]
        if traced {
            let sub = *submitted;
            drop(submitted);
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
        error
    }

    /// How host GPU timestamps map onto the title's GPU clock
    /// ([`crate::queue::Written::GpuTimestamp`]): the queue's calibration
    /// of its timestamps against the CPU's performance counter, which
    /// `Instant` reads too.
    fn clock(&self) -> Option<Clock> {
        use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
        let (mut gpu, mut cpu, mut now, mut cpu_frequency) = (0, 0, 0, 0);
        // SAFETY: plain queries into our locals.
        let frequency = unsafe {
            let frequency = self.queue.GetTimestampFrequency().ok()?;
            self.queue.GetClockCalibration(&mut gpu, &mut cpu).ok()?;
            QueryPerformanceCounter(&mut now);
            QueryPerformanceFrequency(&mut cpu_frequency);
            frequency
        };
        let title_now = crate::queue::gpu_clock();
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        // The title's clock when the host GPU's read `gpu`.
        let since = (i128::from(now) - i128::from(cpu)) * ticks / i128::from(cpu_frequency.max(1));
        (frequency > 0).then_some(Clock { gpu, at: i128::from(title_now) - since, frequency, now: title_now })
    }

    /// The queue's timestamp frequency (diagnostics).
    #[cfg(feature = "diagnostics")]
    fn frequency(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1)
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

/// A host GPU timestamp `gpu` taken when the title's GPU clock read `at`,
/// the host's ticks per second, and the title's clock when taken.
struct Clock {
    gpu: u64,
    at: i128,
    frequency: u64,
    now: u64,
}

impl Clock {
    /// A host GPU timestamp on the title's clock: never past the time it
    /// was read, so bottom-of-pipe timestamps written after stay later.
    fn convert(&self, raw: u64) -> u64 {
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        let since = (i128::from(raw) - i128::from(self.gpu)) * ticks / i128::from(self.frequency);
        (self.at + since).clamp(0, i128::from(self.now)) as u64
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

    /// The noted commands' GPU time, longest first, as (milliseconds, text)
    /// (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn timed(&self, frequency: u64) -> Vec<(f64, String)> {
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
        // The host closes no list with queries running; their title ends,
        // in a later list, are dropped (`end_query`).
        for (heap, kind, index, _) in self.queries.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
            // SAFETY: an open list and a query begun in it.
            unsafe { self.list.EndQuery(&heap, kind, index) };
        }
        for (from, to, bytes) in &self.deferred().read_backs {
            // SAFETY: an open list; buffers of ours it holds.
            unsafe {
                self.list.ResourceBarrier(&[transition(
                    from,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                self.list.CopyBufferRegion(to, 0, from, 0, *bytes);
            }
        }
        // A window of batched indirect clamps ends with the list.
        crate::indirect::end(self);
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

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        *self.deferred() = Deferred::default();
        crate::indirect::reset(self);
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.queries.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
        for pool in [&self.scratch, &self.tables] {
            pool.lock().unwrap_or_else(|e| e.into_inner()).rewind();
        }
        {
            let mut readback = self.readback.lock().unwrap_or_else(|e| e.into_inner());
            (readback.current, readback.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
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
    /// prelude's to prepare; a depth texture the list used before it had a
    /// twin (so none was noted) counts as used on its depth side, written.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
    ) -> Option<bool> {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                let key = std::sync::Arc::as_ptr(depth) as usize;
                if self.states.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&key) {
                    v.insert(crate::twin::Use::new(depth, crate::twin::DEPTH, true)).again(side, writes)
                } else {
                    v.insert(crate::twin::Use::new(depth, side, writes));
                    None
                }
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
    /// states become those the list ends with.
    fn prelude(&self) -> Option<ID3D12CommandList> {
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
        if barriers.is_empty() && copies.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
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
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data or GPU timing on): the device-removed report pairs the
    /// breadcrumbs' marker commands with the texts in order and names the
    /// commands a fault stopped among, and the timing report adds up the
    /// commands' GPU time by pipeline.
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
        // Timed notes are what `VICTORY_GPU_TIMING` reports on; markers are
        // only what the device-removed report pairs with breadcrumbs.
        if noting() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            if dred_on() {
                traced(&what);
                let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
                const UNICODE: u32 = 0;
                // SAFETY: an open list; the marker is copied when recorded.
                unsafe {
                    self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
                };
            }
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Begins query `index` of `heap` (occlusion or statistics, never a
    /// timestamp) unless it is running in the list already. Whether it
    /// began.
    pub(crate) fn begin_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> bool {
        let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        if kind == crate::query::TIMESTAMP || queries.iter().any(|(h, _, i, _)| h == heap && *i == index) {
            return false;
        }
        let kind = D3D12_QUERY_TYPE(kind as i32);
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.BeginQuery(heap, kind, index) };
        queries.push((heap.clone(), kind, index, false));
        drop(queries);
        self.keep(heap);
        true
    }

    /// Ends query `index` of `heap`: a timestamp is taken; another query
    /// ends only if it began in this list, as the type it began as. The
    /// type it ended as, if it did and its result is the host's to give: an
    /// occlusion query with a draw inside that the host did not run counts
    /// too few samples, and a title would cull by it what the draw shows.
    pub(crate) fn end_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> Option<u32> {
        let (kind, missed) = if kind == crate::query::TIMESTAMP {
            (D3D12_QUERY_TYPE(kind as i32), false)
        } else {
            let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
            let at = queries.iter().position(|(h, _, i, _)| h == heap && *i == index)?;
            let (_, kind, _, missed) = queries.swap_remove(at);
            (kind, missed)
        };
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.EndQuery(heap, kind, index) };
        self.keep(heap);
        let occlusion = matches!(kind.0 as u32, crate::query::OCCLUSION | crate::query::BINARY_OCCLUSION);
        (!(missed && occlusion)).then_some(kind.0 as u32)
    }

    /// A draw the title recorded was not run on the host (no pipeline yet,
    /// a shader that does not translate, ...): the queries running around
    /// it did not see it.
    pub(crate) fn missed_draw(&self) {
        for q in self.queries.lock().unwrap_or_else(|e| e.into_inner()).iter_mut() {
            q.3 = true;
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
    pub(crate) fn defer_heap(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, usize::MAX, 0);
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Heap(viewed.clone()));
        }
    }

    /// Shaders of the list read title memory through the page table of
    /// view heap `heap` (`crate::memory::prepare` at submission).
    pub(crate) fn defer_memory(&self, heap: usize) {
        self.deferred().copies.push(Step::Memory(heap));
    }

    /// After the list has run, before its writes go back: `check` says
    /// whether its shaders read title memory with no copy.
    pub(crate) fn defer_missed(&self, check: impl Fn() -> bool + Send + 'static) {
        self.deferred().missed.push(Box::new(check));
    }

    /// At the list's end (when it is closed), `bytes` of `from` (in the
    /// unordered-access state) are copied into the read-back buffer `to`.
    pub(crate) fn read_back_at_close(
        &self,
        from: &ID3D12Resource,
        to: &ID3D12Resource,
        bytes: u64,
    ) {
        self.deferred()
            .read_backs
            .push((from.clone(), to.clone(), bytes));
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes.
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + 'static) {
        self.deferred().after.push(Box::new(work));
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
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of read-back memory (8-byte aligned) until the list is
    /// reset, for what its query resolves write: the buffer, the offset in
    /// it and the CPU pointer, read once the list has run. The buffers are
    /// the list's own copy destinations, whatever the title's commands do
    /// with its buffers' states. Chunks are reused after a reset; a resolve
    /// too big for one gets a buffer of its own until then.
    pub(crate) fn readback(&self, len: u64) -> Option<(ID3D12Resource, u64, *const u8)> {
        let host = host()?;
        let len = len.max(1).checked_next_multiple_of(8)?;
        if len > READBACK_CHUNK {
            let (buffer, data) = host.readback_buffer(len)?;
            name(&buffer, || format!("list read-back {len:#x}"));
            self.deferred().uploads.push(buffer.clone());
            return Some((buffer, 0, data));
        }
        let mut r = self.readback.lock().unwrap_or_else(|e| e.into_inner());
        if r.chunks.get(r.current).is_some() && len > READBACK_CHUNK - r.used {
            (r.current, r.used) = (r.current + 1, 0);
        }
        if r.current == r.chunks.len() {
            let (buffer, data) = host.readback_buffer(READBACK_CHUNK)?;
            name(&buffer, || format!("list read-back {READBACK_CHUNK:#x}"));
            r.chunks.push((buffer, data as usize));
        }
        let (buffer, mapped) = &r.chunks[r.current];
        let result = (buffer.clone(), r.used, (mapped + r.used as usize) as *const u8);
        r.used += len;
        Some(result)
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        self.scratch_at(len).map(|(_, _, gpu, data)| (gpu, data))
    }

    /// As [`HostList::scratch`], with the upload buffer and the offset in
    /// it, for a batched clamp window's table
    /// (`crate::indirect`), from memory of its own: the room the table does
    /// not use is given back when the window closes.
    pub(crate) fn table(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        self.scratch_from(&self.tables, len)
            .map(|(buffer, at, _, data)| (buffer, at, data))
    }

    /// The table [`HostList::table`] gave last, `len` bytes at `at`, needs
    /// only its first `used` bytes: the rest goes to the next table.
    pub(crate) fn give_back_table(&self, at: u64, len: u64, used: u64) {
        let mut s = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        let (Some(len), Some(used)) = (
            len.max(1).checked_next_multiple_of(256),
            used.checked_next_multiple_of(256),
        ) else {
            return;
        };
        if s.used == at + len && used <= len {
            s.used = at + used;
        }
    }

    /// Upload memory the list holds until its next reset and keeps after
    /// it: scratch, clamp tables and buffers of their own (tests check it
    /// stays bounded).
    #[cfg(test)]
    pub(crate) fn upload_held(&self) -> u64 {
        let own: u64 = self
            .deferred()
            .uploads
            .iter()
            // SAFETY: a live buffer.
            .map(|b| unsafe { b.GetDesc() }.Width)
            .sum();
        own + [&self.scratch, &self.tables]
            .iter()
            .map(|p| p.lock().unwrap().held())
            .sum::<u64>()
    }

    /// Scratch memory: its buffer, offset, host GPU address and CPU pointer.
    fn scratch_at(&self, len: u64) -> Option<(ID3D12Resource, u64, u64, *mut u8)> {
        self.scratch_from(&self.scratch, len)
    }

    /// Memory from `pool`: its buffer, offset, host GPU address and CPU
    /// pointer.
    fn scratch_from(
        &self,
        pool: &Mutex<Scratch>,
        len: u64,
    ) -> Option<(ID3D12Resource, u64, u64, *mut u8)> {
        let host = host()?;
        let mut s = pool.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer.clone());
            return Some((buffer, 0, gpu, data));
        }
        while let Some(chunk) = s.chunks.get(s.current) {
            if len <= chunk.capacity - s.used {
                break;
            }
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let next = s.chunks.last().map_or(FIRST_SCRATCH_CHUNK, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(FIRST_SCRATCH_CHUNK).max(next).min(SCRATCH_CHUNK);
            let (buffer, data) = host.upload_buffer(capacity)?;
            name(&buffer, || format!("list scratch {capacity:#x}"));
            keep_resident(&buffer, Residency::Frame);
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            #[cfg(feature = "diagnostics")]
            tracing::debug!(target: "victory_gpu::scratch", list = ?std::ptr::from_ref(self),
                chunk = s.current, requested = len, capacity, "allocated list scratch");
            s.chunks.push(ScratchChunk { buffer, mapped: data as usize, gpu, capacity });
            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used;
        let data = (chunk.mapped + at as usize) as *mut u8;
        let result = (chunk.buffer.clone(), at, chunk.gpu + at, data);
        s.used += len;
        Some(result)
    }

    /// The list's indirect-work budget buffer and its host GPU address, zeroed
    /// at each submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<(&ID3D12Resource, u64)> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                name(&b, || "indirect budget".into());
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some((buffer, unsafe { buffer.GetGPUVirtualAddress() }))
    }

    /// The caller must have waited for the list to finish.
    #[cfg(test)]
    pub(crate) unsafe fn indirect_budget_words(&self) -> [u32; 2] {
        let (_, data) = self.budget.get().unwrap().as_ref().unwrap();
        // SAFETY: the live mapped buffer contains both words; the GPU is done.
        unsafe { (*data as *const [u32; 2]).read() }
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    pub(crate) fn dispatch_args(&self, bytes: u64) -> Option<ID3D12Resource> {
        let mut slot = self.dispatch_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, size)) = &*slot
            && *size >= bytes
        {
            return Some(buffer.clone());
        }
        let size = bytes.max(64 << 10).next_multiple_of(256);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
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
        let mut buffer = None;
        // SAFETY: a valid default-heap buffer description and out pointer.
        unsafe {
            host()?
                .device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .ok()?;
        }
        let buffer: ID3D12Resource = buffer?;
        if let Some((old, _)) = slot.replace((buffer.clone(), size)) {
            self.keep(&old);
        }
        Some(buffer)
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Run(Box::new(work)));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
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
                    Step::Run(_) | Step::Host(_) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(_) => C::StepTextureUs,
                    Step::Heap(_) => C::StepHeapUs,
                    Step::Memory(_) => C::StepMemoryUs,
                },
            );
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
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
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked) => {
                    // The shaders start from the title's data, which a
                    // snapshot must hold (marked ranges keep none).
                    if !marked {
                        batch.flush();
                    }
                    shadow.writing(*address, *len, *marked);
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        staging.refresh();
                    }
                    continue;
                }
                Step::Heap(viewed) => {
                    crate::bindless::refresh(viewed, |s, a, l| batch.add(s, a, l));
                    continue;
                }
                &Step::Memory(heap) => {
                    crate::memory::prepare(heap, |s, a, l| batch.add(s, a, l));
                    continue;
                }
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
///
/// A copy none of whose pages changed since it took them is not asked for
/// at all ([`crate::shadow::Shadow::unchanged`]): lists using title memory
/// ask for every copy there is, a few thousand in the world, of which a
/// submission's writes touch a few.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy's refresh is in `refreshes`, by copy.
    pending: std::collections::HashMap<usize, usize>,
    /// The range of each copy brought up to date since the last title
    /// write, by copy (held, so that its address names no other).
    current: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        // Nothing to copy anywhere in it, now or (but for a write that
        // comes meanwhile, which is the next submission's) at the flush.
        if shadow.unchanged() {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, len as u64);
            }
            return;
        }
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

    /// Does the refreshes; returns the bytes copied. Their copies of pages
    /// no shader writes are made together at the end, on several threads
    /// ([`crate::copy_pool`]); the time per range counts only finding them.
    fn flush(&mut self) -> usize {
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        self.pending.clear();
        let started = std::time::Instant::now();
        let (total, pieces) = crate::copy_pool::queued(|| self.update_all());
        crate::copy_pool::run(pieces);
        PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        total
    }

    /// [`Batch::flush`]'s updates, one per copy.
    fn update_all(&mut self) -> usize {
        let mut total = 0;
        for (shadow, from, to) in self.refreshes.drain(..) {
            let started = std::time::Instant::now();
            let copied = shadow.update(from, to - from);
            #[cfg(feature = "diagnostics")]
            crate::copystat::range(from, to - from, started.elapsed().as_micros() as u64, 0);
            #[cfg(not(feature = "diagnostics"))]
            let _ = started;
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
            Ok(()) => {
                let buffer = buffer?;
                crate::procmem::count_buffer(&buffer, crate::procmem::Kind::Upload, size);
                Some((buffer, data.cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }

    /// A buffer in video memory shaders may write (unordered access), in
    /// the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.buffer_in(
            D3D12_HEAP_TYPE_DEFAULT,
            size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
    }

    /// A buffer in the read-back heap, mapped for reading (what the GPU
    /// copied there, once the submission that copies it has run).
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let buffer = self.buffer_in(
            D3D12_HEAP_TYPE_READBACK,
            size,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, mapped whole.
        match unsafe { buffer.Map(0, None, Some(&mut data)) } {
            Ok(()) => {
                crate::procmem::count_buffer(&buffer, crate::procmem::Kind::Readback, size);
                Some((buffer, data.cast_const().cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    fn buffer_in(
        &self,
        heap: D3D12_HEAP_TYPE,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: heap,
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
            Flags: flags,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, heap = heap.Type.0, "no buffer");
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
    /// Frames since the last log line.
    logged_frames: u64,
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
            logged_frames: 0,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        self.logged_frames += 1;
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
        // Logs video memory use past nine tenths of the budget, once.
        crate::budget::watch();
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
        // Outside the submissions: in none of the times above.
        let waited = crate::pipeline::take_waited_ms() / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let root_views = crate::draw::take_root_view_slots() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        // Over the same frames too.
        let (table_bytes, table_copies) = crate::tables::take_copied();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame; \
                 {waited:.1} ms/frame of draws waiting for pipelines",
                making = crate::pipeline::making()
            );
            // Over every frame since the last line.
            let logged_frames = std::mem::take(&mut self.logged_frames);
            let d = crate::draw::take_dynamic_drops();
            let pool = crate::descriptor::fullest_dynamic().unwrap_or_default();
            // A capped heap's pool (the only kind short of slots) once it has
            // been more than half used, or anything dropped or taken back.
            let busy = pool.total < (1 << 18) && pool.low < pool.total / 2;
            if busy || d != crate::draw::DynamicDrops::default() {
                let f = d.failed_takes;
                tracing::info!(
                    "dynamic descriptor slots: dropped {} draws for root views and {} for tables \
                     in {logged_frames} frames ({:.1}/frame); tables refused {} times with the \
                     slots free but not in a row, {} with too few free; failed takes by size \
                     1: {}, 2-8: {}, 9-32: {}, 33-256: {}, more: {}; {} list runs shorter than \
                     256 slots; fullest view heap {} of {} free, peak {} in use, {} held by \
                     title memory view pairs; {} held by lists that have run; taken back {} \
                     slots of {} lists that had run; lists submitted again before a reset {}, \
                     {} of them not run (their slots taken back)",
                    d.root_views,
                    d.tables,
                    (d.root_views + d.tables) as f64 / logged_frames.max(1) as f64,
                    d.fragmented,
                    d.exhausted,
                    f[0],
                    f[1],
                    f[2],
                    f[3],
                    f[4],
                    d.short_runs,
                    pool.free,
                    pool.total,
                    pool.total - pool.low,
                    pool.pairs,
                    crate::draw::held_by_ran(),
                    d.slots_taken_back,
                    d.taken_back,
                    d.resubmitted,
                    d.not_run,
                );
            }
            // A capped heap's census: where the title keeps its view
            // descriptors and where shaders read them, to size its split
            // between title and dynamic slots (`VICTORY_DYNAMIC_SLOTS`).
            let (writes, reads, fresh) = (
                crate::descriptor::host::TITLE_WRITES.take_text(),
                crate::descriptor::SHADER_READS.take_text(),
                crate::draw::FRESH_TABLES.take_text(),
            );
            let table_slots = crate::draw::take_table_slots();
            if pool.total > 0 && pool.total < (1 << 18) {
                tracing::info!(
                    "view heap census (64 Ki bands from slot 0, since the last line): title writes {writes}; \
                     shader reads (per list) {reads}; fresh tables' slots, by their title table {fresh}; \
                     {table_slots} dynamic slots taken for fresh tables in {logged_frames} frames ({:.0}/frame)",
                    table_slots as f64 / logged_frames.max(1) as f64,
                );
            }
            let [clear_slots, clears_shared, clears_dropped] = crate::draw::take_clear_slots();
            if busy || clear_slots + clears_dropped > 0 {
                tracing::info!(
                    "dynamic descriptor slots for unordered access clears (stand-ins, ambient): \
                     {clear_slots} taken in {logged_frames} frames ({:.1}/frame), {clears_shared} \
                     clears shared a slot of their list, {clears_dropped} not run for want of one",
                    clear_slots as f64 / logged_frames.max(1) as f64,
                );
            }
            if table_copies > 0 {
                tracing::info!(
                    "host lookup tables copied into video memory: {:.2} MB/frame in {:.0} copies/frame",
                    table_bytes as f64 / 1e6 / frames,
                    table_copies as f64 / frames
                );
            }
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            let waiting = crate::resource::waiting_textures();
            let (let_go, let_go_bytes) = crate::resource::let_go_counts();
            let (staging, staging_bytes, staging_let_go, staging_let_go_bytes) = crate::texture::staging_counts();
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures waiting for a first use {waiting}, sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 placed textures let go while idle {let_go} ({:.0} MB); \
                 staging buffers {staging} ({:.0} MB), let go while idle {staging_let_go} ({:.0} MB); \
                 copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                let_go_bytes as f64 / 1e6,
                staging_bytes as f64 / 1e6,
                staging_let_go_bytes as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
            );
            let (marked, made) = crate::descriptor::take_usage();
            tracing::info!(
                "heap usage: {marked} heap slots marked read by shaders, {made} placed-texture views made for them (last 5 s); not made: {:?}, no heap {}; placed textures that failed to make: {}",
                crate::descriptor::host::PLACED_WHY.each_ref().map(|a| a.swap(0, Ordering::Relaxed)),
                crate::descriptor::USAGE.2.swap(0, Ordering::Relaxed),
                crate::descriptor::host::PLACED_FAILED.swap(0, Ordering::Relaxed),
            );
            crate::tiles::status();
            crate::threads::log();
            #[cfg(feature = "diagnostics")]
            {
                crate::procmem::log();
                // Frames since the last of these lines (about 5 s): the
                // tallies below cover them all. `frames` counts only the
                // last 1 s window, as `take_drops` does.
                let window = std::mem::take(&mut self.copy_stats_frames) as f64;
                crate::copystat::log(window);
                crate::bindless::log(window);
                crate::shadow::log_views(window);
                crate::vram::log(window);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
                costs::log(window);
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
            let resized = retry.attempt(Instant::now(), || {
                // SAFETY: no back buffer is held; every submission was waited for.
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
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
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

/// Where host copies go from the two switches' values (`upload`:
/// `VICTORY_GPU_UPLOAD`, `writes`: `VICTORY_GPU_UPLOAD_WRITES`, each off
/// unless set, `=1`) and whether the host has GPU upload heaps: (copies
/// shaders only read in GPU upload heaps, copies shaders write there too).
/// Without the heaps, or without `VICTORY_GPU_UPLOAD`, no copy goes there.
fn placement(upload: bool, writes: bool, heaps: bool) -> (bool, bool) {
    let read_only = upload && heaps;
    (read_only, read_only && writes)
}

/// Bytes of host copies in GPU upload heaps now ([`Host::gpu_upload_buffer`]):
/// charged when one is made, given back when D3D12 destroys it
/// ([`give_back_upload`]).
static UPLOAD_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Called by D3D12 when a host copy in a GPU upload heap is destroyed (its
/// last reference gone: the copy's, lists', and descriptor tables' clones
/// of the resource): its `size` (the callback's data) leaves
/// [`UPLOAD_BYTES`]. Counted for the whole run, the budget ran out after
/// enough buffers came and went and every later copy stayed in system
/// memory (64 copies of 32 MiB on an RTX 3050 Laptop, #2120).
unsafe extern "system" fn give_back_upload(size: *mut std::ffi::c_void) {
    UPLOAD_BYTES.fetch_sub(size as u64, Ordering::Relaxed);
}

/// [`UPLOAD_BYTES`] (tests, and the budget's report of what fills video
/// memory).
pub(crate) fn upload_bytes() -> u64 {
    UPLOAD_BYTES.load(Ordering::Relaxed)
}

/// Video memory host copies may take in GPU upload heaps; the rest stay in
/// system memory. Half of what the rest of the process leaves of the
/// budget the system gives it: the title's render targets and textures
/// come first, and past the budget the system would demote them as soon
/// as these copies. At most 2 GiB (the fixed size of before), or a quarter
/// of the budget on cards where that is more; 2 GiB on a host that does
/// not report a budget.
const UPLOAD_POOL: crate::budget::Pool = crate::budget::Pool {
    part: (1, 2),
    ceiling: 2 << 30,
    large: (1, 4),
};

/// Whether a host copy of `size` bytes may go in a GPU upload heap, with
/// `own` bytes of copies there already: within [`UPLOAD_POOL`]'s limit
/// under `budget`.
fn upload_fits(budget: Option<&crate::budget::Budget>, own: u64, size: u64) -> bool {
    let limit = budget.map_or(UPLOAD_POOL.ceiling, |b| b.limit(&UPLOAD_POOL, own));
    own + size <= limit
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.mapped_buffer(size, false)
    }

    /// A buffer in video memory that the CPU maps through the PCI BAR (a
    /// GPU upload heap, `D3D12_OPTIONS16`): shaders read it as fast as any
    /// video memory (on an RTX 5070 Ti ~720 GB/s against ~23 GB/s from
    /// system memory), the CPU writes it about as fast as system memory,
    /// and reading it from the CPU is very slow (~50 MB/s). For host copies
    /// of title memory that shaders only read. None when the host has no
    /// such heaps (no resizable BAR, older runtimes), `VICTORY_GPU_UPLOAD`
    /// is off, or the copies there would pass their share of the video
    /// memory budget ([`UPLOAD_POOL`]). Kept resident before what the title
    /// streams ([`keep_resident`]).
    pub(crate) fn gpu_upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if !self.gpu_upload() {
            return None;
        }
        let size = size.max(1);
        let budget = crate::budget::get();
        let own = UPLOAD_BYTES.fetch_add(size, Ordering::Relaxed);
        if !upload_fits(budget, own, size) {
            UPLOAD_BYTES.fetch_sub(size, Ordering::Relaxed);
            static FULL: std::sync::Once = std::sync::Once::new();
            FULL.call_once(|| {
                tracing::info!(
                    "host copies past their share of the video memory budget go to system memory"
                )
            });
            return None;
        }
        let Some((buffer, data)) = self.mapped_buffer(size, true) else {
            UPLOAD_BYTES.fetch_sub(size, Ordering::Relaxed);
            return None;
        };
        // The charge goes back when the buffer is destroyed, not when the
        // copy drops: lists and descriptor tables may hold it longer.
        // SAFETY: a live buffer; the callback's data is the size alone.
        let watched = unsafe {
            buffer
                .cast::<windows::Win32::Graphics::Direct3D::ID3DDestructionNotifier>()
                .and_then(|n| {
                    n.RegisterDestructionCallback(Some(give_back_upload), size as *const _)
                })
        };
        if watched.is_err() {
            // Not given back when destroyed, it would count for good.
            drop(buffer);
            UPLOAD_BYTES.fetch_sub(size, Ordering::Relaxed);
            static UNWATCHED: std::sync::Once = std::sync::Once::new();
            UNWATCHED.call_once(|| {
                tracing::info!(
                    "no destruction callback on this host: host copies stay in system memory"
                )
            });
            return None;
        }
        if let Some(b) = budget {
            b.charge(size);
        }
        keep_resident(&buffer, Residency::Frame);
        Some((buffer, data))
    }

    /// Whether host copies that shaders only read go to GPU upload heaps:
    /// `VICTORY_GPU_UPLOAD=1` (off by default) where the host has them
    /// ([`placement`]). Asked once.
    pub(crate) fn gpu_upload(&self) -> bool {
        self.upload_placement().0
    }

    /// Whether host copies that shaders write go to GPU upload heaps too,
    /// with a system memory mirror for the CPU (`crate::shadow`):
    /// `VICTORY_GPU_UPLOAD_WRITES=1` (off by default) with
    /// [`Host::gpu_upload`] ([`placement`]). Asked once.
    pub(crate) fn gpu_upload_writes(&self) -> bool {
        self.upload_placement().1
    }

    /// Where host copies of title buffers go in GPU upload heaps, decided
    /// once; logged once with P-0a's level (`crate::vram`)
    /// (at the host's creation, [`host`]): (copies shaders only read in GPU
    /// upload heaps, copies shaders write there too).
    fn upload_placement(&self) -> (bool, bool) {
        static PLACEMENT: OnceLock<(bool, bool)> = OnceLock::new();
        *PLACEMENT.get_or_init(|| {
            let upload = crate::switch::on("VICTORY_GPU_UPLOAD");
            let writes = crate::switch::on("VICTORY_GPU_UPLOAD_WRITES");
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: a feature query writing the structure of its size.
            let known = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut o).cast(),
                    size_of_val(&o) as u32,
                )
            }
            .is_ok();
            let heaps = known && o.GPUUploadHeapSupported.as_bool();
            let (read_only, written) = placement(upload, writes, heaps);
            tracing::info!(
                vram_copies = crate::vram::level(),
                upload_read_only = read_only,
                upload_written = written,
                gpu_upload_heaps = heaps,
                "host copies of title buffers: in video memory under its budget first \
                 (VICTORY_VRAM_COPIES=2 every copy, =1 read-only ones; off by \
                 default), then in GPU upload heaps (VICTORY_GPU_UPLOAD=1 for read-only \
                 copies, VICTORY_GPU_UPLOAD_WRITES=1 for written ones; off by default), \
                 then system memory"
            );
            (read_only, written)
        })
    }

    /// Copies `regions` (destination, its offset, source, its offset,
    /// bytes) between buffers on the GPU, now, and waits: for the GPU's
    /// writes into copies in video memory to reach their system memory
    /// mirrors before the CPU reads them (`crate::shadow::pull`). Runs in a
    /// list of its own after whatever the queue holds; callable from inside
    /// a submission's copy-back. Buffers start and end in the common state
    /// (buffers enter and leave the copy states implicitly).
    pub(crate) fn copy_now(&self, regions: &[(&ID3D12Resource, u64, &ID3D12Resource, u64, u64)]) {
        if regions.is_empty() {
            return;
        }
        struct Copier {
            allocator: ID3D12CommandAllocator,
            list: ID3D12GraphicsCommandList,
            fence: ID3D12Fence,
            event: HANDLE,
            value: u64,
        }
        // SAFETY: D3D12 objects are free-threaded; the event is waited on
        // under the mutex only.
        unsafe impl Send for Copier {}
        static COPIER: OnceLock<Option<Mutex<Copier>>> = OnceLock::new();
        let copier = COPIER.get_or_init(|| {
            // SAFETY: plain creation; the list is closed until used.
            let made = unsafe {
                (|| -> windows::core::Result<Copier> {
                    let allocator: ID3D12CommandAllocator =
                        self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList =
                        self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                    list.Close()?;
                    let fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                    let event = CreateEventW(None, false, false, None)?;
                    Ok(Copier { allocator, list, fence, event, value: 0 })
                })()
            };
            match made {
                Ok(c) => Some(Mutex::new(c)),
                Err(e) => {
                    tracing::warn!(error = %e, "no host copier: copies in video memory are not read back");
                    None
                }
            }
        });
        let Some(copier) = copier else {
            return;
        };
        let mut c = copier.lock().unwrap_or_else(|e| e.into_inner());
        c.value += 1;
        let value = c.value;
        // SAFETY: the allocator's last list finished (each call waits); the
        // buffers are live and their ranges inside them (the caller's).
        let ran = unsafe {
            c.allocator.Reset().and_then(|()| {
                c.list.Reset(&c.allocator, None)?;
                for &(dst, dst_at, src, src_at, bytes) in regions {
                    c.list.CopyBufferRegion(dst, dst_at, src, src_at, bytes);
                }
                c.list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(c.list.cast()?)]);
                self.queue.Signal(&c.fence, value)?;
                c.fence.SetEventOnCompletion(value, c.event)?;
                WaitForSingleObject(c.event, INFINITE);
                Ok(())
            })
        };
        if let Err(e) = ran {
            tracing::warn!(error = %e, "a host copy did not run");
        }
    }

    fn mapped_buffer(&self, size: u64, upload: bool) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = if upload {
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
                ..Default::default()
            }
        } else {
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            }
        };
        self.mapped_in(size, &heap, upload)
    }

    /// A buffer of `size` bytes in `heap`, mapped, that shaders read and
    /// write (`upload`: counted as in a GPU upload heap). Starts in the
    /// common state.
    pub(crate) fn mapped_in(
        &self,
        size: u64,
        heap: &D3D12_HEAP_PROPERTIES,
        upload: bool,
    ) -> Option<(ID3D12Resource, *mut u8)> {
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
                    heap,
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
            Ok(()) => {
                let buffer = buffer?;
                let kind = if upload {
                    crate::procmem::Kind::GpuUpload
                } else {
                    crate::procmem::Kind::Shared
                };
                crate::procmem::count_buffer(&buffer, kind, size);
                Some((buffer, data.cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, upload, "no shared buffer");
                None
            }
        }
    }
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A host texture with
/// one is held by its owner alone: no view heap slot keeps it
/// (`HostHeap::kept`) and none was retired ([`retire`]).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// The card's memory budget the system gives this process and how much of
/// it the process uses, in bytes (DXGI's local segment); none without an
/// adapter to ask. Past the budget the system moves the card's memory into
/// the PC's.
pub(crate) fn video_memory() -> Option<(u64, u64)> {
    struct Adapter(IDXGIAdapter3);
    // SAFETY: DXGI adapters are free-threaded.
    unsafe impl Send for Adapter {}
    // SAFETY: as above.
    unsafe impl Sync for Adapter {}
    static ADAPTER: OnceLock<Option<Adapter>> = OnceLock::new();
    let adapter = ADAPTER
        .get_or_init(|| {
            let host = host()?;
            // SAFETY: plain DXGI and device queries.
            unsafe {
                let luid = host.device.GetAdapterLuid();
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory
                    .EnumAdapterByLuid::<IDXGIAdapter3>(luid)
                    .ok()
                    .map(Adapter)
            }
        })
        .as_ref()?;
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: a plain query that fills our structure.
    unsafe {
        adapter
            .0
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
    }
    .ok()?;
    Some((info.Budget, info.CurrentUsage))
}

/// Host resources that view heap slots stopped viewing while a submission
/// may have been running ([`retire`]).
static RETIRED: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());

/// Keeps `resource` alive until the GPU work running now (or the next
/// submission's) has run: title threads rewrite view heap slots while the
/// GPU runs, and a shader that took the old descriptor still reads what it
/// views.
pub(crate) fn retire(resource: ID3D12Resource) {
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(resource);
}

/// After a submission has run, with none running: the retired resources go.
pub(crate) fn release_retired() {
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    drop(retired);
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

/// With `VICTORY_GPU_TIMING`: adds one list's noted commands to the cost
/// table, naming its slowest ones when it took over 50 ms on the GPU
/// (diagnostics). `wall_ms` is how long the list's submission took.
#[cfg(feature = "diagnostics")]
fn report_list(times: &[(f64, String)], wall_ms: f64) {
    costs::add(times, wall_ms);
    let total: f64 = times.iter().map(|(ms, _)| ms).sum();
    if total > 50.0 {
        tracing::warn!(
            ms = total,
            wall_ms,
            commands = times.len(),
            "slow host list"
        );
        for (ms, op) in times.iter().take(6) {
            tracing::warn!("slow list command, {ms:.1} ms: {op}");
        }
    }
}

/// What a note runs, for the GPU cost table: the kind of command, its size,
/// and the pipeline it binds ("draw 36x1 [vs 12ab, ps 34cd]", without the
/// views and tables that follow). Notes that name resources rather than a
/// pipeline keep the first words that are not addresses ("copy to",
/// "texture upload").
#[cfg(feature = "diagnostics")]
fn cost_label(text: &str) -> String {
    let head = text.split(" | ").next().unwrap_or(text);
    if head.contains('[') {
        // Addresses differ per draw; drop them so that the same pass adds up.
        let words: Vec<&str> = head
            .split(' ')
            .filter(|w| !w.starts_with("0x") && *w != "at")
            .collect();
        return words.join(" ").chars().take(160).collect();
    }
    let words: Vec<&str> = head
        .split(' ')
        .take_while(|w| w.chars().all(|c| c.is_ascii_alphabetic()))
        .collect();
    words.join(" ")
}

/// With `VICTORY_GPU_TIMING`: the GPU time the noted commands took since the
/// last report, added up by what they ran, so that a frame's cost can be read
/// off by pipeline (diagnostics).
#[cfg(feature = "diagnostics")]
mod costs {
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Commands, their GPU milliseconds, and the lists and wall milliseconds
    /// they came from, since the last report.
    #[derive(Default)]
    struct Tally {
        by_label: HashMap<String, (u64, f64)>,
        noted: f64,
        lists: u64,
        wall: f64,
    }

    static TALLY: Mutex<Option<Tally>> = Mutex::new(None);

    /// Adds one list's noted commands, as (GPU milliseconds, note) longest
    /// first, and how long the list took to run (wall milliseconds).
    pub(super) fn add(times: &[(f64, String)], wall_ms: f64) {
        let mut tally = TALLY.lock().unwrap_or_else(|e| e.into_inner());
        let tally = tally.get_or_insert_with(Tally::default);
        tally.lists += 1;
        tally.wall += wall_ms;
        for (ms, text) in times {
            let entry = tally.by_label.entry(super::cost_label(text)).or_default();
            entry.0 += 1;
            entry.1 += ms;
            tally.noted += ms;
        }
    }

    /// Logs the biggest items per frame since the last call, longest first.
    pub(super) fn log(frames: f64) {
        let Some(mut tally) = TALLY.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        if tally.lists == 0 {
            return;
        }
        let mut items: Vec<(String, (u64, f64))> = tally.by_label.drain().collect();
        items.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let noted = tally.noted.max(1e-6);
        let per_frame = |v: f64| v / frames;
        let commands: u64 = items.iter().map(|(_, (n, _))| *n).sum();
        let waited = if tally.wall > 0.0 {
            format!(", {:.1} ms wall/frame", per_frame(tally.wall))
        } else {
            String::new()
        };
        tracing::info!(
            "GPU cost by pipeline: {:.1} ms/frame over {:.0} lists ({:.0} commands/frame{waited})",
            per_frame(tally.noted),
            per_frame(tally.lists as f64),
            per_frame(commands as f64)
        );
        for (label, (n, ms)) in items.iter().take(16) {
            tracing::info!(
                "{:.2} ms/frame ({:.0}% of noted, {:.3} ms each, {:.1}/frame): {label}",
                per_frame(*ms),
                100.0 * ms / noted,
                ms / (*n).max(1) as f64,
                per_frame(*n as f64)
            );
        }
    }
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

    /// A copy whose memory nothing wrote since it was taken is left out of
    /// the batch (no refresh queued, none remembered); a write the guard
    /// takes or one that is reported brings it back, for the pages written.
    #[test]
    fn a_copy_nothing_wrote_is_not_asked_for() {
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
        assert_eq!((batch.refreshes.len(), batch.flush()), (1, len), "all of it at first");
        // Later lists and submissions: nothing queued, nothing kept.
        batch.add(&copy, p, len);
        batch.title_written();
        batch.add(&copy, p + PAGE, 100);
        let mut next = Batch::default();
        next.add(&copy, p, len);
        assert!(batch.refreshes.is_empty() && next.refreshes.is_empty());
        assert_eq!(next.flush(), 0);
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + 2 * PAGE + 8) as *mut u8).write_volatile(7) };
        next.add(&copy, p, len);
        assert_eq!((next.refreshes.len(), next.flush()), (1, PAGE), "the page written");
        assert_eq!(copy.read_host((p + 2 * PAGE + 8) as u64, 1)[0], 7);
        // A write of our own or the kernel's, reported.
        drop(crate::watch::writing(p + 3 * PAGE, 16));
        next.title_written();
        next.add(&copy, p, len);
        assert_eq!(next.flush(), PAGE, "the page reported");
        next.add(&copy, p, len);
        assert!(next.refreshes.is_empty());
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// A world frame's refreshes, on a made-up heap shaped like the world's
    /// (#400): 36 submissions, each after the title wrote 256 scattered
    /// pages of a 174 MB tracked heap, each declaring 24 ranges of it (of
    /// 64 views) as written by marking shaders, refreshed through `Batch`
    /// as `Host::execute` does (ranges armed, then one flush). Times the
    /// flushes with the copies on one thread and on the workers
    /// (`crate::copy_pool`), and checks every title write reached the copy.
    /// Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture world_frame_refreshes`.
    #[test]
    #[ignore = "benchmark"]
    fn world_frame_refreshes() {
        use crate::watch::PAGE;
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};
        const HEAP: usize = 174 << 20;
        const SUBMISSIONS: usize = 36;
        const DECLARED: usize = 24;
        const VIEWS: usize = 64;
        const TITLE_PAGES: usize = 256;
        const FRAMES: usize = 6;
        let p = crate::watch::tests::pool(HEAP / PAGE);
        crate::shadow::register(p, p as u64, HEAP as u64);
        let Some(copy) = crate::shadow::find(p as u64, HEAP as u64) else {
            crate::shadow::unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        // Views of the heap: 256-byte aligned, 64 KiB to 32 MiB.
        let views: Vec<(usize, usize)> = (0..VIEWS)
            .map(|_| {
                let len = (64 << 10) << next(10);
                let at = next((HEAP - len) / 256) * 256;
                (p + at, len)
            })
            .collect();
        let mut batch = Batch::default();
        batch.add(&copy, p, HEAP);
        batch.flush();
        let mut stamp = 1u64;
        let mut frame = |serial: bool| -> (Duration, usize) {
            crate::copy_pool::SERIAL.store(serial, Ordering::Relaxed);
            let (mut took, mut copied) = (Duration::ZERO, 0);
            for _ in 0..SUBMISSIONS {
                let writes: Vec<usize> = (0..TITLE_PAGES)
                    .map(|_| p + next(HEAP / PAGE) * PAGE + next(PAGE / 8) * 8)
                    .collect();
                for &at in &writes {
                    // SAFETY: 8 aligned bytes of our pool.
                    unsafe { (at as *mut u64).write_volatile(stamp) };
                    stamp += 1;
                }
                let declared: Vec<(usize, usize)> =
                    (0..DECLARED).map(|_| views[next(VIEWS)]).collect();
                let mut batch = Batch::default();
                for &(at, len) in &declared {
                    batch.add(&copy, at, len);
                    copy.writing(at, len, true);
                }
                let started = Instant::now();
                copied += batch.flush();
                took += started.elapsed();
                // Every title write inside a declared range is in the copy.
                for &at in &writes {
                    if declared.iter().any(|&(a, n)| a <= at && at + 8 <= a + n) {
                        // SAFETY: 8 aligned bytes of our pool.
                        let title = unsafe { (at as *const u64).read_volatile() };
                        let host = copy.read_host(at as u64, 2);
                        assert_eq!(
                            u64::from(host[0]) | u64::from(host[1]) << 32,
                            title,
                            "{at:#x}"
                        );
                    }
                }
                for &(at, len) in &declared {
                    copy.copy_back(at, len);
                }
            }
            (took, copied)
        };
        let mut report = |serial: bool| {
            frame(serial);
            let mut ms: Vec<f64> = (0..FRAMES)
                .map(|_| {
                    let (took, copied) = frame(serial);
                    println!(
                        "  {}: {:.1} ms, {} MB copied",
                        if serial { "one thread" } else { "workers" },
                        took.as_secs_f64() * 1e3,
                        copied >> 20
                    );
                    took.as_secs_f64() * 1e3
                })
                .collect();
            ms.sort_by(f64::total_cmp);
            ms[FRAMES / 2]
        };
        let serial = report(true);
        let workers = report(false);
        println!("refresh per frame, median: one thread {serial:.1} ms, workers {workers:.1} ms");
        crate::copy_pool::SERIAL.store(false, Ordering::Relaxed);
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
mod frame_bench;
