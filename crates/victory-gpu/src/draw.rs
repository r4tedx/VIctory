//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, depth bounds, topology, targets) is set on the host
//! list as the title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// None leaves the host list's default bounds (0..1) in place.
    depth_bounds: Option<(f32, f32)>,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
    /// Root SRVs bound as one-view tables in this list: (host heap, title
    /// address) → dynamic slot.
    root_srv_slots: std::collections::HashMap<(usize, u64), u32>,
    /// Root UAVs bound as one-view tables in this list: (host heap, title
    /// address, the host copy viewed or 0 for a null view) → dynamic slot.
    root_uav_slots: std::collections::HashMap<(usize, u64, usize), u32>,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootView {
    Cbv,
    Srv,
    Uav,
}

/// Permanently mapped zero-filled read-only fallback buffer (for unset or
/// unmapped root CBVs/SRVs) and writable sink buffer (for unset or unmapped
/// root UAVs, kept separate so shader writes never dirty the zero buffer).
fn fallback_root_va(uav: bool) -> Option<u64> {
    const SIZE: u64 = 4 << 20;
    static ZERO: std::sync::OnceLock<Option<(ID3D12Resource, u64)>> = std::sync::OnceLock::new();
    static SINK: std::sync::OnceLock<Option<(ID3D12Resource, u64)>> = std::sync::OnceLock::new();
    let slot = if uav { &SINK } else { &ZERO };
    slot.get_or_init(|| {
        let host = crate::host::host()?;
        let (buffer, data) = if uav {
            host.shared_buffer(SIZE)?
        } else {
            host.upload_buffer(SIZE)?
        };
        let label = if uav {
            "fallback sink root uav"
        } else {
            "fallback zero root view"
        };
        crate::host::name(&buffer, || label.into());
        // SAFETY: `data` points to `SIZE` mapped writable bytes, and `buffer` is live.
        let gpu = unsafe {
            data.write_bytes(0, SIZE as usize);
            buffer.GetGPUVirtualAddress()
        };
        Some((buffer, gpu))
    })
    .as_ref()
    .map(|&(_, gpu)| gpu)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. Constant buffers refresh up to the D3D12 64 KiB
/// maximum; SRVs and UAVs refresh up to [`ROOT_SRV_MAX`] / [`ROOT_UAV_MAX`]
/// of the backing buffer (dirty-page watched) and UAVs also schedule
/// write-back of that range (see [`root_len`]).
fn root_view(list: &HostList, address: u64, kind: RootView) -> Option<u64> {
    if address == 0 {
        return None;
    }
    let shadow = crate::shadow::find(address, 1)?;
    let gpu = shadow.gpu(address);
    match kind {
        RootView::Cbv => {
            let len = root_len(shadow.left(address), RootView::Cbv);
            if gpu.is_multiple_of(256) {
                shadow.refresh(list, address, len);
                list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
                return Some(gpu);
            }
            let (gpu, to) = list.scratch(len)?;
            list.defer_fill(to, address as *const u8, len as usize);
            Some(gpu)
        }
        RootView::Srv => {
            if !gpu.is_multiple_of(4) {
                return None;
            }
            let len = root_len(shadow.left(address), RootView::Srv);
            shadow.refresh(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            Some(gpu)
        }
        RootView::Uav => {
            if !gpu.is_multiple_of(4) {
                return None;
            }
            let len = root_len(shadow.left(address), RootView::Uav);
            shadow.refresh(list, address, len);
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
            Some(gpu)
        }
    }
}

fn emit_uav_barrier(list: &HostList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: an open command list and a global UAV barrier.
    unsafe { list.list.ResourceBarrier(&[barrier]) };
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Largest root SRV range refreshed per bind.
const ROOT_SRV_MAX: u64 = 4 << 20;
/// Largest root UAV range refreshed, kept and copied back per submission.
const ROOT_UAV_MAX: u64 = 2 << 20;

/// The root SRV refresh cap in use: [`ROOT_SRV_MAX`], or what
/// `VICTORY_ROOT_SRV_MB` says, in mebibytes (0: the rest of the copy). A
/// page of a host copy that no bind ever refreshed reads zeros, so a
/// program reading a root SRV past the cap gets zeros without any refusal
/// or fault. The switch is there to measure whether the world does: compare
/// a run at the default with one at 0.
fn root_srv_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| root_srv_cap_from(std::env::var("VICTORY_ROOT_SRV_MB").ok().as_deref()))
}

/// [`root_srv_cap`] for the switch's value; anything but a number is the
/// default.
fn root_srv_cap_from(value: Option<&str>) -> u64 {
    match value.and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(0) => u64::MAX,
        Some(mb) => mb.saturating_mul(1 << 20),
        None => ROOT_SRV_MAX,
    }
}

/// The bytes of a root view's buffer copy (`left` remaining after its
/// address) that a bind refreshes: constant buffers up to the D3D12 64 KiB
/// maximum, SRVs and UAVs up to their caps, warning once when a cap cuts.
///
/// shortcut: a root view has no size, and the translated shader does not
/// yet report the range it reads or writes, so SRVs and UAVs are capped
/// rather than taking the rest of the copy: behind most title buffers is
/// one host copy of well over 100 MiB, and a UAV's range is kept and
/// compared on every submission. The UAV cap is the smaller for that.
fn root_len(left: u64, kind: RootView) -> u64 {
    let cap = match kind {
        RootView::Cbv => return left.min(1 << 16),
        RootView::Srv => root_srv_cap(),
        RootView::Uav => ROOT_UAV_MAX,
    };
    if left > cap {
        match kind {
            RootView::Srv => once!(left, cap, "a root SRV cut to its refresh cap"),
            _ => once!(left, cap, "a root UAV cut to its refresh and copy-back cap"),
        }
    }
    left.min(cap)
}

/// What a root UAV at title address `address` views when bound as a table
/// ([`State::root_uav_table`]): the host copy, the view's byte offset in
/// it and its length in bytes, a multiple of 4. The length is what a root
/// UAV descriptor refreshes and copies back ([`root_len`]), so a shader
/// stores nothing that the title does not get back. None (a null view)
/// for a null address, one outside mapped title memory, one whose range
/// holds no dword, and one not on 16 bytes of its copy: the host starts
/// raw views only there (the console on 4), and a view from the 16 bytes
/// below would shift every store and load.
/// shortcut: such an address is not served at all (warned of once); the
/// shader would have to add the difference to each offset.
fn root_uav_range(address: u64) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    if address == 0 {
        return None;
    }
    let shadow = crate::shadow::find(address, 1)?;
    // SAFETY: a live buffer.
    let base = unsafe { shadow.buffer.GetGPUVirtualAddress() };
    let offset = shadow.gpu(address) - base;
    if !offset.is_multiple_of(16) {
        once!(
            address = format_args!("{address:#x}"),
            "a root UAV not on 16 bytes gets a null view: its stores are dropped, its loads read zeros"
        );
        return None;
    }
    let len = root_len(shadow.left(address), RootView::Uav) & !3;
    (len != 0).then_some((shadow, offset, len))
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(texture, record.desc().as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The value of user-data dword `u` of a translated shader: a root
    /// argument, or a bound heap's guard slot.
    fn user_dword(&self, compute: bool, u: &victory_shader::emit::UserData) -> u32 {
        let guard = |samplers: bool| {
            let h = self.heaps[usize::from(samplers)];
            (h != 0)
                .then(|| crate::descriptor::heap(h as This).host())
                .flatten()
                .map_or(0, |h| h.guard_slot())
        };
        match u.root {
            RootType::ResourceHeap => guard(false),
            RootType::SamplerHeap => guard(true),
            _ => self.user_value(compute, u.slot, u.root, u.dword),
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some((min, max));
        if list.depth_bounds.is_none() {
            crate::host::warn_depth_bounds();
        }
        self.apply_depth_bounds(list);
    }

    fn apply_depth_bounds(&self, list: &HostList) {
        if let (Some(list), Some((min, max))) = (&list.depth_bounds, self.depth_bounds) {
            // SAFETY: the host supports depth bounds and the list is open.
            unsafe { list.OMSetDepthBounds(min, max) };
        }
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            // After `bind`: a shader may have written the buffer before.
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            // Also reached after the indirect clamp invalidates the pipeline.
            if !compute {
                self.apply_depth_bounds(list);
            }
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        let mut bound_params = 0u64;
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            let made = self.tables(list, &t.used, compute);
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                return false;
            }
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_dword(compute, u),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, kind) in [
                (&stage.root_cbvs, RootView::Cbv),
                (&stage.root_srvs, RootView::Srv),
                (&stage.root_uavs, RootView::Uav),
            ] {
                for &(slot, param) in views {
                    if param < 64 {
                        if bound_params & (1u64 << param) != 0 {
                            continue;
                        }
                        bound_params |= 1u64 << param;
                    }
                    let uav = kind == RootView::Uav;
                    let table = match kind {
                        RootView::Cbv => false,
                        RootView::Srv => crate::pipeline::host::bounded_root_srvs(),
                        RootView::Uav => crate::pipeline::host::bounded_root_uavs(),
                    };
                    if table {
                        // A one-view table (#493, and the same for UAVs):
                        // an unset or unmapped address gets a null view
                        // there, not a fallback.
                        let address = match self.roots_ref(compute).get(usize::from(slot)) {
                            Some(&Root::View(address)) => Some(address),
                            _ => {
                                once!(slot, "a shader reads a root view the title did not set");
                                None
                            }
                        };
                        let bound = if uav {
                            self.root_uav_table(list, address, param, compute)
                        } else {
                            self.root_srv_table(list, address, param, compute)
                        };
                        if !bound {
                            return false;
                        }
                        continue;
                    }
                    let gpu = match self.roots_ref(compute).get(usize::from(slot)) {
                        Some(&Root::View(address)) if address != 0 => {
                            match root_view(list, address, kind) {
                                Some(gpu) => gpu,
                                None => {
                                    once!(
                                        slot,
                                        address = format_args!("{address:#x}"),
                                        ?kind,
                                        "root view outside mapped title memory; binding zeroed fallback"
                                    );
                                    let Some(gpu) = fallback_root_va(uav) else {
                                        continue;
                                    };
                                    gpu
                                }
                            }
                        }
                        Some(Root::View(_)) => {
                            once!(
                                slot,
                                ?kind,
                                "null root view address; binding zeroed fallback"
                            );
                            let Some(gpu) = fallback_root_va(uav) else {
                                continue;
                            };
                            gpu
                        }
                        _ => {
                            once!(
                                slot,
                                ?kind,
                                "a shader reads a root view the title did not set; binding zeroed fallback"
                            );
                            let Some(gpu) = fallback_root_va(uav) else {
                                continue;
                            };
                            gpu
                        }
                    };
                    // SAFETY: an open list; `gpu` lies in a live host or fallback buffer.
                    unsafe {
                        match (compute, kind) {
                            (false, RootView::Cbv) => {
                                l.SetGraphicsRootConstantBufferView(param, gpu)
                            }
                            (false, RootView::Srv) => {
                                l.SetGraphicsRootShaderResourceView(param, gpu)
                            }
                            (false, RootView::Uav) => {
                                l.SetGraphicsRootUnorderedAccessView(param, gpu)
                            }
                            (true, RootView::Cbv) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, RootView::Srv) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, RootView::Uav) => {
                                l.SetComputeRootUnorderedAccessView(param, gpu)
                            }
                        }
                    }
                }
            }
            if let Some(param) = stage.stride_table {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.stride_table_gpu());
                if let Some(gpu) = gpu {
                    // SAFETY: an open list; the stride table buffer lives with the host heap.
                    unsafe {
                        if compute {
                            l.SetComputeRootShaderResourceView(param, gpu);
                        } else {
                            l.SetGraphicsRootShaderResourceView(param, gpu);
                        }
                    }
                } else {
                    once!("a shader reads the runtime stride table without a host view heap");
                }
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table (see
    /// [`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, or no host copy). One slot per address per list.
    /// False if no dynamic slot could be had (the draw is dropped).
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
    ) -> bool {
        let Some((heap, host)) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find_map(|h| h.host().filter(|x| !x.is_sampler()).map(|x| (h, x)))
        else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let index = match self.root_srv_slots.get(&key) {
            Some(&index) => index,
            None => {
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_srv_slots.insert(key, index);
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                index
            }
        };
        if let Some((a, s)) = &shadow {
            // The bound a root SRV descriptor refreshes: the title's root
            // SRVs are read well past 1 MiB.
            s.refresh(list, *a, root_len(s.left(*a), RootView::Srv));
            // As [`root_view`]: the copy may be in its written state.
            list.buffer_state(&s.buffer, crate::host::BUFFER_READ);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root UAV as a one-view table (see
    /// [`crate::pipeline::host::bounded_root_uavs`]): a raw unordered
    /// access view of the range of the title buffer's host copy that
    /// [`root_uav_range`] gives, or a null view (stores dropped, loads
    /// zero) where it gives none. Does for that range what [`root_view`]
    /// does for a root UAV descriptor: refreshed, copied back to title
    /// memory after the submission, the copy put in its written state. One
    /// slot per address and copy per list. False if no dynamic slot could
    /// be had (the draw or dispatch is dropped).
    fn root_uav_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
    ) -> bool {
        let Some((heap, host)) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find_map(|h| h.host().filter(|x| !x.is_sampler()).map(|x| (h, x)))
        else {
            once!("a root UAV without a bound view heap: the draw or dispatch is dropped");
            return false;
        };
        let view = address.and_then(root_uav_range);
        // By copy as well as address: a view made of an address without a
        // copy (null), or of a copy the title has since released, must
        // not serve a later bind in the list that finds another.
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0),
            view.as_ref().map_or(0, |v| v.0.buffer.as_raw() as usize),
        );
        let index = match self.root_uav_slots.get(&key) {
            Some(&index) => index,
            None => {
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_uav_slots.insert(key, index);
                match &view {
                    Some((s, offset, len)) => host.raw_uav_at(index, &s.buffer, *offset, *len),
                    None => host.null_raw_uav_at(index),
                }
                index
            }
        };
        if let (Some(a), Some((s, _, len))) = (address, &view) {
            // As [`root_view`], for exactly the bytes of the view: what a
            // shader can store is what is copied back.
            s.refresh(list, a, *len);
            s.gpu_writes(list, a, *len);
            list.buffer_state(&s.buffer, crate::host::BUFFER_WRITE);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        // A shader emitted as one console wave per host group runs that
        // many host groups per console group, along x.
        let waves = group_waves(p);
        let Some(x) = x.checked_mul(waves).filter(|&x| x <= 65535 || waves == 1) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch of more host groups than D3D12 runs in a dimension dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        if p.has_root_uavs() {
            emit_uav_barrier(list);
        }
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_dword(compute, u);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage
                .root_cbvs
                .iter()
                .chain(&stage.root_srvs)
                .chain(&stage.root_uavs)
            {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                    // A root UAV bound as a table views less than the rest
                    // of its copy, or nothing.
                    if stage.root_uavs.iter().any(|r| r.0 == slot)
                        && crate::pipeline::host::bounded_root_uavs()
                    {
                        let _ = match root_uav_range(*address) {
                            Some((_, _, len)) => write!(text, " uav view {len:#x} bytes"),
                            None => write!(text, " null uav view"),
                        };
                    }
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    if crate::pipeline::host_pipeline(self.pipeline as This)
                        .is_some_and(|p| p.has_root_uavs())
                    {
                        emit_uav_barrier(list);
                    }
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u, &view);
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    fn tables(&mut self, list: &HostList, used: &[UsedResource], compute: bool) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA, 2,048 samplers anywhere)
        // are made afresh too: the title's slots there have no host slot of
        // their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                self.table_failed = true;
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view);
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES;
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            if crate::pipeline::host_pipeline(self.pipeline as This)
                .is_some_and(|p| p.has_root_uavs())
            {
                emit_uav_barrier(list);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let ran = self.topology != RECTANGLES && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            if crate::pipeline::host_pipeline(self.pipeline as This)
                .is_some_and(|p| p.has_root_uavs())
            {
                emit_uav_barrier(list);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        if kind == INDEXED && self.topology == RECTANGLES {
            once!("indirect indexed rectangles not drawn");
            why("indexed rectangles not supported");
            return false;
        }
        if kind == DRAW && self.topology == RECTANGLES {
            let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_drop(self.pipeline as This);
                return false;
            };
            if p.compute || p.mesh {
                why(&format!(
                    "pipeline mismatch (compute {}, mesh {}, topology {})",
                    p.compute, p.mesh, self.topology
                ));
                return false;
            }
            let Some(signature) = host_signature(DRAW, 16) else {
                why("no host signature");
                return false;
            };
            if max == 0 {
                DRAWN.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            let prepared = match crate::rectangles::expand(
                list,
                p,
                &self.vertex,
                (&args, arguments, stride, max),
                counter.as_ref().map(|(s, at)| (s, *at)),
            ) {
                Ok(p) => p,
                Err(reason) => {
                    why(reason);
                    return false;
                }
            };
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
            if !self.prepare(list, false) {
                why("prepare failed for indirect rectangles");
                return false;
            }
            prepared.before_draw(list);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                self.op_text(
                    format!("indirect rect draw max {max} at {arguments:#x}"),
                    false,
                )
            });
            // SAFETY: an open list with everything set; `prepared.args_buf` is
            // list-owned (records past the title's count are empty draws, so
            // no count buffer), the signature cached.
            unsafe {
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &prepared.args_buf,
                    0,
                    None::<&ID3D12Resource>,
                    0,
                )
            };
            prepared.after_draw(list);
            DRAWN.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        let dispatch = (kind == DISPATCH)
            .then(|| crate::pipeline::host_pipeline(self.pipeline as This))
            .flatten()
            .filter(|p| p.compute);
        let waves = dispatch.map_or(1, group_waves);
        let host_stride = if waves > 1 { 12 } else { stride };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, host_stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // The rewrite and the clamp both bind their own compute pipeline.
        let rewritten;
        let host_args = if waves > 1 {
            if max == 0 {
                return true;
            }
            let Some(buffer) = crate::indirect::split_dispatch(
                list,
                (&args.buffer, args.gpu(arguments)),
                counter.as_ref().map(|(s, at)| (&s.buffer, s.gpu(*at))),
                (stride, max, waves),
            ) else {
                why("split dispatch arguments could not be rewritten");
                return false;
            };
            rewritten = buffer;
            (&rewritten, 0)
        } else {
            if !crate::indirect::bound(list, &args.buffer, args.gpu(arguments), (kind, stride, max)) {
                why("not bounded (no clamp, or arguments not 4-byte aligned)");
                return false;
            }
            (&args.buffer, args.offset(arguments))
        };
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => dispatch.is_some_and(|p| self.bind(list, p, true)),
            _ => false,
        };
        if !ran {
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => crate::pipeline::note_drop(self.pipeline as This),
                Some(p) => why(&format!(
                    "prepare failed (compute {}, mesh {})",
                    p.compute, p.mesh
                )),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        crate::indirect::execute(
            list,
            &signature,
            max,
            host_args,
            counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at))),
        );
        if crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.has_root_uavs())
        {
            emit_uav_barrier(list);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// Host thread groups per console thread group of a compute pipeline
/// (`HostShader::group_waves`).
fn group_waves(p: &HostPipeline) -> u32 {
    p.stages
        .first()
        .map_or(1, |s| s.translated().host.group_waves.max(1))
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
        } else {
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(View::Unordered(r, ..)) => format!("uav {:?}", texture(*r as This)),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::native::{ComputeIo, StageIo};
    use victory_shader::{
        Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    struct Allocation {
        ptr: *mut u8,
    }

    impl Allocation {
        fn watched(len: usize) -> Self {
            // SAFETY: allocates a fresh write-watched virtual range freed in Drop.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            Self { ptr }
        }

        fn reserved_unmapped(len: usize) -> Self {
            // SAFETY: reserves an uncommitted VA range so accesses are unmapped.
            let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) }
                .cast::<u8>();
            assert!(!ptr.is_null());
            Self { ptr }
        }

        fn addr(&self) -> u64 {
            self.ptr as u64
        }
    }

    impl Drop for Allocation {
        fn drop(&mut self) {
            // SAFETY: `self.ptr` came from `VirtualAlloc`.
            unsafe {
                VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
            }
        }
    }

    /// Synthetic GFX10 compute shader reading Root CBV (slot 0, dword 0) and
    /// Root SRV (slot 1, byte offset `+1.5 MiB`), and storing `cbv ^ srv ^
    /// 0xA5A5_5A5A` to Root UAV (slot 2, at `+1.5 MiB + 4 * group_id.x`).
    fn build_root_view_cs() -> (victory_shader::emit::HostShader, Vec<UsedResource>) {
        let s_mov_b32 =
            |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32;
        let s_load_dword = |dst: u8, base: u8, byte_off: u32| -> [u32; 2] {
            [
                0x3D << 26 | (dst as u32) << 6 | (base as u32 >> 1),
                125 << 25 | (byte_off & 0x1F_FFFF),
            ]
        };
        let vop1 = |op: u16, vdst: u8, src0: u16| -> u32 {
            0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        let buffer_load_dword_offen = |vdst: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x0C << 18 | (1 << 12),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8 | vaddr as u32,
            ]
        };
        let buffer_store_dword_offen = |vdata: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x1C << 18 | (1 << 12),
                0x80 << 24
                    | ((srsrc as u32 >> 2) & 0x1F) << 16
                    | (vdata as u32) << 8
                    | vaddr as u32,
            ]
        };

        let mut words = Vec::new();
        // s[0:1] = CBV 0, s[2:3] = SRV 1, s[4:5] = UAV 2, s6 = group id X.
        words.push(vop1(0x001, 3, 6));
        words.extend_from_slice(&s_load_dword(6, 0, 0));
        // SRV descriptor in s[8:11], UAV descriptor in s[12:15].
        words.push(s_mov_b32(8, 2));
        words.push(s_mov_b32(9, 3));
        words.push(s_mov_b32(10, 0xC1));
        words.extend_from_slice(&[s_mov_b32(11, 0xFF), 0x3101_4FAC]);
        words.push(s_mov_b32(12, 4));
        words.push(s_mov_b32(13, 5));
        words.push(s_mov_b32(14, 0xC1));
        words.extend_from_slice(&[s_mov_b32(15, 0xFF), 0x3101_4FAC]);
        let vop2 = |op: u8, vdst: u8, src0: u16, vsrc1: u8| -> u32 {
            (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        // v0 = 0x18_0000 (+1.5 MiB); v1 = srv[0x18_0000]; v2 = s6 ^ v1 ^ 0xA5A5_5A5A.
        words.extend_from_slice(&[vop1(0x001, 0, 0xFF), 0x0018_0000]);
        words.extend_from_slice(&buffer_load_dword_offen(1, 0, 8));
        words.push(vop2(0x1D, 2, 6, 1)); // v_xor_b32 v2, s6, v1
        words.extend_from_slice(&[vop2(0x1D, 2, 0xFF, 2), 0xA5A5_5A5A]); // v_xor_b32 v2, literal, v2
        words.push(vop2(0x1A, 3, 0x82, 3)); // v3 = group id X * 4
        words.push(vop2(0x25, 3, 256, 3)); // v3 += 1.5 MiB
        words.extend_from_slice(&buffer_store_dword_offen(2, 3, 12));
        words.push(0xBF81_0000); // s_endpgm

        let used = vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 4,
                slot: 2,
            },
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 1,
                    user_data: 2,
                    slot: 1,
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 2,
                    user_data: 4,
                    slot: 2,
                },
            ],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: (6 << 1) | (1 << 7),
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = victory_shader::ir::build(&native.code, false, 8).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let res =
            victory_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let c = Container {
            digest: [0; 16],
            version: (1, 0),
            parts: Vec::new(),
        };
        let shader = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res, victory_shader::emit::Target::ANY).unwrap();
        (shader, used)
    }

    /// What [`build_root_uav_cs`]'s program stores in range, before the
    /// exclusive-or with what it loads past the view.
    const UAV_IN_RANGE: u32 = 0x5EED_0001;
    /// What it stores out of range.
    const UAV_OUT_OF_RANGE: u32 = 0x0BAD_0002;
    /// Byte offsets from the root UAV's address: its in-range store, its
    /// load and store past a 2 MiB view, and its store far past any copy.
    const UAV_INSIDE: usize = 0x18_0000;
    const UAV_PAST: usize = 0x28_0000;
    const UAV_FAR: u32 = 0x3000_0000;

    /// Synthetic GFX10 compute shader with one root UAV (slot 0). It loads
    /// the dword at `+2.5 MiB` ([`UAV_PAST`]), stores that exclusive-or
    /// [`UAV_IN_RANGE`] at `+1.5 MiB` ([`UAV_INSIDE`]), then stores
    /// [`UAV_OUT_OF_RANGE`] at `+2.5 MiB` and at `+768 MiB` ([`UAV_FAR`]).
    fn build_root_uav_cs() -> (victory_shader::emit::HostShader, Vec<UsedResource>) {
        let s_mov_b32 =
            |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32;
        let v_mov_b32 =
            |vdst: u8, value: u32| [0x7E00_0000 | (vdst as u32) << 17 | 0x001 << 9 | 0xFF, value];
        let v_xor_b32 = |vdst: u8, value: u32, vsrc1: u8| {
            [
                0x1D << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | 0xFF,
                value,
            ]
        };
        let buffer_load_dword_offen = |vdst: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x0C << 18 | (1 << 12),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8 | vaddr as u32,
            ]
        };
        let buffer_store_dword_offen = |vdata: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x1C << 18 | (1 << 12),
                0x80 << 24
                    | ((srsrc as u32 >> 2) & 0x1F) << 16
                    | (vdata as u32) << 8
                    | vaddr as u32,
            ]
        };

        let mut words = Vec::new();
        // s[0:1] = UAV 0; its descriptor in s[12:15].
        words.push(s_mov_b32(12, 0));
        words.push(s_mov_b32(13, 1));
        words.push(s_mov_b32(14, 0xC1));
        words.extend_from_slice(&[s_mov_b32(15, 0xFF), 0x3101_4FAC]);
        // v1 = uav[+2.5 MiB]; uav[+1.5 MiB] = v1 ^ UAV_IN_RANGE.
        words.extend_from_slice(&v_mov_b32(0, UAV_PAST as u32));
        words.extend_from_slice(&buffer_load_dword_offen(1, 0, 12));
        words.extend_from_slice(&v_xor_b32(2, UAV_IN_RANGE, 1));
        words.extend_from_slice(&v_mov_b32(3, UAV_INSIDE as u32));
        words.extend_from_slice(&buffer_store_dword_offen(2, 3, 12));
        // uav[+2.5 MiB] = uav[+768 MiB] = UAV_OUT_OF_RANGE.
        words.extend_from_slice(&v_mov_b32(4, UAV_OUT_OF_RANGE));
        words.extend_from_slice(&buffer_store_dword_offen(4, 0, 12));
        words.extend_from_slice(&v_mov_b32(5, UAV_FAR));
        words.extend_from_slice(&buffer_store_dword_offen(4, 5, 12));
        words.push(0xBF81_0000); // s_endpgm

        let used = vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 2,
            user_data: 0,
            slot: 0,
        }];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = victory_shader::ir::build(&native.code, false, 8).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let res =
            victory_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let c = Container {
            digest: [0; 16],
            version: (1, 0),
            parts: Vec::new(),
        };
        let target = victory_shader::emit::Target::ANY;
        let shader = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res, target).unwrap();
        assert_eq!(shader.root_uavs, vec![0]);
        (shader, used)
    }

    /// Byte offsets from the root SRV's address that [`build_root_srv_cs`]'s
    /// program loads: under 1 MiB, and at `+1.5 MiB`.
    const SRV_NEAR: usize = 0x8_0000;
    const SRV_FAR: usize = 0x18_0000;
    /// Byte offsets from its root UAV's address where it stores the two.
    const SRV_NEAR_OUT: usize = 16;
    const SRV_FAR_OUT: usize = 32;

    /// Synthetic GFX10 compute shader with a root SRV (slot 0) and a root
    /// UAV (slot 1). It stores the SRV's dwords at [`SRV_NEAR`] and
    /// [`SRV_FAR`] at [`SRV_NEAR_OUT`] and [`SRV_FAR_OUT`] of the UAV,
    /// unchanged.
    fn build_root_srv_cs() -> (victory_shader::emit::HostShader, Vec<UsedResource>) {
        let s_mov_b32 =
            |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32;
        let v_mov_b32 =
            |vdst: u8, value: u32| [0x7E00_0000 | (vdst as u32) << 17 | 0x001 << 9 | 0xFF, value];
        let buffer_load_dword_offen = |vdst: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x0C << 18 | (1 << 12),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8 | vaddr as u32,
            ]
        };
        let buffer_store_dword_offen = |vdata: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x1C << 18 | (1 << 12),
                0x80 << 24
                    | ((srsrc as u32 >> 2) & 0x1F) << 16
                    | (vdata as u32) << 8
                    | vaddr as u32,
            ]
        };

        let mut words = Vec::new();
        // s[0:1] = SRV 0, s[2:3] = UAV 1; their descriptors in s[8:11] and
        // s[12:15].
        words.push(s_mov_b32(8, 0));
        words.push(s_mov_b32(9, 1));
        words.push(s_mov_b32(10, 0xC1));
        words.extend_from_slice(&[s_mov_b32(11, 0xFF), 0x3101_4FAC]);
        words.push(s_mov_b32(12, 2));
        words.push(s_mov_b32(13, 3));
        words.push(s_mov_b32(14, 0xC1));
        words.extend_from_slice(&[s_mov_b32(15, 0xFF), 0x3101_4FAC]);
        // uav[SRV_NEAR_OUT] = srv[SRV_NEAR]; uav[SRV_FAR_OUT] = srv[SRV_FAR].
        for (n, (from, to)) in [(SRV_NEAR, SRV_NEAR_OUT), (SRV_FAR, SRV_FAR_OUT)]
            .into_iter()
            .enumerate()
        {
            let v = 3 * n as u8;
            words.extend_from_slice(&v_mov_b32(v, from as u32));
            words.extend_from_slice(&buffer_load_dword_offen(v + 1, v, 8));
            words.extend_from_slice(&v_mov_b32(v + 2, to as u32));
            words.extend_from_slice(&buffer_store_dword_offen(v + 1, v + 2, 12));
        }
        words.push(0xBF81_0000); // s_endpgm

        let view = |kind, slot: u8| UsedResource {
            kind,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 2,
            user_data: 2 * slot,
            slot,
        };
        let used = vec![view(ResourceType::Srv, 0), view(ResourceType::Uav, 1)];
        let param = |slot: u8| RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: u32::from(slot),
            user_data: 2 * slot,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![param(0), param(1)],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 4 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = victory_shader::ir::build(&native.code, false, 8).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let res =
            victory_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let c = Container {
            digest: [0; 16],
            version: (1, 0),
            parts: Vec::new(),
        };
        let target = victory_shader::emit::Target::ANY;
        let shader = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res, target).unwrap();
        assert_eq!(shader.root_srvs, vec![0]);
        assert_eq!(shader.root_uavs, vec![1]);
        (shader, used)
    }

    /// Whether the host's device is still there.
    fn device_alive(host: &crate::host::Host) -> bool {
        // SAFETY: a live device.
        unsafe { host.device.GetDeviceRemovedReason() }.is_ok()
    }

    #[test]
    fn null_and_unmapped_root_views_use_safe_distinct_fallbacks() {
        assert!(crate::shadow::find(0, 1).is_none());
        assert!(crate::shadow::find(0x1000, 64).is_none());
        assert!(crate::shadow::find(u64::MAX - 16, 1).is_none());
        assert!(crate::shadow::find(u64::MAX - (1 << 20), 1).is_none());

        let unmapped = Allocation::reserved_unmapped(8 << 20);
        let mid = unmapped.addr() + (4 << 20);
        assert!(crate::shadow::find(mid, 64).is_none());

        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().expect("new_list");
        assert!(root_view(&list, 0, RootView::Cbv).is_none());
        assert!(root_view(&list, 0, RootView::Srv).is_none());
        assert!(root_view(&list, 0, RootView::Uav).is_none());
        assert!(root_view(&list, mid, RootView::Srv).is_none());
        assert!(root_view(&list, mid, RootView::Uav).is_none());
        assert!(root_view(&list, u64::MAX - 16, RootView::Srv).is_none());
        assert!(root_view(&list, u64::MAX - 16, RootView::Uav).is_none());

        let zero_va = fallback_root_va(false).expect("zero fallback VA");
        let sink_va = fallback_root_va(true).expect("sink fallback VA");
        assert_ne!(zero_va, 0);
        assert_ne!(sink_va, 0);
        assert_ne!(zero_va, sink_va);
        assert!(zero_va.is_multiple_of(256));
        assert!(sink_va.is_multiple_of(256));
        list.close();
    }

    #[test]
    fn root_view_ranges_are_capped() {
        let big = 1 << 30;
        assert_eq!(root_len(big, RootView::Cbv), 64 << 10);
        assert_eq!(root_len(big, RootView::Srv), root_srv_cap().min(big));
        // The switch: mebibytes, 0 for the whole copy, the default otherwise.
        assert_eq!(root_srv_cap_from(None), ROOT_SRV_MAX);
        assert_eq!(root_srv_cap_from(Some("64")), 64 << 20);
        assert_eq!(root_srv_cap_from(Some(" 0 ")), u64::MAX);
        assert_eq!(root_srv_cap_from(Some("all")), ROOT_SRV_MAX);
        assert_eq!(root_len(big, RootView::Uav), ROOT_UAV_MAX);
        // A shorter remaining extent is taken whole.
        for kind in [RootView::Cbv, RootView::Srv, RootView::Uav] {
            assert_eq!(root_len(4096, kind), 4096);
        }
    }

    #[test]
    fn root_srv_refreshes_dirty_data_past_one_mib() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let len = 2 << 20;
        let alloc = Allocation::watched(len);
        let base = alloc.addr();
        let resource_id = alloc.ptr as usize;
        crate::shadow::register(resource_id, base, len as u64);

        let off_a = 0x18_0000usize; // +1.5 MiB
        let off_b = 0x1C_0000usize; // +1.75 MiB
        // SAFETY: `off_a` and `off_b` lie within the 2 MiB committed allocation.
        unsafe {
            alloc.ptr.add(off_a).cast::<u32>().write(0xCAFE_1234);
        }

        let list = host.new_list().expect("new_list");
        let shadow = crate::shadow::find(base, 4).expect("shadow");
        assert_eq!(
            root_view(&list, base, RootView::Srv),
            Some(shadow.gpu(base))
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(off_a), 0xCAFE_1234);

        // Second submission mutates another page past +1 MiB and verifies
        // incremental dirty-page refresh updates it too.
        list.reset();
        // SAFETY: within the 2 MiB allocation.
        unsafe {
            alloc.ptr.add(off_b).cast::<u32>().write(0xBEEF_5678);
        }
        assert_eq!(
            root_view(&list, base, RootView::Srv),
            Some(shadow.gpu(base))
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(off_a), 0xCAFE_1234);
        assert_eq!(shadow.read_u32(off_b), 0xBEEF_5678);

        crate::shadow::unregister(resource_id, base);
    }

    #[test]
    fn dispatch_binds_fallbacks_and_copies_root_uav_writes_back() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (shader, used) = build_root_view_cs();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        let unmapped = Allocation::reserved_unmapped(8 << 20);
        let mid_unmapped = unmapped.addr() + (4 << 20);

        let len = 2 << 20;
        let srv_buf = Allocation::watched(len);
        let uav_buf = Allocation::watched(len);
        let cbv_buf = Allocation::watched(64 << 10);
        crate::shadow::register(srv_buf.ptr as usize, srv_buf.addr(), len as u64);
        crate::shadow::register(uav_buf.ptr as usize, uav_buf.addr(), len as u64);
        crate::shadow::register(cbv_buf.ptr as usize, cbv_buf.addr(), 64 << 10);

        let off = 0x18_0000usize; // +1.5 MiB
        // SAFETY: offsets are inside the committed allocations.
        unsafe {
            cbv_buf.ptr.cast::<u32>().write(0x1357_2468);
            srv_buf.ptr.add(off).cast::<u32>().write(0xDEAD_BEEF);
        }

        // 1) Dispatch with slot 0 unset, slot 1 = null (0x0), slot 2 = unmapped VA:
        //    the CBV reads the ZERO fallback and the SRV a null view, both as
        //    0; the UAV store at +1.5 MiB must neither fault nor dirty what
        //    they read. As a one-view table the UAV is a null view, which
        //    drops the store; with VICTORY_ROOT_UAV_DIRECT=1 the SINK
        //    fallback absorbs it.
        let list = host.new_list().expect("new_list");
        // Root SRVs and UAVs are one-view tables in the bound view heap.
        let heap = crate::descriptor::test_view_heap(16);
        let mut state = State::default();
        state.set_pipeline(pso);
        state.set_heaps(&[heap]);
        state.set_view(true, 1, 0);
        state.set_view(true, 2, mid_unmapped);
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);

        // 2) Dispatch with CBV & SRV still unbound (verifying they still read
        //    0 after step 1's store!) and UAV bound to `uav_buf`:
        //    uav_buf[+1.5 MiB] must receive 0 ^ 0 ^ 0xA5A5_5A5A = 0xA5A5_5A5A
        //    in CPU title memory.
        list.reset();
        state.reset(pso);
        state.set_heaps(&[heap]);
        state.set_view(true, 2, uav_buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let from_zero = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(
            from_zero, 0xA5A5_5A5A,
            "the store through the unmapped root UAV must not dirty what unset views read"
        );

        // 3) Dispatch with all three root views bound (SRV reading +1.5 MiB and
        //    UAV writing +1.5 MiB) and verify full extent refresh + write-back.
        list.reset();
        state.reset(pso);
        state.set_heaps(&[heap]);
        state.set_view(true, 0, cbv_buf.addr());
        state.set_view(true, 1, srv_buf.addr());
        state.set_view(true, 2, uav_buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let got = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(got, 0x1357_2468 ^ 0xDEAD_BEEF ^ 0xA5A5_5A5A);

        // 4) Indirect dispatch (`execute_indirect`) with Root UAVs bound also
        //    emits its UAV barrier cleanly and copies updated writes back.
        let args_buf = Allocation::watched(64 << 10);
        crate::shadow::register(args_buf.ptr as usize, args_buf.addr(), 64 << 10);
        // SAFETY: updating CBV[0] and writing [1, 1, 1] dispatch args inside committed allocations.
        unsafe {
            cbv_buf.ptr.cast::<u32>().write(0x2468_ACE0);
            args_buf.ptr.cast::<[u32; 3]>().write([1, 1, 1]);
        }
        list.reset();
        state.reset(pso);
        state.set_heaps(&[heap]);
        state.set_view(true, 0, cbv_buf.addr());
        state.set_view(true, 1, srv_buf.addr());
        state.set_view(true, 2, uav_buf.addr());
        assert!(state.execute_indirect(&list, (2, 12, 1), args_buf.addr(), None));
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let indirect_got = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(indirect_got, 0x2468_ACE0 ^ 0xDEAD_BEEF ^ 0xA5A5_5A5A);
        crate::shadow::unregister(args_buf.ptr as usize, args_buf.addr());

        crate::shadow::unregister(srv_buf.ptr as usize, srv_buf.addr());
        crate::shadow::unregister(uav_buf.ptr as usize, uav_buf.addr());
        crate::shadow::unregister(cbv_buf.ptr as usize, cbv_buf.addr());
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(pso, 0x10);
        assert_eq!(release(pso), 0);
    }

    #[test]
    fn indirect_split_dispatch_expands_groups_and_preserves_arguments() {
        let Some(host) = crate::host::host() else {
            eprintln!("no D3D12 host for the split dispatch test");
            return;
        };
        let (mut shader, used) = build_root_view_cs();
        // Test the host launch protocol with a shader that records the
        // host group IDs. The shader crate tests its own wave relabelling.
        shader.group_waves = 2;
        let pso = crate::pipeline::test_compute_pipeline(shader, used);
        let args = Allocation::watched(64 << 10);
        let uav = Allocation::watched(2 << 20);
        crate::shadow::register(args.ptr as usize, args.addr(), 64 << 10);
        crate::shadow::register(uav.ptr as usize, uav.addr(), 2 << 20);
        let source = crate::shadow::find(args.addr(), 64).expect("argument shadow");
        let words = [
            2, 0, 0, 0, // count, then padding before the arguments
            32768, 1, 1, 0xAAAA, // expanded X would exceed the host limit
            1, 1, 1, 0xBBBB, // two host groups
            2, 1, 1, 0xCCCC, // four host groups, beyond the count
        ];
        // SAFETY: the records fit in the committed argument allocation.
        unsafe { args.ptr.cast::<[u32; 16]>().write(words) };

        let list = host.new_list().expect("new_list");
        let heap = crate::descriptor::test_view_heap(16);
        let mut state = State::default();
        for (count, groups) in [(Some(args.addr()), 2), (None, 4)] {
            state.reset(pso);
            state.set_heaps(&[heap]);
            state.set_view(true, 2, uav.addr());
            assert!(state.execute_indirect(&list, (2, 16, 3), args.addr() + 16, count));
            list.close();
            assert_eq!(host.log_messages(), None);
            host.execute(&[&list]);
            for group in 0..4 {
                // SAFETY: the group results are inside the committed UAV.
                let value = unsafe { uav.ptr.add(0x18_0000).cast::<u32>().add(group).read() };
                assert_eq!(value, if group < groups { 0xA5A5_5A5A } else { 0 });
            }
            for (word, expected) in words.iter().enumerate() {
                assert_eq!(source.read_u32(word * 4), *expected, "argument word {word}");
            }
            list.reset();
        }
        crate::shadow::unregister(args.ptr as usize, args.addr());
        crate::shadow::unregister(uav.ptr as usize, uav.addr());
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(pso, 0x10);
        assert_eq!(release(pso), 0);
    }

    /// A root SRV bound as a one-view table has its copy refreshed as far
    /// as [`root_len`] says, not only over its first 1 MiB: a program that
    /// loads the dword 1.5 MiB into the buffer gets what title memory holds
    /// there. With the refresh cut to 1 MiB the page is never copied and
    /// the load reads 0 (the view itself reaches the copy's end either
    /// way), while the load under 1 MiB still reads its dword.
    ///
    /// Only the table's own bind refreshes the SRV's copy here: the output
    /// is a root UAV on a buffer of its own, and nothing else is bound.
    #[test]
    fn a_root_srv_table_refreshes_its_copy_past_one_mib() {
        let Some(host) = crate::host::host() else {
            return;
        };
        if !crate::pipeline::host::bounded_root_srvs() || root_srv_cap() < 2 << 20 {
            // A root descriptor is another path ([`root_view`]), and a cap
            // set under the offset read is the case this test tells apart.
            return;
        }
        let (shader, used) = build_root_srv_cs();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        let len = 2 << 20;
        let srv_buf = Allocation::watched(len);
        let out_buf = Allocation::watched(64 << 10);
        crate::shadow::register(srv_buf.ptr as usize, srv_buf.addr(), len as u64);
        crate::shadow::register(out_buf.ptr as usize, out_buf.addr(), 64 << 10);
        let (near, far, unwritten) = (0x0000_BEA7u32, 0xFACE_0FF5u32, 0xFFFF_FFFFu32);
        // SAFETY: inside the committed allocations.
        unsafe {
            srv_buf.ptr.add(SRV_NEAR).cast::<u32>().write(near);
            srv_buf.ptr.add(SRV_FAR).cast::<u32>().write(far);
            out_buf.ptr.add(SRV_NEAR_OUT).cast::<u32>().write(unwritten);
            out_buf.ptr.add(SRV_FAR_OUT).cast::<u32>().write(unwritten);
        }
        // The copy holds nothing of the buffer before the bind.
        let shadow = crate::shadow::find(srv_buf.addr(), 4).expect("shadow");
        assert_eq!(shadow.read_u32(SRV_NEAR), 0);
        assert_eq!(shadow.read_u32(SRV_FAR), 0);

        let list = host.new_list().expect("new_list");
        let heap = crate::descriptor::test_view_heap(16);
        let mut state = State::default();
        state.set_pipeline(pso);
        state.set_heaps(&[heap]);
        state.set_view(true, 0, srv_buf.addr());
        state.set_view(true, 1, out_buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        assert!(device_alive(host), "the device was removed");

        // SAFETY: inside the committed allocation.
        let out = |at: usize| unsafe { out_buf.ptr.add(at).cast::<u32>().read() };
        let (got_near, got_far) = (out(SRV_NEAR_OUT), out(SRV_FAR_OUT));
        eprintln!("root SRV table: read {got_near:#x} at +0.5 MiB, {got_far:#x} at +1.5 MiB");
        assert_eq!(got_near, near, "the load under 1 MiB");
        assert_eq!(
            got_far, far,
            "the load at +1.5 MiB (0: the page was not refreshed)"
        );
        assert_eq!(shadow.read_u32(SRV_FAR), far);

        crate::shadow::unregister(srv_buf.ptr as usize, srv_buf.addr());
        crate::shadow::unregister(out_buf.ptr as usize, out_buf.addr());
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(pso, 0x10);
        assert_eq!(release(pso), 0);
    }

    /// A root UAV bound as a one-view table ends where its refresh and
    /// copy-back range does: a store past that is dropped and a load past
    /// it reads 0, inside the host copy and far past it, without a fault,
    /// while a store inside in the same dispatch reaches title memory.
    #[test]
    fn a_root_uav_store_past_its_view_is_dropped() {
        let Some(host) = crate::host::host() else {
            return;
        };
        if !crate::pipeline::host::bounded_root_uavs() {
            // A root descriptor has no end: the program's far store would
            // fault the GPU.
            return;
        }
        let (shader, used) = build_root_uav_cs();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        // 3 MiB of title memory: the view is its first 2 MiB (ROOT_UAV_MAX),
        // so +2.5 MiB is past the view and inside the copy.
        let len = 3 << 20;
        let buf = Allocation::watched(len);
        let base = buf.addr();
        crate::shadow::register(buf.ptr as usize, base, len as u64);
        let marker = 0x0C0F_FEE0u32;
        // SAFETY: inside the committed allocation.
        unsafe { buf.ptr.add(UAV_PAST).cast::<u32>().write(marker) };

        // The whole copy brought up to date (as a root SRV: 3 MiB), so that
        // it holds the marker where the program loads and stores past the
        // view; the UAV's own refresh ends with its view.
        let list = host.new_list().expect("new_list");
        let shadow = crate::shadow::find(base, 4).expect("shadow");
        assert_eq!(root_view(&list, base, RootView::Srv), Some(shadow.gpu(base)));
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(UAV_PAST), marker);
        let view = root_uav_range(base).map(|(_, offset, len)| (offset, len));
        assert_eq!(view, Some((0, ROOT_UAV_MAX)));

        list.reset();
        let heap = crate::descriptor::test_view_heap(16);
        let mut state = State::default();
        state.set_pipeline(pso);
        state.set_heaps(&[heap]);
        state.set_view(true, 0, base);
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        assert!(device_alive(host), "the device was removed");

        // SAFETY: inside the committed allocation.
        let title = |at: usize| unsafe { buf.ptr.add(at).cast::<u32>().read() };
        assert_eq!(
            title(UAV_INSIDE),
            UAV_IN_RANGE,
            "the in-range store must reach title memory, with 0 loaded past the view"
        );
        assert_eq!(title(UAV_PAST), marker, "title memory past the view changed");
        assert_eq!(
            shadow.read_u32(UAV_PAST),
            marker,
            "the store past the view landed in the host copy"
        );

        crate::shadow::unregister(buf.ptr as usize, base);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(pso, 0x10);
        assert_eq!(release(pso), 0);
    }

    /// A root UAV address not on 16 bytes of its copy cannot start a raw
    /// view: it gets a null view, so the program's stores are dropped
    /// without a fault. The same program at an address on 16 bytes stores.
    #[test]
    fn a_root_uav_off_16_bytes_gets_a_null_view() {
        let Some(host) = crate::host::host() else {
            return;
        };
        if !crate::pipeline::host::bounded_root_uavs() {
            // As above: the program's far store needs a view with an end.
            return;
        }
        let (shader, used) = build_root_uav_cs();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        let len = 3 << 20;
        let buf = Allocation::watched(len);
        let base = buf.addr();
        crate::shadow::register(buf.ptr as usize, base, len as u64);
        // SAFETY: inside the committed allocation.
        let title = |at: usize| unsafe { buf.ptr.add(at).cast::<u32>().read() };
        let shadow = crate::shadow::find(base, 4).expect("shadow");

        // On 8 bytes: a root UAV descriptor takes it (4 are enough there).
        assert!(root_uav_range(base + 8).is_none());
        let list = host.new_list().expect("new_list");
        assert!(root_view(&list, base + 8, RootView::Uav).is_some());
        list.close();
        host.execute(&[&list]);

        list.reset();
        let heap = crate::descriptor::test_view_heap(16);
        let mut state = State::default();
        state.set_pipeline(pso);
        state.set_heaps(&[heap]);
        state.set_view(true, 0, base + 8);
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        assert!(device_alive(host), "the device was removed");
        for at in [8 + UAV_INSIDE, 8 + UAV_PAST] {
            assert_eq!(title(at), 0, "a store through the null view reached title memory");
            assert_eq!(shadow.read_u32(at), 0, "a store through the null view landed");
        }

        // On 16 bytes: the view starts there and the in-range store lands.
        let range = root_uav_range(base + 16).map(|(_, offset, len)| (offset, len));
        assert_eq!(range, Some((16, ROOT_UAV_MAX)));
        list.reset();
        state.reset(pso);
        state.set_heaps(&[heap]);
        state.set_view(true, 0, base + 16);
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        assert!(device_alive(host), "the device was removed");
        assert_eq!(title(16 + UAV_INSIDE), UAV_IN_RANGE);
        assert_eq!(title(16 + UAV_PAST), 0);
        assert_eq!(shadow.read_u32(16 + UAV_PAST), 0);

        crate::shadow::unregister(buf.ptr as usize, base);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(pso, 0x10);
        assert_eq!(release(pso), 0);
    }
}
