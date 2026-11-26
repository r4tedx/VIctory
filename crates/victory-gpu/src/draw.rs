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
    address: u64,
    size: u32,
    stride: u32,
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
    /// The last [`Self::dispatch`] ran (else its stand-ins take its place).
    dispatch_ran: bool,
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
    /// The title-memory page table made for this list (`crate::memory`)
    /// and its view heap.
    memory: Option<(usize, crate::memory::Map)>,
    /// Views made for root SRVs and UAVs in this list ([`State::root_view`]):
    /// (view heap, host copy, unordered access) → dynamic slot.
    root_slots: std::collections::HashMap<(usize, usize, bool), u32>,
    /// The run of dynamic slots root views and fresh tables take from
    /// (view heap, next slot, end), [`State::take_slots`]: [`LIST_RUN`] at a
    /// time where the heap has a run that long, so that a list's slots lie
    /// together and come back together, not scattered among other lists'
    /// where they leave holes too short for a table; what it has left goes
    /// back when the list closes or takes another run
    /// ([`State::trim_run`]).
    run: (usize, u32, u32),
    /// The dynamic slots the list held when first submitted since its
    /// reset (counted in [`HELD_BY_RAN`] until it is reset or released).
    ran: Option<u64>,
    /// Slots made for unordered access clears in this list
    /// ([`State::clear_view`]): (view heap, view, host resource) → dynamic
    /// slot, so that clears of one view (a stand-in's, at each dispatch of
    /// its program) share one. The slots are among [`Self::dynamic`], held
    /// until the list is reset (which drops this with them) or released;
    /// the list keeps the host resource, so its address names no other.
    /// Made again in each segment of the list, as root views are.
    clear_slots: std::collections::BTreeMap<(usize, View, usize), u32>,
    /// What was set straight onto the host list, for the next segment of
    /// the list ([`State::cut`]).
    fixed: Fixed,
}

/// Fixed-function state the title set, as it set it.
#[derive(Default)]
struct Fixed {
    viewports: Option<Vec<D3D12_VIEWPORT>>,
    scissors: Option<Vec<RECT>>,
    blend_factor: Option<Option<[f32; 4]>>,
    stencil_ref: Option<u32>,
    /// Render target and depth view handles.
    targets: Option<(Vec<u64>, u64)>,
    topology: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            give_back(heap, first, n);
        }
        if let Some(n) = self.ran {
            HELD_BY_RAN.fetch_sub(n, Ordering::Relaxed);
        }
    }
}

/// Gives back `n` dynamic slots from `first` of the view heap keyed
/// `heap` (a title heap's, or [`MEMORY_HEAP`]).
fn give_back(heap: usize, first: u32, n: u32) {
    match (heap, crate::memory::heap()) {
        (MEMORY_HEAP, Some(ours)) => ours.release(first, n),
        _ => crate::descriptor::release_dynamic(heap, first, n),
    }
}

/// The key of our own view heap (`crate::memory::heap`) among the title's.
const MEMORY_HEAP: usize = crate::memory::OURS;

/// The sampler heap we bind when the title binds none (a value no title heap
/// address takes).
const SAMPLERS_OURS: usize = usize::MAX;

/// Dynamic slots a list takes at a time for its root views and fresh
/// tables; fewer, down to what is needed, where the heap has no free run
/// this long (a capped heap's 2^16 dynamic slots held by many lists).
const LIST_RUN: u32 = 256;

/// Why draws and dispatches were dropped for want of dynamic slots: root
/// views, fresh tables. With the list runs taken shorter than
/// [`LIST_RUN`] instead of dropping, and the tables refused while as many
/// slots as they need were free, only not in a row (fragmented), or were
/// not (exhausted).
static DYNAMIC_DROPS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static SHORT_RUNS: AtomicU64 = AtomicU64::new(0);
static TABLE_REFUSALS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
const NO_ROOT_VIEW_SLOTS: usize = 0;
const NO_TABLE_SLOTS: usize = 1;
const FRAGMENTED: usize = 0;
const EXHAUSTED: usize = 1;

/// Dynamic slots held by lists submitted since their last reset (each
/// submission is waited for): once it returns the GPU no longer reads
/// them unless the title submits the list again, but they come back only
/// at the reset.
static HELD_BY_RAN: AtomicU64 = AtomicU64::new(0);

/// What [`take_dynamic_drops`] counts.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct DynamicDrops {
    pub(crate) root_views: u64,
    pub(crate) tables: u64,
    pub(crate) short_runs: u64,
    pub(crate) fragmented: u64,
    pub(crate) exhausted: u64,
}

/// Draws and dispatches dropped since the last call for want of dynamic
/// slots, the list runs taken shorter than [`LIST_RUN`], and table
/// refusals by cause.
pub(crate) fn take_dynamic_drops() -> DynamicDrops {
    let take = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
    DynamicDrops {
        root_views: take(&DYNAMIC_DROPS[NO_ROOT_VIEW_SLOTS]),
        tables: take(&DYNAMIC_DROPS[NO_TABLE_SLOTS]),
        short_runs: take(&SHORT_RUNS),
        fragmented: take(&TABLE_REFUSALS[FRAGMENTED]),
        exhausted: take(&TABLE_REFUSALS[EXHAUSTED]),
    }
}

/// Dynamic slots held now by lists already submitted ([`HELD_BY_RAN`]).
pub(crate) fn held_by_ran() -> u64 {
    HELD_BY_RAN.load(Ordering::Relaxed)
}

/// Dynamic slots taken for root views ([`State::root_view`]) since the last
/// call.
pub(crate) fn take_root_view_slots() -> u64 {
    ROOT_VIEW_SLOTS.swap(0, Ordering::Relaxed)
}
static ROOT_VIEW_SLOTS: AtomicU64 = AtomicU64::new(0);

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

/// Marks the last traced draw as not run on the host, counts it by why
/// (the reason [`not_run`] gave, if its pipeline has a host one) and notes
/// it in the list (diagnostics).
#[cfg(feature = "diagnostics")]
fn dropped(list: &HostList, pipeline: usize, kind: &str) {
    let why = NOT_RUN.with(|w| w.replace(""));
    if why.is_empty() || crate::pipeline::host_pipeline(pipeline as This).is_none() {
        crate::pipeline::note_drop(pipeline as This);
    } else {
        crate::pipeline::note_why(format!("{kind}: {why} {}", crate::pipeline::shaders_of(pipeline as This)));
    }
    list.note_op(|| format!("DROPPED {kind} ({why}) {}", crate::pipeline::shaders_of(pipeline as This)));
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// Why the draw or dispatch this thread last prepared cannot run.
    static NOT_RUN: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Notes why the draw or dispatch being prepared cannot run (diagnostics).
fn not_run(why: &'static str) {
    #[cfg(feature = "diagnostics")]
    NOT_RUN.with(|w| w.set(why));
    #[cfg(not(feature = "diagnostics"))]
    let _ = why;
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. What shaders write through a root UAV goes back
/// to title memory after the submission.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
fn root_view(list: &HostList, address: u64, srv: bool, uav: bool) -> Option<u64> {
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv || uav { 1 << 20 } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || uav || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        if uav {
            // A host root descriptor: no heap view, so its writes are not
            // marked.
            shadow.gpu_writes(list, address, len, false);
        }
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
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
        self.fixed.topology = true;
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

    /// Fixed-function state, straight onto the host list (and kept for the
    /// list's next segment).
    pub(crate) fn viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.fixed.viewports = Some(viewports.to_vec());
        // SAFETY: an open list.
        unsafe { list.list.RSSetViewports(viewports) };
    }

    pub(crate) fn scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.fixed.scissors = Some(rects.to_vec());
        // SAFETY: an open list.
        unsafe { list.list.RSSetScissorRects(rects) };
    }

    pub(crate) fn blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.fixed.blend_factor = Some(factor.copied());
        // SAFETY: an open list.
        unsafe { list.list.OMSetBlendFactor(factor) };
    }

    pub(crate) fn stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.fixed.stencil_ref = Some(reference);
        // SAFETY: an open list.
        unsafe { list.list.OMSetStencilRef(reference) };
    }

    /// The title list goes on in a fresh host list (a segment after a wait,
    /// `crate::list`): the state it set is given to `list` again.
    ///
    /// Replayed: topology, viewports, scissors, blend factor, stencil
    /// reference, render targets and depth (set at once); heaps, root
    /// signatures, pipelines, root arguments, vertex and index buffers are
    /// set by the next draw or dispatch from what this state holds (the
    /// host caches are cleared). Made again in the new segment, so their
    /// submission-time work (refreshes, uploads) runs with it: fresh tables,
    /// root views and the title-memory page table. Not replayed, as nothing
    /// of it is on the host list: depth bounds and shading rate (ignored),
    /// queries (resolved with plain values), the 0x4a0 upload memory (the
    /// title list's own), texture states and twins (each segment has its
    /// own, made from the real states at its submission). Clears and copies
    /// are commands, recorded once in the segment they belong to.
    pub(crate) fn cut(&mut self, list: &HostList) {
        self.host_state_lost();
        self.fresh_tables.clear();
        self.root_slots.clear();
        self.clear_slots.clear();
        self.memory = None;
        let fixed = std::mem::take(&mut self.fixed);
        if fixed.topology {
            self.set_topology(list, self.topology);
        }
        if let Some(v) = &fixed.viewports {
            self.viewports(list, v);
        }
        if let Some(r) = &fixed.scissors {
            self.scissors(list, r);
        }
        if let Some(f) = fixed.blend_factor {
            self.blend_factor(list, f.as_ref());
        }
        if let Some(s) = fixed.stencil_ref {
            self.stencil_ref(list, s);
        }
        if let Some((targets, depth)) = &fixed.targets {
            self.set_targets(list, targets, *depth);
        }
    }

    /// The title pipeline set.
    pub(crate) fn pipeline(&self) -> This {
        self.pipeline as This
    }

    /// Others recorded into the host list (`crate::external`): its heaps,
    /// root signatures and pipeline are set again before the next draw or
    /// dispatch.
    pub(crate) fn host_state_lost(&mut self) {
        // No title heaps (0) differ from what the host list has, too.
        (self.host_pipeline, self.host_roots, self.host_heaps) = (0, [0; 2], [usize::MAX; 2]);
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

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        self.fixed.targets = Some((handles.to_vec(), depth));
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
            crate::capture::note_target(list, record.resource);
            rtvs.push(host.rtv(&texture, record.desc().as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            Some(host.dsv(&texture, record.depth_desc().as_ref()))
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

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value, a root
    /// view's address (title memory).
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::View(address)), RootType::Descriptor) if dword < 2 => {
                (address >> (32 * dword)) as u32
            }
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

    /// The host addresses of the bound view heap's title-memory page table
    /// and the list's page bitmap for it, set up on first use (the table
    /// names every host copy at each submission).
    fn memory_map(&mut self, list: &HostList) -> Option<(u64, u64)> {
        let heap = match self.heaps[0] {
            0 => MEMORY_HEAP,
            h => h,
        };
        if let Some((h, map)) = &self.memory
            && *h == heap
        {
            return Some((map.table, map.dirty));
        }
        let map = crate::memory::bind(list, heap)?;
        let out = (map.table, map.dirty);
        self.memory = Some((heap, map));
        Some(out)
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
        // Diagnostic: each distinct range a title sets, once (at most 16).
        static SEEN: std::sync::Mutex<Vec<(u32, u32)>> = std::sync::Mutex::new(Vec::new());
        {
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            let key = (min.to_bits(), max.to_bits());
            if seen.len() < 16 && !seen.contains(&key) {
                seen.push(key);
                tracing::info!(min, max, pipeline = ?self.pipeline, "depth bounds set");
            }
        }
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
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            not_run("another kind of pipeline");
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
                not_run("index buffer outside title buffers");
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
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
        // A depth copy of ours (`crate::twin`) ran a compute pipeline.
        if list.take_rebind() {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            (self.host_pipeline, self.host_roots[1]) = (p.pso.as_raw() as usize, 0);
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
        // What its shaders write is withheld only if the program uses title
        // memory and a list using it misses (`crate::memory::withholding`).
        let _writer = list.writer_uses_memory(p.stages.iter().any(|s| s.memory.is_some()));
        let l = &list.list;
        // A depth copy of ours (`crate::twin`) since the last one ran a
        // compute pipeline.
        if list.take_rebind() {
            (self.host_pipeline, self.host_roots[1]) = (0, 0);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them. Title memory's views need a view heap: ours where
        // the title binds none (the BVH builder's dispatches may not).
        let memory_heap = self.heaps[0] == 0
            && p.stages.iter().any(|s| s.memory.is_some() || has_root_views(&s.translated().host));
        self.bind_heaps(list, memory_heap);
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
        #[cfg(feature = "diagnostics")]
        crate::capture::binding(&p.name);
        // The bound view heap's mark table and the bitmap shaders mark
        // their writes through views in (`crate::shadow::copy_back_marks`).
        if let Some((table_param, bitmap_param)) = p.marks {
            let heap = match self.heaps[0] {
                0 => crate::memory::heap(),
                h => crate::descriptor::heap(h as This).host(),
            };
            // Shaders using title memory mark in a bitmap of their own, held
            // back while their results are (`crate::memory::withholding`).
            let memory = p.stages.iter().any(|s| s.memory.is_some());
            let (Some(table), Some(bitmap)) = (heap.and_then(|h| h.marks()), crate::shadow::view_writes(memory)) else {
                once!("a shader marking its writes without a view heap: dropped");
                not_run("write marks without a view heap");
                return false;
            };
            list.keep(table);
            // SAFETY: an open list; the table lives while the list holds it,
            // the bitmap for good.
            unsafe {
                let table = table.GetGPUVirtualAddress();
                if compute {
                    l.SetComputeRootShaderResourceView(table_param, table);
                    l.SetComputeRootUnorderedAccessView(bitmap_param, bitmap);
                } else {
                    l.SetGraphicsRootShaderResourceView(table_param, table);
                    l.SetGraphicsRootUnorderedAccessView(bitmap_param, bitmap);
                }
            }
        }
        // Shaders taking views from the heap itself read (or write) views no
        // table names: everything the bound heap's views reach is brought up
        // to date at submission (`crate::bindless`).
        if p.stages.iter().any(|s| s.translated().host.heap) {
            let writes = p.stages.iter().any(|s| s.translated().host.heap_writes);
            if let Some(host) = (self.heaps[0] != 0)
                .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                .flatten()
            {
                list.defer_heap(&host.viewed);
            }
            crate::bindless::note_draw(writes);
        }
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            // The view addresses exist before the stage's tables are made:
            // a table made afresh enters its views' addresses there.
            if stage.view_addresses.is_some() && self.heaps[0] != 0 {
                let _ = crate::descriptor::heap(self.heaps[0] as This)
                    .host()
                    .and_then(|h| h.view_addresses());
            }
            let made = self.tables(list, &t.used, compute);
            // A depth copy of ours made readable a texture the tables hold
            // (`crate::twin`), with a compute pipeline: this one's again
            // (root arguments are set below).
            if list.take_rebind() {
                // SAFETY: an open list, a live root signature and pipeline.
                unsafe {
                    if compute {
                        l.SetComputeRootSignature(&p.root);
                    }
                    l.SetPipelineState(&p.pso);
                }
                self.host_pipeline = p.pso.as_raw() as usize;
                self.host_roots[1] = if compute { root } else { 0 };
            }
            if self.table_failed {
                DYNAMIC_DROPS[NO_TABLE_SLOTS].fetch_add(1, Ordering::Relaxed);
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                not_run("no dynamic slots for its tables");
                return false;
            }
            // Root views a shader reads as title memory: the copies holding
            // their addresses go into the page table.
            let mut roots = Vec::new();
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if u.root == RootType::Descriptor
                        && u.dword == 0
                        && let Some(Root::View(address)) =
                            self.roots_ref(compute).get(usize::from(u.slot))
                        && let Some(shadow) = crate::shadow::find(*address, 1)
                    {
                        let uav = t.used.iter().any(|r| {
                            r.slot == u.slot
                                && r.root == RootType::Descriptor
                                && r.kind == ResourceType::Uav
                        });
                        roots.push((*address, uav.then_some(shadow)));
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None if is_view_offset(u) => {
                            // Where the root address lies in the copy its view shows.
                            let offset = match self.roots_ref(compute).get(usize::from(u.slot)) {
                                Some(&Root::View(address)) => crate::shadow::find(address, 1)
                                    .and_then(|s| u32::try_from(s.offset(address)).ok()),
                                _ => None,
                            };
                            match offset {
                                Some(offset) => offset,
                                None => {
                                    not_run("a root view's offset: no root view, or outside title buffers");
                                    return false;
                                }
                            }
                        }
                        None if is_root_view(u) => {
                            let uav = u.dword == victory_shader::emit::ROOT_UAV_VIEW;
                            match self.root_view(list, compute, u.slot, uav) {
                                Some((slot, _)) => slot,
                                None => return false,
                            }
                        }
                        None => self.user_value(compute, u.slot, u.root, u.dword),
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
            for (views, kind) in [(&stage.root_cbvs, 0), (&stage.root_srvs, 1), (&stage.root_uavs, 2)] {
                for &(slot, param) in views {
                    if kind == 1 && slot == victory_shader::emit::VIEW_STRIDES {
                        // The bound view heap's stride table; without one
                        // the shader would read an unbound root view.
                        let heap = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                            .flatten();
                        let Some(table) = heap.and_then(|h| h.strides()) else {
                            once!("a shader reads view strides without a view heap: dropped");
                            not_run("view strides without a view heap");
                            return false;
                        };
                        list.keep(table);
                        // SAFETY: an open list; the table lives while the list
                        // holds it.
                        unsafe {
                            let gpu = table.GetGPUVirtualAddress();
                            if compute {
                                l.SetComputeRootShaderResourceView(param, gpu);
                            } else {
                                l.SetGraphicsRootShaderResourceView(param, gpu);
                            }
                        }
                        continue;
                    }
                    let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        continue;
                    };
                    let Some(gpu) = root_view(list, *address, kind == 1, kind == 2) else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, kind) {
                            (false, 0) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, 1) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (false, _) => l.SetGraphicsRootUnorderedAccessView(param, gpu),
                            (true, 0) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, 1) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, _) => l.SetComputeRootUnorderedAccessView(param, gpu),
                        }
                    }
                }
            }
            if let Some(param) = stage.view_addresses {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.view_addresses());
                let Some(gpu) = gpu else {
                    once!("a shader reading a view's address dropped: no view heap");
                    not_run("view addresses without a view heap");
                    return false;
                };
                // SAFETY: an open list; the heap's buffer lives with the heap.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, gpu);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, gpu);
                    }
                }
            }
            if let Some(param) = stage.memory {
                let Some((table, dirty)) = self.memory_map(list) else {
                    once!("a shader reading title memory dropped: no view heap or page table");
                    not_run("title memory without a view heap or page table");
                    return false;
                };
                crate::memory::note_use(&p.name);
                // What shaders write through a root UAV goes back by its
                // bytes, as through root views ([`root_view`]).
                for (address, shadow) in &roots {
                    if let (Some(shadow), Some((_, map))) = (shadow, &self.memory) {
                        let len = shadow.left(*address).min(1 << 20);
                        shadow.refresh(list, *address, len);
                        shadow.gpu_writes(list, *address, len, false);
                        map.precise(*address, len);
                    }
                }
                // SAFETY: an open list; list memory alive until it is reset.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, table);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, table);
                    }
                    if let Some(param) = stage.memory_dirty {
                        if compute {
                            l.SetComputeRootUnorderedAccessView(param, dirty);
                        } else {
                            l.SetGraphicsRootUnorderedAccessView(param, dirty);
                        }
                    }
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::binding("");
        true
    }

    /// The heap index of a raw view (`uav`: unordered access) of the host
    /// copy holding the root view the title set at root parameter `slot`,
    /// and the root address's byte offset in it: a dynamic slot of the view
    /// heap the shader indexes, made once per list and copy; the range from
    /// the root address brought up to date at submission
    /// (`victory_shader::emit`, "Root SRVs and root UAVs"). None if the
    /// draw must be dropped: no root view set, or no dynamic slot.
    /// The copy is brought up to date from the root address to its end: a
    /// shader reaches wherever data its root view leads to lie (a vertex
    /// fetch's streams sit at offsets in the root view's own header).
    /// shortcut: what shaders write through a root UAV goes back to title
    /// memory for 1 MiB from its address, as before the views.
    fn root_view(&mut self, list: &HostList, compute: bool, slot: u8, uav: bool) -> Option<(u32, u32)> {
        let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
            once!(slot, "a shader reads a root view the title did not set: dropped");
            not_run("a root view the title did not set");
            return None;
        };
        let Some(shadow) = crate::shadow::find(address, 1) else {
            not_run("a root view outside title buffers");
            return None;
        };
        let Some(offset) = u32::try_from(shadow.offset(address)).ok() else {
            not_run("a root view 4 GiB or more into its copy");
            return None;
        };
        shadow.refresh(list, address, shadow.left(address));
        if uav {
            // Writes through the view are marked wherever they land in the
            // copy; else declared for 1 MiB.
            let len = if shadow.marked() { shadow.left(address) } else { shadow.left(address).min(1 << 20) };
            shadow.gpu_writes(list, address, len, true);
        }
        let (key, host) = match self.heaps[0] {
            0 => (MEMORY_HEAP, crate::memory::heap()),
            h => (
                std::ptr::from_ref(crate::descriptor::heap(h as This)) as usize,
                crate::descriptor::heap(h as This).host(),
            ),
        };
        let Some(host) = host else {
            not_run("a root view without a host view heap");
            return None;
        };
        let copy = std::sync::Arc::as_ptr(&shadow) as usize;
        if let Some(&index) = self.root_slots.get(&(key, copy, uav)) {
            return Some((index, offset));
        }
        let Some(index) = self.take_slots(key, host, 1) else {
            once!("no dynamic slots for root views: dropped");
            DYNAMIC_DROPS[NO_ROOT_VIEW_SLOTS].fetch_add(1, Ordering::Relaxed);
            not_run("no dynamic slots for root views");
            return None;
        };
        ROOT_VIEW_SLOTS.fetch_add(1, Ordering::Relaxed);
        host.raw_view_at(index, Some(&shadow.buffer), 0, shadow.range().1 & !3, uav);
        if uav {
            host.set_mark(index, shadow.mark_entry(shadow.range().0));
        }
        self.root_slots.insert((key, copy, uav), index);
        Some((index, offset))
    }

    /// `n` consecutive dynamic slots of the view heap keyed `key` (`host`)
    /// from the list's run, held until the list is reset: the first. A run
    /// with fewer left is trimmed and another taken, [`LIST_RUN`] slots
    /// (or `n` if more) where the heap has a free run that long, else its
    /// longest free run of at least `n`.
    fn take_slots(&mut self, key: usize, host: &crate::descriptor::host::HostHeap, n: u32) -> Option<u32> {
        let (heap, next, end) = self.run;
        if heap != key || end - next < n {
            self.trim_run();
            let want = LIST_RUN.max(n);
            let (first, got) = host.dynamic_run(want, n)?;
            if got < want {
                once!(got, n, "dynamic slots short: lists take runs under {LIST_RUN} slots");
                SHORT_RUNS.fetch_add(1, Ordering::Relaxed);
            }
            self.dynamic.push((key, first, got));
            self.run = (key, first, first + got);
        }
        let first = self.run.1;
        self.run.1 += n;
        Some(first)
    }

    /// Gives back the slots of the list's run nothing took: the list closed
    /// (it records nothing more until reset) or takes another run. No
    /// command names them, so another list may have them at once.
    fn trim_run(&mut self) {
        let (heap, next, end) = self.run;
        if next >= end {
            return;
        }
        give_back(heap, next, end - next);
        if let Some(run) = self.dynamic.iter_mut().rev().find(|d| d.0 == heap && d.1 + d.2 == end) {
            run.2 = next - run.1;
        }
        self.dynamic.retain(|d| d.2 > 0);
        self.run = (heap, next, next);
    }

    /// The title closed the list.
    pub(crate) fn close(&mut self) {
        self.trim_run();
    }

    /// The list is submitted (closed; its submission is waited for): its
    /// dynamic slots are counted as held by a list that ran until it is
    /// reset. Once per recording.
    pub(crate) fn submitted(&mut self) {
        if self.ran.is_some() {
            return;
        }
        let n = self.dynamic.iter().map(|d| u64::from(d.2)).sum();
        HELD_BY_RAN.fetch_add(n, Ordering::Relaxed);
        self.ran = Some(n);
    }

    /// Gives the host list the title's heaps if it has others.
    /// The heaps a list must have bound before a root signature that indexes
    /// them directly is set: the title's of each kind, and ours for a kind it
    /// binds none of — D3D12 requires a heap of both kinds to be bound before
    /// such a signature (the debug layer's id 1419), and breaking that rule
    /// removes the device on NVIDIA. Compute lists that only read views are
    /// the case: they bind no sampler heap, and every signature we make says
    /// its sampler heap is indexed directly.
    fn wanted(title: [usize; 2], memory_heap: bool) -> [usize; 2] {
        let sampler = if title[1] != 0 {
            title[1]
        } else {
            SAMPLERS_OURS
        };
        let views = if memory_heap { MEMORY_HEAP } else { title[0] };
        [views, sampler]
    }

    /// Tests that bind descriptor heaps of their own: the state takes them
    /// for the ones it would bind for a title that binds none, so it does
    /// not replace them with its fallback sampler heap.
    #[cfg(test)]
    pub(crate) fn heaps_bound_by_caller(&mut self) {
        self.host_heaps = Self::wanted(self.heaps, false);
    }

    /// Gives the host list the title's heaps if it has others.
    fn bind_heaps(&mut self, list: &HostList, memory_heap: bool) {
        let want = Self::wanted(self.heaps, memory_heap);
        if self.host_heaps != want {
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0 && h != SAMPLERS_OURS)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            if memory_heap && let Some(h) = crate::memory::heap() {
                heaps.insert(0, Some(h.heap.clone()));
            }
            if want[1] == SAMPLERS_OURS {
                match crate::descriptor::fallback_samplers() {
                    Some(h) => {
                        tracing::debug!("the title binds no sampler heap: binding ours");
                        heaps.push(Some(h.heap.clone()));
                    }
                    None => {
                        once!("no sampler heap of ours to bind");
                    }
                }
            }
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { list.list.SetDescriptorHeaps(&heaps) };
            self.host_heaps = want;
            self.host_roots = [0; 2];
        }
    }

    /// `ClearUnorderedAccessViewUint` or `...Float` (`float`): the view the
    /// title made at `cpu` (its CPU handle; the GPU one names the same view
    /// in the bound heap) cleared to `values` on the host, through a
    /// dynamic slot of the bound view heap that the list holds until reset.
    pub(crate) fn clear_unordered(
        &mut self,
        list: &HostList,
        cpu: u64,
        values: [u32; 4],
        float: bool,
        rects: &[RECT],
    ) {
        crate::indirect::end(list);
        let Some(view) = crate::descriptor::view(cpu) else {
            once!("an unordered access clear of a slot without a view");
            return;
        };
        self.clear_view(list, &view, values, float, rects, false, None);
    }

    /// The clear of [`Self::clear_unordered`] of `view`; `written`: a
    /// dispatch just wrote it, so unordered access barriers go before and
    /// after the clear. `predicate`: a word the clear alone (not what
    /// readies the view for it) is predicated on, skipped where it holds
    /// zero. Whether it was recorded. Its shader-visible slot is a dynamic
    /// one the list holds until reset, one per view however often the list
    /// clears it (`clear_slots`).
    #[allow(clippy::too_many_arguments)]
    fn clear_view(
        &mut self,
        list: &HostList,
        view: &View,
        values: [u32; 4],
        float: bool,
        rects: &[RECT],
        written: bool,
        predicate: Option<&ID3D12Resource>,
    ) -> bool {
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This));
        let (Some(heap), Some(host)) = (heap, crate::host::host()) else {
            once!("an unordered access clear without a bound view heap");
            return false;
        };
        let Some(visible) = heap.host() else { return false };
        let at = host.cpu_view();
        let Some(resource) = crate::descriptor::host::make_view(view, at) else {
            once!("an unordered access clear of a view the host cannot make");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, view.clone(), resource.as_raw() as usize);
        let slot = match self.clear_slots.get(&key) {
            Some(&slot) => slot,
            None => {
                // From the list's run, as root views and fresh tables take
                // theirs ([`Self::take_slots`]).
                let Some(slot) = self.take_slots(key.0, visible, 1) else {
                    once!("an unordered access clear without a dynamic slot: not run");
                    return false;
                };
                visible.make_at(slot, view);
                self.clear_slots.insert(key, slot);
                slot
            }
        };
        let uav = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        readable(list, &uav, view, false);
        self.bind_heaps(list, false);
        list.keep(&resource);
        let gpu = visible.gpu(slot);
        let barrier = || D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    // SAFETY: borrowed without a reference of its own,
                    // never released (ManuallyDrop); the list keeps it.
                    pResource: unsafe { std::mem::transmute_copy(&resource) },
                }),
            },
        };
        // SAFETY: an open list; both slots view the live resource.
        unsafe {
            if written {
                list.list.ResourceBarrier(&[barrier()]);
            }
            if let Some(predicate) = predicate {
                list.list.SetPredication(predicate, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
            }
            if float {
                let values = values.map(f32::from_bits);
                list.list
                    .ClearUnorderedAccessViewFloat(gpu, at, &resource, &values, rects);
            } else {
                list.list
                    .ClearUnorderedAccessViewUint(gpu, at, &resource, &values, rects);
            }
            if predicate.is_some() {
                list.list.SetPredication(None::<&ID3D12Resource>, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
            }
            if written {
                list.list.ResourceBarrier(&[barrier()]);
            }
        }
        true
    }

    /// After a dispatch of `p`: the stand-ins `VICTORY_AMBIENT` sets for
    /// its program (`crate::ambient`). `indirect`: an indirect dispatch's
    /// most records and its record count ([`Self::stand_in_after_indirect`]).
    fn stand_ins(&mut self, list: &HostList, p: &HostPipeline, indirect: Option<(u32, Records<'_>)>) {
        let rules = crate::ambient::rules();
        if rules.is_empty() {
            return;
        }
        let Some(stage) = p.stages.first() else { return };
        let (program, used) = (&stage.shader.hash, Some(&stage.translated().used[..]));
        match indirect {
            None => {
                self.stand_in(list, program, used, rules, None);
            }
            Some((max, records)) => {
                self.stand_in_after_indirect(list, program, used, rules, max, records);
            }
        }
    }

    /// Whether `VICTORY_AMBIENT` names the program of the bound pipeline:
    /// whether a dispatch of it has stand-ins after it.
    fn names_stand_ins(&self) -> bool {
        let rules = crate::ambient::rules();
        !rules.is_empty()
            && crate::pipeline::host_pipeline(self.pipeline as This)
                .and_then(|p| p.stages.first())
                .is_some_and(|s| rules.iter().any(|r| r.program == s.shader.hash))
    }

    /// [`Self::stand_in`] after an indirect dispatch of at most `max`
    /// records, as many as `records` says. The stand-ins follow shader
    /// work, so where no record runs the views keep what they held: no
    /// records at most, no clears; with a count, the clears are predicated
    /// on it, so a count of zero skips them on the GPU; without one, every
    /// clear. How many were recorded.
    fn stand_in_after_indirect(
        &mut self,
        list: &HostList,
        program: &str,
        used: Option<&[UsedResource]>,
        rules: &[crate::ambient::Rule],
        max: u32,
        records: Records<'_>,
    ) -> usize {
        if max == 0 || !rules.iter().any(|r| r.program == program) {
            return 0;
        }
        let predicate = match records {
            Records::Uncounted => return self.stand_in(list, program, used, rules, None),
            Records::Counted(predicate) => predicate,
            Records::Unknown => {
                once!(program, "stand-ins after an indirect dispatch with a count: no predicate word took it, so they are skipped");
                for _ in rules.iter().filter(|r| r.program == program) {
                    crate::ambient::note(false);
                }
                return 0;
            }
        };
        use crate::host::transition;
        // SAFETY: an open list; our word, common between uses.
        unsafe {
            list.list.ResourceBarrier(&[transition(predicate, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PREDICATION)]);
        }
        let filled = self.stand_in(list, program, used, rules, Some(predicate));
        // SAFETY: as above.
        unsafe {
            list.list.ResourceBarrier(&[transition(predicate, D3D12_RESOURCE_STATE_PREDICATION, D3D12_RESOURCE_STATE_COMMON)]);
        }
        filled
    }

    /// In place of a dispatch that did not run (held back, dropped, or
    /// without a host pipeline): the stand-ins `VICTORY_STAND_IN` sets for
    /// its program (`crate::ambient`). Nothing of the program runs. Its
    /// translation is not asked for: it may have none. `indirect`: for an
    /// indirect dispatch, its most records and the title address of its
    /// record count, if it has one. The stand-ins take the place of the
    /// records it would have run, so as after one that runs
    /// ([`Self::stand_in_after_indirect`]): none for no records at most,
    /// and with a count, predicated on it (its host copy brought up to
    /// date, as the execution would have).
    fn instead(&mut self, list: &HostList, indirect: Option<(u32, Option<u64>)>) {
        let rules = crate::ambient::instead_rules();
        if rules.is_empty() {
            return;
        }
        let Some(program) = crate::pipeline::compute_hash(self.pipeline as This) else {
            return;
        };
        let Some((max, count)) = indirect else {
            self.stand_in(list, program, None, rules, None);
            return;
        };
        let Some(at) = count else {
            self.stand_in_after_indirect(list, program, None, rules, max, Records::Uncounted);
            return;
        };
        if max == 0 || !rules.iter().any(|r| r.program == program) {
            return;
        }
        let Some(copy) = crate::shadow::find(at, 4) else {
            once!(program, "a stand-in in place of an indirect dispatch whose count is outside title buffers: skipped");
            for _ in rules.iter().filter(|r| r.program == program) {
                crate::ambient::note(false);
            }
            return;
        };
        copy.refresh(list, at, 4);
        // The count where the execution would have read it.
        let records = match count_predicate(list, &copy.buffer, copy.offset(at)) {
            Some(predicate) => Records::Counted(predicate),
            None => Records::Unknown,
        };
        self.stand_in_after_indirect(list, program, None, rules, max, records);
    }

    /// The views of `rules` for `program`, whose code uses `used` (none:
    /// not known, as for a program that did not run), cleared to their
    /// values (each clear predicated on `predicate`, if given,
    /// [`Self::clear_view`]): how many were.
    fn stand_in(
        &mut self,
        list: &HostList,
        program: &str,
        used: Option<&[UsedResource]>,
        rules: &[crate::ambient::Rule],
        predicate: Option<&ID3D12Resource>,
    ) -> usize {
        let mut filled = 0;
        for rule in rules.iter().filter(|r| r.program == program) {
            let uavs = || -> Vec<String> {
                used.unwrap_or_default()
                    .iter()
                    .filter(|u| u.kind == ResourceType::Uav && u.root == RootType::Table)
                    .map(|u| format!("s{}+{}", u.slot, u.table_offset_dwords))
                    .collect()
            };
            let first = !rule.seen.swap(true, Ordering::Relaxed);
            let named = used.is_none_or(|used| {
                used.iter().any(|u| {
                    u.kind == ResourceType::Uav
                        && u.root == RootType::Table
                        && (u.slot, u.table_offset_dwords) == (rule.slot, rule.offset)
                })
            });
            if !named {
                if first {
                    tracing::warn!(program, view = %rule.view(), views = ?uavs(), "stand-in: the program has no such unordered view in a table");
                }
                crate::ambient::note(false);
                continue;
            }
            let view = match self.roots_ref(true).get(usize::from(rule.slot)) {
                Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(rule.offset) * 4),
                _ => None,
            };
            // The formats of the view and of the texture it views, the
            // title's; the size, for the log.
            let formats = |view: &View| match view {
                View::Unordered(r, _, words) => crate::resource::title_format(*r as This).map(|own| {
                    // SAFETY: a live resource.
                    let size = crate::resource::host_resource(*r as This).map(|t| unsafe { t.GetDesc() }).map(|d| (d.Width, d.Height));
                    (words.map_or(0, |w| w[0]), own, size)
                }),
                View::Placed(true, texture, words) => Some((words[0], texture[8], None)),
                _ => None,
            };
            let Some((view, (format, own, size))) = view.and_then(|v| formats(&v).map(|f| (v, f))) else {
                if first {
                    tracing::warn!(program, view = %rule.view(), "stand-in: no unordered view of a texture there");
                }
                crate::ambient::note(false);
                continue;
            };
            let format = if format == 0 { own } else { format };
            let fill = crate::ambient::fill(rule.value, format, own);
            if first {
                tracing::info!(
                    program,
                    view = %rule.view(),
                    views = ?uavs(),
                    format,
                    texture_format = own,
                    ?size,
                    value = ?rule.value,
                    ?fill,
                    ran = used.is_some(),
                    "stand-in: first dispatch of the program"
                );
            }
            let recorded = match fill {
                Ok(crate::ambient::Fill::Float(rgba)) => {
                    self.clear_view(list, &view, rgba.map(f32::to_bits), true, &[], true, predicate)
                }
                Ok(crate::ambient::Fill::Uint(words)) => {
                    self.clear_view(list, &view, words, false, &[], true, predicate)
                }
                Err(why) => {
                    if first {
                        tracing::warn!(program, view = %rule.view(), format, texture_format = own, "stand-in: {why}");
                    }
                    false
                }
            };
            crate::ambient::note(recorded);
            filled += usize::from(recorded);
        }
        filled
    }

    /// Runs a dispatch on the host; if it does not run, the stand-ins
    /// `VICTORY_STAND_IN` names for its program take its place
    /// ([`Self::instead`]).
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        // Indirect executions after it clamp what it writes (it or the
        // stand-ins that take its place).
        crate::indirect::end(list);
        self.dispatch_ran = false;
        self.run_dispatch(list, x, y, z);
        if !self.dispatch_ran {
            self.instead(list, None);
        }
    }

    /// [`Self::dispatch`] itself; `dispatch_ran` says whether it ran.
    fn run_dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        // The acceleration-structure builder runs whole or not at all, and
        // what walks its structures waits for it (`pipeline::builder`).
        if let Some(why) = crate::pipeline::held_back(self.pipeline as This) {
            once!(why, "a dispatch held back");
            not_run(why);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
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
            dropped(list, self.pipeline, "dispatch");
            return;
        };
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        self.probe_small(list, true);
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        self.dispatch_ran = true;
        #[cfg(feature = "diagnostics")]
        self.probe_written(list);
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        self.stand_ins(list, p, None);
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
            for u in t.host.user_data.iter().filter(|u| !is_root_view(u)) {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                    // A slot the host never made: what the title's memory
                    // holds there (our record, zeros, or words it wrote itself).
                    if slot == "null" && self.heaps[0] != 0 {
                        let at = crate::descriptor::heap(self.heaps[0] as This).start + u64::from(index) * 32;
                        let mut words = [0u8; 32];
                        if crate::host::read_into(at as *const u8, &mut words) {
                            let w: Vec<u32> =
                                words.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            let _ = write!(text, " (title words {w:x?}");
                            if let Some(view) = crate::descriptor::view(at) {
                                let _ = write!(text, ": {}", describe(Some(&view)));
                            }
                            let _ = write!(text, ")");
                        }
                    }
                }
            }
            let viewed = t.host.user_data.iter().filter(|u| is_root_view(u)).map(|u| u.slot);
            let slots: std::collections::BTreeSet<u8> =
                stage.root_cbvs.iter().chain(&stage.root_srvs).map(|r| r.0).chain(viewed).collect();
            for slot in slots {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    // Its first dwords (a vertex stream header, say).
                    let mut head = [0u8; 16];
                    let head = crate::host::read_into(*address as *const u8, &mut head)
                        .then(|| head.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>());
                    let _ = write!(text, " root{slot}={address:#x} host {host:x?} head {head:x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute {
            let mem = |r: usize| crate::resource::texture_memory(r as This).map_or(0, |(p, _)| p as u64);
            let targets: Vec<String> = self.targets.iter().map(|&t| format!("{t:#x} mem {:#x}", mem(t))).collect();
            let _ = write!(text, " targets {targets:?} depth {:#x} mem {:#x}", self.depth, mem(self.depth));
        }
        text
    }


    /// Whether the pipeline bound may write GPU memory: through unordered
    /// access views (tables, root views, the heap itself), title memory at
    /// addresses it computes, or marked views.
    fn may_write(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| {
            p.marks.is_some()
                || p.stages.iter().any(|s| {
                    let t = s.translated();
                    !s.root_uavs.is_empty()
                        || t.host.memory_writes
                        || t.host.heap_writes
                        || t.host.view_marks
                        || t.used.iter().any(|r| r.kind == ResourceType::Uav)
                })
        })
    }

    /// Before a draw or mesh dispatch whose pipeline may write: the open
    /// window of batched indirect clamps ends, so that executions after it
    /// clamp what it writes.
    fn end_clamps_if_writing(&self, list: &HostList) {
        if list.clamps_open.load(Ordering::Relaxed) && self.may_write() {
            crate::indirect::end(list);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        self.end_clamps_if_writing(list);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            list.missed_draw();
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "mesh");
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
            readable(list, u, &view, true);
        }
    }

    /// The descriptors a shader reads through tables, made from the
    /// title's slots. Constant buffers there are raw views of their data
    /// (translated shaders read them so: the host wants constant buffer
    /// views 256-byte aligned, the console only 16), brought up to date at
    /// submission like other buffer views. Returns the byte offsets of
    /// tables made afresh in dynamic slots, by root parameter.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once.
    fn tables(&mut self, list: &HostList, used: &[UsedResource], compute: bool) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = Vec::new();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
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
                        readable(list, u, &view, true);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = self.take_slots(std::ptr::from_ref(heap) as usize, host, n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                once!(n, "no dynamic slots for a table: a draw or dispatch is dropped");
                let cause = if host.dynamic_free().0 >= n { FRAGMENTED } else { EXHAUSTED };
                TABLE_REFUSALS[cause].fetch_add(1, Ordering::Relaxed);
                self.table_failed = true;
                continue;
            };
            self.fresh_tables.insert(key, first);
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view, true);
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

    /// In a listed frame: each small buffer (64 bytes at most) the draw's
    /// tables name, as title memory now and as its host copy when the list
    /// is submitted (before the GPU runs it) and after it has run.
    #[cfg(feature = "diagnostics")]
    fn probe_small(&self, list: &HostList, compute: bool) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else { return };
        for stage in &p.stages {
            for u in &stage.translated().used {
                let (RootType::Table, Some(Root::Table(table))) =
                    (u.root, self.roots_ref(compute).get(usize::from(u.slot)))
                else {
                    continue;
                };
                let Some(view) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let range = match view {
                    View::Constant(address, size) => Some((address, u64::from(size))),
                    ref v => crate::descriptor::host::buffer_range(v),
                };
                let Some((at, len)) = range.filter(|r| r.1 <= 64) else {
                    continue;
                };
                let Some(shadow) = crate::shadow::find(at, len) else { continue };
                let who = format!("{} {:?} s{}+{} {at:#x}+{len}", stage.shader.hash, u.kind, u.slot, u.table_offset_dwords);
                let (s, w) = (shadow.clone(), who.clone());
                list.defer_host(move || tracing::info!(target: "victory_gpu::trace", "small {w} at submission: {}", s.peek(at, len)));
                list.defer_after(move || tracing::info!(target: "victory_gpu::trace", "small {who} after the list: {}", shadow.peek(at, len)));
            }
        }
    }

    /// In a listed frame, right after a dispatch: each 1D or 3D texture it
    /// writes through a table, read back in the list itself (what the
    /// dispatch left there, before anything later in the frame).
    #[cfg(feature = "diagnostics")]
    fn probe_written(&self, list: &HostList) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let (Some(p), Some(host)) = (crate::pipeline::host_pipeline(self.pipeline as This), crate::host::host()) else {
            return;
        };
        for stage in &p.stages {
            for u in stage.translated().used.iter().filter(|u| u.kind == ResourceType::Uav) {
                let (RootType::Table, Some(Root::Table(table))) = (u.root, self.compute.get(usize::from(u.slot))) else {
                    continue;
                };
                let Some(View::Unordered(r, ..)) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let Some(t) = crate::resource::host_resource(r as This) else { continue };
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D && d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                    continue;
                }
                const COPY_SOURCE: u32 = 0x800;
                const UNORDERED: u32 = 0x8;
                // All UAV writes done before the copy.
                let all = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
                    },
                };
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[all]) };
                crate::list::ensure_state(list, r as This, COPY_SOURCE);
                let read = crate::capture::read_back(&host.device, &list.list, &t);
                crate::list::ensure_state(list, r as This, UNORDERED);
                let Some((buffer, f)) = read else { continue };
                list.keep(&buffer);
                struct Readback(ID3D12Resource);
                // SAFETY: D3D12 objects are free-threaded.
                unsafe impl Send for Readback {}
                // SAFETY: as above.
                unsafe impl Sync for Readback {}
                let buffer = Readback(buffer);
                let who = format!("{} s{}+{} {r:#x} {}x{}x{} f{}", stage.shader.hash, u.slot, u.table_offset_dwords, d.Width, d.Height, d.DepthOrArraySize, d.Format.0);
                list.defer_after(move || {
                    let mut data = std::ptr::null_mut();
                    // SAFETY: the list has run; the mapping spans the first slice.
                    unsafe {
                        if buffer.0.Map(0, None, Some(&mut data)).is_err() {
                            return;
                        }
                        let n = (f.Footprint.RowPitch * f.Footprint.Height) as usize / 4;
                        let w = std::slice::from_raw_parts(data.cast::<u32>(), n);
                        let set = w.iter().filter(|&&x| x != 0).count();
                        let first: Vec<f32> = w.iter().take(8).map(|&x| f32::from_bits(x)).collect();
                        tracing::info!(target: "victory_gpu::trace", "written {who} right after the dispatch: {set} of {n} dwords set, first {first:?}");
                        buffer.0.Unmap(0, None);
                    }
                });
            }
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
                        let view = crate::descriptor::view(handle);
                        let mut text = describe(view.as_ref());
                        // The title's view words, and whether the host slot
                        // the shader reads was made from a view.
                        if let Some(View::Shader(_, Some(w)) | View::Unordered(_, _, Some(w))) = &view {
                            text.push_str(&format!(" words {w:?}"));
                        }
                        let slot = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This))
                            .and_then(|h| {
                                let i = handle.checked_sub(h.start)? / 32;
                                h.host()?.describe_slot(u32::try_from(i).ok()?)
                            });
                        let current: String = view.as_ref().map(|v| format!("{v:?}")).unwrap_or_default().chars().take(120).collect();
                        text.push_str(match slot.as_deref() {
                            Some("null") => " slot null",
                            Some(s) if s.starts_with(&current) => " slot made",
                            Some(_) => " slot stale",
                            None => " slot unknown",
                        });
                        text
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
        self.end_clamps_if_writing(list);
        // Rectangles: completed by the pipeline's geometry shader, else
        // expanded into vertex buffers of their own.
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        let ran = self.prepare(list, false)
            && (!rects || self.rectangles(list, vertices, first) || {
                not_run("rectangle vertices outside their buffer");
                false
            });
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            #[cfg(feature = "diagnostics")]
            self.probe_small(list, false);
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            list.missed_draw();
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "draw");
        }
    }

    /// Whether the bound pipeline's geometry shader completes rectangles.
    fn completes_rectangles(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rectangles)
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
        self.end_clamps_if_writing(list);
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        if rects {
            once!("indexed rectangles not drawn");
            not_run("indexed rectangles");
        }
        let ran = !rects && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            list.missed_draw();
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "indexed");
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. Dispatch records are thread counts in groups of the
    /// size `threads` packs ([`crate::threads::packed`]) unless it is 0.
    /// False if it cannot run on the host.
    /// If an indirect dispatch does not run, the stand-ins
    /// `VICTORY_STAND_IN` names for its program take the place of the
    /// records it would have run ([`Self::instead`]).
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
        threads: u32,
    ) -> bool {
        const DISPATCH: u32 = 2;
        let ran = self.run_indirect(list, (kind, stride, max), arguments, count, threads);
        if !ran && kind == DISPATCH {
            self.instead(list, Some((max, count)));
        }
        ran
    }

    /// [`Self::execute_indirect`] itself.
    fn run_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
        threads: u32,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            list.note_op(|| format!("DROPPED indirect kind {kind} (skipped) {}", crate::pipeline::shaders_of(self.pipeline as This)));
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let pipeline = self.pipeline as This;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| {
            crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
            list.note_op(|| format!("DROPPED indirect kind {kind} max {max} ({what}) {}", crate::pipeline::shaders_of(pipeline)));
        };
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        // A call with no host pipeline cannot run, and that is the common
        // case while the world builds its pipelines: #400 measures 1 400-1 750
        // indirect calls a frame dropped this way. Drop it here, before the
        // argument refresh and the clamp dispatch, which would otherwise copy
        // the arguments and dispatch a compute shader for records nothing
        // reads. Same order as the `is_skipped` check above, for the same
        // reason.
        if let Some(held) = crate::pipeline::held_back(self.pipeline as This) {
            once!(why = held, "an indirect call held back");
            why(held);
            return false;
        }
        if crate::pipeline::host_pipeline(self.pipeline as This).is_none() {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            why("no host pipeline");
            return false;
        }
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
        // A ready multi-wave pipeline needs packed, scaled arguments.
        // Otherwise keep the original clamp-before-prepare order.
        let pipeline = self.pipeline as This;
        let compute = || crate::pipeline::host_pipeline(pipeline).filter(|p| p.compute);
        let dispatch = (kind == DISPATCH).then(compute).flatten();
        let waves = dispatch.map_or(1, group_waves);
        // Thread counts (`VICTORY_DISPATCH_THREADS`) are divided into whole
        // groups where records are copied into a list-owned buffer (the
        // rewrite): the clamp works in place in the title buffer's copy,
        // where it would divide them again at every execution.
        let threads = if dispatch.is_some() { threads } else { 0 };
        let rewrite = waves > 1 || threads != 0;
        let host_stride = if rewrite { 12 } else { stride };
        let Some(mut signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, host_stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        // Zero records execute nothing on the GPU: skip refresh, clamp
        // dispatch, barriers, and pipeline state invalidation.
        if max == 0 {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                self.op_text(format!("indirect kind {kind} max 0 at {arguments:#x}"), kind == DISPATCH)
            });
            if kind == DISPATCH {
                DISPATCHED.fetch_add(1, Ordering::Relaxed);
            } else {
                DRAWN.fetch_add(1, Ordering::Relaxed);
            }
            return true;
        }
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // A multi-wave dispatch's records are rewritten into a list-owned
        // buffer, bounded and scaled (the title's buffers left as they are),
        // after any open window of clamps: like a command that writes GPU
        // memory, it ends the window. The others are clamped in the open
        // window of the list's indirect executions
        // (`crate::indirect::batching`), or on their own here.
        let rewritten = if rewrite {
            crate::indirect::end(list);
            match crate::indirect::dispatch(
                list,
                args.gpu(arguments),
                counter.as_ref().map(|(s, at)| s.gpu(*at)),
                (stride, max, waves),
                threads,
            ) {
                Some(buffer) => Some(buffer),
                // Single-wave records the rewrite does not take (a stride
                // under 12, too many records) run clamped, as group counts.
                None if waves == 1 => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(stride, max, "VICTORY_DISPATCH_THREADS: indirect thread counts not applied (records the rewrite does not take)");
                    });
                    let Some(own) = host_signature(kind, stride) else {
                        why("no host signature");
                        return false;
                    };
                    signature = own;
                    None
                }
                None => {
                    why("dispatch arguments could not be rewritten");
                    return false;
                }
            }
        } else {
            None
        };
        let batched = (rewritten.is_none() && crate::indirect::batching() && max != 0)
            .then(|| {
                // Host addresses as a copy's start and the offset in it.
                let at = |s: &crate::shadow::Shadow, a: u64| (s.gpu(a) - s.offset(a), s.offset(a));
                let count = counter.as_ref().map(|(s, a)| at(s, *a));
                crate::indirect::member(list, at(&args, arguments), count, (kind, stride, max))
            })
            .flatten();
        // Bounded first: the clamp binds a compute pipeline of its own.
        let clamped = match (&rewritten, &batched) {
            (Some(_), _) | (None, Some(_)) => None,
            (None, None) => {
                let Some(clamped) = crate::indirect::bound(
                    list,
                    (&args.buffer, args.gpu(arguments)),
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    (kind, stride, max),
                ) else {
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                Some(clamped)
            }
        };
        // What runs, put back after: the rewritten records (their count in
        // the same buffer), or the list's clamped records.
        let done = |list: &HostList| match (&rewritten, &clamped) {
            (Some(buffer), _) => crate::indirect::done(list, buffer),
            (None, Some(clamped)) => crate::indirect::done(list, clamped),
            (None, None) => {}
        };
        if batched.as_ref().is_none_or(|b| b.opened) {
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
        }
        let ran = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || self.completes_rectangles() || {
                    not_run("rectangles");
                    false
                }) && self.prepare(list, kind == INDEXED)
            }
            MESH => self.prepare_for(list, false, true),
            // A pipeline may become ready during the clamp. It can use
            // these arguments only if their wave count matches.
            DISPATCH => dispatch
                .or_else(compute)
                .is_some_and(|p| group_waves(p) == waves && self.bind(list, p, true)),
            _ => false,
        };
        if !ran {
            done(list);
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => {
                    crate::pipeline::note_drop(self.pipeline as This);
                    list.note_op(|| format!("DROPPED indirect kind {kind} (no host pipeline) {}", crate::pipeline::shaders_of(pipeline)));
                }
                Some(p) => why(&format!(
                    "prepare failed (compute {}, mesh {}): {}",
                    p.compute,
                    p.mesh,
                    NOT_RUN.with(|w| w.replace(""))
                )),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        // Rewritten: packed records of 12 bytes, the count after them.
        // Batched: the clamped records and count, in the list's memory.
        // Else the list's own clamped records (`crate::indirect::bound`) and
        // the title's count.
        let (records, count) = match (&rewritten, &batched, &clamped) {
            (Some(b), _, _) => ((b, 0), Some((b, u64::from(max) * 12))),
            (None, Some(b), _) => ((&b.buffer, b.offset), counter.as_ref().map(|_| (&b.buffer, b.count))),
            (None, None, Some(clamped)) => ((clamped, 0), counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)))),
            (None, None, None) => unreachable!("bounded above"),
        };
        // The count where the execution reads it, for the stand-ins after
        // it: the same value, whatever the records then write.
        let predicate = match &counter {
            Some((s, at)) if kind == DISPATCH && max > 0 && self.names_stand_ins() => {
                count_predicate(list, &s.buffer, s.offset(*at))
            }
            _ => None,
        };
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh) or its own, the signature is
        // cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                records.0,
                records.1,
                count.map(|(b, _)| b),
                count.map_or(0, |(_, at)| at),
            )
        };
        // A listed frame, or any frame for a traced shader: what the records
        // held (after the clamp).
        #[cfg(feature = "diagnostics")]
        {
            let always = !crate::host::traced_shaders().is_empty()
                && crate::host::is_traced(&crate::pipeline::shaders_of(pipeline));
            if always || crate::capture::dump_on() {
                let name = crate::pipeline::shaders_of(pipeline);
                let record_stride = if rewritten.is_some() { 12 } else { stride };
                crate::capture::read_indirect(list, records, count, (kind, record_stride, max), name, always);
            }
        }
        done(list);
        if batched.is_some() && (kind == DISPATCH || self.may_write()) {
            // It may write later executions' records: they clamp after it.
            crate::indirect::end(list);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            if let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) {
                let records = match (&counter, predicate) {
                    (None, _) => Records::Uncounted,
                    (Some(_), Some(predicate)) => Records::Counted(predicate),
                    (Some(_), None) => Records::Unknown,
                };
                self.stand_ins(list, p, Some((max, records)));
            }
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The records an indirect dispatch ran, for the stand-ins after it
/// ([`State::stand_in_after_indirect`]).
#[derive(Clone, Copy)]
enum Records<'a> {
    /// No count: as many as its arguments give, up to its most.
    Uncounted,
    /// As many as its count says, copied into the list's predicate word
    /// ([`count_predicate`]) where the execution read it.
    Counted(&'a ID3D12Resource),
    /// A count no predicate word took.
    Unknown,
}

/// The list's predicate word ([`HostList::predicate`]) holding the 32-bit
/// count at `offset` of `buffer` in its low half (the high half stays
/// zero), copied where this is recorded: just before the execution that
/// reads the count, so the word holds what it read. Common before and
/// after. None without the word.
fn count_predicate<'a>(list: &'a HostList, buffer: &ID3D12Resource, offset: u64) -> Option<&'a ID3D12Resource> {
    use crate::host::transition;
    let predicate = list.predicate()?;
    // SAFETY: an open list; our word, and the count's host copy, which the
    // list keeps.
    unsafe {
        list.list.ResourceBarrier(&[transition(predicate, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
        list.list.CopyBufferRegion(predicate, 0, buffer, offset, 4);
        list.list.ResourceBarrier(&[transition(predicate, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
    }
    Some(predicate)
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
/// copied, textures moved to the state they need. `marked`: a shader's
/// writes through the view mark their pages (every translated shader's do);
/// the host's own (clears) do not.
fn readable(list: &HostList, u: &UsedResource, view: &View, marked: bool) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    // A placed view of memory the GPU writes through another title texture
    // reads that one's host copy.
    if let Some(View::Shader(b, _)) = &placed
        && let Some(a) = crate::resource::alias_source(*b as This)
    {
        crate::texture::alias_copy(list, a, *b as This);
    }
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    // Constant buffers (raw views) to their copy's end at most; others in
    // the copy their host view reads.
    let range = match *view {
        View::Constant(address, size) => crate::shadow::find(address, 1).and_then(|s| {
            let len = u64::from(size).min(s.left(address));
            crate::shadow::find(address, len).map(|s| (s, address, len))
        }),
        _ => crate::descriptor::host::view_copy(view),
    };
    if let Some((shadow, address, len)) = range
        && len > 0
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len, marked);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(&t)
            {
                crate::texture::upload(list, *resource as This, &t);
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
            crate::capture::note_target(list, *resource as This);
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
            // Its console memory: another view of it (a placed one) is
            // another host texture.
            let at = crate::resource::texture_memory(r).map_or(0, |(p, _)| p as u64);
            format!("{}x{} f{} flags {:#x} mem {at:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    // Small buffers (exposure, counters): their title memory now, as floats.
    let small = |range: Option<(u64, u64)>| match range {
        Some((at, len)) if len <= 64 && crate::texture::committed(at as *const u8, len as usize) => {
            // SAFETY: `len` committed bytes of title memory.
            let b = unsafe { std::slice::from_raw_parts(at as *const u8, len as usize) };
            let f: Vec<f32> = b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
            format!(" = {f:?}")
        }
        _ => String::new(),
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This).unwrap_or_else(|| {
            let range = crate::descriptor::host::buffer_range(v);
            format!("buffer {range:?}{}", small(range))
        }),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || {
                let range = crate::descriptor::host::buffer_range(v);
                format!("uav buffer {range:?}{}", small(range))
            },
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are, read
        // through the kernel (asking `VirtualQuery` first cost milliseconds
        // per draw in the title's pools, with every draw noted).
        Some(View::Constant(address, size)) if *size >= 16 => {
            let mut bytes = [0u8; 16];
            if crate::host::read_into(*address as *const u8, &mut bytes) {
                let v: [f32; 4] = std::array::from_fn(|i| f32::from_le_bytes([0, 1, 2, 3].map(|k| bytes[4 * i + k])));
                format!("Constant({address:#x}, {size}) {v:?}")
            } else {
                format!("Constant({address:#x}, {size})")
            }
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// A user-data entry that takes where a root view's address lies in the
/// copy its view shows (not a dword of a root constants parameter).
fn is_view_offset(u: &victory_shader::emit::UserData) -> bool {
    u.root == RootType::Descriptor && u.dword == victory_shader::emit::ROOT_VIEW_OFFSET
}

/// A user-data entry that takes the heap index of a root view's view.
fn is_root_view(u: &victory_shader::emit::UserData) -> bool {
    u.root == RootType::Descriptor && u.dword >= victory_shader::emit::ROOT_SRV_VIEW
}

/// Whether a shader reads root views through views the host makes.
fn has_root_views(t: &victory_shader::emit::HostShader) -> bool {
    t.user_data.iter().any(is_root_view)
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

#[cfg(test)]
mod tests {
    use super::*;

    /// A root signature that indexes the heaps directly needs one of each
    /// kind bound before it is set: our sampler heap stands in for a title
    /// that binds none, which a compute list that only reads views does.
    #[test]
    fn a_list_without_a_sampler_heap_gets_ours() {
        // The title binds both: nothing of ours.
        assert_eq!(State::wanted([0x10, 0x20], false), [0x10, 0x20]);
        // Views only: our sampler heap after them.
        assert_eq!(State::wanted([0x10, 0], false), [0x10, SAMPLERS_OURS]);
        // Neither, with title memory's views needed: ours of both kinds.
        assert_eq!(
            State::wanted([0, 0], true),
            [MEMORY_HEAP, SAMPLERS_OURS]
        );
        // A title sampler heap stays whatever the views are.
        assert_eq!(State::wanted([0, 0x20], true), [MEMORY_HEAP, 0x20]);
    }
    use crate::object::{HRESULT, S_OK};

    /// The fifth dword of a root constants parameter is a constant, not the
    /// offset of a root view.
    #[test]
    fn only_a_root_view_has_a_view_offset() {
        let entry = |root, dword| victory_shader::emit::UserData { index: 0, slot: 0, root, dword };
        assert!(is_view_offset(&entry(RootType::Descriptor, victory_shader::emit::ROOT_VIEW_OFFSET)));
        assert!(!is_view_offset(&entry(RootType::Constants, victory_shader::emit::ROOT_VIEW_OFFSET)));
        assert!(!is_view_offset(&entry(RootType::Descriptor, 0)));
    }
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    /// Reads 16 dwords 64 bytes apart from `offset` past the root view
    /// (heap index in the table) as the translated shaders do: the root
    /// address's offset in the view plus their own.
    const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), RootConstants(num32BitConstants=2, b0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint root; uint offset; }
[RootSignature(RS)]
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(id.x * 4, src.Load(root + offset + id.x * 64));
}
"#;

    /// A depth texture the title denies shader reads, which the host keeps
    /// in its typed format (16- or 32-bit depth), used through an
    /// unordered view: its twin's plane holds the same bytes, copied over
    /// without the conversion and back. A multisample one gets no twin,
    /// and asking for one is no debug layer error.
    #[test]
    fn a_typed_depth_texture_shares_its_bytes_with_its_twin() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D32_FLOAT,
        };
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(1);
        let (out, data) = host.shared_buffer(64 * 64 * 4).unwrap();
        let cs = compile_cs(
            r#"
#define RS "DescriptorTable(UAV(u0)), UAV(u1)"
RWTexture2D<float> d : register(u0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store((id.y * 64 + id.x) * 4, asuint(d[id.xy]));
    d[id.xy] = 0.75;
}
"#,
        );
        // Console format, its unordered view's, the host's; the clear of
        // 0.25 as the shader reads it; 0.75 as the depth texture holds it.
        let cases = [
            (55, 56, DXGI_FORMAT_D16_UNORM, 16384.0 / 65535.0, 49151, 2),
            (40, 41, DXGI_FORMAT_D32_FLOAT, 0.25, 0.75f32.to_bits(), 4),
        ];
        for (console, view, format, read, held, size) in cases {
            let words = texture_words(64, 64, console, 0, 2 | 4 | 8);
            let depth = texture_at_address(title_memory(1 << 20), &words);
            let copy = crate::resource::host_copy(depth).expect("a host depth texture");
            // SAFETY: a live resource.
            let dd = unsafe { copy.texture.GetDesc() };
            assert_eq!(dd.Format, format, "f{console}: the host keeps it typed");
            let uav = View::Unordered(depth as usize, 0, Some([view, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
            slots.make(0, &uav);
            let mut f = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device.GetCopyableFootprints(
                    &dd,
                    0,
                    1,
                    0,
                    Some(&mut f),
                    None,
                    None,
                    Some(&mut total),
                )
            };
            let (raw, raw_data) = host.shared_buffer(total).unwrap();
            let list = host.new_list().unwrap();
            crate::list::ensure_state(&list, depth, 0x10);
            let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            let handle = host.dsv(&copy.texture, Some(&dsv));
            // SAFETY: an open list, the texture in the depth write state.
            unsafe {
                list.list
                    .ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, 0.25, 0, None)
            };
            readable(&list, &used(ResourceType::Uav), &uav, false);
            assert!(!list.take_rebind(), "f{console}: no conversion");
            run(&list, &slots, &cs, 0, Some(&out), (8, 8));
            crate::list::ensure_state(&list, depth, 0x800);
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the copy does not release.
                pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&raw) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
            };
            // SAFETY: an open list, the texture in the copy source state.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
            list.close();
            host.execute(&[&list]);
            for (x, y) in [(0usize, 0usize), (31, 17), (63, 63)] {
                let at = f.Offset as usize + y * f.Footprint.RowPitch as usize + x * size;
                // SAFETY: inside the mapped buffers.
                let (got, bytes) = unsafe {
                    let got = (data.add((y * 64 + x) * 4) as *const f32).read();
                    let bytes = (raw_data.add(at) as *const u32).read_unaligned();
                    (got, if size == 2 { bytes & 0xffff } else { bytes })
                };
                assert!(
                    (got - read).abs() < 1e-7,
                    "f{console} cleared at {x},{y}: {got}"
                );
                assert_eq!(bytes, held, "f{console} written depth at {x},{y}");
            }
        }
        let mut words = texture_words(64, 64, 40, 0, 2 | 4);
        words[9] = 4;
        let depth = texture_at_address(title_memory(1 << 20), &words);
        let copy = crate::resource::host_copy(depth).expect("a host depth texture");
        // SAFETY: a live resource.
        assert_eq!(unsafe { copy.texture.GetDesc() }.SampleDesc.Count, 4);
        let uav = View::Unordered(depth as usize, 0, Some([41, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
        assert!(crate::descriptor::host::make_view(&uav, host.cpu_view()).is_none());
        assert!(copy.twin.get().is_some_and(Option::is_none), "no twin");
        if let Some(e) = host.log_messages() {
            assert!(e.contains("CreatePipelineLibrary"), "debug layer: {e}");
        }
    }

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        compile_cs(SOURCE)
    }

    /// A compute shader with its root signature inside (`RS`).
    fn compile_cs(source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("root"),
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
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: a compiled shader with its root signature inside.
        unsafe {
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

    /// Which pixels of a 64x64 host D32S8 `texture` (title texture `depth`)
    /// pass an EQUAL depth test against a square (two triangles, as the
    /// title's rectangles become) drawn over pixel `k - 1` (row by row) with
    /// z = k times the title's 1/65535 (`0x37800080`), the depth
    /// `VS_TiledLighting` gives draw k: entry `k - 1`.
    fn equal_test_passes(depth: This, texture: &ID3D12Resource) -> Vec<bool> {
        use windows::Win32::Graphics::Dxgi::Common::*;
        const SOURCE: &str = r#"
#define RS "RootFlags(0)"
static const float2 CORNERS[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
[RootSignature(RS)]
float4 vs(uint v : SV_VertexID) : SV_Position {
    uint i = v / 6;
    float2 p = float2(i % 64, i / 64) + CORNERS[v % 6];
    return float4(p.x / 32.0 - 1.0, 1.0 - p.y / 32.0, (float)(i + 1) * asfloat(0x37800080), 1.0);
}
uint ps() : SV_Target { return 1; }
"#;
        let host = crate::host::host().unwrap();
        let blob = |entry: windows::core::PCSTR, target: windows::core::PCSTR| {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names outlive the call.
            unsafe { D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("equal"), None, None, entry, target, 0, 0, &mut code, None) }.unwrap();
            code.unwrap()
        };
        let (vs, ps) = (blob(s!("vs"), s!("vs_5_1")), blob(s!("ps"), s!("ps_5_1")));
        // SAFETY: the blobs' own bytes.
        let bytes = |b: &ID3DBlob| unsafe { D3D12_SHADER_BYTECODE { pShaderBytecode: b.GetBufferPointer(), BytecodeLength: b.GetBufferSize() } };
        // SAFETY: a compiled shader with its root signature inside.
        let root: ID3D12RootSignature = unsafe {
            host.device.CreateRootSignature(0, std::slice::from_raw_parts(vs.GetBufferPointer().cast::<u8>(), vs.GetBufferSize()))
        }
        .unwrap();
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed for the call.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: bytes(&vs),
            PS: bytes(&ps),
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            DSVFormat: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        desc.RTVFormats[0] = DXGI_FORMAT_R32_UINT;
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = true.into();
        desc.DepthStencilState.DepthEnable = true.into();
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
        // SAFETY: a complete description.
        let pso: ID3D12PipelineState = unsafe { host.device.CreateGraphicsPipelineState(&desc) }.unwrap();
        let target_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_UINT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut target: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &target_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut target)
        }
        .unwrap();
        let target = target.unwrap();
        let mut f = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: outputs for one subresource.
        unsafe { host.device.GetCopyableFootprints(&target_desc, 0, 1, 0, Some(&mut f), None, None, Some(&mut total)) };
        let (raw, data) = host.shared_buffer(total).unwrap();
        let list = host.new_list().unwrap();
        crate::list::ensure_state(&list, depth, 0x10);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let (rtv, dsv) = (host.rtv(&target, None), host.dsv(texture, Some(&dsv)));
        // SAFETY: an open list; the target in the render target state, the
        // depth texture in the depth write state; copies into our buffer.
        unsafe {
            let l = &list.list;
            l.ClearRenderTargetView(rtv, &[0.0; 4], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            l.RSSetViewports(&[D3D12_VIEWPORT { Width: 64.0, Height: 64.0, MaxDepth: 1.0, ..Default::default() }]);
            l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { right: 64, bottom: 64, ..Default::default() }]);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.DrawInstanced(64 * 64 * 6, 1, 0, 0);
            l.ResourceBarrier(&[crate::host::transition(&target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&raw),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.close();
        host.execute(&[&list]);
        (0..64 * 64)
            .map(|i| {
                // SAFETY: inside the mapped buffer, by the footprint.
                unsafe { (data.add(f.Offset as usize + (i / 64) * f.Footprint.RowPitch as usize + (i % 64) * 4) as *const u32).read() == 1 }
            })
            .collect()
    }

    /// Dispatches with the console flags `flags` of `counts` (direct, or
    /// one record of an indirect dispatch whose signature argument carries
    /// the flags) to a program of 64-thread groups, with the conversion on:
    /// the groups that ran and the lanes that ran.
    fn marked_groups(indirect: bool, flags: u32, counts: [u32; 3]) -> Option<[u32; 2]> {
        const MARKS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer marks : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint ignored;
    if (lane == 0) {
        marks.InterlockedAdd(0, 1, ignored);
    }
    marks.InterlockedAdd(4, 1, ignored);
}
"#;
        let host = crate::host::host()?;
        let (root, pso) = compile_cs(MARKS);
        let size = Some([64, 1, 1]);
        let (marks, data) = host.shared_buffer(256).unwrap();
        let (args, words) = host.shared_buffer(256).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffers, written before the list runs and
        // read after it has; an open list with everything set.
        unsafe {
            data.cast::<[u32; 2]>().write_volatile([0, 0]);
            words.cast::<[u32; 3]>().write_volatile(counts);
            let clamped = indirect.then(|| {
                let argument = [crate::threads::DISPATCH_SINGLE, flags, 0, 0];
                let threads = crate::threads::indirect_thread_dims(argument).then_some(size).flatten();
                let threads = crate::threads::packed(threads);
                // Single-wave records are clamped into the list's argument
                // buffer (main's 315cea18): thread counts go through the
                // rewrite into a list-owned buffer.
                if threads == 0 {
                    let at = args.GetGPUVirtualAddress();
                    crate::indirect::bound(&list, (&args, at), None, (2, 12, 1)).unwrap()
                } else {
                    crate::indirect::dispatch(&list, args.GetGPUVirtualAddress(), None, (12, 1, 1), threads).unwrap()
                }
            });
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRootUnorderedAccessView(0, marks.GetGPUVirtualAddress());
            match &clamped {
                Some(buffer) => {
                    let signature = host_signature(2, 12).unwrap();
                    list.list.ExecuteIndirect(&signature, 1, buffer, 0, None, 0);
                    crate::indirect::done(&list, buffer);
                }
                None => {
                    let [x, y, z] = crate::threads::as_groups(true, size, flags, counts);
                    list.list.Dispatch(x, y, z);
                }
            }
            list.close();
            host.execute(&[&list]);
            Some(data.cast::<[u32; 2]>().read_volatile())
        }
    }

    /// With `USE_THREAD_DIMENSIONS` (0x20), 1000 threads in groups of 64
    /// launch 16 groups, directly and through indirect execution; the last
    /// group runs whole (1024 lanes, the shortcut in `crate::threads`).
    /// Without the flag the counts stay group counts.
    #[test]
    fn thread_dimensions_launch_whole_groups() {
        for indirect in [false, true] {
            let Some(flagged) = marked_groups(indirect, 0x20, [1000, 1, 1]) else {
                return; // no host GPU here
            };
            assert_eq!(flagged, [16, 1024], "indirect {indirect}");
            let plain = marked_groups(indirect, 0, [1000, 1, 1]).unwrap();
            assert_eq!(plain, [1000, 64_000], "indirect {indirect}");
        }
    }

    /// A title buffer of `bytes` in memory of its own.
    fn buffer(bytes: u64) -> This {
        let desc: [u32; 14] = [1, 0, 0, 0, bytes as u32, (bytes >> 32) as u32, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// Root views (`victory_shader::emit`, "Root SRVs and root UAVs") through
    /// our device: a raw view of the host copy holding the root address,
    /// brought up to date from there to the copy's end, so data 1.5 MiB past
    /// the root address (past the 1 MiB the old root views brought up to
    /// date) reads right; reads past the copy's end read 0 (a host root
    /// descriptor read whatever lay there). Two root addresses in one copy
    /// share its view.
    #[test]
    fn root_views_reach_their_copy_and_read_zeros_past_it() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const MIB: u64 = 1 << 20;
        let (src, dst) = (buffer(4 * MIB), buffer(256));
        let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_uav(null, dst, null, &uav, title_heap.start + 32);
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        let src = src_bytes as u64;
        let cases = [
            // Data 1.5 MiB past the root address.
            (src + 0x100, 0x18_0000u32, (0..16).map(|k| 0x5000 + k).collect::<Vec<u32>>()),
            // The copy's last 256 bytes: four lanes inside, the rest past it.
            (src + 4 * MIB - 256, 0, (0..16).map(|k| if k < 4 { 0x7000 + k } else { 0 }).collect()),
        ];
        for (round, (address, offset, want)) in cases.into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // The title writes after recording, before submitting.
            let mut state = State::default();
            state.set_heaps(&[heap]);
            state.set_view(true, 0, address);
            let (index, at) = state.root_view(&list, true, 0, false).expect("a root view");
            assert_eq!(u64::from(at), address - src, "the root address's offset in the copy");
            state.set_view(true, 1, address - 0x40);
            let other = state.root_view(&list, true, 1, false).expect("a root view");
            assert_eq!(other, (index, at - 0x40), "one view per copy");
            let constants = [at, offset];
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(index));
                list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(1));
                list.list.SetComputeRoot32BitConstants(2, 2, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            for k in 0..16u64 {
                let at = address + u64::from(offset) + 64 * k;
                if at + 4 <= src + 4 * MIB {
                    let value = if round == 0 { 0x5000 + k as u32 } else { 0x7000 + k as u32 };
                    // SAFETY: inside the title buffer.
                    unsafe { (at as *mut u32).write(value) };
                }
            }
            host.execute(&[&list]);
            assert_eq!(out.read_host(dst_bytes as u64, 16), want, "round {round}");
        }
    }

    /// A root view takes a run shorter than 256 slots where the heap has
    /// no longer one, and is made instead of dropping its draw; what the
    /// run has left comes back when the list closes, the rest when it is
    /// reset after running; with no slot at all the draw is dropped and
    /// counted.
    #[test]
    fn root_views_take_short_runs_and_their_slots_come_back() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let heap_host = crate::descriptor::heap(heap).host().expect("a host view heap");
        let (free, total) = heap_host.dynamic_free();
        assert_eq!(free, total, "a new heap's dynamic slots are free");
        // Every dynamic slot held by other lists but 5 in a row.
        let all = heap_host.dynamic(total).unwrap();
        heap_host.release(all + 10, 5);
        let address = |b: This| crate::resource::buffer_bytes(b).unwrap().0 as u64;
        let (a, b) = (address(buffer(256)), address(buffer(256)));
        take_dynamic_drops();

        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[heap]);
        state.set_view(true, 0, a);
        let (index, _) = state.root_view(&list, true, 0, false).expect("a root view from the short run");
        assert_eq!(index, all + 10);
        assert_eq!(heap_host.dynamic_free().0, 0, "the run took all 5");
        list.close();
        state.close();
        assert_eq!(heap_host.dynamic_free().0, 4, "the 4 no view took come back at close");
        assert_eq!(take_dynamic_drops(), DynamicDrops { short_runs: 1, ..Default::default() }, "one short run, no drops");

        // No slot left: the draw is dropped, and counted with its reason.
        let other = host.new_list().unwrap();
        let held = heap_host.dynamic(4).unwrap();
        let mut starved = State::default();
        starved.set_heaps(&[heap]);
        starved.set_view(true, 0, b);
        #[cfg(feature = "diagnostics")]
        NOT_RUN.with(|w| w.set(""));
        assert_eq!(starved.root_view(&other, true, 0, false), None);
        #[cfg(feature = "diagnostics")]
        assert_eq!(NOT_RUN.with(|w| w.replace("")), "no dynamic slots for root views");
        assert_eq!(take_dynamic_drops(), DynamicDrops { root_views: 1, ..Default::default() });
        drop(starved);
        other.close();
        heap_host.release(held, 4);

        // The list ran (submissions are waited for); reset, its slot is free.
        host.execute(&[&list]);
        list.reset();
        state.reset(null);
        assert_eq!(heap_host.dynamic_free().0, 5);
        heap_host.release(all, total);
        assert_eq!(heap_host.dynamic_free(), (total, total));
    }

    /// Two lists making 8-slot tables in turn take them from runs of their
    /// own, so when one is reset its slots come back as one free run, not
    /// as 8-slot holes between the other's: a 200-slot table fits then.
    /// A run's unused tail comes back at close, and a submitted list's
    /// slots are counted as held by a list that ran until its reset.
    #[test]
    fn tables_of_two_lists_come_back_as_one_run() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let key = std::ptr::from_ref(title_heap) as usize;
        let host = title_heap.host().expect("a host view heap");
        let (_, total) = host.dynamic_free();
        // Every dynamic slot held elsewhere but 512 in a row.
        let all = host.dynamic(total).unwrap();
        let base = all + 1000;
        host.release(base, 512);

        let (mut a, mut b) = (State::default(), State::default());
        let (mut tables_a, mut tables_b) = (Vec::new(), Vec::new());
        for _ in 0..16 {
            tables_a.push(a.take_slots(key, host, 8).expect("a table of 8"));
            tables_b.push(b.take_slots(key, host, 8).expect("a table of 8"));
        }
        assert_eq!(tables_a, (0..16).map(|k| base + 8 * k).collect::<Vec<_>>(), "one run");
        assert_eq!(tables_b, (0..16).map(|k| base + 256 + 8 * k).collect::<Vec<_>>(), "another");
        assert_eq!(host.dynamic_free().0, 0);
        a.close();
        b.close();
        assert_eq!(host.dynamic_free().0, 256, "both runs' tails come back at close");

        let held = held_by_ran();
        a.submitted();
        a.submitted();
        assert_eq!(held_by_ran(), held + 128, "a's tables, counted once");
        drop(a);
        assert_eq!(held_by_ran(), held, "back at the reset");
        let mut c = State::default();
        assert_eq!(c.take_slots(key, host, 200), Some(base), "a's slots and its tail are one run");
        drop((b, c));
        assert_eq!(host.dynamic_free().0, 512);
        host.release(all, total);
    }

    /// Title memory of `bytes` (64 KiB aligned) for textures at an address.
    fn title_memory(bytes: usize) -> u64 {
        let layout = std::alloc::Layout::from_size_align(bytes, 1 << 16).unwrap();
        // SAFETY: a non-zero size; leaked, as title memory outlives a test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    /// A 2D resource description as words: size, format, layout, flags.
    fn texture_words(width: u32, height: u32, format: u32, layout: u32, flags: u32) -> [u32; 14] {
        [3, 0, 0, 0, width, 0, height, 1 | 1 << 16, format, 1, 0, layout, flags, 0]
    }

    /// A title texture (resource object) at `address`.
    fn texture_at_address(address: u64, words: &[u32; 14]) -> This {
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut out = std::ptr::null();
        let made = crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
        assert_eq!(made, S_OK);
        out
    }

    /// A placed shader view (device 0x230) of a 2D texture at `address`.
    fn placed_view(address: u64, words: [u32; 14]) -> View {
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[words[8], 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, words, desc)
    }

    fn used(kind: ResourceType) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }
    }

    /// Slots of a shader-visible view heap of ours.
    struct Slots(ID3D12DescriptorHeap, u32);

    impl Slots {
        fn new(n: u32) -> Self {
            let host = crate::host::host().unwrap();
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: n,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain creation.
            let heap = unsafe { host.device.CreateDescriptorHeap(&desc) }.unwrap();
            // SAFETY: a plain query.
            let step = unsafe { host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
            Slots(heap, step)
        }

        fn make(&self, i: u32, view: &View) {
            // SAFETY: a live heap.
            let mut at = unsafe { self.0.GetCPUDescriptorHandleForHeapStart() };
            at.ptr += (i * self.1) as usize;
            assert!(crate::descriptor::host::make_view(view, at).is_some(), "slot {i} made");
        }

        fn gpu(&self, i: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
            // SAFETY: a live shader-visible heap.
            let mut at = unsafe { self.0.GetGPUDescriptorHandleForHeapStart() };
            at.ptr += u64::from(i * self.1);
            at
        }
    }

    /// Records a dispatch of `pso` over `groups` with the table at slot
    /// `table` as parameter 0 and, if given, root UAV `out` as parameter 1.
    fn run(
        list: &HostList,
        slots: &Slots,
        (root, pso): &(ID3D12RootSignature, ID3D12PipelineState),
        table: u32,
        out: Option<&ID3D12Resource>,
        groups: (u32, u32),
    ) {
        // SAFETY: an open list and live objects.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(slots.0.clone())]);
            list.list.SetComputeRootSignature(root);
            list.list.SetPipelineState(pso);
            list.list.SetComputeRootDescriptorTable(0, slots.gpu(table));
            if let Some(out) = out {
                list.list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            }
            list.list.Dispatch(groups.0, groups.1, 1);
        }
    }

    const READ: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u1)"
Texture2D<float4> t : register(t0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store4((id.y * 64 + id.x) * 16, asuint(t.Load(int3(id.xy, 0))));
}
"#;

    /// One texture memory written through an unordered view of a resource
    /// object in one format and read through a placed view in another: a
    /// castable pair shares one host texture (no copy), a pair of other
    /// shapes over the same bytes (block-compressed against plain) is two,
    /// bridged by a copy at the read.
    #[test]
    fn a_texture_written_in_one_format_reads_in_another_through_a_placed_view() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Format written and size, what the shader writes, format read and
        // size, the texel expected, whether one host texture.
        type Case = (u32, u32, &'static str, u32, u32, [f32; 4], bool);
        let cases: [Case; 2] = [
            // R32_UINT holding R9G9B9E5 (1, 0.5, 0.25): the sky's tables.
            (
                42,
                64,
                "RWTexture2D<uint> o : register(u0); void put(uint2 p) { o[p] = 256u | 128u << 9 | 64u << 18 | 16u << 27; }",
                67,
                64,
                [1.0, 0.5, 0.25, 1.0],
                true,
            ),
            // R32G32_UINT blocks read as BC1: solid red blocks.
            (
                17,
                16,
                "RWTexture2D<uint2> o : register(u0); void put(uint2 p) { o[p] = uint2(0xF800, 0); }",
                71,
                64,
                [1.0, 0.0, 0.0, 1.0],
                false,
            ),
        ];
        for (round, (written, size, put, format, read_size, want, shared)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 20);
            let object = texture_at_address(address, &texture_words(size, size, written, 0x105, 4));
            let view = placed_view(address, texture_words(read_size, read_size, format, 0x105, 0));
            let Some(View::Shader(placed, _)) = crate::descriptor::host::placed_texture(&view) else {
                panic!("round {round}: a placed texture");
            };
            let a = crate::resource::host_copy(object).unwrap();
            let b = crate::resource::host_copy(placed as This).unwrap();
            assert_eq!(std::sync::Arc::ptr_eq(&a, &b), shared, "round {round}: one host texture");
            let unordered = View::Unordered(object as usize, 0, Some([written, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
            slots.make(0, &unordered);
            slots.make(1, &view);
            let write = compile_cs(&format!(
                "#define RS \"DescriptorTable(UAV(u0))\"\n{put}\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ put(id.xy); }}"
            ));
            let list = host.new_list().unwrap();
            let copies = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed);
            readable(&list, &used(ResourceType::Uav), &unordered, false);
            run(&list, &slots, &write, 0, None, (size / 8, size / 8));
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 1, Some(&out), (read_size / 8, read_size / 8));
            list.close();
            host.execute(&[&list]);
            let copied = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed) > copies;
            assert_eq!(copied, !shared, "round {round}: a copy only where not shared");
            let n = read_size as usize;
            for i in [0usize, 7, n * 9 + 33, n * n - 1] {
                let (x, y) = (i % n, i / n);
                // SAFETY: inside the mapped output.
                let got = unsafe { (data.add((y * 64 + x) * 16) as *const [f32; 4]).read() };
                assert_eq!(got, want, "round {round} texel {x},{y}");
            }
        }
    }

    /// A texture the title wrote, read through tables by two lists of one
    /// submission, is uploaded once; the next submission uploads it again
    /// only after the title writes it (run perf7 uploaded the same textures
    /// up to 24 times a frame).
    #[test]
    fn a_texture_is_uploaded_once_per_submission_and_only_when_written() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Runtime memory the change journal follows; RGBA8 64x64, linear.
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(64, 64, 28, 0x100, 0));
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(desc));
        slots.make(0, &view);
        let submit = |byte: Option<u8>| {
            if let Some(byte) = byte {
                // SAFETY: the texture's memory, 256-byte rows of 64 texels.
                unsafe { std::ptr::write_bytes(address as *mut u8, byte, 64 * 64 * 4) };
            }
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            for list in &lists {
                readable(list, &used(ResourceType::Srv), &view, false);
                run(list, &slots, &read, 0, Some(&out), (8, 8));
                list.close();
            }
            let before = crate::texture::uploads(object);
            host.execute(&[&lists[0], &lists[1]]);
            // SAFETY: inside the mapped output.
            let texel = unsafe { (data as *const [f32; 4]).read() };
            (crate::texture::uploads(object) - before, texel[0])
        };
        let (n, x) = submit(Some(0x40));
        assert_eq!(n, 1, "one upload for two lists");
        assert!((x - 64.0 / 255.0).abs() < 1e-6);
        assert_eq!(submit(None).0, 0, "not written: not uploaded");
        let (n, x) = submit(Some(0x80));
        assert_eq!(n, 1, "written: uploaded again");
        assert!((x - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(submit(Some(0x80)).0, 0, "the same bytes in a shared page: not uploaded");
    }

    /// A list recorded before another whose host copy writes a texture it
    /// reads, but run after it, reads the host's texels: whether its upload
    /// from title memory is still due is decided when its submission is
    /// made, after the writing list's (`texture::host_writes`), not when it
    /// was recorded.
    #[test]
    fn a_list_run_after_a_host_write_keeps_the_host_texels() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Runtime memory the change journal follows; RGBA8 64x64, linear.
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(64, 64, 28, 0x100, 0));
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(desc));
        slots.make(0, &view);
        let reading = || {
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            list
        };
        // SAFETY: inside the mapped output.
        let texel = || unsafe { (data as *const [f32; 4]).read() }[0];
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        let write = |byte: u8| unsafe { std::ptr::write_bytes(address as *mut u8, byte, 64 * 64 * 4) };
        write(0x40);
        host.execute(&[&reading()]);
        assert!((texel() - 64.0 / 255.0).abs() < 1e-6, "uploaded: {}", texel());
        // The title writes the memory again and a list reading the texture
        // is recorded: its upload is due.
        write(0x80);
        let late = reading();
        // A list recorded after it, holding a host copy into the texture
        // (its mark; what it copies is the host's business), runs first.
        let copying = host.new_list().unwrap();
        crate::texture::host_writes(&copying, object);
        copying.close();
        host.execute(&[&copying]);
        host.execute(&[&late]);
        assert!(
            (texel() - 64.0 / 255.0).abs() < 1e-6,
            "the host's texels kept, not title memory uploaded over them: {}",
            texel()
        );
    }

    /// A texture fed from title memory given up (`resource::evict`) while a
    /// recorded list names it: the list still reads it, and the next use
    /// makes it again with its bytes from title memory.
    #[test]
    fn a_texture_given_up_is_made_again_with_its_bytes_and_recorded_lists_keep_it() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(64, 64, 28, 0x100, 0));
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, 64 * 64 * 4) };
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(desc));
        let record = || {
            slots.make(0, &view);
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            list
        };
        // SAFETY: inside the mapped output.
        let texel = || unsafe { (data as *const [f32; 4]).read() }[0];
        let list = record();
        assert!(crate::resource::give_up_now(object), "fed from title memory");
        assert!(crate::resource::made_copy(object).is_none(), "its title texture let go");
        host.execute(&[&list]);
        assert!((texel() - 64.0 / 255.0).abs() < 1e-6, "the recorded list read it");
        // SAFETY: inside the mapped output.
        unsafe { std::ptr::write_bytes(data, 0, 16) };
        let list = record();
        host.execute(&[&list]);
        assert!((texel() - 64.0 / 255.0).abs() < 1e-6, "made again with its bytes");
        assert_eq!(crate::texture::uploads(object), 1, "uploaded into the new one");
    }

    /// Textures brought up in one submission share staging space (parts
    /// of one upload buffer): each reads its own bytes.
    #[test]
    fn textures_uploaded_together_each_read_their_own_bytes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ);
        let outs: Vec<_> = (0..2).map(|_| host.shared_buffer(64 * 64 * 16).unwrap()).collect();
        let pool = crate::watch::tests::pool(2);
        let views: Vec<View> = (0..2usize)
            .map(|k| {
                let address = (pool + k * crate::watch::PAGE) as u64;
                // SAFETY: the texture's memory in our pool, 256-byte rows of 64 texels.
                unsafe { std::ptr::write_bytes(address as *mut u8, 0x40 * (k as u8 + 1), 64 * 64 * 4) };
                let object = texture_at_address(address, &texture_words(64, 64, 28, 0x100, 0));
                let mut desc = [0u32; 10];
                desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                View::Shader(object as usize, Some(desc))
            })
            .collect();
        let lists: Vec<_> = (0..2usize)
            .map(|k| {
                slots.make(k as u32, &views[k]);
                let list = host.new_list().unwrap();
                readable(&list, &used(ResourceType::Srv), &views[k], false);
                run(&list, &slots, &read, k as u32, Some(&outs[k].0), (8, 8));
                list.close();
                list
            })
            .collect();
        host.execute(&[&lists[0], &lists[1]]);
        for (k, (_, data)) in outs.iter().enumerate() {
            // SAFETY: inside the mapped output.
            let texel = unsafe { (*data as *const [f32; 4]).read() };
            assert!((texel[0] - (0x40 * (k + 1)) as f32 / 255.0).abs() < 1e-6, "texture {k}: {}", texel[0]);
        }
    }

    /// A texture written in a page that holds only its own bytes is uploaded
    /// without hashing them (a movie frame rewritten every frame), so even
    /// the same bytes again are uploaded.
    #[test]
    fn a_texture_owning_its_pages_is_uploaded_on_any_write() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // RGBA8 128x128, linear: one whole 64 KiB page.
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(128, 128, 28, 0x100, 0));
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(desc));
        slots.make(0, &view);
        let submit = |byte: u8| {
            // SAFETY: the texture's memory, one page.
            unsafe { std::ptr::write_bytes(address as *mut u8, byte, crate::watch::PAGE) };
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            let before = crate::texture::uploads(object);
            host.execute(&[&list]);
            // SAFETY: inside the mapped output.
            let texel = unsafe { (data as *const [f32; 4]).read() };
            (crate::texture::uploads(object) - before, texel[0])
        };
        assert_eq!(submit(0x40).0, 1);
        assert_eq!(submit(0x40).0, 1, "its own page written: uploaded");
        let (n, x) = submit(0x80);
        assert_eq!(n, 1);
        assert!((x - 128.0 / 255.0).abs() < 1e-6, "a reused staging buffer carries the new bytes: {x}");
    }

    /// A title write into one mip of a texture uploads that mip alone (the
    /// pages of the others did not change), and the others keep their
    /// contents.
    #[test]
    fn a_write_into_one_mip_uploads_that_mip_alone() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // RGBA8 512x512 with two mips, linear, in memory the journal follows.
        let address = crate::watch::tests::pool(32) as u64;
        let mut words = texture_words(512, 512, 28, 0x100, 0);
        words[7] = 1 | 2 << 16;
        let object = texture_at_address(address, &words);
        let (_, desc) = crate::resource::texture_memory(object).unwrap();
        let layout = crate::resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        let mip1 = layout.mips[1].offset as usize;
        // SAFETY: the texture's memory (its layout's size).
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, layout.size as usize) };
        let mut view = [0u32; 10];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 2]);
        let view = View::Shader(object as usize, Some(view));
        slots.make(0, &view);
        let submit = || {
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            let before = crate::texture::copied(object);
            host.execute(&[&list]);
            // SAFETY: inside the mapped output.
            let texel = unsafe { (data as *const [f32; 4]).read() };
            (crate::texture::copied(object) - before, texel[0])
        };
        assert_eq!(submit().0, 2, "both mips at first");
        // SAFETY: mip 1's bytes (256x256 texels of 4 bytes).
        unsafe { std::ptr::write_bytes((address as usize + mip1) as *mut u8, 0x80, 256 * 256 * 4) };
        let (n, x) = submit();
        assert_eq!(n, 1, "mip 1 alone");
        assert!((x - 64.0 / 255.0).abs() < 1e-6, "mip 0 kept: {x}");
    }

    /// The ambient stand-in (`crate::ambient`): a program writes three
    /// unordered views of a title table; after it, the views the rules name
    /// for its hash hold their values (an R32_UINT texture the R9G9B9E5
    /// packing, an RGBA16F one the colour as halves), the view no rule of
    /// its names keeps what it wrote, and a program no rule names changes
    /// nothing.
    #[test]
    fn the_ambient_stand_in_fills_the_named_views_after_the_named_program() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const N: u32 = 16;
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        // R32_UINT, RGBA16F, R32_UINT.
        let formats = [42u32, 10, 42];
        let textures: Vec<This> = formats
            .iter()
            .map(|&f| texture_at_address(title_memory(1 << 20), &texture_words(N, N, f, 0x105, 4)))
            .collect();
        let uavs: Vec<View> = textures
            .iter()
            .zip(formats)
            .map(|(&t, f)| View::Unordered(t as usize, 0, Some([f, 4, 0, 0, 0, 0, 0, 0, 0, 0])))
            .collect();
        for (i, (&t, f)) in textures.iter().zip(formats).enumerate() {
            let desc = [f, 4, 0, 0, 0, 0, 0, 0, 0, 0];
            crate::descriptor::create_uav(null, t, null, &desc, title_heap.start + 32 * i as u64);
        }
        let slots = Slots::new(6);
        for (i, v) in uavs.iter().enumerate() {
            slots.make(i as u32, v);
        }
        let reads: Vec<View> = textures
            .iter()
            .zip(formats)
            .map(|(&t, f)| View::Shader(t as usize, Some([f, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0])))
            .collect();
        for (i, v) in reads.iter().enumerate() {
            slots.make(3 + i as u32, v);
        }
        let write = compile_cs(
            r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=3))"
RWTexture2D<uint> a : register(u0);
RWTexture2D<float4> b : register(u1);
RWTexture2D<uint> c : register(u2);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    a[id.xy] = 0xdeadbeef;
    b[id.xy] = float4(9, 9, 9, 9);
    c[id.xy] = 0xdeadbeef;
}
"#,
        );
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=3)), UAV(u1)"
Texture2D<uint> a : register(t0);
Texture2D<float4> b : register(t1);
Texture2D<uint> c : register(t2);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int3 at = int3(id.xy, 0);
    uint i = (id.y * 16 + id.x) * 32;
    o.Store2(i, uint2(a.Load(at), c.Load(at)));
    o.Store4(i + 16, asuint(b.Load(at)));
}
"#,
        );
        let (out, data) = host.shared_buffer(u64::from(N * N) * 32).unwrap();
        let rules = crate::ambient::parse(
            "aaaaaaaaaaaaaaaa:s1+0:rgb9e5=0.6,0.588,0.564; AAAAAAAAAAAAAAAA:1+8:0.25,0.5,1; bbbbbbbbbbbbbbbb:s1+16:0x12345678",
        )
        .unwrap();
        let program_uses: Vec<UsedResource> = (0..3)
            .map(|i| UsedResource { slot: 1, table_offset_dwords: 8 * i, ..used(ResourceType::Uav) })
            .collect();
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[heap]);
        state.set_table(true, 1, title_heap.start);
        for v in &uavs {
            readable(&list, &used(ResourceType::Uav), v, false);
        }
        run(&list, &slots, &write, 0, None, (N / 8, N / 8));
        let before = crate::ambient::filled();
        assert_eq!(state.stand_in(&list, "aaaaaaaaaaaaaaaa", Some(&program_uses), &rules, None), 2, "the two views of its rules");
        assert_eq!(state.stand_in(&list, "cccccccccccccccc", Some(&program_uses), &rules, None), 0, "a program no rule names");
        assert_eq!(crate::ambient::filled() - before, 2);
        for v in &reads {
            readable(&list, &used(ResourceType::Srv), v, false);
        }
        run(&list, &slots, &read, 3, Some(&out), (N / 8, N / 8));
        list.close();
        host.execute(&[&list]);
        let grey = crate::ambient::rgb9e5([0.6, 0.588, 0.564]);
        for i in [0usize, 5, 7 * 16 + 9, 255] {
            // SAFETY: inside the mapped output.
            let (words, colour) = unsafe {
                (
                    (data.add(i * 32) as *const [u32; 2]).read(),
                    (data.add(i * 32 + 16) as *const [f32; 4]).read(),
                )
            };
            assert_eq!(words, [grey, 0xdeadbeef], "texel {i}: the packed grey, and the unnamed view as written");
            assert_eq!(colour, [0.25, 0.5, 1.0, 1.0], "texel {i}: the colour as halves");
        }
    }

    /// The ambient stand-in after an indirect dispatch clears only where
    /// records ran: a view is seeded with a
    /// sentinel, an indirect dispatch of a program that writes it runs,
    /// then its stand-in. With at most one record and a count of zero the
    /// sentinel survives; with a count of one the record writes the view
    /// and the stand-in clears it after; with at most no records nothing
    /// changes; with no count the stand-in clears. The count sits at an
    /// offset that is not 8-byte aligned with a nonzero word after it, so a
    /// 64-bit predicate read in place would get it wrong.
    #[test]
    fn the_ambient_stand_in_follows_only_indirect_dispatches_that_ran_records() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const N: u32 = 16;
        const SENTINEL: u32 = 0x5e5e_5e5e;
        const RECORD: u32 = 0xdead_beef;
        const STAND_IN: u32 = 0x1234_5678;
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        // R32_UINT.
        let texture = texture_at_address(title_memory(1 << 20), &texture_words(N, N, 42, 0x105, 4));
        let desc = [42, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        let uav = View::Unordered(texture as usize, 0, Some(desc));
        crate::descriptor::create_uav(null, texture, null, &desc, title_heap.start);
        let srv = View::Shader(texture as usize, Some([42, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0]));
        let slots = Slots::new(2);
        slots.make(0, &uav);
        slots.make(1, &srv);
        let writes = |value: u32| {
            compile_cs(&format!(
                r#"
#define RS "DescriptorTable(UAV(u0))"
RWTexture2D<uint> a : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    a[id.xy] = {value}u;
}}
"#
            ))
        };
        let (seed, record) = (writes(SENTINEL), writes(RECORD));
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u1)"
Texture2D<uint> t : register(t0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store((id.y * 16 + id.x) * 4, t.Load(int3(id.xy, 0)));
}
"#,
        );
        let (out, data) = host.shared_buffer(u64::from(N * N) * 4).unwrap();
        let signature = host_signature(2, 12).expect("a dispatch signature");
        let (arguments, words) = host.shared_buffer(12).unwrap();
        // SAFETY: the mapped record, written before any list reads it.
        unsafe { (words as *mut [u32; 3]).write([N / 8, N / 8, 1]) };
        let (counts, count_words) = host.shared_buffer(16).unwrap();
        let rules = crate::ambient::parse(&format!("aaaaaaaaaaaaaaaa:s1+0:{STAND_IN:#x}")).unwrap();
        let program_uses = [UsedResource { slot: 1, ..used(ResourceType::Uav) }];
        let uav_barrier = || D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
            },
        };
        // (most records, count, the view after, stand-ins recorded)
        let cases = [
            (1, Some(0), SENTINEL, 1),
            (1, Some(1), STAND_IN, 1),
            (0, None, SENTINEL, 0),
            (1, None, STAND_IN, 1),
        ];
        for (max, count, want, recorded) in cases {
            if let Some(n) = count {
                // SAFETY: the mapped words, written before the list runs.
                unsafe { (count_words as *mut [u32; 4]).write([1, n, u32::MAX, u32::MAX]) };
            }
            let list = host.new_list().unwrap();
            let mut state = State::default();
            state.set_heaps(&[heap]);
            state.set_table(true, 1, title_heap.start);
            readable(&list, &used(ResourceType::Uav), &uav, false);
            run(&list, &slots, &seed, 0, None, (N / 8, N / 8));
            // SAFETY: an open list and live objects.
            unsafe {
                list.list.ResourceBarrier(&[
                    uav_barrier(),
                    crate::host::transition(&arguments, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
                ]);
                list.list.SetComputeRootSignature(&record.0);
                list.list.SetPipelineState(&record.1);
                list.list.SetComputeRootDescriptorTable(0, slots.gpu(0));
            }
            // The count where the execution reads it.
            let records = match count {
                Some(_) => Records::Counted(count_predicate(&list, &counts, 4).expect("a predicate word")),
                None => Records::Uncounted,
            };
            // SAFETY: an open list and live objects.
            unsafe {
                list.list.ExecuteIndirect(&signature, max, &arguments, 0, count.map(|_| &counts), count.map_or(0, |_| 4));
                list.list.ResourceBarrier(&[crate::host::transition(
                    &arguments,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            assert_eq!(
                state.stand_in_after_indirect(&list, "aaaaaaaaaaaaaaaa", Some(&program_uses), &rules, max, records),
                recorded,
                "max {max} count {count:?}: stand-ins recorded"
            );
            readable(&list, &used(ResourceType::Srv), &srv, false);
            run(&list, &slots, &read, 1, Some(&out), (N / 8, N / 8));
            list.close();
            host.execute(&[&list]);
            for i in [0usize, 5, 7 * 16 + 9, 255] {
                // SAFETY: inside the mapped output, after the list ran.
                let got = unsafe { (data.add(i * 4) as *const u32).read() };
                assert_eq!(got, want, "max {max} count {count:?} texel {i}: {got:#x}");
            }
        }
    }

    /// Clears of one view in one list share one dynamic slot (a stand-in
    /// clears its view at each dispatch of its program): eight clears of a
    /// view take one, a clear of another view one more, and the views hold
    /// their last clears; the slots come back when the list is reset.
    #[test]
    fn clears_of_one_view_in_a_list_share_a_dynamic_slot() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const N: u32 = 16;
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let heap_host = crate::descriptor::heap(heap).host().expect("a host view heap");
        // R32_UINT.
        let textures: Vec<This> = (0..2)
            .map(|_| texture_at_address(title_memory(1 << 20), &texture_words(N, N, 42, 0x105, 4)))
            .collect();
        let uavs: Vec<View> = textures
            .iter()
            .map(|&t| View::Unordered(t as usize, 0, Some([42, 4, 0, 0, 0, 0, 0, 0, 0, 0])))
            .collect();
        let slots = Slots::new(2);
        for (i, &t) in textures.iter().enumerate() {
            slots.make(i as u32, &View::Shader(t as usize, Some([42, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0])));
        }
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2)), UAV(u1)"
Texture2D<uint> a : register(t0);
Texture2D<uint> b : register(t1);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int3 at = int3(id.xy, 0);
    o.Store2((id.y * 16 + id.x) * 8, uint2(a.Load(at), b.Load(at)));
}
"#,
        );
        let (out, data) = host.shared_buffer(u64::from(N * N) * 8).unwrap();
        let free = heap_host.dynamic_free();
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[heap]);
        for k in 1..=8 {
            assert!(state.clear_view(&list, &uavs[0], [k, 0, 0, 0], false, &[], true, None), "clear {k}");
        }
        assert_eq!(heap_host.dynamic_free(), free - 1, "eight clears of one view, one slot");
        assert!(state.clear_view(&list, &uavs[1], [0x77, 0, 0, 0], false, &[], true, None));
        assert_eq!(heap_host.dynamic_free(), free - 2, "another view, another slot");
        assert!(state.clear_view(&list, &uavs[0], [0x55, 0, 0, 0], false, &[], true, None));
        assert_eq!(heap_host.dynamic_free(), free - 2, "the first view's slot again");
        for t in &textures {
            let srv = View::Shader(*t as usize, Some([42, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0]));
            readable(&list, &used(ResourceType::Srv), &srv, false);
        }
        run(&list, &slots, &read, 0, Some(&out), (N / 8, N / 8));
        list.close();
        host.execute(&[&list]);
        for i in [0usize, 5, 7 * 16 + 9, 255] {
            // SAFETY: inside the mapped output, after the list ran.
            let words = unsafe { (data.add(i * 8) as *const [u32; 2]).read() };
            assert_eq!(words, [0x55, 0x77], "texel {i}: each view's last clear");
        }
        list.reset();
        state.reset(null);
        assert_eq!(heap_host.dynamic_free(), free, "the slots come back at reset");
        // A new recording takes a slot afresh.
        state.set_heaps(&[heap]);
        assert!(state.clear_view(&list, &uavs[0], [1, 0, 0, 0], false, &[], true, None));
        assert_eq!(heap_host.dynamic_free(), free - 1);
        list.close();
        host.execute(&[&list]);
        list.reset();
        state.reset(null);
    }

    /// A title compute pipeline of `bytes`, which do not translate: it never
    /// gets a host pipeline, so its dispatches do not run (the gate), and
    /// its program's hash.
    fn gated_compute(bytes: &'static [u8]) -> (This, String) {
        #[repr(C)]
        struct ComputeDesc {
            root: This,
            code: *const u8,
            size: usize,
        }
        let desc = ComputeDesc { root: std::ptr::null(), code: bytes.as_ptr(), size: bytes.len() };
        let mut out = std::ptr::null();
        let made = crate::pipeline::create_compute(
            std::ptr::null(),
            std::ptr::from_ref(&desc).cast(),
            &crate::resource::PAGEABLE_IIDS[0],
            &mut out,
        );
        assert_eq!(made, S_OK);
        (out, blake3::hash(bytes).to_hex()[..16].to_owned())
    }

    /// `VICTORY_STAND_IN` (`crate::ambient`): dispatches of programs that
    /// cannot run (no host pipeline: the gate a held-back program meets
    /// too) clear the views their rules name in their place, direct and
    /// indirect; a program no rule names and an indirect draw clear
    /// nothing, and nothing of the gated programs runs.
    #[test]
    fn stand_ins_take_the_place_of_dispatches_that_do_not_run() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const N: u32 = 16;
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        // R16_FLOAT (a shadow mask, say), R8_UINT, R16_FLOAT.
        let formats = [54u32, 62, 54];
        let textures: Vec<This> = formats
            .iter()
            .map(|&f| texture_at_address(title_memory(1 << 20), &texture_words(N, N, f, 0x105, 4)))
            .collect();
        let uavs: Vec<View> = textures
            .iter()
            .zip(formats)
            .map(|(&t, f)| View::Unordered(t as usize, 0, Some([f, 4, 0, 0, 0, 0, 0, 0, 0, 0])))
            .collect();
        for (i, (&t, f)) in textures.iter().zip(formats).enumerate() {
            let desc = [f, 4, 0, 0, 0, 0, 0, 0, 0, 0];
            crate::descriptor::create_uav(null, t, null, &desc, title_heap.start + 32 * i as u64);
        }
        let slots = Slots::new(6);
        for (i, v) in uavs.iter().enumerate() {
            slots.make(i as u32, v);
        }
        let reads: Vec<View> = textures
            .iter()
            .zip(formats)
            .map(|(&t, f)| View::Shader(t as usize, Some([f, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0])))
            .collect();
        for (i, v) in reads.iter().enumerate() {
            slots.make(3 + i as u32, v);
        }
        // What the views held before: last frame's, say.
        let write = compile_cs(
            r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=3))"
RWTexture2D<float> a : register(u0);
RWTexture2D<uint> b : register(u1);
RWTexture2D<float> c : register(u2);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    a[id.xy] = 0.25;
    b[id.xy] = 7;
    c[id.xy] = 0.25;
}
"#,
        );
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=3)), UAV(u1)"
Texture2D<float> a : register(t0);
Texture2D<uint> b : register(t1);
Texture2D<float> c : register(t2);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int3 at = int3(id.xy, 0);
    o.Store3((id.y * 16 + id.x) * 12, uint3(asuint(a.Load(at)), b.Load(at), asuint(c.Load(at))));
}
"#,
        );
        let (out, data) = host.shared_buffer(u64::from(N * N) * 12).unwrap();
        let (direct, direct_hash) = gated_compute(b"stand-in test: a program that does not translate (direct)");
        let (indirect, indirect_hash) = gated_compute(b"stand-in test: a program that does not translate (indirect)");
        let (unnamed, _) = gated_compute(b"stand-in test: a program no rule names");
        assert!(crate::pipeline::host_pipeline(direct).is_none(), "the gate");
        let rules: &'static [crate::ambient::Rule] = Vec::leak(
            crate::ambient::parse(&format!(
                "{direct_hash}:s1+0:1,1,1; {direct_hash}:s1+8:mask; {indirect_hash}:s1+16:0.5,0.5,0.5"
            ))
            .unwrap(),
        );
        crate::ambient::TEST_INSTEAD.with(|t| t.set(Some(rules)));
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[heap]);
        state.set_table(true, 1, title_heap.start);
        for v in &uavs {
            readable(&list, &used(ResourceType::Uav), v, false);
        }
        run(&list, &slots, &write, 0, None, (N / 8, N / 8));
        let filled = crate::ambient::filled();
        let ran = dispatched();
        state.set_pipeline(direct);
        state.dispatch(&list, 1, 1, 1);
        assert_eq!(crate::ambient::filled() - filled, 2, "both views of the direct program's rules");
        state.set_pipeline(indirect);
        assert!(!state.execute_indirect(&list, (2, 12, 1), 0, None), "the indirect dispatch does not run");
        assert_eq!(crate::ambient::filled() - filled, 3, "the view of the indirect program's rule");
        state.set_pipeline(direct);
        assert!(!state.execute_indirect(&list, (0, 16, 1), 0, None));
        state.set_pipeline(unnamed);
        state.dispatch(&list, 1, 1, 1);
        assert_eq!(crate::ambient::filled() - filled, 3, "nothing for an indirect draw or a program no rule names");
        assert_eq!(dispatched(), ran, "nothing of the programs ran");
        crate::ambient::TEST_INSTEAD.with(|t| t.set(None));
        for v in &reads {
            readable(&list, &used(ResourceType::Srv), v, false);
        }
        run(&list, &slots, &read, 3, Some(&out), (N / 8, N / 8));
        list.close();
        host.execute(&[&list]);
        for i in [0usize, 5, 7 * 16 + 9, 255] {
            // SAFETY: inside the mapped output.
            let [a, b, c] = unsafe { (data.add(i * 12) as *const [u32; 3]).read() };
            assert_eq!((f32::from_bits(a), b, f32::from_bits(c)), (1.0, 0xff, 0.5), "texel {i}");
        }
        let release = |o: This| crate::object::slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for p in [direct, indirect, unnamed] {
            assert_eq!(release(p), 0);
        }
    }

    /// `VICTORY_STAND_IN` in place of an indirect dispatch that does not
    /// run stands in for the records it would have run: with at most one
    /// record and a count of zero (in a title buffer, at an offset that is
    /// not 8-byte aligned, a nonzero word after it) the view keeps its
    /// sentinel; with a count of one it is cleared; with at most no records
    /// it keeps the sentinel; with no count it is cleared.
    #[test]
    fn stand_ins_in_place_of_indirect_dispatches_follow_their_record_counts() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const N: u32 = 16;
        const SENTINEL: u32 = 0x5e5e_5e5e;
        const STAND_IN: u32 = 0x1234_5678;
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        // R32_UINT.
        let texture = texture_at_address(title_memory(1 << 20), &texture_words(N, N, 42, 0x105, 4));
        let desc = [42, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        let uav = View::Unordered(texture as usize, 0, Some(desc));
        crate::descriptor::create_uav(null, texture, null, &desc, title_heap.start);
        let srv = View::Shader(texture as usize, Some([42, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0]));
        let slots = Slots::new(2);
        slots.make(0, &uav);
        slots.make(1, &srv);
        let seed = compile_cs(&format!(
            r#"
#define RS "DescriptorTable(UAV(u0))"
RWTexture2D<uint> a : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    a[id.xy] = {SENTINEL}u;
}}
"#
        ));
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u1)"
Texture2D<uint> t : register(t0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store((id.y * 16 + id.x) * 4, t.Load(int3(id.xy, 0)));
}
"#,
        );
        let (out, data) = host.shared_buffer(u64::from(N * N) * 4).unwrap();
        // The title buffer holding the count.
        let counts = title_memory(1 << 16);
        crate::shadow::register(counts as usize, counts, 1 << 16);
        let (gated, hash) = gated_compute(b"stand-in test: an indirect dispatch that does not translate");
        let rules: &'static [crate::ambient::Rule] =
            Vec::leak(crate::ambient::parse(&format!("{hash}:s1+0:{STAND_IN:#x}")).unwrap());
        crate::ambient::TEST_INSTEAD.with(|t| t.set(Some(rules)));
        // (most records, count, the view after)
        let cases = [(1, Some(0), SENTINEL), (1, Some(1), STAND_IN), (0, None, SENTINEL), (1, None, STAND_IN)];
        for (max, count, want) in cases {
            if let Some(n) = count {
                // SAFETY: the title buffer's words, written before the list runs.
                unsafe { (counts as *mut [u32; 4]).write([1, n, u32::MAX, u32::MAX]) };
            }
            let list = host.new_list().unwrap();
            let mut state = State::default();
            state.set_heaps(&[heap]);
            state.set_table(true, 1, title_heap.start);
            readable(&list, &used(ResourceType::Uav), &uav, false);
            run(&list, &slots, &seed, 0, None, (N / 8, N / 8));
            state.set_pipeline(gated);
            let ran = state.execute_indirect(&list, (2, 12, max), 0, count.map(|_| counts + 4));
            assert!(!ran, "max {max} count {count:?}: the dispatch does not run");
            readable(&list, &used(ResourceType::Srv), &srv, false);
            run(&list, &slots, &read, 1, Some(&out), (N / 8, N / 8));
            list.close();
            host.execute(&[&list]);
            for i in [0usize, 5, 7 * 16 + 9, 255] {
                // SAFETY: inside the mapped output, after the list ran.
                let got = unsafe { (data.add(i * 4) as *const u32).read() };
                assert_eq!(got, want, "max {max} count {count:?} texel {i}: {got:#x}");
            }
        }
        crate::ambient::TEST_INSTEAD.with(|t| t.set(None));
        crate::shadow::unregister(counts as usize, counts);
        let release = |o: This| crate::object::slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(gated), 0);
    }

    /// A depth texture with stencil (the console's 16-bit depth, the
    /// host's 32-bit float) written through unordered views of its planes,
    /// then read as depth and stencil: the 16-bit depth arrives as the
    /// float the title's shaders compute for it, the stencil as written,
    /// and a draw at that depth passes an EQUAL test. Rendered as depth (a
    /// clear), then used through the unordered view: back in 16 bits.
    #[test]
    fn a_depth_texture_written_through_unordered_views_reads_as_depth() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let address = title_memory(1 << 22);
        let depth = texture_at_address(address, &texture_words(64, 64, 118, 0, 2));
        let copy = crate::resource::host_copy(depth).expect("a host depth texture");
        let planes = [
            View::Unordered(depth as usize, 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0])),
            View::Unordered(depth as usize, 0, Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0])),
        ];
        slots.make(0, &planes[0]);
        slots.make(1, &planes[1]);
        let reads = [
            View::Shader(depth as usize, Some([119, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0])),
            View::Shader(depth as usize, Some([120, 4, 0x1688, 0, 0, 1, 1, 0, 0, 0])),
        ];
        slots.make(2, &reads[0]);
        slots.make(3, &reads[1]);
        let write = compile_cs(
            r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
RWTexture2D<float> d : register(u0);
RWTexture2D<uint> s : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    d[id.xy] = (1 + id.x + 64 * id.y) * (1.0 / 65535.0);
    s[id.xy] = (id.x ^ id.y) & 0xff;
}
"#,
        );
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2)), UAV(u1)"
Texture2D<float> d : register(t0);
Texture2D<uint2> s : register(t1);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store2((id.y * 64 + id.x) * 8, uint2(asuint(d.Load(int3(id.xy, 0))), s.Load(int3(id.xy, 0)).g));
}
"#,
        );
        let (out, data) = host.shared_buffer(64 * 64 * 8).unwrap();
        // The depth texture's planes as bytes, read back after.
        // SAFETY: a live resource.
        let dd = unsafe { copy.texture.GetDesc() };
        let mut f = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        let mut total = 0;
        // SAFETY: outputs for two subresources.
        unsafe { host.device.GetCopyableFootprints(&dd, 0, 2, 0, Some(f.as_mut_ptr()), None, None, Some(&mut total)) };
        let (raw, raw_data) = host.shared_buffer(total).unwrap();
        let list = host.new_list().unwrap();
        for v in &planes {
            readable(&list, &used(ResourceType::Uav), v, false);
        }
        run(&list, &slots, &write, 0, None, (8, 8));
        for v in &reads {
            readable(&list, &used(ResourceType::Srv), v, false);
        }
        assert!(list.take_rebind(), "the conversion ran in the list");
        run(&list, &slots, &read, 2, Some(&out), (8, 8));
        crate::list::ensure_state(&list, depth, 0x800);
        let at = |r: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let to = |fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&raw) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        // SAFETY: an open list, the texture in the copy source state.
        unsafe {
            list.list.CopyTextureRegion(&to(f[0]), 0, 0, 0, &at(&copy.texture, 0), None);
            list.list.CopyTextureRegion(&to(f[1]), 0, 0, 0, &at(&copy.texture, 1), None);
        }
        list.close();
        host.execute(&[&list]);
        // The title's float for 16-bit value u (its 1/65535 is 0x37800080).
        let float = |u: u32| (u as f32 * f32::from_bits(0x3780_0080)).to_bits();
        for (x, y) in [(0usize, 0usize), (5, 0), (63, 2), (17, 40), (63, 63)] {
            let k = (1 + x + 64 * y) as u32;
            // SAFETY: inside the mapped buffers.
            let [depth_bits, stencil] = unsafe { (data.add((y * 64 + x) * 8) as *const [u32; 2]).read() };
            let got = f32::from_bits(depth_bits);
            assert!((got - k as f32 / 65535.0).abs() < 1e-7, "depth at {x},{y}: {got}");
            assert_eq!(stencil, ((x ^ y) & 0xff) as u32, "stencil at {x},{y}");
            // SAFETY: inside the mapped buffer, by the footprints.
            let (word, byte) = unsafe {
                (
                    (raw_data.add(f[0].Offset as usize + y * f[0].Footprint.RowPitch as usize + x * 4) as *const u32).read(),
                    raw_data.add(f[1].Offset as usize + y * f[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(word, float(k), "depth at {x},{y}");
            assert_eq!(u32::from(byte), ((x ^ y) & 0xff) as u32);
        }
        // Tested EQUAL against a point at the depth the title's tiled
        // lighting gives value k (k times its 1/65535), as the console's
        // 16-bit depth passes it for every k. In 24-bit depth on the RX 6700
        // XT only k = 258 of 1..=4096 passed.
        let passed = equal_test_passes(depth, &copy.texture);
        let failed: Vec<u32> = (1..=4096).filter(|&k| !passed[k as usize - 1]).collect();
        assert!(failed.is_empty(), "EQUAL fails at {failed:?}");
        // Rendered as depth (a clear of both planes), then used through
        // the unordered view: copied to the twin, converted back.
        list.reset();
        crate::list::ensure_state(&list, depth, 0x10);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let handle = host.dsv(&copy.texture, Some(&dsv));
        // SAFETY: an open list, the texture in the depth write state.
        unsafe {
            list.list.ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5, 0x11, None)
        };
        readable(&list, &used(ResourceType::Uav), &planes[0], false);
        let twin = copy.twin.get().and_then(Option::as_ref).expect("a twin");
        let mut g = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        for (p, plane) in twin.planes.iter().enumerate() {
            // SAFETY: a live resource.
            let pd = unsafe { plane.texture.GetDesc() };
            // SAFETY: an output for one subresource.
            unsafe { host.device.GetCopyableFootprints(&pd, 0, 1, f[p].Offset, Some(&mut g[p]), None, None, None) };
            if let Some(b) = list.move_texture(plane, 0x800) {
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[b]) };
            }
        }
        // SAFETY: an open list, the twin's planes in the copy source state.
        unsafe {
            list.list.CopyTextureRegion(&to(g[0]), 0, 0, 0, &at(&twin.planes[0].texture, 0), None);
            list.list.CopyTextureRegion(&to(g[1]), 0, 0, 0, &at(&twin.planes[1].texture, 0), None);
        }
        list.close();
        host.execute(&[&list]);
        for (x, y) in [(0usize, 0usize), (63, 63)] {
            // SAFETY: inside the mapped buffer, by the footprints.
            let (half, byte) = unsafe {
                (
                    (raw_data.add(g[0].Offset as usize + y * g[0].Footprint.RowPitch as usize + x * 2) as *const u16).read(),
                    raw_data.add(g[1].Offset as usize + y * g[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(half, 32768, "16-bit depth at {x},{y}");
            assert_eq!(byte, 0x11, "stencil at {x},{y}");
        }
    }


    /// Unordered-access-only 1D and 3D title textures (the colour grading
    /// bakes: a 4096 R32_FLOAT curve, a 64^3 RGBA16F table) keep what a
    /// dispatch writes through their unordered views, whole (a 3D view of
    /// every depth slice, -1), and a shader view reads it.
    #[test]
    fn unordered_1d_and_3d_textures_keep_their_writes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Description words (dimension, width, height, depth | mips << 16,
        // format), the unordered view's and shader view's words, the
        // shaders' bodies, groups, a texel to check and its value.
        type Case = ([u32; 5], [u32; 10], [u32; 10], &'static str, &'static str, (u32, u32), [f32; 4]);
        let cases: [Case; 2] = [
            (
                [2, 4096, 1, 1 | 1 << 16, 41],
                [41, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [41, 2, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture1D<float> o : register(u0); void put(uint3 p) { if (p.y == 0 && p.z == 0) o[p.x] = p.x * 0.25; }",
                "Texture1D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int2(p.y * 64 + p.x, 0)); }",
                (512, 1),
                [(2 * 64 + 7) as f32 * 0.25, 0.0, 0.0, 1.0],
            ),
            (
                [4, 64, 64, 64 | 1 << 16, 10],
                [10, 8, 0, 0, u32::MAX, 0, 0, 0, 0, 0],
                [10, 8, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture3D<float4> o : register(u0); void put(uint3 p) { o[p] = float4(p, 1) / 64; }",
                "Texture3D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int4(p.x, p.y, 40, 0)); }",
                (8, 8),
                [7.0 / 64.0, 2.0 / 64.0, 40.0 / 64.0, 1.0 / 64.0],
            ),
        ];
        for (round, (d, uav, srv, put, get, groups, want)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 22);
            let words = [d[0], 0, 0, 0, d[1], 0, d[2], d[3], d[4], 1, 0, 0, 4, 0];
            let texture = texture_at_address(address, &words);
            let unordered = View::Unordered(texture as usize, 0, Some(uav));
            let shader = View::Shader(texture as usize, Some(srv));
            slots.make(0, &unordered);
            slots.make(1, &shader);
            let write = compile_cs(&format!(
                "#define RS \"DescriptorTable(UAV(u0))\"\n{put}\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ for (uint z = 0; z < 64; z++) put(uint3(id.xy, z)); }}"
            ));
            let read = compile_cs(&format!(
                "#define RS \"DescriptorTable(SRV(t0)), UAV(u1)\"\n{get}\nRWByteAddressBuffer o : register(u1);\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ o.Store4((id.y * 64 + id.x) * 16, asuint(get(id))); }}"
            ));
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Uav), &unordered, false);
            run(&list, &slots, &write, 0, None, groups);
            readable(&list, &used(ResourceType::Srv), &shader, false);
            run(&list, &slots, &read, 1, Some(&out), (8, 8));
            list.close();
            host.execute(&[&list]);
            // SAFETY: inside the mapped output: texel (7, 2) of the read.
            let got = unsafe { (data.add((2 * 64 + 7) * 16) as *const [f32; 4]).read() };
            assert_eq!(got, want, "round {round}");
        }
    }

    /// A structured view the title places off its buffer's elements (the
    /// culling constants at a ring position) is made over a copy of its own
    /// bytes; so is one on an element but not on 16 bytes, since the host
    /// makes it raw (from marquezinii's review of #1762: element 1 of the
    /// 456-byte constants); one on an element and on 16 bytes views the
    /// buffer's copy.
    #[test]
    fn structured_views_off_their_buffers_elements_get_their_own_copy() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let src = buffer(1 << 16);
        let (bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let start = bytes as u64;
        // SAFETY: a plain CPU-side heap of one view.
        let cpu: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .unwrap();
        // SAFETY: a live heap.
        let at = unsafe { cpu.GetCPUDescriptorHandleForHeapStart() };
        // Four 456-byte elements placed at `address` (resource and view
        // descriptions as the title passes them to device 0x230).
        let view = |address: u64| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4 * 456, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 4, 456, 0, 0]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let width = |address: u64| {
            let made = crate::descriptor::host::make_view(&view(address), at).expect("a view");
            // SAFETY: a live buffer.
            unsafe { made.GetDesc() }.Width
        };
        assert_eq!(width(start + 4 * 456), 1 << 16, "on an element and 16 bytes: the buffer's copy");
        assert_ne!(width(start + 456), 1 << 16, "on an element only: a copy of its own bytes (in a shared buffer)");
        assert_ne!(width(start + 100), 1 << 16, "off: a copy of its own bytes (in a shared buffer)");
        // Draws refresh the copy the view shows: its own for element 1.
        let (copy, _, _) = crate::descriptor::host::view_copy(&view(start + 456)).unwrap();
        assert_eq!(copy.range(), (start + 456, 4 * 456), "element 1's copy is its own");
        drop(copy);
        let (copy, address, len) = crate::descriptor::host::view_copy(&view(start + 100)).unwrap();
        assert_eq!((copy.range(), address, len), ((start + 100, 4 * 456), start + 100, 4 * 456));
        drop(copy);
        // A slot made from the view holds its copy: after more than a
        // thousand other views' copies, draws still find (and refresh) the
        // copy the slot shows; rewritten, the slot lets it go.
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let slots = crate::descriptor::heap(heap).host().expect("a host view heap");
        slots.make_at(0, &view(start + 100));
        let shown = std::sync::Arc::downgrade(&crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0);
        for k in 1..1200 {
            if (100 + 4 * k) % 456 != 0 {
                crate::descriptor::host::view_copy(&view(start + 100 + 4 * k)).unwrap();
            }
        }
        let found = crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0;
        assert!(shown.upgrade().is_some_and(|s| std::sync::Arc::ptr_eq(&s, &found)), "the slot's copy");
        drop(found);
        slots.make_null(0);
        assert!(shown.upgrade().is_none(), "released with the slot");
    }
}
