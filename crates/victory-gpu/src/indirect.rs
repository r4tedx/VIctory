//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! With `VICTORY_BATCH_CLAMPS=1` ([`batching`]) the clamps of a window of
//! indirect executions with nothing between them that may write GPU memory
//! run together, before the first: no dispatch or barrier of ours between
//! the executions, the same records clamped and charged in the same order.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::host::{HostList, host, transition};

// The title's shadow buffers stay UAVs, including when arguments and count
// are in the same buffer. Only the packed destination is written here.
const DISPATCH: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
cbuffer Dispatch : register(b0) {
    uint stride; uint records; uint waves; uint has_count; uint first; uint phase;
};
RWByteAddressBuffer source : register(u0);
RWByteAddressBuffer count : register(u1);
RWByteAddressBuffer args : register(u2);
RWByteAddressBuffer budget : register(u3);
static const uint BUDGET = 1u << 21; // in groups of 64 threads
static const uint GROUPS = 1u << 20;

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // The legacy clamp charges the first word in threads, including draws
    // whose charges need not be multiples of 64. Keep it intact and use a
    // second word in groups so even thousands of pending charges fit.
    if (phase != 1) {
        if (id.x != 0) return;
        uint threads = budget.Load(0);
        uint initial = (min(threads, BUDGET * 64) + 63) / 64;
        if (phase == 0) {
            budget.Store(4, initial);
        } else if (threads < BUDGET * 64) {
            budget.Store(0, threads + (budget.Load(4) - initial) * 64);
        }
        return;
    }
    uint record = first + id.x;
    if (record >= records) return;
    uint active = has_count == 0 ? records : min(records, count.Load(0));
    if (record == 0) args.Store(records * 12, active);
    uint3 g = 0;
    if (record < active) {
        g = min(source.Load3(record * stride), uint3(65535, 65535, 65535));
        g.x = g.x <= 65535u / waves ? g.x * waves : 0;
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = 0;
        }
        uint cost = g.x * g.y * g.z;
        if (cost != 0) {
            uint used;
            budget.InterlockedAdd(4, cost, used);
            // Pending refunds can cause conservative refusals. The counter
            // includes all accepted work and cannot wrap within a batch.
            if (used + cost > BUDGET) {
                budget.InterlockedAdd(4, 0u - cost);
                g = 0;
            }
        }
    }
    args.Store3(record * 12, g);
}
"#;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u1), UAV(u0)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
// The title's records, read only; the clamped ones go to the list's own.
ByteAddressBuffer args : register(t0);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` into `dest`; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        if (kind == 1) {
            dest.Store3(at + 8, args.Load3(at + 8));
        } else {
            dest.Store2(at + 8, args.Load2(at + 8));
        }
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            dest.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make(SOURCE) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make(source: &str) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let code = compile(source, s!("main"))?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = bytes(&code);
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    args: (&ID3D12Resource, u64),
    count: Option<&ID3D12Resource>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let (args_res, gpu) = args;
    let (Some(c), Some((budget_res, budget))) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    let l = &list.list;
    if records == 0 {
        // Zero records run no clamp dispatch and need no UAV barriers; keep
        // the `COMMON -> INDIRECT_ARGUMENT` state paired with `done()`.
        // SAFETY: an open list and live buffer in COMMON.
        unsafe {
            l.ResourceBarrier(&[transition(
                &out,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
        return Some(out);
    }
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // a title buffer's copy the caller's list holds.
    unsafe {
        // Order earlier title UAV writes into `args` (and any distinct
        // `count` buffer) without issuing a global NULL-resource UAV flush.
        let out_to_uav = transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        match count.filter(|c_res| c_res.as_raw() != args_res.as_raw()) {
            Some(count_res) => l.ResourceBarrier(&[
                uav_barrier_for(args_res),
                uav_barrier_for(count_res),
                out_to_uav,
            ]),
            None => l.ResourceBarrier(&[uav_barrier_for(args_res), out_to_uav]),
        }
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // One group walks every record.
        l.Dispatch(1, 1, 1);
        // `out` transitions UAV -> INDIRECT_ARGUMENT; `budget` only needs a
        // targeted UAV barrier for the next clamp in this list or submission.
        l.ResourceBarrier(&[
            uav_barrier_for(budget_res),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

/// Bounds and scales dispatches into a list-owned buffer of packed records,
/// followed by one word containing the effective count.
/// The source and count remain usable by later GPU writers.
pub(crate) fn dispatch(
    list: &HostList,
    source: u64,
    count: Option<u64>,
    (stride, records, waves): (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let groups = records.div_ceil(64);
    if groups == 0
        || groups > 65535
        || waves < 2
        || !source.is_multiple_of(4)
        || count.is_some_and(|at| !at.is_multiple_of(4))
        || stride < 12
        || !stride.is_multiple_of(4)
        || u64::from(records - 1) * u64::from(stride) + 11 > u64::from(u32::MAX)
    {
        return None;
    }
    static REWRITE: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    let rewrite = REWRITE
        .get_or_init(|| {
            make(DISPATCH)
                .inspect_err(|e| tracing::warn!(error = %e, "no dispatch argument rewrite"))
                .ok()
        })
        .as_ref()?;
    let (_, budget) = list.budget()?;
    let output = list.dispatch_args(u64::from(records) * 12 + 4)?;
    let mut constants = [stride, records, waves, u32::from(count.is_some()), 0, 0];
    // At most 2^21 groups carried in and 2^20 per record, even if every
    // refusal is still awaiting its refund. Each batch stays below 2^32.
    const BATCH_GROUPS: u32 = 63;
    // SAFETY: the caller kept the source and count alive and checked their
    // extents. The scratch buffer has one full record per thread.
    unsafe {
        let l = &list.list;
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&rewrite.root);
        l.SetPipelineState(&rewrite.pso);
        l.SetComputeRootUnorderedAccessView(1, source);
        l.SetComputeRootUnorderedAccessView(2, count.unwrap_or(source));
        l.SetComputeRootUnorderedAccessView(3, output.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, budget);
        let launch = |constants: &[u32; 6], groups| {
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.Dispatch(groups, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);
        };
        launch(&constants, 1);
        constants[5] = 1;
        for group in (0..groups).step_by(BATCH_GROUPS as usize) {
            constants[4] = group * 64;
            launch(&constants, (groups - group).min(BATCH_GROUPS));
        }
        constants[5] = 2;
        launch(&constants, 1);
        l.ResourceBarrier(&[transition(
            &output,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
    Some(output)
}

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// Whether indirect clamps are batched per window of indirect executions
/// (`VICTORY_BATCH_CLAMPS=1`; off by default, so every execution has its
/// own clamp, as [`bound`] records it).
///
/// A window opens at an indirect execution and records, there, one clamp
/// pass for it and every indirect execution after it until a command that
/// may write GPU memory: a dispatch, an unordered access clear, a draw or
/// indirect execution whose pipeline writes (it is the window's last), an
/// execution the batch cannot take, or the list's end. An execution the
/// window has no room for opens the next one. Nothing between the pass and
/// an execution of its window writes GPU memory, so the pass reads what the
/// execution's own clamp would read at its place, after every earlier
/// write (one barrier on all of them, as [`bound`] has). The pass copies
/// each record, clamped, to memory of the list's, where the executions read
/// it with no dispatch or barrier of ours between them: one indirect
/// dispatch clamps (a command per run of executions whose records lie in
/// one buffer, which it takes as a root view), then one group charges the
/// list's budget record by record in recorded order, as [`bound`] does call
/// by call.
///
/// shortcut: the clamped records and counts are copies: the title's
/// memory keeps what it wrote (the in-place clamp rewrites it).
pub(crate) fn batching() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_BATCH_CLAMPS"))
}

const BATCH_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u4), CBV(b1), SRV(t0)"
// One run of the window: consecutive executions whose records lie in one
// buffer (and counts, those that have one, in one buffer): its first
// execution and how many, its first record among the window's and how
// many.
cbuffer Run : register(b0) { uint first_entry; uint entries; uint first_record; uint records; };
// The run's buffers, from their start.
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer count_src : register(u1);
// The window's clamped records and counts.
RWByteAddressBuffer clamped_args : register(u2);
// Per record of the window: where it is in `clamped_args`, and its cost.
RWByteAddressBuffer charged : register(u3);
RWByteAddressBuffer budget : register(u4);
// The window's runs and records.
cbuffer Window : register(b1) { uint runs; uint total; };
// Per execution of the window, 32 bytes: kind, record stride, whether a
// distinct count bounds the clamp, its first record among the window's;
// where its records and count go in `clamped_args` (no count: ~0), and
// where they are in the run's buffers.
ByteAddressBuffer executions : register(t0);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
// Copies the record from `src` to `clamped_args`. Only active records
// are clamped and charged, using the in-place clamp's bounds.
uint clamp_and_copy(uint kind, uint from, uint to, bool active) {
    uint cost = 0;
    if (kind <= 1) {
        uint4 args = src.Load4(from);
        if (active) {
            args.x = min(args.x, 1u << 24);
            args.y = min(args.y, (1u << 26) / max(args.x, 1u));
            cost = args.x * args.y;
        }
        clamped_args.Store4(to, args);
        if (kind == 1) {
            clamped_args.Store(to + 16, src.Load(from + 16));
        }
    } else {
        uint3 g = src.Load3(from);
        if (active) {
            if (kind == 2) {
                g = min(g, uint3(65535, 65535, 65535));
                if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
                    g = uint3(0, 0, 0);
                }
                cost = g.x * g.y * g.z * 64;
            } else {
                g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
                cost = g.x * g.y * g.z * 128;
            }
        }
        clamped_args.Store3(to, g);
    }
    return cost;
}
// A lane per record of a run (a command of the window's indirect
// dispatch). Each execution's records go to places of their own, so the
// commands need no order among them.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void gather(uint id : SV_DispatchThreadID) {
    if (id >= records) {
        return;
    }
    uint at = first_record + id;
    // The run's last execution whose first record is at or before it.
    uint lo = first_entry;
    uint hi = first_entry + entries - 1;
    while (lo < hi) {
        uint mid = (lo + hi + 1) / 2;
        if (executions.Load(mid * 32 + 12) <= at) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    uint4 e = executions.Load4(lo * 32);
    uint4 where = executions.Load4(lo * 32 + 16);
    uint record = at - e.w;
    bool distinct_count = e.z != 0;
    bool active = true;
    if (where.y != 0xffffffffu && (distinct_count || record == 0)) {
        // Use the same GPU-loaded count that record zero copies for
        // ExecuteIndirect. The inline clamp only honors distinct counts.
        uint count = count_src.Load(where.w);
        if (record == 0) {
            clamped_args.Store(where.y, count);
        }
        active = !distinct_count || record < count;
    }
    uint offset = record * e.y;
    uint to = where.x + offset;
    uint cost = clamp_and_copy(e.x, where.z + offset, to, active);
    charged.Store2(at * 8, uint2(to, cost));
}
// Above any budget left: sums saturate here and still compare right.
static const uint CAP = BUDGET + 1;
// Records a lane charges in a tile of the group's 64 lanes.
static const uint EACH = 16;
groupshared uint sums[2][64];
// One group charges the window's records in order, 1024 at a time, 16
// consecutive ones a lane: a record runs if every record up to it fits in
// what is left; the first that does not, and every later one with a cost,
// run empty. That is the in-place clamp's walk, as prefix sums: each lane
// sums its records, the lanes' sums are scanned, and each lane walks its
// records again from the sum before it.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void charge(uint lane : SV_GroupIndex) {
    uint left = BUDGET - min(budget.Load(0), BUDGET);
    for (uint base = 0; base < total; base += 64 * EACH) {
        uint first = base + lane * EACH;
        uint own = 0;
        for (uint k = 0; k < EACH && first + k < total; k++) {
            own = min(own + min(charged.Load((first + k) * 8 + 4), CAP), CAP);
        }
        uint p = 0;
        sums[0][lane] = own;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint d = 1; d < 64; d <<= 1) {
            uint v = sums[p][lane];
            if (lane >= d) {
                v = min(v + sums[p][lane - d], CAP);
            }
            sums[1 - p][lane] = v;
            GroupMemoryBarrierWithGroupSync();
            p = 1 - p;
        }
        uint sum = lane == 0 ? 0 : sums[p][lane - 1];
        for (uint j = 0; j < EACH && first + j < total; j++) {
            uint2 r = charged.Load2((first + j) * 8);
            sum = min(sum + min(r.y, CAP), CAP);
            // Zero-cost records, including those past the count, stay as gathered.
            if (r.y != 0 && sum > left) {
                // Draws keep their layout with no vertices; groups become
                // zero.
                clamped_args.Store(r.x, 0);
            }
        }
        uint all = sums[p][63];
        left = all > left ? 0 : left - all;
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

/// The batch's pipelines (one root signature) and the command signature of
/// its clamp pass.
struct Batched {
    root: ID3D12RootSignature,
    gather: ID3D12PipelineState,
    charge: ID3D12PipelineState,
    signature: ID3D12CommandSignature,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Batched {}
// SAFETY: as above.
unsafe impl Sync for Batched {}

/// Bytes of one command of the clamp pass (a run): four constants, the
/// source and count root views, the dispatch.
const COMMAND: u64 = 48;
/// Bytes of one execution's description.
const EXECUTION: u64 = 32;
/// Where the commands start in a window's table, after its constants (a
/// root constant buffer, 256-byte aligned); the executions follow them.
const COMMANDS_AT: u64 = 256;
/// Runs a window has room for: the clamp pass's most commands. Small, as
/// an indirect call may cost by its most commands (WARP's does): a window
/// of executions of many buffers ends sooner.
const RUNS: u32 = 16;
/// Where the executions start in a window's table.
const EXECUTIONS_AT: u64 = COMMANDS_AT + RUNS as u64 * COMMAND;
/// Executions a window has room for at first and at most: the next window
/// gets more each time one fills.
const FIRST_ENTRIES: u32 = 64;
const MOST_ENTRIES: u32 = 8192;

fn batched() -> Option<&'static Batched> {
    static BATCHED: std::sync::OnceLock<Option<Batched>> = std::sync::OnceLock::new();
    BATCHED
        .get_or_init(|| match make_batched() {
            Ok(b) => Some(b),
            Err(e) => {
                tracing::warn!(error = %e, "no batched argument clamp: clamps run one by one");
                None
            }
        })
        .as_ref()
}

fn compile(source: &str, entry: windows::core::PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            entry,
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

/// A blob's bytes.
///
/// # Safety
/// The slice lives as long as the blob.
unsafe fn bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own bytes.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

fn make_batched() -> windows::core::Result<Batched> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let gather = compile(BATCH_SOURCE, s!("gather"))?;
    let charge = compile(BATCH_SOURCE, s!("charge"))?;
    // SAFETY: the blobs' own bytes, alive through the calls; the root
    // signature is inside the shaders (the same in both); the argument
    // descriptions live through the call.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes(&gather))?;
        let pso = |blob: &ID3DBlob| {
            device.CreateComputePipelineState::<ID3D12PipelineState>(
                &D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes(blob).as_ptr().cast(),
                        BytecodeLength: bytes(blob).len(),
                    },
                    ..Default::default()
                },
            )
        };
        let (gather, charge) = (pso(&gather)?, pso(&charge)?);
        let view = |index| D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                UnorderedAccessView: D3D12_INDIRECT_ARGUMENT_DESC_0_4 {
                    RootParameterIndex: index,
                },
            },
        };
        let arguments = [
            D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                        RootParameterIndex: 0,
                        DestOffsetIn32BitValues: 0,
                        Num32BitValuesToSet: 4,
                    },
                },
            },
            view(1),
            view(2),
            D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                ..Default::default()
            },
        ];
        let mut signature: Option<ID3D12CommandSignature> = None;
        device.CreateCommandSignature(
            &D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: COMMAND as u32,
                NumArgumentDescs: arguments.len() as u32,
                pArgumentDescs: arguments.as_ptr(),
                NodeMask: 0,
            },
            &root,
            &mut signature,
        )?;
        let signature = signature.ok_or_else(windows::core::Error::empty)?;
        Ok(Batched {
            root,
            gather,
            charge,
            signature,
        })
    }
}

/// GPU-only buffers a list hands out front to back, reused after a reset:
/// the chunks and their sizes (in the unordered access state outside a
/// window), the one in use and the bytes used in it.
#[derive(Default)]
struct Pool {
    chunks: Vec<(ID3D12Resource, u64)>,
    current: usize,
    used: u64,
}

impl Pool {
    /// The rest of a chunk with room for `need` bytes: the chunk, the offset
    /// and the bytes from there to its end. Chunks start at `first` bytes
    /// and double up to `most` (or what `need` takes).
    fn rest(&mut self, need: u64, (first, most): (u64, u64)) -> Option<(ID3D12Resource, u64, u64)> {
        while let Some(&(_, capacity)) = self.chunks.get(self.current) {
            if need <= capacity - self.used {
                break;
            }
            (self.current, self.used) = (self.current + 1, 0);
        }
        if self.current == self.chunks.len() {
            let next = self.chunks.last().map_or(first, |c| (2 * c.1).min(most));
            let capacity = need.checked_next_multiple_of(first)?.max(next);
            self.chunks.push((gpu_buffer(capacity)?, capacity));
            self.used = 0;
        }
        let (buffer, capacity) = &self.chunks[self.current];
        Some((buffer.clone(), self.used, capacity - self.used))
    }

    /// `bytes` from the offset [`Pool::rest`] gave are taken (root views
    /// after them start 256-byte aligned).
    fn take(&mut self, bytes: u64) {
        let capacity = self.chunks.get(self.current).map_or(0, |c| c.1);
        self.used = (self.used + bytes.next_multiple_of(256)).min(capacity);
    }
}

/// A GPU-only buffer shaders write, in the common state (buffers move out
/// of it implicitly).
fn gpu_buffer(size: u64) -> Option<ID3D12Resource> {
    let device = &host()?.device;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    let made = unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    match made {
        Ok(()) => {
            let buffer = buffer?;
            crate::host::name(&buffer, || format!("batched indirect clamps {size:#x}"));
            Some(buffer)
        }
        Err(e) => {
            tracing::warn!(error = %e, size, "no buffer for batched indirect clamps");
            None
        }
    }
}

/// Record chunks: 64 KiB, doubling up to 4 MiB.
const RECORDS: (u64, u64) = (64 << 10, 4 << 20);
/// Charge chunks (8 bytes a record): 16 KiB, doubling up to 1 MiB.
const CHARGES: (u64, u64) = (16 << 10, 1 << 20);

/// A list's batched clamps: the open window, the memory windows take, and
/// the room the next window's table gets.
#[derive(Default)]
pub(crate) struct Batch {
    window: Option<Window>,
    records: Pool,
    charges: Pool,
    room: u32,
}

/// The open window: its table (upload memory the list holds: the window's
/// constants, a command per run, then a description per execution, written
/// as executions come), where its records go and how much room it has.
struct Window {
    table: usize,
    room: u32,
    entries: u32,
    total: u32,
    /// Commands so far, and the last one's source and count buffers (host
    /// GPU addresses; none when no execution of it has a count).
    runs: u32,
    run: (u64, Option<u64>),
    /// Records of the last run.
    run_records: u32,
    /// The chunk the window's records go to and their offset there, the
    /// bytes written and the bytes it may write.
    records: (ID3D12Resource, u64),
    written: u64,
    space: u64,
    /// Records the window's charges have room for.
    charge_room: u64,
}

// SAFETY: the table is a persistent mapping of the list's upload memory;
// D3D12 objects are free-threaded.
unsafe impl Send for Batch {}

/// Where an execution of a window reads: the buffer, the offset of its
/// first record and of its count (when it has one), and whether its window
/// opened with it (the clamp pass replaced the list's compute root
/// signature and pipeline: callers bind theirs again).
pub(crate) struct Member {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) offset: u64,
    pub(crate) count: u64,
    pub(crate) opened: bool,
}

/// Adds, to `list`'s open window (opened here if there is none or it has no
/// room), the clamp of `records` records of `kind` `stride` bytes apart at
/// `source` (a buffer's host GPU address and the offset in it), with their
/// count at `count` (likewise) if the execution has one. None if the batch
/// cannot take them: no window is open then, so the caller clamps them with
/// [`bound`], in order.
pub(crate) fn member(
    list: &HostList,
    source: (u64, u64),
    count: Option<(u64, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<Member> {
    let words = match kind {
        0 => 4,
        1 => 5,
        2 | 10 => 3,
        _ => 0,
    };
    // Its records, then its count.
    let bytes = u64::from(stride) * u64::from(records) + if count.is_some() { 4 } else { 0 };
    // Match bound's resource comparison before adjusting either base address.
    let distinct_count = count.is_some_and(|(base, _)| base != source.0);
    // Offsets the pass's 32 bits reach; past them the address starts a
    // buffer of its own.
    let near = |(base, at): (u64, u64), len: u64| {
        if at + len <= u64::from(u32::MAX) {
            (base, at)
        } else {
            (base + at, 0)
        }
    };
    let source = near(source, u64::from(stride) * u64::from(records));
    let count = count.map(|c| near(c, 4));
    let fits = words != 0
        && records != 0
        && stride >= words * 4
        && stride.is_multiple_of(4)
        && (source.0 + source.1).is_multiple_of(4)
        && count.is_none_or(|(base, at)| (base + at).is_multiple_of(4))
        // A run's groups, and offsets in the window's memory.
        && records.div_ceil(64) <= 65535
        && bytes <= RECORDS.1;
    let (Some(b), true) = (batched(), fits) else {
        end(list);
        return None;
    };
    let Some((_, budget)) = list.budget() else {
        end(list);
        return None;
    };
    let mut guard = list.clamps.lock().unwrap_or_else(|e| e.into_inner());
    let batch = &mut *guard;
    // The last run goes on if its buffers are this execution's: records in
    // the same buffer, and a count (if any) in the same buffer as its
    // others'.
    let joins = |w: &Window| {
        w.runs > 0
            && w.run.0 == source.0
            && count.is_none_or(|(base, _)| w.run.1.is_none_or(|c| c == base))
            && w.run_records + records <= 65535 * 64
    };
    let room = batch.window.as_ref().map(|w| {
        let filled = w.entries == w.room;
        let full = filled
            || (w.runs == RUNS && !joins(w))
            || w.written + bytes > w.space
            || u64::from(w.total) + u64::from(records) > w.charge_room;
        (full, filled.then_some(w.room))
    });
    if let Some((true, filled)) = room {
        if let Some(room) = filled {
            batch.room = (room * 4).min(MOST_ENTRIES);
        }
        close(list, batch);
    }
    let opened = batch.window.is_none();
    if opened {
        let Some(w) = open(list, batch, b, budget, bytes, records) else {
            drop(guard);
            return None;
        };
        batch.window = Some(w);
        list.clamps_open
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let w = batch.window.as_mut()?;
    let at = w.written;
    let count_at = count.map(|_| at + u64::from(stride) * u64::from(records));
    let execution = [
        kind,
        stride,
        u32::from(distinct_count),
        w.total,
        at as u32,
        count_at.map_or(u32::MAX, |c| c as u32),
        source.1 as u32,
        count.map_or(0, |c| c.1 as u32),
    ];
    let joins = joins(w);
    let table = w.table;
    let command = |i: u32| (table + (COMMANDS_AT + u64::from(i) * COMMAND) as usize) as *mut u32;
    // SAFETY: the commands and descriptions lie in the window's table (its
    // rooms checked above), upload memory the list holds; the GPU reads them
    // once the list is submitted.
    unsafe {
        let executions = (table + EXECUTIONS_AT as usize) as *mut u32;
        let to = executions.add(w.entries as usize * EXECUTION as usize / 4);
        std::ptr::copy_nonoverlapping(execution.as_ptr(), to, 8);
        if !joins {
            let c = command(w.runs);
            c.cast::<[u32; 4]>().write([w.entries, 0, w.total, 0]);
            c.add(4).cast::<u64>().write_unaligned(source.0);
            c.add(6).cast::<u64>().write_unaligned(source.0);
            w.runs += 1;
            w.run = (source.0, None);
            w.run_records = 0;
        }
        let c = command(w.runs - 1);
        if let Some((base, _)) = count
            && w.run.1.is_none()
        {
            c.add(6).cast::<u64>().write_unaligned(base);
            w.run.1 = Some(base);
        }
        w.run_records += records;
        *c.add(1) += 1;
        *c.add(3) = w.run_records;
        c.add(8)
            .cast::<[u32; 3]>()
            .write_unaligned([w.run_records.div_ceil(64), 1, 1]);
    }
    w.entries += 1;
    w.total += records;
    w.written += bytes;
    Some(Member {
        buffer: w.records.0.clone(),
        offset: w.records.1 + at,
        count: w.records.1 + count_at.unwrap_or(0),
        opened,
    })
}

/// Opens a window whose first execution takes `bytes` and `records`: its
/// table and memory, and its clamp pass recorded here. None if there is no
/// memory for it (nothing is recorded then).
fn open(
    list: &HostList,
    batch: &mut Batch,
    b: &Batched,
    budget: u64,
    bytes: u64,
    records: u32,
) -> Option<Window> {
    let room = batch.room.max(FIRST_ENTRIES);
    let (table, at, data) = list.scratch_in(EXECUTIONS_AT + u64::from(room) * EXECUTION)?;
    let (records_buffer, records_at, space) = batch.records.rest(bytes, RECORDS)?;
    let (charges, charges_at, charge_bytes) =
        batch.charges.rest(u64::from(records) * 8, CHARGES)?;
    // SAFETY: the list's upload memory; no executions or records yet (the
    // window's end writes how many).
    unsafe { data.cast::<[u32; 2]>().write([0, 0]) };
    let l = &list.list;
    // SAFETY: an open list; our root signature, pipelines and command
    // signature; the table is upload memory the list holds, the chunks are
    // the list's (unordered access outside windows), the budget its word.
    unsafe {
        let table_gpu = table.GetGPUVirtualAddress() + at;
        #[cfg(feature = "diagnostics")]
        list.note_op(|| "batched indirect clamps".to_string());
        // Earlier shader writes (the title's argument passes, earlier
        // clamps' budget) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&b.root);
        l.SetPipelineState(&b.gather);
        l.SetComputeRootUnorderedAccessView(3, records_buffer.GetGPUVirtualAddress() + records_at);
        l.SetComputeRootUnorderedAccessView(4, charges.GetGPUVirtualAddress() + charges_at);
        l.SetComputeRootUnorderedAccessView(5, budget);
        l.SetComputeRootConstantBufferView(6, table_gpu);
        l.SetComputeRootShaderResourceView(7, table_gpu + EXECUTIONS_AT);
        // A command per run of the window: the table's first word, written
        // when the window ends.
        l.ExecuteIndirect(&b.signature, RUNS, &table, at + COMMANDS_AT, &table, at);
        l.ResourceBarrier(&[uav_barrier()]);
        // It reads only the root arguments set above, which the commands
        // leave as they were.
        l.SetPipelineState(&b.charge);
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                &records_buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    Some(Window {
        table: data as usize,
        room,
        entries: 0,
        total: 0,
        runs: 0,
        run: (0, None),
        run_records: 0,
        records: (records_buffer, records_at),
        written: 0,
        space,
        charge_room: charge_bytes / 8,
    })
}

/// Ends `list`'s open window, if any: before a command that may write GPU
/// memory, and when the list closes. Cheap when none is open.
pub(crate) fn end(list: &HostList) {
    if !list.clamps_open.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let mut batch = list.clamps.lock().unwrap_or_else(|e| e.into_inner());
    close(list, &mut batch);
}

/// The window's counts go into its table, and its records back to
/// unordered access, where the next window writes.
fn close(list: &HostList, batch: &mut Batch) {
    list.clamps_open
        .store(false, std::sync::atomic::Ordering::Relaxed);
    let Some(w) = batch.window.take() else { return };
    // SAFETY: the window's table, upload memory the list holds; an open list
    // and the list's chunk.
    unsafe {
        (w.table as *mut [u32; 2]).write([w.runs, w.total]);
        list.list.ResourceBarrier(&[transition(
            &w.records.0,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
    }
    batch.records.take(w.written);
    batch.charges.take(u64::from(w.total) * 8);
}

/// After the list's last submission ran: its memory is free again (the
/// chunks are kept for the next recording).
pub(crate) fn reset(list: &HostList) {
    list.clamps_open
        .store(false, std::sync::atomic::Ordering::Relaxed);
    let mut guard = list.clamps.lock().unwrap_or_else(|e| e.into_inner());
    let batch = &mut *guard;
    batch.window = None;
    for pool in [&mut batch.records, &mut batch.charges] {
        (pool.current, pool.used) = (0, 0);
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

/// A UAV barrier scoped to `resource` (non-null `pResource`).
fn uav_barrier_for(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    #[cfg(test)]
    tests::UAV_BARRIERS.with(|l| l.borrow_mut().push(resource.as_raw() as usize));
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed COM pointer; `ManuallyDrop` never releases it,
                // and the caller keeps `resource` alive through `ResourceBarrier`.
                pResource: std::mem::ManuallyDrop::new(Some(unsafe { std::mem::transmute_copy(resource) })),
            }),
        },
    }
}

#[cfg(test)]
#[path = "indirect_tests.rs"]
mod regressions;

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    thread_local! {
        pub(super) static UAV_BARRIERS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    fn take_uav_barriers() -> Vec<usize> {
        UAV_BARRIERS.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, (&buffer, gpu), None, spec).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, (&title, gpu), None, (2, 12, 1)).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }


    /// The batched clamp's shaders compile with the system's compiler and
    /// the host takes them and their command signature.
    #[test]
    fn batched_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        if let Err(e) = super::make_batched() {
            panic!("{e}");
        }
    }

    /// Words in a record of `kind`.
    fn words(kind: u32) -> usize {
        match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        }
    }

    /// A compute pipeline of the test's own (a 5.1 shader with its root
    /// signature inside, entry `main`).
    fn pipeline(source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let device = &super::host().unwrap().device;
        let code = super::compile(source, windows::core::s!("main")).unwrap();
        // SAFETY: the blob's own bytes; plain creation.
        unsafe {
            let root: ID3D12RootSignature =
                device.CreateRootSignature(0, super::bytes(&code)).unwrap();
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: super::bytes(&code).as_ptr().cast(),
                        BytecodeLength: super::bytes(&code).len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// The host signature of plain indirect dispatches, 12 bytes apart.
    fn dispatches() -> ID3D12CommandSignature {
        let device = &super::host().unwrap().device;
        let arguments = [D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        }];
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: plain creation; the description lives through the call.
        unsafe {
            device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 12,
                        NumArgumentDescs: 1,
                        pArgumentDescs: arguments.as_ptr(),
                        NodeMask: 0,
                    },
                    None,
                    &mut signature,
                )
                .unwrap();
        }
        signature.unwrap()
    }

    /// Records, in `list`, copies of `(buffer, offset, bytes)` ranges of the
    /// batch's memory (unordered access, no window open) one after another
    /// into a read-back buffer: the buffer and where each copy starts.
    fn read_back(
        list: &super::HostList,
        ranges: &[(&ID3D12Resource, u64, u64)],
    ) -> (ID3D12Resource, Vec<u64>) {
        let device = &super::host().unwrap().device;
        let total: u64 = ranges.iter().map(|r| r.2).sum();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total.max(4),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut back: Option<ID3D12Resource> = None;
        let mut distinct: Vec<&ID3D12Resource> = Vec::new();
        for r in ranges {
            if !distinct.iter().any(|d| d.as_raw() == r.0.as_raw()) {
                distinct.push(r.0);
            }
        }
        use windows::core::Interface;
        let (uav, copy) = (
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let mut starts = Vec::new();
        // SAFETY: plain creation; an open list and live buffers, put back in
        // the state they had.
        unsafe {
            device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut back,
                )
                .unwrap();
            let back = back.as_ref().unwrap();
            let to: Vec<_> = distinct
                .iter()
                .map(|b| super::transition(b, uav, copy))
                .collect();
            list.list.ResourceBarrier(&to);
            let mut at = 0;
            for &(buffer, offset, bytes) in ranges {
                list.list.CopyBufferRegion(back, at, buffer, offset, bytes);
                starts.push(at);
                at += bytes;
            }
            let from: Vec<_> = distinct
                .iter()
                .map(|b| super::transition(b, copy, uav))
                .collect();
            list.list.ResourceBarrier(&from);
        }
        list.keep(back.as_ref().unwrap());
        (back.unwrap(), starts)
    }

    /// The words of a read-back buffer after the list ran.
    fn words_of(back: &ID3D12Resource, at: u64, n: usize) -> Vec<u32> {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer the finished list wrote; read inside it.
        unsafe {
            back.Map(0, None, Some(&mut data)).unwrap();
            let words =
                std::slice::from_raw_parts(data.cast::<u8>().add(at as usize).cast::<u32>(), n)
                    .to_vec();
            back.Unmap(0, None);
            words
        }
    }

    /// Executions of a test: kind, records, the words they start with.
    type Call = (u32, u32, Vec<u32>);

    /// Mixed executions: light records, and every tenth execution starts
    /// with one of about 2^24 vertex instances or threads, so the budget
    /// runs out about two thirds of the way; garbage group counts (clamped
    /// to none) and records of no cost along the way.
    fn mixed_calls(n: usize) -> Vec<Call> {
        let mut seed = 0x2545_f491_u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        (0..n)
            .map(|i| {
                let kind = [0, 1, 2, 10][next() as usize % 4];
                let records = match next() % 8 {
                    0 => 1 + next() % 150,
                    _ => 1 + next() % 6,
                };
                let mut data = Vec::new();
                for r in 0..records {
                    let heavy = r == 0 && i % 10 == 3;
                    let garbage = i % 13 == 5;
                    match kind {
                        0 | 1 if heavy => data.extend([1 << 12, 1 << 12]),
                        0 | 1 => data.extend([1 + next() % 16, 1 + next() % 16]),
                        2 if heavy => data.extend([1024, 256, 1]),
                        2 if garbage => data.extend([u32::MAX; 3]),
                        2 => data.extend([1 + next() % 4, 1 + next() % 4, next() % 3]),
                        _ if heavy || garbage => data.extend([u32::MAX, u32::MAX, 1]),
                        _ => data.extend([1 + next() % 64, next() % 2, 1]),
                    }
                    // A draw's other words, copied as they are.
                    for _ in 2..words(kind).max(2) {
                        if kind <= 1 {
                            data.push(next());
                        }
                    }
                }
                (kind, records, data)
            })
            .collect()
    }

    /// Writes the calls' records into one shared buffer, each at a 256-byte
    /// boundary: the buffer, its words and each call's offset.
    fn fill(calls: &[Call]) -> (ID3D12Resource, *mut u32, Vec<u64>) {
        let host = super::host().unwrap();
        let mut offsets = Vec::new();
        let mut at = 0u64;
        for (_, _, data) in calls {
            offsets.push(at);
            at += (data.len() as u64 * 4).next_multiple_of(256);
        }
        let (buffer, data) = host.shared_buffer(at.max(256)).unwrap();
        let data = data.cast::<u32>();
        for ((_, _, words), &offset) in calls.iter().zip(&offsets) {
            for (i, &w) in words.iter().enumerate() {
                // SAFETY: inside the buffer (sized above), before any list runs.
                unsafe { data.add(offset as usize / 4 + i).write_volatile(w) };
            }
        }
        (buffer, data, offsets)
    }

    /// Batched clamps leave every record as the clamps one by one do, the
    /// budget included (spent on the way, so active records run empty from
    /// there). The 150 batched executions mix counted and uncounted calls
    /// across several windows, with clamps one by one before and after them.
    /// The batched source records stay as written.
    #[test]
    fn batched_clamps_leave_what_clamps_one_by_one_leave() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let mut calls = mixed_calls(160);
        // The first batched call executes one record. Charging both of its
        // remaining records would exhaust the budget in the first window.
        calls[5] = (2, 3, vec![65_536, 1, 1, 1024, 1024, 1, 1024, 1024, 1]);
        let counts: Vec<_> = calls
            .iter()
            .enumerate()
            .map(|(i, &(_, records, _))| match i % 12 {
                1 => Some(records / 2),
                2 => Some(0),
                3 => Some(records + 1),
                5 if i == 5 => Some(1),
                _ => None,
            })
            .collect();
        let (count_buffer, count_data) = host.shared_buffer(counts.len() as u64 * 4).unwrap();
        for (i, &count) in counts.iter().enumerate() {
            // SAFETY: a mapped buffer sized for all counts, before submission.
            unsafe {
                count_data
                    .cast::<u32>()
                    .add(i)
                    .write_volatile(count.unwrap_or(0))
            };
        }
        // SAFETY: a live buffer.
        let count_gpu = unsafe { count_buffer.GetGPUVirtualAddress() };
        let count_offset = |i: usize| counts[i].map(|_| i as u64 * 4);
        let inline_count = |i| count_offset(i).map(|at| (&count_buffer, count_gpu + at));
        let inline = |i: usize| !(5..155).contains(&i);
        // One by one, in place.
        let (old, old_data, offsets) = fill(&calls);
        let list = host.new_list().unwrap();
        // SAFETY: a live buffer.
        let old_gpu = unsafe { old.GetGPUVirtualAddress() };
        for (i, ((kind, records, _), &offset)) in calls.iter().zip(&offsets).enumerate() {
            let spec = (*kind, words(*kind) as u32 * 4, *records);
            let count = inline_count(i);
            let clamped = super::bound(&list, &old, count, old_gpu + offset, spec).unwrap();
            super::done(&list, &old, count.map(|(buffer, _)| buffer), clamped);
        }
        let (old_budget, old_budget_at) = read_back(&list, &[(list.budget().unwrap().0, 0, 4)]);
        list.close();
        host.execute(&[&list]);
        // Batched, but the first five and the last five.
        let (new, new_data, _) = fill(&calls);
        let batched = host.new_list().unwrap();
        // SAFETY: a live buffer.
        let new_gpu = unsafe { new.GetGPUVirtualAddress() };
        let mut members = Vec::new();
        for (i, ((kind, records, _), &offset)) in calls.iter().zip(&offsets).enumerate() {
            let spec = (*kind, words(*kind) as u32 * 4, *records);
            if inline(i) {
                super::end(&batched);
                let count = inline_count(i);
                let clamped = super::bound(&batched, &new, count, new_gpu + offset, spec).unwrap();
                super::done(&batched, &new, count.map(|(buffer, _)| buffer), clamped);
                members.push(None);
            } else {
                // The first window fills (one run); then 30 executions name
                // their records from their own start, not the buffer's: a
                // run each, past a window's room for runs.
                let source = match i {
                    100..130 => (new_gpu + offset, 0),
                    _ => (new_gpu, offset),
                };
                let count = count_offset(i).map(|at| (count_gpu, at));
                members.push(Some(super::member(&batched, source, count, spec).unwrap()));
            }
        }
        super::end(&batched);
        assert!(
            batched.clamps.lock().unwrap().room > super::FIRST_ENTRIES,
            "more than one window"
        );
        let ranges: Vec<_> = members
            .iter()
            .zip(&calls)
            .filter_map(|(m, (kind, records, _))| {
                m.as_ref().map(|m| {
                    (
                        &m.buffer,
                        m.offset,
                        u64::from(*records) * words(*kind) as u64 * 4,
                    )
                })
            })
            .chain([(batched.budget().unwrap().0, 0, 4)])
            .collect();
        let (back, starts) = read_back(&batched, &ranges);
        batched.close();
        host.execute(&[&batched]);
        // Draws of the test all have vertices: only the budget empties them.
        let mut ran = 0;
        let mut empty = 0;
        let mut copied = starts.iter();
        for (i, ((kind, records, written), &offset)) in calls.iter().zip(&offsets).enumerate() {
            let at = offset as usize / 4;
            // SAFETY: inside the buffers, after the lists ran.
            let want: Vec<u32> = (0..written.len())
                .map(|w| unsafe { old_data.add(at + w).read_volatile() })
                .collect();
            let got = if inline(i) {
                // SAFETY: as above.
                (0..written.len())
                    .map(|w| unsafe { new_data.add(at + w).read_volatile() })
                    .collect()
            } else {
                // The title's memory keeps what it wrote.
                // SAFETY: as above.
                let kept: Vec<u32> = (0..written.len())
                    .map(|w| unsafe { new_data.add(at + w).read_volatile() })
                    .collect();
                assert_eq!(&kept, written, "execution {i}: its source changed");
                words_of(&back, *copied.next().unwrap(), written.len())
            };
            assert_eq!(
                got, want,
                "execution {i} (kind {kind}, count {:?})",
                counts[i]
            );
            if *kind <= 1 {
                let active = counts[i].unwrap_or(*records).min(*records) as usize;
                for record in want.chunks(words(*kind)).take(active) {
                    if record[0] == 0 {
                        empty += 1;
                    } else {
                        ran += 1;
                    }
                }
            }
        }
        // The budget ran out on the way.
        assert!(ran > 100 && empty > 100, "ran {ran}, empty {empty}");
        let want_budget = words_of(&old_budget, old_budget_at[0], 1);
        assert_eq!(want_budget, [1 << 27]);
        assert_eq!(words_of(&back, *copied.next().unwrap(), 1), want_budget);
    }

    /// A window ends before a write to later executions' records (as a
    /// title dispatch ends it): the next one clamps what was written.
    /// Without the end the window's pass, recorded before the write, would
    /// read what was there before (shown, so the end is what matters).
    #[test]
    fn clamps_read_what_a_dispatch_before_them_wrote() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (root, pso) = pipeline(
            "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n\
             [RootSignature(RS)][numthreads(1, 1, 1)] void main() { b.Store3(0, uint3(5, 6, 7)); }",
        );
        let run = |ends: bool| {
            let (buffer, data) = host.shared_buffer(512).unwrap();
            let data = data.cast::<u32>();
            // SAFETY: our mapped buffer, before the list runs.
            unsafe {
                for i in 0..128 {
                    data.add(i).write_volatile(1);
                }
            }
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let list = host.new_list().unwrap();
            let first = super::member(&list, (gpu, 0), None, (2, 12, 1)).unwrap();
            assert!(first.opened);
            if ends {
                super::end(&list);
            }
            // SAFETY: an open list, our pipeline and buffer.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootUnorderedAccessView(0, gpu + 256);
                list.list.Dispatch(1, 1, 1);
            }
            let second = super::member(&list, (gpu, 256), None, (2, 12, 1)).unwrap();
            assert_eq!(second.opened, ends);
            super::end(&list);
            let (back, starts) = read_back(
                &list,
                &[
                    (&first.buffer, first.offset, 12),
                    (&second.buffer, second.offset, 12),
                ],
            );
            list.close();
            host.execute(&[&list]);
            (words_of(&back, starts[0], 3), words_of(&back, starts[1], 3))
        };
        assert_eq!(run(true), (vec![1, 1, 1], vec![5, 6, 7]));
        assert_eq!(run(false), (vec![1, 1, 1], vec![1, 1, 1]));
    }

    /// Executions read the window's records and counts: draws, a mesh
    /// dispatch and dispatches in one window; the dispatches run with their
    /// count (3 of 4 records: 1 + 2 + 3 groups) and without (all 4: 10).
    /// The last dispatches' count lies in another buffer (a run of its own):
    /// 1 of 2 records, 5 groups. A draw count in the argument buffer is
    /// copied and leaves the clamp's record limit at the supplied maximum.
    #[test]
    fn executions_read_the_windows_records_and_counts() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (root, pso) = pipeline(
            "#define RS \"UAV(u0)\"\nRWByteAddressBuffer c : register(u0);\n\
             [RootSignature(RS)][numthreads(1, 1, 1)] void main() { uint o; c.InterlockedAdd(0, 1, o); }",
        );
        let signature = dispatches();
        let calls: Vec<Call> = vec![
            (0, 2, vec![u32::MAX, 3, 9, 8, 30, 2, 4, 5]),
            (10, 1, vec![70_000, 5, 6]),
            (1, 1, vec![12, u32::MAX, 1, 2, 3]),
            (2, 4, vec![1, 1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1]),
            (2, 2, vec![5, 1, 1, 6, 1, 1]),
        ];
        let (buffer, args_data, offsets) = fill(&calls);
        let draw_count_at = offsets[0] + calls[0].2.len() as u64 * 4;
        let (counts, count) = host.shared_buffer(256).unwrap();
        let (others, other) = host.shared_buffer(256).unwrap();
        let (counter, counted) = host.shared_buffer(256).unwrap();
        // SAFETY: our mapped buffers, before the list runs; live buffers.
        let (gpu, count_gpu, counter_gpu) = unsafe {
            // The draw count fits in the padding after its records.
            args_data
                .add(draw_count_at as usize)
                .cast::<u32>()
                .write_volatile(0);
            count.cast::<u32>().write_volatile(3);
            other.cast::<u32>().add(1).write_volatile(1);
            counted.cast::<u32>().write_volatile(0);
            (
                buffer.GetGPUVirtualAddress(),
                counts.GetGPUVirtualAddress(),
                counter.GetGPUVirtualAddress(),
            )
        };
        // SAFETY: a live buffer.
        let other_gpu = unsafe { others.GetGPUVirtualAddress() };
        let list = host.new_list().unwrap();
        let members: Vec<_> = calls
            .iter()
            .zip(&offsets)
            .enumerate()
            .map(|(i, ((kind, records, _), &offset))| {
                let count = match i {
                    0 => Some((gpu, draw_count_at)),
                    3 => Some((count_gpu, 0)),
                    4 => Some((other_gpu, 4)),
                    _ => None,
                };
                super::member(
                    &list,
                    (gpu, offset),
                    count,
                    (*kind, words(*kind) as u32 * 4, *records),
                )
                .unwrap()
            })
            .collect();
        assert!(members[0].opened && members[1..].iter().all(|m| !m.opened));
        let (draw, m, last) = (&members[0], &members[3], &members[4]);
        // SAFETY: an open list, our pipeline, signature and buffers; the
        // window's records are indirect arguments while it is open.
        unsafe {
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRootUnorderedAccessView(0, counter_gpu);
            list.list
                .ExecuteIndirect(&signature, 4, &m.buffer, m.offset, &m.buffer, m.count);
            list.list
                .ExecuteIndirect(&signature, 4, &m.buffer, m.offset, None, 0);
            list.list.ExecuteIndirect(
                &signature,
                2,
                &last.buffer,
                last.offset,
                &last.buffer,
                last.count,
            );
        }
        super::end(&list);
        let ranges: Vec<_> = members
            .iter()
            .zip(&calls)
            .map(|(m, (kind, records, _))| {
                (
                    &m.buffer,
                    m.offset,
                    u64::from(*records) * words(*kind) as u64 * 4,
                )
            })
            .chain([
                (&m.buffer, m.count, 4),
                (&last.buffer, last.count, 4),
                (&draw.buffer, draw.count, 4),
            ])
            .collect();
        let (back, starts) = read_back(&list, &ranges);
        list.close();
        host.execute(&[&list]);
        // SAFETY: our mapped buffer, after the list ran.
        assert_eq!(unsafe { counted.cast::<u32>().read_volatile() }, 6 + 10 + 5);
        let got: Vec<Vec<u32>> = starts
            .iter()
            .zip(&ranges)
            .map(|(&at, r)| words_of(&back, at, r.2 as usize / 4))
            .collect();
        assert_eq!(got[0], [1 << 24, 3, 9, 8, 30, 2, 4, 5]);
        assert_eq!(got[1], [65535, 1, 1]);
        assert_eq!(got[2], [12, (1 << 26) / 12, 1, 2, 3]);
        assert_eq!(got[3], [1, 1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1]);
        assert_eq!(got[4], [5, 1, 1, 6, 1, 1]);
        assert_eq!(
            (got[5].as_slice(), got[6].as_slice()),
            ([3].as_slice(), [1].as_slice())
        );
        assert_eq!(got[7], [0]);
    }

    /// A list runs its batched clamps again at each submission (the budget
    /// zeroed each time, the records the title's again), and after a reset
    /// it records into the memory it has: no more of it.
    #[test]
    fn batched_clamps_run_again_and_reuse_their_memory() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let calls: Vec<Call> = (0..100).map(|_| (0, 1, vec![1 << 24, 4, 0, 0])).collect();
        let (buffer, _, offsets) = fill(&calls);
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let list = host.new_list().unwrap();
        let mut chunks = 0;
        for round in 0..3 {
            let members: Vec<_> = offsets
                .iter()
                .map(|&o| super::member(&list, (gpu, o), None, (0, 16, 1)).unwrap())
                .collect();
            super::end(&list);
            let ranges: Vec<_> = members.iter().map(|m| (&m.buffer, m.offset, 16)).collect();
            let (back, starts) = read_back(&list, &ranges);
            list.close();
            for _ in 0..2 {
                host.execute(&[&list]);
                // 2^26 vertex instances a record: two fit the budget.
                let ran: Vec<bool> = starts
                    .iter()
                    .map(|&at| words_of(&back, at, 1)[0] != 0)
                    .collect();
                assert_eq!(ran.iter().filter(|&&r| r).count(), 2, "round {round}");
                assert!(ran[0] && ran[1]);
            }
            let batch = list.clamps.lock().unwrap();
            let now = batch.records.chunks.len() + batch.charges.chunks.len();
            drop(batch);
            if round > 0 {
                assert_eq!(now, chunks, "round {round}: more memory");
            }
            chunks = now;
            list.reset();
        }
        list.close();
    }

    /// Before and after: 1,500 indirect dispatches of no groups, each with
    /// its own clamp and with batched clamps, on this machine's host
    /// adapter (WARP for these tests). Prints the GPU time; run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_batched_clamps() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        const N: u32 = 1500;
        let (title_root, title_pso) = pipeline(
            "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n\
             [RootSignature(RS)][numthreads(1, 1, 1)] void main() {}",
        );
        let signature = dispatches();
        let (args, _) = host.shared_buffer(u64::from(N) * 12).unwrap();
        let (sink, _) = host.shared_buffer(256).unwrap();
        let device = &host.device;
        let mut heap: Option<ID3D12QueryHeap> = None;
        // SAFETY: plain creation.
        unsafe {
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 2,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .unwrap();
        }
        let heap = heap.unwrap();
        let stamps = {
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 16,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                device
                    .CreateCommittedResource(
                        &heap_props,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut out,
                    )
                    .unwrap()
            };
            out.unwrap()
        };
        let frequency = host.timestamp_frequency() as f64;
        // SAFETY: live buffers.
        let (a, s) = unsafe { (args.GetGPUVirtualAddress(), sink.GetGPUVirtualAddress()) };
        let list = host.new_list().unwrap();
        list.close();
        // 0: a clamp each; 1: batched; 2: no clamps; 3, 4: the clamps of 0
        // and 1 without the executions.
        let run = |variant: u32| -> (f64, f64) {
            let (batched, executes) = (variant == 1 || variant == 4, variant <= 2);
            let mut gpu = Vec::new();
            let mut cpu = Vec::new();
            for _ in 0..7 {
                list.reset();
                let started = std::time::Instant::now();
                let l = &list.list;
                // SAFETY: an open list, our pipelines, signature and buffers.
                unsafe {
                    l.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    let title = || {
                        l.SetComputeRootSignature(&title_root);
                        l.SetPipelineState(&title_pso);
                        l.SetComputeRootUnorderedAccessView(0, s);
                    };
                    title();
                    for i in 0..N {
                        let at = u64::from(i) * 12;
                        if variant == 2 {
                            l.ResourceBarrier(&[super::transition(
                                &args,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                            )]);
                            l.ExecuteIndirect(&signature, 1, &args, at, None, 0);
                            l.ResourceBarrier(&[super::transition(
                                &args,
                                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            )]);
                        } else if batched {
                            let m = super::member(&list, (a, at), None, (2, 12, 1)).unwrap();
                            if m.opened {
                                title();
                            }
                            if executes {
                                l.ExecuteIndirect(&signature, 1, &m.buffer, m.offset, None, 0);
                            }
                        } else {
                            let clamped = super::bound(&list, &args, None, a + at, (2, 12, 1)).unwrap();
                            title();
                            if executes {
                                l.ExecuteIndirect(&signature, 1, &args, at, None, 0);
                            }
                            super::done(&list, &args, None, clamped);
                        }
                    }
                    super::end(&list);
                    l.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    l.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &stamps, 0);
                }
                list.close();
                cpu.push(started.elapsed().as_secs_f64() * 1e3);
                host.execute(&[&list]);
                let mut data = std::ptr::null_mut();
                // SAFETY: the read-back buffer the finished list wrote.
                unsafe {
                    stamps.Map(0, None, Some(&mut data)).unwrap();
                    let t = std::slice::from_raw_parts(data.cast::<u64>(), 2);
                    gpu.push(t[1].saturating_sub(t[0]) as f64 * 1e3 / frequency);
                    stamps.Unmap(0, None);
                }
            }
            gpu.sort_by(f64::total_cmp);
            cpu.sort_by(f64::total_cmp);
            (gpu[3], cpu[3])
        };
        let [each, batched, none, each_clamps, batched_clamps] = [0, 1, 2, 3, 4].map(run);
        eprintln!(
            "BENCH {N} indirect dispatches of no groups, median of 7, GPU ms (recording ms): a clamp each \
             {:.2} ({:.2}); batched {:.2} ({:.2}); no clamps {:.2} ({:.2}); the clamps alone: a clamp each \
             {:.2} ({:.2}), batched {:.2} ({:.2})",
            each.0,
            each.1,
            batched.0,
            batched.1,
            none.0,
            none.1,
            each_clamps.0,
            each_clamps.1,
            batched_clamps.0,
            batched_clamps.1
        );
    }

    /// Clamps emit targeted UAV barriers only on `args`, any distinct `count`
    /// buffer, and `budget` (never a global NULL-resource flush), and zero
    /// records emit no UAV barriers while keeping `out` transitions balanced.
    #[test]
    fn scoped_uav_barriers_and_zero_record_skip() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (args, args_data) = host.shared_buffer(256).unwrap();
        let (count, _) = host.shared_buffer(256).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            args_data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            args.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        let list = host.new_list().unwrap();
        let budget_key = key(list.budget().unwrap().0);

        // 1. No count buffer -> [args, budget].
        take_uav_barriers();
        issued();
        let out = super::bound(&list, (&args, gpu), None, (2, 12, 1)).unwrap();
        super::done(&list, &out);
        assert_eq!(take_uav_barriers(), [key(&args), budget_key]);
        let mut states: std::collections::HashMap<usize, u32> = [(key(&out), 0)].into_iter().collect();
        replay(&issued(), &mut states).unwrap();
        assert_eq!(states[&key(&out)], 0);

        // 2. Count buffer aliased with args -> deduplicated [args, budget].
        take_uav_barriers();
        let out = super::bound(&list, (&args, gpu), Some(&args), (2, 12, 1)).unwrap();
        super::done(&list, &out);
        assert_eq!(take_uav_barriers(), [key(&args), budget_key]);

        // 3. Distinct count buffer -> [args, count, budget].
        take_uav_barriers();
        let out = super::bound(&list, (&args, gpu), Some(&count), (2, 12, 1)).unwrap();
        super::done(&list, &out);
        assert_eq!(take_uav_barriers(), [key(&args), key(&count), budget_key]);

        // 4. Zero records -> no UAV barriers, balanced COMMON -> INDIRECT -> COMMON.
        take_uav_barriers();
        issued();
        let out = super::bound(&list, (&args, gpu), Some(&count), (2, 12, 0)).unwrap();
        super::done(&list, &out);
        assert!(take_uav_barriers().is_empty());
        let mut states: std::collections::HashMap<usize, u32> = [(key(&out), 0)].into_iter().collect();
        replay(&issued(), &mut states).unwrap();
        assert_eq!(states[&key(&out)], 0);

        // Execute the list once, then execute again after completion (never
        // re-submitting an in-flight list in the same ExecuteCommandLists call).
        list.close();
        host.execute(&[&list]);
        host.execute(&[&list]);
        let mut p = std::ptr::null_mut();
        // SAFETY: our mapped argument buffer, after execution completed.
        let clamped = unsafe {
            out.Map(0, None, Some(&mut p)).unwrap();
            p.cast::<[u32; 3]>().read_volatile()
        };
        assert_eq!(clamped, [65_535, 1, 1]);
    }
}
