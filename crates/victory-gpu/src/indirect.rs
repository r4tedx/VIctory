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
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint has_count; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
// Distinct indirect count buffer (bound to `args` when unused). Reading it
// promotes a fresh `COMMON` count buffer to `UNORDERED_ACCESS` before the
// post-clamp `UNORDERED_ACCESS -> INDIRECT_ARGUMENT` transition, and bounds
// clamping to the records `ExecuteIndirect` can actually run.
RWByteAddressBuffer count_buf : register(u2);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` in place; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
groupshared uint limit;
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
        limit = has_count != 0 ? min(records, count_buf.Load(0)) : records;
    }
    GroupMemoryBarrierWithGroupSync();
    uint active = limit;
    for (uint tile = 0; tile < active / 64 + min(active % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < active) {
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
            args.Store(at, 0);
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
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
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
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
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

/// Result of [`bound`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Clamped {
    /// Zero records: nothing was recorded on the list and `buffer` stayed in
    /// its previous state (no pipeline rebind or [`done`] transition needed).
    Empty,
    /// The clamp ran and `buffer` is now in `INDIRECT_ARGUMENT`: rebind the
    /// compute root signature and pipeline, and call [`done`] after execution.
    Ready,
}

impl Clamped {
    /// Whether [`bound`] dispatched the clamp and transitioned `buffer` to
    /// `INDIRECT_ARGUMENT`.
    pub(crate) fn transitioned(self) -> bool {
        self == Self::Ready
    }
}

/// Which host buffer a scoped indirect barrier targets (never a wildcard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Args,
    Count,
}

/// The barriers and transitions planned for one [`bound`] / [`done`] pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plan {
    /// Scoped UAV barriers emitted before the clamp dispatch (and the exact
    /// same buffers transitioned `UAV <-> INDIRECT_ARGUMENT` around execution).
    buffers: [Target; 2],
    buffers_len: usize,
    /// Whether the clamp dispatches, bundles a post-clamp `budget` UAV barrier
    /// (covering both subsequent clamps in the list and repeated list
    /// executions in `Host::execute`), and transitions `buffers`.
    transition: bool,
}

impl Plan {
    /// Plans barriers and transitions for `records` indirect records.
    /// `count_ptr` is the raw resource pointer of the count buffer, if any
    /// (deduplicated against `args_ptr`).
    fn for_call(records: u32, args_ptr: usize, count_ptr: Option<usize>) -> Self {
        if records == 0 {
            return Self {
                buffers: [Target::Args; 2],
                buffers_len: 0,
                transition: false,
            };
        }
        let mut buffers = [Target::Args; 2];
        let mut len = 1;
        if count_ptr.is_some_and(|c| c != args_ptr) {
            buffers[len] = Target::Count;
            len += 1;
        }
        Self {
            buffers,
            buffers_len: len,
            transition: true,
        }
    }

    fn pre_uav(&self) -> &[Target] {
        &self.buffers[..self.buffers_len]
    }

    fn transitions(&self) -> &[Target] {
        if self.transition {
            &self.buffers[..self.buffers_len]
        } else {
            &[]
        }
    }

    fn post_budget_uav(&self) -> bool {
        self.transition
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves `buffer` (and any distinct `count` buffer) ready in
/// `INDIRECT_ARGUMENT`. `count` is `(resource, gpu_address)` of the indirect
/// count buffer, if the call gives one. Replaces the list's compute root
/// signature and pipeline when returning [`Clamped::Ready`] (callers bind
/// theirs after). `None` if the clamp is not available or an address/stride is
/// not 4-byte aligned.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    count: Option<(&ID3D12Resource, u64)>,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> Option<Clamped> {
    if records == 0 {
        return Some(Clamped::Empty);
    }
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    if let Some((_, count_gpu)) = count
        && !count_gpu.is_multiple_of(4)
    {
        return None;
    }
    let (c, (budget_res, budget)) = (clamp()?, list.budget()?);
    let plan = Plan::for_call(
        records,
        buffer.as_raw() as usize,
        count.map(|(r, _)| r.as_raw() as usize),
    );
    if !plan.transition {
        return Some(Clamped::Empty);
    }
    let distinct_count = count.filter(|(r, _)| r.as_raw() != buffer.as_raw());
    let resource_for = |target: Target| match target {
        Target::Args => buffer,
        Target::Count => distinct_count.expect("count buffer planned").0,
    };
    let l = &list.list;
    let constants = [kind, stride, records, u32::from(distinct_count.is_some())];
    // SAFETY: an open list; our root signature, pipeline and live buffers
    // the caller and list hold.
    unsafe {
        // Earlier shader writes to the argument (and distinct count) buffers
        // first, scoped to their resources so unrelated UAV waves keep running.
        match plan.pre_uav() {
            [a] => l.ResourceBarrier(&[uav_barrier_for(resource_for(*a))]),
            [a, b] => l.ResourceBarrier(&[
                uav_barrier_for(resource_for(*a)),
                uav_barrier_for(resource_for(*b)),
            ]),
            _ => unreachable!(),
        }
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, distinct_count.map_or(gpu, |(_, g)| g));
        // One group walks every record.
        l.Dispatch(1, 1, 1);
        // Transitioning UAV -> INDIRECT_ARGUMENT synchronizes the clamp's UAV
        // writes to `buffer` and puts any distinct `count` buffer (promoted to
        // UAV by `u2` above if it started in `COMMON`) in `INDIRECT_ARGUMENT`.
        // Bundling `budget`'s UAV barrier in the same batch also serializes
        // repeated executions of this list in one `ExecuteCommandLists` call.
        let to_arg = |t: Target| {
            transition(
                resource_for(t),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )
        };
        debug_assert!(plan.post_budget_uav());
        match plan.transitions() {
            [a] => l.ResourceBarrier(&[to_arg(*a), uav_barrier_for(&budget_res)]),
            [a, b] => l.ResourceBarrier(&[to_arg(*a), to_arg(*b), uav_barrier_for(&budget_res)]),
            _ => unreachable!(),
        }
    }
    Some(Clamped::Ready)
}

/// After the execution: `buffer` (and any distinct `count` buffer) back to
/// unordered access, where the title's passes keep using them. No-op when
/// `clamped` is [`Clamped::Empty`] (nothing was transitioned).
pub(crate) fn done(
    list: &HostList,
    buffer: &ID3D12Resource,
    count: Option<&ID3D12Resource>,
    clamped: Clamped,
) {
    if !clamped.transitioned() {
        return;
    }
    let to_uav = |r: &ID3D12Resource| {
        transition(
            r,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )
    };
    let distinct_count = count.filter(|r| r.as_raw() != buffer.as_raw());
    // SAFETY: an open list and live buffers in INDIRECT_ARGUMENT.
    unsafe {
        match distinct_count {
            None => list.list.ResourceBarrier(&[to_uav(buffer)]),
            Some(count_res) => list.list.ResourceBarrier(&[to_uav(buffer), to_uav(count_res)]),
        }
    }
}

/// A UAV barrier scoped to `resource` (non-null `pResource`).
fn uav_barrier_for(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{Clamped, Plan, Target};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::Interface;

    /// Zero records skips pre-clamp barriers, clamp dispatch, transitions,
    /// budget barrier, and restore even when a count buffer is present.
    #[test]
    fn zero_records_skips_clamp_and_barriers() {
        let empty = Plan::for_call(0, 0x1000, None);
        assert!(empty.pre_uav().is_empty());
        assert!(empty.transitions().is_empty());
        assert!(!empty.post_budget_uav());

        let with_count = Plan::for_call(0, 0x1000, Some(0x2000));
        assert!(with_count.pre_uav().is_empty());
        assert!(with_count.transitions().is_empty());
        assert!(!with_count.post_budget_uav());
        assert!(!Clamped::Empty.transitioned());
        assert!(Clamped::Ready.transitioned());
    }

    /// Non-zero records plan only scoped pre-clamp UAV barriers and matching
    /// `UAV <-> INDIRECT_ARGUMENT` transitions (deduplicating aliased count
    /// buffers), plus a post-clamp budget UAV barrier bundled into the same
    /// batch so repeated list executions stay synchronized.
    #[test]
    fn barrier_plan_scopes_pre_uav_targets() {
        let plain = Plan::for_call(4, 0x1000, None);
        assert_eq!(plain.pre_uav(), &[Target::Args]);
        assert_eq!(plain.transitions(), &[Target::Args]);
        assert!(plain.post_budget_uav());

        let distinct_count = Plan::for_call(8, 0x1000, Some(0x2000));
        assert_eq!(distinct_count.pre_uav(), &[Target::Args, Target::Count]);
        assert_eq!(distinct_count.transitions(), &[Target::Args, Target::Count]);
        assert!(distinct_count.post_budget_uav());

        let same_count = Plan::for_call(8, 0x1000, Some(0x1000));
        assert_eq!(same_count.pre_uav(), &[Target::Args]);
        assert_eq!(same_count.transitions(), &[Target::Args]);
        assert!(same_count.post_budget_uav());
    }

    /// `uav_barrier_for` constructs a scoped UAV barrier with a non-null
    /// `pResource` pointing at the exact resource (never a global wildcard).
    #[test]
    fn scoped_uav_barrier_sets_non_null_resource() {
        let mut dummy = 0u64;
        let raw = (&raw mut dummy).cast::<core::ffi::c_void>();
        // SAFETY: wrapped in ManuallyDrop so `Release` is never called, and
        // never dereferenced — only the pointer value is copied and compared.
        let fake = std::mem::ManuallyDrop::new(unsafe { ID3D12Resource::from_raw(raw) });
        let barrier = super::uav_barrier_for(&fake);
        assert_eq!(barrier.Type, D3D12_RESOURCE_BARRIER_TYPE_UAV);
        assert_eq!(barrier.Flags, D3D12_RESOURCE_BARRIER_FLAG_NONE);
        // SAFETY: `barrier.Type` is `D3D12_RESOURCE_BARRIER_TYPE_UAV`.
        let scoped = unsafe { &barrier.Anonymous.UAV.pResource }
            .as_ref()
            .expect("scoped UAV barrier must set pResource");
        assert_eq!(scoped.as_raw(), raw);
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
    /// `calls` times in one list: each record's first word after (zero when
    /// it runs empty). One record more follows them and must stay as
    /// written. `None` without a host GPU.
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
            for _ in 0..calls {
                let state = super::bound(&list, &buffer, None, gpu, spec).unwrap();
                assert_eq!(state, if records == 0 { Clamped::Empty } else { Clamped::Ready });
                super::done(&list, &buffer, None, state);
            }
            list.close();
            host.execute(&[&list]);
            host.drain();
            for at in all - per..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()]);
            }
            let first = |i: usize| data.add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Zero-record calls leave buffer bytes untouched and do not emit a
    /// mismatched restore transition.
    #[test]
    fn zero_records_leaves_buffer_untouched() {
        let Some(empty) = clamped(0, &[u32::MAX], 0, 2) else {
            return; // no host GPU here
        };
        assert!(empty.is_empty());
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

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    /// A distinct `count` buffer — whether starting fresh in `COMMON` or left
    /// in `UNORDERED_ACCESS` by an earlier pass — transitions cleanly into
    /// `INDIRECT_ARGUMENT` for `ExecuteIndirect` and restores back to
    /// `UNORDERED_ACCESS` in `done`, while repeated execution of the same list
    /// stays synchronized on `budget`.
    #[test]
    fn distinct_count_buffer_transitions_for_execute_indirect_and_restores() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let arg = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg,
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: valid command signature description for Dispatch.
        unsafe {
            host.device
                .CreateCommandSignature(&sig_desc, None, &mut sig)
                .unwrap();
        }
        let sig = sig.unwrap();

        let (args, args_data) = host.shared_buffer(256).unwrap();
        let (scratch, _) = host.shared_buffer(256).unwrap();
        let (count_uav, count_uav_data) = host.shared_buffer(256).unwrap();
        let (count_common, count_common_data) = host.shared_buffer(256).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: mapped host buffers written before submission and read after.
        unsafe {
            let a = args_data.cast::<u32>();
            // Record 0: 0 groups (no-op Dispatch via ExecuteIndirect); record 1: garbage past count=1.
            a.add(0).write_volatile(0);
            a.add(1).write_volatile(0);
            a.add(2).write_volatile(0);
            a.add(3).write_volatile(u32::MAX);
            a.add(4).write_volatile(1);
            a.add(5).write_volatile(1);

            let cu = count_uav_data.cast::<u32>();
            cu.add(0).write_volatile(1);
            cu.add(1).write_volatile(1);
            cu.add(2).write_volatile(1);

            let cc = count_common_data.cast::<u32>();
            cc.add(0).write_volatile(1);

            let args_gpu = args.GetGPUVirtualAddress();
            let scratch_gpu = scratch.GetGPUVirtualAddress();
            let count_uav_gpu = count_uav.GetGPUVirtualAddress();
            let count_common_gpu = count_common.GetGPUVirtualAddress();

            // Put `count_uav` into `UNORDERED_ACCESS` first (reproducing a
            // count buffer written by a prior UAV pass or earlier indirect call).
            let prep = super::bound(&list, &count_uav, None, count_uav_gpu, (2, 12, 1)).unwrap();
            assert_eq!(prep, Clamped::Ready);
            super::done(&list, &count_uav, None, prep);

            // Case 1: distinct `count` already in `UNORDERED_ACCESS`.
            let s1 = super::bound(
                &list,
                &args,
                Some((&count_uav, count_uav_gpu)),
                args_gpu,
                (2, 12, 2),
            )
            .unwrap();
            assert_eq!(s1, Clamped::Ready);
            // Point the active compute PSO's UAV slots away from `args`/`count`
            // (which are now in `INDIRECT_ARGUMENT`) before `ExecuteIndirect`.
            list.list.SetComputeRootUnorderedAccessView(1, scratch_gpu);
            list.list.SetComputeRootUnorderedAccessView(3, scratch_gpu);
            list.list
                .ExecuteIndirect(&sig, 2, &args, 0, Some(&count_uav), 0);
            super::done(&list, &args, Some(&count_uav), s1);

            // Case 2: distinct `count` starting fresh in `COMMON` (promoted by u2).
            let s2 = super::bound(
                &list,
                &args,
                Some((&count_common, count_common_gpu)),
                args_gpu,
                (2, 12, 2),
            )
            .unwrap();
            assert_eq!(s2, Clamped::Ready);
            list.list.SetComputeRootUnorderedAccessView(1, scratch_gpu);
            list.list.SetComputeRootUnorderedAccessView(3, scratch_gpu);
            list.list
                .ExecuteIndirect(&sig, 2, &args, 0, Some(&count_common), 0);
            super::done(&list, &args, Some(&count_common), s2);

            list.close();
            // Execute the same list twice in one batch to exercise post-clamp
            // `budget` UAV synchronization across repeated executions.
            host.execute(&[&list, &list]);
            host.drain();

            // Record 0 ran empty (0 groups); record 1 was past `count = 1`.
            assert_eq!(a.add(0).read_volatile(), 0);
            assert_eq!(a.add(3).read_volatile(), u32::MAX);
        }
    }
}
