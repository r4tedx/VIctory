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
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
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

/// Which host buffer a scoped indirect UAV barrier targets (never a wildcard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Args,
    Count,
    Budget,
}

/// The barriers and transition planned for one [`bound`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plan {
    /// Scoped UAV barriers emitted before the clamp dispatch.
    pre_uav: [Target; 3],
    pre_uav_len: usize,
    /// Whether the clamp dispatches and emits the single post-clamp
    /// `UAV -> INDIRECT_ARGUMENT` transition (and therefore requires [`done`]).
    transition: bool,
}

impl Plan {
    /// Plans barriers for `records` indirect records. `count_ptr` is the raw
    /// resource pointer of the count buffer, if any (deduplicated against
    /// `args_ptr`). `budget_written` is true when an earlier clamp on this
    /// list already wrote the budget buffer.
    fn for_call(
        records: u32,
        args_ptr: usize,
        count_ptr: Option<usize>,
        budget_written: bool,
    ) -> Self {
        if records == 0 {
            return Self {
                pre_uav: [Target::Args; 3],
                pre_uav_len: 0,
                transition: false,
            };
        }
        let mut pre_uav = [Target::Args; 3];
        let mut len = 1;
        if count_ptr.is_some_and(|c| c != args_ptr) {
            pre_uav[len] = Target::Count;
            len += 1;
        }
        if budget_written {
            pre_uav[len] = Target::Budget;
            len += 1;
        }
        Self {
            pre_uav,
            pre_uav_len: len,
            transition: true,
        }
    }

    fn pre_uav(&self) -> &[Target] {
        &self.pre_uav[..self.pre_uav_len]
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. `count`
/// is the indirect count buffer, if the call gives one (synchronized before
/// execution when it is a different host buffer than `buffer`, without
/// changing its state). Replaces the list's compute root signature and
/// pipeline when returning [`Clamped::Ready`] (callers bind theirs after).
/// `None` if the clamp is not available or the arguments are unaligned.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    count: Option<&ID3D12Resource>,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> Option<Clamped> {
    if records == 0 {
        return Some(Clamped::Empty);
    }
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let (c, (budget_res, budget, budget_written)) = (clamp()?, list.budget()?);
    let plan = Plan::for_call(
        records,
        buffer.as_raw() as usize,
        count.map(|r| r.as_raw() as usize),
        budget_written,
    );
    if !plan.transition {
        return Some(Clamped::Empty);
    }
    let barrier_for = |target: Target| match target {
        Target::Args => uav_barrier_for(buffer),
        Target::Count => uav_barrier_for(count.expect("count buffer planned")),
        Target::Budget => uav_barrier_for(budget_res),
    };
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and live buffers
    // the caller and list hold.
    unsafe {
        // Earlier shader writes (the title's argument/count passes, and any
        // prior clamp's budget write in this list) first, scoped to their
        // buffers so unrelated UAV waves keep running.
        match plan.pre_uav() {
            [a] => l.ResourceBarrier(&[barrier_for(*a)]),
            [a, b] => l.ResourceBarrier(&[barrier_for(*a), barrier_for(*b)]),
            [a, b, d] => l.ResourceBarrier(&[barrier_for(*a), barrier_for(*b), barrier_for(*d)]),
            _ => unreachable!(),
        }
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        // One group walks every record.
        l.Dispatch(1, 1, 1);
        // Transitioning UAV -> INDIRECT_ARGUMENT already synchronizes the
        // clamp's UAV writes to `buffer`; `budget` is synchronized before the
        // next clamp on this list.
        l.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
    Some(Clamped::Ready)
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it. No-op when `clamped` is [`Clamped::Empty`]
/// (the buffer was never transitioned).
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource, clamped: Clamped) {
    if !clamped.transitioned() {
        return;
    }
    // SAFETY: an open list and a live buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
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

    /// Zero records skips pre-clamp barriers, clamp dispatch, transition, and
    /// restore even when a count buffer and dirty budget are present.
    #[test]
    fn zero_records_skips_clamp_and_barriers() {
        let empty = Plan::for_call(0, 0x1000, None, false);
        assert!(empty.pre_uav().is_empty());
        assert!(!empty.transition);

        let with_count_and_budget = Plan::for_call(0, 0x1000, Some(0x2000), true);
        assert!(with_count_and_budget.pre_uav().is_empty());
        assert!(!with_count_and_budget.transition);
        assert!(!Clamped::Empty.transitioned());
        assert!(Clamped::Ready.transitioned());
    }

    /// Non-zero records plan only scoped pre-clamp UAV barriers (deduplicating
    /// aliased count buffers, adding the budget only after an earlier clamp)
    /// and a single post-clamp transition with no redundant UAV barrier.
    #[test]
    fn barrier_plan_scopes_pre_uav_targets() {
        let first = Plan::for_call(4, 0x1000, None, false);
        assert_eq!(first.pre_uav(), &[Target::Args]);
        assert!(first.transition);

        let second = Plan::for_call(4, 0x1000, None, true);
        assert_eq!(second.pre_uav(), &[Target::Args, Target::Budget]);
        assert!(second.transition);

        let distinct_count = Plan::for_call(8, 0x1000, Some(0x2000), false);
        assert_eq!(distinct_count.pre_uav(), &[Target::Args, Target::Count]);
        assert!(distinct_count.transition);

        let distinct_count_and_budget = Plan::for_call(8, 0x1000, Some(0x2000), true);
        assert_eq!(
            distinct_count_and_budget.pre_uav(),
            &[Target::Args, Target::Count, Target::Budget]
        );
        assert!(distinct_count_and_budget.transition);

        let same_count_first = Plan::for_call(8, 0x1000, Some(0x1000), false);
        assert_eq!(same_count_first.pre_uav(), &[Target::Args]);

        let same_count_second = Plan::for_call(8, 0x1000, Some(0x1000), true);
        assert_eq!(same_count_second.pre_uav(), &[Target::Args, Target::Budget]);
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
                super::done(&list, &buffer, state);
            }
            list.close();
            host.execute(&[&list]);
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
}
