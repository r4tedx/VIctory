use super::*;
use crate::draw::State;
use victory_shader::emit::{self, HostShader, Target};
use victory_shader::native::{ComputeIo, RootParam, RootType, StageIo};
use victory_shader::{Container, HwStage, NativeProgram, io, ir, resource, structure};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

#[derive(Clone, Copy)]
enum Output {
    StoreLane,
    AddLane,
    Count,
}

fn lane_shader(target: Target, output: Output) -> HostShader {
    // Two guest waves per group; output at (group_x + 3 * group_y) * 128
    // plus the guest thread's index.
    let mut words: Vec<u32> = vec![
        0x9308_FF01,
        512, // s8 = group_x * 512
        0x9309_FF02,
        1536,        // s9 = group_y * 1536
        0x8008_0908, // s_add_u32 s8, s8, s9
        0xBE82_0300,
        0xBE83_0380,
        0xF408_0101,
        0xFA00_0000, // load UAV descriptor
        0xBF8C_C07F, // wait for descriptor
    ];
    match output {
        Output::StoreLane | Output::AddLane => words.extend([
            0xD760_0009,
            0x0001_5100, // v_readlane_b32 s9, v0, 40
            0x7E04_0209, // v_mov_b32 v2, s9
        ]),
        Output::Count => words.push(0x7E04_0281), // v_mov_b32 v2, 1
    }
    words.extend([
        0x3406_0082,
        0x4A06_0608, // v3 = (v0 << 2) + s8
        match output {
            Output::StoreLane => 0xE070_1000, // buffer_store_dword
            Output::AddLane | Output::Count => 0xE0C8_1000, // buffer_atomic_add
        },
        0x8001_0203, // v2, v3, s[4:7], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (1 << 1) | (1 << 7) | (1 << 8),
            threads: [128, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let container = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let program = ir::build(&native.code, native.wave64, native.vgprs).unwrap();

    let io = io::raise(&program, &native).unwrap();
    let structure = structure::structure(&program).unwrap();
    let resources =
        resource::track(&program, &native.layout, native.stage, Some(&structure)).unwrap();
    emit::emit(
        &container, &native, &program, &structure, &io, &resources, target,
    )
    .unwrap()
}

fn test_host() -> Option<&'static crate::host::Host> {
    let host = host();
    if host.is_none() {
        assert!(
            !std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1")
                && !std::env::var("VICTORY_ADAPTER").is_ok_and(|v| !v.is_empty()),
            "requested D3D12 adapter is unavailable"
        );
        eprintln!("no D3D12 test host; skipped");
    }
    host
}

fn is_radeon_igpu(host: &crate::host::Host) -> bool {
    let mut architecture = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
    // SAFETY: queries on the live device, its DXGI adapter and a sized out buffer.
    unsafe {
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| {
                factory.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid())
            })
            .and_then(|adapter| adapter.GetDesc1());
        adapter.is_ok_and(|desc| desc.VendorId == 0x1002)
            && host
                .device
                .CheckFeatureSupport(
                    D3D12_FEATURE_ARCHITECTURE1,
                    (&raw mut architecture).cast(),
                    size_of_val(&architecture) as u32,
                )
                .is_ok()
            && architecture.UMA.as_bool()
    }
}

const ADD_ONE: &str = r#"
RWByteAddressBuffer dst : register(u0);
[RootSignature("UAV(u0)")]
[numthreads(1, 1, 1)]
void main() {
    uint old;
    dst.InterlockedAdd(0, 1, old);
}
"#;

fn add_one(list: &HostList, worker: &Clamp, address: u64) {
    // SAFETY: an open list and a live UAV holding at least one word.
    unsafe {
        list.list.ResourceBarrier(&[uav_barrier()]);
        list.list.SetComputeRootSignature(&worker.root);
        list.list.SetPipelineState(&worker.pso);
        list.list.SetComputeRootUnorderedAccessView(0, address);
        list.list.Dispatch(1, 1, 1);
    }
}

struct TitleBuffer {
    data: Box<[u32]>,
    address: u64,
}

impl TitleBuffer {
    fn new<const N: usize>(words: [u32; N]) -> Self {
        let mut data: Box<[u32]> = Box::new(words);
        let address = data.as_mut_ptr() as u64;
        crate::shadow::register(address as usize, address, size_of_val(&*data) as u64);
        Self { data, address }
    }
}

impl Drop for TitleBuffer {
    fn drop(&mut self) {
        crate::shadow::unregister(self.data.as_mut_ptr() as usize, self.address);
    }
}

fn bind_output(host: &crate::host::Host, list: &HostList, buffer: &ID3D12Resource, lanes: usize) {
    // SAFETY: each heap and mapped UAV stays live through submission.
    unsafe {
        let heap = |kind| -> ID3D12DescriptorHeap {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .unwrap()
        };
        let view_heap = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        host.device.CreateUnorderedAccessView(
            buffer,
            None::<&ID3D12Resource>,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: lanes as u32,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            }),
            view_heap.GetCPUDescriptorHandleForHeapStart(),
        );
        let sampler_heap = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        list.keep(&view_heap);
        list.keep(&sampler_heap);
        list.list
            .SetDescriptorHeaps(&[Some(view_heap), Some(sampler_heap)]);
    }
}

fn release_pipeline(pipeline: crate::object::This) {
    let release =
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(pipeline, 0x10);
    assert_eq!(release(pipeline), 0);
}

#[test]
fn execute_indirect_preserves_two_384_lane_reference_cases() {
    let Some(host) = test_host() else { return };
    let target = host.target();
    if !target.runs(32) && !target.runs(64) {
        eprintln!("D3D12 wave lanes {target:?} cannot execute the translated lane fixture");
        return;
    }
    let shader = lane_shader(target, Output::StoreLane);
    let waves = if target.runs(64) { 1 } else { 2 };
    assert_eq!(shader.group_waves, waves);
    eprintln!("D3D12 wave lanes {target:?}, groups per guest group {waves}");
    let pipeline = crate::pipeline::test_compute_pipeline(shader);
    let arguments = TitleBuffer::new([0, 0, 0, 0, 2, 1, 1, 0xfeed, 3, 1, 1, 0xbeef]);
    let address = arguments.address;
    let bytes = size_of_val(&*arguments.data) as u64;
    let shadow = crate::shadow::find(address, bytes).unwrap();
    let list = host.new_list().unwrap();
    let worker = make(ADD_ONE).unwrap();
    shadow.refresh(&list, address, bytes);
    shadow.gpu_writes(&list, address + 4, 16, false);
    add_one(&list, &worker, shadow.gpu(address + 4));
    add_one(&list, &worker, shadow.gpu(address + 16));

    const LANES: usize = 384;
    let outputs =
        std::array::from_fn::<_, 2, _>(|_| host.shared_buffer((LANES * 4) as u64).unwrap());
    let mut state = State::default();
    state.reset(pipeline);
    for (buffer, mapped) in &outputs {
        // SAFETY: the buffer has space for all output lanes.
        unsafe { mapped.write_bytes(0, LANES * 4) };
        bind_output(host, &list, buffer, LANES);
        state.heaps_bound_by_caller();
        assert!(state.execute_indirect(&list, (2, 16, 2), address + 16, Some(address + 4)));
    }
    // A writer later in the same list exercises restoration of UAV state.
    add_one(&list, &worker, shadow.gpu(address + 16));
    list.close();
    host.execute_checked(&list);
    release_pipeline(pipeline);
    for (case, (_, output)) in outputs.iter().enumerate() {
        // SAFETY: submission has completed and the mapped buffer still lives.
        let lanes = unsafe { std::slice::from_raw_parts(output.cast::<u32>(), LANES) };
        for (lane, &value) in lanes.iter().enumerate() {
            assert_eq!(
                value,
                40 + 64 * ((lane / 64) % 2) as u32,
                "case {case}, lane {lane}"
            );
        }
    }
    // Inspect the actual source, including count and padding, after both
    // executions and the second producer.
    let mut data = std::ptr::null_mut();
    // SAFETY: the source is a mapped, CPU-visible shadow; the list finished.
    unsafe {
        shadow.buffer.Map(0, None, Some(&mut data)).unwrap();
        assert_eq!(
            *data.cast::<[u32; 12]>(),
            [0, 1, 0, 0, 4, 1, 1, 0xfeed, 3, 1, 1, 0xbeef]
        );
        shadow.buffer.Unmap(0, None);
    }
}

#[test]
fn execute_indirect_runs_both_counted_records() {
    let Some(host) = test_host() else { return };
    let target = host.target();
    for output in [Output::Count, Output::AddLane] {
        let reads_lane = matches!(output, Output::AddLane);
        if reads_lane && !target.runs(32) && !target.runs(64) {
            eprintln!("D3D12 wave lanes {target:?} cannot execute the translated lane fixture");
            continue;
        }
        let shader = lane_shader(target, output);
        let waves = if reads_lane && !target.runs(64) { 2 } else { 1 };
        assert_eq!(shader.group_waves, waves);
        let pipeline = crate::pipeline::test_compute_pipeline(shader);
        // Each record touches its own group along X or Y. Both contribute
        // to the origin. Only the counted-out record would touch X=2.
        let arguments = TitleBuffer::new([2, 1, 1, 11, 1, 2, 1, 22, 3, 1, 1, 33]);
        let count = TitleBuffer::new([2]);
        const LANES: usize = 512;
        let (buffer, mapped) = host.shared_buffer((LANES * 4) as u64).unwrap();
        // SAFETY: the buffer has space for all output lanes.
        unsafe { mapped.write_bytes(0, LANES * 4) };
        let list = host.new_list().unwrap();
        bind_output(host, &list, &buffer, LANES);
        let mut state = State::default();
        state.reset(pipeline);
        state.heaps_bound_by_caller();
        assert!(state.execute_indirect(&list, (2, 16, 3), arguments.address, Some(count.address)));
        list.close();
        host.execute_checked(&list);
        release_pipeline(pipeline);
        // SAFETY: submission completed and the output is still mapped.
        let lanes = unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), LANES) };
        let per_record = |lane| {
            if reads_lane {
                40 + 64 * ((lane / 64) % 2) as u32
            } else {
                1
            }
        };
        // This driver behavior was reproduced on the base path. Accept the
        // exact first-record output on Radeon UMA at k=1; keep k>1 strict.
        if waves == 1
            && is_radeon_igpu(host)
            && lanes
                .iter()
                .enumerate()
                .all(|(lane, &value)| value == per_record(lane) * [1, 1, 0, 0][lane / 128])
        {
            eprintln!("Radeon iGPU: base k=1 path executed only the first record");
            continue;
        }
        for (lane, &value) in lanes.iter().enumerate() {
            let expected = per_record(lane) * [2, 1, 0, 1][lane / 128];
            assert_eq!(value, expected, "k={waves}, lane {lane}");
        }
        eprintln!("D3D12 wave lanes {target:?}, both records executed at k={waves}");
    }
}

/// A dispatch with no pipeline set is dropped before its arguments are
/// refreshed or clamped (the play tree's order, for the calls made while the
/// world builds its pipelines): nothing is recorded for it.
#[test]
fn drops_a_dispatch_without_a_pipeline_before_the_clamp() {
    let Some(host) = test_host() else { return };
    let arguments = TitleBuffer::new([u32::MAX, 1, 1]);
    let list = host.new_list().unwrap();
    #[cfg(feature = "diagnostics")]
    drop(crate::pipeline::take_drops());
    assert!(!State::default().execute_indirect(&list, (2, 12, 1), arguments.address, None));
    #[cfg(feature = "diagnostics")]
    assert!(
        crate::pipeline::take_drops()
            .iter()
            .any(|(why, _)| why == "no pipeline set")
    );
    list.close();
    host.execute_checked(&list);
    assert_eq!(*arguments.data, [u32::MAX, 1, 1]);
}

#[test]
fn reads_arguments_written_earlier_in_the_list_at_k1_and_k2() {
    let Some(host) = test_host() else { return };
    let worker = make(ADD_ONE).unwrap();
    let desc = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
        ..Default::default()
    };
    for waves in [1, 2] {
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: one complete dispatch argument description.
        unsafe {
            host.device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: if waves == 1 { 16 } else { 12 },
                        NumArgumentDescs: 1,
                        pArgumentDescs: &desc,
                        NodeMask: 0,
                    },
                    None,
                    &mut sig,
                )
                .unwrap();
        }
        let sig = sig.unwrap();
        let list = host.new_list().unwrap();
        let original = [2u32, 1, 1, 0xfeed, 1, 1, 1, 0xbeef, 1];
        let (source, source_cpu) = host.shared_buffer(256).unwrap();
        let (hits, hits_cpu) = host.shared_buffer(256).unwrap();
        // SAFETY: both mapped buffers have enough space for these writes.
        let (source_gpu, hits_gpu) = unsafe {
            source_cpu.cast::<[u32; 9]>().write(original);
            hits_cpu.cast::<u32>().write(0);
            (source.GetGPUVirtualAddress(), hits.GetGPUVirtualAddress())
        };
        add_one(&list, &worker, source_gpu);
        let (args, count_offset) = if waves == 1 {
            let count = Some((&source, source_gpu + 32));
            let clamped = bound(&list, &source, count, source_gpu, (2, 16, 2));
            assert_eq!(clamped, Some(Clamped::Ready));
            (source.clone(), 32)
        } else {
            (
                dispatch(&list, source_gpu, Some(source_gpu + 32), (16, 2, waves), 0).unwrap(),
                24,
            )
        };
        // SAFETY: the clamp put the records and count in the indirect
        // state. The worker only writes the live hits buffer.
        unsafe {
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            list.list
                .ExecuteIndirect(&sig, 2, &args, 0, Some(&args), count_offset);
        }
        // The clamp's count is in the same buffer; the rewrite's in its own.
        done(&list, &args, None, Clamped::Ready);
        add_one(&list, &worker, source_gpu);
        list.close();
        host.execute_checked(&list);
        // SAFETY: the host has waited for the GPU to finish; both buffers live.
        unsafe {
            assert_eq!(hits_cpu.cast::<u32>().read(), 3 * waves, "k={waves}");
            let mut expected = original;
            expected[0] = 4;
            assert_eq!(source_cpu.cast::<[u32; 9]>().read(), expected, "k={waves}");
        }
    }
}

#[test]
fn dispatch_limits_charge_only_scaled_host_groups() {
    let Some(host) = test_host() else { return };
    let list = host.new_list().unwrap();
    // A draw leaves a fractional group in the shared budget. A cap-sized
    // record then fits, another is refunded, and a small record still fits.
    // The final draw spends the exact remainder.
    let (draws, draw_data) = host.shared_buffer(256).unwrap();
    let remaining = (1 << 26) - 1 - 128;
    // SAFETY: the mapped buffer has space for the two draw records.
    let draw_address = unsafe {
        draw_data
            .cast::<[u32; 8]>()
            .write([1, 1, 0, 0, 1, remaining, 0, 0]);
        draws.GetGPUVirtualAddress()
    };
    let clamped = bound(&list, &draws, None, draw_address, (0, 16, 1));
    assert_eq!(clamped, Some(Clamped::Ready));
    done(&list, &draws, None, Clamped::Ready);
    let original = [
        [512u32, 1024, 1, 11],
        [32768, 1, 1, 22],
        [512, 1025, 1, 33],
        [512, 1024, 1, 44],
        [1, 1, 1, 55],
    ];
    let (source, input) = host.shared_buffer(256).unwrap();
    let (readback, output) = host.shared_buffer(256).unwrap();
    // SAFETY: both the padded records and the count fit in the source.
    let address = unsafe {
        input.cast::<[[u32; 4]; 5]>().write(original);
        input.add(80).cast::<u32>().write(0);
        source.GetGPUVirtualAddress()
    };
    let calls = [
        (64, 1, Some(address + 80)),
        (0, 3, None),
        (48, 1, None),
        (64, 1, None),
        (64, 1, None),
    ];
    let mut dst = 0;
    for (index, (at, records, count)) in calls.into_iter().enumerate() {
        if index == calls.len() - 1 {
            let clamped = bound(&list, &draws, None, draw_address + 16, (0, 16, 1));
            assert_eq!(clamped, Some(Clamped::Ready));
            done(&list, &draws, None, Clamped::Ready);
        }
        let packed = dispatch(&list, address + at, count, (16, records, 2), 0).unwrap();
        let bytes = u64::from(records) * 12;
        // SAFETY: the scratch buffer is in INDIRECT_ARGUMENT after the
        // rewrite and the nonoverlapping readbacks fit in the mapped buffer.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &packed,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list
                .CopyBufferRegion(&readback, dst, &packed, 0, bytes);
            list.list.ResourceBarrier(&[transition(
                &packed,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        dst += bytes;
    }
    list.close();
    host.execute_checked(&list);
    let expected = [
        [0, 0, 0],
        [1024, 1024, 1],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [2, 1, 1],
        [0, 0, 0],
    ];
    // SAFETY: submission has completed and the mapped buffers still live.
    unsafe {
        assert_eq!(output.cast::<[[u32; 3]; 7]>().read(), expected);
        assert_eq!(input.cast::<[[u32; 4]; 5]>().read(), original);
        assert_eq!(
            draw_data.cast::<[u32; 8]>().read(),
            [1, 1, 0, 0, 1, remaining, 0, 0]
        );
    }
}

/// One submission, with rewrites only. The large packed dispatches are never executed.
#[test]
fn rewrite_many_records_once() {
    let Some(host) = test_host() else { return };
    const RECORDS: u32 = 6000;
    const STRIDE: u32 = 20;
    const WAVES: u32 = 3;
    let regular: Vec<_> = (0..RECORDS)
        .map(|i| [1 + i % 7, 1 + (i / 7) % 3, 1, i, !i])
        .collect();
    let cap = [341, 1025, 1]; // 1,048,575 host groups at k=3.
    let contended: Vec<_> = (0..RECORDS)
        .map(|i| [cap[0], cap[1], cap[2], i, !i])
        .collect();
    let input: Vec<_> = regular.iter().chain(&contended).copied().collect();
    let counts = [4100_u32, 4032, RECORDS];
    let calls = [
        (64, None, false),
        (RECORDS, None, false),
        (RECORDS, Some(0), false),
        (4096, Some(1), false),
        (RECORDS, Some(2), true),
    ];
    let bytes = |records| u64::from(records) * 12 + 4;
    let (source, source_cpu) = host.shared_buffer(size_of_val(&*input) as u64).unwrap();
    let (count, count_cpu) = host.shared_buffer(size_of_val(&counts) as u64).unwrap();
    let (readback, readback_cpu) = host
        .shared_buffer(calls.iter().map(|&(records, _, _)| bytes(records)).sum())
        .unwrap();
    // SAFETY: the mapped buffers have space for all their input words.
    let (source_gpu, count_gpu) = unsafe {
        source_cpu
            .cast::<[u32; 5]>()
            .copy_from_nonoverlapping(input.as_ptr(), input.len());
        count_cpu.cast::<[u32; 3]>().write(counts);
        (source.GetGPUVirtualAddress(), count.GetGPUVirtualAddress())
    };
    let list = host.new_list().unwrap();
    let mut at = 0;
    let mut first_buffer = None;
    for (records, count_index, contend) in calls {
        let source_at = source_gpu
            + if contend {
                u64::from(RECORDS * STRIDE)
            } else {
                0
            };
        let count_at = count_index.map(|i| count_gpu + i as u64 * 4);
        let packed = dispatch(&list, source_at, count_at, (STRIDE, records, WAVES), 0).unwrap();
        // SAFETY: packed is in INDIRECT_ARGUMENT and readback ranges are disjoint.
        unsafe {
            let address = packed.GetGPUVirtualAddress();
            if records == 64 {
                first_buffer = Some(address);
            } else {
                assert_ne!(
                    Some(address),
                    first_buffer,
                    "scratch allocation should have grown"
                );
            }
            list.list.ResourceBarrier(&[transition(
                &packed,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list
                .CopyBufferRegion(&readback, at, &packed, 0, bytes(records));
            list.list.ResourceBarrier(&[transition(
                &packed,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        at += bytes(records);
    }
    list.close();
    host.execute_checked(&list);

    // SAFETY: submission completed; the mappings cover the original extents.
    let words = unsafe { std::slice::from_raw_parts(readback_cpu.cast::<u32>(), at as usize / 4) };
    let mut at = 0;
    let mut charged = 0;
    for (records, count_index, contend) in calls {
        let active = count_index.map_or(records, |i| counts[i]);
        let end = at + records as usize * 3;
        assert_eq!(words[end], active, "effective count");
        let mut accepted = 0;
        let (packed, []) = words[at..end].as_chunks::<3>() else {
            unreachable!("readback contains whole records");
        };
        for (i, &record) in packed.iter().enumerate() {
            let expected = if contend {
                let scaled = [cap[0] * WAVES, cap[1], cap[2]];
                if record == scaled {
                    accepted += 1;
                    scaled
                } else {
                    [0; 3]
                }
            } else if i < active as usize {
                let [x, y, z, _, _] = regular[i];
                [x * WAVES, y, z]
            } else {
                [0; 3]
            };
            assert_eq!(record, expected, "record {i} of {records}");
            charged += expected.into_iter().product::<u32>();
        }
        if contend {
            assert_eq!(accepted, 1, "exactly one large record fits");
        }
        at = end + 1;
    }
    // SAFETY: the submitted list has finished and all three buffers are live.
    unsafe {
        assert_eq!(list.indirect_budget_words(), [charged * 64, charged]);
        assert_eq!(
            std::slice::from_raw_parts(source_cpu.cast::<[u32; 5]>(), input.len()),
            input
        );
        assert_eq!(count_cpu.cast::<[u32; 3]>().read(), counts);
    }
}
