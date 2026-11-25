//! End-to-end tests of waterfall loops over per-lane descriptors: the
//! synthetic GFX10 programs of `resource::tests` lowered to DXIL, validated
//! by `dxil.dll` when present, and run on the host GPU and WARP, each lane
//! reading the resource its own descriptor names.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0 (textures, or with `buffers` structured buffers of dwords, at entries
/// 0 and 1, a raw UAV at entry 2) and the table of samplers at s1.
fn build(words: &[u32], buffers: bool) -> HostShader {
    let lowered = lower_as(words, buffers, false).expect("lower");
    // One access with each lane's own heap index, no wave operation.
    assert_eq!(lowered.nonuniform_handles, 1);
    assert!(!lowered.wave_ops);
    host_shader(lowered)
}

/// The wave32 compute program of 32 threads [`build`] lowers.
fn native(words: &[u32], buffers: bool) -> NativeProgram {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: user_data,
    };
    let views = if buffers {
        range(ResourceType::Srv, 0, 4, 4, 0)
    } else {
        range(ResourceType::Srv, 0, 8, 0, 0)
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            views,
            range(ResourceType::Uav, 16, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 0, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// [`build`]'s program lowered in the per-lane form, or with `console_cf`
/// in the console flow (`lower::CONSOLE_FLOW`).
fn lower_as(
    words: &[u32],
    buffers: bool,
    console_cf: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    lower_for(&native(words, buffers), super::Target::ANY, console_cf)
}

fn lower_for(
    native: &NativeProgram,
    target: super::Target,
    console_cf: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    let console = structure::structure_console(&p).expect("structure_console");
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: if console_cf { &console } else { &st },
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target,
        console_cf,
        per_lane_only: false,
    })
}

/// The host shader of `lowered`, validated by `dxil.dll` when present.
fn host_shader(lowered: lower::Lowered) -> HostShader {
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        group_waves: lowered.group_waves,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's 32 dwords.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    // One GPU test at a time, with the VALU tests: in parallel on the
    // hardware adapter and WARP the test process sometimes died of an
    // access violation (7 of 24 runs of these four tests).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Both tables start at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// The red channel each lane sampled from two 1x1 textures, red 1.0 at
/// heap index 0 and 2.0 at 1.
fn sampled(shader: &HostShader) -> Vec<(&'static str, Vec<f32>)> {
    let first = [[1.0f32, 0.0, 0.0, 1.0]];
    let second = [[2.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &first),
        View::Texture(1, 1, &second),
        View::RawUav(32),
    ];
    run(shader, &views)
        .into_iter()
        .map(|(label, words)| (label, words.into_iter().map(f32::from_bits).collect()))
        .collect()
}

#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let shader = build(WATERFALL_JOINED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 took the first texture, the others the second.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    use crate::resource::tests::WATERFALL_JOINED_MIXED_WAVE32;
    let shader = build(WATERFALL_JOINED_MIXED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes
        // below 24 picked the first, the rest the second.
        let want: Vec<f32> = (0..32)
            .map(|l| match l {
                0..16 => [1.0, 2.0][l & 1],
                16..24 => 1.0,
                _ => 2.0,
            })
            .collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
    let shader = build(WATERFALL_JOINED_BUFFER_WAVE32, true);
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(4),
        },
        View::RawUav(32),
    ];
    for (label, got) in run(&shader, &views) {
        // Dword `lane` of the first buffer below 16, of the second from 16
        // to 23; lanes from 24 on load and store nothing.
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let shader = build(WATERFALL_INDEXED_WAVE32, false);
    let results = sampled(&shader);
    assert!(!results.is_empty());
    for (label, red) in results {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    use crate::resource::tests::cmpx_waterfall;
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        let shader = build(&cmpx_waterfall(false, folded), false);
        let results = sampled(&shader);
        assert!(!results.is_empty());
        for (label, red) in results {
            assert_eq!(red, want, "{label}, folded {folded}");
        }
    }
}

/// In the console flow every lane reaches a join by the edge the wave took,
/// so the edge does not say which descriptor a lane holds. Lowered there,
/// all 32 lanes sampled the second texture where lanes 0-15 hold the first
/// (RTX 5070).
/// The console flow refuses it by name, and `emit` keeps the per-lane form
/// where that translates.
#[test]
fn the_console_flow_refuses_descriptors_joined_per_lane() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    match lower_as(WATERFALL_JOINED_WAVE32, false, true) {
        Err(crate::ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, lower::LANE_PHI_IN_CONSOLE_FLOW);
        }
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("lowered in the console flow"),
    }
}

/// [`WATERFALL_JOINED_WAVE32`] with a named lane read (the sample's second
/// coordinate comes from lane 5), which with the arms' branches asks for
/// the console flow on every host.
///
/// [`WATERFALL_JOINED_WAVE32`]: crate::resource::tests::WATERFALL_JOINED_WAVE32
fn joined_with_a_lane_read() -> Vec<u32> {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    const MOV: u32 = 0x7E06_0302; // v_mov_b32 v3, v2
    let at = WATERFALL_JOINED_WAVE32
        .iter()
        .position(|&w| w == MOV)
        .expect("the coordinate's copy");
    let mut words = WATERFALL_JOINED_WAVE32[..at].to_vec();
    words.extend([
        0xD760_002C,
        (128 + 5) << 9 | 0x100, // v_readlane_b32 s44, v0, 5
        0x7E06_022C,            // v_mov_b32 v3, s44
    ]);
    words.extend(&WATERFALL_JOINED_WAVE32[at + 1..]);
    words
}

/// A program that takes the console flow only for its lane reads, with a
/// descriptor joined per lane: it asks for the console flow, the console
/// flow refuses the join, and `emit` keeps the per-lane form.
#[test]
fn joined_descriptors_keep_the_per_lane_form_when_the_console_flow_is_asked_for() {
    let words = joined_with_a_lane_read();
    let why = |l: Result<lower::Lowered, crate::ShaderError>| match l {
        Err(crate::ShaderError::Unsupported { what, .. }) => what,
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("lowered"),
    };
    assert_eq!(why(lower_as(&words, false, false)), lower::CONSOLE_FLOW);
    assert_eq!(
        why(lower_as(&words, false, true)),
        lower::LANE_PHI_IN_CONSOLE_FLOW
    );
    let native = native(&words, false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let container = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&container).expect("container");
    for target in [
        super::Target::ANY,
        super::Target {
            lanes_min: 32,
            lanes_max: 32,
        },
        super::Target {
            lanes_min: 32,
            lanes_max: 64,
        },
    ] {
        let shader = super::emit(&c, &native, &p, &st, &io, &res, target).expect("emit");
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
}

/// [`WATERFALL_JOINED_WAVE32`] as a wave64 program (the masks in SGPR
/// pairs, encodings from llvm-mc for gfx1030 wave64), with `first_lane`
/// the sample's second coordinate read from the first lane's id
/// (`v_readfirstlane_b32 s44, v0`, outside the waterfall loop).
///
/// [`WATERFALL_JOINED_WAVE32`]: crate::resource::tests::WATERFALL_JOINED_WAVE32
fn joined_wave64(first_lane: bool) -> Vec<u32> {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let mut words = Vec::new();
    for &w in WATERFALL_JOINED_WAVE32 {
        match w {
            0x7E06_0302 if first_lane => words.extend([
                0x7E58_0500, // v_readfirstlane_b32 s44, v0
                0x7E06_022C, // v_mov_b32 v3, s44
            ]),
            0xBE94_3C6A => words.push(0xBE94_246A), // s_and_saveexec_b64 s[20:21], vcc
            0x8914_147E => words.push(0x8994_147E), // s_xor_b64 s[20:21], exec, s[20:21]
            0xBE94_3D14 => words.push(0xBE94_2514), // s_or_saveexec_b64 s[20:21], s[20:21]
            0x897E_147E => words.push(0x89FE_147E), // s_xor_b64 exec, exec, s[20:21]
            0x887E_147E => words.push(0x88FE_147E), // s_or_b64 exec, exec, s[20:21]
            0xBE97_037E => words.push(0xBEAE_047E), // s_mov_b64 s[46:47], exec
            0xD4E2_0015 => words.push(0xD4E2_0030), // v_cmp_eq_u64_e64 s[48:49], ...
            0xD4E2_0016 => words.push(0xD4E2_0032), // v_cmp_eq_u64_e64 s[50:51], ...
            0x8715_1615 => words.push(0x87B0_3230), // s_and_b64 s[48:49], s[48:49], s[50:51]
            0xBE95_3C15 => words.push(0xBEB0_2430), // s_and_saveexec_b64 s[48:49], s[48:49]
            0x897E_157E => words.push(0x89FE_307E), // s_xor_b64 exec, exec, s[48:49]
            0xBEFE_0317 => words.push(0xBEFE_042E), // s_mov_b64 exec, s[46:47]
            w => words.push(w),
        }
    }
    words
}

/// On a 32..32 host, a native wave64 group of 33-64 threads whose only
/// cross-lane operation outside the waterfall is a first-lane read of a
/// varying value asks for the console flow by that read alone; the console
/// flow refuses the descriptor joined per lane, and `emit` keeps the
/// per-lane form, declaring no wave size.
#[test]
fn a_first_lane_read_with_joined_descriptors_falls_back_to_the_per_lane_form() {
    let nvidia = super::Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    for threads in [48u32, 64] {
        let program = |first_lane| {
            let mut cs = native(&joined_wave64(first_lane), false);
            cs.wave64 = true;
            cs.io = StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [threads, 1, 1],
            });
            cs
        };
        assert!(lower_for(&program(false), nvidia, false).is_ok());
        let cs = program(true);
        for (console_cf, expected) in [
            (false, lower::CONSOLE_FLOW),
            (true, lower::LANE_PHI_IN_CONSOLE_FLOW),
        ] {
            match lower_for(&cs, nvidia, console_cf) {
                Err(crate::ShaderError::Unsupported { what, .. }) => assert_eq!(what, expected),
                Err(e) => panic!("{threads} threads: {e}"),
                Ok(_) => panic!("{threads} threads: expected {expected}"),
            }
        }
        let mut shader = super::tests::emit_for(&cs, nvidia).expect("per-lane retry");
        assert_eq!(super::split_tests::wave_size(&shader.dxil), None);
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil)
                .expect("DXIL validation failed");
        }
    }
}
