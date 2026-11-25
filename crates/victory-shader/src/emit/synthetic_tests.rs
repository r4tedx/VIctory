//! Synthetic GFX10 programs of what the 2020 compiler does not produce,
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU and WARP. Gathers at a level (`image_gather4_l`, `_l_o`), which
//! DXIL lacks: at level 0 the host's own gather (`image_gather4_lz`) of the
//! same coordinates is the reference on the hardware adapter (RDNA2, as the
//! console): it rounds the exact product of coordinate and size to 1/256 of
//! a texel, half up, which WARP does not. Lane permutes (`ds_bpermute_b32`).

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Per lane (32, wave32): the record [offset word, u, v, level] at entry 1
/// of the table at s0 (a 2D texture at entry 0, a raw UAV at entry 2); the
/// level converted from an integer (so it is whole). Gathers red at the
/// level, with the offset, and at level 0 (`lz`) into 12 dwords at lane × 48.
fn program(whole: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0502,
        0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        // The culling shaders' sampler: border, transparent black, levels
        // 0-12, linear filters.
        0xB008_01B6, // s_movk_i32 s8, 0x1b6
        0xBE89_03FF,
        0x00C0_0000, // s_mov_b32 s9, 0xc00000
        0xBE8A_03FF,
        0x1850_0000, // s_mov_b32 s10, 0x18500000
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x3420_0084, // v_lshlrev_b32 v16, 4, v0
        0x1628_00B0, // v_mul_u32_u24 v20, 48, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE038_1000,
        0x8005_0C10, // buffer_load_dwordx4 v[12:15], v16, s[20:23], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
    ];
    if whole {
        w.push(0x7E1E_0D0F); // v_cvt_f32_u32 v15, v15
    }
    w.extend([
        0xF110_0108,
        0x0043_040D, // image_gather4_l v[4:7], v[13:15], s[12:19], s[8:11]
        0xF150_0108,
        0x0043_080C, // image_gather4_l_o v[8:11], v[12:15], s[12:19], s[8:11]
        0xF11C_0108,
        0x0043_100D, // image_gather4_lz v[16:19], v[13:14], s[12:19], s[8:11]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_1000,
        0x800A_0414, // buffer_store_dwordx4 v[4:7], v20, s[40:43], 0 offen
        0xE078_1010,
        0x800A_0814, // buffer_store_dwordx4 v[8:11], v20, s[40:43], 0 offen offset:16
        0xE078_1020,
        0x800A_1014, // buffer_store_dwordx4 v[16:19], v20, s[40:43], 0 offen offset:32
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn build(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Srv, 8, 4, 16),
            range(ResourceType::Uav, 16, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let lowered = lowered(&native)?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| super::StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    })
}

fn lowered(native: &NativeProgram) -> Result<lower::Lowered, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
    })
}

/// The first texel of a footprint (see `Lower::gather_first`).
fn first(u: f32, size: u32) -> i64 {
    let u = if u.is_nan() { 0.0 } else { u.clamp(-2.0, 2.0) };
    ((f64::from(u) * f64::from(size) * 512.0).floor() as i64 - 255) >> 9
}

#[test]
fn gathers_at_a_level_match_the_hardware() {
    // 24×16 then 12×8, 6×4, 3×2; red = level × 1000 + y × 32 + x + 1.
    let mut texels = Vec::new();
    for l in 0..4 {
        let (w, h) = (24 >> l, 16 >> l);
        for y in 0..h {
            for x in 0..w {
                texels.push([(l * 1000 + y * 32 + x + 1) as f32, 0.0, 0.0, 1.0]);
            }
        }
    }
    let texel = |l: u32, x: i64, y: i64| -> f32 {
        let (w, h) = (24i64 >> l, 16i64 >> l);
        let skip: i64 = (0..l).map(|k| (24i64 >> k) * (16 >> k)).sum();
        if (0..w).contains(&x) && (0..h).contains(&y) {
            texels[(skip + y * w + x) as usize][0]
        } else {
            0.0
        }
    };
    // Records: lanes 0-15 at level 0 where rounding decides (coordinates
    // whose f32 product with 24 lands on the other side of a texel's
    // threshold than the exact one, ties of 1/512 on the 16 texels of y,
    // NaN, infinities, zeros); lanes 16-31 at other levels and offsets.
    let mut records: Vec<[u32; 4]> = Vec::new();
    let rounded = |u: f32| ((f64::from(u * 24.0) * 512.0).floor() as i64 - 255) >> 9;
    let mut straddling = (1..24).filter_map(|i| {
        let u0 = ((f64::from(i) + 127.5 / 256.0) / 24.0) as f32;
        (-1024i32..1024)
            .map(|d| f32::from_bits(u0.to_bits().wrapping_add_signed(d)))
            .find(|&u| rounded(u) != first(u, 24))
    });
    for k in 0..16 {
        let tie = (f64::from(k - 3) + 0.5 - 1.0 / 512.0) / 16.0;
        let u = match k {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => -0.0,
            _ => straddling.next().unwrap(),
        };
        records.push([0, u.to_bits(), (tie as f32).to_bits(), 0]);
    }
    for k in 0..16u32 {
        let level = [1, 2, 3, 7][k as usize % 4];
        let offset = [0x0000, 0x3F01, 0x0302, 0x3E3D][k as usize / 4];
        let (u, v) = (
            (k as f32 * 0.77 + 0.3) / 13.0 - 0.1,
            (k as f32 * 0.41 + 0.2) / 9.0,
        );
        records.push([offset, u.to_bits(), v.to_bits(), level]);
    }
    let input: Vec<u32> = records.iter().flatten().copied().collect();
    let shader = build(&program(true)).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Mipped(24, 16, 4, &texels),
        View::Buffer {
            data: &input,
            stride: Stride::Structured(16),
        },
        View::RawUav(32 * 12),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        ran += 1;
        for (lane, r) in records.iter().enumerate() {
            let got = |k: usize| -> Vec<f32> {
                out[0][lane * 12 + 4 * k..][..4]
                    .iter()
                    .map(|&b| f32::from_bits(b))
                    .collect()
            };
            let [offset, u, v, level] = *r;
            let level = level.min(3);
            let (w, h) = (24 >> level, 16 >> level);
            let at = |ox: i64, oy: i64| {
                let (x, y) = (
                    first(f32::from_bits(u), w) + ox,
                    first(f32::from_bits(v), h) + oy,
                );
                vec![
                    texel(level, x, y + 1),
                    texel(level, x + 1, y + 1),
                    texel(level, x + 1, y),
                    texel(level, x, y),
                ]
            };
            let sext = |b: u32| i64::from((b as i32) << 26 >> 26);
            assert_eq!(got(0), at(0, 0), "{label}: lane {lane}");
            assert_eq!(
                got(1),
                at(sext(offset), sext(offset >> 8)),
                "{label}: lane {lane}, offset"
            );
            // WARP's own gather truncates where the hardware rounds.
            if lane < 16 && !warp {
                assert_eq!(got(2), got(0), "{label}: lane {lane} against its gather");
            }
        }
    }
    assert!(ran > 0, "no device");
}

#[test]
fn gathers_at_a_level_that_may_be_a_fraction_stop() {
    let Err(crate::ShaderError::Unsupported { what, .. }) = build(&program(false)) else {
        panic!("translated a gather at a level read from memory");
    };
    assert_eq!(what, "a gather at a level that may not be whole");
}

#[test]
fn permutes_read_the_lane_named_and_zero_from_lanes_left_out() {
    // Lanes below 24 run the permute; each reads lane (13 × lane + 5) mod 32
    // (its address in dwords, offset 20 bytes) of 1000 + lane.
    let shader = build(&[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x1602_00B4, // v_mul_u32_u24 v1, 52, v0
        0x4A04_00FF,
        0x0000_03E8, // v_add_nc_u32 v2, 1000, v0
        0x7E06_02FF,
        0x0000_DEAD, // v_mov_b32 v3, 0xdead
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0xDACC_0014,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ])
    .expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let want: Vec<u32> = (0..32)
        .map(|l| match (13 * l + 5) % 32 {
            _ if l >= 24 => 0xDEAD,
            src if src < 24 => 1000 + src,
            _ => 0,
        })
        .collect();
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // The program declares the console's wave of 32, which WARP refuses.
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}

/// A lane mask joined with data at a uniform loop's header, as the title's
/// wave-uniform BVH walks push lane masks to memory and pop them
/// (`shadowmask:CS_TraceShadowRays_Checkerboard`): the register holds the
/// mask on entry and a number from the back edge, read both as a number and
/// as a mask.
#[test]
fn a_mask_joined_with_data_at_a_uniform_loop_is_its_number() {
    // s6 = lanes below 24, shifted right by 4 and passed through a VGPR in
    // each of two iterations; v4 takes each lane's bit of it on every pass.
    let shader = build(&[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x8706_7E6A, // s_and_b32 s6, vcc_lo, exec_lo
        0xBE87_0380, // s_mov_b32 s7, 0
        0xD501_0004,
        0x0019_0280, // header: v_cndmask_b32 v4, 0, 1, s6
        0x9006_8406, // s_lshr_b32 s6, s6, 4
        0x7E0A_0206, // v_mov_b32 v5, s6
        0x7E0C_0505, // v_readfirstlane_b32 s6, v5
        0x8007_8107, // s_add_u32 s7, s7, 1
        0xBF0A_8207, // s_cmp_lt_u32 s7, 2
        0xBF85_FFF8, // s_cbranch_scc1 header
        0x7E06_0206, // v_mov_b32 v3, s6
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xE070_1080,
        0x800A_0410, // buffer_store_dword v4, v16, s[40:43], 0 offen offset:128
        0xBF81_0000, // s_endpgm
    ])
    .expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(64),
    ];
    let mut want = vec![0x00FF_FFFF >> 8; 32];
    want.extend((0..32).map(|lane| u32::from(lane < 20)));
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}

/// A lane mask a compare makes inside a region the other lanes skip, joined
/// with data in the same register on the skip edge (a reused register): the
/// skipping lanes read 0, their bit of the compare, as on the console. A
/// mask holding lanes outside its edge (the saved EXEC) stays refused.
#[test]
fn a_compare_mask_joined_with_data_on_the_skip_edge_is_0_there() {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
        0xBF88_0002, // s_cbranch_execz +2
        0xD4C4_0006,
        0x0002_0088, // v_cmp_gt_u32 s6, 8, v0
        0xBEFE_0308, // s_mov_b32 exec_lo, s8
        0xD501_0003,
        0x0019_0280, // v_cndmask_b32 v3, 0, 1, s6
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let shader = build(&words).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let want: Vec<u32> = (0..32).map(|lane| u32::from(lane < 8)).collect();
    {
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            let Ok((root, pso)) = host.compute_pipeline(&shader) else {
                assert!(warp, "{label}: no pipeline for a wave of 32");
                eprintln!("{label}: no pipeline for a wave of 32; skipped");
                continue;
            };
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(out[0], want, "{label}");
        }
    }
    // The saved EXEC instead (every lane): the skipping lanes' bit would be
    // 1 on the console.
    words[10..12].copy_from_slice(&[0xBE86_0308, 0xBF80_0000]); // s_mov_b32 s6, s8; s_nop 0
    let Err(crate::ShaderError::Unsupported { what, .. }) = build(&words) else {
        panic!("a mask holding lanes off its edge joined with data translated");
    };
    assert!(what.contains("lane-mask join"), "{what}");
}

/// A register array (v3-v14) indexed through M0 in waterfall loops, as
/// `tiled_lighting:CS_PrescalarizeLightLoop` builds its per-tile light
/// masks: each lane's index (clamped to 0-11 by `v_med3_u32`) becomes M0 in
/// the loop pass that handles the lanes holding it; a `v_movrels` reads the
/// lane's array element, a `v_movreld` writes it back with the lane's bit
/// added. Twice, so the second pass reads what the first wrote.
#[test]
fn relative_moves_in_waterfall_loops_reach_each_lane_s_element() {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    ];
    w.extend((3..15u32).map(|k| 0x7E00_0280 | k << 17)); // v_mov_b32 vk, 0
    w.extend([
        0x2C38_0081, // v_lshrrev_b32 v28, 1, v0
        0xD559_001C,
        0x022D_011C, // v_med3_u32 v28, v28, 0, 11
        0x7E3A_0281, // v_mov_b32 v29, 1
        0x343A_3B00, // v_lshlrev_b32 v29, v0, v29
    ]);
    let pass = [
        0xBEA4_037E, // s_mov_b32 s36, exec_lo
        0x7E4C_051C, // read: v_readfirstlane_b32 s38, v28
        0xD4C2_0006,
        0x0002_3826, // v_cmp_eq_u32 s6, s38, v28
        0xBE86_3C06, // s_and_saveexec_b32 s6, s6
        0xBEFC_0326, // s_mov_b32 m0, s38
        0x7E3E_8703, // v_movrels_b32 v31, v3
        0x897E_067E, // s_xor_b32 exec_lo, exec_lo, s6
        0xBF89_FFF8, // s_cbranch_execnz read
        0xBEFE_0324, // s_mov_b32 exec_lo, s36
        0x383A_3B1F, // v_or_b32 v29, v31, v29
        0x7E4C_051C, // write: v_readfirstlane_b32 s38, v28
        0xD4C2_0006,
        0x0002_3826, // v_cmp_eq_u32 s6, s38, v28
        0xBE86_3C06, // s_and_saveexec_b32 s6, s6
        0xBEFC_0326, // s_mov_b32 m0, s38
        0x7E06_851D, // v_movreld_b32 v3, v29
        0x897E_067E, // s_xor_b32 exec_lo, exec_lo, s6
        0xBF89_FFF8, // s_cbranch_execnz write
        0xBEFE_0324, // s_mov_b32 exec_lo, s36
    ];
    w.extend(pass);
    w.extend([0x7E3A_02FF, 0x8000_0000]); // v_mov_b32 v29, 0x80000000
    w.extend(pass);
    for k in 3..15u32 {
        // buffer_store_dword vk, v16, s[40:43], 0 offen offset:128 * (k - 3)
        w.extend([0xE070_1000 | (128 * (k - 3)), 0x800A_0010 | k << 8]);
    }
    w.push(0xBF81_0000); // s_endpgm
    let shader = build(&w).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(12 * 32),
    ];
    let want: Vec<u32> = (0..12u32)
        .flat_map(|k| (0..32u32).map(move |lane| if (lane >> 1).min(11) == k { 1 << lane | 0x8000_0000 } else { 0 }))
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}

#[test]
fn ordered_pixel_sections_use_rasterizer_ordered_views() {
    // A pixel program's ordered section (POPS) as the title's compiler
    // writes it: the collision id after the primitive mask (s10, user data
    // ending at s9), a wait loop on the exiting wave id, a read-modify-write
    // of a UAV, the end of the section.
    let code: Vec<u32> = vec![
        0xBE84_0308, // s_mov_b32 s4, s8
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0x9000_9F0A, // s_lshr_b32 s0, s10, 31
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
        0x8102_81EF, // s_add_i32 s2, src_pops_exiting_wave_id, 1
        0xBF08_8502, // s_cmp_gt_u32 s2, 5
        0x8582_807E, // s_cselect_b64 s[2:3], exec, 0
        0x8880_0200, // s_or_b64 s[0:1], s[0:1], s[2:3]
        0x87EA_007E, // s_and_b64 vcc, exec, s[0:1]
        0xBF87_0004, // s_cbranch_vccnz +4
        0xBF8E_0001, // s_sleep 1
        0x8100_81EF, // s_add_i32 s0, src_pops_exiting_wave_id, 1
        0xBF0A_8500, // s_cmp_lt_u32 s0, 5
        0xBF84_FFFC, // s_cbranch_scc0 -4
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_3F88,
        0x0003_0200, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm dlc glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x0604_04F2, // v_add_f32 v2, 1.0, v2
        0xF020_1F08,
        0x0003_0200, // image_store v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_1890,
        0x0000_0000, // exp null off, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    let native = |db_shader_control| NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 8,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 8,
            slot: 0,
        }],
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control,
            patches: Vec::new(),
        },
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let l = lowered(&native(1 << 16)).expect("lower");
    assert!(l.rovs);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
    // Without the ordered mode the collision id is no value the host has.
    let Err(crate::ShaderError::Unsupported { what, .. }) = lowered(&native(0)) else {
        panic!("translated a collision id outside an ordered section");
    };
    assert_eq!(what, "a hardware SGPR input");
}

/// A program whose lanes read their row neighbours, in a console group of
/// `[x, y, z]` threads: each thread takes its flat id `x + X * (y + Y * z)`
/// from its thread id, reads lane `^ 15`'s (`v_mov_b32_dpp row_xmask:15`)
/// and stores it at its own flat id. On the console thread `f` stores
/// `f ^ 15` (its wave holds flat ids in order).
fn reads_its_row_neighbour(shape: [u32; 3], wave64: bool) -> NativeProgram {
    let [x, y, z] = shape;
    assert!(x <= 64, "an inline multiplier");
    let ids = if z > 1 { 2 } else { 1 };
    let mut code: Vec<u32> = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0000,                               // s_load_dwordx4 s[40:43], s[4:5], 0x0
        0x0B << 25 | 3 << 17 | 1 << 9 | (128 + x), // v_mul_u32_u24 v3, x, v1
        0x4A06_0700,                               // v_add_nc_u32 v3, v0, v3
    ];
    if z > 1 {
        code.extend([
            0x160A_04FF,
            x * y,       // v_mul_u32_u24 v5, x * y, v2
            0x4A06_0705, // v_add_nc_u32 v3, v5, v3
        ]);
    }
    code.extend([
        0x7E0E_02FA,
        0xFF01_6F03, // v_mov_b32_dpp v7, v3 row_xmask:15 row_mask:0xf bank_mask:0xf
        0x3408_0682, // v_lshlrev_b32 v4, 2, v3
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0704, // buffer_store_dword v7, v4, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        // One user SGPR; thread ids x and y, and z for 3-D groups.
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | ids << 11,
            threads: shape,
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The host group a lowered compute program declares (`PSV0`) and its
/// wave size, if any.
fn host_group(dxil: &[u8]) -> ([u32; 3], Option<u32>) {
    let c = crate::dxbc::Container::parse(dxil).expect("container");
    let d = c.part(crate::dxbc::FourCC::PSV0).expect("PSV0").data;
    let at = |o: usize| crate::dxbc::u32_at(d, 4 + o).expect("PSV0 runtime info");
    let wave = (at(16) == at(20) && at(16) != 0).then(|| at(16));
    ([at(36), at(40), at(44)], wave)
}

/// Shapes whose host waves an RTX 5070 Ti fills in 16 by 2 tiles (#2126's
/// measurements: 2-D groups 24, 32 or 64 wide, or not a power of two
/// wide), rows, and 3-D blocks.
const GROUPS: [[u32; 3]; 9] = [
    [8, 8, 1],
    [32, 2, 1],
    [32, 4, 1],
    [64, 2, 1],
    [24, 8, 1],
    [12, 16, 1],
    [16, 16, 1],
    [4, 4, 4],
    [8, 8, 2],
];

/// Threads relabelled so that their host lane is their console lane run in
/// a 1-D host group, whose waves every host fills with runs of flat ids in
/// order; the console's x, y and z come from the flat id. So each thread
/// gets its own id and reads its console neighbour's, whatever a host does
/// with 2-D groups.
#[test]
fn relabelled_threads_run_in_a_one_dimensional_host_group() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // WARP only: it refuses `WaveSize(32)`, so the waves of 32 only lower.
    let host = Host::new(true).ok();
    if host.is_none() {
        eprintln!("no SM 6.6 WARP device; lowering checked only");
    }
    let mut wrong = Vec::new();
    for wave64 in [false, true] {
        for shape in GROUPS {
            let n: u32 = shape.iter().product();
            let wave = if wave64 { 64 } else { 32 };
            if !n.is_multiple_of(wave) {
                continue;
            }
            let l = lowered(&reads_its_row_neighbour(shape, wave64)).expect("lower");
            assert_eq!(
                host_group(&l.dxil),
                ([n, 1, 1], Some(wave)),
                "{shape:?} wave{wave}: the host group"
            );
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut l.dxil.clone())
                    .unwrap_or_else(|e| panic!("{shape:?} wave{wave}: {e}"));
            }
            let Some(host) = &host else { continue };
            let cs = HostShader {
                stage: HwStage::Cs,
                dxil: l.dxil,
                user_data: l.user_data,
                samplers: l.samplers,
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                memory: l.memory,
                memory_writes: l.memory_writes,
                view_addresses: l.view_addresses,
                heap: l.heap,
                heap_writes: l.heap_writes,
                view_marks: l.view_marks,
            };
            let Ok((root, pso)) = host.compute_pipeline(&cs) else {
                eprintln!(
                    "{shape:?} wave{wave}: WARP refuses WaveSize({wave}); lowering checked only"
                );
                continue;
            };
            let ud = vec![0u32; cs.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(n)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{shape:?} wave{wave}: {e}"));
            let want: Vec<u32> = (0..n).map(|f| f ^ 15).collect();
            eprintln!("{shape:?} wave{wave}: ran on WARP");
            if out[0] != want {
                wrong.push(format!(
                    "{shape:?} wave{wave}: {:?}",
                    &out[0][..n.min(64) as usize]
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:?}");
}

/// The relabelling (`Lower::thread_id`) over host layouts: the flat ids each
/// host wave holds, in lane order, give each thread `first & !31` plus its
/// lane. In order (a 1-D group on every host measured) every id is taken
/// once; in 16 by 2 tiles (2-D groups 32 wide on an RTX 5070 Ti, #2126) two
/// waves take the same ids. Models the host for waves of 32, which WARP does
/// not run.
#[test]
fn the_relabelling_needs_waves_of_flat_ids_in_order() {
    let relabel = |waves: &[Vec<u32>]| {
        let mut ids: Vec<u32> = waves
            .iter()
            .flat_map(|w| {
                let base = w[0] & !31;
                (0..w.len() as u32).map(move |lane| base + lane)
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    };
    for [x, y] in [[32, 2], [32, 8], [64, 4], [16, 16]] {
        let n = x * y;
        let in_order: Vec<Vec<u32>> = (0..n).step_by(32).map(|b| (b..b + 32).collect()).collect();
        assert_eq!(relabel(&in_order), n as usize, "{x}x{y} in order");
        let mut tiles = Vec::new();
        for by in (0..y).step_by(2) {
            for bx in (0..x).step_by(16) {
                tiles.push(
                    (0..2)
                        .flat_map(|dy| (0..16).map(move |dx| (by + dy) * x + bx + dx))
                        .collect::<Vec<u32>>(),
                );
            }
        }
        // Tiles are rows only when a row is 16 wide.
        let taken = relabel(&tiles);
        if x == 16 {
            assert_eq!(taken, n as usize, "{x}x{y} in tiles");
        } else {
            assert_eq!(taken, n as usize / 2, "{x}x{y} in tiles");
        }
    }
}
