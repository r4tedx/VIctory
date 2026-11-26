//! Console wave64 compute programs on hosts with and without 64-lane waves:
//! hand-assembled GFX10 code, dispatched on every hardware adapter, checked
//! lane by lane against what the console's lane layout gives. On a host
//! without 64-lane waves, programs whose lanes read only their own 32-lane
//! half run as two host waves of 32 (see `lower::Lower::new`).

use super::host::{Dispatch, Host, Root, View, adapters};
use super::tests::{compute, emit_for};
use crate::native::{RootParam, RootType, StageIo};

const S_ENDPGM: u32 = 0xBF81_0000;

/// The wave size a DXIL container's `PSV0` requires (minimum = maximum).
fn wave_size(dxil: &[u8]) -> Option<u32> {
    let c = crate::Container::parse(dxil).ok()?;
    let psv = c.part(crate::FourCC::PSV0)?.data;
    let u = |at: usize| crate::dxbc::u32_at(psv, at).ok();
    let (min, max) = (u(4 + 16)?, u(4 + 20)?);
    (min == max).then_some(min)
}

/// A program over `threads` x 1 x 1 lanes per group with a UAV table in
/// user SGPR 0 and the group id in s1: lane t of group g stores the
/// result of `body` (v0 = t in, v2 out) at dword g * threads + t.
fn program(body: &[u32], threads: u32) -> crate::NativeProgram {
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0x9308_FF01,
        threads * 4, // s_mul_i32 s8, s1, threads * 4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x4A06_0608, // v_add_nc_u32 v3, s8, v3
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ]);
    let mut p = compute(&words, true);
    p.layout = vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    // One user SGPR, then the group id x.
    p.io = StageIo::Cs(crate::native::ComputeIo {
        rsrc2: 1 << 1 | 1 << 7,
        threads: [threads, 1, 1],
    });
    p
}

/// Swaps the 16-lane rows of each 32 (`ds_swizzle` xor 16), then mirrors
/// each row (DPP): lane t ends with t ^ 31, every read inside its half.
const IN_HALF: &[u32] = &[
    0xD8D4_401F,
    0x0100_0000, // ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM, and 0x1f, xor 0x10)
    0x7E04_02FA,
    0xFF01_4001, // v_mov_b32_dpp v2, v1 row_mirror row_mask:0xf bank_mask:0xf
];

fn run(host: &Host, native: &crate::NativeProgram, groups: u32) -> Result<Vec<u32>, String> {
    let target = host.target();
    let cs = emit_for(native, target).map_err(|e| e.to_string())?;
    // The validator, where one is installed: drivers take some modules it
    // refuses.
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil.clone())
            .map_err(|e| format!("validation: {e}"))?;
    }
    let (root, pso) = host.compute_pipeline(&cs)?;
    let data = vec![0; (cs.user_data_dwords() as usize).max(1)];
    let StageIo::Cs(io) = &native.io else {
        unreachable!()
    };
    let n = io.threads.iter().product::<u32>() * groups;
    let views = [View::RawUav(n)];
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(data))],
            views: &views,
            samplers: &[],
            // One host group per console wave where the shader says so.
            groups: [groups * cs.group_waves, 1, 1],
        },
    )?;
    Ok(out.into_iter().next().unwrap_or_default())
}

fn hardware_adapters() -> Vec<String> {
    if std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1") {
        eprintln!("VICTORY_WARP=1; skipping hardware adapters");
        return Vec::new();
    }
    adapters().unwrap_or_else(|e| {
        eprintln!("no DXGI; skipped: {e}");
        Vec::new()
    })
}

#[test]
fn in_half_lane_reads_match_the_console_on_every_adapter() {
    let mut ran = 0;
    for name in hardware_adapters() {
        let Ok(host) = Host::on(&name) else {
            eprintln!("{name}: no SM 6.6 device; skipped");
            continue;
        };
        let target = host.target();
        // One console wave, two per group, and a group whose second wave
        // has its lower half only.
        for threads in [64, 128, 32] {
            let p = program(IN_HALF, threads);
            let cs = emit_for(&p, target).unwrap_or_else(|e| panic!("{name}: {e}"));
            let split = !target.runs(64);
            assert_eq!(
                wave_size(&cs.dxil),
                Some(if split { 32 } else { 64 }),
                "{name}: the declared wave size"
            );
            let groups = 3;
            let out = run(&host, &p, groups).unwrap_or_else(|e| panic!("{name}: {e}"));
            let want: Vec<u32> = (0..groups)
                .flat_map(|_| (0..threads).map(|t| t ^ 31))
                .collect();
            assert_eq!(out, want, "{name}: {threads} threads (split {split})");
            eprintln!("{name}: {threads} threads per group exact (split {split})");
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no hardware adapter; skipped");
    }
}

/// 3-D groups: the console forms waves from the flat thread id (x
/// fastest). Lane f (f = x + X * (y + Y * z)) stores f ^ 31 at dword f.
fn program_3d(threads: [u32; 3]) -> crate::NativeProgram {
    program_3d_of(
        threads,
        &[
            0xD8D4_401F,
            0x0100_0004, // ds_swizzle_b32 v1, v4 offset:swizzle(BITMASK_PERM, and 0x1f, xor 0x10)
            0x7E04_02FA,
            0xFF01_4001, // v_mov_b32_dpp v2, v1 row_mirror
        ],
    )
}

/// [`program_3d`] with `core` between the flat id (in v4) and the store of
/// v2 at dword f.
fn program_3d_of(threads: [u32; 3], core: &[u32]) -> crate::NativeProgram {
    let [x, y, _] = threads;
    let (sx, sxy) = (x.trailing_zeros(), (x * y).trailing_zeros());
    assert!(x.is_power_of_two() && y.is_power_of_two());
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000,               // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F,               // s_waitcnt lgkmcnt(0)
        0x340A_0200 | (128 + sx),  // v_lshlrev_b32 v5, log2 X, v1
        0x340C_0400 | (128 + sxy), // v_lshlrev_b32 v6, log2 XY, v2
        0x4A08_0B00,               // v_add_nc_u32 v4, v0, v5
        0x4A08_0D04,               // v_add_nc_u32 v4, v4, v6
    ];
    words.extend_from_slice(core);
    words.extend([
        0x3406_0882, // v_lshlrev_b32 v3, 2, v4
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ]);
    let mut p = compute(&words, true);
    p.layout = program(&[], 64).layout;
    // One user SGPR; thread ids x, y and z in v0..v2.
    p.io = StageIo::Cs(crate::native::ComputeIo {
        rsrc2: 1 << 1 | 2 << 11,
        threads,
    });
    p
}

#[test]
fn three_dimensional_groups_keep_the_consoles_lanes() {
    for name in hardware_adapters() {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        let split = !host.target().runs(64);
        for threads in [
            [4, 4, 4],
            [8, 8, 2],
            [2, 4, 8],
            [16, 2, 4],
            [8, 8, 4],
            [4, 8, 8],
        ] {
            let p = program_3d(threads);
            let n: u32 = threads.iter().product();
            let out = run(&host, &p, 1).unwrap_or_else(|e| panic!("{name}: {e}"));
            let want: Vec<u32> = (0..n).map(|f| f ^ 31).collect();
            assert_eq!(out, want, "{name}: {threads:?} (split {split})");
            eprintln!("{name}: {threads:?} exact (split {split})");
        }
    }
}

/// Runs `body` (v0 = t in, v2 out) on every hardware adapter in groups of
/// `threads` and checks lane t against `want(t)`; on a host without
/// 64-lane waves it must have translated in the console flow.
fn console_flow_case(label: &str, body: &[u32], threads: u32, want: impl Fn(u32) -> u32) {
    flow_case(label, body, threads, want, true);
}

/// [`console_flow_case`], checking hosts with 64-lane waves (the per-lane
/// form) only when `native` says.
fn flow_case(label: &str, body: &[u32], threads: u32, want: impl Fn(u32) -> u32, native: bool) {
    program_case(label, &program(body, threads), want, native);
}

/// [`flow_case`] of a program built by the caller.
fn program_case(label: &str, p: &crate::NativeProgram, want: impl Fn(u32) -> u32, native: bool) {
    let StageIo::Cs(io) = &p.io else {
        unreachable!()
    };
    let threads = io.threads[0];
    for name in hardware_adapters() {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        let split = !host.target().runs(64);
        let groups = 2;
        let out = run(&host, p, groups).unwrap_or_else(|e| panic!("{name}: {label}: {e}"));
        let expect: Vec<u32> = (0..groups).flat_map(|_| (0..threads).map(&want)).collect();
        if !split && !native {
            if out != expect {
                eprintln!("{name}: {label}: the per-lane form differs (known)");
            }
            continue;
        }
        assert_eq!(
            out, expect,
            "{name}: {label}, {threads} threads (split {split})"
        );
        eprintln!("{name}: {label}, {threads} threads exact (split {split})");
    }
}

/// A mask counted whole: 40 lanes of the console wave, 32 in one half and
/// 8 in the other.
#[test]
fn masks_read_whole_count_both_halves() {
    let body = [
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("s_bcnt1 of lanes below 40", &body, 64, |_| 40);
    // 48 threads: the upper half is partial.
    console_flow_case("s_bcnt1 of lanes below 40", &body, 48, |_| 40);
    // 32 threads: no upper half at all.
    console_flow_case("s_bcnt1 of lanes below 40", &body, 32, |_| 32);
}

/// A lane of the other half: lane 40 is the upper half's lane 8.
#[test]
fn lane_reads_cross_halves() {
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("v_readlane 40", &body, 64, |_| 40);
}

/// A branch only lanes below 20 take (all in the lower half), with a lane
/// read inside: the upper half has no lane there but still meets the lower
/// one at the exchange.
#[test]
fn exchanges_inside_a_branch_one_half_skips() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0003, // s_cbranch_execz +3
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    console_flow_case("if (lane < 20) v2 = readlane(10)", &body, 64, |t| {
        if t < 20 { 10 } else { 1 }
    });
}

/// The console flow is for one console wave per host group, or one
/// control flow for a whole group. A group of several console waves that
/// share no LDS is emitted as that many host groups of 64 threads; one
/// whose waves share LDS or meet at an `s_barrier` as one host group of
/// 32-lane waves (`lower::groups`); one whose size is no multiple of 64
/// still stops.
#[test]
fn console_flow_is_for_one_console_wave_per_group() {
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let nvidia = super::Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    let amd = super::Target {
        lanes_min: 32,
        lanes_max: 64,
    };
    assert_eq!(
        emit_for(&program(&body, 64), nvidia).unwrap().group_waves,
        1
    );
    let two = emit_for(&program(&body, 128), nvidia).unwrap();
    assert_eq!(two.group_waves, 2);
    assert_eq!(wave_size(&two.dxil), Some(32));
    assert_eq!(
        emit_for(&program(&body, 1024), nvidia).unwrap().group_waves,
        16
    );
    // A host with 64-lane waves keeps the console's groups.
    assert_eq!(emit_for(&program(&body, 128), amd).unwrap().group_waves, 1);
    let stops = |p: &crate::NativeProgram| {
        let e = emit_for(p, nvidia).expect_err("stops").to_string();
        assert!(
            e.contains("in a group of more than 64 threads"),
            "the reason: {e}"
        );
    };
    stops(&program(&body, 96));
    let mut with_barrier = body.to_vec();
    with_barrier.push(0xBF8A_0000); // s_barrier
    for p in [program_lds(&body, 128), program(&with_barrier, 128)] {
        let one = emit_for(&p, nvidia).unwrap();
        assert_eq!(one.group_waves, 1);
        assert_eq!(wave_size(&one.dxil), Some(32));
    }
}

/// Groups of two and four console waves as host groups of one wave each,
/// two console groups per dispatch. v0 is the thread id in the console's
/// group and s1 its group id, which the store address is built from: lane
/// 40 of each wave holds 64 * wave + 40, and a mask of the lanes whose id
/// in the wave is below 40 counts 40 in every wave.
#[test]
fn groups_of_several_console_waves_run_as_one_host_group_each() {
    let read = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let count = [
        0x3602_00BF, // v_and_b32 v1, 63, v0
        0x7D88_02A8, // v_cmp_gt_u32 vcc, 40, v1
        0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    for threads in [128, 256] {
        console_flow_case("v_readlane 40 in each wave", &read, threads, |t| {
            (t & !63) + 40
        });
        console_flow_case("s_bcnt1 of 40 lanes in each wave", &count, threads, |_| 40);
    }
    // 2-D and 3-D groups: the console forms its waves from the flat thread
    // id, which the three ids give (v4); lane 40 of each wave holds
    // 64 * wave + 40.
    for name in hardware_adapters() {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        for threads in [[16, 16, 1], [8, 16, 1], [8, 8, 2], [4, 4, 8], [32, 32, 1]] {
            let read = [
                0xD760_0009,
                0x0001_5104, // v_readlane_b32 s9, v4, 40
                0x7E04_0209, // v_mov_b32 v2, s9
            ];
            let p = program_3d_of(threads, &read);
            let n: u32 = threads.iter().product();
            let out = run(&host, &p, 1).unwrap_or_else(|e| panic!("{name}: {threads:?}: {e}"));
            let want: Vec<u32> = (0..n).map(|f| (f & !63) + 40).collect();
            assert_eq!(out, want, "{name}: {threads:?}");
            eprintln!("{name}: v_readlane 40 in each wave, {threads:?} exact");
        }
    }
    // A whole-wave section in each wave: the sum of its thread ids.
    let mut sum = reduction_scan(37, 0x80);
    sum.push(0x7E04_020A); // v_mov_b32 v2, s10
    console_flow_case("a whole-wave sum in each wave", &sum, 128, |t| {
        2016 + 64 * 64 * (t / 64)
    });
}

/// A loop lane t runs t times, with a lane read of the other half in
/// every iteration: the lower half's lanes are done after 31 iterations,
/// the upper half's go on to 63, and the console wave loops while any lane
/// does. After it, s9 holds lane 40's id, as the last iteration read it.
///
/// The per-lane form got this wrong on every host (#225); hosts with
/// 64-lane waves now run it in the console flow too, without exchanges.
#[test]
fn loops_run_while_any_lane_of_the_console_wave_does() {
    let body = [
        0xBE89_0380, // s_mov_b32 s9, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0006, // s_cbranch_execz end
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFFA, // s_cbranch_execnz loop
        // end:
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ];
    console_flow_case("loop t times reading lane 40", &body, 64, |t| t + 40);
}

/// DPP's row mask names the console's rows: with `row_mask:0x3` only
/// rows 0 and 1 (lanes 0 to 31) write, so the upper half keeps its value
/// although its host rows are 0 and 1 too.
#[test]
fn dpp_row_masks_name_console_rows() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7E04_02FA,
        0x3F00_E400, // v_mov_b32_dpp v2, v0 quad_perm:[0,1,2,3] row_mask:0x3 bank_mask:0xf
    ];
    console_flow_case(
        "DPP row_mask 0x3",
        &body,
        64,
        |t| if t < 32 { t } else { 1 },
    );
}

/// `v_readfirstlane` reads the console wave's first lane: lane 0 here, in
/// the upper half too. With DPP in the program it used to be split in
/// halves, where the upper half read its own first lane (32).
#[test]
fn first_lane_reads_are_the_console_waves() {
    let body = [
        0x7E06_02FA,
        0xFF00_E400, // v_mov_b32_dpp v3, v0 quad_perm:[0,1,2,3]
        0x7E12_0500, // v_readfirstlane_b32 s9, v0
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("DPP and v_readfirstlane", &body, 64, |_| 0);
}

/// `v_writelane` ignores EXEC: inside a branch only lanes below 20 take,
/// it still writes lane 40. The per-lane form never had lane 40 there.
#[test]
fn lane_writes_reach_lanes_outside_the_branch() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0004, // s_cbranch_execz +4
        0xBE89_03FF,
        0x0000_004D, // s_mov_b32 s9, 77
        0xD761_0002,
        0x0001_5009, // v_writelane_b32 v2, s9, 40
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    console_flow_case("if (lane < 20) writelane(40, 77)", &body, 64, |t| {
        if t == 40 { 77 } else { 1 }
    });
}

/// The loop form compilers emit: lanes that are done are added to an exit
/// mask that starts as 0, EXEC loses them, and the mask restores EXEC
/// after the loop. Lane t runs t + 1 iterations with a read of lane 40 in
/// each; afterwards every lane has s9 = 40 and counts all 64 lanes in the
/// exit mask.
#[test]
fn loops_with_an_exit_mask_grown_from_zero() {
    let body = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0480, // s_mov_b64 s[10:11], 0
        0xBE89_0380, // s_mov_b32 s9, 0
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
        0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
        0xBF89_FFF9, // s_cbranch_execnz loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBE8C_100A, // s_bcnt1_i32_b64 s12, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
        0x4A04_040C, // v_add_nc_u32 v2, s12, v2
    ];
    console_flow_case("exit-mask loop reading lane 40", &body, 64, |_| 40 + 64);
}

/// [`program`] with 512 bytes of LDS.
fn program_lds(body: &[u32], threads: u32) -> crate::NativeProgram {
    let mut p = program(body, threads);
    let StageIo::Cs(io) = &mut p.io else {
        unreachable!()
    };
    io.rsrc2 |= 1 << 15;
    p
}

/// One console wave sharing LDS without `s_barrier`: the console runs each
/// instruction in all 64 lanes before the next, so a lane reads what a lane
/// of the other half wrote one instruction earlier. Lane t writes t to
/// dword t, then reads dword 63 - t; then writes that plus 50 to dword t
/// and reads dword t ^ 32.
#[test]
fn one_wave_shares_lds_in_instruction_order() {
    let body = [
        0x4C0A_00BF, // v_sub_nc_u32 v5, 63, v0
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x340C_0A82, // v_lshlrev_b32 v6, 2, v5
        0xD834_0000,
        0x0000_0004, // ds_write_b32 v4, v0
        0xD8D8_0000,
        0x0700_0006, // ds_read_b32 v7, v6
        0x4A0E_0EB2, // v_add_nc_u32 v7, 50, v7
        0xD834_0000,
        0x0000_0704, // ds_write_b32 v4, v7
        0x3A10_00A0, // v_xor_b32 v8, 32, v0
        0x3410_1082, // v_lshlrev_b32 v8, 2, v8
        0xD8D8_0000,
        0x0200_0008, // ds_read_b32 v2, v8
    ];
    // Dword u holds (63 - u) + 50 after the second write.
    program_case(
        "LDS in instruction order",
        &program_lds(&body, 64),
        |t| (63 - (t ^ 32)) + 50,
        true,
    );
}

/// The same sharing with the halves pulled apart: only the lower half does
/// 48 buffer stores before its LDS write, so without a barrier the upper
/// half's read of the lower half's dword can come first. Lane t writes t to
/// dword t and reads dword 63 - t.
///
/// On an RTX 5070 this test and the one above also pass with the barrier
/// before each LDS access left out: that card keeps the two waves in step
/// for these programs. They check that the form gives the console's result,
/// not that the barrier is what gives it; D3D12 promises no order between a
/// group's waves without one. `lds_written_and_read_in_different_masked_blocks`
/// has a case that does fail without it.
#[test]
fn lds_order_holds_when_one_half_is_busy() {
    let mut body = vec![
        0x3412_0082, // v_lshlrev_b32 v9, 2, v0
        0x4A12_1208, // v_add_nc_u32 v9, s8, v9
        0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    ];
    for _ in 0..48 {
        // buffer_store_dword v0, v9, s[4:7], 0 offen
        body.extend_from_slice(&[0xE070_1000, 0x8001_0009]);
    }
    body.extend_from_slice(&[
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4C0A_00BF, // v_sub_nc_u32 v5, 63, v0
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x340C_0A82, // v_lshlrev_b32 v6, 2, v5
        0xD834_0000,
        0x0000_0004, // ds_write_b32 v4, v0
        0xD8D8_0000,
        0x0200_0006, // ds_read_b32 v2, v6
    ]);
    program_case(
        "LDS order with a busy lower half",
        &program_lds(&body, 64),
        |t| 63 - t,
        true,
    );
}

/// The diagnostic summary of the module emitted last: what it declares and
/// the operations it calls, nothing of the console program.
#[test]
fn the_last_modules_summary_names_its_group_wave_size_and_operations() {
    let nvidia = super::Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let mut p = program(&body, 64);
    let StageIo::Cs(io) = &mut p.io else {
        unreachable!()
    };
    io.threads = [8, 8, 1];
    emit_for(&p, nvidia).unwrap();
    let summary = super::last_module_summary();
    eprintln!("{summary}");
    assert!(
        summary.starts_with("threads 8x8x1, wave size 32; ops: "),
        "{summary}"
    );
    for op in ["barrier", "waveReadLaneAt", "flattenedThreadIdInGroup"] {
        assert!(summary.contains(op), "{op} in {summary}");
    }
}

/// Sum lane IDs within rows, add the preceding row, then carry lane 31's
/// inclusive result into the upper half. Lane 32 must start at 528.
#[test]
fn wave64_inclusive_prefix_joins_both_host_waves() {
    let body = [
        0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
        0x4A02_02FA,
        0xFF01_1101, // v_add_nc_u32 v1, v1, v1 row_shr:1
        0x4A02_02FA,
        0xFF01_1201, // row_shr:2
        0x4A02_02FA,
        0xFF01_1401, // row_shr:4
        0x4A02_02FA,
        0xFF01_1801, // row_shr:8
        0xD778_0002,
        0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E06_02FA,
        0xAF00_E402, // v_mov_b32 v3, v2, row_mask:0xa
        0x4A02_0701, // v_add_nc_u32 v1, v1, v3
        0xD760_0009,
        0x0001_3F01, // v_readlane_b32 s9, v1, 31
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBEFF_03C1, // s_mov_b32 exec_hi, -1
        0x4A02_0209, // v_add_nc_u32 v1, s9, v1
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
        0x7E04_0301, // v_mov_b32 v2, v1
    ];
    console_flow_case("whole-wave inclusive sum", &body, 64, |n| n * (n + 1) / 2);
}

/// A scan of the lane ids with `opcode` (a VOP2 operation) as its step and
/// `ident` (an inline constant operand) in the lanes outside the mask, as
/// in the test above, then a read of lane 63: the reduction over the whole
/// console wave, left in s10. It uses v1 to v3 as scratch in every lane.
fn reduction_scan(opcode: u32, ident: u32) -> Vec<u32> {
    reduction_scan_of(opcode, ident, None)
}

/// [`reduction_scan`] with the identity as a literal when `literal` gives
/// one (`ident` is then the literal operand, 0xFF).
fn reduction_scan_of(opcode: u32, ident: u32, literal: Option<u32>) -> Vec<u32> {
    let op = opcode << 25;
    let mut words = vec![
        0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0000 | ident, // v_cndmask_b32 v1, ident, v0, s[0:1]
    ];
    words.extend(literal);
    words.extend([
        op | 0x0002_02FA,
        0xFF01_1101, // v_<op> v1, v1, v1 row_shr:1
        op | 0x0002_02FA,
        0xFF01_1201, // row_shr:2
        op | 0x0002_02FA,
        0xFF01_1401, // row_shr:4
        op | 0x0002_02FA,
        0xFF01_1801, // row_shr:8
        0xD778_0002,
        0x0305_8301,         // v_permlanex16_b32 v2, v1, -1, -1
        0x7E06_0200 | ident, // v_mov_b32 v3, ident
    ]);
    words.extend(literal);
    words.extend([
        0x7E06_02FA,
        0xAF00_E402,      // v_mov_b32 v3, v2 row_mask:0xa
        op | 0x0002_0701, // v_<op> v1, v1, v3
        0xD760_0009,
        0x0001_3F01,      // v_readlane_b32 s9, v1, 31
        0xBEFE_0380,      // s_mov_b32 exec_lo, 0
        0xBEFF_03C1,      // s_mov_b32 exec_hi, -1
        op | 0x0002_0209, // v_<op> v1, s9, v1
        0xD760_000A,
        0x0001_7F01, // v_readlane_b32 s10, v1, 63
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
    ]);
    words
}

/// Reductions over the whole console wave need both halves: the sum of the
/// lane ids, their maximum (lane 63, in the upper half), their minimum
/// (lane 0, in the lower half) and their bitwise or.
#[test]
fn whole_wave_reductions_join_both_host_waves() {
    // (operation, VOP2 opcode, identity operand, the console's result)
    for (label, opcode, ident, want) in [
        ("v_add_nc_u32", 37, 0x80, 2016),
        ("v_max_u32", 20, 0x80, 63),
        ("v_min_u32", 19, 0xC1, 0),
        ("v_or_b32", 28, 0x80, 63),
    ] {
        let mut body = reduction_scan(opcode, ident);
        body.push(0x7E04_020A); // v_mov_b32 v2, s10
        console_flow_case(&format!("whole-wave {label}"), &body, 64, |_| want);
    }
}

/// The same sum over a mask that straddles the halves: lanes below 40 (32
/// in the lower half, 8 in the upper). The other lanes keep their value.
#[test]
fn masked_whole_wave_sum_straddles_the_halves() {
    // The result goes through v5: the scan leaves no old value in v2.
    let mut body = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    ];
    body.extend(reduction_scan(37, 0x80));
    body.extend([
        0x7E0A_020A, // v_mov_b32 v5, s10
        0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
        0x7E04_0305, // v_mov_b32 v2, v5
    ]);
    // 0 + 1 + ... + 39 = 780.
    console_flow_case("masked whole-wave sum", &body, 64, |t| {
        if t < 40 { 780 } else { 1 }
    });
}

/// A whole-wave section crosses the halves in a group of exactly one
/// console wave, and needs no crossing in a group of 32 threads or fewer
/// (the lower half alone): with a partial upper half it still stops.
#[test]
fn whole_wave_sections_in_partial_groups_still_stop() {
    let nvidia = super::Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    let mut body = reduction_scan(37, 0x80);
    body.push(0x7E04_020A); // v_mov_b32 v2, s10
    for threads in [64, 32, 16] {
        assert!(
            emit_for(&program(&body, threads), nvidia).is_ok(),
            "{threads} threads"
        );
    }
    for threads in [48, 33] {
        assert_eq!(
            emit_for(&program(&body, threads), nvidia),
            Err(crate::ShaderError::Unsupported {
                offset: 0,
                what: "a whole-wave section of a wave64 on a host without 64-lane waves",
            }),
            "{threads} threads"
        );
    }
}

/// A wave64 group of 32 threads or fewer is the console wave's lower half
/// alone: its upper lanes are in no mask, so a whole-wave reduction is the
/// one host wave's, with nothing exchanged. (With the halves combined as in
/// a full wave, a minimum or an AND would take the absent half's 0.)
#[test]
fn whole_wave_reductions_in_a_group_of_one_half() {
    // (operation, VOP2 opcode, identity operand, the console's result per
    // thread count)
    for (label, opcode, ident, want) in [
        ("v_add_nc_u32", 37, 0x80, [496, 120]),
        ("v_max_u32", 20, 0x80, [31, 15]),
        ("v_min_u32", 19, 0xC1, [0, 0]),
        ("v_or_b32", 28, 0x80, [31, 15]),
    ] {
        for (threads, want) in [32, 16].into_iter().zip(want) {
            let mut body = reduction_scan(opcode, ident);
            body.push(0x7E04_020A); // v_mov_b32 v2, s10
            console_flow_case(&format!("whole-wave {label}"), &body, threads, |_| want);
        }
    }
}

/// One console wave whose LDS write and read sit in different exec-masked
/// blocks (the pattern measured on upstream PR #189): lanes write dword t,
/// then lanes 0..31 read dword t + 32 in a later masked block. The console
/// runs the wave in lockstep, so the read sees the write.
///
/// `delay` has only lanes 32..63 write, after 4096 iterations of a
/// scalar-counted loop in their own masked block, so a lower half that was
/// not held back would read first.
///
/// In the console flow both halves are in step here twice over: a branch on
/// EXEC is decided by both halves through an exchange, which is a group
/// barrier, and each LDS access has a group barrier before it. So these two
/// also pass on an RTX 5070 with the second left out; the case without
/// branches in the test below does not.
fn lds_across_masked_blocks(delay: bool) -> Vec<u32> {
    let mut body = vec![
        0x7E04_0287, // v_mov_b32 v2, 7
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x4A0A_0087, // v_add_nc_u32 v5, 7, v0
    ];
    if delay {
        body.extend_from_slice(&[
            0x7D82_009F, // v_cmp_lt_u32 vcc, 31, v0
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
            0xBF88_000A, // s_cbranch_execz join
            0xBE89_0380, // s_mov_b32 s9, 0
            // loop:
            0x160A_0AFF,
            40503,       // v_mul_u32_u24 v5, 40503, v5
            0x4A0A_0A81, // v_add_nc_u32 v5, 1, v5
            0x8109_8109, // s_add_i32 s9, s9, 1
            0xBF0A_FF09,
            4096,        // s_cmp_lt_u32 s9, 4096
            0xBF85_FFF9, // s_cbranch_scc1 loop
            0xD834_0000,
            0x0000_0504, // ds_write_b32 v4, v5
            // join:
            0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        ]);
    } else {
        body.extend_from_slice(&[
            0xD834_0000,
            0x0000_0504, // ds_write_b32 v4, v5
        ]);
    }
    body.extend_from_slice(&[
        0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0002, // s_cbranch_execz join
        0xD8D8_0080,
        0x0200_0004, // ds_read_b32 v2, v4 offset:128
        // join:
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ]);
    body
}

#[test]
fn lds_written_and_read_in_different_masked_blocks() {
    program_case(
        "LDS written by every lane, then read by the lower half",
        &program_lds(&lds_across_masked_blocks(false), 64),
        |t| if t < 32 { t + 39 } else { 7 },
        true,
    );
    // What a writer leaves: 4096 times v = v_mul_u32_u24(40503, v) + 1.
    let late = |t: u32| {
        (0..4096).fold(t + 7, |v, _| {
            (v & 0xFF_FFFF).wrapping_mul(40503).wrapping_add(1)
        })
    };
    program_case(
        "LDS written late by the upper half, then read by the lower",
        &program_lds(&lds_across_masked_blocks(true), 64),
        |t| if t < 32 { late(t + 32) } else { 7 },
        true,
    );
    // The same masks with no branch around either access: nothing but the
    // barrier before each LDS access orders the halves. With it left out,
    // the lower half of an RTX 5070 reads 0 here every time (20 of 20 runs):
    // this is the case that shows the barrier is needed.
    let masked = [
        0x7E04_0287, // v_mov_b32 v2, 7
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x4A0A_0087, // v_add_nc_u32 v5, 7, v0
        0x7D82_009F, // v_cmp_lt_u32 vcc, 31, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xD834_0000,
        0x0000_0504, // ds_write_b32 v4, v5
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xD8D8_0080,
        0x0200_0004, // ds_read_b32 v2, v4 offset:128
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    program_case(
        "LDS written by the upper half and read by the lower, no branches",
        &program_lds(&masked, 64),
        |t| if t < 32 { t + 39 } else { 7 },
        true,
    );
}

#[test]
fn backward_permute_wraps_within_each_half() {
    let body = [0x3406_0082, 0xDACC_0004, 0x0200_0003];
    console_flow_case("backward permute", &body, 64, |lane| {
        (lane & !31) | ((lane + 1) & 31)
    });
}

#[test]
fn backward_permute_respects_both_exec_masks() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xDACC_0004,
        0x0200_0003, // ds_bpermute_b32 v2, v3, v0 offset:4
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    console_flow_case("masked backward permute", &body, 64, |lane| match lane {
        0..19 => lane + 1,
        19 => 0,
        _ => 1,
    });
}

/// `ds_bpermute_b32` where lane t reads v0 of lane t + 1 of its half, in the
/// masks, group sizes and forms the two tests above leave out.
#[test]
fn backward_permute_in_other_masks_groups_and_forms() {
    let next = |t: u32| (t & !31) | ((t + 1) & 31);
    // EXEC over part of each half: lanes 0..21 and 42..63.
    let straddle = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x4A02_0096, // v_add_nc_u32 v1, 22, v0
        0x3602_02BF, // v_and_b32 v1, 63, v1
        0x7D88_02AC, // v_cmp_gt_u32 vcc, 44, v1
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xDACC_0004,
        0x0200_0003, // ds_bpermute_b32 v2, v3, v0 offset:4
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    let active = |t: u32| !(22..42).contains(&t);
    console_flow_case(
        "backward permute under a mask over both halves",
        &straddle,
        64,
        |t| match (active(t), active(next(t))) {
            (false, _) => 1,
            (true, true) => next(t),
            (true, false) => 0,
        },
    );
    let plain = [
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xDACC_0004,
        0x0200_0003, // ds_bpermute_b32 v2, v3, v0 offset:4
    ];
    // Two console waves per group, then a group whose upper half has 16
    // lanes (a source lane that is not there reads 0), then one half only.
    console_flow_case("backward permute, two console waves", &plain, 128, next);
    console_flow_case("backward permute, 48 threads", &plain, 48, |t| {
        if next(t) < 48 { next(t) } else { 0 }
    });
    console_flow_case("backward permute, 32 threads", &plain, 32, next);
    // Beside a lane read across the halves, which takes the console flow on
    // a host without 64-lane waves.
    let with_read = [
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xDACC_0004,
        0x0200_0003, // ds_bpermute_b32 v2, v3, v0 offset:4
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ];
    console_flow_case("backward permute beside a lane read", &with_read, 64, |t| {
        next(t) + 40
    });
}

/// Float reductions over the whole console wave: the lane ids as floats,
/// their sum, their maximum (lane 63, in the upper half) and their minimum
/// (lane 0, in the lower half), with the identities the operations need
/// (0.0, -inf, +inf). Each half's result crosses as its bits.
#[test]
fn float_whole_wave_reductions_join_both_host_waves() {
    // (operation, VOP2 opcode, identity operand, its literal, the result)
    for (label, opcode, ident, literal, want) in [
        ("v_add_f32", 3, 0x80, None, 2016),
        ("v_max_f32", 16, 0xFF, Some(0xFF80_0000), 63),
        ("v_min_f32", 15, 0xFF, Some(0x7F80_0000), 0),
    ] {
        let mut body = vec![0x7E00_0D00]; // v_cvt_f32_u32 v0, v0
        body.extend(reduction_scan_of(opcode, ident, literal));
        body.extend([
            0x7E00_0F00, // v_cvt_u32_f32 v0, v0
            0x7E04_020A, // v_mov_b32 v2, s10
            0x7E04_0F02, // v_cvt_u32_f32 v2, v2
        ]);
        console_flow_case(&format!("whole-wave {label}"), &body, 64, |_| want);
    }
}

/// A float sum over a mask that straddles the halves (lanes below 40), and
/// the inclusive float prefix sum of the lane ids, whose upper half starts
/// from the lower half's total.
#[test]
fn masked_and_prefix_float_sums_cross_the_halves() {
    let mut body = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0x7E00_0D00, // v_cvt_f32_u32 v0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    ];
    body.extend(reduction_scan(3, 0x80));
    body.extend([
        0x7E0A_020A, // v_mov_b32 v5, s10
        0x7E0A_0F05, // v_cvt_u32_f32 v5, v5
        0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
        0x7E00_0F00, // v_cvt_u32_f32 v0, v0
        0x7E04_0305, // v_mov_b32 v2, v5
    ]);
    console_flow_case("masked whole-wave float sum", &body, 64, |t| {
        if t < 40 { 780 } else { 1 }
    });
    let prefix = [
        0x7E00_0D00, // v_cvt_f32_u32 v0, v0
        0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
        0x0602_02FA,
        0xFF01_1101, // v_add_f32 v1, v1, v1 row_shr:1
        0x0602_02FA,
        0xFF01_1201, // row_shr:2
        0x0602_02FA,
        0xFF01_1401, // row_shr:4
        0x0602_02FA,
        0xFF01_1801, // row_shr:8
        0xD778_0002,
        0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E06_02FA,
        0xAF00_E402, // v_mov_b32 v3, v2 row_mask:0xa
        0x0602_0701, // v_add_f32 v1, v1, v3
        0xD760_0009,
        0x0001_3F01, // v_readlane_b32 s9, v1, 31
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBEFF_03C1, // s_mov_b32 exec_hi, -1
        0x0602_0209, // v_add_f32 v1, s9, v1
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
        0x7E04_0F01, // v_cvt_u32_f32 v2, v1
        0x7E00_0F00, // v_cvt_u32_f32 v0, v0
    ];
    console_flow_case("whole-wave inclusive float sum", &prefix, 64, |n| {
        n * (n + 1) / 2
    });
}

/// A select whose selector is a number, not a compare result: a constant
/// in each mask word, and in the second program a word with a bit that
/// depends on the group (s8 is 256 in group 1). The lane takes its own
/// bit of the console's mask: lanes 32..63 of a wave64 the high word's,
/// each lane of a wave32 the one word's.
#[test]
fn selects_on_masks_held_as_numbers() {
    let constant = [
        0xBE8A_03FF,
        0x0F0F_0F0F, // s_mov_b32 s10, 0x0f0f0f0f
        0xBE8B_03FF,
        0x3333_3333, // s_mov_b32 s11, 0x33333333
        0x7E02_0287, // v_mov_b32 v1, 7
        0xD501_0002,
        0x002A_0101, // v_cndmask_b32 v2, v1, v0, s[10:11]
    ];
    let built = [
        0xBE8A_03FF,
        0x0F0F_0F0F, // s_mov_b32 s10, 0x0f0f0f0f
        0x880A_080A, // s_or_b32 s10, s10, s8
        0xBE8B_03FF,
        0x3333_3333, // s_mov_b32 s11, 0x33333333
        0x7E02_0287, // v_mov_b32 v1, 7
        0xD501_0002,
        0x002A_0101, // v_cndmask_b32 v2, v1, v0, s[10:11]
    ];
    for name in hardware_adapters() {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        for (label, body, by_group) in [("constant", &constant[..], false), ("built", &built, true)]
        {
            // (128 threads of a wave64: two console waves, one host group
            // each where the host has no 64-lane waves.)
            for (wave64, threads) in [(true, 64), (false, 64), (false, 32), (true, 128)] {
                let mut p = program(body, threads);
                p.wave64 = wave64;
                let groups = 2;
                let out = run(&host, &p, groups)
                    .unwrap_or_else(|e| panic!("{name}: {label} wave64 {wave64}: {e}"));
                let want: Vec<u32> = (0..groups)
                    .flat_map(|g| {
                        // s8 = g * threads * 4: bit 8 in group 1 of 64, bit 7 of 32.
                        let lo = 0x0F0F_0F0F | if by_group { g * threads * 4 } else { 0 };
                        (0..threads).map(move |t| {
                            let (word, lane) = if wave64 && t % 64 >= 32 {
                                (0x3333_3333, t % 64 - 32)
                            } else {
                                (lo, t % 32)
                            };
                            if word >> lane & 1 != 0 { t } else { 7 }
                        })
                    })
                    .collect();
                assert_eq!(
                    out, want,
                    "{name}: a select on a {label} mask, wave64 {wave64}, {threads} threads"
                );
                eprintln!(
                    "{name}: a select on a {label} mask, wave64 {wave64}, {threads} threads exact"
                );
            }
        }
    }
}

#[test]
fn numeric_selects_beside_in_half_reads() {
    let mut body = IN_HALF.to_vec();
    body.extend([
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_03C1, // s_mov_b32 s11, -1
    ]);
    let select = body.len();
    body.extend([
        0xD501_0001,
        0x0029_0F02, // v_cndmask_b32 v1, v2, 7, s[10:11] (unused)
    ]);
    let wave32 = super::Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    let unused = program(&body, 128);
    let shader = emit_for(&unused, wave32).unwrap();
    assert_eq!(wave_size(&shader.dxil), Some(32));
    program_case(
        "unused numeric select beside in-half reads",
        &unused,
        |t| t ^ 31,
        true,
    );

    body[select] = 0xD501_0002; // write v2, the stored result
    let used = program(&body, 64);
    assert!(emit_for(&used, wave32).is_ok());
    program_case(
        "used numeric select beside in-half reads",
        &used,
        |t| if t < 32 { t ^ 31 } else { 7 },
        true,
    );
}

/// A lane read inside a divergent branch in a program that also has a
/// whole-wave section: the lane read takes the console flow on every host,
/// and the section is lowered there as it is elsewhere. Lanes below 20
/// read lane 10; every lane then adds the wave's sum of lane ids.
#[test]
fn lane_reads_in_branches_beside_a_whole_wave_section() {
    let mut body = reduction_scan(37, 0x80); // s10 = 2016
    body.extend([
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0003, // s_cbranch_execz +3
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
        0x4A04_040A, // v_add_nc_u32 v2, s10, v2
    ]);
    console_flow_case(
        "a lane read in a branch beside a whole-wave section",
        &body,
        64,
        |t| 2016 + if t < 20 { 10 } else { 1 },
    );
}

/// EXEC restored dword by dword from a mask: the structure pass holds such
/// a mask as a number, and that number is known only as the mask's own
/// dword. The lowering used to follow that in a circle until the stack
/// ran out; it stops with a reason now.
#[test]
fn a_mask_copied_dword_by_dword_stops_instead_of_recursing() {
    let body = [
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0x878C_6A7E, // s_and_b64 s[12:13], exec, vcc
        0xBEFE_030C, // s_mov_b32 exec_lo, s12
        0xBEFF_030D, // s_mov_b32 exec_hi, s13
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    ];
    // In the console flow (a host without 64-lane waves) and per lane.
    for (lanes_min, lanes_max, why) in [
        (32, 32, "an opaque lane mask"),
        (32, 64, "a lane mask holding lanes elsewhere read as data"),
    ] {
        let target = super::Target {
            lanes_min,
            lanes_max,
        };
        let e = emit_for(&program(&body, 64), target)
            .expect_err("stops")
            .to_string();
        assert!(e.contains(why), "{lanes_min}..{lanes_max}: {e}");
    }
}

/// `s_*_saveexec` writes the old EXEC to its scalar destination and the new
/// EXEC to EXEC, and sets SCC to whether the *new* EXEC is non-zero. In
/// console flow, `s_cbranch_scc0` after `s_and_saveexec_b64` must branch on
/// the new EXEC (0 here), not the saved old EXEC (all lanes active).
#[test]
fn saveexec_scc_tests_new_exec_not_saved_exec() {
    let body = [
        0xBE89_03AA, // s_mov_b32 s9, 42
        0x7D88_0080, // v_cmp_gt_u32 vcc, 0, v0 (false in every lane)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc (old != 0, new == 0 -> SCC = 0)
        0xBF84_0003, // s_cbranch_scc0 +3 (must skip the body)
        0xD760_000E,
        0x0001_1500, // v_readlane_b32 s14, v0, 10 (takes the console flow)
        0xBE89_03BC, // s_mov_b32 s9, 60 (must not run)
        0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("saveexec SCC tests new EXEC", &body, 64, |_| 42);
}

/// A loop that runs while `s_and_saveexec` leaves a lane: SCC is the new
/// EXEC, so the wave leaves when its last lane does. Lane t stays for t + 1
/// iterations and s9 counts the wave's: 64. Branching on the saved EXEC
/// ran the loop once more, with no lane in it.
#[test]
fn loops_leave_on_the_new_exec_of_saveexec() {
    let body = [
        0xD760_000E,
        0x0001_1500, // v_readlane_b32 s14, v0, 10 (takes the console flow)
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0xBE89_0380, // s_mov_b32 s9, 0
        // loop:
        0x8009_8109, // s_add_u32 s9, s9, 1
        0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
        0x7D82_0280, // v_cmp_lt_u32 vcc, 0, v1
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF85_FFFB, // s_cbranch_scc1 loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("a loop leaving on saveexec's SCC", &body, 64, |_| 64);
}

/// A 64-lane group filters lanes through two nested EXEC branches, writes the
/// smallest surviving element ID and a mask of participating halves to LDS,
/// then reads both words with EXEC restored. The two groups use distinct IDs.
#[test]
fn nested_exec_masks_keep_shared_visibility_results_across_halves() {
    let Ok(mut names) = adapters() else { return };
    // Host::on chooses by name; repeated DXGI descriptions open the same device.
    names.sort_unstable();
    names.dedup();
    for name in names {
        let Ok(host) = Host::on(&name) else { continue };
        if !host.target().runs(32) && !host.target().runs(64) {
            continue;
        }
        // The outer mask is lane < end, the inner mask is lane >= start.
        for (start, end) in [
            (0u32, 0u32),
            (8, 9),
            (40, 41),
            (31, 33),
            (0, 64),
            (40, 32),
            (63, 64),
            (32, 32),
        ] {
            let body = [
                0x7E02_0280, // v_mov_b32 v1, 0
                0x7E0A_02FF,
                0x0000_00FF, // v_mov_b32 v5, 0xff
                0x7E0C_0280, // v_mov_b32 v6, 0
                0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0xBF88_0002, // s_cbranch_execz init_done
                0xD838_0100,
                0x0006_0501, // ds_write2_b32 v1, v5, v6 offset1:1
                0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0xBF8A_0000, // s_barrier
                0x8F0E_8601, // s_lshl_b32 s14, s1, 6
                0x4A0E_000E, // v_add_nc_u32 v7, s14, v0
                0x7E0A_0281, // v_mov_b32 v5, 1
                0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0
                0x020C_0A82, // v_cndmask_b32 v6, 2, v5, vcc
                0x7D88_0080 | end, // v_cmp_gt_u32 vcc, end, v0
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0xBF88_0008, // s_cbranch_execz outer_done
                0x7D86_0080 | start, // v_cmp_le_u32 vcc, start, v0
                0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
                0xBF88_0004, // s_cbranch_execz inner_done
                0xD81C_0000,
                0x0000_0701, // ds_min_u32 v1, v7
                0xD828_0004,
                0x0000_0601, // ds_or_b32 v1, v6 offset:4
                0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
                0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0xBF8A_0000, // s_barrier
                0xD8DC_0100,
                0x0400_0001, // ds_read2_b32 v[4:5], v1 offset1:1
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0x3408_0882, // v_lshlrev_b32 v4, 2, v4
                0x3804_0905, // v_or_b32 v2, v5, v4
            ];
            let values = run(&host, &program_lds(&body, 64), 2)
                .unwrap_or_else(|e| panic!("{name}, [{start}, {end}): {e}"));
            assert_eq!(values.len(), 128);
            for (group, lanes) in values.as_chunks::<64>().0.iter().enumerate() {
                let mut first = 255;
                let mut halves = 0;
                for lane in start..end {
                    first = first.min(group as u32 * 64 + lane);
                    halves |= 1 << (lane / 32);
                }
                let want = (first << 2) | halves;
                for (lane, actual) in lanes.iter().enumerate() {
                    assert_eq!(
                        *actual, want,
                        "{name}, [{start}, {end}), group {group}, lane {lane}"
                    );
                }
            }
            eprintln!("{name}: [{start}, {end}) matched 128 lane results");
        }
    }
}

/// A select on an inline constant mask (here 5: lanes 0 and 2) tests the
/// console lane's bit, like one on a mask held in scalars
/// (`numeric_selects_beside_in_half_reads`). A program split in halves has
/// only its host lane for it (lanes 32 and 34 would take the bits of lanes
/// 0 and 2): on a host without 64-lane waves the select is deferred and,
/// its result being used, the program takes the console flow instead.
#[test]
fn selects_on_constant_masks_beside_in_half_reads() {
    let mut body = IN_HALF.to_vec();
    body.extend_from_slice(&[
        0xD501_0002,
        0x0216_0101, // v_cndmask_b32 v2, v1, v0, 5
    ]);
    let p = program(&body, 64);
    let wave32 = super::Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    let halves = emit_for(&p, wave32).unwrap();
    assert_eq!(wave_size(&halves.dxil), Some(32));
    // One host wave is the console wave: the lane tests its own bit.
    let native = emit_for(&p, super::Target::ANY).unwrap();
    assert_eq!(wave_size(&native.dxil), Some(64));
    // v1 is t ^ 16 (the swizzle), v0 is t.
    program_case(
        "a select on a constant mask beside in-half reads",
        &p,
        |t| if t == 0 || t == 2 { t } else { t ^ 16 },
        true,
    );
}

/// The exit mask of a loop its lanes leave at different iterations, counted
/// after the loop: all 64 lanes are in it. Per lane, each lane counted what
/// its own path saw (1 on a host with 64-lane waves). Fixture from the
/// scalar-state patch posted on upstream PR #439, after PR #269's.
#[test]
fn loop_exit_mask_counted_after_the_loop() {
    let body = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0  (lane i loops i + 1 times)
        0xBE8A_0480, // s_mov_b64 s[10:11], 0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1 (v1 - 1)
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
        0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
        0xBF89_FFFB, // s_cbranch_execnz loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBE89_100A, // s_bcnt1_i32_b64 s9, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("loop_exit_mask_counted_after_the_loop", &body, 64, |_| 64);
}

/// A mask grown inside a branch only lanes 32-63 take and counted after it:
/// lanes 0-15 and 32-47. Per lane the program stopped (`a lane mask holding
/// lanes elsewhere read as data`). Same source as the test above.
#[test]
fn mask_grown_inside_a_branch_counted_after_it() {
    let body = [
        0x7D88_0090, // v_cmp_gt_u32 vcc, 16, v0      (lanes 0-15)
        0xBE8A_046A, // s_mov_b64 s[10:11], vcc
        0x7D86_00A0, // v_cmp_le_u32 vcc, 32, v0      (lanes 32-63)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0002, // s_cbranch_execz -> join
        0x7D88_00B0, // v_cmp_gt_u32 vcc, 48, v0      (of those: 32-47)
        0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
        0x88FE_0C7E, // join: s_or_b64 exec, exec, s[12:13]
        0xBE89_100A, // s_bcnt1_i32_b64 s9, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case(
        "mask_grown_inside_a_branch_counted_after_it",
        &body,
        64,
        |_| 32,
    );
}

/// The wave32 form of `loop_exit_mask_counted_after_the_loop`, with one
/// dword of mask per wave.
fn wave32_exit_mask_program(threads: u32) -> crate::NativeProgram {
    let body = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0  (lane i loops i + 1 times)
        0xBE8A_0380, // s_mov_b32 s10, 0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
        0x880A_0A6A, // s_or_b32 s10, vcc_lo, s10
        0x8A7E_0A7E, // s_andn2_b32 exec_lo, exec_lo, s10
        0xBF89_FFFB, // s_cbranch_execnz loop
        0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
        0xBE89_0F0A, // s_bcnt1_i32_b32 s9, s10
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let mut p = program(&body, threads);
    p.wave64 = false;
    p
}

#[test]
fn wave32_loop_exit_mask_counted_after_the_loop() {
    let p = wave32_exit_mask_program(32);
    program_case("wave32 loop exit mask counted", &p, |_| 32, true);
}

#[test]
fn wave32_loop_exit_mask_counts_a_partial_second_wave() {
    let p = wave32_exit_mask_program(48);
    let groups = 3;
    let expected: Vec<u32> = (0..groups)
        .flat_map(|_| (0..48).map(|lane| if lane < 32 { 32 } else { 16 }))
        .collect();
    for name in hardware_adapters() {
        let Ok(host) = Host::on(&name) else { continue };
        // This case checks native Wave32 execution. WARP can report
        // incorrect active-lane counts for a partial wave.
        if host.reported_target().is_some_and(|target| target.runs(32)) {
            let out = run(&host, &p, groups).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(out, expected, "{name}: partial Wave32 exit mask");
            eprintln!("{name}: all {} lane counts exact", out.len());
        } else {
            eprintln!("{name}: native Wave32 support not reported; skipped");
        }
    }
}

// ---- groups of several console waves sharing LDS (`lower::groups`) ----

/// [`program`] with `granules` of 512 bytes of LDS.
fn program_lds_of(body: &[u32], threads: u32, granules: u32) -> crate::NativeProgram {
    let mut p = program(body, threads);
    let StageIo::Cs(io) = &mut p.io else {
        unreachable!()
    };
    io.rsrc2 |= granules << 15;
    p
}

const NVIDIA: super::Target = super::Target {
    lanes_min: 32,
    lanes_max: 32,
};

/// Emitted for a 32-lane host as one host group of 32-lane waves, and
/// valid where a validator is installed.
fn one_host_group(p: &crate::NativeProgram, label: &str) {
    let cs = emit_for(p, NVIDIA).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(cs.group_waves, 1, "{label}");
    assert_eq!(wave_size(&cs.dxil), Some(32), "{label}");
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil.clone())
            .unwrap_or_else(|e| panic!("{label}: validation: {e}"));
    }
}

/// Lane t writes t to dword t; after an `s_barrier` it reads the dword of
/// the same lane in the other console wave (t ^ 64) and adds what lane 40
/// of its own console wave holds (`v_readlane`, the other half).
const LDS_ACROSS_WAVES: &[u32] = &[
    0x3408_0082, // v_lshlrev_b32 v4, 2, v0
    0xD834_0000,
    0x0000_0004, // ds_write_b32 v4, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0x3A0A_00C0, // v_xor_b32 v5, 64, v0
    0x340A_0A82, // v_lshlrev_b32 v5, 2, v5
    0xD8D8_0000,
    0x0600_0005, // ds_read_b32 v6, v5
    0xD760_0009,
    0x0001_5100, // v_readlane_b32 s9, v0, 40
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x4A04_0C09, // v_add_nc_u32 v2, s9, v6
];

/// Waves above the first end at once (EXEC empty: `s_cbranch_execz` to
/// `s_endpgm`), before the first wave's `s_barrier`; the first wave
/// shares LDS between its halves across it: lane t writes t + 1 to dword
/// t and reads dword t ^ 32, then adds lane 40's id. The branch skips the
/// rest of the body and the store [`program`] appends.
fn early_end() -> Vec<u32> {
    let rest: &[u32] = &[
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x4A0E_0081, // v_add_nc_u32 v7, 1, v0
        0xD834_0000,
        0x0000_0704, // ds_write_b32 v4, v7
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x3A0A_00A0, // v_xor_b32 v5, 32, v0
        0x340A_0A82, // v_lshlrev_b32 v5, 2, v5
        0xD8D8_0000,
        0x0200_0005, // ds_read_b32 v2, v5
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ];
    // Past the rest, then [`program`]'s address arithmetic (2) and store (2).
    let skip = rest.len() as u32 + 4;
    let mut w = vec![
        0x7D88_00C0,        // v_cmp_gt_u32 vcc, 64, v0
        0xBE8A_246A,        // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0000 | skip, // s_cbranch_execz s_endpgm
    ];
    w.extend_from_slice(rest);
    w
}

/// Thread 0 clears dword 0; after an `s_barrier`, lanes below 20 of the
/// group (all in the first console wave: the others skip the region, EXEC
/// empty) add 1 to it with an LDS atomic; after another, every lane reads
/// it and adds lane 40's id in its console wave.
const SKIPPED_BY_SOME_WAVES: &[u32] = &[
    0xD760_0009,
    0x0001_5100, // v_readlane_b32 s9, v0, 40
    0x7E0E_0280, // v_mov_b32 v7, 0
    0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
    0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    0xD834_0000,
    0x0000_0707, // ds_write_b32 v7, v7
    0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
    0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    0xBF88_0003, // s_cbranch_execz +3
    0x7E10_0281, // v_mov_b32 v8, 1
    0xD800_0000,
    0x0000_0807, // ds_add_u32 v7, v8
    0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0xD8D8_0000,
    0x0200_0007, // ds_read_b32 v2, v7
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x4A04_0409, // v_add_nc_u32 v2, s9, v2
];

/// A branch each console wave takes by its own count of lanes below 20
/// (`s_bcnt1`, SCC): the first wave goes in and writes its count to its
/// dword (1 + wave), the second does not, its EXEC full. Thread 0 first
/// puts 7 in both dwords; after an `s_barrier` each lane reads the other
/// wave's dword: the second wave's 7 is still there.
const OWN_COUNT_REGION: &[u32] = &[
    0x7E0E_0280, // v_mov_b32 v7, 0
    0x7E12_0287, // v_mov_b32 v9, 7
    0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
    0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    0xD838_0201,
    0x0009_0907, // ds_write2_b32 v7, v9, v9 offset0:1 offset1:2
    0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
    0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
    0xBF84_0006, // s_cbranch_scc0 +6
    0x2C14_0086, // v_lshrrev_b32 v10, 6, v0
    0x4A14_1481, // v_add_nc_u32 v10, 1, v10
    0x3414_1482, // v_lshlrev_b32 v10, 2, v10
    0x7E16_0209, // v_mov_b32 v11, s9
    0xD834_0000,
    0x0000_0B0A, // ds_write_b32 v10, v11
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0x2C18_0086, // v_lshrrev_b32 v12, 6, v0
    0x3A18_1881, // v_xor_b32 v12, 1, v12
    0x4A18_1881, // v_add_nc_u32 v12, 1, v12
    0x3418_1882, // v_lshlrev_b32 v12, 2, v12
    0xD8D8_0000,
    0x0200_000C, // ds_read_b32 v2, v12
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

/// Groups of two and four console waves sharing LDS across an `s_barrier`
/// emit for a 32-lane host as one host group, valid: the waves exchange
/// through LDS, a wave reads a lane of its other half, waves end before
/// the barrier, skip a region by EXEC or by a count of their own.
#[test]
fn groups_sharing_lds_emit_as_one_host_group() {
    one_host_group(
        &program_lds_of(LDS_ACROSS_WAVES, 128, 1),
        "across waves, 128",
    );
    one_host_group(
        &program_lds_of(LDS_ACROSS_WAVES, 256, 2),
        "across waves, 256",
    );
    one_host_group(&program_lds_of(&early_end(), 128, 1), "early end");
    one_host_group(
        &program_lds_of(SKIPPED_BY_SOME_WAVES, 256, 1),
        "skipped region",
    );
    one_host_group(&program_lds_of(OWN_COUNT_REGION, 128, 1), "own count");
}

/// What a group of several console waves cannot run as one control flow
/// stops with its reason: a loop its waves may leave at different
/// iterations, an `s_barrier` in a region only some waves run, a branch
/// with code on both sides that the waves may take differently.
#[test]
fn groups_sharing_lds_stop_where_waves_part() {
    let read = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
    ];
    let why = |body: &[u32]| {
        let mut w = read.to_vec();
        w.extend_from_slice(body);
        match emit_for(&program_lds_of(&w, 128, 1), NVIDIA) {
            Err(crate::ShaderError::Unsupported { what, .. }) => what,
            Err(e) => panic!("{e}"),
            Ok(_) => panic!("emitted"),
        }
    };
    let lane_loop = [
        0x7E04_0280, // v_mov_b32 v2, 0
        0x4A04_0481, // loop: v_add_nc_u32 v2, 1, v2
        0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFFC, // s_cbranch_execnz loop
    ];
    assert_eq!(why(&lane_loop), super::lower::groups::LOOP_LEFT_APART);
    let barrier_inside = [
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0001, // s_cbranch_execz +1
        0xBF8A_0000, // s_barrier
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    ];
    assert_eq!(
        why(&barrier_inside),
        super::lower::groups::BARRIER_IN_REGION
    );
    let both_sides = [
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0x87EA_6A7E, // s_and_b64 vcc, exec, vcc
        0xBF86_0002, // s_cbranch_vccz else
        0x7E04_0281, // v_mov_b32 v2, 1
        0xBF82_0001, // s_branch end
        0x7E04_0282, // else: v_mov_b32 v2, 2
    ];
    assert_eq!(why(&both_sides), super::lower::groups::BRANCH_APART);
}

/// The same programs dispatched on every 32-lane hardware adapter
/// (none under `VICTORY_WARP=1`), two console groups each, checked lane
/// by lane against the console.
#[test]
fn groups_sharing_lds_match_the_console_on_every_adapter() {
    for (threads, granules) in [(128, 1), (256, 2)] {
        program_case(
            "LDS across console waves",
            &program_lds_of(LDS_ACROSS_WAVES, threads, granules),
            |t| (t ^ 64) + (t & !63) + 40,
            true,
        );
        program_case(
            "a region some console waves skip",
            &program_lds_of(SKIPPED_BY_SOME_WAVES, threads, granules),
            |t| 20 + (t & !63) + 40,
            true,
        );
    }
    program_case(
        "waves that end before the barrier",
        &program_lds_of(&early_end(), 128, 1),
        |t| if t < 64 { (t ^ 32) + 1 + 40 } else { 0 },
        true,
    );
    program_case(
        "a region by each wave's own count",
        &program_lds_of(OWN_COUNT_REGION, 128, 1),
        |t| if t < 64 { 7 } else { 20 },
        true,
    );
}
