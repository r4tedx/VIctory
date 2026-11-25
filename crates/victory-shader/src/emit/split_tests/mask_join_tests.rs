//! Scalar lane-mask joins in Wave64 compute groups split across two host waves.
//! Instruction encodings are from llvm-mc for GFX1030 in wave64 mode.
use super::{Host, adapters, emit_for, program, run, wave_size};
use crate::emit::Target;

const LM_READ_LO: u32 = 0x7E04_0214; // v_mov_b32 v2, s20
const LM_READ_HI: u32 = 0x7E04_0215; // v_mov_b32 v2, s21

/// `before`, then `inside` in the lanes below 45 (`s_and_saveexec_b64
/// s[10:11], vcc`, skipped by `s_cbranch_execz` when none), EXEC
/// restored, then `after`.
fn lm_branch(before: &[u32], inside: &[u32], after: &[u32]) -> Vec<u32> {
    let mut b = vec![0x7E04_0280]; // v_mov_b32 v2, 0
    b.extend_from_slice(before);
    b.extend([
        0x7D88_00AD,                       // v_cmp_gt_u32 vcc, 45, v0
        0xBE8A_246A,                       // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0000 | inside.len() as u32, // s_cbranch_execz
    ]);
    b.extend_from_slice(inside);
    b.push(0x88FE_0A7E); // s_or_b64 exec, exec, s[10:11]
    b.extend_from_slice(after);
    b
}

fn lm_join(init: u32, read: u32) -> Vec<u32> {
    lm_branch(
        &[init],
        &[
            0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
            0xBE94_046A, // s_mov_b64 s[20:21], vcc
        ],
        &[read],
    )
}

const LM_ZERO: u32 = 0xBE94_0480; // s_mov_b64 s[20:21], 0
const LM_ONES: u32 = 0xBE94_04C1; // s_mov_b64 s[20:21], -1

/// Both arms of an if/else (lanes below 45 / the rest) write s[20:21]:
/// the then arm lanes below 40, the else arm lanes above 50; the wave
/// leaves through the else arm: lo 0, hi 0xFFF80000 in every lane.
fn lm_if_else(read: u32) -> Vec<u32> {
    vec![
        0x7E04_0280, // v_mov_b32 v2, 0
        LM_ZERO,
        0x7D88_00AD, // v_cmp_gt_u32 vcc, 45, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0x898A_0A7E, // s_xor_b64 s[10:11], exec, s[10:11]
        0xBF88_0002, // s_cbranch_execz 2
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE94_046A, // s_mov_b64 s[20:21], vcc
        0xBE8A_250A, // s_or_saveexec_b64 s[10:11], s[10:11]
        0x89FE_0A7E, // s_xor_b64 exec, exec, s[10:11]
        0xBF88_0002, // s_cbranch_execz 2
        0x7D82_00B2, // v_cmp_lt_u32 vcc, 50, v0
        0xBE94_046A, // s_mov_b64 s[20:21], vcc
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        read,
    ]
}

/// Runs `p` in 2 groups of 64 on every adapter that splits a wave64 and
/// checks every word against `want(group, t)`.
fn lm_groups(label: &str, p: &crate::NativeProgram, want: impl Fn(u32, u32) -> u32) {
    let Ok(names) = adapters() else {
        eprintln!("no DXGI; skipped");
        return;
    };
    let expect: Vec<u32> = (0..2)
        .flat_map(|g| (0..64).map(move |t| (g, t)))
        .map(|(g, t)| want(g, t))
        .collect();
    for name in names {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        if !host.target().runs(32) || host.target().runs(64) {
            continue;
        }
        let out = run(&host, p, 2).unwrap_or_else(|e| panic!("{name}: {label}: {e}"));
        assert_eq!(out, expect, "{name}: {label}");
        eprintln!("{name}: {label}: both groups exact");
    }
}

/// [`lm_branch`] whose branch holds lanes below 45 in group 0 and no lane
/// in group 1 (45 - 45 * group id): group 1's wave skips it whole.
fn lm_branch_by_group(before: &[u32], inside: &[u32], after: &[u32]) -> Vec<u32> {
    let mut b = lm_branch(before, inside, after);
    let at = b
        .iter()
        .position(|&w| w == 0x7D88_00AD)
        .expect("the branch compare");
    b.splice(
        at..=at,
        [
            0x930D_FF01,
            0xFFFF_FFD3, // s_mul_i32 s13, s1, -45
            0x810D_AD0D, // s_add_i32 s13, s13, 45
            0x7D88_000D, // v_cmp_gt_u32 vcc, s13, v0
        ],
    );
    b
}

#[test]
fn lm64_joined_masks_follow_the_taken_edge() {
    for (read, active) in [(LM_READ_LO, u32::MAX), (LM_READ_HI, 0xff)] {
        for (init, skipped) in [(LM_ZERO, 0), (LM_ONES, u32::MAX)] {
            let body = lm_branch_by_group(&[init], &[0x7D88_00A8, 0xBE94_046A], &[read]);
            lm_groups(
                "mask joined with a constant",
                &program(&body, 64),
                |group, _| {
                    if group == 0 { active } else { skipped }
                },
            );
        }
        let expected = if read == LM_READ_LO { 0 } else { 0xfff8_0000 };
        lm_groups(
            "mask written on both edges",
            &program(&lm_if_else(read), 64),
            |_, _| expected,
        );
    }
}

/// A lane-mask join inside a loop body (no back edge into the join itself):
/// each of two passes zeroes s[20:21], sets it to vcc (lanes below
/// 40 + 10 * pass, under the branch's lanes below 45) in the branch, and
/// xors the joined high dword into v2: 0xFF ^ 0x1FFF in every lane.
const LM_LOOP_BODY_JOIN: &[u32] = &[
    0x7E04_0280, // v_mov_b32 v2, 0
    LM_ZERO,
    0xBE8E_0380, // s_mov_b32 s14, 0
    LM_ZERO,     // loop:
    0x7D88_00AD, // v_cmp_gt_u32 vcc, 45, v0
    0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    0xBF88_0004, // s_cbranch_execz +4
    0x930F_8A0E, // s_mul_i32 s15, s14, 10
    0x810F_A80F, // s_add_i32 s15, s15, 40
    0x7D88_000F, // v_cmp_gt_u32 vcc, s15, v0
    0xBE94_046A, // s_mov_b64 s[20:21], vcc
    0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    0x3A04_0415, // v_xor_b32 v2, s21, v2
    0x800E_810E, // s_add_u32 s14, s14, 1
    0xBF0A_820E, // s_cmp_lt_u32 s14, 2
    0xBF85_FFF3, // s_cbranch_scc1 loop
];

#[test]
fn lm64_join_in_a_loop_body() {
    lm_groups(
        "join in a loop body, hi",
        &program(LM_LOOP_BODY_JOIN, 64),
        |_, _| 0x1F00,
    );
}

/// What lowering gives a program: its declared wave size, or why it stops.
fn outcome(p: &crate::NativeProgram, lanes_min: u32, lanes_max: u32) -> String {
    match emit_for(
        p,
        Target {
            lanes_min,
            lanes_max,
        },
    ) {
        Ok(s) => format!("wave {:?}", wave_size(&s.dxil)),
        Err(e) => format!("stops: {e}"),
    }
}

/// The mask-join programs above as wave32, as native wave64 on hosts that
/// run 64-lane waves, and in groups of 128: the console-flow join arm is
/// for split wave64 groups of one wave only. Elsewhere each keeps the
/// outcome it had before the arm (pinned from the wave-per-group base).
#[test]
fn lm64_join_outcomes_outside_the_split() {
    const ELSEWHERE: &str = "a lane mask holding lanes elsewhere read as data is not supported";
    const AS_DATA: &str = "a lane mask read as data is not supported";
    const SIZE: &str =
        "GFX10 code at 0x0: a console wave size the host does not run is not supported";
    // (name, body, where wave32 stops, where wave64 stops)
    let bodies: [(&str, Vec<u32>, u32, u32); 5] = [
        ("join from 0", lm_join(LM_ZERO, LM_READ_LO), 0x2c, 0x48),
        ("join from -1", lm_join(LM_ONES, LM_READ_LO), 0x2c, 0x48),
        ("if/else", lm_if_else(LM_READ_LO), 0x30, 0x60),
        (
            "by group",
            lm_branch_by_group(&[LM_ZERO], &[0x7D88_00A8, 0xBE94_046A], &[LM_READ_LO]),
            0x38,
            0x54,
        ),
        ("loop body", LM_LOOP_BODY_JOIN.to_vec(), 0x34, 0x64),
    ];
    for (name, body, at32, at64) in &bodies {
        for wave64 in [false, true] {
            for threads in [64, 128] {
                for (lo, hi) in [(32, 32), (32, 64), (64, 64)] {
                    let mut p = program(body, threads);
                    p.wave64 = wave64;
                    let want = match (wave64, threads, hi) {
                        (true, 64, 32) => "wave Some(32)".to_string(),
                        (true, ..) => format!("stops: GFX10 code at {at64:#x}: {AS_DATA}"),
                        (false, _, 64) if lo == 64 => format!("stops: {SIZE}"),
                        (false, ..) => format!("stops: GFX10 code at {at32:#x}: {ELSEWHERE}"),
                    };
                    let got = outcome(&p, lo, hi);
                    assert_eq!(
                        got, want,
                        "{name}, wave64={wave64}, {threads} threads, {lo}..{hi}"
                    );
                }
            }
        }
    }
}

/// A separate lane read can already route a multi-wave group to console
/// flow, with one host group per console wave. Keep its previous refusal.
#[test]
fn lm64_multi_wave_group_with_another_lane_read_keeps_its_mask_join_refusal() {
    let mut body = vec![0xD760_0009, 0x0001_5100]; // v_readlane_b32 s9, v0, 40
    body.extend(lm_join(LM_ZERO, LM_READ_HI));
    let got = outcome(&program(&body, 128), 32, 32);
    assert!(got.contains("a lane mask read as data"), "{got}");
}
