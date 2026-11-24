//! Review probes for traced buffer addresses: hand-assembled compute programs
//! that read a root SRV (or a table descriptor) through one address shape
//! each and store what they read to a UAV, compared with the dwords the
//! console would read (computed by hand from the same inputs).
//!
//! Layout: root constants in `s0..s3`, a table at `s4`, root SRV A in
//! `s[6:7]`, root SRV B in `s[8:9]`. The UAV descriptor is in `s[12:15]`;
//! bodies use `s16` and up, `v5` and up, and leave the result in `v0`.

use super::super::host::Stride;
use super::*;

const A: fn(u32) -> u32 = |i| 0xA000_0000 + i;
const B: fn(u32) -> u32 = |i| 0xB000_0000 + i;
const DWORDS: u32 = 64;

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], soffset, offset`.
const fn s_load_x4(dst: u8, base: u8, soffset: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        (soffset as u32) << 25 | (offset & 0x1F_FFFF),
    ]
}

const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32 & 0x7F) << 16 | (src1 as u32) << 8 | src0 as u32
}
const fn sopp(op: u8, simm16: i16) -> u32 {
    0xBF80_0000 | (op as u32 & 0x7F) << 16 | simm16 as u16 as u32
}
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}
const fn vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | (sdst as u32 & 0x7F) << 8 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}
const fn global_load_dword(vdst: u8, vaddr: u8, saddr: u8, offset: i16) -> [u32; 2] {
    [
        0x37 << 26 | 0x0C << 18 | 2 << 14 | (offset as u32 & 0xFFF),
        (vdst as u32) << 24 | (saddr as u32 & 0x7F) << 16 | vaddr as u32,
    ]
}
/// MUBUF `buffer_load_dword vdata, v[index], s[srsrc:srsrc+3], 0` with idxen.
const fn buffer_load_dword_idx(vdata: u8, vindex: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x0C << 18 | 1 << 13,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vindex as u32,
    ]
}

const fn s_addk_i32(sdst: u8, imm: i16) -> u32 {
    0xB000_0000 | 0x0F << 23 | (sdst as u32 & 0x7F) << 16 | imm as u16 as u32
}

/// `global_load_dword v0, v5, s[16:17]` with `v5 = 0`.
fn load_s16() -> Vec<u32> {
    let l = global_load_dword(0, 5, 16, 0);
    vec![vop1(0x001, 5, 0x80), l[0], l[1]]
}

/// `s_cmp_eq_u32 s0, 0`, then `then` when `s0 != 0` and `other` when it is 0.
fn diamond(then: &[u32], other: &[u32]) -> Vec<u32> {
    let mut w = vec![
        sopc(0x06, 0, 0x80),
        sopp(0x05, then.len() as i16 + 1),
    ];
    w.extend_from_slice(then);
    w.push(sopp(0x02, other.len() as i16));
    w.extend_from_slice(other);
    w
}

/// The table's used ranges and the byte offset of the output UAV's descriptor.
struct Tables {
    used: Vec<UsedResource>,
    out: u32,
}

fn build(body: &[u32], tables: Option<&Tables>) -> Result<HostShader, String> {
    build_with_threads(body, tables, 1)
}

fn build_with_threads(
    body: &[u32],
    tables: Option<&Tables>,
    threads: u32,
) -> Result<HostShader, String> {
    let mut body = body.to_vec();
    body.extend(output_store(0, threads));
    build_program(&probe_program(&body, tables, threads), super::super::Target::ANY)
}

fn output_store(vdata: u8, threads: u32) -> [u32; 2] {
    let mut store = buffer_store_dword(vdata, 12, 0);
    if threads > 1 {
        store[0] |= 1 << 12; // offen
        store[1] |= 7;       // vaddr = v7
    }
    store
}

fn probe_program(body: &[u32], tables: Option<&Tables>, threads: u32) -> NativeProgram {
    let mut words = Vec::new();
    if threads > 1 {
        // v0 starts as the thread id in X. v7 is reserved for its byte
        // offset in the output UAV.
        words.push(vop2(0x1A, 7, 0x82, 0)); // v_lshlrev_b32 v7, 2, v0
    }
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_x4(12, 4, 125, tables.map_or(0, |t| t.out)));
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i)));
    }
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let rp = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let srv = |user_data, slot| UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data,
        slot,
    };
    let mut used = match tables {
        Some(t) => t.used.clone(),
        None => vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
    };
    used.extend([srv(6, 2), srv(8, 3)]);
    used.push(UsedResource {
        kind: ResourceType::Uav,
        ..srv(10, 4)
    });
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            rp(RootType::Constants, 0, 4, 0),
            rp(RootType::Table, 4, 1, 1),
            rp(RootType::Descriptor, 6, 2, 2),
            rp(RootType::Descriptor, 8, 2, 3),
            rp(RootType::Descriptor, 10, 2, 4),
        ],
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 12 << 1,
            threads: [threads, 1, 1],
        }),
        code,
    }
}

fn build_program(native: &NativeProgram, target: super::super::Target) -> Result<HostShader, String> {
    let p = ir::build(&native.code, native.wave64, native.vgprs)
        .map_err(|e| format!("ir::build: {e:?}"))?;
    // The order `emit::translate` has: the tracer's refusal is reported
    // before one of structuring.
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())
        .map_err(|e| format!("refused by the tracer: {e:?}"))?;
    let io = io::raise(&p, native).map_err(|e| format!("io::raise: {e:?}"))?;
    let st = st.map_err(|e| format!("structure: {e:?}"))?;
    let container = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&container).map_err(|e| format!("container: {e:?}"))?;
    let shader = super::super::emit(&c, native, &p, &st, &io, &res, target)
        .map_err(|e| format!("refused by lowering: {e:?}"))?;
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy)
            .map_err(|e| format!("DXIL validation failed: {e}"))?;
    }
    Ok(shader)
}

/// Dispatches once per case on the hardware adapter and WARP; returns
/// `(adapter, case, value read back)`.
fn run(
    shader: &HostShader,
    cases: &[[u32; 4]],
    views: &[View],
    out_view: usize,
) -> Vec<(&'static str, [u32; 4], Result<u32, String>)> {
    run_with_readback(shader, cases, views, None, |o| match out_view {
        // The root UAV: the index of the one dword written, and
        // the dword, as `index << 24 | low bits`.
        usize::MAX => {
            let c = o.last().expect("a root UAV");
            let hits: Vec<usize> = (0..c.len()).filter(|&k| c[k] != 0).collect();
            match hits[..] {
                [k] => (k as u32) << 24 | (c[k] & 0xFF_FFFF),
                _ => u32::MAX,
            }
        }
        k => o[k][0],
    })
}

fn run_with_readback<T>(
    shader: &HostShader,
    cases: &[[u32; 4]],
    views: &[View],
    wave_size: Option<u32>,
    readback: impl Fn(Vec<Vec<u32>>) -> T,
) -> Vec<(&'static str, [u32; 4], Result<T, String>)> {
    assert!(shader.root_cbvs.is_empty(), "unexpected root CBV");
    let a: Vec<u32> = (0..DWORDS).map(A).collect();
    let b: Vec<u32> = (0..DWORDS).map(B).collect();
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        if let Some(wave) = wave_size {
            let target = host.target();
            if !target.runs(wave) {
                eprintln!(
                    "{label} wave sizes {}..{} exclude wave{wave}; skipped",
                    target.lanes_min, target.lanes_max
                );
                continue;
            }
        }
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(x) => x,
            Err(e) => {
                out.push((label, [0; 4], Err(format!("compute_pipeline: {e}"))));
                continue;
            }
        };
        for &inputs in cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    (root, _) => super::super::host::guard(root, views.len(), 0)
                        .unwrap_or_else(|| panic!("unexpected user data {:?}", (root, u.slot))),
                };
            }
            let mut roots = vec![(0, Root::Constants(ud))];
            let mut slots: Vec<u8> = shader
                .root_srvs
                .iter()
                .chain(&shader.root_uavs)
                .copied()
                .collect();
            slots.sort_unstable();
            for (k, slot) in slots.iter().enumerate() {
                let view = match slot {
                    2 => Root::View {
                        srv: true,
                        data: a.clone(),
                    },
                    3 => Root::View {
                        srv: true,
                        data: b.clone(),
                    },
                    4 => Root::Uav(DWORDS),
                    other => panic!("unexpected root view slot {other}"),
                };
                roots.push((k as u32 + 1, view));
            }
            let r = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots,
                        views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .map(&readback);
            out.push((label, inputs, r));
        }
    }
    assert!(wave_size.is_some() || !out.is_empty(), "no SM 6.6 device");
    out
}

enum Want {
    /// The dword the console reads, from the root constants.
    Reads(fn([u32; 4]) -> u32),
    /// The console program is not one we expect to translate.
    Refused,
}

struct Probe {
    name: &'static str,
    body: Vec<u32>,
    cases: Vec<[u32; 4]>,
    want: Want,
}

fn probes() -> Vec<Probe> {
    let cat = |parts: &[&[u32]]| parts.concat();
    let lit = |x: u32| x;
    let mut v = Vec::new();
    let mut add = |name, body: Vec<u32>, cases: &[[u32; 4]], want| {
        v.push(Probe {
            name,
            body,
            cases: cases.to_vec(),
            want,
        });
    };
    let one = [[0, 0, 0, 0]];

    // --- a root address plus a constant ---
    add(
        "A + 8 (s_add_u32, s_addc_u32)",
        cat(&[&[sop2(0x00, 16, 6, 0x88), sop2(0x04, 17, 7, 0x80)], &load_s16()]),
        &one,
        Want::Reads(|_| A(2)),
    );
    add(
        "A + 16 - 4 (s_addk_i32 twice)",
        cat(&[
            &[s_mov_b32(16, 6), s_addk_i32(16, 16), s_addk_i32(16, -4), s_mov_b32(17, 7)],
            &load_s16(),
        ]),
        &one,
        Want::Reads(|_| A(3)),
    );
    add(
        "A + 0x100 - 0x100 + 4 (s_sub_u32 of a constant)",
        cat(&[
            &[
                sop2(0x00, 16, 6, 0xFF),
                lit(0x100),
                sop2(0x01, 16, 16, 0xFF),
                lit(0xFC),
                s_mov_b32(17, 7),
            ],
            &load_s16(),
        ]),
        &one,
        Want::Reads(|_| A(1)),
    );

    // --- plus a register computed from data ---
    let dyn_cases = [[0, 0, 0, 0], [4, 0, 0, 0], [20, 8, 4, 0], [8, 36, 12, 0]];
    add(
        "A + s0",
        cat(&[&[sop2(0x00, 16, 6, 0), sop2(0x04, 17, 7, 0x80)], &load_s16()]),
        &dyn_cases,
        Want::Reads(|i| A(i[0] / 4)),
    );
    add(
        "(A + s0) + s1",
        cat(&[
            &[sop2(0x00, 16, 6, 0), sop2(0x00, 16, 16, 1), sop2(0x04, 17, 7, 0x80)],
            &load_s16(),
        ]),
        &dyn_cases,
        Want::Reads(|i| A((i[0] + i[1]) / 4)),
    );
    {
        // global_load_dword v0, v3 (= s2), s[16:17], offset 4
        let l = global_load_dword(0, 3, 16, 4);
        add(
            "(A + s0) + s1, VGPR offset s2, immediate 4",
            vec![
                sop2(0x00, 16, 6, 0),
                sop2(0x00, 16, 16, 1),
                sop2(0x04, 17, 7, 0x80),
                l[0],
                l[1],
            ],
            &dyn_cases,
            Want::Reads(|i| A((i[0] + i[1] + i[2] + 4) / 4)),
        );
    }
    add(
        "A + ((s0 << 2) + (s1 & 60)) (one term computed from data)",
        cat(&[
            &[
                sop2(0x1E, 18, 0, 0x82),  // s_lshl_b32 s18, s0, 2
                sop2(0x0E, 19, 1, 0xBC),  // s_and_b32 s19, s1, 60
                sop2(0x00, 18, 18, 19),   // s18 += s19
                sop2(0x00, 16, 6, 18),    // s16 = A + s18
                sop2(0x04, 17, 7, 0x80),
            ],
            &load_s16(),
        ]),
        &[[0, 0, 0, 0], [3, 0, 0, 0], [2, 0x7C, 0, 0], [5, 0xFFFF_FF0C, 0, 0]],
        Want::Reads(|i| A(((i[0] << 2) + (i[1] & 60)) / 4)),
    );

    // --- masked or shifted terms ---
    add(
        "(A + (s1 & 60)) + s2 (masked term, two terms)",
        cat(&[
            &[
                sop2(0x0E, 19, 1, 0xBC),
                sop2(0x00, 16, 6, 19),
                sop2(0x00, 16, 16, 2),
                sop2(0x04, 17, 7, 0x80),
            ],
            &load_s16(),
        ]),
        &[[0, 0, 0, 0], [0, 0x7C, 8, 0], [0, 0xFFFF_FF0C, 16, 0]],
        Want::Reads(|i| A(((i[1] & 60) + i[2]) / 4)),
    );
    for (name, op, k) in [
        ("(s0 << 1) + A (s_lshl1_add_u32)", 0x2Eu8, 1u32),
        ("(s0 << 2) + A (s_lshl2_add_u32)", 0x2F, 2),
        ("(s0 << 3) + A (s_lshl3_add_u32)", 0x30, 3),
        ("(s0 << 4) + A (s_lshl4_add_u32)", 0x31, 4),
    ] {
        let cases = [[0, 0, 0, 0], [4, 0, 0, 0], [6, 0, 0, 0]];
        add(
            name,
            cat(&[&[sop2(op, 16, 0, 6), s_mov_b32(17, 7)], &load_s16()]),
            &cases,
            match k {
                1 => Want::Reads(|i| A((i[0] << 1) / 4)),
                2 => Want::Reads(|i| A((i[0] << 2) / 4)),
                3 => Want::Reads(|i| A((i[0] << 3) / 4)),
                _ => Want::Reads(|i| A((i[0] << 4) / 4)),
            },
        );
    }
    add(
        "((s0 << 2) + A) + s1, then - s2 (s_sub_u32 of data)",
        cat(&[
            &[
                sop2(0x2F, 16, 0, 6),
                sop2(0x00, 16, 16, 1),
                sop2(0x01, 16, 16, 2),
                s_mov_b32(17, 7),
            ],
            &load_s16(),
        ]),
        &[[4, 8, 4, 0], [9, 0, 16, 0], [1, 40, 12, 0]],
        Want::Reads(|i| A(((i[0] << 2) + i[1] - i[2]) / 4)),
    );
    add(
        "(A << 0 shifted position) s_lshl2_add_u32 s16, A, s0",
        cat(&[&[sop2(0x2F, 16, 6, 0), s_mov_b32(17, 7)], &load_s16()]),
        &one,
        Want::Refused,
    );
    add(
        "A & 0xFFFFFFF0 (masked base)",
        cat(&[
            &[sop2(0x0E, 16, 6, 0xFF), lit(0xFFFF_FFF0), s_mov_b32(17, 7)],
            &load_s16(),
        ]),
        &one,
        Want::Refused,
    );
    add(
        "((A | 2) + s0) + s1 (aligned or, then two terms)",
        cat(&[
            &[
                sop2(0x10, 16, 6, 0x82),
                sop2(0x00, 16, 16, 0),
                sop2(0x00, 16, 16, 1),
                s_mov_b32(17, 7),
            ],
            &load_s16(),
        ]),
        &[[2, 0, 0, 0], [6, 8, 0, 0]],
        Want::Reads(|i| A((2 + i[0] + i[1]) / 4)),
    );
    {
        // v5 = A.lo + v1 (s0); v6 = A.hi + carry; v5 = (v2 << 2) + v5; load v[5:6].
        let add_co = vop3b(0x30F, 5, 106, sgpr(6), vgpr(1), 0);
        let add_ci = vop3b(0x128, 6, 106, sgpr(7), 0x80, sgpr(106));
        let lshl = vop3(0x346, 5, vgpr(2), 0x82, vgpr(5));
        let add3 = vop3(0x36D, 5, vgpr(3), vgpr(5), vgpr(4));
        let l = global_load_dword(0, 5, 125, 0);
        add(
            "VGPR pair: v_add3(s2, (s1 << 2) + (A + s0), s3)",
            vec![
                add_co[0], add_co[1], add_ci[0], add_ci[1], lshl[0], lshl[1], add3[0], add3[1],
                l[0], l[1],
            ],
            &[[0, 0, 0, 0], [8, 1, 4, 12], [4, 5, 0, 8]],
            Want::Reads(|i| A((i[0] + (i[1] << 2) + i[2] + i[3]) / 4)),
        );
    }

    // --- addresses joined by a phi ---
    let join_cases = [[0, 0, 0, 0], [1, 12, 0, 0], [7, 20, 4, 0]];
    add(
        "phi(A + 4, A + 8)",
        cat(&[
            &diamond(&[sop2(0x00, 16, 6, 0x84)], &[sop2(0x00, 16, 6, 0x88)]),
            &[s_mov_b32(17, 7)],
            &load_s16(),
        ]),
        &join_cases,
        Want::Reads(|i| A(if i[0] == 0 { 2 } else { 1 })),
    );
    add(
        "phi(A + s1, A + 8)",
        cat(&[
            &diamond(&[sop2(0x00, 16, 6, 1)], &[sop2(0x00, 16, 6, 0x88)]),
            &[s_mov_b32(17, 7)],
            &load_s16(),
        ]),
        &join_cases,
        Want::Reads(|i| A(if i[0] == 0 { 2 } else { i[1] / 4 })),
    );
    add(
        "phi(A + 4, A + 8) + s1 + s2 after the join",
        cat(&[
            &diamond(&[sop2(0x00, 16, 6, 0x84)], &[sop2(0x00, 16, 6, 0x88)]),
            &[
                sop2(0x00, 16, 16, 1),
                sop2(0x00, 16, 16, 2),
                s_mov_b32(17, 7),
            ],
            &load_s16(),
        ]),
        &join_cases,
        Want::Reads(|i| A((if i[0] == 0 { 8 } else { 4 } + i[1] + i[2]) / 4)),
    );
    add(
        "phi(A, B) (both halves joined)",
        cat(&[
            &diamond(
                &[s_mov_b32(16, 6), s_mov_b32(17, 7)],
                &[s_mov_b32(16, 8), s_mov_b32(17, 9)],
            ),
            &load_s16(),
        ]),
        &join_cases,
        Want::Reads(|i| if i[0] == 0 { B(0) } else { A(0) }),
    );
    add(
        "phi(A + 4, B + 4) low half only, high half A's",
        cat(&[
            &diamond(&[sop2(0x00, 16, 6, 0x84)], &[sop2(0x00, 16, 8, 0x84)]),
            &[s_mov_b32(17, 7)],
            &load_s16(),
        ]),
        &join_cases,
        Want::Refused,
    );
    add(
        "phi(A + 4, s1) (one arm no address)",
        cat(&[
            &diamond(&[sop2(0x00, 16, 6, 0x84)], &[s_mov_b32(16, 1)]),
            &[s_mov_b32(17, 7)],
            &load_s16(),
        ]),
        &join_cases,
        Want::Refused,
    );
    {
        // s16 = A + (s0 << 2); loop 4 times: v0 += [s16]; s16 += s1.
        let l = global_load_dword(6, 5, 16, 0);
        add(
            "loop: p = A + (s0 << 2); 4 x { sum += [p]; p += s1 }",
            vec![
                sop2(0x2F, 16, 0, 6),
                s_mov_b32(17, 7),
                s_mov_b32(18, 0x80),
                vop1(0x001, 0, 0x80),
                vop1(0x001, 5, 0x80),
                l[0],
                l[1],
                vop2(0x25, 0, vgpr(0), 6),
                sop2(0x00, 16, 16, 1),
                sop2(0x04, 17, 17, 0x80),
                sop2(0x00, 18, 18, 0x81),
                sopc(0x0A, 18, 0x84),
                sopp(0x05, -8),
            ],
            &[[0, 4, 0, 0], [2, 8, 0, 0], [5, 0, 0, 0], [3, 12, 0, 0]],
            Want::Reads(|i| {
                (0..4).fold(0u32, |s, k| s.wrapping_add(A(i[0] + k * i[1] / 4)))
            }),
        );
    }
    {
        // Loop whose step is chosen inside the loop: p += (k == s2) ? 8 : 4.
        let l = global_load_dword(6, 5, 16, 0);
        add(
            "loop: 4 x { sum += [p]; p += (k == s2 ? 8 : 4) }",
            vec![
                sop2(0x2F, 16, 0, 6),        // p = A + (s0 << 2)
                s_mov_b32(17, 7),
                s_mov_b32(18, 0x80),         // k = 0
                vop1(0x001, 0, 0x80),
                vop1(0x001, 5, 0x80),
                l[0],                        // loop:
                l[1],
                vop2(0x25, 0, vgpr(0), 6),
                sopc(0x06, 18, 2),           // s_cmp_eq_u32 k, s2
                sopp(0x05, 2),               // scc1 -> +8 arm
                sop2(0x00, 16, 16, 0x84),    // p += 4
                sopp(0x02, 1),
                sop2(0x00, 16, 16, 0x88),    // p += 8
                sop2(0x00, 18, 18, 0x81),    // k += 1
                sopc(0x0A, 18, 0x84),        // k < 4
                sopp(0x05, -11),
            ],
            &[[0, 0, 9, 0], [0, 0, 0, 0], [3, 0, 2, 0], [1, 0, 3, 0]],
            Want::Reads(|i| {
                let (mut at, mut s) = (i[0], 0u32);
                for k in 0..4 {
                    s = s.wrapping_add(A(at));
                    at += if k == i[2] { 2 } else { 1 };
                }
                s
            }),
        );
    }

    // --- a term that could be the base or the offset ---
    add(
        "A.lo + B.lo, high half A's",
        cat(&[&[sop2(0x00, 16, 6, 8), sop2(0x04, 17, 7, 0x80)], &load_s16()]),
        &one,
        Want::Refused,
    );
    add(
        "table + A.lo, high half A's",
        cat(&[&[sop2(0x00, 16, 4, 6), sop2(0x04, 17, 7, 0x80)], &load_s16()]),
        &one,
        Want::Refused,
    );
    {
        let l = global_load_dword(0, 5, 6, 0);
        add(
            "A with a VGPR offset holding B.lo",
            vec![vop1(0x001, 5, sgpr(8)), l[0], l[1]],
            &one,
            Want::Refused,
        );
    }
    add(
        "s0 - A.lo",
        cat(&[&[sop2(0x01, 16, 0, 6), s_mov_b32(17, 7)], &load_s16()]),
        &one,
        Want::Refused,
    );
    add(
        "(A + s0) - A.lo (a difference, no address)",
        cat(&[
            &[sop2(0x00, 16, 6, 0), sop2(0x01, 16, 16, 6), s_mov_b32(17, 7)],
            &load_s16(),
        ]),
        &one,
        Want::Refused,
    );
    add(
        "(A + 16) - s0",
        cat(&[
            &[sop2(0x00, 16, 6, 0x90), sop2(0x01, 16, 16, 0), s_mov_b32(17, 7)],
            &load_s16(),
        ]),
        &[[0, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0]],
        Want::Reads(|i| A((16 - i[0]) / 4)),
    );
    add(
        "A.lo + s0 with B's high half",
        cat(&[
            &[s_mov_b32(22, 7), sop2(0x00, 16, 6, 0), s_mov_b32(17, 9)],
            &load_s16(),
        ]),
        &one,
        Want::Refused,
    );
    add(
        // The high-half shortcut (older than this change): any sum holding
        // A's high half passes as its carry, so the stray addend is dropped.
        "A.lo + s0 with high half B.hi + A.hi + carry (shortcut: read as A + s0)",
        cat(&[&[sop2(0x00, 16, 6, 0), sop2(0x04, 17, 9, 7)], &load_s16()]),
        &one,
        Want::Reads(|i| A(i[0] / 4)),
    );
    {
        let l = global_load_dword(0, 5, 0, 0);
        add(
            "an address in two root constants s[0:1]",
            vec![vop1(0x001, 5, 0x80), l[0], l[1]],
            &one,
            Want::Refused,
        );
    }
    add(
        "A.hi used as the low half (s7 + s0, high s7)",
        cat(&[&[sop2(0x00, 16, 7, 0), s_mov_b32(17, 7)], &load_s16()]),
        &one,
        Want::Refused,
    );
    v
}

#[test]
fn address_shape_probes() {
    let mut wrong = Vec::new();
    for p in probes() {
        let shader = match build(&p.body, None) {
            Ok(s) => s,
            Err(why) => {
                let note = match p.want {
                    Want::Refused => "as expected",
                    Want::Reads(_) => "console reads a value; we refuse (no wrong value)",
                };
                eprintln!("PROBE | {} | REFUSED | {why} | {note}", p.name);
                continue;
            }
        };
        let rows = run(&shader, &p.cases, &[View::RawUav(1)], 0);
        let mut ok = true;
        let mut detail = Vec::new();
        for (label, inputs, got) in rows {
            match (&p.want, got) {
                (Want::Reads(f), Ok(got)) => {
                    let want = f(inputs);
                    if got != want {
                        ok = false;
                    }
                    detail.push(format!("{label} {inputs:?}: got {got:#010x} want {want:#010x}"));
                }
                (Want::Refused, Ok(got)) => {
                    ok = false;
                    detail.push(format!("{label} {inputs:?}: translated, read {got:#010x}"));
                }
                (_, Err(e)) => {
                    ok = false;
                    detail.push(format!("{label} {inputs:?}: {e}"));
                }
            }
        }
        eprintln!(
            "PROBE | {} | {} | {}",
            p.name,
            if ok { "READS RIGHT" } else { "WRONG" },
            detail.join("; ")
        );
        if !ok {
            wrong.push(p.name);
        }
    }
    assert!(wrong.is_empty(), "probes with wrong results: {wrong:#?}");
}

/// Prints one row per probe result and returns whether every read matched.
fn report<T: PartialEq + std::fmt::Debug>(
    name: &str,
    rows: Vec<(&'static str, [u32; 4], Result<T, String>)>,
    want: impl Fn([u32; 4]) -> T,
) -> bool {
    let mut ok = true;
    let mut detail = Vec::new();
    for (label, inputs, got) in rows {
        match got {
            Ok(got) => {
                let want = want(inputs);
                ok &= got == want;
                detail.push(format!("{label} {inputs:?}: got {got:08x?} want {want:08x?}"));
            }
            Err(e) => {
                ok = false;
                detail.push(format!("{label} {inputs:?}: {e}"));
            }
        }
    }
    eprintln!(
        "PROBE | {name} | {} | {}",
        if ok { "READS RIGHT" } else { "WRONG" },
        detail.join("; ")
    );
    ok
}

fn assert_reason(name: &str, why: &str, reason: &str) {
    eprintln!("PROBE | {name} | REFUSED | {why}");
    assert!(why.contains(reason), "{name}: expected {reason:?}, got {why}");
}

/// Save EXEC in s20 and leave only lanes where `vsrc == 0` active.
fn exec_if_zero(vsrc: u8) -> [u32; 2] {
    [
        0x7C00_0000 | 0xC2 << 17 | (vsrc as u32) << 9 | 0x80, // v_cmp_eq_u32 vcc, 0, vsrc
        0xBE80_0000 | 20 << 16 | 0x3C << 8 | 0x6A,           // s_and_saveexec_b32 s20, vcc_lo
    ]
}

const RESTORE_EXEC: u32 = sop2(0x10, 126, 126, 20); // s_or_b32 exec_lo, exec_lo, s20
const MASKED: &str = "an address written under an EXEC mask over another base";

/// Lanes selected by EXEC take `A + s1 (+ s3)`. The others keep the old
/// address, `old_base + s2`, where `old_base` names root SRV A or B.
fn exec_masked_select(two_terms: bool, old_base: u8, mask_vgpr: u8) -> Vec<u32> {
    let add_old = vop3b(0x30F, 5, 106, sgpr(old_base), vgpr(3), 0);
    let ci_old = vop3b(0x128, 6, 106, sgpr(old_base + 1), 0x80, sgpr(106));
    let add_a = vop3b(0x30F, 5, 106, sgpr(6), vgpr(2), 0); // v5 = A.lo + s1
    let ci_a = vop3b(0x128, 6, 106, sgpr(7), 0x80, sgpr(106)); // v6 = A.hi + carry
    let more = vop2(0x25, 5, vgpr(4), 5); // v5 = s3 + v5
    let load = global_load_dword(0, 5, 125, 0);
    let mut w = vec![add_old[0], add_old[1], ci_old[0], ci_old[1]];
    w.extend(exec_if_zero(mask_vgpr));
    w.extend([add_a[0], add_a[1], ci_a[0], ci_a[1]]);
    if two_terms {
        w.push(more);
    }
    w.extend([RESTORE_EXEC, load[0], load[1]]);
    w
}

const MASK_CASES: [[u32; 4]; 4] = [[0, 8, 20, 4], [0, 0, 0, 0], [1, 8, 20, 4], [5, 16, 12, 8]];

/// One computed term: the shape the tracer took before the change too.
#[test]
fn exec_masked_address_one_term() {
    let name = "exec-masked A + s1 over B + s2";
    match build(&exec_masked_select(false, 8, 1), None) {
        Err(why) => assert_reason(name, &why, MASKED),
        Ok(shader) => {
            let rows = run(&shader, &MASK_CASES, &[View::RawUav(1)], 0);
            let ok = report(name, rows, |i| {
                if i[0] == 0 { A(i[1] / 4) } else { B(i[2] / 4) }
            });
            assert!(ok, "the lanes the mask leaves out read the wrong buffer");
        }
    }
}

/// Two computed terms: a shape only the wider tracer takes.
#[test]
fn exec_masked_address_two_terms() {
    let name = "exec-masked (A + s1) + s3 over B + s2";
    match build(&exec_masked_select(true, 8, 1), None) {
        Err(why) => assert_reason(name, &why, MASKED),
        Ok(shader) => {
            let rows = run(&shader, &MASK_CASES, &[View::RawUav(1)], 0);
            let ok = report(name, rows, |i| {
                if i[0] == 0 { A((i[1] + i[3]) / 4) } else { B(i[2] / 4) }
            });
            assert!(ok, "the lanes the mask leaves out read the wrong buffer");
        }
    }
}

fn table_range(kind: ResourceType, at: u32, stride: u32) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride,
        dwords: 8,
        user_data: 4,
        slot: 1,
    }
}

fn table_probe(
    name: &str,
    body: &[u32],
    cases: &[[u32; 4]],
    want: impl Fn([u32; 4]) -> u32,
) -> Result<(), String> {
    let tables = Tables {
        used: vec![
            table_range(ResourceType::Srv, 0, 8),
            table_range(ResourceType::Srv, 8, 16),
            table_range(ResourceType::Uav, 24, 0),
        ],
        out: 96,
    };
    let shader = build(body, Some(&tables))?;
    let decoy: Vec<u32> = (0..DWORDS).map(|i| 0xDEC0_0000 + i).collect();
    let x: Vec<u32> = (0..DWORDS).map(|i| 0x1111_0000 + i).collect();
    let y: Vec<u32> = (0..DWORDS).map(|i| 0x2222_0000 + i).collect();
    let views = [
        View::Buffer {
            data: &decoy,
            stride: Stride::Structured(8),
        },
        View::Buffer {
            data: &x,
            stride: Stride::Structured(16),
        },
        View::Buffer {
            data: &y,
            stride: Stride::Structured(16),
        },
        View::RawUav(1),
    ];
    let rows = run(&shader, cases, &views, 0);
    let ok = report(name, rows, want);
    assert!(ok, "the descriptor was read with another range's record size");
    Ok(())
}

/// A table descriptor whose address two edges join: `table + 32` or
/// `table + 64`, both in a range of structured SRVs with 16-byte records.
/// The range at the table's start holds 8-byte records. Element `s1` of the
/// chosen buffer is read with an indexed load.
#[test]
fn table_descriptor_joined_by_a_phi_keeps_its_range() {
    let l = s_load_x4(24, 20, 125, 0);
    let b = buffer_load_dword_idx(0, 2, 24); // index v2 = s1
    let body = [
        diamond(&[sop2(0x00, 20, 4, 0xA0)], &[sop2(0x00, 20, 4, 0xC0)]),
        vec![s_mov_b32(21, 0x80), l[0], l[1], b[0], b[1]],
    ]
    .concat();
    table_probe(
        "table: phi(T + 32, T + 64), 16-byte records, element s1",
        &body,
        &[[0, 0, 0, 0], [0, 1, 0, 0], [1, 2, 0, 0], [1, 3, 0, 0]],
        |i| (if i[0] == 0 { 0x2222_0000 } else { 0x1111_0000 }) + i[1] * 4,
    )
    .expect("the joined descriptor stays in one table range");
}

/// The same table descriptor through two computed terms: `(T + 32 + s2) + s3`.
#[test]
fn table_descriptor_with_two_terms_keeps_its_range() {
    let l = s_load_x4(24, 20, 125, 0);
    let b = buffer_load_dword_idx(0, 2, 24);
    let body = vec![
        sop2(0x00, 20, 4, 0xA0), // T + 32
        sop2(0x00, 20, 20, 2),   // + s2
        sop2(0x00, 20, 20, 3),   // + s3
        s_mov_b32(21, 0x80),
        l[0],
        l[1],
        b[0],
        b[1],
    ];
    table_probe(
        "table: (T + 32 + s2) + s3, 16-byte records, element s1",
        &body,
        &[[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 32, 0], [0, 3, 16, 16]],
        |i| (if i[2] + i[3] == 32 { 0x2222_0000 } else { 0x1111_0000 }) + i[1] * 4,
    )
    .expect("the descriptor with two computed terms stays in one table range");
}

/// The one-term form of the same descriptor, which the tracer took before:
/// `(T + 32) + s2`.
#[test]
fn table_descriptor_with_one_term_keeps_its_range() {
    let l = s_load_x4(24, 20, 125, 0);
    let b = buffer_load_dword_idx(0, 2, 24);
    let body = vec![
        sop2(0x00, 20, 4, 0xA0),
        sop2(0x00, 20, 20, 2),
        s_mov_b32(21, 0x80),
        l[0],
        l[1],
        b[0],
        b[1],
    ];
    let name = "table: (T + 32) + s2, 16-byte records, element s1";
    if let Err(why) = table_probe(
        name,
        &body,
        &[[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 32, 0], [0, 3, 32, 0]],
        |i| (if i[2] == 32 { 0x2222_0000 } else { 0x1111_0000 }) + i[1] * 4,
    ) {
        eprintln!("PROBE | {name} | REFUSED | {why}");
    }
}

/// A root UAV written through `(C + s0) + s1` with a VGPR offset `s2`:
/// the console writes dword `(s0 + s1 + s2) / 4` of C.
#[test]
fn root_uav_store_through_two_terms_and_a_vgpr_offset() {
    // global_store_dword v7, v3 (= s2), s[16:17]
    let store = [
        0x37 << 26 | 0x1C << 18 | (2 << 14),
        16u32 << 16 | 7 << 8 | 3,
    ];
    let body = vec![
        sop2(0x00, 16, 10, 0),
        sop2(0x00, 16, 16, 1),
        sop2(0x04, 17, 11, 0x80),
        vop1(0x001, 7, 0xFF),
        0x00C0_FFEE,
        store[0],
        store[1],
        vop1(0x001, 0, 0x80),
    ];
    let name = "store: (C + s0) + s1, VGPR offset s2";
    match build(&body, None) {
        Err(why) => eprintln!("PROBE | {name} | REFUSED | {why}"),
        Ok(shader) => {
            let cases = [[0, 0, 0, 0], [4, 8, 0, 0], [8, 4, 12, 0], [16, 0, 4, 0]];
            let rows = run(&shader, &cases, &[View::RawUav(1)], usize::MAX);
            let ok = report(name, rows, |i| ((i[0] + i[1] + i[2]) / 4) << 24 | 0xC0_FFEE);
            assert!(ok, "the store landed on another dword");
        }
    }
}

/// The two edges name buffers with different record sizes. No single range
/// supplies a stride for the joined indexed load.
#[test]
fn table_descriptor_joined_by_a_phi_crosses_ranges() {
    let l = s_load_x4(24, 20, 125, 0);
    let b = buffer_load_dword_idx(0, 2, 24); // index v2 = s1
    let body = [
        diamond(&[sop2(0x00, 20, 4, 0x80)], &[sop2(0x00, 20, 4, 0xA0)]),
        vec![s_mov_b32(21, 0x80), l[0], l[1], b[0], b[1]],
    ]
    .concat();
    let name = "table: phi(T, T + 32), 8- and 16-byte records, element s1";
    let why = table_probe(name, &body, &[[0, 2, 0, 0], [1, 2, 0, 0]], |i| {
        if i[0] == 0 { 0x1111_0000 + i[1] * 4 } else { 0xDEC0_0000 + i[1] * 2 }
    })
    .expect_err("the phi names two table ranges");
    assert_reason(name, &why, "a table descriptor at an address that does not settle its range");
}

/// Runs two lanes of one group and compares their outputs in thread-id order.
fn two_lane_probe(name: &str, body: &[u32], want: impl Fn([u32; 4]) -> [u32; 2]) {
    let shader = build_with_threads(body, None, 2)
        .unwrap_or_else(|why| panic!("{name}: {why}"));
    let cases = [[0, 8, 20, 4], [0, 16, 12, 8]];
    let rows = run_with_readback(&shader, &cases, &[View::RawUav(2)], None, |o| [o[0][0], o[0][1]]);
    assert!(report(name, rows, want), "a lane read the wrong dword");
}

/// Lane 0 takes `A + s1` and lane 1 keeps `A + s2`.
#[test]
fn exec_masked_address_one_term_keeps_the_same_base() {
    two_lane_probe(
        "exec-masked A + s1 over A + s2",
        &exec_masked_select(false, 6, 0),
        |i| [A(i[1] / 4), A(i[2] / 4)],
    );
}

/// Lane 0 takes `(A + s1) + s3` and lane 1 keeps `A + s2` across both writes.
#[test]
fn exec_masked_address_two_terms_keeps_the_same_base() {
    two_lane_probe(
        "exec-masked (A + s1) + s3 over A + s2",
        &exec_masked_select(true, 6, 0),
        |i| [A((i[1] + i[3]) / 4), A(i[2] / 4)],
    );
}

/// Both lanes load through the unmasked address `A + s3`. The separate VGPR
/// byte offset is plain data: lane 0 takes `s2` and lane 1 keeps `s1`.
#[test]
fn exec_masked_data_is_a_plain_offset() {
    let mut body = vec![
        sop2(0x00, 16, 6, 3),    // s16 = A.lo + s3
        sop2(0x04, 17, 7, 0x80), // s17 = A.hi + carry
        vop1(0x001, 5, vgpr(2)), // v5 = s1
    ];
    body.extend(exec_if_zero(0));
    body.extend([vop1(0x001, 5, vgpr(3)), RESTORE_EXEC]); // v5 = s2 in lane 0
    body.extend(global_load_dword(0, 5, 16, 0));
    two_lane_probe("exec-masked data offset beside A + s3", &body, |i| {
        [A((i[2] + i[3]) / 4), A((i[1] + i[3]) / 4)]
    });
}

const SENTINEL: u32 = 0x51_E171;

/// Seed the output in all lanes, then replace it only under the address
/// write's EXEC. The wave64 form uses LDS so wave32 hosts need console flow.
fn same_mask_program(threads: u32, counted_loop: bool, load_after_restore: bool) -> NativeProgram {
    let wave64 = threads == 64;
    let store = output_store(8, threads);
    let mut body = vec![vop1(0x001, 8, 0xFF), SENTINEL]; // v8 = sentinel
    if wave64 {
        body.extend([
            0xD834_0000, 0x0000_0807, // ds_write_b32 v7, v8
            0xD8D8_0000, 0x0800_0007, // ds_read_b32 v8, v7
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ]);
    }
    body.extend(store);
    body.extend([
        vop1(0x001, 5, vgpr(2)), // old v5 = s1
        vop1(0x001, 6, vgpr(3)), // old v6 = s2
    ]);
    if wave64 {
        body.extend([
            0x7D88_00A1, // v_cmp_gt_u32 vcc, 33, v0
            0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        ]);
    } else {
        body.extend(exec_if_zero(0));
    }
    let skip = body.len();
    if wave64 {
        body.push(0); // s_cbranch_execz restore
    }
    if counted_loop {
        body.push(s_mov_b32(16, 0x82)); // s16 = 2
    }
    let loop_start = body.len();
    if counted_loop && !load_after_restore {
        body.push(vop2(0x26, 9, vgpr(5), 6)); // v9 = v5 - v6, use the loop-carried data
    }
    body.extend([
        vop1(0x001, 5, sgpr(6)), // v5 = A.lo
        vop1(0x001, 6, sgpr(7)), // v6 = A.hi
    ]);
    let load = global_load_dword(8, 5, 125, 12); // v8 = A(3)
    if !load_after_restore {
        body.extend(load);
        if counted_loop {
            body.extend([
                vop2(0x25, 8, vgpr(9), 8), // add the difference (zero on the second trip)
                vop1(0x001, 5, vgpr(2)),  // reuse both halves as data
                vop1(0x001, 6, vgpr(2)),
            ]);
        }
        body.extend(store);
    }
    if counted_loop {
        body.push(sop2(0x01, 16, 16, 0x81)); // s_sub_u32 s16, s16, 1
        body.push(sopc(0x06, 16, 0x80)); // s_cmp_eq_u32 s16, 0
        body.push(sopp(0x04, loop_start as i16 - body.len() as i16 - 1)); // s_cbranch_scc0 loop
    }
    if wave64 {
        body[skip] = sopp(0x08, (body.len() - skip - 1) as i16);
    }
    body.push(if wave64 {
        0x88FE_147E // s_or_b64 exec, exec, s[20:21]
    } else {
        RESTORE_EXEC
    });
    if load_after_restore {
        body.extend(load);
        body.extend(store);
    }
    let mut native = probe_program(&body, None, threads);
    if wave64 {
        native.wave64 = true;
        let StageIo::Cs(io) = &mut native.io else {
            unreachable!()
        };
        io.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

fn same_mask_probe(name: &str, threads: u32, counted_loop: bool) {
    let console = threads == 64;
    let (target, wave_size) = if console {
        (
            super::super::Target {
                lanes_min: 32,
                lanes_max: 32,
            },
            Some(32),
        )
    } else {
        (super::super::Target::ANY, None)
    };
    let native = same_mask_program(threads, counted_loop, false);
    let shader = build_program(&native, target).unwrap_or_else(|why| panic!("{name}: {why}"));
    let rows = run_with_readback(
        &shader,
        &[[0, 11, 4, 0]],
        &[View::RawUav(threads)],
        wave_size,
        |o| o.into_iter().next().unwrap(),
    );
    if rows.is_empty() {
        eprintln!("PROBE | {name} | TRANSLATED | no adapter supports the required wave size");
        return;
    }
    let active = if console { 33 } else { 1 };
    let want = |_| {
        (0..threads)
            .map(|lane| if lane < active { A(3) } else { SENTINEL })
            .collect()
    };
    assert!(report(name, rows, want), "a lane read the wrong dword or lost its sentinel");
}

#[test]
fn exec_address_loaded_under_the_same_mask() {
    same_mask_probe("same EXEC", 2, false);
}

#[test]
fn exec_address_loaded_under_the_same_mask_in_a_loop() {
    same_mask_probe("same EXEC in a loop", 2, true);
}

#[test]
fn exec_address_loaded_under_the_same_mask_in_a_console_loop() {
    same_mask_probe("same EXEC in a console loop", 64, true);
}

#[test]
fn exec_address_loaded_after_the_loop_restores_exec() {
    let native = same_mask_program(2, true, true);
    let why = build_program(&native, super::super::Target::ANY)
        .expect_err("the restored lanes have a data value in the address VGPRs");
    assert_reason("loop restores EXEC before the load", &why, MASKED);
}
