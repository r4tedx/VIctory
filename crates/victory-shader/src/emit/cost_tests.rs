//! What the translation's own code costs on a GPU: synthetic GFX10 compute
//! programs made of one thing the lowering adds to every use (here: the
//! write marks of stores through unordered buffer views, `emit` "Write
//! marks"), dispatched over many thread groups between two timestamp
//! queries. The benchmark is ignored by default and names its adapter
//! (`VICTORY_BENCH_ADAPTER`: `warp`, or part of an adapter's description),
//! so it never runs on a card nobody chose.
//!
//! A time from WARP says how much work the CPU rasteriser has; it does not
//! predict a graphics card, whose compiler and memory system differ. Times
//! compare two translations of one program on one adapter, nothing else.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
/// Inline constant 0.
const ZERO: u16 = 128;
const V_LSHLREV_B32: u32 = 0x1A;
const BUFFER_STORE_DWORD: u32 = 0x1C;
/// Lanes of a thread group.
const LANES: u32 = 32;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dword*` (`op` 2: four dwords) at `s[base:base+1] + offset`.
const fn s_load(op: u32, sdata: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
        125 << 25 | offset,
    ]
}

const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
}

/// MUBUF `op` of `vdata` at `vaddr + byte_off` through `s[srsrc:srsrc+3]`.
const fn mubuf(op: u32, vdata: u8, vaddr: u8, srsrc: u8, byte_off: u32) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | 1 << 12 | (byte_off & 0xFFF),
        0x80 << 24 | (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// A compute program of 32 lanes that stores `lane << 2` to `stores` rows
/// of an unordered buffer view from a table, each lane at its own dword of
/// the row: row `k` starts at dword `32 k`.
fn stores(stores: u32, view_marks: bool) -> HostShader {
    assert!(stores <= 32, "the store's offset has 12 bits");
    let mut w = vec![s_mov_b32(5, ZERO as u8)]; // s[4:5]: the table at s4
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the view
    w.push(vop2(V_LSHLREV_B32, 3, 128 + 2, 0)); // v3 = lane << 2
    w.push(S_WAITCNT_ALL);
    for k in 0..stores {
        w.extend(mubuf(BUFFER_STORE_DWORD, 3, 3, 8, 4 * LANES * k));
    }
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 1,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [LANES, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks,
    })
    .expect("lower");
    assert_eq!(l.view_marks, view_marks);
    let shader = HostShader {
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    shader
}

/// Dispatches `shader` (made by [`stores`] with `rows` stores) over
/// `groups` thread groups `runs` times, checks what it wrote each time, and
/// returns the dispatch's GPU seconds, least first. The view is marked from
/// bit 100, its first byte at the start of a 64 KiB page.
fn time(host: &Host, shader: &HostShader, rows: u32, groups: u32, runs: usize) -> Vec<f64> {
    let (root, pso) = host.compute_pipeline(shader).expect("pipeline");
    let ud = vec![0u32; shader.user_data_dwords() as usize];
    let want: Vec<u32> = (0..rows * LANES).map(|i| (i % LANES) << 2).collect();
    let mut seconds = Vec::new();
    for _ in 0..runs {
        let mut roots = vec![(0, Root::Constants(ud.clone()))];
        if shader.view_marks {
            roots.push((
                1,
                Root::View {
                    srv: true,
                    data: vec![100, 0],
                },
            ));
            roots.push((2, Root::Uav(8)));
        }
        let (out, s) = host
            .dispatch_timed(
                &root,
                &pso,
                &Dispatch {
                    roots,
                    views: &[View::RawUav(rows * LANES)],
                    samplers: &[],
                    groups: [groups, 1, 1],
                },
            )
            .expect("dispatch");
        assert_eq!(out[0], want, "the stores");
        if shader.view_marks {
            let marked: Vec<usize> = (0..out[1].len() * 32)
                .filter(|&b| out[1][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, [100], "the page marked");
        }
        seconds.push(s);
    }
    seconds.sort_by(f64::total_cmp);
    seconds
}

/// The stores land and mark their page under the timed dispatch too (the
/// benchmark's own check, on WARP, at a size that takes no time).
#[test]
fn timed_stores_land_and_mark() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("no SM 6.6 WARP device; skipped");
        return;
    };
    for marks in [false, true] {
        let seconds = time(&host, &stores(4, marks), 4, 2, 1);
        assert!(seconds[0] >= 0.0);
    }
}

#[test]
#[ignore = "benchmark: VICTORY_BENCH_ADAPTER=warp cargo test --release -p victory-shader \
            gpu_time_of_marked_stores -- --ignored --nocapture"]
fn gpu_time_of_marked_stores() {
    let name = std::env::var("VICTORY_BENCH_ADAPTER")
        .expect("VICTORY_BENCH_ADAPTER names the adapter: warp, or part of its description");
    let groups: u32 = std::env::var("VICTORY_BENCH_GROUPS")
        .ok()
        .and_then(|g| g.parse().ok())
        .unwrap_or(4096);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (host, adapter) = Host::named(&name).expect("the adapter");
    println!("adapter: {adapter}; {groups} groups of {LANES} lanes; least and median of 7 runs");
    for rows in [1, 8, 32] {
        for marks in [false, true] {
            let shader = stores(rows, marks);
            // One run first: the driver compiles and pages in.
            time(&host, &shader, rows, groups, 1);
            let s = time(&host, &shader, rows, groups, 7);
            println!(
                "{rows:>2} stores, marks {}: {:>7} bytes of DXIL, {:>9.3} ms least, {:>9.3} ms median",
                if marks { "on " } else { "off" },
                shader.dxil.len(),
                s[0] * 1e3,
                s[s.len() / 2] * 1e3,
            );
        }
    }
}
