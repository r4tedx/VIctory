//! Ray intersection (`image_bvh_intersect_ray`, `image_bvh64_intersect_ray`):
//! hand-assembled GFX10 programs over synthetic nodes, dispatched on the
//! default adapter and on WARP and compared with a CPU model. Everything
//! here is synthetic: no node, ray or program comes from a title, and
//! nothing was compared with console hardware.
//!
//! The CPU model ([`cpu`]) is written from the RDNA2 ISA (8.2.10) and AMD's
//! GPURT (MIT; `IntersectCommon.hlsl` and `shadersClean/common/gfx10`). The
//! geometric expectations of `fixtures_*` are given as numbers, apart from
//! the model.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::lower::bvh::{self as why, VIEW_DWORD1, VIEW_DWORD3};
use crate::ShaderError;
use crate::gfx10::{Fields, Operand, decode};
use crate::native::{RootParam, RootType, StageIo};

const S_ENDPGM: u32 = 0xBF81_0000;
const NONE: u32 = u32::MAX;
/// SGPRs of the three descriptors, and the heap index of each view.
const BVH: u8 = 4;
const RAYS: u8 = 12;
const OUT: u8 = 16;
/// What the result VGPRs hold before the instruction.
const PRESET: u32 = 0xABCD_0000;
const THREADS: u32 = 64;

/// The CPU model: what the instruction returns for a node in a view.
mod cpu {
    pub const NONE: u32 = u32::MAX;

    #[derive(Debug, Clone, Copy)]
    pub struct Ray {
        pub extent: f32,
        pub origin: [f32; 3],
        pub dir: [f32; 3],
        pub inv: [f32; 3],
    }

    /// A half as a float, exactly (denormals, infinities and NaNs included).
    pub fn unhalf(h: u16) -> f32 {
        let (sign, exp, frac) = (h >> 15, h >> 10 & 0x1F, u32::from(h & 0x3FF));
        let magnitude = match exp {
            0 => frac as f32 * (-24f32).exp2(),
            31 => f32::from_bits(0x7F80_0000 | frac << 13),
            e => (1.0 + frac as f32 / 1024.0) * (f32::from(e) - 15.0).exp2(),
        };
        if sign == 1 { -magnitude } else { magnitude }
    }

    /// The byte offset of the node `pointer` names, when the node lies
    /// whole in a view of `view` bytes: 64-bit arithmetic throughout.
    pub fn node_offset(pointer: u64, view: u64, node: u64) -> Option<u64> {
        let offset = u128::from(pointer >> 3) * 64;
        (offset + u128::from(node) <= u128::from(view)).then_some(offset as u64)
    }

    /// The four result dwords for `pointer` in `view` (dwords).
    pub fn intersect(view: &[u32], pointer: u64, ray: &Ray, sort: bool, grow: u32) -> [u32; 4] {
        let ty = pointer & 7;
        let node = if ty == 5 { 128 } else { 64 };
        let Some(at) = node_offset(pointer, view.len() as u64 * 4, node) else {
            return [NONE; 4];
        };
        let w = &view[at as usize / 4..][..node as usize / 4];
        let f = |k: usize| f32::from_bits(w[k]);
        match ty {
            5 => {
                let corners = std::array::from_fn(|n| std::array::from_fn(|j| f(4 + 6 * n + j)));
                boxes([w[0], w[1], w[2], w[3]], corners, ray, sort, grow)
            }
            4 => {
                let half = |k: usize| unhalf((w[4 + k / 2] >> (16 * (k % 2))) as u16);
                let corners = std::array::from_fn(|n| std::array::from_fn(|j| half(6 * n + j)));
                boxes([w[0], w[1], w[2], w[3]], corners, ray, sort, grow)
            }
            0 | 1 => {
                let v = |n: usize| [f(3 * n), f(3 * n + 1), f(3 * n + 2)];
                let (a, b, c) = if ty == 0 {
                    (v(0), v(1), v(2))
                } else {
                    (v(1), v(3), v(2))
                };
                triangle(a, b, c, (w[15] >> (8 * ty)) as u8, ray)
            }
            _ => [NONE; 4],
        }
    }

    fn boxes(
        children: [u32; 4],
        corners: [[f32; 6]; 4],
        ray: &Ray,
        sort: bool,
        grow: u32,
    ) -> [u32; 4] {
        let mut child = children;
        let mut key = [0f32; 4];
        for n in 0..4 {
            let (mut near, mut far) = (f32::NEG_INFINITY, f32::INFINITY);
            let mut nan = [true; 2];
            for axis in 0..3 {
                let lo = (corners[n][axis] - ray.origin[axis]) * ray.inv[axis];
                let hi = (corners[n][3 + axis] - ray.origin[axis]) * ray.inv[axis];
                let (enter, leave) = if ray.inv[axis] >= 0.0 {
                    (lo, hi)
                } else {
                    (hi, lo)
                };
                // A maximum or minimum with a NaN is the other operand.
                near = if axis == 0 { enter } else { near.max(enter) };
                far = if axis == 0 { leave } else { far.min(leave) };
                nan = [nan[0] && enter.is_nan(), nan[1] && leave.is_nan()];
            }
            let (mut from, mut to) = (near.max(0.0), far.min(ray.extent));
            if nan[0] || nan[1] {
                (from, to) = (f32::INFINITY, f32::NEG_INFINITY);
            }
            // (A NaN end is no hit.)
            let hit = from <= to * (1.0 + grow as f32 * 5.960_464_5e-8);
            if !hit {
                child[n] = NONE;
            }
            key[n] = from;
        }
        if sort {
            for (a, b) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
                if child[b] != NONE && key[b] < key[a] || child[a] == NONE {
                    child.swap(a, b);
                    key.swap(a, b);
                }
            }
        }
        child
    }

    fn triangle(a: [f32; 3], b: [f32; 3], c: [f32; 3], id: u8, ray: &Ray) -> [u32; 4] {
        let sub = |x: [f32; 3], y: [f32; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
        let cross = |x: [f32; 3], y: [f32; 3]| {
            [
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ]
        };
        let dot = |x: [f32; 3], y: [f32; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        let (e1, e2, e3) = (sub(b, a), sub(c, a), sub(ray.origin, a));
        let (s1, s2) = (cross(ray.dir, e2), cross(e3, e1));
        let (mut t_num, mut denom) = (dot(e2, s2), dot(s1, e1));
        let (i_num, j_num) = (dot(e3, s1), dot(ray.dir, s2));
        let (t, u, v) = (t_num / denom, i_num / denom, j_num / denom);
        if u < 0.0 || u > 1.0 || v < 0.0 || u + v > 1.0 || t < 0.0 {
            (t_num, denom) = (f32::INFINITY, 1.0);
        }
        let coordinates = [denom - i_num - j_num, i_num, j_num, 0.0];
        let pick = |sel: u8| coordinates[usize::from(sel & 3)];
        [t_num, denom, pick(id), pick(id >> 2)].map(f32::to_bits)
    }
}

/// A float as a half; it must be one exactly.
fn half(f: f32) -> u32 {
    static HALVES: std::sync::OnceLock<std::collections::HashMap<u32, u16>> =
        std::sync::OnceLock::new();
    let halves = HALVES.get_or_init(|| {
        (0..=u16::MAX)
            .map(|h| (cpu::unhalf(h).to_bits(), h))
            .collect()
    });
    let h = halves
        .get(&f.to_bits())
        .unwrap_or_else(|| panic!("{f} is no half"));
    u32::from(*h)
}

/// The fields of the instruction as encoded.
#[derive(Debug, Clone, Copy)]
struct Mimg {
    wide: bool,
    a16: bool,
    /// One VGPR per address, listed (NSA), in an order that is not theirs.
    nsa: bool,
    dmask: u32,
    dim: u32,
    unorm: bool,
    r128: bool,
    d16: bool,
    tfe: bool,
    lwe: bool,
}

impl Mimg {
    const fn of(wide: bool, a16: bool) -> Self {
        Mimg {
            wide,
            a16,
            nsa: false,
            dmask: 0xF,
            dim: 0,
            unorm: true,
            r128: true,
            d16: false,
            tfe: false,
            lwe: false,
        }
    }

    /// Address VGPRs: RDNA2 ISA, table 48.
    fn operands(self) -> usize {
        usize::from(self.wide) + if self.a16 { 8 } else { 11 }
    }

    /// The VGPR of each address operand: v4 on, shuffled with NSA.
    fn regs(self) -> Vec<u8> {
        let n = self.operands() as u8;
        (0..n)
            .map(|k| 4 + if self.nsa { (k * 5) % n } else { k })
            .collect()
    }

    /// The instruction returning in `data..data + 4` from descriptor `rsrc`.
    fn words(self, data: u8, rsrc: u8) -> Vec<u32> {
        let regs = self.regs();
        let extra = if self.nsa {
            (regs.len() - 1).div_ceil(4)
        } else {
            0
        };
        let bit = |b: bool, n: u32| u32::from(b) << n;
        let mut w = vec![
            0xF000_0000
                | (0x66 | u32::from(self.wide)) << 18
                | 1
                | (extra as u32) << 1
                | self.dim << 3
                | self.dmask << 8
                | bit(self.unorm, 12)
                | bit(self.r128, 15)
                | bit(self.tfe, 16)
                | bit(self.lwe, 17),
            u32::from(regs[0])
                | u32::from(data) << 8
                | u32::from(rsrc / 4) << 16
                | bit(self.a16, 30)
                | bit(self.d16, 31),
        ];
        w.resize(2 + extra, 0);
        if self.nsa {
            for (k, &r) in regs[1..].iter().enumerate() {
                w[2 + k / 4] |= u32::from(r) << (8 * (k % 4));
            }
        }
        w
    }

    /// The address operands of a ray, in the instruction's order.
    fn ray(self, pointer: u64, ray: &cpu::Ray) -> Vec<u32> {
        let mut w = vec![pointer as u32];
        if self.wide {
            w.push((pointer >> 32) as u32);
        }
        w.push(ray.extent.to_bits());
        w.extend(ray.origin.map(f32::to_bits));
        if self.a16 {
            let h: Vec<u32> = ray.dir.iter().chain(&ray.inv).map(|&f| half(f)).collect();
            w.extend(h.chunks(2).map(|p| p[0] | p[1] << 16));
        } else {
            w.extend(ray.dir.map(f32::to_bits));
            w.extend(ray.inv.map(f32::to_bits));
        }
        assert_eq!(w.len(), self.operands());
        w
    }
}

/// Where dwords 1 and 3 of the BVH descriptor come from.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Flags {
    /// The descriptor as the table load gives it.
    Loaded,
    /// Set by the code.
    Set(u32, u32),
    /// Dword 1 from a user SGPR: no constant.
    Computed,
    /// Dword 1 the bit reverse of 1 (box sorting, no growing), dword 3
    /// this constant.
    Reversed(u32),
}

#[derive(Debug, Clone, Copy)]
struct Form {
    mimg: Mimg,
    flags: Flags,
    /// Run the instruction only in lanes whose record says so (EXEC).
    masked: bool,
    /// Dword 0 of the descriptor is set by the code.
    constant_base: bool,
}

impl Form {
    const fn of(wide: bool, a16: bool) -> Self {
        Form {
            mimg: Mimg::of(wide, a16),
            flags: Flags::Loaded,
            masked: false,
            constant_base: false,
        }
    }

    fn sorted(self, sort: bool) -> Self {
        let flags = Flags::Set(
            VIEW_DWORD1 & !(1 << 31) | u32::from(sort) << 31,
            VIEW_DWORD3,
        );
        Form { flags, ..self }
    }
}

fn s_load_x4(dst: u8, offset: u32) -> [u32; 2] {
    [0xF408_0001 | u32::from(dst) << 6, 0xFA00_0000 | offset]
}

fn s_mov_literal(dst: u8, value: u32) -> [u32; 2] {
    [0xBE80_03FF | u32::from(dst) << 16, value]
}

fn v_mov_literal(dst: u8, value: u32) -> [u32; 2] {
    [0x7E00_02FF | u32::from(dst) << 17, value]
}

fn v_mov_sgpr(dst: u8, src: u8) -> u32 {
    0x7E00_0200 | u32::from(dst) << 17 | u32::from(src)
}

fn load_x4(dst: u8, addr: u8, rsrc: u8, offset: u32) -> [u32; 2] {
    let w1 = 0x8000_0000 | u32::from(rsrc / 4) << 16 | u32::from(dst) << 8 | u32::from(addr);
    [0xE038_1000 | offset, w1]
}

fn store_x4(src: u8, addr: u8, rsrc: u8) -> [u32; 2] {
    let w1 = 0x8000_0000 | u32::from(rsrc / 4) << 16 | u32::from(src) << 8 | u32::from(addr);
    [0xE078_1000, w1]
}

/// The descriptor loads: the BVH, the rays and the output from heap
/// entries 0, 1 and 2 of the table in user SGPR 0; then lane t of group g
/// has its record's byte offset in v1 and its result's in v2.
fn prologue() -> Vec<u32> {
    let mut w = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
    ];
    w.extend(s_load_x4(BVH, 0x00));
    w.extend(s_load_x4(RAYS, 0x20));
    w.extend(s_load_x4(OUT, 0x40));
    w.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x9308_FF01,
        THREADS * 64, // s_mul_i32 s8, s1, 64 * threads
        0x9309_FF01,
        THREADS * 16, // s_mul_i32 s9, s1, 16 * threads
        0x3402_0086,  // v_lshlrev_b32 v1, 6, v0
        0x3404_0084,  // v_lshlrev_b32 v2, 4, v0
        0x4A02_0208,  // v_add_nc_u32 v1, s8, v1
        0x4A04_0409,  // v_add_nc_u32 v2, s9, v2
    ]);
    w
}

/// A wave64 compute program of `THREADS` lanes.
fn finish(words: &[u32]) -> crate::NativeProgram {
    let mut p = crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64: true,
        vgprs: 24,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(crate::native::ComputeIo {
            rsrc2: 0,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
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
        threads: [THREADS, 1, 1],
    });
    p
}

/// Lane t of group g intersects the ray in record `g * 64 + t` of the ray
/// buffer (16 dwords: VGPR v4 + k of the instruction's addresses is dword
/// k, dword 12 whether the lane runs it under `masked`) and stores the four
/// result dwords.
fn program(form: &Form) -> crate::NativeProgram {
    let mut w = prologue();
    match form.flags {
        Flags::Loaded => {}
        Flags::Set(d1, d3) => {
            w.extend(s_mov_literal(BVH + 1, d1));
            w.extend(s_mov_literal(BVH + 3, d3));
        }
        // s_mov_b32 s5, s0
        Flags::Computed => w.push(0xBE80_0300 | u32::from(BVH + 1) << 16),
        Flags::Reversed(d3) => {
            w.push(0xBE80_0B81 | u32::from(BVH + 1) << 16); // s_brev_b32 s5, 1
            w.extend(s_mov_literal(BVH + 3, d3));
        }
    }
    if form.constant_base {
        w.extend(s_mov_literal(BVH, 0));
    }
    for k in 0..3 {
        w.extend(load_x4(4 + 4 * k, 1, RAYS, 16 * u32::from(k)));
    }
    w.extend(load_x4(20, 1, RAYS, 48));
    w.push(0xBF8C_0000); // s_waitcnt vmcnt(0)
    for k in 0..4 {
        w.extend(v_mov_literal(16 + k, PRESET + u32::from(k)));
    }
    if form.masked {
        w.extend([
            0x7D8A_2880, // v_cmp_ne_u32 vcc, 0, v20
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        ]);
    }
    w.extend(form.mimg.words(16, BVH));
    if form.masked {
        w.push(0x88FE_0A7E); // s_or_b64 exec, exec, s[10:11]
    }
    w.extend(store_x4(16, 2, OUT));
    w.push(S_ENDPGM);
    finish(&w)
}

/// One ray of a dispatch.
#[derive(Debug, Clone, Copy)]
struct Case {
    pointer: u64,
    ray: cpu::Ray,
    /// Whether the lane runs the instruction (with [`Form::masked`]).
    runs: bool,
}

fn ray(origin: [f32; 3], dir: [f32; 3], extent: f32) -> cpu::Ray {
    cpu::Ray {
        extent,
        origin,
        dir,
        inv: dir.map(|d| 1.0 / d),
    }
}

/// The ray buffer of `cases`, padded to whole groups with lanes that do
/// not run.
fn records(form: &Form, cases: &[Case]) -> Vec<u32> {
    let regs = form.mimg.regs();
    let mut out = vec![0; cases.len().next_multiple_of(THREADS as usize) * 16];
    for (case, record) in cases.iter().zip(out.chunks_mut(16)) {
        for (operand, &reg) in form.mimg.ray(case.pointer, &case.ray).iter().zip(&regs) {
            record[usize::from(reg - 4)] = *operand;
        }
        record[12] = u32::from(case.runs);
    }
    out
}

/// The stages [`super::translate`] runs, on a program without a container.
fn emit_for(native: &crate::NativeProgram) -> Result<super::HostShader, ShaderError> {
    let c = crate::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = crate::structure::structure(&p);
    let res = crate::resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = crate::io::raise(&p, native)?;
    let st = st?;
    super::emit(&c, native, &p, &st, &io, &res)
}

fn translate(p: &crate::NativeProgram) -> Result<super::HostShader, &'static str> {
    emit_for(p).map_err(|e| match e {
        ShaderError::Unsupported { what, .. } => what,
        e => panic!("{e}"),
    })
}

/// Runs `p` on `host` over the views `bvh` and `rays`, in `groups` groups:
/// the output dwords.
fn run(
    host: &Host,
    p: &crate::NativeProgram,
    bvh: &[u32],
    rays: &[u32],
    groups: u32,
) -> Result<Vec<u32>, String> {
    let cs = emit_for(p).map_err(|e| e.to_string())?;
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil.clone())
            .map_err(|e| format!("validation: {e}"))?;
    }
    let (root, pso) = host.compute_pipeline(&cs)?;
    let data = vec![0; (cs.user_data_dwords() as usize).max(1)];
    let raw = |data| View::Buffer {
        data,
        stride: Stride::Raw,
    };
    let views = [raw(bvh), raw(rays), View::RawUav(groups * THREADS * 4)];
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(data))],
            views: &views,
            samplers: &[],
            groups: [groups, 1, 1],
        },
    )?;
    Ok(out.into_iter().next().unwrap_or_default())
}

/// Runs `f` on the default adapter and on WARP where each has an SM 6.6
/// device (one dispatching test at a time); says so when neither has.
fn on_each_adapter(mut f: impl FnMut(&str, &Host)) {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (name, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("{name}: no SM 6.6 device; skipped");
            continue;
        };
        f(name, &host);
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no SM 6.6 device; skipped");
    }
}

/// The results of `cases` under `form` on `host`, four dwords each.
fn intersect_on(host: &Host, form: &Form, bvh: &[u32], cases: &[Case]) -> Vec<[u32; 4]> {
    let rays = records(form, cases);
    let groups = (rays.len() / 16) as u32 / THREADS;
    let out =
        run(host, &program(form), bvh, &rays, groups).unwrap_or_else(|e| panic!("{form:?}: {e}"));
    out.chunks(4)
        .take(cases.len())
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect()
}

/// What the flags of `form` make the model do: (sort, grow).
fn model_flags(form: &Form) -> (bool, u32) {
    let d1 = match form.flags {
        Flags::Set(d1, _) => d1,
        Flags::Reversed(_) => 1 << 31,
        _ => VIEW_DWORD1,
    };
    (d1 >> 31 != 0, d1 >> 23 & 0xFF)
}

/// Checks `cases` under `form` on every adapter against the model: lanes
/// that do not run keep [`PRESET`].
fn parity(label: &str, form: &Form, bvh: &[u32], cases: &[Case]) {
    let (sort, grow) = model_flags(form);
    // The form's own operands: halves with A16, and no high pointer dword
    // in the 32-bit form.
    let mut own = if form.mimg.a16 {
        halved(cases)
    } else {
        cases.to_vec()
    };
    if !form.mimg.wide {
        for c in &mut own {
            c.pointer &= 0xFFFF_FFFF;
        }
    }
    let cases = &own[..];
    on_each_adapter(|name, host| {
        let got = intersect_on(host, form, bvh, cases);
        let mut counts = [0usize; 3];
        for (n, (case, got)) in cases.iter().zip(&got).enumerate() {
            let want = if case.runs || !form.masked {
                cpu::intersect(bvh, case.pointer, &case.ray, sort, grow)
            } else {
                std::array::from_fn(|k| PRESET + k as u32)
            };
            assert_eq!(
                *got, want,
                "{name}: {label}: ray {n} {case:x?}: got {got:x?}, the model {want:x?}"
            );
            let ran = case.runs || !form.masked;
            let boxes = matches!(case.pointer & 7, 4 | 5);
            counts[0] += usize::from(ran && boxes && want != [NONE; 4]);
            counts[1] += usize::from(ran && !boxes && hit(want, f32::INFINITY).is_some());
            counts[2] += usize::from(!ran);
        }
        eprintln!(
            "{name}: {label}: {} rays exact (synthetic; {} boxes hit, {} triangles hit, {} lanes not run)",
            cases.len(),
            counts[0],
            counts[1],
            counts[2]
        );
    });
}

fn floats(dst: &mut [u32], values: &[f32]) {
    for (d, v) in dst.iter_mut().zip(values) {
        *d = v.to_bits();
    }
}

fn box32(children: [u32; 4], corners: [[f32; 6]; 4]) -> [u32; 32] {
    let mut w = [0; 32];
    w[..4].copy_from_slice(&children);
    for (n, c) in corners.iter().enumerate() {
        floats(&mut w[4 + 6 * n..], c);
    }
    w
}

fn box16(children: [u32; 4], corners: [[f32; 6]; 4]) -> [u32; 16] {
    let mut w = [0; 16];
    w[..4].copy_from_slice(&children);
    for (n, c) in corners.iter().enumerate() {
        for (k, pair) in c.chunks(2).enumerate() {
            w[4 + 3 * n + k] = half(pair[0]) | half(pair[1]) << 16;
        }
    }
    w
}

fn triangles(vertices: [[f32; 3]; 4], id: u32) -> [u32; 16] {
    let mut w = [0; 16];
    for (n, v) in vertices.iter().enumerate() {
        floats(&mut w[3 * n..], v);
    }
    w[15] = id;
    w
}

/// The boxes of the reference's sorted-children fixture, children 8, 16, 24, 32.
const FIXTURE_BOXES: [[f32; 6]; 4] = [
    [-1.0, -1.0, 8.0, 1.0, 1.0, 9.0],
    [-1.0, -1.0, 2.0, 1.0, 1.0, 3.0],
    [2.0, 2.0, 1.0, 3.0, 3.0, 2.0],
    [-1.0, -1.0, 5.0, 1.0, 1.0, 6.0],
];

/// The reference's triangle: A (0, 0, 4), B (4, 0, 4), C (0, 2, 4).
const FIXTURE_TRIANGLE: [[f32; 3]; 4] = [
    [0.0, 0.0, 4.0],
    [4.0, 0.0, 4.0],
    [0.0, 2.0, 4.0],
    [4.0, 2.0, 4.0],
];

/// Identity barycentrics for both triangles: I from 1, J from 2.
const IDENTITY: u32 = 0x0909;

/// A synthetic structure of nine 64-byte nodes (576 bytes):
///
/// | node | what |
/// |------|------|
/// | 0    | zeros |
/// | 1-2  | a float box node (the reference's boxes) |
/// | 3    | a half box node |
/// | 4    | the reference's triangle and its mirror, identity barycentrics |
/// | 5    | a triangle pair whose id rotates the barycentrics |
/// | 6-7  | a float box node with an inverted box and a child of all ones |
/// | 8    | a triangle pair: the view's last node, its id the last dword |
fn scene() -> Vec<u32> {
    let mut w = vec![0; 16];
    w.extend(box32([8, 16, 24, 32], FIXTURE_BOXES));
    w.extend(box16(
        [0x51, 0x52, 0x53, 0x54],
        [
            [-4.0, -4.0, -4.0, 4.0, 4.0, 4.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [-2.0, -2.0, 6.0, 2.0, 2.0, 8.0],
            [1.0, -3.0, -1.0, 3.0, -1.0, 1.0],
        ],
    ));
    w.extend(triangles(FIXTURE_TRIANGLE, IDENTITY));
    // Triangle 0 returns (J, rest), triangle 1 (rest, I); the bits above
    // the selectors (double sided, opaque...) change nothing.
    w.extend(triangles(
        [
            [-2.0, -2.0, -3.0],
            [2.0, -2.0, -3.0],
            [-2.0, 2.0, -3.0],
            [2.0, 2.0, -5.0],
        ],
        0xB492,
    ));
    w.extend(box32(
        [0x1234_5678, 0xBBBB_0000, NONE, 0xAAAA_0005],
        [
            [-6.0, -1.0, -1.0, -4.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            [4.0, -1.0, -1.0, 6.0, 1.0, 1.0],
            [-8.0, -8.0, -8.0, 8.0, 8.0, 8.0],
        ],
    ));
    w.extend(triangles(
        [
            [0.0, 0.0, -4.0],
            [0.0, 4.0, -4.0],
            [4.0, 0.0, -4.0],
            [4.0, 4.0, -4.0],
        ],
        0x0609,
    ));
    assert_eq!(w.len(), 144);
    w
}

const fn pointer(node: u64, ty: u64) -> u64 {
    node << 3 | ty
}

/// Node pointers into [`scene`]: every node as what it is, the kinds that
/// return all ones (triangle types 2 and 3, the user types 6 and 7), and
/// nodes that do not lie in the view.
const POINTERS: [u64; 20] = [
    pointer(1, 5),
    pointer(3, 4),
    pointer(4, 0),
    pointer(4, 1),
    pointer(5, 0),
    pointer(5, 1),
    pointer(6, 5),
    pointer(8, 0),
    pointer(8, 1),
    pointer(0, 5),
    pointer(4, 2),
    pointer(4, 3),
    pointer(1, 6),
    pointer(1, 7),
    // A 128-byte node at the last 64 bytes, and nodes past the end.
    pointer(8, 5),
    pointer(9, 0),
    pointer(9, 4),
    // Bit 29 and up: the byte offset passes 32 bits.
    0x2000_0000 | pointer(1, 5),
    0x8000_0000 | pointer(4, 0),
    0xFFFF_FFFF,
];

/// A deterministic generator (a linear congruential one).
struct Rng(u64);

impl Rng {
    fn next(&mut self, n: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % u64::from(n)) as u32
    }
}

/// A box of points: its minimum and maximum corners.
type Range = [[f32; 3]; 2];

/// Anywhere around [`scene`].
const ANYWHERE: Range = [[-6.0; 3], [6.0; 3]];

/// Rays for the nodes of [`scene`]: the pointer, the box origins are drawn
/// from and the box of targets. The boxed origins send rays through
/// several boxes of a node, in another order than the node lists them.
const AIMS: [(u64, Range, Range); 13] = [
    (
        pointer(1, 5),
        [[-1.0, -1.0, -6.0], [1.0, 1.0, 1.0]],
        [[-1.0, -1.0, 8.0], [1.0, 1.0, 9.0]],
    ),
    (
        pointer(1, 5),
        [[-1.0, -1.0, 10.0], [1.0, 1.0, 12.0]],
        [[-1.0, -1.0, 1.0], [1.0, 1.0, 2.0]],
    ),
    (
        pointer(1, 5),
        ANYWHERE,
        [[-1.0, -1.0, 1.0], [3.0, 3.0, 9.0]],
    ),
    (pointer(3, 4), ANYWHERE, [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
    (
        pointer(3, 4),
        [[-2.0, -2.0, 9.0], [2.0, 2.0, 12.0]],
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
    ),
    (
        pointer(6, 5),
        [[7.0, -1.0, -1.0], [9.0, 1.0, 1.0]],
        [[-6.0, -1.0, -1.0], [-4.0, 1.0, 1.0]],
    ),
    (
        pointer(6, 5),
        ANYWHERE,
        [[-6.0, -1.0, -1.0], [6.0, 1.0, 1.0]],
    ),
    (pointer(4, 0), ANYWHERE, [[0.0, 0.0, 4.0], [4.0, 2.0, 4.0]]),
    (pointer(4, 1), ANYWHERE, [[0.0, 0.0, 4.0], [4.0, 2.0, 4.0]]),
    (
        pointer(5, 0),
        ANYWHERE,
        [[-2.0, -2.0, -4.0], [2.0, 2.0, -3.0]],
    ),
    (
        pointer(5, 1),
        ANYWHERE,
        [[-2.0, -2.0, -5.0], [2.0, 2.0, -3.0]],
    ),
    (
        pointer(8, 0),
        ANYWHERE,
        [[0.0, 0.0, -4.0], [4.0, 4.0, -4.0]],
    ),
    (
        pointer(8, 1),
        ANYWHERE,
        [[0.0, 0.0, -4.0], [4.0, 4.0, -4.0]],
    ),
];

/// `count` rays against [`scene`]. Two in three go to its box and triangle
/// nodes, from anywhere toward a point of the node's geometry, the rest to
/// every pointer of [`POINTERS`] in turn. Origins and targets lie on odd
/// eighths: never on a box plane or a vertex (so no product of zero and
/// infinity), and the directions are multiples of a quarter, which are
/// halves exactly and keep the triangle test's products exact whatever the
/// host contracts. The extent is in units of the way to the target. With
/// `high`, some pointers get a high dword.
fn cases(count: usize, high: bool) -> Vec<Case> {
    const EXTENTS: [f32; 6] = [0.5, 0.75, 1.5, 4.0, 64.0, f32::INFINITY];
    let mut rng = Rng(0x5EED);
    (0..count)
        .map(|n| {
            // An odd eighth in `lo..=hi + 1/8`.
            let mut odd = |lo: f32, hi: f32| {
                let steps = ((hi - lo) * 4.0) as u32 + 1;
                lo + (2 * rng.next(steps) + 1) as f32 * 0.125
            };
            let mut within =
                |b: Range| -> [f32; 3] { std::array::from_fn(|k| odd(b[0][k], b[1][k])) };
            let (mut pointer, origin, target) = match n % 3 {
                0 => {
                    let p = POINTERS[n / 3 % POINTERS.len()];
                    (p, within(ANYWHERE), within([[-3.0; 3], [3.0; 3]]))
                }
                _ => {
                    let (p, from, to) = AIMS[(n - n / 3) % AIMS.len()];
                    (p, within(from), within(to))
                }
            };
            if high && n % 7 == 3 {
                pointer |= u64::from(1 + rng.next(3)) << 32;
            }
            let mut dir: [f32; 3] = std::array::from_fn(|k| target[k] - origin[k]);
            if dir == [0.0; 3] {
                dir[2] = -0.0;
                dir[0] = 0.25;
            }
            Case {
                pointer,
                ray: ray(origin, dir, EXTENTS[rng.next(6) as usize]),
                runs: n % 5 != 1,
            }
        })
        .collect()
}

/// The nearest half to `f`.
fn nearest_half(f: f32) -> f32 {
    if !f.is_finite() {
        return f;
    }
    (0..=u16::MAX)
        .map(cpu::unhalf)
        .filter(|h| h.is_finite())
        .min_by(|a, b| (a - f).abs().total_cmp(&(b - f).abs()))
        .expect("halves")
}

/// `cases` as an A16 instruction takes them: the direction is halves as it
/// is, the inverse rounds to halves.
fn halved(cases: &[Case]) -> Vec<Case> {
    let mut known = std::collections::HashMap::new();
    let mut round = |f: f32| *known.entry(f.to_bits()).or_insert_with(|| nearest_half(f));
    cases
        .iter()
        .map(|c| {
            let mut c = *c;
            c.ray.inv = c.ray.inv.map(&mut round);
            c
        })
        .collect()
}

/// A triangle result as a hit the caller accepts: `t` within the extent
/// (the caller's test, as GPURT's traversal makes it), then the two
/// barycentrics.
fn hit(r: [u32; 4], extent: f32) -> Option<(f32, f32, f32)> {
    let [t_num, denom, i_num, j_num] = r.map(f32::from_bits);
    let t = t_num / denom;
    (t.is_finite() && t >= 0.0 && t <= extent).then_some((t, i_num / denom, j_num / denom))
}

// ---- no GPU ----

/// The decoder's fields for the four forms (RDNA2 ISA, table 48): 11, 8,
/// 12 and 9 address VGPRs, four result VGPRs, a four-dword descriptor.
#[test]
fn bvh_instructions_decode_with_their_operand_counts() {
    for (wide, a16, count) in [
        (false, false, 11),
        (false, true, 8),
        (true, false, 12),
        (true, true, 9),
    ] {
        for nsa in [false, true] {
            let mimg = Mimg {
                nsa,
                ..Mimg::of(wide, a16)
            };
            let mut words = mimg.words(16, BVH);
            assert_eq!(
                words.len(),
                if nsa {
                    2 + (count - 1usize).div_ceil(4)
                } else {
                    2
                }
            );
            words.push(S_ENDPGM);
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let insts = decode(&code).unwrap();
            assert_eq!(insts.len(), 2, "{mimg:?}");
            let name = if wide {
                "image_bvh64_intersect_ray"
            } else {
                "image_bvh_intersect_ray"
            };
            assert_eq!(insts[0].name(), Some(name));
            let Fields::Image {
                data,
                addr,
                rsrc,
                sampler,
                dmask,
                dim,
                unorm,
                r128,
                a16: packed,
                d16,
                ..
            } = insts[0].fields(true)
            else {
                panic!("not an image instruction");
            };
            assert_eq!(data, Operand::Vector { reg: 16, dwords: 4 });
            assert_eq!(
                rsrc,
                Operand::Scalar {
                    code: BVH,
                    dwords: 4
                }
            );
            assert_eq!(sampler, None);
            assert_eq!(
                (dmask, dim, unorm, r128, packed, d16),
                (0xF, 0, true, true, a16, false)
            );
            let regs: Vec<u8> = addr
                .iter()
                .flat_map(|a| match *a {
                    Operand::Vector { reg, dwords } => (0..dwords).map(move |k| reg + k),
                    other => panic!("{other:?}"),
                })
                .collect();
            assert_eq!(regs, mimg.regs(), "{mimg:?}");
            assert_eq!(regs.len(), count);
            assert_eq!(addr.len(), if nsa { count } else { 1 });
        }
    }
}

/// The reference's range fixture as arithmetic: a pointer of 13 is the
/// node at byte 64, which a view must hold whole, and the high dword
/// counts.
#[test]
fn bvh_node_offsets_are_64_bit() {
    assert_eq!(cpu::node_offset(13, 192, 128), Some(64));
    assert_eq!(cpu::node_offset(13, 191, 128), None);
    assert_eq!(cpu::node_offset(1 << 32 | 13, 192, 128), None);
    assert_eq!(
        cpu::node_offset(1 << 32 | 13, (1 << 35) + 192, 128),
        Some((1 << 35) + 64)
    );
    // The top three bits are part of the node index too.
    assert_eq!(cpu::node_offset(u64::MAX, u64::MAX, 64), None);
}

/// The reference's geometric fixtures against the model alone.
#[test]
fn bvh_model_gives_the_reference_fixtures() {
    let node = box32([8, 16, 24, 32], FIXTURE_BOXES);
    let along_z = ray([0.0; 3], [0.0, 0.0, 1.0], 7.0);
    assert_eq!(
        cpu::intersect(&node, 5, &along_z, true, 0),
        [16, 32, NONE, NONE]
    );
    assert_eq!(
        cpu::intersect(&node, 5, &along_z, false, 0),
        [NONE, 16, NONE, 32]
    );
    let tri = triangles(FIXTURE_TRIANGLE, IDENTITY);
    let at = |origin, extent| {
        let r = ray(origin, [0.0, 0.0, 2.0], extent);
        hit(cpu::intersect(&tri, 0, &r, true, 0), extent)
    };
    assert_eq!(at([1.0, 0.5, 0.0], 3.0), Some((2.0, 0.25, 0.25)));
    assert_eq!(at([1.0, 0.5, 0.0], 1.0), None);
    assert_eq!(at([3.0, 1.0, 0.0], 3.0), None);
    for h in [
        0u16, 1, 0x3FF, 0x400, 0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x8000,
    ] {
        assert_eq!(half(cpu::unhalf(h)), u32::from(h));
    }
    assert_eq!(cpu::unhalf(0x3C00), 1.0);
    assert_eq!(cpu::unhalf(0xC400), -4.0);
    assert_eq!(cpu::unhalf(1), (-24f32).exp2());
}

/// What the generated rays exercise, by the model: the parity tests mean
/// little unless boxes are hit in an order sorting changes, triangles are
/// hit and missed, and nodes are refused.
#[test]
fn bvh_generated_rays_cover_the_cases() {
    let bvh = scene();
    for high in [false, true] {
        let rays = cases(256, high);
        let (mut reordered, mut several, mut none_hit) = (0, 0, 0);
        let (mut hits, mut misses, mut rotated, mut refused) = (0, 0, 0, 0);
        for c in &rays {
            let sorted = cpu::intersect(&bvh, c.pointer, &c.ray, true, 6);
            let unsorted = cpu::intersect(&bvh, c.pointer, &c.ray, false, 6);
            let in_view = cpu::node_offset(c.pointer, 576, 64).is_some();
            match c.pointer & 7 {
                4 | 5 if sorted != [NONE; 4] => {
                    let hits =
                        |r: [u32; 4]| r.into_iter().filter(|c| *c != NONE).collect::<Vec<_>>();
                    several += usize::from(sorted[1] != NONE);
                    reordered += usize::from(hits(sorted) != hits(unsorted));
                }
                4 | 5 if in_view && c.pointer != pointer(8, 5) => none_hit += 1,
                0 | 1 if in_view => match hit(sorted, f32::INFINITY) {
                    Some(_) => {
                        hits += 1;
                        rotated += usize::from(c.pointer >> 3 == 5 || c.pointer == pointer(8, 1));
                    }
                    None => misses += 1,
                },
                _ => {}
            }
            refused += usize::from(sorted == [NONE; 4] && !matches!(c.pointer & 7, 4 | 5));
        }
        eprintln!(
            "high {high}: boxes: {several} with several hits, {reordered} that sorting reorders, {none_hit} without a hit"
        );
        eprintln!(
            "high {high}: triangles: {hits} hits ({rotated} with rotated barycentrics), {misses} misses; {refused} other all-ones"
        );
        assert!(several >= 30 && reordered >= 20 && none_hit >= 10, "boxes");
        assert!(hits >= 20 && rotated >= 5 && misses >= 20, "triangles");
        assert!(refused >= 20, "refused nodes");
    }
}

/// A float box node whose boxes the ray leaves just before it enters them:
/// by 2^-22, 2^-17 and 2^-12 of the distance, and one it crosses.
fn slivers() -> [u32; 32] {
    let gap = |e: i32| [1.0 + (e as f32).exp2(), 0.0, -1.0, 2.0, 1.0, 1.0];
    box32(
        [0xC0, 0xC1, 0xC2, 0xC3],
        [
            gap(-22),
            gap(-17),
            gap(-12),
            [0.5, 0.0, -1.0, 2.0, 1.0, 1.0],
        ],
    )
}

/// The ray of [`slivers`]: it enters each box through x and has left
/// through y at distance 1.
fn sliver_ray() -> cpu::Ray {
    ray([0.0; 3], [1.0, 1.0, 0.0], 8.0)
}

/// The children [`slivers`] returns by the ULPs the far end grows by.
const SLIVER_RESULTS: [(u32, [u32; 4]); 3] = [
    (0, [0xC3, NONE, NONE, NONE]),
    (6, [0xC3, 0xC0, NONE, NONE]),
    (255, [0xC3, 0xC0, 0xC1, NONE]),
];

#[test]
fn bvh_model_grows_boxes_by_the_descriptors_ulps() {
    for (grow, want) in SLIVER_RESULTS {
        assert_eq!(
            cpu::intersect(&slivers(), 5, &sliver_ray(), true, grow),
            want,
            "{grow}"
        );
    }
}

/// Every form translates, and the module passes the validator where one is
/// installed.
#[test]
fn bvh_forms_translate() {
    for (wide, a16) in [(false, false), (false, true), (true, false), (true, true)] {
        for nsa in [false, true] {
            for masked in [false, true] {
                let mut form = Form::of(wide, a16);
                form.mimg.nsa = nsa;
                form.masked = masked;
                let cs = translate(&program(&form)).unwrap_or_else(|e| panic!("{form:?}: {e}"));
                if let Ok(mut v) = super::validator::Validator::load(
                    super::validator::Validator::default_path().as_deref(),
                ) {
                    v.validate(&mut cs.dxil.clone())
                        .unwrap_or_else(|e| panic!("{form:?}: {e}"));
                }
            }
        }
    }
}

/// What the ISA forbids in the instruction's fields (8.2.10) stops the
/// translation by name: the store of the result needs it.
#[test]
fn bvh_instruction_fields_the_isa_forbids_are_refused_by_name() {
    let base = Mimg::of(false, false);
    let cases = [
        (
            Mimg {
                r128: false,
                ..base
            },
            why::NOT_R128,
        ),
        (Mimg { dmask: 0x7, ..base }, why::DMASK),
        (Mimg { dmask: 0, ..base }, why::DMASK),
        (
            Mimg {
                unorm: false,
                ..base
            },
            why::NORMALIZED,
        ),
        (Mimg { dim: 1, ..base }, why::DIM),
        (Mimg { tfe: true, ..base }, why::TFE_LWE),
        (Mimg { lwe: true, ..base }, why::TFE_LWE),
        // (d16 halves the result VGPRs: the store still needs them.)
        (Mimg { d16: true, ..base }, why::D16),
    ];
    for (mimg, reason) in cases {
        let form = Form {
            mimg,
            ..Form::of(false, false)
        };
        assert_eq!(translate(&program(&form)).err(), Some(reason), "{mimg:?}");
    }
}

/// The descriptor's flags (RDNA2 ISA, table 49) that are not supported
/// stop the translation by name.
#[test]
fn bvh_descriptor_flags_that_are_not_supported_are_refused_by_name() {
    let cases = [
        // Types other than 8.
        (Flags::Set(VIEW_DWORD1, 1 << 24), why::TYPE),
        (Flags::Set(VIEW_DWORD1, 9 << 28 | 1 << 24), why::TYPE),
        // Triangle return mode 0.
        (Flags::Set(VIEW_DWORD1, 8 << 28), why::RETURN_MODE),
        // Reserved bits 54:40, 119:106 and 122:121.
        (Flags::Set(VIEW_DWORD1 | 1 << 8, VIEW_DWORD3), why::RESERVED),
        (
            Flags::Set(VIEW_DWORD1 | 1 << 22, VIEW_DWORD3),
            why::RESERVED,
        ),
        (
            Flags::Set(VIEW_DWORD1, VIEW_DWORD3 | 1 << 10),
            why::RESERVED,
        ),
        (
            Flags::Set(VIEW_DWORD1, VIEW_DWORD3 | 1 << 23),
            why::RESERVED,
        ),
        (
            Flags::Set(VIEW_DWORD1, VIEW_DWORD3 | 1 << 25),
            why::RESERVED,
        ),
        (
            Flags::Set(VIEW_DWORD1, VIEW_DWORD3 | 1 << 26),
            why::RESERVED,
        ),
        // Flags that are not constants.
        (Flags::Computed, why::FLAGS),
    ];
    for (flags, reason) in cases {
        let form = Form {
            flags,
            ..Form::of(false, false)
        };
        assert_eq!(translate(&program(&form)).err(), Some(reason), "{flags:x?}");
    }
    // What is supported: the base address's high bits, any box growing,
    // sorting on or off, big_page, the size's high bits.
    for flags in [
        Flags::Set(0xFF, VIEW_DWORD3),
        Flags::Set(0xFF << 23, VIEW_DWORD3),
        Flags::Set(1 << 31, VIEW_DWORD3),
        Flags::Set(0, VIEW_DWORD3 | 1 << 27),
        Flags::Set(0, VIEW_DWORD3 | 0x3FF),
        Flags::Reversed(VIEW_DWORD3 | 1 << 27 | 0x3FF),
    ] {
        let form = Form {
            flags,
            ..Form::of(false, false)
        };
        assert!(translate(&program(&form)).is_ok(), "{flags:x?}");
    }
}

/// A descriptor the code sets whole, base address included (one over all
/// of memory, with absolute node pointers, say), names no view.
#[test]
fn bvh_descriptor_with_a_constant_base_is_refused_by_name() {
    let form = Form {
        constant_base: true,
        flags: Flags::Set(VIEW_DWORD1, VIEW_DWORD3),
        ..Form::of(false, false)
    };
    assert_eq!(translate(&program(&form)).err(), Some(why::CONSTANT_BASE));
}

/// A program storing dwords `first..first + 4` (SGPRs) of the BVH
/// descriptor, with the instruction present so the load is a BVH's.
fn descriptor_dump(sgprs: [u8; 4]) -> crate::NativeProgram {
    let mut w = prologue();
    for k in 0..3 {
        w.extend(load_x4(4 + 4 * k, 1, RAYS, 16 * u32::from(k)));
    }
    w.extend(Mimg::of(false, false).words(20, BVH));
    for (k, s) in sgprs.into_iter().enumerate() {
        w.push(v_mov_sgpr(16 + k as u8, s));
    }
    w.extend(store_x4(16, 2, OUT));
    w.push(S_ENDPGM);
    finish(&w)
}

/// The base address of a table-loaded BVH descriptor has no host value:
/// reading dword 0 as a number stops the translation by name.
#[test]
fn bvh_descriptor_base_read_as_data_is_refused_by_name() {
    let p = descriptor_dump([BVH, BVH + 1, BVH + 2, BVH + 3]);
    assert_eq!(translate(&p).err(), Some(why::BASE));
}

// ---- on the default adapter and WARP ----

/// The four descriptor SGPRs of a table-loaded BVH descriptor name a
/// bounded raw view: dword 2 is the node count less one from the view's
/// own size (192 bytes: 2; 576 bytes: 8), dwords 1 and 3 the flags the
/// translation gives such a descriptor.
#[test]
fn bvh_descriptor_dwords_come_from_the_bounded_view() {
    let p = descriptor_dump([BVH + 1, BVH + 2, BVH + 3, BVH + 2]);
    let rays = vec![0; THREADS as usize * 16];
    on_each_adapter(|name, host| {
        for (dwords, size) in [(48, 2), (144, 8), (16, 0), (15, 0)] {
            let bvh = vec![0; dwords];
            let out = run(host, &p, &bvh, &rays, 1).unwrap_or_else(|e| panic!("{name}: {e}"));
            for lane in out.chunks(4) {
                assert_eq!(
                    lane,
                    [VIEW_DWORD1, size, VIEW_DWORD3, size],
                    "{name}: {dwords} dwords"
                );
            }
            eprintln!(
                "{name}: a raw view of {} bytes: size field {size} (synthetic)",
                dwords * 4
            );
        }
    });
}

/// The reference's fixtures through the translated instruction, with the
/// expectations given as numbers: the float box node at byte 64 of a
/// 192-byte view (its last byte the view's last), sorted and not; the same
/// node in a view one dword short, and behind a high pointer dword; a ray
/// from inside a box along one of its planes' directions, and one outside;
/// the triangle hit, the same beyond its extent, and a miss.
#[test]
fn bvh_reference_fixtures_through_the_instruction() {
    let mut view = vec![0; 16];
    view.extend(box32([8, 16, 24, 32], FIXTURE_BOXES));
    let along_z = ray([0.0; 3], [0.0, 0.0, 1.0], 7.0);
    let node = Case {
        pointer: 13,
        ray: along_z,
        runs: true,
    };
    // A box around the origin and three the ray cannot meet.
    let far = [100.0, 100.0, 100.0, 101.0, 101.0, 101.0];
    let around = box32(
        [0x77, 1, 2, 3],
        [[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], far, far, far],
    );
    let parallel = |x: f32| Case {
        pointer: 5,
        ray: ray([x, 0.0, 0.0], [-0.0, 0.0, 1.0], 3.0),
        runs: true,
    };
    let tri = triangles(FIXTURE_TRIANGLE, IDENTITY);
    let shot = |origin, extent| Case {
        pointer: 0,
        ray: ray(origin, [0.0, 0.0, 2.0], extent),
        runs: true,
    };
    let shots = [
        shot([1.0, 0.5, 0.0], 3.0),
        shot([1.0, 0.5, 0.0], 1.0),
        shot([3.0, 1.0, 0.0], 3.0),
    ];
    on_each_adapter(|name, host| {
        let narrow = Form::of(false, false);
        let one = |form: &Form, bvh: &[u32], case: Case| intersect_on(host, form, bvh, &[case])[0];
        // Sorted children (the loaded descriptor sorts), then in node order.
        assert_eq!(one(&narrow, &view, node), [16, 32, NONE, NONE], "{name}");
        assert_eq!(
            one(&narrow.sorted(true), &view, node),
            [16, 32, NONE, NONE],
            "{name}"
        );
        assert_eq!(
            one(&narrow.sorted(false), &view, node),
            [NONE, 16, NONE, 32],
            "{name}"
        );
        // The view's last byte: one dword less and the node is not in it.
        assert_eq!(one(&narrow, &view[..47], node), [NONE; 4], "{name}");
        // The wide pointer: the same node, and not with a high dword.
        let wide = Form::of(true, false);
        assert_eq!(one(&wide, &view, node), [16, 32, NONE, NONE], "{name}");
        let high = Case {
            pointer: 1 << 32 | 13,
            ..node
        };
        assert_eq!(one(&wide, &view, high), [NONE; 4], "{name}");
        // Inside and parallel: entered at 0; outside: missed.
        assert_eq!(
            one(&narrow, &around, parallel(0.0)),
            [0x77, NONE, NONE, NONE],
            "{name}"
        );
        assert_eq!(one(&narrow, &around, parallel(2.0)), [NONE; 4], "{name}");
        // The triangle: t = 2, u = v = 0.25; beyond the extent; a miss.
        let hits: Vec<_> = shots
            .iter()
            .map(|s| hit(one(&narrow, &tri, *s), s.ray.extent))
            .collect();
        assert_eq!(hits, [Some((2.0, 0.25, 0.25)), None, None], "{name}");
        eprintln!("{name}: the reference fixtures hold (synthetic)");
    });
}

/// Float and half box nodes, triangles and triangle pairs, the kinds that
/// return all ones and nodes out of the view, against the model, sorted
/// (as loaded, and set) and unsorted, with the 32-bit pointer and full
/// width rays.
#[test]
fn bvh_full_width_rays_match_the_model() {
    let (bvh, rays) = (scene(), cases(256, false));
    let form = Form::of(false, false);
    parity("sorted, as loaded", &form, &bvh, &rays);
    parity("sorted", &form.sorted(true), &bvh, &rays);
    parity("unsorted", &form.sorted(false), &bvh, &rays);
    // Box growing of 0 and of 255 ULPs.
    for grow in [0u32, 255] {
        let grown = Form {
            flags: Flags::Set(1 << 31 | grow << 23, VIEW_DWORD3),
            ..form
        };
        parity(&format!("sorted, {grow} ULPs"), &grown, &bvh, &rays);
    }
}

/// Box growing (descriptor bits 62:55) through the instruction, with the
/// expectations given as numbers: 0 ULPs (what a loaded descriptor has,
/// as the title's own descriptors do), 6 and 255.
#[test]
fn bvh_box_growing_is_the_descriptors() {
    let case = Case {
        pointer: 5,
        ray: sliver_ray(),
        runs: true,
    };
    on_each_adapter(|name, host| {
        for (grow, want) in SLIVER_RESULTS {
            let form = Form {
                flags: Flags::Set(1 << 31 | grow << 23, VIEW_DWORD3),
                ..Form::of(false, false)
            };
            let got = intersect_on(host, &form, &slivers(), &[case])[0];
            assert_eq!(got, want, "{name}: {grow} ULPs");
        }
        let loaded = intersect_on(host, &Form::of(false, false), &slivers(), &[case])[0];
        assert_eq!(loaded, SLIVER_RESULTS[0].1, "{name}: as loaded");
        eprintln!("{name}: boxes grow by 0, 6 and 255 ULPs as set (synthetic)");
    });
}

/// The direction and its inverse as halves (A16).
#[test]
fn bvh_a16_rays_match_the_model() {
    let (bvh, rays) = (scene(), cases(256, false));
    let form = Form::of(false, true);
    parity("a16 sorted", &form, &bvh, &rays);
    parity("a16 unsorted", &form.sorted(false), &bvh, &rays);
}

/// The 64-bit pointer: a high dword of 0 reads the same nodes, any other
/// puts the node out of the view; with full-width and A16 rays.
#[test]
fn bvh64_rays_match_the_model() {
    let (bvh, rays) = (scene(), cases(256, true));
    assert!(rays.iter().any(|c| c.pointer >> 32 != 0));
    for a16 in [false, true] {
        let form = Form::of(true, a16);
        let label = if a16 { "bvh64 a16" } else { "bvh64" };
        parity(&format!("{label} sorted"), &form, &bvh, &rays);
        parity(
            &format!("{label} unsorted"),
            &form.sorted(false),
            &bvh,
            &rays,
        );
    }
}

/// Addresses in VGPRs that are not consecutive (NSA).
#[test]
fn bvh_nsa_rays_match_the_model() {
    let (bvh, rays) = (scene(), cases(128, true));
    for (wide, a16) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut form = Form::of(wide, a16);
        form.mimg.nsa = true;
        parity(&format!("nsa wide {wide} a16 {a16}"), &form, &bvh, &rays);
    }
}

/// Flags the code sets with a bit reverse of a constant (box sorting in
/// dword 1) are constants: the rays sort as with the same flags moved in.
#[test]
fn bvh_flags_from_a_bit_reverse_sort() {
    let (bvh, rays) = (scene(), cases(256, false));
    let form = Form {
        flags: Flags::Reversed(VIEW_DWORD3),
        ..Form::of(false, false)
    };
    parity("flags from a bit reverse", &form, &bvh, &rays);
}

/// EXEC: lanes it disables keep their result VGPRs.
#[test]
fn bvh_honors_exec() {
    let (bvh, rays) = (scene(), cases(256, true));
    assert!(rays.iter().any(|c| c.runs) && rays.iter().any(|c| !c.runs));
    for (wide, a16) in [(false, false), (true, true)] {
        let form = Form {
            masked: true,
            ..Form::of(wide, a16)
        };
        parity(&format!("masked wide {wide} a16 {a16}"), &form, &bvh, &rays);
    }
}
