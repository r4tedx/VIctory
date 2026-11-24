//! Ray intersection: `image_bvh_intersect_ray` and
//! `image_bvh64_intersect_ray` (RDNA2 ISA 8.2.10).
//!
//! The instruction takes a ray and a node pointer in VGPRs and a 128-bit
//! BVH descriptor in four SGPRs, fetches the node and returns four dwords.
//! AMD's ISA gives the operand layout, the restrictions on the instruction
//! fields and the descriptor fields (table 49); it does not give the node
//! layouts or what a test returns beyond naming the results. Those follow
//! AMD's GPURT (MIT), `src/shadersClean/common/gfx10` (node layouts) and
//! the software path of `src/shaders/IntersectCommon.hlsl`
//! (`image_bvh64_intersect_ray_base` at ray-tracing IP 1.1, which is what
//! this hardware generation has):
//!
//! * a node pointer is a node type in bits 2:0 over the node's index in
//!   64-byte units: the node is at `(pointer >> 3) * 64` bytes from the
//!   descriptor's base address;
//! * type 5, a 128-byte box node of four children: child pointers in
//!   dwords 0-3, then per child a minimum and a maximum corner as three
//!   floats each. Type 4, the same in 64 bytes with the corners as halves
//!   (min x, min y | min z, max x | max y, max z). The result is the four
//!   child pointers, `0xFFFFFFFF` for a child the ray misses, hits first and
//!   ordered by entry distance when the descriptor enables sorting;
//! * types 0 and 1, the two triangles of a 64-byte triangle node (vertices
//!   0, 1, 2 and 1, 3, 2 of the four at bytes 0, 12, 24 and 36; the triangle
//!   id at byte 60). The result is `t` as a numerator and a denominator,
//!   then the barycentric numerators I and J, rotated as the triangle id's
//!   byte for that triangle says; a miss returns `t` = infinity over 1;
//! * anything else returns four `0xFFFFFFFF`.
//!
//! What the host binds: the descriptor is traced to its table or heap
//! entry by dword 0 alone ([`crate::resource`]), and the view at that index
//! is a raw buffer (`ByteAddressBuffer`) that starts at the descriptor's
//! base address and spans the BVH. The view's size is the bound: a node
//! that does not lie in it whole is not fetched and returns four
//! `0xFFFFFFFF`. The node offset is taken as the 64-bit number it is, so a
//! pointer beyond 32 bits of bytes is out of bounds rather than wrapped.
//!
//! The descriptor's flags are dwords 1 and 3 as the program holds them,
//! which must be constants here: set by the code, or the ones
//! [`Lower::bvh_descriptor_data`] gives a descriptor loaded from a table.
// shortcut: nothing here was compared with hardware. Open against it: the
// NaN and equality edge cases of the box test, the triangle test (GPURT's
// software path is not watertight and does not cull by the ray extent; the
// hardware's may differ in the last bits), triangle types 2 and 3 (the
// hardware has four triangles per node, GPURT's software path only two and
// returns -1 for the others, as this does), what an out-of-bounds node
// returns, and the descriptor's size field (the view's size is used).

use super::{Lower, Val, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Reg};
use crate::native::ResourceType;
use crate::resource::{Site, Source};

pub(crate) const NOT_R128: Why = "a ray intersection with a 256-bit descriptor (r128 clear)";
pub(crate) const DMASK: Why = "a ray intersection that does not return four dwords (dmask)";
pub(crate) const D16: Why = "a ray intersection with 16-bit results (d16)";
pub(crate) const NORMALIZED: Why = "a ray intersection with normalized coordinates (unorm clear)";
pub(crate) const DIM: Why = "a ray intersection on a view that is not 1D (dim)";
pub(crate) const TFE_LWE: Why = "a ray intersection with tfe or lwe";
pub(crate) const FLAGS: Why = "a BVH descriptor whose flag dwords are not constants";
pub(crate) const TYPE: Why = "a BVH descriptor of another type (not 8)";
pub(crate) const RESERVED: Why = "a BVH descriptor with reserved bits set";
pub(crate) const RETURN_MODE: Why =
    "a BVH descriptor with triangle return mode 0 (triangle id and hit status)";
/// A constant base is refused: the lowering reads node `n` at byte `64 * n`
/// of the view in the slot dword 0 names, which assumes the view starts at
/// the structure's base. A table-loaded base that is not the structure's
/// start (absolute 64-bit node pointers over base 0, GPURT's
/// `(bvhAddress >> 3) + nodePointer`) can't be told apart and reads as
/// misses.
pub(crate) const CONSTANT_BASE: Why = "a BVH descriptor with a constant base address";
pub(crate) const BASE: Why = "a BVH descriptor's base address read as data";
const RAY: Why = "a ray operand";

/// Dword 1 of a BVH descriptor loaded from a table, read as a number: box
/// sorting enabled (bit 31) and no box growing (bits 30:23), with the high
/// bits of the base address 0. This is what the title's own descriptors
/// hold: Leonida's census of the 716 intersection sites (issue #274) found
/// `box_grow` 0 and `box_sort_en` 1 at every one, and sampled programs set
/// dword 1 to 0x8000_0000 (`s_brev_b32 s1, 1`).
pub(crate) const VIEW_DWORD1: u32 = 1 << 31;
/// Dword 3 of the same: type 8 (bits 31:28) and triangle return mode 1
/// (bit 24: I and J numerators, what GPURT's traversal reads), with the
/// high bits of the size 0. The same census found type 8 and return mode 1
/// at all 716 sites.
// shortcut: return mode 1 is taken, not read: a table-loaded descriptor
// with mode 0 would want the triangle id and hit status in results 2 and 3
// (ISA table 50) and gets the I and J numerators instead.
pub(crate) const VIEW_DWORD3: u32 = 8 << 28 | 1 << 24;

/// A ray as the instruction's address VGPRs hold it, as floats.
struct Ray {
    extent: V,
    origin: [V; 3],
    dir: [V; 3],
    inv: [V; 3],
}

impl Lower<'_> {
    fn ibin(&mut self, op: Bin, a: V, b: V) -> V {
        self.s.m.bin(self.cur, op, a, b)
    }

    fn icmp(&mut self, p: u8, a: V, b: V) -> V {
        self.s.m.cmp(self.cur, p, a, b)
    }

    fn pick(&mut self, c: V, t: V, f: V) -> V {
        self.s.m.select(self.cur, c, t, f)
    }

    fn fconst(&mut self, f: f32) -> V {
        self.s.m.f32(f.to_bits())
    }

    /// `n` floats (1 to 4) at byte `at` of raw buffer `h`.
    fn raw_floats(&mut self, h: V, at: V, n: u32) -> Vec<V> {
        let c = self.cur;
        let mask = self.s.m.i8((1u8 << n) - 1);
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, at, u, mask, align],
        );
        (0..n).map(|k| self.s.m.extract(c, r, k)).collect()
    }

    /// `dwords` floats from byte `first` of the node at byte `node`.
    fn node_floats(&mut self, h: V, node: V, first: u32, dwords: u32) -> Vec<V> {
        let mut out = Vec::new();
        for k in (0..dwords).step_by(4) {
            let off = self.s.m.i32(first + 4 * k);
            let at = self.ibin(Bin::Add, node, off);
            out.extend(self.raw_floats(h, at, (dwords - k).min(4)));
        }
        out
    }

    /// The size in bytes of the raw buffer view `h`.
    fn view_bytes(&mut self, h: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let undef = self.s.m.undef(i32t);
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, undef]);
        self.s.m.extract(c, d, 0)
    }

    /// The raw handle of the view a BVH descriptor names: a UAV when the
    /// title's used-resource entry says so, else an SRV. Both the
    /// intersection and a read of the descriptor's dwords as numbers go
    /// through here, so a program reads one slot as one kind of view.
    // shortcut: the slot must hold a raw view of the console's node memory
    // starting at the structure's base (see the host notes in `emit/mod.rs`).
    // Nothing on the GPU side makes one yet, and PR #702 would make a host
    // acceleration structure there instead; with a null view every node is
    // out of bounds and the intersection returns all ones (a miss).
    fn bvh_handle(&mut self, site: Site, src: &Source) -> Result<V, Why> {
        let uav = self
            .used_kind(src)
            .is_some_and(|u| u.kind == ResourceType::Uav);
        self.handle(site, src, false, (11 | if uav { 0x1000 } else { 0 }, 0))
    }

    /// A BVH descriptor load (`s_load_dword*` from a table or heap) whose
    /// dwords the code may read as numbers, like
    /// [`Lower::descriptor_data`]: dwords 1 and 3 are the flags of
    /// [`VIEW_DWORD1`] and [`VIEW_DWORD3`], dword 2 the node count less one
    /// (RDNA2 ISA, table 49) from the host view's size.
    // shortcut: dword 0 (and with it the base address, in 256-byte units
    // across dwords 0 and 1) has no host value: a program that computes
    // the structure's address from it stops there.
    pub(super) fn bvh_descriptor_data(
        &mut self,
        inst: &Inst,
        site: Site,
        src: &Source,
        (code, count, first): (u8, u32, u8),
    ) -> Result<(), Why> {
        let h = self.bvh_handle(site, src)?;
        let bytes = self.view_bytes(h);
        let (six, one, zero) = (self.s.m.i32(6), self.s.m.i32(1), self.s.m.i32(0));
        let nodes = self.ibin(Bin::LShr, bytes, six);
        let less = self.ibin(Bin::Sub, nodes, one);
        let none = self.icmp(pred::EQ, nodes, zero);
        let size = self.pick(none, zero, less);
        let words = [
            (1, self.s.m.i32(VIEW_DWORD1)),
            (2, size),
            (3, self.s.m.i32(VIEW_DWORD3)),
        ];
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(BASE));
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn bvh(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            rsrc,
            dmask,
            dim,
            unorm,
            r128,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        // The restrictions of RDNA2 ISA 8.2.10; the sampler field is a
        // placeholder.
        for (bad, why) in [
            (!*r128, NOT_R128),
            (*dmask != 0xF, DMASK),
            (*d16, D16),
            (!*unorm, NORMALIZED),
            (*dim != 0, DIM),
            (*tfe || *lwe, TFE_LWE),
        ] {
            if bad {
                return Err(why);
            }
        }
        let wide = inst.name() == "image_bvh64_intersect_ray";
        let site = (b, i);
        let guard = self.guard(b, i)?;

        // The descriptor's flags (RDNA2 ISA, table 49).
        let Operand::Scalar { code, .. } = *rsrc else {
            return Err("a descriptor not in SGPRs");
        };
        let mut flags = [0u32; 2];
        for (k, flag) in [1u8, 3].into_iter().zip(&mut flags) {
            let v = inst.read(Reg::S(code + k)).ok_or(FLAGS)?;
            let x = self.i32_of(v)?;
            *flag = self.s.m.int_value(x).ok_or(FLAGS)? as u32;
        }
        let [w1, w3] = flags;
        if w3 >> 28 != 8 {
            return Err(TYPE);
        }
        // Bits 54:40, 119:106 and 122:121; big_page (123) changes nothing here.
        if w1 & 0x007F_FF00 != 0 || w3 & 0x06FF_FC00 != 0 {
            return Err(RESERVED);
        }
        if w3 >> 24 & 1 == 0 {
            return Err(RETURN_MODE);
        }
        let (sort, grow) = (w1 >> 31 != 0, w1 >> 23 & 0xFF);

        // The address VGPRs in order (one range, or one each with NSA):
        // the node pointer (two dwords for the 64-bit form), the extent,
        // the origin, then the direction and its inverse, as three floats
        // each or with a16 as six halves, two per VGPR, low half first.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg.wrapping_add(k)));
            }
        }
        let word = |l: &mut Self, k: usize| -> Result<V, Why> {
            let r = *regs.get(k).ok_or(RAY)?;
            l.i32_of(inst.read(Reg::V(r)).ok_or(RAY)?)
        };
        let float = |l: &mut Self, k: usize| -> Result<V, Why> {
            let r = *regs.get(k).ok_or(RAY)?;
            l.float_of(inst.read(Reg::V(r)).ok_or(RAY)?)
        };
        let lo = word(self, 0)?;
        let hi = if wide { Some(word(self, 1)?) } else { None };
        let at = 1 + usize::from(wide);
        let extent = float(self, at)?;
        let mut origin = [extent; 3];
        for (k, o) in origin.iter_mut().enumerate() {
            *o = float(self, at + 1 + k)?;
        }
        let mut rest = [extent; 6];
        for (k, x) in rest.iter_mut().enumerate() {
            *x = if *a16 {
                let mut h = word(self, at + 4 + k / 2)?;
                if k % 2 == 1 {
                    let k16 = self.s.m.i32(16);
                    h = self.ibin(Bin::LShr, h, k16);
                }
                let f32t = self.s.t.f32;
                self.s
                    .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h])
            } else {
                float(self, at + 4 + k)?
            };
        }
        let ray = Ray {
            extent,
            origin,
            dir: [rest[0], rest[1], rest[2]],
            inv: [rest[3], rest[4], rest[5]],
        };

        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked BVH descriptor")?;
        if matches!(src, Source::Constant(_)) {
            return Err(CONSTANT_BASE);
        }
        let h = self.bvh_handle(site, src)?;
        let bytes = self.view_bytes(h);

        // The node's byte offset is (pointer >> 3) * 64: the pointer
        // without its type, shifted left by 3, as a 64-bit number (67 bits
        // for the 64-bit form). It fits the 32 bits of a view's size when
        // the pointer's high dword and the top three bits of its low dword
        // are 0; only then is it compared with the size, less the node's
        // own, so that the whole node lies in the view.
        // shortcut: a view of 4 GiB or more reads as its size modulo 2^32.
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (zero, seven, not_type) = (k(self, 0), k(self, 7), k(self, !7));
        let (three, top_shift) = (k(self, 3), k(self, 29));
        let ty = self.ibin(Bin::And, lo, seven);
        let top = self.ibin(Bin::LShr, lo, top_shift);
        let mut fits = self.icmp(pred::EQ, top, zero);
        if let Some(hi) = hi {
            let none = self.icmp(pred::EQ, hi, zero);
            fits = self.ibin(Bin::And, fits, none);
        }
        let node = self.ibin(Bin::And, lo, not_type);
        let node = self.ibin(Bin::Shl, node, three);
        let (t4, t5, two) = (k(self, 4), k(self, 5), k(self, 2));
        let box32 = self.icmp(pred::EQ, ty, t5);
        let box16 = self.icmp(pred::EQ, ty, t4);
        let triangle = self.icmp(pred::IULT, ty, two);
        let (big, small) = (k(self, 128), k(self, 64));
        let size = self.pick(box32, big, small);
        let room = self.icmp(pred::IUGE, bytes, size);
        let last = self.ibin(Bin::Sub, bytes, size);
        let within = self.icmp(pred::IULE, node, last);
        let mut live = self.ibin(Bin::And, fits, room);
        live = self.ibin(Bin::And, live, within);
        // Only lanes EXEC enables fetch.
        if let Some(g) = guard {
            live = self.ibin(Bin::And, live, g);
        }

        let mut out = [k(self, u32::MAX); 4];
        let join = |l: &mut Self, at, inside: [V; 4], out: &mut [V; 4]| {
            let pairs: Vec<(V, V)> = inside.into_iter().zip(*out).collect();
            out.copy_from_slice(&l.end_guarded(at, &pairs));
        };

        // Box nodes of floats.
        let cond = self.ibin(Bin::And, live, box32);
        let at = self.begin_guarded(Some(cond));
        let words = self.node_floats(h, node, 0, 28);
        let children: [V; 4] = std::array::from_fn(|n| self.as_int(words[n]));
        let corners: [[V; 6]; 4] =
            std::array::from_fn(|n| std::array::from_fn(|j| words[4 + 6 * n + j]));
        let r = self.bvh_boxes(children, &corners, &ray, sort, grow);
        join(self, at, r, &mut out);

        // Box nodes of halves.
        let cond = self.ibin(Bin::And, live, box16);
        let at = self.begin_guarded(Some(cond));
        let words = self.node_floats(h, node, 0, 16);
        let children: [V; 4] = std::array::from_fn(|n| self.as_int(words[n]));
        let k16 = k(self, 16);
        let mut halves = Vec::new();
        for &w in &words[4..] {
            let w = self.as_int(w);
            let high = self.ibin(Bin::LShr, w, k16);
            for h in [w, high] {
                let f32t = self.s.t.f32;
                halves.push(
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]),
                );
            }
        }
        let corners: [[V; 6]; 4] =
            std::array::from_fn(|n| std::array::from_fn(|j| halves[6 * n + j]));
        let r = self.bvh_boxes(children, &corners, &ray, sort, grow);
        join(self, at, r, &mut out);

        // Triangles 0 and 1 of a triangle node.
        let cond = self.ibin(Bin::And, live, triangle);
        let at = self.begin_guarded(Some(cond));
        let r = self.bvh_triangle(h, node, ty, &ray);
        join(self, at, r, &mut out);

        let Operand::Vector { reg, .. } = *data else {
            return Err("a ray intersection result");
        };
        for (n, x) in out.into_iter().enumerate() {
            self.write(inst, Reg::V(reg.wrapping_add(n as u8)), x, guard);
        }
        Ok(())
    }

    /// The ray against the four boxes of a box node (GPURT's
    /// `fast_intersect_bbox` and `IntersectNodeBvh4`): `corners` are each
    /// child's minimum then maximum. The child pointers, `0xFFFFFFFF` for
    /// a miss; with `sort`, hits first by entry distance.
    fn bvh_boxes(
        &mut self,
        children: [V; 4],
        corners: &[[V; 6]; 4],
        ray: &Ray,
        sort: bool,
        grow: u32,
    ) -> [V; 4] {
        let (fzero, inf, ninf) = (
            self.fconst(0.0),
            self.fconst(f32::INFINITY),
            self.fconst(f32::NEG_INFINITY),
        );
        // The far end grows by `grow` ULPs (2^-24 each) to stay
        // conservative against the triangle test.
        let grown = self.fconst(1.0 + grow as f32 * 5.960_464_5e-8);
        let none = self.s.m.i32(u32::MAX);
        let mut child = children;
        let mut key = [fzero; 4];
        for n in 0..4 {
            let (mut near, mut far) = (None, None);
            for axis in 0..3 {
                let (o, inv) = (ray.origin[axis], ray.inv[axis]);
                let lo = self.ibin(Bin::Sub, corners[n][axis], o);
                let hi = self.ibin(Bin::Sub, corners[n][3 + axis], o);
                let (lo, hi) = (self.ibin(Bin::Mul, lo, inv), self.ibin(Bin::Mul, hi, inv));
                let forward = self.icmp(pred::OGE, inv, fzero);
                let enter = self.pick(forward, lo, hi);
                let leave = self.pick(forward, hi, lo);
                near = Some(match near {
                    Some(x) => self.binary(35, x, enter),
                    None => enter,
                });
                far = Some(match far {
                    Some(x) => self.binary(36, x, leave),
                    None => leave,
                });
            }
            let (near, far) = (near.expect("three axes"), far.expect("three axes"));
            let (a, z) = (
                self.icmp(pred::UNO, near, near),
                self.icmp(pred::UNO, far, far),
            );
            let nan = self.ibin(Bin::Or, a, z);
            let from = self.binary(35, near, fzero);
            let to = self.binary(36, far, ray.extent);
            let from = self.pick(nan, inf, from);
            let to = self.pick(nan, ninf, to);
            // Equal ends count as a hit: rounding snaps short spans together.
            let to = self.ibin(Bin::Mul, to, grown);
            let hit = self.icmp(pred::OLE, from, to);
            child[n] = self.pick(hit, children[n], none);
            key[n] = from;
        }
        if sort {
            // A sorting network of five exchanges: a missed child moves
            // behind, a nearer hit forward.
            for (a, b) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
                let b_hit = self.icmp(pred::NE, child[b], none);
                let nearer = self.icmp(pred::OLT, key[b], key[a]);
                let swap = self.ibin(Bin::And, b_hit, nearer);
                let a_miss = self.icmp(pred::EQ, child[a], none);
                let swap = self.ibin(Bin::Or, swap, a_miss);
                (child[a], child[b]) = (
                    self.pick(swap, child[b], child[a]),
                    self.pick(swap, child[a], child[b]),
                );
                (key[a], key[b]) = (
                    self.pick(swap, key[b], key[a]),
                    self.pick(swap, key[a], key[b]),
                );
            }
        }
        child
    }

    /// The ray against triangle `ty` (0 or 1) of the triangle node at byte
    /// `node` (GPURT's `FetchTriangleFromNode`, `fast_intersect_triangle`
    /// and `SwizzleBarycentrics`): the numerator and denominator of `t`,
    /// then the numerators of the barycentrics the triangle id selects.
    fn bvh_triangle(&mut self, h: V, node: V, ty: V, ray: &Ray) -> [V; 4] {
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (zero, one) = (k(self, 0), k(self, 1));
        // Triangle 0 is vertices 0, 1, 2; triangle 1 is 1, 3, 2.
        let second = self.icmp(pred::EQ, ty, one);
        let (k0, k12, k24, k36, k60) = (zero, k(self, 12), k(self, 24), k(self, 36), k(self, 60));
        let first = self.pick(second, k12, k0);
        let middle = self.pick(second, k36, k12);
        let vertex = |l: &mut Self, off: V| -> [V; 3] {
            let at = l.ibin(Bin::Add, node, off);
            let v = l.raw_floats(h, at, 3);
            [v[0], v[1], v[2]]
        };
        let (a, b, c) = (vertex(self, first), vertex(self, middle), vertex(self, k24));
        let at = self.ibin(Bin::Add, node, k60);
        let id = self.raw_floats(h, at, 1)[0];
        let id = self.as_int(id);

        let sub = |l: &mut Self, x: [V; 3], y: [V; 3]| -> [V; 3] {
            std::array::from_fn(|n| l.ibin(Bin::Sub, x[n], y[n]))
        };
        let cross = |l: &mut Self, x: [V; 3], y: [V; 3]| -> [V; 3] {
            std::array::from_fn(|n| {
                let (p, q) = ((n + 1) % 3, (n + 2) % 3);
                let (u, v) = (l.ibin(Bin::Mul, x[p], y[q]), l.ibin(Bin::Mul, x[q], y[p]));
                l.ibin(Bin::Sub, u, v)
            })
        };
        let dot = |l: &mut Self, x: [V; 3], y: [V; 3]| -> V {
            let p: [V; 3] = std::array::from_fn(|n| l.ibin(Bin::Mul, x[n], y[n]));
            let s = l.ibin(Bin::Add, p[0], p[1]);
            l.ibin(Bin::Add, s, p[2])
        };
        let (e1, e2, e3) = (sub(self, b, a), sub(self, c, a), sub(self, ray.origin, a));
        let s1 = cross(self, ray.dir, e2);
        let s2 = cross(self, e3, e1);
        let t_num = dot(self, e2, s2);
        let denom = dot(self, s1, e1);
        let i_num = dot(self, e3, s1);
        let j_num = dot(self, ray.dir, s2);
        let t = self.ibin(Bin::SDiv, t_num, denom);
        let u = self.ibin(Bin::SDiv, i_num, denom);
        let v = self.ibin(Bin::SDiv, j_num, denom);
        let (fzero, fone, inf) = (
            self.fconst(0.0),
            self.fconst(1.0),
            self.fconst(f32::INFINITY),
        );
        let uv = self.ibin(Bin::Add, u, v);
        let mut miss = self.icmp(pred::OLT, u, fzero);
        for (p, x, y) in [
            (pred::OGT, u, fone),
            (pred::OLT, v, fzero),
            (pred::OGT, uv, fone),
            (pred::OLT, t, fzero),
        ] {
            let m = self.icmp(p, x, y);
            miss = self.ibin(Bin::Or, miss, m);
        }
        // shortcut: GPURT's software path. Leonida's RX 6800 probe (issue
        // #274) measured hardware results scaled by k = |d[kz]|^3 on a hit,
        // and copysign(inf, k * det) in word 0 on a miss with the other
        // three kept; a shader comparing t / denominator sees the same
        // acceptance either way, but word 1's sign on a miss differs.
        let t_num = self.pick(miss, inf, t_num);
        let denom = self.pick(miss, fone, denom);

        // The triangle id holds a byte per triangle: bits 1:0 and 3:2 say
        // which of (1 - I - J, I, J), times the denominator, returns as I
        // and J (the builder may have rotated the vertices).
        let rest = self.ibin(Bin::Sub, denom, i_num);
        let rest = self.ibin(Bin::Sub, rest, j_num);
        let (three, eight, two) = (k(self, 3), k(self, 8), k(self, 2));
        let shift = self.ibin(Bin::Mul, ty, eight);
        let byte = self.ibin(Bin::LShr, id, shift);
        let mut picked = [fzero; 2];
        for (n, p) in picked.iter_mut().enumerate() {
            let sel = if n == 0 {
                byte
            } else {
                self.ibin(Bin::LShr, byte, two)
            };
            let sel = self.ibin(Bin::And, sel, three);
            // (Selector 3 names no coordinate: it reads 0.)
            let mut x = fzero;
            for (code, value) in [(2, j_num), (1, i_num), (0, rest)] {
                let code = k(self, code);
                let is = self.icmp(pred::EQ, sel, code);
                x = self.pick(is, value, x);
            }
            *p = x;
        }
        [t_num, denom, picked[0], picked[1]].map(|f| self.as_int(f))
    }
}
