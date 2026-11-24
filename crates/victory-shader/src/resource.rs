//! Resource tracking (decision 0003): where each descriptor a program uses
//! comes from, in terms of the root signature.
//!
//! Root parameters arrive in user SGPRs ([`RootParam`]): a descriptor table
//! as the 32-bit heap address of its range (the code sets the high half to
//! 0), a root descriptor as a 64-bit buffer address, root constants as
//! themselves. A root signature with dynamic resources (SM 6.6) adds the
//! 32-bit base addresses of both heaps; a descriptor is then loaded from
//! one of them at an index the code computes (from a root constant, say). The code loads table descriptors from heap memory with
//! `s_load_dword*`, and builds a buffer descriptor around a root
//! descriptor's address itself. Each descriptor operand and each scalar
//! memory address is traced through the SSA values back to those
//! parameters, by data flow: copies, phis that agree, and additions.

use std::cell::{OnceCell, RefCell};
use std::collections::{HashMap, HashSet};

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand, Valu};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, RootParam, RootType};
use crate::structure::{Bool, LaneExit, Structured};

/// An instruction: block and index in [`crate::ir::Block::insts`].
pub type Site = (usize, usize);

/// Relative byte offset `value - base` for multi-term, scaled, subtracted, or
/// phi-joined addresses, plus an optional extra dynamic offset operand (such as
/// SMEM `soffset` or a `global_*` VGPR offset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelOffset {
    pub value: Value,
    pub base: Value,
    pub extra_dynamic: Option<Value>,
    /// The constants added to `base` on the way to `value`: the least their
    /// sum can be, and how far above that it can lie over the phi edges (and
    /// writes under an EXEC mask) that reach `value`. `None` where no constant
    /// bounds it: a run-time term subtracted, a pointer stepped round a loop.
    /// A table's range is picked by it, as by [`Address::offset`] otherwise.
    pub constant: Option<(u32, u32)>,
}

/// A byte address: a root parameter's, plus a constant and at most one
/// value the program computes (an index times a stride, say).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    /// Index into the layout, or [`CODE`].
    pub param: usize,
    /// Added modulo 2^32 (with carry into the high half of a root
    /// descriptor's address).
    pub offset: u32,
    pub dynamic: Option<Value>,
    pub rel: Option<RelOffset>,
}

/// [`Address::param`] of the program's own code: a constant table the
/// compiler appends after the code and reads through `s_getpc_b64`; the
/// offset is then the byte offset in the code.
pub const CODE: usize = usize::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Dwords `first..` of what the scalar load at `load` reads (see
    /// [`Resources::scalar`] for its address).
    Loaded { load: Site, first: u8 },
    /// A buffer descriptor built around an address: `bits` OR-ed into
    /// dword 1 (stride, swizzle), dwords 2 and 3 as given.
    Built {
        address: Address,
        bits: u32,
        words: [u32; 2],
    },
    /// Fixed in the code (an immediate sampler, say).
    Constant(Vec<u32>),
    /// Chosen by control flow: one source per predecessor of `block`, in
    /// [`crate::ir::Block::preds`] order.
    Phi { block: usize, args: Vec<Source> },
    /// Chosen by SCC (`scc`, the value an `s_cselect` or `s_cmov` of each
    /// dword reads): the source when it is set, then when it is clear.
    Select { scc: Value, args: Vec<Source> },
    /// Dwords `first..` of what the vector load at `load` (`global_load_*`)
    /// reads, each lane at its own address (see [`Resources::scalar`]); the
    /// dwords past that load come from loads of the bytes that follow. A
    /// descriptor that differs between lanes.
    Fetched { load: Site, first: u8 },
    /// Picked per lane by `v_cndmask_b32` reading the lane mask whose low
    /// dword is `mask`: the source where the lane's bit is set, then where it
    /// is clear.
    Lanes { mask: Value, args: Vec<Source> },
    /// Per-lane dwords chosen by control flow (VGPRs joined where lanes
    /// that took different paths meet): one source per predecessor of
    /// `block`, in [`crate::ir::Block::preds`] order, `None` on an edge no
    /// lane that accesses the descriptor takes. A descriptor that differs
    /// between lanes.
    LanePhi {
        block: usize,
        args: Vec<Option<Source>>,
    },
}

impl Source {
    /// The sources this one picks from (itself unless a phi).
    pub fn leaves(&self) -> Vec<&Source> {
        match self {
            Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } => {
                args.iter().flat_map(Source::leaves).collect()
            }
            Source::LanePhi { args, .. } => {
                args.iter().flatten().flat_map(Source::leaves).collect()
            }
            s => vec![s],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Image, buffer or `s_buffer_load` resource.
    Resource,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub site: Site,
    pub role: Role,
    pub source: Source,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    /// Every scalar memory access by address (`s_load_*`, `s_store_*`), and
    /// every global memory access (`global_*`, the dynamic term per lane).
    pub scalar: Vec<(Site, Address)>,
    pub descriptors: Vec<Descriptor>,
    /// Root arguments: value on entry, layout index, dword within the parameter.
    pub params: Vec<(Value, usize, u32)>,
    pub waterfall: Waterfall,
}

/// Waterfall loops over descriptors that differ between lanes: each pass
/// reads the first active lane's descriptor with `v_readfirstlane_b32`,
/// compares it with every lane's own (`v_cmp_eq`), and runs the access with
/// the lanes that match. Their accesses take each lane's own descriptor
/// ([`Source::Fetched`], [`Source::Lanes`]), so the first lane's dwords are
/// each lane's own too: the compares hold in every lane and one pass serves
/// all lanes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Waterfall {
    /// The compares, true in every lane.
    pub compares: Vec<Site>,
    /// The `v_readfirstlane_b32` results they compare, each the lane's own
    /// source dword.
    pub reads: Vec<Value>,
}

/// Traces every descriptor and scalar address of `p`, which runs as
/// `stage`, to the root parameters in `layout`. Fails on the first one it
/// cannot trace. `lanes`, `p` as one lane runs it, tells which paths the
/// lanes that access a per-lane descriptor took and which writes they ran;
/// without it, per-lane descriptors chosen by control flow are refused.
///
/// An address that passes through a VGPR depends on which lanes each write
/// ran in: without `lanes` the program is structured
/// ([`crate::structure::structure`]) the first time one does.
/// [`track_console`] is for a program lowered from the console's form.
pub fn track(
    p: &Program,
    layout: &[RootParam],
    stage: HwStage,
    lanes: Option<&Structured>,
) -> Result<Resources, ShaderError> {
    trace(Tracer::new(p, layout, stage, lanes, lanes.map(|l| (l, false)))?)
}

/// [`track`] for a program lowered from `console`, the form every lane of
/// the console wave runs through every block
/// ([`crate::structure::structure_console`]), where EXEC alone masks each
/// VGPR write. `lanes` is the program as one lane runs it, as for [`track`].
pub fn track_console(
    p: &Program,
    layout: &[RootParam],
    stage: HwStage,
    lanes: &Structured,
    console: &Structured,
) -> Result<Resources, ShaderError> {
    trace(Tracer::new(p, layout, stage, Some(lanes), Some((console, true)))?)
}

fn trace(t: Tracer) -> Result<Resources, ShaderError> {
    let p = t.p;
    let mut out = Resources {
        params: p
            .inputs
            .iter()
            .filter_map(|&(reg, v)| {
                let (param, dword) = t.param_of(reg)?;
                Some((v, param, dword))
            })
            .collect(),
        ..Resources::default()
    };
    // Group accesses in the same loop so a shared mask can select several
    // descriptors together.
    let mut first_lane_at = std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>)>::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let fail = |what| ShaderError::Unsupported {
                offset: inst.machine.offset,
                what,
            };
            let mut first_lane = Vec::new();
            let mut descriptor = |op: &Operand, role| -> Result<(), ShaderError> {
                let source = t.descriptor(inst, op, b).map_err(fail)?;
                first_lane.extend(t.first_lane_words(inst, op).unwrap_or_default());
                out.descriptors.push(Descriptor { site, role, source });
                Ok(())
            };
            match &inst.fields {
                Fields::Image { rsrc, sampler, .. } => {
                    descriptor(rsrc, Role::Resource)?;
                    if let Some(s) = sampler {
                        descriptor(s, Role::Sampler)?;
                    }
                }
                Fields::Buffer { rsrc, .. } => descriptor(rsrc, Role::Resource)?,
                Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_") => {
                    descriptor(base, Role::Resource)?;
                }
                Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } if inst.name().contains("_load_") || inst.name().contains("_store_") => {
                    let address = t.smem_address(inst, base, *offset, soffset).map_err(fail)?;
                    out.scalar.push((site, address));
                }
                Fields::Flat {
                    addr,
                    saddr,
                    offset,
                    ..
                } if inst.name().starts_with("global_") => {
                    let address = t
                        .flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
                        .map_err(fail)?;
                    out.scalar.push((site, address));
                }
                _ => {}
            }
            if !first_lane.is_empty() {
                let (accesses, reads) = first_lane_at.entry(b).or_default();
                accesses.push(i);
                reads.extend(first_lane);
            }
        }
    }
    // The proof needs the per-lane form. Without it nothing is emitted, and
    // its failure is the one to report, so the reads stay unproven here.
    if t.lanes.is_none() {
        return Ok(out);
    }
    for (block, (accesses, mut reads)) in first_lane_at {
        reads.sort_unstable();
        reads.dedup();
        let proof = t
            .waterfall(block, &accesses, &reads)
            .map_err(|what| ShaderError::Unsupported {
                offset: t.inst((block, accesses[0])).machine.offset,
                what,
            })?;
        out.waterfall.compares.extend(proof.compares);
        out.waterfall.reads.extend(proof.reads);
    }
    Ok(out)
}

/// A dword after copy propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Const(u32),
    Value(Value),
}

/// A 32-bit sum: optional root parameter (as its first dword), constant,
/// optional computed term, and whether non-linear/multi-term arithmetic was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sum {
    base: Option<Value>,
    /// With a base and `multi`: the least the constants added to it can sum
    /// to.
    offset: u32,
    dynamic: Option<Value>,
    multi: bool,
    /// With a base: how far above `offset` the constants can sum to, over
    /// the phi edges and masked writes that reach the value (0: `offset`
    /// exactly). `None` where no constant bounds them.
    spread: Option<u32>,
}

/// The stretch of constants (least, spread) that covers `a` and `b`; modulo
/// 2^32, the shorter way round.
fn stretch(a: (u32, Option<u32>), b: (u32, Option<u32>)) -> (u32, Option<u32>) {
    let (first, second) = if b.0.wrapping_sub(a.0) as i32 >= 0 {
        (a, b)
    } else {
        (b, a)
    };
    let gap = second.0.wrapping_sub(first.0);
    let spread = first
        .1
        .zip(second.1)
        .and_then(|(x, y)| Some(x.max(gap.checked_add(y)?)));
    (first.0, spread)
}

/// Why an address is refused when an EXEC mask kept its register from some
/// lanes that go on to use it, and those lanes hold no address of its base.
const MASKED: &str = "an address written under an EXEC mask over another base";

/// What reads a value.
#[derive(Debug, Clone, Copy)]
enum Reader {
    Phi(usize, Value),
    Inst(Site),
}

#[derive(Debug, Clone, Copy)]
struct Trace {
    sum: Sum,
    /// Shallowest active phi index in `visiting` depended on, if any.
    cycle: Option<usize>,
}

impl Trace {
    fn has_ptr(&self) -> bool {
        self.sum.base.is_some() || self.cycle.is_some()
    }
}

struct Tracer<'a> {
    p: &'a Program,
    layout: &'a [RootParam],
    stage: HwStage,
    /// Defining instruction per value (phis and inputs have none).
    defs: Vec<Option<Site>>,
    /// Block and index of each phi value.
    phis: Vec<Option<(usize, usize)>>,
    /// Register per input value.
    inputs: Vec<Option<Reg>>,
    origin: Vec<Term>,
    lanes: Option<&'a Structured>,
    /// Reverse-postorder index per block.
    rpo: Vec<usize>,
    /// Memoized non-cyclic sums by value.
    sums: RefCell<HashMap<Value, Sum>>,
    /// The structure the program is lowered from, and whether it is the
    /// console's ([`track_console`]).
    given: Option<(&'a Structured, bool)>,
    /// The program's per-lane structure when none is given, made on first
    /// use; `None` for a program that does not structure (never lowered).
    own: OnceCell<Option<Structured>>,
    /// Readers per value, made on first use.
    readers: OnceCell<HashMap<Value, Vec<Reader>>>,
    /// Memoized [`Self::kept`] by VGPR write.
    kept: RefCell<HashMap<Value, Option<Value>>>,
    /// VGPR writes whose address [`Self::masked`] dropped.
    lost: RefCell<HashSet<Value>>,
}

/// What lanes are known to satisfy: each [`Bool`] where its flag is set,
/// its negation where not.
type Known = Vec<(Bool, bool)>;

const MAX_ROUNDS: usize = 64;
const MAX_JOINS: usize = 8;

impl<'a> Tracer<'a> {
    fn new(
        p: &'a Program,
        layout: &'a [RootParam],
        stage: HwStage,
        lanes: Option<&'a Structured>,
        given: Option<(&'a Structured, bool)>,
    ) -> Result<Self, ShaderError> {
        let all = || {
            p.inputs
                .iter()
                .map(|i| i.1)
                .chain(p.blocks.iter().flat_map(|b| {
                    b.phis
                        .iter()
                        .map(|phi| phi.value)
                        .chain(b.insts.iter().flat_map(|i| i.writes.iter().map(|w| w.1)))
                }))
        };
        let n = all().map(|v| v.0 as usize + 1).max().unwrap_or(0);
        let mut defs = vec![None; n];
        let mut phis = vec![None; n];
        let mut inputs = vec![None; n];
        for &(reg, v) in &p.inputs {
            inputs[v.0 as usize] = Some(reg);
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phis[phi.value.0 as usize] = Some((b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs[v.0 as usize] = Some((b, i));
                }
            }
        }
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (n, b) in crate::ir::reverse_postorder(&p.blocks)
            .into_iter()
            .enumerate()
        {
            rpo[b] = n;
        }
        let mut t = Tracer {
            p,
            layout,
            stage,
            defs,
            phis,
            inputs,
            origin: Vec::new(),
            lanes,
            rpo,
            sums: RefCell::new(HashMap::new()),
            given,
            own: OnceCell::new(),
            readers: OnceCell::new(),
            kept: RefCell::new(HashMap::new()),
            lost: RefCell::new(HashSet::new()),
        };
        t.origin = t.propagate_copies(n)?;
        Ok(t)
    }

    /// Optimistic copy propagation: a move's result is its source's
    /// origin, a phi's the one origin its arguments agree on.
    fn propagate_copies(&self, n: usize) -> Result<Vec<Term>, ShaderError> {
        let mut o: Vec<Option<Term>> = vec![None; n];
        for &(_, v) in &self.p.inputs {
            o[v.0 as usize] = Some(Term::Value(v));
        }
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for &b in &order {
                let block = &self.p.blocks[b];
                for phi in &block.phis {
                    let own = Term::Value(phi.value);
                    let mut args = phi
                        .args
                        .iter()
                        .filter_map(|a| o[a.0 as usize])
                        .filter(|&t| t != own);
                    let new = args
                        .next()
                        .map(|first| if args.all(|t| t == first) { first } else { own });
                    if new.is_some() && new != o[phi.value.0 as usize] {
                        o[phi.value.0 as usize] = new;
                        changed = true;
                    }
                }
                for inst in &block.insts {
                    for (k, &(_, v)) in inst.writes.iter().enumerate() {
                        let new = match copy_source(inst, k).or_else(|| self.restored(inst)) {
                            Some(Moved::Const(c)) => Some(Term::Const(c)),
                            Some(Moved::Value(s)) => o[s.0 as usize],
                            None => Some(Term::Value(v)),
                        };
                        if new.is_some() && new != o[v.0 as usize] {
                            o[v.0 as usize] = new;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                return Ok(o
                    .into_iter()
                    .enumerate()
                    .map(|(v, t)| t.unwrap_or(Term::Value(Value(v as u32))))
                    .collect());
            }
        }
        Err(ShaderError::Unsupported {
            offset: 0,
            what: "copies that do not settle",
        })
    }

    fn inst(&self, (b, i): Site) -> &'a Inst {
        &self.p.blocks[b].insts[i]
    }

    fn def(&self, v: Value) -> Option<&'a Inst> {
        self.defs[v.0 as usize].map(|s| self.inst(s))
    }

    fn origin(&self, v: Value) -> Term {
        self.origin[v.0 as usize]
    }

    /// The origin of a one-dword operand as `inst` reads it.
    fn term(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code))?)),
            Operand::Const(c) => inline(c).map(Term::Const),
            Operand::Literal(x) => Some(Term::Const(x)),
            _ => None,
        }
    }

    /// Layout index and dword of the root parameter `reg` holds on entry.
    fn param_of(&self, reg: Reg) -> Option<(usize, u32)> {
        let code = match reg {
            Reg::S(code) if !self.p.spilled => code,
            Reg::Spill(code) => code,
            _ => return None,
        };
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let first = self.stage.user_sgpr(p.user_data);
            let dword = u32::from(code).checked_sub(first.into())?;
            (dword < p.sgprs).then_some((i, dword))
        })
    }

    /// The SOP2 operands of `v` if an `s_add` with one of `names` wrote it.
    fn sop2(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, [Term; 2])> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) || inst.writes.first()?.1 != v {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        Some((
            inst,
            [
                self.term(inst, src.first()?)?,
                self.term(inst, src.get(1)?)?,
            ],
        ))
    }

    fn is_base_root(&self, param: usize) -> bool {
        matches!(
            self.layout[param].root,
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap | RootType::Descriptor
        )
    }

    fn or_align_limit(&self, base: Value) -> Option<u32> {
        if self.getpc(base).is_some() {
            return Some(4);
        }
        let reg = self.inputs[base.0 as usize]?;
        let (param, 0) = self.param_of(reg)? else {
            return None;
        };
        match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => Some(16),
            RootType::Descriptor => Some(4),
            _ => None,
        }
    }

    fn valu_ops(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, &'a Valu)> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) {
            return None;
        }
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() || f.clamp || f.omod != 0
        {
            return None;
        }
        let writes_v = match f.dst? {
            Operand::Vector { reg, .. } => inst.write(Reg::V(reg)) == Some(v),
            Operand::Scalar { code, .. } => inst.write(Reg::S(code)) == Some(v),
            _ => false,
        };
        writes_v.then_some((inst, f))
    }

    fn valu_bin(&self, v: Value, names: &[&str]) -> Option<[Term; 2]> {
        let (inst, f) = self.valu_ops(v, names)?;
        if !matches!(f.dst, Some(Operand::Vector { .. })) {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    fn valu_tri(&self, v: Value, names: &[&str]) -> Option<[Term; 3]> {
        let (inst, f) = self.valu_ops(v, names)?;
        if !matches!(f.dst, Some(Operand::Vector { .. })) {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
            self.vterm(inst, f.src.get(2)?)?,
        ])
    }

    /// Whether a VALU source is wave-uniform (a scalar/constant, or copied from
    /// one through unmodified `v_mov_b32`s).
    fn uniform_valu_src(&self, mut inst: &'a Inst, mut op: &'a Operand) -> bool {
        loop {
            match *op {
                // A spill argument is one dword for the wave, like a scalar.
                Operand::Scalar { .. }
                | Operand::Spill(_)
                | Operand::Const(_)
                | Operand::Literal(_) => return true,
                Operand::Vector { reg, .. } => {
                    let Some(val) = inst.read(Reg::V(reg)) else {
                        return false;
                    };
                    let Some((next, f)) = self.valu_ops(val, &["v_mov_b32"]) else {
                        return false;
                    };
                    if self.kept(val).is_some() {
                        return false;
                    }
                    let Some(src) = f.src.first() else {
                        return false;
                    };
                    inst = next;
                    op = src;
                }
                Operand::Attr { .. } => return false,
            }
        }
    }

    /// Address-local `v_readfirstlane_b32` or `v_readlane_b32` reading `v`.
    fn lane_read(&self, v: Value) -> Option<(Term, bool)> {
        let (inst, f) = self.valu_ops(v, &["v_readfirstlane_b32", "v_readlane_b32"])?;
        if !matches!(f.dst, Some(Operand::Scalar { .. })) {
            return None;
        }
        let src0 = f.src.first()?;
        let uniform = self.uniform_valu_src(inst, src0);
        let t = self.vterm(inst, src0)?;
        Some((t, uniform))
    }

    /// A VA-preserving low-bit mask (`s_and_b32` / `v_and_b32` with `mask & 0xFF == 0xFF`).
    fn va_masked(&self, v: Value) -> Option<Term> {
        let [a, b] = self
            .sop2(v, &["s_and_b32"])
            .map(|(_, t)| t)
            .or_else(|| self.valu_bin(v, &["v_and_b32"]))?;
        match [a, b] {
            [x, Term::Const(mask)] | [Term::Const(mask), x] if mask & 0xFF == 0xFF => Some(x),
            _ => None,
        }
    }

    fn strip_va_mask(&self, mut t: Term) -> Term {
        while let Term::Value(v) = t {
            if self.kept(v).is_some() {
                break;
            } else if let Some(x) = self.va_masked(v) {
                t = x;
            } else if let Some((x, _)) = self.lane_read(v) {
                t = x;
            } else {
                break;
            }
        }
        t
    }

    fn sum(&self, t: Term) -> Sum {
        self.sum_rec(t, &mut Vec::new(), &mut Vec::new()).sum
    }

    /// The structure the program is lowered from and whether it is the
    /// console's: the one given, else the program's per-lane structure.
    fn structured(&self) -> Option<(&Structured, bool)> {
        self.given.or_else(|| {
            let own = || crate::structure::structure(self.p).ok();
            Some((self.own.get_or_init(own).as_ref()?, false))
        })
    }

    /// The lane's EXEC bit at the vector instruction at `site`.
    fn exec(&self, (b, i): Site) -> Option<Bool> {
        self.structured()?.0.execs[b].get(i).copied().flatten()
    }

    /// Whether the vector instruction at `site` runs under an EXEC mask that
    /// leaves out lanes that go on: per lane, lanes that reach it; in the
    /// console's form, where every lane reaches every block, any lane.
    fn narrowed(&self, site: Site) -> bool {
        match self.structured() {
            Some((_, true)) => self.exec(site).is_some_and(|e| e != Bool::TRUE),
            Some((st, false)) => st.guards[site.0][site.1].is_some(),
            None => false,
        }
    }

    /// For a VGPR write under an EXEC mask: the register's old value, which
    /// the lanes the mask leaves out keep, if any of them can go on to read
    /// the register. `None` for a write whose readers all run in lanes that
    /// ran it too (a load under the same mask, say): they see what it wrote.
    fn kept(&self, v: Value) -> Option<Value> {
        let site = self.defs[v.0 as usize]?;
        let inst = self.inst(site);
        let &(reg @ Reg::V(_), _) = inst.writes.iter().find(|w| w.1 == v)? else {
            return None;
        };
        let old = inst.read(reg)?;
        if let Some(&kept) = self.kept.borrow().get(&v) {
            return kept;
        }
        let kept = (self.narrowed(site) && self.leaks(v, self.exec(site), &mut Vec::new()))
            .then_some(old);
        self.kept.borrow_mut().insert(v, kept);
        kept
    }

    /// Whether `x` can be read in a lane where `cover` (an EXEC bit; `None`:
    /// nothing) did not hold: by an instruction whose EXEC does not imply
    /// it, through a phi, or carried on in the result of a later masked
    /// write to its register.
    fn leaks(&self, x: Value, cover: Option<Bool>, seen: &mut Vec<(Value, bool)>) -> bool {
        if seen.contains(&(x, cover.is_some())) {
            return false;
        }
        seen.push((x, cover.is_some()));
        let readers = self.readers.get_or_init(|| {
            let mut readers: HashMap<Value, Vec<Reader>> = HashMap::new();
            for (b, block) in self.p.blocks.iter().enumerate() {
                for phi in &block.phis {
                    for &a in &phi.args {
                        readers.entry(a).or_default().push(Reader::Phi(b, phi.value));
                    }
                }
                for (i, inst) in block.insts.iter().enumerate() {
                    for &(_, r) in &inst.reads {
                        readers.entry(r).or_default().push(Reader::Inst((b, i)));
                    }
                }
            }
            readers
        });
        let Some(readers) = readers.get(&x) else {
            return false;
        };
        readers.iter().any(|&r| match r {
            Reader::Phi(b, phi) => {
                // A loop can redefine mask atoms on the back edge. Keep
                // coverage only if all its atoms predate the join.
                let cover = cover.filter(|&c| {
                    self.structured()
                        .is_some_and(|(st, _)| st.bools.defined_before(c, self.rpo[b]))
                });
                self.leaks(phi, cover, seen)
            }
            Reader::Inst(site) => {
                let inst = self.inst(site);
                // Lane reads by index take a lane whatever EXEC is.
                let covered = || {
                    let (Some((st, _)), Some(c), Some(e)) = (self.structured(), cover, self.exec(site))
                    else {
                        return false;
                    };
                    !matches!(inst.name(), "v_readlane_b32" | "v_writelane_b32")
                        && st.bools.within(e, c)
                };
                if operand(inst, x) && !covered() {
                    return true;
                }
                // The lanes its own mask leaves out carry `x` on in its result.
                self.narrowed(site)
                    && inst.writes.iter().any(|&(reg, y)| {
                        matches!(reg, Reg::V(_))
                            && inst.read(reg) == Some(x)
                            && self.leaks(y, cover, seen)
                    })
            }
        })
    }

    /// `new`, the trace of what the instruction writing `v` computes, as the
    /// register holds it after a write under an EXEC mask whose other lanes
    /// go on to read it ([`Self::kept`]): an address stays one only over an
    /// address of the same base, which the lowering then picks per lane;
    /// anything else is the register as it is.
    fn masked(
        &self,
        v: Value,
        new: Trace,
        visiting: &mut Vec<Value>,
        frames: &mut Vec<HashMap<Value, Trace>>,
    ) -> Trace {
        let Some(old) = self.kept(v) else {
            return new;
        };
        let as_is = Trace {
            sum: Sum {
                base: None,
                offset: 0,
                dynamic: Some(v),
                multi: false,
                spread: Some(0),
            },
            cycle: None,
        };
        if !new.has_ptr() {
            return as_is;
        }
        let old = self.sum_rec(self.vorigin(old), visiting, frames);
        let same = new.sum.base.zip(old.sum.base).is_none_or(|(a, b)| a == b);
        if !old.has_ptr() || !same {
            self.lost.borrow_mut().insert(v);
            return as_is;
        }
        let (offset, spread) = match (new.sum.base, old.sum.base) {
            (Some(_), Some(_)) => stretch(
                (new.sum.offset, new.sum.spread),
                (old.sum.offset, old.sum.spread),
            ),
            // One side is still being traced round a loop.
            _ => (0, None),
        };
        Trace {
            sum: Sum {
                base: new.sum.base.or(old.sum.base),
                offset,
                dynamic: Some(v),
                multi: true,
                spread,
            },
            cycle: match (new.cycle, old.cycle) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
        }
    }

    /// The source of the `v_mov_b32` that wrote VGPR value `v`.
    fn moved(&self, v: Value) -> Option<Term> {
        let (inst, f) = self.valu_ops(v, &["v_mov_b32"])?;
        if !matches!(f.dst, Some(Operand::Vector { .. })) {
            return None;
        }
        self.vterm(inst, f.src.first()?)
    }

    /// Whether the terms `v` is computed from include an address dropped at
    /// a write under an EXEC mask ([`Self::masked`]).
    fn mask_lost(&self, v: Value, seen: &mut Vec<Value>) -> bool {
        if self.lost.borrow().contains(&v) {
            return true;
        }
        if seen.contains(&v) {
            return false;
        }
        seen.push(v);
        let terms: Vec<Option<Term>> = if let Some((b, i)) = self.phis[v.0 as usize] {
            let args = &self.p.blocks[b].phis[i].args;
            args.iter().map(|&a| Some(self.vorigin(a))).collect()
        } else {
            match self.def(v).map(|inst| (inst, &inst.fields)) {
                Some((inst, Fields::Sop { dst, src, .. })) => src
                    .iter()
                    .chain(dst.iter().filter(|_| inst.name() == "s_addk_i32"))
                    .map(|s| self.term(inst, s))
                    .collect(),
                Some((inst, Fields::Valu(f))) => {
                    f.src.iter().map(|s| self.vterm(inst, s)).collect()
                }
                _ => Vec::new(),
            }
        };
        terms
            .into_iter()
            .any(|t| matches!(t, Some(Term::Value(x)) if self.mask_lost(x, seen)))
    }

    fn sum_rec(
        &self,
        t: Term,
        visiting: &mut Vec<Value>,
        frames: &mut Vec<HashMap<Value, Trace>>,
    ) -> Trace {
        let v = match t {
            Term::Const(c) => {
                return Trace {
                    sum: Sum {
                        base: None,
                        offset: c,
                        dynamic: None,
                        multi: false,
                        spread: Some(0),
                    },
                    cycle: None,
                };
            }
            Term::Value(v) => v,
        };
        if let Some(&sum) = self.sums.borrow().get(&v) {
            return Trace { sum, cycle: None };
        }
        if let Some(depth) = visiting.iter().position(|&x| x == v) {
            return Trace {
                sum: Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                    multi: true,
                    spread: None,
                },
                cycle: Some(depth),
            };
        }
        if let Some(&cached) = frames.last().and_then(|f| f.get(&v)) {
            return cached;
        }
        let (res, cache_global) = self.sum_value(v, visiting, frames);
        let res = self.masked(v, res, visiting, frames);
        if cache_global && res.cycle.is_none() {
            self.sums.borrow_mut().insert(v, res.sum);
        } else if let Some(frame) = frames.last_mut() {
            frame.insert(v, res);
        }
        res
    }

    fn sum_value(
        &self,
        v: Value,
        visiting: &mut Vec<Value>,
        frames: &mut Vec<HashMap<Value, Trace>>,
    ) -> (Trace, bool) {
        let plain_dynamic = Trace {
            sum: Sum {
                base: None,
                offset: 0,
                dynamic: Some(v),
                multi: false,
                spread: Some(0),
            },
            cycle: None,
        };
        if let Some(reg) = self.inputs[v.0 as usize] {
            return match self.param_of(reg) {
                Some((param, 0)) if self.is_base_root(param) => (
                    Trace {
                        sum: Sum {
                            base: Some(v),
                            offset: 0,
                            dynamic: None,
                            multi: false,
                            spread: Some(0),
                        },
                        cycle: None,
                    },
                    true,
                ),
                _ => (plain_dynamic, true),
            };
        }
        if self.getpc(v).is_some() {
            return (
                Trace {
                    sum: Sum {
                        base: Some(v),
                        offset: 0,
                        dynamic: None,
                        multi: false,
                        spread: Some(0),
                    },
                    cycle: None,
                },
                true,
            );
        }
        if let Some((phi_b, phi_i)) = self.phis[v.0 as usize] {
            let my_depth = visiting.len();
            visiting.push(v);
            frames.push(HashMap::new());
            let mut shared_base: Option<Value> = None;
            let mut outer_cycle: Option<usize> = None;
            let mut touched_outer = false;
            // The constants over the edges; none once a loop steps the pointer.
            let mut span: Option<(u32, Option<u32>)> = None;
            let mut looped = false;
            let mut ok = !self.p.blocks[phi_b].phis[phi_i].args.is_empty();
            for &arg in &self.p.blocks[phi_b].phis[phi_i].args {
                let e = self.sum_rec(self.vorigin(arg), visiting, frames);
                if let Some(d) = e.cycle
                    && d < my_depth
                {
                    touched_outer = true;
                }
                if !e.has_ptr() {
                    ok = false;
                    break;
                }
                looped |= e.cycle.is_some();
                if let Some(b) = e.sum.base {
                    if shared_base.is_some_and(|sb| sb != b) {
                        ok = false;
                        break;
                    }
                    shared_base = Some(b);
                    let edge = (e.sum.offset, e.sum.spread);
                    span = Some(span.map_or(edge, |s| stretch(s, edge)));
                }
                if let Some(d) = e.cycle
                    && d < my_depth
                {
                    outer_cycle = Some(outer_cycle.map_or(d, |o| o.min(d)));
                }
            }
            frames.pop();
            visiting.pop();
            let res = if ok {
                if let Some(base) = shared_base {
                    let (offset, spread) = span.unwrap_or((0, None));
                    Trace {
                        sum: Sum {
                            base: Some(base),
                            offset,
                            dynamic: Some(v),
                            multi: true,
                            spread: spread.filter(|_| !looped),
                        },
                        cycle: outer_cycle,
                    }
                } else if let Some(d) = outer_cycle {
                    Trace {
                        sum: Sum {
                            base: None,
                            offset: 0,
                            dynamic: None,
                            multi: true,
                            spread: None,
                        },
                        cycle: Some(d),
                    }
                } else {
                    plain_dynamic
                }
            } else {
                plain_dynamic
            };
            return (res, !touched_outer);
        }

        let combine_add = |x: Trace, y: Trace| -> Trace {
            if x.has_ptr() && y.has_ptr() {
                return plain_dynamic;
            }
            if !x.has_ptr() && !y.has_ptr() {
                if x.sum.dynamic.is_some() && y.sum.dynamic.is_some() {
                    return plain_dynamic;
                }
                return Trace {
                    sum: Sum {
                        base: None,
                        offset: x.sum.offset.wrapping_add(y.sum.offset),
                        dynamic: x.sum.dynamic.or(y.sum.dynamic),
                        multi: false,
                        spread: Some(0),
                    },
                    cycle: None,
                };
            }
            let (p, o) = if x.has_ptr() { (x, y) } else { (y, x) };
            let multi = p.sum.multi || (p.sum.dynamic.is_some() && o.sum.dynamic.is_some());
            let dynamic = if multi {
                Some(v)
            } else {
                p.sum.dynamic.or(o.sum.dynamic)
            };
            Trace {
                sum: Sum {
                    base: p.sum.base,
                    offset: p.sum.offset.wrapping_add(o.sum.offset),
                    dynamic,
                    multi,
                    spread: p.sum.spread,
                },
                cycle: p.cycle,
            }
        };

        let Some(inst) = self.def(v) else {
            return (plain_dynamic, true);
        };

        if let Some(src_t) = self.moved(v) {
            return (self.sum_rec(src_t, visiting, frames), true);
        }

        if let Some((src_t, uniform)) = self.lane_read(v) {
            let inner = self.sum_rec(src_t, visiting, frames);
            if !inner.has_ptr() {
                return if inner.sum.dynamic.is_none() {
                    (inner, true)
                } else {
                    (plain_dynamic, true)
                };
            }
            if uniform || (!inner.sum.multi && inner.sum.dynamic.is_none() && inner.cycle.is_none())
            {
                return (inner, true);
            }
            return (
                Trace {
                    sum: Sum {
                        base: inner.sum.base,
                        offset: inner.sum.offset,
                        dynamic: Some(v),
                        multi: true,
                        spread: inner.sum.spread,
                    },
                    cycle: inner.cycle,
                },
                true,
            );
        }

        if let Some([a, b]) = self
            .sop2(v, &["s_add_u32", "s_add_i32"])
            .map(|(_, t)| t)
            .or_else(|| self.valu_bin(v, &["v_add_co_u32", "v_add_nc_u32", "v_add_nc_i32"]))
        {
            let x = self.sum_rec(a, visiting, frames);
            let y = self.sum_rec(b, visiting, frames);
            return (combine_add(x, y), true);
        }

        if inst.name() == "s_addk_i32"
            && let Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                imm: Some(imm),
                ..
            } = &inst.fields
            && inst.writes.first().is_some_and(|w| w.1 == v)
            && let Some(prev) = inst.read(Reg::S(*code)).map(|p| self.origin(p))
        {
            let x = self.sum_rec(prev, visiting, frames);
            let k = Trace {
                sum: Sum {
                    base: None,
                    offset: i32::from(*imm as i16) as u32,
                    dynamic: None,
                    multi: false,
                    spread: Some(0),
                },
                cycle: None,
            };
            return (combine_add(x, k), true);
        }

        if let Some((inst, [a, b])) = self.sop2(
            v,
            &[
                "s_lshl1_add_u32",
                "s_lshl2_add_u32",
                "s_lshl3_add_u32",
                "s_lshl4_add_u32",
            ],
        ) {
            let shift = u32::from(inst.name().as_bytes()[6] - b'0');
            let x = self.sum_rec(a, visiting, frames);
            let y = self.sum_rec(b, visiting, frames);
            if x.has_ptr() {
                return (plain_dynamic, true);
            }
            if !y.has_ptr() {
                if x.sum.dynamic.is_none() && y.sum.dynamic.is_none() {
                    return (
                        Trace {
                            sum: Sum {
                                base: None,
                                offset: x.sum.offset.wrapping_shl(shift).wrapping_add(y.sum.offset),
                                dynamic: None,
                                multi: false,
                                spread: Some(0),
                            },
                            cycle: None,
                        },
                        true,
                    );
                }
                return (plain_dynamic, true);
            }
            if x.sum.dynamic.is_none() {
                let k = Trace {
                    sum: Sum {
                        base: None,
                        offset: x.sum.offset.wrapping_shl(shift),
                        dynamic: None,
                        multi: false,
                        spread: Some(0),
                    },
                    cycle: None,
                };
                return (combine_add(y, k), true);
            }
            return (
                Trace {
                    sum: Sum {
                        base: y.sum.base,
                        offset: y.sum.offset,
                        dynamic: Some(v),
                        multi: true,
                        spread: y.sum.spread,
                    },
                    cycle: y.cycle,
                },
                true,
            );
        }

        if let Some([a, sh, b]) = self.valu_tri(v, &["v_lshl_add_u32"]) {
            let ta = self.sum_rec(a, visiting, frames);
            let tsh = self.sum_rec(sh, visiting, frames);
            let tb = self.sum_rec(b, visiting, frames);
            if ta.has_ptr() || tsh.has_ptr() {
                return (plain_dynamic, true);
            }
            if !tb.has_ptr() {
                if ta.sum.dynamic.is_none() && tsh.sum.dynamic.is_none() && tb.sum.dynamic.is_none()
                {
                    return (
                        Trace {
                            sum: Sum {
                                base: None,
                                offset: ta
                                    .sum
                                    .offset
                                    .wrapping_shl(tsh.sum.offset & 31)
                                    .wrapping_add(tb.sum.offset),
                                dynamic: None,
                                multi: false,
                                spread: Some(0),
                            },
                            cycle: None,
                        },
                        true,
                    );
                }
                return (plain_dynamic, true);
            }
            if ta.sum.dynamic.is_none() && tsh.sum.dynamic.is_none() {
                let k = Trace {
                    sum: Sum {
                        base: None,
                        offset: ta.sum.offset.wrapping_shl(tsh.sum.offset & 31),
                        dynamic: None,
                        multi: false,
                        spread: Some(0),
                    },
                    cycle: None,
                };
                return (combine_add(tb, k), true);
            }
            return (
                Trace {
                    sum: Sum {
                        base: tb.sum.base,
                        offset: tb.sum.offset,
                        dynamic: Some(v),
                        multi: true,
                        spread: tb.sum.spread,
                    },
                    cycle: tb.cycle,
                },
                true,
            );
        }

        if let Some([a, b, c]) = self.valu_tri(v, &["v_add3_u32"]) {
            let ta = self.sum_rec(a, visiting, frames);
            let tb = self.sum_rec(b, visiting, frames);
            let tc = self.sum_rec(c, visiting, frames);
            let ptrs =
                usize::from(ta.has_ptr()) + usize::from(tb.has_ptr()) + usize::from(tc.has_ptr());
            if ptrs > 1 {
                return (plain_dynamic, true);
            }
            return (combine_add(combine_add(ta, tb), tc), true);
        }

        if let Some((minuend, subtrahend)) = self
            .sop2(v, &["s_sub_u32", "s_sub_i32"])
            .map(|(_, [a, b])| (a, b))
            .or_else(|| {
                self.valu_bin(v, &["v_sub_co_u32", "v_sub_nc_u32"])
                    .map(|[a, b]| (a, b))
            })
            .or_else(|| {
                self.valu_bin(v, &["v_subrev_co_u32", "v_subrev_nc_u32"])
                    .map(|[a, b]| (b, a))
            })
        {
            let m = self.sum_rec(minuend, visiting, frames);
            let s = self.sum_rec(subtrahend, visiting, frames);
            if s.has_ptr() {
                return (plain_dynamic, true);
            }
            if !m.has_ptr() {
                if m.sum.dynamic.is_none() && s.sum.dynamic.is_none() {
                    return (
                        Trace {
                            sum: Sum {
                                base: None,
                                offset: m.sum.offset.wrapping_sub(s.sum.offset),
                                dynamic: None,
                                multi: false,
                                spread: Some(0),
                            },
                            cycle: None,
                        },
                        true,
                    );
                }
                return (plain_dynamic, true);
            }
            // A constant taken off moves the constants; a run-time term
            // taken off leaves nothing that bounds them.
            return (
                Trace {
                    sum: Sum {
                        base: m.sum.base,
                        offset: m.sum.offset.wrapping_sub(s.sum.offset),
                        dynamic: Some(v),
                        multi: true,
                        spread: m.sum.spread.filter(|_| s.sum.dynamic.is_none()),
                    },
                    cycle: m.cycle,
                },
                true,
            );
        }

        if let Some([a, b]) = self
            .sop2(v, &["s_or_b32"])
            .map(|(_, t)| t)
            .or_else(|| self.valu_bin(v, &["v_or_b32"]))
        {
            let x = self.sum_rec(a, visiting, frames);
            let y = self.sum_rec(b, visiting, frames);
            let or_align = |p: Trace, k: Trace| -> Option<Trace> {
                if p.cycle.is_some() || p.sum.multi || p.sum.dynamic.is_some() {
                    return None;
                }
                let base = p.sum.base?;
                if k.has_ptr() || k.sum.dynamic.is_some() {
                    return None;
                }
                let c = k.sum.offset;
                let lim = self.or_align_limit(base)?;
                ((p.sum.offset & c) == 0 && c < lim).then_some(Trace {
                    sum: Sum {
                        base: Some(base),
                        offset: p.sum.offset | c,
                        dynamic: None,
                        multi: false,
                        spread: Some(0),
                    },
                    cycle: None,
                })
            };
            if let Some(res) = or_align(x, y).or_else(|| or_align(y, x)) {
                return (res, true);
            }
            return (plain_dynamic, true);
        }

        (plain_dynamic, true)
    }

    /// The `s_getpc_b64` writing `lo` as its low dword.
    fn getpc(&self, lo: Value) -> Option<&'a Inst> {
        let inst = self.def(lo)?;
        (inst.name() == "s_getpc_b64" && inst.writes.first().is_some_and(|w| w.1 == lo))
            .then_some(inst)
    }

    /// The origin of a VALU operand.
    fn vterm(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(self.vorigin(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// A VGPR value's origin through `v_mov_b32` copies, as every lane that
    /// reads the register holds it: a copy made under an EXEC mask whose
    /// other lanes go on to read the register ([`Self::kept`]) is not
    /// followed.
    fn vorigin(&self, v: Value) -> Term {
        self.copied(v, true)
    }

    /// A VGPR value's origin through `v_mov_b32` copies: with `every_lane`
    /// as [`Self::vorigin`] gives it, else in the lanes that ran the copies.
    fn copied(&self, v: Value, every_lane: bool) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && !(every_lane && self.kept(v).is_some())
            && let Some(t) = f.src.first().and_then(|s| match *s {
                Operand::Vector { reg, .. } => {
                    Some(self.copied(inst.read(Reg::V(reg))?, every_lane))
                }
                op => self.term(inst, &op),
            })
        {
            return t;
        }
        self.origin(v)
    }

    /// A global memory address: an SGPR pair plus a VGPR offset, or a VGPR
    /// pair.
    fn flat_address(
        &self,
        inst: &Inst,
        addr: Option<&Operand>,
        saddr: Option<&Operand>,
        offset: i16,
    ) -> Result<Address, &'static str> {
        let Some(&Operand::Vector { reg, .. }) = addr else {
            return Err("a global address not in VGPRs");
        };
        let off = offset as i32 as u32;
        match saddr {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("an address not in SGPRs");
                };
                let v = inst.read(Reg::V(reg)).ok_or("a global address")?;
                let mut extra = self.sum(self.vorigin(v));
                if extra.base.is_some() {
                    return Err("an offset from a root parameter");
                }
                extra.offset = extra.offset.wrapping_add(off);
                self.address(lo, hi, extra)
            }
            _ => {
                let word = |k| inst.read(Reg::V(reg + k)).map(|v| self.vorigin(v));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("a global address");
                };
                self.address(
                    lo,
                    hi,
                    Sum {
                        base: None,
                        offset: off,
                        dynamic: None,
                        multi: false,
                        spread: Some(0),
                    },
                )
            }
        }
    }

    /// The address `lo` and `hi` hold, plus `extra` (an instruction's
    /// immediate and offset register).
    fn address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        let s = self.sum(lo);
        let base = s.base.ok_or_else(|| self.why_no_base(s.dynamic))?;
        let use_rel = s.multi || (s.dynamic.is_some() && extra.dynamic.is_some());
        let lo_val = match lo {
            Term::Value(v) => v,
            Term::Const(_) => unreachable!("base implies value"),
        };
        let rel = use_rel.then_some(RelOffset {
            value: lo_val,
            base,
            extra_dynamic: extra.dynamic,
            constant: s.spread.map(|spread| (s.offset, spread)),
        });
        let dynamic = if rel.is_some() {
            extra.dynamic.or(Some(lo_val))
        } else {
            s.dynamic.or(extra.dynamic)
        };
        if let Some(pc) = self.getpc(base) {
            // A table in the code: the base is the next instruction's offset.
            let high = pc.writes.get(1).map(|w| w.1);
            let want = high.map_or(Term::Const(0), Term::Value);
            if !self.high_half(hi, high)
                && !(high.is_some() && self.carried(hi, want, &mut Vec::new()))
            {
                return Err("an address with an untracked high half");
            }
            let offset = if rel.is_some() {
                (pc.machine.offset as u32 + 4).wrapping_add(extra.offset)
            } else {
                (pc.machine.offset as u32 + 4)
                    .wrapping_add(s.offset)
                    .wrapping_add(extra.offset)
            };
            return Ok(Address {
                param: CODE,
                offset,
                dynamic,
                rel,
            });
        }
        let reg = self.inputs[base.0 as usize].ok_or("an address not from a root parameter")?;
        let (param, _) = self
            .param_of(reg)
            .ok_or("an address not from a root parameter")?;
        let high = match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => None,
            RootType::Descriptor => {
                let code = self.stage.user_sgpr(self.layout[param].user_data) + 1;
                let high = match reg {
                    Reg::Spill(_) => Reg::Spill(code),
                    _ => Reg::S(code),
                };
                let input = self.p.inputs.iter().find(|i| i.0 == high);
                Some(input.ok_or("a root descriptor without its high half")?.1)
            }
            _ => return Err("an address from a root parameter that is no address"),
        };
        // shortcut: with a computed term, a high half that adds anything to the
        // base's is taken as the carry of that term (offsets below 4 GB).
        let want = high.map_or(Term::Const(0), Term::Value);
        if !self.high_half(hi, high)
            && !(dynamic.is_some() && self.carried(hi, want, &mut Vec::new()))
        {
            return Err("an address with an untracked high half");
        }
        let offset = if rel.is_some() {
            extra.offset
        } else {
            s.offset.wrapping_add(extra.offset)
        };
        Ok(Address {
            param,
            offset,
            dynamic,
            rel,
        })
    }

    /// `hi` is `high` (0 for a table), possibly plus a carry.
    fn high_half(&self, hi: Term, high: Option<Value>) -> bool {
        let want = high.map_or(Term::Const(0), Term::Value);
        let hi = self.strip_va_mask(hi);
        if hi == want {
            return true;
        }
        let Term::Value(v) = hi else {
            return false;
        };
        // Lanes an EXEC mask left with the old high half use that one.
        if let Some(old) = self.kept(v)
            && !self.high_half(self.vorigin(old), high)
        {
            return false;
        }
        if let Some(src) = self.moved(v) {
            return self.high_half(src, high);
        }
        let Some([a, b]) = self
            .sop2(v, &["s_addc_u32"])
            .map(|(_, t)| t)
            .or_else(|| self.valu_bin(v, &["v_add_co_ci_u32"]))
        else {
            return false;
        };
        let (a, b) = (self.strip_va_mask(a), self.strip_va_mask(b));
        a == Term::Const(0) && self.high_half(b, high)
            || b == Term::Const(0) && self.high_half(a, high)
    }

    /// `hi` adds something (with carry) to `want` (`high` or 0 for a table),
    /// through carry-adds, adds, subtractions (minuend), VA-preserving masks,
    /// and high-half phis.
    fn carried(&self, hi: Term, want: Term, seen: &mut Vec<Value>) -> bool {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Carry {
            No,
            Cycle,
            Yes,
        }
        fn best(a: Carry, b: Carry) -> Carry {
            match (a, b) {
                (Carry::Yes, _) | (_, Carry::Yes) => Carry::Yes,
                (Carry::Cycle, _) | (_, Carry::Cycle) => Carry::Cycle,
                _ => Carry::No,
            }
        }
        fn rec(t: &Tracer<'_>, hi: Term, want: Term, seen: &mut Vec<Value>) -> Carry {
            let hi = t.strip_va_mask(hi);
            if hi == want {
                return Carry::Yes;
            }
            let Term::Value(v) = hi else {
                return Carry::No;
            };
            if seen.contains(&v) {
                return Carry::Cycle;
            }
            // Lanes an EXEC mask left with the old high half use that one.
            if let Some(old) = t.kept(v) {
                seen.push(v);
                let old = rec(t, t.vorigin(old), want, seen);
                seen.pop();
                if old == Carry::No {
                    return Carry::No;
                }
            }
            if let Some(src) = t.moved(v) {
                return rec(t, src, want, seen);
            }
            if let Some((phi_b, phi_i)) = t.phis[v.0 as usize] {
                if t.p.blocks[phi_b].phis[phi_i].args.is_empty() {
                    return Carry::No;
                }
                seen.push(v);
                let mut any_yes = false;
                for &arg in &t.p.blocks[phi_b].phis[phi_i].args {
                    match rec(t, t.vorigin(arg), want, seen) {
                        Carry::No => {
                            seen.pop();
                            return Carry::No;
                        }
                        Carry::Yes => any_yes = true,
                        Carry::Cycle => {}
                    }
                }
                seen.pop();
                return if any_yes { Carry::Yes } else { Carry::Cycle };
            }
            if let Some([a, b]) = t
                .sop2(v, &["s_addc_u32", "s_add_u32", "s_add_i32"])
                .map(|(_, args)| args)
                .or_else(|| {
                    t.valu_bin(
                        v,
                        &[
                            "v_add_co_ci_u32",
                            "v_add_nc_u32",
                            "v_add_co_u32",
                            "v_add_nc_i32",
                        ],
                    )
                })
            {
                return best(rec(t, a, want, seen), rec(t, b, want, seen));
            }
            if let Some(minuend) = t
                .sop2(v, &["s_subb_u32", "s_sub_u32", "s_sub_i32"])
                .map(|(_, [a, _])| a)
                .or_else(|| {
                    t.valu_bin(v, &["v_sub_co_ci_u32", "v_sub_nc_u32", "v_sub_co_u32"])
                        .map(|[a, _]| a)
                })
                .or_else(|| {
                    t.valu_bin(
                        v,
                        &["v_subrev_co_ci_u32", "v_subrev_nc_u32", "v_subrev_co_u32"],
                    )
                    .map(|[_, b]| b)
                })
            {
                return rec(t, minuend, want, seen);
            }
            Carry::No
        }
        rec(self, hi, want, seen) == Carry::Yes
    }

    fn why_no_base(&self, dynamic: Option<Value>) -> &'static str {
        let Some(v) = dynamic else {
            return "a constant address";
        };
        if self.mask_lost(v, &mut Vec::new()) {
            return MASKED;
        }
        if let Some(reg) = self.inputs[v.0 as usize] {
            return if self.param_of(reg).is_some() {
                "an address from a root parameter that is no address"
            } else {
                "an address in an SGPR that is no root parameter"
            };
        }
        if self
            .leaves(v)
            .any(|l| self.def(l).is_some_and(|i| i.name() == "s_getpc_b64"))
        {
            return "an address relative to the program counter";
        }
        match self.def(v).map(|i| i.name()) {
            Some(n) if n.starts_with("s_load_") => "an address loaded from memory",
            None => "an address from a phi",
            _ => "an address not from a root parameter",
        }
    }

    /// The terms of the additions that compute `v`.
    fn leaves(&self, v: Value) -> Box<dyn Iterator<Item = Value> + '_> {
        match self.sop2(v, &["s_add_u32", "s_add_i32"]) {
            Some((_, terms)) => Box::new(terms.into_iter().flat_map(move |t| match t {
                Term::Value(x) => self.leaves(x),
                Term::Const(_) => Box::new(std::iter::empty()),
            })),
            None => Box::new(std::iter::once(v)),
        }
    }

    fn smem_address(
        &self,
        inst: &Inst,
        base: &Operand,
        offset: i32,
        soffset: &Operand,
    ) -> Result<Address, &'static str> {
        let Operand::Scalar { code, .. } = *base else {
            return Err("an address not in SGPRs");
        };
        let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
        let (Some(lo), Some(hi)) = (word(0), word(1)) else {
            return Err("an address not in SGPRs");
        };
        let mut extra = self.sum(
            self.term(inst, soffset)
                .ok_or("an unknown offset operand")?,
        );
        if extra.base.is_some() {
            return Err("an offset from a root parameter");
        }
        extra.offset = extra.offset.wrapping_add(offset as u32);
        self.address(lo, hi, extra)
    }

    /// The source of the descriptor operand `op` of `inst`, in block `at`.
    fn descriptor(&self, inst: &Inst, op: &Operand, at: usize) -> Result<Source, &'static str> {
        let Operand::Scalar { code, dwords } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let words = (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a descriptor not in SGPRs")?;
        // Only the lanes that reach the access read the descriptor.
        let mut known = Vec::from_iter(self.lanes.map(|l| (l.reach[at], true)));
        self.classify(&words, &mut Vec::new(), &mut known)
    }

    /// The source of a descriptor's words; `joins` are the phi blocks
    /// being resolved, `known` holds in every lane that reads them.
    fn classify(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        if let Some(source) = self.phi(words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.select(words, joins, known)? {
            return Ok(source);
        }
        let read = |w: &Term| match *w {
            Term::Value(v) => self.first_lane_source(v),
            Term::Const(_) => None,
        };
        if let Some(sources) = words.iter().map(read).collect::<Option<Vec<_>>>() {
            // Read from the first active lane, as a waterfall loop does: the
            // dwords each lane holds.
            let lanes: Vec<Term> = sources.into_iter().map(Term::Value).collect();
            return self.per_lane(&lanes, joins, known);
        }
        if words.iter().any(|w| read(w).is_some()) {
            return Err("a descriptor partly read from the first lane");
        }
        if let Some(consts) = words
            .iter()
            .map(|w| match *w {
                Term::Const(c) => Some(c),
                Term::Value(_) => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            return Ok(Source::Constant(consts));
        }
        if let Some(source) = self.loaded(words) {
            return Ok(source);
        }
        if let [lo, hi, Term::Const(w2), Term::Const(w3)] = *words {
            let (mut hi, mut bits) = (hi, 0);
            while let Term::Value(v) = hi {
                if let Some((x, more)) = self.set_bits(v) {
                    (hi, bits) = (x, bits | more);
                } else if let Some(x) = self.va_masked(v) {
                    hi = x;
                } else {
                    break;
                }
            }
            let address = self.address(
                lo,
                hi,
                Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                    multi: false,
                    spread: Some(0),
                },
            )?;
            return Ok(Source::Built {
                address,
                bits,
                words: [w2, w3],
            });
        }
        let loaded = |w: &Term| {
            matches!(*w, Term::Value(v)
                if self.def(v).is_some_and(|i| i.name().starts_with("s_load_")))
        };
        Err(if words.iter().any(loaded) {
            "a descriptor changed after loading"
        } else {
            "a descriptor neither loaded nor built"
        })
    }

    /// Words that phis of one block join, traced per predecessor (words
    /// defined before the join hold on every edge).
    fn phi(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        // Each level multiplies the sources by the predecessor count.
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let args = (0..self.p.blocks[block].preds.len())
            .map(|j| {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => self.origin(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.classify(&edge, joins, known)
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Phi { block, args: args? }))
    }

    /// Words that `s_cselect` (or `s_cmov`) picks by one SCC value, traced
    /// per outcome (other words are the same for both).
    fn select(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(scc) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != scc) {
            return Err("a descriptor chosen by several conditions");
        }
        // Each level doubles the sources; selects count like joins (a
        // marker no block index reaches).
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| self.classify(&side(set), joins, known))
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Select { scc, args: args? }))
    }

    /// For `v` written by `s_cselect_b32`/`_b64` or `s_cmov_b32`/`_b64`:
    /// the SCC it reads and the dword it is when SCC is set, then clear.
    fn picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            return None;
        };
        let k = inst.writes.iter().position(|w| w.1 == v)? as u8;
        let wide = match inst.name() {
            "s_cselect_b32" | "s_cmov_b32" => false,
            "s_cselect_b64" | "s_cmov_b64" => true,
            _ => return None,
        };
        // A dword of a 32- or 64-bit operand: 64-bit inline integers are
        // sign-extended; 64-bit literals and float constants are not
        // followed.
        let dword = |op: &Operand| match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code + k))?)),
            Operand::Const(c) if !wide => inline(c).map(Term::Const),
            Operand::Const(c @ 128..=208) => {
                let low = inline(c)?;
                Some(Term::Const(if k == 0 {
                    low
                } else {
                    ((low as i32) >> 31) as u32
                }))
            }
            Operand::Literal(x) if !wide => Some(Term::Const(x)),
            _ => None,
        };
        let set = dword(src.first()?)?;
        let clear = if inst.name().starts_with("s_cmov") {
            dword(dst.as_ref()?)?
        } else {
            dword(src.get(1)?)?
        };
        Some((inst.read(Reg::Scc)?, set, clear))
    }

    /// `v` as some other dword with constant bits set.
    fn set_bits(&self, v: Value) -> Option<(Term, u32)> {
        if let Some((_, [x, Term::Const(bits)] | [Term::Const(bits), x])) =
            self.sop2(v, &["s_or_b32"])
        {
            return Some((x, bits));
        }
        let inst = self.def(v)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        match (inst.name(), self.term(inst, src.first()?)?) {
            ("s_bitset1_b32", Term::Const(bit)) => {
                Some((self.origin(inst.read(Reg::S(*code))?), 1 << (bit & 31)))
            }
            _ => None,
        }
    }

    /// For a `v_readfirstlane_b32` result: the VGPR value it reads.
    fn first_lane_source(&self, v: Value) -> Option<Value> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readfirstlane_b32" || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let Some(&Operand::Vector { reg, .. }) = f.src.first() else {
            return None;
        };
        inst.read(Reg::V(reg))
    }

    /// The `v_readfirstlane_b32` results the descriptor operand `op` of
    /// `inst` is, when every dword is one.
    fn first_lane_words(&self, inst: &Inst, op: &Operand) -> Option<Vec<Value>> {
        let Operand::Scalar { code, dwords } = *op else {
            return None;
        };
        (0..dwords)
            .map(|k| match self.origin(inst.read(Reg::S(code + k))?) {
                Term::Value(v) => self.first_lane_source(v).map(|_| v),
                Term::Const(_) => None,
            })
            .collect()
    }

    /// `t` through VGPR copies (of VGPRs or of scalars) and copies.
    fn lane_term(&self, mut t: Term) -> Term {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            // A descriptor's dwords, as the lanes that copied them hold them.
            let next = self.copied(v, false);
            if next == t {
                break;
            }
            t = next;
        }
        t
    }

    /// Whether `t` is a value each lane holds its own of (one in VGPRs).
    fn varies(&self, t: Term) -> bool {
        let Term::Value(v) = t else { return false };
        let reg = match (self.inputs[v.0 as usize], self.phis[v.0 as usize]) {
            (Some(reg), _) => Some(reg),
            (None, Some((b, i))) => Some(self.p.blocks[b].phis[i].reg),
            (None, None) => self
                .def(v)
                .and_then(|i| i.writes.iter().find(|w| w.1 == v))
                .map(|w| w.0),
        };
        matches!(reg, Some(Reg::V(_)))
    }

    /// [`Self::lane_term`], where every lane in `known` runs each VGPR copy
    /// it follows.
    fn lane_value(&self, mut t: Term, known: &Known) -> Result<Term, &'static str> {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            let next = self.vstep(v);
            if next == t {
                break;
            }
            if self.varies(t)
                && let Some(site) = self.defs[v.0 as usize]
            {
                self.runs(site, known)?;
            }
            t = next;
        }
        Ok(t)
    }

    /// One copy of [`Self::vorigin`]: what the `v_mov_b32` writing `v`
    /// reads, or `v`'s origin.
    fn vstep(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.raw(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A VALU operand as `inst` reads it: a VGPR's value itself, other
    /// operands' origins.
    fn raw(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(Term::Value(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// Fails unless every lane in `known` that reaches the vector
    /// instruction at `site` runs it: one with its EXEC bit clear there
    /// keeps what the registers held before.
    fn runs(&self, site: Site, known: &Known) -> Result<(), &'static str> {
        let Some(lanes) = self.lanes else {
            return Ok(());
        };
        let Some(&Some(guard)) = lanes.guards.get(site.0).and_then(|g| g.get(site.1)) else {
            return Ok(());
        };
        let mut skips = known.clone();
        skips.extend([(lanes.reach[site.0], true), (guard, false)]);
        if lanes.bools.meet(&skips) {
            return Err("a per-lane descriptor written where some lanes may keep the old one");
        }
        Ok(())
    }

    /// The source of descriptor dwords each lane holds its own of (a
    /// waterfall loop reads them from the first active lane): the same in
    /// every lane, loaded per lane from memory a heap address points to, or
    /// picked per lane between such sources, by a lane mask or by the path
    /// the lane took. `known` holds in every lane that accesses them.
    fn per_lane(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        let words = words
            .iter()
            .map(|&w| self.lane_value(w, known))
            .collect::<Result<Vec<_>, _>>()?;
        let words = &words[..];
        let Some(&lane) = words.iter().find(|&&w| self.varies(w)) else {
            return self.classify(words, joins, known);
        };
        if let Some(source) = self.fetched(words, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_select(words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_phi(words, joins, known)? {
            return Ok(source);
        }
        let Term::Value(v) = lane else { unreachable!() };
        let name = self.def(v).map(|i| i.name());
        Err(match name {
            _ if self.inputs[v.0 as usize].is_some() => {
                "a descriptor read from the first lane of an input VGPR"
            }
            None => "a descriptor read from the first lane of a phi",
            Some(n) if n.starts_with("ds_read") => {
                "a descriptor read from the first lane of an LDS read"
            }
            Some(n) if n.starts_with("buffer_load") => {
                "a descriptor read from the first lane of a buffer load"
            }
            Some(n) if n.starts_with("global_load") || n.starts_with("flat_load") => {
                "a descriptor read from the first lane of a partial or mixed load"
            }
            Some(_) => "a descriptor read from the first lane of computed VGPRs",
        })
    }

    /// Dwords of `global_load_dword*` that line up: `first..` of one load,
    /// then the bytes that follow (an image descriptor takes two
    /// `global_load_dwordx4`).
    fn fetched(&self, words: &[Term], known: &Known) -> Result<Option<Source>, &'static str> {
        let at = |w: &Term| -> Option<(Site, usize)> {
            let Term::Value(v) = *w else { return None };
            let site = self.defs[v.0 as usize]?;
            let inst = self.inst(site);
            if !inst.name().starts_with("global_load_dword") {
                return None;
            }
            let k = inst
                .writes
                .iter()
                .filter(|w| matches!(w.0, Reg::V(_)))
                .position(|w| w.1 == v)?;
            Some((site, k))
        };
        let Some(sites) = words.iter().map(at).collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let address = |site: Site| {
            let inst = self.inst(site);
            let Fields::Flat {
                addr,
                saddr,
                offset,
                ..
            } = &inst.fields
            else {
                return Err("a global load");
            };
            self.flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
        };
        let (load, first) = sites[0];
        let a0 = address(load)?;
        for (k, &(site, dword)) in sites.iter().enumerate() {
            self.runs(site, known)?;
            let a = address(site)?;
            let want = a0.offset.wrapping_add(4 * (first + k) as u32);
            if (a.param, a.dynamic, a.rel) != (a0.param, a0.dynamic, a0.rel)
                || a.offset.wrapping_add(4 * dword as u32) != want
            {
                return Err("a descriptor read from the first lane of loads that do not line up");
            }
        }
        Ok(Some(Source::Fetched {
            load,
            first: first as u8,
        }))
    }

    /// Words that `v_cndmask_b32` picks by one lane mask, traced per outcome
    /// (other words are the same for both).
    fn lane_select(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.lane_picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(mask) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != mask) {
            return Err("a descriptor chosen per lane by several masks");
        }
        for (w, p) in words.iter().zip(&picks) {
            if let (Term::Value(v), Some(_)) = (*w, p)
                && let Some(site) = self.defs[v.0 as usize]
            {
                self.runs(site, known)?;
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        // Each side holds where the lane's bit of the mask is set, or clear.
        let bit = self.lanes.and_then(|l| l.masks.get(&mask).copied());
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| {
                let n = known.len();
                known.extend(bit.map(|b| (b, set)));
                let arg = self.per_lane(&side(set), joins, known);
                known.truncate(n);
                arg
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Lanes { mask, args: args? }))
    }

    /// For `v` written by a plain `v_cndmask_b32`: the low dword of the
    /// lane mask it reads, and the dword it is where the lane's bit is set
    /// (src1), then clear (src0).
    fn lane_picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_cndmask_b32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.clamp
            || f.omod != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let [clear, set, Operand::Scalar { code, .. }] = f.src[..] else {
            return None;
        };
        let mask = inst.read(Reg::S(code))?;
        Some((mask, self.raw(inst, &set)?, self.raw(inst, &clear)?))
    }

    /// Per-lane words that phis of one block join, as an if/else on a
    /// per-lane condition leaves them: traced on each edge into the block
    /// that lanes in `known` can take (other words hold on every edge).
    /// Lanes take their own paths ([`Structured`]), so each gets the value
    /// of the edge it came by; an edge they cannot take is `None` (a flow
    /// block's phi joins an arm's dwords with what lanes that skipped the
    /// arm held before, which only they read).
    fn lane_phi(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        let preds = &self.p.blocks[block].preds;
        if preds.iter().any(|&q| self.rpo[q] >= self.rpo[block]) {
            // shortcut: what a lane holds after some passes is not traced.
            return Err("a descriptor read from the first lane of a phi at a loop header");
        }
        let Some(lanes) = self.lanes else {
            return Err("a descriptor read from the first lane of a phi, without the lanes' paths");
        };
        // The other words must be there before the join: defined in a
        // block that dominates it, which (both dominating the access) is
        // one the join does not reach.
        for w in words {
            if let Term::Value(v) = *w
                && phi_of(w).is_none()
                && let Some((d, _)) = self.defs[v.0 as usize]
                && (d == block || self.reaches(block, d))
            {
                return Err("a descriptor partly joined by control flow");
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let mut args = Vec::new();
        for (j, &q) in preds.iter().enumerate() {
            let n = known.len();
            known.push((lanes.reach[q], true));
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = lanes.exits[q]
                && taken != not_taken
            {
                known.push((when, taken == block));
            }
            let arg = if lanes.bools.meet(known) {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => Term::Value(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.per_lane(&edge, joins, known).map(Some)
            } else {
                Ok(None)
            };
            known.truncate(n);
            match arg {
                Ok(a) => args.push(a),
                Err(e) => {
                    joins.pop();
                    return Err(e);
                }
            }
        }
        joins.pop();
        Ok(Some(Source::LanePhi { block, args }))
    }

    /// Whether a path leads from block `from` to block `to` (to `from`
    /// itself only through a back edge).
    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut seen = vec![false; self.p.blocks.len()];
        let mut stack = self.p.blocks[from].exit.successors();
        while let Some(x) = stack.pop() {
            if x == to {
                return true;
            }
            if std::mem::replace(&mut seen[x], true) {
                continue;
            }
            stack.extend(self.p.blocks[x].exit.successors());
        }
        false
    }

    /// Prove a single-block waterfall at the descriptor access. In each
    /// iteration the access serves exactly the matching lanes, and the
    /// back edge keeps exactly the other lanes active. Otherwise, why not.
    ///
    // shortcut: only the accesses are proven. Any other per-lane work in the
    // loop is taken to run once for each lane, as the access does; a body
    // whose effect adds up over the passes a lane waits through (a counter
    // before the mask is narrowed) would differ. Compilers put nothing but
    // the reads, the compares and the access in these loops.
    fn waterfall(
        &self,
        b: usize,
        accesses: &[usize],
        reads: &[Value],
    ) -> Result<Waterfall, &'static str> {
        const LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
        const SPLIT: &str = "a waterfall loop nested in another or split over blocks";
        const SELECT: &str = "a waterfall loop whose compares do not select the access's lanes";
        const RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
        let access = accesses[0];
        let block = &self.p.blocks[b];
        let lanes = self.lanes.ok_or(LOOP)?;
        let exit = block.insts.last().ok_or(LOOP)?;
        if !matches!(block.exit, Exit::Branch { taken, not_taken } if taken == b && not_taken != b)
            || exit.name() != "s_cbranch_execnz"
        {
            return Err(LOOP);
        }
        let mask = |low: Option<Value>, high: Option<Value>| {
            let high = if self.p.wave64 { Some(high?) } else { None };
            lanes.mask_bits(low?, high)
        };
        let exec = |inst: &Inst| mask(inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1)));
        // The reads belong to this loop's own block: a loop around this one
        // (one per operand, as LLVM nests them for a per-lane image and
        // sampler) is not followed.
        if reads
            .iter()
            .any(|v| self.defs[v.0 as usize].is_none_or(|at| at.0 != b))
        {
            return Err(SPLIT);
        }
        let selected = exec(&block.insts[access]).ok_or(SELECT)?;
        if accesses
            .iter()
            .any(|&i| exec(&block.insts[i]) != Some(selected))
        {
            return Err(SELECT);
        }
        let retry = exec(exit).ok_or(RETRY)?;
        let first = self.defs[reads[0].0 as usize].ok_or(LOOP)?;
        let remaining = exec(self.inst(first)).ok_or(SELECT)?;
        if reads.iter().any(|v| {
            self.defs[v.0 as usize]
                .is_none_or(|at| at.1 >= access || exec(self.inst(at)) != Some(remaining))
        }) {
            return Err(SELECT);
        }
        let reach = (lanes.reach[b], true);
        let meet = |conditions: &[(Bool, bool)]| lanes.bools.meet(conditions);
        let mut out = Waterfall::default();
        let mut required = vec![(remaining, true)];
        for (i, inst) in block.insts[..access].iter().enumerate() {
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            let Some(ty) = inst
                .name()
                .strip_prefix("v_cmp_eq_")
                .or(inst.name().strip_prefix("v_cmpx_eq_"))
            else {
                continue;
            };
            let dwords = match ty {
                "u32" | "i32" => 1,
                "u64" | "i64" => 2,
                _ => continue,
            };
            if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
                continue;
            }
            // The first lane's dwords on one side, the lane's own on the
            // other.
            let pair = |first: &Operand, own: &Operand| -> Option<Vec<Value>> {
                let (&Operand::Scalar { code, .. }, &Operand::Vector { reg, .. }) = (first, own)
                else {
                    return None;
                };
                (0..dwords)
                    .map(|k| {
                        let Term::Value(x) = self.origin(inst.read(Reg::S(code + k))?) else {
                            return None;
                        };
                        let lane = inst.read(Reg::V(reg + k))?;
                        let (_, read) = self.defs[x.0 as usize]?;
                        let same = reads.contains(&x)
                            && read < i
                            && self.lane_term(Term::Value(self.first_lane_source(x)?))
                                == self.lane_term(Term::Value(lane));
                        same.then_some(x)
                    })
                    .collect()
            };
            let (Some(a), Some(c)) = (f.src.first(), f.src.get(1)) else {
                continue;
            };
            if let Some(xs) = pair(a, c).or_else(|| pair(c, a)) {
                let code = if inst.name().starts_with("v_cmpx_") {
                    EXEC
                } else if let Some(Operand::Scalar { code, .. }) = f.dst {
                    code
                } else {
                    continue;
                };
                let Some(matched) = mask(inst.write(Reg::S(code)), inst.write(Reg::S(code + 1))) else {
                    continue;
                };
                if meet(&[reach, (selected, true), (matched, false)]) {
                    continue;
                }
                required.push((matched, true));
                out.compares.push((b, i));
                for x in xs {
                    if !out.reads.contains(&x) {
                        out.reads.push(x);
                    }
                }
            }
        }
        if reads.iter().any(|v| !out.reads.contains(v))
            || meet(&[reach, (selected, true), (remaining, false)])
        {
            return Err(SELECT);
        }
        // The access must take every match.
        required.extend([reach, (selected, false)]);
        if meet(&required) {
            return Err(SELECT);
        }
        // The retry must keep exactly the lanes that were eligible but did
        // not perform the access.
        if meet(&[reach, (retry, true), (remaining, false)])
            || meet(&[reach, (retry, true), (selected, true)])
            || meet(&[reach, (remaining, true), (selected, false), (retry, false)])
        {
            return Err(RETRY);
        }
        Ok(out)
    }

    /// Consecutive dwords of one scalar load.
    fn loaded(&self, words: &[Term]) -> Option<Source> {
        let Term::Value(v0) = words[0] else {
            return None;
        };
        let load = self.defs[v0.0 as usize]?;
        let inst = self.inst(load);
        if !inst.name().starts_with("s_load_") {
            return None;
        }
        let first = inst.writes.iter().position(|w| w.1 == v0)?;
        let same = words.iter().enumerate().all(|(k, w)| {
            inst.writes
                .get(first + k)
                .is_some_and(|x| *w == Term::Value(x.1))
        });
        same.then_some(Source::Loaded {
            load,
            first: first as u8,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Moved {
    Const(u32),
    Value(Value),
}

impl Tracer<'_> {
    /// For `v_readlane_b32` of a constant lane: the SGPR spilled to that
    /// lane, found by following the VGPR's `v_writelane_b32` writes.
    fn restored(&self, inst: &Inst) -> Option<Moved> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readlane_b32" {
            return None;
        }
        let [Operand::Vector { reg, .. }, lane] = v.src[..] else {
            return None;
        };
        self.spilled(inst.read(Reg::V(reg))?, lane_of(lane)?, &mut Vec::new())
            .ok()
            .flatten()
    }

    /// What lane `lane` of VGPR value `v` holds: `Ok(None)` for a phi in
    /// `seen` (already being resolved), `Err` if not a spilled SGPR.
    fn spilled(&self, v: Value, lane: u32, seen: &mut Vec<Value>) -> Result<Option<Moved>, ()> {
        if let Some((b, i)) = self.phis[v.0 as usize] {
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &a in &self.p.blocks[b].phis[i].args {
                if let Some(m) = self.spilled(a, lane, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            return Ok(found);
        }
        let inst = self.def(v).ok_or(())?;
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        if inst.name() != "v_writelane_b32" {
            return Err(());
        }
        if lane_of(*at).ok_or(())? != lane {
            return self.spilled(inst.read(Reg::V(reg)).ok_or(())?, lane, seen);
        }
        match *value {
            Operand::Scalar { code, .. } if code != NULL => {
                inst.read(Reg::S(code)).map(Moved::Value)
            }
            Operand::Literal(x) => Some(Moved::Const(x)),
            Operand::Const(c) => inline(c).map(Moved::Const),
            _ => None,
        }
        .map(Some)
        .ok_or(())
    }
}

/// Whether `inst` reads VGPR value `x` as an operand, not only as the old
/// value of a register it writes (which the lanes outside EXEC keep).
fn operand(inst: &Inst, x: Value) -> bool {
    let ops: Vec<&Operand> = match &inst.fields {
        Fields::Valu(f)
            if f.sdwa.is_none()
                && f.dpp.is_none()
                && !inst.name().starts_with("v_movrel")
                && !inst.name().starts_with("v_swap")
                && inst.name() != "v_writelane_b32" =>
        {
            f.src.iter().collect()
        }
        Fields::Flat {
            addr, data, saddr, ..
        } => [addr, data, saddr].into_iter().flatten().collect(),
        // shortcut: any read by another instruction counts as one.
        _ => return true,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Vector { reg, dwords } => {
            (0..dwords).any(|k| inst.read(Reg::V(reg + k)) == Some(x))
        }
        _ => false,
    })
}

/// A lane select given as a constant.
fn lane_of(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(c) => inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// What dword `k` of a move's result copies.
fn copy_source(inst: &Inst, k: usize) -> Option<Moved> {
    let Fields::Sop { src, imm, .. } = &inst.fields else {
        return None;
    };
    if inst.name() == "s_movk_i32" {
        return Some(Moved::Const(i32::from((*imm)? as i16) as u32));
    }
    let src = src.first()?;
    match (inst.name(), *src) {
        ("s_mov_b32", Operand::Spill(code)) => inst.read(Reg::Spill(code)).map(Moved::Value),
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { code, .. }) if code != NULL => {
            inst.read(Reg::S(code + k as u8)).map(Moved::Value)
        }
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { .. }) => Some(Moved::Const(0)),
        ("s_mov_b32", Operand::Literal(x)) => Some(Moved::Const(x)),
        ("s_mov_b32", Operand::Const(c)) => inline(c).map(Moved::Const),
        // shortcut: only the one-operand ways the compiler makes a constant
        // that has no inline form; fold more if descriptors need it.
        ("s_brev_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(x.reverse_bits())),
        ("s_not_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(!x)),
        // 64-bit inline integers are sign-extended.
        ("s_mov_b64", Operand::Const(c @ 128..=208)) => {
            let low = inline(c)?;
            Some(Moved::Const(if k == 0 {
                low
            } else {
                ((low as i32) >> 31) as u32
            }))
        }
        _ => None,
    }
}

/// The 32-bit value of an inline constant operand.
pub(crate) fn inline(code: u8) -> Option<u32> {
    let float = |f: f32| Some(f.to_bits());
    match code {
        128..=192 => Some(u32::from(code) - 128),
        193..=208 => Some((192 - i32::from(code)) as u32),
        240 => float(0.5),
        241 => float(-0.5),
        242 => float(1.0),
        243 => float(-1.0),
        244 => float(2.0),
        245 => float(-2.0),
        246 => float(4.0),
        247 => float(-4.0),
        248 => Some(0x3E22_F983),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::native::RootType::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn track_words(words: &[u32], layout: &[RootParam], stage: HwStage) -> Resources {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        track(&p, layout, stage, lanes(&p).as_ref()).unwrap()
    }

    /// `p` as one lane runs it, if it structures.
    fn lanes(p: &Program) -> Option<Structured> {
        crate::structure::structure(p).ok()
    }

    fn param(root: RootType, user_data: u8, sgprs: u32) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        }
    }

    fn at(param: usize, offset: u32) -> Address {
        Address {
            param,
            offset,
            dynamic: None,
            rel: None,
        }
    }

    fn try_track_words(
        words: &[u32],
        layout: &[RootParam],
        stage: HwStage,
    ) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        track(&p, layout, stage, lanes(&p).as_ref())
    }

    const fn s_mov(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
    }
    const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
        0x8000_0000
            | (op as u32 & 0x7F) << 23
            | (sdst as u32 & 0x7F) << 16
            | (src1 as u32) << 8
            | src0 as u32
    }
    const fn s_addk_i32(sdst: u8, imm: i16) -> u32 {
        0xB000_0000 | 0x0F << 23 | (sdst as u32 & 0x7F) << 16 | imm as u16 as u32
    }
    const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
        0xBF00_0000 | (op as u32 & 0x7F) << 16 | (src1 as u32) << 8 | src0 as u32
    }
    const fn sopp(op: u8, simm16: i16) -> u32 {
        0xBF80_0000 | (op as u32 & 0x7F) << 16 | simm16 as u16 as u32
    }
    const fn s_load_x4(dst: u8, base: u8, soffset: u8, offset: i32) -> [u32; 2] {
        [
            0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
            (soffset as u32) << 25 | (offset as u32 & 0x1F_FFFF),
        ]
    }
    const fn s_buf_load(dst: u8, base: u8) -> [u32; 2] {
        [
            0x3D << 26 | 0x08 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
            125 << 25,
        ]
    }
    const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
        0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
    }
    const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }
    const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
        ]
    }
    const fn vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | (sdst as u32 & 0x7F) << 8 | vdst as u32,
            (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
        ]
    }
    const fn global_load(vdst: u8, vaddr: u8, saddr: u8, offset: i16) -> [u32; 2] {
        [
            0x37 << 26 | 0x0C << 18 | 2 << 14 | (offset as u32 & 0xFFF),
            (vdst as u32) << 24 | (saddr as u32 & 0x7F) << 16 | vaddr as u32,
        ]
    }
    const fn sgpr(n: u8) -> u16 {
        n as u16
    }
    const fn vgpr(n: u8) -> u16 {
        256 + n as u16
    }

    #[test]
    fn table_descriptors_are_loaded_from_their_range() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE89_0380, // s_mov_b32 s9, 0
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF408_0204,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
                0xF40C_0001,
                0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
                0xF080_0F08,
                0x0040_0002, // image_sample v[0:3], v[2:3], s[0:7], s[8:11]
                S_ENDPGM,
            ],
            &[param(Table, 8, 1), param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0)), ((0, 4), at(1, 0))]);
        let roles: Vec<_> = r.descriptors.iter().map(|d| (d.role, &d.source)).collect();
        assert_eq!(
            roles,
            [
                (
                    Role::Resource,
                    &Source::Loaded {
                        load: (0, 4),
                        first: 0
                    }
                ),
                (
                    Role::Sampler,
                    &Source::Loaded {
                        load: (0, 3),
                        first: 0
                    }
                ),
            ]
        );
    }

    #[test]
    fn root_descriptors_are_built_in_code() {
        let r = track_words(
            &[
                0x8801_FF0B,
                0x0010_0000, // s_or_b32 s1, s11, 0x100000
                0xBE80_030A, // s_mov_b32 s0, s10
                0xBE82_03C1, // s_mov_b32 s2, -1
                0xBE83_03FF,
                0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
                0xF430_0000,
                0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
                S_ENDPGM,
            ],
            &[param(Descriptor, 10, 2)],
            HwStage::Vs,
        );
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        assert_eq!(
            r.params.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
            [(0, 0), (0, 1)]
        );
    }

    #[test]
    fn spilled_root_descriptor_uses_its_logical_high_half() {
        let code: Vec<u8> = [
            0xF404_0280u32, // s_load_dwordx2 s[10:11], s[0:1], 0
            0xFA00_0000,
            0xBE85_0311, // s_mov_b32 s5, physical s17
            0x8801_FF0B,
            0x0010_0000, // s_or_b32 s1, s11, 0x100000
            0xBE80_030A, // s_mov_b32 s0, s10
            0xBE82_03C1, // s_mov_b32 s2, -1
            0xBE83_03FF,
            0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
            0xF430_0000,
            0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = crate::ir::build_with(&code, false, 256, true).unwrap();
        assert!(p.inputs.iter().any(|&(reg, _)| reg == Reg::S(17)));
        let r = track(&p, &[param(Descriptor, 16, 2)], HwStage::Cs, lanes(&p).as_ref()).unwrap();
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        let input = |reg| p.inputs.iter().find(|&&(r, _)| r == reg).unwrap().1;
        assert_eq!(
            r.params,
            [(input(Reg::Spill(16)), 0, 0), (input(Reg::Spill(17)), 0, 1)]
        );
    }

    #[test]
    fn heap_indices_from_root_constants_in_a_merged_stage() {
        // User data starts at s8: the constant (index 6) is s14, the heap
        // (index 10) s18.
        let r = track_words(
            &[
                0x8F08_850E, // s_lshl_b32 s8, s14, 5
                0xBE93_0380, // s_mov_b32 s19, 0
                0xF408_0209,
                0x1000_0000, // s_load_dwordx4 s[8:11], s[18:19], s8
                0xE000_2000,
                0x8002_0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
                S_ENDPGM,
            ],
            &[param(Constants, 6, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ms,
        );
        let index = r.params.iter().find(|p| p.1 == 0).unwrap().0;
        let shift = r.scalar[0].1.dynamic.unwrap();
        assert_eq!(r.scalar[0].1.param, 1);
        // The computed term is the shifted constant, not the constant itself.
        assert_ne!(shift, index);
        assert_eq!(
            r.descriptors[0].source,
            Source::Loaded {
                load: (0, 2),
                first: 0
            }
        );
    }

    #[test]
    fn spilled_sgprs_are_followed_through_vgpr_lanes() {
        let r = track_words(
            &[
                0xD761_0000,
                0x0001_0609, // v_writelane_b32 v0, s9, 3
                0xD760_0002,
                0x0001_0700, // v_readlane_b32 s2, v0, 3
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF40C_0001,
                0xFA00_0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0x20))]);
    }

    #[test]
    fn descriptors_joined_by_control_flow_are_traced_per_edge() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE83_0380, // s_mov_b32 s3, 0
                0xBF06_8001, // s_cmp_eq_u32 s1, 0
                0xBF85_0003, // s_cbranch_scc1 -> 0x1c
                0xF408_0001,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
                0xBF82_0002, // s_branch -> 0x24
                0xF408_0001,
                0xFA00_0010, // 0x1c: s_load_dwordx4 s[0:3], s[2:3], 0x10
                0xF420_0100,
                0xFA00_0000, // 0x24: s_buffer_load_dword s4, s[0:3], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Phi { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        let offsets: Vec<_> = args
            .iter()
            .map(|a| match a {
                Source::Loaded { load, .. } => {
                    r.scalar.iter().find(|s| s.0 == *load).unwrap().1.offset
                }
                a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets.len(), 2);
        assert!(
            offsets.contains(&0) && offsets.contains(&0x10),
            "{offsets:?}"
        );
    }

    #[test]
    fn descriptors_selected_by_scc_are_traced_per_outcome() {
        // Two buffer descriptors loaded from the table at s9 and one
        // picked with s_cselect_b64 per half, as the compiler makes a
        // conditional choice of resources without a branch.
        let r = track_words(
            &[
                0xBE8C_0309, // s_mov_b32 s12, s9
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xF408_0006,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
                0xF408_0106,
                0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
                0xBF06_800E, // s_cmp_eq_u32 s14, 0
                0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
                0x858A_0602, // s_cselect_b64 s[10:11], s[2:3], s[6:7]
                0xF420_0404,
                0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Select { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        // SCC set picks the first operand.
        assert_eq!(
            args[..],
            [
                Source::Loaded {
                    load: (0, 2),
                    first: 0
                },
                Source::Loaded {
                    load: (0, 3),
                    first: 0
                },
            ]
        );
        assert_eq!(r.scalar, [((0, 2), at(0, 0)), ((0, 3), at(0, 0x10))]);
    }

    #[test]
    fn a_selection_that_mixes_two_descriptors_fails() {
        // The high half picked from the first load whatever SCC is: when
        // clear, the descriptor is half of one load and half of the other.
        let code: Vec<u8> = [
            0xBE8C_0309, // s_mov_b32 s12, s9
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xF408_0006,
            0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
            0xF408_0106,
            0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
            0xBF06_800E, // s_cmp_eq_u32 s14, 0
            0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
            0x858A_0202, // s_cselect_b64 s[10:11], s[2:3], s[2:3]
            0xF420_0404,
            0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 36,
                what: "a descriptor changed after loading"
            }
        );
    }

    #[test]
    fn untraceable_addresses_fail_with_a_reason() {
        let code: Vec<u8> = [0xF408_0001, 0xFA00_0000, S_ENDPGM] // s_load_dwordx4 s[0:3], s[2:3]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 0,
                what: "an address in an SGPR that is no root parameter"
            }
        );
    }

    /// A waterfall loop over an image descriptor each lane loads from the
    /// table at s0 (wave32, assembled with llvm-mc for gfx1030): lane i
    /// reads entry i with two `global_load_dwordx4`, the loop reads the
    /// first active lane's descriptor into s[12:19], samples with the lanes
    /// whose descriptor it is, and repeats for the others. The sampler comes
    /// from the table at s1; the result goes to LDS so the sample is needed.
    pub(crate) const WATERFALL_FETCHED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// [`WATERFALL_FETCHED_WAVE32`] in wave64: 64-bit lane masks.
    pub(crate) const WATERFALL_FETCHED_WAVE64: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0016,
        0x0002_080C, // v_cmp_eq_u64_e64 s[22:23], s[12:13], v[4:5]
        0xD4E2_0018,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s[24:25], s[14:15], v[6:7]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1010, // v_cmp_eq_u64_e64 s[24:25], s[16:17], v[8:9]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1412, // v_cmp_eq_u64_e64 s[24:25], s[18:19], v[10:11]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xBE96_2416, // s_and_saveexec_b64 s[22:23], s[22:23]
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x89FE_167E, // s_xor_b64 exec, exec, s[22:23]
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The same waterfall over one of two image descriptors loaded from the
    /// table at s0, picked per lane by `v_cndmask_b32` (lanes below 16 take
    /// the first).
    pub(crate) const WATERFALL_PICKED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x7c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x7c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The waterfall loop of [`WATERFALL_PICKED_WAVE32`] over one of two
    /// image descriptors loaded from the table at s0, chosen by control flow
    /// as the compiler lowers an if/else on a per-lane condition: lanes below
    /// 16 move the first into v[4:11], the others the second, each arm
    /// skipped when no lane takes it. The sample goes to the buffer at dword
    /// 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_JOINED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x70
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBE94_3D14, // 0x70: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x9c
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0x887E_147E, // 0x9c: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0xa4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xa4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// [`WATERFALL_JOINED_WAVE32`] with other sources on each path: lanes
    /// below 16 load table entry (lane & 1) with two `global_load_dwordx4`,
    /// the others pick per lane with `v_cndmask_b32` (lanes below 24 the
    /// first entry, the rest the second).
    pub(crate) const WATERFALL_JOINED_MIXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x3602_0081, // v_and_b32 v1, 1, v0
        0x3402_0285, // v_lshlrev_b32 v1, 5, v1
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x68
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xBE94_3D14, // 0x68: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0011, // s_cbranch_execz -> 0xb8
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0x887E_147E, // 0xb8: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E18_0504, // 0xc4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xc4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop whose descriptor changes on each pass: v[4:11] holds
    /// the first table entry on entry and the second after every pass, so
    /// the first lane's dwords are phis of the loop header.
    pub(crate) const WATERFALL_LOOPED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0x64: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0xBF89_FFDE, // s_cbranch_execnz -> 0x64
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop over one of two buffer descriptors (four dwords)
    /// loaded from the table at s0, chosen by nested if/else: lanes below 16
    /// move the first into v[4:7], lanes from 16 to 23 the second, and the
    /// others keep what v[4:7] held. Only lanes below 24 (the same mask)
    /// run the loop, which loads dword `lane` of the chosen buffer, and
    /// store it to the buffer at dword 16 of the table.
    pub(crate) const WATERFALL_JOINED_BUFFER_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0602,
        0xFA00_0000, // s_load_dwordx4 s[24:27], s[4:5], 0x0
        0xF408_0802,
        0xFA00_0020, // s_load_dwordx4 s[32:35], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xD4C4_001E,
        0x0002_0098, // v_cmp_gt_u32_e64 s30, 24, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x50
        0x7E08_0218, // 0x40: v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0xBE94_3D14, // 0x50: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0007, // s_cbranch_execz -> 0x78
        0xBE9F_3C1E, // s_and_saveexec_b32 s31, s30
        0xBF88_0004, // s_cbranch_execz -> 0x74
        0x7E08_0220, // 0x64: v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x887E_1F7E, // 0x74: s_or_b32 exec_lo, exec_lo, s31
        0x887E_147E, // 0x78: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_3C1E, // s_and_saveexec_b32 s23, s30
        0xBF88_0013, // s_cbranch_execz -> 0xd0
        0xBE9D_037E, // s_mov_b32 s29, exec_lo
        0x7E18_0504, // 0x88: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xE030_1000,
        0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFF1, // s_cbranch_execnz -> 0x88
        0xBEFE_031D, // s_mov_b32 exec_lo, s29
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0x887E_177E, // 0xd0: s_or_b32 exec_lo, exec_lo, s23
        0xBF81_0000, // s_endpgm
    ];

    fn waterfall(words: &[u32], wave64: bool) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, wave64, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        track(&p, &layout, HwStage::Cs, lanes(&p).as_ref())
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_per_lane_loads_are_traced_to_them() {
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
        ] {
            let r = waterfall(words, wave64).unwrap();
            let image = r
                .descriptors
                .iter()
                .find(|d| d.role == Role::Resource)
                .unwrap();
            let Source::Fetched { load, first: 0 } = image.source else {
                panic!("{:?}", image.source)
            };
            // The first of the two loads: entry v0 * 32 of the table at s0.
            let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
            assert_eq!((a.param, a.offset), (0, 0));
            assert!(a.dynamic.is_some());
            // The sampler is the same for every lane.
            let sampler = r.descriptors.iter().find(|d| d.role == Role::Sampler);
            assert!(matches!(
                sampler.unwrap().source,
                Source::Loaded { first: 0, .. }
            ));
            // Four 64-bit compares of the first lane's dwords with the lane's
            // own; the eight reads become the lane's own dwords.
            assert_eq!(r.waterfall.compares.len(), 4);
            assert_eq!(r.waterfall.reads.len(), 8);
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_a_per_lane_choice_are_traced_per_lane() {
        let r = waterfall(WATERFALL_PICKED_WAVE32, false).unwrap();
        let image = &r.descriptors[0];
        let Source::Lanes { args, .. } = &image.source else {
            panic!("{:?}", image.source)
        };
        // Lanes with their bit set take the first load (offset 0), the
        // others the second (0x20).
        let offsets: Vec<u32> = args
            .iter()
            .map(|a| match *a {
                Source::Loaded { load, first: 0 } => {
                    r.scalar.iter().find(|s| s.0 == load).unwrap().1.offset
                }
                ref a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_first_lane_read_outside_a_loop_is_no_waterfall() {
        // The waterfall without its back edge: the s_cbranch_execnz is an
        // s_nop, so lanes the first one does not match never get a pass.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let back = words.iter().position(|&w| w == 0xBF89_FFE6).unwrap();
        words[back] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane outside a waterfall loop",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// Why resource tracking refuses `words`.
    fn refusal(words: &[u32], wave64: bool) -> &'static str {
        match waterfall(words, wave64) {
            Err(ShaderError::Unsupported { what, .. }) => what,
            other => panic!("not refused: {other:?}"),
        }
    }

    const NO_LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
    const NOT_SELECTED: &str = "a waterfall loop whose compares do not select the access's lanes";
    const BAD_RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
    const SPLIT: &str = "a waterfall loop nested in another or split over blocks";

    /// The index of `word` in `words`.
    fn word_at(words: &[u32], word: u32) -> usize {
        words.iter().position(|&w| w == word).unwrap()
    }

    #[test]
    fn a_first_lane_read_without_its_compare_is_no_waterfall() {
        // The loop without the compare of dwords 0 and 1: nothing makes
        // the first lane's dwords the lane's own there.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let cmp = word_at(&words, 0x0002_080C);
        words[cmp - 1..=cmp].copy_from_slice(&[0xBF80_0000, 0xBF80_0000]);
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_compare_that_does_not_mask_the_access_is_no_waterfall() {
        // The compares are all there, but their result never narrows EXEC:
        // every lane samples with the first lane's descriptor (reported by
        // ifBars on upstream PR #566).
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let save = word_at(&words, 0xBE95_3C15);
        words[save] = 0xBE95_037E; // s_mov_b32 s21, exec_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn an_access_under_a_further_mask_is_no_waterfall() {
        // The access runs for the matching lanes that VCC also holds, yet
        // the retry removes every matching lane.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let nop = word_at(&words, 0xBE95_3C15) + 1;
        assert_eq!(words[nop], 0xBF80_0000);
        words[nop] = 0x877E_6A7E; // s_and_b32 exec_lo, exec_lo, vcc_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_loop_that_drops_unmatched_lanes_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0xBEFE_0380; // s_mov_b32 exec_lo, 0
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_retry_that_keeps_other_lanes_than_those_waiting_is_no_waterfall() {
        // The retry keeps the lanes of VCC that did not just sample, not
        // the waiting ones: lanes 16 and up never sample on the console.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0x8A7E_7E6A; // s_andn2_b32 exec_lo, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_wave64_retry_of_one_half_is_no_waterfall() {
        // Only the low half of EXEC loses the lanes just served.
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let retry = word_at(&words, 0x89FE_167E);
        words[retry] = 0x897E_167E; // s_xor_b32 exec_lo, exec_lo, s22
        assert_eq!(refusal(&words, true), BAD_RETRY);
    }

    #[test]
    fn a_descriptor_used_after_exec_is_restored_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let sample = word_at(&words, 0xF09C_0F08);
        let restore = word_at(&words, 0xBEFE_0317);
        let image = [words[sample], words[sample + 1]];
        words[sample..sample + 2].fill(0xBF80_0000); // Keep the branch offsets.
        words.splice(restore + 1..restore + 1, image);
        assert_eq!(refusal(&words, false), NO_LOOP);
    }

    #[test]
    fn a_wave64_access_with_exec_hi_overwritten_is_no_waterfall() {
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let save = word_at(&words, 0xBE96_2416);
        words[save + 1] = 0xBEFF_03C1; // s_mov_b32 exec_hi, -1
        assert_eq!(refusal(&words, true), NOT_SELECTED);
    }

    /// The waterfall with `v_cmpx_eq` narrowing EXEC itself, and the retry
    /// as `todo & !EXEC` (`folded`: in one `s_andn2_wrexec`), as LLVM's
    /// AMDGPU backend writes it for GFX10 and later.
    pub(crate) fn cmpx_waterfall(wave64: bool, folded: bool) -> Vec<u32> {
        let (base, todo, restore): (&[u32], u32, u32) = if wave64 {
            (WATERFALL_FETCHED_WAVE64, 0xBE94_047E, 0xBEFE_0414)
        } else {
            (WATERFALL_JOINED_WAVE32, 0xBE97_037E, 0xBEFE_0317)
        };
        let reads = word_at(base, 0x7E18_0504);
        let sample = word_at(base, 0xF09C_0F08);
        let mut w = base[..=word_at(base, todo)].to_vec();
        // s_mov_b32 s21, exec_lo / s_mov_b64 s[22:23], exec
        w.push(if wave64 { 0xBE96_047E } else { 0xBE95_037E });
        w.extend(&base[reads..reads + 8]); // v_readfirstlane_b32 s12..s19
        w.extend([
            0x7DE4_080C, // v_cmpx_eq_u64 s[12:13], v[4:5]
            0x7DE4_0C0E, // v_cmpx_eq_u64 s[14:15], v[6:7]
            0x7DE4_1010, // v_cmpx_eq_u64 s[16:17], v[8:9]
            0x7DE4_1412, // v_cmpx_eq_u64 s[18:19], v[10:11]
        ]);
        w.extend(&base[sample..sample + 2]); // image_sample_lz
        match (wave64, folded) {
            // s_andn2_wrexec_b32 s21, s21; s_cbranch_execnz
            (false, true) => w.extend([0xBE95_4715, 0xBF89_FFF0]),
            // s_andn2_b32 s21, s21, exec_lo; s_mov_b32 exec_lo, s21
            (false, false) => w.extend([0x8A15_7E15, 0xBEFE_0315, 0xBF89_FFEF]),
            // s_andn2_wrexec_b64 s[22:23], s[22:23]
            (true, true) => w.extend([0xBE96_3A16, 0xBF89_FFF0]),
            // s_andn2_b64 s[22:23], s[22:23], exec; s_mov_b64 exec, s[22:23]
            (true, false) => w.extend([0x8A96_7E16, 0xBEFE_0416, 0xBF89_FFEF]),
        }
        w.extend(&base[word_at(base, restore)..]);
        w
    }

    #[test]
    fn a_waterfall_narrowed_by_v_cmpx_is_one() {
        for wave64 in [false, true] {
            for folded in [false, true] {
                let r = waterfall(&cmpx_waterfall(wave64, folded), wave64)
                    .unwrap_or_else(|e| panic!("wave64 {wave64} folded {folded}: {e:?}"));
                assert_eq!(r.waterfall.compares.len(), 4, "{wave64} {folded}");
                assert_eq!(r.waterfall.reads.len(), 8, "{wave64} {folded}");
            }
        }
    }

    #[test]
    fn a_v_cmpx_waterfall_whose_retry_drops_lanes_is_none() {
        let mut words = cmpx_waterfall(false, false);
        let retry = word_at(&words, 0x8A15_7E15);
        words[retry] = 0x8A15_7E6A; // s_andn2_b32 s21, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_waterfall_nested_in_another_is_refused_by_name() {
        // An image whose resource and sampler are both per lane, with one
        // loop per operand, the one for the sampler inside the one for the
        // resource (what the SelectionDAG legalisation in LLVM would give).
        // The console gives each lane its own pair. Refused for now: the
        // access of the inner loop reads the first-lane dwords of the outer
        // loop, which another block holds.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x3602_0081, // v_and_b32 v1, 1, v0
            0x3402_0285, // v_lshlrev_b32 v1, 5, v1
            0x3622_0080, // v_and_b32 v17, 0, v0
            0x3422_2284, // v_lshlrev_b32 v17, 4, v17
            0xDC38_8000,
            0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
            0xDC38_8010,
            0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
            0xDC38_8000,
            0x1406_0011, // global_load_dwordx4 v[20:23], v17, s[6:7]
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xBE94_037E, // s_mov_b32 s20, exec_lo
            // L1, the loop over the resource:
            0x7E18_0504, // v_readfirstlane_b32 s12, v4
            0x7E1A_0505,
            0x7E1C_0506,
            0x7E1E_0507,
            0x7E20_0508,
            0x7E22_0509,
            0x7E24_050A,
            0x7E26_050B, // v_readfirstlane_b32 s19, v11
            0xD4E2_0015,
            0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
            0xD4E2_0016,
            0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xBE95_3C15, // s_and_saveexec_b32 s21, s21
            0xBE9A_037E, // s_mov_b32 s26, exec_lo
            // L2, the loop over the sampler:
            0x7E10_0514, // v_readfirstlane_b32 s8, v20
            0x7E12_0515, // v_readfirstlane_b32 s9, v21
            0x7E14_0516, // v_readfirstlane_b32 s10, v22
            0x7E16_0517, // v_readfirstlane_b32 s11, v23
            0xD4E2_0018,
            0x0002_2808, // v_cmp_eq_u64_e64 s24, s[8:9], v[20:21]
            0xD4E2_0019,
            0x0002_2C0A, // v_cmp_eq_u64_e64 s25, s[10:11], v[22:23]
            0x8718_1918, // s_and_b32 s24, s24, s25
            0xBE98_3C18, // s_and_saveexec_b32 s24, s24
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            0x897E_187E, // s_xor_b32 exec_lo, exec_lo, s24
            0xBF89_FFF2, // s_cbranch_execnz L2
            0xBEFE_031A, // s_mov_b32 exec_lo, s26
            0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
            0xBF89_FFDA, // s_cbranch_execnz L1
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        assert_eq!(refusal(&words, false), SPLIT);
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_lds_fail_naming_it() {
        // The waterfall over descriptors each lane read from LDS:
        // ds_read_b128 v[4:7], v1 and ds_read_b128 v[8:11], v1 offset:16
        // in place of the global loads.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        words[5..9].copy_from_slice(&[0xDBFC_0000, 0x0400_0001, 0xDBFC_0010, 0x0800_0001]);
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of an LDS read",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// The block of `p` at byte `offset`.
    fn block_at(words: &[u32], offset: usize) -> usize {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        p.blocks.iter().position(|b| b.offset == offset).unwrap()
    }

    /// The table offset of what the scalar load of `s` reads.
    fn table_offset(r: &Resources, s: &Source) -> u32 {
        match *s {
            Source::Loaded { load, first: 0 } => {
                r.scalar.iter().find(|x| x.0 == load).unwrap().1.offset
            }
            ref s => panic!("{s:?}"),
        }
    }

    #[test]
    fn root_constants_dword0_is_offset_not_base() {
        // s0 = RootType::Constants (dword 0), s4 = Table, s[6:7] = Descriptor.
        let layout = [
            param(Constants, 0, 2),
            param(Table, 4, 1),
            param(Descriptor, 6, 2),
        ];
        // Added to Table (s_add_u32 s8, s0, s4) and Descriptor (s_add_u32 s10, s0, s6; s_addc_u32 s11, s7, 0),
        // plus passed directly as soffset and global vaddr.
        let load_table = s_load_x4(12, 8, 125, 0);
        let load_soffset = s_load_x4(16, 4, 0, 0);
        let load_desc = global_load(0, 0, 10, 0);
        let load_vaddr = global_load(1, 0, 6, 0);
        let r = track_words(
            &[
                sop2(0x00, 8, 0, 4), // s_add_u32 s8, s0, s4
                s_mov(9, 0x80),      // s_mov_b32 s9, 0
                load_table[0],
                load_table[1],
                s_mov(5, 0x80), // s_mov_b32 s5, 0
                load_soffset[0],
                load_soffset[1],
                sop2(0x00, 10, 0, 6),    // s_add_u32 s10, s0, s6
                sop2(0x04, 11, 7, 0x80), // s_addc_u32 s11, s7, 0
                vop1(0x001, 0, sgpr(0)), // v_mov_b32 v0, s0
                load_desc[0],
                load_desc[1],
                load_vaddr[0],
                load_vaddr[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        assert_eq!(r.scalar.len(), 4);
        assert_eq!(r.scalar[0].1.param, 1);
        assert!(r.scalar[0].1.dynamic.is_some());
        assert_eq!(r.scalar[1].1.param, 1);
        assert!(r.scalar[1].1.dynamic.is_some());
        assert_eq!(r.scalar[2].1.param, 2);
        assert!(r.scalar[2].1.dynamic.is_some());
        assert_eq!(r.scalar[3].1.param, 2);
        assert!(r.scalar[3].1.dynamic.is_some());

        // RootType::Constants dword 0 alone is NOT a base address.
        let only_const = s_load_x4(8, 0, 125, 0);
        let err = try_track_words(
            &[s_mov(1, 0x80), only_const[0], only_const[1], S_ENDPGM],
            &[param(Constants, 0, 2)],
            HwStage::Cs,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 4,
                what: "an address from a root parameter that is no address"
            }
        );
    }

    #[test]
    fn two_term_dynamic_addresses_produce_rel_offset() {
        let layout = [
            param(Constants, 0, 2), // s0 = dyn1, s1 = dyn2
            param(Table, 4, 1),     // s4 = table
        ];
        // 1) (base + dyn1) + dyn2 via two s_add_u32s.
        // 2) (base + 0x10 + dyn1) with soffset = dyn2 and immediate offset = 0x20.
        let load1 = s_load_x4(12, 8, 125, 0);
        let load2 = s_load_x4(16, 10, 1, 0x20);
        let r = track_words(
            &[
                sop2(0x00, 8, 4, 0), // s_add_u32 s8, s4, s0
                sop2(0x00, 8, 8, 1), // s_add_u32 s8, s8, s1
                s_mov(9, 0x80),      // s_mov_b32 s9, 0
                load1[0],
                load1[1],
                sop2(0x00, 10, 4, 0x90), // s_add_u32 s10, s4, 16
                sop2(0x00, 10, 10, 0),   // s_add_u32 s10, s10, s0
                s_mov(11, 0x80),         // s_mov_b32 s11, 0
                load2[0],
                load2[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        let base_val = r.params.iter().find(|p| p.1 == 1).unwrap().0;
        let dyn2_val = r.params.iter().find(|p| p.1 == 0 && p.2 == 1).unwrap().0;

        let a0 = r.scalar[0].1;
        assert_eq!(a0.param, 1);
        assert_eq!(a0.offset, 0);
        assert!(a0.dynamic.is_some());
        let rel0 = a0.rel.expect("two-term add needs rel");
        assert_eq!(rel0.base, base_val);
        assert_eq!(rel0.extra_dynamic, None);
        assert_eq!(a0.dynamic, Some(rel0.value));

        let a1 = r.scalar[1].1;
        assert_eq!(a1.param, 1);
        // s_offset (0x10) is inside rel_offset(value, base), so offset holds only extra (0x20).
        assert_eq!(a1.offset, 0x20);
        let rel1 = a1.rel.expect("lo dyn + soffset dyn needs rel");
        assert_eq!(rel1.base, base_val);
        assert_eq!(rel1.extra_dynamic, Some(dyn2_val));
        assert_eq!(a1.dynamic, Some(dyn2_val));
    }

    #[test]
    fn scaled_subtracted_add3_addk_and_aligned_or_addresses() {
        let layout = [
            param(Constants, 0, 2),  // s0 = dyn1, s1 = dyn2
            param(Table, 4, 1),      // s4 = table
            param(Descriptor, 6, 2), // s[6:7] = desc
        ];
        // Positive cases:
        // - s_addk_i32 (+0x10 and -0x4)
        // - s_lshl{1..4}_add_u32 (base on src1)
        // - v_lshl_add_u32 (base on src2)
        // - v_add3_u32 (base + dyn1 + dyn2)
        // - s_sub_u32, s_sub_i32, v_sub_co_u32, v_sub_nc_u32, v_subrev_co_u32, v_subrev_nc_u32 (minuend)
        // - aligned s_or_b32 (table | 8, (table + 8) | 4, desc | 2) and v_or_b32 (table | 8)
        // - uniform v_mov_b32 -> v_readfirstlane_b32 and v_readlane_b32
        let mut words = vec![
            s_mov(9, 0x80), // s9 = 0 (table hi)
            s_mov(8, 4),    // s8 = table
            s_addk_i32(8, 16),
            s_addk_i32(8, -4),
        ];
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [0] offset = 12
        for op in [0x2Eu8, 0x2F, 0x30, 0x31] {
            words.push(sop2(op, 8, 0, 4));
            words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [1..=4]
        }
        // v_lshl_add_u32 v2, s0, 2, s4 -> global_load v3, v[2:3], off (with v3 = 0)
        words.push(vop1(0x001, 3, 0x80)); // v3 = 0
        words.extend_from_slice(&vop3(0x346, 2, sgpr(0), 0x82, sgpr(4)));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [5]
        // v_add3_u32 v2, s0, s4, s1 -> global_load
        words.extend_from_slice(&vop3(0x36D, 2, sgpr(0), sgpr(4), sgpr(1)));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [6]
        // s_sub_u32 and s_sub_i32 on table minuend
        words.push(sop2(0x01, 8, 4, 0));
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [7]
        words.push(sop2(0x03, 8, 4, 0));
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [8]
        // v_sub_co_u32, v_sub_nc_u32, v_subrev_co_u32, v_subrev_nc_u32
        words.push(vop1(0x001, 5, sgpr(0))); // v5 = dyn1
        words.push(vop1(0x001, 6, sgpr(4))); // v6 = table
        words.extend_from_slice(&vop3b(0x310, 2, 106, vgpr(6), vgpr(5), 0));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [9]
        words.push(vop2(0x26, 2, vgpr(6), 5)); // v_sub_nc_u32 v2, v6, v5
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [10]
        words.extend_from_slice(&vop3b(0x319, 2, 106, vgpr(5), vgpr(6), 0));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [11]
        words.push(vop2(0x27, 2, vgpr(5), 6)); // v_subrev_nc_u32 v2, v5, v6
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [12]
        // Aligned OR: table | 8, (table + 8) | 4, desc | 2, v_or_b32(table, 8)
        words.push(sop2(0x10, 8, 4, 0x88)); // s_or_b32 s8, s4, 8
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [13] offset 8
        words.push(sop2(0x10, 8, 8, 0x84)); // s_or_b32 s8, s8, 4
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [14] offset 12
        words.push(sop2(0x10, 10, 6, 0x82)); // s_or_b32 s10, s6, 2
        words.push(s_mov(11, 7));
        words.extend_from_slice(&global_load(4, 3, 10, 0)); // [15] desc offset 2
        words.push(vop2(0x1C, 2, sgpr(4), 7)); // v7 needs to be 8:
        words.pop();
        words.push(vop1(0x001, 7, 0x88)); // v7 = 8
        words.push(vop2(0x1C, 2, sgpr(4), 7)); // v_or_b32 v2, s4, v7? wait v7 is not Const in copy-prop
        // Use inline constant 8 as src0 in v_or_b32 v2, 8, v6 (v6 = table)
        words.pop();
        words.push(vop2(0x1C, 2, 0x88, 6));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [16] table offset 8
        // Uniform v_mov_b32 -> v_readfirstlane_b32 / v_readlane_b32
        words.push(vop1(0x001, 7, sgpr(4))); // v7 = s4
        words.push(vop1(0x002, 8, vgpr(7))); // v_readfirstlane_b32 s8, v7
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [17]
        words.extend_from_slice(&vop3(0x360, 8, vgpr(7), 0x80, 0)); // v_readlane_b32 s8, v7, 0
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [18]
        words.push(S_ENDPGM);

        let r = track_words(&words, &layout, HwStage::Cs);
        assert_eq!(r.scalar.len(), 19);
        assert_eq!(r.scalar[0].1, at(1, 12));
        for idx in 1..=6 {
            assert!(r.scalar[idx].1.rel.is_some(), "scalar[{idx}] missing rel");
        }
        for idx in 7..=12 {
            assert!(r.scalar[idx].1.rel.is_some(), "sub[{idx}] missing rel");
        }
        assert_eq!(r.scalar[13].1, at(1, 8));
        assert_eq!(r.scalar[14].1, at(1, 12));
        assert_eq!(r.scalar[15].1, at(2, 2));
        assert_eq!(r.scalar[16].1, at(1, 8));
        assert_eq!(r.scalar[17].1, at(1, 0));
        assert_eq!(r.scalar[18].1, at(1, 0));

        // Negative cases that MUST fail:
        let bad_cases: &[&[u32]] = &[
            // Base on shifted src0 of s_lshl1_add_u32
            &[
                s_mov(9, 0x80),
                sop2(0x2E, 8, 4, 0),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
            // Base on shifted src0 of v_lshl_add_u32
            &[
                vop1(0x001, 3, 0x80),
                vop3(0x346, 2, sgpr(4), 0x82, sgpr(0))[0],
                vop3(0x346, 2, sgpr(4), 0x82, sgpr(0))[1],
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                S_ENDPGM,
            ],
            // Two bases in v_add3_u32
            &[
                vop1(0x001, 3, 0x80),
                vop3(0x36D, 2, sgpr(4), sgpr(6), sgpr(0))[0],
                vop3(0x36D, 2, sgpr(4), sgpr(6), sgpr(0))[1],
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                S_ENDPGM,
            ],
            // Base on subtrahend (const - base) in s_sub_u32
            &[
                s_mov(9, 0x80),
                sop2(0x01, 8, 0x90, 4),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
            // Base on subtrahend in v_subrev_nc_u32 (vsrc1 - src0 = 16 - table)
            &[
                vop1(0x001, 3, 0x80),
                vop1(0x001, 5, 0x90),      // v5 = 16
                vop2(0x27, 2, sgpr(4), 5), // v2 = v5 - s4
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                S_ENDPGM,
            ],
            // Base on subtrahend in v_sub_nc_u32 (src0 - vsrc1 = 16 - table)
            &[
                vop1(0x001, 3, 0x80),
                vop1(0x001, 5, sgpr(4)), // v5 = table
                vop2(0x26, 2, 0x90, 5),  // v2 = 16 - v5
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                S_ENDPGM,
            ],
            // Unaligned / out-of-range OR: table | 16
            &[
                s_mov(9, 0x80),
                sop2(0x10, 8, 4, 0x90),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
            // Overlapping OR: (table + 4) | 4
            &[
                s_mov(9, 0x80),
                sop2(0x00, 8, 4, 0x84),
                sop2(0x10, 8, 8, 0x84),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
            // Descriptor | 4 (>= 4 limit)
            &[
                sop2(0x10, 10, 6, 0x84),
                s_mov(11, 7),
                s_load_x4(12, 10, 125, 0)[0],
                s_load_x4(12, 10, 125, 0)[1],
                S_ENDPGM,
            ],
            // Dynamic term + OR: (table + dyn) | 4
            &[
                s_mov(9, 0x80),
                sop2(0x00, 8, 4, 0),
                sop2(0x10, 8, 8, 0x84),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
        ];
        for (idx, words) in bad_cases.iter().enumerate() {
            assert!(
                try_track_words(words, &layout, HwStage::Cs).is_err(),
                "bad case {idx} unexpectedly succeeded"
            );
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_an_if_else_join_are_traced_per_path() {
        let words = WATERFALL_JOINED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The join after the else arm: lanes that ran it (lanes from 16 on)
        // take the second entry (0x20); the others come through the flow
        // block, whose join gives lanes that ran the then arm the first one
        // (0). No lane that skipped the then arm comes that way, so the
        // dwords they held before (the inputs) are not traced.
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x9c));
        let [Some(flow), Some(other)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        let Source::LanePhi { block, args } = flow else {
            panic!("{flow:?}")
        };
        assert_eq!(*block, block_at(words, 0x70));
        let [None, Some(then)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // The loop's compares are of the joined dwords; still recognised.
        assert_eq!(r.waterfall.compares.len(), 4);
        assert_eq!(r.waterfall.reads.len(), 8);
        // Without the lanes' paths the join is refused.
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        let err = track(&p, &layout, HwStage::Cs, None).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi, without the lanes' paths",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn high_half_carry_va_mask_and_table_carry() {
        let layout = [
            param(Constants, 0, 2),  // s0, s1
            param(Table, 4, 1),      // s4
            param(Descriptor, 6, 2), // s[6:7]
        ];
        // 1) Constant-offset v_add_co_u32 + v_add_co_ci_u32(s7, 0, vcc).
        // 2) Table 64-bit dynamic carry: s_mov_b32 s9, 0; s_add_u32 s8, s4, s0; s_addc_u32 s9, s9, s1.
        // 3) VA mask s_and_b32 hi, 0xFFFF on global_load_dword.
        // 4) VA mask s_and_b32 hi, 0xFFFF inside Source::Built before s_or_b32 0x100000.
        let r = track_words(
            &[
                vop3b(0x30F, 2, 106, sgpr(6), 0x90, 0)[0],
                vop3b(0x30F, 2, 106, sgpr(6), 0x90, 0)[1],
                vop3b(0x128, 3, 106, sgpr(7), 0x80, sgpr(106))[0],
                vop3b(0x128, 3, 106, sgpr(7), 0x80, sgpr(106))[1],
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                s_mov(9, 0x80),
                sop2(0x00, 8, 4, 0),
                sop2(0x04, 9, 9, 1),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                sop2(0x00, 10, 6, 0),
                sop2(0x04, 11, 7, 0x80),
                sop2(0x0E, 11, 11, 0xFF),
                0x0000_FFFF, // s_and_b32 s11, s11, 0xFFFF
                vop1(0x001, 5, 0x80),
                global_load(4, 5, 10, 0)[0],
                global_load(4, 5, 10, 0)[1],
                sop2(0x0E, 1, 7, 0xFF),
                0x0000_FFFF, // s_and_b32 s1, s7, 0xFFFF
                sop2(0x10, 1, 1, 0xFF),
                0x0010_0000, // s_or_b32 s1, s1, 0x100000
                s_mov(0, 6),
                s_mov(2, 0xC1),
                s_mov(3, 0xFF),
                0x1101_4FAC,
                s_buf_load(0, 0)[0],
                s_buf_load(0, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        assert_eq!(r.scalar[0].1, at(2, 16));
        assert_eq!(r.scalar[1].1.param, 1);
        assert!(r.scalar[1].1.dynamic.is_some());
        assert_eq!(r.scalar[2].1.param, 2);
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(2, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC],
            }
        );

        // Invalid VA mask (0x0F -> 0x0F & 0xFF != 0xFF) MUST fail.
        let err_mask = try_track_words(
            &[
                sop2(0x0E, 9, 7, 0x8F), // s_and_b32 s9, s7, 15
                s_mov(8, 6),
                vop1(0x001, 5, 0x80),
                global_load(4, 5, 8, 0)[0],
                global_load(4, 5, 8, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        )
        .unwrap_err();
        assert_eq!(
            err_mask,
            ShaderError::Unsupported {
                offset: 12,
                what: "an address with an untracked high half"
            }
        );

        // Unrelated high half SGPR MUST fail (keep s7 referenced so Descriptor high half exists).
        let err_hi = try_track_words(
            &[
                s_mov(12, 7),
                s_mov(8, 6),
                s_mov(9, 0), // unrelated constant s0 as hi
                vop1(0x001, 5, 0x80),
                global_load(4, 5, 8, 0)[0],
                global_load(4, 5, 8, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        )
        .unwrap_err();
        assert_eq!(
            err_hi,
            ShaderError::Unsupported {
                offset: 16,
                what: "an address with an untracked high half"
            }
        );
    }

    #[test]
    fn descriptors_joined_from_a_per_lane_load_and_a_per_lane_choice_are_traced_per_path() {
        let words = WATERFALL_JOINED_MIXED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0xb8));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::Lanes { args: picked, .. }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        // The then arm loads each lane's entry from the table at s0.
        let [None, Some(Source::Fetched { load, first: 0 })] = flow[..] else {
            panic!("{flow:?}")
        };
        let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
        assert_eq!((a.param, a.offset), (0, 0));
        assert!(a.dynamic.is_some());
        // The else arm picks the first entry where vcc is set.
        let offsets: Vec<u32> = picked.iter().map(|s| table_offset(&r, s)).collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_descriptor_read_from_the_first_lane_of_a_loop_phi_fails_naming_it() {
        let err = waterfall(WATERFALL_LOOPED_WAVE32, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi at a loop header",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_per_lane_descriptor_some_lanes_do_not_write_fails() {
        // The else arm of WATERFALL_JOINED_WAVE32 not skipped (its
        // s_cbranch_execz an s_nop): it runs with EXEC clear in the lanes
        // of the then arm, which keep the first entry; the second is not
        // every lane's.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let skip = words.iter().rposition(|&w| w == 0xBF88_0008).unwrap();
        words[skip] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a per-lane descriptor written where some lanes may keep the old one",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_buffer_descriptor_joined_per_lane_is_traced_on_the_paths_of_lanes_that_load() {
        let words = WATERFALL_JOINED_BUFFER_WAVE32;
        let r = waterfall(words, false).unwrap();
        let buffer = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The outer join: lanes from the else arm's inner join, or (below
        // 16) through the flow block.
        let Source::LanePhi { block, args } = &buffer.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x78));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::LanePhi {
                block: inner,
                args: nested,
            }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        let [None, Some(then)] = &flow[..] else {
            panic!("{flow:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // Lanes from 24 on skip the inner arm and keep the inputs, but do
        // not load: that edge is not traced.
        assert_eq!(*inner, block_at(words, 0x74));
        let [None, Some(other)] = &nested[..] else {
            panic!("{nested:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        assert_eq!(r.waterfall.compares.len(), 2);
        assert_eq!(r.waterfall.reads.len(), 4);
    }

    #[test]
    fn phi_joined_and_loop_carried_addresses() {
        let layout = [
            param(Constants, 0, 1),  // s0
            param(Descriptor, 2, 2), // s[2:3]
            param(Descriptor, 4, 2), // s[4:5]
        ];
        // 1) If/else diamond joining (s2 + 4) and (s2 + 8).
        let r_if = track_words(
            &[
                sopc(0x06, 0, 0x80),    // s_cmp_eq_u32 s0, 0
                sopp(0x05, 2),          // s_cbranch_scc1 -> +2
                sop2(0x00, 8, 2, 0x84), // s8 = s2 + 4
                sopp(0x02, 1),          // s_branch -> join
                sop2(0x00, 8, 2, 0x88), // s8 = s2 + 8
                s_mov(9, 3),            // join: s9 = s3
                vop1(0x001, 0, 0x80),
                global_load(1, 0, 8, 0)[0],
                global_load(1, 0, 8, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        let base1 = r_if.params.iter().find(|p| p.1 == 1 && p.2 == 0).unwrap().0;
        let rel_if = r_if.scalar[0].1.rel.expect("if/else phi needs rel");
        assert_eq!(r_if.scalar[0].1.param, 1);
        assert_eq!(rel_if.base, base1);

        // 2) Loop-carried pointer induction phi (s_add_u32 + s_addc_u32 back-edge).
        let r_loop = track_words(
            &[
                s_mov(8, 2),                // s8 = s2
                s_mov(9, 3),                // s9 = s3
                s_mov(10, 0x80),            // s10 = 0
                vop1(0x001, 0, 0x80),       // v0 = 0
                global_load(1, 0, 8, 0)[0], // loop:
                global_load(1, 0, 8, 0)[1],
                sop2(0x00, 8, 8, 0x84), // s8 += 4
                sop2(0x04, 9, 9, 0x80), // s9 += carry
                sop2(0x00, 10, 10, 0x81),
                sopc(0x0A, 10, 0x84),
                sopp(0x05, -7), // s_cbranch_scc1 loop
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        let base_loop = r_loop
            .params
            .iter()
            .find(|p| p.1 == 1 && p.2 == 0)
            .unwrap()
            .0;
        let rel_loop = r_loop.scalar[0].1.rel.expect("loop phi needs rel");
        assert_eq!(r_loop.scalar[0].1.param, 1);
        assert_eq!(rel_loop.base, base_loop);

        // 3) Negative: mismatched-base phi joining s2 (param 1) and s4 (param 2) MUST fail.
        let err = try_track_words(
            &[
                sopc(0x06, 0, 0x80), // s_cmp_eq_u32 s0, 0
                sopp(0x05, 3),       // s_cbranch_scc1 -> else
                s_mov(8, 2),
                s_mov(9, 3),
                sopp(0x02, 2), // s_branch -> join
                s_mov(8, 4),
                s_mov(9, 5),
                vop1(0x001, 0, 0x80),
                global_load(1, 0, 8, 0)[0],
                global_load(1, 0, 8, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 32,
                what: "an address from a phi"
            }
        );
    }
}
