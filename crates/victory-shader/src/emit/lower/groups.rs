//! A console group of several wave64 waves that share LDS or meet at
//! `s_barrier`, on a host without 64-lane waves: the whole group is one
//! host group, each console wave two 32-lane host waves in the console flow
//! ([`super::CONSOLE_FLOW`]). LDS and `s_barrier` map to the host group's
//! own; what a console wave does as a whole (its branches, lane masks read
//! as numbers, lane reads, whole-wave sections) goes through the exchange
//! of its two halves, one LDS dword per host wave, between group barriers.
//!
//! A group barrier holds only where every thread of the group gets to it,
//! so the group runs one control flow, the same in every thread:
//!
//! * a branch on an SCC compare of values that are the same in every
//!   console wave of the group (user SGPRs, the group id, scalar loads at
//!   such addresses, scalar arithmetic on them, loop counters) is taken the
//!   same way by every thread as it is;
//! * a branch that skips a region (its other successor is its immediate
//!   post-dominator) is taken into the region when some console wave of the
//!   group goes in, by a vote of the group's waves. A console wave that
//!   would have skipped it runs it too, with every vector instruction off
//!   (its guard holds only in the waves that went in), and arrives at the
//!   join with the values it would have arrived with: each phi there takes,
//!   in such a wave, its value from the skipping edge;
//! * a branch to an `s_endpgm` taken when EXEC is empty ends the group only
//!   when every console wave would end; the waves that would have ended go
//!   on with every vector instruction off.
//!
//! The console's own barriers then pair as they do on the console: a
//! console wave that ended takes no part in them there, and one that goes
//! on here does nothing they order. What does not fit stops with a reason:
//! a loop the waves of a group may leave at different iterations, a branch
//! with code on both sides that the waves may take differently, an
//! `s_barrier` in a region only some waves run (the console would pair it
//! with another barrier of the waves that skip it), a scalar load whose
//! address may differ between the waves in code a wave runs only because
//! others do.

use std::collections::{HashMap, HashSet};

use super::Lower;
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::emit::shader::flags;
use crate::gfx10::Fields;
use crate::ir::{EXEC, Exit, Program, Reg, VCC, Value};

type Why = &'static str;

pub(crate) const LOOP_LEFT_APART: Why = "a loop the console waves of a group may leave at different iterations, on a host without 64-lane waves";
pub(crate) const BRANCH_APART: Why = "a branch the console waves of a group may take different ways, with code on both sides, on a host without 64-lane waves";
pub(crate) const BARRIER_IN_REGION: Why = "an s_barrier in a region only some console waves of a group run, on a host without 64-lane waves";
const EXIT_NOT_EXEC: Why = "a program end some console waves of a group take on a test other than an empty EXEC, on a host without 64-lane waves";
const SCALAR_LOAD_APART: Why = "a scalar load at an address that may differ between console waves, in code a wave runs because others do";
const UNSTRUCTURED: Why = "a region of a branch entered or left other than through the branch and its join, in a group of several console waves";
const NO_EXIT: Why = "a loop no console wave leaves, in a group of several console waves";
const NO_LDS_ROOM: Why =
    "LDS with no room for the exchange of a group of several console waves (32 KiB on the host)";

/// How the group takes a block's conditional branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Branch {
    /// Every thread decides the same: an SCC compare of values the same in
    /// every console wave of the group.
    Uniform,
    /// `region`, whose only predecessor is the branch's block, runs what
    /// `join`, the branch's immediate post-dominator, is reached without:
    /// the group goes in if some console wave would.
    Skip { region: usize, join: usize },
    /// The other successor holds only `s_endpgm` and is taken when EXEC is
    /// empty: the group ends there only if every console wave would.
    Exit { go_on: usize },
}

/// What [`plan`] found.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    /// 32-lane host waves in the group (two per console wave).
    pub host_waves: u32,
    /// Per IR block that ends in a conditional branch.
    pub branches: HashMap<usize, Branch>,
    /// Immediate dominators of the IR blocks.
    pub idom: Vec<usize>,
    /// The `region` or `go_on` successor of a [`Branch::Skip`] or
    /// [`Branch::Exit`]: that branch's block.
    pub entered_from: HashMap<usize, usize>,
    /// Per join of [`Branch::Skip`]s: their blocks and regions, innermost
    /// first.
    pub joins: HashMap<usize, Vec<(usize, usize)>>,
}

impl Plan {
    /// Whether IR block `a` dominates `b`.
    pub(crate) fn dominates(&self, a: usize, mut b: usize) -> bool {
        loop {
            if a == b {
                return true;
            }
            let d = self.idom[b];
            if d == b || d == usize::MAX {
                return false;
            }
            b = d;
        }
    }
}

/// Whether `b` holds nothing but `s_endpgm`.
fn bare_end(p: &Program, b: usize) -> bool {
    let block = &p.blocks[b];
    block.exit == Exit::End
        && block.insts.len() == 1
        && block.insts[0].name().starts_with("s_endpgm")
}

/// The form a group of `threads` threads (several console waves, `lds`
/// bytes of the title's LDS) runs in, or why it cannot. `uniform` holds the
/// program inputs that are the same in every console wave of a group (user
/// SGPRs, the group id).
pub(crate) fn plan(
    p: &Program,
    uniform: &HashSet<Value>,
    threads: u32,
    lds: u32,
) -> Result<Plan, Why> {
    let host_waves = threads / 32;
    if lds + 4 * host_waves > 32768 {
        return Err(NO_LDS_ROOM);
    }
    let n = p.blocks.len();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; n];
    for (k, &b) in order.iter().enumerate() {
        index[b] = k;
    }
    let idom = crate::ir::idoms(&p.blocks, &order, &index);
    if order.iter().any(|&b| p.blocks[b].exit == Exit::Indirect) {
        return Err("an indirect jump");
    }
    let ipdom = post_dominators(p, &order)?;
    let dom = |a: usize, mut b: usize| loop {
        if a == b {
            break true;
        }
        let d = idom[b];
        if d == b || d == usize::MAX {
            break false;
        }
        b = d;
    };
    // Branches and the values the same in every wave depend on each other
    // (a phi at the join of a skipped region is not, as the waves that
    // skipped it take the other edge's value): from no such joins, until
    // the joins found stop growing.
    let mut split_joins = HashSet::new();
    let branches = loop {
        let same = uniform_values(p, uniform, &split_joins);
        let branches = classify(p, &order, &index, &dom, &ipdom, &same)?;
        let joins: HashSet<usize> = branches
            .values()
            .filter_map(|b| match *b {
                Branch::Skip { join, .. } => Some(join),
                _ => None,
            })
            .collect();
        if joins == split_joins {
            break (branches, same);
        }
        split_joins = joins;
    };
    let (branches, same) = branches;
    let mut entered_from = HashMap::new();
    let mut joins: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let depth = |mut b: usize| {
        let mut d = 0;
        while idom[b] != b && idom[b] != usize::MAX {
            b = idom[b];
            d += 1;
        }
        d
    };
    for (&x, &branch) in &branches {
        let (entry, join) = match branch {
            Branch::Uniform => continue,
            Branch::Skip { region, join } => (region, Some(join)),
            Branch::Exit { go_on } => (go_on, None),
        };
        if p.blocks[entry].preds != [x] {
            return Err(UNSTRUCTURED);
        }
        let scope: Vec<usize> = order.iter().copied().filter(|&d| dom(entry, d)).collect();
        for &d in &scope {
            for s in p.blocks[d].exit.successors() {
                let inside = dom(entry, s);
                let ok = match join {
                    Some(j) => inside || s == j,
                    None => inside || bare_end(p, s),
                };
                if !ok {
                    return Err(UNSTRUCTURED);
                }
            }
            for inst in &p.blocks[d].insts {
                if join.is_some() && inst.name() == "s_barrier" {
                    return Err(BARRIER_IN_REGION);
                }
                if matches!(inst.fields, Fields::Smem { .. })
                    && inst.reads.iter().any(|r| !same.contains(&r.1))
                {
                    return Err(SCALAR_LOAD_APART);
                }
                // Scalar stores would take effect in every wave.
                if matches!(inst.fields, Fields::Smem { .. })
                    && !(inst.name().starts_with("s_load")
                        || inst.name().starts_with("s_buffer_load"))
                {
                    return Err("a scalar memory write in a group of several console waves");
                }
            }
        }
        entered_from.insert(entry, x);
        if let Some(j) = join {
            if p.blocks[j]
                .preds
                .iter()
                .any(|&q| index[q] != usize::MAX && index[j] <= index[q])
            {
                return Err(UNSTRUCTURED);
            }
            joins.entry(j).or_default().push((x, entry));
        }
    }
    for list in joins.values_mut() {
        list.sort_by_key(|&(x, _)| std::cmp::Reverse(depth(x)));
    }
    Ok(Plan {
        host_waves,
        branches,
        idom,
        entered_from,
        joins,
    })
}

/// Each reachable block's immediate post-dominator; `usize::MAX` stands for
/// the program's end (past every `s_endpgm`).
fn post_dominators(p: &Program, order: &[usize]) -> Result<Vec<usize>, Why> {
    let n = p.blocks.len();
    let end = n;
    let words = (n + 1).div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom = vec![full; n + 1];
    let set = |s: &mut Vec<u64>, b: usize| s[b / 64] |= 1 << (b % 64);
    let has = |s: &[u64], b: usize| s[b / 64] >> (b % 64) & 1 != 0;
    pdom[end] = vec![0; words];
    set(&mut pdom[end], end);
    let succs = |b: usize| match p.blocks[b].exit {
        Exit::End => vec![end],
        e => e.successors(),
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            let mut s = vec![u64::MAX; words];
            for t in succs(b) {
                for (w, x) in s.iter_mut().zip(&pdom[t]) {
                    *w &= x;
                }
            }
            set(&mut s, b);
            if s != pdom[b] {
                pdom[b] = s;
                changed = true;
            }
        }
    }
    let count = |s: &[u64]| s.iter().map(|w| w.count_ones()).sum::<u32>();
    let mut ipdom = vec![usize::MAX; n];
    for &b in order {
        // A block from which no path ends keeps every block.
        if !has(&pdom[b], end) {
            return Err(NO_EXIT);
        }
        let k = count(&pdom[b]);
        ipdom[b] = (0..=n)
            .find(|&d| d != b && has(&pdom[b], d) && count(&pdom[d]) + 1 == k)
            .map_or(usize::MAX, |d| if d == end { usize::MAX } else { d });
    }
    Ok(ipdom)
}

/// The scalar values every console wave of the group holds the same: the
/// largest set closed under scalar arithmetic and scalar loads on values in
/// it (none reading EXEC), and phis whose arguments are in it, except at
/// `split_joins` (where the waves that skipped a region take their own
/// value).
fn uniform_values(
    p: &Program,
    inputs: &HashSet<Value>,
    split_joins: &HashSet<usize>,
) -> HashSet<Value> {
    let masks = [EXEC, EXEC + 1, VCC, VCC + 1];
    let mask_reg = |r: Reg| matches!(r, Reg::S(c) if masks.contains(&c));
    let mut same: HashSet<Value> = inputs.clone();
    for (b, block) in p.blocks.iter().enumerate() {
        if !split_joins.contains(&b) {
            for phi in &block.phis {
                if matches!(phi.reg, Reg::S(_) | Reg::Scc) && !mask_reg(phi.reg) {
                    same.insert(phi.value);
                }
            }
        }
        for inst in &block.insts {
            let name = inst.name();
            let candidate = match inst.fields {
                Fields::Sop { .. } => !matches!(
                    name,
                    "s_getreg_b32" | "s_memtime" | "s_memrealtime" | "s_sendmsg" | "s_sendmsghalt"
                ),
                Fields::Smem { .. } => {
                    name.starts_with("s_load") || name.starts_with("s_buffer_load")
                }
                _ => false,
            };
            if candidate && !inst.reads.iter().any(|r| mask_reg(r.0)) {
                for &(r, v) in &inst.writes {
                    if !mask_reg(r) && !matches!(r, Reg::V(_)) {
                        same.insert(v);
                    }
                }
            }
        }
    }
    // Take out what reads a value not in the set, until nothing changes.
    let mut changed = true;
    while changed {
        changed = false;
        for block in &p.blocks {
            for phi in &block.phis {
                if same.contains(&phi.value) && phi.args.iter().any(|a| !same.contains(a)) {
                    same.remove(&phi.value);
                    changed = true;
                }
            }
            for inst in &block.insts {
                let writes_same = inst.writes.iter().any(|w| same.contains(&w.1));
                if writes_same && inst.reads.iter().any(|r| !same.contains(&r.1)) {
                    for w in &inst.writes {
                        same.remove(&w.1);
                    }
                    changed = true;
                }
            }
        }
    }
    same
}

/// The form of each conditional branch (see [`Branch`]).
fn classify(
    p: &Program,
    order: &[usize],
    index: &[usize],
    dom: &dyn Fn(usize, usize) -> bool,
    ipdom: &[usize],
    same: &HashSet<Value>,
) -> Result<HashMap<usize, Branch>, Why> {
    let mut out = HashMap::new();
    for &b in order {
        let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
            continue;
        };
        let Some(last) = p.blocks[b].insts.last() else {
            return Err("a branch without its instruction");
        };
        let name = last.name();
        let uniform = matches!(name, "s_cbranch_scc0" | "s_cbranch_scc1")
            && last.read(Reg::Scc).is_some_and(|v| same.contains(&v));
        let back = [taken, not_taken]
            .iter()
            .any(|&s| index[s] <= index[b] && dom(s, b));
        if uniform || taken == not_taken {
            out.insert(b, Branch::Uniform);
            continue;
        }
        if back {
            return Err(LOOP_LEFT_APART);
        }
        // To `s_endpgm` where EXEC is empty.
        let end = match name {
            "s_cbranch_execz" => Some((taken, not_taken)),
            "s_cbranch_execnz" => Some((not_taken, taken)),
            _ => None,
        };
        if let Some((e, go_on)) = end
            && bare_end(p, e)
        {
            out.insert(b, Branch::Exit { go_on });
            continue;
        }
        if bare_end(p, taken) || bare_end(p, not_taken) {
            return Err(EXIT_NOT_EXEC);
        }
        let join = ipdom[b];
        let branch = if join == taken {
            Branch::Skip {
                region: not_taken,
                join,
            }
        } else if join == not_taken {
            Branch::Skip {
                region: taken,
                join,
            }
        } else {
            return Err(BRANCH_APART);
        };
        out.insert(b, branch);
    }
    Ok(out)
}

/// The exchange slots and the group's votes ([`Lower`] in a group of
/// several console waves).
impl Lower<'_> {
    /// The group-shared dwords of [`Self::exchange`]: one per host wave,
    /// at least two.
    pub(super) fn xchg_slots(&mut self) -> V {
        if let Some(g) = self.xchg {
            return g;
        }
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let arr = self.s.m.array((threads / 32).max(2), self.s.t.i32);
        let g = self.s.m.global("xchg", arr, 3);
        self.xchg = Some(g);
        g
    }

    /// This thread's host wave in the group (console flow in halves: a
    /// console wave's lower half is host wave 2w, its upper half 2w + 1).
    pub(super) fn host_wave(&mut self) -> V {
        let f = self.console_flat();
        let e = self.s.entry;
        let k5 = self.s.m.i32(5);
        self.s.m.bin(e, Bin::LShr, f, k5)
    }

    /// In a group of several console waves: whether `bit`
    /// holds in some thread of the group, the same in every thread. Each
    /// host wave puts whether it holds in one of its lanes in its slot;
    /// after a group barrier, lane n of every host wave reads slot n. Every
    /// thread of the group gets here, as to [`Self::exchange`].
    pub(super) fn group_any(&mut self, at: usize, bit: V) -> V {
        let (i1, void) = (self.s.t.i1, self.s.t.void);
        let any = self
            .s
            .op(at, "waveAnyTrue", Attr::NoUnwind, i1, 113, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let (one, z) = (self.s.m.i32(1), self.s.m.i32(0));
        let word = self.s.m.select(at, any, one, z);
        let slots = self.xchg_slots();
        let hw = self.host_wave();
        let mine = self.s.m.element(at, slots, hw);
        self.s.m.store(at, mine, word);
        let mode = self.s.m.i32(1 | 8);
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let lane = self.s.op(
            at,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let waves = self.groups.as_ref().map_or(2, |g| g.host_waves);
        let n = self.s.m.i32(waves);
        let inside = self.s.m.cmp(at, pred::IULT, lane, n);
        let slot = self.s.m.select(at, inside, lane, z);
        let p = self.s.m.element(at, slots, slot);
        let w = self.s.m.load(at, p);
        let w = self.s.m.select(at, inside, w, z);
        let set = self.s.m.cmp(at, pred::NE, w, z);
        let r = self
            .s
            .op(at, "waveAnyTrue", Attr::NoUnwind, i1, 113, &[set]);
        // All have read before the slots are written again.
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        r
    }

    /// The group's way at block `b`'s branch ([`Branch::Skip`],
    /// [`Branch::Exit`]), from `taken`, this console wave's own
    /// test: into the region (or on) if `along_taken` says the taken edge
    /// leads there. A wave votes for its own way where it runs `b` for
    /// itself; the group goes that way if some wave votes for it.
    pub(super) fn group_vote(&mut self, b: usize, taken: V, along_taken: bool) -> V {
        let c = self.cur;
        let own = if along_taken {
            taken
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, taken, t)
        };
        let vote = match self.here[b] {
            Some(h) => self.s.m.bin(c, Bin::And, own, h),
            None => own,
        };
        self.group_votes.insert(b, (own, vote));
        self.group_any(c, vote)
    }

    /// In a group of several console waves: the incoming values of a phi at
    /// IR block `b` (one per pred in `preds`), where `b` joins regions some
    /// console waves did not go into ([`Branch::Skip`]). A region's
    /// edge carries, in a wave that did not go in, the value of the edge
    /// that skips it, `bypass` (valid in the skipping block), innermost
    /// region first.
    pub(super) fn group_fix(
        &mut self,
        b: usize,
        preds: &[usize],
        bypass: &[Option<V>],
        mut incoming: Vec<(V, usize)>,
    ) -> Vec<(V, usize)> {
        let Some(g) = &self.groups else {
            return incoming;
        };
        let Some(list) = g.joins.get(&b).cloned() else {
            return incoming;
        };
        for (k, &q) in preds.iter().enumerate() {
            for &(x, region) in &list {
                let g = self.groups.as_ref().expect("set");
                if !g.dominates(region, q) {
                    continue;
                }
                let skipped = preds.iter().position(|&p| p == x).and_then(|kx| bypass[kx]);
                let (Some(skipped), Some(&(own, _))) = (skipped, self.group_votes.get(&x)) else {
                    continue;
                };
                let qb = incoming[k].1;
                incoming[k].0 = self.s.m.select(qb, own, incoming[k].0, skipped);
            }
        }
        incoming
    }
}
