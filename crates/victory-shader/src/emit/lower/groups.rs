//! A console group of several wave64 waves that share LDS or meet at
//! `s_barrier`, on a host without 64-lane waves: the whole group is one
//! host group, each console wave two 32-lane host waves in the console flow
//! ([`super::CONSOLE_FLOW`]). LDS and `s_barrier` map to the host group's
//! own; what a console wave does as a whole (its branches, lane masks read
//! as numbers, lane reads, whole-wave sections) goes through the exchange
//! of its two halves, one LDS dword per host wave, between group barriers.
//!
//! A group barrier holds only where every thread of the group gets to it,
//! so the group runs one control flow, the same in every thread. Each IR
//! block runs in a console wave "for itself" where `here` holds (vector
//! instructions take effect only there); elsewhere the wave runs it with
//! every vector instruction off:
//!
//! * a branch on an SCC compare of values that are the same in every
//!   console wave of the group (user SGPRs, the group id, scalar loads at
//!   such addresses, scalar arithmetic on them, loop counters, a first-lane
//!   read the group makes its first wave's, below) is taken the same way by
//!   every thread as it is;
//! * a branch that skips a region (its other successor is its immediate
//!   post-dominator) is taken into the region when some console wave of the
//!   group goes in, by a vote of the group's waves. A console wave that
//!   would have skipped it runs it too, with every vector instruction off,
//!   and arrives at the join with the values it would have arrived with:
//!   each phi there takes, in such a wave, its value from the skipping edge;
//! * a region at most one console wave of the group can go into (its EXEC
//!   holds only lanes of that wave: a compare of the thread id with a
//!   constant) is the control flow of that wave: every branch inside it,
//!   loops and branches with code on both sides included, goes the way that
//!   wave goes, by the same vote;
//! * a loop the waves may leave at different iterations, with no
//!   `s_barrier` in it, runs while some wave is still in it: a wave that
//!   left stays in with every vector instruction off (`here` is the loop's
//!   "alive" bit) and keeps, for after the loop, the values it left with
//!   (each value used after the loop is "frozen": held in a phi of the
//!   loop's header where the wave is out). Its exits are votes: the group
//!   leaves when no wave stays. With several exits, a flag per exit says
//!   which one a wave took, and the loop's join takes that exit's values;
//! * a branch to an `s_endpgm` ends the group only when every console wave
//!   would end; the waves that would have ended go on with every vector
//!   instruction off.
//!
//! The console's own barriers then pair as they do on the console: a
//! console wave that ended takes no part in them there, and one that goes
//! on here does nothing they order; barriers stay out of the regions and
//! loops some waves run for others, so every console wave meets each one
//! the same number of times. Within a span between two barriers, the order
//! the group runs the waves' code in is one the console may run it in.
//!
//! A first-lane read (`v_readfirstlane_b32`) of a buffer load whose address
//! is the same in every lane of every wave (no per-lane offset or index,
//! one descriptor for the group), with all lanes on at both and no store
//! that could come before the load, is the first console wave's in every
//! wave: the console may run every wave's load at the same moment, before
//! any store of the group, and then all read the same dword.
//!
//! What does not fit stops with a reason: an `s_barrier` in a region only
//! some waves run or in a loop they may leave at different iterations, a
//! branch with code on both sides that the waves may take differently, a
//! loop whose exits lead to different code, a scalar load whose address
//! may differ between the waves in code a wave runs only because others do,
//! a lane mask made in such a loop and used after it.

use std::collections::{HashMap, HashSet};

use super::{Lower, Val};
use crate::emit::bitcode::{Attr, Bin, Term, V, pred};
use crate::emit::shader::flags;
use crate::gfx10::{Fields, Operand};
use crate::io::Input;
use crate::ir::{EXEC, Exit, Inst, Program, Reg, VCC, Value};

type Why = &'static str;

pub(crate) const BARRIER_IN_LOOP: Why = "an s_barrier in a loop the console waves of a group may leave at different iterations, on a host without 64-lane waves";
pub(crate) const BRANCH_APART: Why = "a branch the console waves of a group may take different ways, with code on both sides, on a host without 64-lane waves";
pub(crate) const BARRIER_IN_REGION: Why = "an s_barrier in a region only some console waves of a group run, on a host without 64-lane waves";
pub(crate) const EXITS_APART: Why = "a loop the console waves of a group may leave at different iterations, by exits with different code after them";
const EXIT_OFF_PATH: Why = "an exit of a loop the console waves of a group may leave at different iterations, off the path every iteration takes";
const EXIT_TWO_LOOPS: Why = "an exit from two loops at once that the console waves of a group may take at different iterations";
pub(crate) const MASK_OUT_OF_LOOP: Why = "a lane mask from a loop the console waves of a group may leave at different iterations, used after it";
const VALUE_OUT_OF_LOOP: Why = "a value from a loop the console waves of a group may leave at different iterations, used after it where it is not held";
pub(crate) const SCALAR_LOAD_APART: Why = "a scalar load at an address that may differ between console waves, in code a wave runs because others do";
const UNSTRUCTURED: Why = "a region of a branch entered or left other than through the branch and its join, in a group of several console waves";
const NO_EXIT: Why = "a loop no console wave leaves, in a group of several console waves";
const IRREDUCIBLE: Why = "irreducible control flow in a group of several console waves";
pub(crate) const NO_LDS_ROOM: Why = "LDS with no room for the exchange of a group of several console waves (the title's LDS and 4 bytes per 32-lane host wave over the host's 32,768 bytes)";

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
    /// The other successor holds only `s_endpgm`: the group ends there only
    /// if every console wave would.
    Exit { go_on: usize },
    /// Inside a region at most one console wave runs for itself: the way
    /// that wave goes.
    Follow,
    /// An exit of the divergent loop `lp` ([`Plan::loops`]): the group goes
    /// on to `stay` (in the loop) while some console wave would.
    Leave { stay: usize, lp: usize },
}

/// A loop the console waves of a group may leave at different iterations.
#[derive(Debug, Clone)]
pub(crate) struct Loop {
    pub header: usize,
    /// Per IR block: in the loop.
    pub body: Vec<bool>,
    /// The exiting blocks, in the order every iteration reaches them.
    pub exits: Vec<usize>,
    /// Per exit, the predecessor of [`Self::join`] its path ends in (the
    /// exiting block, or the last block of no effect after it).
    pub ends: Vec<usize>,
    /// Where every exit's path arrives (`usize::MAX`: the program's end).
    pub join: usize,
    /// The values used after the loop that it holds for the waves that
    /// left it (defined in it or in loops inside it), and whether each is
    /// an SCC bit.
    pub held: Vec<(Value, bool)>,
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
    /// The `region`, `go_on` or `stay` successor of a [`Branch::Skip`],
    /// [`Branch::Exit`] or [`Branch::Leave`] (other than a loop's header):
    /// that branch's block.
    pub entered_from: HashMap<usize, usize>,
    /// Per join of [`Branch::Skip`]s: their blocks and regions, innermost
    /// first.
    pub joins: HashMap<usize, Vec<(usize, usize)>>,
    /// Divergent loops; their headers.
    pub loops: Vec<Loop>,
    pub header_of: HashMap<usize, usize>,
    /// Blocks after a divergent loop whose immediate dominator is in it:
    /// they hold for the waves the loop's entry held for.
    pub exit_here: HashMap<usize, usize>,
    /// Per value defined in a divergent loop, the innermost such loop.
    pub def_loop: HashMap<Value, usize>,
    /// Values used after a divergent loop: the innermost loop freezing
    /// each.
    pub freeze: HashMap<Value, usize>,
    /// First-lane reads that take the group's first console wave's value.
    pub broadcast: HashSet<Value>,
}

impl Plan {
    /// Whether IR block `a` dominates `b`.
    pub(crate) fn dominates(&self, a: usize, b: usize) -> bool {
        dominates(&self.idom, a, b)
    }
}

fn dominates(idom: &[usize], a: usize, mut b: usize) -> bool {
    loop {
        if a == b {
            return true;
        }
        let d = idom[b];
        if d == b || d == usize::MAX {
            return false;
        }
        b = d;
    }
}

/// Whether `b` holds nothing but `s_endpgm`.
fn bare_end(p: &Program, b: usize) -> bool {
    let block = &p.blocks[b];
    block.exit == Exit::End
        && block.insts.len() == 1
        && block.insts[0].name().starts_with("s_endpgm")
}

/// Whether `b` does nothing but go on to its one successor.
fn no_effect(p: &Program, b: usize) -> Option<usize> {
    let block = &p.blocks[b];
    let Exit::Jump(t) = block.exit else {
        return None;
    };
    let quiet = |i: &Inst| {
        let n = i.name();
        n == "s_nop" || n == "s_branch" || n.starts_with("s_waitcnt") || n == "s_inst_prefetch"
    };
    (block.phis.is_empty() && block.insts.iter().all(quiet)).then_some(t)
}

fn is_store(inst: &Inst) -> bool {
    let n = inst.name();
    matches!(
        inst.fields,
        Fields::Buffer { .. } | Fields::Flat { .. } | Fields::Image { .. } | Fields::Smem { .. }
    ) && (n.contains("store") || n.contains("atomic"))
}

/// The form a group of `threads` threads (several console waves, `lds`
/// bytes of the title's LDS) runs in, or why it cannot. `uniform` holds the
/// program inputs that are the same in every console wave of a group (user
/// SGPRs, the group id); `io` the program's inputs.
pub(crate) fn plan(
    p: &Program,
    uniform: &HashSet<Value>,
    threads: [u32; 3],
    lds: u32,
    io: &[(Value, Input)],
) -> Result<Plan, Why> {
    let n_threads = threads.iter().product::<u32>();
    let host_waves = n_threads / 32;
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
    if crate::ir::irreducible_edge(&p.blocks, &order, &index).is_some() {
        return Err(IRREDUCIBLE);
    }
    let ipdom = post_dominators(p, &order)?;
    let cfg = Cfg {
        p,
        order: &order,
        index: &index,
        idom: &idom,
        ipdom: &ipdom,
        loops: natural_loops(p, &order, &index, &idom),
    };
    // The thread id along x, in a group of one dimension: the console
    // thread's flat id.
    let tid = (threads[1] == 1 && threads[2] == 1)
        .then(|| {
            io.iter()
                .find(|(_, i)| matches!(i, Input::ThreadIdInGroup(0)))
                .map(|&(v, _)| v)
        })
        .flatten();
    let lanes = Lanes::new(p, n_threads, tid);
    let mut broadcast = broadcast_candidates(p, &order, &lanes);
    // What the solution needs in the group's own flow: each first-lane
    // read of one dword, each value found the same in every lane and wave.
    // Where the solution leaves one elsewhere, again without it.
    let mut excluded = HashSet::new();
    loop {
        let (plan, same, inv) = cfg.solve(uniform, &broadcast, &excluded, &lanes, host_waves)?;
        let keep: HashSet<Value> = broadcast
            .iter()
            .copied()
            .filter(|&r| cfg.broadcast_holds(&plan, &same, r))
            .collect();
        let bad: HashSet<usize> = inv
            .at
            .values()
            .copied()
            .filter(|&b| !cfg.own_flow(&plan, b))
            .collect();
        if keep.len() == broadcast.len() && bad.is_subset(&excluded) {
            return Ok(plan);
        }
        broadcast = keep;
        excluded.extend(bad);
    }
}

/// A natural loop.
struct NatLoop {
    header: usize,
    body: Vec<bool>,
    latches: Vec<usize>,
}

/// Natural loops by header, outermost first.
fn natural_loops(p: &Program, order: &[usize], index: &[usize], idom: &[usize]) -> Vec<NatLoop> {
    let n = p.blocks.len();
    let mut out: Vec<NatLoop> = Vec::new();
    for &q in order {
        for s in p.blocks[q].exit.successors() {
            if !dominates(idom, s, q) {
                continue;
            }
            let k = match out.iter().position(|l| l.header == s) {
                Some(k) => k,
                None => {
                    out.push(NatLoop {
                        header: s,
                        body: vec![false; n],
                        latches: Vec::new(),
                    });
                    out.len() - 1
                }
            };
            out[k].latches.push(q);
            out[k].body[s] = true;
            let mut stack = vec![q];
            while let Some(x) = stack.pop() {
                if out[k].body[x] {
                    continue;
                }
                out[k].body[x] = true;
                stack.extend(
                    p.blocks[x]
                        .preds
                        .iter()
                        .copied()
                        .filter(|&r| index[r] != usize::MAX),
                );
            }
        }
    }
    let size = |l: &NatLoop| l.body.iter().filter(|&&x| x).count();
    out.sort_by_key(|l| std::cmp::Reverse(size(l)));
    out
}

struct Cfg<'a> {
    p: &'a Program,
    order: &'a [usize],
    index: &'a [usize],
    idom: &'a [usize],
    ipdom: &'a [usize],
    loops: Vec<NatLoop>,
}

impl Cfg<'_> {
    fn dom(&self, a: usize, b: usize) -> bool {
        dominates(self.idom, a, b)
    }

    /// The innermost natural loop holding `b`.
    fn innermost(&self, b: usize) -> Option<usize> {
        // Outermost first: the last one found is the innermost.
        (0..self.loops.len()).rev().find(|&l| self.loops[l].body[b])
    }

    /// The forms of the branches and the values the same in every wave;
    /// they depend on each other (a phi at the join of a skipped region, or
    /// a value used after a loop the waves left apart, is not the same in
    /// every wave), so from none such until what is split stops growing.
    fn solve(
        &self,
        uniform: &HashSet<Value>,
        broadcast: &HashSet<Value>,
        excluded: &HashSet<usize>,
        lanes: &Lanes,
        host_waves: u32,
    ) -> Result<(Plan, HashSet<Value>, Invariant), Why> {
        let mut split_joins = HashSet::new();
        let mut split_values = HashSet::new();
        loop {
            let (same, inv) = self.same_and_invariant(
                uniform,
                broadcast,
                excluded,
                lanes,
                &split_joins,
                &split_values,
            );
            let plan = self.forms(&same, &inv, lanes, host_waves, broadcast)?;
            let mut grew = false;
            for &j in plan.joins.keys() {
                grew |= split_joins.insert(j);
            }
            for l in &plan.loops {
                if l.join != usize::MAX {
                    grew |= split_joins.insert(l.join);
                }
                for &(v, _) in &l.held {
                    grew |= split_values.insert(v);
                }
            }
            if !grew {
                return Ok((plan, same, inv));
            }
        }
    }

    /// The scalar values the same in every wave, with the first-lane reads
    /// of values the same in every lane and wave among them, until those
    /// stop growing.
    fn same_and_invariant(
        &self,
        uniform: &HashSet<Value>,
        broadcast: &HashSet<Value>,
        excluded: &HashSet<usize>,
        lanes: &Lanes,
        split_joins: &HashSet<usize>,
        split_values: &HashSet<Value>,
    ) -> (HashSet<Value>, Invariant) {
        let mut reads: HashSet<Value> = HashSet::new();
        loop {
            let fixed: HashSet<Value> = broadcast.union(&reads).copied().collect();
            let inputs: HashSet<Value> = uniform.union(&fixed).copied().collect();
            let same = uniform_values(self.p, &inputs, &fixed, split_joins, split_values);
            let inv = invariant(self.p, self.order, &same, lanes, excluded);
            if inv.reads.is_subset(&reads) {
                return (same, inv);
            }
            reads.extend(inv.reads.iter().copied());
        }
    }

    /// Whether `b` runs in the group's own flow: in every console wave for
    /// itself, outside every region, divergent loop and part after one.
    fn own_flow(&self, plan: &Plan, mut d: usize) -> bool {
        loop {
            if plan.entered_from.contains_key(&d)
                || plan.header_of.contains_key(&d)
                || plan.exit_here.contains_key(&d)
            {
                return false;
            }
            match self.idom[d] {
                x if x == d || x == usize::MAX => return true,
                x => d = x,
            }
        }
    }

    fn forms(
        &self,
        same: &HashSet<Value>,
        inv: &Invariant,
        lanes: &Lanes,
        host_waves: u32,
        broadcast: &HashSet<Value>,
    ) -> Result<Plan, Why> {
        let p = self.p;
        let mut forms: HashMap<usize, Result<Branch, Why>> = HashMap::new();
        // Natural loop index -> divergent loop index.
        let mut divergent: HashMap<usize, usize> = HashMap::new();
        for &b in self.order {
            let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
                continue;
            };
            forms.insert(b, self.classify(b, taken, not_taken, same, inv));
        }
        // Regions at most one console wave goes into: their branches follow
        // that wave. Outermost first, by depth.
        let mut skips: Vec<(usize, usize)> = forms
            .iter()
            .filter_map(|(&b, f)| match f {
                Ok(Branch::Skip { region, .. }) => Some((b, *region)),
                _ => None,
            })
            .collect();
        skips.sort_by_key(|&(b, _)| self.index[b]);
        let mut followed = vec![false; p.blocks.len()];
        for (b, region) in skips {
            if followed[b] || lanes.entering_waves(p, b, region) > 1 {
                continue;
            }
            for &d in self.order {
                if d != b && self.dom(region, d) {
                    followed[d] = true;
                    if forms.contains_key(&d) {
                        forms.insert(d, Ok(Branch::Follow));
                    }
                }
            }
        }
        // Exits of loops the waves may leave apart.
        for &b in self.order {
            let Some(Ok(Branch::Leave { .. })) = forms.get(&b) else {
                continue;
            };
            let nat = self.innermost(b).expect("a loop exit is in a loop");
            let next = divergent.len();
            let lp = *divergent.entry(nat).or_insert(next);
            if let Some(Ok(Branch::Leave { lp: l, .. })) = forms.get_mut(&b) {
                *l = lp;
            }
        }
        let mut order_err = None;
        for &b in self.order {
            if let Some(Err(why)) = forms.get(&b) {
                order_err.get_or_insert(*why);
            }
        }
        if let Some(why) = order_err {
            return Err(why);
        }
        let mut branches: HashMap<usize, Branch> = forms
            .into_iter()
            .map(|(b, f)| (b, f.expect("checked")))
            .collect();
        let mut nats: Vec<usize> = divergent.keys().copied().collect();
        nats.sort_by_key(|&nat| divergent[&nat]);
        let mut loops = Vec::new();
        let mut header_of = HashMap::new();
        let mut exit_here = HashMap::new();
        let mut entered_from = HashMap::new();
        for &nat in &nats {
            let lp = divergent[&nat];
            let l =
                self.divergent_loop(nat, lp, &mut branches, &mut entered_from, &mut exit_here)?;
            header_of.insert(l.header, lp);
            loops.push(l);
        }
        // What each divergent loop holds for after it.
        let mut def_loop = HashMap::new();
        let mut freeze = HashMap::new();
        self.values_after(&mut loops, &nats, &divergent, &mut def_loop, &mut freeze);
        let mut joins: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        let depth = |mut b: usize| {
            let mut d = 0;
            while self.idom[b] != b && self.idom[b] != usize::MAX {
                b = self.idom[b];
                d += 1;
            }
            d
        };
        for &x in self.order {
            let Some(&branch) = branches.get(&x) else {
                continue;
            };
            let (entry, join) = match branch {
                Branch::Uniform | Branch::Follow | Branch::Leave { .. } => continue,
                Branch::Skip { region, join } => (region, Some(join)),
                Branch::Exit { go_on } => (go_on, None),
            };
            if p.blocks[entry].preds != [x] {
                return Err(UNSTRUCTURED);
            }
            for &d in self.order.iter().filter(|&&d| self.dom(entry, d)) {
                for s in p.blocks[d].exit.successors() {
                    let inside = self.dom(entry, s);
                    let ok = match join {
                        Some(j) => inside || s == j,
                        None => inside || bare_end(p, s),
                    };
                    if !ok {
                        return Err(UNSTRUCTURED);
                    }
                }
                if join.is_some() && p.blocks[d].insts.iter().any(|i| i.name() == "s_barrier") {
                    return Err(BARRIER_IN_REGION);
                }
                scalar_loads_together(p, d, same)?;
            }
            entered_from.insert(entry, x);
            if let Some(j) = join {
                if p.blocks[j]
                    .preds
                    .iter()
                    .any(|&q| self.index[q] != usize::MAX && self.index[j] <= self.index[q])
                {
                    return Err(UNSTRUCTURED);
                }
                if !followed[x] {
                    joins.entry(j).or_default().push((x, entry));
                }
            }
        }
        for l in &loops {
            for (d, _) in l.body.iter().enumerate().filter(|(_, i)| **i) {
                scalar_loads_together(p, d, same)?;
            }
        }
        for list in joins.values_mut() {
            list.sort_by_key(|&(x, _)| std::cmp::Reverse(depth(x)));
        }
        Ok(Plan {
            host_waves,
            branches,
            idom: self.idom.to_vec(),
            entered_from,
            joins,
            loops,
            header_of,
            exit_here,
            def_loop,
            freeze,
            broadcast: broadcast.clone(),
        })
    }

    /// The form of the conditional branch ending block `b`, or why it has
    /// none.
    fn classify(
        &self,
        b: usize,
        taken: usize,
        not_taken: usize,
        same: &HashSet<Value>,
        inv: &Invariant,
    ) -> Result<Branch, Why> {
        let p = self.p;
        let Some(last) = p.blocks[b].insts.last() else {
            return Err("a branch without its instruction");
        };
        let name = last.name();
        // A VCC test of a compare of values the same in every lane and
        // wave, all lanes on: all its lanes or none, in every wave.
        let vcc_same = || {
            (0..if p.wave64 { 2 } else { 1 }).all(|k| {
                last.read(Reg::S(VCC + k))
                    .is_some_and(|v| inv.masks.contains(&v))
            })
        };
        let uniform = matches!(name, "s_cbranch_scc0" | "s_cbranch_scc1")
            && last.read(Reg::Scc).is_some_and(|v| same.contains(&v))
            || matches!(name, "s_cbranch_vccz" | "s_cbranch_vccnz") && vcc_same();
        if uniform || taken == not_taken {
            return Ok(Branch::Uniform);
        }
        // An exit of the innermost loop around the branch.
        if let Some(l) = self.innermost(b) {
            let body = &self.loops[l].body;
            match (body[taken], body[not_taken]) {
                (true, false) => return Ok(Branch::Leave { stay: taken, lp: 0 }),
                (false, true) => {
                    return Ok(Branch::Leave {
                        stay: not_taken,
                        lp: 0,
                    });
                }
                _ => {}
            }
        }
        // An end some waves take: no wave that ends takes part in a later
        // barrier, and one that goes on here does nothing.
        if bare_end(p, taken) && !bare_end(p, not_taken) {
            return Ok(Branch::Exit { go_on: not_taken });
        }
        if bare_end(p, not_taken) && !bare_end(p, taken) {
            return Ok(Branch::Exit { go_on: taken });
        }
        let join = self.ipdom[b];
        if join == taken {
            Ok(Branch::Skip {
                region: not_taken,
                join,
            })
        } else if join == not_taken {
            Ok(Branch::Skip {
                region: taken,
                join,
            })
        } else {
            Err(BRANCH_APART)
        }
    }

    /// Checks natural loop `nat` as divergent loop `lp` and finds its
    /// exits and join.
    fn divergent_loop(
        &self,
        nat: usize,
        lp: usize,
        branches: &mut HashMap<usize, Branch>,
        entered_from: &mut HashMap<usize, usize>,
        exit_here: &mut HashMap<usize, usize>,
    ) -> Result<Loop, Why> {
        let p = self.p;
        let l = &self.loops[nat];
        let parent = (0..nat)
            .rev()
            .find(|&k| self.loops[k].body[l.header] && k != nat);
        let mut exits = Vec::new();
        for &b in self.order.iter().filter(|&&b| l.body[b]) {
            if p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier") {
                return Err(BARRIER_IN_LOOP);
            }
            let out: Vec<usize> = p.blocks[b]
                .exit
                .successors()
                .into_iter()
                .filter(|&s| !l.body[s])
                .collect();
            if out.is_empty() {
                continue;
            }
            let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
                return Err(EXIT_OFF_PATH);
            };
            // An exit some waves may take: every one of the loop's exits.
            let stay = if l.body[taken] { taken } else { not_taken };
            if !l.body[stay] {
                return Err(EXIT_OFF_PATH);
            }
            if let Some(k) = parent
                && out.iter().any(|&s| !self.loops[k].body[s])
            {
                return Err(EXIT_TWO_LOOPS);
            }
            if !l.latches.iter().all(|&q| self.dom(b, q)) {
                return Err(EXIT_OFF_PATH);
            }
            if stay != l.header {
                if p.blocks[stay].preds != [b] {
                    return Err(UNSTRUCTURED);
                }
                entered_from.insert(stay, b);
            }
            branches.insert(b, Branch::Leave { stay, lp });
            exits.push(b);
        }
        if exits.is_empty() {
            return Err(NO_EXIT);
        }
        // Each exit's path, past blocks of no effect, to the join.
        let mut join = None;
        let mut ends = Vec::new();
        for &b in &exits {
            let s = p.blocks[b]
                .exit
                .successors()
                .into_iter()
                .find(|&s| !l.body[s])
                .expect("an exit");
            let (mut end, mut at) = (b, s);
            while let Some(t) = no_effect(p, at).filter(|_| p.blocks[at].preds == [end]) {
                exit_here.insert(at, lp);
                (end, at) = (at, t);
            }
            let at = if bare_end(p, at) { usize::MAX } else { at };
            if *join.get_or_insert(at) != at {
                return Err(EXITS_APART);
            }
            ends.push(end);
        }
        let join = join.expect("an exit");
        if join != usize::MAX {
            if self.idom[join] != usize::MAX && l.body[self.idom[join]]
                || exit_here.get(&self.idom[join]) == Some(&lp)
            {
                exit_here.insert(join, lp);
            }
            // The join's phis take a wave's values from the exit it took.
            for &q in &p.blocks[join].preds {
                if self.index[q] != usize::MAX && l.body[q] && !exits.contains(&q) {
                    return Err(EXITS_APART);
                }
            }
        }
        Ok(Loop {
            header: l.header,
            body: l.body.clone(),
            exits,
            ends,
            join,
            held: Vec::new(),
        })
    }

    /// The values each divergent loop holds for after it: those defined in
    /// it (or in loops inside it) that an instruction or a phi outside it
    /// reads.
    fn values_after(
        &self,
        loops: &mut [Loop],
        nats: &[usize],
        divergent: &HashMap<usize, usize>,
        def_loop: &mut HashMap<Value, usize>,
        freeze: &mut HashMap<Value, usize>,
    ) {
        let p = self.p;
        if loops.is_empty() {
            return;
        }
        // Innermost divergent loop of each block: the smallest holding it.
        let inner = |b: usize| -> Option<usize> {
            nats.iter()
                .filter(|&&nat| self.loops[nat].body[b])
                .min_by_key(|&&nat| self.loops[nat].body.iter().filter(|&&x| x).count())
                .map(|nat| divergent[nat])
        };
        // Per value defined in a divergent loop: its block, and whether it
        // is an SCC bit.
        let mut info: HashMap<Value, (usize, bool)> = HashMap::new();
        for &b in self.order {
            let Some(lp) = inner(b) else { continue };
            for phi in &p.blocks[b].phis {
                def_loop.insert(phi.value, lp);
                info.insert(phi.value, (b, phi.reg == Reg::Scc));
            }
            for inst in &p.blocks[b].insts {
                for &(r, v) in &inst.writes {
                    def_loop.insert(v, lp);
                    info.insert(v, (b, r == Reg::Scc));
                }
            }
        }
        // Reads outside the innermost loop of the definition. (EXEC is a
        // lane mask: what the lowering makes of it is checked there.)
        let mut uses = Vec::new();
        for &b in self.order {
            for inst in &p.blocks[b].insts {
                for &(r, v) in &inst.reads {
                    let exec = matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1);
                    if !exec && def_loop.get(&v).is_some_and(|&lp| !loops[lp].body[b]) {
                        uses.push((v, b));
                    }
                }
            }
            for phi in &p.blocks[b].phis {
                for &v in &phi.args {
                    if def_loop.get(&v).is_some_and(|&lp| !loops[lp].body[b]) {
                        uses.push((v, b));
                    }
                }
            }
        }
        for (v, at) in uses {
            let (d, bit) = info[&v];
            freeze.entry(v).or_insert(def_loop[&v]);
            // Every divergent loop around the definition the use is
            // outside of holds it.
            for l in loops.iter_mut() {
                if l.body[d] && !l.body[at] && !l.held.iter().any(|h| h.0 == v) {
                    l.held.push((v, bit));
                }
            }
        }
    }

    /// Whether first-lane read `r` may take the group's first wave's value
    /// under `plan`: the load's descriptor and offset are the same in every
    /// wave, and both instructions run in the group's own flow.
    fn broadcast_holds(&self, plan: &Plan, same: &HashSet<Value>, r: Value) -> bool {
        let p = self.p;
        let Some((b, i)) = find_def(p, r) else {
            return false;
        };
        let src = p.blocks[b].insts[i]
            .reads
            .iter()
            .find(|x| matches!(x.0, Reg::V(_)))
            .map(|x| x.1);
        let Some((lb, li)) = src.and_then(|v| find_def(p, v)) else {
            return false;
        };
        let load = &p.blocks[lb].insts[li];
        let Fields::Buffer { rsrc, soffset, .. } = &load.fields else {
            return false;
        };
        let reads_same = |o: &Operand, dwords: u8| match *o {
            Operand::Scalar { code, .. } if code == crate::gfx10::NULL => true,
            Operand::Scalar { code, .. } => (0..dwords).all(|k| {
                load.read(Reg::S(code + k))
                    .is_some_and(|v| same.contains(&v))
            }),
            Operand::Const(_) | Operand::Literal(_) => true,
            _ => false,
        };
        reads_same(rsrc, 4)
            && reads_same(soffset, 1)
            && self.own_flow(plan, b)
            && self.own_flow(plan, lb)
    }
}

fn find_def(p: &Program, v: Value) -> Option<(usize, usize)> {
    p.blocks.iter().enumerate().find_map(|(b, block)| {
        block
            .insts
            .iter()
            .position(|i| i.writes.iter().any(|w| w.1 == v))
            .map(|i| (b, i))
    })
}

/// In code a wave runs because others do: no scalar load at an address
/// that may differ between the waves, no scalar store.
fn scalar_loads_together(p: &Program, d: usize, same: &HashSet<Value>) -> Result<(), Why> {
    for inst in &p.blocks[d].insts {
        if !matches!(inst.fields, Fields::Smem { .. }) {
            continue;
        }
        if inst.reads.iter().any(|r| !same.contains(&r.1)) {
            return Err(SCALAR_LOAD_APART);
        }
        // Scalar stores would take effect in every wave.
        if !(inst.name().starts_with("s_load") || inst.name().starts_with("s_buffer_load")) {
            return Err("a scalar memory write in a group of several console waves");
        }
    }
    Ok(())
}

/// The first-lane reads of a buffer load at one address for every lane and
/// wave, all lanes on at both, with no store that may come before the load.
fn broadcast_candidates(p: &Program, order: &[usize], lanes: &Lanes) -> HashSet<Value> {
    let mut out = HashSet::new();
    if !p.wave64 {
        return out;
    }
    for &b in order {
        for inst in &p.blocks[b].insts {
            if inst.name() != "v_readfirstlane_b32" {
                continue;
            }
            let Some(&(_, r)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_))) else {
                continue;
            };
            let Some(src) = inst
                .reads
                .iter()
                .find(|x| matches!(x.0, Reg::V(_)))
                .map(|x| x.1)
            else {
                continue;
            };
            let Some((lb, li)) = find_def(p, src) else {
                continue;
            };
            let load = &p.blocks[lb].insts[li];
            let fixed = matches!(
                load.fields,
                Fields::Buffer {
                    offen: false,
                    idxen: false,
                    lds: false,
                    tfe: false,
                    ..
                }
            ) && load.name().starts_with("buffer_load_");
            if !fixed || !lanes.all_on(inst) || !lanes.all_on(load) {
                continue;
            }
            if !no_store_before(p, lb, li) {
                continue;
            }
            out.insert(r);
        }
    }
    out
}

/// Whether no store or atomic can run before instruction `i` of block `b`
/// (in any block from which `b` is reached, or earlier in `b`).
fn no_store_before(p: &Program, b: usize, i: usize) -> bool {
    if p.blocks[b].insts[..i].iter().any(is_store) {
        return false;
    }
    let mut seen = vec![false; p.blocks.len()];
    let mut stack: Vec<usize> = p.blocks[b].preds.clone();
    while let Some(x) = stack.pop() {
        if std::mem::replace(&mut seen[x], true) {
            continue;
        }
        if p.blocks[x].insts.iter().any(is_store) {
            return false;
        }
        stack.extend(p.blocks[x].preds.iter().copied());
    }
    true
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
/// `split_joins` (where the waves that skipped a region, or left a loop at
/// another iteration, take their own value) and `split_values` (used after
/// such a loop). `fixed` are first-lane reads the group makes the same.
fn uniform_values(
    p: &Program,
    inputs: &HashSet<Value>,
    fixed: &HashSet<Value>,
    split_joins: &HashSet<usize>,
    split_values: &HashSet<Value>,
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
    for v in split_values {
        same.remove(v);
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
                let writes_same = inst
                    .writes
                    .iter()
                    .any(|w| same.contains(&w.1) && !fixed.contains(&w.1));
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

// ---- values the same in every lane of every console wave ----

/// Values the same in every lane of every console wave of the group, and
/// where each is defined: VGPRs, compare results (all lanes or none), and
/// first-lane reads of such VGPRs.
#[derive(Debug, Default)]
struct Invariant {
    vgprs: HashSet<Value>,
    masks: HashSet<Value>,
    reads: HashSet<Value>,
    at: HashMap<Value, usize>,
}

/// Pure lane arithmetic: the same operands give the same result in every
/// lane.
const PURE: &[&str] = &[
    "v_mov_b32",
    "v_add_nc_u32",
    "v_sub_nc_u32",
    "v_subrev_nc_u32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_not_b32",
    "v_lshlrev_b32",
    "v_lshrrev_b32",
    "v_ashrrev_i32",
    "v_ffbh_u32",
    "v_ffbl_b32",
    "v_ffbh_i32",
    "v_cndmask_b32",
    "v_min_u32",
    "v_max_u32",
    "v_min_i32",
    "v_max_i32",
    "v_mul_lo_u32",
    "v_bfe_u32",
    "v_lshl_add_u32",
    "v_add_lshl_u32",
    "v_add3_u32",
    "v_lshl_or_b32",
    "v_and_or_b32",
    "v_or3_b32",
];

/// The inline constant or literal an operand names.
fn constant(o: &Operand) -> Option<u32> {
    match *o {
        Operand::Const(c @ 128..=192) => Some(u32::from(c - 128)),
        Operand::Const(c @ 193..=208) => Some((-i32::from(c - 192)) as u32),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// [`Invariant`] of a program, given the scalar values `same` in every
/// wave and leaving out what blocks `excluded` define. Its sources: inline
/// constants and literals, scalars in `same`, and LDS dwords at a constant
/// address. All lanes are on at each definition, and (checked against the
/// plan, [`plan`]) each runs in the group's own flow. An LDS read there is
/// one point of the group's LDS order: the console flow in halves puts a
/// group barrier before every LDS access (`Lower::lds`), which every
/// thread meets, so every lane of every wave reads the dword as all
/// accesses before it left it and none after.
fn invariant(
    p: &Program,
    order: &[usize],
    same: &HashSet<Value>,
    lanes: &Lanes,
    excluded: &HashSet<usize>,
) -> Invariant {
    let mut inv = Invariant::default();
    // A VGPR set to a constant: the address of an LDS access.
    let mut konst: HashSet<Value> = HashSet::new();
    for block in &p.blocks {
        for inst in &block.insts {
            if let Fields::Valu(f) = &inst.fields
                && inst.name() == "v_mov_b32"
                && f.sdwa.is_none()
                && f.dpp.is_none()
                && f.src.first().and_then(constant).is_some()
                && let Some(&(Reg::V(_), v)) = inst.writes.first()
            {
                konst.insert(v);
            }
        }
    }
    let operand = |inv: &Invariant, inst: &Inst, o: &Operand| match *o {
        Operand::Const(_) | Operand::Literal(_) => true,
        Operand::Scalar { code, .. } if code == crate::gfx10::NULL => true,
        Operand::Scalar { code, dwords } => (0..dwords.max(1)).all(|k| {
            inst.read(Reg::S(code + k))
                .is_some_and(|v| same.contains(&v) || inv.masks.contains(&v))
        }),
        Operand::Vector { reg, dwords } => (0..dwords.max(1)).all(|k| {
            inst.read(Reg::V(reg + k))
                .is_some_and(|v| inv.vgprs.contains(&v))
        }),
        _ => false,
    };
    loop {
        let mut changed = false;
        for &b in order.iter().filter(|b| !excluded.contains(b)) {
            for inst in &p.blocks[b].insts {
                if !lanes.all_on(inst) {
                    continue;
                }
                let name = inst.name();
                let (set, regs): (u8, Vec<Value>) = match &inst.fields {
                    Fields::Valu(f) => {
                        let plain = f.sdwa.is_none()
                            && f.dpp.is_none()
                            && f.neg == 0
                            && f.abs == 0
                            && f.omod == 0
                            && !f.clamp
                            && f.carry.is_none();
                        if !plain || !f.src.iter().all(|o| operand(&inv, inst, o)) {
                            continue;
                        }
                        let s = |r: &Reg| matches!(r, Reg::S(_));
                        let writes = |want: fn(&Reg) -> bool| {
                            inst.writes
                                .iter()
                                .filter(|w| want(&w.0))
                                .map(|w| w.1)
                                .collect()
                        };
                        if name == "v_readfirstlane_b32" {
                            (2, writes(|r| matches!(r, Reg::S(_))))
                        } else if name.starts_with("v_cmp_") {
                            (
                                1,
                                inst.writes
                                    .iter()
                                    .filter(|w| s(&w.0))
                                    .map(|w| w.1)
                                    .collect(),
                            )
                        } else if PURE.contains(&name)
                            && inst.writes.iter().all(|w| matches!(w.0, Reg::V(_)))
                        {
                            (0, writes(|r| matches!(r, Reg::V(_))))
                        } else {
                            continue;
                        }
                    }
                    Fields::Ds {
                        addr: Some(Operand::Vector { reg, .. }),
                        gds: false,
                        ..
                    } if name == "ds_read_b32" => {
                        if !inst.read(Reg::V(*reg)).is_some_and(|v| konst.contains(&v)) {
                            continue;
                        }
                        (
                            0,
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1)
                                .collect(),
                        )
                    }
                    _ => continue,
                };
                for v in regs {
                    let dest = match set {
                        0 => &mut inv.vgprs,
                        1 => &mut inv.masks,
                        _ => &mut inv.reads,
                    };
                    if dest.insert(v) {
                        inv.at.insert(v, b);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return inv;
        }
    }
}

// ---- which console waves a lane mask can hold lanes of ----

type Bits = Vec<u64>;

/// Bounds of lane masks as sets of the group's threads (from the program's
/// EXEC and compares of the thread id with constants): a mask dword holds
/// at least `lower` and at most `upper`.
struct Lanes<'a> {
    p: &'a Program,
    threads: u32,
    tid: Option<Value>,
    launch: [Option<Value>; 2],
    def: HashMap<Value, (usize, usize)>,
    /// Values a phi or an instruction defines.
    defined: HashSet<Value>,
    /// The bounds found so far (a fixpoint, from "no value yet").
    est: HashMap<Value, (Bits, Bits)>,
}

impl<'a> Lanes<'a> {
    fn new(p: &'a Program, threads: u32, tid: Option<Value>) -> Self {
        let launch =
            [EXEC, EXEC + 1].map(|r| p.inputs.iter().find(|i| i.0 == Reg::S(r)).map(|i| i.1));
        let mut def = HashMap::new();
        let mut defined = HashSet::new();
        for (b, block) in p.blocks.iter().enumerate() {
            defined.extend(block.phis.iter().map(|phi| phi.value));
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    def.insert(v, (b, i));
                    defined.insert(v);
                }
            }
        }
        let mut l = Lanes {
            p,
            threads,
            tid,
            launch,
            def,
            defined,
            est: HashMap::new(),
        };
        l.solve();
        l
    }

    /// The bounds of every scalar dword, by iteration from "no value yet"
    /// (lower: every lane, upper: none) up: each step is monotone (more
    /// possible values never narrow a bound), so it ends, at bounds that
    /// hold for every value the program can make.
    fn solve(&mut self) {
        let p = self.p;
        loop {
            let mut changed = false;
            for block in &p.blocks {
                for phi in &block.phis {
                    let Reg::S(c) = phi.reg else { continue };
                    let k = c as usize & 1;
                    let mut lo = self.universe(k);
                    let mut hi = vec![0; self.words()];
                    for &a in &phi.args {
                        let (al, ah) = self.of(a, k);
                        for w in 0..lo.len() {
                            lo[w] &= al[w];
                            hi[w] |= ah[w];
                        }
                    }
                    changed |=
                        self.est.insert(phi.value, (lo.clone(), hi.clone())) != Some((lo, hi));
                }
                for inst in &block.insts {
                    for &(r, v) in &inst.writes {
                        let Reg::S(code) = r else { continue };
                        let k = code as usize & 1;
                        let x = self
                            .compute(v, k)
                            .unwrap_or_else(|| (vec![0; self.words()], self.universe(k)));
                        changed |= self.est.insert(v, x.clone()) != Some(x);
                    }
                }
            }
            if !changed {
                return;
            }
        }
    }

    fn words(&self) -> usize {
        (self.threads as usize).div_ceil(64)
    }

    /// The threads mask dword `k` (0: lanes 0-31, 1: 32-63) can hold.
    fn universe(&self, k: usize) -> Bits {
        self.set_of(|t| (t % 64) / 32 == k as u32)
    }

    fn set_of(&self, f: impl Fn(u32) -> bool) -> Bits {
        let mut s = vec![0; self.words()];
        for t in 0..self.threads {
            if f(t) {
                s[t as usize / 64] |= 1 << (t % 64);
            }
        }
        s
    }

    /// Whether EXEC holds every lane at `inst` (the program's own, or a
    /// copy of it).
    fn all_on(&self, inst: &Inst) -> bool {
        (0..2).all(|k| {
            inst.read(Reg::S(EXEC + k as u8)).is_some_and(|v| {
                let (lo, _) = self.of(v, k);
                lo == self.universe(k)
            })
        })
    }

    /// How many console waves can go into the region of the branch ending
    /// block `b` (its successor `region`).
    fn entering_waves(&self, p: &Program, b: usize, region: usize) -> usize {
        let waves = self.threads.div_ceil(64) as usize;
        let Some(last) = p.blocks[b].insts.last() else {
            return waves;
        };
        let Exit::Branch { taken, .. } = p.blocks[b].exit else {
            return waves;
        };
        let (reg, zero_taken) = match last.name() {
            "s_cbranch_execz" => (EXEC, true),
            "s_cbranch_execnz" => (EXEC, false),
            "s_cbranch_vccz" => (VCC, true),
            "s_cbranch_vccnz" => (VCC, false),
            _ => return waves,
        };
        // Into the region where the mask is not empty, or where it is.
        let into_nonempty = (region == taken) != zero_taken;
        let mut lower = vec![0; self.words()];
        let mut upper = vec![0; self.words()];
        for k in 0..2 {
            let Some(v) = last.read(Reg::S(reg + k as u8)) else {
                return waves;
            };
            let (lo, hi) = self.of(v, k);
            for w in 0..lower.len() {
                lower[w] |= lo[w];
                upper[w] |= hi[w];
            }
        }
        (0..waves)
            .filter(|&w| {
                if into_nonempty {
                    upper[w] != 0
                } else {
                    lower[w] == 0
                }
            })
            .count()
    }

    /// Bounds of value `v`, a mask dword `k`: the program's EXEC exactly,
    /// one not found yet nothing (during [`Self::solve`]), an input or what
    /// no rule covers anything.
    fn of(&self, v: Value, k: usize) -> (Bits, Bits) {
        if self.launch[k] == Some(v) {
            let u = self.universe(k);
            return (u.clone(), u);
        }
        match self.est.get(&v) {
            Some(x) => x.clone(),
            None if self.defined.contains(&v) => (self.universe(k), vec![0; self.words()]),
            None => (vec![0; self.words()], self.universe(k)),
        }
    }

    fn compute(&self, v: Value, k: usize) -> Option<(Bits, Bits)> {
        let p = self.p;
        let &(b, i) = self.def.get(&v)?;
        let inst = &p.blocks[b].insts[i];
        let &(Reg::S(code), _) = inst.writes.iter().find(|w| w.1 == v)? else {
            return None;
        };
        if code as usize & 1 != k {
            return None;
        }
        let name = inst.name();
        let u = self.universe(k);
        let not = |x: &Bits| -> Bits { x.iter().zip(&u).map(|(a, m)| !a & m).collect() };
        let and = |x: &Bits, y: &Bits| -> Bits { x.iter().zip(y).map(|(a, b)| a & b).collect() };
        let or = |x: &Bits, y: &Bits| -> Bits { x.iter().zip(y).map(|(a, b)| a | b).collect() };
        // (lower, upper) of a combination, from both operands'.
        let and2 = |a: (Bits, Bits), b: (Bits, Bits)| (and(&a.0, &b.0), and(&a.1, &b.1));
        let or2 = |a: (Bits, Bits), b: (Bits, Bits)| (or(&a.0, &b.0), or(&a.1, &b.1));
        let not1 = |a: (Bits, Bits)| (not(&a.1), not(&a.0));
        if name.starts_with("v_cmp") {
            let exec = self.of(inst.read(Reg::S(EXEC + k as u8))?, k);
            let Fields::Valu(f) = &inst.fields else {
                return None;
            };
            // (Operand modifiers would change what is compared.)
            let plain = f.sdwa.is_none() && f.dpp.is_none() && f.neg == 0 && f.abs == 0;
            let cmp = self.compare(inst, name, &f.src, k).filter(|_| plain);
            return Some(match cmp {
                Some(c) => and2((c.clone(), c), exec),
                None => (vec![0; self.words()], exec.1),
            });
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let operand =
            |l: &Self, n: usize| -> Option<(Bits, Bits)> { l.operand(inst, src.get(n)?, k) };
        let exec_old = |l: &Self| -> Option<(Bits, Bits)> {
            Some(l.of(inst.read(Reg::S(EXEC + k as u8))?, k))
        };
        let writes_exec = code == EXEC + k as u8;
        Some(match name {
            "s_mov_b64" => operand(self, 0)?,
            "s_and_b64" => and2(operand(self, 0)?, operand(self, 1)?),
            "s_or_b64" => or2(operand(self, 0)?, operand(self, 1)?),
            "s_andn2_b64" | "s_andn2_wrexec_b64" => {
                and2(operand(self, 0)?, not1(operand(self, 1)?))
            }
            "s_andn1_wrexec_b64" => and2(not1(operand(self, 0)?), operand(self, 1)?),
            "s_not_b64" => not1(operand(self, 0)?),
            "s_and_saveexec_b64"
            | "s_or_saveexec_b64"
            | "s_andn2_saveexec_b64"
            | "s_andn1_saveexec_b64" => {
                let old = exec_old(self)?;
                if !writes_exec {
                    old
                } else {
                    let s = operand(self, 0)?;
                    match name {
                        "s_and_saveexec_b64" => and2(s, old),
                        "s_or_saveexec_b64" => or2(s, old),
                        "s_andn2_saveexec_b64" => and2(s, not1(old)),
                        _ => and2(not1(s), old),
                    }
                }
            }
            _ => return None,
        })
    }

    /// Dword `k` of a 64-bit scalar operand.
    fn operand(&self, inst: &Inst, o: &Operand, k: usize) -> Option<(Bits, Bits)> {
        match *o {
            Operand::Scalar { code, .. } => {
                let r = code.checked_add(k as u8)?;
                if r as usize & 1 != k {
                    return None;
                }
                let v = inst.read(Reg::S(r))?;
                Some(self.of(v, k))
            }
            Operand::Const(c) => {
                let x: i64 = match c {
                    128..=192 => i64::from(c - 128),
                    193..=208 => -i64::from(c - 192),
                    _ => return None,
                };
                let word = (x >> (32 * k)) as u32;
                let s = self.set_of(|t| (t % 64) / 32 == k as u32 && word >> (t % 32) & 1 != 0);
                Some((s.clone(), s))
            }
            _ => None,
        }
    }

    /// The exact lanes of an integer compare of the thread id with a
    /// constant, in mask dword `k`.
    fn compare(&self, inst: &Inst, name: &str, src: &[Operand], k: usize) -> Option<Bits> {
        let rest = name
            .strip_prefix("v_cmpx_")
            .or_else(|| name.strip_prefix("v_cmp_"))?;
        let (op, ty) = rest.split_once('_')?;
        let signed = match ty {
            "u32" => false,
            "i32" => true,
            _ => return None,
        };
        let tid = self.tid?;
        let value = |o: &Operand| -> Option<Result<i64, ()>> {
            match *o {
                Operand::Vector { reg, .. } if inst.read(Reg::V(reg)) == Some(tid) => Some(Err(())),
                Operand::Const(c @ 128..=192) => Some(Ok(i64::from(c - 128))),
                Operand::Const(c @ 193..=208) => Some(Ok(-i64::from(c - 192))),
                Operand::Literal(x) => Some(Ok(i64::from(x))),
                _ => None,
            }
        };
        let (a, b) = (value(src.first()?)?, value(src.get(1)?)?);
        let as_ty = |x: i64| {
            if signed {
                i64::from(x as i32)
            } else {
                i64::from(x as u32)
            }
        };
        let cmp = |x: i64, y: i64| match op {
            "eq" => Some(x == y),
            "ne" | "lg" => Some(x != y),
            "lt" => Some(x < y),
            "le" => Some(x <= y),
            "gt" => Some(x > y),
            "ge" => Some(x >= y),
            _ => None,
        };
        cmp(0, 0)?;
        let at = |t: u32, o: Result<i64, ()>| as_ty(o.unwrap_or(i64::from(t)));
        Some(self.set_of(|t| (t % 64) / 32 == k as u32 && cmp(at(t, a), at(t, b)).unwrap_or(false)))
    }
}

// ---- lowering ----

/// A phi made during lowering and the incoming values gathered for it.
type Pending = (V, Vec<(V, usize)>);

/// What a group of several console waves keeps while it lowers.
#[derive(Debug, Default)]
pub(crate) struct State {
    /// Per divergent loop: its "alive" phi (the waves still in it) and the
    /// incoming values gathered for it.
    alive: HashMap<usize, Pending>,
    /// Per divergent loop: what `here` held at its entry.
    entry_here: HashMap<usize, Option<V>>,
    /// Per (divergent loop, value): the phi holding the value for the waves
    /// that left the loop.
    held: HashMap<(usize, Value), Pending>,
    /// Per (divergent loop, exit): whether the wave left by it.
    flags: HashMap<(usize, usize), Pending>,
    /// Per exiting block: the waves leaving there now.
    leaving: HashMap<usize, V>,
    /// Per value used after a divergent loop: the value, held where the
    /// wave has left.
    frozen: HashMap<Value, Val>,
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

    /// The group's way at block `b`'s branch, from `taken`, this console
    /// wave's own test: toward the side `along_taken` says the taken edge
    /// leads to. A wave votes for its own way where it runs `b` for itself;
    /// the group goes that way if some wave votes for it.
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

    /// The terminator of block `b`'s conditional branch, `cond` this
    /// console wave's own test, in the group's form ([`Branch`]).
    pub(super) fn group_branch(
        &mut self,
        b: usize,
        cond: V,
        taken: usize,
        not_taken: usize,
    ) -> Term {
        let form = self
            .groups
            .as_ref()
            .and_then(|g| g.branches.get(&b))
            .copied();
        match form {
            // The group's vote: in if some wave goes in, on if some wave
            // goes on, the way the one wave that can be here goes.
            Some(Branch::Skip { region, join }) => {
                let g = self.group_vote(b, cond, region == taken);
                self.branch_term(b, g, region, join)
            }
            Some(Branch::Exit { go_on }) => {
                let end = if go_on == taken { not_taken } else { taken };
                let g = self.group_vote(b, cond, go_on == taken);
                self.branch_term(b, g, go_on, end)
            }
            Some(Branch::Follow) => {
                let g = self.group_vote(b, cond, true);
                self.branch_term(b, g, taken, not_taken)
            }
            Some(Branch::Leave { stay, .. }) => {
                let out = if stay == taken { not_taken } else { taken };
                let g = self.group_vote(b, cond, stay == taken);
                let c = self.cur;
                let (own, _) = self.group_votes[&b];
                let t = self.s.m.bool(true);
                let gone = self.s.m.bin(c, Bin::Xor, own, t);
                let leaving = match self.here[b] {
                    Some(h) => self.s.m.bin(c, Bin::And, gone, h),
                    None => gone,
                };
                self.gs.leaving.insert(b, leaving);
                self.branch_term(b, g, stay, out)
            }
            _ => self.branch_term(b, cond, taken, not_taken),
        }
    }

    /// At the start of IR block `b`: for which console waves it runs. A
    /// divergent loop's header: its "alive" phi, and the phis holding
    /// values and exits for the waves that left. After the loop: what held
    /// at its entry. A block a branch's vote goes into: the waves that
    /// voted. Any other block: as its dominator.
    pub(super) fn group_enter(&mut self, b: usize) {
        if !self.console_cf {
            return;
        }
        let Some(g) = &self.groups else { return };
        let dom = self.group_base(b);
        let here = if let Some(&lp) = g.header_of.get(&b) {
            self.gs.entry_here.insert(lp, dom);
            Some(self.loop_phis(lp))
        } else if let Some(x) = g.entered_from.get(&b) {
            self.group_votes.get(x).map(|v| v.1)
        } else {
            dom
        };
        self.here[b] = here;
    }

    /// Divergent loop `lp`'s phis in its header, made on first use (the
    /// edges into the header are lowered before it): its "alive" bit, the
    /// values it holds, and with several exits which one a wave took.
    fn loop_phis(&mut self, lp: usize) -> V {
        if let Some(&(alive, _)) = self.gs.alive.get(&lp) {
            return alive;
        }
        let g = self.groups.as_ref().expect("set");
        let l = &g.loops[lp];
        let (held, exits) = (l.held.clone(), l.exits.len());
        let blk = self.block[l.header].expect("reachable");
        let (i1, i32t) = (self.s.t.i1, self.s.t.i32);
        let alive = self.s.m.phi(blk, i1);
        self.gs.alive.insert(lp, (alive, Vec::new()));
        for (v, bit) in held {
            let h = self.s.m.phi(blk, if bit { i1 } else { i32t });
            self.gs.held.insert((lp, v), (h, Vec::new()));
        }
        // (One exit: every wave that left took it.)
        for e in (0..exits).filter(|_| exits > 1) {
            let f = self.s.m.phi(blk, i1);
            self.gs.flags.insert((lp, e), (f, Vec::new()));
        }
        alive
    }

    /// For which console waves IR block `b` runs before its own branch's
    /// vote or loop: after a divergent loop it is after (its immediate
    /// dominator in the loop), what held at that loop's entry; else as its
    /// immediate dominator.
    fn group_base(&self, b: usize) -> Option<V> {
        let g = self.groups.as_ref()?;
        match g.exit_here.get(&b) {
            Some(lp) => self.gs.entry_here.get(lp).copied().flatten(),
            None => match g.idom[b] {
                d if d == b || d == usize::MAX => None,
                d => self.here[d],
            },
        }
    }

    /// An edge into loop header `to` (IR), from DXIL block `via`: forward
    /// from IR block `from` (`back` false) or a back edge from the latch
    /// `from`. A divergent loop's phis take what the edge brings.
    pub(super) fn group_edge(&mut self, from: usize, to: usize, via: usize, back: bool) {
        let Some(g) = &self.groups else { return };
        let Some(&lp) = g.header_of.get(&to) else {
            return;
        };
        let l = &g.loops[lp];
        let (exits, held) = (l.exits.clone(), l.held.clone());
        // The waves coming in: those the header holds for before its loop;
        // going round again, at a latch that is an exit, those that vote to
        // stay.
        let alive = match (back, g.branches.get(&from)) {
            (false, _) => self.group_base(to),
            (true, Some(Branch::Leave { stay, .. })) if *stay == to => {
                self.group_votes.get(&from).map(|v| v.1).or(self.here[from])
            }
            (true, _) => self.here[from],
        };
        let alive = alive.unwrap_or_else(|| self.s.m.bool(true));
        self.loop_phis(lp);
        self.gs
            .alive
            .get_mut(&lp)
            .expect("header")
            .1
            .push((alive, via));
        // The enclosing divergent loop's held value, or nothing yet.
        let outer = |l: &Self, v: Value| {
            let g = l.groups.as_ref().expect("set");
            g.loops
                .iter()
                .enumerate()
                .filter(|&(k, o)| k != lp && o.body[to] && o.held.iter().any(|h| h.0 == v))
                .max_by_key(|&(_, o)| std::cmp::Reverse(o.body.iter().filter(|&&x| x).count()))
                .map(|(k, _)| l.gs.held[&(k, v)].0)
        };
        for (v, bit) in held {
            let x = if back {
                match self.gs.frozen.get(&v) {
                    Some(Val::I32(x) | Val::Bit(x)) => *x,
                    _ => self.gs.held[&(lp, v)].0,
                }
            } else {
                match outer(self, v) {
                    Some(h) => h,
                    None if bit => self.s.m.bool(false),
                    None => self.s.m.i32(0),
                }
            };
            self.gs
                .held
                .get_mut(&(lp, v))
                .expect("held")
                .1
                .push((x, via));
        }
        for (e, &x) in exits.iter().enumerate().filter(|_| exits.len() > 1) {
            let f = if back {
                let (old, _) = self.gs.flags[&(lp, e)];
                match self.gs.leaving.get(&x) {
                    Some(&now) => self.s.m.bin(via, Bin::Or, old, now),
                    None => old,
                }
            } else {
                self.s.m.bool(false)
            };
            self.gs
                .flags
                .get_mut(&(lp, e))
                .expect("flag")
                .1
                .push((f, via));
        }
    }

    /// At the end of IR block `b` (before its exit): the values it defines
    /// that are used after a divergent loop, held where the wave has left.
    /// (A value with no data form, a lane mask, is not: a read of it after
    /// the loop stops there, [`Self::group_value`].)
    pub(super) fn group_leave(&mut self, b: usize) {
        if !self.console_cf {
            return;
        }
        let Some(g) = &self.groups else { return };
        if g.freeze.is_empty() {
            return;
        }
        let p = self.x.p;
        let defined = p.blocks[b].phis.iter().map(|phi| phi.value).chain(
            p.blocks[b]
                .insts
                .iter()
                .flat_map(|i| i.writes.iter().map(|w| w.1)),
        );
        let list: Vec<(Value, usize)> = defined
            .filter_map(|v| g.freeze.get(&v).map(|&lp| (v, lp)))
            .collect();
        let c = self.cur;
        for (v, lp) in list {
            let Some(&(h, _)) = self.gs.held.get(&(lp, v)) else {
                continue;
            };
            let bit = self.s.m.type_of(h) == self.s.t.i1;
            let here = match self.here[b] {
                Some(h) => h,
                None => self.s.m.bool(true),
            };
            let held = match self.vals.get(&v) {
                Some(&Val::I32(x)) if !bit => Val::I32(self.s.m.select(c, here, x, h)),
                Some(&Val::Bit(x)) if bit => Val::Bit(self.s.m.select(c, here, x, h)),
                _ => continue,
            };
            self.gs.frozen.insert(v, held);
        }
    }

    /// A value read while lowering IR block `cur_ir`: the held one where
    /// it comes from a divergent loop the block is after.
    pub(super) fn group_value(&self, v: Value) -> Result<Option<Val>, Why> {
        let Some(g) = &self.groups else {
            return Ok(None);
        };
        let Some(&lp) = g.def_loop.get(&v) else {
            return Ok(None);
        };
        if self.cur_ir == usize::MAX || g.loops[lp].body[self.cur_ir] {
            return Ok(None);
        }
        match self.gs.frozen.get(&v) {
            Some(&x) => Ok(Some(x)),
            None => Err(VALUE_OUT_OF_LOOP),
        }
    }

    /// A lane mask's atom, defined by `v`, read while lowering `cur_ir`:
    /// none from a divergent loop the block is after.
    pub(super) fn group_mask(&self, v: Value) -> Result<(), Why> {
        let Some(g) = &self.groups else {
            return Ok(());
        };
        match g.def_loop.get(&v) {
            Some(&lp) if self.cur_ir != usize::MAX && !g.loops[lp].body[self.cur_ir] => {
                Err(MASK_OUT_OF_LOOP)
            }
            _ => Ok(()),
        }
    }

    /// A first-lane read the group takes from its first console wave
    /// ([`Plan::broadcast`]): every host wave puts its own in its slot, and
    /// after a group barrier all read host wave 0's.
    pub(super) fn group_first(&mut self, inst: &Inst, r: V) -> V {
        let wanted = self
            .groups
            .as_ref()
            .is_some_and(|g| inst.writes.iter().any(|w| g.broadcast.contains(&w.1)));
        if !wanted || !self.console_cf {
            return r;
        }
        let (c, void) = (self.cur, self.s.t.void);
        let slots = self.xchg_slots();
        let hw = self.host_wave();
        let mine = self.s.m.element(c, slots, hw);
        self.s.m.store(c, mine, r);
        let mode = self.s.m.i32(1 | 8);
        self.s
            .op(c, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let z = self.s.m.i32(0);
        let first = self.s.m.element(c, slots, z);
        let x = self.s.m.load(c, first);
        self.s
            .op(c, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        x
    }

    /// In a group of several console waves: the incoming values of phi `i`
    /// of IR block `b` (`(b, i)`) (`mask`: a lane-mask phi; `used`: one some lane
    /// reads), one per pred in `preds`.
    /// Where `b` joins regions some console waves did not go into
    /// ([`Branch::Skip`]), a region's edge carries, in a wave that did not
    /// go in, the value of the edge that skips it, `bypass` (valid in the
    /// skipping block), innermost region first. Where `b` is the join of a
    /// divergent loop, an exit's edge carries, in each wave, the value of
    /// the exit it took, held since.
    pub(super) fn group_fix(
        &mut self,
        (b, i): (usize, usize),
        mask: bool,
        used: bool,
        preds: &[usize],
        bypass: &[Option<V>],
        mut incoming: Vec<(V, usize)>,
    ) -> Result<Vec<(V, usize)>, Why> {
        if !self.console_cf {
            return Ok(incoming);
        }
        let Some(g) = &self.groups else {
            return Ok(incoming);
        };
        let p = self.x.p;
        let phi = &p.blocks[b].phis[i];
        let arg_of = |q: usize| {
            p.blocks[b]
                .preds
                .iter()
                .position(|&x| x == q)
                .map(|n| phi.args[n])
        };
        // (A phi no lane reads takes any value.)
        let looped: Vec<usize> = (0..g.loops.len())
            .filter(|&l| used && g.loops[l].join == b)
            .collect();
        for lp in looped {
            let l = self.groups.as_ref().expect("set").loops[lp].clone();
            // Whether the argument the phi takes from exit `x`'s path comes
            // from inside the loop.
            let inside = |s: &Self, a: Value| s.def_block.get(&a).is_some_and(|&d| l.body[d]);
            for (k, &q) in preds.iter().enumerate() {
                let Some(e) = l.ends.iter().position(|&x| x == q) else {
                    continue;
                };
                let args: Vec<Value> = l
                    .ends
                    .iter()
                    .map(|&x| arg_of(x).ok_or(EXITS_APART))
                    .collect::<Result<_, _>>()?;
                if mask {
                    if args.iter().any(|&a| inside(self, a)) {
                        return Err(MASK_OUT_OF_LOOP);
                    }
                    continue;
                }
                let (own, qb) = incoming[k];
                // The value of exit `x`'s argument for the waves that took
                // it, valid on exit `e`'s edge: as computed for this edge,
                // from before the loop, held as the wave left (an exit no
                // later than `e` in the iteration), or held in the header.
                let value = |s: &mut Self, x: usize| -> Result<V, Why> {
                    let a = args[x];
                    if !inside(s, a) {
                        return match s.vals.get(&a) {
                            _ if x == e => Ok(own),
                            Some(&Val::I32(v) | &Val::Bit(v)) => Ok(v),
                            _ => Err(VALUE_OUT_OF_LOOP),
                        };
                    }
                    if x <= e {
                        match s.gs.frozen.get(&a) {
                            Some(&Val::I32(v) | &Val::Bit(v)) => Ok(v),
                            _ => Err(VALUE_OUT_OF_LOOP),
                        }
                    } else {
                        s.gs.held
                            .get(&(lp, a))
                            .map(|h| h.0)
                            .ok_or(VALUE_OUT_OF_LOOP)
                    }
                };
                let mut chain = value(self, e)?;
                for x in (0..l.exits.len()).rev().filter(|&x| x != e) {
                    let (old, _) = self.gs.flags[&(lp, x)];
                    let took = match self.gs.leaving.get(&l.exits[x]) {
                        Some(&now) if x < e => self.s.m.bin(qb, Bin::Or, old, now),
                        _ => old,
                    };
                    let there = value(self, x)?;
                    chain = self.s.m.select(qb, took, there, chain);
                }
                incoming[k].0 = chain;
            }
        }
        let Some(list) = self.groups.as_ref().and_then(|g| g.joins.get(&b)).cloned() else {
            return Ok(incoming);
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
        Ok(incoming)
    }

    /// Fills the divergent loops' phis.
    pub(super) fn group_phis(&mut self) {
        let alive = std::mem::take(&mut self.gs.alive);
        let held = std::mem::take(&mut self.gs.held);
        let flags = std::mem::take(&mut self.gs.flags);
        for (phi, incoming) in alive
            .into_values()
            .chain(held.into_values())
            .chain(flags.into_values())
        {
            self.s.m.set_phi(phi, incoming);
        }
    }
}

#[cfg(test)]
mod tests {
    //! The forms [`plan`] finds, on hand-assembled GFX10 programs of two
    //! console waves (128 threads): which branches the group votes on,
    //! follows or takes as they are, which loops it runs while some wave is
    //! still in them and what it holds for after them, and what stops.

    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn ir(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::ir::build(&code, true, 256).unwrap()
    }

    /// The plan for 128 threads, user SGPR 0 the same in every wave.
    fn plan_of(p: &Program) -> Result<Plan, Why> {
        let input = |r: Reg| p.inputs.iter().find(|i| i.0 == r).map(|i| i.1);
        let uniform: HashSet<Value> = input(Reg::S(0)).into_iter().collect();
        let io: Vec<(Value, Input)> = input(Reg::V(0))
            .map(|v| (v, Input::ThreadIdInGroup(0)))
            .into_iter()
            .collect();
        plan(p, &uniform, [128, 1, 1], 512, &io)
    }

    /// The (first) block holding the instruction at word `w`.
    fn block_at(p: &Program, w: usize) -> usize {
        p.blocks
            .iter()
            .position(|b| b.insts.iter().any(|i| i.machine.offset == 4 * w))
            .unwrap()
    }

    /// The instruction at word `w`.
    fn inst_at(p: &Program, w: usize) -> &Inst {
        p.blocks[block_at(p, w)]
            .insts
            .iter()
            .find(|i| i.machine.offset == 4 * w)
            .unwrap()
    }

    /// Each lane counts up to its thread id (at least 1); a wave leaves
    /// when all its lanes have: wave 0 after 63 iterations, wave 1 after
    /// 127. EXEC is restored after the loop, whose count is then read.
    const LANE_LOOP: &[u32] = &[
        0x7E04_0280, // v_mov_b32 v2, 0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0x4A04_0481, // loop: v_add_nc_u32 v2, 1, v2
        0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFFC, // s_cbranch_execnz loop
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0x7E06_0302, // v_mov_b32 v3, v2
        S_ENDPGM,
    ];

    /// The same count, two steps an iteration, with an exit after each.
    const TWO_EXITS: &[u32] = &[
        0x7E04_0280, // v_mov_b32 v2, 0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0x4A04_0481, // loop: v_add_nc_u32 v2, 1, v2
        0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF88_0004, // s_cbranch_execz out
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFF8, // s_cbranch_execnz loop
        0xBEFE_040A, // out: s_mov_b64 exec, s[10:11]
        0x7E06_0302, // v_mov_b32 v3, v2
        S_ENDPGM,
    ];

    #[test]
    fn a_loop_the_waves_leave_apart_runs_while_one_is_in_it() {
        let p = ir(LANE_LOOP);
        let g = plan_of(&p).unwrap();
        assert_eq!(g.loops.len(), 1);
        let l = &g.loops[0];
        let latch = block_at(&p, 5);
        assert_eq!(l.exits, vec![latch]);
        assert!(matches!(g.branches[&latch], Branch::Leave { lp: 0, .. }));
        // The count the loop leaves with is held for after it.
        let count = inst_at(&p, 2).write(Reg::V(2)).unwrap();
        assert!(l.held.iter().any(|&(v, bit)| v == count && !bit));
        assert_eq!(g.freeze.get(&count), Some(&0));
    }

    #[test]
    fn a_loop_with_two_exits_holds_each_exits_values() {
        let p = ir(TWO_EXITS);
        let g = plan_of(&p).unwrap();
        let l = &g.loops[0];
        assert_eq!(l.exits, vec![block_at(&p, 5), block_at(&p, 9)]);
        assert_eq!(l.join, block_at(&p, 10));
        for w in [2, 6] {
            let v = inst_at(&p, w).write(Reg::V(2)).unwrap();
            assert!(l.held.iter().any(|h| h.0 == v), "word {w}");
        }
    }

    #[test]
    fn what_a_loop_the_waves_leave_apart_cannot_hold_stops() {
        // A barrier inside: the waves would meet it different numbers of
        // times.
        let mut w = LANE_LOOP.to_vec();
        w.insert(5, 0xBF8A_0000); // s_barrier
        w[6] = 0xBF89_FFFB; // s_cbranch_execnz loop
        assert_eq!(plan_of(&ir(&w)).unwrap_err(), BARRIER_IN_LOOP);
        // Exits whose paths run different code before they meet.
        let mut w = TWO_EXITS.to_vec();
        w[5] = 0xBF88_0005; // s_cbranch_execz past the move below
        w.insert(10, 0x7E04_0285); // v_mov_b32 v2, 5 (the second exit only)
        assert_eq!(plan_of(&ir(&w)).unwrap_err(), EXITS_APART);
    }

    /// Only lanes below 64 (the first console wave) go into the region; in
    /// it, a branch with code on both sides.
    const FIRST_WAVE_REGION: &[u32] = &[
        0x7E04_0287, // v_mov_b32 v2, 7
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0x7DA8_00C0, // v_cmpx_gt_u32 exec, 64, v0
        0xBF88_0007, // s_cbranch_execz after
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0x87EA_6A7E, // s_and_b64 vcc, exec, vcc
        0xBF86_0002, // s_cbranch_vccz else
        0x7E04_0281, // v_mov_b32 v2, 1
        0xBF82_0001, // s_branch end
        0x7E04_0282, // else: v_mov_b32 v2, 2
        0x4A04_0481, // end: v_add_nc_u32 v2, 1, v2
        0xBEFE_040A, // after: s_mov_b64 exec, s[10:11]
        S_ENDPGM,
    ];

    #[test]
    fn a_region_one_wave_can_go_into_follows_that_wave() {
        let p = ir(FIRST_WAVE_REGION);
        let g = plan_of(&p).unwrap();
        assert!(matches!(g.branches[&block_at(&p, 3)], Branch::Skip { .. }));
        assert_eq!(g.branches[&block_at(&p, 6)], Branch::Follow);
        // Both waves can go in where lanes below 80 do: both sides apart.
        let mut w = FIRST_WAVE_REGION.to_vec();
        w[2] = 0x7DA8_00FF; // v_cmpx_gt_u32 exec, lit, v0
        w.insert(3, 80);
        w[4] = 0xBF88_0007;
        assert_eq!(plan_of(&ir(&w)).unwrap_err(), BRANCH_APART);
    }

    #[test]
    fn the_lanes_of_a_thread_id_compare_bound_the_waves() {
        let p = ir(FIRST_WAVE_REGION);
        let tid = p.inputs.iter().find(|i| i.0 == Reg::V(0)).unwrap().1;
        let lanes = Lanes::new(&p, 128, Some(tid));
        let b = block_at(&p, 3);
        let region = match p.blocks[b].exit {
            Exit::Branch { not_taken, .. } => not_taken,
            _ => unreachable!(),
        };
        assert_eq!(lanes.entering_waves(&p, b, region), 1);
        let exec = p.blocks[b]
            .insts
            .last()
            .unwrap()
            .read(Reg::S(EXEC))
            .unwrap();
        let (lo, hi) = lanes.of(exec, 0);
        // Lanes 0-31 of the first wave, exactly.
        assert_eq!(lo, vec![0xFFFF_FFFF, 0]);
        assert_eq!(hi, lo);
    }

    /// A first-lane read of one dword (no per-lane address) gives a loop
    /// its count; an `s_barrier` in the loop.
    const COUNTED_BY_ONE_DWORD: &[u32] = &[
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_0FFC,
        0x8001_0100, // buffer_load_dword v1, off, s[4:7], 0 offset:4092
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E12_0501, // v_readfirstlane_b32 s9, v1
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBF8A_0000, // loop: s_barrier
        0x810A_810A, // s_add_i32 s10, s10, 1
        0xBF04_090A, // s_cmp_lt_i32 s10, s9
        0xBF85_FFFC, // s_cbranch_scc1 loop
        S_ENDPGM,
    ];

    #[test]
    fn a_first_lane_read_of_one_dword_is_the_groups() {
        let p = ir(COUNTED_BY_ONE_DWORD);
        let g = plan_of(&p).unwrap();
        let read = inst_at(&p, 8).write(Reg::S(9)).unwrap();
        assert!(g.broadcast.contains(&read));
        assert_eq!(g.branches[&block_at(&p, 13)], Branch::Uniform);
        assert!(g.loops.is_empty());
        // A store that may come first (another wave's, further on): the
        // waves may read different dwords, and the barrier is then in a
        // loop they may leave apart.
        let mut w = COUNTED_BY_ONE_DWORD.to_vec();
        w.splice(5..5, [0xE070_0FFC, 0x8001_0100]); // buffer_store_dword v1, off, s[4:7], 0 offset:4092
        assert_eq!(plan_of(&ir(&w)).unwrap_err(), BARRIER_IN_LOOP);
    }

    /// A loop counted by an LDS dword every lane reads at one address (a
    /// first-lane read of it), with an `s_barrier`; before it, a VCC test
    /// of a compare of that dword.
    const COUNTED_BY_AN_LDS_DWORD: &[u32] = &[
        0x7E06_0280, // v_mov_b32 v3, 0
        0xD8D8_0010,
        0x0300_0003, // ds_read_b32 v3, v3 offset:16
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7D88_0682, // v_cmp_gt_u32 vcc, 2, v3
        0x7E12_0503, // v_readfirstlane_b32 s9, v3
        0xBF87_0005, // s_cbranch_vccnz end
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBF8A_0000, // loop: s_barrier
        0x810A_810A, // s_add_i32 s10, s10, 1
        0xBF04_090A, // s_cmp_lt_i32 s10, s9
        0xBF85_FFFC, // s_cbranch_scc1 loop
        S_ENDPGM,    // end
    ];

    #[test]
    fn an_lds_dword_every_lane_reads_counts_for_the_group() {
        let p = ir(COUNTED_BY_AN_LDS_DWORD);
        let g = plan_of(&p).unwrap();
        assert_eq!(g.branches[&block_at(&p, 6)], Branch::Uniform);
        assert_eq!(g.branches[&block_at(&p, 11)], Branch::Uniform);
        assert!(g.loops.is_empty());
        // Read at a per-lane address instead: the waves may count apart,
        // and the loop holds a barrier.
        let mut w = COUNTED_BY_AN_LDS_DWORD.to_vec();
        w[0] = 0x7E06_0300; // v_mov_b32 v3, v0
        assert_eq!(plan_of(&ir(&w)).unwrap_err(), BARRIER_IN_LOOP);
    }

    #[test]
    fn an_end_on_a_count_ends_only_the_waves_that_take_it() {
        let w = [
            0x7D88_00C0, // v_cmp_gt_u32 vcc, 64, v0
            0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
            0xBF84_0003, // s_cbranch_scc0 end
            0x7E04_0281, // v_mov_b32 v2, 1
            0xBF8A_0000, // s_barrier
            S_ENDPGM,
            S_ENDPGM, // end
        ];
        let p = ir(&w);
        let g = plan_of(&p).unwrap();
        assert!(matches!(g.branches[&block_at(&p, 2)], Branch::Exit { .. }));
    }

    #[test]
    fn a_group_with_no_lds_room_for_the_exchange_stops() {
        let p = ir(LANE_LOOP);
        // 1024 threads: 32 host waves, 128 bytes of exchange.
        let none = HashSet::new();
        assert_eq!(
            plan(&p, &none, [1024, 1, 1], 32768, &[]).unwrap_err(),
            NO_LDS_ROOM
        );
        // 128 bytes fewer of the title's: room.
        assert!(plan(&p, &none, [1024, 1, 1], 32640, &[]).is_ok());
    }
}
