//! One instruction for one value. The lowering writes code as the console
//! instructions name it, one after another: a register read as a float is
//! cast again at each read, a descriptor's handle is made and its index
//! clamped again in each block, an address is shifted and masked again for
//! each access. Where an instruction computes what an earlier one computed
//! (the same operation on the same operands) and every path to it has run
//! that earlier one, its readers take the earlier result and it is left
//! out.
//!
//! The pass is exact, and narrow on purpose:
//!
//! * Only instructions whose result is a function of their operands alone
//!   take part: integer arithmetic and compares, integer selects, width
//!   casts and bit casts, `extractvalue`, `getelementptr`, handle creation
//!   and annotation, and the integer `dx.op`s declared `readnone`. A
//!   resource's dimensions do not change while a shader runs, so
//!   `getDimensions` of one handle and level is one value too.
//! * Nothing that computes with floats takes part (float arithmetic,
//!   compares, conversions, float `dx.op`s): a driver may contract or
//!   reorder them by how they are used, so one instruction read twice is
//!   not known to give what two gave.
//! * Of what reads memory, only loads from the host's own tables take
//!   part (the page table, the view addresses, the view marks and the
//!   view strides: `emit` "Title memory", "Write marks", "Buffer views in
//!   the heap"). No shader writes them and the host does not change them
//!   under a list that runs, so one entry read twice is one value. Every
//!   other load, sample or wave operation is asked again wherever the
//!   lowering asked: a store or another thread may have come between.
//!
//! What it leaves out is always a copy of an instruction that stays, with
//! the same operands. So no operation and no declaration loses its last
//! use (the shader flags and the declarations stay as they were), and no
//! value loses its last reader.

use std::collections::HashMap;

use super::{Attr, Bin, Cast, Module, Op, Term, Ty, TypeDef, V, successors};

/// What an instruction computes: its operation and operands.
#[derive(PartialEq, Eq, Hash, Clone)]
enum Key {
    Bin(u8, V, V),
    Cast(u8, V, Ty),
    Cmp(u8, V, V),
    Select(V, V, V),
    Extract(V, u32),
    Gep(Ty, V, Vec<V>),
    Call(u32, Vec<V>),
}

/// The calls beside the integer `readnone` ones whose result is a function
/// of their operands: handles, and a resource's dimensions.
const CALLS: [&str; 4] = [
    "dx.op.createHandleFromBinding",
    "dx.op.createHandleFromHeap",
    "dx.op.annotateHandle",
    "dx.op.getDimensions",
];

/// How the host's tables are read.
const TABLE_LOAD: &str = "dx.op.rawBufferLoad.f32";

impl Module {
    /// Leaves out the instructions that compute a value an instruction
    /// before them on every path already holds (module notes). `tables` are
    /// the handles of the host's tables, whose raw loads are values too.
    /// Returns how many instructions it left out.
    pub(crate) fn reuse(&mut self, tables: &[V]) -> usize {
        if self.blocks.is_empty() {
            return 0;
        }
        let children = self.dominator_tree();
        // The value each instruction left out is read as.
        let mut same: HashMap<u32, V> = HashMap::new();
        let mut held: HashMap<Key, V> = HashMap::new();
        // The keys held, in the order taken: a block's go when it is left.
        let mut taken: Vec<Key> = Vec::new();
        enum Step {
            Enter(usize),
            Leave(usize),
        }
        let mut steps = vec![Step::Enter(0)];
        while let Some(step) = steps.pop() {
            let b = match step {
                Step::Enter(b) => b,
                Step::Leave(mark) => {
                    for key in taken.drain(mark..) {
                        held.remove(&key);
                    }
                    continue;
                }
            };
            steps.push(Step::Leave(taken.len()));
            let mut body = std::mem::take(&mut self.blocks[b].body);
            body.retain(|&i| {
                // Its operands were made on every path here, so those left
                // out are known by now.
                rewrite(&mut self.insts[i as usize].op, &same);
                let Some(key) = self.key(i, tables, &same) else {
                    return true;
                };
                match held.get(&key) {
                    Some(&earlier) => {
                        same.insert(i, earlier);
                        false
                    }
                    None => {
                        held.insert(key.clone(), V::Inst(i));
                        taken.push(key);
                        true
                    }
                }
            });
            self.blocks[b].body = body;
            steps.extend(children[b].iter().map(|&c| Step::Enter(c)));
        }
        if same.is_empty() {
            return 0;
        }
        // Phis read values made after them (a loop's), and branches read
        // their condition: both once every block is done.
        for b in 0..self.blocks.len() {
            for k in 0..self.blocks[b].phis.len() {
                let i = self.blocks[b].phis[k];
                rewrite(&mut self.insts[i as usize].op, &same);
            }
            if let Some(Term::CondBr(c, ..)) = &mut self.blocks[b].term {
                read(c, &same);
            }
        }
        same.len()
    }

    /// What instruction `i` computes, if it takes part.
    fn key(&self, i: u32, tables: &[V], same: &HashMap<u32, V>) -> Option<Key> {
        let inst = &self.insts[i as usize];
        let int = |t: Ty| matches!(self.types[t.0 as usize], TypeDef::Int(_));
        Some(match &inst.op {
            Op::Bin(op, a, b) if int(inst.ty) => {
                // The operands of a commutative operation in one order.
                let swap = matches!(op, Bin::Add | Bin::Mul | Bin::And | Bin::Or | Bin::Xor);
                let (a, b) = if swap && b < a { (*b, *a) } else { (*a, *b) };
                Key::Bin(*op as u8, a, b)
            }
            Op::Cast(op, v)
                if matches!(op, Cast::Trunc | Cast::ZExt | Cast::SExt | Cast::Bitcast) =>
            {
                Key::Cast(*op as u8, *v, inst.ty)
            }
            // Integer predicates.
            Op::Cmp(p, a, b) if *p >= 32 => Key::Cmp(*p, *a, *b),
            Op::Select(c, t, f) if int(inst.ty) => Key::Select(*c, *t, *f),
            Op::Extract(v, index) => Key::Extract(*v, *index),
            Op::Gep(ty, p, index) => Key::Gep(*ty, *p, index.clone()),
            Op::Call(f, args) => {
                let func = &self.funcs[*f as usize];
                let integer = func.attr == Some(Attr::ReadNone) && func.name.ends_with(".i32");
                // A raw load's first argument after the opcode is its handle.
                let table = func.name == TABLE_LOAD
                    && args.get(1).is_some_and(|h| {
                        tables.iter().any(|&t| {
                            let mut t = t;
                            read(&mut t, same);
                            t == *h
                        })
                    });
                if !integer && !table && !CALLS.contains(&func.name.as_str()) {
                    return None;
                }
                Key::Call(*f, args.clone())
            }
            _ => return None,
        })
    }

    /// The blocks each block immediately dominates, of those `main`'s
    /// entry (block 0) reaches.
    fn dominator_tree(&self) -> Vec<Vec<usize>> {
        let n = self.blocks.len();
        // Reverse postorder, and each block's place in it.
        let mut seen = vec![false; n];
        let mut post = Vec::new();
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            top.1 += 1;
            match successors(self.blocks[b].term).get(k) {
                Some(&s) if !seen[s] => {
                    seen[s] = true;
                    stack.push((s, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        let mut place = vec![usize::MAX; n];
        for (k, &b) in post.iter().enumerate() {
            place[b] = k;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &post {
            for &s in successors(self.blocks[b].term).iter() {
                preds[s].push(b);
            }
        }
        // Cooper, Harvey and Kennedy's iteration over reverse postorder.
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &post[1..] {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    if new == usize::MAX {
                        new = p;
                        continue;
                    }
                    let (mut x, mut y) = (p, new);
                    while x != y {
                        while place[x] > place[y] {
                            x = idom[x];
                        }
                        while place[y] > place[x] {
                            y = idom[y];
                        }
                    }
                    new = x;
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let mut children = vec![Vec::new(); n];
        for &b in &post[1..] {
            children[idom[b]].push(b);
        }
        children
    }
}

/// Reads through the instructions left out.
fn read(v: &mut V, same: &HashMap<u32, V>) {
    if let V::Inst(i) = *v
        && let Some(&earlier) = same.get(&i)
    {
        *v = earlier;
    }
}

fn rewrite(op: &mut Op, same: &HashMap<u32, V>) {
    if same.is_empty() {
        return;
    }
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) | Op::Store(a, b) | Op::AtomicRmw(_, a, b) => {
            read(a, same);
            read(b, same);
        }
        Op::Cast(_, v) | Op::Extract(v, _) | Op::Load(v) => read(v, same),
        Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
            read(a, same);
            read(b, same);
            read(c, same);
        }
        Op::Call(_, args) => args.iter_mut().for_each(|a| read(a, same)),
        Op::Phi(incoming) => incoming.iter_mut().for_each(|x| read(&mut x.0, same)),
        Op::Gep(_, p, index) => {
            read(p, same);
            index.iter_mut().for_each(|i| read(i, same));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Attr, Bin, Cast, Module, Rmw, Term, V, pred};
    use crate::emit::counts;

    /// A module with the functions the tests call: a handle from the heap,
    /// a load through it (which reads memory), a float operation, a sink.
    struct Fixture {
        m: Module,
        handle: V,
        load: V,
        float: V,
        sink: V,
    }

    fn fixture() -> Fixture {
        let mut m = Module::default();
        let (void, i1, i32t, f32t) = (m.void(), m.int(1), m.int(32), m.float());
        let i8t = m.int(8);
        let i8p = m.ptr(i8t);
        let handle_t = m.named_struct("dx.types.Handle", vec![i8p]);
        let ty = m.func_ty(handle_t, vec![i32t, i32t, i1, i1]);
        let handle = m.function(
            "dx.op.createHandleFromHeap",
            ty,
            Some(Attr::ReadNone),
            false,
        );
        let ty = m.func_ty(i32t, vec![i32t, handle_t, i32t]);
        let load = m.function("dx.op.load.i32", ty, Some(Attr::ReadOnly), false);
        let ty = m.func_ty(f32t, vec![i32t, f32t]);
        let float = m.function("dx.op.unary.f32", ty, Some(Attr::ReadNone), false);
        let ty = m.func_ty(void, vec![i32t, i32t]);
        let sink = m.function("dx.op.sink", ty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.function("main", main_ty, None, true);
        Fixture {
            m,
            handle,
            load,
            float,
            sink,
        }
    }

    fn kinds(m: &Module) -> Vec<(String, u64)> {
        counts::count(&m.write())
            .expect("counts")
            .kinds
            .into_iter()
            .collect()
    }

    fn kind(m: &Module, name: &str) -> u64 {
        kinds(m)
            .into_iter()
            .find(|(k, _)| k == name)
            .map_or(0, |(_, n)| n)
    }

    /// What the lowering writes for two reads through one descriptor in one
    /// block: the index clamped twice, two handles, the address made twice.
    /// One of each stays; both loads stay, and read the one handle.
    #[test]
    fn a_block_keeps_one_of_each_value() {
        let mut f = fixture();
        let m = &mut f.m;
        let b = m.new_block();
        let (index, last, four, no, zero) =
            (m.i32(7), m.i32(100), m.i32(4), m.bool(false), m.i32(0));
        let x = m.call(b, f.load, vec![zero, V::Const(0), zero]);
        let mut loads = Vec::new();
        for _ in 0..2 {
            let scaled = m.bin(b, Bin::Mul, x, four);
            let over = m.cmp(b, pred::IUGT, scaled, last);
            let clamped = m.select(b, over, last, scaled);
            let h = m.call(b, f.handle, vec![index, clamped, no, no]);
            let at = m.bin(b, Bin::Add, four, scaled);
            loads.push(m.call(b, f.load, vec![zero, h, at]));
        }
        let sum = m.bin(b, Bin::Add, loads[0], loads[1]);
        let _ = m.call(b, f.sink, vec![zero, sum]);
        m.terminate(b, Term::Ret);
        let before = counts::count(&m.write()).expect("counts").instructions;
        assert_eq!(m.reuse(&[]), 5);
        let after = counts::count(&m.write()).expect("counts");
        assert_eq!(after.instructions, before - 5);
        assert_eq!(kind(m, "dx.op.createHandleFromHeap"), 1);
        assert_eq!(kind(m, "dx.op.load.i32"), 3, "loads are asked again");
        assert_eq!(
            (kind(m, "mul"), kind(m, "icmp"), kind(m, "select")),
            (1, 1, 1)
        );
        // The two loads' sum and the address: `4 + scaled` once.
        assert_eq!(kind(m, "add"), 2);
        assert_eq!(m.reuse(&[]), 0, "nothing left to leave out");
    }

    /// A value made before a branch serves both sides and the join; a value
    /// made on one side serves neither the other side nor the join.
    #[test]
    fn a_value_serves_only_what_it_dominates() {
        let mut f = fixture();
        let m = &mut f.m;
        let (entry, then, other, join) =
            (m.new_block(), m.new_block(), m.new_block(), m.new_block());
        let (zero, one, two) = (m.i32(0), m.i32(1), m.i32(2));
        let x = m.call(entry, f.load, vec![zero, V::Const(0), zero]);
        let above = m.bin(entry, Bin::Add, x, one);
        let c = m.cmp(entry, pred::EQ, above, zero);
        m.terminate(entry, Term::CondBr(c, then, other));
        for b in [then, other, join] {
            // `x + 1` again (the entry's), and `x << 2` (nobody's above).
            let again = m.bin(b, Bin::Add, x, one);
            let side = m.bin(b, Bin::Shl, x, two);
            let both = m.bin(b, Bin::Xor, again, side);
            let _ = m.call(b, f.sink, vec![zero, both]);
            m.terminate(b, if b == join { Term::Ret } else { Term::Br(join) });
        }
        assert_eq!(m.reuse(&[]), 3, "the three later `x + 1`");
        assert_eq!(kind(m, "add"), 1);
        assert_eq!(
            kind(m, "shl"),
            3,
            "one per block: no side dominates another"
        );
        assert_eq!(kind(m, "xor"), 3);
    }

    /// Floats, memory and atomics are left as written: two equal float
    /// operations, two equal loads, two equal atomics all stay; the bit
    /// casts around them are one each.
    #[test]
    fn floats_and_memory_are_left_alone() {
        let mut f = fixture();
        let m = &mut f.m;
        let (i32t, f32t) = (m.int(32), m.float());
        let arr = m.array(4, i32t);
        let lds = m.global("lds", arr, 3);
        let b = m.new_block();
        let (zero, one) = (m.i32(0), m.i32(1));
        let x = m.call(b, f.load, vec![zero, V::Const(0), zero]);
        let mut out = Vec::new();
        for _ in 0..2 {
            let fx = m.cast(b, Cast::Bitcast, x, f32t);
            let product = m.bin(b, Bin::Mul, fx, fx);
            let root = m.call(b, f.float, vec![zero, product]);
            let less = m.cmp(b, pred::OLT, root, fx);
            let picked = m.select(b, less, root, fx);
            let bits = m.cast(b, Cast::Bitcast, picked, i32t);
            let at = m.element(b, lds, zero);
            let old = m.load(b, at);
            let added = m.atomic(b, Rmw::Add, at, one);
            let converted = m.cast(b, Cast::FpToUi, picked, i32t);
            let all = m.bin(b, Bin::Add, bits, old);
            let all = m.bin(b, Bin::Add, all, added);
            out.push(m.bin(b, Bin::Add, all, converted));
        }
        let _ = m.call(b, f.sink, vec![out[0], out[1]]);
        m.terminate(b, Term::Ret);
        // Left out: the second bit cast of `x` and the second address.
        assert_eq!(m.reuse(&[]), 2);
        assert_eq!(kind(m, "mul"), 2, "float multiplies");
        assert_eq!(kind(m, "dx.op.unary.f32"), 2);
        assert_eq!((kind(m, "fcmp"), kind(m, "select")), (2, 2));
        assert_eq!(
            (kind(m, "load"), kind(m, "atomicrmw"), kind(m, "fptoui")),
            (2, 2, 2)
        );
        assert_eq!(kind(m, "getelementptr"), 1);
        // `x` as a float once; each pick back to bits (two picks).
        assert_eq!(kind(m, "bitcast"), 3);
    }

    /// A load from one of the host's tables is a value: the same entry read
    /// again, in the block and in one it dominates, is the first read. The
    /// same load through any other handle stays.
    #[test]
    fn a_table_entry_is_read_once() {
        let mut m = Module::default();
        let (void, i1, i32t) = (m.void(), m.int(1), m.int(32));
        let i8t = m.int(8);
        let i8p = m.ptr(i8t);
        let handle_t = m.named_struct("dx.types.Handle", vec![i8p]);
        let ty = m.func_ty(handle_t, vec![i32t, i32t, i1, i1]);
        let handle = m.function(
            "dx.op.createHandleFromHeap",
            ty,
            Some(Attr::ReadNone),
            false,
        );
        let ty = m.func_ty(i32t, vec![i32t, handle_t, i32t]);
        let load = m.function("dx.op.rawBufferLoad.f32", ty, Some(Attr::ReadOnly), false);
        let ty = m.func_ty(void, vec![i32t, i32t]);
        let sink = m.function("dx.op.sink", ty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.function("main", main_ty, None, true);
        let (entry, next) = (m.new_block(), m.new_block());
        let (zero, one, eight, no) = (m.i32(0), m.i32(1), m.i32(8), m.bool(false));
        let table = m.call(entry, handle, vec![zero, zero, no, no]);
        let data = m.call(entry, handle, vec![zero, one, no, no]);
        for b in [entry, next] {
            for h in [table, data] {
                let x = m.call(b, load, vec![zero, h, eight]);
                let y = m.call(b, load, vec![zero, h, eight]);
                let _ = m.call(b, sink, vec![x, y]);
            }
            m.terminate(
                b,
                if b == entry {
                    Term::Br(next)
                } else {
                    Term::Ret
                },
            );
        }
        // Of the table's four loads one stays; the data's four all stay.
        assert_eq!(m.reuse(&[table]), 3);
        assert_eq!(kind(&m, "dx.op.rawBufferLoad.f32"), 5);
        // Without the table named, a load is a load.
        assert_eq!(m.reuse(&[]), 0);
    }

    /// In a loop the header's value serves the body in the same iteration,
    /// the phi that carries a left-out value around reads the one kept, and
    /// so does the branch.
    #[test]
    fn loops_read_the_value_kept() {
        let mut f = fixture();
        let m = &mut f.m;
        let i32t = m.int(32);
        let (entry, head, body, exit) =
            (m.new_block(), m.new_block(), m.new_block(), m.new_block());
        let (zero, one, ten) = (m.i32(0), m.i32(1), m.i32(10));
        m.terminate(entry, Term::Br(head));
        let i = m.phi(head, i32t);
        let carried = m.phi(head, i32t);
        let next = m.bin(head, Bin::Add, i, one);
        m.terminate(head, Term::Br(body));
        let again = m.bin(body, Bin::Add, one, i);
        let more = m.cmp(body, pred::IULT, again, ten);
        let _ = m.call(body, f.sink, vec![zero, carried]);
        m.terminate(body, Term::CondBr(more, head, exit));
        m.set_phi(i, vec![(zero, entry), (next, body)]);
        m.set_phi(carried, vec![(zero, entry), (again, body)]);
        m.terminate(exit, Term::Ret);
        assert_eq!(m.reuse(&[]), 1, "`1 + i` is `i + 1`");
        assert_eq!(kind(m, "add"), 1);
        assert_eq!(kind(m, "phi"), 2);
        // Written without a forward reference: the phi and the compare read
        // the header's sum.
        let _ = m.write();
    }
}
