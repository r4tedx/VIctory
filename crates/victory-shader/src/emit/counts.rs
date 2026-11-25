//! What an emitted shader's `main` holds, counted from the container's
//! bitcode: the instructions the host's compiler is given, whatever made
//! them, so two revisions of the translator compare by reading what each
//! wrote. It reads the unabbreviated records `bitcode` writes and refuses
//! anything else (a module DXC wrote has abbreviations).
//!
//! The counts are static: one per instruction written, however often it
//! runs. They rank what the translation adds and show what a change takes
//! out; they are no measure of time.

use std::collections::BTreeMap;

use crate::dxbc::{Container, FourCC};

type Why = &'static str;

/// The instructions of one shader's `main`, or of several added up.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    /// Basic blocks.
    pub blocks: u64,
    /// Instructions, terminators among them.
    pub instructions: u64,
    /// The same by kind: LLVM's name of the instruction (the integer name
    /// for both forms of a binary operation: the record does not tell
    /// `add` from `fadd`), and for a call the function called.
    pub kinds: BTreeMap<String, u64>,
}

impl Counts {
    pub fn add(&mut self, other: &Counts) {
        self.blocks += other.blocks;
        self.instructions += other.instructions;
        for (kind, n) in &other.kinds {
            *self.kinds.entry(kind.clone()).or_default() += n;
        }
    }
}

/// Counts the `main` of the shader in `container` (a DXIL container as
/// [`super::HostShader::dxil`] holds).
pub fn of(container: &[u8]) -> Result<Counts, Why> {
    let c = Container::parse(container).map_err(|_| "no container")?;
    let part = c.part(FourCC::DXIL).ok_or("no DXIL part")?.data;
    // The program header: version, size, "DXIL", DXIL version, then the
    // bitcode's offset from the magic and its size.
    let dword = |at: usize| -> Result<usize, Why> {
        let b = part.get(at..at + 4).ok_or("a truncated program header")?;
        Ok(u32::from_le_bytes(b.try_into().expect("4")) as usize)
    };
    let start = 8 + dword(16)?;
    let bitcode = part
        .get(start..start + dword(20)?)
        .ok_or("bitcode past the part")?;
    count(bitcode)
}

const MODULE_BLOCK: u64 = 8;
const CONSTANTS_BLOCK: u64 = 11;
const FUNCTION_BLOCK: u64 = 12;
const VALUE_SYMTAB_BLOCK: u64 = 14;
const TYPE_BLOCK: u64 = 17;

/// Counts the one function body of a module `bitcode::Module::write` wrote.
pub(crate) fn count(bitcode: &[u8]) -> Result<Counts, Why> {
    if bitcode.get(..4) != Some(b"BC\xC0\xDE") {
        return Err("no bitcode");
    }
    let mut r = Bits {
        data: bitcode,
        at: 32,
    };
    let mut m = Module::default();
    // The stream holds the module block, entered at the top-level width.
    if r.fixed(2)? != 1 || r.vbr(8)? != MODULE_BLOCK {
        return Err("no module block");
    }
    let width = r.vbr(4)? as u32;
    r.align();
    r.fixed(32)?;
    m.block(&mut r, MODULE_BLOCK, width)?;
    let mut counts = m.counts;
    for (callee, n) in m.calls {
        let name = m.names.remove(&callee).unwrap_or_else(|| "call".into());
        *counts.kinds.entry(name).or_default() += n;
    }
    Ok(counts)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Type {
    Void,
    /// A function type and its return type.
    Function(u64),
    Other,
}

#[derive(Default)]
struct Module {
    types: Vec<Type>,
    /// Value ids run through globals, functions, constants, then the
    /// instructions that have a value.
    globals: u64,
    functions: u64,
    constants: u64,
    next: u64,
    names: BTreeMap<u64, String>,
    /// Calls by the value id called.
    calls: BTreeMap<u64, u64>,
    counts: Counts,
}

impl Module {
    /// Reads block `id` up to its end; `r` is just past its length word.
    fn block(&mut self, r: &mut Bits, id: u64, width: u32) -> Result<(), Why> {
        if !(2..=32).contains(&width) {
            return Err("a block's abbreviation width");
        }
        if id == FUNCTION_BLOCK {
            self.next = self.globals + self.functions + self.constants;
        }
        let mut ops = Vec::new();
        loop {
            match r.fixed(width)? {
                0 => {
                    r.align();
                    return Ok(());
                }
                1 => {
                    let inner = r.vbr(8)?;
                    let inner_width = r.vbr(4)? as u32;
                    r.align();
                    let words = r.fixed(32)? as usize;
                    // The module's own tables; anything else is passed over
                    // (attributes, metadata, a function's inner blocks).
                    let read = id == MODULE_BLOCK
                        && matches!(
                            inner,
                            TYPE_BLOCK | CONSTANTS_BLOCK | VALUE_SYMTAB_BLOCK | FUNCTION_BLOCK
                        );
                    if read {
                        self.block(r, inner, inner_width)?;
                    } else {
                        r.at += words * 32;
                    }
                }
                3 => {
                    let code = r.vbr(6)?;
                    let n = r.vbr(6)?;
                    ops.clear();
                    for _ in 0..n {
                        ops.push(r.vbr(6)?);
                    }
                    self.record(id, code, &ops)?;
                }
                _ => return Err("an abbreviated record"),
            }
        }
    }

    fn record(&mut self, block: u64, code: u64, ops: &[u64]) -> Result<(), Why> {
        let op = |i: usize| ops.get(i).copied().ok_or("a short record");
        match (block, code) {
            // NUMENTRY and a struct's name make no type.
            (TYPE_BLOCK, 1 | 19) => {}
            (TYPE_BLOCK, 2) => self.types.push(Type::Void),
            (TYPE_BLOCK, 21) => self.types.push(Type::Function(op(1)?)),
            (TYPE_BLOCK, _) => self.types.push(Type::Other),
            (MODULE_BLOCK, 7) => self.globals += 1,
            (MODULE_BLOCK, 8) => self.functions += 1,
            (MODULE_BLOCK, _) => {}
            // SETTYPE makes no value.
            (CONSTANTS_BLOCK, 1) => {}
            (CONSTANTS_BLOCK, _) => self.constants += 1,
            (VALUE_SYMTAB_BLOCK, 1) => {
                let name = ops[1.min(ops.len())..]
                    .iter()
                    .map(|&c| c as u8 as char)
                    .collect();
                self.names.insert(op(0)?, name);
            }
            (VALUE_SYMTAB_BLOCK, _) => {}
            (FUNCTION_BLOCK, 1) => self.counts.blocks += op(0)?,
            (FUNCTION_BLOCK, _) => self.instruction(code, ops)?,
            _ => {}
        }
        Ok(())
    }

    fn instruction(&mut self, code: u64, ops: &[u64]) -> Result<(), Why> {
        let op = |i: usize| ops.get(i).copied().ok_or("a short record");
        // The kind, and whether the instruction has a value.
        let (kind, value) = match code {
            2 => (
                *[
                    "add", "sub", "mul", "udiv", "sdiv", "urem", "srem", "shl", "lshr", "ashr",
                    "and", "or", "xor",
                ]
                .get(op(2)? as usize)
                .ok_or("a binary opcode")?,
                true,
            ),
            3 => (
                *[
                    "trunc", "zext", "sext", "fptoui", "fptosi", "uitofp", "sitofp", "fptrunc",
                    "fpext", "ptrtoint", "inttoptr", "bitcast",
                ]
                .get(op(2)? as usize)
                .ok_or("a cast opcode")?,
                true,
            ),
            10 => ("ret", false),
            11 => ("br", false),
            16 => ("phi", true),
            20 => ("load", true),
            26 => ("extractvalue", true),
            28 => (if op(2)? < 32 { "fcmp" } else { "icmp" }, true),
            29 => ("select", true),
            38 => ("atomicrmw", true),
            43 => ("getelementptr", true),
            44 => ("store", false),
            46 => ("cmpxchg", true),
            34 => {
                // [attributes, calling convention, function type, callee
                // (relative), arguments...]
                let callee = self
                    .next
                    .checked_sub(op(3)?)
                    .ok_or("a call of a later value")?;
                *self.calls.entry(callee).or_default() += 1;
                let void = match self.types.get(op(2)? as usize) {
                    Some(&Type::Function(ret)) => self.types.get(ret as usize) == Some(&Type::Void),
                    _ => return Err("a call without its function type"),
                };
                self.counts.instructions += 1;
                self.next += u64::from(!void);
                return Ok(());
            }
            _ => return Err("an instruction record of another kind"),
        };
        self.counts.instructions += 1;
        *self.counts.kinds.entry(kind.into()).or_default() += 1;
        self.next += u64::from(value);
        Ok(())
    }
}

/// The bitstream: bytes read from bit 0 up.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    /// The next `width` bits (at most 32).
    fn fixed(&mut self, width: u32) -> Result<u64, Why> {
        let byte = self.at / 8;
        if self.at + width as usize > self.data.len() * 8 {
            return Err("truncated bitcode");
        }
        let mut buf = [0u8; 8];
        let end = (byte + 8).min(self.data.len());
        buf[..end - byte].copy_from_slice(&self.data[byte..end]);
        let v = u64::from_le_bytes(buf) >> (self.at % 8);
        self.at += width as usize;
        Ok(v & ((1u64 << width) - 1))
    }

    fn vbr(&mut self, width: u32) -> Result<u64, Why> {
        let hi = 1u64 << (width - 1);
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let chunk = self.fixed(width)?;
            if shift >= 64 {
                return Err("a number past 64 bits");
            }
            v |= (chunk & (hi - 1)) << shift;
            if chunk & hi == 0 {
                return Ok(v);
            }
            shift += width - 1;
        }
    }

    fn align(&mut self) {
        self.at = self.at.next_multiple_of(32);
    }
}

#[cfg(test)]
mod tests {
    use super::super::bitcode::{Attr, Bin, Cast, Module, Rmw, Term, pred};
    use super::super::shader::Shader;
    use super::*;

    fn kinds(c: &Counts) -> Vec<(&str, u64)> {
        c.kinds.iter().map(|(k, &n)| (k.as_str(), n)).collect()
    }

    /// Every instruction the writer has, in a loop with a phi nothing reads
    /// (which the writer leaves out) and an empty latch (which it folds).
    #[test]
    fn counts_what_the_writer_wrote() {
        let mut m = Module::default();
        let (void, i1, i32t, f32t) = (m.void(), m.int(1), m.int(32), m.float());
        let load_ty = m.func_ty(i32t, vec![i32t]);
        let load = m.function("dx.op.load.i32", load_ty, Some(Attr::ReadOnly), false);
        let store_ty = m.func_ty(void, vec![i32t, i32t]);
        let store = m.function("dx.op.store.i32", store_ty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.function("main", main_ty, None, true);
        let arr = m.array(4, i32t);
        let lds = m.global("lds", arr, 3);
        let (entry, head, body, latch, exit) = (
            m.new_block(),
            m.new_block(),
            m.new_block(),
            m.new_block(),
            m.new_block(),
        );
        let (zero, one, four) = (m.i32(0), m.i32(1), m.i32(4));
        m.terminate(entry, Term::Br(head));
        let i = m.phi(head, i32t);
        let unread = m.phi(head, i32t);
        m.terminate(head, Term::Br(body));
        let x = m.call(body, load, vec![i]);
        let y = m.call(body, load, vec![x]);
        let sum = m.bin(body, Bin::Add, x, y);
        let shifted = m.bin(body, Bin::Shl, sum, one);
        let f = m.cast(body, Cast::Bitcast, shifted, f32t);
        let nan = m.cmp(body, pred::UNO, f, f);
        let big = m.cmp(body, pred::IUGT, shifted, four);
        let both = m.bin(body, Bin::And, nan, big);
        let picked = m.select(body, both, zero, shifted);
        let _ = m.call(body, store, vec![i, picked]);
        let at = m.element(body, lds, i);
        let old = m.load(body, at);
        m.store(body, at, picked);
        let _ = m.atomic(body, Rmw::Add, at, old);
        let _ = m.cmpxchg(body, at, old, picked);
        let next = m.bin(body, Bin::Add, i, one);
        let more = m.cmp(body, pred::IULT, next, four);
        m.terminate(body, Term::CondBr(more, latch, exit));
        m.terminate(latch, Term::Br(head));
        m.set_phi(i, vec![(zero, entry), (next, latch)]);
        m.set_phi(unread, vec![(zero, entry), (next, latch)]);
        m.terminate(exit, Term::Ret);
        let _ = i1;
        let c = count(&m.write()).expect("counts");
        // Entry, header, body and exit: the latch is folded into the body.
        assert_eq!(c.blocks, 4);
        assert_eq!(
            kinds(&c),
            [
                ("add", 2),
                ("and", 1),
                ("atomicrmw", 1),
                ("bitcast", 1),
                ("br", 3),
                ("cmpxchg", 1),
                ("dx.op.load.i32", 2),
                ("dx.op.store.i32", 1),
                ("extractvalue", 1),
                ("fcmp", 1),
                ("getelementptr", 1),
                ("icmp", 2),
                ("load", 1),
                ("phi", 1),
                ("ret", 1),
                ("select", 1),
                ("shl", 1),
                ("store", 1),
            ]
        );
        assert_eq!(c.instructions, c.kinds.values().sum::<u64>());
        assert_eq!(c.instructions, 23);
    }

    /// Through a whole container, as a shader is handed to the host.
    #[test]
    fn reads_a_container() {
        let mut s = Shader::new(true);
        let entry = s.entry;
        s.m.terminate(entry, Term::Ret);
        let c = of(&s.finish()).expect("counts");
        assert_eq!((c.blocks, c.instructions), (1, 1));
        assert_eq!(kinds(&c), [("ret", 1)]);
    }

    #[test]
    fn sums() {
        let one = Counts {
            blocks: 2,
            instructions: 3,
            kinds: [("add".to_owned(), 2), ("ret".to_owned(), 1)].into(),
        };
        let mut total = one.clone();
        total.add(&one);
        assert_eq!((total.blocks, total.instructions), (4, 6));
        assert_eq!(kinds(&total), [("add", 4), ("ret", 2)]);
    }

    /// What is no module of ours is refused, not miscounted.
    #[test]
    fn refuses_other_streams() {
        assert_eq!(count(b"BC"), Err("no bitcode"));
        assert_eq!(count(b"BC\xC0\xDE"), Err("truncated bitcode"));
        // A module block whose first entry defines an abbreviation (id 2).
        let mut bytes = b"BC\xC0\xDE".to_vec();
        // ENTER_SUBBLOCK (1, 2 bits), block 8 (vbr8), width 3 (vbr4).
        let head: u32 = 1 | 8 << 2 | 3 << 10;
        bytes.extend(head.to_le_bytes());
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(2u32.to_le_bytes());
        assert_eq!(count(&bytes), Err("an abbreviated record"));
        let truncated = {
            let mut m = Module::default();
            let void = m.void();
            let main_ty = m.func_ty(void, vec![]);
            m.function("main", main_ty, None, true);
            let b = m.new_block();
            m.terminate(b, Term::Ret);
            let mut w = m.write();
            w.truncate(w.len() - 8);
            w
        };
        assert_eq!(count(&truncated), Err("truncated bitcode"));
    }
}
