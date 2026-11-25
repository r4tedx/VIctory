//! Console dispatch flags. `DispatchX(x, y, z, flags)` and the console's
//! single-pass indirect dispatch (signature type 105) carry flags with the
//! bit layout of the hardware's `COMPUTE_DISPATCH_INITIATOR` (Mesa's gfx10
//! register tables). Bit 5, `USE_THREAD_DIMENSIONS`, makes the counts
//! threads, not groups: the command processor launches ceil(threads /
//! group size) groups per dimension (and with bit 1, `PARTIAL_TG_EN`, the
//! last one with only the threads left, as radv programs unaligned
//! dispatches). The host's `Dispatch` takes groups, so such a dispatch
//! passed on as it is launches about group-size times too many groups.
//!
//! `VICTORY_DISPATCH_FLAGS_LOG=1` counts every `DispatchX` and every
//! indirect dispatch by shader and flags, and logs the counts with the
//! frame report (every 5 s).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::pipeline::Shader;

/// `COMPUTE_DISPATCH_INITIATOR.USE_THREAD_DIMENSIONS`.
pub(crate) const USE_THREAD_DIMENSIONS: u32 = 0x20;

/// The console's single-pass dispatch, as an indirect argument type.
pub(crate) const DISPATCH_SINGLE: u32 = 105;

/// Whether an indirect signature argument (type, then three fields) asks
/// for thread counts.
/// shortcut: where the flags sit in a type-105 argument is not known; the
/// first field is assumed. The counter logs every field so a run can tell.
pub(crate) fn indirect_thread_dims(argument: [u32; 4]) -> bool {
    argument[0] == DISPATCH_SINGLE && argument[1] & USE_THREAD_DIMENSIONS != 0
}

/// Groups per dimension for `threads` threads in groups of `size`.
pub(crate) fn groups(threads: [u32; 3], size: [u32; 3]) -> [u32; 3] {
    std::array::from_fn(|i| threads[i].div_ceil(size[i].max(1)))
}

fn logging() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_DISPATCH_FLAGS_LOG").is_ok_and(|v| v == "1"))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    /// The shader's address (shaders live as long as the process).
    shader: usize,
    indirect: bool,
    /// Direct: the flags, then zeros. Indirect: the signature's argument.
    words: [u32; 4],
}

struct Entry {
    hash: String,
    /// The console program's group size.
    size: Option<[u32; 3]>,
    calls: u64,
    /// Direct: the counts as given (first, smallest, largest), the groups
    /// they make as group counts, and as thread counts.
    first: [u32; 3],
    min: [u32; 3],
    max: [u32; 3],
    groups: u64,
    as_threads: u64,
    /// Indirect: the largest record count.
    records: u32,
}

static COUNTS: Mutex<Option<HashMap<Key, Entry>>> = Mutex::new(None);

fn note(shader: Option<&Shader>, indirect: bool, words: [u32; 4], counts: Option<[u32; 3]>, records: u32) {
    let key = Key { shader: shader.map_or(0, |s| std::ptr::from_ref(s) as usize), indirect, words };
    let mut all = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    let e = all.get_or_insert_with(HashMap::new).entry(key).or_insert_with(|| Entry {
        hash: shader.map_or_else(|| "none".into(), |s| s.hash.clone()),
        size: shader.and_then(Shader::compute_threads),
        calls: 0,
        first: counts.unwrap_or_default(),
        min: [u32::MAX; 3],
        max: [0; 3],
        groups: 0,
        as_threads: 0,
        records: 0,
    });
    e.calls += 1;
    e.records = e.records.max(records);
    if let Some(c) = counts {
        let product = |g: [u32; 3]| g.iter().map(|&n| u64::from(n)).product::<u64>();
        e.min = std::array::from_fn(|i| e.min[i].min(c[i]));
        e.max = std::array::from_fn(|i| e.max[i].max(c[i]));
        e.groups += product(c);
        if let Some(size) = e.size {
            e.as_threads += product(groups(c, size));
        }
    }
}

/// Counts a `DispatchX` of `counts` with `flags` (when logging).
pub(crate) fn note_direct(shader: Option<&Shader>, flags: u32, counts: [u32; 3]) {
    if logging() {
        note(shader, false, [flags, 0, 0, 0], Some(counts), 0);
    }
}

/// Counts an indirect dispatch of up to `records` records with signature
/// argument `argument` (when logging).
pub(crate) fn note_indirect(shader: Option<&Shader>, argument: [u32; 4], records: u32) {
    if logging() {
        note(shader, true, argument, None, records);
    }
}

/// Logs the counts since the last call, one line per shader and flags.
pub(crate) fn log() {
    if !logging() {
        return;
    }
    let Some(all) = COUNTS.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let mut all: Vec<_> = all.into_iter().collect();
    all.sort_by_key(|(k, e)| (k.indirect, std::cmp::Reverse(e.calls)));
    for (k, e) in all {
        if k.indirect {
            tracing::info!(
                hash = %e.hash,
                argument = ?k.words,
                thread_dims = indirect_thread_dims(k.words),
                calls = e.calls,
                group = ?e.size,
                records = e.records,
                "indirect dispatch flags (5 s)"
            );
        } else {
            tracing::info!(
                hash = %e.hash,
                flags = %format_args!("{:#x}", k.words[0]),
                thread_dims = k.words[0] & USE_THREAD_DIMENSIONS != 0,
                calls = e.calls,
                group = ?e.size,
                first = ?e.first,
                min = ?e.min,
                max = ?e.max,
                groups = e.groups,
                as_threads = e.as_threads,
                "DispatchX flags (5 s)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_counts_round_up_to_whole_groups() {
        assert_eq!(groups([1000, 1, 1], [64, 1, 1]), [16, 1, 1]);
        assert_eq!(groups([64, 1, 1], [64, 1, 1]), [1, 1, 1]);
        assert_eq!(groups([65, 9, 3], [64, 8, 2]), [2, 2, 2]);
        assert_eq!(groups([0, 12, 9], [64, 4, 4]), [0, 3, 3]);
        assert_eq!(groups([u32::MAX, 1, 1], [64, 1, 1]), [1 << 26, 1, 1]);
    }

    #[test]
    fn only_the_single_pass_type_with_bit_5_asks_for_threads() {
        assert!(indirect_thread_dims([DISPATCH_SINGLE, 0x20, 0, 0]));
        assert!(indirect_thread_dims([DISPATCH_SINGLE, 0x22, 0, 0]));
        assert!(!indirect_thread_dims([DISPATCH_SINGLE, 0, 60000, 0]));
        assert!(!indirect_thread_dims([2, 0x20, 0, 0]));
    }
}
