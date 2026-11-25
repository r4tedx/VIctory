# play/main: the main fork build

`play/main` is the build to play and test on. Each tested improvement lands
here as one merge commit, so any single one can be reverted on its own
(`git revert -m 1 <merge>`). This file lists what is in it, where each part
came from, the switch that turns it on (if any), and the evidence for it.
The evidence is from the user's own supervised runs on an RTX 5070 (run
names such as L4Ma). Tests are synthetic: the title was not used in them.

Base: upstream main 940d32f, plus the "layer 4" stack (layer4-final 07db99ea,
upstream #1152 as layer4-final-b 562c2673, layer4-final-fixes 5522be22). The
layer 4 stack's own merge commits list what it holds.

## Merged parts, in order

| # | Part | Source branch @ head | Upstream PR | Switch | Merge on play/main |
|---|------|----------------------|-------------|--------|--------------------|
| 0 | Memory cuts: a process-memory line every 5 s, idle staging buffers released after 2 s and 2 frames (and after a frame past 512 MiB), snapshot pool capped at 512 MiB, depth-bounds pipeline lines | `play/layer4-memory-cuts` @ 54dda729 (base of this branch) | none (fork only) | none: always on; the memory line only in `diagnostics` builds | (base) |
| 1 | Video-memory copies of written title buffers with a mirror, upload-heap budget share, residency priorities (#2222 with #2130 and #2120); streamed textures take memory level by level (#2389); DispatchX flag counter and whole-group fix | `gpu-run/layer4-2130` @ 7872cb40 | #2222, #2130, #2120, #2389 (0xkn0wh3r3 and others); DispatchX counter and fix: fork only (DexNom co-author) | `VICTORY_GPU_UPLOAD=1` (upstream's), then `VICTORY_GPU_UPLOAD_WRITES=1` for copies that shaders write; `VICTORY_LEVELS_ON_DEMAND=1` (with `VICTORY_LEVELS_MIN_KB`), `VICTORY_DISPATCH_FLAGS_LOG=1`, `VICTORY_DISPATCH_THREADS=1`; batched clamps stay behind `VICTORY_BATCH_CLAMPS=1` | d1af3a61 |
| 2 | The BVH builder stays held back on hosts without 64-lane waves, even with every pipeline made | `play/bvh-builder-32lane-guard` @ 4087079c | none; guards upstream #2118's gate | none: on 32-lane hosts only | 1b9a9367 |
| 3 | GPU cost by pipeline in the 5 s report | `play/layer4-gpu-cost` @ 073c778e | #1735 (8ec5de1a, Nixitov) | `VICTORY_GPU_TIMING=1` (off by default; diagnostics builds); `VICTORY_NAME_SHADERS` | 404c7e0e |
| 4 | Loose title memory copied in 256 KiB windows instead of 4 MiB, with window reuse and a lookup past many smaller buffers | `play/layer4-memory-cuts-2088` @ 50cfd988 | #2088 (76cdef61, with #402's window reuse) | none: always on | 75b3f75b (last, so it reverts alone) |

## Evidence

- **0, memory cuts (L4Ma, exe at 54dda729):** ran the full 600 s with no
  memory-guard stop, 0 errors, panics or device removals. Last 12 lines
  771 ms a frame (1.30 fps): prep 183, GPU 322, copy-back 189 ms. First run
  to reach player control. Its memory line showed loose windows at
  5,520 MB of 6,558 MB of host copies, which is why part 4 follows.
- **1, layer4-2130 (L4Ha, L4Hb):** L4Ha (DispatchX fix off) last 12 lines
  892 ms; first fully correct-looking world frames; no errors or removals.
  L4Hb (`VICTORY_DISPATCH_THREADS=1`) GPU 345 to 339 ms: no speed change.
  The fix is for correctness (thread counts launch whole groups), not
  speed. L4Ha's counter found about 63.5 M surplus groups a run on one
  shader with thread-count records.
- **2, 32-lane guard:** on 32-lane hosts the builder's wave64 programs run
  as 32-lane halves; the whole builder hung an RTX 4070 SUPER within two
  minutes (#2118 thread, moe-miau; device removed in its submission). In
  the user's runs the builder was held back anyway, because not every
  pipeline was made (made=48 refused=5 in L4Ma). The guard keeps it held
  back on such hosts once all 53 are made too, with its own log line ("held
  back: the host has no 64-lane waves"). No title run has reached made=53
  yet, so the guard has not been seen to act there.
- **3, #1735 (L4Ca):** ran the full 600 s with `VICTORY_GPU_TIMING=1`,
  0 errors or removals. Frame times are not valid in that mode, because
  each list is timed on its own. Its table puts compute far ahead of draws
  (top: one 788-group dispatch at 109 ms a frame, batched clamps 62 ms).
  With the switch unset, nothing is timed.
- **4, #2088:** run L4Mb is to test it. Its author's lookup change
  b73bb8b4 (spikes and stretched triangles) is reverted on the branch
  (76cdef61 is the exact inverse of b73bb8b4). None of the names b73bb8b4
  added (`BIG_LEN`, `SMALL_SPAN`, `BIG`, `enclosing_split`, `holding`) is
  in this tree.

## Variants (branches off play/main, NOT part of it)

Each is play/main plus one commit, so it reverts alone. Each has its own
build for A/B runs.

| Variant | Adds | Switch | Evidence and risk |
|---------|------|--------|-------------------|
| `play/main-tables-vram` | fork `gpu/tables-in-vram` 2a2a602c (F1): the host lookup tables shaders read on memory accesses (page tables, view strides, view marks, view addresses) in video memory | `VICTORY_TABLES_VRAM=1` (a GPU upload heap where the host has one, else as `=copy`) or `=copy` (default heap, with copies at submissions); unset: system memory as before | Branch head 001945dc. F1 only (a cherry-pick of 2a2a602c), ported to layer 4's view-address layout (8 bytes for every title and dynamic slot); not merged with its base, upstream main da82e444, which would bring 81 upstream commits and a new shader cache key. Build, tests and runs: to follow. |
| `play/main-groups` | fork `shader/wave64-lds-groups` fa2dcfdd: a compute group of several console waves sharing LDS runs as one host group of 32-lane wave pairs | none: on hosts without 64-lane waves | To follow. Changes victory-shader, so the shader cache key changes (a new cache pass). Never run on hardware: WARP cannot run 32-lane waves, so whether LDS is seen across waves is unproven. |

## Merge notes

Each merge commit's message lists the same details in full.

- **Already in, so skipped:** nothing. Before each merge, `git merge-base
  --is-ancestor` showed that none of the four source heads was in play/main.
  Parts 2 and 3 are each layer4-2130 plus their own commit(s), so after
  merge 1 they add only those. Part 4 is memory-cuts plus #2088, so it adds
  only #2088.
- **Merge 1 conflict (shadow.rs, `Drop for Shadow`):** both sides kept. The
  memory cuts' counters (`procmem::SNAPSHOTS_IN_USE`, `procmem::COPIES`)
  and layer4-2130's `COPY_BYTES` (the budget's report) are each taken off.
  Their increments in `Shadow::new` had merged cleanly. Both counters stay:
  the 5 s memory line reads one, and the VRAM budget reads the other.
- **Merges 2 and 3:** no conflicts.
- **Merge 4:** no conflicts, plus one fix in the merge commit itself.
  #2088's test helper `entry()` now sets `read_only: false`, the `Entry`
  field layer 4 added for #2120's read-only copies. Without it, 50cfd988's
  own tests do not build (E0063).
- **Overlaps git merged without conflict, checked by hand:**
  - shadow.rs: #2130's video-memory copies with a mirror (stale runs,
    `freshen`) never touch the snapshot pool, so the memory cuts' snapshot
    accounting and 512 MiB cap cover every snapshot. #2088's `enclosing()`
    replaces the 64-entry scan in `find()`. Its `MAX_SPAN` is raised in
    `register_as()`, which `register()` and #2120's `register_read_only()`
    both go through. `LOOSE_SPAN` is raised at the one insert into `LOOSE`.
    The memory cuts' `held_copies()` reads `LOOSE`'s `(start, len)` keys,
    which #2088 keeps. The copy-backs (#2242, from layer 4; #2280 is not
    in, because layer 4 reverted it in e9159245) work on a copy once it is
    found, so the new lookup changes which copy is found, not how it is
    copied back. The 256 KiB windows do mean more and smaller loose copies
    for the copy-backs to walk. That is what run L4Mb is to measure.
  - indirect.rs: layer4-2130 adds the DispatchX `threads` argument (with
    #2168's clamps and the batched clamps already in layer 4). The memory
    cuts add only a test (outside code ends the batched-clamp window), and
    it calls neither `dispatch()` nor `execute_indirect()`.
  - host.rs `Meter`: the memory cuts' `procmem::log()`, layer4-2130's
    tiles and threads logs, and #1735's `costs::log()` all stay.
- **victory-shader:** `git diff --stat fork/gpu-run/layer4-2130 HEAD --
  crates/victory-shader` is empty. play/main translates shaders exactly as
  layer4-2130 does, so a shader cache from layer4-2130 builds still hits.

## Tests

victory-gpu `--lib` tests on WARP (`VICTORY_D3D12_WARP=1`, `--features diagnostics`, `--test-threads=1`), at 404c7e0e (before part 4) and at 75b3f75b: results to follow in a later commit to this file.

## Build

`cargo build --locked --profile diag -p victory --features diagnostics`
with `CARGO_TARGET_DIR=C:\Users\PC\.t3\targets\play-main` (seeded from
an earlier layer 4 target dir). Every crate from victory-shader up to
victory was recompiled.

- At 75b3f75b: `targets\play-main\diag\victory.exe` (copy:
  `victory-75b3f75b.exe`), sha256
  `aa3689e98bc2d9cb23559caa0d97041f5605e9d9394b828295108be231221a46`.
  Commits after it that change only this file leave the exe as it is.
- `cargo clippy --locked -p victory -p victory-gpu --all-targets --features
  victory/diagnostics,victory-gpu/diagnostics -- -D warnings`: clean.
