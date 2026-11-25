# play/main: the main fork build

`play/main` is the build to play and test on. Each tested improvement lands
here as one commit (a merge, or a cherry-pick), so any single one can be
reverted on its own (`git revert -m 1 <merge>`, or `git revert <commit>`). This file lists what is in it, where each part
came from, the switch that turns it on (if any), and the evidence for it.
The evidence is from the user's own supervised runs on an RTX 5070 (run
names such as L4Ma). Tests are synthetic: the title was not used in them.

Base: upstream main 940d32f, plus the "layer 4" stack (layer4-final 07db99ea,
upstream #1152 as layer4-final-b 562c2673, layer4-final-fixes 5522be22). The
layer 4 stack's own merge commits list what it holds.

**Current build:** `C:\Users\PC\.t3\targets\play-main\diag\victory-99d3e306.exe`,
sha256 `5847b7b3c244dced6f2282d2f47c18f0a9febdb0f829309f17a6d6b47316b5fd`
(the code of aeeaed59: later commits change only tests or this file).

## Merged parts, in order

| # | Part | Source branch @ head | Upstream PR | Switch | Merge on play/main |
|---|------|----------------------|-------------|--------|--------------------|
| 0 | Memory cuts: a process-memory line every 5 s, idle staging buffers released after 2 s and 2 frames (and after a frame past 512 MiB), snapshot pool capped at 512 MiB, depth-bounds pipeline lines | `play/layer4-memory-cuts` @ 54dda729 (base of this branch) | none (fork only) | none: always on; the memory line only in `diagnostics` builds | (base) |
| 1 | Video-memory copies of written title buffers with a mirror, upload-heap budget share, residency priorities (#2222 with #2130 and #2120); streamed textures take memory level by level (#2389); DispatchX flag counter and whole-group fix | `gpu-run/layer4-2130` @ 7872cb40 | #2222, #2130, #2120, #2389 (0xkn0wh3r3 and others); DispatchX counter and fix: fork only (DexNom co-author) | `VICTORY_GPU_UPLOAD=1` (upstream's), then `VICTORY_GPU_UPLOAD_WRITES=1` for copies that shaders write; `VICTORY_LEVELS_ON_DEMAND=1` (with `VICTORY_LEVELS_MIN_KB`), `VICTORY_DISPATCH_FLAGS_LOG=1`, `VICTORY_DISPATCH_THREADS=1`; batched clamps stay behind `VICTORY_BATCH_CLAMPS=1` | d1af3a61 |
| 2 | The BVH builder stays held back on hosts without 64-lane waves, even with every pipeline made | `play/bvh-builder-32lane-guard` @ 4087079c | none; guards upstream #2118's gate | none: on 32-lane hosts only | 1b9a9367 |
| 3 | GPU cost by pipeline in the 5 s report | `play/layer4-gpu-cost` @ 073c778e | #1735 (8ec5de1a, Nixitov) | `VICTORY_GPU_TIMING=1` (off by default; diagnostics builds); `VICTORY_NAME_SHADERS` | 404c7e0e |
| 4 | Loose title memory copied in 256 KiB windows instead of 4 MiB, with window reuse and a lookup past many smaller buffers | `play/layer4-memory-cuts-2088` @ 50cfd988 | #2088 (76cdef61, with #402's window reuse) | none: always on | 75b3f75b; **reverted** in 64136f63 (copy blow-up in the open world; the memory win is kept for a later, fixed form) |
| 5 | GPU cost by pipeline (part 3) is per frame of the 5 s it covers, not of the last 1 s window: about 3x too high before | play/main itself (found by the hot-programs delegate) | reported on #1735 (issuecomment-7174304614) | `VICTORY_GPU_TIMING=1`, as part 3 | e92714a3 (a plain commit) |
| 6 | Root views, fresh tables and unordered access clears take their dynamic slots from one run per list, so a list's slots lie together and come back together; the dynamic-slot line splits refused tables by cause | fork `play/layer4-dynamic-tables` @ 7d9a8467 (layer-4 port of `gpu/dynamic-slots-tables` 921df625) | none (fork only) | none: always on | 99d3e306 (a cherry-pick) |
| 7 | Test: clears of one view share one dynamic slot, counted out of the list's run (part 6 lends a list 256 slots at once) | play/main itself | none | none | aeeaed59 |

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
  With the switch unset, nothing is timed. **Correction:** the per-frame
  figures in that table are about 3x too high. #1735's report divided a
  5 s tally by the frames of the last 1 s window. The order of the items
  stands; their per-frame sizes do not. play/main e92714a3 fixes the
  report (found by the hot-programs delegate; see #1735's thread).
- **4, #2088, reverted:** in the user's same-save A/B in the safehouse,
  PM1 (75b3f75b, with #2088) against PM0 (404c7e0e, without it), #2088's
  256 KiB windows took host copies of title memory to about 6 GB a frame,
  against about 0.55 GB without. Preparing averaged 981 ms against 377 ms,
  and the median frame was 428 ms longer: PM0 - PM1 95% bootstrap CI
  [-569, -346] ms. It did save about 2.9 GB of memory. 64136f63 reverts
  it; the memory win is kept for a later, fixed form. Its author's lookup change
  b73bb8b4 (spikes and stretched triangles) is reverted on the branch
  (76cdef61 is the exact inverse of b73bb8b4). None of the names b73bb8b4
  added (`BIG_LEN`, `SMALL_SPAN`, `BIG`, `enclosing_split`, `holding`) is
  in this tree.
- **6, dynamic tables:** in the same runs, in the open world, about 1,000
  to 1,500 draws and dispatches a frame were dropped with "no dynamic
  slots for its tables" (the biggest share) and "no dynamic slots for root
  views". play/main already had a6cd2b68 (root views take a shorter run
  rather than drop), but not this. PM2 measured it: see below.

## Runs on play/main

The user's supervised runs on the RTX 5070, all from the same save in the
safehouse, each 600 s. Comparisons by the GPU thread's script; window
means of the 5 s meter lines; median deltas with 95% bootstrap CIs
(4,000 resamples per run). Full tables:
`C:\Users\PC\.t3\artifacts\gpu-runs\2026-11-25\PM0\compare-PM1-PM0.md`
and `...\PM2\compare-PM0-PM2.md`.

| Run | Build (exe sha256) | Frame ms mean / p50 / p95 | Preparing / GPU / copy-back ms | Drops/frame: no host pipeline / indirect |
|-----|--------------------|---------------------------|-------------------------------|------------------------------------------|
| PM1 | 75b3f75b, with #2088 (aa3689e9…1a46) | 1,860 / 1,482 / 2,822 | 981 / 196 / 132 | 814 / 1,189 |
| PM0 | 404c7e0e, without #2088 (db330702…51f5) | 1,218 / 1,054 / 2,419 | 377 / 183 / 126 | 1,064 / 1,584 |
| PM2 | 99d3e306, #2088 reverted + part 6 (5847b7b3…b5fd) | 847 / 851 / 918 | 232 / 225 / 121 | 400 / 310 |

- PM0 against PM1: median frame 428 ms shorter, CI [-569, -346] ms.
  #2088's copies cost about 600 ms of preparing a frame, so part 4 was
  reverted.
- PM2 against PM0: median 204 ms shorter, CI [-275, -144] ms. p95 is
  918 against 2,419 ms, and no stutter samples against 7%. Preparing is
  232 against 377 ms. GPU time rose 43 ms, because more is drawn: indirect
  drops fell from 1,584 to 310 a frame. No errors and no device removal.
- Still dropped in PM2, a frame: the BVH builder held back, about 327 (by
  design: 48 of 53 pipelines made); "no dynamic slots for its tables",
  about 197 (indirect kinds 2, 10 and 1 the most); compute shader
  unavailable, 107.

## Variants (branches off play/main, NOT part of it)

Each is play/main plus one commit (play/main-shaders: two merges), so each
part reverts alone. Each has its own target dir and build for A/B runs.
None of them has a title run yet.

| Variant | Adds | Switch | Evidence and risk |
|---------|------|--------|-------------------|
| `play/main-tables-vram` | fork `gpu/tables-in-vram` 2a2a602c (F1): the host lookup tables shaders read on memory accesses (page tables, view strides, view marks, view addresses) in video memory | `VICTORY_TABLES_VRAM=1` (a GPU upload heap where the host has one, else as `=copy`) or `=copy` (default heap, with copies at submissions); unset: system memory as before | Head c49ef675 on play/main aeeaed59 (first forms 001945dc, on 75b3f75b, and 46d71b97, on 99d3e306; neither built). F1 only, a cherry-pick of 2a2a602c, ported to layer 4's view-address layout (8 bytes for every title and dynamic slot, as play/main's shaders read them). Not merged with its base, upstream main da82e444: that would bring 81 upstream commits and a new shader cache key. victory-shader is unchanged, so the cache hits. Build, tests and runs: built into `targets\play-main-tables-vram` (fresh seed, workspace fingerprints cleared, recompiled; clippy clean): `diag\victory-c49ef675.exe`, sha256 `8ca1fda0ea58bec602aa6db5e202c844074c443b14e6dcc7bde60b0267d4afdb`. The first port missed four layer 4 uses of the old (buffer, mapping) tuples; c49ef675 ports them, with test-only `Table::read` and `Table::mapped`. WARP tests: `tables::`, `memory::` and `descriptor::host::` (22 tests) pass in each mode, `VICTORY_TABLES_VRAM` unset, `=1` and `=copy`: 22 / 0 / 0 each, `tables::tests::shaders_read_what_the_cpu_wrote_in_every_mode` among them. WARP has no GPU upload heaps, so `=1` runs as `=copy` there; the upload-heap form is untested until a run on the RTX 5070. |
| `play/main-shaders` | two merges: fork `shader/wave64-lds-groups` fa2dcfdd (a compute group of several console waves sharing LDS runs as one host group of 32-lane wave pairs), then fork `shader/implicit-sample-optin` 780fc968 (compute samples with implicit derivatives in the console flow in halves, at a level) | groups: none, on hosts without 64-lane waves; implicit samples: `VICTORY_IMPLICIT_SAMPLE_COMPUTE=1` | Heads 122182dd (after the groups merge) and f965f807 (after implicit-optin), on play/main 99d3e306; each merge reverts alone. Groups: offline 32-32 listing 25,725 -> 25,734 validated (+9 / -0), including all 7 remaining cascade-shadow programs. Implicit-optin: 32-32 +13 validated, 0 lost. victory-gpu `shader_cache::` and `pipeline::library` on WARP: 22 passed / 0 failed / 1 ignored (a measurement), `implicit_sample_changes_the_key_only_when_enabled` among them. Both change victory-shader, so the shader cache key changes (a new cache pass). Neither has run on hardware: WARP cannot run 32-lane waves, so whether LDS is seen across waves is unproven. With `VICTORY_IMPLICIT_SAMPLE_COMPUTE=1`, multi-wave LDS groups may take implicit samples too: the guard is one check. That combination is untested. `translate --cache --host-lanes` (a3fee8e2) is in. Build and tests: built into `targets\play-main-shaders` (fresh seed, workspace fingerprints cleared, every crate from victory-shader up recompiled; clippy clean for victory, victory-gpu and victory-shader with `host`): `diag\victory-f965f807.exe`, sha256 `2215a877adc6250e595ceaa24270b7a7e3ec6516f7b01dcda11ba2832c268692`. victory-shader `--features host --lib` on WARP: 325 passed / 2 failed / 3 ignored at 99d3e306, 331 / 2 / 3 at f965f807. The same 2 fail at both heads, both WARP-only: `gathers_at_a_level_match_the_hardware` (WARP's gather differs by 1) and `permutes_read_the_lane_named_and_zero_from_lanes_left_out` (no pipeline for a 32-lane wave on WARP). The 6 new tests pass: three groups split tests, `compute_samples_with_implicit_derivatives_are_at_a_level`, and two in emit::tests. `groups_sharing_lds_match_the_console_on_every_adapter` passes trivially on WARP (no hardware adapter is listed), so cross-wave LDS is still unproven. `translate --help` lists `--cache` and `--host-lanes`. `translate --cache --host-lanes 32-32` on an empty directory, with a scratch cache, exits 0 ("host waves 32-32 lanes ... 0 containers"). |
| `play/main-copyback` | fork `play/layer4-copy-back-spans` e6f9c522: copy-backs write their spans with plain stores (`rep movsb`, watch::store) instead of a kernel copy each (#2242's per-span cost, about 73 us a page in game); plus a fix of ours for pages with `PAGE_GUARD` | none | Head a4ed2b0e on play/main aeeaed59; one commit. Its safety argument holds: a faulting `rep movsb` can be restarted, and the handler skips it only for a write inside the stored range at that instruction; the kernel copy writes the rest, as before. The one gap, a guard page, raised STATUS_GUARD_PAGE_VIOLATION, which e6f9c522 did not take (a crash where the kernel copy failed quietly). It now puts the guard bit back and lets the kernel copy go on (test `stores_leave_guard_pages_as_the_kernel_copy_does`). The branch's later 513a525c is not taken. Built into `targets\play-main-copyback` (fresh seed, fingerprints cleared; clippy clean): `diag\victory-a4ed2b0e.exe`, sha256 `f01388f7ac87adeef37dbfa592db17b8c4f3d61113062a80b1099f43eb3d166e`. victory-gpu only, so the cache hits. WARP tests: `shadow::` and `watch::` 31 passed / 0 failed / 2 ignored, both store tests among them (`stores_leave_unmapped_and_read_only_pages_alone`, `stores_leave_guard_pages_as_the_kernel_copy_does`); the whole lib 278 / 0 / 10 (play/main's 276 plus the two). |

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
    for the copy-backs to walk. Runs PM0 and PM1 measured it, and part 4
    was reverted (see "Runs on play/main").
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

victory-gpu `--lib` tests on WARP (`VICTORY_D3D12_WARP=1`, `--features
diagnostics`, `--test-threads=1`, through heavy-run), at each head:

| Head | Passed | Failed | Ignored |
|------|--------|--------|---------|
| 404c7e0e (parts 0 to 3) | 275 | 0 | 10 |
| 75b3f75b (with part 4, #2088) | 280 | 0 | 10 |
| 99d3e306 (part 4 reverted, part 6) | 275 | 1 | 10 |
| aeeaed59 (part 7) | 276 | 0 | 10 |

- The 5 more tests are #2088's own, all in shadow.rs:
  `a_parent_is_found_past_many_children`, `loose_memory_must_be_committed`,
  `loose_windows_are_reused_for_ranges_inside_them`,
  `overlapping_windows_agree_through_title_memory` and
  `registered_buffers_widen_the_search`. No other test was added or lost,
  and none fails at those two heads.
- The 10 ignored tests are benchmarks and measurements, marked so in the
  code, at every head.
- The run used the WARP device: the tests that unwrap `host()`
  (bindless, descriptor::host) ran and passed.
- The stand-in counter flake (fixed on #2619 only) did not show in any
  run. The three stand-in tests passed every time. That is one run per
  head, so it does not rule the flake out.
- At 99d3e306 the 5 tests of #2088 are gone with it, and part 6 adds
  `tables_of_two_lists_come_back_as_one_run` (passed). The one failure
  was `clears_of_one_view_in_a_list_share_a_dynamic_slot` (from
  5522be22): "eight clears of one view, one slot", left 261888, right
  262143. The heap's free count fell by 256, not 1, because part 6 lends
  a list a whole run of 256 slots (`LIST_RUN`) at the first slot it
  takes. The clears still share one slot. Part 7 counts the slots taken
  out of the list's run instead, and checks that the rest of the run
  comes back when the list closes.
- e92714a3 (the GPU-cost report fix, below) changes only a diagnostics
  log line in `Meter::frame`. Its build ran clippy clean; the lib tests
  were not run again for it.

## Build

`cargo build --locked --profile diag -p victory --features diagnostics`
with `CARGO_TARGET_DIR=C:\Users\PC\.t3\targets\play-main` (seeded from
an earlier layer 4 target dir). Every crate from victory-shader up to
victory was recompiled.

- At 75b3f75b: `targets\play-main\diag\victory-75b3f75b.exe`, sha256
  `aa3689e98bc2d9cb23559caa0d97041f5605e9d9394b828295108be231221a46`
  (the build the user's run PM1 used).
- At e92714a3 (with part 5): `targets\play-main\diag\victory-e92714a3.exe`, sha256
  `3c718b1f6bc083a7d9e220df6d859bb9590ca3f1f0bad264b00a8c5aef175646`.
- At 99d3e306 (part 4 reverted, part 6): `targets\play-main\diag\victory-99d3e306.exe`
  (also `diag\victory.exe`), sha256
  `5847b7b3c244dced6f2282d2f47c18f0a9febdb0f829309f17a6d6b47316b5fd`.
  victory-gpu, victory-hle and victory were recompiled. Part 7 changes
  only a test, so this exe stands for it too.
  Commits that change only this file or tests leave the exe as it is.
- `cargo clippy --locked -p victory -p victory-gpu --all-targets --features
  victory/diagnostics,victory-gpu/diagnostics -- -D warnings`: clean at
  every head built.
