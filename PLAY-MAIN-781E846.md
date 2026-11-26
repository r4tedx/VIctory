# play/main-781e846: play/main's parts on upstream main 781e846d

This branch is upstream main **781e846d** plus the parts of fork `play/main`,
for an A/B against play/main (75b3f75b exe) at the same save and with the same
switches. Tests are synthetic. The title was not used in them.

It was built in two stages:

1. **`play/main-13f91ca`** (fork, head `bc0685f8`): main 13f91cae, then play/main's
   parts in play/main's order. Each part is its own merge (upstream PRs, at the
   head play/main had or at the author's rebase onto 13f91ca/67e8ae1), cherry-pick
   (fork commits), or applied change (play/main merges whose source branch sits
   on the layer-4 stack: `git diff <play/main first parent> <merge>`). Every
   commit message names its source and its conflicts.
2. **`play/main-781e846`** (fork, head `77093be0`, plus this file):
   `play/main-13f91ca` with upstream main 781e846d merged on top (one merge,
   `77093be0`). This retargeting was asked by the research thread because the
   branch was not built yet when main moved. Upstream main has since moved on
   (842b0cb). This branch stays on 781e846d.

`git log --first-parent 13f91cae..play/main-781e846` lists every step.

## Build

- `cargo build --locked --profile diag -p victory --features diagnostics`, with
  `CARGO_TARGET_DIR=C:\Users\PC\.t3\targets\play-main-13f91ca`. The target dir was
  seeded by copying `targets\play-main`.
- At `77093be0`: `targets\play-main-13f91ca\diag\victory.exe` (copy
  `victory-77093be0.exe`), sha256
  `3fcf97d918aba4d58046093b73ac9474250580450c0782927674b9466fa3d95b`, built
  2026-11-26 03:54:30. The seeded exe was play/main's `aa3689e9…`.
  victory-shader, victory-gpu, victory-hle and victory were recompiled
  (victory-shader rlib 03:54:08).
- At `c2fbf277` (#2629 on top, victory crates cleaned first, and victory-shader,
  victory-gpu, victory-hle and victory recompiled): `diagictory.exe` (copy
  `victory-c2fbf277.exe`), sha256
  `5711e32beda7a33b6ed378d0058c74b1530157fcb6210f60dd4d2b3d6febdb1f`, built
  04:39:07. This exe adds only the opt-in VICTORY_STALE_CHECK diagnostics over
  `victory-77093be0.exe`.
- `cargo clippy --locked -p victory -p victory-gpu -p victory-shader --all-targets
  --features victory/diagnostics,victory-gpu/diagnostics -- -D warnings`: clean at
  `bc0685f8` and at `77093be0`.
- **victory-shader differs from play/main.** This branch has main's translator
  changes, main's #573 bpermute lowering, #2143 and main's read marks, so the
  shader cache key changes. A cache pass is needed before a timed run.

## Parts carried (in order)

| Part | Source used | Notes |
|---|---|---|
| #158 depth bounds | author head 1a45c5e3 (49dd0808 + a test drop, main merged in) | |
| #159 queries | b7249e58 (= play/main) | its read-back renamed `query_readback` (main added `HostList::readback`) |
| #2111 pipeline wait budget | 0ef11e5e (= play/main) | its figure moved into main's report block at 781e846d |
| #1762 raw structured views | 4eb5d1ed (550cac2b, our c43e934c, 57aa0549, f6999942) | test adapted to main's shared view buffers (6cc6aa35) |
| #2027, #1315, #2134, #2115, #2147, #1428, #2043, #1751, #2180, #1778, #2129, #454 | play/main's heads | #2147, #1778: main has them since 781e846d (main's version kept) |
| #1763, #1133, #1906, #1849, #2051, #1933, #2098, #1978, #2167, #2085, #2168 | author rebases onto 13f91ca/67e8ae1 | #2168's rebase is on main's list-owned clamp arguments (315cea18). #1906: main's since 781e846d |
| wave64 stack (#223 #233 #320 #439) | 5a11cf48 (= play/main) | combined with main's uniform joins, lane-mask numbers and quad helpers; emit/lower.rs test modules moved to `emit/lower/tests.rs` and `select_tests.rs` (the file passed the 512 KiB asset-policy limit) |
| d1530bcb, a3fee8e2 | play/main first-parent commits | VICTORY_VIEW_HEAP_CAP; `translate --cache --host-lanes` |
| #1819, #2118, #2175, #2242, #2276, #2173, #2163, #2031, #2311, #2143, #2245, #2294, #2295, #1152 | play/main's heads | #2031, #2143, #1152: main's since 781e846d |
| #2269 last-dword marks | a8bc74c8 | applied to main's two-bitmap copy_back_marks (main still cleared the whole last dword) |
| batched clamps (VICTORY_BATCH_CLAMPS) | fork 1bf7c72a..31047d78 as one change | on main's list-owned clamp arguments: without a window, execute_indirect is main's clamp (+#2168) |
| #2619/#2620 stand-ins | author rebase af1973b2 | includes the clears-share-a-slot fix (= 5522be22) and 67320f31 (VICTORY_AMBIENT after an indirect dispatch clears only where records ran: new vs play/main, opt-in) |
| wave-per-group + #2306 | play/main merge b12f7683 as one change | the rewrite of multi-wave indirect dispatches takes no clamp; rect.rs 15ac13f1 left out (see below) |
| library saves, dynamic slots (a6cd2b68) | play/main merges 83a7b32a, abe0ed49 as changes | dynamic-slot drop reasons are main's |
| **921df625 fresh tables from a list run** | play/main 99d3e306 (= 7d9a8467) | as the GPU thread asked: cut indirect drops 1,584 to 310 a frame on play/main |
| memory cuts | 4e54ef04, 716cfb9f, 3a09a080 | the process-memory line counts no texture staging apart (main pools staging) |
| cee9e495 | fork | outside code ends the batched-clamp window |
| #2222 with #2130/#2120 | author rebase 4495caf0 | combined with #2242 as on play/main (mirror freshened before, pushed after each copy-back) |
| #2389 levels on demand | 29f572aa (same three commits, main merged in) | |
| DispatchX counter and fix | d7656117, 38d7330c, 7992d7ad, 7872cb40 | tests on main's clamp API |
| BVH 32-lane guard | 4087079c | |
| #2136 remainder | new commit bc0685f8 | f85763e0's two race fixes main lacked at 13f91ca (main merged its own as #2455/#2239 after 781e846d) |
| Build fix-ups | 1d716df8 | places where parts written for different bases meet |

## Left out, and why

| Part (play/main) | Superseded by / reason |
|---|---|
| #1989 UAV flags every stage (0f11320b) | main a1a2077a |
| #1883 traversal lane masks (a31d8ccb) | main e8e9f0c2 (+ff474638): the same programs, refused at per-lane joins on purpose |
| #2136 937f329a and f85763e0's first part | main 19533398, cc0e153b, a3b0f36e |
| #1962 heap-use marks (0c791faa…) and our marks-once-per-wave fix (97bb0e1c) | main's read marks 52d14b10, 2d6c0dfc, 80d8cf53, a1a2077a |
| #1962 memory set: #1873, #2221, #2161, 47bb09c9 | main 97721d1d/8f2a6f14, a7381ed6, 69d1e8e7, fdf34ab7/8a3d9759 |
| 740abe89 staging sweep | tunes 47bb09c9. Main has no per-texture staging buffers |
| c0be5b2f snapshot cap | main 2d4aa557 (identical) |
| 360b8557 quad helpers | main 4015a182 |
| a54e0c84 rect #2366, and 15ac13f1 | main 8ff79135 (#2366 closed for it) |
| #1735 GPU cost by pipeline (8ec5de1a, e92714a3) | main 1365fe11 + ed887f24 (VICTORY_GPU_TIMING per-pipeline table). VICTORY_NAME_SHADERS is not carried |
| #2280 | reverted on play/main too |
| **#2088** 256 KiB loose windows | dropped on the GPU thread's A/B: host copies ~0.55 to ~6 GB a frame, preparing 377 to 981 ms. **Main 781e846d does not carry #2088.** |
| 26fd2190 VICTORY_RT_MISS diagnostic | opt-in diagnostic. It conflicts with main's BVH lowering (moved to lower/bvh.rs) and shader-cache key. Not ported |
| #2150 | not in play/main either. Main's 286a00e9/850b66ff are in. #2150's remaining value was not re-measured |
| async #1923 | not in play/main. Not added |

## 781e846d overlaps (decisions)

- **Copy-back write-back: #2054 (main 4106550f) vs #2242.** #2242's path stays,
  and the branch keeps one write-back path. #2242 writes back only the runs of
  bytes the GPU changed, so a title write that lands on the page during the
  copy-back survives. #2054 merges the title's and the GPU's bytes and writes the
  whole piece back. That is faster, but it writes over such a title write.
  Its `merge_writeback` (and its test) is removed, because nothing calls it.
- **Dirty bitmaps: #1865 (main cfc23447) vs #2175.** #2175's video-memory bitmaps
  stay: on NVIDIA, the system-memory bitmap's atomics removed the device.
  #1865's pool of system-memory bitmaps and its pool tests are removed. Its
  `has_missed` helper and its skip of zero words are used.
- **Wave64 bpermute halves: #573 (main 0665a0e8) vs the wave64 stack's ab7882de.**
  Main's lowering is kept, with its tests.
- **Library save: #2007 (main 321a4bc1).** #2129's mapped-file write, which
  allocates nothing the size of the library, stays with #2180 and library saves.
- **Shader test host: #244 (main bde8e7de).** Main's shared, serialized WARP
  device is kept, and the wave64 stack's `Host::on` (VICTORY_ADAPTER) is added
  on top of it.
- **Main's squashed copies of our parts** (#2147, #2143, #2031, #1906, #1778,
  #1152, #1763): main's version is kept where the two differed.

## Switches

The same as play/main (PLAY-MAIN.md): VICTORY_GPU_UPLOAD(+_WRITES),
VICTORY_LEVELS_ON_DEMAND, VICTORY_DISPATCH_FLAGS_LOG, VICTORY_DISPATCH_THREADS,
VICTORY_BATCH_CLAMPS, VICTORY_AMBIENT, VICTORY_STAND_IN, VICTORY_VIEW_HEAP_CAP,
VICTORY_NO_WATCHDOG. VICTORY_GPU_TIMING is now main's.

## WARP tests at 77093be0

Commands:
- `VICTORY_D3D12_WARP=1 cargo test --locked -p victory-gpu --features diagnostics
  --lib -- --test-threads=1`
- `VICTORY_WARP=1 cargo test --locked -p victory-shader --lib --
  --test-threads=1 --skip split_tests --skip lane_read_loop_tests --skip
  each_adapter_gets_what_it_runs`

Both ran through heavy-run, once each.

**victory-gpu: 333 passed, 5 failed, 8 ignored.** The failures:
- `memory::tests::marked_pages_go_back_after_each_run` (#2175's test)
- `query::tests::query_resolves_after_unordered_access_to_destination` (#159's)
- `indirect::tests::batched_clamps_leave_what_clamps_one_by_one_leave` (batched
  clamps, compared with main's clamp)
- `draw::tests::clears_of_one_view_in_a_list_share_a_dynamic_slot` (the stand-in
  branch's test, which counts slots one by one; a6cd2b68 takes runs)
- `descriptor::host::tests::placed_uav_textures_create_real_host_uavs` (main's,
  from #1273)

Main 13f91cae's baseline from another thread (mem-vram, same command, 3 runs):
198/0, 197/1 and 197/1, with the flakes
`memory::tests::a_submission_that_missed_leaves_the_copies_at_once` and
`descriptor::host::tests::a_slot_rewritten_while_its_view_is_prepared_views_the_new_record`.
This branch has 338 tests against 199.

**victory-shader: 316 passed, 7 failed, 3 ignored, 41 filtered.**
- Three are known WARP failures on main 13f91ca (reported on #2724):
  `a_mask_joined_with_data_at_a_uniform_loop_is_its_number`,
  `gathers_at_a_level_match_the_hardware`,
  `permutes_read_the_lane_named_and_zero_from_lanes_left_out`.
- Four are new here:
  - `emit::counts::tests::counts_what_the_writer_wrote` (#2115)
  - `emit::lower::tests::a_mask_joined_with_data_needs_a_uniform_join` and
    `emit::lower::tests::a_scalar_result_needs_no_exec` (main's tests)
  - `emit::synthetic_tests::permutes_mask_high_address_bits_and_divergent_branches`
    (main's #573 test)

  These are main's tests meeting the wave64 stack's lowering, or #2115 meeting
  main's writer.

Each failure was run again alone (`--exact`, WARP):
- `marked_pages_go_back_after_each_run` **passes alone**. It is an
  order-dependent failure in the full run, not reproduced.
- `batched_clamps_leave_what_clamps_one_by_one_leave`: "execution 5 (kind 2,
  count Some(1))". The batched clamp leaves records past a distinct count
  as written (31047d78 honours counts). Main's clamp clamps every record up to
  the most. Records past the count do not run either way. This is a difference
  in the test's expectation, opt-in path only.
- `clears_of_one_view_in_a_list_share_a_dynamic_slot`: free slots 262143, not
  free - 1. a6cd2b68 takes dynamic slots in runs of 256, so the test's
  one-slot count does not hold on this line.
- `query_resolves_after_unordered_access_to_destination`: "the UAV dispatch
  wrote its draw arguments" is not seen. This is #159's test. Its
  indirect-argument setup assumed the in-place clamp; it was adapted to main's
  clamp for this branch, and the adaptation is still wrong.
- `placed_uav_textures_create_real_host_uavs`: "srv case 0": a placed SRV slot
  is not kept after `mirror(.., true)`. This is main's #1273 test. **Open**:
  not yet traced. It may be a real difference between this line's view-heap
  code and main's for placed views.
- victory-shader `a_mask_joined_with_data_needs_a_uniform_join` and
  `a_scalar_result_needs_no_exec`: lower() returns the wave64 stack's
  "a wave64 program whose halves exchange lanes (console flow)" retry. These
  main tests call lower() directly, where emit() would retry in the console
  flow: a test harness mismatch.
- `permutes_mask_high_address_bits_and_divergent_branches`: "no pipeline for a
  wave of 32" on WARP. WARP has no 32-lane waves (as for split_tests).
- `counts_what_the_writer_wrote` (#2115): the instruction counts differ from
  its expectation after main's DXIL writer changes (286a00e9, 850b66ff).

None of these was run on hardware. play/main's own suite results were never
recorded (PLAY-MAIN.md: "to follow").

## #2629 stale check (c2fbf277)

fork mem/stale-check bba19981 (upstream #2629, VICTORY_STALE_CHECK,
diagnostics builds only, off unless set) is taken as a **cherry-pick**, not a
merge. Its base, upstream f3578e38, is 24 commits past this line's 781e846d,
and a merge would bring those in. It needs none of them: on 781e846d it
builds, clippy is clean, and its WARP tests pass (`stale`: 2 passed;
`shadow::`: 23 passed). One conflict, in shadow.rs: #2130's test helper of the
same name is renamed `mirror_stale_pages`.

Two of those 24 upstream commits are main's own versions of fixes this line
has: #2455 (a page whose alias cannot be guarded stays changed) and #2239
(save a page's old bytes before any guard is opened). Both match bc0685f8's
port of #2136's f85763e0. They come in with the next retarget.

## Not in this branch yet

The follow-up PRs for main's new code are not merged yet:
- hold.rs: #2605, #2608, #2611;
- eviction: #2601, #2707, #2598;
- read marks: #2576, #2724, #2668;
- #2639, #2612;
- the texture worker pool #2786;
- extras: #2634, Belobug ff920f22;

Their review findings, as read for this branch:
- #2601: the pin race is fixed in its head 883db7ad.
- #2612: the remap race is fixed in d687c71.
- #2605: the VALVE stall risk still applies (an unreadable wait holds its queue
  up to 10 s; watch the `forced` count).
- #2608 + #2605 need `flush_after: &|_| false` in #2605's test.
- #2707 and #2598 change the same `HostHeap::evict` and overlap.
