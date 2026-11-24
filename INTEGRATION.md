# Tester integration branch

This branch puts many open upstream pull requests together in one tree, so
that testers who run the title can build once and measure the fixes
together. It is **not an official release**. The maintainer has not
reviewed or merged any of these PRs, and each may still change or be
turned down upstream.

- Base: upstream `main` at `ffe310e` since update 7 (merged in as one
  commit; before that `d2a4bae`). Mentions of `d2a4bae` below describe
  runs and merges made on the old base.
- Every PR comes in as its own merge commit (`git log --first-parent`), so
  you can find, revert or leave out each one.
- Before inclusion, every PR was built and tested on one Windows 11
  machine (NVIDIA RTX 5070, 32-lane waves only; WARP). The title was not
  used in these tests: the integration machine has not run this tree with
  the title. One tester has run an earlier head of it with the title (see
  "Reports from the title"); nobody has reported a run of this head. It
  makes no claim about results in the title.
- Update 3 adds #493, #483, #206, #469, #459, #454 and #579, and brings
  the wave64 stack up to #439's head `e4d899f`.
- Update 4 adds #566 (with the rest of #307) and brings the wave64 stack
  to #439's head `055aef9`.
- Update 5 adds #453 (pixel barycentrics), with one fix commit after it.
- Update 6 adds #587 (mesh output bounds) with tests, and the switch
  `VICTORY_HEAP_GUARD=0`. **On AMD cards read "Known problem on AMD"
  below before running this tree.**
- Update 7 takes upstream `main` `ffe310e` in (116 commits, #566 among
  them, which is no longer a PR this tree adds), and adds
  r4tedx/VIctory#10 (`VICTORY_PRESENT_1440P`) and a fix for
  level-of-detail queries of array textures. **Fewer programs translate
  than at update 5, on purpose: see "What update 7 costs" below.**
- Update 8 adds geometry programs: the title's 195 single-stream
  triangle geometry shaders translate and validate (two commits written
  for this tree, see "Changes made in this branch"). Nothing else
  changes in the offline count.
- Update 9 adds the black-scene work: #704 (unknown-stride buffers,
  on #193), #489 (root arguments from the spill table), #294 (root
  UAVs) and #346 (indirect rectangle draws). **With it the title's
  instance-culling compute programs translate for the first time: read
  "What update 9 changes at run time" before comparing runs.**
- Update 10 adds wave64 compute groups of more than 64 threads on
  32-lane cards (one host group per console wave; direct dispatches
  only). 20 more compute programs validate, most of them cascade-shadow
  programs; none is lost.

## What update 9 changes at run time

Counted offline over the title's shaders (same method as below): 205
more programs validate than at update 8 and none is lost (24,933 of
25,889 for a 32-lane host). `a structured buffer of unknown stride` goes
from 180 programs to 0, and 25 compute programs get past `a program the
console structure cannot hold`. Of the 60 programs with "Cull" in their
name, 60 validate (30 before): the 30 `InstanceCulling` compute programs
were all stopped at unknown stride.

Those programs have not run in the title on this tree. One tester's own
port of #704 onto this tree at update 5 (likecrank-dotcom, on #704, RTX
5060 Ti) had four of them translated and with pipelines in the title,
and the run ended in `DEVICE_HUNG` after 111 s on a compute dispatch,
where the tree without #704 held 600 to 900 s on other cards. That is
one run of another integration, taken before the zero-element guard
below, so it shows a direction to check, not a fault of this update.

A second report, on an RTX 5080 (MichonGoddijn231849, issue #841): an
older branch with the same additions as update 9, on this tree at update
5, hung about a minute into the world at a mesh dispatch (mesh
`51ff151f75cd5759`, pixel `7fe75645c406cac6`), where the tree at update
5 alone ran about six minutes. Both programs translate without update
9's additions, so it is not known which addition brings the hang back.
Update 9 has not been run there.

If a run at update 9 ends sooner than one at update 8:

- compare with the head before it, `da605bc`;
- run once with `VICTORY_SKIP_PIPELINES=<hashes>` naming the compute
  programs the log shows as newly translated (the four that tester saw:
  `17af15c4015cb4b6,55679a0547649d0b,fdd56d97a806bcca,89d08e561897d4e5`),
  and say whether the run then lasts as before.

## What update 7 costs

Counted offline over the title's shaders (`victory translate --host-lanes
32-32 --validate --list`, 25,889 programs, program by program): 24,746
validated at update 5 and 24,533 at update 7. 221 that validated stop
now, and 8 are new (the 3 pixel programs of the level-of-detail fix, 5
compute programs through `main`'s rule for SGPRs the hardware does not
initialize).

All 221 come from one commit of `main` (`3dd467c`, "a phi that only
another read phi consumes is live too"): with that rule alone switched
off in an experimental build, the same count loses none. Before it, a
value with no host form (a lane mask read as data, a descriptor word, the
first half of an interpolation) that reached a read through two joins was
taken for dead and replaced by zero; now the program is refused with that
value's reason. So these programs did translate, with a zero where the
console has a value, and how many of them drew right is not known. None
of the 221 validates on plain `main` either. By reason: `a lane mask
read as data` 108, `a structured buffer of unknown stride` 57, `the
first half of an interpolation` 20, `a program the console structure
cannot hold` 17, others 19.

If a run at update 7 shows less than one at update 5 or 6, this is the
first thing to suspect, and it is worth reporting.

## Known problem on AMD (update 3 and later)

A tester (nullmarsh, issue #293) reports that on an RX 9060 XT (driver
32.0.32029.2010) the title dies about two seconds into the world with an
access violation inside `amdxc64.dll`, in 3 of 3 runs at update 4, from
this tree's merge of #483 (`7ef2c41`) on. With that merge reverted,
update 5 ran 364 s in the world without a device loss. NVIDIA runs of
this tree do not show it. The cause is not known. #483 merged another
way onto `main` `ffe310e` ran 453 s on an RX 9070 XT with the same
driver (h0useadv, on #483), so the suspects are what this tree's merge
adds: the guard on #307's per-lane heap index, and where the guard slot
sits.

- **To run on AMD:** `git revert -m 1 7ef2c41` (it reverts cleanly at
  update 5, per that report).
- **What is known since (nullmarsh, update 6):** the fault is in
  `CreateComputePipelineState`, so it is AMD's compiler on the shaders
  this merge changes, not a read of the guard slot. `VICTORY_HEAP_GUARD=0`
  does not help, as expected for a fault at pipeline creation. It moves
  to another compute program when one is left out (three were named).
  An RX 6750 XT ran update 5 for 10 minutes without it (mpDom, issue
  #841), so it is not every AMD card or driver.
- `VICTORY_HEAP_GUARD=0` stays as a switch: the same shaders, with
  `u32::MAX` passed as the guard, so no index is redirected.
- Looked at offline, the three programs are variants of one very large
  compute program from the wave64 console flow (about 830 KB of DXIL,
  1,480 blocks and 735 heap handles each), and the guard puts a compare
  and a select in front of every handle. **To try (since `da605bc`):**
  `VICTORY_HEAP_GUARD_FORM=umin` bounds the index with one `UMin`
  instead (the same value), and `VICTORY_HEAP_GUARD_FORM=off` emits no
  bound, as before #483. It is read when a program is translated, so
  each run needs a shader cache of its own. Both forms pass #483's GPU
  test here and validate the same programs; whether either gets past
  AMD's compiler is not known.

## Reports from the title

These are testers' own reports, from their own copies of the title and
with other builds than this tree. They were not repeated here.

- **A set without the world-entry GPU hang on three NVIDIA cards.**
  Upstream `main` `d2a4bae` plus #493, #483, #189, #89, #205, #206, #469,
  #459 and #454, diagnostics builds run with
  `VICTORY_SHADER_LOOP_LIMIT=32`: no device removal on an RTX 5080
  (Zignixx, 7 to 10 minutes per run), an RTX 3080 (SlopComp, 12.5 minutes
  in the world) and an RTX 3070 (Vyral1978, about 14 minutes in the
  world). The 3D scene was black in all of them. Upstream issue #71,
  comments 7123097017, 7123060678 and 7123087811.
- **First run of this tree with the title** (nullmarsh, upstream issue
  #293, comment 7125407618; tree at `7670a8a`, before updates 3 and 4;
  AMD RX 9060 XT, opening cutscene, compared with `d2a4bae` + #135 +
  #365): the device was lost 251 and 153 s into the world against 40 to
  94 s on the comparison build, each time `0x887A0006`; 178 unique
  refused programs against 372 to 387; 1,292 draws dropped per frame
  against 968 to 1,183. The largest rows per frame: `an address in an
  SGPR that is no root parameter` 891, `a descriptor read from the first
  lane of a phi` 306 (the reason #566 handles, new in update 4), `a
  structured buffer of unknown stride` 226. World frames were black
  apart from the opening-credit text.
- **How this tree differs from that set.** It has all of those PRs but
  #189. In #189's place it carries the wave64 stack (#223, #233, #320,
  #439, #180), which cannot be in one tree with #189, and it has the
  other PRs listed below. It also keeps 2^18 dynamic descriptor slots on
  a capped view heap, where that set had 2^16 (see "Changes made in this
  branch"). So a run of this tree with `VICTORY_SHADER_LOOP_LIMIT=32`
  tells whether the result still holds here. Nobody has made that run
  with this tree.
- **One tester's run with the wave64 stack in #189's place** (their own
  merge, not this tree): on an RTX 5080, `d2a4bae` + #439 at `f653ea4`
  + #493, #483, #89, #205, #206, #469, #459, #454 against the same set
  with #189, loop limit 32, 262,144 dynamic slots on both, about 6
  minutes in the world, one run each. Neither lost the device. With
  the stack 958 draws per frame were dropped for want of a host
  pipeline against 1,470, and the richest 2560x1440 G-buffer target
  was 67.9% covered against 10.5%; the frame took 1,233 ms against
  850 ms. With the stack the four instance-culling compute programs
  named on issue #71 stop at `a structured buffer of unknown stride`
  (Zignixx, upstream PR #439, comment 7123794194).
- **If it hangs, try `VICTORY_SKIP_WAVE64=mesh,compute`** (#579,
  diagnostics build). It leaves out draws and dispatches of mesh and
  compute programs written for 64-lane waves, the compute programs the
  wave64 stack translates among them. Testers report that the hang goes
  away with it: an RTX 5070 Ti under vkd3d-proton (deadkid14, `d2a4bae` +
  #177 + #454 + #579: a hang about 5 minutes after the world load
  without the switch, more than 10 minutes in the world with it, one run
  each; issue #71, comment 7122920254), an RTX 5090 (wbkepler, PR #579)
  and an RTX 4080 (Proudnoob4, issue #71, comment 7123228803).
  deadkid14 and Proudnoob4 also report a black screen in the world.
- **#136 makes every frame black** (left out of this tree). Two testers
  bisected it: on an RTX 5080 a build with it is black from the first
  frame, loading screens included, and the build just before it shows
  the loading art (Zignixx, issue #71, comment 7123097017); on an RTX
  4080 `main` + #175 shows the loading art and adding #136 makes every
  captured frame black (Proudnoob4, comment 7123145062).

## What is in this tree

Grouped by what each PR is meant to fix. The number is the upstream PR,
and the name is its author's GitHub login.

### GPU: title buffer states and copies

- #334 (realhenners): transition host copies of title memory between
  shader writes and reads (one buffer-state tracker)
- #261 (Maraco91): copy-back puts back only what shaders wrote, in two
  more cases
- #322 (jsthope): copy into title memory without first asking for its
  region

### GPU: descriptors and descriptor heaps

- #205 (asdaasdsaddf): null texture UAVs get the title's view dimension
- #365 (FIocker): replay pending descriptors before publishing the host
  heap
- #336 (hophaver), at `3c81420`: drop draws whose fresh tables cannot be
  made; 2^18 dynamic slots. Its later commit `e8d3b64` (2^16 on a capped
  heap) is not taken; see "Changes made in this branch".
- #493 (hophaver), head `c48b240`: each root SRV is bound as a one-view
  table of its buffer's host copy, so reads past the copy return zeros.
  A root SRV whose address is not on 16 bytes reads zeros (logged once
  as `a root SRV not on 16 bytes reads zeros`), and one used with no
  view heap bound drops its draw. `VICTORY_ROOT_SRV_DIRECT=1` goes back
  to root descriptors.
- #483 (Vaximous), head `65a161d`: a heap index computed from data stays
  inside the heap (an index at or past the heap's guard slot reads that
  slot, which is null in a view heap)
- #206 (AnoHunted), head `05981dd`: a structured buffer view that does
  not start on an element of its host copy becomes a raw view of the
  same bytes (a shader that indexes such a view reads it as structured,
  which D3D12 leaves undefined)
- #123 (9hvn9yt2vk-netizen): refuse raw buffer views that do not start
  on 16 bytes
- #251 (tcurtsinger): free descriptor tables may span the search cursor
- #86 (faultzonsol): cap a sampler heap past the host's limit, with
  dynamic slots for tables past it
- #83 (FIocker): bound the null sampler staging heap (combined with #86
  by a small fix in its review branch)

### GPU: pipelines and the pipeline cache

- #347 (AugustoZFJ): ask for an empty pipeline library with a null blob
- #352 (rhuantac): mesh pipelines load from the pipeline library (they
  name their topology)
- #358 (Dufftwt): a shader cache that cannot keep an entry still gives
  its translation
- #275 (wbkepler): atomic adaptive pipeline wait budget, non-blocking
  pipeline library saves
- #178 (Ortomanic): compile the indirect-argument clamp with DXC when
  D3DCompile refuses it (Wine/Proton)
- #158 (r4tedx): the depth bounds test reaches the host, on adapters that
  support it

### GPU: textures and presentation

- #89 (HenriqueSBueno): packed 16-bit render targets become R8G8B8A8
  where the GPU cannot render to them
- #331 (AnoHunted): sRGB and linear views of one texture each read with
  their own curve
- #260 (Dufftwt): a half-float rounding carry raises the exponent
  instead of halving the value
- #211 (Yaekai): composite several PresentX planes (Series S black UI and
  menus)
- #469 (AnoHunted), head `c47c963`: a depth target that is also
  unordered access gets that access through a twin texture, copied to
  and from the depth in each list that uses it

### Shaders: instructions checked against AMD's ISA reference

- #48 (flapwarp78): 64-bit texture atomics, and proven shader branches
  followed
- #25 (artor410): packed typed-buffer formats decoded
- #169 (Dufftwt): `tbuffer_load_format_d16` loads pack two 16-bit
  channels per VGPR. Its test fails on the integration machine's AMD
  Radeon integrated GPU, before and after update 3 (a 16-bit UNORM 1.0
  comes out as the f16 `0x3BFF`, not `0x3C00`); it passes on the RTX
  5070.
- #276 (Dufftwt): `s_ttracedata_imm` is SOPP opcode 0x28
- #168 (Dufftwt): SCC of `s_min`, `s_max` and `s_lshl1-4_add_u32` as the
  ISA sets it
- #110 (lordon1a): DXIL IsNaN in `med3`, so Intel Xe does not fold
  `fcmp uno`
- #154 (lordon1a): DXIL IsNaN for unordered, ordered and self float
  compares
- #98 (tedin7): `v_permlane16_b32` in whole-wave sections

### Shaders: descriptor tracking, signatures and validator fixes

- #307 (AntoninMichelq), head `6285e19`: waterfall loops over per-lane
  descriptors, and descriptors chosen by `s_cselect`. Its last two
  commits (`077ec8b`, `6285e19`) came in with #566.
- r4tedx/VIctory#10 (mpDom), head `d6897fa`: with
  `VICTORY_PRESENT_1440P` set (diagnostics builds), the last 2560x1440
  colour target the frame wrote is presented in place of the title's own
  output, which stays black in the world while those targets hold the
  scene (issue #841). Not run in the title here.
- #566 (r4tedx), merged upstream since (in `main` from update 7 on;
  listed for the history), head `6c63dea`: per-lane descriptors chosen by control
  flow (an if/else whose arms each set the descriptor, traced on the
  paths the accessing lanes take), and a waterfall loop is proven on the
  lane masks before the first lane's descriptor is taken as each lane's
  own; each failed proof has its own reason. Its commit `b1c8306` (the
  proof) is by fuzeofficlal. Stacked on #307.
- #453 (AntoninMichelq), head `87e9d41`: a pixel program's barycentrics
  read as values come from `SV_Barycentrics`, and interpolation with
  computed barycentrics is evaluated from the vertex values. See
  "Changes made in this branch" for the fix that follows it.
- #704 (r4tedx, on #193 by n-romanovich), at `fb8ea43`: a buffer whose
  stride the container does not name is read through a raw view the
  host makes beside every structured or typed buffer view, with a table
  that gives each heap slot its stride and raw view. Fitted to this
  tree's heap as title slots, dynamic slots, the guard slot, then the
  raw views.
- #489 (wrothgaril), head `279eac1` (contains #302): a compute program
  whose root arguments all sit past the 16 user SGPRs reads them from
  the spill table, as they were on entry.
- #294 (dadou000), head `0050ce5`: root UAVs, and root views refreshed
  further into their buffer. In this tree a root UAV is bound as a
  bounded one-view table, as root SRVs are (#493).
  `VICTORY_ROOT_SRV_MB=<mebibytes>` sets how much of a root SRV's copy
  a bind refreshes (0: the rest of the copy).
- #346 (BastienRdz), head `7ff1184`: indirect rectangle-list draws are
  expanded on the GPU, adapted to #334's buffer-state tracker.
- #587 (FibbS6), head `1052a90`: a mesh lane's vertex and primitive
  writes are kept inside the declared maxima, primitives no lane writes
  are culled first, and a barrier only some lanes reach is emitted
  without the thread-group sync. See "Changes made in this branch".
- #309 (AntoninMichelq): SGPRs reloaded from VGPR lanes are read as the
  spilled value
- #306 (wbkepler): DXIL validator failures (signatures, PSV0,
  implicit derivatives in compute). Its image-address part is replaced
  by #160's; see "Changes made in this branch".
- #333 (SkippySteve): native root parameters stay inside the SGPRs; the
  thread-id count is clamped; each blob refusal is named
- #105 (tedin7): a loop that branches back to the entry instruction
  translates
- #288 (Yaekai): cap the BDD memo table, bail out early on overflow
- #343 (pedrofort): reject overlapping DXBC container parts
- #353 (FIocker): shader info shows stage I/O even when decoding fails
- #326 (SkippySteve): DXBC/DXIL parser and sampler/SP3 robustness fixes
- #259 (desplmfao): a descriptor's level count is queried only when
  word 3 is read
- #179 (TG0000): interpolation halves indexed on demand
- #176 (Yaekai): packed fp16 pixel inputs paired with unlisted signature
  rows
- #226 (gingertonik): test that an image read as float and used by an
  atomic is annotated twice
- #286 (tarcisioads): accept known `D3D_MIN_PRECISION` signature
  elements
- #459 (AnoHunted), head `dc838d6`: mesh output maxima come down to the
  lane count when the outputs pass D3D12's 32 KiB. A tester measured
  that it clears none of the 53 mesh programs over the cap in the
  title's shaders, whose lane count is already the maximum (Proudnoob4,
  upstream issue #201, comment 7122527604). It changes nothing for them.

### Shaders: image sampling

- #160 (r4tedx), head `25e917b`: texel offsets, comparison and gather
  forms, gradient samples of cube images, gathers at a level known only
  at run time, partial-channel image stores, and the a16 address layout
  (bias as an f16 in its own dword; derivatives 16-bit only in `_g16`
  opcodes)

### Shaders: wave64 compute on 32-lane hosts (NVIDIA)

- #223 (r4tedx): translate for the host's wave sizes
- #233 (r4tedx), at `3098672`: wave64 compute runs in halves or in the
  console flow on hosts without 64-lane waves
- #320 (r4tedx): lane reads and writes where lanes branch differently
  run in the console flow, on every host
- #439 (r4tedx), head `055aef9`: one console wave sharing LDS runs in
  the console flow. It carries peter-bonanni's commit `9b73d66`
  ("carry whole-wave integer scans across two 32-lane waves"):
  whole-wave sections across the two halves. It also carries `416714c`
  from Raxstreams's pull request r4tedx/VIctory#2 (git author Peter
  Bonanni): `ds_bpermute_b32` reads within the lane's own 32-lane half.
  Float whole-wave sums, minima and maxima cross the two halves
  (`965ce30`).
- #180 (permadiego), head `5ff639e`: twin descriptor loads, and quad
  gradients for compute samples

These come in together from the branch `integration/wave64-with-180` at
`24b3bcc`, which merges #180 into #439's head. That commit has a single
parent, so #180's changes are in it but its own commits are not. In
this tree, the shader tests ran on the integration machine's RTX 5070
and also on an AMD Radeon integrated GPU (64-lane waves). #439's five
commits after `e49302a` (`a32c177`, `8876ed3`, `416714c`, `f653ea4`,
`965ce30`) were picked onto the tree in update 3, in two merge commits,
and its docs-only commit `2467891` and test-only commit `e6936e3` after them,
then `d2189ff` (compute selects take masks held as numbers), `9e98b2b`
(by ifBars, from pull request r4tedx/VIctory#6: the inputs of console
mask joins carried around a loop are resolved after the first walk) and
`901eef9` (its test runs on every hardware adapter), then `e4d899f` (a
mask copied dword by dword in the console flow stops with `an opaque
lane mask` where the translator overflowed its stack; a lane read
beside a whole-wave section takes the console flow),
and the shader tests ran again on both GPUs. Its `055aef9` (a numeric
select nothing consumes no longer asks for the console's wave size) is
in this tree as MichaelSmithAI's original commit `3d3f32e`, merged from
pull request r4tedx/VIctory#5; after that merge the tests ran on the RTX
5070 only.

**Validation failures, fixed by #180.** A whole-title
`translate --validate` run found #320 adds 20 compute modules that fail
DXIL validation (43 with #439); the driver still creates pipelines from
them. All are compute `image_sample` emitted as `dx.op.sample`: an
implicit-derivative sample, which validators 1.6 and 1.7 refuse outside
pixel shaders. `main` emits 22 such modules (#233 emits the same 22).
#180 takes the level from each quad of the console's lanes as explicit
gradients (`sampleGrad`), which these validators accept.

**Measured in the title** by testers on an RTX 5070 Ti, with the whole
stack (#223, #233, #320, #439, #180) on `main`: 70 of the 119 compute
programs the driver refused now run (26 with #320 alone), and none that
ran on `main` was lost. Image results were not checked.

**Open report on #233:** on Windows 10 19044 with an RTX 3080 (driver
32.0.15.9649) and #175, NVIDIA's shader compiler faulted on some newly
accepted compute pipelines at world entry, one pipeline was refused with
`E_OUTOFMEMORY`, and then the device was removed. Three other NVIDIA
machines with newer drivers did not show it.

### Test stability

- #433 (r4tedx): upstream's VALU GPU tests run one at a time. Without
  it, `cargo test` sometimes never finishes, with several GPU tests stuck
  together. On the integration machine, before #433, 4 of 12 runs of the
  shader tests hung; with it, 0 of 12 did.
- At update 10, one run of the same command: 526 passed, 0 failed, 7
  ignored.
- At update 9, one run of the same command: 525 passed, 0 failed, 7
  ignored.
- At update 8, one run of the same command with the machine at 100% CPU
  from other work: 491 passed, 1 failed, 7 ignored. The failure is
  `frame::tests::waits_end_on_ticks_and_count_frames`, which expects
  consecutive 5 ms ticks and skipped ten or more at a time; it failed 3
  of 3 on its own at that load and passed 3 of 3 on its own at update 6.
  The update does not touch its file. The three geometry tests pass,
  one of them a draw on the GPU compared with a reference draw.
- At update 7, one run of the same command after `main` was merged in:
  487 passed, 0 failed, 7 ignored; and one more at the update's head
  (see the commit that adds this line for its numbers).
- At update 6, one run of the same command with other threads' builds
  on the CPU (85% load): 442 passed, 2 failed, 5 ignored. The two are
  `frame::tests::waits_end_on_ticks_and_count_frames` (a timing test)
  and `xgamesave::tests::a_kill_at_any_step_leaves_old_or_new`; each
  passed 3 of 3 runs on its own afterwards, and neither is in code this
  update changes.
- At update 5, one run of `cargo test --workspace --no-fail-fast --
  --test-threads=4` on the RTX 5070 with no other GPU work started from
  here: 442 passed, 5 ignored.
- **Still not stable at update 4.** `cargo test --workspace
  --no-fail-fast` passed at the #566 merge (430 passed, 5 ignored). With
  r4tedx/VIctory#5 on top, 14 runs of the shader tests on the RTX 5070,
  while other GPU work ran on the same PC: 11 passed (200 tests), one
  ended with no test named, one never finished (16 image and wave64
  split tests stuck together), and in one the driver removed the device
  under 13 image and split tests that were running (`The GPU device
  instance has been suspended`; the NVIDIA driver logged `Error occurred
  on GPUID` at that second). The tests in flight were already in this
  tree before update 4, and none of #566's was among them; which test
  or what else on the PC set it off is not known. The runs were stopped
  there. These are test programs, not the title.

### Switches that are off unless set

- #454 (ruvenesh), head `d2c93fa`: with `VICTORY_NO_WATCHDOG` set (to any
  value), the loader patches out the two places where the title's own
  watchdog ends the process, in build 1.0.2.1 only. Without the variable
  nothing changes.
- #579 (wbkepler), head `1658c84`: in diagnostics builds,
  `VICTORY_SKIP_WAVE64=<stage>,...` (`mesh`, `compute`, `vertex`,
  `pixel`) omits draws and dispatches whose pipeline has a program of a
  listed stage written for 64-lane waves

## Changes made in this branch

These are the only edits made here beyond the PRs themselves. Each one
is described in its merge or fix commit.

- **#160 with #306 (image addresses):** #160's address layout is kept,
  with #306's other parts. #306 read an a16 bias as a 32-bit float and
  packed derivatives with a16, which does not match the hardware (LLVM's
  `gfx10_asm_mimg.s` has `image_sample_d ... 2D a16` with five address
  dwords).
- **Level clamp and validator 1.8** (commit `Shaders: a level clamp
  declares tiled resources under validator 1.8`): #306 declares validator
  1.8 for compute shaders with implicit derivatives. From 1.8 on, a sample
  with a level clamp needs the TiledResources flag, so #160's `_cl` samples
  in compute failed validation. The flag is now set in that case.
- **#352 with #158 (mesh pipeline stream):** #352's triangle topology
  and cache key are kept inside #158's shared stream builder, in the mesh
  stage only, so vertex pipelines with depth bounds don't name a topology
  twice.
- **#233 with #307 / #309:** lane-spill instructions (#309) still don't
  count as lane reads in #233's new test. #307's waterfall read of
  `v_readfirstlane` comes before #233's console-flow form.
- **Mechanical fixes for #233's new arguments** in tests: `emit/lower.rs`,
  `emit/image_tests.rs`, `emit/mod.rs`, `sgpr_spill.rs`.
- **Merge base of the wave64 stack:** #233 is in this tree as `1eb9995`,
  the same tree as `3098672` but another commit, so git took `e0865d2`
  (#223) as the base of `integration/wave64-with-180`. Each file was
  merged with `3098672` as the base instead.
- **#98 with `9b73d66` (`v_permlane16` / `v_permlanex16` in whole-wave
  sections):** #98 models them only with every lane active, `9b73d66`
  also with `fi` set (an inactive lane read as it is). Both conditions
  are kept, for both instructions.
- **#320 with #309:** #320's test for lane reads and writes in a program
  whose lanes branch differently counts live instructions only, as #233's
  does: dead SGPR-spill lane instructions do not send a program to the
  console flow. #307's waterfall arm of `v_readfirstlane` still comes
  first; #320 adds no arm of its own.
- **Compute samples with implicit derivatives (#180 with #306 and #160):**
  #180 decides how compute `image_sample` and `image_sample_b` are
  emitted, in all of #160's forms (texel offset, level clamp, cube, a16).
  It takes the console's own quads (lanes 4k to 4k+3), whereas the host's
  compute derivatives follow the thread layout (2x2 in 2D groups). Its
  arm now reads the bias at #160's address (after an offset, an f16 with
  a16) and keeps #160's level clamp; its test for which programs need
  the console's lanes uses #160's form parser instead of two names.
  #306's validator 1.8 declaration is kept only for what still uses the
  host's derivatives in compute: comparison samples (SM 6.6 has no
  comparison sample with gradients). The TiledResources fix above (a
  level clamp under 1.8) now matters only for them. #306's test of a
  compute `image_sample` now asserts that the module does not declare
  1.8, and still validates it and creates the pipeline. A new twin test
  with `image_sample_c` asserts 1.8.
- **#309's spilled-mask test with #320:** without forwarding, the
  per-lane form still stops with "a lane mask holding lanes elsewhere
  read as data", but the program now runs in the console flow. The test
  checks both, and checks that forwarded spills do not ask for the
  console flow. It uses a test-only `emit::lower_compute`.
- **Mechanical fixes for #320's `per_lane_only`** in two tests that build
  `lower::Parts` by hand, in `emit/lower.rs` (#307's `lower_wave`, #226's
  annotation test). `emit/mod.rs`: module order, with `split_tests`
  before `threading`.
- **Dynamic slots on a capped view heap (#493 with #336's `e8d3b64`):**
  #493 carries #336's later commit, which gives a view heap the host
  caps (NVIDIA: 1,000,000 views, where the title asks for 1,310,720)
  2^16 dynamic slots in place of 2^18, to mirror more of the title's
  slots. That commit's sizing is not taken: a capped heap keeps 2^18, as
  this tree had. The reason is one tester's measurement in the title on
  an RTX 5080 (Zignixx, upstream issue #71, comment 7123097017, one run
  per row): with 65,536 dynamic slots 22-30% of the requests for slots
  were refused and about 1,680 mesh dispatches per frame were dropped;
  with 262,144 none were, the peak in use was 132,014, and
  `title slot past the host heap` never fired in a 7-minute run. A new
  test, `a_capped_view_heap_keeps_its_dynamic_slots`, holds this. On the
  integration machine the heap is capped at 737,855 title slots, 262,144
  dynamic ones and the guard slot.
- **#493 with #86 / #83 / #251 (`HostHeap`):** this tree's heap layout is
  kept (the capped sampler heap with its dynamic slots, the free-table
  search); from #493 only its own additions are taken.
- **#493 with #334 (root SRV tables):** the host copy a root SRV table
  views is moved to its read state, as #334 does for root views. #493
  was written without #334's buffer states.
- **#483 with #86 and #336 (the guard slot):** a view heap is one slot
  larger, and that slot, after the dynamic ones and outside the pool
  they are handed out from, is the guard (slot 999,999 on a capped
  heap). #483 placed it by a constant; here the pool's size is the
  heap's own (#86). A sampler heap's guard is its last slot with the
  dynamic ones counted, not #483's last title slot, which would have
  clamped away the sampler tables #86 makes in the dynamic slots. New
  test: `the_guard_slot_is_past_every_dynamic_slot`.
- **#483 with #307 (per-lane heap indices):** #307's per-lane heap
  index, which #483 did not know, goes through #483's guard too.
- **#483 with #160's tests:** the image tests fill the new guard dwords
  (`emit/image_tests.rs`).
- **#206 with #123:** #206's choice (on an element, raw view of the same
  bytes, or refused) comes first, then #123's refusal of a raw view that
  does not start on 16 bytes.
- **#469 with #334 (`HostList`):** #334's buffer states and #469's twins
  are both kept. When a list closes, the twins are copied back first,
  then the buffers go back to the common state.
- **#459's tests:** moved to the `tests` module of `emit/mod.rs` (git had
  put them in #48's `atomic_tests`), with #223's `Target` argument.
- **#439's new commits with #180:** in `Lower::new`, the instructions
  that read lanes of the same half are #180's implicit-derivative
  samples and `416714c`'s `ds_bpermute_b32`, both.
- **#439's `e4d899f` with #309 and the mask-SCC change:** its test for a
  lane read or write outside a whole-wave section looks at live
  instructions only, as this tree's earlier test did (dead SGPR-spill
  lane instructions do not send a program to the console flow). In
  `Lower`, its `opaque_open` and the tree's `mask_scc` are both kept.

- **Merge base of #566:** the tree held #307 as its first two commits
  (`5ca7d9a`, `52637df`, the same files as `ffa6635`), so git took
  upstream `main` as the base. Each file was merged with `ffa6635` as
  the base instead.
- **#566 with the wave64 stack (`structure.rs`):** `Structured` keeps the
  stack's `execs`, `others` and `pairs` and #566's `reach` and
  `paired_masks`. `pairs` (one high dword per low, no lane bits) and
  `paired_masks` (the lane bit of each exact pair) are not the same
  information and stay separate.
- **#566 with #483 (per-lane heap indices):** the index joined over the
  edges is made of indices that already went through #483's guard;
  #566's GPU tests fill the guard dwords.
- **Mechanical fixes for #566's `resource::track` argument:** every
  caller passes the per-lane form (`emit/mod.rs`, `emit/lower.rs`,
  `emit/image_tests.rs`, `emit/lane_read_loop_tests.rs`,
  `sgpr_spill.rs`, and the test of r4tedx/VIctory#5); #566's GPU test
  fills #233's and #320's fields of `lower::Parts`.
- **#566 with the console flow (#233, #320, #439)** (commit `Shaders:
  the console flow refuses per-lane descriptors chosen by control
  flow`): in the console flow every lane reaches a join by the wave's
  edge, so #566's phi gave every lane one descriptor (measured: all 32
  lanes the second texture). The console flow refuses such a program by
  name and it keeps the per-lane form. New test:
  `joined_descriptors_keep_the_per_lane_form_when_the_console_flow_is_asked_for`.
- **#566's GPU tests run one at a time** (commit `Tests: #566's GPU
  tests run one at a time`): in parallel the test binary sometimes died
  of an access violation (7 of 24 runs); with a lock, as #433's, 0 of 36
  runs of those tests alone.
- **#453's conflicts:** in `emit/lower.rs` the interpolation's packing
  keeps #176's `half_inputs` lookup with #453's `total` closure after
  it; in `io.rs`, #179's `Halves` made on demand with #453's
  `push_pair`, and both sides' tests; #453's test helper takes #566's
  argument of `resource::track` and #223's `Target`.
- **A fix after #453** (commit `Shaders: at most one perspective and one
  linear SV_Barycentrics, none of the console's`): #453 emitted modules
  `dxil.dll` refuses in two cases. A program reading the barycentrics of
  two sample positions of one kind (centre and centroid) now stops with
  `barycentrics of two sample positions`, and an `SV_Barycentrics` row
  in the console's own signature is no longer carried over. Two tests,
  each failing without its half. In review, every probe #453 drew on the
  RTX 5070 matched values computed on the CPU. On WARP, programs with
  both a perspective and a linear `SV_Barycentrics` drew wrong values,
  and so did DXC's own module for the same shader, so that is WARP's.
- **#587's conflict and one change to it:** its `divergent_barriers`
  sits beside #453's `interp_mode` in `emit/lower.rs` (both kept). Its
  fence-only barriers are not applied in the console flow of the wave64
  stack (#233, #320, #439): there every lane of a wave runs every block,
  so the lanes reach a barrier together and the stack relies on the
  sync. #587 came without tests; two are added (primitives past the
  lanes are culled first, for four pairs of lane count and maximum, each
  validated; a barrier under a per-lane branch has mode 8 and one every
  lane reaches mode 9), with a test-only record of a module's `dx.op`
  calls.
- **`VICTORY_HEAP_GUARD=0`** (`draw.rs`): see "Known problem on AMD".
- **The merge of `main` `ffe310e` (update 7):** conflicts in seven files;
  the merge commit's message lists each. In short: both sides in
  `list.rs`, `native.rs` and `emit/shader.rs`; in `emit/lower.rs` the
  tree's IsNaN compares with `main`'s `olt | ogt` for `v_cmp_lg`, #160's
  sample forms in place of `main`'s list of sample names and of its
  immediate-offset block, and `main`'s zero for an SGPR the hardware
  does not initialize beside the tree's barycentric inputs; the test
  files merged item by item with all of both sides' tests.
- **Geometry programs (update 8; two commits by this project's shader
  work, not an upstream PR yet):** `Shaders: translate uncompressed
  native FSSM containers` (`native.rs`; the geometry containers need
  it) and `Shaders: translate single-stream triangle geometry programs`
  (new `geometry.rs`; the geometry stage reaches DXIL as a geometry
  shader, with the console's copy vertex shader traced for what each
  vertex holds). Counted offline over the title's shaders: 195 `Gs`
  programs that stopped at `an address in an SGPR that is no root
  parameter` validate, and no other program changes. Other stages take
  the same path as before (checked in an independent review). Two known
  limits from that review: a container with a geometry slot or an
  uncompressed entry that does not parse now fails as a whole, where the
  slot or entry was skipped before; and every vertex pipeline's stream
  now names an empty geometry shader, which should be the same pipeline
  to Direct3D. Not run in the title: nothing is known yet about what
  these programs draw there. A third commit offered with them (leave a
  lane mask's dword undefined at a data join) is **not** in the tree:
  it turned refusals into modules by handing out an undefined value
  where the console uses the mask's value.
- **Groups of several console waves (update 10):** the wave64 work's
  `shader/wave-per-group` at `252a095`, taken as the difference from the
  stack's head and applied three-way, because the stack is not a merge
  parent of this tree. A wave64 compute program whose group is several
  console waves runs as one host group of 64 threads per wave, and a
  dispatch of (x, y, z) groups runs (x * waves, y, z). An indirect
  dispatch of such a shader is still dropped, and the title sends its
  cascade-shadow passes through indirect execution (a tester on #439),
  so this update alone will not make those passes run. Programs that
  need LDS or a barrier across the group's waves stay refused (20 of
  the 43 that were stopped for group size validate now).
- **The black-scene branch (update 9)** came in as one merge of work
  built on this tree at update 5; its own merge commits say what each
  PR's conflicts were. In short: #294's root views keep #493's tables
  for SRVs and tell #334's tracker the state each needs; #346's raw
  state transitions of the title's buffers go through that tracker (its
  expansion pass with the tracker is not covered by a test, only by a
  title draw); #704's raw view pool sits after #483's guard slot and the
  stride table has a zero entry for the guard. Merging it into update 8
  conflicted in four places, all with the geometry path: the tree's
  `NativeProgram::translation_ir` builds with the spill table for
  stages other than geometry.
- **#193's later safety changes, by hand (update 9):** #704's current
  head carries #193's `f99b792`; merging that head conflicts with this
  tree's heap layout in a dozen places, so its changes are applied to
  the tree's own code: a buffer view of no elements is not made (a
  tester's debug-layer run showed `CreateUnorderedAccessView` refusing
  one and the device removed with `0x887A0001` right after), a shader
  that reads the stride table while no view heap is bound is not run,
  and a named stride that is no multiple of four is refused.
- **`VICTORY_HEAP_GUARD_FORM`** (`emit/lower.rs`, since `da605bc`): see
  "Known problem on AMD".
- **Limits of the geometry programs (update 8)**, from an outside review
  against another translator's measurements (c4k3ss, issue #293): one
  stream; triangle input only; no `EMIT_CUT` message; no control flow in
  the geometry body or past the copy shader's stream predicate; 16-bit
  inputs and outputs refused; a vertex slot rewritten after its EMIT
  translates with the earlier value; the record offsets rest on two
  internal checks that hold on all 195 programs. Whether each geometry
  program links with its vertex stage is not checked: the count
  validates programs one at a time.
- **A level-of-detail query passes its spatial coordinates only**
  (`emit/lower.rs`; the fix is MichaelSmithAI's, from #453's
  discussion): `image_get_lod` of an array texture passed the slice as a
  coordinate and `dxil.dll` refused the module. A test for a 1D-array
  and a 2D-array query fails without it. The 3 pixel programs of the
  title that hit it validate now.

## Left out on purpose

- #442 (AnoHunted): reads barycentrics as values, as #453's first commit
  does; it refuses interpolation with computed barycentrics, which #453
  lowers, so the tree takes #453
- #136: a16 bias read as f32; `image_gather4_l` ignores the sampler's
  address mode; 31 `E_OUTOFMEMORY` refusals in a title run. Two testers
  have since bisected it as making every frame black, loading screens
  included (Zignixx on an RTX 5080 and Proudnoob4 on an RTX 4080,
  upstream issue #71, comments 7123097017 and 7123145062)
- #317: MSAA descriptors report 1 sample instead of log2(samples)
- #269: its new tests fail on 32-lane hosts (NVIDIA)
- #318: a16 also packs derivatives, which the hardware does not
- #138: a16 derivatives and bias layout (as #318); covered by #160
- #332: partial-channel image stores fail DXIL validation (typed UAV
  stores must write all four components); covered by #160
- #342: partial typed buffer stores fail DXIL validation, as #332
- #305: missing the descriptor-heap 64-bit atomics flag; #48 (included)
  covers the same instructions
- #360: `image_gather4_l` ignores the level; sample offsets are not
  immediates
- #200: `v_div_fmas_f32` scales by 2^64 instead of 2^32
- #40: writes through all `MaxCommandCount` records of the title's
  argument buffer; #242's design is preferred
- #242: reads the title's argument buffer in the wrong resource state
  (updated upstream since, not re-reviewed)
- #282: refuses rectangle lists whose start vertex is not a multiple of
  3 (updated upstream since, not re-reviewed)
- #190: overlaps #178 (included) for the Wine/Proton indirect clamp
- #301: drops back-face stencil masks and the depth-bounds flag (updated
  upstream since); would also need combining with #158
- #55: its fallback can remove render-target use from a texture that
  still gets render-target views
- #184: does not compile on current `main` (`Step::Texture` not covered)
- #174: the diagnostics build does not compile (`tick`)
- #321: skips a frame event when asked between a tick and its offset
- #185: a draft
- #135: overlaps #336 and #251 (included) on the dynamic descriptor
  pool
- #367: closed as a duplicate of #334
- #159 (r4tedx): needs adapting to #334's state tracker first
- #189 (alfycs): another way to run wave64 compute on 32-lane hosts. It
  and the wave64 stack in this tree (#223, #233, #320, #439) change the
  same lowering and cannot be in one tree.
- #578 (wbkepler): refuses every wave64 mesh and compute program on
  hosts without 64-lane waves, the compute programs the wave64 stack
  translates among them. #579's switch (included) does the same on
  request.

## What this tree is NOT expected to fix

- **The world hang itself is not known to be fixed here.** In-game runs
  of #334 still ended in `DXGI_ERROR_DEVICE_HUNG`. Testers report no
  hang with another set of PRs (see "Reports from the title"); whether
  this tree behaves the same has not been run.
- **The black 3D scene.** The runs without a hang still showed a black
  scene in the world.
- **Some top rows of issue #293's table** ("a descriptor neither loaded
  nor built", "a scalar memory read as data"). No PR in this tree fixes
  them. "A structured buffer of unknown stride" is fixed since update 9
  (#704), and the geometry programs of "an address in an SGPR that is
  no root parameter" since update 8.
- **Faults on textures the title freed earlier.**

## How to build

Requirements are the same as in the README (Windows 10/11 x64, AVX2, a
D3D12 GPU with shader model 6.6 and mesh shaders, Rust as pinned).

```powershell
git switch integration/testers
cargo build --release -p victory
# for device-removal reports, the diagnostics build:
cargo build --profile diag -p victory --features diagnostics
```

The diagnostics build is at `target\diag\victory.exe`, the release build
at `target\release\victory.exe`. Run it as the README says. A fresh
`VICTORY_CACHE_DIR` keeps shaders cached by other builds out of the
comparison.

## What makes a useful report

Post on the PR you are testing, or on the issue that asked for the run.
Please include:

- **GPU, driver version and Windows build** (for example "RTX 3080,
  32.0.15.9649, Windows 10 19044").
- **The frame-rate line's dropped counts** (draws and dispatches dropped
  without a host pipeline, and how many were still being made).
- **The device-removal block from `crashes.log`**, if the device was
  removed (reason, DRED breadcrumbs, the command it names).
- **The drop summary by reason**, as in issue #293 (`dropped N/frame
  without a host pipeline: <stage> <hash>` joined with the
  `shader does not translate ... error=` reason), so it can be compared
  with that table.
- **The counts from `victory translate --validate --pipelines --list`**
  over the title's shaders: pipelines created, pipelines refused by the
  driver, modules failing validation. This finds a faulting pipeline
  without running the world.
- **Any `nvgpucomp64.dll` first-chance exception or `E_OUTOFMEMORY`
  pipeline refusal**, with the shader hash from the log.
- **Whether the world shows geometry**, in words.
- **The variables you set**: `VICTORY_SHADER_LOOP_LIMIT`,
  `VICTORY_SKIP_WAVE64`, `VICTORY_NO_WATCHDOG`, `VICTORY_ROOT_SRV_DIRECT`,
  `VICTORY_HEAP_GUARD`, `VICTORY_HEAP_GUARD_FORM`, `VICTORY_ROOT_SRV_MB`,
  `VICTORY_SKIP_PIPELINES`.
- **For the black scene:** a target dump with `VICTORY_ROOT_SRV_MB` at
  its default and at 0, to see whether the G-buffer fills when root
  SRVs are refreshed to the end of their copy.
- **These log lines, if they appear**: `a root SRV not on 16 bytes reads
  zeros`, `a root SRV without a bound view heap`, `title slot past the
  host heap` (with its highest index), and the `dropped=` count of
  `a draw or dispatch dropped: no dynamic slots for its tables`.
- Which PRs you left out or added, if you changed the tree.

Post **counts and messages only**. Never post game files, shader dumps,
extracted data, or screenshots of the title.
