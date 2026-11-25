# Tester integration branch on main

This branch puts a few open upstream pull requests together on top of the
current upstream `main`, so that testers who run the title can build once and
measure them together. It is **not an official release**; the maintainer has
not merged any of these PRs, and each may still change or be turned down.

It replaces `integration/testers`, whose last update is `4744af9` (update 10).
That tree was based on `ffe310e`, before the maintainer's large drop of
2026-11-24 and the PRs merged after it; it is frozen, not deleted.

- Base: upstream `main` at `940d32f`.
- Every PR comes in as its own merge commit (`git log --first-parent`), so
  you can find, revert or leave out each one. No edits were made beyond the
  PRs themselves. Two conflicts, both text: the README's "missing" row,
  which #158 and #159 both edit, carries both clauses; and the DXIL line of
  `translate`'s summary keeps #1315's "vertex, geometry and pixel programs"
  with #2115's wording for checks that were not asked for.
- Only heads their authors or owners name as green on `main` are taken, and
  a head goes in only when no program that validates on `main` is lost.

## What is in it

| PR | Author | Head | What it does |
|---|---|---|---|
| #158 | r4tedx | `49dd080` | the depth bounds test reaches the host, on adapters that support it |
| #159 | r4tedx | `b7249e5` | occlusion, pipeline statistics and timestamp queries run on the host |
| #2111 | r4tedx | `0ef11e5` | one pipeline wait budget per frame for all recording threads; the frame line prints how long draws waited for pipelines |
| #1762 | claude-GTA6 | `4eb5d1e` | structured buffer views become raw views of the same bytes on the host (NVIDIA reads 0 through raw loads of structured views) |
| #1883 | powplusplus | `a31d8cc` | lane masks a traversal keeps as numbers |
| #2027 | fiestorra | `34d0b4e` | a compute shader samples at level 0, not with derivatives |
| #1315 | r4tedx | `0d45a59` | single-stream triangle geometry programs |
| #2136 | r4tedx | `937f329` | a host copy nothing wrote is not asked for at submission (less CPU time preparing a frame) |
| #2147 | r4tedx | `f869790` | ranges shaders may write are armed as runs of pages, not page by page |
| #2134 | r4tedx | `efb35b5` | a benchmark of a world frame's preparing at the published volumes (tests only) |
| #2115 | r4tedx | `4d34d76` | `translate --count`: the instructions of the emitted shaders, to compare builds |
| #1428 | fuzeofficlal | `0e3796e` | `OMSetRenderTargets` reads no handle for zero targets, and bounds the count |
| #2043 | Deilach | `7686b6f` | a pixel discard in a loop whose exit depends on it kills its lanes, as the console does |
| #1989 | BastienRodz | `0f11320` | a vertex (or mesh, geometry) program writing through views declares the UAV flags the validator requires |

Left out for now: #2150 (one instruction per value on the finished
module). It rewrites every emitted shader for a module-size gain; it has
had one hardware run by a tester since. It is kept out so that the first
supervised game run uses shader output as close to `main` as possible.

Not in it yet: the wave64 stack (#223, #233, #320, #439: compute programs
written for 64-lane waves, on NVIDIA's 32-lane waves). Its head on `main`
is `5a11cf4`, with its implicit-sample and recursion guards; it goes in
with the next check round. Until then an NVIDIA card refuses those
programs on this branch as on `main`.

## Known risks on NVIDIA (crash safety)

- **Heap indices past a capped heap.** Translated shaders clamp
  descriptor-heap indices below 2^21, but NVIDIA caps a shader-visible
  view heap at 1,000,000, so an index between the two still passes the
  heap's end. Upstream #1777 aims at this but is not taken yet: it opens
  the default hardware adapter in every process that translates, and has
  a slot-overlap case at one heap size (both reported on #1777).
- **Driver-level shader error reporting** (the Nsight Aftermath Monitor
  set to Global) turned every world load into a device hang for a tester
  on upstream #71. Check it is off before a run.
- Running out of per-draw descriptor slots does not bind unwritten slots:
  `main` drops that draw or dispatch (logged as `a draw or dispatch
  dropped: no dynamic slots for its tables`).

## How it was checked

On one Windows 11 PC (NVIDIA RTX 5070, AMD Radeon integrated GPU, WARP):

- `cargo build --release -p victory`, the test builds and
  `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `victory-gpu` tests on WARP (diagnostics build, `VICTORY_D3D12_WARP=1`,
  one test per process): 168 passed, 0 failed.
- `victory-shader` tests with `VICTORY_WARP=1`, one test per process: 268
  passed, 2 failed. The two failures are hardware-only checks that
  `VICTORY_WARP=1` sends to WARP (`gathers_at_a_level_match_the_hardware`,
  and `permutes_read_the_lane_named_and_zero_from_lanes_left_out`, which
  needs a device that accepts a wave of 32); they fail the same way on
  `main` under `VICTORY_WARP=1`, and no PR here touches them.
- **Local title-data diagnostic:** `victory translate shaders.rpf --validate
  --list --jobs 6` on this PC's copy of the title (offline, CPU only, counts
  and refusal reasons only), program by program against the same count of
  `main`: **1,842 programs gained, 0 lost**: all 867 vertex programs
  `main` emits but the validator refuses (#1989), all 195 geometry
  programs (#1315), and 780 compute programs (#1883 and #2027 together:
  735 that `main` emits but the validator refuses, 40 ray-intersection
  programs and 5 more). Every program of that archive that does not validate on this
  branch does not validate on `main` either.
- The speed numbers of #2136 and #2111 in their PRs come from synthetic
  benchmarks, not from the game; times from the title are what they need.
- Nothing from this branch was run on the RTX 5070, and **the title was not
  run** with it. It makes no claim about results in the game.

## What a useful report from the title contains

- the branch head, your GPU and driver, Windows version;
- whether the world loads and whether it is lit, and how long it runs;
- the frame line's times (preparing, on the GPU, waiting for pipelines);
- any device removal with its DRED output, and the System log's display
  driver events (`nvlddmkm`) at that time. On NVIDIA, check first that no
  driver-level shader error reporting (the Nsight Aftermath Monitor set to
  Global) is on: a tester on upstream issue #71 found it turns every world
  load into a device hang.

Counts, messages and hashes only: never game files, shader dumps or
captures.

## `integration/play`: playable candidate for NVIDIA

`integration/play` is this branch (`915978c`) plus the NVIDIA boot, heap
and format PRs and the wave64 stack, so that one build on a 32-lane NVIDIA
card gets as far as the ztrx fork does. It is a **candidate**: built and
checked only (`cargo check --locked -p victory --features diagnostics`
after every merge, then `cargo build --locked --profile diag -p victory
--features diagnostics`); no test suite was run on it and the title has
not been run with it yet. Every PR is its own merge commit
(`git log --first-parent`), authors' commits unchanged.

### Merged, in order

| PR | Head | What it does |
|---|---|---|
| #1763 | `50d91b1` | typed UAVs in formats the host does not support are refused, whatever their family or castability |
| #1751 | `4521591` | a host that caps view heaps (NVIDIA, 1,000,000) keeps the larger dynamic range |
| #1133 | `cae16e6` | translated shaders clamp view indices to the host's cap, whose last slot stays null |
| #2180 | `e1bf670` | UAV/render-target flags a format cannot take are stripped or widened; an early-list-only save writes no library |
| #1778 | `ba67548` | a sampler heap of our own is bound when the title binds none |
| #2129 | `2ef0c75` | the pipeline library file is mapped, not read into memory (intermittent device loss) |
| #1906 | `bcd0d3d` | view addresses for tables made afresh in dynamic slots |
| #1849 | `de95e51` | free memory is not the runtime's to watch (world-load deadlock) |
| #454 | `85c04d1` | opt-in `VICTORY_NO_WATCHDOG` hook for the title's watchdog (unset: off) |
| #223, #233, #320, #439 | `5a11cf4` | wave64 compute on 32-lane hosts, in halves and in the console flow, with the implicit-sample and recursion guards (fork `shader/console-flow-lds`; not `831cdb3`) |

Conflicts resolved at the merges (each merge commit says how): #1133 with
#1751 (both heap branches kept, #1133's guard slot on each) and with
#158's depth-bounds query; #2129 with #2180 in the library save (#2129's
mapped write, #2180's skip when only the early list changed); #1906 with
#1133 (two tests at one place); the wave64 stack with #2115 (`--count`
and `--host-lanes` both), #1133 (`view_limit` and `ops`), #1883 (mask
joins: the stack's rule in the console flow, #1883's per-lane cases
elsewhere, a numeric mask's lane bit refused in halves) and #2043 (the
killing form for trapped discards first, then the per-lane form, the
console flow, the killing form for discards).

### Considered and left out

- #89 (packed 16-bit formats 86/115 as R8G8B8A8): based on `35b054e`,
  before the maintainer's drop; eight conflicting hunks in
  `descriptor/host.rs`, `list.rs` and `texture.rs`, two of them large.
  #2180 is taken instead; it strips or widens the flags rather than
  widening the format.
- #1777 and #1779 (views past a capped heap): #1133 is taken. It clamps
  in the shader to a null last slot of the capped heap, keeps the larger
  dynamic range #1751 asks for, and is based on `940d32f`; #1777 is based
  on the older `70564ae`, and #1779 rewrites ten files (884 lines). Like
  #1777, #1133 creates the host device when the shader cache key is first
  computed (also in `victory translate --cache`).

### Ray tracing (not changed here)

The ztrx and fxceless forks run with the title's ray-tracing structure
builder off: the compute pipelines of `XboxBvhBuilder.dll` (the programs
with a spilled user-data table) are dropped at dispatch unless
`VICTORY_BVH_BUILDER=1` (ztrx `crates/victory-gpu/src/pipeline.rs:285`
`builder_runs`, called from `draw.rs:1677` and `draw.rs:2278`; detection
`of_builder`, `pipeline.rs:238`, needs that fork's
`victory_shader::spill`). The programs that walk the structures still
run there. Upstream `main` has no such gate and no `spill` module, so it
is not a small port and is not in this branch. For the hang in compute
`f607780aa8e502d0` (the probe shadow trace), a diagnostics build of this
branch takes `VICTORY_SKIP_PIPELINES=f607780aa8e502d0`, as `main` does.

### Build

`CARGO_TARGET_DIR=C:\Users\PC\.t3\targets\play`, `cargo build --locked
--profile diag -p victory --features diagnostics` at `0ab7ca3`:
`diag\victory.exe`, sha256
`5b8bf42def4777099794b9a9ce2ef64330c0edf0a2a0a9dd507ee3de0e155a30`.

## Layer 2 (`gpu-run/layer2`)

Built on `gpu-run/layer1-2118` at `2fde53b`: this file's list above, plus
#2136 `f85763e`, #2147 `f2ca086`, #1819, #2168 and #2118 (the BVH builder
gate reverted). Twelve more upstream PRs, each as its own `--no-ff` merge
in this order. Every head is the PR's current head as of 2026-11-25, the
same SHA the plan named. `cargo check --locked -p victory --features
diagnostics` passed after each merge.

| PR | Author | Head | What it does |
|---|---|---|---|
| #2175 | wbkepler | `82ae414a` | title memory page bitmaps live in video memory |
| #2242 | wbkepler | `09595753` | copy-backs put back only the bytes the GPU changed |
| #2280 | wbkepler | `7ef96fa9` | copy-backs compare a block at a time (stacked on #2242) |
| #2051 | 0xkn0wh3r3 | `e004f357` | a submission's refreshes copy on several threads |
| #2120 | 0xkn0wh3r3 | `772c5670` | copies of read-only title buffers in GPU upload heaps; opt-in, `VICTORY_GPU_UPLOAD=1`, **off by default** |
| #2276 | denkencar-sketch | `14e872b7` | a depth texture used before it got a twin is copied to the twin in the list |
| #2173 | fuzeofficlal | `cdd11bd2` | typed D16/D32 and multisample depth twins |
| #1933 | mgunes1311-lab | `4be6b961` | depth-twin UAVs stay in concrete UAV formats; default views |
| #2098 | moe-miau | `619b694a` | an acceleration structure's view address is its location |
| #2163 | 0xkn0wh3r3 | `cd4402f4` | BCPack: BC7 dictionary indices read where each entry has them |
| #2031 | 0xkn0wh3r3 | `4036fbe4` | heap samplers keep point mips when anisotropic, as static samplers do |
| #1978 | 0xkn0wh3r3 | `392b3963` | no texture layouts or name copies per draw for unchanged answers |

### Conflicts

- #2175 with #159 in `crates/victory-gpu/src/host.rs`, both behaviours kept.
  `HostList::close` first ends the queries still running (#159), then
  makes #2175's read-back copies (a barrier from unordered access to copy
  source, then the copy), both before the diagnostics timestamp end.
  `Host::readback_buffer` is #2175's (on its `buffer_in` helper,
  `pub(crate)`, returns `*const u8`), with its `uav_buffer`; #159's
  caller `HostList::readback` passes that pointer on without its old
  `cast_const`.
- #2051, tests only, `host.rs`: #2136's test
  `a_copy_nothing_wrote_is_not_asked_for` and #2051's ignored benchmark
  `world_frame_refreshes` were added at the same place; both are kept.
- #2098, doc comment only, `descriptor/host.rs` `write_address`: #1906's
  clause (the caller keeps `index` below `HostHeap::address_slots`) and
  #2098's (an acceleration structure's address is written too) are both in it.

No PR was skipped.

### Shader translator

`git diff 2fde53b 3797434 --stat -- crates/victory-shader` is empty: no
PR in this layer changes `victory-shader`, so the layer 1 shader cache
still holds. The layer touches `crates/victory-gpu` (11 files) and
`crates/victory-formats/src/bcpack.rs`. The one change in
`pipeline/host.rs` (#2031) moves the anisotropic point-mip support
query into `sampler::aniso_point_mip` and leaves the result as it was.

### Build

`CARGO_TARGET_DIR=C:\Users\PC\.t3\targets\layer2` (seeded from
`targets\layer1-2118`, its `diag\victory.exe` deleted first), `cargo
build --locked --profile diag -p victory --features diagnostics` at
`3797434`: `diag\victory.exe`, sha256
`f159c53db4397226259cfeb2303dd09ec95fbee8b999085f0ad37465df97d95f`.

### Unit tests (WARP, through heavy-run)

- `victory-formats` (#2163's crate): 39 passed, 0 failed, 5 ignored.
- `victory-gpu`: the lib tests do not compile, and the cause is already in
  layer 1 (`2fde53b`), not in this layer. #159's test in `query.rs`
  (lines 573-574) calls `indirect::bound` and `indirect::done` with the
  arguments they had before #2168 (`cd29144`) added `count` and
  `clamped`. Layer 2 changes neither file. The production code is fine
  (`cargo check` and the diag build pass). Fixing that test means
  bringing #159 up to #2168's API.

## Layer 4 (`gpu-run/layer4`)

Base: `gpu-run/bis-no2242` at `e9159245`. That is layer 2 (`533217b7`)
with #2280 and #2242 reverted. Each piece is one `git merge --no-ff` and
keeps its authorship. After each merge, `cargo check --locked -p victory
--features diagnostics` passed through `heavy-run --light`, with
`CARGO_TARGET_DIR=C:\Users\PC\.t3\targets\layer4` seeded from
`targets\bis-no2242`.

| # | Piece | Source SHA | Merge | Conflicts |
|---|-------|-----------|-------|-----------|
| 1 | rect.rs edge-triangle fix (upstream PR #2366) | `a54e0c84` | `0dfa952d` | none |
| 2 | pixel quad reads reach helper lanes | `360b8557` | `f5ceef7b` | `emit/lower.rs`, see below |
| 3 | #1962 rebased, with test fixes | `36871fe7` | `6d30bb90` | none |
| 4 | #1962 marks fix (`fix/1962-marks-once-per-wave`) | `288cbd38` | not merged | held, see below |
| 5 | upstream #2311: BVH-walking pipelines wait until the builder has run whole | `481a9aec` | `ebb682dc` | none |
| 6 | upstream #2167: memory_dirty atomic skipped when the bit is already set | `6ab9c988` | `c0d7654b` | none (#2175 was already in the base) |
| 7 | fork `gpu/batched-clamps-on-2168`: opt-in `VICTORY_BATCH_CLAMPS=1` | `31047d78` | `39e8c797` | `victory-gpu/src/host.rs`, see below |
| 8 | upstream #2085: early depth test for pixel shaders that write buffers | `3c76d58d` | `5a2082cb` | none |
| 9 | upstream #2143: smaller constant tables | `a68ae559` | `552e410a` | none |
| 10 | upstream #2245: mesh primitive indices written once | `2e26c46a` | `2cbd1d59` | none |

After the merges: `d3caa2d1` "Tests: layer 4's shader tests compile
again". This commit changes test code only.

### Skipped

- Piece 4, `288cbd38` (#1962 marks: once per wave, test before OR). Held
  because of a translator panic, and a fix is pending. On trees with the
  wave64 stack, which includes this base, it makes the translator panic
  with "forward reference outside a phi".

### Conflict resolutions

- Piece 2, `crates/victory-shader/src/emit/lower.rs`: the base's
  `console_read_lane` (console flow) and 360b8557's new `helpers_read`
  doc comment were added at the same place. Both are kept: first
  `console_read_lane` as the base has it, then 360b8557's comment and its
  `helpers_read` (`quad && self.s.pixel`, without the discard condition).
- Piece 7, `crates/victory-gpu/src/host.rs` `HostList::close`: the base's
  query ends and deferred read-back copies, and the batched clamps'
  `crate::indirect::end(self)`. Both are kept: first the base's query
  ends and read-back copies, then the clamp window end. They touch
  different buffers. `indirect::end` only writes the window's counts
  table and moves its records back to unordered access.
- `d3caa2d1` (test-compile fix): #2167's `synthetic_mark_dxil` and
  #2085's early-depth tests build `Parts` and `HostShader` the way main
  has them. On this tree `Parts` also takes `target`, `console_cf`,
  `per_lane_only` and `rays_miss`, `Lower::new` takes `numeric_selects`,
  and `HostShader` takes `heap_usage` (#1962). The fix fills these in with
  their defaults (`Target::ANY`, `false`, `l.heap_usage`). After it,
  `cargo check --locked -p victory-shader -p victory-gpu --tests` passes.

### Shader translator

`git diff --stat e9159245 <layer 4 head> -- crates/victory-shader`: 13
files, 1271 insertions, 146 deletions (`emit/bitcode.rs`, `host.rs`,
`lower.rs`, `memory_tests.rs`, `mod.rs`, `rect.rs`, `shader.rs`,
`synthetic_tests.rs`, and one or two lines in each of `bvh_tests.rs`,
`semantics_tests.rs`, `stride_tests.rs`, `valu_tests.rs`,
`waterfall_tests.rs`). The translator changes (pieces 1, 2, 3, 6, 8, 9,
10), so the shader cache needs a new pass for this layer.

### Build

`cargo build --locked --profile diag -p victory --features diagnostics`
at `d3caa2d1`, through `heavy-run --light`, into `targets\layer4`:
`diag\victory.exe`, sha256
`2118b3f5864bdc573f79366d396052b142ab6abaf9549eecc4c69e9b4a8a9fc2`.
This file only changes docs, so the binary matches this layer's head.

### Tests

No unit tests were run for this layer. Only test compilation was
checked (above). #2085's new synthetic tests loop over `("hardware",
false), ("warp", true)`. Run them only with hardware turned off, or not
at all, while the hardware hold is in place.

## Layer 4 final (`gpu-run/layer4-final`)

Base: `gpu-run/layer4` at `01c5d98d`. Each piece is one `git merge
--no-ff` and keeps its authorship. After each merge, `cargo check --locked
-p victory --features diagnostics` passed through `heavy-run --light`,
with `CARGO_TARGET_DIR=C:\Users\PC\.t3\targets\layer4-final`.

| # | Piece | Source SHA | Merge | Conflicts |
|---|-------|-----------|-------|-----------|
| 1 | upstream #2294: a typed depth format in a shader resource view is made in the readable colour format | `298a0067` | `0245d605` | `descriptor/host.rs`, tests only |
| 2 | upstream #2269: marks in the last dword of the written-pages bitmap are not discarded | `a8bc74c8` | `2dc089a2` | `shadow.rs`, tests only |
| 3 | upstream #2295: the fifth dword of a root constants parameter is read as a constant | `ec236e64` | `972ed409` | none |
| 4 | fork `gpu/ambient-stand-in`: opt-in `VICTORY_AMBIENT` | `2f829fb4` | `c43cf5f6` | none |
| 5 | fork `gpu-run/layer4-wpg`: wave-per-group `6551f220` on layer 4, rect clippy fix `15ac13f1`, upstream #2306 `a4057f25` | `a47125be` | `b12f7683` | none here (its own are in its merge `4330dba8`) |
| 6 | fork `play/layer4-library-saves`: the pipeline library is saved after 256 new pipelines or 60 s | `9baeb2de` | `83a7b32a` | none |
| 7 | fork `gpu/dynamic-slots-no-drops` `a6cd2b68`, ported as `play/layer4-dynamic-slots` | `d0e2d2c3` | `abe0ed49` | `descriptor/host.rs`, `draw.rs`, see below |
| 8 | fork `play/layer4-1962-memory`: #1873 `81ebd7c6`, #2221 `df15bd1e` (in place of #1969), #2161 `e04431f7`, idle staging buffers released | `47bb09c9` | `80440ae9` | none |
| 9 | fork `gpu/shadow-stand-in`: opt-in `VICTORY_STAND_IN` | `669f7ff5` | `6188353e` | `draw.rs`, see below |

Between pieces 7 and 8: `cd2d17ec` "GPU tests: the root constant pixel
test follows layer 4's shader API". This commit changes test code only.

### Left out

- `play/layer4-2130-2389` (#2130 with #2222, and #2389): not final when
  this layer was handed over. It conflicts with piece 8 in `texture.rs`
  (`Staging::refresh_now`, the `Staging` fields) and `resource.rs` (the
  `placed` argument of `shared_texture` belongs in #2221's
  `host_texture()`), and with piece 1 in a test module. It is to follow
  as its own head.
- Upstream #2306 was not merged on its own: piece 5 carries it.

### Conflict resolutions

- Pieces 1 and 2: each side added a test at the same place in a test
  module. Both tests are kept.
- Piece 7 (in `d0e2d2c3`): `descriptor/host.rs`, layer 4's
  `fallback_samplers()` and the fix's `Slots` allocator were added at
  the same place; both are kept, `fallback_samplers()` first.
  `draw.rs`, layer 4's `SAMPLERS_OURS` next to `ROOT_VIEW_RUN`, whose
  doc comment the fix rewrote; both are kept. The fix branch's own merge
  of #2295 adds nothing here (piece 3).
- Piece 9, `draw.rs` `dispatch()`: layer 4's batched clamps end the
  list's open clamp window at the top of `dispatch()`; `669f7ff5` makes
  `dispatch()` a wrapper over `run_dispatch()`. Both are kept: the
  wrapper ends the window, runs `run_dispatch()`, and runs the stand-ins
  if it did not run. One line was added that neither side has:
  `instead()` ends an open clamp window before its stand-in clear, since
  on this tree every command that writes GPU memory does. It is reached
  only with `VICTORY_STAND_IN` set.
- `cd2d17ec`: #2295's test builds `HostShader` and calls `emit` and
  `hand_out` the way main has them. On this tree `emit` takes a
  `Target`, `HostShader` has `heap_usage` (#1962) and `group_waves`
  (piece 5), and `hand_out` takes the program's code. The fix fills in
  `Target::ANY`, `false`, `1` and `&[]`.

### Shader translator

Only piece 5 changes `crates/victory-shader` (13 files, 557 insertions,
105 deletions against `01c5d98d`), so the shader cache needs a new pass
for this layer. `git diff a47125be <head> -- crates/victory-shader
crates/victory-pipeline apps/victory` is empty: the translator is the
one of `gpu-run/layer4-wpg`, and a cache made for that head serves this
one.

Offline, `victory translate --host-lanes 32-32 --validate --list --jobs
6` over the title's shader archive (local title data, CPU only), base
`01c5d98d` against the binary built at `80440ae9`: 25,889 programs in
both; validated 25,701 before, 25,725 after; 0 lost; 24 gained (21 were
"console wave64 lanes in a group of more than 64 threads", 3 "a
whole-wave section of a wave64"). The listing matches the one of
`a47125be` program by program (0 lost, 0 gained). The binary of the
head, `6188353e`, was not listed itself; it differs from `80440ae9` in
`crates/victory-gpu` only.

### Build

`cargo build --locked --profile diag -p victory --features diagnostics`
at `6188353e`, through `heavy-run --light`, into `targets\layer4-final`:
`diag\victory.exe`, sha256
`5a9c0b14238d4775e5fb2bb78ab125d47f037e78d9a578feb7664b894b11be31`.
This file only changes docs, so the binary matches this layer's head.

### Switches

New and opt-in (unset: no change): `VICTORY_AMBIENT` (piece 4),
`VICTORY_STAND_IN` (piece 9). New and on by default: the staging and
placed-texture sweeps of piece 8 (`VICTORY_STAGING_SWEEP=0`,
`VICTORY_TEXTURE_SWEEP=0` turn them off for an A/B) and the library-save
thresholds of piece 6 (`VICTORY_LIBRARY_SAVE_MIN`, default 256;
`VICTORY_LIBRARY_SAVE_SECS`, default 60).

### Tests and review

Checked on this layer: `cargo check --locked -p victory-gpu -p
victory-shader --features victory-gpu/diagnostics --tests` passes at
`6188353e`. Run on the software adapter, on the port branch `d0e2d2c3`
with the test fix above applied by hand: the three tests of piece 7,
#2295's two tests and the two older dynamic-slot tests (7 passed). No
suite was run on this layer's head; each piece's own suite results are
with its branch. The title was not run.

A read-only review of the merged draw path at `abe0ed49` (pieces 1 to
7) found no conflict between the pieces. It noted that each
`VICTORY_AMBIENT` clear takes one more dynamic slot, held until the list
resets, and that `external::record` (already in layer 4) does not end an
open clamp window. Pieces 8 and 9 were not reviewed that way.
