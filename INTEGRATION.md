# Tester integration branch

This branch puts many open upstream pull requests together in one tree, so
that testers who run the title can build once and measure the fixes
together. It is **not an official release**. The maintainer has not
reviewed or merged any of these PRs, and each may still change or be
turned down upstream.

- Base: upstream `main` at `d2a4bae` (Marvisak/VIctory, "Merge pull request
  #129").
- Every PR comes in as its own merge commit (`git log --first-parent`), so
  you can find, revert or leave out each one.
- Before inclusion, every PR was built and tested on one Windows 11
  machine (NVIDIA RTX 5070, 32-lane waves only; WARP). That machine has no
  copy of the title. **Nobody has run this tree with the title yet.** It
  makes no claim about results in the title.

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
- #336 (hophaver): drop draws whose fresh tables cannot be made; 2^18
  dynamic slots
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

### Shaders: instructions checked against AMD's ISA reference

- #48 (flapwarp78): 64-bit texture atomics, and proven shader branches
  followed
- #25 (artor410): packed typed-buffer formats decoded
- #169 (Dufftwt): `tbuffer_load_format_d16` loads pack two 16-bit
  channels per VGPR
- #276 (Dufftwt): `s_ttracedata_imm` is SOPP opcode 0x28
- #168 (Dufftwt): SCC of `s_min`, `s_max` and `s_lshl1-4_add_u32` as the
  ISA sets it
- #110 (lordon1a): DXIL IsNaN in `med3`, so Intel Xe does not fold
  `fcmp uno`
- #154 (lordon1a): DXIL IsNaN for unordered, ordered and self float
  compares
- #98 (tedin7): `v_permlane16_b32` in whole-wave sections

### Shaders: descriptor tracking, signatures and validator fixes

- #307 (AntoninMichelq): waterfall loops over per-lane descriptors, and
  descriptors chosen by `s_cselect` (with a one-file fix to
  `apps/victory/src/translate.rs` so the CLI builds)
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
- #136 (mpDom, from `8ae232f`): only the descriptor-use selection and
  record-count reconstruction for descriptors used by `s_buffer_load_dword*`.
  The ordinary scalar load changes, a16 sampling, gather emulation, phi,
  signature, interpolation, conservative depth and LOD changes are excluded.
- #179 (TG0000): interpolation halves indexed on demand
- #176 (Yaekai): packed fp16 pixel inputs paired with unlisted signature
  rows
- #226 (gingertonik): test that an image read as float and used by an
  atomic is annotated twice
- #286 (tarcisioads): accept known `D3D_MIN_PRECISION` signature
  elements

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

The wave64 part is **#223 + #233 only**. #320 and #439, which build on
#233, are left out. Both are drafts: a whole-title
`translate --validate` run found #320 adds 20 compute modules that fail
DXIL validation (the driver still creates pipelines from them).

**Open report on #233:** on Windows 10 19044 with an RTX 3080 (driver
32.0.15.9649) and #175, NVIDIA's shader compiler faulted on some newly
accepted compute pipelines at world entry, one pipeline was refused with
`E_OUTOFMEMORY`, and then the device was removed. Three other NVIDIA
machines with newer drivers did not show it. #233 also emits 22 modules
that fail DXIL validation, but `main` already emits all 22.

### Test stability

- #433 (r4tedx): upstream's VALU GPU tests run one at a time. Without
  it, `cargo test` sometimes never finishes, with several GPU tests stuck
  together. On the integration machine, before #433, 4 of 12 runs of the
  shader tests hung; with it, 0 of 12 did.

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
- **#136 scalar buffer descriptor review:** retry descriptor data lowering
  only for programs refused by the existing path. Require a known, aligned
  table entry and use the same raw or structured view for the scalar load
  and record count.

## Left out on purpose

- #302: a spill-table load reads a register's current value, not its
  entry value (follow-up in #489)
- #193: its fallback reads a structured view through a raw handle,
  which returns 0 on NVIDIA; a title run with it (and other PRs) ended in
  a 0x887A0001 device removal
- Rest of #136: a16 bias read as f32; `image_gather4_l` ignores the
  sampler's address mode; missing phi inputs become zero. The full PR
  had 31 `E_OUTOFMEMORY` refusals in a title run.
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
- #346: needs adapting to #334's buffer-state tracker
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
- #294: root views refresh the whole rest of a large buffer copy
  (updated upstream since, not re-reviewed)
- #367: closed as a duplicate of #334
- #159 (r4tedx): needs adapting to #334's state tracker first
- #320, #439 (r4tedx): drafts, 20 modules failing validation on the
  title (see above)

## What this tree is NOT expected to fix

- **The world hang itself.** In-game runs of #334 still ended in
  `DXGI_ERROR_DEVICE_HUNG`, and nothing reviewed has been shown to stop
  the hang. This tree removes known errors around it, not the hang.
- **The top rows of issue #293's table** ("an address in an SGPR that is
  no root parameter", "a descriptor neither loaded nor built", "a
  structured buffer of unknown stride"). The #136 excerpt addresses
  "a scalar memory read as data" when the data is a known buffer
  descriptor's record count.
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
- Which PRs you left out or added, if you changed the tree.

Post **counts and messages only**. Never post game files, shader dumps,
extracted data, or screenshots of the title.
