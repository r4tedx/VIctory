//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it: [`Target`] says
//! which wave sizes the host runs, and a shader that needs another one does
//! not translate for it (rather than being refused at pipeline creation).
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed. An entry of root type `ResourceHeap` (`SamplerHeap`)
//!   is a guard slot of the bound view (sampler) heap, past the title's
//!   console dwords: every heap index the code computes at or above it
//!   reads that slot instead (an index past a heap hangs some GPUs). The
//!   host keeps a view heap's guard slot null and gives no table or title
//!   descriptor a slot at or above it; any sampler slot will do.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs, root SRVs and root UAVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` / `u<i>` in [`ROOT_CBV_SPACE`], `i` the title's root
//!   parameter index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`],
//!   [`HostShader::root_uavs`]); the address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). Grand Theft Auto VI's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** use a scalar integer UAV. The host binds R32_UINT
//!   for 32-bit operations and R32G32_UINT for 64-bit operations. The latter
//!   require SM 6.6 atomic support on typed and descriptor-heap resources.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

mod bitcode;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(test, windows))]
mod image_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
#[cfg(all(test, windows))]
mod lane_read_loop_tests;
mod lower;
mod sampler;
mod shader;
#[cfg(all(test, windows))]
mod split_tests;
mod threading;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{
    ROOT_CBV_SPACE, RawWindow, SAMPLER_SPACE, STRIDE_SPACE, StrideEntry, USER_DATA_SPACE,
};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// Root UAVs (raw buffers) the shader reads or writes: `u<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_uavs: Vec<u8>,
    /// True when the shader reads the runtime stride table (`t0` in
    /// [`STRIDE_SPACE`]).
    pub stride_table: bool,
    /// Compute: host thread groups per console thread group (1 unless the
    /// console's group is several wave64 waves emitted as one host group of
    /// 64 threads each, on a host without 64-lane waves). A dispatch of
    /// (x, y, z) console groups must run (x * group_waves, y, z) host
    /// groups: host group id x is console group x / group_waves, wave
    /// x % group_waves.
    pub group_waves: u32,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table`, or `ResourceHeap` / `SamplerHeap` for a
    /// heap's guard slot (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// The wave sizes (lanes per wave) the host GPU runs, as D3D12 reports them
/// (`D3D12_FEATURE_DATA_D3D12_OPTIONS1::WaveLaneCountMin` / `Max`). A
/// compute shader that declares the console's wave size translates only
/// for a host whose range holds it: NVIDIA's range is 32..32, so a console
/// wave64 program that reads other lanes needs 64-lane waves it cannot get.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Target {
    pub lanes_min: u32,
    pub lanes_max: u32,
}

impl Target {
    /// No limit known: every wave size is taken to run (the host refuses
    /// a pipeline it cannot run).
    pub const ANY: Target = Target {
        lanes_min: 4,
        lanes_max: 128,
    };

    /// Whether the host runs waves of `lanes`.
    pub fn runs(self, lanes: u32) -> bool {
        (self.lanes_min..=self.lanes_max).contains(&lanes)
    }

    /// The range as bytes, for cache keys.
    pub fn key(self) -> [u8; 8] {
        let mut k = [0; 8];
        k[..4].copy_from_slice(&self.lanes_min.to_le_bytes());
        k[4..].copy_from_slice(&self.lanes_max.to_le_bytes());
        k
    }
}

impl Default for Target {
    fn default() -> Self {
        Target::ANY
    }
}

/// A diagnostic: what the module this thread emitted last declares and
/// calls (thread-group size, wave size, `dx.op` operations), for reports of
/// a module the validator refuses. Nothing of the console program is in it.
pub fn last_module_summary() -> String {
    shader::last_summary()
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container, for any host ([`Target::ANY`]).
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_for(container, Target::ANY)
}

/// [`translate`] for a host that runs the wave sizes of `target`.
pub fn translate_for(container: &[u8], target: Target) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    // A compute program whose root arguments all sit past the 16 user SGPRs
    // finds them through the spill table `s[0:1]` points to.
    let mut p = ir::build_with(
        &native.patched_code(),
        native.wave64,
        native.vgprs,
        native.spilled(),
    )?;
    crate::sgpr_spill::forward(&mut p);
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit(&c, &native, &p, &st, &io, &res, target)
}

/// Emits a program whose earlier stages already ran, for `target`.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    target: Target,
) -> Result<HostShader, ShaderError> {
    let emit =
        |p: &ir::Program, st: &structure::Structured, io: &io::Io, res: &resource::Resources| {
            let first = emit_program(c, native, p, st, io, res, target);
            if matches!(
                first,
                Err(ShaderError::Unsupported {
                    what: lower::FIRST_INTERP_HALF,
                    ..
                })
            ) && let Ok(explicit) = io::raise_interps(p, native, true)
            {
                return emit_program(c, native, p, st, &explicit, res, target).or(first);
            }
            first
        };
    let original = emit(p, st, io, res);
    // Retry unsupported programs only. Threading can separate an unknown
    // register value from its apparent use on a mutually exclusive path.
    if !matches!(original, Err(ShaderError::Unsupported { .. })) {
        return original;
    }
    let edges = threading::constant_edges(p, st);
    if edges.is_empty() {
        return original;
    }
    let mut p = ir::thread_edges(p, &edges)?;
    let lower_threaded = |p: &ir::Program| {
        let st = structure::structure(p)?;
        let io = io::raise(p, native)?;
        let res = resource::track(p, &native.layout, native.stage, Some(&st))?;
        emit(p, &st, &io, &res)
    };
    let threaded = lower_threaded(&p);
    // Threading rebuilds SSA from registers. Forward spill reloads when
    // the threaded form still fails, preserving existing translations
    // that use the explicit reloads.
    if matches!(threaded, Err(ShaderError::Unsupported { .. }))
        && crate::sgpr_spill::forward(&mut p)
    {
        return lower_threaded(&p).or(threaded);
    }
    threaded
}

/// Tests: the first lowering of a compute program alone, without the
/// console flow it may stop to ask for ([`lower::CONSOLE_FLOW`]), or the
/// per-lane form only.
#[cfg(test)]
pub(crate) fn lower_compute(
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    per_lane_only: bool,
) -> Result<(), ShaderError> {
    lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        target: Target::ANY,
        console_cf: false,
        per_lane_only,
    })
    .map(|_| ())
}

#[cfg(test)]
pub(crate) use lower::CONSOLE_FLOW;

fn emit_program(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    target: Target,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    // Lowering declares the `SV_Barycentrics` a program reads, with the mode
    // it reads them in: a row of the console's own would be a second one.
    inputs.retain(|e| e.kind != 28);
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say. Conservative depth outputs
    // (`SV_DepthGreaterEqual`/`SV_DepthLessEqual`, kinds 18/19) require
    // `SV_Position` to use `linear_noperspective_centroid` (5) rather than
    // `linear_noperspective` (4) unless the shader runs at sample frequency.
    let conservative_depth = pixel && outputs.iter().any(|o| matches!(o.kind, 18 | 19));
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated.iter_mut() {
        e.interp = match (e.kind, e.comp) {
            (3, _) => {
                if conservative_depth {
                    5
                } else {
                    4
                }
            }
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            // Until a read says which mode (`lower`).
            (28, _) => 2,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    harmonize_row_interp(interpolated);
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (Grand Theft Auto VI's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let mut parts = lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
        target,
        console_cf: false,
        per_lane_only: false,
    };
    // Addresses traced for the console flow: EXEC masks more VGPR writes
    // where every lane runs every block.
    let console_res;
    let l = match lower::lower(&parts) {
        Err(ShaderError::Unsupported { what, .. }) if what == lower::CONSOLE_FLOW => {
            let console = structure::structure_console(p);
            let lowered = match &console {
                Ok(console) => {
                    match resource::track_console(p, &native.layout, native.stage, st, console) {
                        Ok(traced) => {
                            console_res = traced;
                            parts.st = console;
                            parts.res = &console_res;
                            parts.console_cf = true;
                            lower::lower(&parts)
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match e {
                        ShaderError::Unsupported { what, .. } => what,
                        _ => "a program the console structure cannot hold",
                    },
                }),
            };
            match lowered {
                Ok(l) => l,
                // What the console flow cannot lower keeps the per-lane form
                // where that translates; else the console flow's reason.
                Err(e) => {
                    parts.st = st;
                    parts.res = res;
                    parts.console_cf = false;
                    parts.per_lane_only = true;
                    lower::lower(&parts).map_err(|_| e)?
                }
            }
        }
        l => l?,
    };
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        stride_table: l.stride_table,
        group_waves: l.group_waves,
    })
}

/// DXIL requires every signature element packed into the same row to share an
/// interpolation mode. Integer attributes and `SV_IsFrontFace` can only be
/// `Constant` (1), so any float sibling packed into their row must be `1` as
/// well; after pixel-shader lowering, unread siblings also inherit the actual
/// interpolation mode of a read element in the same row.
pub(super) fn harmonize_row_interp(elements: &mut [container::SigElement]) {
    let mut rows: Vec<u32> = elements.iter().filter_map(|e| e.row).collect();
    rows.sort_unstable();
    rows.dedup();
    for row in rows {
        let forced_constant = elements
            .iter()
            .any(|e| e.row == Some(row) && (e.kind == 13 || (e.kind == 0 && e.comp != 9)));
        let mode = if forced_constant {
            Some(1)
        } else {
            elements
                .iter()
                .find(|e| e.row == Some(row) && e.usage != 0 && e.interp != 0)
                .map(|e| e.interp)
        };
        if let Some(m) = mode {
            for e in elements.iter_mut() {
                if e.row == Some(row) && e.interp != 0 && e.usage == 0 {
                    e.interp = m;
                }
            }
        }
    }
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        // DXBC encodes HLSL min16*/min10* types with a 32-bit ComponentType
        // (1..=3) and a non-zero D3D_MIN_PRECISION at +28 (1 FLOAT_16,
        // 2 FLOAT_2_8, 4 SINT_16, 5 UINT_16, 240 ANY_16, 241 ANY_10).
        // DXIL declares them as 32-bit with MinPrecision = 0.
        let min_prec = u32_at(part, at + 28)?;
        if !matches!(min_prec, 0 | 1 | 2 | 4 | 5 | 240 | 241) {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "an unknown minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::Target;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res, Target::ANY) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A pixel shader writing a constant colour. Every host builds it:
    /// `constant_color_is_a_well_formed_container` checks it without D3D12,
    /// and the Windows tests run it through dxil.dll and a device.
    fn constant_color() -> Vec<u8> {
        use super::bitcode::Attr;
        use super::shader::Shader;

        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    /// What D3D12 and the validator check before reading the bitcode, without
    /// either: the container parses, its size field is its length, its parts
    /// are the ones a pixel shader carries, and the digest is the one
    /// `container::hash` gives for the bytes after it.
    #[test]
    fn constant_color_is_a_well_formed_container() {
        let c = constant_color();
        let parsed = Container::parse(&c).unwrap();
        assert_eq!(crate::dxbc::u32_at(&c, 0x18).unwrap() as usize, c.len());
        let fourccs: Vec<_> = parsed.parts.iter().map(|p| p.fourcc.0).collect();
        assert_eq!(
            fourccs,
            [*b"SFI0", *b"ISG1", *b"OSG1", *b"PSV0", *b"HASH", *b"DXIL"]
        );
        let dxil = parsed.part(crate::dxbc::FourCC(*b"DXIL")).unwrap();
        assert_eq!(&dxil.data[8..12], b"DXIL");
        assert_eq!(parsed.digest, super::container::hash(&c[20..]));
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            stride_table: false,
            group_waves: 1,
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                stride_table: false,
                group_waves: 1,
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res, Target::ANY);
    }

    /// A compute program of 64 threads in one group: `words` as GFX10 code.
    pub(super) fn compute(words: &[u32], wave64: bool) -> crate::NativeProgram {
        compute_of(words, wave64, 64)
    }

    /// [`compute`] with `threads` threads in the group.
    pub(super) fn compute_of(words: &[u32], wave64: bool, threads: u32) -> crate::NativeProgram {
        crate::NativeProgram {
            stage: crate::HwStage::Cs,
            wave64,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 0,
                threads: [threads, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    pub(super) fn emit_for(
        native: &crate::NativeProgram,
        target: Target,
    ) -> Result<super::HostShader, crate::ShaderError> {
        let c = crate::Container {
            digest: [0; 16],
            version: (1, 0),
            parts: vec![],
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs)?;
        let st = crate::structure::structure(&p);
        let res = crate::resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
        let io = crate::io::raise(&p, native)?;
        let st = st?;
        super::emit(&c, native, &p, &st, &io, &res, target)
    }

    const NVIDIA: Target = Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    const RDNA: Target = Target {
        lanes_min: 32,
        lanes_max: 64,
    };

    #[test]
    fn console_lanes_translate_only_for_hosts_with_that_wave_size() {
        let words = [
            0xD760_0000,
            0x0001_0B00, // v_readlane_b32 s0, v0, 5
            0x7E02_0200, // v_mov_b32 v1, s0
            0xBF81_0000, // s_endpgm
        ];
        // One console wave per group: in the console flow on NVIDIA.
        let wave64 = compute(&words, true);
        assert!(emit_for(&wave64, Target::ANY).is_ok());
        assert!(emit_for(&wave64, RDNA).is_ok());
        assert!(emit_for(&wave64, NVIDIA).is_ok());
        // Two per group: each console wave is a host group of its own on
        // NVIDIA.
        let two = compute_of(&words, true, 128);
        assert_eq!(emit_for(&two, RDNA).map(|s| s.group_waves), Ok(1));
        assert_eq!(emit_for(&two, NVIDIA).map(|s| s.group_waves), Ok(2));
        // A group that is no whole number of console waves still stops.
        let partial = compute_of(&words, true, 96);
        assert!(emit_for(&partial, RDNA).is_ok());
        assert_eq!(
            emit_for(&partial, NVIDIA).map(|s| s.group_waves),
            Err(crate::ShaderError::Unsupported {
                offset: 0,
                what: "console wave64 lanes in a group of more than 64 threads on a host without 64-lane waves",
            })
        );
        // The same code in waves of 32 runs on either.
        let wave32 = compute(&words, false);
        assert!(emit_for(&wave32, NVIDIA).is_ok());
        assert!(emit_for(&wave32, RDNA).is_ok());
    }

    #[test]
    #[cfg(windows)]
    fn what_translates_for_the_host_creates_a_pipeline() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let target = host.target();
        eprintln!("host waves {}-{}", target.lanes_min, target.lanes_max);
        let words = [0xD760_0000, 0x0001_0B00, 0x7E02_0200, 0xBF81_0000];
        for wave64 in [false, true] {
            match emit_for(&compute_of(&words, wave64, 128), target) {
                Ok(cs) => {
                    if let Err(e) = host.compute_pipeline(&cs) {
                        panic!("wave64 {wave64}: translated for the host, refused by it: {e}");
                    }
                }
                Err(e) => {
                    assert!(wave64 && !target.runs(64), "{e}");
                    // What it replaces: the host refuses the console's
                    // wave size at pipeline creation.
                    let any = emit_for(&compute_of(&words, true, 128), Target::ANY).unwrap();
                    assert!(host.compute_pipeline(&any).is_err());
                }
            }
        }
    }

    /// On every hardware adapter: a wave64 program that reads other lanes
    /// translates in groups of one console wave (natively or in the console
    /// flow) and the adapter takes it; in groups of two it is one host
    /// group where the adapter runs 64-lane waves and two where it does not.
    /// On a machine with an AMD and an NVIDIA GPU this checks both answers.
    #[test]
    #[cfg(windows)]
    fn each_adapter_gets_what_it_runs() {
        let Ok(names) = super::host::adapters() else {
            eprintln!("no DXGI; skipped");
            return;
        };
        let words = [0xD760_0000, 0x0001_0B00, 0x7E02_0200, 0xBF81_0000];
        for name in names {
            let Ok(host) = super::host::Host::on(&name) else {
                eprintln!("{name}: no SM 6.6 device; skipped");
                continue;
            };
            let target = host.target();
            eprintln!("{name}: waves {}-{}", target.lanes_min, target.lanes_max);
            let one =
                emit_for(&compute(&words, true), target).unwrap_or_else(|e| panic!("{name}: {e}"));
            host.compute_pipeline(&one)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let two = emit_for(&compute_of(&words, true, 128), target)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                two.group_waves,
                if target.runs(64) { 1 } else { 2 },
                "{name}"
            );
            host.compute_pipeline(&two)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    #[test]
    fn programs_reading_no_other_lanes_translate_for_any_wave_size() {
        let words = [
            0x7E02_0300, // v_mov_b32 v1, v0
            0xBF81_0000, // s_endpgm
        ];
        assert!(emit_for(&compute(&words, true), NVIDIA).is_ok());
    }

    #[cfg(windows)]
    fn validate_if_available(dxil: &[u8]) {
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let mut copy = dxil.to_vec();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy)
            .unwrap_or_else(|e| panic!("dxil.dll rejected shader:\n{e}"));
        assert_eq!(copy[4..20], ours[..], "container hash disagrees");
    }

    fn emit_synthetic(
        parts: &[(FourCC, &[u8])],
        native: &NativeProgram,
    ) -> Result<HostShader, ShaderError> {
        let bytes = crate::dxbc::build(parts);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, native).unwrap();
        emit(&c, native, &p, &st, &io, &res, Target::ANY)
    }

    fn scalar_descriptor_program(kind: crate::ResourceType, stride: u32) -> NativeProgram {
        use crate::native::{ResourceType, RootParam, UsedResource};
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF408_0100, 0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
            0xF420_0202, 0xFA00_000C, // s_buffer_load_dword s8, s[4:7], 12
            0xF408_0300, 0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 32
            0x7E00_0208, // v_mov_b32 v0, s8 (the scalar load)
            0x7E02_0206, // v_mov_b32 v1, s6 (descriptor dword 2)
            0xE070_0000, 0x8003_0000, // buffer_store_dword v0, off, s[12:15], 0
            0xE070_0004, 0x8003_0100, // buffer_store_dword v1, off, s[12:15], 0 offset:4
            0xBF81_0000, // s_endpgm
        ];
        let input = UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride,
            dwords: 4,
            user_data: 0,
            slot: 0,
        };
        NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 2,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![
                input,
                UsedResource {
                    kind: ResourceType::Uav,
                    stride: 0,
                    table_offset_dwords: 8,
                    ..input
                },
            ],
            io: StageIo::Cs(ComputeIo { rsrc2: 1 << 1, threads: [1, 1, 1] }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    /// The user data of a shader whose one table starts at byte `table` of
    /// the view heap, dispatched with `views` views: the table's address and
    /// the heap's guard slot (the null slot the dispatch keeps after them).
    #[cfg(windows)]
    fn table_user_data(cs: &HostShader, table: u32, views: usize) -> Vec<u32> {
        let mut data = vec![0; cs.user_data_dwords().max(1) as usize];
        for u in &cs.user_data {
            data[u.index as usize] = match u.root {
                RootType::Table => table,
                root => super::host::guard(root, views, 0)
                    .unwrap_or_else(|| panic!("user data of {root:?}")),
            };
        }
        data
    }

    #[test]
    fn scalar_buffer_descriptor_data_loads_values_and_record_count() {
        for (stride, count) in [(0, 24), (8, 3)] {
            let native = scalar_descriptor_program(crate::ResourceType::Srv, stride);
            let cs = emit_synthetic(&[], &native).unwrap();
            #[cfg(windows)]
            {
                use super::host::{Dispatch, Host, Root, Stride, View};
                validate_if_available(&cs.dxil);
                for (backend, warp) in [("hardware", false), ("WARP", true)] {
                    let Ok(host) = Host::new(warp) else {
                        eprintln!("no SM 6.6 {backend} device; skipped dispatch");
                        continue;
                    };
                    let (root, pso) = host.compute_pipeline(&cs).unwrap();
                    let view_stride = if stride == 0 {
                        Stride::Raw
                    } else {
                        Stride::Structured(stride)
                    };
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![(0, Root::Constants(table_user_data(&cs, 32, 3)))],
                                views: &[
                                    View::Buffer { data: &[0; 8], stride: view_stride },
                                    View::Buffer { data: &[11, 22, 33, 44, 55, 66], stride: view_stride },
                                    View::RawUav(2),
                                ],
                                samplers: &[],
                                groups: [1, 1, 1],
                            },
                        )
                        .unwrap();
                    assert_eq!(out, [vec![44, count]], "{backend}, stride {stride}");
                    eprintln!("{backend}, stride {stride}: loaded 44 and record count {count}");
                }
            }
            #[cfg(not(windows))]
            let _ = (cs, count);
        }
    }

    #[test]
    fn scalar_buffer_descriptor_data_requires_a_known_buffer_view() {
        use crate::ResourceType::{Cbv, Srv};
        let cbv = scalar_descriptor_program(Cbv, 0);
        let mut wrong_slot = scalar_descriptor_program(Srv, 0);
        wrong_slot.used[0].slot = 1;
        let mut unaligned = scalar_descriptor_program(Srv, 0);
        unaligned.code[8..12].copy_from_slice(&0xFA00_0010u32.to_le_bytes());
        let invalid_stride = scalar_descriptor_program(Srv, 6);
        for (native, what) in [
            (cbv, "a constant buffer's descriptor read as data"),
            (wrong_slot, "a scalar buffer descriptor without a known table view"),
            (unaligned, "a scalar buffer descriptor without a known table view"),
            (invalid_stride, "a scalar buffer descriptor with an unsupported stride"),
        ] {
            assert_eq!(
                emit_synthetic(&[], &native),
                Err(ShaderError::Unsupported { offset: 44, what })
            );
        }
    }

    /// Programs around a reconstructed scalar view that could translate
    /// wrongly: each reads back the right values on the hosts or is refused.
    #[cfg(windows)]
    mod scalar_views {
        use super::*;
        use crate::ResourceType::{Srv, Uav};
        use crate::emit::host::{Dispatch, Host, Root, Stride, View};

        const S_ENDPGM: u32 = 0xBF81_0000;
        const NO_SOFFSET: u32 = 125;
        const S_LOAD_DWORDX4: u32 = 2;
        const S_BUFFER_LOAD_DWORD: u32 = 8;
        const S_BUFFER_LOAD_DWORDX4: u32 = 10;
        /// `s_add_u32` and `s_lshl_b32`.
        const ADD: u32 = 0;
        const SHL: u32 = 0x1E;

        /// `s_mov_b32 s[sdst], n` for a constant up to 64.
        const fn s_mov(sdst: u32, n: u32) -> u32 {
            0xBE80_0300 | sdst << 16 | (128 + n)
        }

        /// SOP2 `op s[sdst], a, b` (an SGPR, or 128 + n for the constant n).
        const fn sop2(op: u32, sdst: u32, a: u32, b: u32) -> u32 {
            0x8000_0000 | op << 23 | sdst << 16 | b << 8 | a
        }

        /// SMEM `op s[sdata..], s[sbase..], soffset, offset`.
        const fn smem(op: u32, sdata: u32, sbase: u32, offset: u32, soffset: u32) -> [u32; 2] {
            [
                0x3D << 26 | op << 18 | sdata << 6 | sbase >> 1,
                soffset << 25 | offset,
            ]
        }

        /// `v_mov_b32 v[vdst], src` (an SGPR, or 128 + n for the constant n).
        const fn v_mov(vdst: u32, src: u32) -> u32 {
            0x7E00_0200 | vdst << 17 | src
        }

        /// `buffer_store_dword v[vdata], off, s[srsrc..], 0 offset:offset`.
        const fn store(vdata: u32, srsrc: u32, offset: u32) -> [u32; 2] {
            [
                0xE070_0000 | offset,
                0x80 << 24 | (srsrc >> 2) << 16 | vdata << 8,
            ]
        }

        /// The descriptor at the table's byte 0 into `s[4:7]`, the output
        /// UAV's at byte 32 into `s[12:15]`.
        const LOADS: [u32; 5] = {
            let d = smem(S_LOAD_DWORDX4, 4, 0, 0, NO_SOFFSET);
            let out = smem(S_LOAD_DWORDX4, 12, 0, 32, NO_SOFFSET);
            [s_mov(1, 0), d[0], d[1], out[0], out[1]]
        };

        /// `code` in [`scalar_descriptor_program`]'s layout: a table at
        /// `s0` holding a buffer view of `kind` and `stride` at byte 0 and
        /// a raw UAV at byte 32.
        fn translate(
            code: &[&[u32]],
            kind: crate::ResourceType,
            stride: u32,
        ) -> Result<HostShader, ShaderError> {
            let mut native = scalar_descriptor_program(kind, stride);
            native.vgprs = 4;
            native.code = code
                .iter()
                .copied()
                .flatten()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            emit_synthetic(&[], &native)
        }

        /// What the UAVs among `views` hold after `cs` runs on each
        /// available host, its table at heap slot 1.
        fn run(cs: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<Vec<u32>>)> {
            // One dispatch at a time, as in `valu_tests::run_cases`.
            static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            validate_if_available(&cs.dxil);
            let mut out = Vec::new();
            for (backend, warp) in [("hardware", false), ("WARP", true)] {
                let Ok(host) = Host::new(warp) else {
                    eprintln!("no SM 6.6 {backend} device; skipped dispatch");
                    continue;
                };
                let (root, pso) = host.compute_pipeline(cs).unwrap();
                let user_data = table_user_data(cs, 32, views.len());
                let got = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(user_data))],
                            views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap();
                out.push((backend, got));
            }
            out
        }

        /// A decoy at heap slot 0, six dwords viewed with `stride` at slot 1
        /// and an output UAV of `out` dwords at slot 2.
        fn run_over_six_dwords(
            cs: &HostShader,
            stride: u32,
            out: u32,
        ) -> Vec<(&'static str, Vec<u32>)> {
            let stride = match stride {
                0 => Stride::Raw,
                n => Stride::Structured(n),
            };
            let views = [
                View::Buffer { data: &[0; 8], stride },
                View::Buffer { data: &[11, 22, 33, 44, 55, 66], stride },
                View::RawUav(out),
            ];
            run(cs, &views)
                .into_iter()
                .map(|(backend, mut uavs)| (backend, uavs.remove(0)))
                .collect()
        }

        /// A descriptor the program stores through is a UAV: scalar loads
        /// of it, before and after the store, read through the same view,
        /// whether lowering reconstructs the descriptor (the scalar load is
        /// its first use) or takes the table's word that it is a UAV.
        #[test]
        fn a_scalar_load_of_a_descriptor_the_program_stores_through() {
            let early = smem(S_BUFFER_LOAD_DWORD, 8, 4, 8, NO_SOFFSET);
            let program = |first_use_is_scalar: bool| -> [Vec<u32>; 5] {
                [
                    LOADS.to_vec(),
                    // s8 = dword 2 of the buffer before the store, or 0.
                    if first_use_is_scalar { early.to_vec() } else { vec![s_mov(8, 0)] },
                    // Dword 2 of the buffer = 55, then s9 reads it back.
                    [&[v_mov(2, 128 + 55)][..], &store(2, 4, 8)].concat(),
                    smem(S_BUFFER_LOAD_DWORD, 9, 4, 8, NO_SOFFSET).to_vec(),
                    // Out: s8, s9 and the descriptor's dword 2 (its bytes).
                    [
                        &[v_mov(0, 8), v_mov(1, 9), v_mov(3, 6)][..],
                        &store(0, 12, 0),
                        &store(1, 12, 4),
                        &store(3, 12, 8),
                        &[S_ENDPGM],
                    ]
                    .concat(),
                ]
            };
            for first_use_is_scalar in [true, false] {
                let code = program(first_use_is_scalar);
                let code: Vec<&[u32]> = code.iter().map(Vec::as_slice).collect();
                let cs = translate(&code, Uav, 0).unwrap();
                let views = [
                    View::Buffer { data: &[0; 8], stride: Stride::Raw },
                    View::RawUav(4),
                    View::RawUav(3),
                ];
                for (backend, got) in run(&cs, &views) {
                    assert_eq!(
                        got,
                        [vec![0, 0, 55, 0], vec![0, 55, 16]],
                        "{backend}, scalar use first: {first_use_is_scalar}"
                    );
                }
            }
            // A structured UAV: the store wants a raw view, the scalar load
            // the table's structured one.
            let code = program(true);
            let code: Vec<&[u32]> = code.iter().map(Vec::as_slice).collect();
            assert_eq!(
                translate(&code, Uav, 8),
                Err(ShaderError::Unsupported {
                    offset: 76,
                    what: "incompatible scalar and vector views of a buffer descriptor",
                })
            );
        }

        /// A scalar load ignores the low two bits of its address: at byte
        /// 13 it reads dword 3, from an immediate or a register offset or
        /// their sum, and a wide load at byte 6 reads dwords 1 to 4 across
        /// the elements of a structured view. This holds the values read,
        /// not the masking: the hardware host and WARP both round an
        /// unaligned load down themselves.
        #[test]
        fn scalar_loads_of_a_reconstructed_descriptor_at_unaligned_offsets() {
            let end: [&[u32]; 4] = [
                &[v_mov(0, 8), v_mov(1, 6)],
                &store(0, 12, 0),
                &store(1, 12, 4),
                &[S_ENDPGM],
            ];
            let cases: [(&str, [u32; 2], u32); 3] = [
                ("immediate 13", smem(S_BUFFER_LOAD_DWORD, 8, 4, 13, NO_SOFFSET), 44),
                ("register 13", smem(S_BUFFER_LOAD_DWORD, 8, 4, 0, 9), 44),
                ("immediate 6 + register 13", smem(S_BUFFER_LOAD_DWORD, 8, 4, 6, 9), 55),
            ];
            for (stride, count) in [(0, 24), (8, 3)] {
                for (name, load, value) in &cases {
                    let code = [&LOADS[..], &[s_mov(9, 13)], load, end[0], end[1], end[2], end[3]];
                    let cs = translate(&code, Srv, stride).unwrap();
                    for (backend, got) in run_over_six_dwords(&cs, stride, 2) {
                        assert_eq!(got, [*value, count], "{backend}, stride {stride}, {name}");
                    }
                }
                let code: [&[u32]; 10] = [
                    &LOADS,
                    &smem(S_BUFFER_LOAD_DWORDX4, 8, 4, 6, NO_SOFFSET),
                    &[v_mov(0, 8), v_mov(1, 9), v_mov(2, 10), v_mov(3, 11)],
                    &store(0, 12, 0),
                    &store(1, 12, 4),
                    &store(2, 12, 8),
                    &store(3, 12, 12),
                    &[v_mov(0, 6)],
                    &store(0, 12, 16),
                    &[S_ENDPGM],
                ];
                let cs = translate(&code, Srv, stride).unwrap();
                for (backend, got) in run_over_six_dwords(&cs, stride, 5) {
                    assert_eq!(got, [22, 33, 44, 55, count], "{backend}, stride {stride}, wide");
                }
            }
        }

        /// One load of the descriptor gives the record count that bounds a
        /// loop, a twin load of the same dwords inside the loop gives the
        /// descriptor each pass reads a dword through: the count comes from
        /// the view the twin's scalar loads use.
        #[test]
        fn a_twin_load_in_a_loop_reads_the_descriptor_the_count_came_from() {
            // Raw: byte offsets 0, 4 .. below the 24 bytes. Structured: the
            // first dword of each of the 3 records of 8 bytes.
            for (stride, shift, step, sum, count) in [(0, 0, 4, 231, 24), (8, 3, 1, 99, 3)] {
                let code: [&[u32]; 9] = [
                    &LOADS,                                      // words 0-4
                    &[s_mov(20, 0), s_mov(21, 0)],               // 5-6: index, sum
                    &smem(S_LOAD_DWORDX4, 16, 0, 0, NO_SOFFSET), // 7-8: the twin
                    &[sop2(SHL, 22, 20, 128 + shift)],           // 9: byte offset
                    &smem(S_BUFFER_LOAD_DWORD, 8, 16, 0, 22),    // 10-11
                    &[
                        sop2(ADD, 21, 21, 8),          // 12: sum += the dword
                        sop2(ADD, 20, 20, 128 + step), // 13: the next index
                        0xBF0A_0614,                   // 14: s_cmp_lt_u32 s20, s6
                        0xBF85_FFF7,                   // 15: s_cbranch_scc1, to 7
                        v_mov(0, 21),
                        v_mov(1, 6),
                    ],
                    &store(0, 12, 0),
                    &store(1, 12, 4),
                    &[S_ENDPGM],
                ];
                let cs = translate(&code, Srv, stride).unwrap();
                for (backend, got) in run_over_six_dwords(&cs, stride, 2) {
                    assert_eq!(got, [sum, count], "{backend}, stride {stride}");
                }
            }
        }
    }

    fn sig_elem(
        name: &str,
        index: u32,
        kind: u8,
        comp: u8,
        row: Option<u32>,
        col: u8,
        cols: u8,
    ) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 0,
            row,
            col,
            cols,
            usage: 0,
            half: false,
        }
    }

    #[test]
    fn conservative_depth_sets_position_centroid_and_validates() {
        for (depth_name, depth_kind) in
            [("SV_DepthLessEqual", 18u8), ("SV_DepthGreaterEqual", 19u8)]
        {
            // v_mov_b32 v0, 0.5; exp mrtz v0, off, off, off done; s_endpgm
            let words = [
                0x7E00_02F0u32, // v_mov_b32 v0, 0.5
                0xF800_0881,    // exp mrtz (target 8), en=0x1, done=1
                0x0000_0000,    // vsrc0=v0
                0xBF81_0000,    // s_endpgm
            ];
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: StageIo::Ps {
                    inputs: vec![],
                    input_addr: 0,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let isg1 =
                container::signature(&[sig_elem("SV_Position", 0, 3, 9, Some(0), 0, 4)], true);
            let osg1 =
                container::signature(&[sig_elem(depth_name, 0, depth_kind, 9, None, 0, 1)], false);
            let host =
                emit_synthetic(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)], &native).unwrap();
            #[cfg(windows)]
            validate_if_available(&host.dxil);
        }
    }

    #[test]
    fn row_interp_harmonizes_packed_elements_and_validates() {
        // 1) Unit check on harmonize_row_interp for packed uint + float and unread siblings.
        let mut elems = vec![
            SigElement {
                interp: 1,
                ..sig_elem("COLOR", 1, 0, 5, Some(1), 0, 1)
            },
            SigElement {
                interp: 2,
                ..sig_elem("COLOR", 2, 0, 9, Some(1), 1, 1)
            },
            SigElement {
                interp: 1,
                usage: 1,
                ..sig_elem("TEXCOORD", 0, 0, 9, Some(2), 0, 2)
            },
            SigElement {
                interp: 2,
                ..sig_elem("TEXCOORD", 1, 0, 9, Some(2), 2, 2)
            },
        ];
        harmonize_row_interp(&mut elems);
        assert_eq!(elems[1].interp, 1);
        assert_eq!(elems[3].interp, 1);

        // 2) Full PS with SV_IsFrontFace + unread float in row 1 AND a read
        //    flat float + unread float in row 2 -> validates with dxil.dll.
        // v_interp_mov_f32 v0, p0, attr0.x; exp mrt0 v0, v0, v0, v0 done; s_endpgm
        let words = [
            0xC802_0002u32, // v_interp_mov_f32 v0, p0, attr0.x
            0xF800_080F,    // exp mrt0, en=0xF, done=1
            0x0000_0000,    // vsrc0..3 = v0
            0xBF81_0000,    // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 2,
                    mask: 0x3,
                    attr: 0,
                    flat: true,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let isg1 = container::signature(
            &[
                sig_elem("SV_Position", 0, 3, 9, Some(0), 0, 4),
                sig_elem("TEXCOORD", 0, 0, 9, Some(1), 0, 1),
                sig_elem("SV_IsFrontFace", 0, 13, 5, Some(1), 1, 1),
                sig_elem("TEXCOORD", 1, 0, 9, Some(2), 0, 2),
                sig_elem("TEXCOORD", 2, 0, 9, Some(2), 2, 2),
            ],
            true,
        );
        let osg1 = container::signature(&[sig_elem("SV_Target", 0, 16, 9, Some(0), 0, 4)], false);
        let host =
            emit_synthetic(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)], &native).unwrap();
        #[cfg(windows)]
        validate_if_available(&host.dxil);
    }

    #[test]
    fn signature_part_preserves_element_order_within_row_and_validates() {
        // Two elements in row 1 where the earlier element has a higher column
        // (col=3, w) than the later element (col=0, xyz).
        let row1_elems = [
            sig_elem("SV_Position", 0, 3, 9, Some(0), 0, 4),
            sig_elem("TEXCOORD", 3, 0, 9, Some(1), 3, 1),
            sig_elem("NORMAL", 0, 0, 9, Some(1), 0, 3),
        ];
        let isg1 = container::signature(&row1_elems, true);
        let parsed = signature(&isg1).unwrap();
        assert_eq!(parsed[1].name, "TEXCOORD");
        assert_eq!(parsed[2].name, "NORMAL");

        // Validate through full PS emission so dxil.dll verifies ISG1 matches
        // the DXIL signature metadata byte for byte.
        // v_mov_b32 v0, 0; exp mrt0 v0, v0, v0, v0 done; s_endpgm
        let ps_words = [0x7E00_0280u32, 0xF800_080F, 0x0000_0000, 0xBF81_0000];
        let ps_native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: ps_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let osg1 = container::signature(&[sig_elem("SV_Target", 0, 16, 9, Some(0), 0, 4)], false);
        let ps =
            emit_synthetic(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)], &ps_native).unwrap();
        #[cfg(windows)]
        validate_if_available(&ps.dxil);

        // Also validate a VS whose OSG1 has the same out-of-column row order.
        // v_mov_b32 v0, 0; exp pos0 v0..v0 done; exp param0 v0..v0; s_endpgm
        let vs_words = [
            0x7E00_0280u32,
            0xF800_00CF, // exp pos0 (target 12), en=0xF
            0x0000_0000,
            0xF800_0A0F, // exp param0 (target 32), en=0xF, done=1
            0x0000_0000,
            0xBF81_0000,
        ];
        let vs_native = NativeProgram {
            stage: HwStage::Vs,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Vs {
                inputs: vec![],
                outputs: vec![crate::native::VsOutput {
                    usage: crate::native::Usage::Generic,
                    row: 1,
                    param: 0,
                    fp16: false,
                    pack2: false,
                }],
            },
            code: vs_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let vs_osg1 = container::signature(&row1_elems, false);
        let vs = emit_synthetic(&[(FourCC::OSG1, &vs_osg1)], &vs_native).unwrap();
        #[cfg(windows)]
        validate_if_available(&vs.dxil);
    }

    /// A compute program (8x8x1) of `words` sampling a 2D texture from a
    /// root table, translated.
    fn compute_sample(words: &[u32]) -> HostShader {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![crate::native::RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 8,
                dwords: 8,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [8, 8, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        emit_synthetic(&[], &native).unwrap()
    }

    /// The PSV0 runtime info size, 48 bytes up to validator 1.7, 52 (info
    /// version 3) for validator 1.8.
    fn psv0_size(cs: &HostShader) -> u32 {
        let parsed = Container::parse(&cs.dxil).unwrap();
        let psv0 = parsed.part(FourCC::PSV0).unwrap().data;
        crate::dxbc::u32_at(psv0, 0).unwrap()
    }

    #[cfg(windows)]
    #[test]
    fn texture_array_lod_uses_only_spatial_coordinates() {
        let target = sig_elem("SV_Target", 0, 16, 9, Some(0), 0, 4);
        let output = container::signature(&[target], false);
        // 1D array and 2D array.
        for dim in [4, 5] {
            let words = [
                0xBE81_0380u32, // s_mov_b32 s1, 0
                0xF40C_0100,
                0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], 0
                0xBE8C_0380, // s_mov_b32 s12, 0
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xBE8E_0380, // s_mov_b32 s14, 0
                0xBE8F_0380, // s_mov_b32 s15, 0
                0x7E00_02F0, // v_mov_b32 v0, 0.5
                0x7E02_02F0, // v_mov_b32 v1, 0.5
                0x7E04_0280, // v_mov_b32 v2, 0
                0xF180_0100 | dim << 3,
                0x0061_0300, // image_get_lod v3, v[0:2], s[4:11], s[12:15]
                0xF800_180F,
                0x0303_0303, // exp mrt0 v3, v3, v3, v3 done
                0xBF81_0000,
            ];
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![crate::native::RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![crate::native::UsedResource {
                    kind: crate::native::ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 8,
                    dwords: 8,
                    user_data: 0,
                    slot: 0,
                }],
                io: StageIo::Ps {
                    inputs: vec![],
                    input_addr: 0,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let shader = emit_synthetic(&[(FourCC::OSG1, &output)], &native).unwrap();
            validate_if_available(&shader.dxil);
        }
    }

    #[test]
    fn compute_shader_implicit_sample_takes_quad_gradients_and_validates() {
        // Synthetic CS (8x8x1): loads a 2D texture SRV from a root table (s0)
        // and samples it with an immediate sampler (s[8:11]) via `image_sample`.
        // s_load_dwordx8 s[0:7], s[12:13], 0x0 (table in s12)
        // Wait: in resource::track, a 1-SGPR table pointer uses s_load_dwordx8
        // from s[0:1] when layout has RootType::Table at user_data=0!
        // Encode:
        //   s_mov_b32 s1, 0             (high half of 64-bit table base)
        //   s_load_dwordx8 s[4:11], s[0:1], 0x0
        //   s_mov_b32 s12, 0; s_mov_b32 s13, 0; s_mov_b32 s14, 0; s_mov_b32 s15, 0
        //   v_mov_b32 v0, 0.5; v_mov_b32 v1, 0.5
        //   image_sample v0, v[0:1], s[4:11], s[12:15] dmask:0x1 dim:SQ_RSRC_IMG_2D
        //   s_endpgm
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0100,    // s_load_dwordx8 s[4:11], s[0:1], 0
            0xFA00_0000,    // soffset=NULL, offset=0
            0xBE8C_0380,    // s_mov_b32 s12, 0
            0xBE8D_0380,    // s_mov_b32 s13, 0
            0xBE8E_0380,    // s_mov_b32 s14, 0
            0xBE8F_0380,    // s_mov_b32 s15, 0
            0x7E00_02F0,    // v_mov_b32 v0, 0.5
            0x7E02_02F0,    // v_mov_b32 v1, 0.5
            // MIMG image_sample (op=0x20): w0 = 0xF000_0000 | (0x20 << 18) | (1 << 8) | (1 << 3)
            0xF080_0108,
            // w1: vaddr=v0, vdata=v0, srsrc=s[4:11] (1), ssamp=s[12:15] (3)
            (3 << 21) | (1 << 16),
            0xBF81_0000, // s_endpgm
        ];
        // The console quad's differences as gradients: no implicit
        // derivatives, so no validator 1.8 (PSV0 RuntimeInfo2, 48 bytes).
        let cs = compute_sample(&words);
        assert_eq!(psv0_size(&cs), 48);
        #[cfg(windows)]
        {
            validate_if_available(&cs.dxil);
            if let Ok(host) = super::host::Host::new(false) {
                host.compute_pipeline(&cs).unwrap();
            }
        }
    }

    #[test]
    fn compute_shader_implicit_comparison_sample_uses_valver_1_8_and_validates() {
        // As above with `image_sample_c` (reference in v0, coordinates in
        // v[1:2]): SM 6.6 has no comparison sample with gradients, so the
        // host's implicit derivatives, which DXIL takes in compute from
        // validator 1.8 on.
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0100,    // s_load_dwordx8 s[4:11], s[0:1], 0
            0xFA00_0000,    // soffset=NULL, offset=0
            0xBE8C_0380,    // s_mov_b32 s12, 0
            0xBE8D_0380,    // s_mov_b32 s13, 0
            0xBE8E_0380,    // s_mov_b32 s14, 0
            0xBE8F_0380,    // s_mov_b32 s15, 0
            0x7E00_02F0,    // v_mov_b32 v0, 0.5
            0x7E02_02F0,    // v_mov_b32 v1, 0.5
            0x7E04_02F0,    // v_mov_b32 v2, 0.5
            // MIMG image_sample_c (op=0x28): w0 = 0xF000_0000 | (0x28 << 18) | (1 << 8) | (1 << 3)
            0xF0A0_0108,
            // w1: vaddr=v0, vdata=v0, srsrc=s[4:11] (1), ssamp=s[12:15] (3)
            (3 << 21) | (1 << 16),
            0xBF81_0000, // s_endpgm
        ];
        let cs = compute_sample(&words);
        // PSV0 must use RuntimeInfo3 (52 bytes) for validator 1.8.
        assert_eq!(psv0_size(&cs), 52);
        #[cfg(windows)]
        {
            validate_if_available(&cs.dxil);
            if let Ok(host) = super::host::Host::new(false) {
                host.compute_pipeline(&cs).unwrap();
            }
        }
    }

    #[test]
    fn biased_16bit_image_sample_unpacks_f32_bias_and_f16_coords_and_validates() {
        // Synthetic PS executing `image_sample_b v0, v[1:2], s[4:11], s[12:15]
        // dmask:0x1 dim:SQ_RSRC_IMG_2D a16` where v1 is a 32-bit f32 bias
        // (-1.0) and v2 holds packed f16 (u, v) = (0.5, 0.5), then exporting v0.
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0100,    // s_load_dwordx8 s[4:11], s[0:1], 0
            0xFA00_0000,    // soffset=NULL, offset=0
            0xBE8C_0380,    // s_mov_b32 s12, 0
            0xBE8D_0380,    // s_mov_b32 s13, 0
            0xBE8E_0380,    // s_mov_b32 s14, 0
            0xBE8F_0380,    // s_mov_b32 s15, 0
            0x7E02_02F3,    // v_mov_b32 v1, -1.0 (32-bit f32 bias)
            0x7E04_02FF,    // v_mov_b32 v2, lit(0x3800_3800) (packed f16 0.5, 0.5)
            0x3800_3800,
            // MIMG image_sample_b (op=0x25): dmask=0x1, dim=1 (2D), a16=1 (bit 30 of w1)
            0xF000_0000 | (0x25 << 18) | (1 << 8) | (1 << 3),
            (1 << 30) | (3 << 21) | (1 << 16) | 1, // vaddr=v1 (v[1:2]), vdata=v0
            0xF800_080F,                           // exp mrt0 v0, v0, v0, v0 done
            0x0000_0000,
            0xBF81_0000, // s_endpgm
        ];
        // Sanity-check the synthetic encoding decodes as `image_sample_b ... a16`.
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let decoded = crate::gfx10::decode(&code).unwrap();
        let sample = decoded
            .iter()
            .find(|i| i.name() == Some("image_sample_b"))
            .expect("image_sample_b decoded");
        assert!(matches!(
            sample.fields(false),
            crate::gfx10::Fields::Image {
                a16: true,
                dim: 1,
                ..
            }
        ));

        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![crate::native::RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 8,
                dwords: 8,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code,
        };
        let osg1 = container::signature(&[sig_elem("SV_Target", 0, 16, 9, Some(0), 0, 4)], false);
        let ps = emit_synthetic(&[(FourCC::OSG1, &osg1)], &native).unwrap();
        #[cfg(windows)]
        validate_if_available(&ps.dxil);
    }

    #[test]
    fn signature_accepts_known_min_precision_and_rejects_unknown() {
        let base = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        };
        for (min_prec, comp_type, expected_comp) in [
            (1u32, 3u32, 9u8), // D3D_MIN_PRECISION_FLOAT_16
            (2, 3, 9),         // D3D_MIN_PRECISION_FLOAT_2_8
            (4, 2, 4),         // D3D_MIN_PRECISION_SINT_16
            (5, 1, 5),         // D3D_MIN_PRECISION_UINT_16
            (240, 3, 9),       // D3D_MIN_PRECISION_ANY_16
            (241, 3, 9),       // D3D_MIN_PRECISION_ANY_10
        ] {
            let mut part = super::container::signature(std::slice::from_ref(&base), true);
            // Row starts at offset 8: +16 is ComponentType, +28 is MinPrecision.
            part[8 + 16..8 + 20].copy_from_slice(&comp_type.to_le_bytes());
            part[8 + 28..8 + 32].copy_from_slice(&min_prec.to_le_bytes());
            let parsed = super::signature(&part).unwrap();
            assert_eq!(parsed.len(), 1);
            assert_eq!(parsed[0].comp, expected_comp);
            assert!(!parsed[0].half);
        }

        let mut bad = super::container::signature(&[base], true);
        bad[8 + 28..8 + 32].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            super::signature(&bad),
            Err(ShaderError::Unsupported {
                offset: 0,
                what: "an unknown minimum-precision signature element",
            })
        );
    }

    /// A mesh program of `threads` lanes declaring `max` vertices and
    /// primitives; each lane exports its vertex (the position, then
    /// `vertex_rows` four-component rows) and its primitive (`prim_rows`
    /// rows), translated.
    fn mesh(
        threads: u32,
        max: u32,
        vertex_rows: u8,
        prim_rows: u8,
    ) -> Result<HostShader, ShaderError> {
        use crate::native::{MeshIo, Usage, VsOutput};
        let element = |name: &str, row: u8, kind: u8| SigElement {
            name: name.into(),
            index: row.into(),
            kind,
            comp: 9,
            interp: 0,
            row: Some(row.into()),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        };
        let mut vertex = vec![element("SV_Position", 0, 3)];
        vertex.extend((1..=vertex_rows).map(|r| element("TEXCOORD", r, 0)));
        let prims: Vec<SigElement> = (0..prim_rows).map(|r| element("PRIMDATA", r, 0)).collect();
        let mut words: Vec<u32> = vec![
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_02F2, // v_mov_b32 v1, 1.0
            0xF800_00CF, // exp pos0 v0, v0, v0, v1
            0x0100_0000,
        ];
        let mut outputs = Vec::new();
        for n in 0..vertex_rows + prim_rows {
            // exp param<n> v1, v1, v1, v1; the last ones per-primitive.
            words.extend([0xF800_000F | (32 + u32::from(n)) << 4, 0x0101_0101]);
            let primitive = n >= vertex_rows;
            outputs.push(VsOutput {
                usage: Usage::Generic,
                row: if primitive { n - vertex_rows } else { n + 1 },
                param: if primitive { 0x80 | n } else { n },
                fp16: false,
                pack2: false,
            });
        }
        words.extend([
            0xF800_0941, // exp prim v0, off, off, off done
            0x0000_0000,
            0xBF81_0000, // s_endpgm
        ]);
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: true,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs,
                clip_cntl: 0,
                threads,
                max_vertices: max,
                max_primitives: max,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let osg1 = super::container::signature(&vertex, false);
        let psg1 = super::container::signature(&prims, false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1), (FourCC::PSG1, &psg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        emit(&c, &native, &p, &st, &io, &res, Target::ANY)
    }

    /// The vertex and primitive maxima a translated mesh shader's `PSV0`
    /// declares (bytes 12 and 14 of the runtime info, after its size).
    fn psv_maxima(dxil: &[u8]) -> (u16, u16) {
        let c = Container::parse(dxil).unwrap();
        let d = c.part(FourCC::PSV0).unwrap().data;
        let half = |at: usize| u16::from_le_bytes([d[4 + at], d[5 + at]]);
        (half(12), half(14))
    }

    /// What `dxil.dll` says of a container; `None` without it.
    #[cfg(windows)]
    fn validated(dxil: &[u8]) -> Option<Result<(), String>> {
        let path = super::validator::Validator::default_path();
        let mut v = super::validator::Validator::load(path.as_deref()).ok()?;
        Some(v.validate(&mut dxil.to_vec()))
    }

    #[cfg(not(windows))]
    fn validated(_: &[u8]) -> Option<Result<(), String>> {
        None
    }

    /// 64 lanes declaring 256 vertices and primitives, with 28 vertex and 8
    /// primitive output components: 36 KiB at those maxima, past the 32 KiB
    /// D3D12 allows (the validator refused it: "vertex plus primitive output
    /// size is greater than 32768"). The lanes write vertices and primitives
    /// 0-63 alone, so the maxima come down to 64. Outputs that fit keep the
    /// console's maxima.
    #[test]
    fn mesh_maxima_past_the_lanes_come_down_to_fit() {
        for (rows, maxima) in [((6, 2), (64, 64)), ((1, 1), (256, 256))] {
            let s = mesh(64, 256, rows.0, rows.1).unwrap();
            match validated(&s.dxil) {
                Some(r) => r.unwrap_or_else(|e| panic!("{rows:?}: {e}")),
                None => eprintln!("no dxil.dll; validation skipped"),
            }
            assert_eq!(psv_maxima(&s.dxil), maxima, "{rows:?}");
        }
    }

    /// 128 lanes, 256 vertices and primitives, 64 vertex and 8 primitive
    /// output components: past 32 KiB even at the lane count. Still
    /// translated as before, with the maxima down to the lanes.
    #[test]
    fn mesh_outputs_past_32_kib_at_the_lane_count_keep_the_lane_maxima() {
        let s = mesh(128, 256, 15, 2).unwrap();
        assert_eq!(psv_maxima(&s.dxil), (128, 128));
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use crate::native::{ComputeIo, RootParam, StageIo};

    #[test]
    fn heap_texture_atomic_widths() {
        // A synthetic compute shader loads a 2D UAV from a root table,
        // compares or adds at (0, 0), then uses the returned register(s).
        for (opcode, mask, wide) in [
            (16u32, 3, false),
            (16, 15, true),
            (17, 1, false),
            (17, 3, true),
        ] {
            let words = [
                0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
                0xF40C_0200,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[0:1], 0
                0x7E00_0280,
                0x7E02_0280, // v0, v1 = coordinates
                0x7E04_0281,
                0x7E06_0282,
                0x7E08_0283,
                0x7E0A_0284, // v2..v5 = 1..4
                0xF000_2008 | opcode << 18 | mask << 8,
                0x0002_0200,
                // Use the old value in a second atomic at the same location.
                0xF044_0008 | (if wide { 3 } else { 1 }) << 8,
                0x0002_0200,
                0xBF81_0000,
            ];
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 6,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [32, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.code, false, native.vgprs).unwrap();
            let bytes = crate::dxbc::build(&[]);
            let st = structure::structure(&p).unwrap();
            let shader = emit(
                &Container::parse(&bytes).unwrap(),
                &native,
                &p,
                &st,
                &io::raise(&p, &native).unwrap(),
                &resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap(),
                Target::ANY,
            )
            .unwrap();
            let container = Container::parse(&shader.dxil).unwrap();
            let features = u64::from_le_bytes(
                container
                    .part(FourCC::SFI0)
                    .unwrap()
                    .data
                    .try_into()
                    .unwrap(),
            );
            let required = 0x1040_8000; // heap atomics, typed atomics, Int64Ops
            assert_eq!(features & required, if wide { required } else { 0 });
            #[cfg(windows)]
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone()).unwrap();
            }
        }
    }

    /// A compute program over a table at `s4` (user data dwords `s0..s3`
    /// before it): a shader view at table dword 0 (and whatever follows up
    /// to `uav`), an unordered access view at table dword `uav`, neither
    /// with a stride known here.
    fn unknown_stride_program(
        words: &[u32],
        uav: u32,
        compute: crate::native::ComputeIo,
    ) -> super::lower::Lowered {
        let used = |kind, table_offset_dwords| crate::UsedResource {
            kind,
            root: crate::RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![
                used(crate::ResourceType::Srv, 0),
                used(crate::ResourceType::Uav, uav),
            ],
            io: crate::native::StageIo::Cs(compute),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let mut l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            target: super::Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
        .unwrap();
        assert!(l.stride_table);
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil).unwrap();
        }
        l
    }

    /// One thread, user SGPRs `s0..s4`.
    const ONE_THREAD: crate::native::ComputeIo = crate::native::ComputeIo {
        rsrc2: 5 << 1,
        threads: [1, 1, 1],
    };

    #[cfg(windows)]
    fn compute_shader(l: super::lower::Lowered) -> super::HostShader {
        super::HostShader {
            stage: super::HwStage::Cs,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            stride_table: l.stride_table,
            group_waves: l.group_waves,
        }
    }

    /// The user data of `cs` for a dispatch over `views` views: `s0`, the
    /// table at heap slot 0, and the heap guard ([`super::host::guard`]:
    /// the null slot the dispatch keeps after its views), which every heap
    /// index the program computes is held under. The raw views the table
    /// names are among the views, so below the guard as well.
    #[cfg(windows)]
    fn user_data(cs: &super::HostShader, s0: u32, views: usize) -> Vec<u32> {
        let mut data = vec![0; cs.user_data_dwords().max(5) as usize];
        data[0] = s0;
        for u in &cs.user_data {
            if let Some(guard) = super::host::guard(u.root, views, 0) {
                data[u.index as usize] = guard;
            }
        }
        data
    }

    /// The hardware adapter and WARP, those there are.
    #[cfg(windows)]
    fn hosts() -> Vec<(&'static str, super::host::Host)> {
        [("the default adapter", false), ("WARP", true)]
            .into_iter()
            .filter_map(|(name, warp)| Some((name, super::host::Host::new(warp).ok()?)))
            .collect()
    }

    /// What the GPU side makes for a buffer view of `count` elements of
    /// `stride` bytes from element `first` of the buffer in heap slot `of`:
    /// the raw view of those bytes (to go in heap slot `raw_slot`) and the
    /// slot's runtime stride table entry.
    #[cfg(windows)]
    fn raw_view_of(
        of: usize,
        raw_slot: u32,
        (first, count, stride): (u32, u32, u32),
    ) -> (super::host::View<'static>, super::StrideEntry) {
        let window = super::RawWindow::covering(
            u64::from(first * stride),
            u64::from(count * stride),
        )
        .unwrap();
        (
            super::host::View::RawAlias {
                of,
                first: window.first as u32,
                dwords: window.dwords,
            },
            super::StrideEntry {
                stride,
                raw_slot: Some(raw_slot),
                bias: window.bias,
            },
        )
    }

    /// A buffer read without a stride known at translation reads the bytes
    /// the console reads, whatever view the title put in the slot: raw,
    /// structured of any stride, or typed. The shader reads the raw view the
    /// runtime stride table names for the slot (the slot itself for a raw
    /// one), never the slot's own view through a handle of another kind,
    /// which the RTX 5070 answers with other dwords than WARP and a Radeon
    /// (121 for the 107 of the structured view of 12 bytes below).
    ///
    /// Checked for every kind on `host`: a load, a store and atomics.
    #[cfg(windows)]
    fn unknown_stride_views_read_their_bytes(name: &str, host: &super::host::Host) {
        use super::StrideEntry;
        use super::host::{Dispatch, Root, Stride, View};
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT;

        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x7E00_0281, // v_mov_b32 v0, 1
            0x7E02_0288, // v_mov_b32 v1, 8
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E06_028A, // v_mov_b32 v3, 10
            0xE0C8_3000,
            0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
            0xE0C8_3000,
            0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        let cs = compute_shader(unknown_stride_program(words, 8, ONE_THREAD));
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let data: Vec<u32> = (100..164).collect();
        // The view in the slot, its stride, and the elements it has of the
        // buffer (first, count); none: a raw view, read in its own slot.
        let kinds = [
            ("raw", Stride::Raw, 4, None),
            // (The console's descriptor carries a stride for raw views too.)
            ("raw, stride 12", Stride::Raw, 12, None),
            ("structured 12", Stride::Structured(12), 12, Some((0, 21))),
            ("structured 16", Stride::Structured(16), 16, Some((0, 16))),
            ("structured 20", Stride::Structured(20), 20, Some((0, 12))),
            ("typed RGBA32F", Stride::Typed, 16, Some((0, 16))),
            (
                "typed R32_UINT",
                Stride::Format(DXGI_FORMAT_R32_UINT, 4),
                4,
                Some((0, 64)),
            ),
            // Views starting 4 and 12 bytes into their raw views.
            (
                "structured 12 from element 3",
                Stride::Structured(12),
                12,
                Some((3, 10)),
            ),
            (
                "structured 20 from element 3",
                Stride::Structured(20),
                20,
                Some((3, 6)),
            ),
        ];
        for (kind, view, stride, elements) in kinds {
            // Heap: the shader view, the unordered one, their raw views.
            let mut views = vec![
                View::Buffer {
                    data: &data,
                    stride: view,
                },
                View::BufferUav(64, view),
            ];
            let mut table = Vec::new();
            for slot in 0..2 {
                let entry = match elements {
                    Some((first, count)) => {
                        let (raw, entry) = raw_view_of(slot, 2 + slot as u32, (first, count, stride));
                        views.push(raw);
                        entry
                    }
                    None => StrideEntry {
                        stride,
                        ..StrideEntry::default()
                    },
                };
                table.extend(entry.words());
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(user_data(&cs, 0, views.len()))),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: table,
                                },
                            ),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let first = elements.map_or(0, |e| e.0);
            let at = |index: u32, off: u32| (((first + index) * stride + off) / 4) as usize;
            // The load of [2, 4], stored at [1, 8]; 10 added twice at [2, 0].
            let mut want = vec![0; 64];
            want[at(1, 8)] = data[at(2, 4)];
            want[at(2, 0)] = 20;
            assert_eq!(out[0], want, "{kind} on {name}");
        }
    }

    #[test]
    fn structured_buffer_unknown_stride_uses_runtime_stride_table() {
        // Translated and validated on every host; run where there is a GPU.
        let _ = unknown_stride_program(
            &[
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF408_0202,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
                0x7E00_0282, // v_mov_b32 v0, 2
                0x7E02_0284, // v_mov_b32 v1, 4
                0xE030_3000,
                0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
                0xBF81_0000, // s_endpgm
            ],
            8,
            ONE_THREAD,
        );
        #[cfg(windows)]
        for (name, host) in hosts() {
            unknown_stride_views_read_their_bytes(name, &host);
        }
    }

    /// The heap slot of an unknown-stride read loaded from a buffer at run
    /// time, per thread group: each group's lanes read the bytes of the
    /// view in the slot their group chose (a structured view of 12 bytes in
    /// one, of 16 in the other), since the table entry is looked up by the
    /// same computed slot the handle is made from. A lane reading past the
    /// chosen view's elements reads zero, not the buffer's next bytes: the
    /// raw view ends where the view does.
    #[test]
    fn unknown_stride_views_chosen_by_a_loaded_heap_index() {
        let words: &[u32] = &[
            0xBE94_0304, // s_mov_b32 s20, s4
            0xBE95_0380, // s_mov_b32 s21, 0
            0xF408_040A,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
            0x8F07_8205, // s_lshl_b32 s7, s5, 2
            0xF420_0188,
            0x0E00_0000, // s_buffer_load_dword s6, s[16:19], s7
            0x8F06_8506, // s_lshl_b32 s6, s6, 5
            0xF408_020A,
            0x0C00_0000, // s_load_dwordx4 s[8:11], s[20:21], s6
            0xF408_030A,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[20:21], 0x20
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x8F07_8405, // s_lshl_b32 s7, s5, 4
            0x7E02_0207, // v_mov_b32 v1, s7
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        // Four threads (v0: the thread's id), the group's id in s5 after
        // the user SGPRs s0..s4.
        let l = unknown_stride_program(
            words,
            8,
            crate::native::ComputeIo {
                rsrc2: 5 << 1 | 1 << 7,
                threads: [4, 1, 1],
            },
        );
        #[cfg(not(windows))]
        let _ = l;
        #[cfg(windows)]
        {
            use super::StrideEntry;
            use super::host::{Dispatch, Root, Stride, View};
            let cs = compute_shader(l);
            let (a, b): (Vec<u32>, Vec<u32>) = ((100..164).collect(), (200..264).collect());
            // Slot 2: three elements of 12 bytes; slot 3: four of 16 bytes
            // from element 1; their raw views in slots 4 and 5.
            let (raw_a, entry_a) = raw_view_of(2, 4, (0, 3, 12));
            let (raw_b, entry_b) = raw_view_of(3, 5, (1, 4, 16));
            let raw = StrideEntry {
                stride: 4,
                ..StrideEntry::default()
            };
            let table: Vec<u32> = [raw, raw, entry_a, entry_b]
                .into_iter()
                .flat_map(StrideEntry::words)
                .collect();
            // Lane `l` reads [l, 4]: of the first view dwords 1, 4, 7 and
            // (past its three elements) nothing, of the second 5, 9, 13, 17.
            let of_a = [101, 104, 107, 0];
            let of_b = [205, 209, 213, 217];
            for (name, host) in hosts() {
                let (root, pso) = host.compute_pipeline(&cs).unwrap();
                // The slot each of the two groups reads, in the buffer in
                // slot 0.
                for (slots, want) in [([2, 3], [of_a, of_b]), ([3, 2], [of_b, of_a])] {
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![
                                    (0, Root::Constants(user_data(&cs, 0, 6))),
                                    (
                                        1,
                                        Root::View {
                                            srv: true,
                                            data: table.clone(),
                                        },
                                    ),
                                ],
                                views: &[
                                    View::Buffer {
                                        data: &slots,
                                        stride: Stride::Raw,
                                    },
                                    View::RawUav(8),
                                    View::Buffer {
                                        data: &a,
                                        stride: Stride::Structured(12),
                                    },
                                    View::Buffer {
                                        data: &b,
                                        stride: Stride::Structured(16),
                                    },
                                    raw_a,
                                    raw_b,
                                ],
                                samplers: &[],
                                groups: [2, 1, 1],
                            },
                        )
                        .unwrap();
                    assert_eq!(out[0], want.concat(), "slots {slots:?} on {name}");
                }
            }
        }
    }

    /// The descriptor of an unknown-stride read chosen by control flow: the
    /// table entry read is the chosen slot's.
    #[test]
    fn unknown_stride_descriptor_chosen_by_control_flow() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0003, // s_cbranch_scc1 -> 0x18
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xBF82_0002, // s_branch -> 0x20
            0xF408_0202,
            0xFA00_0020, // 0x18: s_load_dwordx4 s[8:11], s[4:5], 0x20
            0xF408_0302,
            0xFA00_0040, // 0x20: s_load_dwordx4 s[12:15], s[4:5], 0x40
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x7E00_0281, // v_mov_b32 v0, 1
            0x7E02_0288, // v_mov_b32 v1, 8
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        let l = unknown_stride_program(words, 16, ONE_THREAD);
        #[cfg(not(windows))]
        let _ = l;
        #[cfg(windows)]
        {
            use super::StrideEntry;
            use super::host::{Dispatch, Root, Stride, View};
            let cs = compute_shader(l);
            let (a, b): (Vec<u32>, Vec<u32>) = ((100..164).collect(), (200..264).collect());
            // Slots 0 and 1: structured views of 12 and 16 bytes, their raw
            // views in slots 3 and 4; slot 2: a raw unordered view.
            let (raw_a, entry_a) = raw_view_of(0, 3, (0, 21, 12));
            let (raw_b, entry_b) = raw_view_of(1, 4, (0, 16, 16));
            let raw = StrideEntry {
                stride: 4,
                ..StrideEntry::default()
            };
            let table: Vec<u32> = [entry_a, entry_b, raw]
                .into_iter()
                .flat_map(StrideEntry::words)
                .collect();
            for (name, host) in hosts() {
                let (root, pso) = host.compute_pipeline(&cs).unwrap();
                // s0: 1 takes the first load, 0 the second.
                for (s0, want) in [(1, 107), (0, 209)] {
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![
                                    (0, Root::Constants(user_data(&cs, s0, 5))),
                                    (
                                        1,
                                        Root::View {
                                            srv: true,
                                            data: table.clone(),
                                        },
                                    ),
                                ],
                                views: &[
                                    View::Buffer {
                                        data: &a,
                                        stride: Stride::Structured(12),
                                    },
                                    View::Buffer {
                                        data: &b,
                                        stride: Stride::Structured(16),
                                    },
                                    View::RawUav(16),
                                    raw_a,
                                    raw_b,
                                ],
                                samplers: &[],
                                groups: [1, 1, 1],
                            },
                        )
                        .unwrap();
                    // Stored at [1, 8] of the raw view: stride 4, dword 3.
                    assert_eq!(out[0][3], want, "s0 = {s0} on {name}");
                }
            }
        }
    }

    #[test]
    fn stride_entries_name_the_raw_view_and_its_bias() {
        use super::{RawWindow, StrideEntry};
        // No raw slot: the slot's own view, whatever the table held before.
        assert_eq!(StrideEntry::default().words(), [0, 0]);
        let own = StrideEntry {
            stride: 4,
            raw_slot: None,
            bias: 0,
        };
        assert_eq!(own.words(), [4, 0]);
        let aliased = StrideEntry {
            stride: 12,
            raw_slot: Some(0),
            bias: 12,
        };
        assert_eq!(aliased.words(), [12, 1 | 12 << 28]);
        // A view on 16 bytes starts its raw view; one 4 to 12 bytes past
        // starts it early, and the raw view covers the bytes before too.
        let window = |first, dwords, bias| {
            Some(RawWindow {
                first,
                dwords,
                bias,
            })
        };
        assert_eq!(RawWindow::covering(32, 24), window(8, 6, 0));
        assert_eq!(RawWindow::covering(36, 24), window(8, 7, 4));
        assert_eq!(RawWindow::covering(44, 22), window(8, 9, 12));
        assert_eq!(RawWindow::covering(0, 0), window(0, 1, 0));
        // Off a dword: no raw view reads it.
        assert_eq!(RawWindow::covering(34, 24), None);
        assert_eq!(RawWindow::covering(16, 4 << 32), None);
    }
}

#[cfg(test)]
mod barycentric_tests {
    use super::container::SigElement;
    use super::*;
    use crate::native::{PsInput, StageIo, Usage};

    /// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, LINEAR_CENTER.
    const PERSP_CENTER: u32 = 1 << 1;
    const LINEAR_CENTER: u32 = 1 << 5;

    /// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
    fn exp_mrt0(r: u32) -> [u32; 2] {
        [0xF800_180F, r * 0x0101_0101]
    }

    /// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
    fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
        3 << 25 | dst << 17 | b << 9 | (256 + a)
    }

    fn element(name: &str, kind: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    /// Translates a pixel program with the input VGPRs `input_addr`
    /// enables and one float4 TEXCOORD0 attribute (attr 0, row 0).
    fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
        pixel_with(words, input_addr, &[])
    }

    /// [`pixel`] with `more` rows in the console's input signature.
    fn pixel_with(
        words: &[u32],
        input_addr: u32,
        more: &[SigElement],
    ) -> Result<HostShader, ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut target = element("SV_Target", 16, 0);
        target.usage = 15;
        let mut rows = vec![element("TEXCOORD", 0, 0)];
        rows.extend_from_slice(more);
        let isg1 = container::signature(&rows, true);
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let st = structure::structure(&p);
        let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
        let io = io::raise(&p, &native)?;
        let st = st?;
        emit(&c, &native, &p, &st, &io, &res, Target::ANY)
    }

    /// The translation's `SFI0` features and `ISG1` rows.
    fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
        let c = Container::parse(&shader.dxil).unwrap();
        let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        (
            u64::from_le_bytes(sfi0),
            signature(c.part(FourCC::ISG1).unwrap().data).unwrap(),
        )
    }

    fn validate(shader: &HostShader) {
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        }
        let _ = shader;
    }

    /// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as
    /// plain floats: they are channels y and z of `SV_Barycentrics`, the
    /// weights of vertices 1 and 2.
    #[test]
    fn barycentrics_read_as_values_are_sv_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!((bary[0].index, bary[0].col, bary[0].cols), (0, 0, 3));
        // Not packed: the validator wants packing location -1.
        assert_eq!(bary[0].row, None);
        // Only y and z (I and J) are read.
        let isg1 = Container::parse(&shader.dxil).unwrap();
        let isg1 = isg1.part(FourCC::ISG1).unwrap().data;
        let rw: Vec<u8> = (0..2).map(|k| isg1[8 + 32 * k + 25]).collect();
        assert!(rw.contains(&0b110), "{rw:?}");
        validate(&shader);
    }

    /// Two kinds of barycentrics read directly: one element per mode.
    #[test]
    fn two_barycentric_modes_get_two_elements() {
        // v0-v1 perspective i, j; v2-v3 linear i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | LINEAR_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        let mut indices: Vec<u32> = inputs
            .iter()
            .filter(|e| e.kind == 28)
            .map(|e| e.index)
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1], "{inputs:?}");
        validate(&shader);
    }

    /// Centre and centroid, both perspective: DXIL takes one perspective
    /// and one linear `SV_Barycentrics`, and the validator refuses a module
    /// with two of one kind, so the program stops with a reason.
    #[test]
    fn two_sample_positions_of_one_kind_are_refused_by_name() {
        const PERSP_CENTROID: u32 = 1 << 2;
        // v0-v1 the centre's i, j; v2-v3 the centroid's.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        match pixel(&words, PERSP_CENTER | PERSP_CENTROID) {
            Err(ShaderError::Unsupported { what, .. }) => {
                assert_eq!(what, "barycentrics of two sample positions");
            }
            Err(e) => panic!("{e}"),
            Ok(_) => panic!("translated"),
        }
    }

    /// A console signature that has an `SV_Barycentrics` row of its own,
    /// under a program that only interpolates and one that reads i and j:
    /// the row is not carried over (lowering declares what is read), and
    /// both modules validate.
    #[test]
    fn a_console_barycentrics_row_is_not_carried_over() {
        let mut row = element("SV_Barycentrics", 28, 1);
        row.cols = 3;
        let [e0, e1] = exp_mrt0(2);
        let interpolates = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel_with(&interpolates, PERSP_CENTER, std::slice::from_ref(&row)).unwrap();
        let (_, inputs) = parts(&shader);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);

        let reads = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel_with(&reads, PERSP_CENTER, &[row]).unwrap();
        let (_, inputs) = parts(&shader);
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!(bary[0].row, None, "{inputs:?}");
        validate(&shader);
    }

    /// I and J only fed to `v_interp_*` stay host interpolation: no
    /// barycentrics are declared.
    #[test]
    fn interpolation_alone_declares_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_eq!(features & 0x2_0000, 0);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);
    }

    /// Both: the attribute is interpolated and J is used as a value.
    #[test]
    fn interpolation_and_a_barycentric_value_together() {
        let [e0, e1] = exp_mrt0(3);
        let words = [
            0xC808_0000,        // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,        // v_interp_p2_f32 v2, v1, attr0.x
            v_add_f32(3, 2, 1), // v3 = attribute + j
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }
    /// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
    fn texcoord_mode(shader: &HostShader) -> u8 {
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let row = [1, 0, 0x44, 0, 3];
        let at = psv.windows(5).position(|w| w == row).unwrap();
        psv[at + 5]
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads
    /// the attribute's vertices: TEXCOORD0 becomes constant, read with
    /// `attributeAtVertex`; i and j are `SV_Barycentrics` values.
    #[test]
    fn computed_barycentrics_read_the_vertices() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// Weights from constants alone: the vertices, no `SV_Barycentrics`.
    #[test]
    fn constant_weights_need_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// The same attribute interpolated with the loaded i and j and with a
    /// computed j: both from its vertices, the first weighed with
    /// `SV_Barycentrics`.
    #[test]
    fn computed_and_loaded_barycentrics_on_one_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
            0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
            0x1006_02F0, // v_mul_f32 v3, 0.5, v1
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// P10 of an attribute the program interpolates is P1 - P0.
    #[test]
    fn p10_of_an_interpolated_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
        // Alone, what the hardware holds is not known.
        let [e0, e1] = exp_mrt0(4);
        let words = [0xC812_0000, e0, e1, S_ENDPGM];
        assert!(pixel(&words, PERSP_CENTER).is_err());
    }

    #[test]
    fn a_first_interpolation_half_can_be_exported_directly() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xc808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            e0,
            e1,          // exp mrt0, v2, v2, v2, v2
            0xbf81_0000, // s_endpgm
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        assert!(inputs.iter().any(|e| e.kind == 28));
        validate(&shader);
    }

    #[test]
    fn a_threaded_pixel_shader_forwards_a_scalar_reload() {
        let [e0, e1] = exp_mrt0(3);
        // A wave-uniform branch skips the spill. Its flag also skips the
        // reload, so the spill supplies the only value the reload can read.
        let words = [
            0x7e06_0500, // v_readfirstlane_b32 s3, v0
            0x7e02_0280, // v_mov_b32 v1, 0
            0xbf06_8003, // s_cmp_eq_u32 s3, 0
            0xbf85_0004, // s_cbranch_scc1 join
            0xbe84_0381, // s_mov_b32 s4, 1
            0xd761_0002,
            0x0001_0004, // v_writelane_b32 v2, s4, 0
            0x7e02_0281, // v_mov_b32 v1, 1
            0x7d84_0280, // join: v_cmp_eq_u32 vcc, 0, v1
            0xbf87_0004, // s_cbranch_vccnz constant
            0xd760_0005,
            0x0001_0102, // v_readlane_b32 s5, v2, 0
            0x7e06_0205, // v_mov_b32 v3, s5
            0xbf82_0001, // s_branch export
            0x7e06_0280, // constant: v_mov_b32 v3, 0
            e0,
            e1,          // export: exp mrt0, v3, v3, v3, v3
            0xbf81_0000, // s_endpgm
        ];
        validate(&pixel(&words, PERSP_CENTER).unwrap());
    }

    #[test]
    fn a_conditional_pixel_export_is_guarded() {
        // Only lanes for which v0 is zero export a colour. This is not the
        // final `done` export, so inactive lanes must keep their old output.
        let words: [u32; 6] = [
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE80_3C6A, // s_and_saveexec_b32 s0, vcc_lo
            0xF800_000F,
            0x0302_0102, // exp mrt0, v2, v1, v2, v3
            0x887E_007E, // s_or_b32 exec_lo, exec_lo, s0
            0xBF81_0000, // s_endpgm
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![],
                input_addr: 1 << 8,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let position = SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 1,
            half: false,
        };
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let isg1 = super::container::signature(&[position], true);
        let osg1 = super::container::signature(&[target], false);
        let bytes =
            crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let emitted = super::emit(&c, &native, &p, &st, &io, &res, super::Target::ANY).unwrap();

        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut emitted.dxil.clone()).unwrap();
        }
        #[cfg(not(windows))]
        drop(emitted);
    }
}
