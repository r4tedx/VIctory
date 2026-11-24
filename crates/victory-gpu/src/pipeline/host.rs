//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, STRIDE_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, PCSTR};

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    depth_bounds: bool,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            depth_bounds: false,
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => {
                    self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned();
                    self.depth_bounds = false;
                }
                21 => {
                    // As before, read this console subobject with the PC layout.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth_bounds = d.DepthBoundsTestEnable.as_bool();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::target_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate
                )
            })
            .collect();
        let depth_bounds = if self.depth_bounds { " depth bounds" } else { "" };
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}{depth_bounds}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    fn depth_with_bounds(&self, host: &crate::host::Host) -> Option<D3D12_DEPTH_STENCIL_DESC1> {
        if !self.depth_bounds {
            return None;
        }
        if !host.depth_bounds_supported {
            crate::host::warn_depth_bounds();
            return None;
        }
        let d = self.depth;
        Some(D3D12_DEPTH_STENCIL_DESC1 {
            DepthEnable: d.DepthEnable,
            DepthWriteMask: d.DepthWriteMask,
            DepthFunc: d.DepthFunc,
            StencilEnable: d.StencilEnable,
            StencilReadMask: d.StencilReadMask,
            StencilWriteMask: d.StencilWriteMask,
            FrontFace: d.FrontFace,
            BackFace: d.BackFace,
            DepthBoundsTestEnable: true.into(),
        })
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// Whether root SRVs are bound as one-view tables of the buffer's host
/// copy (reads past its end return zeros) rather than root descriptors
/// (unbounded: a mesh shader's vertex fetch, which has no record limit,
/// read past the copy into memory the host had freed). Set
/// `VICTORY_ROOT_SRV_DIRECT=1` to go back.
pub(crate) fn bounded_root_srvs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_SRV_DIRECT").is_none())
}

/// Whether root UAVs are bound as one-view tables of the buffer's host
/// copy rather than root descriptors. A root descriptor has no size: a
/// store or load past the copy is a GPU fault or hang, where the console's
/// view has a record count (stores past it are dropped, loads return
/// zeros) and its programs rely on that; their offsets come from title
/// data, which is stale or zero when an earlier pass was dropped. The
/// view covers the range a bind refreshes and copies back to title memory,
/// so no store lands outside what the title gets back. Set
/// `VICTORY_ROOT_UAV_DIRECT=1` to go back; a switch of its own, apart from
/// `VICTORY_ROOT_SRV_DIRECT`, so that a run can tell which of the two
/// changes what it sees.
pub(crate) fn bounded_root_uavs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_UAV_DIRECT").is_none())
}

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    pub(crate) root_uavs: Vec<(u8, u32)>,
    /// Root parameter of the runtime stride table (`t0`, [`STRIDE_SPACE`]).
    pub(crate) stride_table: Option<u32>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    pub(crate) fn has_root_uavs(&self) -> bool {
        self.stages.iter().any(|s| !s.root_uavs.is_empty())
    }

    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "geometry" | "mesh"))
        {
            return refuse(
                "only vertex, geometry, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let geometry = find("geometry");
        if mesh.is_some() && geometry.is_some() {
            return refuse("geometry and mesh stages in one pipeline", shaders);
        }
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages.extend(geometry.map(|gs| (gs, D3D12_SHADER_VISIBILITY_GEOMETRY)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse(
                "the host GPU has no mesh shaders (or no shader model 6.6)",
                shaders,
            );
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        // Ranges the table parameters point at, alive until the root
        // signature is made.
        let mut ranges: Vec<Box<D3D12_DESCRIPTOR_RANGE1>> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        let mut root_views: Vec<((i32, u8), u32)> = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                stride_table: None,
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
                ),
            ] {
                for &slot in slots {
                    if kind == D3D12_ROOT_PARAMETER_TYPE_SRV && bounded_root_srvs() {
                        // A one-view table: the view's size bounds the reads,
                        // which a root descriptor leaves unbounded.
                        ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
                            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                            NumDescriptors: 1,
                            BaseShaderRegister: u32::from(slot),
                            RegisterSpace: ROOT_CBV_SPACE,
                            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                                | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                            OffsetInDescriptorsFromTableStart: 0,
                        }));
                        let range: *const D3D12_DESCRIPTOR_RANGE1 = &**ranges.last()?;
                        out.push((slot, params.len() as u32));
                        params.push(D3D12_ROOT_PARAMETER1 {
                            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                                    NumDescriptorRanges: 1,
                                    pDescriptorRanges: range,
                                },
                            },
                            ShaderVisibility: visibility,
                        });
                        continue;
                    }
                    let param = if kind == D3D12_ROOT_PARAMETER_TYPE_UAV && bounded_root_uavs() {
                        push_root_uav_table(
                            &mut params,
                            &mut ranges,
                            &mut root_views,
                            slot,
                            visibility,
                        )
                    } else {
                        push_root_view(&mut params, &mut root_views, kind, slot, visibility)
                    };
                    if !out.contains(&(slot, param)) {
                        out.push((slot, param));
                    }
                }
            }
            if t.stride_table {
                stage.stride_table = Some(params.len() as u32);
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: STRIDE_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        let (root, root_blob) =
            match root_signature(&host.device, &params, &samplers, compute.is_some()) {
                Ok(root) => root,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host root signature refused");
                    return None;
                }
            };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        let depth_with_bounds = g.depth_with_bounds(host);
        if mesh.is_some() {
            // "triangles": the stream names its topology since it did not
            // load from the library without one (see `MeshStages`); a
            // new name keeps such entries, stored before, from being
            // looked up again.
            let name = super::library::name(
                "mesh triangles",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let ms = MeshStages::new(code(0));
            let made = match depth_with_bounds {
                Some(depth) => stream_pipeline(
                    &host.device, &root, ms, code(1), g,
                    Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth),
                    &name,
                ),
                None => stream_pipeline(
                    &host.device, &root, ms, code(1), g,
                    Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
                    &name,
                ),
            };
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
            });
        }

        let pixel = stages
            .iter()
            .position(|(_, vis)| *vis == D3D12_SHADER_VISIBILITY_PIXEL)
            .map_or_else(Default::default, code);
        let gs = stages
            .iter()
            .position(|(_, vis)| *vis == D3D12_SHADER_VISIBILITY_GEOMETRY)
            .map_or_else(Default::default, code);
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: pixel,
            GS: gs,
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles on
            // the host (see `draw`).
            PrimitiveTopologyType: match g.topology.0 {
                RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        fn dxil(stage: Option<&Arc<Shader>>) -> &[u8] {
            stage
                .and_then(|s| s.translated())
                .map_or(&[], |s| s.host.dxil.as_slice())
        }
        let mut bytecodes = vec![dxil(find("vertex")), dxil(find("pixel"))];
        if geometry.is_some() {
            bytecodes.push(dxil(geometry));
        }
        let name = super::library::name("graphics", &root_blob, &bytecodes, &state);
        let made = if let Some(depth) = depth_with_bounds {
            let vertex = VertexStream {
                vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, desc.VS),
                gs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, desc.GS),
                input: Sub(
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
                    desc.InputLayout,
                ),
                strip_cut: Sub(
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
                    desc.IBStripCutValue,
                ),
                topology: Sub(
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                    desc.PrimitiveTopologyType,
                ),
            };
            stream_pipeline(
                &host.device, &root, vertex, desc.PS, g,
                Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth),
                &name,
            )
        } else {
            super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
            )
        };
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: g.slots(),
            inputs: g.elements(),
        })
    }
}

/// A stream subobject: its type, then its description, pointer-aligned.
#[repr(C, align(8))]
struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

#[repr(C)]
struct VertexStream {
    vs: Sub<D3D12_SHADER_BYTECODE>,
    gs: Sub<D3D12_SHADER_BYTECODE>,
    input: Sub<D3D12_INPUT_LAYOUT_DESC>,
    strip_cut: Sub<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>,
    topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
}

/// The mesh stage of a stream, and its topology.
///
/// The stream names a triangle topology, which D3D12 does not need for a
/// mesh shader: without it the pipeline library refuses to load a stored
/// pipeline with the very stream it was made from (`E_INVALIDARG`; debug
/// layer message 972, "does not match the one used to create the PSO"), so
/// every mesh pipeline was compiled again, and its store then refused (970,
/// "already exists"). The translation makes triangle mesh shaders only
/// (`mesh_prologue` refuses others), and the debug layer wants the two to
/// agree (1323).
#[repr(C)]
struct MeshStages {
    ms: Sub<D3D12_SHADER_BYTECODE>,
    topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
}

impl MeshStages {
    fn new(ms: D3D12_SHADER_BYTECODE) -> Self {
        MeshStages {
            ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
            topology: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ),
        }
    }
}

/// A mesh pipeline from the title's graphics state without depth bounds:
/// the pipeline-state stream form (no input layout).
#[cfg(test)]
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    stream_pipeline(
        device,
        root,
        MeshStages::new(ms),
        ps,
        g,
        Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        name,
    )
}

/// The stream form used by mesh pipelines (stages [`MeshStages`]) and
/// vertex pipelines with depth bounds ([`VertexStream`]).
fn stream_pipeline<S, D>(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    stages: S,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    depth: Sub<D>,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    #[repr(C)]
    struct Stream<S, D> {
        root: Sub<*mut std::ffi::c_void>,
        stages: S,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
    }
    let mut stream = Stream {
        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
        stages,
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend),
        mask: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            g.sample_mask,
        ),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth,
        targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
        dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of_val(&stream),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

/// Adds or reuses a root view parameter for `(kind, slot)`, widening its
/// visibility to `ALL` when shared across multiple shader stages so shared
/// root descriptors do not consume extra root signature dwords.
fn push_root_view(
    params: &mut Vec<D3D12_ROOT_PARAMETER1>,
    seen: &mut Vec<((i32, u8), u32)>,
    kind: D3D12_ROOT_PARAMETER_TYPE,
    slot: u8,
    visibility: D3D12_SHADER_VISIBILITY,
) -> u32 {
    if let Some(existing) = shared_root_view(params, seen, kind, slot, visibility) {
        return existing;
    }
    let index = params.len() as u32;
    seen.push(((kind.0, slot), index));
    params.push(D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: u32::from(slot),
                RegisterSpace: ROOT_CBV_SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: visibility,
    });
    index
}

/// The parameter an earlier stage added for `(kind, slot)`, its visibility
/// widened to `ALL` if this stage's differs.
fn shared_root_view(
    params: &mut [D3D12_ROOT_PARAMETER1],
    seen: &[((i32, u8), u32)],
    kind: D3D12_ROOT_PARAMETER_TYPE,
    slot: u8,
    visibility: D3D12_SHADER_VISIBILITY,
) -> Option<u32> {
    let &(_, existing) = seen.iter().find(|&&(k, _)| k == (kind.0, slot))?;
    if params[existing as usize].ShaderVisibility != visibility {
        params[existing as usize].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    Some(existing)
}

/// Adds or reuses the parameter of root UAV `slot` as a one-view table
/// (see [`bounded_root_uavs`]): one UAV range at the register and space
/// the root descriptor had, so the shader is the same either way. Shared
/// between stages as [`push_root_view`] shares a descriptor (the title
/// sets one address for the slot, so one view serves every stage); a
/// table costs one root signature dword where a descriptor costs two.
/// `ranges` keeps the range alive until the root signature is made.
fn push_root_uav_table(
    params: &mut Vec<D3D12_ROOT_PARAMETER1>,
    ranges: &mut Vec<Box<D3D12_DESCRIPTOR_RANGE1>>,
    seen: &mut Vec<((i32, u8), u32)>,
    slot: u8,
    visibility: D3D12_SHADER_VISIBILITY,
) -> u32 {
    let kind = D3D12_ROOT_PARAMETER_TYPE_UAV;
    if let Some(existing) = shared_root_view(params, seen, kind, slot, visibility) {
        return existing;
    }
    let range = Box::new(D3D12_DESCRIPTOR_RANGE1 {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 1,
        BaseShaderRegister: u32::from(slot),
        RegisterSpace: ROOT_CBV_SPACE,
        Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
            | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
        OffsetInDescriptorsFromTableStart: 0,
    });
    let index = params.len() as u32;
    seen.push(((kind.0, slot), index));
    params.push(D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                NumDescriptorRanges: 1,
                pDescriptorRanges: &raw const *range,
            },
        },
        ShaderVisibility: visibility,
    });
    ranges.push(range);
    index
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    use super::*;

    /// A triangle mesh shader of our own and a pixel shader for it.
    const MS: &str = "struct V { float4 p : SV_Position; };
        [outputtopology(\"triangle\")] [numthreads(1, 1, 1)]
        void main(out vertices V v[3], out indices uint3 t[1]) {
            SetMeshOutputCounts(3, 1);
            v[0].p = float4(0, 0, 0, 1);
            v[1].p = float4(1, 0, 0, 1);
            v[2].p = float4(0, 1, 0, 1);
            t[0] = uint3(0, 1, 2);
        }";
    const PS: &str = "float4 main() : SV_Target { return 1; }";

    /// The PC `dxc.exe` (`VICTORY_PC_DXC`, else the newest Windows SDK's).
    fn dxc() -> Option<PathBuf> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?
                .with_file_name("dxc.exe"),
        };
        dxc.exists().then_some(dxc)
    }

    fn compile(dxc: &Path, dir: &Path, src: &str, profile: &str) -> Vec<u8> {
        let (hlsl, out) = (
            dir.join(format!("{profile}.hlsl")),
            dir.join(format!("{profile}.dxil")),
        );
        std::fs::write(&hlsl, src).unwrap();
        let status = std::process::Command::new(dxc)
            .args(["-nologo", "-T", profile, "-E", "main", "-Fo"])
            .arg(&out)
            .arg(&hlsl)
            .status()
            .unwrap();
        assert!(status.success(), "dxc failed for {profile}");
        std::fs::read(out).unwrap()
    }

    fn code(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        }
    }

    /// A mesh pipeline made once is loaded from the library the next time,
    /// not compiled again (a stream without a topology never loads).
    #[test]
    fn a_mesh_pipeline_loads_from_the_library() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let name = super::super::library::name("mesh test", &root_blob, &[&ms, &ps], &g.describe());
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, &name).expect("made");
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, &name).expect("made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the mesh pipeline was compiled again instead of loaded from the library"
        );
    }
}

#[cfg(test)]
mod root_view_tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter, IDXGIFactory4,
    };

    #[test]
    fn root_views_deduplicate_across_stages_and_create_on_warp() {
        let mut params = Vec::new();
        let mut seen = Vec::new();

        // Vertex stage declares CBV 0, SRV 1, UAV 2.
        let v_cbv = push_root_view(
            &mut params,
            &mut seen,
            D3D12_ROOT_PARAMETER_TYPE_CBV,
            0,
            D3D12_SHADER_VISIBILITY_VERTEX,
        );
        let v_srv = push_root_view(
            &mut params,
            &mut seen,
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            1,
            D3D12_SHADER_VISIBILITY_VERTEX,
        );
        let v_uav = push_root_view(
            &mut params,
            &mut seen,
            D3D12_ROOT_PARAMETER_TYPE_UAV,
            2,
            D3D12_SHADER_VISIBILITY_VERTEX,
        );
        assert_eq!((v_cbv, v_srv, v_uav), (0, 1, 2));
        assert_eq!(params[2].ShaderVisibility, D3D12_SHADER_VISIBILITY_VERTEX);

        // Pixel stage shares CBV 0 and UAV 2, declares SRV 2 (same slot,
        // different kind -> separate parameter), and pixel-only UAV 5.
        let p_cbv = push_root_view(
            &mut params,
            &mut seen,
            D3D12_ROOT_PARAMETER_TYPE_CBV,
            0,
            D3D12_SHADER_VISIBILITY_PIXEL,
        );
        let p_uav_shared = push_root_view(
            &mut params,
            &mut seen,
            D3D12_ROOT_PARAMETER_TYPE_UAV,
            2,
            D3D12_SHADER_VISIBILITY_PIXEL,
        );
        let p_srv_same_slot = push_root_view(
            &mut params,
            &mut seen,
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            2,
            D3D12_SHADER_VISIBILITY_PIXEL,
        );
        let p_uav_only = push_root_view(
            &mut params,
            &mut seen,
            D3D12_ROOT_PARAMETER_TYPE_UAV,
            5,
            D3D12_SHADER_VISIBILITY_PIXEL,
        );

        assert_eq!(p_cbv, v_cbv);
        assert_eq!(p_uav_shared, v_uav);
        assert_eq!(p_srv_same_slot, 3);
        assert_eq!(p_uav_only, 4);
        assert_eq!(params.len(), 5);
        assert_eq!(params[0].ShaderVisibility, D3D12_SHADER_VISIBILITY_ALL);
        assert_eq!(params[1].ShaderVisibility, D3D12_SHADER_VISIBILITY_VERTEX);
        assert_eq!(params[2].ShaderVisibility, D3D12_SHADER_VISIBILITY_ALL);
        assert_eq!(params[3].ShaderVisibility, D3D12_SHADER_VISIBILITY_PIXEL);
        assert_eq!(params[4].ShaderVisibility, D3D12_SHADER_VISIBILITY_PIXEL);

        // Create the root signature on WARP (and on host() when available).
        // SAFETY: plain DXGI/D3D12 device creation on the WARP software adapter.
        let warp_device = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>())
                .and_then(|a| {
                    let mut dev: Option<ID3D12Device> = None;
                    D3D12CreateDevice(&a, D3D_FEATURE_LEVEL_12_0, &mut dev)?;
                    dev.ok_or_else(windows::core::Error::empty)
                })
        };
        if let Ok(dev) = warp_device {
            root_signature(&dev, &params, &[], false)
                .expect("WARP CreateRootSignature with deduplicated CBV/SRV/UAV root views");
        }
        if let Some(h) = host() {
            root_signature(&h.device, &params, &[], false)
                .expect("host CreateRootSignature with deduplicated CBV/SRV/UAV root views");
        }
    }

    /// A root UAV as a one-view table is shared between the stages that
    /// name it, as a root UAV descriptor is, and such a root signature is
    /// made on WARP and on the host.
    #[test]
    fn root_uav_tables_deduplicate_across_stages_and_create_on_warp() {
        let mut params = Vec::new();
        let mut ranges = Vec::new();
        let mut seen = Vec::new();
        let (vertex, pixel) = (
            D3D12_SHADER_VISIBILITY_VERTEX,
            D3D12_SHADER_VISIBILITY_PIXEL,
        );
        let uav = D3D12_ROOT_PARAMETER_TYPE_UAV;

        // Vertex stage: CBV 0 (a descriptor), UAVs 2 and 3 (tables).
        let cbv = push_root_view(&mut params, &mut seen, D3D12_ROOT_PARAMETER_TYPE_CBV, 0, vertex);
        let v_uav = push_root_uav_table(&mut params, &mut ranges, &mut seen, 2, vertex);
        let v_only = push_root_uav_table(&mut params, &mut ranges, &mut seen, 3, vertex);
        // Pixel stage: shares UAV 2, has UAV 5 of its own.
        let p_uav = push_root_uav_table(&mut params, &mut ranges, &mut seen, 2, pixel);
        let p_only = push_root_uav_table(&mut params, &mut ranges, &mut seen, 5, pixel);
        // A stage naming a slot twice gets the one parameter.
        let again = push_root_uav_table(&mut params, &mut ranges, &mut seen, 5, pixel);

        assert_eq!((cbv, v_uav, v_only, p_uav, p_only, again), (0, 1, 2, 1, 3, 3));
        assert_eq!(params.len(), 4);
        assert_eq!(ranges.len(), 3);
        assert_eq!(params[1].ShaderVisibility, D3D12_SHADER_VISIBILITY_ALL);
        assert_eq!(params[2].ShaderVisibility, vertex);
        assert_eq!(params[3].ShaderVisibility, pixel);
        assert_eq!(seen[1], ((uav.0, 2), 1));
        for (param, slot) in [(1usize, 2u32), (2, 3), (3, 5)] {
            let p = &params[param];
            assert_eq!(p.ParameterType, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE);
            // SAFETY: a table parameter, its range kept alive in `ranges`.
            let range = unsafe {
                assert_eq!(p.Anonymous.DescriptorTable.NumDescriptorRanges, 1);
                &*p.Anonymous.DescriptorTable.pDescriptorRanges
            };
            assert_eq!(range.RangeType, D3D12_DESCRIPTOR_RANGE_TYPE_UAV);
            assert_eq!(range.NumDescriptors, 1);
            assert_eq!(range.BaseShaderRegister, slot);
            assert_eq!(range.RegisterSpace, ROOT_CBV_SPACE);
        }

        // SAFETY: plain DXGI/D3D12 device creation on the WARP software adapter.
        let warp_device = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>())
                .and_then(|a| {
                    let mut dev: Option<ID3D12Device> = None;
                    D3D12CreateDevice(&a, D3D_FEATURE_LEVEL_12_0, &mut dev)?;
                    dev.ok_or_else(windows::core::Error::empty)
                })
        };
        if let Ok(dev) = warp_device {
            root_signature(&dev, &params, &[], false)
                .expect("WARP CreateRootSignature with root UAV tables");
        }
        if let Some(h) = host() {
            root_signature(&h.device, &params, &[], false)
                .expect("host CreateRootSignature with root UAV tables");
        }
    }
}
