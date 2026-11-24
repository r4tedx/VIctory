//! Indirect rectangle-list draws from end to end: a title pipeline made from
//! a stream, hand-made vertices and draw records in memory registered as
//! title buffers, `State::execute_indirect` (the clamp, the expansion of
//! [`crate::rectangles`], the draw on the rewritten arguments), and the
//! render target read back.

use super::*;
use crate::draw::{State, VertexView};
use crate::host::{Host, HostList};
use crate::object::{S_OK, slot};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, PCSTR, s};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

/// The render target's width and height in pixels.
const SIZE: usize = 8;

/// The target's pixels as read back: rows of R8G8B8A8.
type Image = [[[u8; 4]; SIZE]; SIZE];

const CLEAR: [u8; 4] = [0, 0, 0, 255];
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const YELLOW: [u8; 4] = [255, 255, 0, 255];

/// The pixels (`x0..x1`, `y0..y1`) of the three rectangles and their colours.
const A: ([usize; 4], [u8; 4]) = ([1, 1, 5, 4], RED);
const B: ([usize; 4], [u8; 4]) = ([5, 4, 8, 7], GREEN);
const C: ([usize; 4], [u8; 4]) = ([0, 5, 2, 8], BLUE);

/// One vertex as the pipeline's input layout reads it: a position in clip
/// space, then a colour (12 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    colour: [u8; 4],
}

/// A vertex at a point given in pixels (the target is 8 by 8, y down).
fn vertex(x: f32, y: f32, colour: [u8; 4]) -> Vertex {
    let half = SIZE as f32 / 2.0;
    Vertex {
        position: [x / half - 1.0, 1.0 - y / half],
        colour,
    }
}

/// The three vertices a title gives for a rectangle: top left, top right,
/// bottom right. The fourth corner (bottom left) is `v0 + v2 - v1`. Clockwise,
/// since the pipeline culls back faces: a corner taken in another order
/// would cull half of the rectangle.
fn rectangle(([x0, y0, x1, y1], colour): ([usize; 4], [u8; 4])) -> [Vertex; 3] {
    let (x0, y0, x1, y1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
    [
        vertex(x0, y0, colour),
        vertex(x1, y0, colour),
        vertex(x1, y1, colour),
    ]
}

fn cleared() -> Image {
    [[CLEAR; SIZE]; SIZE]
}

/// Colours a rectangle's pixels in the expected image.
fn fill(image: &mut Image, ([x0, y0, x1, y1], colour): ([usize; 4], [u8; 4])) {
    for row in &mut image[y0..y1] {
        row[x0..x1].fill(colour);
    }
}

/// Colours the pixels of the plain triangle (corners at pixels (0, 0),
/// (4.5, 0) and (0, 4.5)): those whose centre has x + y below 4.5. No edge
/// passes through a pixel centre, so the coverage is exact; drawn as a
/// rectangle instead, it would also cover pixels up to (3, 3).
fn fill_triangle(image: &mut Image) {
    for (y, row) in image.iter_mut().enumerate() {
        for (x, pixel) in row.iter_mut().enumerate() {
            if x + y <= 3 {
                *pixel = YELLOW;
            }
        }
    }
}

/// An image as text, one letter per pixel, for a failed comparison.
fn picture(image: &Image) -> String {
    let letter = |pixel: &[u8; 4]| match *pixel {
        CLEAR => '.',
        RED => 'A',
        GREEN => 'B',
        BLUE => 'C',
        YELLOW => 'T',
        _ => '?',
    };
    image
        .iter()
        .map(|row| row.iter().map(letter).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Fails with both images and the values of the pixels that differ.
fn assert_image(case: &str, got: &Image, want: &Image) {
    if got == want {
        eprintln!("{case}: as expected\n{}", picture(got));
        return;
    }
    let mut wrong = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            if got[y][x] != want[y][x] {
                wrong.push(format!(
                    "({x}, {y}): read {:?}, expected {:?}",
                    got[y][x], want[y][x]
                ));
            }
        }
    }
    panic!(
        "{case}: read back\n{}\nexpected\n{}\n{}",
        picture(got),
        picture(want),
        wrong.join("\n")
    );
}

/// Write-watched memory registered as a title buffer, as the draw tests'
/// `Allocation::watched` with `crate::shadow::register` is, so that
/// `crate::shadow::find` gives it a host copy. Unregistered and freed when
/// dropped, also when an assertion fails.
struct TitleMemory {
    data: *mut u8,
    len: usize,
}

impl TitleMemory {
    fn new(len: usize) -> Self {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: a fresh write-watched range of our own, freed in Drop.
        let data = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!data.is_null());
        crate::shadow::register(data as usize, data as u64, len as u64);
        Self { data, len }
    }

    /// Writes `values` at `offset`; their title address.
    fn put<T: Copy>(&self, offset: usize, values: &[T]) -> u64 {
        assert!(offset + size_of_val(values) <= self.len);
        // SAFETY: inside the committed range (checked above).
        unsafe {
            let to = self.data.add(offset);
            std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), to, size_of_val(values));
            to as u64
        }
    }
}

impl Drop for TitleMemory {
    fn drop(&mut self) {
        crate::shadow::unregister(self.data as usize, self.data as u64);
        // SAFETY: `self.data` came from `VirtualAlloc`.
        unsafe { VirtualFree(self.data.cast(), 0, MEM_RELEASE) };
    }
}

/// A shader of the test as host code (FXC), standing in for a translated
/// title shader with no bindings.
fn host_shader(stage: &'static str, entry: PCSTR, target: PCSTR) -> Arc<Shader> {
    const SOURCE: &str = r#"
        struct Corner { float4 position : SV_Position; float4 colour : COLOR; };
        Corner vs(float2 position : POSITION, float4 colour : COLOR) {
            Corner corner;
            corner.position = float4(position, 0.5, 1);
            corner.colour = colour;
            return corner;
        }
        float4 ps(Corner corner) : SV_Target { return corner.colour; }
    "#;
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; `code` is the
    // output pointer.
    unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            PCSTR::null(),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            None,
        )
        .unwrap();
    }
    let code = code.unwrap();
    // SAFETY: the blob owns these bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    let hw_stage = match stage {
        "vertex" => victory_shader::HwStage::Vs,
        _ => victory_shader::HwStage::Ps,
    };
    let shader = shader(stage, bytes);
    shader.translated.get_or_init(|| {
        Some(Translated {
            host: victory_shader::emit::HostShader {
                stage: hw_stage,
                dxil: bytes.to_vec(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                stride_table: false,
                group_waves: 1,
            },
            used: Vec::new(),
        })
    });
    shader
}

/// A title pipeline from a stream, as a title makes one for rectangles: the
/// two shaders, no depth test, a position and a colour per vertex from
/// vertex buffer 0, the console's rectangle topology type and one
/// R8G8B8A8_UNORM target. Culling stays at the default (back faces).
fn rectangle_pipeline() -> This {
    #[repr(C)]
    struct Depth(u32, D3D12_DEPTH_STENCIL_DESC);
    #[repr(C)]
    struct Targets(u32, D3D12_RT_FORMAT_ARRAY);
    #[repr(C)]
    struct Stream {
        vs_kind: u64,
        vs: Bytecode,
        ps_kind: u64,
        ps: Bytecode,
        depth: Depth,
        input_kind: u64,
        input: D3D12_INPUT_LAYOUT_DESC,
        topology: [u32; 2],
        targets: Targets,
    }
    let vs = host_shader("vertex", s!("vs"), s!("vs_5_1"));
    let ps = host_shader("pixel", s!("ps"), s!("ps_5_1"));
    let element = |name, format, offset| D3D12_INPUT_ELEMENT_DESC {
        SemanticName: name,
        Format: format,
        AlignedByteOffset: offset,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        ..Default::default()
    };
    let elements = [
        element(s!("POSITION"), DXGI_FORMAT_R32G32_FLOAT, 0),
        element(s!("COLOR"), DXGI_FORMAT_R8G8B8A8_UNORM, 8),
    ];
    let keep = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let mut formats = D3D12_RT_FORMAT_ARRAY {
        NumRenderTargets: 1,
        ..Default::default()
    };
    formats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    // The console's rectangle topology type (`RECTANGLES` in `host.rs`).
    const RECTANGLE_TYPE: u32 = 19;
    let stream = Stream {
        vs_kind: 1,
        vs: Bytecode {
            code: vs.container.as_ptr(),
            size: vs.container.len(),
        },
        ps_kind: 2,
        ps: Bytecode {
            code: ps.container.as_ptr(),
            size: ps.container.len(),
        },
        depth: Depth(
            11,
            D3D12_DEPTH_STENCIL_DESC {
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
        ),
        input_kind: 12,
        input: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: elements.as_ptr(),
            NumElements: elements.len() as u32,
        },
        topology: [14, RECTANGLE_TYPE],
        targets: Targets(15, formats),
    };
    let desc = StreamDesc {
        size: size_of_val(&stream),
        stream: (&raw const stream).cast(),
    };
    let null = std::ptr::null();
    let mut pso = null;
    assert_eq!(
        create_stream(null, &desc, &PIPELINE_IIDS[0], &mut pso),
        S_OK
    );
    let made = super::test_host_pipeline(pso).expect("host refused the rectangle pipeline");
    // What the expansion goes by: both inputs per vertex, from slot 0.
    assert_eq!((made.slots, made.inputs.len()), (1, 2));
    pso
}

/// A host list with a draw state of its own, the target its submissions
/// draw to and the buffer the target is read back through.
struct Scene<'a> {
    host: &'a Host,
    list: HostList,
    state: State,
    pso: This,
    target: ID3D12Resource,
    readback: ID3D12Resource,
    /// The title's vertex buffer view for slot 0.
    vertices: VertexView,
    /// The list ran before, so it is reset first.
    ran: bool,
}

/// Bytes per row of the read-back copy (D3D12's row alignment).
const ROW_PITCH: usize = 256;

impl<'a> Scene<'a> {
    fn new(host: &'a Host, pso: This, vertices: VertexView) -> Self {
        let samples = DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        };
        let target_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: SIZE as u64,
            Height: SIZE as u32,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: samples,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let buffer_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: (ROW_PITCH * SIZE) as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: samples,
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let resource = |desc: &D3D12_RESOURCE_DESC, kind, state| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: valid resource descriptions and output pointer.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        desc,
                        state,
                        None,
                        &mut made,
                    )
                    .unwrap();
            }
            made.unwrap()
        };
        Scene {
            host,
            list: host.new_list().expect("host command list"),
            state: State::default(),
            pso,
            target: resource(
                &target_desc,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ),
            readback: resource(
                &buffer_desc,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ),
            vertices,
            ran: false,
        }
    }

    /// One submission of the list: the target cleared, what `draws` records
    /// with the title's state (the pipeline and vertex buffer set), and the
    /// target copied out. Returns the pixels.
    fn render(&mut self, draws: impl FnOnce(&HostList, &mut State)) -> Image {
        if self.ran {
            self.list.reset();
        }
        self.ran = true;
        self.state.reset(self.pso);
        self.state.set_vertex_buffers(0, &[self.vertices]);
        let l = &self.list.list;
        let clear = CLEAR.map(|c| f32::from(c) / 255.0);
        // A view is valid until it is recorded, so one per submission.
        let rtv = self.host.rtv(&self.target, None);
        // SAFETY: an open list; the target is in RENDER_TARGET.
        unsafe {
            l.OMSetRenderTargets(1, Some(&rtv), false, None);
            l.ClearRenderTargetView(rtv, &clear, None);
        }
        crate::draw::viewports(
            &self.list,
            &[D3D12_VIEWPORT {
                Width: SIZE as f32,
                Height: SIZE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }],
        );
        crate::draw::scissors(
            &self.list,
            &[RECT {
                right: SIZE as i32,
                bottom: SIZE as i32,
                ..Default::default()
            }],
        );
        draws(&self.list, &mut self.state);

        let mut from = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            ..Default::default()
        };
        let mut to = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: SIZE as u32,
                        Height: SIZE as u32,
                        Depth: 1,
                        RowPitch: ROW_PITCH as u32,
                    },
                },
            },
            ..Default::default()
        };
        let moved = |before, after| crate::host::transition(&self.target, before, after);
        // SAFETY: the locations borrow live resources, the copy covers the
        // whole target, and the target ends in RENDER_TARGET again.
        unsafe {
            from.pResource = std::mem::transmute_copy(&self.target);
            to.pResource = std::mem::transmute_copy(&self.readback);
            l.ResourceBarrier(&[moved(
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            l.ResourceBarrier(&[moved(
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
        }
        self.list.close();
        self.host.execute(&[&self.list]);

        let mut image = cleared();
        let mut mapped = std::ptr::null_mut();
        let all = D3D12_RANGE {
            Begin: 0,
            End: ROW_PITCH * SIZE,
        };
        // SAFETY: the GPU copy finished; each row holds `SIZE` pixels of four
        // bytes at its pitch.
        unsafe {
            self.readback.Map(0, Some(&all), Some(&mut mapped)).unwrap();
            for (y, row) in image.iter_mut().enumerate() {
                *row = mapped
                    .cast::<u8>()
                    .add(y * ROW_PITCH)
                    .cast::<[[u8; 4]; SIZE]>()
                    .read_unaligned();
            }
            self.readback.Unmap(0, Some(&D3D12_RANGE::default()));
        }
        image
    }
}

/// The console's rectangle list and the triangle list, as a title sets
/// them on its command list (`RECTANGLES` in `draw.rs`).
const RECTANGLE_LIST: u32 = 17;
const TRIANGLE_LIST: u32 = 4;

/// ExecuteIndirect's kind for a non-indexed draw, and its record size.
const DRAW: u32 = 0;
const RECORD: u32 = 16;

/// Draws the cases and checks each one's pixels. With `shared`, the
/// vertices, the draw records and the record count lie in one title buffer
/// (one host copy that the passes write and the draws read, as in a title's
/// big heap); without, in three. `check` is called after each submission
/// with the case's name.
fn draw_cases(host: &Host, pso: This, shared: bool, check: &dyn Fn(&str)) {
    const LEN: usize = 64 << 10;
    let first = TitleMemory::new(LEN);
    let others = (!shared).then(|| (TitleMemory::new(LEN), TitleMemory::new(LEN)));
    let (args_memory, count_memory) = match &others {
        Some((args, count)) => (args, count),
        None => (&first, &first),
    };

    // Vertices 0..3 are rectangle A; 3 is a stray vertex no draw starts at,
    // so that B and C start at 4, not a multiple of three; 4..7 B; 7..10 C;
    // 10..13 the plain triangle.
    let stray = vertex(8.0, 0.0, [255, 255, 255, 255]);
    let triangle = [
        vertex(0.0, 0.0, YELLOW),
        vertex(4.5, 0.0, YELLOW),
        vertex(0.0, 4.5, YELLOW),
    ];
    let mut vertices = Vec::new();
    vertices.extend(rectangle(A));
    vertices.push(stray);
    vertices.extend(rectangle(B));
    vertices.extend(rectangle(C));
    vertices.extend(triangle);
    let view = VertexView {
        address: first.put(0x100, &vertices),
        size: size_of_val(&vertices[..]) as u32,
        stride: size_of::<Vertex>() as u32,
    };
    // Two records of (vertex count, instance count, start vertex, start
    // instance): A, then B and C as two rectangles of one record.
    let records: [[u32; 4]; 2] = [[3, 1, 0, 0], [6, 1, 4, 0]];
    let args = args_memory.put(0x400, &records);
    let second = args + u64::from(RECORD);
    // The title's count: one record.
    let count = count_memory.put(0x800, &[1u32]);

    let where_ = if shared {
        "one title buffer"
    } else {
        "three title buffers"
    };
    let mut scene = Scene::new(host, pso, view);
    let mut case = |name: &str, want: &Image, draws: &dyn Fn(&HostList, &mut State)| {
        let name = format!("{name} ({where_})");
        let got = scene.render(draws);
        check(&name);
        assert_image(&name, &got, want);
    };
    let indirect = |list: &HostList, state: &mut State, max, at, count| {
        state.set_topology(list, RECTANGLE_LIST);
        assert!(
            state.execute_indirect(list, (DRAW, RECORD, max), at, count),
            "the indirect rectangle draw did not run on the host"
        );
    };

    // (a) One record: A alone, both of its halves.
    let mut want = cleared();
    fill(&mut want, A);
    case("one record", &want, &|list, state| {
        indirect(list, state, 1, args, None);
    });

    // (b) Two records in one execution: A, then B and C from vertex 4.
    let mut want = cleared();
    fill(&mut want, A);
    fill(&mut want, B);
    fill(&mut want, C);
    case("two records", &want, &|list, state| {
        indirect(list, state, 2, args, None);
    });

    // (c) Two records, of which the title's count buffer lets one run.
    let mut want = cleared();
    fill(&mut want, A);
    case("two records, count of one", &want, &|list, state| {
        indirect(list, state, 2, args, Some(count));
    });

    // (d) In one list: rectangles, then a plain triangle from the same
    // vertex buffer (read again after the passes had it as written), then
    // rectangles again (the list's scratch reused, the buffer written
    // again after a draw read it).
    let mut want = cleared();
    fill(&mut want, A);
    fill_triangle(&mut want);
    fill(&mut want, B);
    fill(&mut want, C);
    case(
        "rectangles, a triangle, rectangles",
        &want,
        &|list, state| {
            indirect(list, state, 1, args, None);
            state.set_topology(list, TRIANGLE_LIST);
            state.draw(list, 3, 1, 10, 0);
            indirect(list, state, 1, second, None);
        },
    );

    // (e) A plain draw first, so the list starts with the vertex buffer in
    // its read state, then rectangles with the count buffer over it.
    let mut want = cleared();
    fill_triangle(&mut want);
    fill(&mut want, A);
    case("a triangle, rectangles", &want, &|list, state| {
        state.set_topology(list, TRIANGLE_LIST);
        state.draw(list, 3, 1, 10, 0);
        indirect(list, state, 2, args, Some(count));
    });
}

/// Set in the child process that runs the test under the debug layer.
const CHILD: &str = "VICTORY_TEST_RECT_INDIRECT_CHILD";

/// The debug layer's errors since the last call (warnings are printed).
fn layer_errors(messages: &ID3D12InfoQueue) -> Vec<String> {
    let mut errors = Vec::new();
    // SAFETY: each message is read into a buffer of the size the queue
    // reports, then the queue is emptied.
    unsafe {
        for i in 0..messages.GetNumStoredMessages() {
            let mut size = 0;
            if messages.GetMessage(i, None, &mut size).is_err() || size == 0 {
                continue;
            }
            let mut bytes = vec![0u64; size.div_ceil(8)];
            let m = bytes.as_mut_ptr().cast::<D3D12_MESSAGE>();
            if messages.GetMessage(i, Some(m), &mut size).is_err() {
                continue;
            }
            let text = std::slice::from_raw_parts((*m).pDescription, (*m).DescriptionByteLength);
            let text = format!("id {}: {}", (*m).ID.0, String::from_utf8_lossy(text));
            if (*m).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                errors.push(text);
            } else if (*m).Severity.0 <= D3D12_MESSAGE_SEVERITY_WARNING.0 {
                eprintln!("debug layer warning: {text}");
            }
        }
        messages.ClearStoredMessages();
    }
    errors
}

/// Rectangles drawn through `State::execute_indirect` reach the target:
/// one record, two records (the second from a start vertex that is not a
/// multiple of three, holding two rectangles), two records cut to one by a
/// count buffer, and rectangle draws before and after a plain draw from the
/// same vertex buffer in one list. Each case is checked pixel by pixel:
/// inside a rectangle its colour, also in the half the fourth corner adds,
/// outside the clear colour.
///
/// The pixels cannot show the states the list's buffer tracker gives the
/// title's copies (drivers do not act on them), so where the D3D12 debug
/// layer is installed the test runs once more in a child process with the
/// layer on, which checks every barrier's state before against what the
/// list did, and fails on any error it reports.
#[test]
fn indirect_rectangles_reach_the_target() {
    let child = std::env::var_os(CHILD).is_some();
    if child {
        // The layer is process-wide and must be on before the host device
        // is made; the child runs this test alone.
        let mut layer: Option<ID3D12Debug> = None;
        // SAFETY: plain queries, before any device exists in this process.
        if unsafe { D3D12GetDebugInterface(&mut layer) }.is_err() {
            eprintln!("skipped the debug layer run: no D3D12 debug layer");
            return;
        }
        // SAFETY: as above.
        unsafe { layer.unwrap().EnableDebugLayer() };
    }
    let Some(host) = crate::host::host() else {
        eprintln!("skipped indirect rectangles: no host D3D12 device");
        return;
    };
    let messages: Option<ID3D12InfoQueue> = child.then(|| host.device.cast().unwrap());
    let check = |case: &str| {
        if let Some(messages) = &messages {
            let errors = layer_errors(messages);
            assert!(errors.is_empty(), "{case}: debug layer errors: {errors:#?}");
        }
    };
    let pso = rectangle_pipeline();
    check("the pipeline");
    draw_cases(host, pso, false, &check);
    draw_cases(host, pso, true, &check);
    assert_eq!(slot::<extern "system" fn(This) -> u32>(pso, 0x10)(pso), 0);
    if child {
        eprintln!("debug layer run: no errors");
        return;
    }

    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "pipeline::rect_indirect_test::indirect_rectangles_reach_the_target",
            "--exact",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    eprintln!("{text}");
    assert!(out.status.success(), "the run under the debug layer failed");
    assert!(text.contains("1 passed"), "the child ran no test");
}
