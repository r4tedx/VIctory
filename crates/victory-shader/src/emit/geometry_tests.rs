//! A synthetic GS loads three different vertices, moves the first one,
//! and emits a triangle strip. Compare its actual rasterized coverage
//! with the same operation compiled to DXIL by the Windows SDK.

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn geometry_part() -> Vec<u8> {
    let mut words = vec![
        // Establish EXEC from the merged GS lane count in s3.
        0x9404ff03,
        0x80008,
        0x92fe8004,
        0xbf06c004,
        0x85fe7ec1,
        // Read the ring page from the internal table, then add the wave offset.
        0xbe8a0308,
        0x870bff09,
        255,
        0xf4000305,
        0xfa0000b8,
        0x8f0c880c,
        0x810c020c,
        // Lane within the wave; v5 = wave base + lane * 48 bytes.
        0xd7650004,
        0x000100c1,
        0xd7660004,
        0x000208c1,
        0xd5430005,
        (12 << 18) | (260 << 9) | 176,
        // Unpack the first, second, and third vertex offsets to v6/v7/v8.
        0x340c0082,
        0x360c0cff,
        0x3fffc,
        0x340e00f9,
        0x05860682,
        0x34100282,
        0x361010ff,
        0x3fffc,
    ];
    for v in 0..3 {
        let addr = 6 + v;
        words.extend([0xd8dc0100, 10 << 24 | addr, 0xd8dc0302, 12 << 24 | addr]);
        if v == 0 {
            // v_add_f32 v10, 0.5, v10
            words.push(0x06000000 | 10 << 17 | 10 << 9 | 240);
        }
        for pair in 0..2 {
            let k = 4 * v + 2 * pair;
            let src = 10 + 2 * pair;
            words.extend([
                0xd8380000 | (k + 1) << 8 | k,
                (src + 1) << 16 | src << 8 | 5,
            ]);
        }
        words.push(0xbf900022); // emit
    }
    words.extend([0xbf900012, 0xbf900003, 0xbf810000]); // cut, done, endpgm
    let copy = bytes(&[
        0x8701ff01,
        255,
        0x34000082,
        0xf4000000,
        0xfa0000b8,
        0xd7460006,
        0x04011000, // byte address of this vertex
        0xd8dc0100,
        6,
        0xd8dc0302,
        2 << 24 | 6,
        0xf80008cf,
        0x03020100,
        0xbf810000, // exp pos0, v0..v3
    ]);
    crate::native::tests::geometry_part(&bytes(&words), &copy)
}

#[cfg(windows)]
#[test]
fn geometry_vertices_match_reference_draw() {
    use super::{HostShader, Target, container, host, translate_for, validator::Validator};
    use crate::{Container, FourCC, HwStage};
    let Some(dxc) = Validator::default_path()
        .map(|p| p.with_file_name("dxc.exe"))
        .filter(|p| p.exists())
    else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("victory-gs-draw-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let hlsl = dir.join("triangle.hlsl");
    std::fs::write(
        &hlsl,
        r#"
        struct V { float4 p: SV_Position; };
        V vs(float4 p: POSITION) { V v; v.p=p; return v; }
        [maxvertexcount(3)] void gs(triangle V v[3], inout TriangleStream<V> s) {
            v[0].p.x += 0.5; s.Append(v[0]); s.Append(v[1]); s.Append(v[2]); s.RestartStrip();
        }
        float4 ps(): SV_Target { return float4(1, 0, 0, 1); }
    "#,
    )
    .unwrap();
    let compile = |entry, stage| {
        let output = dir.join(format!("{entry}.dxil"));
        let proc = std::process::Command::new(&dxc)
            .args(["-T", &format!("{entry}_6_6"), "-E", entry, "-Fo"])
            .arg(&output)
            .arg(&hlsl)
            .output()
            .unwrap();
        assert!(
            proc.status.success(),
            "{}",
            String::from_utf8_lossy(&proc.stderr)
        );
        HostShader {
            stage,
            dxil: std::fs::read(output).unwrap(),
            ..host::stub_ps()
        }
    };
    let vs = compile("vs", HwStage::Vs);
    let ps = compile("ps", HwStage::Ps);
    let reference = compile("gs", HwStage::Gs);
    let native = geometry_part();
    let packed = Container::parse(&reference.dxil).unwrap();
    let parts: Vec<_> = packed
        .parts
        .iter()
        .map(|p| {
            (
                &p.fourcc.0,
                if p.fourcc == FourCC::DXIL {
                    &native
                } else {
                    p.data
                },
            )
        })
        .collect();
    let mut gs = translate_for(&container::build(&parts), Target::ANY).unwrap();
    Validator::load(Validator::default_path().as_deref())
        .unwrap()
        .validate(&mut gs.dxil)
        .unwrap();
    let vertices = [
        -0.75, -0.75, 0.5, 1.0, -0.75, 0.75, 0.5, 1.0, 0.75, -0.75, 0.5, 1.0,
    ];
    let draw = host::Draw {
        roots: vec![],
        textures: &[],
        samplers: &[],
        vertices: &vertices,
        stride: 16,
        size: (16, 16),
        mesh_groups: 0,
    };
    for warp in [false, true] {
        let Ok(host) = host::Host::new(warp) else {
            continue;
        };
        let run = |stage| {
            let (root, pso) = host.geometry_pipeline(&vs, stage, &ps).unwrap();
            host.draw(&root, &pso, &draw).unwrap()
        };
        let expected = run(&reference);
        let covered = expected.iter().filter(|p| p[0] == 1.0).count();
        assert!(
            (10..190).contains(&covered),
            "reference covered {covered}/256 pixels"
        );
        assert_eq!(run(&gs), expected);
    }
    for name in ["triangle.hlsl", "vs.dxil", "ps.dxil", "gs.dxil"] {
        std::fs::remove_file(dir.join(name)).unwrap();
    }
    std::fs::remove_dir(dir).unwrap();
}

#[test]
fn geometry_copy_rejects_untraced_reads_and_exports() {
    let native = crate::native_programs(&geometry_part())
        .unwrap()
        .pop()
        .unwrap();
    let program = native.translation_ir().unwrap();
    let crate::native::StageIo::Gs(mut geometry) = native.io else {
        panic!("missing GS data");
    };
    let original = geometry.copy.code.clone();
    // The first paired load now reads at the uninitialised v17 instead
    // of the copy vertex address in v6.
    geometry.copy.code[8 * 4..9 * 4].copy_from_slice(&17u32.to_le_bytes());
    assert!(crate::geometry::raise(&program, &geometry).is_err());
    geometry.copy.code = original;
    // Redirect an export source to an unwritten VGPR.
    let last_export_sources = geometry.copy.code.len() - 8;
    geometry.copy.code[last_export_sources] = 19;
    assert!(crate::geometry::raise(&program, &geometry).is_err());
    // Unconditional early termination cannot be reported as a successful
    // copy of an empty output signature.
    geometry.copy.code = bytes(&[0xbf810000]);
    assert!(crate::geometry::raise(&program, &geometry).is_err());
}

#[test]
fn geometry_io_keeps_component_order_and_constants_in_the_copy_export() {
    use crate::geometry::{Component, StreamOp};
    let native = crate::native_programs(&geometry_part())
        .unwrap()
        .pop()
        .unwrap();
    let program = native.translation_ir().unwrap();
    let crate::native::StageIo::Gs(mut geometry) = native.io else {
        panic!("missing GS data");
    };
    geometry.copy.code.truncate(geometry.copy.code.len() - 4);
    // v_mov_b32 v4, 1.0; exp param0, v2, v0, v4, v1; endpgm
    geometry
        .copy
        .code
        .extend(bytes(&[0x7e0802f2, 0xf800020f, 0x01040002, 0xbf810000]));
    let streams = crate::geometry::raise(&program, &geometry).unwrap();
    let mut vertices = 0;
    for (index, operation) in streams.messages.values().enumerate() {
        match operation {
            StreamOp::Emit(outputs) => {
                let [pos, param] = outputs.as_slice() else {
                    panic!("copy exports");
                };
                assert_eq!(
                    param.channels,
                    [
                        pos.channels[2],
                        pos.channels[0],
                        Some(Component::Constant(1f32.to_bits())),
                        pos.channels[1]
                    ]
                );
                vertices += 1;
            }
            StreamOp::Cut => assert_eq!(index, 3),
        }
    }
    assert_eq!(vertices, 3);
}
