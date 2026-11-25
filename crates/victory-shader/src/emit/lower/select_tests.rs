use super::super::emit;
use crate::native::{ComputeIo, RootParam};
use crate::{Container, FourCC, HwStage, NativeProgram, ResourceType, RootType, UsedResource};

#[test]
fn selects_from_independently_written_mask_words() {
    let words = [
        0xbe81_0380u32, // s1 = 0 (table address high)
        0xf408_0200,
        0xfa00_0000, // load UAV into s[8:11]
        0xbe84_03ff,
        0x8000_0000, // s4 = lane 31
        0xbe85_0381, // s5 = lane 32 in wave64
        0x7e02_028b,
        0x7e04_029d, // v1 = 11, v2 = 29
        0xd501_0003,
        0x0012_0501, // v_cndmask_b32 v3, v1, v2, s[4:5]
        0x3402_0082, // v1 = 4 * v0
        0xe070_1000,
        0x8002_0301, // store v3 to UAV at v1
        0xbf81_0000,
    ];
    let container = crate::dxbc::build(&[]);
    let c = Container::parse(&container).unwrap();
    for (wave64, uses_select, threads) in [
        (false, true, 32),
        (true, true, 64),
        (true, false, 128),
        (true, false, 64),
        (true, true, 128),
    ] {
        let mut words = words;
        if !uses_select {
            words[12] = 0x8002_0101; // store v1 instead of the select result
        }
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: crate::native::StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [threads, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let wave32_only = super::super::Target {
            lanes_min: 32,
            lanes_max: 32,
        };
        for target in [wave32_only, super::super::Target::ANY] {
            let result = emit(&c, &native, &p, &st, &io, &res, target);
            if uses_select && threads > 64 && !target.runs(64) {
                assert!(matches!(
                    result,
                    Err(crate::ShaderError::Unsupported {
                        what: "console wave64 lanes in a group of more than 64 threads on a host without 64-lane waves",
                        ..
                    })
                ));
                continue;
            }
            let shader = result.unwrap();
            let emitted = Container::parse(&shader.dxil).unwrap();
            let features = emitted.part(FourCC::SFI0).unwrap();
            let wave_ops = u64::from_le_bytes(features.data.try_into().unwrap()) & 0x4000 != 0;
            assert_eq!(wave_ops, uses_select);
        }
    }
}
