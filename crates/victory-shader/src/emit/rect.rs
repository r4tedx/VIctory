//! The console's rectangle lists on the host. A rectangle comes as three
//! corners in any order, and the hardware adds the fourth across the
//! diagonal from the corner between the other two, for the position and
//! every attribute: `a + b - c`, `c` the corner facing the longest side.
//! Shaders differ: a full-screen rectangle can come as top left, bottom
//! left, bottom right (`c` second), a blit or tile as top left, bottom left,
//! top right (`c` first, as Mesa's own blits give them). The host has no
//! such primitive; a geometry shader made for the vertex shader that feeds
//! it takes each triangle `(v0, v1, v2)` and emits it with its completion
//! `(a, b, v3)`, both turning the same way. Integer outputs (which can only
//! be flat) keep `v0`'s value at the fourth corner.

use super::bitcode::{Attr, Bin, Term, pred};
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The geometry shader that completes the rectangles of the vertex shader
/// container `vs` (its output signature is the shader's input and output).
pub fn rectangle_gs(vs: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(vs)?;
    let part = c
        .part(FourCC::OSG1)
        .ok_or(ShaderError::Malformed("a vertex shader without outputs"))?;
    let mut elements = super::signature(part.data)?;
    elements.retain(|e| e.row.is_some());
    for e in &mut elements {
        e.usage = (1 << e.cols) - 1;
    }
    let mut s = Shader::new(false);
    s.gs = Some(6);
    let b = s.entry;
    // Corner k's values, element by element, component by component.
    let mut corners: Vec<Vec<Vec<super::bitcode::V>>> = Vec::new();
    for k in 0..3u32 {
        let mut values = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let (ty, suffix) = if e.comp == 9 {
                (s.t.f32, ".f32")
            } else {
                (s.t.i32, ".i32")
            };
            let mut comps = Vec::new();
            for col in 0..e.cols {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(k)];
                comps.push(s.op(b, &format!("loadInput{suffix}"), Attr::ReadNone, ty, 4, &args));
            }
            values.push(comps);
        }
        corners.push(values);
    }
    // Which corner is between the other two: the one facing the longest
    // side, in screen space. Without a position (or with ties, a degenerate
    // rectangle) the second one.
    let (first, third) = match elements
        .iter()
        .position(|e| e.kind == 3 && e.comp == 9 && e.cols >= 2)
    {
        Some(p) => {
            let xy: Vec<_> = (0..3)
                .map(|k| {
                    let c = &corners[k][p];
                    let (x, y) = (c[0], c[1]);
                    if c.len() >= 4 {
                        (
                            s.m.bin(b, Bin::SDiv, x, c[3]),
                            s.m.bin(b, Bin::SDiv, y, c[3]),
                        )
                    } else {
                        (x, y)
                    }
                })
                .collect();
            let mut side = |i: usize, j: usize| {
                let dx = s.m.bin(b, Bin::Sub, xy[i].0, xy[j].0);
                let dy = s.m.bin(b, Bin::Sub, xy[i].1, xy[j].1);
                let (xx, yy) = (s.m.bin(b, Bin::Mul, dx, dx), s.m.bin(b, Bin::Mul, dy, dy));
                s.m.bin(b, Bin::Add, xx, yy)
            };
            let (d01, d02, d12) = (side(0, 1), side(0, 2), side(1, 2));
            let mut longest = |d: _, e: _, f: _| {
                let over_e = s.m.cmp(b, pred::OGT, d, e);
                let over_f = s.m.cmp(b, pred::OGT, d, f);
                s.m.bin(b, Bin::And, over_e, over_f)
            };
            (Some(longest(d12, d01, d02)), Some(longest(d01, d02, d12)))
        }
        None => (None, None),
    };
    // Corner k's value, picked by the corner between the others: the
    // completion is `(a, b, a + b - c)`, `(v0, v2, ..)` when it is the
    // second, `(v2, v1, ..)` when the first, `(v1, v0, ..)` when the third.
    let pick = |m: &mut super::bitcode::Module, [if_first, if_second, if_third]: [_; 3]| {
        let v = match third {
            Some(t) => m.select(b, t, if_third, if_second),
            None => if_second,
        };
        match first {
            Some(f) => m.select(b, f, if_first, v),
            None => v,
        }
    };
    let mut completion: [Vec<Vec<_>>; 3] = Default::default();
    for (id, e) in elements.iter().enumerate() {
        let [mut a, mut bs, mut fourth] = [Vec::new(), Vec::new(), Vec::new()];
        let columns = corners[0][id].iter().zip(&corners[1][id]).zip(&corners[2][id]);
        for ((&v0, &v1), &v2) in columns.take(usize::from(e.cols)) {
            let (va, vb) = (pick(&mut s.m, [v2, v0, v1]), pick(&mut s.m, [v1, v2, v0]));
            a.push(va);
            bs.push(vb);
            fourth.push(if e.comp == 9 {
                let c = pick(&mut s.m, [v0, v1, v2]);
                let sum = s.m.bin(b, Bin::Add, va, vb);
                s.m.bin(b, Bin::Sub, sum, c)
            } else {
                v0
            });
        }
        completion[0].push(a);
        completion[1].push(bs);
        completion[2].push(fourth);
    }
    corners.extend(completion);
    for triangle in [[0, 1, 2], [3, 4, 5]] {
        for k in triangle {
            for (id, e) in elements.iter().enumerate() {
                let suffix = if e.comp == 9 { ".f32" } else { ".i32" };
                for col in 0..e.cols {
                    let v = corners[k][id][usize::from(col)];
                    let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), v];
                    s.op(b, &format!("storeOutput{suffix}"), Attr::NoUnwind, s.t.void, 5, &args);
                }
            }
            let stream = [s.m.i8(0)];
            s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &stream);
        }
        let stream = [s.m.i8(0)];
        s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &stream);
    }
    s.inputs = elements.clone();
    s.outputs = elements;
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::{Attr, Term};
    use super::super::container::SigElement;
    use super::super::host::{Draw, Host};
    use super::super::shader::Shader;

    fn element(name: &str, kind: u8, interp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// A shader copying its inputs to its outputs (vertex), or writing
    /// target 0 from TEXCOORD (pixel).
    fn shader(pixel: bool) -> Vec<u8> {
        let mut s = Shader::new(pixel);
        let b = s.entry;
        let (inputs, outputs) = if pixel {
            (
                vec![element("SV_Position", 3, 4, 0, 4), element("TEXCOORD", 0, 2, 1, 2)],
                vec![element("SV_Target", 16, 0, 0, 4)],
            )
        } else {
            (
                vec![element("POSITION", 0, 0, 0, 4), element("TEXCOORD", 0, 0, 1, 2)],
                vec![element("SV_Position", 3, 0, 0, 4), element("TEXCOORD", 0, 0, 1, 2)],
            )
        };
        let load = |s: &mut Shader, id: u32, col: u8| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
        };
        let values: Vec<_> = if pixel {
            let one = s.m.f32(1f32.to_bits());
            vec![load(&mut s, 1, 0), load(&mut s, 1, 1), one, one]
        } else {
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
                .into_iter()
                .map(|(id, c)| load(&mut s, id, c))
                .collect()
        };
        let mut k = 0;
        for (id, e) in outputs.iter().enumerate() {
            for col in 0..e.cols {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), values[k]];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
                k += 1;
            }
        }
        s.inputs = inputs;
        s.outputs = outputs;
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    /// One rectangle (top left, bottom left, bottom right) with a
    /// coordinate attribute: every pixel of the square is drawn, the
    /// attribute interpolated over it, the top-right half included.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        // The corner between the other two second.
        draws_the_square(&[[-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]);
    }

    /// Tiles and blits give the corners as top left, bottom left, top
    /// right: the corner between the others first, its fourth the bottom
    /// right. Completed across the wrong diagonal (`v0 + v2 - v1`), the
    /// second triangle lands on the square above, and the bottom-right half
    /// is never drawn.
    #[test]
    fn rectangles_with_the_corner_first_complete_across_their_diagonal() {
        draws_the_square(&[[-0.5, 0.5], [-0.5, -0.5], [0.5, 0.5]]);
    }

    /// The corner between the others third (top right, bottom left, top
    /// left): the fourth is the bottom right as well.
    #[test]
    fn rectangles_with_the_corner_third_complete_across_their_diagonal() {
        draws_the_square(&[[0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]]);
    }

    /// Draws one rectangle of the given corners (clip x, y; the attribute
    /// is the same x, y) into 32x32 and checks every pixel: the square
    /// `|x|, |y| < 0.5` holds its coordinates, the rest nothing.
    fn draws_the_square(corners: &[[f32; 2]; 3]) {
        let (vs, ps) = (shader(false), shader(true));
        let gs = super::rectangle_gs(&vs).unwrap();
        if let Ok(mut v) =
            super::super::validator::Validator::load(super::super::validator::Validator::default_path().as_deref())
        {
            for c in [&vs, &gs, &ps] {
                v.validate(&mut c.clone()).unwrap();
            }
        }
        let vertices: Vec<f32> = corners
            .iter()
            .flat_map(|&[x, y]| [x, y, 0.5, 1.0, x, y])
            .collect();
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else { continue };
            let root = host.root_signature_for(&[]).unwrap();
            let pso = host.pipeline_gs(&root, &vs, &gs, &ps).unwrap();
            let draw = Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &vertices,
                stride: 24,
                size: (32, 32),
                mesh_groups: 0,
            };
            let texels = host.draw(&root, &pso, &draw).unwrap();
            for (i, t) in texels.iter().enumerate() {
                let (x, y) = ((i % 32) as f32, (i / 32) as f32);
                let (u, v) = ((x + 0.5) / 16.0 - 1.0, 1.0 - (y + 0.5) / 16.0);
                let inside = u.abs() < 0.5 && v.abs() < 0.5;
                let want = if inside { [u, v, 1.0, 1.0] } else { [0.0; 4] };
                let near = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4);
                assert!(near, "warp {warp} pixel ({x}, {y}): {t:?}, want {want:?}");
            }
        }
    }
}
