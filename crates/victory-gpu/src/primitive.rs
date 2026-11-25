//! Console primitive types the host lacks. A rectangle list gives three
//! corners per rectangle in any order, and the hardware adds the fourth
//! across the diagonal from the corner between the other two: `a + b - c`,
//! `c` the corner facing the longest side (Grand Theft Auto VI's
//! full-screen rectangles come as top left, bottom left, bottom right, `c`
//! second; tiles as top left, bottom left, top right, `c` first). Pipelines
//! whose geometry shader completes them (`victory_shader::emit::rect`)
//! draw them as they are; for the rest each rectangle becomes two
//! triangles here, with the fourth vertex made from the other three on the
//! CPU, by the same rule.
//!
//! shortcut: the fourth vertex is computed from vertex inputs, not from
//! vertex shader outputs as the hardware does; that is the same for vertex
//! shaders linear in their inputs (blits, 2D quads). The corner between the
//! others is found from the input named as the position, not from the
//! screen-space position; for a rectangle in its own input space that is
//! the same corner. Without such an input, or with a tie (a degenerate
//! rectangle), it is the second, as before. Inputs in formats
//! [`components`] does not know keep the first vertex's value, and so do
//! components past the end of the vertex (an element that reaches into the
//! next one, which the layout allows).

/// Where the rectangles' corners are: the position input's vertices (the
/// same rectangles as the ones expanded, in its own buffer), its byte
/// offset in each and its format.
#[derive(Clone, Copy)]
pub(crate) struct Position<'a> {
    pub(crate) src: &'a [u8],
    pub(crate) stride: usize,
    pub(crate) offset: u32,
    pub(crate) format: u32,
}

/// Expands `rects` rectangles of `stride`-byte vertices from `src` into six
/// vertices each in `dst`: `(v0, v1, v2)` and its completion
/// `(a, b, a + b - c)`, `c` the corner between the other two (see
/// [`between`]), so both triangles turn the same way: `(v0, v2, ..)` when
/// it is the second, `(v2, v1, ..)` when the first, `(v1, v0, ..)` when the
/// third. `elements` are (byte offset, format).
pub(crate) fn expand_rects(
    src: &[u8],
    dst: &mut [u8],
    stride: usize,
    elements: &[(u32, u32)],
    position: Option<Position<'_>>,
) {
    if stride == 0 {
        return;
    }
    let (Some(in_rect), Some(out_rect)) = (stride.checked_mul(3), stride.checked_mul(6)) else {
        return;
    };
    for (r, (rect, out)) in src
        .chunks_exact(in_rect)
        .zip(dst.chunks_exact_mut(out_rect))
        .enumerate()
    {
        let v = |i: usize| &rect[i * stride..(i + 1) * stride];
        let [a, b, c] = match position.map_or(1, |p| between(p, r)) {
            0 => [2, 1, 0],
            2 => [1, 0, 2],
            _ => [0, 2, 1],
        };
        for (slot, i) in [0, 1, 2, a, b].into_iter().enumerate() {
            out[slot * stride..(slot + 1) * stride].copy_from_slice(v(i));
        }
        let fourth = &mut out[5 * stride..];
        fourth.copy_from_slice(v(0));
        for &(offset, format) in elements {
            let at = offset as usize;
            let Some((n, kind)) = components(format) else {
                continue;
            };
            // Only the components inside the vertex: the rest belong to the
            // next one.
            let inside = stride.saturating_sub(at) / kind.bytes();
            for k in 0..n.min(inside) {
                let get = |i: usize| kind.read(&v(i)[at..], k);
                let value = get(a) + get(b) - get(c);
                kind.write(&mut fourth[at..], k, value);
            }
        }
    }
}

/// Which corner of rectangle `rect` is between the other two: the one
/// facing the longest side, from x, y of the position (divided by w when it
/// has one), as the geometry shader does. The second when the position has
/// fewer than two components inside its vertex or lies outside its
/// buffer, in a format [`components`] does not know, or on a tie or an
/// unordered comparison (a zero w).
fn between(p: Position<'_>, rect: usize) -> usize {
    let Some((n, kind)) = components(p.format) else {
        return 1;
    };
    let at = p.offset as usize;
    let inside = n.min(p.stride.saturating_sub(at) / kind.bytes());
    let corner = |i: usize| -> Option<(f32, f32)> {
        let start = rect.checked_mul(3)?.checked_add(i)?.checked_mul(p.stride)?;
        let vertex = p.src.get(start..start.checked_add(p.stride)?)?;
        let get = |k: usize| kind.read(&vertex[at..], k);
        let (x, y) = (get(0), get(1));
        Some(if inside >= 4 {
            (x / get(3), y / get(3))
        } else {
            (x, y)
        })
    };
    if inside < 2 {
        return 1;
    }
    let (Some(c0), Some(c1), Some(c2)) = (corner(0), corner(1), corner(2)) else {
        return 1;
    };
    let side = |(ax, ay): (f32, f32), (bx, by): (f32, f32)| {
        let (dx, dy) = (ax - bx, ay - by);
        dx * dx + dy * dy
    };
    let (d01, d02, d12) = (side(c0, c1), side(c0, c2), side(c1, c2));
    if d12 > d01 && d12 > d02 {
        0
    } else if d01 > d02 && d01 > d12 {
        2
    } else {
        1
    }
}

#[derive(Clone, Copy)]
enum Kind {
    F32,
    F16,
    Unorm8,
    Unorm16,
    Snorm8,
    Snorm16,
}

/// Component count and kind of the vertex formats rectangles use.
fn components(format: u32) -> Option<(usize, Kind)> {
    Some(match format {
        2 => (4, Kind::F32),
        6 => (3, Kind::F32),
        16 => (2, Kind::F32),
        41 => (1, Kind::F32),
        10 => (4, Kind::F16),
        34 => (2, Kind::F16),
        54 => (1, Kind::F16),
        11 => (4, Kind::Unorm16),
        35 => (2, Kind::Unorm16),
        56 => (1, Kind::Unorm16),
        28 | 29 | 87 | 91 => (4, Kind::Unorm8),
        49 => (2, Kind::Unorm8),
        61 => (1, Kind::Unorm8),
        13 => (4, Kind::Snorm16),
        37 => (2, Kind::Snorm16),
        58 => (1, Kind::Snorm16),
        31 => (4, Kind::Snorm8),
        51 => (2, Kind::Snorm8),
        63 => (1, Kind::Snorm8),
        _ => return None,
    })
}

impl Kind {
    fn bytes(self) -> usize {
        match self {
            Kind::F32 => 4,
            Kind::F16 | Kind::Unorm16 | Kind::Snorm16 => 2,
            Kind::Unorm8 | Kind::Snorm8 => 1,
        }
    }

    fn read(self, at: &[u8], c: usize) -> f32 {
        let b = &at[c * self.bytes()..];
        match self {
            Kind::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Kind::F16 => half_to_f32(u16::from_le_bytes([b[0], b[1]])),
            Kind::Unorm16 => f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
            Kind::Unorm8 => f32::from(b[0]) / 255.0,
            Kind::Snorm16 => (f32::from(i16::from_le_bytes([b[0], b[1]])) / 32767.0).max(-1.0),
            Kind::Snorm8 => (f32::from(b[0] as i8) / 127.0).max(-1.0),
        }
    }

    fn write(self, at: &mut [u8], c: usize, v: f32) {
        let n = self.bytes();
        let b = &mut at[c * n..(c + 1) * n];
        match self {
            Kind::F32 => b.copy_from_slice(&v.to_le_bytes()),
            Kind::F16 => b.copy_from_slice(&f32_to_half(v).to_le_bytes()),
            Kind::Unorm16 => {
                b.copy_from_slice(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes())
            }
            Kind::Unorm8 => b[0] = (v.clamp(0.0, 1.0) * 255.0).round() as u8,
            Kind::Snorm16 => {
                b.copy_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
            }
            Kind::Snorm8 => b[0] = (v.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8,
        }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: scale the mantissa.
        (0, m) => {
            let v = m as f32 * 2f32.powi(-24);
            return if sign != 0 { -v } else { v };
        }
        (31, m) => sign | 0x7f80_0000 | m << 13,
        (e, m) => sign | (e + 112) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = bits & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00; // too large: infinity
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // below half the smallest subnormal: zero
        }
        // Half subnormals are `m * 2^-24`. An f32 in this range is still
        // normal (exponent field >= 1), so its significand has the implicit
        // leading 1; shift it into the half mantissa, rounding to even.
        let sig = (1u32 << 23) | man;
        let shift = (14 - exp) as u32;
        let halfway = 1u32 << (shift - 1);
        let q = sig >> shift;
        let r = sig & ((1u32 << shift) - 1);
        let m = q + u32::from(r > halfway || (r == halfway && (q & 1) != 0));
        return sign | m as u16;
    }
    // Rounded to nearest: a carry out of the mantissa is added into the
    // exponent (`2 - 2^-12` is 2.0); at most it reaches infinity, 0x7c00.
    let half = (exp as u32) << 10;
    sign | (half + ((man + 0x1000) >> 13)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    const TL: [f32; 2] = [-1.0, 1.0];
    const BL: [f32; 2] = [-1.0, -1.0];
    const BR: [f32; 2] = [1.0, -1.0];
    const TR: [f32; 2] = [1.0, 1.0];

    fn floats(vertices: &[&[f32]]) -> Vec<u8> {
        vertices
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|x| x.to_le_bytes())
            .collect()
    }

    /// Vertex `i` of `dst` as floats.
    fn vertex(dst: &[u8], stride: usize, i: usize) -> Vec<f32> {
        let (floats, _) = dst[i * stride..(i + 1) * stride].as_chunks::<4>();
        floats.iter().map(|&b| f32::from_le_bytes(b)).collect()
    }

    /// Expands one rectangle of float vertices (one element of `format` at
    /// 0) whose position is the element `position` (offset, format) of the
    /// same vertices, and gives the six vertices out.
    fn expand(corners: &[&[f32]], format: u32, position: Option<(u32, u32)>) -> Vec<Vec<f32>> {
        let src = floats(corners);
        let stride = corners[0].len() * 4;
        let mut dst = vec![0u8; 6 * stride];
        let position = position.map(|(offset, format)| Position {
            src: &src,
            stride,
            offset,
            format,
        });
        expand_rects(&src, &mut dst, stride, &[(0, format)], position);
        (0..6).map(|i| vertex(&dst, stride, i)).collect()
    }

    /// Twice the signed area of a triangle's x, y.
    fn turn(t: &[Vec<f32>]) -> f32 {
        let (a, b, c) = (&t[0], &t[1], &t[2]);
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }

    /// The corner between the other two (the right angle) second, first or
    /// third: the completion is `(a, b, a + b - c)` across the diagonal
    /// from it, the bottom right or top right of the square, and turns the
    /// same way as the rectangle's own triangle. Other vertex buffers (here
    /// texture coordinates) are completed by the same corner, found in the
    /// position's buffer.
    #[test]
    fn rectangles_complete_across_the_diagonal_from_their_right_angle() {
        let uv = |[x, y]: [f32; 2]| [(x + 1.0) / 2.0, (1.0 - y) / 2.0];
        for (corners, want) in [
            ([TL, BL, BR], [TL, BR, TR]),
            ([TL, BL, TR], [TR, BL, BR]),
            ([TR, BL, TL], [BL, TR, BR]),
        ] {
            let out = expand(&corners.each_ref().map(|c| c.as_slice()), 16, Some((0, 16)));
            assert_eq!(out[..3], corners.map(|c| c.to_vec())[..], "{corners:?}");
            assert_eq!(out[3..], want.map(|c| c.to_vec())[..], "{corners:?}");
            assert!(
                turn(&out[..3]) * turn(&out[3..]) > 0.0,
                "{corners:?}: {out:?}"
            );

            let position = floats(&corners.each_ref().map(|c| c.as_slice()));
            let coords = floats(&corners.map(uv).each_ref().map(|c| c.as_slice()));
            let mut dst = vec![0u8; 6 * 8];
            let position = Position {
                src: &position,
                stride: 8,
                offset: 0,
                format: 16,
            };
            expand_rects(&coords, &mut dst, 8, &[(0, 16)], Some(position));
            let got: Vec<_> = (3..6).map(|i| vertex(&dst, 8, i)).collect();
            assert_eq!(got, want.map(|c| uv(c).to_vec()), "{corners:?}");
        }
    }

    /// A float4 position is divided by w first. Undivided, two sides of the
    /// corners below tie (taken as a degenerate rectangle: the second
    /// corner); divided, the first corner is the square's right angle.
    #[test]
    fn rectangle_corners_are_compared_after_the_divide_by_w() {
        let corners: [&[f32]; 3] = [
            &[-4.0, 4.0, 0.0, 4.0],
            &[-1.0, -1.0, 0.0, 1.0],
            &[1.0, 1.0, 0.0, 1.0],
        ];
        let out = expand(&corners, 2, Some((0, 2)));
        assert_eq!(out[3], corners[2]);
        assert_eq!(out[4], corners[1]);
        assert_eq!(out[5], [4.0, -4.0, 0.0, -2.0], "v1 + v2 - v0");
        // Only x, y (float2 at 0): a tie, the second corner.
        let out = expand(&corners, 2, Some((0, 16)));
        assert_eq!(out[3], corners[0]);
        assert_eq!(out[4], corners[2]);
        assert_eq!(out[5], [-2.0, 6.0, 0.0, 4.0], "v0 + v2 - v1");
    }

    /// The old rule (the second corner between the others, fourth
    /// `v0 + v2 - v1`) where the corners can't be compared: no position, one
    /// of a single component, of an unknown format, past the end of the
    /// vertex or outside its buffer, a tie, or a zero w (unordered
    /// comparisons).
    /// The corners are a tile's (top left, bottom left, top right), whose
    /// right angle is first.
    #[test]
    fn rectangles_without_comparable_corners_complete_from_the_second() {
        let second = |out: &[Vec<f32>], corners: &[&[f32]]| {
            let fourth: Vec<f32> = (0..corners[0].len())
                .map(|k| corners[0][k] + corners[2][k] - corners[1][k])
                .collect();
            assert_eq!(out[3], corners[0]);
            assert_eq!(out[4], corners[2]);
            assert_eq!(out[5], fourth);
        };
        let tile: [&[f32]; 3] = [&TL, &BL, &TR];
        for position in [
            None,
            Some((0, 41)),
            Some((0, 999)),
            Some((8, 16)),
            Some((4, 16)),
        ] {
            second(&expand(&tile, 16, position), &tile);
        }
        // The position's buffer holds no whole rectangle.
        let src = floats(&tile);
        let mut dst = vec![0u8; 6 * 8];
        let short = Position {
            src: &src[..20],
            stride: 8,
            offset: 0,
            format: 16,
        };
        expand_rects(&src, &mut dst, 8, &[(0, 16)], Some(short));
        second(
            &(0..6).map(|i| vertex(&dst, 8, i)).collect::<Vec<_>>(),
            &tile,
        );
        // A tie: the first and the third side equally long.
        let tie: [&[f32]; 3] = [&[-4.0, 4.0], &BL, &TR];
        second(&expand(&tie, 16, Some((0, 16))), &tie);
        // A zero w at the second corner: x / w and y / w are not numbers.
        let zero_w: [&[f32]; 3] = [
            &[-1.0, 1.0, 0.0, 1.0],
            &[0.0, -1.0, 0.0, 0.0],
            &[1.0, 1.0, 0.0, 1.0],
        ];
        second(&expand(&zero_w, 2, Some((0, 2))), &zero_w);
    }

    /// A rectangle of float2 positions and RGBA8 colors, given as top
    /// left, bottom left, bottom right: the fourth corner completes the
    /// parallelogram at the top right.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        let vertex = |x: f32, y: f32, r: u8| {
            let mut v = Vec::new();
            v.extend(x.to_le_bytes());
            v.extend(y.to_le_bytes());
            v.extend([r, 0, 0, 255]);
            v
        };
        let src: Vec<u8> = [
            vertex(-1.0, 1.0, 0),
            vertex(-1.0, -1.0, 100),
            vertex(1.0, -1.0, 150),
        ]
        .concat();
        let mut dst = vec![0u8; 6 * 12];
        expand_rects(&src, &mut dst, 12, &[(0, 16), (8, 28)], None);
        assert_eq!(dst[..36], src[..]);
        assert_eq!(dst[36..48], src[..12]);
        assert_eq!(dst[48..60], src[24..36]);
        assert_eq!(dst[60..], vertex(1.0, 1.0, 50)[..]);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x3e00), 1.5);
        assert_eq!(half_to_f32(0xc000), -2.0);
    }

    /// A rounding that carries out of the mantissa raises the exponent
    /// (the carry used to be or-ed into an odd exponent, halving the value).
    #[test]
    fn halves_round_up_into_the_next_power_of_two() {
        assert_eq!(f32_to_half(2.0 - 2f32.powi(-12)), 0x4000);
        assert_eq!(f32_to_half(0.5 - 2f32.powi(-13)), 0x3800);
        assert_eq!(f32_to_half(65520.0), 0x7c00, "past the largest half");
        // An R16_FLOAT element at 1.0, 2^-12 and 1.0: the fourth corner is
        // 2 - 2^-12, which rounds to 2.0.
        let src: Vec<u8> = [0x3c00_u16, 0x0c00, 0x3c00]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let mut dst = vec![0u8; 6 * 2];
        expand_rects(&src, &mut dst, 2, &[(0, 54)], None);
        assert_eq!(dst[10..], 0x4000_u16.to_le_bytes());
    }

    /// An element that reaches past the stride (float4 at 4 in 8-byte
    /// vertices, RGBA8 at the stride itself): the components inside get
    /// the fourth corner, the rest keep the first vertex's bytes. The old
    /// code indexed past the vertex and panicked.
    #[test]
    fn elements_past_the_stride_are_left_alone() {
        let vertex = |x: f32, y: f32| [x.to_le_bytes(), y.to_le_bytes()].concat();
        let src = [vertex(0.0, 2.0), vertex(1.0, 3.0), vertex(5.0, 7.0)].concat();
        let mut dst = vec![0u8; 6 * 8];
        expand_rects(&src, &mut dst, 8, &[(0, 16), (4, 2), (8, 28)], None);
        assert_eq!(dst[..40], [&src[..], &src[..8], &src[16..]].concat()[..]);
        assert_eq!(dst[40..], vertex(4.0, 6.0)[..]);
    }

    #[test]
    fn normalized_attributes_are_interpolated_in_their_own_range() {
        let vertex = |rg: [u8; 2], u16_value: u16, u8_value: u8, snorm8: i8, snorm16: i16| {
            [
                &rg[..],
                &u16_value.to_le_bytes(),
                &[u8_value, snorm8 as u8],
                &snorm16.to_le_bytes(),
            ]
            .concat()
        };
        let src = [
            vertex([3, 250], 1000, 10, i8::MIN, -23000),
            vertex([8, 200], 500, 3, -120, -18000),
            vertex([20, 253], 10000, 100, -120, -32000),
        ]
        .concat();
        let mut dst = [0; 6 * 8];
        expand_rects(
            &src,
            &mut dst,
            8,
            &[(0, 49), (2, 56), (4, 61), (5, 63), (6, 58)],
            None,
        );
        assert_eq!(dst[40..], vertex([15, 255], 10500, 107, -127, -32767));
    }

    /// `half_to_f32` reads subnormals, so an f16 rectangle output must be
    /// able to write them back instead of flushing every one to zero.
    #[test]
    fn half_subnormals_convert_in_both_directions() {
        assert_eq!(f32_to_half(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_half(-2f32.powi(-24)), 0x8001);
        assert_eq!(f32_to_half(2f32.powi(-23)), 0x0002);
        assert_eq!(f32_to_half(1023.0 * 2f32.powi(-24)), 0x03ff);
        // Ties go to an even mantissa: 0/1 and 1/2.
        assert_eq!(f32_to_half(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_half(3.0 * 2f32.powi(-25)), 0x0002);
        // Below half the smallest subnormal, and normal values, are unchanged.
        assert_eq!(f32_to_half(2f32.powi(-26)), 0x0000);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn zero_or_overflowing_stride_is_ignored_without_panicking() {
        let mut dst = [0xAAu8; 16];
        expand_rects(&[1, 2, 3], &mut dst, 0, &[(0, 16)], None);
        assert_eq!(dst, [0xAAu8; 16]);
        expand_rects(&[1, 2, 3], &mut dst, usize::MAX / 2, &[(0, 16)], None);
        assert_eq!(dst, [0xAAu8; 16]);
    }

    #[test]
    fn unorm16_unorm8_clamping_and_unknown_formats_expand_correctly() {
        // Vertex layout (12 bytes):
        //   0..4: R16G16_UNORM (format 35)
        //   4..8: R8G8B8A8_UNORM (format 28) — tests clamping below 0 and above 1
        //   8..12: unknown format (999) — must keep v0's bytes verbatim
        let v = |u0: u16, u1: u16, c: [u8; 4], raw: [u8; 4]| {
            let mut b = Vec::with_capacity(12);
            b.extend(u0.to_le_bytes());
            b.extend(u1.to_le_bytes());
            b.extend(c);
            b.extend(raw);
            b
        };
        let src = [
            v(
                10_000,
                50_000,
                [10, 200, 100, 255],
                [0xDE, 0xAD, 0xBE, 0xEF],
            ),
            v(20_000, 10_000, [100, 50, 100, 255], [1, 2, 3, 4]),
            v(30_000, 20_000, [20, 200, 100, 255], [5, 6, 7, 8]),
        ]
        .concat();
        let mut dst = vec![0u8; 6 * 12];
        expand_rects(&src, &mut dst, 12, &[(0, 35), (4, 28), (8, 999)], None);
        let fourth = &dst[60..72];
        // u0: 10000 + 30000 - 20000 = 20000; u1: 50000 + 20000 - 10000 = 60000
        assert_eq!(u16::from_le_bytes([fourth[0], fourth[1]]), 20_000);
        assert_eq!(u16::from_le_bytes([fourth[2], fourth[3]]), 60_000);
        // c0: 10 + 20 - 100 -> clamped to 0; c1: 200 + 200 - 50 -> clamped to 255
        assert_eq!(&fourth[4..8], &[0, 255, 100, 255]);
        // Unknown format keeps v0 bytes
        assert_eq!(&fourth[8..12], &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn multiple_rectangles_and_f32_f16_vector_formats_expand_in_sequence() {
        // Two rectangles of R32G32B32_FLOAT (format 6, 12 bytes) + R16G16_FLOAT (format 34, 4 bytes) = 16 bytes
        let v = |p: [f32; 3], uv: [f32; 2]| {
            let mut b = Vec::with_capacity(16);
            for x in p {
                b.extend(x.to_le_bytes());
            }
            for x in uv {
                b.extend(f32_to_half(x).to_le_bytes());
            }
            b
        };
        let src = [
            // Rect 0
            v([0.0, 1.0, 0.5], [0.0, 0.0]),
            v([0.0, 0.0, 0.5], [0.0, 1.0]),
            v([2.0, 0.0, 0.5], [1.0, 1.0]),
            // Rect 1
            v([10.0, 20.0, 1.0], [0.25, 0.5]),
            v([10.0, 10.0, 1.0], [0.25, 1.0]),
            v([14.0, 10.0, 1.0], [0.75, 1.0]),
        ]
        .concat();
        let mut dst = vec![0u8; 12 * 16];
        expand_rects(&src, &mut dst, 16, &[(0, 6), (12, 34)], None);
        assert_eq!(&dst[5 * 16..6 * 16], &v([2.0, 1.0, 0.5], [1.0, 0.0])[..]);
        assert_eq!(
            &dst[11 * 16..12 * 16],
            &v([14.0, 20.0, 1.0], [0.75, 0.5])[..]
        );
    }
}
