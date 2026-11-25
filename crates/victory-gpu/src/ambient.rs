//! A constant in place of what one program writes to one of its unordered
//! views: a stand-in, opt-in at run time, for an output whose inputs the
//! host never fills (light that rays would gather while the structures
//! they walk are not built, say).
//!
//! `VICTORY_AMBIENT=<program>:<view>:<value>`, rules separated by `;`:
//! - `<program>`: the 16-hex-digit shader hash the log prints for the
//!   compute program (`compute 0123456789abcdef`);
//! - `<view>`: the unordered view as the console shader addresses it, root
//!   parameter and offset in dwords in its descriptor table, `s1+176` or
//!   `1+176` (the notation of the diagnostics' table listings). That
//!   address comes from the console code itself, so it holds across
//!   translator versions, where an order of the views would not;
//! - `<value>`: a colour `r,g,b[,a]` (alpha 1 if left out), optionally
//!   with the packing of an integer view: `rgb9e5=`, `r11g11b10=` or
//!   `rgba16f=`; or a raw word `0x...`, `mask` (all ones) or `zero` for
//!   an integer view of any width, which keeps each channel's low bits
//!   (`zero` clears a float view too). A view in a float or normalized
//!   format takes the colour as it is.
//!
//! After every dispatch of the program, direct or indirect, the view is
//! cleared to the value (`ClearUnorderedAccessView*`) between unordered
//! access barriers. Nothing happens while the variable is unset.
//! The idea and the use of R9G9B9E5 as 32-bit words come from the
//! Belobug fork's `ambient.rs` (0603123, 4f45ab8, GPL-3.0-or-later),
//! which matches its program by name; here the user names it. The masks
//! and zeros for narrower integer outputs follow ScarlettPort's stand-in
//! encodings (`emit/stand_in.rs`, GPL-3.0-or-later).
//! shortcut: the same value everywhere, every frame.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How an integer view holds a colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Packing {
    Rgb9e5,
    R11g11b10,
    Rgba16f,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Value {
    /// A colour, and the packing an integer view takes (none: by the
    /// texture's format).
    Colour([f32; 4], Option<Packing>),
    /// A word for an integer view (its low bits per channel).
    Raw(u32),
}

/// One stand-in: program, view, value.
#[derive(Debug)]
pub(crate) struct Rule {
    /// Lowercase, as logged.
    pub(crate) program: String,
    pub(crate) slot: u8,
    pub(crate) offset: u32,
    pub(crate) value: Value,
    /// The first match was logged.
    pub(crate) seen: AtomicBool,
}

impl Rule {
    /// The view as written in the variable and the logs.
    pub(crate) fn view(&self) -> String {
        format!("s{}+{}", self.slot, self.offset)
    }
}

/// The rules in `text` (the variable's value).
pub(crate) fn parse(text: &str) -> Result<Vec<Rule>, String> {
    text.split(';')
        .map(str::trim)
        .filter(|r| !r.is_empty())
        .map(|rule| parse_rule(rule).map_err(|e| format!("{rule:?}: {e}")))
        .collect()
}

fn parse_rule(rule: &str) -> Result<Rule, String> {
    let mut parts = rule.splitn(3, ':');
    let (Some(program), Some(view), Some(value)) = (parts.next(), parts.next(), parts.next())
    else {
        return Err("not <program>:<view>:<value>".into());
    };
    let program = program.trim().to_ascii_lowercase();
    if program.len() != 16 || !program.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("the program is not a 16-hex-digit hash".into());
    }
    let view = view.trim();
    let view = view.strip_prefix(['s', 'S']).unwrap_or(view);
    let Some((slot, offset)) = view.split_once('+') else {
        return Err("the view is not <root parameter>+<dwords>".into());
    };
    let slot = slot
        .trim()
        .parse::<u8>()
        .map_err(|_| "the view's root parameter is not a number")?;
    let offset = offset
        .trim()
        .parse::<u32>()
        .map_err(|_| "the view's offset is not a number")?;
    Ok(Rule {
        program,
        slot,
        offset,
        value: parse_value(value.trim())?,
        seen: AtomicBool::new(false),
    })
}

fn parse_value(value: &str) -> Result<Value, String> {
    match value.to_ascii_lowercase().as_str() {
        "mask" => return Ok(Value::Raw(u32::MAX)),
        "zero" => return Ok(Value::Raw(0)),
        _ => {}
    }
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        return u32::from_str_radix(hex, 16)
            .map(Value::Raw)
            .map_err(|_| "the raw word is not 32-bit hex".into());
    }
    let (packing, colour) = match value.split_once('=') {
        Some((name, colour)) => {
            let packing = match name.trim().to_ascii_lowercase().as_str() {
                "rgb9e5" => Packing::Rgb9e5,
                "r11g11b10" => Packing::R11g11b10,
                "rgba16f" => Packing::Rgba16f,
                other => return Err(format!("no packing {other:?} (rgb9e5, r11g11b10, rgba16f)")),
            };
            (Some(packing), colour)
        }
        None => (None, value),
    };
    let channels = colour
        .split(',')
        .map(|c| {
            c.trim()
                .parse::<f32>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 0.0)
        })
        .collect::<Option<Vec<f32>>>()
        .ok_or("the colour is not 3 or 4 finite numbers of at least 0")?;
    let rgba = match channels[..] {
        [r, g, b] => [r, g, b, 1.0],
        [r, g, b, a] => [r, g, b, a],
        _ => return Err("the colour is not 3 or 4 numbers".into()),
    };
    Ok(Value::Colour(rgba, packing))
}

/// The rules `VICTORY_AMBIENT` sets, read once; none if unset or wrong.
pub(crate) fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| {
        let Ok(text) = std::env::var("VICTORY_AMBIENT") else {
            return Vec::new();
        };
        match parse(&text) {
            Ok(rules) => {
                for r in &rules {
                    tracing::info!(program = %r.program, view = %r.view(), value = ?r.value, "ambient stand-in set");
                }
                rules
            }
            Err(e) => {
                tracing::warn!(error = %e, "VICTORY_AMBIENT not understood: no stand-in");
                Vec::new()
            }
        }
    })
}

/// What the clear writes: floats for a float or normalized view, words for
/// an integer one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Fill {
    Float([f32; 4]),
    Uint([u32; 4]),
}

/// DXGI formats of integer views (UINT and SINT).
const INTEGER: [u32; 21] = [
    3, 4, 7, 8, 12, 14, 17, 18, 25, 30, 32, 36, 38, 42, 43, 50, 52, 57, 59, 62, 64,
];
/// DXGI formats of float and normalized views a float clear converts to.
const FLOAT: [u32; 24] = [
    2, 6, 10, 11, 13, 16, 24, 26, 28, 31, 34, 35, 37, 41, 49, 51, 54, 56, 58, 61, 63, 65, 87, 88,
];
const R9G9B9E5: u32 = 67;
const R11G11B10: u32 = 26;
const RGBA16F: u32 = 10;

/// The clear of `value` into a view of `view` format of a texture of
/// `texture` format (DXGI numbers, the title's), or why there is none.
pub(crate) fn fill(value: Value, view: u32, texture: u32) -> Result<Fill, &'static str> {
    if FLOAT.contains(&view) {
        return match value {
            Value::Colour(rgba, _) => Ok(Fill::Float(rgba)),
            Value::Raw(0) => Ok(Fill::Float([0.0; 4])),
            Value::Raw(_) => Err("a raw word other than zero for a view that is not of integers"),
        };
    }
    if !INTEGER.contains(&view) {
        return Err("a view format a clear does not take");
    }
    let rgba = match value {
        // The clear keeps each channel's low bits (all ones stays a mask).
        Value::Raw(word) => return Ok(Fill::Uint([word; 4])),
        Value::Colour(rgba, packing) => match packing.or(match texture {
            R9G9B9E5 => Some(Packing::Rgb9e5),
            R11G11B10 => Some(Packing::R11g11b10),
            RGBA16F => Some(Packing::Rgba16f),
            _ => None,
        }) {
            Some(packing) => (rgba, packing),
            None => {
                return Err(
                    "an integer view of a texture whose format does not say the packing: name it",
                );
            }
        },
    };
    let ([r, g, b, a], packing) = rgba;
    let half = |v: f32| small_float(v, 10);
    match (packing, view) {
        (Packing::Rgb9e5, 42 | 43) => Ok(Fill::Uint([rgb9e5([r, g, b]); 4])),
        (Packing::R11g11b10, 42 | 43) => Ok(Fill::Uint([r11g11b10([r, g, b]); 4])),
        // R16G16B16A16_UINT: a half per channel.
        (Packing::Rgba16f, 12) => Ok(Fill::Uint([half(r), half(g), half(b), half(a)])),
        // R32G32_UINT: two halves per word.
        (Packing::Rgba16f, 17) => Ok(Fill::Uint([
            half(r) | half(g) << 16,
            half(b) | half(a) << 16,
            0,
            0,
        ])),
        _ => Err("a packing whose size is not the view's"),
    }
}

/// R9G9B9E5_SHAREDEXP as the D3D specification packs it: a 5-bit shared
/// exponent (bias 15) over three 9-bit mantissas, channels clamped to
/// [0, 65408], rounded to nearest.
pub(crate) fn rgb9e5(rgb: [f32; 3]) -> u32 {
    const MAX: f64 = 65408.0;
    let c = rgb.map(|v| {
        if v.is_nan() {
            0.0
        } else {
            f64::from(v).clamp(0.0, MAX)
        }
    });
    let most = c[0].max(c[1]).max(c[2]);
    // floor(log2(most)), exact for the powers of two; -16 at least.
    let floor_log2 = if most > 0.0 {
        most.log2().floor() as i32
    } else {
        -16
    };
    let floor_log2 = match floor_log2 {
        // log2 may round up just under a power of two.
        e if 2f64.powi(e) > most => e - 1,
        e => e,
    };
    let mut exponent = floor_log2.max(-16) + 16;
    let mut scale = 2f64.powi(24 - exponent);
    if (most * scale + 0.5).floor() >= 512.0 {
        exponent += 1;
        scale *= 0.5;
    }
    let m = c.map(|v| ((v * scale + 0.5).floor() as u32).min(511));
    m[0] | m[1] << 9 | m[2] << 18 | (exponent as u32) << 27
}

/// R11G11B10_FLOAT: 11-, 11- and 10-bit floats without sign.
pub(crate) fn r11g11b10([r, g, b]: [f32; 3]) -> u32 {
    small_float(r, 6) | small_float(g, 6) << 11 | small_float(b, 5) << 22
}

/// `v` as a float without sign of a 5-bit exponent (bias 15) and
/// `mantissa` bits, rounded to nearest even (a half's bits for 10, its
/// sign 0). Negative and NaN give 0; too large and infinity the largest
/// finite value (shortcut: not infinity, which a stand-in light must not
/// be).
pub(crate) fn small_float(v: f32, mantissa: u32) -> u32 {
    let largest = (30 << mantissa) | ((1 << mantissa) - 1);
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    if exponent > 15 {
        return largest;
    }
    // The significand with its leading 1 (f32 subnormals are far below
    // the smallest of these floats: they round to 0), in units of
    // 2^(exponent - 23).
    let significand = u64::from(bits & 0x7f_ffff | 0x80_0000);
    // Bits to drop for units of the target's last place: 2^(e - mantissa)
    // for normals, 2^(-14 - mantissa) below.
    let shift = (23 - mantissa as i32 + (-14 - exponent).max(0)) as u32;
    if shift > 40 {
        return 0;
    }
    let mut out = significand >> shift;
    let rest = significand & ((1 << shift) - 1);
    let half = 1u64 << (shift - 1);
    if rest > half || (rest == half && out & 1 == 1) {
        out += 1;
    }
    // Normals: the leading 1 becomes the exponent field (a carry out of the
    // mantissa raises it, as it must).
    let out = if exponent >= -14 {
        out - (1 << mantissa) + (((exponent + 15) as u64) << mantissa)
    } else {
        out
    };
    (out as u32).min(largest)
}

static FILLED: AtomicU64 = AtomicU64::new(0);
static SKIPPED: AtomicU64 = AtomicU64::new(0);

// The stand-ins recorded on this thread (tests): tests run in parallel,
// each on its own thread, and record their stand-ins there.
#[cfg(test)]
thread_local! {
    static TEST_FILLED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// A stand-in recorded (`filled`) or not, logged every 5 s with the counts
/// so far.
pub(crate) fn note(filled: bool) {
    if filled {
        FILLED.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        TEST_FILLED.with(|n| n.set(n.get() + 1));
    } else {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
    }
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let now = Instant::now();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    match *last {
        Some(t) if now.duration_since(t) < Duration::from_secs(5) => {}
        Some(_) => {
            tracing::info!(
                filled = FILLED.load(Ordering::Relaxed),
                skipped = SKIPPED.load(Ordering::Relaxed),
                "ambient stand-in: views filled so far"
            );
            *last = Some(now);
        }
        None => *last = Some(now),
    }
}

/// How many stand-ins this thread recorded (tests): not the run's count,
/// which tests running alongside add to.
#[cfg(test)]
pub(crate) fn filled() -> u64 {
    TEST_FILLED.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference R9G9B9E5 encoder written another way: the smallest
    /// shared exponent whose rounded mantissas all fit in 9 bits.
    fn reference_rgb9e5(rgb: [f32; 3]) -> u32 {
        let c = rgb.map(|v| f64::from(v).clamp(0.0, 65408.0));
        for e in 0..32u32 {
            let unit = 2f64.powi(e as i32 - 24);
            let m = c.map(|v| (v / unit + 0.5).floor() as u32);
            if m.iter().all(|&m| m <= 511) {
                return m[0] | m[1] << 9 | m[2] << 18 | e << 27;
            }
        }
        unreachable!("65408 fits at exponent 31")
    }

    fn decode_rgb9e5(w: u32) -> [f64; 3] {
        let unit = 2f64.powi((w >> 27) as i32 - 24);
        [w & 511, (w >> 9) & 511, (w >> 18) & 511].map(|m| f64::from(m) * unit)
    }

    /// The code nearest `v` (ties to the even code), found by decoding
    /// every code of the small float.
    fn reference_small(v: f32, mantissa: u32) -> u32 {
        let decode = |code: u32| {
            let (e, m) = (
                (code >> mantissa) as i32,
                f64::from(code & ((1 << mantissa) - 1)),
            );
            let one = f64::from(1u32 << mantissa);
            if e == 0 {
                m / one * 2f64.powi(-14)
            } else {
                (1.0 + m / one) * 2f64.powi(e - 15)
            }
        };
        let largest = (30 << mantissa) | ((1 << mantissa) - 1);
        let v = f64::from(v).min(decode(largest));
        (0..=largest)
            .min_by(|&a, &b| {
                let (da, db) = ((decode(a) - v).abs(), (decode(b) - v).abs());
                da.partial_cmp(&db).unwrap().then((a & 1).cmp(&(b & 1)))
            })
            .unwrap()
    }

    #[test]
    fn rgb9e5_packs_as_the_reference_encoder() {
        // The sky's tables in the castable test: (1, 0.5, 0.25).
        assert_eq!(
            rgb9e5([1.0, 0.5, 0.25]),
            256 | 128 << 9 | 64 << 18 | 16 << 27
        );
        let colours = [
            [0.0, 0.0, 0.0],
            [0.6, 0.588, 0.564],
            [1.0, 1.0, 1.0],
            [0.25, 0.5, 4.0],
            [65408.0, 1.0, 0.0],
            [1e9, 0.0, 0.0],
            [511.75, 3.0, 0.001],
            [1.999, 0.01, 0.3],
            [1e-7, 2e-8, 0.0],
            [3.0e-5, 1.5e-5, 0.0],
        ];
        for c in colours {
            let w = rgb9e5(c);
            assert_eq!(w, reference_rgb9e5(c), "{c:?}");
            // Each channel within half a unit of the shared exponent.
            let unit = 2f64.powi((w >> 27) as i32 - 24);
            for (got, want) in decode_rgb9e5(w).iter().zip(c) {
                assert!(
                    (got - f64::from(want).min(65408.0)).abs() <= unit / 2.0,
                    "{c:?}: {got} for {want}"
                );
            }
        }
    }

    #[test]
    fn small_floats_round_to_the_nearest_code() {
        assert_eq!(small_float(1.0, 10), 0x3c00);
        assert_eq!(small_float(0.6, 10), 0x38cd);
        assert_eq!(small_float(65504.0, 10), 0x7bff);
        assert_eq!(small_float(1e9, 10), 0x7bff);
        assert_eq!(small_float(f32::INFINITY, 10), 0x7bff);
        assert_eq!(small_float(-1.0, 10), 0);
        assert_eq!(small_float(2f32.powi(-24), 10), 1);
        assert_eq!(small_float(1.0, 6), 15 << 6);
        assert_eq!(small_float(1.0, 5), 15 << 5);
        for v in [
            0.6f32,
            0.588,
            0.1,
            3.3,
            1000.5,
            6.1e-5,
            2.0e-6,
            1.0e-9,
            0.99951,
            2047.9,
            1.0 + 1.0 / 2048.0,
        ] {
            for m in [5, 6, 10] {
                assert_eq!(small_float(v, m), reference_small(v, m), "{v} in {m} bits");
            }
        }
        assert_eq!(
            r11g11b10([1.0, 1.0, 1.0]),
            15 << 6 | (15 << 6) << 11 | (15 << 5) << 22
        );
    }

    #[test]
    fn rules_parse_and_refuse() {
        let rules =
            parse(" 0123456789ABCDEF:s1+176:rgb9e5=0.6,0.5,0.4 ; fedcba9876543210:2+8:0x3c00 ;")
                .unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(
            (rules[0].program.as_str(), rules[0].slot, rules[0].offset),
            ("0123456789abcdef", 1, 176)
        );
        assert_eq!(
            rules[0].value,
            Value::Colour([0.6, 0.5, 0.4, 1.0], Some(Packing::Rgb9e5))
        );
        assert_eq!(rules[0].view(), "s1+176");
        assert_eq!(rules[1].value, Value::Raw(0x3c00));
        assert_eq!(
            parse("0123456789abcdef:0+0:1,2,3,0.5").unwrap()[0].value,
            Value::Colour([1.0, 2.0, 3.0, 0.5], None)
        );
        assert!(parse("").unwrap().is_empty());
        assert_eq!(
            parse("0123456789abcdef:1+136:mask").unwrap()[0].value,
            Value::Raw(u32::MAX)
        );
        assert_eq!(
            parse("0123456789abcdef:1+192:ZERO").unwrap()[0].value,
            Value::Raw(0)
        );
        for wrong in [
            "0123456789abcde:1+176:1,1,1",
            "0123456789abcdeg:1+176:1,1,1",
            "0123456789abcdef:1:1,1,1",
            "0123456789abcdef:1+x:1,1,1",
            "0123456789abcdef:1+176:1,1",
            "0123456789abcdef:1+176:-1,1,1",
            "0123456789abcdef:1+176:nan,1,1",
            "0123456789abcdef:1+176:rgb10=1,1,1",
            "0123456789abcdef:1+176",
        ] {
            assert!(parse(wrong).is_err(), "{wrong}");
        }
    }

    #[test]
    fn fills_follow_the_view_and_texture_formats() {
        let grey = Value::Colour([0.5, 0.5, 0.5, 1.0], None);
        let packed = |p| Value::Colour([0.5, 0.5, 0.5, 1.0], Some(p));
        assert_eq!(
            fill(grey, RGBA16F, RGBA16F),
            Ok(Fill::Float([0.5, 0.5, 0.5, 1.0]))
        );
        assert_eq!(
            fill(grey, R11G11B10, R11G11B10),
            Ok(Fill::Float([0.5, 0.5, 0.5, 1.0]))
        );
        // R32_UINT of an R9G9B9E5 texture: packed by the texture's format.
        assert_eq!(
            fill(grey, 42, R9G9B9E5),
            Ok(Fill::Uint([rgb9e5([0.5; 3]); 4]))
        );
        assert_eq!(
            fill(grey, 42, R11G11B10),
            Ok(Fill::Uint([r11g11b10([0.5; 3]); 4]))
        );
        // R32_UINT of an R32_UINT texture: only with the packing named.
        assert!(fill(grey, 42, 42).is_err());
        assert_eq!(
            fill(packed(Packing::Rgb9e5), 42, 42),
            Ok(Fill::Uint([rgb9e5([0.5; 3]); 4]))
        );
        assert_eq!(fill(Value::Raw(7), 42, 42), Ok(Fill::Uint([7; 4])));
        assert!(fill(Value::Raw(7), RGBA16F, RGBA16F).is_err());
        // Masks and zeros for integer views of any width; zero for floats.
        assert_eq!(
            fill(Value::Raw(u32::MAX), 62, 62),
            Ok(Fill::Uint([u32::MAX; 4]))
        );
        assert_eq!(fill(Value::Raw(0), 57, 57), Ok(Fill::Uint([0; 4])));
        assert_eq!(fill(Value::Raw(0), 54, 54), Ok(Fill::Float([0.0; 4])));
        assert_eq!(
            fill(grey, 12, RGBA16F),
            Ok(Fill::Uint([0x3800, 0x3800, 0x3800, 0x3c00]))
        );
        assert_eq!(
            fill(grey, 17, RGBA16F),
            Ok(Fill::Uint([0x3800_3800, 0x3c00_3800, 0, 0]))
        );
        assert!(fill(packed(Packing::Rgba16f), 42, 42).is_err());
        assert!(fill(grey, R9G9B9E5, R9G9B9E5).is_err());
    }
}
