//! ASERT difficulty rule (spec/node.md section 6.2).
//!
//! Integer implementation of the aserti3-2d algorithm (absolutely scheduled
//! exponentially rising targets, designed by Mark Lundeberg); the reference
//! runs it is checked against are in `node/vectors/asert_reference_runs.txt`.
//! The target is an exponential of the schedule drift: it halves when the chain is one
//! half-life (600 s) ahead of the 60 s schedule and doubles when it is one
//! half-life behind. The anchor is the genesis block: its `bits` is the anchor
//! target and its timestamp the time base. The genesis block has no parent,
//! so the block at height h has its parent at h - 1 and the schedule counts
//! h - 1 spacings (see `Params::next_bits`).
//!
//! The rule is consensus-critical and must be bit-exact on every node: it uses
//! integers only, no floating point and no clock. The product of the anchor
//! target and the factor is computed in 320 bits, so the arithmetic is exact
//! for every compact target; below 2^239 it equals the reference arithmetic.
//! Precondition carried over from the reference: the anchor target is at most
//! the pow limit, and abs(time_diff - spacing * height_diff) < 2^47, which
//! keeps the 16.16 fixed-point exponent inside an i64.

use neb_block::{Target, TargetError};

/// Target block spacing in seconds.
pub const SPACING: i64 = 60;
/// Half-life of the exponential in seconds (10 blocks at the target spacing).
pub const HALF_LIFE: i64 = 600;

/// The smallest target the NEB compact form can express (mantissa 0x800000).
pub const MIN_TARGET_BITS: u32 = 0x0080_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifficultyError {
    /// The anchor or pow limit is not a normalised compact target.
    BadBits(TargetError),
    /// The anchor target is easier than the pow limit.
    AnchorAboveLimit,
}

impl std::fmt::Display for DifficultyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DifficultyError::BadBits(e) => write!(f, "bad compact target: {e}"),
            DifficultyError::AnchorAboveLimit => write!(f, "anchor target above the pow limit"),
        }
    }
}

impl std::error::Error for DifficultyError {}

/// A 256-bit unsigned value as four little-endian u64 limbs, the width of a
/// target. Products that can exceed 256 bits go through `mul_u64_wide`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct U256([u64; 4]);

impl U256 {
    const ZERO: U256 = U256([0; 4]);
    const ONE: U256 = U256([1, 0, 0, 0]);

    fn from_target(t: &Target) -> U256 {
        let mut limbs = [0u64; 4];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&t.0[i * 8..i * 8 + 8]);
            *limb = u64::from_le_bytes(b);
        }
        U256(limbs)
    }

    fn to_target(self) -> Target {
        let mut out = [0u8; 32];
        for (i, limb) in self.0.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
        }
        Target(out)
    }

    fn is_zero(&self) -> bool {
        self.0 == [0; 4]
    }

    fn bit_len(&self) -> u32 {
        for i in (0..4).rev() {
            if self.0[i] != 0 {
                return 64 * i as u32 + (64 - self.0[i].leading_zeros());
            }
        }
        0
    }

    fn gt(&self, other: &U256) -> bool {
        for i in (0..4).rev() {
            if self.0[i] != other.0[i] {
                return self.0[i] > other.0[i];
            }
        }
        false
    }

    /// The product with `f` as the low 256 bits and the limb above them
    /// (320 bits in total; the product is below 2^320 because f < 2^64).
    fn mul_u64_wide(&self, f: u64) -> (U256, u64) {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for (o, &a) in out.iter_mut().zip(&self.0) {
            let p = a as u128 * f as u128 + carry;
            *o = p as u64;
            carry = p >> 64;
        }
        (U256(out), carry as u64)
    }

    /// Shift left, or None when any bit would leave the 256-bit range.
    fn shl_checked(&self, n: u64) -> Option<U256> {
        if self.is_zero() {
            return Some(U256::ZERO);
        }
        if self.bit_len() as u64 + n > 256 {
            return None;
        }
        let limb = (n / 64) as usize;
        let bit = (n % 64) as u32;
        let mut out = [0u64; 4];
        if bit == 0 {
            for i in (0..4 - limb).rev() {
                out[i + limb] = self.0[i];
            }
        } else {
            for i in (0..4 - limb).rev() {
                let mut v = self.0[i] << bit;
                if i > 0 {
                    v |= self.0[i - 1] >> (64 - bit);
                }
                out[i + limb] = v;
            }
        }
        Some(U256(out))
    }

    /// `(high:self) >> n` for a 320-bit value, or None when the result does
    /// not fit 256 bits.
    fn shr_wide(&self, high: u64, n: u64) -> Option<U256> {
        if n >= 320 {
            return Some(U256::ZERO);
        }
        let limbs = [self.0[0], self.0[1], self.0[2], self.0[3], high];
        let (limb, bit) = ((n / 64) as usize, (n % 64) as u32);
        let mut out = [0u64; 5];
        for i in 0..5 - limb {
            let mut v = limbs[i + limb] >> bit;
            if bit > 0 && i + limb + 1 < 5 {
                v |= limbs[i + limb + 1] << (64 - bit);
            }
            out[i] = v;
        }
        if out[4] != 0 {
            return None;
        }
        Some(U256([out[0], out[1], out[2], out[3]]))
    }

    /// Logical shift right; shifts beyond 256 bits yield zero. Used only by
    /// the test transcription of the compact conversion of the reference runs.
    #[cfg(test)]
    fn shr(&self, n: u64) -> U256 {
        if n >= 256 {
            return U256::ZERO;
        }
        let limb = (n / 64) as usize;
        let bit = (n % 64) as u32;
        let mut out = [0u64; 4];
        if bit == 0 {
            out[..4 - limb].copy_from_slice(&self.0[limb..]);
        } else {
            for (i, o) in out[..4 - limb].iter_mut().enumerate() {
                let mut v = self.0[i + limb] >> bit;
                if i + limb + 1 < 4 {
                    v |= self.0[i + limb + 1] << (64 - bit);
                }
                *o = v;
            }
        }
        U256(out)
    }
}

/// The reference form of the algorithm: the next target from the anchor target.
///
/// `height_diff` is the parent height minus the anchor height, `time_diff`
/// the parent time minus the anchor time base (both in the reference's
/// convention, where the anchor is itself a block with a parent). The result
/// is clamped to `[1, pow_limit]`. Used with the reference parameters to
/// replay the reference runs; the chain calls `Params::next_bits`.
pub fn core_next_target(
    ref_target: &Target,
    spacing: i64,
    half_life: i64,
    time_diff: i64,
    height_diff: i64,
    pow_limit: &Target,
) -> Target {
    debug_assert!(height_diff >= 0);
    debug_assert!((time_diff - spacing * height_diff).abs() < (1 << 47));
    let drift = time_diff as i128 - spacing as i128 * (height_diff as i128 + 1);
    core_from_drift(ref_target, half_life, drift, pow_limit)
}

/// The exponent, shift and factor part of the algorithm, from the drift in
/// seconds: `time_diff - spacing * (height_diff + 1)` for the reference shape,
/// or `time_diff - spacing * height_diff` for the genesis anchor (the genesis
/// block consumed no schedule slot). Result = target * 2^(drift / half_life).
fn core_from_drift(ref_target: &Target, half_life: i64, drift: i128, pow_limit: &Target) -> Target {
    let t = U256::from_target(ref_target);
    let pl = U256::from_target(pow_limit);

    // Exponent in 16.16 fixed point; the division truncates toward zero as in
    // the reference. The bound on |drift| keeps it inside an i64.
    let exponent = (drift * 65536) / half_life as i128;
    let exponent: i64 = exponent.try_into().expect("precondition bounds the exponent");

    // Integer part (arithmetic shift rounds toward minus infinity) and
    // fraction in [0, 65536), so frac >= 0 for negative exponents too.
    let mut shifts = exponent >> 16;
    let frac = (exponent & 0xFFFF) as u128;
    // Cubic approximation of 2^(frac/65536) - 1 scaled by 2^16: the
    // coefficients are those of the reference, in 48-bit fixed point, and
    // (1 << 47) rounds the final shift to nearest.
    let poly =
        (195766423245049u128 * frac + 971821376u128 * frac * frac + 5127u128 * frac * frac * frac + (1u128 << 47))
            >> 48;
    let factor = 65536u64 + poly as u64;

    // factor is in [2^16, 2^17), so factor / 2^16 is the multiplier in [1, 2).
    // A target below 2^256 times a 17-bit factor needs up to 273 bits, which
    // fits the 320-bit product; a result that does not fit 256 bits after the
    // shift is above any pow limit and clamps to it.
    let (low, high) = t.mul_u64_wide(factor);
    // Remove the 2^16 scale of the factor.
    shifts -= 16;
    let mut next = if shifts <= 0 {
        low.shr_wide(high, -shifts as u64).unwrap_or(pl)
    } else if high != 0 {
        pl
    } else {
        low.shl_checked(shifts as u64).unwrap_or(pl)
    };
    // The reference clamps a zero result to 1 and a result above the limit
    // to the limit.
    if next.is_zero() {
        next = U256::ONE;
    } else if next.gt(&pl) {
        next = pl;
    }
    next.to_target()
}

/// Parameters of the difficulty rule. They come from the network parameters
/// (spec/node.md section 6.6); all nodes of a network must agree on them.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub spacing: i64,
    pub half_life: i64,
    /// Compact form of the easiest allowed target.
    pub pow_limit_bits: u32,
    /// Compact target and timestamp of the genesis block, the schedule anchor.
    pub anchor_bits: u32,
    pub anchor_time: u32,
}

impl Params {
    /// The compact target of the block whose parent is at `parent_height`
    /// with timestamp `parent_time` (genesis is height 0). Errors only on
    /// invalid parameters; any heights and times are accepted.
    pub fn next_bits(&self, parent_height: u32, parent_time: u32) -> Result<u32, DifficultyError> {
        let anchor = Target::from_bits(self.anchor_bits).map_err(DifficultyError::BadBits)?;
        let limit = Target::from_bits(self.pow_limit_bits).map_err(DifficultyError::BadBits)?;
        let anchor_u = U256::from_target(&anchor);
        let limit_u = U256::from_target(&limit);
        if anchor_u.gt(&limit_u) {
            return Err(DifficultyError::AnchorAboveLimit);
        }
        // The genesis block has no parent: the schedule counts parent_height
        // spacings, not parent_height + 1 (spec/node.md section 6.2).
        let drift = (parent_time as i128 - self.anchor_time as i128) - self.spacing as i128 * parent_height as i128;
        let target = core_from_drift(&anchor, self.half_life, drift, &limit);
        // Targets below 0x800000 have no normalised compact form, so they
        // clamp up to it.
        let min = Target::from_bits(MIN_TARGET_BITS).expect("constant target");
        let target = if U256::from_target(&target).gt(&U256::from_target(&min)) { target } else { min };
        target.to_bits().map_err(DifficultyError::BadBits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u256_from_u64(v: u64) -> U256 {
        U256([v, 0, 0, 0])
    }

    // Compact conversions of the reference runs (their pow limit is 0x1d00ffff),
    // to replay those runs.

    const REFERENCE_MAX_BITS: u32 = 0x1D00_FFFF;

    fn reference_bits_to_target(bits: u32) -> U256 {
        let size = bits >> 24;
        assert!(size <= 0x1D);
        let word = bits & 0x00FF_FFFF;
        assert!((0x8000..=0x7F_FFFF).contains(&word));
        if size <= 3 {
            u256_from_u64((word >> (8 * (3 - size))) as u64)
        } else {
            u256_from_u64(word as u64).shl_checked((8 * (size - 3)) as u64).unwrap()
        }
    }

    fn reference_target_to_bits(target: &U256) -> u32 {
        let max = reference_bits_to_target(REFERENCE_MAX_BITS);
        let mut t = *target;
        assert!(!t.is_zero());
        if t.gt(&max) {
            t = max;
        }
        let size = t.bit_len().div_ceil(8);
        let mut compact = if size <= 3 {
            t.shl_checked((8 * (3 - size)) as u64).unwrap().0[0]
        } else {
            t.shr((8 * (size - 3)) as u64).0[0]
        };
        let mut size = size;
        if compact & 0x0080_0000 != 0 {
            compact >>= 8;
            size += 1;
        }
        assert!(compact == (compact & 0x007F_FFFF));
        assert!(size < 256);
        compact as u32 | (size << 24)
    }

    #[derive(Default)]
    struct RunHeader {
        anchor_bits: u32,
        anchor_height: i64,
        anchor_time: i64,
        genesis_bits: Option<u32>,
        genesis_time: Option<u32>,
        pow_limit_bits: Option<u32>,
    }

    /// One line of a run, as in the vector file: iteration, height, time and target bits.
    type Line = (u64, u64, u64, u32);
    type Run = (RunHeader, Vec<Line>);

    fn parse_vectors(text: &str) -> Vec<Run> {
        let mut runs = Vec::new();
        let mut header: Option<RunHeader> = None;
        let mut lines: Vec<Line> = Vec::new();
        let flush = |header: &mut Option<RunHeader>, lines: &mut Vec<Line>, runs: &mut Vec<Run>| {
            if let Some(h) = header.take() {
                runs.push((h, std::mem::take(lines)));
            }
        };
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("## ") {
                let rest = rest.trim_start();
                if rest.starts_with("description:") {
                    flush(&mut header, &mut lines, &mut runs);
                    header = Some(RunHeader::default());
                    continue;
                }
                if header.is_none() {
                    header = Some(RunHeader::default());
                }
                let h = header.as_mut().unwrap();
                if let Some(v) = rest.strip_prefix("anchor nBits: ") {
                    h.anchor_bits = u32::from_str_radix(v.trim_start_matches("0x"), 16).unwrap();
                } else if let Some(v) = rest.strip_prefix("anchor height: ") {
                    h.anchor_height = v.parse().unwrap();
                } else if let Some(v) = rest.strip_prefix("anchor parent time: ") {
                    h.anchor_time = v.parse().unwrap();
                } else if let Some(v) = rest.strip_prefix("genesis bits: ") {
                    h.genesis_bits = Some(u32::from_str_radix(v.trim_start_matches("0x"), 16).unwrap());
                } else if let Some(v) = rest.strip_prefix("genesis time: ") {
                    h.genesis_time = Some(v.parse().unwrap());
                } else if let Some(v) = rest.strip_prefix("pow limit bits: ") {
                    h.pow_limit_bits = Some(u32::from_str_radix(v.trim_start_matches("0x"), 16).unwrap());
                }
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 4
                && let [i, h, t, bits] = parts[..]
            {
                lines.push((
                    i.parse().unwrap(),
                    h.parse().unwrap(),
                    t.parse().unwrap(),
                    u32::from_str_radix(bits.trim_start_matches("0x"), 16).unwrap(),
                ));
            }
        }
        flush(&mut header, &mut lines, &mut runs);
        runs
    }

    // Bit-exactness against the reference runs: every line of their vector file
    // (half-life 2 days, spacing 600 s) must give the same compact target.
    #[test]
    fn matches_the_reference_runs() {
        let text = include_str!("../../../vectors/asert_reference_runs.txt");
        let runs = parse_vectors(text);
        assert_eq!(runs.len(), 8, "the eight deterministic runs");
        let pow_limit = reference_bits_to_target(REFERENCE_MAX_BITS);
        let mut checked = 0usize;
        for (h, lines) in &runs {
            let anchor = reference_bits_to_target(h.anchor_bits);
            for (_i, height, time, want) in lines {
                let target = core_next_target(
                    &anchor.to_target(),
                    600,
                    2 * 24 * 3600,
                    (*time).wrapping_sub(h.anchor_time as u64) as i64,
                    (*height).wrapping_sub(h.anchor_height as u64) as i64,
                    &pow_limit.to_target(),
                );
                let got = reference_target_to_bits(&U256::from_target(&target));
                assert_eq!(got, *want, "height {height} time {time}");
                checked += 1;
            }
        }
        assert!(checked > 10_000, "all vector lines were replayed ({checked})");
    }

    // Bit-exactness of the genesis-anchored rule against vectors/neb_asert_s0.txt.
    #[test]
    fn matches_the_neb_vector_set() {
        let text = include_str!("../../../vectors/neb_asert_s0.txt");
        let runs = parse_vectors(text);
        assert_eq!(runs.len(), 7, "the seven NEB runs");
        let mut checked = 0usize;
        for (h, lines) in &runs {
            let params = Params {
                spacing: SPACING,
                half_life: HALF_LIFE,
                pow_limit_bits: h.pow_limit_bits.expect("neb header"),
                anchor_bits: h.genesis_bits.expect("neb header"),
                anchor_time: h.genesis_time.expect("neb header"),
            };
            for (_i, height, time, want) in lines {
                // Each vector line gives the parent: (height, time).
                let got = params.next_bits(*height as u32, *time as u32).unwrap();
                assert_eq!(got, *want, "height {height} time {time}");
                checked += 1;
            }
        }
        assert_eq!(checked, 89, "all NEB vector lines were replayed");
    }

    // Blocks exactly on schedule leave the target unchanged.
    #[test]
    fn steady_blocks_keep_the_target() {
        let params = Params {
            spacing: SPACING,
            half_life: HALF_LIFE,
            pow_limit_bits: 0xC280_0000,
            anchor_bits: 0xBB80_0000,
            anchor_time: 1_800_000_000,
        };
        for h in 1u32..40 {
            let t = 1_800_000_000u32 + 60 * (h - 1);
            assert_eq!(params.next_bits(h - 1, t).unwrap(), 0xBB80_0000, "height {h}");
        }
    }

    // Clamp to the compact-form floor and to the pow limit.
    #[test]
    fn clamps_and_floor() {
        let params = Params {
            spacing: SPACING,
            half_life: HALF_LIFE,
            pow_limit_bits: 0xC280_0000,
            anchor_bits: 0xBB80_0000,
            anchor_time: 1_800_000_000,
        };
        // Far ahead of schedule: the target would fall below the format floor.
        assert_eq!(params.next_bits(0, 1_800_000_000 - 1_000_000).unwrap(), MIN_TARGET_BITS);
        // A long stall: the target would exceed the pow limit.
        assert_eq!(params.next_bits(0, 1_800_000_000 + 86_400).unwrap(), 0xC280_0000);
    }

    // Above 2^239 the 320-bit product keeps the result exact: one half-life of
    // drift halves or doubles the target.
    #[test]
    fn easy_targets_above_2_pow_239_are_exact_too() {
        // 0xE880_0000 is 2^255, the easiest target of the compact form.
        let params = Params {
            spacing: SPACING,
            half_life: HALF_LIFE,
            pow_limit_bits: 0xE880_0000,
            anchor_bits: 0xE600_0000 | 0x0080_0000, // 2^253
            anchor_time: 1_800_000_000,
        };
        let anchor = params.anchor_bits;
        for h in 1u32..40 {
            let t = 1_800_000_000u32 + 60 * (h - 1);
            assert_eq!(params.next_bits(h - 1, t).unwrap(), anchor, "height {h}");
        }
        // A stall of a day clamps to the limit; a parent a half-life ahead of
        // schedule halves the target.
        assert_eq!(params.next_bits(0, 1_800_000_000 + 86_400).unwrap(), 0xE880_0000);
        let fast = params.next_bits(5, 1_800_000_000 + 300 - 600).unwrap();
        assert_eq!(Target::from_bits(fast).unwrap().shifted_left(1), Some(Target::from_bits(anchor).unwrap()));
        // A parent a half-life behind schedule doubles the target.
        assert_eq!(params.next_bits(5, 1_800_000_000 + 300 + 600).unwrap(), 0xE780_0000);
    }

    // Invalid launch parameters are errors, not clamped.
    #[test]
    fn rejects_bad_parameters() {
        let bad_bits = Params {
            spacing: SPACING,
            half_life: HALF_LIFE,
            pow_limit_bits: 0xC280_0000,
            anchor_bits: 0x007F_FFFF,
            anchor_time: 0,
        };
        assert!(matches!(bad_bits.next_bits(0, 0), Err(DifficultyError::BadBits(_))));
        let anchor_above_limit = Params {
            spacing: SPACING,
            half_life: HALF_LIFE,
            pow_limit_bits: 0xBB80_0000,
            anchor_bits: 0xC280_0000,
            anchor_time: 0,
        };
        assert_eq!(anchor_above_limit.next_bits(0, 60), Err(DifficultyError::AnchorAboveLimit));
        let bad_anchor = Params {
            spacing: SPACING,
            half_life: HALF_LIFE,
            pow_limit_bits: 0x0000_0001u32, // mantissa below the minimum
            anchor_bits: 0xBB80_0000,
            anchor_time: 0,
        };
        assert!(matches!(bad_anchor.next_bits(0, 60), Err(DifficultyError::BadBits(_))));
    }
}
