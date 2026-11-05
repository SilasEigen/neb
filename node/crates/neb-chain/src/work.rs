//! Cumulative chain work (spec/node.md section 6.5): the expected number
//! of proof-of-work attempts one block costs at its `bits`,
//! `floor(2^256 / (target + 1))`, and the sum of those values along a chain.
//! Chain selection compares branches by cumulative work, not by height.
//! The arithmetic is exact integer arithmetic on four u64 limbs.

use neb_block::{Target, TargetError};

/// A 256-bit unsigned integer, four little-endian u64 limbs; one unit is one
/// expected attempt. The smallest target is 2^23, so one block is worth at
/// most about 2^233 and the sum overflows 2^256 only after 2^23 blocks of
/// that difficulty, which no chain reaches (hence `Add` does not check).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Hash)]
pub struct Work(pub [u64; 4]);

impl Work {
    pub const ZERO: Work = Work([0; 4]);

    /// The work of one block at `bits` (spec/node.md section 6.5):
    /// `floor(2^256 / (target + 1))`, computed exactly. Errors when `bits` is
    /// not a normalised compact target.
    pub fn of_bits(bits: u32) -> Result<Work, TargetError> {
        let target = Target::from_bits(bits)?;
        // Divisor = target + 1 in five limbs. The target is at most
        // 0xFFFFFF << 232 = 2^256 - 2^232, so target + 1 < 2^256: the carry
        // out of the fourth limb never happens and the fifth limb stays zero.
        let mut divisor = [0u64; 5];
        for (i, limb) in divisor.iter_mut().take(4).enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&target.0[8 * i..8 * i + 8]);
            *limb = u64::from_le_bytes(b);
        }
        let mut carry = true;
        for limb in divisor.iter_mut().take(4) {
            let (v, c) = limb.overflowing_add(carry as u64);
            *limb = v;
            carry = c;
        }
        debug_assert!(!carry);
        // Binary long division of 2^256, one dividend bit per step from bit
        // 255 down to bit 0. The only set bit of 2^256 is bit 256, which is
        // consumed by the initial remainder 1; the remainder stays below the
        // divisor, so it fits five limbs after the shift. The quotient is
        // below 2^256 because the divisor is at least 2^23 + 1.
        let mut rem = [1u64, 0, 0, 0, 0];
        let mut quotient = [0u64; 4];
        for i in (0..256).rev() {
            let mut shifted_out = 0u64;
            for limb in rem.iter_mut() {
                let next_out = *limb >> 63;
                *limb = (*limb << 1) | shifted_out;
                shifted_out = next_out;
            }
            if cmp5(&rem, &divisor).is_ge() {
                sub5(&mut rem, &divisor);
                quotient[i / 64] |= 1u64 << (i % 64);
            }
        }
        Ok(Work(quotient))
    }

    /// 32 bytes, little-endian: the encoding in the index of `neb-store`.
    pub fn to_le_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, limb) in self.0.iter().enumerate() {
            out[8 * i..8 * i + 8].copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    pub fn from_le_bytes(bytes: &[u8; 32]) -> Work {
        let mut limbs = [0u64; 4];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[8 * i..8 * i + 8]);
            *limb = u64::from_le_bytes(b);
        }
        Work(limbs)
    }
}

impl std::ops::Add for Work {
    type Output = Work;

    /// Addition modulo 2^256; the wrap point is out of reach (see the type).
    fn add(self, other: Work) -> Work {
        let mut out = [0u64; 4];
        let mut carry = 0u64;
        for ((o, a), b) in out.iter_mut().zip(self.0).zip(other.0) {
            let (v, c1) = a.overflowing_add(b);
            let (v, c2) = v.overflowing_add(carry);
            *o = v;
            carry = u64::from(c1 || c2);
        }
        Work(out)
    }
}

impl Ord for Work {
    fn cmp(&self, other: &Work) -> std::cmp::Ordering {
        for i in (0..4).rev() {
            if self.0[i] != other.0[i] {
                return self.0[i].cmp(&other.0[i]);
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl PartialOrd for Work {
    fn partial_cmp(&self, other: &Work) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Compares two 320-bit values given as five little-endian limbs.
fn cmp5(a: &[u64; 5], b: &[u64; 5]) -> std::cmp::Ordering {
    for i in (0..5).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    std::cmp::Ordering::Equal
}

/// `a -= b` over five limbs; the caller guarantees `a >= b`.
fn sub5(a: &mut [u64; 5], b: &[u64; 5]) {
    let mut borrow = 0u64;
    for i in 0..5 {
        let (v1, b1) = a[i].overflowing_sub(b[i]);
        let (v2, b2) = v1.overflowing_sub(borrow);
        a[i] = v2;
        borrow = (b1 as u64) | (b2 as u64);
    }
    debug_assert_eq!(borrow, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limbs(v: [u64; 4]) -> Work {
        Work(v)
    }

    // Exact floor values, including the limb-spanning case and the invalid bits.
    #[test]
    fn known_work_values() {
        // For example, 2^232: floor(2^256 / (2^232 + 1)) = 2^24 - 1
        assert_eq!(Work::of_bits(0xD180_0000), Ok(limbs([0x00FF_FFFF, 0, 0, 0])));
        // 2^217: floor(2^256 / (2^217 + 1)) = 2^39 - 1
        assert_eq!(Work::of_bits(0xC280_0000), Ok(limbs([0x7F_FFFF_FFFF, 0, 0, 0])));
        // 2^216: 2^40 - 1
        assert_eq!(Work::of_bits(0xC180_0000), Ok(limbs([0xFF_FFFF_FFFF, 0, 0, 0])));
        // The largest target the format can express (2^256 - 2^232): work 1.
        assert_eq!(Work::of_bits(0xE8FF_FFFF), Ok(limbs([1, 0, 0, 0])));
        // The smallest target (2^23) gives a value spanning all four limbs;
        // equal to `(1 << 256) // ((1 << 23) + 1)` in Python.
        let small = Work::of_bits(0x0080_0000).expect("valid bits");
        assert_eq!(
            small.0,
            [0x0001_FFFF_FC00_0007, 0xFFC0_0000_7FFF_FF00, 0x07FF_FFF0_0000_1FFF, 0x0000_01FF_FFFC_0000]
        );
        assert_eq!(
            small.to_le_bytes(),
            [
                0x07, 0x00, 0x00, 0xFC, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0xFF, 0xFF, 0x7F, 0x00, 0x00, 0xC0, 0xFF, 0xFF,
                0x1F, 0x00, 0x00, 0xF0, 0xFF, 0xFF, 0x07, 0x00, 0x00, 0xFC, 0xFF, 0xFF, 0x01, 0x00, 0x00
            ]
        );
        assert!(Work::of_bits(0xC200_0000).is_err(), "mantissa below 0x800000");
        assert!(Work::of_bits(0xE900_0000).is_err(), "shift above 232");
    }

    // Chain selection relies on work growing as the target shrinks.
    #[test]
    fn work_is_monotonic_in_difficulty() {
        // A smaller target is harder and must carry more work.
        let mut previous = Work::ZERO;
        for shift in (0..=232).rev().step_by(7) {
            let w = Work::of_bits((shift << 24) | 0x80_0000).expect("valid bits");
            assert!(w > previous, "shift {shift}");
            previous = w;
        }
    }

    // Byte encoding used by the index round-trips; addition carries between limbs.
    #[test]
    fn byte_roundtrip_and_addition() {
        let a = Work::of_bits(0xD180_0000).unwrap();
        let b = Work::of_bits(0xC280_0000).unwrap();
        let bytes = b.to_le_bytes();
        assert_eq!(Work::from_le_bytes(&bytes), b);
        let sum = a + b;
        // 0xFFFFFF + 0x7FFFFFFFFF = 0x8000FFFFFE, no carry between limbs.
        assert_eq!(sum, limbs([0x80_00FF_FFFE, 0, 0, 0]));
        // Carries propagate across limbs; the sum wraps at 2^256.
        let top = limbs([u64::MAX, u64::MAX, u64::MAX, u64::MAX]);
        let one = limbs([1, 0, 0, 0]);
        assert_eq!(top + one, limbs([0, 0, 0, 0]));
        assert!(top > one);
        assert_eq!(Work::ZERO.to_le_bytes(), [0u8; 32]);
    }
}
