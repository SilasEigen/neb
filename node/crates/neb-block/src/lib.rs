//! Block header and compact target (spec/node.md section 4.1).
//!
//! The header is 80 bytes with little-endian fields at fixed offsets, so its
//! encoding is canonical by construction. The block id is BLAKE3(header),
//! the same digest the proof of work uses as its key (spec/shares.md section
//! 11.1). This crate checks only the shape of a header and a target; the
//! rules that need chain context (parent, time, bits of the difficulty
//! algorithm) belong to neb-chain.

/// Header size in bytes: 4 + 32 + 32 + 4 + 4 + 4.
pub const HEADER_SIZE: usize = 80;

/// A block header. The proof of work hashes exactly these 80 bytes with the
/// nonce at offsets 76..80, so field order and offsets cannot change.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Header {
    pub version: u32,
    pub parent: [u8; 32],
    pub body_root: [u8; 32],
    pub time: u32,
    pub bits: u32,
    pub nonce: u32,
}

impl Header {
    /// Layout: version 0..4, parent 4..36, body_root 36..68, time 68..72,
    /// bits 72..76, nonce 76..80; integers little-endian.
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut out = [0u8; HEADER_SIZE];
        out[0..4].copy_from_slice(&self.version.to_le_bytes());
        out[4..36].copy_from_slice(&self.parent);
        out[36..68].copy_from_slice(&self.body_root);
        out[68..72].copy_from_slice(&self.time.to_le_bytes());
        out[72..76].copy_from_slice(&self.bits.to_le_bytes());
        out[76..80].copy_from_slice(&self.nonce.to_le_bytes());
        out
    }

    pub fn decode(buf: &[u8; HEADER_SIZE]) -> Header {
        let mut parent = [0u8; 32];
        parent.copy_from_slice(&buf[4..36]);
        let mut body_root = [0u8; 32];
        body_root.copy_from_slice(&buf[36..68]);
        Header {
            version: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            parent,
            body_root,
            time: u32::from_le_bytes(buf[68..72].try_into().unwrap()),
            bits: u32::from_le_bytes(buf[72..76].try_into().unwrap()),
            nonce: u32::from_le_bytes(buf[76..80].try_into().unwrap()),
        }
    }

    /// Block id: BLAKE3 of the 80 encoded bytes (spec/node.md section 4.1).
    pub fn id(&self) -> [u8; 32] {
        *blake3::hash(&self.encode()).as_bytes()
    }
}

impl From<[u8; HEADER_SIZE]> for Header {
    fn from(buf: [u8; HEADER_SIZE]) -> Header {
        Header::decode(&buf)
    }
}

impl From<Header> for [u8; HEADER_SIZE] {
    fn from(h: Header) -> [u8; HEADER_SIZE] {
        h.encode()
    }
}

/// `bits = (shift << 24) | mantissa`, `shift` in 0..=232. The largest target
/// is 0xFFFFFF << 232, which has 256 bits.
pub const SHIFT_MAX: u32 = 232;
/// `mantissa` in 0x800000..=0xFFFFFF (top bit of the 24 set). With this range
/// each target has one encoding, and there is no sign bit.
pub const MANTISSA_MIN: u32 = 0x0080_0000;
pub const MANTISSA_MAX: u32 = 0x00FF_FFFF;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetError {
    /// `shift` above `SHIFT_MAX`.
    ShiftTooLarge,
    /// `mantissa` outside `MANTISSA_MIN..=MANTISSA_MAX`.
    MantissaOutOfRange,
    /// The value is below 2^23 and has no normalised compact form.
    NotNormalizable,
}

impl std::fmt::Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetError::ShiftTooLarge => write!(f, "shift above 232"),
            TargetError::MantissaOutOfRange => write!(f, "mantissa outside 0x800000..=0xFFFFFF"),
            TargetError::NotNormalizable => write!(f, "target has no normalised compact form"),
        }
    }
}

impl std::error::Error for TargetError {}

/// A 256-bit target as a little-endian byte array.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Target(pub [u8; 32]);

impl Target {
    /// Decodes the compact form `mantissa << shift`. Out-of-range shift or
    /// mantissa is an error, so only the normalised form decodes.
    pub fn from_bits(bits: u32) -> Result<Target, TargetError> {
        let shift = bits >> 24;
        let mantissa = bits & MANTISSA_MAX;
        if shift > SHIFT_MAX {
            return Err(TargetError::ShiftTooLarge);
        }
        if !(MANTISSA_MIN..=MANTISSA_MAX).contains(&mantissa) {
            return Err(TargetError::MantissaOutOfRange);
        }
        // The residual shift is below 8, so acc has at most 24 + 7 = 31 bits;
        // the byte loop cannot pass index 31 because shift <= 232.
        let mut out = [0u8; 32];
        let mut acc = (mantissa as u64) << (shift % 8);
        let mut idx = (shift / 8) as usize;
        while acc != 0 {
            out[idx] = (acc & 0xFF) as u8;
            acc >>= 8;
            idx += 1;
        }
        Ok(Target(out))
    }

    /// The normalised compact form, or an error for targets below 2^23.
    /// Bits below the 24-bit mantissa are dropped, so the result is exact
    /// only for targets that came from `from_bits`.
    pub fn to_bits(&self) -> Result<u32, TargetError> {
        let top = self.0.iter().rposition(|&b| b != 0).ok_or(TargetError::NotNormalizable)?;
        // Index of the highest set bit; the mantissa keeps it as bit 23.
        let topbits = 8 * top as u32 + (7 - self.0[top].leading_zeros());
        if topbits < 23 {
            return Err(TargetError::NotNormalizable);
        }
        let shift = topbits - 23;
        let mantissa = self.shr_small(shift) as u32;
        Ok((shift << 24) | mantissa)
    }

    /// True when the 256-bit little-endian value `hash` does not exceed this
    /// target (spec/node.md section 4.1: valid when `ticket <= target`).
    /// Compares from the most significant byte (index 31) down.
    pub fn admits(&self, hash: &[u8; 32]) -> bool {
        for i in (0..32).rev() {
            if hash[i] != self.0[i] {
                return hash[i] < self.0[i];
            }
        }
        true
    }

    /// This target times 2^n, or `None` when the product needs more than 256
    /// bits. The zero target maps to itself.
    pub fn shifted_left(&self, n: u32) -> Option<Target> {
        let Some(top) = self.0.iter().rposition(|&b| b != 0) else {
            return Some(*self);
        };
        let bit_len = 8 * top as u32 + (8 - self.0[top].leading_zeros());
        if bit_len + n > 256 {
            return None;
        }
        let (bytes, bits) = ((n / 8) as usize, n % 8);
        let mut out = [0u8; 32];
        for i in 0..32 - bytes {
            out[i + bytes] |= self.0[i] << bits;
            if bits > 0 && i + bytes + 1 < 32 {
                out[i + bytes + 1] |= self.0[i] >> (8 - bits);
            }
        }
        Some(Target(out))
    }

    /// The 48 bits of this target starting at bit `n`, shifted down. Callers
    /// need 24 bits, so 6 bytes cover any bit offset within a byte.
    fn shr_small(&self, n: u32) -> u64 {
        let byte = (n / 8) as usize;
        let bit = n % 8;
        let mut acc: u64 = 0;
        for i in (0..6).rev() {
            let idx = byte + i;
            let b = if idx < 32 { self.0[idx] } else { 0 };
            acc = (acc << 8) | b as u64;
        }
        acc >> bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex32(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        out
    }

    // Byte offsets and little-endian order of every header field.
    #[test]
    fn header_encoding_offsets() {
        let mut h = Header {
            version: 0x0102_0304,
            time: 0x5566_7788,
            bits: 0x9988_7766,
            nonce: 0xDDEE_FF00,
            ..Header::default()
        };
        h.parent[0] = 0xAA;
        h.body_root[31] = 0xBB;
        let b = h.encode();
        assert_eq!(&b[0..4], &[0x04, 0x03, 0x02, 0x01]);
        assert_eq!(b[4], 0xAA);
        assert_eq!(b[35], 0x00);
        assert_eq!(b[67], 0xBB);
        assert_eq!(&b[68..72], &[0x88, 0x77, 0x66, 0x55]);
        assert_eq!(&b[72..76], &[0x66, 0x77, 0x88, 0x99]);
        assert_eq!(&b[76..80], &[0x00, 0xFF, 0xEE, 0xDD]);
    }

    // decode(encode(h)) == h through all conversions.
    #[test]
    fn header_roundtrip() {
        let mut h = Header { version: 1, ..Header::default() };
        for i in 0..32 {
            h.parent[i] = i as u8;
            h.body_root[i] = 255 - i as u8;
        }
        h.time = 1_700_000_000;
        h.bits = 0x0080_0000;
        h.nonce = 0xDEAD_BEEF;
        let buf = h.encode();
        assert_eq!(Header::decode(&buf), h);
        assert_eq!(Header::from(buf), h);
        let buf2: [u8; 80] = h.into();
        assert_eq!(buf2, buf);
    }

    #[test]
    fn header_id_known_answers() {
        // Known answers of BLAKE3 over 80 bytes, reproduced by the Python
        // implementation in pow/oracle.
        assert_eq!(Header::default().id(), hex32("6ba212aad7f6eb4e216ceaf270778639e81c565938360246a45e1f51cf2aaf3a"));
        // Header whose byte i equals i.
        let mut buf = [0u8; 80];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = i as u8;
        }
        let h2 = Header::decode(&buf);
        assert_eq!(h2.id(), hex32("4a6847ef66bce6c14ca05ec8f8ad7383bf2931f4bcfd0373c18e059e93f9def6"));
    }

    // Decoded targets at the shift boundaries, checked byte by byte.
    #[test]
    fn target_from_bits_hand_cases() {
        let t = Target::from_bits(0x0080_0000).unwrap();
        assert_eq!(t.0[2], 0x80);
        assert!(t.0[..2].iter().all(|&b| b == 0) && t.0[3..].iter().all(|&b| b == 0));

        let t = Target::from_bits(0x0180_0000).unwrap(); // shift 1
        assert_eq!(t.0[3], 0x01);
        assert!(t.0[..3].iter().all(|&b| b == 0) && t.0[4..].iter().all(|&b| b == 0));

        let t = Target::from_bits(0xE8FF_FFFF).unwrap(); // shift 232, max mantissa
        assert_eq!(&t.0[29..32], &[0xFF, 0xFF, 0xFF]);
        assert!(t.0[..29].iter().all(|&b| b == 0));

        // 0x800000 << 212 = 2^235: the only set bit is bit 3 of byte 29.
        let t = Target::from_bits(0xD480_0000).unwrap();
        assert_eq!(t.0[29], 0x08);
        assert!(t.0.iter().enumerate().all(|(i, &b)| i == 29 || b == 0));
    }

    // Non-normalised or out-of-range bits are invalid, so a target has one encoding.
    #[test]
    fn target_rejects_bad_bits() {
        assert_eq!(Target::from_bits(0xE980_0000), Err(TargetError::ShiftTooLarge));
        assert_eq!(Target::from_bits(0x007F_FFFF), Err(TargetError::MantissaOutOfRange));
        assert_eq!(Target::from_bits(0x0000_0001), Err(TargetError::MantissaOutOfRange));
        assert_eq!(Target::from_bits(0x0000_0000), Err(TargetError::MantissaOutOfRange));
    }

    // decode then encode is the identity on normalised bits.
    #[test]
    fn target_roundtrips_through_bits() {
        for bits in [0x0080_0000u32, 0x00FF_FFFF, 0x0180_0000, 0x5AC0_0001, 0xE8FF_FFFF, 0x01FF_FFFF, 0xE880_0000] {
            let t = Target::from_bits(bits).unwrap();
            assert_eq!(t.to_bits(), Ok(bits), "bits {bits:#010x}");
        }
    }

    // Values below 2^23 cannot fill the 24-bit mantissa.
    #[test]
    fn target_below_2_pow_23_is_not_normalizable() {
        assert_eq!(
            Target([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                .to_bits(),
            Err(TargetError::NotNormalizable)
        );
        assert_eq!(Target([0u8; 32]).to_bits(), Err(TargetError::NotNormalizable));
    }

    // Overflow past 256 bits returns None instead of wrapping.
    #[test]
    fn shifted_left_multiplies_or_reports_overflow() {
        let t = Target::from_bits(0x0180_0000).unwrap(); // 0x800000 << 1
        assert_eq!(t.shifted_left(5).unwrap(), Target::from_bits(0x0680_0000).unwrap());
        assert_eq!(t.shifted_left(0).unwrap(), t);
        assert_eq!(Target::from_bits(0xE880_0000).unwrap().shifted_left(1), None);
        // 2^242 has 243 bits: shifting by 13 reaches 2^255 (256 bits), by 14 overflows.
        let big = Target::from_bits(0xDB80_0000).unwrap(); // 0x800000 << 219 = 2^242
        assert_eq!(big.shifted_left(13).unwrap(), Target::from_bits(0xE880_0000).unwrap());
        assert_eq!(big.shifted_left(14), None);
        // A shift that is not a whole number of bytes carries across byte borders.
        let odd = Target::from_bits(0x0080_0001).unwrap();
        assert_eq!(odd.shifted_left(3).unwrap().to_bits().unwrap(), 0x0380_0001);
    }

    // Equality passes (ticket <= target); order is decided by the highest differing byte.
    #[test]
    fn target_admits() {
        let t = Target::from_bits(0x0080_0000).unwrap(); // target = 0x800000
        let equal = {
            let mut h = [0u8; 32];
            h[2] = 0x80;
            h
        };
        let below = {
            let mut h = [0u8; 32];
            h[2] = 0x7F;
            h[0] = 0xFF; // low bytes do not matter once a higher byte is smaller
            h
        };
        let above = {
            let mut h = [0u8; 32];
            h[2] = 0x80;
            h[3] = 0x01;
            h
        };
        assert!(t.admits(&equal));
        assert!(t.admits(&below));
        assert!(!t.admits(&above));
    }
}
