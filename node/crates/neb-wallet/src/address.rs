//! bech32m addresses for the 32-byte `lock_data` of a type 1 lock
//! (spec/node.md section 5.2).
//!
//! The payload is the BLAKE3 hash of the Ed25519 public key. There is no
//! witness-version byte: the prefix `neb`, the separator `1`, 52 data
//! characters (256 bits as 5-bit values, the last 4 bits zero padding) and a
//! 6-character checksum are the whole address, 62 characters in all.

/// Human-readable prefix of the network, lowercase.
pub const ADDRESS_PREFIX: &str = "neb";

/// The bech32 alphabet: the character at index `v` encodes the 5-bit value `v`.
const CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
/// bech32m checksum constant. bech32 uses 1, so an address
/// does not verify as the other variant.
const BECH32M_CONST: u32 = 0x2bc8_30a3;
/// Generator coefficients of the bech32 checksum (a Bose-Chaudhuri-Hocquenghem error-correcting code).
const GENERATOR: [u32; 5] = [0x3b6a_57b2, 0x2650_8e6d, 0x1ea1_19fa, 0x3d42_33dd, 0x2a14_62b3];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddressError {
    Empty,
    MixedCase,
    NoSeparator,
    BadCharacter(char),
    BadChecksum,
    BadLength,
    BadPadding,
    WrongPrefix,
}

impl std::fmt::Display for AddressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddressError::Empty => write!(f, "empty address"),
            AddressError::MixedCase => write!(f, "mixed case address"),
            AddressError::NoSeparator => write!(f, "missing separator"),
            AddressError::BadCharacter(c) => write!(f, "bad character {c}"),
            AddressError::BadChecksum => write!(f, "bad checksum"),
            AddressError::BadLength => write!(f, "payload is not 32 bytes"),
            AddressError::BadPadding => write!(f, "bad padding"),
            AddressError::WrongPrefix => write!(f, "wrong address prefix"),
        }
    }
}

impl std::error::Error for AddressError {}

/// bech32 checksum polynomial over 5-bit values. The 30-bit state moves up by
/// 5 bits per value; the 5 bits that leave at the top select generator words.
fn polymod(values: &[u8]) -> u32 {
    let mut chk = 1u32;
    for value in values {
        let top = chk >> 25;
        chk = ((chk & 0x01ff_ffff) << 5) ^ (*value as u32);
        for (i, generator) in GENERATOR.iter().enumerate() {
            if (top >> i) & 1 == 1 {
                chk ^= generator;
            }
        }
    }
    chk
}

/// bech32: the high 3 bits of each prefix byte, a zero, the low 5 bits of
/// each prefix byte.
fn hrp_expand(hrp: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(hrp.len() * 2 + 1);
    for b in hrp.bytes() {
        out.push(b >> 5);
    }
    out.push(0);
    for b in hrp.bytes() {
        out.push(b & 31);
    }
    out
}

/// The six 5-bit checksum values, most significant first.
fn create_checksum(hrp: &str, data: &[u8]) -> [u8; 6] {
    let mut values = hrp_expand(hrp);
    values.extend_from_slice(data);
    values.extend_from_slice(&[0; 6]);
    let mod_value = polymod(&values) ^ BECH32M_CONST;
    let mut out = [0u8; 6];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = ((mod_value >> (5 * (5 - i))) & 31) as u8;
    }
    out
}

fn verify_checksum(hrp: &str, data: &[u8]) -> bool {
    let mut values = hrp_expand(hrp);
    values.extend_from_slice(data);
    polymod(&values) == BECH32M_CONST
}

/// Regroups `from`-bit values into `to`-bit values, most significant bit
/// first (bech32 `convertbits`). With `pad`, a partial last group is padded
/// with zero bits. Without it, `None` is returned for a leftover of `from` or
/// more bits or for non-zero leftover bits, so every byte string has exactly
/// one encoding. `acc` is not masked: bits already emitted shift out of the
/// `u32` and are never read.
fn convert_bits(data: &[u8], from: u32, to: u32, pad: bool) -> Option<Vec<u8>> {
    let mut acc = 0u32;
    let mut bits = 0u32;
    let mut out = Vec::new();
    let max_value = (1u32 << from) - 1;
    let max_out = (1u32 << to) - 1;
    for value in data {
        if (*value as u32) > max_value {
            return None;
        }
        acc = (acc << from) | *value as u32;
        bits += from;
        while bits >= to {
            bits -= to;
            out.push(((acc >> bits) & max_out) as u8);
        }
    }
    if pad {
        if bits > 0 {
            out.push(((acc << (to - bits)) & max_out) as u8);
        }
    } else if bits >= from || ((acc << (to - bits)) & max_out) != 0 {
        return None;
    }
    Some(out)
}

/// Encodes 32 bytes of lock data as a lowercase address. `prefix` must be
/// lowercase.
pub fn encode_address(prefix: &str, lock_data: &[u8; 32]) -> String {
    let data = convert_bits(lock_data, 8, 5, true).expect("8 to 5 conversion");
    let checksum = create_checksum(prefix, &data);
    let mut out = String::with_capacity(prefix.len() + 1 + data.len() + 6);
    out.push_str(prefix);
    out.push('1');
    for value in data.iter().chain(checksum.iter()) {
        out.push(CHARSET[*value as usize] as char);
    }
    out
}

/// Decodes an address of the network with the given lowercase `prefix`. An
/// all-uppercase address is accepted, a mixed-case one is not (bech32). The
/// prefix is compared before the checksum, so a valid address of another
/// network is `WrongPrefix`, not `BadChecksum`. The checksum detects any
/// change of up to 4 characters of an address of this length.
pub fn decode_address(prefix: &str, text: &str) -> Result<[u8; 32], AddressError> {
    if text.is_empty() {
        return Err(AddressError::Empty);
    }
    let lower = text.to_ascii_lowercase();
    if text != lower && text != text.to_ascii_uppercase() {
        return Err(AddressError::MixedCase);
    }
    // The separator is the last `1` (the data alphabet has none), with at
    // least the 6 checksum characters after it and a non-empty prefix before.
    let Some(pos) = lower.rfind('1') else {
        return Err(AddressError::NoSeparator);
    };
    if pos == 0 || pos + 7 > lower.len() {
        return Err(AddressError::NoSeparator);
    }
    let hrp = &lower[..pos];
    if hrp != prefix {
        return Err(AddressError::WrongPrefix);
    }
    let mut values = Vec::with_capacity(lower.len() - pos - 1);
    for c in lower[pos + 1..].chars() {
        let value = CHARSET.iter().position(|x| *x as char == c).ok_or(AddressError::BadCharacter(c))?;
        values.push(value as u8);
    }
    if !verify_checksum(hrp, &values) {
        return Err(AddressError::BadChecksum);
    }
    values.truncate(values.len() - 6);
    let bytes = convert_bits(&values, 5, 8, false).ok_or(AddressError::BadPadding)?;
    if bytes.len() != 32 {
        return Err(AddressError::BadLength);
    }
    Ok(bytes.try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Round trip, and the all-uppercase form that bech32 allows.
    #[test]
    fn roundtrip_and_prefix() {
        let lock = [0x42u8; 32];
        let address = encode_address(ADDRESS_PREFIX, &lock);
        assert!(address.starts_with("neb1"));
        assert_eq!(decode_address(ADDRESS_PREFIX, &address).unwrap(), lock);
        assert_eq!(decode_address(ADDRESS_PREFIX, &address.to_uppercase()).unwrap(), lock);
    }

    // A changed checksum character, a valid address of another prefix and an
    // address without data must all be refused.
    #[test]
    fn rejects_tampering_and_wrong_prefix() {
        let lock = [7u8; 32];
        let mut chars: Vec<char> = encode_address(ADDRESS_PREFIX, &lock).chars().collect();
        let last = chars.len() - 1;
        chars[last] = if chars[last] == 'q' { 'p' } else { 'q' };
        let bad: String = chars.into_iter().collect();
        assert!(decode_address(ADDRESS_PREFIX, &bad).is_err());
        let other = encode_address("xxx", &lock);
        assert_eq!(decode_address(ADDRESS_PREFIX, &other), Err(AddressError::WrongPrefix));
        assert!(decode_address(ADDRESS_PREFIX, "neb1").is_err());
    }
}
