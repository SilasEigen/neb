//! Canonical encoding primitives (spec/node.md section 3).
//!
//! Fixed-width integers are little-endian. Variable-length integers (varints)
//! use a one-byte prefix: values below 0xFD are the byte itself; 0xFD, 0xFE
//! and 0xFF announce a u16, u32 or u64 that follows. Only the shortest form
//! is valid, so every value has exactly one encoding and hashes of encoded
//! structures are unambiguous. Every read is bounded by the buffer.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// A read needs more bytes than remain.
    UnexpectedEnd,
    /// A varint is not in its shortest form.
    NonCanonical,
    /// Bytes remain after the value was fully decoded.
    TrailingBytes,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::UnexpectedEnd => write!(f, "unexpected end of input"),
            CodecError::NonCanonical => write!(f, "non-canonical variable-length integer"),
            CodecError::TrailingBytes => write!(f, "trailing bytes after the value"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Longest varint: prefix 0xFF plus eight bytes.
pub const MAX_VARINT_BYTES: usize = 9;

/// Appends the shortest varint encoding of `v` (1, 3, 5 or 9 bytes).
pub fn write_varint(out: &mut Vec<u8>, v: u64) {
    if v < 0xFD {
        out.push(v as u8);
    } else if v <= 0xFFFF {
        out.push(0xFD);
        out.extend_from_slice(&(v as u16).to_le_bytes());
    } else if v <= 0xFFFF_FFFF {
        out.push(0xFE);
        out.extend_from_slice(&(v as u32).to_le_bytes());
    } else {
        out.push(0xFF);
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Length in bytes of the shortest varint encoding of `v`.
pub fn varint_len(v: u64) -> usize {
    if v < 0xFD {
        1
    } else if v <= 0xFFFF {
        3
    } else if v <= 0xFFFF_FFFF {
        5
    } else {
        9
    }
}

/// A cursor over a byte slice; every read checks the remaining length first.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Returns the next `n` bytes. `n` comes from untrusted input, so the
    /// end offset is computed with a checked add.
    pub fn read_exact(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        let end = self.pos.checked_add(n).ok_or(CodecError::UnexpectedEnd)?;
        if end > self.buf.len() {
            return Err(CodecError::UnexpectedEnd);
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    pub fn read_u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.read_exact(1)?[0])
    }

    pub fn read_u16(&mut self) -> Result<u16, CodecError> {
        let b: [u8; 2] = self.read_exact(2)?.try_into().unwrap();
        Ok(u16::from_le_bytes(b))
    }

    pub fn read_u32(&mut self) -> Result<u32, CodecError> {
        let b: [u8; 4] = self.read_exact(4)?.try_into().unwrap();
        Ok(u32::from_le_bytes(b))
    }

    pub fn read_u64(&mut self) -> Result<u64, CodecError> {
        let b: [u8; 8] = self.read_exact(8)?.try_into().unwrap();
        Ok(u64::from_le_bytes(b))
    }

    /// Reads a varint; a form longer than the shortest is `NonCanonical`.
    pub fn read_varint(&mut self) -> Result<u64, CodecError> {
        let first = self.read_u8()?;
        match first {
            0..=0xFC => Ok(first as u64),
            0xFD => {
                let v = self.read_u16()? as u64;
                if v < 0xFD { Err(CodecError::NonCanonical) } else { Ok(v) }
            }
            0xFE => {
                let v = self.read_u32()? as u64;
                if v <= 0xFFFF { Err(CodecError::NonCanonical) } else { Ok(v) }
            }
            0xFF => {
                let v = self.read_u64()?;
                if v <= 0xFFFF_FFFF { Err(CodecError::NonCanonical) } else { Ok(v) }
            }
        }
    }

    /// Fails when bytes remain: a canonical decoding consumes its input exactly.
    pub fn expect_done(&self) -> Result<(), CodecError> {
        if self.pos == self.buf.len() { Ok(()) } else { Err(CodecError::TrailingBytes) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: u64) -> (Vec<u8>, u64) {
        let mut out = Vec::new();
        write_varint(&mut out, v);
        let mut r = Reader::new(&out);
        let got = r.read_varint().unwrap();
        r.expect_done().unwrap();
        (out, got)
    }

    // Every value has one encoding and its length equals `varint_len`.
    #[test]
    fn varint_roundtrip_and_length() {
        for v in [0u64, 1, 0xFC, 0xFD, 0xFF, 0x100, 0xFFFF, 0x1_0000, 0xFFFF_FFFF, 0x1_0000_0000, u64::MAX] {
            let (enc, got) = roundtrip(v);
            assert_eq!(got, v, "value {v}");
            assert_eq!(enc.len(), varint_len(v), "length of {v}");
        }
    }

    // Length steps at 0xFD, 0x10000 and 0x100000000.
    #[test]
    fn varint_lengths_at_boundaries() {
        assert_eq!(varint_len(0xFC), 1);
        assert_eq!(varint_len(0xFD), 3);
        assert_eq!(varint_len(0xFFFF), 3);
        assert_eq!(varint_len(0x1_0000), 5);
        assert_eq!(varint_len(0xFFFF_FFFF), 5);
        assert_eq!(varint_len(0x1_0000_0000), 9);
    }

    // Malleability: a value encoded in a longer form would give a second
    // byte string, and a second hash, for the same structure.
    #[test]
    fn varint_rejects_longer_forms() {
        let cases: [&[u8]; 4] = [
            &[0xFD, 0x00, 0x00],
            &[0xFD, 0xFC, 0x00],
            &[0xFE, 0xFF, 0xFF, 0x00, 0x00],
            &[0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ];
        for c in cases {
            let mut r = Reader::new(c);
            assert_eq!(r.read_varint(), Err(CodecError::NonCanonical), "bytes {c:02x?}");
        }
    }

    // A prefix without its payload is an error, never a short read.
    #[test]
    fn varint_rejects_truncation() {
        for c in [&[0xFD][..], &[0xFD, 0x01][..], &[0xFE, 0x01, 0x02, 0x03][..], &[0xFF, 1, 2, 3, 4, 5, 6][..], &[][..]]
        {
            let mut r = Reader::new(c);
            assert_eq!(r.read_varint(), Err(CodecError::UnexpectedEnd), "bytes {c:02x?}");
        }
    }

    // Trailing bytes after a decoded structure are an error.
    #[test]
    fn expects_exact_consumption() {
        let buf = [0x01u8, 0x02];
        let mut r = Reader::new(&buf);
        assert_eq!(r.read_varint().unwrap(), 1);
        assert_eq!(r.expect_done(), Err(CodecError::TrailingBytes));
        let mut r2 = Reader::new(&buf);
        assert_eq!(r2.read_u8().unwrap(), 1);
        assert_eq!(r2.read_u8().unwrap(), 2);
        assert_eq!(r2.expect_done(), Ok(()));
    }

    // Byte order of fixed-width integers and the bound at the buffer end.
    #[test]
    fn integer_reads_are_little_endian_and_bounded() {
        let buf = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut r = Reader::new(&buf);
        assert_eq!(r.read_u32().unwrap(), 0x0403_0201);
        assert_eq!(r.read_u32().unwrap(), 0x0807_0605);
        assert_eq!(r.read_u8(), Err(CodecError::UnexpectedEnd));
    }
}
