//! Blocks: the 80-byte header plus the body (spec/node.md section 4.2,
//! spec/shares.md section 11.3).
//!
//! Wire layout: `header (80) | tx_count (varint) | transactions |
//! share_count (varint) | share records (40 bytes each)`. `body_root` commits
//! to the whole body, so a header fixes the transactions, the coinbase
//! extranonce and the share list. Decoding enforces the shape rules that need
//! no chain state; the rules that need state are in `state`.

use crate::tx::{Transaction, h};
use neb_block::Header;
use neb_codec::{Reader, write_varint};

/// Limit on the serialised block (header and body) in bytes (spec/node.md
/// section 5.4).
pub const MAX_BLOCK_BYTES: usize = 1_000_000;

/// Most share records one block lists (spec/shares.md section 11).
pub const MAX_SHARES_PER_BLOCK: usize = 256;

/// Bytes of a share record (spec/shares.md section 11.2).
pub const SHARE_RECORD_BYTES: usize = 40;

/// A share of a round (spec/shares.md section 11.2): the fields of a header
/// that a miner chooses, `body_root (32) || LE32(time) || LE32(nonce)`. The
/// round fixes the rest of the header (the parent, the bits and the version),
/// so a record is bound to the parent of the round and not to the header of
/// the block that wins it. Records order as byte strings.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ShareRecord {
    pub body_root: [u8; 32],
    pub time: u32,
    pub nonce: u32,
}

impl ShareRecord {
    pub fn to_bytes(&self) -> [u8; SHARE_RECORD_BYTES] {
        let mut out = [0u8; SHARE_RECORD_BYTES];
        out[..32].copy_from_slice(&self.body_root);
        out[32..36].copy_from_slice(&self.time.to_le_bytes());
        out[36..].copy_from_slice(&self.nonce.to_le_bytes());
        out
    }

    pub fn from_bytes(buf: &[u8; SHARE_RECORD_BYTES]) -> ShareRecord {
        ShareRecord {
            body_root: buf[..32].try_into().unwrap(),
            time: u32::from_le_bytes(buf[32..36].try_into().unwrap()),
            nonce: u32::from_le_bytes(buf[36..].try_into().unwrap()),
        }
    }

    /// The share header of this record in the round whose headers have the
    /// parent `parent` and the bits `bits`; its id is the key of the share.
    pub fn header(&self, parent: [u8; 32], bits: u32) -> Header {
        Header { version: 1, parent, body_root: self.body_root, time: self.time, bits, nonce: self.nonce }
    }

    /// The record of a header (its parent, bits and version are not kept).
    pub fn of_header(header: &Header) -> ShareRecord {
        ShareRecord { body_root: header.body_root, time: header.time, nonce: header.nonce }
    }
}

impl PartialOrd for ShareRecord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ShareRecord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_bytes().cmp(&other.to_bytes())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Block {
    pub header: Header,
    pub txs: Vec<Transaction>,
    /// Share records of the round of the parent (spec/shares.md section 11.2),
    /// strictly increasing as byte strings.
    pub shares: Vec<ShareRecord>,
}

/// The list rules of spec/shares.md section 11.2 that need no chain state:
/// at most 256 records, strictly increasing (so without duplicate). Strict
/// order gives the list one encoding, so `shares_root` is canonical.
pub fn check_shares_list(shares: &[ShareRecord]) -> Result<(), &'static str> {
    if shares.len() > MAX_SHARES_PER_BLOCK {
        return Err("more than 256 share records");
    }
    if shares.windows(2).any(|w| w[0] >= w[1]) {
        return Err("share records are not strictly increasing");
    }
    Ok(())
}

/// BLAKE3 of the records, 40 bytes each (spec/shares.md section 11.3); the
/// empty list hashes the empty input.
pub fn shares_root(shares: &[ShareRecord]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for r in shares {
        hasher.update(&r.to_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// `H("body", LE32(count), merkle_root, shares_root)` (spec/node.md section
/// 4.2); `h` concatenates the tag and the parts without separator.
pub fn body_root_of(tx_count: u32, merkle_root: &[u8; 32], shares_root: &[u8; 32]) -> [u8; 32] {
    h(b"body", &[&tx_count.to_le_bytes(), merkle_root, shares_root])
}

impl Block {
    /// Serialised block; the length is what the size limit counts.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = self.header.encode().to_vec();
        write_varint(&mut out, self.txs.len() as u64);
        for tx in &self.txs {
            out.extend_from_slice(&tx.encode());
        }
        write_varint(&mut out, self.shares.len() as u64);
        for r in &self.shares {
            out.extend_from_slice(&r.to_bytes());
        }
        out
    }

    /// Decodes a complete block. Errors on a short header, a transaction count
    /// of 0 or above u32::MAX, non-canonical varints, a malformed share list
    /// and trailing bytes. Does not check sizes or commitments.
    pub fn decode(buf: &[u8]) -> Result<Block, String> {
        if buf.len() < 80 {
            return Err("shorter than a header".into());
        }
        let header = Header::decode(buf[..80].try_into().unwrap());
        let mut r = Reader::new(&buf[80..]);
        let count = r.read_varint().map_err(|e| e.to_string())?;
        if count == 0 || count > u32::MAX as u64 {
            return Err("transaction count out of range".into());
        }
        // The count is untrusted: cap the preallocation. Each decode consumes
        // input, so a false count fails at the end of the buffer.
        let mut txs = Vec::with_capacity((count as usize).min(4096));
        for _ in 0..count {
            txs.push(Transaction::decode_from(&mut r)?);
        }
        let n = r.read_varint().map_err(|e| e.to_string())?;
        if n > MAX_SHARES_PER_BLOCK as u64 {
            return Err("more than 256 share records".into());
        }
        // n <= 256 here, so the allocation is bounded.
        let mut shares = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let bytes: [u8; SHARE_RECORD_BYTES] =
                r.read_exact(SHARE_RECORD_BYTES).map_err(|e| e.to_string())?.try_into().unwrap();
            shares.push(ShareRecord::from_bytes(&bytes));
        }
        check_shares_list(&shares)?;
        r.expect_done().map_err(|e| e.to_string())?;
        Ok(Block { header, txs, shares })
    }

    pub fn id(&self) -> [u8; 32] {
        self.header.id()
    }

    /// Commitment of the header to the body (spec/node.md section 4.2). The
    /// Merkle tree has leaves `H("leaf", BLAKE3(full transaction bytes))`
    /// (witnesses included) and nodes `H("node", left, right)`; an odd last
    /// node is promoted unchanged. The count and the share list are bound by
    /// the outer hash, which also separates trees of different sizes.
    pub fn body_root(&self) -> [u8; 32] {
        let mut level: Vec<[u8; 32]> =
            self.txs.iter().map(|t| h(b"leaf", &[blake3::hash(&t.encode()).as_bytes()])).collect();
        let shares = shares_root(&self.shares);
        if level.is_empty() {
            return body_root_of(0, &[0u8; 32], &shares);
        }
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut i = 0;
            while i + 1 < level.len() {
                next.push(h(b"node", &[&level[i], &level[i + 1]]));
                i += 2;
            }
            if i < level.len() {
                next.push(level[i]);
            }
            level = next;
        }
        body_root_of(self.txs.len() as u32, &level[0], &shares)
    }

    /// True when the header's `body_root` matches the body.
    pub fn body_ok(&self) -> bool {
        self.body_root() == self.header.body_root
    }
}
