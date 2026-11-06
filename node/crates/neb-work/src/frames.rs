//! Binary frames of the work service (spec/node.md section 7.1): the framing,
//! the message types and the codec of each payload. The node and the miners
//! share this code, so both sides decode the same bytes. The peer is a local
//! miner process, but every decoder bounds its allocations as for a hostile
//! peer.
//!
//! A frame is `length (u32) || type (u8) || payload`, little-endian, where
//! `length` counts the type byte and the payload. Every decoder is canonical:
//! it rejects a short payload, an out-of-range field and trailing bytes.
//! A framing error is fatal for the session (the receiver closes the
//! connection); the receivers are in `neb-node` (`daemon`) and `neb-cpuminer`.

use neb_block::Header;
use neb_chain::block::body_root_of;
use neb_chain::tx::h;
use neb_codec::{Reader, write_varint};
use neb_pow::shares::{Model, PACKED_WEIGHTS};

/// Frame types of spec/node.md section 7.1 (miner to node: 1, 4, 6; node to
/// miner: 2, 5, 7).
pub const MSG_HELLO: u8 = 1;
pub const MSG_JOB: u8 = 2;
/// Reserved (spec/node.md section 11): receivers ignore it.
pub const MSG_SET_SHARE_TARGET: u8 = 3;
pub const MSG_SUBMIT: u8 = 4;
pub const MSG_RESULT: u8 = 5;
pub const MSG_SHARE: u8 = 6;
pub const MSG_MODEL: u8 = 7;

/// Largest `length` field, type byte included: 2 MiB (2 097 152 bytes), about
/// 40 times the model frame. It bounds what a peer can make the receiver buffer.
pub const MAX_FRAME: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    /// The length field is 0 or above `MAX_FRAME`, or a count or string exceeds its bound.
    TooLong,
    /// The type byte is not in 1..=7.
    BadType(u8),
    /// The payload does not decode.
    BadShape(String),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::TooLong => write!(f, "frame over 2 MiB"),
            FrameError::BadType(t) => write!(f, "unknown frame type {t}"),
            FrameError::BadShape(e) => write!(f, "bad payload: {e}"),
        }
    }
}

impl std::error::Error for FrameError {}

/// Builds `length || type || payload` with `length = 1 + payload.len()`. The
/// caller keeps the payload below `MAX_FRAME`; the largest frame sent is the
/// model frame (1 + 1 + 32 + 50816 bytes).
pub fn frame(msg_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    out.extend_from_slice(&((payload.len() + 1) as u32).to_le_bytes());
    out.push(msg_type);
    out.extend_from_slice(payload);
    out
}

/// The front of a receive buffer: a complete frame, or how much is missing.
pub enum FrameOut<'a> {
    /// Bytes still missing before the next call can make progress: first the
    /// 4-byte length field, then the rest of the frame.
    Need(usize),
    /// A complete frame; `consumed` (`4 + length`) is what to drop from the buffer.
    Message { msg_type: u8, payload: &'a [u8], consumed: usize },
}

/// Parses the first frame of `buf`, without copying. The length field is
/// checked as soon as its 4 bytes are present, so an oversized frame is refused
/// before its payload is waited for. The type byte is checked once the whole
/// frame has arrived.
pub fn parse_frame(buf: &[u8]) -> Result<FrameOut<'_>, FrameError> {
    if buf.len() < 4 {
        return Ok(FrameOut::Need(4 - buf.len()));
    }
    let len = u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize;
    // Refuse before waiting for the payload: `len` counts the type byte, so 0
    // cannot be a frame, and a large `len` would make the receiver buffer it.
    if len == 0 || len > MAX_FRAME {
        return Err(FrameError::TooLong);
    }
    if buf.len() < 4 + len {
        return Ok(FrameOut::Need(4 + len - buf.len()));
    }
    let msg_type = buf[4];
    if !(MSG_HELLO..=MSG_MODEL).contains(&msg_type) {
        return Err(FrameError::BadType(msg_type));
    }
    Ok(FrameOut::Message { msg_type, payload: &buf[5..4 + len], consumed: 4 + len })
}

/// A varint length followed by that many bytes (spec/node.md section 3).
fn write_bytes(out: &mut Vec<u8>, b: &[u8]) {
    write_varint(out, b.len() as u64);
    out.extend_from_slice(b);
}

/// Inverse of `write_bytes`; a length above `MAX_FRAME` is refused before the read.
fn read_bytes(r: &mut Reader) -> Result<Vec<u8>, FrameError> {
    let n = r.read_varint().map_err(|e| FrameError::BadShape(e.to_string()))?;
    if n > MAX_FRAME as u64 {
        return Err(FrameError::TooLong);
    }
    Ok(r.read_exact(n as usize).map_err(|e| FrameError::BadShape(e.to_string()))?.to_vec())
}

/// First frame of a miner: the version of the work service protocol (1) and a
/// name for the node log. The node never branches on either.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hello {
    pub version: u16,
    pub name: String,
}

impl Hello {
    /// Payload only, without the frame header.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.version.to_le_bytes());
        write_bytes(&mut out, self.name.as_bytes());
        out
    }
    pub fn decode(p: &[u8]) -> Result<Hello, FrameError> {
        let mut r = Reader::new(p);
        let version = r.read_u16().map_err(|e| FrameError::BadShape(e.to_string()))?;
        let name = String::from_utf8(read_bytes(&mut r)?).map_err(|_| FrameError::BadShape("name not utf8".into()))?;
        r.expect_done().map_err(|e| FrameError::BadShape(e.to_string()))?;
        Ok(Hello { version, name })
    }
}

/// The job of spec/node.md section 7.1: everything a miner needs except its own
/// extranonce, time and nonce. The coinbase is `prefix || extranonce || suffix`.
/// `branch` is the Merkle path of the coinbase leaf (position 0) through the
/// other transactions of the template; its entries are opaque hashes that the
/// miner folds in order with `H("node", h, s)`.
/// The model the attempts are evaluated on is the one of the last `model` frame.
/// Times are Unix seconds, `u32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    /// Names the job in `Submit` and `ShareMsg` frames.
    pub id: u64,
    /// The node sets it in every job.
    pub clean: bool,
    /// Header version, 1.
    pub version: u32,
    pub parent: [u8; 32],
    /// Unix seconds. A header time must lie in `[time_start, time_start + time_window]`.
    pub time_start: u32,
    pub time_window: u32,
    /// Compact block target (spec/node.md section 4.1).
    pub bits: u32,
    /// `target(bits)` as a 256-bit little-endian integer.
    pub target: [u8; 32],
    /// The share target of `bits` (spec/shares.md section 11.2), same encoding.
    pub share_target: [u8; 32],
    pub coinbase_prefix: Vec<u8>,
    pub coinbase_suffix: Vec<u8>,
    /// Bytes of extranonce, 1 to 32.
    pub extranonce_size: u8,
    pub branch: Vec<[u8; 32]>,
    /// Transactions in the template, coinbase included: part of the
    /// `body_root` commitment.
    pub tx_count: u32,
    /// The commitment to the share list of the block (spec/shares.md
    /// section 11.3), part of `body_root`.
    pub shares_root: [u8; 32],
    /// Height of the block that the job builds.
    pub height: u32,
    /// Unix time after which a miner that rolls the extranonce or the time
    /// stops using the job.
    pub expiry: u32,
}

impl Job {
    /// The `body_root` for this extranonce: the coinbase leaf
    /// `H("leaf", BLAKE3(coinbase))` folded with the branch, then
    /// `H("body", LE32(tx_count), merkle_root, shares_root)` (spec/node.md
    /// sections 4.2 and 7.1).
    pub fn body_root(&self, extranonce: &[u8]) -> [u8; 32] {
        let mut coinbase =
            Vec::with_capacity(self.coinbase_prefix.len() + extranonce.len() + self.coinbase_suffix.len());
        coinbase.extend_from_slice(&self.coinbase_prefix);
        coinbase.extend_from_slice(extranonce);
        coinbase.extend_from_slice(&self.coinbase_suffix);
        let mut acc = h(b"leaf", &[blake3::hash(&coinbase).as_bytes()]);
        for sibling in &self.branch {
            acc = h(b"node", &[&acc, sibling]);
        }
        body_root_of(self.tx_count, &acc, &self.shares_root)
    }

    /// The 80-byte header of an attempt (spec/node.md section 4.1); the miner
    /// hashes it with BLAKE3 to get the key of the attempt.
    pub fn header(&self, extranonce: &[u8], time: u32, nonce: u32) -> Header {
        Header {
            version: self.version,
            parent: self.parent,
            body_root: self.body_root(extranonce),
            time,
            bits: self.bits,
            nonce,
        }
    }

    /// Fields in the order of spec/node.md section 7.1.
    fn payload(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.id.to_le_bytes());
        out.push(self.clean as u8);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.parent);
        out.extend_from_slice(&self.time_start.to_le_bytes());
        out.extend_from_slice(&self.time_window.to_le_bytes());
        out.extend_from_slice(&self.bits.to_le_bytes());
        out.extend_from_slice(&self.target);
        out.extend_from_slice(&self.share_target);
        write_bytes(&mut out, &self.coinbase_prefix);
        write_bytes(&mut out, &self.coinbase_suffix);
        out.push(self.extranonce_size);
        write_varint(&mut out, self.branch.len() as u64);
        for b in &self.branch {
            out.extend_from_slice(b);
        }
        out.extend_from_slice(&self.tx_count.to_le_bytes());
        out.extend_from_slice(&self.shares_root);
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.expiry.to_le_bytes());
        out
    }

    /// The complete frame, header included.
    pub fn encode(&self) -> Vec<u8> {
        frame(MSG_JOB, &self.payload())
    }

    /// Rejects `clean` other than 0 or 1, `extranonce_size` outside 1..=32, a
    /// branch of more than 4096 entries, `tx_count` 0 and trailing bytes.
    pub fn decode(p: &[u8]) -> Result<Job, FrameError> {
        let mut r = Reader::new(p);
        let shape = |e: neb_codec::CodecError| FrameError::BadShape(e.to_string());
        let id = r.read_u64().map_err(shape)?;
        let clean = match r.read_u8().map_err(shape)? {
            0 => false,
            1 => true,
            _ => return Err(FrameError::BadShape("clean is not 0 or 1".into())),
        };
        let version = r.read_u32().map_err(shape)?;
        let parent: [u8; 32] = r.read_exact(32).map_err(shape)?.try_into().unwrap();
        let time_start = r.read_u32().map_err(shape)?;
        let time_window = r.read_u32().map_err(shape)?;
        let bits = r.read_u32().map_err(shape)?;
        let target: [u8; 32] = r.read_exact(32).map_err(shape)?.try_into().unwrap();
        let share_target: [u8; 32] = r.read_exact(32).map_err(shape)?.try_into().unwrap();
        let coinbase_prefix = read_bytes(&mut r)?;
        let coinbase_suffix = read_bytes(&mut r)?;
        let extranonce_size = r.read_u8().map_err(shape)?;
        if extranonce_size == 0 || extranonce_size > 32 {
            return Err(FrameError::BadShape("extranonce size out of range".into()));
        }
        let n = r.read_varint().map_err(shape)?;
        // Checked before the allocation below: the entry count is attacker-chosen.
        if n > 4096 {
            return Err(FrameError::TooLong);
        }
        let mut branch = Vec::with_capacity(n as usize);
        for _ in 0..n {
            branch.push(r.read_exact(32).map_err(shape)?.try_into().unwrap());
        }
        let tx_count = r.read_u32().map_err(shape)?;
        if tx_count == 0 {
            return Err(FrameError::BadShape("zero transactions".into()));
        }
        let shares_root: [u8; 32] = r.read_exact(32).map_err(shape)?.try_into().unwrap();
        let height = r.read_u32().map_err(shape)?;
        let expiry = r.read_u32().map_err(shape)?;
        r.expect_done().map_err(shape)?;
        Ok(Job {
            id,
            clean,
            version,
            parent,
            time_start,
            time_window,
            bits,
            target,
            share_target,
            coinbase_prefix,
            coinbase_suffix,
            extranonce_size,
            branch,
            tx_count,
            shares_root,
            height,
            expiry,
        })
    }
}

/// A solution: the extranonce, time and nonce that complete the job `id` into a
/// header that meets the block target. Decoding checks the shape only; the node
/// checks the extranonce length, the time window and the proof of work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submit {
    pub id: u64,
    pub extranonce: Vec<u8>,
    pub time: u32,
    pub nonce: u32,
}

impl Submit {
    /// The complete frame, header included.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.id.to_le_bytes());
        write_bytes(&mut out, &self.extranonce);
        out.extend_from_slice(&self.time.to_le_bytes());
        out.extend_from_slice(&self.nonce.to_le_bytes());
        frame(MSG_SUBMIT, &out)
    }
    pub fn decode(p: &[u8]) -> Result<Submit, FrameError> {
        let mut r = Reader::new(p);
        let shape = |e: neb_codec::CodecError| FrameError::BadShape(e.to_string());
        let id = r.read_u64().map_err(shape)?;
        let extranonce = read_bytes(&mut r)?;
        let time = r.read_u32().map_err(shape)?;
        let nonce = r.read_u32().map_err(shape)?;
        r.expect_done().map_err(shape)?;
        Ok(Submit { id, extranonce, time, nonce })
    }
}

/// The answer to a `Submit`; `code` is the result code of spec/node.md section
/// 7.1 (0 accepted, 1 bad submission, 2 bad header, 3 bad state).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultMsg {
    pub accepted: bool,
    pub code: u8,
}

impl ResultMsg {
    /// The complete frame, header included.
    pub fn encode(&self) -> Vec<u8> {
        frame(MSG_RESULT, &[self.accepted as u8, self.code])
    }
    pub fn decode(p: &[u8]) -> Result<ResultMsg, FrameError> {
        if p.len() != 2 {
            return Err(FrameError::BadShape("result is two bytes".into()));
        }
        // Any `accepted` byte other than 1 reads as false.
        Ok(ResultMsg { accepted: p[0] == 1, code: p[1] })
    }
}

/// A frame of the reserved type 3 with a 32-byte target (spec/node.md section 11).
pub fn set_share_target(target: &[u8; 32]) -> Vec<u8> {
    frame(MSG_SET_SHARE_TARGET, target)
}

/// A share found by a miner on a job: a nonce of the reference header of the
/// job whose ticket meets the share target (spec/shares.md section 11.5).
/// The node does not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShareMsg {
    pub job_id: u64,
    pub nonce: u32,
}

impl ShareMsg {
    /// The complete frame, header included.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12);
        out.extend_from_slice(&self.job_id.to_le_bytes());
        out.extend_from_slice(&self.nonce.to_le_bytes());
        frame(MSG_SHARE, &out)
    }

    pub fn decode(p: &[u8]) -> Result<ShareMsg, FrameError> {
        let mut r = Reader::new(p);
        let shape = |e: neb_codec::CodecError| FrameError::BadShape(e.to_string());
        let job_id = r.read_u64().map_err(shape)?;
        let nonce = r.read_u32().map_err(shape)?;
        r.expect_done().map_err(shape)?;
        Ok(ShareMsg { job_id, nonce })
    }
}

/// The model frame: `slot (1) || weights_hash (32) || packed weights (50 816 bytes)`
/// (spec/shares.md section 11.5): the slot of the model, which names the dataset the
/// attempts use, then the 4-bit weights of both layers, packed. The node sends it
/// before every job. Residuals are not sent.
pub fn model_frame(slot: u8, model: &Model) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 32 + PACKED_WEIGHTS);
    out.push(slot);
    out.extend_from_slice(&model.weights_hash());
    out.extend_from_slice(&model.packed_weights());
    frame(MSG_MODEL, &out)
}

/// The slot and the weights of a model frame payload, as a model with zero residuals.
/// Fails when the size is wrong or the weights do not hash to the announced
/// `weights_hash`. The node is trusted, so the hash detects a damaged or
/// truncated frame, not a forged one.
pub fn decode_model(p: &[u8]) -> Result<(u8, Model), FrameError> {
    if p.len() != 1 + 32 + PACKED_WEIGHTS {
        return Err(FrameError::BadShape("model frame has the wrong size".into()));
    }
    let model = Model::from_packed_weights(&p[33..]).map_err(FrameError::BadShape)?;
    if model.weights_hash() != p[1..33] {
        return Err(FrameError::BadShape("weights do not match their hash".into()));
    }
    Ok((p[0], model))
}
