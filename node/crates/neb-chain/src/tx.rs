//! Transactions, locks and witnesses (spec/node.md section 5).
//!
//! Serialisation: `version (u8) | n_in (varint) | inputs | n_out (varint) |
//! outputs`; an input is `prev_txid (32) | prev_index (u32 LE) | witness_len
//! (varint) | witness`, an output `amount (u64 LE) | lock_type (u8) |
//! lock_data (32)`. `txid` commits to the serialisation without witnesses;
//! the block commits to the full bytes (section 4.2). A witness can thus be
//! re-encoded without changing the txid, which is why a signature covers the
//! txid and not the witness. Amounts are u64 base units of 10^-8 coin.
//!
//! This module checks the shape and the signatures of a transaction. Rules
//! that need the UTXO set (existence, maturity, fees) are in `state`.

use neb_codec::{Reader, write_varint};

/// One coin: 10^8 base units.
pub const COIN: u64 = 100_000_000;

/// Limits of an ordinary transaction (spec/node.md section 5.4); the
/// coinbase has exactly one input.
pub const MAX_INPUTS: usize = 255;
pub const MAX_OUTPUTS: usize = 255;
/// Serialised size in bytes, witnesses included.
pub const MAX_TX_BYTES: usize = 100_000;

/// `lock_data = BLAKE3(public key)`; witness `public key (32) | signature (64)`.
pub const LOCK_SINGLE: u8 = 1;
/// m-of-n, 1 <= m <= n <= 15; see `check_witness` for the witness layout.
pub const LOCK_MULTISIG: u8 = 2;

/// `H(tag, parts...)` of spec/node.md section 3: BLAKE3 over the tag bytes
/// followed by the parts, without separator. The tags are distinct fixed
/// strings, so values of different kinds cannot collide.
pub fn h(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(tag);
    for p in parts {
        hasher.update(p);
    }
    *hasher.finalize().as_bytes()
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TxIn {
    pub prev_txid: [u8; 32],
    pub prev_index: u32,
    /// Not interpreted for a coinbase; otherwise see `check_witness`.
    pub witness: Vec<u8>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TxOut {
    pub amount: u64,
    pub lock_type: u8,
    pub lock_data: [u8; 32],
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Transaction {
    /// Must be 1 for an ordinary transaction.
    pub version: u8,
    pub inputs: Vec<TxIn>,
    pub outputs: Vec<TxOut>,
}

impl Transaction {
    /// Full serialisation, witnesses included: what the block commits to.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.version);
        write_varint(&mut out, self.inputs.len() as u64);
        for i in &self.inputs {
            out.extend_from_slice(&i.prev_txid);
            out.extend_from_slice(&i.prev_index.to_le_bytes());
            write_varint(&mut out, i.witness.len() as u64);
            out.extend_from_slice(&i.witness);
        }
        write_varint(&mut out, self.outputs.len() as u64);
        for o in &self.outputs {
            out.extend_from_slice(&o.amount.to_le_bytes());
            out.push(o.lock_type);
            out.extend_from_slice(&o.lock_data);
        }
        out
    }

    /// Serialisation without the witness fields: what `txid` commits to.
    pub fn encode_no_witness(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.version);
        write_varint(&mut out, self.inputs.len() as u64);
        for i in &self.inputs {
            out.extend_from_slice(&i.prev_txid);
            out.extend_from_slice(&i.prev_index.to_le_bytes());
        }
        write_varint(&mut out, self.outputs.len() as u64);
        for o in &self.outputs {
            out.extend_from_slice(&o.amount.to_le_bytes());
            out.push(o.lock_type);
            out.extend_from_slice(&o.lock_data);
        }
        out
    }

    /// `txid = H("txid", serialisation without witnesses)` (spec/node.md
    /// section 5.1). The coinbase is the exception: its witness (height and
    /// extranonce) is committed too, so that every coinbase txid is unique
    /// (section 5.3). A coinbase carries no signature, so there is no
    /// signature malleability to exclude.
    pub fn txid(&self) -> [u8; 32] {
        if self.is_coinbase() { h(b"txid", &[&self.encode()]) } else { h(b"txid", &[&self.encode_no_witness()]) }
    }

    /// Signature message of one input (spec/node.md section 5.2): binds the
    /// chain, the transaction, the input position and the exact output spent
    /// (amount, lock), so a signature cannot be replayed on another chain,
    /// input or output. Integers are little-endian.
    pub fn sighash(&self, chain_id: &[u8; 32], input_index: u32, spent: &TxOut) -> [u8; 32] {
        h(
            b"sighash",
            &[
                chain_id,
                &self.txid(),
                &input_index.to_le_bytes(),
                &spent.amount.to_le_bytes(),
                &[spent.lock_type],
                &spent.lock_data,
            ],
        )
    }

    /// Decodes one transaction that fills `buf` exactly.
    pub fn decode(buf: &[u8]) -> Result<Transaction, String> {
        let mut r = Reader::new(buf);
        let tx = Self::decode_from(&mut r)?;
        r.expect_done().map_err(|e| e.to_string())?;
        Ok(tx)
    }

    /// Reads one transaction, leaving the reader right after it (a block holds
    /// a sequence of transactions). Counts above 1 000 000 are refused early;
    /// the limits of section 5.4 are enforced by the caller. The vectors
    /// are preallocated for at most 4096 entries, whatever the count.
    pub fn decode_from(r: &mut Reader) -> Result<Transaction, String> {
        let version = r.read_u8().map_err(|e| e.to_string())?;
        let n_in = r.read_varint().map_err(|e| e.to_string())?;
        if n_in > 1_000_000 {
            return Err("input count out of range".into());
        }
        let mut inputs = Vec::with_capacity((n_in as usize).min(4096));
        for _ in 0..n_in {
            let prev_txid: [u8; 32] = r.read_exact(32).map_err(|e| e.to_string())?.try_into().unwrap();
            let prev_index = r.read_u32().map_err(|e| e.to_string())?;
            let wlen = r.read_varint().map_err(|e| e.to_string())?;
            let witness = r.read_exact(wlen as usize).map_err(|e| e.to_string())?.to_vec();
            inputs.push(TxIn { prev_txid, prev_index, witness });
        }
        let n_out = r.read_varint().map_err(|e| e.to_string())?;
        if n_out > 1_000_000 {
            return Err("output count out of range".into());
        }
        let mut outputs = Vec::with_capacity((n_out as usize).min(4096));
        for _ in 0..n_out {
            let amount = r.read_u64().map_err(|e| e.to_string())?;
            let lock_type = r.read_u8().map_err(|e| e.to_string())?;
            let lock_data: [u8; 32] = r.read_exact(32).map_err(|e| e.to_string())?.try_into().unwrap();
            outputs.push(TxOut { amount, lock_type, lock_data });
        }
        Ok(Transaction { version, inputs, outputs })
    }

    /// Shape test of spec/node.md section 5.3: one input with an all-zero
    /// `prev_txid` and `prev_index` 0xFFFFFFFF.
    pub fn is_coinbase(&self) -> bool {
        self.inputs.len() == 1 && self.inputs[0].prev_txid == [0u8; 32] && self.inputs[0].prev_index == 0xFFFF_FFFF
    }

    /// `lock_data` of an m-of-n lock: `H("multisig", m, n, keys in order)`
    /// with m and n one byte each (spec/node.md section 5.2).
    pub fn multisig_lock_data(m: u8, keys: &[[u8; 32]]) -> [u8; 32] {
        let mut parts: Vec<u8> = Vec::with_capacity(2 + 32 * keys.len());
        parts.push(m);
        parts.push(keys.len() as u8);
        for k in keys {
            parts.extend_from_slice(k);
        }
        h(b"multisig", &[&parts])
    }

    /// Checks the witness of input `input_index` against the output it spends
    /// (spec/node.md section 5.2). `chain_id` is the genesis block id.
    ///
    /// Single key: 96 bytes, public key then signature. Multisig: `m (u8) |
    /// n (u8) | n public keys (32 each) | m entries of key index (u8) and
    /// signature (64)`. Lengths and the lock commitment are checked before any
    /// signature, because they are cheap and signatures are not. The verify
    /// call is the cofactorless Ed25519 equation of section 3 (not the strict
    /// variant), as the specification requires.
    pub fn check_witness(&self, chain_id: &[u8; 32], input_index: u32, spent: &TxOut) -> Result<(), String> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};
        let msg = self.sighash(chain_id, input_index, spent);
        let w = &self.inputs[input_index as usize].witness;
        match spent.lock_type {
            LOCK_SINGLE => {
                if w.len() != 96 {
                    return Err("single-key witness must be 32 + 64 bytes".into());
                }
                let key: [u8; 32] = w[..32].try_into().unwrap();
                let sig: [u8; 64] = w[32..].try_into().unwrap();
                if *blake3::hash(&key).as_bytes() != spent.lock_data {
                    return Err("public key does not match the lock".into());
                }
                let vk = VerifyingKey::from_bytes(&key).map_err(|_| "public key not on the curve")?;
                let sig = Signature::from_bytes(&sig);
                vk.verify(&msg, &sig).map_err(|_| "signature does not verify".to_string())
            }
            LOCK_MULTISIG => {
                if w.len() < 2 {
                    return Err("multisig witness too short".into());
                }
                let m = w[0];
                let n = w[1];
                if n == 0 || n > 15 || m == 0 || m > n {
                    return Err("multisig parameters out of range".into());
                }
                let need = 2 + 32 * n as usize + 65 * m as usize;
                if w.len() != need {
                    return Err("multisig witness length does not match m and n".into());
                }
                let keys: Vec<[u8; 32]> =
                    (0..n as usize).map(|i| w[2 + 32 * i..2 + 32 * (i + 1)].try_into().unwrap()).collect();
                let mut parts: Vec<u8> = vec![m, n];
                for k in &keys {
                    parts.extend_from_slice(k);
                }
                if h(b"multisig", &[&parts]) != spent.lock_data {
                    return Err("multisig parameters do not match the lock".into());
                }
                // Strictly increasing indexes below n: no key can sign twice.
                let mut last: i16 = -1;
                for j in 0..m as usize {
                    let base = 2 + 32 * n as usize + 65 * j;
                    let idx = w[base] as i16;
                    let sig: [u8; 64] = w[base + 1..base + 65].try_into().unwrap();
                    if idx <= last || idx >= n as i16 {
                        return Err("signature indexes must increase".into());
                    }
                    last = idx;
                    let vk =
                        VerifyingKey::from_bytes(&keys[idx as usize]).map_err(|_| "public key not on the curve")?;
                    let sig = Signature::from_bytes(&sig);
                    vk.verify(&msg, &sig).map_err(|_| "signature does not verify".to_string())?;
                }
                Ok(())
            }
            _ => Err("unknown lock type".into()),
        }
    }
}
