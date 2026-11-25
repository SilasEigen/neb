//! The chain state as bytes, so that a node does not replay its whole chain at every start.
//!
//! The bytes hold what the blocks alone do not give at once: the unspent outputs, the three sums of spec/node.md
//! section 6.7, what the last `MAX_REORG` blocks need to be disconnected, and the kept model levels. They name the
//! chain and the tip they belong to, and end with a BLAKE3 hash of all that precedes it.
//!
//! Local storage only: no rule depends on these bytes. A reader that finds anything unexpected returns an error, and
//! the caller replays the chain from the genesis block instead, which gives the same state.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use neb_codec::Reader;
use neb_pow::shares::{Arch, Model};

use super::{ChainParams, ChainState, MAX_REORG, OutPoint, Undo, Utxo};
use crate::block::Block;
use crate::model::{MODEL_DEPTH, Models};

const MAGIC: &[u8; 8] = b"NEBSTAT1";
/// Magic, chain id, height and tip id.
const HEAD: usize = 8 + 32 + 4 + 32;
const HASH: usize = 32;

fn put_outpoint(out: &mut Vec<u8>, op: &OutPoint) {
    out.extend_from_slice(&op.txid);
    out.extend_from_slice(&op.index.to_le_bytes());
}

fn put_utxo(out: &mut Vec<u8>, u: &Utxo) {
    out.extend_from_slice(&u.amount.to_le_bytes());
    out.push(u.lock_type);
    out.extend_from_slice(&u.lock_data);
    out.extend_from_slice(&u.height.to_le_bytes());
    out.push(u.coinbase as u8);
}

type Read<T> = Result<T, String>;

fn take<'a>(r: &mut Reader<'a>, n: usize) -> Read<&'a [u8]> {
    r.read_exact(n).map_err(|e| e.to_string())
}

fn u32_of(r: &mut Reader) -> Read<u32> {
    r.read_u32().map_err(|e| e.to_string())
}

fn u128_of(r: &mut Reader) -> Read<u128> {
    Ok(u128::from_le_bytes(take(r, 16)?.try_into().expect("16 bytes")))
}

fn outpoint(r: &mut Reader) -> Read<OutPoint> {
    Ok(OutPoint { txid: take(r, 32)?.try_into().expect("32 bytes"), index: u32_of(r)? })
}

fn utxo(r: &mut Reader) -> Read<Utxo> {
    let amount = r.read_u64().map_err(|e| e.to_string())?;
    let lock_type = r.read_u8().map_err(|e| e.to_string())?;
    let lock_data = take(r, 32)?.try_into().expect("32 bytes");
    let height = u32_of(r)?;
    let coinbase = match r.read_u8().map_err(|e| e.to_string())? {
        0 => false,
        1 => true,
        _ => return Err("an output flag is neither 0 nor 1".into()),
    };
    Ok(Utxo { amount, lock_type, lock_data, height, coinbase })
}

/// A count read from the file, refused when the bytes left cannot hold that many items of `each` bytes: a damaged
/// count never sizes an allocation.
fn count(r: &mut Reader, each: usize) -> Read<usize> {
    let n = u32_of(r)? as usize;
    if n.saturating_mul(each) > r.remaining() {
        return Err("a count is larger than the file".into());
    }
    Ok(n)
}

impl ChainState {
    /// The state as bytes; see the module. The models shared by several levels are written once.
    pub fn snapshot(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.chain_id);
        out.extend_from_slice(&self.height().to_le_bytes());
        out.extend_from_slice(&self.tip_id());
        for sum in [self.unspent_total, self.coinbase_total, self.fees_total] {
            out.extend_from_slice(&sum.to_le_bytes());
        }
        out.extend_from_slice(&(self.utxos.len() as u32).to_le_bytes());
        for (op, u) in &self.utxos {
            put_outpoint(&mut out, op);
            put_utxo(&mut out, u);
        }
        let kept = self.undos.len().min(MAX_REORG as usize);
        out.extend_from_slice(&(kept as u32).to_le_bytes());
        for undo in &self.undos[self.undos.len() - kept..] {
            out.extend_from_slice(&(undo.spent.len() as u32).to_le_bytes());
            for (op, u) in &undo.spent {
                put_outpoint(&mut out, op);
                put_utxo(&mut out, u);
            }
            out.extend_from_slice(&(undo.created.len() as u32).to_le_bytes());
            for op in &undo.created {
                put_outpoint(&mut out, op);
            }
            out.extend_from_slice(&undo.coinbase_out.to_le_bytes());
            out.extend_from_slice(&undo.fees.to_le_bytes());
        }
        // A level shares with its parent every model its block did not update: each model is written once, and a
        // level is the list of the positions of its models.
        let mut table: Vec<&Arc<Model>> = Vec::new();
        let mut levels: Vec<Vec<u32>> = Vec::new();
        for level in &self.models {
            let mut positions = Vec::new();
            for model in level.iter() {
                let at = table.iter().position(|m| Arc::ptr_eq(m, model)).unwrap_or_else(|| {
                    table.push(model);
                    table.len() - 1
                });
                positions.push(at as u32);
            }
            levels.push(positions);
        }
        out.extend_from_slice(&(table.len() as u32).to_le_bytes());
        for model in table {
            out.extend_from_slice(&model.arch().id().unwrap_or(0).to_le_bytes());
            out.extend(model.weights().iter().map(|w| *w as u8));
            for r in model.residuals() {
                out.extend_from_slice(&r.to_le_bytes());
            }
        }
        out.extend_from_slice(&(levels.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.params.slots() as u32).to_le_bytes());
        for position in levels.iter().flatten() {
            out.extend_from_slice(&position.to_le_bytes());
        }
        let hash = blake3::hash(&out);
        out.extend_from_slice(hash.as_bytes());
        out
    }

    /// The height of the tip that `bytes` name, when they start as a state does. Nothing else is checked.
    pub fn snapshot_height(bytes: &[u8]) -> Option<u32> {
        (bytes.len() >= HEAD + HASH && &bytes[..8] == MAGIC)
            .then(|| u32::from_le_bytes(bytes[40..44].try_into().expect("4 bytes")))
    }

    /// The state that `bytes` hold, for the best chain `blocks` from the genesis block to the tip they name. Fails
    /// on bytes that are damaged, that belong to another chain or another tip, or whose content does not add up;
    /// the caller then replays the chain. `verify_pow` is true, as in a state built from the genesis block.
    pub fn from_snapshot(params: ChainParams, blocks: Vec<Block>, bytes: &[u8]) -> Result<ChainState, String> {
        if bytes.len() < HEAD + HASH || &bytes[..8] != MAGIC {
            return Err("not a state file".into());
        }
        let (body, hash) = bytes.split_at(bytes.len() - HASH);
        if blake3::hash(body).as_bytes() != hash {
            return Err("the hash of the state file does not match".into());
        }
        let mut r = Reader::new(&body[8..]);
        let chain_id: [u8; 32] = take(&mut r, 32)?.try_into().expect("32 bytes");
        let height = u32_of(&mut r)?;
        let tip: [u8; 32] = take(&mut r, 32)?.try_into().expect("32 bytes");
        let genesis = blocks.first().ok_or("no block")?;
        if chain_id != genesis.id() {
            return Err("the state belongs to another chain".into());
        }
        if blocks.len() != height as usize + 1 || blocks[height as usize].id() != tip {
            return Err("the state belongs to another tip".into());
        }
        let (unspent_total, coinbase_total, fees_total) = (u128_of(&mut r)?, u128_of(&mut r)?, u128_of(&mut r)?);
        let mut utxos = BTreeMap::new();
        let mut sum: u128 = 0;
        for _ in 0..count(&mut r, 36 + 46)? {
            let (op, u) = (outpoint(&mut r)?, utxo(&mut r)?);
            sum += u.amount as u128;
            if utxos.insert(op, u).is_some() {
                return Err("an output is listed twice".into());
            }
        }
        if sum != unspent_total || unspent_total + fees_total != coinbase_total {
            return Err("the sums of the state do not add up".into());
        }
        let mut undos = Vec::new();
        for _ in 0..count(&mut r, 4 + 4 + 32)? {
            let mut spent = Vec::new();
            for _ in 0..count(&mut r, 36 + 46)? {
                spent.push((outpoint(&mut r)?, utxo(&mut r)?));
            }
            let mut created = Vec::new();
            for _ in 0..count(&mut r, 36)? {
                created.push(outpoint(&mut r)?);
            }
            undos.push(Undo { spent, created, coinbase_out: u128_of(&mut r)?, fees: u128_of(&mut r)? });
        }
        if undos.len() > height as usize {
            return Err("more undo records than blocks".into());
        }
        let mut table: Vec<Arc<Model>> = Vec::new();
        for _ in 0..count(&mut r, 4)? {
            let arch = match u32_of(&mut r)? {
                0 => Arch::V1,
                id => Arch::from_id(id).ok_or("unknown model architecture")?,
            };
            let n = arch.weights();
            let weights: Vec<i8> = take(&mut r, n)?.iter().map(|b| *b as i8).collect();
            let mut residuals = Vec::with_capacity(n);
            for _ in 0..n {
                residuals.push(u32_of(&mut r)? as i32);
            }
            table.push(Arc::new(Model::from_parts(arch, &weights, &residuals)?));
        }
        let level_count = u32_of(&mut r)? as usize;
        let k = u32_of(&mut r)? as usize;
        if k != params.slots() || level_count == 0 || level_count > (MODEL_DEPTH as usize + 1).min(blocks.len()) {
            return Err("the model levels do not fit this chain".into());
        }
        let mut models: VecDeque<Models> = VecDeque::new();
        for _ in 0..level_count {
            let mut level = Vec::with_capacity(k);
            for _ in 0..k {
                level.push(
                    table.get(u32_of(&mut r)? as usize).ok_or("a level names a model that is not listed")?.clone(),
                );
            }
            models.push_back(Arc::from(level));
        }
        r.expect_done().map_err(|e| e.to_string())?;
        Ok(ChainState {
            params,
            chain_id,
            verify_pow: true,
            blocks,
            undos,
            models,
            utxos,
            unspent_total,
            coinbase_total,
            fees_total,
        })
    }
}
