//! The mempool: unconfirmed transactions that a block template may include.
//!
//! Local policy, not consensus: the checks mirror the transaction rules of
//! spec/node.md section 6.4 so that an accepted transaction is valid in the
//! next block, and the size is bounded (spec/node.md section 8.3). Only
//! transactions that spend confirmed outputs are admitted; a transaction
//! that spends an unconfirmed output is rejected as `UnknownInput`. No fee
//! minimum applies: candidates are ordered by absolute fee. Iteration uses
//! ordered maps and a txid tie-break, so templates are deterministic.

use std::collections::BTreeMap;

use neb_chain::OutPoint;
use neb_chain::state::{COINBASE_MATURITY, ChainState};
use neb_chain::tx::{LOCK_MULTISIG, LOCK_SINGLE, MAX_INPUTS, MAX_OUTPUTS, MAX_TX_BYTES, Transaction};

/// Limit of serialised transaction bytes held: 300 MiB (spec/node.md section
/// 8.3).
pub const MAX_MEMPOOL_BYTES: usize = 300 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
/// Why `Mempool::accept` refused a transaction. None of them is a ban reason
/// by itself: a transaction may be stale because a block just arrived.
pub enum MempoolError {
    CoinbaseShaped,
    Duplicate([u8; 32]),
    Shape(&'static str),
    /// The input is not in the UTXO set of the tip, or is unconfirmed.
    UnknownInput(OutPoint),
    /// Another mempool transaction, or the same one, spends the input.
    Conflict(OutPoint),
    Immature(OutPoint),
    Witness {
        index: usize,
        why: String,
    },
    Overspend,
    Full,
}

impl std::fmt::Display for MempoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MempoolError::CoinbaseShaped => write!(f, "a coinbase-shaped transaction"),
            MempoolError::Duplicate(id) => write!(f, "transaction {:02x}{:02x}.. already in the mempool", id[0], id[1]),
            MempoolError::Shape(e) => write!(f, "shape: {e}"),
            MempoolError::UnknownInput(op) => write!(f, "input {op:?} not confirmed"),
            MempoolError::Conflict(op) => write!(f, "input {op:?} already spent in the mempool"),
            MempoolError::Immature(op) => write!(f, "input {op:?} immature"),
            MempoolError::Witness { index, why } => write!(f, "input {index}: {why}"),
            MempoolError::Overspend => write!(f, "outputs above inputs"),
            MempoolError::Full => write!(f, "mempool is full"),
        }
    }
}

impl std::error::Error for MempoolError {}

/// Transactions by txid, with an index from each spent outpoint to its spender
/// so that conflicts are found without a scan. Not synchronised; the caller
/// serialises access.
pub struct Mempool {
    txs: BTreeMap<[u8; 32], Transaction>,
    spent: BTreeMap<OutPoint, [u8; 32]>,
    /// Serialised size of all held transactions.
    bytes: usize,
    /// Bound on `bytes`.
    pub max_bytes: usize,
}

impl Default for Mempool {
    fn default() -> Self {
        Self::new()
    }
}

impl Mempool {
    pub fn new() -> Mempool {
        Mempool { txs: BTreeMap::new(), spent: BTreeMap::new(), bytes: 0, max_bytes: MAX_MEMPOOL_BYTES }
    }

    pub fn with_limit(max_bytes: usize) -> Mempool {
        Mempool { max_bytes, ..Mempool::new() }
    }

    pub fn len(&self) -> usize {
        self.txs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.txs.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn tx(&self, id: &[u8; 32]) -> Option<&Transaction> {
        self.txs.get(id)
    }

    /// The transaction ids in the mempool, sorted.
    pub fn txids(&self) -> Vec<[u8; 32]> {
        self.txs.keys().copied().collect()
    }

    /// Checks a transaction against the tip of `state` and the mempool, then
    /// adds it; returns the fee in base units. Order: shape, duplicate and
    /// size, which cost nothing, then per input the lookup, maturity (for the
    /// next block) and the signature, which is the expensive step.
    pub fn accept(&mut self, state: &ChainState, tx: &Transaction) -> Result<u64, MempoolError> {
        if tx.is_coinbase() {
            return Err(MempoolError::CoinbaseShaped);
        }
        if tx.version != 1 {
            return Err(MempoolError::Shape("version is not 1"));
        }
        if tx.inputs.is_empty() || tx.inputs.len() > MAX_INPUTS {
            return Err(MempoolError::Shape("input count out of range"));
        }
        if tx.outputs.is_empty() || tx.outputs.len() > MAX_OUTPUTS {
            return Err(MempoolError::Shape("output count out of range"));
        }
        let bytes = tx.encode().len();
        if bytes > MAX_TX_BYTES {
            return Err(MempoolError::Shape("transaction over 100 000 bytes"));
        }
        let id = tx.txid();
        if self.txs.contains_key(&id) {
            return Err(MempoolError::Duplicate(id));
        }
        if self.bytes + bytes > self.max_bytes {
            return Err(MempoolError::Full);
        }
        let next_height = state.height() + 1;
        let mut sum_in: u128 = 0;
        let mut spent_here: Vec<OutPoint> = Vec::new();
        for (index, input) in tx.inputs.iter().enumerate() {
            let op = OutPoint { txid: input.prev_txid, index: input.prev_index };
            if spent_here.contains(&op) || self.spent.contains_key(&op) {
                return Err(MempoolError::Conflict(op));
            }
            let utxo = state.utxo(&op).ok_or(MempoolError::UnknownInput(op))?;
            if utxo.coinbase && next_height < utxo.height + COINBASE_MATURITY {
                return Err(MempoolError::Immature(op));
            }
            tx.check_witness(&state.chain_id, index as u32, &utxo.as_txout())
                .map_err(|why| MempoolError::Witness { index, why })?;
            spent_here.push(op);
            sum_in += utxo.amount as u128;
        }
        for out in &tx.outputs {
            if out.amount == 0 {
                return Err(MempoolError::Shape("zero-amount output"));
            }
            if out.lock_type != LOCK_SINGLE && out.lock_type != LOCK_MULTISIG {
                return Err(MempoolError::Shape("unknown lock type"));
            }
        }
        let sum_out: u128 = tx.outputs.iter().map(|o| o.amount as u128).sum();
        if sum_out > sum_in {
            return Err(MempoolError::Overspend);
        }
        let fee = (sum_in - sum_out) as u64;
        for op in &spent_here {
            self.spent.insert(*op, id);
        }
        self.bytes += bytes;
        self.txs.insert(id, tx.clone());
        Ok(fee)
    }

    /// Transactions for a template: highest absolute fee first, txid as
    /// tie-break, taken greedily within `budget` serialised bytes (a
    /// transaction that does not fit is skipped, a smaller one may follow).
    /// Transactions whose inputs left the UTXO set (after a block connects) are
    /// left out. Maturity and signatures are not rechecked.
    pub fn candidates(&self, state: &ChainState, budget: usize) -> Vec<Transaction> {
        let mut with_fee: Vec<(u128, [u8; 32], &Transaction)> = self
            .txs
            .values()
            .filter_map(|tx| {
                let mut sum_in: u128 = 0;
                for input in &tx.inputs {
                    let op = OutPoint { txid: input.prev_txid, index: input.prev_index };
                    sum_in += state.utxo(&op)?.amount as u128;
                }
                let sum_out: u128 = tx.outputs.iter().map(|o| o.amount as u128).sum();
                Some((sum_in.saturating_sub(sum_out), tx.txid(), tx))
            })
            .collect();
        with_fee.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut chosen = Vec::new();
        let mut used = 0usize;
        for (_, _, tx) in with_fee {
            let bytes = tx.encode().len();
            if used + bytes <= budget {
                used += bytes;
                chosen.push(tx.clone());
            }
        }
        chosen
    }

    /// Removes the transactions of a connected block and every mempool
    /// transaction that spends an input the block spent. Quadratic in the
    /// block's input count times the mempool size, which is small in practice.
    pub fn remove_confirmed(&mut self, txs: &[Transaction]) {
        let mut gone: Vec<[u8; 32]> = Vec::new();
        let mut freed: Vec<OutPoint> = Vec::new();
        for tx in txs {
            if tx.is_coinbase() {
                continue;
            }
            let id = tx.txid();
            if self.txs.remove(&id).is_some() {
                self.bytes -= tx.encode().len();
                gone.push(id);
            }
            for input in &tx.inputs {
                freed.push(OutPoint { txid: input.prev_txid, index: input.prev_index });
            }
        }
        if freed.is_empty() {
            return;
        }
        let conflicted: Vec<[u8; 32]> =
            self.spent.iter().filter(|(op, _)| freed.contains(op)).map(|(_, id)| *id).collect();
        for id in &conflicted {
            if let Some(tx) = self.txs.remove(id) {
                self.bytes -= tx.encode().len();
            }
        }
        for id in gone.iter().chain(conflicted.iter()) {
            self.spent.retain(|_, v| v != id);
        }
    }
}
