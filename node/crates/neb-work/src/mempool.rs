//! The mempool: unconfirmed transactions that a block template may include.
//!
//! Local policy, not consensus: the checks mirror the transaction rules of
//! spec/node.md section 6.4 so that an accepted transaction is valid in the
//! next block, and the size is bounded (spec/node.md section 8.3). Only
//! transactions that spend confirmed outputs are admitted; a transaction
//! that spends an unconfirmed output is rejected as `UnknownInput`. A
//! transaction pays at least `MIN_FEE_PER_BYTE` per serialised byte. When the
//! pool is full, a transaction of a higher fee rate evicts the lowest ones;
//! candidates are ordered by fee rate. Iteration uses ordered maps and a txid
//! tie-break, so templates are deterministic.

use std::collections::{BTreeMap, BTreeSet};

use neb_chain::OutPoint;
use neb_chain::state::{COINBASE_MATURITY, ChainState};
use neb_chain::tx::{LOCK_MULTISIG, LOCK_SINGLE, MAX_INPUTS, MAX_OUTPUTS, MAX_TX_BYTES, Transaction};

/// Limit of serialised transaction bytes held: 300 MiB (spec/node.md section
/// 8.3).
pub const MAX_MEMPOOL_BYTES: usize = 300 * 1024 * 1024;

/// Smallest fee a transaction pays per serialised byte, in base units. Local
/// policy: a block may still include a transaction below it. The fee of the
/// wallet (10 000 base units) covers a transaction of up to 10 000 bytes.
pub const MIN_FEE_PER_BYTE: u64 = 1;

/// The fee rate of `fee` for `bytes`, in base units per 10^6 bytes, the key the
/// pool orders and evicts by.
fn fee_rate(fee: u64, bytes: usize) -> u128 {
    fee as u128 * 1_000_000 / bytes.max(1) as u128
}

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
    /// The fee is below `MIN_FEE_PER_BYTE` per byte.
    LowFee,
    /// The pool is full of transactions of an equal or higher fee rate.
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
            MempoolError::LowFee => write!(f, "fee below {MIN_FEE_PER_BYTE} base unit per byte"),
            MempoolError::Full => write!(f, "mempool is full of transactions with an equal or higher fee rate"),
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
    /// The fee rate of each held transaction, and the same pairs ordered by rate
    /// (lowest first) for eviction.
    rates: BTreeMap<[u8; 32], u128>,
    by_rate: BTreeSet<(u128, [u8; 32])>,
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
        Mempool {
            txs: BTreeMap::new(),
            spent: BTreeMap::new(),
            rates: BTreeMap::new(),
            by_rate: BTreeSet::new(),
            bytes: 0,
            max_bytes: MAX_MEMPOOL_BYTES,
        }
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
    /// size, which cost nothing, then per input the lookup and maturity (for
    /// the next block), the fee and the room in the pool, and last the
    /// signatures, which are the expensive step. When the pool is full, the
    /// transactions of the lowest fee rates leave it if the new one pays a
    /// higher rate than each of them.
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
        let next_height = state.height() + 1;
        let mut sum_in: u128 = 0;
        let mut spent_here: Vec<OutPoint> = Vec::new();
        for input in &tx.inputs {
            let op = OutPoint { txid: input.prev_txid, index: input.prev_index };
            if spent_here.contains(&op) || self.spent.contains_key(&op) {
                return Err(MempoolError::Conflict(op));
            }
            let utxo = state.utxo(&op).ok_or(MempoolError::UnknownInput(op))?;
            if utxo.coinbase && next_height < utxo.height + COINBASE_MATURITY {
                return Err(MempoolError::Immature(op));
            }
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
        if (fee as u128) < bytes as u128 * MIN_FEE_PER_BYTE as u128 {
            return Err(MempoolError::LowFee);
        }
        let rate = fee_rate(fee, bytes);
        let evict = self.room_for(bytes, rate).ok_or(MempoolError::Full)?;
        for (index, op) in spent_here.iter().enumerate() {
            let utxo = state.utxo(op).expect("looked up above");
            tx.check_witness(&state.chain_id, index as u32, &utxo.as_txout())
                .map_err(|why| MempoolError::Witness { index, why })?;
        }
        for old in evict {
            self.drop_tx(&old);
        }
        for op in &spent_here {
            self.spent.insert(*op, id);
        }
        self.bytes += bytes;
        self.txs.insert(id, tx.clone());
        self.rates.insert(id, rate);
        self.by_rate.insert((rate, id));
        Ok(fee)
    }

    /// The transactions to evict, lowest fee rate first, so that `bytes` more
    /// fit; `None` when that would evict a transaction whose rate is not below
    /// `rate`.
    fn room_for(&self, bytes: usize, rate: u128) -> Option<Vec<[u8; 32]>> {
        let mut free = self.max_bytes.checked_sub(self.bytes)?;
        let mut evict = Vec::new();
        let mut lowest = self.by_rate.iter();
        while free < bytes {
            let (r, id) = lowest.next()?;
            if *r >= rate {
                return None;
            }
            free += self.txs[id].encode().len();
            evict.push(*id);
        }
        Some(evict)
    }

    /// Removes one transaction and its index entries; returns it.
    fn drop_tx(&mut self, id: &[u8; 32]) -> Option<Transaction> {
        let tx = self.txs.remove(id)?;
        self.bytes -= tx.encode().len();
        for input in &tx.inputs {
            let op = OutPoint { txid: input.prev_txid, index: input.prev_index };
            if self.spent.get(&op) == Some(id) {
                self.spent.remove(&op);
            }
        }
        if let Some(rate) = self.rates.remove(id) {
            self.by_rate.remove(&(rate, *id));
        }
        Some(tx)
    }

    /// Transactions for a template: highest fee rate first, txid as
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
                let fee = sum_in.saturating_sub(sum_out).min(u64::MAX as u128) as u64;
                Some((fee_rate(fee, tx.encode().len()), tx.txid(), tx))
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
    /// transaction that spends an input the block spent. Each removal looks up
    /// the transaction's own inputs: O(block inputs * log of the pool size).
    pub fn remove_confirmed(&mut self, txs: &[Transaction]) {
        for tx in txs {
            if tx.is_coinbase() {
                continue;
            }
            self.drop_tx(&tx.txid());
            for input in &tx.inputs {
                let op = OutPoint { txid: input.prev_txid, index: input.prev_index };
                if let Some(conflict) = self.spent.get(&op).copied() {
                    self.drop_tx(&conflict);
                }
            }
        }
    }
}
