//! Chain state: the UTXO set, connect and disconnect of blocks, the
//! conservation invariant and reorganisations (spec/node.md sections 6.1,
//! 6.4, 6.5 and 6.7), plus the model of the recent levels of the best chain
//! (spec/shares.md section 11.4).
//!
//! Usage contract. `check_header` applies the header rules to a child of the
//! tip. `connect_block` then checks the body, the proof of work and the share
//! list, and applies the transactions; callers that already verified a block
//! use `connect_verified`. `ChainState` is a single-owner value without
//! locking. Everything is deterministic: ordered maps only, no clock (the
//! current time is a parameter), no floating point.
//!
//! The genesis block is taken from the network parameters and is not checked
//! beyond the shape needed to build the state (spec/node.md section 6.6).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

/// A block removed by a reorganisation: its id and its model.
pub type RemovedBlock = ([u8; 32], Arc<Model>);

use neb_block::{Header, Target};
use neb_difficulty::Params as DifficultyParams;
use neb_pow::shares::{Model, ShareBuffers, SharedDataset};

use crate::block::{Block, MAX_BLOCK_BYTES, MAX_SHARES_PER_BLOCK, ShareRecord};
use crate::model::{self, MODEL_DEPTH, ProofError, share_key, share_target};
use crate::tx::{COIN, LOCK_MULTISIG, LOCK_SINGLE, MAX_INPUTS, MAX_OUTPUTS, MAX_TX_BYTES, TxOut};

/// A coinbase output created at height c is spendable in a block of height
/// c + 100 or later (spec/node.md section 5.3).
pub const COINBASE_MATURITY: u32 = 100;
/// Most blocks a reorganisation may remove from the best chain (spec/node.md
/// section 6.5): one hour at the 60 s spacing.
pub const MAX_REORG: u32 = 60;
/// Blocks per halving era: 312.5 days at 60 s per block.
const HALVING: u32 = 450_000;

/// Block reward in base units (spec/node.md section 6.3): 20 coins, shifted
/// right once per era. The guard keeps the shift below 64; the value is
/// already 0 from era 31 on.
pub fn subsidy(height: u32) -> u64 {
    let era = height / HALVING;
    if era >= 64 { 0 } else { (20 * COIN) >> era }
}

/// Project share of a reward, `floor(reward * 5 / 100)` in base units
/// (spec/node.md section 6.3). The product stays below 2^64 because a reward
/// is at most 20 * 10^8.
pub fn project_share(reward: u64) -> u64 {
    reward * 5 / 100
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
/// Reference to an output: the txid of its transaction and its position.
pub struct OutPoint {
    pub txid: [u8; 32],
    pub index: u32,
}

#[derive(Clone, PartialEq, Eq, Debug)]
/// An unspent output with what maturity needs.
pub struct Utxo {
    pub amount: u64,
    pub lock_type: u8,
    pub lock_data: [u8; 32],
    /// Height of the block that created it.
    pub height: u32,
    pub coinbase: bool,
}

impl Utxo {
    pub fn as_txout(&self) -> TxOut {
        TxOut { amount: self.amount, lock_type: self.lock_type, lock_data: self.lock_data }
    }
}

#[derive(Clone, Debug)]
/// Network parameters the rules depend on (spec/node.md section 6.6).
pub struct ChainParams {
    pub difficulty: DifficultyParams,
    /// `lock_data` of the multisig lock that the project-share output must
    /// carry (spec/node.md section 6.3).
    pub project_share_lock: [u8; 32],
    /// The samples the shares of the network draw from (spec/shares.md
    /// section 4).
    pub dataset: SharedDataset,
}

#[derive(Debug)]
/// Why a block cannot be connected. Any of them makes the block invalid for
/// the current tip, except that `Conservation` and `Genesis` indicate an
/// internal inconsistency or bad parameters rather than a bad block.
pub enum ChainError {
    Decode(String),
    /// The parent of the block is not the current tip (or the reorganisation
    /// ancestor).
    ParentMismatch,
    BadBody(&'static str),
    BodyRootMismatch,
    BodyTooLarge,
    TxTooLarge {
        index: usize,
    },
    UnknownInput(OutPoint),
    DoubleSpend(OutPoint),
    Immature(OutPoint),
    Witness {
        index: usize,
        why: String,
    },
    BadCoinbase(&'static str),
    RewardTooHigh,
    ReorgTooDeep,
    TimeNotAdvancing,
    Conservation,
    Genesis(&'static str),
    Proof(ProofError),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::Decode(e) => write!(f, "decode: {e}"),
            ChainError::ParentMismatch => write!(f, "parent is not the tip"),
            ChainError::BadBody(e) => write!(f, "bad body: {e}"),
            ChainError::BodyRootMismatch => write!(f, "body_root does not commit to the body"),
            ChainError::BodyTooLarge => write!(f, "body over 1 000 000 bytes"),
            ChainError::TxTooLarge { index } => write!(f, "transaction {index} over 100 000 bytes"),
            ChainError::UnknownInput(op) => write!(f, "input {op:?} not in the utxo set"),
            ChainError::DoubleSpend(op) => write!(f, "input {op:?} spent twice"),
            ChainError::Immature(op) => write!(f, "coinbase input {op:?} not mature"),
            ChainError::Witness { index, why } => write!(f, "input {index}: {why}"),
            ChainError::BadCoinbase(why) => write!(f, "coinbase: {why}"),
            ChainError::RewardTooHigh => write!(f, "coinbase pays more than reward plus fees"),
            ChainError::ReorgTooDeep => write!(f, "reorganisation over 60 blocks"),
            ChainError::TimeNotAdvancing => write!(f, "block time not after the median"),
            ChainError::Conservation => write!(f, "conservation invariant broken"),
            ChainError::Genesis(why) => write!(f, "genesis: {why}"),
            ChainError::Proof(e) => write!(f, "proof: {e}"),
        }
    }
}

impl std::error::Error for ChainError {}

#[derive(Debug)]
/// Header rules 1 to 5 of spec/node.md section 6.1, in the order they are
/// checked.
pub enum HeaderError {
    Version,
    UnknownParent,
    Difficulty(String),
    BadBits,
    StaleTime,
    FutureTime,
}

impl std::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeaderError::Version => write!(f, "header version is not 1"),
            HeaderError::UnknownParent => write!(f, "parent is not the tip"),
            HeaderError::Difficulty(e) => write!(f, "difficulty: {e}"),
            HeaderError::BadBits => write!(f, "bits is not the value the chain computes"),
            HeaderError::StaleTime => write!(f, "time not after the median of the last 11"),
            HeaderError::FutureTime => write!(f, "time more than 120 s ahead"),
        }
    }
}

impl std::error::Error for HeaderError {}

/// What a disconnect needs to undo one connected block. The block itself is
/// kept in `ChainState::blocks`.
#[derive(Debug)]
struct Undo {
    spent: Vec<(OutPoint, Utxo)>,
    created: Vec<OutPoint>,
    /// Sum of the coinbase outputs (counted in `coinbase_total`).
    coinbase_out: u128,
    /// Fees of the ordinary transactions (counted in `fees_total`).
    fees: u128,
}

/// The best chain from the genesis block to the tip, with its UTXO set.
pub struct ChainState {
    pub params: ChainParams,
    /// The genesis block id, used as `chain_id` in signature messages
    /// (spec/node.md section 5.2).
    pub chain_id: [u8; 32],
    /// When false the targets of the block and of its shares are not compared
    /// (tests of the other rules); the model step still runs. Always true in
    /// the node.
    pub verify_pow: bool,
    blocks: Vec<Block>,
    undos: Vec<Undo>,
    /// The model of each of the last `MODEL_DEPTH + 1` blocks of the best
    /// chain, the tip last (spec/shares.md section 11.4).
    models: VecDeque<Arc<Model>>,
    utxos: BTreeMap<OutPoint, Utxo>,
    // The three sums of spec/node.md section 6.7: U, C and F. u128 holds the
    // sum of any number of u64 amounts that can occur.
    unspent_total: u128,
    coinbase_total: u128,
    fees_total: u128,
}

impl ChainState {
    /// Builds the state from the genesis block. The genesis header must carry
    /// the anchor target and time of the difficulty parameters, and its body
    /// must match `body_root`. Its proof of work, reward and outputs are not
    /// checked (spec/node.md section 6.6); the initial model is derived from
    /// the genesis id.
    pub fn from_genesis(params: ChainParams, genesis: Block) -> Result<ChainState, ChainError> {
        if genesis.header.bits != params.difficulty.anchor_bits {
            return Err(ChainError::Genesis("bits is not the anchor target"));
        }
        if genesis.header.time != params.difficulty.anchor_time {
            return Err(ChainError::Genesis("time is not the anchor time"));
        }
        if !genesis.body_ok() {
            return Err(ChainError::Genesis("body_root does not commit to the body"));
        }
        let coinbase = genesis.txs.first().ok_or(ChainError::Genesis("no coinbase"))?;
        if !coinbase.is_coinbase() {
            return Err(ChainError::Genesis("first transaction is not the coinbase"));
        }
        let mut utxos = BTreeMap::new();
        let mut total: u128 = 0;
        let txid = coinbase.txid();
        for (index, out) in coinbase.outputs.iter().enumerate() {
            utxos.insert(
                OutPoint { txid, index: index as u32 },
                Utxo {
                    amount: out.amount,
                    lock_type: out.lock_type,
                    lock_data: out.lock_data,
                    height: 0,
                    coinbase: true,
                },
            );
            total += out.amount as u128;
        }
        let state = ChainState {
            chain_id: genesis.id(),
            verify_pow: true,
            models: VecDeque::from([Arc::new(Model::from_seed(&genesis.id()))]),
            blocks: vec![genesis],
            undos: Vec::new(),
            utxos,
            unspent_total: total,
            coinbase_total: total,
            fees_total: 0,
            params,
        };
        if !state.conservation() {
            return Err(ChainError::Conservation);
        }
        Ok(state)
    }

    /// Height of the tip; the genesis block has height 0.
    pub fn height(&self) -> u32 {
        (self.blocks.len() - 1) as u32
    }

    pub fn tip(&self) -> &Block {
        self.blocks.last().expect("genesis is always present")
    }

    pub fn tip_id(&self) -> [u8; 32] {
        self.tip().id()
    }

    pub fn tip_time(&self) -> u32 {
        self.tip().header.time
    }

    /// The best chain's UTXO set.
    pub fn utxo(&self, op: &OutPoint) -> Option<&Utxo> {
        self.utxos.get(op)
    }

    pub fn utxos(&self) -> &BTreeMap<OutPoint, Utxo> {
        &self.utxos
    }

    pub fn unspent_total(&self) -> u128 {
        self.unspent_total
    }

    pub fn coinbase_total(&self) -> u128 {
        self.coinbase_total
    }

    pub fn fees_total(&self) -> u128 {
        self.fees_total
    }

    /// The invariant of spec/node.md section 6.7: unspent amounts `U` plus all
    /// fees `F` equal all coinbase outputs `C`. A fee leaves the UTXO set when
    /// paid and re-enters it only when a coinbase claims it.
    pub fn conservation(&self) -> bool {
        self.unspent_total + self.fees_total == self.coinbase_total
    }

    /// The model of the block at `height` of the best chain, when it is among
    /// the last `MODEL_DEPTH + 1` kept levels.
    pub fn model_at(&self, height: u32) -> Option<Arc<Model>> {
        let back = self.height().checked_sub(height)? as usize;
        let index = self.models.len().checked_sub(1 + back)?;
        self.models.get(index).cloned()
    }

    pub fn tip_model(&self) -> Arc<Model> {
        self.models.back().expect("the genesis model is always present").clone()
    }

    /// Header rules 1 to 5 of spec/node.md section 6.1 for a child of the
    /// tip, cheapest first: version, parent, `bits` from the difficulty rule,
    /// then the time window. `now` is the node's Unix time in seconds, a
    /// local input: it only bounds how far ahead a block may be, so nodes
    /// with different clocks can disagree on a block that is near the limit
    /// until the limit passes. The proof of work and the shares need the
    /// model and are checked by `verify_block`.
    pub fn check_header(&self, header: &Header, now: u32) -> Result<(), HeaderError> {
        if header.version != 1 {
            return Err(HeaderError::Version);
        }
        if header.parent != self.tip_id() {
            return Err(HeaderError::UnknownParent);
        }
        let want = self
            .params
            .difficulty
            .next_bits(self.height(), self.tip_time())
            .map_err(|e| HeaderError::Difficulty(e.to_string()))?;
        if header.bits != want {
            return Err(HeaderError::BadBits);
        }
        // Median of the last 11 times (all of them when fewer exist): the
        // element at index n / 2 of the sorted times.
        let mut times: Vec<u32> = self.blocks.iter().rev().take(11).map(|b| b.header.time).collect();
        times.sort_unstable();
        let median = times[times.len() / 2];
        if header.time <= median {
            return Err(HeaderError::StaleTime);
        }
        if u64::from(header.time) > u64::from(now) + 120 {
            return Err(HeaderError::FutureTime);
        }
        Ok(())
    }

    /// The model, the bits and the share target of the round whose share
    /// headers have the parent `q` (spec/shares.md sections 11.2 and 11.5): `q`
    /// is the tip (the round of the next block's child, on the model of the
    /// tip) or the parent of the tip (the round the next block lists, on the
    /// model of the tip's parent). `None` for any other parent.
    pub fn share_round(&self, q: &[u8; 32]) -> Option<(Arc<Model>, u32, Target)> {
        let height = self.height();
        let (model, bits) = if *q == self.tip_id() {
            let bits = self.params.difficulty.next_bits(height, self.tip_time()).ok()?;
            (self.tip_model(), bits)
        } else if height >= 1 && *q == self.tip().header.parent {
            (self.model_at(height - 1)?, self.tip().header.bits)
        } else {
            return None;
        };
        Some((model, bits, share_target(bits).ok()?))
    }

    /// Whether `record` is a valid share of the round whose headers have the
    /// parent `q` (one ticket evaluation); `None` when `q` is neither the tip
    /// nor its parent. When the proof of work is not checked (tests) every
    /// record of a known round is valid.
    pub fn check_share(&self, q: &[u8; 32], record: &ShareRecord) -> Option<bool> {
        let (model, bits, target) = self.share_round(q)?;
        if !self.verify_pow {
            return Some(true);
        }
        let ticket = ShareBuffers::new().ticket(&model, &*self.params.dataset, &share_key(record, *q, bits));
        Some(target.admits(&ticket))
    }

    /// The records among `candidates` that are valid shares of the round of
    /// the tip (headers whose parent is the tip's parent, evaluated on the
    /// model of the tip's parent), for the share list of a child block:
    /// sorted, without duplicate, at most 256. Empty when the tip is the
    /// genesis block. Costs one ticket per candidate.
    pub fn valid_shares(&self, candidates: impl IntoIterator<Item = ShareRecord>) -> Vec<ShareRecord> {
        let mut records: Vec<ShareRecord> = candidates.into_iter().collect();
        records.sort_unstable();
        records.dedup();
        if self.height() == 0 {
            return Vec::new();
        }
        let q = self.tip().header.parent;
        records.retain(|r| self.check_share(&q, r) == Some(true));
        records.truncate(MAX_SHARES_PER_BLOCK);
        records
    }

    /// Proof of work and share list of a child of the tip (spec/shares.md
    /// sections 11.1 and 11.2); returns the model of the block. This is the
    /// expensive part of validation: one ticket for the block and one per
    /// share. It does not touch the state.
    pub fn verify_block(&self, block: &Block) -> Result<Arc<Model>, ChainError> {
        let height = self.height();
        let model_pp = self.model_at(height.saturating_sub(1)).expect("the tip and its parent keep their models");
        model::verify_block(
            &self.params,
            self.verify_pow,
            block,
            &self.tip().header,
            height == 0,
            &self.tip_model(),
            &model_pp,
        )
        .map_err(ChainError::Proof)
    }

    /// Connects a child of the tip: body checks, proof of work and shares,
    /// then the transactions, the coinbase and the conservation invariant, in
    /// the order of spec/node.md section 6.1 (steps 6 to 10). The caller has
    /// checked the header (see `check_header`). On error the state is
    /// unchanged (see `apply_block` for `Conservation`).
    pub fn connect_block(&mut self, block: &Block) -> Result<(), ChainError> {
        self.check_body(block)?;
        let model = self.verify_block(block)?;
        self.apply_block(block, model)
    }

    /// As `connect_block` for a block whose proof and shares were already
    /// verified with the same tip, which gave `model`; avoids evaluating the
    /// tickets twice.
    pub fn connect_verified(&mut self, block: &Block, model: Arc<Model>) -> Result<(), ChainError> {
        self.check_body(block)?;
        self.apply_block(block, model)
    }

    /// Checks that need no UTXO and no model, cheapest first: parent, body
    /// root, size, coinbase shape and the project-share output (spec/node.md
    /// sections 5.3 and 6.3).
    fn check_body(&self, block: &Block) -> Result<(), ChainError> {
        let height = self.height() + 1;
        if block.header.parent != self.tip_id() {
            return Err(ChainError::ParentMismatch);
        }
        if block.txs.is_empty() {
            return Err(ChainError::BadBody("no transactions"));
        }
        if !block.body_ok() {
            return Err(ChainError::BodyRootMismatch);
        }
        if block.encode().len() > MAX_BLOCK_BYTES {
            return Err(ChainError::BodyTooLarge);
        }
        let coinbase = &block.txs[0];
        if !coinbase.is_coinbase() {
            return Err(ChainError::BadCoinbase("first transaction is not the coinbase"));
        }
        for tx in &block.txs[1..] {
            if tx.is_coinbase() {
                return Err(ChainError::BadCoinbase("a second coinbase-shaped transaction"));
            }
        }

        let reward = subsidy(height);
        let share = project_share(reward);
        if share > 0 {
            if coinbase.outputs.len() != 2 {
                return Err(ChainError::BadCoinbase("the coinbase must have two outputs"));
            }
            let second = &coinbase.outputs[1];
            if second.amount != share
                || second.lock_type != LOCK_MULTISIG
                || second.lock_data != self.params.project_share_lock
            {
                return Err(ChainError::BadCoinbase("the project-share output is wrong"));
            }
        } else if coinbase.outputs.len() != 1 {
            return Err(ChainError::BadCoinbase("one output once the share rounds to zero"));
        }
        Ok(())
    }

    /// Transaction rules (spec/node.md section 6.4) and state update. All
    /// checks run on the pre-block UTXO set before anything is changed, so an
    /// output created in this block cannot be spent in it. Per transaction:
    /// cheap shape checks first, then per input the lookup, the maturity and
    /// the signature (the expensive step). Every error except `Conservation`
    /// is returned before the state is changed; `Conservation` cannot occur
    /// for checked amounts.
    fn apply_block(&mut self, block: &Block, model: Arc<Model>) -> Result<(), ChainError> {
        let height = self.height() + 1;
        let coinbase = &block.txs[0];
        let reward = subsidy(height);

        // Ordinary transactions first: they produce the fees that the coinbase
        // may claim.
        let mut spend: Vec<(OutPoint, Utxo)> = Vec::new();
        let mut created: Vec<(OutPoint, Utxo)> = Vec::new();
        let mut fees: u128 = 0;
        let mut spent_here: Vec<OutPoint> = Vec::new();
        for (index, tx) in block.txs.iter().enumerate().skip(1) {
            if tx.version != 1 {
                return Err(ChainError::BadBody("transaction version is not 1"));
            }
            if tx.inputs.is_empty() || tx.inputs.len() > MAX_INPUTS {
                return Err(ChainError::BadBody("input count out of range"));
            }
            if tx.outputs.is_empty() || tx.outputs.len() > MAX_OUTPUTS {
                return Err(ChainError::BadBody("output count out of range"));
            }
            if tx.encode().len() > MAX_TX_BYTES {
                return Err(ChainError::TxTooLarge { index });
            }
            for out in &tx.outputs {
                if out.amount == 0 {
                    return Err(ChainError::BadBody("zero-amount output"));
                }
                if out.lock_type != LOCK_SINGLE && out.lock_type != LOCK_MULTISIG {
                    return Err(ChainError::BadBody("unknown lock type"));
                }
            }
            let mut sum_in: u128 = 0;
            let mut sum_out: u128 = 0;
            for (i, input) in tx.inputs.iter().enumerate() {
                let op = OutPoint { txid: input.prev_txid, index: input.prev_index };
                if spent_here.contains(&op) {
                    return Err(ChainError::DoubleSpend(op));
                }
                let utxo = self.utxos.get(&op).cloned().ok_or(ChainError::UnknownInput(op))?;
                if utxo.coinbase && height < utxo.height + COINBASE_MATURITY {
                    return Err(ChainError::Immature(op));
                }
                tx.check_witness(&self.chain_id, i as u32, &utxo.as_txout())
                    .map_err(|why| ChainError::Witness { index: i, why })?;
                spent_here.push(op);
                sum_in += utxo.amount as u128;
                spend.push((op, utxo));
            }
            for out in &tx.outputs {
                sum_out += out.amount as u128;
            }
            if sum_out > sum_in {
                return Err(ChainError::BadBody("outputs above inputs"));
            }
            fees += sum_in - sum_out;
            let txid = tx.txid();
            for (index, out) in tx.outputs.iter().enumerate() {
                created.push((
                    OutPoint { txid, index: index as u32 },
                    Utxo {
                        amount: out.amount,
                        lock_type: out.lock_type,
                        lock_data: out.lock_data,
                        height,
                        coinbase: false,
                    },
                ));
            }
        }

        // The coinbase pays at most reward plus fees; the difference is
        // destroyed (spec/node.md section 5.3). Sums are u128: at most 255
        // inputs of u64 amounts per transaction cannot overflow.
        let mut coinbase_out: u128 = 0;
        for out in &coinbase.outputs {
            if out.amount == 0 {
                return Err(ChainError::BadCoinbase("zero-amount output"));
            }
            if out.lock_type != LOCK_SINGLE && out.lock_type != LOCK_MULTISIG {
                return Err(ChainError::BadCoinbase("unknown lock type"));
            }
            coinbase_out += out.amount as u128;
        }
        if coinbase_out > reward as u128 + fees {
            return Err(ChainError::RewardTooHigh);
        }
        let cbid = coinbase.txid();
        for (index, out) in coinbase.outputs.iter().enumerate() {
            created.push((
                OutPoint { txid: cbid, index: index as u32 },
                Utxo { amount: out.amount, lock_type: out.lock_type, lock_data: out.lock_data, height, coinbase: true },
            ));
        }

        // A created outpoint that already exists (in the UTXO set after the
        // spent outputs leave it, or among the outputs of this block) is a
        // duplicate (spec/node.md section 6.4). It is detected before the
        // first mutation, so an error leaves the state unchanged.
        let spent_set: BTreeSet<OutPoint> = spend.iter().map(|(op, _)| *op).collect();
        let mut created_set: BTreeSet<OutPoint> = BTreeSet::new();
        for (op, _) in &created {
            if (self.utxos.contains_key(op) && !spent_set.contains(op)) || !created_set.insert(*op) {
                return Err(ChainError::DoubleSpend(*op));
            }
        }

        // Apply: remove the spent, add the created.
        for (op, _) in &spend {
            self.utxos.remove(op);
        }
        for (op, utxo) in &created {
            self.utxos.insert(*op, utxo.clone());
        }
        // Fees leave the UTXO set; the coinbase outputs, which may claim them,
        // enter it.
        self.unspent_total = self.unspent_total + coinbase_out - fees;
        self.coinbase_total += coinbase_out;
        self.fees_total += fees;
        if !self.conservation() {
            return Err(ChainError::Conservation);
        }
        self.undos.push(Undo {
            spent: spend,
            created: created.iter().map(|(op, _)| *op).collect(),
            coinbase_out,
            fees,
        });
        self.blocks.push(block.clone());
        self.models.push_back(model);
        // Keep the last MODEL_DEPTH + 1 levels (spec/shares.md section 11.4).
        while self.models.len() > MODEL_DEPTH as usize + 1 {
            self.models.pop_front();
        }
        Ok(())
    }

    /// Undoes the tip: the exact inverse of `connect_block` on the UTXO set
    /// and the three sums. The model of the removed block is dropped and the
    /// deeper models are not regenerated, so the number of kept models shrinks
    /// by one. When fewer than two models remain (the tip and its parent are
    /// needed by `verify_block`), the kept models are rebuilt by replay.
    pub fn disconnect_tip(&mut self) -> Result<(), ChainError> {
        self.disconnect_one()?;
        if self.models.len() < 2.min(self.blocks.len()) {
            self.rebuild_models()?;
        }
        Ok(())
    }

    /// Rebuilds the kept models by replaying the shares of every block of the
    /// best chain from the genesis model (spec/shares.md section 11.4). The
    /// proofs were checked when the blocks were connected, so targets are not
    /// compared. Cost: the gradients of every listed share of the chain.
    fn rebuild_models(&mut self) -> Result<(), ChainError> {
        let mut models: VecDeque<Arc<Model>> = VecDeque::from([Arc::new(Model::from_seed(&self.chain_id))]);
        for h in 1..self.blocks.len() {
            let p = models.back().expect("the genesis model is present").clone();
            let pp = if h >= 2 { models[models.len() - 2].clone() } else { p.clone() };
            let next =
                model::verify_block(&self.params, false, &self.blocks[h], &self.blocks[h - 1].header, h == 1, &p, &pp)
                    .map_err(ChainError::Proof)?;
            models.push_back(next);
            while models.len() > MODEL_DEPTH as usize + 1 {
                models.pop_front();
            }
        }
        self.models = models;
        Ok(())
    }

    /// `disconnect_tip` without the model rebuild.
    fn disconnect_one(&mut self) -> Result<(), ChainError> {
        if self.height() == 0 {
            return Err(ChainError::BadBody("cannot disconnect the genesis"));
        }
        self.blocks.pop().expect("height > 0");
        let undo = self.undos.pop().expect("an undo per block");
        self.models.pop_back();
        for op in &undo.created {
            if self.utxos.remove(op).is_none() {
                return Err(ChainError::Conservation);
            }
        }
        for (op, utxo) in undo.spent {
            if self.utxos.insert(op, utxo).is_some() {
                return Err(ChainError::Conservation);
            }
        }
        let cb = undo.coinbase_out;
        let fees = undo.fees;
        self.unspent_total = self.unspent_total - cb + fees;
        self.coinbase_total -= cb;
        self.fees_total -= fees;
        if !self.conservation() {
            return Err(ChainError::Conservation);
        }
        Ok(())
    }

    /// Replaces the blocks above `ancestor_height` by `branch` (spec/node.md
    /// section 6.5). The first branch block must name the ancestor as its
    /// parent; at most `MAX_REORG` blocks may be removed. The branch blocks
    /// get the full `connect_block` checks, except the header rules. The
    /// caller decides by cumulative work whether to reorganise. When a branch
    /// block is invalid the former best chain is reconnected and the error returned.
    /// On success returns the removed blocks' ids and models, tip first.
    pub fn reorg(&mut self, ancestor_height: u32, branch: &[Block]) -> Result<Vec<RemovedBlock>, ChainError> {
        if ancestor_height >= self.height() {
            return Err(ChainError::BadBody("ancestor is not below the tip"));
        }
        if self.height() - ancestor_height > MAX_REORG {
            return Err(ChainError::ReorgTooDeep);
        }
        if branch.is_empty() {
            return Err(ChainError::BadBody("empty branch"));
        }
        let ancestor_id = self.blocks[ancestor_height as usize].id();
        if branch[0].header.parent != ancestor_id {
            return Err(ChainError::ParentMismatch);
        }
        // Earlier reorganisations to a shorter chain can leave fewer than
        // the last `MODEL_DEPTH + 1` models; the removed models and the
        // models of the ancestor are needed below.
        if self.models.len() < (self.blocks.len()).min(MODEL_DEPTH as usize + 1) {
            self.rebuild_models()?;
        }
        let old: Vec<Block> = self.blocks[ancestor_height as usize + 1..].to_vec();
        let removed: Vec<([u8; 32], Arc<Model>)> = (ancestor_height + 1..=self.height())
            .rev()
            .map(|h| (self.id_at(h), self.model_at(h).expect("models of the last levels are rebuilt above")))
            .collect();
        while self.height() > ancestor_height {
            self.disconnect_one()?;
        }
        for block in branch {
            if let Err(e) = self.connect_block(block) {
                while self.height() > ancestor_height {
                    self.disconnect_one()?;
                }
                for b in &old {
                    self.connect_block(b).expect("the old chain connects again");
                }
                return Err(e);
            }
        }
        Ok(removed)
    }

    /// The id of the block at `height` of the best chain; panics above the tip.
    pub fn id_at(&self, height: u32) -> [u8; 32] {
        self.blocks[height as usize].id()
    }
}
