//! Block templates and jobs (spec/node.md section 7.1, spec/shares.md section
//! 11.5). `Work` builds the coinbase and the transaction set of the next block,
//! hands the miner a `Job`, keeps the share records that miners and peers
//! report, and rebuilds the full block from a submission.
//!
//! The miner never gets the coinbase whole. The node cuts the encoded coinbase
//! around the extranonce into a prefix and a suffix and sends the Merkle path of
//! the coinbase leaf, so a miner changes the extranonce and recomputes
//! `body_root` without parsing a transaction.
//!
//! A share is bound to the parent of its header, not to the header (spec/shares.md
//! section 11.2). A nonce reported on a job becomes the record of the reference
//! header of the job (extranonce zero, time `time_start`) with that nonce.
//! Records are kept per parent, and a job for a child of the tip lists those
//! kept for the parent of the tip, whichever miner found them.
//!
//! `Work` has no lock of its own: the node calls it under its state lock. All
//! of it is local policy except the block it produces, which the chain checks
//! as any other block.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use neb_block::{Header, Target};
use neb_pow::shares::{Model, ShareBuffers};

use crate::frames::Job;
use crate::mempool::Mempool;
use neb_chain::Block;
use neb_chain::block::{ShareRecord, body_root_of, shares_root};
use neb_chain::model::{share_key, share_target};
use neb_chain::state::{ChainState, project_share, subsidy};
use neb_chain::tx::{LOCK_MULTISIG, LOCK_SINGLE, Transaction, TxIn, TxOut, h};

/// Bytes of mempool transactions in a template: the block limit of 1 000 000
/// bytes (spec/node.md section 5.4) minus 10 752 for the rest of the block. The
/// rest is at most 80 (header) + 3 (transaction count) + 158 (coinbase with a
/// 32-byte extranonce: 1 + 1 + (32 + 4 + 1 + 36) + 1 + 2 * 41) + 3 (share
/// count) + 40 * 256 (share list) = 10 484 bytes.
const BODY_BUDGET: usize = 1_000_000 - 10_752;
/// Share records kept for one parent (spec/shares.md section 11.5).
pub const MAX_STORED_SHARES: usize = 1024;
/// Bytes of extranonce in the coinbase (spec/node.md section 5.3).
pub const DEFAULT_EXTRANONCE: u8 = 16;
/// Jobs, and share rounds, that the service keeps; the oldest goes first. A
/// `Submit` or `ShareMsg` for an older job is unknown. The bound limits memory:
/// a template holds up to 1 MB of transactions, so about 64 MB at most.
const KEPT_JOBS: usize = 64;

/// Merkle leaf of a serialised transaction: `H("leaf", BLAKE3(bytes))`.
fn leaf_of(bytes: &[u8]) -> [u8; 32] {
    h(b"leaf", &[blake3::hash(bytes).as_bytes()])
}

/// The Merkle root of spec/node.md section 4.2 over a non-empty leaf list: a
/// node is `H("node", left, right)` and an odd node at a level is promoted
/// unchanged, never duplicated. Panics on an empty list (a block has a coinbase).
pub fn merkle_root_of(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(!leaves.is_empty());
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
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
    level[0]
}

/// The Merkle path of the coinbase leaf (position 0), siblings from the bottom
/// level up, given the leaves of the other transactions. The coinbase leaf is
/// unknown until the miner picks an extranonce, so the tree is built with a hole
/// at position 0 and only the nodes that do not depend on the hole are computed.
/// Folding any coinbase leaf with the path gives the root of `merkle_root_of`.
pub fn branch_for_coinbase(other_leaves: &[[u8; 32]]) -> Vec<[u8; 32]> {
    let mut level: Vec<Option<[u8; 32]>> = Vec::with_capacity(other_leaves.len() + 1);
    level.push(None); // the coinbase, not yet computed
    for l in other_leaves {
        level.push(Some(*l));
    }
    let mut branch = Vec::new();
    while level.len() > 1 {
        // The node paired with position 0 is the next entry of the path.
        if let Some(Some(s)) = level.get(1) {
            branch.push(*s);
        }
        let mut next: Vec<Option<[u8; 32]>> = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i + 1 < level.len() {
            next.push(match (level[i], level[i + 1]) {
                (None, _) => None, // the coinbase node, still unknown
                (Some(a), Some(b)) => Some(h(b"node", &[&a, &b])),
                (Some(_), None) => unreachable!("no hole exists outside index 0"),
            });
            i += 2;
        }
        if i < level.len() {
            next.push(level[i]);
        }
        level = next;
    }
    branch
}

/// A job together with the data to rebuild its block.
pub struct Template {
    pub job: Job,
    /// Transactions of the block. Index 0 is a coinbase with a zero extranonce
    /// that stands for the real one, which the miner supplies through the job
    /// prefix and suffix.
    pub txs: Vec<Transaction>,
    /// The share list of the block (spec/shares.md section 11.2).
    pub shares: Vec<ShareRecord>,
    /// The model the job's block works on (its slot in the state of the parent), on
    /// which the attempts and shares of the job are evaluated.
    pub model: Arc<Model>,
    /// The index of the dataset of the job in the network file, sent to the miner in the model frame: the slot of the
    /// model, or the dataset of an upgrade of that slot (spec/shares.md sections 11.5 and 11.7).
    pub slot: u8,
    /// That dataset.
    pub dataset: neb_pow::shares::SharedDataset,
    /// The reference header of the job: all-zero extranonce, `time_start`,
    /// nonce 0. A nonce reported on the job is the record of this header with
    /// that nonce (spec/shares.md section 11.5).
    pub base: Header,
}

/// The jobs handed out and the share records kept, newest `KEPT_JOBS` of each.
#[derive(Default)]
pub struct Work {
    jobs: BTreeMap<u64, Template>,
    next_id: u64,
    /// Extranonce bytes of the jobs to come.
    pub extranonce_size: u8,
    /// The valid share records, by the parent of their share header: the ones
    /// miners reported and the ones peers relayed. A job for a child of the tip
    /// lists the records kept for the parent of the tip (spec/shares.md
    /// section 11.5), so a block lists the shares of every miner that reaches
    /// the node.
    rounds: BTreeMap<[u8; 32], BTreeSet<ShareRecord>>,
    /// Parents in order of first record, oldest first.
    round_order: VecDeque<[u8; 32]>,
}

/// The outputs of a coinbase: the miner's output, then the project-share output
/// of `share` base units when it is not zero (spec/node.md section 5.3).
pub(crate) fn coinbase_outputs(
    miner_amount: u64,
    share: u64,
    miner_lock: [u8; 32],
    project_lock: [u8; 32],
) -> Vec<TxOut> {
    let mut outputs = vec![TxOut { amount: miner_amount, lock_type: LOCK_SINGLE, lock_data: miner_lock }];
    if share > 0 {
        outputs.push(TxOut { amount: share, lock_type: LOCK_MULTISIG, lock_data: project_lock });
    }
    outputs
}

/// The block of a job for one extranonce, time and nonce. Fails, cheapest check
/// first, on an extranonce of another length than the job's, a time outside
/// `[time_start, time_start + time_window]`, or a coinbase that does not decode
/// or is not coinbase-shaped. The proof of work is not checked here.
fn build_block(
    job: &Job,
    txs: &[Transaction],
    shares: &[ShareRecord],
    extranonce: &[u8],
    time: u32,
    nonce: u32,
) -> Result<Block, String> {
    if extranonce.len() != job.extranonce_size as usize {
        return Err("extranonce length does not match the job".into());
    }
    if time < job.time_start || u64::from(time) > u64::from(job.time_start) + u64::from(job.time_window) {
        return Err("time outside the job window".into());
    }
    let mut cb_bytes = Vec::with_capacity(job.coinbase_prefix.len() + extranonce.len() + job.coinbase_suffix.len());
    cb_bytes.extend_from_slice(&job.coinbase_prefix);
    cb_bytes.extend_from_slice(extranonce);
    cb_bytes.extend_from_slice(&job.coinbase_suffix);
    let coinbase = Transaction::decode(&cb_bytes).map_err(|e| format!("coinbase: {e}"))?;
    if !coinbase.is_coinbase() {
        return Err("the rebuilt transaction is not a coinbase".into());
    }
    // Index 0 of the template is the placeholder coinbase.
    let mut all = vec![coinbase];
    all.extend(txs[1..].iter().cloned());
    let leaves: Vec<[u8; 32]> = all.iter().map(|tx| leaf_of(&tx.encode())).collect();
    let root = merkle_root_of(&leaves);
    let body_root = body_root_of(all.len() as u32, &root, &shares_root(shares));
    let header = Header { version: job.version, parent: job.parent, body_root, time, bits: job.bits, nonce };
    Ok(Block { header, txs: all, shares: shares.to_vec() })
}

impl Work {
    /// An empty service with job id 1 next and the default extranonce size.
    pub fn new() -> Work {
        Work { next_id: 1, extranonce_size: DEFAULT_EXTRANONCE, ..Work::default() }
    }

    /// The template of job `id` while it is among the newest `KEPT_JOBS`.
    pub fn job(&self, id: u64) -> Option<&Template> {
        self.jobs.get(&id)
    }

    /// Drops job `id`; its share round is kept.
    pub fn forget(&mut self, id: u64) {
        self.jobs.remove(&id);
    }

    /// Builds the job for the child of the tip and returns its id. The coinbase
    /// pays `miner_lock` the reward plus the fees minus the project share, and
    /// the project share to the network's lock (spec/node.md sections 5.3 and
    /// 6.3). Transactions come from the mempool, highest fee first, within
    /// `BODY_BUDGET` bytes. The share list is the records kept for the parent
    /// of the tip, checked again on the chain, sorted and cut to 256. `now` is
    /// the node's clock in Unix seconds.
    pub fn make_job(
        &mut self,
        state: &ChainState,
        mempool: &Mempool,
        miner_lock: [u8; 32],
        now: u32,
    ) -> Result<u64, String> {
        let height = state.height() + 1;
        let reward = subsidy(height);
        let share = project_share(reward);
        let chosen = mempool.candidates(state, BODY_BUDGET);
        let fees: u128 = chosen
            .iter()
            .map(|tx| {
                let mut sum_in: u128 = 0;
                for input in &tx.inputs {
                    let op = neb_chain::OutPoint { txid: input.prev_txid, index: input.prev_index };
                    sum_in += state.utxo(&op).map(|u| u.amount as u128).unwrap_or(0);
                }
                let sum_out: u128 = tx.outputs.iter().map(|o| o.amount as u128).sum();
                sum_in.saturating_sub(sum_out)
            })
            .sum();
        // Candidate fees are valid inputs minus outputs; the sum fits u64 in
        // practice, but the output amount is a u64, so check the cast.
        let miner_amount = reward as u128 + fees - share as u128;
        if miner_amount > u64::MAX as u128 {
            return Err("reward plus fees overflows".into());
        }

        let outputs = coinbase_outputs(miner_amount as u64, share, miner_lock, state.params.project_share_lock);
        let coinbase_tx = |ex: u8| {
            let mut witness = Vec::new();
            witness.extend_from_slice(&height.to_le_bytes());
            witness.extend_from_slice(&vec![ex; self.extranonce_size as usize]);
            Transaction {
                version: 1,
                inputs: vec![TxIn { prev_txid: [0u8; 32], prev_index: 0xFFFF_FFFF, witness }],
                outputs: outputs.clone(),
            }
        };
        // Encode the coinbase with extranonce bytes 0x00 and with 0xFF. The first
        // differing byte starts the extranonce; cutting its `n` bytes out gives
        // the prefix and the suffix. Both encodings have the same length, as
        // the witness length depends only on the extranonce size.
        let zeros = coinbase_tx(0x00);
        let ones = coinbase_tx(0xFF);
        let bytes_a = zeros.encode();
        let bytes_b = ones.encode();
        if bytes_a.len() != bytes_b.len() {
            return Err("placeholder changed the coinbase length".into());
        }
        let off = bytes_a
            .iter()
            .zip(bytes_b.iter())
            .position(|(a, b)| a != b)
            .ok_or("placeholder extranonce not found in the coinbase")?;
        let n = self.extranonce_size as usize;
        let coinbase_prefix = bytes_a[..off].to_vec();
        let coinbase_suffix = bytes_a[off + n..].to_vec();

        let other_leaves: Vec<[u8; 32]> = chosen.iter().map(|tx| leaf_of(&tx.encode())).collect();
        let branch = branch_for_coinbase(&other_leaves);

        let bits = state.params.difficulty.next_bits(state.height(), state.tip_time()).map_err(|e| e.to_string())?;
        let target = Target::from_bits(bits).map_err(|e| e.to_string())?;
        let round_target = share_target(bits).map_err(|e| e.to_string())?;
        // Records kept for the parent of the tip; `valid_shares` checks them on
        // the model of the tip's parent, sorts them and cuts the list to 256.
        let kept = self.rounds.get(&state.tip().header.parent).cloned().unwrap_or_default();
        let shares = state.valid_shares(kept);
        // The later of the clock and the parent time plus one (spec/node.md
        // section 7.1), so the header time passes the median rule when the
        // clock lags. A job on the same tip as the newest one starts one second
        // after it, so it gives new nonces even when the clock has not moved past
        // that time; at most 60 s ahead of the clock (the future limit is 120 s).
        let after_newest = self
            .jobs
            .last_key_value()
            .filter(|(_, t)| t.job.parent == state.tip_id())
            .map_or(0, |(_, t)| t.job.time_start.saturating_add(1).min(now.saturating_add(60)));
        let starting_time = now.max(state.tip_time().saturating_add(1)).max(after_newest);
        let id = self.next_id;
        self.next_id += 1;
        let tx_count = (chosen.len() + 1) as u32;
        let job = Job {
            id,
            clean: true,
            version: 1,
            parent: state.tip_id(),
            time_start: starting_time,
            time_window: 3600,
            bits,
            target: target.0,
            share_target: round_target.0,
            coinbase_prefix,
            coinbase_suffix,
            extranonce_size: self.extranonce_size,
            branch,
            tx_count,
            shares_root: shares_root(&shares),
            height,
            // Local policy: 300 s after creation. The time window of 3600 s is
            // fixed by the spec.
            expiry: now.saturating_add(300),
        };
        let mut txs = vec![zeros];
        txs.extend(chosen);
        let base =
            build_block(&job, &txs, &shares, &vec![0u8; job.extranonce_size as usize], job.time_start, 0)?.header;
        let slot = state.params.dataset_index_at(height) as u8;
        let dataset = state.params.dataset_at(height).clone();
        self.jobs.insert(id, Template { job, txs, shares, model: state.tip_model(), slot, dataset, base });
        // Ids increase, so the first key is the oldest job.
        while self.jobs.len() > KEPT_JOBS {
            self.jobs.pop_first();
        }
        Ok(id)
    }

    /// Records a share that a miner reports on job `id` (spec/shares.md section
    /// 11.5): the record of the reference header of the job with `nonce`, kept
    /// for the parent of that header. Unless `check` is false, its ticket on the
    /// model of the job must meet the share target of the job; this costs one
    /// ticket evaluation. Returns the parent and the record when the record is
    /// new, for the node to relay, and `None` for a duplicate or when the
    /// parent already holds `MAX_STORED_SHARES` records. Fails on an unknown job
    /// or a ticket above the share target.
    pub fn add_share(&mut self, id: u64, nonce: u32, check: bool) -> Result<Option<([u8; 32], ShareRecord)>, String> {
        let template = self.jobs.get(&id).ok_or("unknown job")?;
        let record = ShareRecord { body_root: template.base.body_root, time: template.base.time, nonce };
        let parent = template.base.parent;
        if check {
            let key = share_key(&record, parent, template.base.bits);
            let ticket = ShareBuffers::new().ticket(&template.model, &*template.dataset, &key);
            if !Target(template.job.share_target).admits(&ticket) {
                return Err("the ticket is above the share target".into());
            }
        }
        Ok(self.store(parent, record).then_some((parent, record)))
    }

    /// Whether `record` is already kept for `parent`.
    pub fn has_record(&self, parent: &[u8; 32], record: &ShareRecord) -> bool {
        self.rounds.get(parent).is_some_and(|r| r.contains(record))
    }

    /// Keeps records that the caller has checked (a `shares` message of a peer)
    /// for `parent`; returns the ones that were new, in the order given.
    pub fn add_records(&mut self, parent: [u8; 32], records: &[ShareRecord]) -> Vec<ShareRecord> {
        records.iter().copied().filter(|r| self.store(parent, *r)).collect()
    }

    /// The records kept for `parent`, in byte order.
    pub fn records(&self, parent: &[u8; 32]) -> Vec<ShareRecord> {
        self.rounds.get(parent).map(|r| r.iter().copied().collect()).unwrap_or_default()
    }

    /// Inserts a record; false for a duplicate or a full parent. The oldest
    /// parents go first once more than `KEPT_JOBS` are kept.
    fn store(&mut self, parent: [u8; 32], record: ShareRecord) -> bool {
        if !self.rounds.contains_key(&parent) {
            self.round_order.push_back(parent);
            while self.round_order.len() > KEPT_JOBS {
                if let Some(old) = self.round_order.pop_front() {
                    self.rounds.remove(&old);
                }
            }
        }
        let round = self.rounds.entry(parent).or_default();
        if round.len() >= MAX_STORED_SHARES {
            return false;
        }
        round.insert(record)
    }

    /// Rebuilds the block of job `id` from a submission: the coinbase from the
    /// job's prefix, the extranonce and the job's suffix, then the job's other
    /// transactions and share list. Only the shape is checked (see
    /// `build_block`); the caller runs the checks of spec/node.md section 6.1
    /// and connects the block.
    pub fn submit(&self, id: u64, extranonce: &[u8], time: u32, nonce: u32) -> Result<Block, String> {
        let template = self.jobs.get(&id).ok_or("unknown job")?;
        build_block(&template.job, &template.txs, &template.shares, extranonce, time, nonce)
    }
}
