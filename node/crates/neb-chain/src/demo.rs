//! Builders for demo chains used by the tests and by the self test of
//! `neb-node`: parameter sets, a genesis block, coinbases and a miner that
//! scans nonces on the synthetic dataset. Not consensus code; the locks are
//! fixed public byte strings and offer no protection.

use std::sync::Arc;

use crate::block::{Block, MAX_SHARES_PER_BLOCK, ShareRecord};
use crate::model::{share_key, share_target};
use crate::state::{ChainParams, ChainState, project_share, subsidy};
use crate::tx::{LOCK_MULTISIG, LOCK_SINGLE, Transaction, TxIn, TxOut};
use neb_block::{Header, Target};
use neb_pow::shares::{ShareBuffers, SyntheticDataset};

/// Parameters whose anchor target (2^217) doubles as the pow limit, so blocks
/// at the 60 s spacing keep the same target. The target is too hard to mine in
/// a test: use them with `verify_pow` false or for rules that ignore the proof.
pub fn demo_params(anchor_time: u32, project_lock: [u8; 32]) -> ChainParams {
    ChainParams {
        difficulty: neb_difficulty::Params {
            spacing: neb_difficulty::SPACING,
            half_life: neb_difficulty::HALF_LIFE,
            pow_limit_bits: 0xC280_0000,
            anchor_bits: 0xC280_0000,
            anchor_time,
        },
        project_share_lock: project_lock,
        datasets: vec![Arc::new(SyntheticDataset::new())],
        upgrades: Vec::new(),
    }
}

/// Block target 2^250: one attempt in 64 meets it. The share target is 2^255,
/// so one attempt in two is a share, and a test can mine a block.
pub const EASY_BITS: u32 = 0xE380_0000;

/// Pow limit of the easy parameters: 2^255, the easiest compact target.
pub const EASY_LIMIT_BITS: u32 = 0xE880_0000;

/// Parameters for tests that mine real blocks on the synthetic dataset: the
/// anchor is `EASY_BITS` and the limit `EASY_LIMIT_BITS`. The limit equals the
/// share target of the anchor, so the share target never saturates.
pub fn easy_params(anchor_time: u32, project_lock: [u8; 32]) -> ChainParams {
    ChainParams {
        difficulty: neb_difficulty::Params {
            spacing: neb_difficulty::SPACING,
            half_life: neb_difficulty::HALF_LIFE,
            pow_limit_bits: EASY_LIMIT_BITS,
            anchor_bits: EASY_BITS,
            anchor_time,
        },
        project_share_lock: project_lock,
        datasets: vec![Arc::new(SyntheticDataset::new())],
        upgrades: Vec::new(),
    }
}

/// The genesis of `easy_params`.
pub fn easy_genesis(anchor_time: u32) -> Block {
    genesis_at(anchor_time, EASY_BITS, MINER_LOCK)
}

/// Mines a child of the tip of `state`: scans nonces from 0 until the ticket
/// on the model of the tip meets the block target, and lists `shares` (the
/// round of the tip). Returns the block and the shares of its own round (at
/// most 256 records up to the solution), for the next block to list. The
/// header time is 60 s after the tip, so the target stays at the anchor.
/// Loops until a solution is found.
pub fn mine_next(state: &ChainState, txs: Vec<Transaction>, shares: Vec<ShareRecord>) -> (Block, Vec<ShareRecord>) {
    let mut block = block_with_shares(state.tip_id(), state.tip_time() + 60, 0, 0, txs, shares);
    block.header.bits = state.params.difficulty.next_bits(state.height(), state.tip_time()).expect("difficulty");
    let target = Target::from_bits(block.header.bits).expect("bits");
    let round_target = share_target(block.header.bits).expect("bits");
    let model = state.tip_model();
    let mut buffers = ShareBuffers::new();
    let mut round = Vec::new();
    for nonce in 0.. {
        let record = ShareRecord { body_root: block.header.body_root, time: block.header.time, nonce };
        let key = share_key(&record, block.header.parent, block.header.bits);
        let ticket = buffers.ticket(&model, &**state.params.dataset_at(state.height() + 1), &key);
        if round_target.admits(&ticket) && round.len() < MAX_SHARES_PER_BLOCK {
            round.push(record);
        }
        if target.admits(&ticket) {
            block.header.nonce = nonce;
            break;
        }
    }
    // A list is in byte order, which is not the order of the nonces.
    round.sort_unstable();
    (block, round)
}

/// `mine_next` with a coinbase that pays the reward less the project share.
pub fn mine_plain(state: &ChainState, shares: Vec<ShareRecord>) -> (Block, Vec<ShareRecord>) {
    let height = state.height() + 1;
    let reward = subsidy(height);
    mine_next(state, vec![coinbase(height, MINER_LOCK, PROJECT_LOCK, reward - project_share(reward))], shares)
}

/// A coinbase paying `miner_amount` to `miner_lock` and the project share to
/// `share_lock`. The witness is `height (u32 LE) | 8 zero bytes` as in
/// spec/node.md section 5.3. The second output is always present, so it is
/// valid only while the project share is above zero.
pub fn coinbase(height: u32, miner_lock: [u8; 32], share_lock: [u8; 32], miner_amount: u64) -> Transaction {
    let share = project_share(subsidy(height));
    let mut witness = Vec::new();
    witness.extend_from_slice(&height.to_le_bytes());
    witness.extend_from_slice(&[0u8; 8]);
    Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [0u8; 32], prev_index: 0xFFFF_FFFF, witness }],
        outputs: vec![
            TxOut { amount: miner_amount, lock_type: LOCK_SINGLE, lock_data: miner_lock },
            TxOut { amount: share, lock_type: LOCK_MULTISIG, lock_data: share_lock },
        ],
    }
}

/// A block without shares; `body_root` is computed.
pub fn block(parent: [u8; 32], time: u32, bits: u32, nonce: u32, txs: Vec<Transaction>) -> Block {
    block_with_shares(parent, time, bits, nonce, txs, Vec::new())
}

/// A block with the given body and share list; `body_root` is computed.
pub fn block_with_shares(
    parent: [u8; 32],
    time: u32,
    bits: u32,
    nonce: u32,
    txs: Vec<Transaction>,
    shares: Vec<ShareRecord>,
) -> Block {
    let mut b = Block { header: Header { version: 1, parent, body_root: [0u8; 32], time, bits, nonce }, txs, shares };
    b.header.body_root = b.body_root();
    b
}

/// Lock data of the demo miner and project outputs; no key corresponds to them.
pub const MINER_LOCK: [u8; 32] = [0x11; 32];
pub const PROJECT_LOCK: [u8; 32] = [0x22; 32];

/// The genesis of `demo_params`: a coinbase paying the reward of height 0 less
/// the project share to `MINER_LOCK`, and the share to `PROJECT_LOCK`.
pub fn genesis(anchor_time: u32) -> Block {
    genesis_at(anchor_time, 0xC280_0000, MINER_LOCK)
}

/// The genesis of `demo_params` with the miner output on `miner_lock`, so that
/// a test can spend it with a real key. The genesis output matures at height 100.
pub fn genesis_to(anchor_time: u32, miner_lock: [u8; 32]) -> Block {
    genesis_at(anchor_time, 0xC280_0000, miner_lock)
}

/// A genesis at the target `bits` and time `anchor_time`.
pub fn genesis_at(anchor_time: u32, bits: u32, miner_lock: [u8; 32]) -> Block {
    let reward = subsidy(0);
    block([0u8; 32], anchor_time, bits, 0, vec![coinbase(0, miner_lock, PROJECT_LOCK, reward - project_share(reward))])
}
