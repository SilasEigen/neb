//! Tests of the share rules (spec/shares.md section 11): the list encoding, the
//! detached share header, the model step checked against an independent
//! replay, rejected lists and reorganisations. They mine on the easy
//! parameters of `demo`.

use crate::block::{
    Block, MAX_SHARES_PER_BLOCK, SHARE_RECORD_BYTES, ShareRecord, body_root_of, check_shares_list, shares_root,
};
use crate::demo;
use crate::model::{ProofError, share_key, share_target};
use crate::state::{ChainError, ChainState, project_share, subsidy};
use neb_pow::shares::{Dataset, Model, N1, N2, ShareBuffers, share_eval};
use std::sync::Arc;

const T0: u32 = 1_800_000_000;

/// A state on `easy_params` with target comparison: a block takes about 64
/// attempts and a round yields about 32 shares.
fn easy_state() -> ChainState {
    let params = demo::easy_params(T0, demo::PROJECT_LOCK);
    ChainState::from_genesis(params, demo::easy_genesis(T0)).expect("genesis")
}

/// A record with a body root filled with `body`.
fn rec(body: u8, time: u32, nonce: u32) -> ShareRecord {
    ShareRecord { body_root: [body; 32], time, nonce }
}

/// `count` records in strictly increasing byte order (the first two bytes of
/// the body root count in big endian).
fn ascending(count: usize) -> Vec<ShareRecord> {
    (0..count)
        .map(|i| {
            let mut body_root = [0u8; 32];
            body_root[0] = (i >> 8) as u8;
            body_root[1] = i as u8;
            ShareRecord { body_root, time: 0, nonce: 0 }
        })
        .collect()
}

/// Mines and connects `count` blocks, each listing the shares of the round of
/// the previous one.
fn mine_chain(state: &mut ChainState, count: u32) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut round = Vec::new();
    for _ in 0..count {
        let (block, next) = demo::mine_plain(state, round);
        state.connect_block(&block).expect("a mined block connects");
        round = next;
        blocks.push(block);
    }
    blocks
}

/// Models replayed with only the share step library, independent of the chain
/// code: the shares listed by block k are the headers whose parent is the
/// parent of block k - 1, with its bits; they are evaluated on the model of
/// block k - 2 and their summed gradients update the model of block k - 1.
fn replay_models(state: &ChainState, blocks: &[Block]) -> Vec<Model> {
    let ds = &*state.params.datasets[0];
    let mut models = vec![Model::from_seed(&state.chain_id)];
    let mut headers = vec![demo::easy_genesis(T0).header];
    for block in blocks {
        let n = models.len();
        let mut next = models[n - 1].clone();
        if !block.shares.is_empty() {
            let on = &models[n.saturating_sub(2)];
            let (p, pp) = (headers[n - 1], headers[n.saturating_sub(2)]);
            let (mut gs1, mut gs2) = (vec![0i32; N1], vec![0i32; N2]);
            for rec in &block.shares {
                let share_header = neb_block::Header {
                    version: 1,
                    parent: pp.id(),
                    body_root: rec.body_root,
                    time: rec.time,
                    bits: p.bits,
                    nonce: rec.nonce,
                };
                let key = *blake3::hash(&share_header.encode()).as_bytes();
                let r = share_eval(on, ds, &key);
                gs1.iter_mut().zip(&r.g1).for_each(|(s, g)| *s += g);
                gs2.iter_mut().zip(&r.g2).for_each(|(s, g)| *s += g);
            }
            next.apply_step(&gs1, &gs2);
        }
        models.push(next);
        headers.push(block.header);
    }
    models
}

/// Number of dataset samples that `model` classifies correctly.
fn correct(model: &Model, ds: &dyn Dataset) -> usize {
    (0..ds.n())
        .filter(|&i| {
            let (x, label) = ds.get(i);
            model.predict(&x) == label
        })
        .count()
}

// Share list: committed by body_root, encoded as varint count and 40-byte records, and non-canonical forms are refused.
#[test]
fn the_share_list_is_committed_and_encoded_canonically() {
    let cb = demo::coinbase(1, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000);
    let mk =
        |shares: Vec<ShareRecord>| demo::block_with_shares([0u8; 32], T0, 0xC280_0000, 0, vec![cb.clone()], shares);
    let list = vec![rec(1, 10, 3), rec(1, 10, 9), rec(2, 5, 40)];
    let mut other = list.clone();
    other[2].nonce = 41;
    let (none, a, b) = (mk(vec![]), mk(list.clone()), mk(other));
    assert_ne!(none.body_root(), a.body_root());
    assert_ne!(a.body_root(), b.body_root());
    assert_eq!(a.body_root(), mk(list.clone()).body_root());
    assert_eq!(shares_root(&[]), *blake3::hash(b"").as_bytes());
    let concat: Vec<u8> = list.iter().flat_map(|r| r.to_bytes()).collect();
    assert_eq!(concat.len(), 3 * SHARE_RECORD_BYTES);
    assert_eq!(shares_root(&list), *blake3::hash(&concat).as_bytes());
    // body_root = H("body", LE32(count), merkle, shares_root).
    let merkle = crate::tx::h(b"leaf", &[blake3::hash(&cb.encode()).as_bytes()]);
    assert_eq!(a.body_root(), body_root_of(1, &merkle, &shares_root(&list)));

    for block in [&none, &a, &b] {
        assert_eq!(&Block::decode(&block.encode()).unwrap(), block);
    }
    // The share list ends the block: a one-byte varint count, then 40-byte records.
    let bytes = a.encode();
    let n = bytes.len();
    let tail_len = 1 + 3 * SHARE_RECORD_BYTES;
    assert_eq!(bytes[n - tail_len], 3u8);
    assert_eq!(&bytes[n - tail_len + 1..n - tail_len + 1 + SHARE_RECORD_BYTES], &list[0].to_bytes()[..]);
    assert_eq!(&bytes[n - SHARE_RECORD_BYTES..], &list[2].to_bytes()[..]);
    assert_eq!(&list[2].to_bytes()[..32], &[2u8; 32]);
    assert_eq!(&list[2].to_bytes()[32..36], &5u32.to_le_bytes());
    assert_eq!(&list[2].to_bytes()[36..], &40u32.to_le_bytes());

    // Records order as byte strings, which is not the order of their numbers.
    assert!(rec(1, 10, 9) < rec(2, 5, 3), "the body root comes first");
    assert!(rec(1, 256, 0) < rec(1, 1, 0), "LE32 bytes: 256 is 00 01 00 00 and 1 is 01 00 00 00");

    // Every other form is refused.
    let tail = |recs: &[ShareRecord]| {
        let mut out = bytes[..n - tail_len].to_vec();
        neb_codec::write_varint(&mut out, recs.len() as u64);
        recs.iter().for_each(|r| out.extend_from_slice(&r.to_bytes()));
        out
    };
    assert!(Block::decode(&tail(&list)).is_ok());
    assert!(Block::decode(&tail(&[list[1], list[0], list[2]])).is_err(), "decreasing");
    assert!(Block::decode(&tail(&[list[0], list[0], list[2]])).is_err(), "duplicate");
    assert!(Block::decode(&tail(&[rec(1, 256, 0), rec(1, 1, 0)])).is_ok(), "ascending bytes, descending numbers");
    assert!(Block::decode(&tail(&[rec(1, 1, 0), rec(1, 256, 0)])).is_err(), "ascending numbers, descending bytes");
    let max = ascending(256);
    let too_many = ascending(257);
    assert!(Block::decode(&tail(&too_many)).is_err(), "257 records");
    assert!(Block::decode(&tail(&max)).is_ok(), "256 records");
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(Block::decode(&trailing).is_err());
    assert!(Block::decode(&bytes[..n - 1]).is_err(), "a record cut short");
    let mut padded_count = bytes[..n - tail_len].to_vec();
    padded_count.extend_from_slice(&[0xFD, 3, 0]);
    padded_count.extend_from_slice(&bytes[n - tail_len + 1..]);
    assert!(Block::decode(&padded_count).is_err(), "a count in a longer form");
    assert!(check_shares_list(&max).is_ok() && check_shares_list(&too_many).is_err());
}

// Consensus check of the model step: the models kept by the state equal a replay that uses only the share step library.
#[test]
fn mined_blocks_with_real_shares_connect_and_match_an_independent_replay() {
    let mut state = easy_state();
    let blocks = mine_chain(&mut state, 14);
    assert_eq!(state.height(), 14);
    assert!(blocks[0].shares.is_empty(), "height 1 lists no share");
    assert!(blocks[1..].iter().all(|b| !b.shares.is_empty()), "every later block lists shares");
    let total: usize = blocks.iter().map(|b| b.shares.len()).sum();
    assert!(total > 100, "about 32 shares per block, got {total}");
    assert!(state.conservation());

    let models = replay_models(&state, &blocks);
    for (height, want) in models.iter().enumerate() {
        if let Some(got) = state.model_at(height as u32) {
            assert_eq!(got.hashes(), want.hashes(), "model of height {height}");
        }
    }
    assert_eq!(state.tip_model().hashes(), models[14].hashes());
    assert_ne!(models[14].weights_hash(), models[0].weights_hash(), "the model moved");

    // A second state connecting the same blocks ends on the same model.
    let mut again = easy_state();
    for b in &blocks {
        again.connect_block(b).expect("replay");
    }
    assert_eq!(again.tip_model().hashes(), state.tip_model().hashes());
    assert_eq!(again.utxos(), state.utxos());
}

/// Mines a block that lists `shares` and returns the error of connecting it.
fn refuse(state: &mut ChainState, shares: Vec<ShareRecord>) -> ChainError {
    let (block, _) = demo::mine_plain(state, shares);
    state.connect_block(&block).expect_err("the block must be refused")
}

/// Splits the records `(body, time, nonce)` for the nonces below `limit` into
/// those that meet and those that miss the share target of the round of the
/// tip (headers with the parent of the tip), evaluated on `model`. Each side
/// is sorted as byte strings, so it is a valid list.
fn split_records(
    state: &ChainState,
    model: &Model,
    body: u8,
    time: u32,
    limit: u32,
) -> (Vec<ShareRecord>, Vec<ShareRecord>) {
    let tip = state.tip().header;
    let target = share_target(tip.bits).unwrap();
    let mut buffers = ShareBuffers::new();
    let (mut good, mut bad): (Vec<_>, Vec<_>) = (0..limit).map(|n| rec(body, time, n)).partition(|r| {
        target.admits(&buffers.ticket(model, &*state.params.datasets[0], &share_key(r, tip.parent, tip.bits)))
    });
    good.sort();
    bad.sort();
    (good, bad)
}

// Each way to break the share rules (bad ticket, bad list, wrong commitment, bad block ticket) is refused and leaves the state unchanged.
#[test]
fn shares_that_break_a_rule_are_rejected() {
    let mut state = easy_state();
    mine_chain(&mut state, 5);
    let model_pp = state.model_at(state.height() - 1).unwrap();
    let (good, bad) = split_records(&state, &model_pp, 0xAB, T0, 60);
    assert!(good.len() >= 8 && bad.len() >= 8);
    let before = (state.height(), state.tip_model().hashes());

    // A list of valid shares passes the verification.
    let (ok_block, _) = demo::mine_plain(&state, good[..8].to_vec());
    assert!(state.verify_block(&ok_block).is_ok());

    // A share with a ticket above the share target.
    let mut with_bad = good[..5].to_vec();
    with_bad.push(bad[0]);
    with_bad.sort_unstable();
    match refuse(&mut state, with_bad) {
        ChainError::Proof(ProofError::BadShare(r)) => assert_eq!(r, bad[0]),
        other => panic!("expected BadShare, got {other:?}"),
    }
    // A duplicate, a decreasing pair and 257 records.
    for list in [vec![good[0], good[0]], vec![good[2], good[1]], ascending(MAX_SHARES_PER_BLOCK + 1)] {
        match refuse(&mut state, list) {
            ChainError::Proof(ProofError::Shape(_)) => {}
            other => panic!("expected a list error, got {other:?}"),
        }
    }
    // The list is committed by body_root: swapping it breaks the commitment.
    let (mut swapped, _) = demo::mine_plain(&state, good[..8].to_vec());
    swapped.shares = good[1..9].to_vec();
    assert!(matches!(state.connect_block(&swapped), Err(ChainError::BodyRootMismatch)));

    // A block whose ticket is above its target.
    let (mut weak, _) = demo::mine_plain(&state, good[..8].to_vec());
    let block_target = neb_block::Target::from_bits(weak.header.bits).unwrap();
    let model_p = state.tip_model();
    let mut buffers = ShareBuffers::new();
    weak.header.nonce = (0u32..)
        .find(|&n| {
            weak.header.nonce = n;
            !block_target.admits(&buffers.ticket(&model_p, &*state.params.datasets[0], &weak.id()))
        })
        .unwrap();
    assert!(matches!(state.connect_block(&weak), Err(ChainError::Proof(ProofError::BadPow))));

    assert_eq!((state.height(), state.tip_model().hashes()), before, "nothing changed");
}

// Height 1 follows the genesis block, which has no round.
#[test]
fn the_block_of_height_one_lists_no_share() {
    let mut state = easy_state();
    match refuse(&mut state, vec![rec(1, T0, 1)]) {
        ChainError::Proof(ProofError::Shape(_)) => {}
        other => panic!("expected a list error at height 1, got {other:?}"),
    }
    assert_eq!(state.height(), 0);
}

// Shares are evaluated on the model of the grandparent; a record valid only on an older model is invalid.
#[test]
fn a_share_computed_on_a_stale_model_is_refused() {
    let mut state = easy_state();
    mine_chain(&mut state, 8);
    let h = state.height();
    let fresh = state.model_at(h - 1).unwrap();
    let stale = state.model_at(h - 3).unwrap();
    assert_ne!(fresh.weights_hash(), stale.weights_hash(), "the weights moved in two blocks");
    let (on_fresh, _) = split_records(&state, &fresh, 0xAB, T0, 400);
    let (on_stale, _) = split_records(&state, &stale, 0xAB, T0, 400);
    let record = *on_stale.iter().find(|r| !on_fresh.contains(r)).expect("a record valid only on the stale model");
    match refuse(&mut state, vec![record]) {
        ChainError::Proof(ProofError::BadShare(r)) => assert_eq!(r, record),
        other => panic!("expected BadShare, got {other:?}"),
    }
    // A record valid on the grandparent's model (the right one) is accepted.
    let (block, _) = demo::mine_plain(&state, vec![on_fresh[0]]);
    assert!(state.connect_block(&block).is_ok());
}

// A share is bound to the parent of the round, not to the header: a header with another body root and another time is a valid share.
#[test]
fn a_header_of_another_miner_with_the_same_parent_is_a_valid_share() {
    let mut state = easy_state();
    let mut blocks = mine_chain(&mut state, 6);
    let model_pp = state.model_at(state.height() - 1).unwrap();
    let tip = state.tip().header;
    // Another miner: another coinbase (body root) and another time.
    let (foreign, _) = split_records(&state, &model_pp, 0xEE, tip.time + 7, 80);
    assert!(foreign.len() >= 10);
    assert!(foreign.iter().all(|r| r.body_root != tip.body_root && r.time != tip.time));
    let list = foreign[..10].to_vec();
    for r in &list {
        assert_eq!(state.check_share(&tip.parent, r), Some(true));
    }
    let (block, _) = demo::mine_plain(&state, list);
    state.connect_block(&block).expect("foreign shares of the round connect");
    blocks.push(block);
    let models = replay_models(&state, &blocks);
    assert_eq!(state.tip_model().hashes(), models[7].hashes(), "the model of the independent replay");
    assert_ne!(models[7].hashes(), models[6].hashes(), "the foreign shares moved the model");
}

// The header of the tip and the header of a competing block with the same parent are both shares of the round.
#[test]
fn the_header_of_the_tip_and_of_a_competing_block_are_shares_of_the_round() {
    let mut state = easy_state();
    let blocks = mine_chain(&mut state, 6);
    let tip = state.tip().header;
    // A competing block: the same parent and list as the tip, another coinbase.
    let mut probe = easy_state();
    for b in &blocks[..5] {
        probe.connect_block(b).unwrap();
    }
    let height = probe.height() + 1;
    let reward = subsidy(height);
    let cb = demo::coinbase(height, [0x55; 32], demo::PROJECT_LOCK, reward - project_share(reward));
    let (rival, _) = demo::mine_next(&probe, vec![cb], blocks[5].shares.clone());
    assert_eq!(rival.header.parent, tip.parent);
    assert_ne!(rival.id(), state.tip_id());
    let list = {
        let mut l = vec![ShareRecord::of_header(&tip), ShareRecord::of_header(&rival.header)];
        l.sort();
        l
    };
    for r in &list {
        assert_eq!(state.check_share(&tip.parent, r), Some(true));
    }
    let (block, _) = demo::mine_plain(&state, list.clone());
    state.connect_block(&block).expect("the two headers are valid shares");
    assert_eq!(block.shares, list);
}

// The store holds the tip's round and the parent's round; a record valid for one round is not valid for the other.
#[test]
fn a_record_of_another_round_is_refused() {
    let mut state = easy_state();
    mine_chain(&mut state, 6);
    let (tip, id) = (state.tip().header, state.tip_id());
    assert!(state.share_round(&id).is_some());
    assert!(state.share_round(&tip.parent).is_some());
    assert!(state.share_round(&[9u8; 32]).is_none());
    assert_eq!(state.check_share(&[9u8; 32], &rec(1, T0, 1)), None);
    // A record that meets the target for headers whose parent is the tip but not for the parent of the tip.
    let record = (0..200)
        .map(|n| rec(0xAB, tip.time, n))
        .find(|r| state.check_share(&id, r) == Some(true) && state.check_share(&tip.parent, r) == Some(false))
        .expect("a record valid for the next round only");
    match refuse(&mut state, vec![record]) {
        ChainError::Proof(ProofError::BadShare(r)) => assert_eq!(r, record),
        other => panic!("expected BadShare, got {other:?}"),
    }
}

// The list of a child block is sorted, without duplicate, at most 256, and every record is valid.
#[test]
fn valid_shares_sorts_deduplicates_and_truncates() {
    let mut state = easy_state();
    mine_chain(&mut state, 4);
    let (tip, q) = (state.tip().header, state.tip().header.parent);
    let mut candidates: Vec<ShareRecord> = (0..800).map(|n| rec(0xCD, tip.time, n)).collect();
    candidates.extend((0..100).map(|n| rec(0xCD, tip.time, n)));
    candidates.reverse();
    let list = state.valid_shares(candidates);
    assert_eq!(list.len(), MAX_SHARES_PER_BLOCK, "about 400 of 800 records are valid, the list is cut at 256");
    assert!(list.windows(2).all(|w| w[0] < w[1]));
    assert!(list.iter().all(|r| state.check_share(&q, r) == Some(true)));
    assert!(state.valid_shares(Vec::new()).is_empty());
}

/// Mines a branch of `count` blocks on the tip of `probe` (which advances with
/// it), paying `lock`; the first block lists the valid shares of the round of
/// the tip among the nonces 0..200.
fn branch_on(probe: &mut ChainState, count: u32, lock: [u8; 32]) -> Vec<Block> {
    let mut branch = Vec::new();
    let time = probe.tip_time();
    let mut round = probe.valid_shares((0..200).map(|n| rec(0xAB, time, n)));
    for _ in 0..count {
        let height = probe.height() + 1;
        let reward = subsidy(height);
        let cb = demo::coinbase(height, lock, demo::PROJECT_LOCK, reward - project_share(reward));
        let (block, next) = demo::mine_next(probe, vec![cb], round);
        probe.connect_block(&block).unwrap();
        round = next;
        branch.push(block);
    }
    branch
}

// The models after a reorganisation equal those of a full replay of the new chain (spec/shares.md section 11.4).
#[test]
fn a_reorganisation_lands_on_the_model_of_a_full_replay() {
    let mut state = easy_state();
    let main = mine_chain(&mut state, 7);
    let mut probe = easy_state();
    for b in &main[..4] {
        probe.connect_block(b).unwrap();
    }
    let branch = branch_on(&mut probe, 4, [0x33; 32]);
    assert!(!branch[0].shares.is_empty());

    let old_tip = state.tip_id();
    let removed = state.reorg(4, &branch).expect("reorg");
    assert_eq!(removed.len(), 3);
    assert_eq!(removed[0].0, old_tip);
    assert_eq!(state.height(), 8);
    assert_eq!(state.tip_id(), probe.tip_id());

    let mut replay = easy_state();
    for b in main[..4].iter().chain(branch.iter()) {
        replay.connect_block(b).expect("replay");
    }
    for h in 0..=8 {
        assert_eq!(state.model_at(h).map(|m| m.hashes()), replay.model_at(h).map(|m| m.hashes()), "height {h}");
    }
    assert_eq!(state.utxos(), replay.utxos());
    assert!(state.conservation());

    // The removed models are the models the former best chain had.
    let mut old = easy_state();
    for b in &main {
        old.connect_block(b).unwrap();
    }
    assert_eq!(removed[0].1[0].hashes(), old.tip_model().hashes());
    assert_eq!(removed[2].1[0].hashes(), old.model_at(5).unwrap().hashes());
}

// An invalid block in the branch leaves the former best chain, UTXO set and model intact.
#[test]
fn a_failed_reorganisation_restores_the_chain_and_its_model() {
    let mut state = easy_state();
    let main = mine_chain(&mut state, 6);
    let mut probe = easy_state();
    for b in &main[..3] {
        probe.connect_block(b).unwrap();
    }
    let mut branch = branch_on(&mut probe, 4, [0x44; 32]);
    // The third block of the branch gets a nonce that fails the proof.
    let model = probe.model_at(5).unwrap();
    let target = neb_block::Target::from_bits(branch[2].header.bits).unwrap();
    let mut buffers = ShareBuffers::new();
    let ds = Arc::clone(&state.params.datasets[0]);
    branch[2].header.nonce = (0u32..)
        .find(|&n| {
            branch[2].header.nonce = n;
            !target.admits(&buffers.ticket(&model, &*ds, &branch[2].id()))
        })
        .unwrap();
    let before = (state.tip_id(), state.tip_model().hashes(), state.utxos().clone());
    assert!(matches!(state.reorg(3, &branch), Err(ChainError::Proof(_))));
    assert_eq!(state.tip_id(), before.0);
    assert_eq!(state.tip_model().hashes(), before.1);
    assert_eq!(state.utxos(), &before.2);
    assert_eq!(state.height(), 6);
}

// The shares train the model: accuracy on the synthetic dataset rises.
#[test]
fn the_model_classifies_better_after_24_blocks_of_shares() {
    let mut state = easy_state();
    let initial = Model::from_seed(&state.chain_id);
    let blocks = mine_chain(&mut state, 25);
    let shares: usize = blocks.iter().map(|b| b.shares.len()).sum();
    assert!(shares >= 24 * 8, "{shares} shares in 25 blocks");
    let ds = Arc::clone(&state.params.datasets[0]);
    let (before, after) = (correct(&initial, &*ds), correct(&state.tip_model(), &*ds));
    assert!(after > before + 200, "{before} of {} before, {after} after", ds.n());
}
