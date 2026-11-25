//! Tests of the chain state written to bytes and read back (`state/snapshot.rs`): the state that is read is the
//! state that was written, it goes on like a replayed one, and bytes that do not fit are refused.

use crate::block::Block;
use crate::demo;
use crate::state::{ChainError, ChainState, MAX_REORG, project_share, subsidy};
use crate::tests::{demo_chain, new_state};

const T0: u32 = 1_800_000_000;

fn one_block(parent: [u8; 32], height: u32, salt: u32) -> Block {
    let reward = subsidy(height);
    let coinbase = demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward));
    demo::block(parent, T0 + 9000 + salt, 0xC280_0000, 7000 + salt, vec![coinbase])
}

// Blocks with real shares, so every block changes a model: a state read from the bytes written at block 6 and
// carried to block 9 is, byte for byte, the state replayed from the genesis block to block 9.
#[test]
fn a_state_read_back_goes_on_like_the_replayed_one() {
    let genesis = demo::easy_genesis(T0);
    let params = || demo::easy_params(T0, demo::PROJECT_LOCK);
    let mut replayed = ChainState::from_genesis(params(), genesis.clone()).expect("genesis");
    let mut blocks = Vec::new();
    let mut round = Vec::new();
    for _ in 0..9 {
        let (block, next) = demo::mine_plain(&replayed, round);
        replayed.connect_block(&block).expect("a mined block connects");
        round = next;
        blocks.push(block);
    }

    let mut early = ChainState::from_genesis(params(), genesis.clone()).expect("genesis");
    for block in &blocks[..6] {
        early.connect_block(block).expect("connect");
    }
    let bytes = early.snapshot();
    assert_eq!(ChainState::snapshot_height(&bytes), Some(6));
    let mut chain = vec![genesis];
    chain.extend_from_slice(&blocks[..6]);
    let mut read = ChainState::from_snapshot(params(), chain, &bytes).expect("the state is read back");
    assert!(read.snapshot() == bytes, "what is read is what was written");
    assert_eq!((read.height(), read.tip_id()), (6, early.tip_id()));
    assert_eq!(crate::models_hash(&read.tip_models()), crate::models_hash(&early.tip_models()));

    for block in &blocks[6..] {
        read.connect_block(block).expect("a block connects on the state that was read");
    }
    assert!(read.snapshot() == replayed.snapshot(), "the two states differ at block 9");
    assert!(read.conservation());
    for h in 0..=9 {
        let (a, b) = (read.model_at(h).expect("level kept"), replayed.model_at(h).expect("level kept"));
        assert_eq!((a.weights_hash(), a.residual_hash()), (b.weights_hash(), b.residual_hash()), "height {h}");
    }
}

// A state read back reorganises as the replayed one does, within the `MAX_REORG` blocks whose records it keeps.
#[test]
fn a_state_read_back_reorganises_like_the_replayed_one() {
    let mut replayed = new_state();
    let blocks = demo_chain(&mut replayed, 130);
    let mut chain = vec![demo::genesis(T0)];
    chain.extend_from_slice(&blocks);
    let mut read = ChainState::from_snapshot(demo::demo_params(T0, demo::PROJECT_LOCK), chain, &replayed.snapshot())
        .expect("read");
    read.verify_pow = false;

    let branch = [one_block(blocks[99].id(), 101, 1)];
    assert!(matches!(read.reorg(130 - MAX_REORG - 1, &branch), Err(ChainError::ReorgTooDeep)));
    read.reorg(100, &branch).expect("reorg on the state that was read");
    replayed.reorg(100, &branch).expect("reorg on the replayed state");
    assert_eq!((read.height(), read.tip_id()), (101, branch[0].id()));
    // The replayed state keeps more records of old blocks, so the bytes differ; what the rules read is the same.
    assert!(read.utxos() == replayed.utxos());
    assert_eq!(
        (read.unspent_total(), read.coinbase_total(), read.fees_total(), read.tip_id()),
        (replayed.unspent_total(), replayed.coinbase_total(), replayed.fees_total(), replayed.tip_id())
    );
    for h in 60..=101 {
        let (a, b) = (read.model_at(h).expect("level kept"), replayed.model_at(h).expect("level kept"));
        assert!((a.weights_hash(), a.residual_hash()) == (b.weights_hash(), b.residual_hash()), "height {h}");
    }
    assert!(read.conservation());
}

// Bytes that are damaged, cut short, of another tip or of another chain give no state.
#[test]
fn bytes_that_do_not_fit_are_refused() {
    let mut state = new_state();
    let blocks = demo_chain(&mut state, 12);
    let bytes = state.snapshot();
    let params = || demo::demo_params(T0, demo::PROJECT_LOCK);
    let mut chain = vec![demo::genesis(T0)];
    chain.extend_from_slice(&blocks);
    assert!(ChainState::from_snapshot(params(), chain.clone(), &bytes).is_ok());

    for at in [0, 9, 50, bytes.len() / 2, bytes.len() - 1] {
        let mut damaged = bytes.clone();
        damaged[at] ^= 1;
        assert!(ChainState::from_snapshot(params(), chain.clone(), &damaged).is_err(), "byte {at}");
    }
    for len in [0, 40, bytes.len() - 1] {
        assert!(ChainState::from_snapshot(params(), chain.clone(), &bytes[..len]).is_err(), "length {len}");
    }
    assert!(ChainState::snapshot_height(&bytes[..40]).is_none());
    // one block fewer, one block more, another block at the tip
    assert!(ChainState::from_snapshot(params(), chain[..12].to_vec(), &bytes).is_err());
    let mut longer = chain.clone();
    longer.push(one_block(state.tip_id(), 13, 1));
    assert!(ChainState::from_snapshot(params(), longer, &bytes).is_err());
    let mut other_tip = chain[..12].to_vec();
    other_tip.push(one_block(blocks[10].id(), 12, 2));
    assert!(ChainState::from_snapshot(params(), other_tip, &bytes).is_err());
    // the same heights on another genesis block
    let mut other = vec![demo::genesis(T0 + 1)];
    other.extend_from_slice(&blocks);
    assert!(ChainState::from_snapshot(demo::demo_params(T0 + 1, demo::PROJECT_LOCK), other, &bytes).is_err());
}

// After a reorganisation to a shorter chain, the next one keeps the levels it has when they cover what it needs:
// a level far below the tip stays absent, where a rebuild from the genesis block would bring it back.
#[test]
fn a_reorganisation_rebuilds_the_models_only_when_it_needs_them() {
    let mut state = new_state();
    let blocks = demo_chain(&mut state, 130);
    let first = [one_block(blocks[69].id(), 71, 1)];
    state.reorg(70, &first).expect("to a shorter chain");
    assert!(state.model_at(5).is_none(), "fewer levels are kept after the shorter chain");
    let second = [one_block(blocks[69].id(), 71, 2)];
    state.reorg(70, &second).expect("one block deep");
    assert_eq!(state.tip_id(), second[0].id());
    assert!(state.model_at(5).is_none(), "the models were rebuilt from the genesis block");
    assert!(state.model_at(71).is_some() && state.model_at(70).is_some() && state.model_at(69).is_some());
}
