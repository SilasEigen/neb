//! Tests of a dataset upgrade (spec/shares.md section 11.7) on a chain of two model slots mined on the easy
//! parameters of `demo`: up to the upgrade a node with the upgrade builds the same chain as a node without it, the
//! slot of the upgrade starts from its new model in the state of the block before the upgrade height, the blocks
//! from that height on are mined and checked on the new dataset, and a replay from the genesis gives the same state.

use std::sync::Arc;

use crate::block::Block;
use crate::demo;
use crate::model::{models_hash, upgrade_model};
use crate::state::{ChainParams, ChainState, Upgrade};
use neb_pow::shares::{Arch, FileDataset, SharedDataset, SyntheticDataset};

const T0: u32 = 1_800_000_000;
const H: u32 = 5;

/// 64 samples unlike the synthetic dataset: the attempts of the upgraded slot use it from height `H` on.
fn other_dataset() -> SharedDataset {
    let n = 64u32;
    let mut bytes = b"NMD1".to_vec();
    bytes.extend(n.to_le_bytes());
    for i in 0..n {
        bytes.extend((0..784u32).map(|p| ((i * 31 + p * 7) % 8) as u8));
        bytes.push((i % 2) as u8);
    }
    Arc::new(FileDataset::from_bytes(&bytes).expect("an NMD1 dataset"))
}

/// Two slots on the synthetic dataset; with `upgrade`, slot 1 moves to `other_dataset` at height `H`.
fn params(upgrade: bool) -> ChainParams {
    let mut p = demo::easy_params(T0, demo::PROJECT_LOCK);
    let synthetic: SharedDataset = Arc::new(SyntheticDataset::new());
    p.datasets = vec![synthetic.clone(), synthetic];
    if upgrade {
        p.datasets.push(other_dataset());
        p.upgrades = vec![Upgrade { height: H, slot: 1 }];
    }
    p
}

/// Mines and connects `count` blocks, each listing the round of its parent; returns the blocks and the state hash
/// after each.
fn mine(state: &mut ChainState, count: u32) -> (Vec<Block>, Vec<[u8; 32]>) {
    let (mut blocks, mut hashes, mut round) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..count {
        let (block, next) = demo::mine_plain(state, round);
        state.connect_block(&block).expect("a mined block connects");
        round = next;
        hashes.push(models_hash(&state.tip_models()));
        blocks.push(block);
    }
    (blocks, hashes)
}

#[test]
fn an_upgrade_is_well_formed_only_at_a_height_of_its_slot() {
    let mut p = params(true);
    assert_eq!(p.slots(), 2);
    assert_eq!(p.check_upgrades(), Ok(()));
    p.upgrades = vec![Upgrade { height: 6, slot: 1 }];
    assert!(p.check_upgrades().is_err(), "6 is a height of slot 0");
    p.upgrades = vec![Upgrade { height: 1, slot: 1 }];
    assert!(p.check_upgrades().is_err(), "the first block cannot start an upgrade");
    p.upgrades = vec![Upgrade { height: 5, slot: 1 }, Upgrade { height: 3, slot: 1 }];
    assert!(p.check_upgrades().is_err(), "more upgrades than the datasets they bring, and not increasing");
}

#[test]
fn the_dataset_of_a_height_and_of_a_state_follow_the_upgrade() {
    let p = params(true);
    // blocks of slot 1 below H work on dataset 1, from H on on dataset 2; slot 0 never changes
    assert_eq!((1..=8).map(|h| p.dataset_index_at(h)).collect::<Vec<_>>(), [1, 0, 1, 0, 2, 0, 2, 0]);
    assert!(Arc::ptr_eq(p.dataset_at(H), &p.datasets[2]));
    // the state of block H - 2 still holds the old model of slot 1; the state of block H - 1 holds the new one
    assert_eq!(p.dataset_index_of(1, H - 2), 1);
    assert_eq!(p.dataset_index_of(1, H - 1), 2);
    assert_eq!(p.dataset_index_of(0, H + 3), 0);
    let new = upgrade_model(&p.upgrades[0], Arch::V1);
    assert_eq!(p.initial_model_of(&[7; 32], 1, H - 1).hashes(), new.hashes());
    assert_ne!(p.initial_model_of(&[7; 32], 1, H - 2).hashes(), new.hashes());
}

#[test]
fn a_node_with_the_upgrade_builds_the_same_chain_until_the_upgrade() {
    let genesis = demo::easy_genesis(T0);
    let mut old = ChainState::from_genesis(params(false), genesis.clone()).expect("genesis");
    let mut new = ChainState::from_genesis(params(true), genesis).expect("genesis");
    let (old_blocks, old_hashes) = mine(&mut old, H - 1);
    let (new_blocks, new_hashes) = mine(&mut new, H - 1);
    // the same blocks: a miner or a node may take the upgrade at any time before the height
    assert_eq!(
        old_blocks.iter().map(Block::id).collect::<Vec<_>>(),
        new_blocks.iter().map(Block::id).collect::<Vec<_>>()
    );
    // the same states up to block H - 2, then slot 1 holds the new model
    assert_eq!(old_hashes[..H as usize - 2], new_hashes[..H as usize - 2]);
    let (o, n) = (old.tip_models(), new.tip_models());
    assert_eq!(o[0].hashes(), n[0].hashes());
    assert_eq!(n[1].hashes(), upgrade_model(&Upgrade { height: H, slot: 1 }, Arch::V1).hashes());
    assert_ne!(o[1].hashes(), n[1].hashes());
}

#[test]
fn the_upgraded_slot_is_mined_and_trained_on_its_new_dataset() {
    let genesis = demo::easy_genesis(T0);
    let mut state = ChainState::from_genesis(params(true), genesis.clone()).expect("genesis");
    let (blocks, _) = mine(&mut state, H + 3);
    let start = upgrade_model(&Upgrade { height: H, slot: 1 }, Arch::V1).hashes();
    // block H + 1 lists the shares of round H, mined on the new model and dataset, and steps slot 1 away from it
    assert_ne!(state.tip_models()[1].hashes(), start);
    // a node that replays the chain from the genesis block gets the same state
    let mut again = ChainState::from_genesis(params(true), genesis.clone()).expect("genesis");
    for b in &blocks {
        again.connect_block(b).expect("replay");
    }
    assert_eq!(models_hash(&again.tip_models()), models_hash(&state.tip_models()));
    // a node without the upgrade refuses the chain from height H on: the proof or the shares of a block are checked
    // on another model and dataset
    let mut old = ChainState::from_genesis(params(false), genesis).expect("genesis");
    let refused = blocks.iter().position(|b| old.connect_block(b).is_err()).expect("an old node refuses a block");
    assert!(refused as u32 + 1 >= H, "block {} refused before the upgrade", refused + 1);
}

/// Both slots move to nebhash v2 one block apart, as on the network: slot 0 to KEP1 at height 6, slot 1 to RET2 at 7.
fn params_v2() -> ChainParams {
    use neb_pow::shares::SyntheticDataset2;
    let mut p = params(false);
    p.datasets.push(Arc::new(SyntheticDataset2::new(Arch::Kep1)));
    p.datasets.push(Arc::new(SyntheticDataset2::new(Arch::Ret2)));
    p.upgrades = vec![Upgrade { height: 6, slot: 0 }, Upgrade { height: 7, slot: 1 }];
    p
}

#[test]
fn both_slots_move_to_the_convolutional_networks_and_keep_training() {
    let p = params_v2();
    assert_eq!(p.check_upgrades(), Ok(()));
    let genesis = demo::easy_genesis(T0);
    let mut state = ChainState::from_genesis(p.clone(), genesis.clone()).expect("genesis");
    let (blocks, hashes) = mine(&mut state, 12);
    let tip = state.tip_models();
    assert_eq!((tip[0].arch(), tip[1].arch()), (Arch::Kep1, Arch::Ret2));
    // both new models moved away from their initial state (blocks 7 to 12 list shares of the v2 rounds)
    let k0 = upgrade_model(&Upgrade { height: 6, slot: 0 }, Arch::Kep1).hashes();
    let r0 = upgrade_model(&Upgrade { height: 7, slot: 1 }, Arch::Ret2).hashes();
    assert_ne!(tip[0].hashes(), k0);
    assert_ne!(tip[1].hashes(), r0);
    // a replay from the genesis block reaches every intermediate state
    let mut again = ChainState::from_genesis(p, genesis.clone()).expect("genesis");
    for (b, h) in blocks.iter().zip(&hashes) {
        again.connect_block(b).expect("replay");
        assert_eq!(models_hash(&again.tip_models()), *h);
    }
    // a node of the old rules refuses the chain at the first v2 block, not before
    let mut old = ChainState::from_genesis(params(false), genesis).expect("genesis");
    let refused = blocks.iter().position(|b| old.connect_block(b).is_err()).expect("an old node refuses a block");
    assert_eq!(refused + 1, 6);
}
