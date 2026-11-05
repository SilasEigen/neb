//! Tests of the chain crate: locks and witnesses, the body root, block and
//! transaction rules, the conservation invariant and reorganisations; the
//! share rules are in `tests_shares`.
//!
//! Most tests set `verify_pow` to false so that blocks can carry arbitrary
//! nonces; the tests that mine use the easy parameters of `demo`.

use crate::block::Block;
use crate::demo;
use crate::state::{COINBASE_MATURITY, ChainError, ChainState, MAX_REORG, OutPoint, project_share, subsidy};
use crate::tx::{COIN, LOCK_MULTISIG, LOCK_SINGLE, MAX_TX_BYTES, Transaction, TxIn, TxOut};
use ed25519_dalek::{Signer, SigningKey};

const T0: u32 = 1_800_000_000;

/// A state on `demo_params` without target comparison: blocks carry arbitrary
/// nonces.
fn new_state() -> ChainState {
    let params = demo::demo_params(T0, demo::PROJECT_LOCK);
    let mut state = ChainState::from_genesis(params, demo::genesis(T0)).expect("genesis");
    state.verify_pow = false;
    state
}

/// A state whose genesis miner output belongs to `sk`, so it can be spent.
fn new_state_keyed(sk: &SigningKey) -> ChainState {
    let params = demo::demo_params(T0, demo::PROJECT_LOCK);
    let mut state = ChainState::from_genesis(params, demo::genesis_to(T0, lock_of(sk))).expect("genesis");
    state.verify_pow = false;
    state
}

/// Connects `count` plain blocks, 60 s apart, and returns them.
fn demo_chain(state: &mut ChainState, count: u32) -> Vec<Block> {
    let mut added = Vec::new();
    for i in 1..=count {
        let height = state.height() + 1;
        let reward = subsidy(height);
        let nonce = 1000 + i;
        let blk = demo::block(
            state.tip_id(),
            T0 + 60 * i,
            0xC280_0000,
            nonce,
            vec![demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward))],
        );
        state.connect_block(&blk).expect("connect");
        added.push(blk);
    }
    added
}

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn lock_of(sk: &SigningKey) -> [u8; 32] {
    *blake3::hash(&sk.verifying_key().to_bytes()).as_bytes()
}

fn sign_single(sk: &SigningKey, tx: &Transaction, chain_id: &[u8; 32], spent: &TxOut) -> Vec<u8> {
    let msg = tx.sighash(chain_id, 0, spent);
    let sig = sk.sign(&msg);
    let mut w = Vec::new();
    w.extend_from_slice(&sk.verifying_key().to_bytes());
    w.extend_from_slice(&sig.to_bytes());
    w
}

/// A transaction spending `op` (locked to `sk`) into one output of `out_amount`;
/// the difference is the fee.
fn spend(state: &ChainState, op: OutPoint, sk: &SigningKey, out_amount: u64) -> Transaction {
    let spent = state.utxo(&op).expect("utxo").as_txout();
    let mut tx = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: op.txid, prev_index: op.index, witness: Vec::new() }],
        outputs: vec![TxOut { amount: out_amount, lock_type: LOCK_SINGLE, lock_data: lock_of(sk) }],
    };
    tx.inputs[0].witness = sign_single(sk, &tx, &state.chain_id, &spent);
    tx
}

// The txid excludes witnesses: changing a witness leaves it unchanged.
#[test]
fn tx_roundtrip_and_txid_ignores_witnesses() {
    let sk = key(7);
    let mut tx = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [3u8; 32], prev_index: 1, witness: vec![9u8; 96] }],
        outputs: vec![TxOut { amount: 5_000, lock_type: LOCK_SINGLE, lock_data: lock_of(&sk) }],
    };
    let decoded = Transaction::decode(&tx.encode()).expect("decode");
    assert_eq!(decoded, tx);
    let id1 = tx.txid();
    tx.inputs[0].witness = vec![1u8; 96];
    assert_eq!(tx.txid(), id1);
    assert_eq!(Transaction::decode(&tx.encode()).unwrap(), tx);
}

// Single-key lock: only the right key with a valid signature over the spent output passes.
#[test]
fn single_key_witness() {
    let sk = key(7);
    let other = key(8);
    let spent = TxOut { amount: 1_000, lock_type: LOCK_SINGLE, lock_data: lock_of(&sk) };
    let tx = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [0u8; 32], prev_index: 0, witness: Vec::new() }],
        outputs: vec![TxOut { amount: 900, lock_type: LOCK_SINGLE, lock_data: lock_of(&other) }],
    };
    let chain_id = [5u8; 32];
    let good = sign_single(&sk, &tx, &chain_id, &spent);
    assert!(tx.check_witness(&chain_id, 0, &spent).is_err(), "empty witness");
    let mut signed = tx.clone();
    signed.inputs[0].witness = good;
    assert!(signed.check_witness(&chain_id, 0, &spent).is_ok());
    let mut bad = signed.clone();
    bad.inputs[0].witness[96 - 1] ^= 1;
    assert!(bad.check_witness(&chain_id, 0, &spent).is_err(), "tampered signature");
    let mut wrong_key = signed.clone();
    wrong_key.inputs[0].witness[..32].copy_from_slice(&other.verifying_key().to_bytes());
    assert!(wrong_key.check_witness(&chain_id, 0, &spent).is_err(), "key does not match the lock");
    // The signature is bound to the spent output.
    let spent2 = TxOut { amount: 1_001, lock_type: LOCK_SINGLE, lock_data: lock_of(&sk) };
    assert!(signed.check_witness(&chain_id, 0, &spent2).is_err());
}

// Multisig 2-of-3: the signature indexes must strictly increase and every signature must verify.
#[test]
fn multisig_two_of_three() {
    let k1 = key(1);
    let k2 = key(2);
    let k3 = key(3);
    let keys = [k1.verifying_key().to_bytes(), k2.verifying_key().to_bytes(), k3.verifying_key().to_bytes()];
    let lock_data = Transaction::multisig_lock_data(2, &keys);
    let spent = TxOut { amount: 1_000, lock_type: LOCK_MULTISIG, lock_data };
    let tx = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [0u8; 32], prev_index: 0, witness: Vec::new() }],
        outputs: vec![TxOut { amount: 500, lock_type: LOCK_SINGLE, lock_data: [0u8; 32] }],
    };
    let chain_id = [6u8; 32];
    let msg = tx.sighash(&chain_id, 0, &spent);
    let mut w = vec![2u8, 3u8];
    for k in &keys {
        w.extend_from_slice(k);
    }
    for (idx, k) in [(0u8, &k1), (2u8, &k3)] {
        w.push(idx);
        w.extend_from_slice(&k.sign(&msg).to_bytes());
    }
    let mut signed = tx.clone();
    signed.inputs[0].witness = w.clone();
    assert!(signed.check_witness(&chain_id, 0, &spent).is_ok());
    // Swap the two (index, signature) entries: indexes then decrease.
    let (a, b) = (w.len() - 130, w.len() - 65);
    let head = &w[..a];
    let first = &w[a..b];
    let second = &w[b..];
    let mut swapped = head.to_vec();
    swapped.extend_from_slice(second);
    swapped.extend_from_slice(first);
    let mut resequenced = signed.clone();
    resequenced.inputs[0].witness = swapped;
    assert!(resequenced.check_witness(&chain_id, 0, &spent).is_err(), "indexes must increase");
    // A tampered signature fails.
    let mut tampered = signed.clone();
    let n = tampered.inputs[0].witness.len();
    tampered.inputs[0].witness[n - 1] ^= 1;
    assert!(tampered.check_witness(&chain_id, 0, &spent).is_err());
    // A different m changes the expected witness length.
    let mut wrong_m = signed.clone();
    wrong_m.inputs[0].witness[0] = 3;
    assert!(wrong_m.check_witness(&chain_id, 0, &spent).is_err());
}

// The height in the coinbase witness keeps coinbase txids unique.
#[test]
fn coinbase_txids_are_unique_per_height() {
    let a = demo::coinbase(1, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000);
    let b = demo::coinbase(2, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000);
    assert_ne!(a.txid(), b.txid());
    // Same height and content give the same txid.
    let c = demo::coinbase(1, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000);
    assert_eq!(a.txid(), c.txid());
}

// Any change to a transaction changes body_root; an odd Merkle leaf is promoted, not duplicated.
#[test]
fn body_root_commits_to_the_body() {
    let cb = demo::coinbase(0, demo::MINER_LOCK, demo::PROJECT_LOCK, 19);
    let tx_a = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [1u8; 32], prev_index: 0, witness: vec![] }],
        outputs: vec![TxOut { amount: 1, lock_type: LOCK_SINGLE, lock_data: [0u8; 32] }],
    };
    let mut tx_b = tx_a.clone();
    tx_b.outputs[0].amount = 2;
    let b1 = demo::block([0u8; 32], T0, 0xC280_0000, 0, vec![cb.clone(), tx_a.clone()]);
    let b2 = demo::block([0u8; 32], T0, 0xC280_0000, 0, vec![cb.clone(), tx_b]);
    let b3 = demo::block([0u8; 32], T0, 0xC280_0000, 0, vec![cb.clone(), tx_a.clone()]);
    assert_eq!(b1.body_root(), b3.body_root());
    assert_ne!(b1.body_root(), b2.body_root());
    assert!(b1.body_ok());
    let mut tampered = b1.clone();
    tampered.txs[1].outputs[0].amount += 1;
    assert!(!tampered.body_ok());
    // Three transactions: the odd leaf is promoted, so [cb, a, a] and [cb, a] differ.
    let b4 = demo::block([0u8; 32], T0, 0xC280_0000, 0, vec![cb.clone(), tx_a.clone(), tx_a.clone()]);
    let b5 = demo::block([0u8; 32], T0, 0xC280_0000, 0, vec![cb, tx_a]);
    assert_ne!(b4.body_root(), b5.body_root());
    assert_eq!(Block::decode(&b4.encode()).unwrap(), b4);
}

// A coinbase output is unspendable before 100 blocks; after that its fee is claimed and U + F = C holds.
#[test]
fn chain_happy_path_conservation_and_maturity() {
    let sk = key(9);
    let mut state = new_state_keyed(&sk);
    demo_chain(&mut state, 3);
    assert_eq!(state.height(), 3);
    assert_eq!(state.unspent_total(), 4 * 20 * 100_000_000); // 4 blocks x 20 coins
    assert!(state.conservation());
    // The genesis coinbase output is not mature at height 4.
    let genesis_cb = demo::genesis_to(T0, lock_of(&sk)).txs[0].txid();
    let op = OutPoint { txid: genesis_cb, index: 0 };
    let sp = spend(&state, op, &sk, 18 * 100_000_000);
    let height = state.height() + 1;
    let blk = demo::block(
        state.tip_id(),
        T0 + 60 * 4,
        0xC280_0000,
        77,
        vec![
            demo::coinbase(
                height,
                demo::MINER_LOCK,
                demo::PROJECT_LOCK,
                subsidy(height) - project_share(subsidy(height)),
            ),
            sp.clone(),
        ],
    );
    match state.connect_block(&blk) {
        Err(ChainError::Immature(_)) => {}
        other => panic!("expected Immature, got {other:?}"),
    }
    assert!(state.conservation());
    assert_eq!(state.height(), 3);
    // Past maturity the same output is spendable; the coinbase claims 1 coin of fee.
    demo_chain(&mut state, 100);
    assert!(state.height() >= COINBASE_MATURITY);
    let sp = spend(&state, op, &sk, 18 * 100_000_000);
    let height = state.height() + 1;
    let fee_transfer = sp.clone();
    let blk = demo::block(
        state.tip_id(),
        state.tip_time() + 60,
        0xC280_0000,
        78,
        vec![
            demo::coinbase(
                height,
                demo::MINER_LOCK,
                demo::PROJECT_LOCK,
                subsidy(height) - project_share(subsidy(height)) + 100_000_000,
            ),
            fee_transfer,
        ],
    );
    state.connect_block(&blk).expect("mature spend");
    assert!(state.conservation());
    assert_eq!(state.fees_total(), 100_000_000);
    assert_eq!(state.unspent_total(), state.coinbase_total() - state.fees_total());
}

// Invalid blocks (reward, share output, body_root, coinbase shape, unknown input) are refused before the state changes.
#[test]
fn connect_rejects_bad_blocks_and_keeps_state_untouched() {
    let mut state = new_state();
    demo_chain(&mut state, 1);
    let height = state.height() + 1;
    let good =
        demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, subsidy(height) - project_share(subsidy(height)));
    let tip = state.tip_id();
    let mk = move |txs: Vec<Transaction>| demo::block(tip, T0 + 3600, 0xC280_0000, 5, txs);

    let over = {
        let mut cb = good.clone();
        cb.outputs[0].amount += 1;
        mk(vec![cb])
    };
    assert!(matches!(state.connect_block(&over), Err(ChainError::RewardTooHigh)));

    let wrong_share = {
        let mut cb = good.clone();
        cb.outputs[1].amount += 1;
        cb.outputs[0].amount -= 1;
        mk(vec![cb])
    };
    assert!(matches!(state.connect_block(&wrong_share), Err(ChainError::BadCoinbase(_))));

    let bad_root = {
        let mut b = mk(vec![good.clone()]);
        b.header.body_root[0] ^= 1;
        b
    };
    assert!(matches!(state.connect_block(&bad_root), Err(ChainError::BodyRootMismatch)));

    let plain_tx = |seed: u8| Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [seed; 32], prev_index: 0, witness: vec![0u8; 96] }],
        outputs: vec![TxOut { amount: 1, lock_type: LOCK_SINGLE, lock_data: [0u8; 32] }],
    };
    let no_coinbase = mk(vec![plain_tx(9)]);
    assert!(matches!(state.connect_block(&no_coinbase), Err(ChainError::BadCoinbase(_))));

    // An input that is not in the UTXO set.
    let blk = mk(vec![good.clone(), plain_tx(7)]);
    assert!(matches!(state.connect_block(&blk), Err(ChainError::UnknownInput(_))));

    // Nothing rejected so far changed the state.
    assert_eq!(state.height(), 1);
    assert!(state.conservation());

    // A second coinbase-shaped transaction is refused.
    let second_cb = {
        let mut w = Vec::new();
        w.extend_from_slice(&7u32.to_le_bytes());
        w.extend_from_slice(&[0u8; 8]);
        Transaction {
            version: 1,
            inputs: vec![TxIn { prev_txid: [0u8; 32], prev_index: 0xFFFF_FFFF, witness: w }],
            outputs: vec![TxOut { amount: 1, lock_type: LOCK_SINGLE, lock_data: [0u8; 32] }],
        }
    };
    let blk = mk(vec![good.clone(), second_cb]);
    assert!(matches!(state.connect_block(&blk), Err(ChainError::BadCoinbase(_))));
}

// Two transactions of one block spending the same output.
#[test]
fn double_spend_in_one_block_is_rejected() {
    let sk = key(9);
    let mut state = new_state_keyed(&sk);
    demo_chain(&mut state, 101);
    let genesis_cb = demo::genesis_to(T0, lock_of(&sk)).txs[0].txid();
    let op = OutPoint { txid: genesis_cb, index: 0 };
    let a = spend(&state, op, &sk, 5 * COIN);
    let b = spend(&state, op, &sk, 4 * COIN);
    let height = state.height() + 1;
    let blk = demo::block(
        state.tip_id(),
        state.tip_time() + 60,
        0xC280_0000,
        9,
        vec![
            demo::coinbase(
                height,
                demo::MINER_LOCK,
                demo::PROJECT_LOCK,
                subsidy(height) - project_share(subsidy(height)),
            ),
            a,
            b,
        ],
    );
    assert!(matches!(state.connect_block(&blk), Err(ChainError::DoubleSpend(_))));
    assert!(state.conservation());
}

// After a reorganisation the UTXO set and sums equal those of a chain built directly on the new branch.
#[test]
fn reorg_lands_on_the_exact_state_of_the_new_branch() {
    let mut state = new_state();
    let a_blocks = demo_chain(&mut state, 4); // heights 1..=4
    assert_eq!(state.height(), 4);

    // A competing branch from height 2, one block longer.
    let mut b_blocks = Vec::new();
    {
        let mut probe = new_state();
        for blk in &a_blocks[..2] {
            probe.connect_block(blk).unwrap();
        }
        for i in 3u32..=5 {
            let height = i;
            let reward = subsidy(height);
            let blk = demo::block(
                probe.tip_id(),
                T0 + 1000 + 60 * i,
                0xC280_0000,
                500 + i,
                vec![demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward))],
            );
            probe.connect_block(&blk).unwrap();
            b_blocks.push(blk);
        }
    }
    let tip_before = state.tip_id();
    state.reorg(2, &b_blocks).expect("reorg");
    assert_eq!(state.height(), 5);
    assert_ne!(state.tip_id(), tip_before);
    assert!(state.conservation());

    // The same state, built without a reorganisation.
    let mut reference = new_state();
    for blk in &a_blocks[..2] {
        reference.connect_block(blk).unwrap();
    }
    for blk in &b_blocks {
        reference.connect_block(blk).unwrap();
    }
    assert_eq!(state.utxos(), reference.utxos());
    assert_eq!(state.unspent_total(), reference.unspent_total());
    assert_eq!(state.coinbase_total(), reference.coinbase_total());
}

// A branch removing more than MAX_REORG blocks is refused whatever its length.
#[test]
fn reorg_deeper_than_the_limit_is_refused() {
    let mut state = new_state();
    let blocks = demo_chain(&mut state, MAX_REORG + 2);
    // A branch from height 1 would remove MAX_REORG + 1 blocks.
    let mut branch = Vec::new();
    let mut probe = new_state();
    probe.connect_block(&blocks[0]).unwrap();
    let mut parent = probe.tip_id();
    for i in 2..=(MAX_REORG + 3) {
        let height = i;
        let reward = subsidy(height);
        let blk = demo::block(
            parent,
            T0 + 5000 + 60 * i,
            0xC280_0000,
            9000 + i,
            vec![demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward))],
        );
        parent = blk.id();
        branch.push(blk);
    }
    assert!(matches!(state.reorg(1, &branch), Err(ChainError::ReorgTooDeep)));
    assert!(state.conservation());
}

// Reorganisations to shorter chains leave fewer than MODEL_DEPTH + 1 models; the next ones still work (spec/shares.md section 11.4).
#[test]
fn repeated_reorganisations_to_shorter_chains_rebuild_the_models() {
    let mut state = new_state();
    let blocks = demo_chain(&mut state, 130);
    let one_block_on = |parent_blocks: &[Block], salt: u32| -> Vec<Block> {
        let mut probe = new_state();
        for b in parent_blocks {
            probe.connect_block(b).unwrap();
        }
        let height = probe.height() + 1;
        let reward = subsidy(height);
        vec![demo::block(
            probe.tip_id(),
            T0 + 9000 + salt,
            0xC280_0000,
            7000 + salt,
            vec![demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward))],
        )]
    };
    let first = one_block_on(&blocks[..70], 1);
    state.reorg(70, &first).expect("first reorg");
    assert_eq!(state.height(), 71);
    let mut chain: Vec<Block> = blocks[..70].to_vec();
    chain.extend(first);
    let second = one_block_on(&chain[..11], 2);
    state.reorg(11, &second).expect("second reorg");
    assert_eq!(state.height(), 12);
    assert!(state.model_at(12).is_some() && state.model_at(11).is_some());
    state.disconnect_tip().expect("disconnect");
    state.disconnect_tip().expect("disconnect");
    assert!(state.model_at(state.height()).is_some() && state.model_at(state.height() - 1).is_some());
    assert!(state.conservation());
}

// A block creating an outpoint that exists leaves the state untouched.
#[test]
fn a_duplicate_created_outpoint_is_refused_before_any_change() {
    let mut state = new_state();
    let blocks = demo_chain(&mut state, 1);
    let before = (state.utxos().clone(), state.unspent_total(), state.coinbase_total(), state.height());
    let mut copy = blocks[0].clone();
    copy.header.parent = state.tip_id();
    // Same coinbase transaction as block 1: its txid is already in the UTXO set.
    assert!(matches!(state.connect_block(&copy), Err(ChainError::DoubleSpend(_))));
    assert_eq!((state.utxos().clone(), state.unspent_total(), state.coinbase_total(), state.height()), before);
}

// A transaction that announces a million inputs or outputs but holds none fails to decode.
#[test]
fn huge_counts_fail_without_input_bytes() {
    let mut bytes = vec![1u8, 0xFE];
    bytes.extend_from_slice(&1_000_000u32.to_le_bytes());
    assert!(Transaction::decode(&bytes).is_err());
}

// Each header rule (version, parent, bits, median time, 120 s future bound) has its own error.
#[test]
fn header_checks_reject_and_a_good_header_passes() {
    use crate::state::HeaderError;
    use neb_block::Header;
    let state = new_state();
    let now = T0 + 600;
    let base = Header {
        version: 1,
        parent: state.tip_id(),
        body_root: demo::genesis(T0).header.body_root,
        time: T0 + 61,
        bits: 0xC280_0000,
        nonce: 12345,
    };
    let mut h = base;
    h.version = 2;
    assert!(matches!(state.check_header(&h, now), Err(HeaderError::Version)));
    let mut h = base;
    h.parent = [9u8; 32];
    assert!(matches!(state.check_header(&h, now), Err(HeaderError::UnknownParent)));
    let mut h = base;
    h.bits = 0xC280_0001;
    assert!(matches!(state.check_header(&h, now), Err(HeaderError::BadBits)));
    let mut h = base;
    h.time = T0; // equal to the median (the genesis time)
    assert!(matches!(state.check_header(&h, now), Err(HeaderError::StaleTime)));
    let mut h = base;
    h.time = now + 121;
    assert!(matches!(state.check_header(&h, now), Err(HeaderError::FutureTime)));
    // The proof of work is not part of the header rules.
    assert!(state.check_header(&base, now).is_ok());
}

// Oversize transaction and zero-amount output.
#[test]
fn transaction_limits_hold() {
    let mut state = new_state();
    demo_chain(&mut state, 1);
    let height = state.height() + 1;
    // A transaction of more than 100 000 bytes.
    let big = {
        let sk = key(4);
        Transaction {
            version: 1,
            inputs: vec![TxIn { prev_txid: [0u8; 32], prev_index: 0, witness: vec![0u8; MAX_TX_BYTES] }],
            outputs: vec![TxOut { amount: 1, lock_type: LOCK_SINGLE, lock_data: lock_of(&sk) }],
        }
    };
    let cb =
        demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, subsidy(height) - project_share(subsidy(height)));
    let blk = demo::block(state.tip_id(), T0 + 3600, 0xC280_0000, 4, vec![cb, big]);
    assert!(matches!(state.connect_block(&blk), Err(ChainError::TxTooLarge { .. })));
    // A coinbase output of zero.
    let mut cb0 =
        demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, subsidy(height) - project_share(subsidy(height)));
    cb0.outputs[0].amount = 0;
    let blk = demo::block(state.tip_id(), T0 + 3600, 0xC280_0000, 4, vec![cb0]);
    assert!(matches!(state.connect_block(&blk), Err(ChainError::BadCoinbase(_))));
}

// The window of kept models is the tip and 100 levels below it.
#[test]
fn models_of_the_last_hundred_levels_are_kept() {
    let mut state = new_state();
    demo_chain(&mut state, 130);
    assert!(state.model_at(130).is_some());
    assert!(state.model_at(30).is_some());
    assert!(state.model_at(29).is_none());
    assert!(state.model_at(131).is_none());
}
