//! Tests of the work service: frame codecs, mempool admission, the Merkle
//! path of the coinbase, templates, submissions and share bookkeeping. The
//! chain is built with `neb_chain::demo`; most tests turn target comparison
//! off, the share test mines on the easy parameters.

use crate::frame;
use crate::frames::{
    FrameOut, Hello, Job, MSG_JOB, MSG_MODEL, MSG_SHARE, ResultMsg, ShareMsg, Submit, decode_model, model_frame,
    parse_frame,
};
use crate::mempool::{Mempool, MempoolError};
use crate::template::{MAX_STORED_SHARES, Work, branch_for_coinbase, coinbase_outputs, merkle_root_of};
use ed25519_dalek::{Signer, SigningKey};
use neb_chain::ShareRecord;
use neb_chain::demo;
use neb_chain::state::{ChainState, OutPoint, project_share, subsidy};
use neb_chain::tx::{LOCK_SINGLE, Transaction, TxIn, TxOut, h};

const T0: u32 = 1_800_000_000;

// A project share of 0 base units gives a coinbase with one output (spec/node.md section 5.3).
#[test]
fn the_coinbase_has_one_output_when_the_project_share_is_zero() {
    assert_eq!(coinbase_outputs(19, 0, [1; 32], [2; 32]).len(), 1);
    let two = coinbase_outputs(15, 4, [1; 32], [2; 32]);
    assert_eq!(two.len(), 2);
    assert_eq!((two[1].amount, two[1].lock_data), (4, [2; 32]));
}

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn lock_of(sk: &SigningKey) -> [u8; 32] {
    *blake3::hash(&sk.verifying_key().to_bytes()).as_bytes()
}

/// A chain whose genesis miner output belongs to `sk`; target comparison is off.
fn chain_with(sk: &SigningKey) -> ChainState {
    let params = demo::demo_params(T0, demo::PROJECT_LOCK);
    let mut state = ChainState::from_genesis(params, demo::genesis_to(T0, lock_of(sk))).expect("genesis");
    state.verify_pow = false;
    state
}

/// Connects `n` plain blocks.
fn demo_blocks(state: &mut ChainState, n: u32) {
    for i in 1..=n {
        let height = state.height() + 1;
        let reward = subsidy(height);
        let blk = demo::block(
            state.tip_id(),
            T0 + 60 * i,
            0xC280_0000,
            100 + i,
            vec![demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward))],
        );
        state.connect_block(&blk).expect("connect");
    }
}

fn sign_single(sk: &SigningKey, tx: &Transaction, chain_id: &[u8; 32], spent: &TxOut) -> Vec<u8> {
    let msg = tx.sighash(chain_id, 0, spent);
    let sig = sk.sign(&msg);
    let mut w = Vec::new();
    w.extend_from_slice(&sk.verifying_key().to_bytes());
    w.extend_from_slice(&sig.to_bytes());
    w
}

/// A signed transaction spending `op` (locked to `sk`) into one output.
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

fn leaf_of(bytes: &[u8]) -> [u8; 32] {
    h(b"leaf", &[blake3::hash(bytes).as_bytes()])
}

// Every frame type decodes to what was encoded; partial input asks for more bytes; a bad type or an oversize length is an error.
#[test]
fn frames_roundtrip() {
    let hello = Hello { version: 1, name: "neb-miner 0.1.0".into() };
    let f = frame(crate::frames::MSG_HELLO, &hello.encode());
    match parse_frame(&f).unwrap() {
        FrameOut::Message { msg_type, payload, consumed } => {
            assert_eq!(msg_type, crate::frames::MSG_HELLO);
            assert_eq!(consumed, f.len());
            assert_eq!(Hello::decode(payload).unwrap(), hello);
        }
        _ => panic!("expected a message"),
    }
    // Partial frames ask for the missing byte count.
    match parse_frame(&f[..2]).unwrap() {
        FrameOut::Need(n) => assert_eq!(n, 2),
        _ => panic!("expected Need"),
    }
    match parse_frame(&f[..6]).unwrap() {
        FrameOut::Need(n) => assert_eq!(n, f.len() - 6),
        _ => panic!("expected Need"),
    }

    let job = Job {
        id: 7,
        clean: true,
        version: 1,
        parent: [3u8; 32],
        time_start: T0,
        time_window: 3600,
        bits: 0xC280_0000,
        target: [9u8; 32],
        share_target: [10u8; 32],
        coinbase_prefix: vec![1, 2, 3],
        coinbase_suffix: vec![4, 5],
        extranonce_size: 16,
        branch: vec![[1u8; 32], [2u8; 32]],
        tx_count: 3,
        shares_root: [8u8; 32],
        height: 42,
        expiry: T0 + 300,
    };
    let jf = job.encode();
    match parse_frame(&jf).unwrap() {
        FrameOut::Message { msg_type, payload, .. } => {
            assert_eq!(msg_type, MSG_JOB);
            assert_eq!(Job::decode(payload).unwrap(), job);
        }
        _ => panic!(),
    }

    let sub = Submit { id: 7, extranonce: vec![0xAB; 16], time: T0 + 10, nonce: 99 };
    match parse_frame(&sub.encode()).unwrap() {
        FrameOut::Message { payload, .. } => assert_eq!(Submit::decode(payload).unwrap(), sub),
        _ => panic!(),
    }
    let res = ResultMsg { accepted: true, code: 0 };
    match parse_frame(&res.encode()).unwrap() {
        FrameOut::Message { payload, .. } => assert_eq!(ResultMsg::decode(payload).unwrap(), res),
        _ => panic!(),
    }
    let share = ShareMsg { job_id: 7, nonce: 0xDEAD_BEEF };
    match parse_frame(&share.encode()).unwrap() {
        FrameOut::Message { msg_type, payload, .. } => {
            assert_eq!(msg_type, MSG_SHARE);
            assert_eq!(ShareMsg::decode(payload).unwrap(), share);
            assert!(ShareMsg::decode(&payload[1..]).is_err());
        }
        _ => panic!(),
    }
    // A bad frame type, and a length field above the 2 MiB bound.
    assert!(parse_frame(&frame(9, &[])).is_err());
    let mut short = job.encode();
    short[..4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(parse_frame(&short).is_err());
}

// Each admission rule has its own error; the accepted transaction reports its fee.
#[test]
fn mempool_accept_and_reject() {
    let sk = key(9);
    let mut state = chain_with(&sk);
    demo_blocks(&mut state, 100);
    let mut pool = Mempool::with_limit(10_000_000);
    let genesis_out = OutPoint { txid: demo::genesis_to(T0, lock_of(&sk)).txs[0].txid(), index: 0 };

    // A tampered signature; nothing is stored.
    let mut bad = spend(&state, genesis_out, &sk, 18 * 100_000_000);
    let n = bad.inputs[0].witness.len();
    bad.inputs[0].witness[n - 1] ^= 1;
    assert!(matches!(pool.accept(&state, &bad), Err(MempoolError::Witness { .. })));

    // Outputs above inputs, correctly signed.
    let mut over = spend(&state, genesis_out, &sk, 19 * 100_000_000);
    over.outputs[0].amount = 20 * 100_000_000;
    let spent = state.utxo(&genesis_out).unwrap().as_txout();
    over.inputs[0].witness = sign_single(&sk, &over, &state.chain_id, &spent);
    assert_eq!(pool.accept(&state, &over), Err(MempoolError::Overspend));

    // An unknown input (built by hand: `spend` reads the UTXO).
    let tx_unknown = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [7u8; 32], prev_index: 0, witness: vec![0u8; 96] }],
        outputs: vec![TxOut { amount: 1, lock_type: LOCK_SINGLE, lock_data: lock_of(&sk) }],
    };
    assert!(matches!(pool.accept(&state, &tx_unknown), Err(MempoolError::UnknownInput(_))));

    // A coinbase-shaped transaction.
    let cb = demo::coinbase(2, demo::MINER_LOCK, demo::PROJECT_LOCK, 1);
    assert_eq!(pool.accept(&state, &cb), Err(MempoolError::CoinbaseShaped));

    // A valid one.
    let tx = spend(&state, genesis_out, &sk, 18 * 100_000_000);
    let fee = pool.accept(&state, &tx).expect("accept");
    assert_eq!(fee, 100_000_000);
    assert_eq!(pool.len(), 1);

    // The same transaction twice.
    assert_eq!(pool.accept(&state, &tx), Err(MempoolError::Duplicate(tx.txid())));

    // The same input with other outputs conflicts with the first.
    let conflict = spend(&state, genesis_out, &sk, 17 * 100_000_000);
    assert!(matches!(pool.accept(&state, &conflict), Err(MempoolError::Conflict(_))));
}

// A miner folding its coinbase leaf with the branch must get the node's Merkle root for any transaction count.
#[test]
fn branch_matches_the_full_root_for_every_size() {
    for n in 0..6usize {
        let others: Vec<[u8; 32]> = (0..n).map(|i| [(i as u8) + 1; 32]).collect();
        let cb_leaf = [0xEEu8; 32];
        let mut full = vec![cb_leaf];
        full.extend_from_slice(&others);
        let full_root = merkle_root_of(&full);
        let branch = branch_for_coinbase(&others);
        let mut hh = cb_leaf;
        for s in &branch {
            hh = h(b"node", &[&hh, s]);
        }
        assert_eq!(hh, full_root, "size {n}");
    }
}

// End to end: job, miner-side root, submit, connect, mempool cleanup.
#[test]
fn template_submit_roundtrip_connects() {
    let sk = key(9);
    let mut state = chain_with(&sk);
    demo_blocks(&mut state, 100);
    let mut pool = Mempool::with_limit(10_000_000);
    let genesis_out = OutPoint { txid: demo::genesis_to(T0, lock_of(&sk)).txs[0].txid(), index: 0 };
    let tx = spend(&state, genesis_out, &sk, 18 * 100_000_000);
    pool.accept(&state, &tx).expect("accept");

    let mut work = Work::new();
    let t = T0 + 300;
    let id = work.make_job(&state, &pool, lock_of(&key(2)), t).expect("job");
    let template = work.job(id).unwrap();
    let job = template.job.clone();
    assert_eq!(job.branch.len(), 1, "one mempool transaction -> one sibling");

    // Emulate the miner: pick an extranonce and fold the branch.
    let ex = vec![0xABu8; job.extranonce_size as usize];
    let mut cb_bytes = job.coinbase_prefix.clone();
    cb_bytes.extend_from_slice(&ex);
    cb_bytes.extend_from_slice(&job.coinbase_suffix);
    let mut hh = leaf_of(&cb_bytes);
    for s in &job.branch {
        hh = h(b"node", &[&hh, s]);
    }
    let full: Vec<[u8; 32]> =
        std::iter::once(leaf_of(&cb_bytes)).chain(template.txs[1..].iter().map(|tx| leaf_of(&tx.encode()))).collect();
    assert_eq!(hh, merkle_root_of(&full), "the miner's root equals the node's");

    // Submit and connect.
    let block = work.submit(id, &ex, job.time_start + 5, 4242).expect("submit");
    assert!(block.body_ok());
    state.connect_block(&block).expect("connect");
    assert!(state.conservation());
    pool.remove_confirmed(&block.txs);
    assert_eq!(pool.len(), 0);
    assert_eq!(state.height(), 101);
}

// A submission with a wrong extranonce length, a time outside the window or an unknown job is refused.
#[test]
fn submit_rejects_bad_shapes() {
    let sk = key(9);
    let mut state = chain_with(&sk);
    demo_blocks(&mut state, 1);
    let pool = Mempool::new();
    let mut work = Work::new();
    let id = work.make_job(&state, &pool, lock_of(&key(2)), T0 + 300).expect("job");
    let job = &work.job(id).unwrap().job;
    assert!(work.submit(id, &[0u8; 1], job.time_start, 1).is_err(), "wrong extranonce length");
    assert!(work.submit(id, &[0u8; 16], job.time_start + job.time_window + 1, 1).is_err(), "time outside the window");
    assert!(work.submit(999, &[0u8; 16], job.time_start, 1).is_err(), "unknown job");
}

// The model frame round-trips, and damage or truncation is detected by the hash and the size.
#[test]
fn the_model_frame_carries_the_weights_and_checks_their_hash() {
    let model = neb_pow::shares::Model::from_seed(&[5u8; 32]);
    let f = model_frame(&model);
    let FrameOut::Message { msg_type, payload, .. } = parse_frame(&f).unwrap() else { panic!() };
    assert_eq!(msg_type, MSG_MODEL);
    assert_eq!(payload.len(), 32 + 50_816);
    let copy = decode_model(payload).unwrap();
    assert_eq!(copy.weights_hash(), model.weights_hash());
    let mut bad = payload.to_vec();
    bad[40] ^= 1;
    assert!(decode_model(&bad).is_err());
    assert!(decode_model(&payload[..payload.len() - 1]).is_err());
}

// The header built by the miner from the job equals the header of the node's block.
#[test]
fn the_miner_assembles_the_header_of_the_node() {
    let sk = key(9);
    let mut state = chain_with(&sk);
    demo_blocks(&mut state, 3);
    let mut work = Work::new();
    let id = work.make_job(&state, &Mempool::new(), lock_of(&key(2)), T0 + 300).expect("job");
    let job = work.job(id).unwrap().job.clone();
    let ex = vec![0x5Au8; job.extranonce_size as usize];
    let block = work.submit(id, &ex, job.time_start + 7, 1234).unwrap();
    assert_eq!(job.header(&ex, job.time_start + 7, 1234), block.header);
    assert_eq!(job.shares_root, neb_chain::block::shares_root(&block.shares));
    assert!(block.body_ok());
    // The share target is the block target times 32 (saturating).
    let want = neb_chain::share_target(job.bits).unwrap();
    assert_eq!(job.share_target, want.0);
}

// Shares: invalid or unknown ones are refused, duplicates are not counted, and a block built on the reference header makes the next job list the reported nonces sorted.
#[test]
fn reported_shares_are_checked_kept_and_listed_by_the_next_job() {
    let mut state = demo_easy_state();
    let ds = std::sync::Arc::clone(&state.params.dataset);
    let (b1, round1) = demo::mine_plain(&state, vec![]);
    state.connect_block(&b1).unwrap();
    let mut work = Work::new();
    let pool = Mempool::new();
    let id = work.make_job(&state, &pool, lock_of(&key(2)), T0 + 300).unwrap();
    let template = work.job(id).unwrap();
    let (job, attack, base) = (template.job.clone(), template.model.clone(), template.base);
    assert_eq!(attack.hashes(), state.tip_model().hashes());
    assert_eq!(base, job.header(&vec![0u8; job.extranonce_size as usize], job.time_start, 0));
    assert!(!round1.is_empty());

    // Nonces that meet and that miss the share target of the job.
    let target = neb_block::Target(job.share_target);
    let mut buffers = neb_pow::shares::ShareBuffers::new();
    let (good, bad): (Vec<u32>, Vec<u32>) = (0..40u32)
        .partition(|&n| target.admits(&buffers.ticket(&attack, &*ds, &neb_block::Header { nonce: n, ..base }.id())));
    assert!(good.len() >= 5 && !bad.is_empty());
    // A reported nonce is the record of the reference header, kept for its parent (the tip at the time of the job).
    let rec = |nonce: u32| ShareRecord { body_root: base.body_root, time: base.time, nonce };
    assert!(work.add_share(id, bad[0], &*ds, true).is_err(), "above the share target");
    assert!(work.add_share(999, good[0], &*ds, true).is_err(), "unknown job");
    assert_eq!(work.add_share(id, good[3], &*ds, true), Ok(Some((base.parent, rec(good[3])))));
    assert_eq!(work.add_share(id, good[0], &*ds, true), Ok(Some((base.parent, rec(good[0])))));
    assert_eq!(work.add_share(id, good[3], &*ds, true), Ok(None), "a duplicate");
    assert!(work.has_record(&base.parent, &rec(good[3])) && !work.has_record(&base.parent, &rec(bad[0])));

    // The winner is mined on the reference header, so the next job lists the
    // reported shares, sorted, plus the winner (its ticket meets the block
    // target, hence also the easier share target).
    let winner = (0u32..)
        .find(|&n| {
            neb_block::Target(job.target).admits(&buffers.ticket(
                &attack,
                &*ds,
                &neb_block::Header { nonce: n, ..base }.id(),
            ))
        })
        .unwrap();
    work.add_share(id, winner, &*ds, true).expect("the winner meets the share target");
    let block = work.submit(id, &vec![0u8; job.extranonce_size as usize], job.time_start, winner).unwrap();
    assert_eq!(block.header, neb_block::Header { nonce: winner, ..base });
    state.connect_block(&block).expect("the block connects");
    let next = work.make_job(&state, &pool, lock_of(&key(2)), T0 + 400).unwrap();
    let listed = work.job(next).unwrap().shares.clone();
    let mut want = vec![rec(good[0]), rec(good[3]), rec(winner)];
    want.sort_unstable();
    want.dedup();
    assert_eq!(listed, want);
    assert!(want.contains(&ShareRecord::of_header(&block.header)), "the winner is a record of its own round");
    // The block built from that job carries the list.
    let block2 = work.submit(next, &[1u8; 16], work.job(next).unwrap().job.time_start, 0).unwrap();
    assert_eq!(block2.shares, want);
}

// Records of other miners (relayed by peers) are kept by parent, listed by the next job with the local ones, and the store is bounded.
#[test]
fn records_of_other_miners_are_kept_by_parent_listed_and_bounded() {
    let mut state = demo_easy_state();
    let ds = std::sync::Arc::clone(&state.params.dataset);
    let (b1, round1) = demo::mine_plain(&state, vec![]);
    state.connect_block(&b1).unwrap();
    let (b2, _) = demo::mine_plain(&state, round1);
    state.connect_block(&b2).unwrap();
    let q = b1.id();
    // Valid shares of the round of b2: headers whose parent is b1, with another body root and another time.
    let mut foreign: Vec<ShareRecord> = (0..60)
        .map(|n| ShareRecord { body_root: [0xEE; 32], time: b2.header.time + 3, nonce: n })
        .filter(|r| state.check_share(&q, r) == Some(true))
        .collect();
    foreign.sort_unstable();
    assert!(foreign.len() >= 10);

    let mut work = Work::new();
    assert_eq!(work.add_records(q, &foreign), foreign, "all new");
    assert!(work.add_records(q, &foreign).is_empty(), "a second time, none is new");
    assert_eq!(work.records(&q), foreign);
    assert!(work.has_record(&q, &foreign[0]) && !work.has_record(&[7; 32], &foreign[0]));

    let pool = Mempool::new();
    let id = work.make_job(&state, &pool, lock_of(&key(2)), T0 + 500).unwrap();
    let template = work.job(id).unwrap();
    assert_eq!(template.shares, foreign, "the job lists the records of miners that are not its own");
    let (job, base, model) = (template.job.clone(), template.base, template.model.clone());
    // The block built from that job connects: the chain accepts the headers of other miners as shares.
    let mut buffers = neb_pow::shares::ShareBuffers::new();
    let winner = (0u32..)
        .find(|&n| {
            neb_block::Target(job.target).admits(&buffers.ticket(
                &model,
                &*ds,
                &neb_block::Header { nonce: n, ..base }.id(),
            ))
        })
        .unwrap();
    let block = work.submit(id, &vec![0u8; job.extranonce_size as usize], job.time_start, winner).unwrap();
    state.connect_block(&block).expect("a block that lists the shares of another miner connects");

    // At most MAX_STORED_SHARES records for one parent, whatever they are.
    let many: Vec<ShareRecord> =
        (0..MAX_STORED_SHARES as u32 + 5).map(|n| ShareRecord { body_root: [1; 32], time: 0, nonce: n }).collect();
    assert_eq!(work.add_records([9; 32], &many).len(), MAX_STORED_SHARES);
    // At most 64 parents: the oldest go first.
    for p in 0..70u8 {
        work.add_records([100 + p; 32], &many[..1]);
    }
    assert!(!work.has_record(&[100; 32], &many[0]), "the oldest parent was dropped");
    assert!(work.has_record(&[169; 32], &many[0]), "the newest parent is kept");
}

fn demo_easy_state() -> ChainState {
    ChainState::from_genesis(demo::easy_params(T0, demo::PROJECT_LOCK), demo::easy_genesis(T0)).expect("genesis")
}
