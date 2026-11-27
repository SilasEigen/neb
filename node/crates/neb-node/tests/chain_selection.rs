//! Chain selection (spec/node.md section 6.5): branches are compared by
//! cumulative work, never by height, and only strictly greater work replaces
//! the tip. `verify_pow` is false so that blocks need no ticket search; every
//! other header rule, including the ASERT `bits`, still runs.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use neb_chain::state::{project_share, subsidy};
use neb_chain::{Block, Work, demo};
use neb_node::params::NetworkParams;
use neb_node::{BlockOutcome, Node, NodeConfig};

const T0: u32 = 1_700_000_000;
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-select-{tag}-{}-{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn test_network() -> NetworkParams {
    NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, 0x02],
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
        dataset_names: &[],
        genesis: demo::genesis(T0),
        chain: demo::demo_params(T0, demo::PROJECT_LOCK),
    }
}

fn open(dir: &Path) -> Node {
    Node::open(NodeConfig { network: test_network(), data_dir: dir.to_path_buf(), verify_pow: false }).expect("node")
}

fn coinbase(height: u32) -> neb_chain::Transaction {
    let reward = subsidy(height);
    demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward))
}

/// The time of a "hard" branch: one second between blocks, far faster than
/// the 60 s schedule, so ASERT lowers the target block by block.
fn fast_time(height: u32) -> u32 {
    T0 + height
}

/// The time of an "easy" branch: exactly on schedule, so every block keeps
/// the anchor target, which is also the pow limit, and has the same work.
fn steady_time(height: u32) -> u32 {
    T0 + 60 * height
}

/// A chain from the genesis whose block times follow `time_of` and whose bits
/// follow the chain's own difficulty rule, as a real miner would produce.
fn build_from_genesis(
    net: &NetworkParams,
    tip_height: u32,
    time_of: impl Fn(u32) -> u32,
    nonce_base: u32,
) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut parent = demo::genesis(T0).id();
    let mut parent_time = T0;
    for h in 1..=tip_height {
        let bits = net.chain.difficulty.next_bits(h - 1, parent_time).expect("difficulty");
        let time = time_of(h);
        let block = demo::block(parent, time, bits, nonce_base + h, vec![coinbase(h)]);
        parent = block.id();
        parent_time = time;
        blocks.push(block);
    }
    blocks
}

fn branch_work(blocks: &[Block]) -> Work {
    blocks.iter().fold(Work::ZERO, |w, b| w + Work::of_bits(b.header.bits).expect("bits"))
}

// Attack scenario: a long chain of cheap blocks must not displace a shorter
// chain of expensive ones, and an equal-work branch must not move the tip.
#[test]
fn a_longer_lighter_branch_does_not_replace_a_heavier_one() {
    let dir = temp_dir("lighter");
    let node = open(&dir);
    let net = test_network();

    // hard branch: 12 blocks one second apart
    let hard = build_from_genesis(&net, 12, fast_time, 1000);
    for block in &hard {
        match node.accept_block(block.clone()).expect("accept") {
            BlockOutcome::Connected { .. } => {}
            other => panic!("expected Connected, got {other:?}"),
        }
    }
    assert_eq!(node.height(), 12);
    let hard_tip = node.tip_id();
    let hard_work = branch_work(&hard);

    // easy branch: 13 steady blocks, taller by one and lighter in total
    let easy = build_from_genesis(&net, 13, steady_time, 2000);
    assert!(easy.len() > hard.len());
    assert!(branch_work(&easy) < hard_work, "the easy branch is taller and lighter");
    for block in &easy {
        match node.accept_block(block.clone()).expect("accept") {
            BlockOutcome::Side { .. } => {}
            other => panic!("expected Side, got {other:?}"),
        }
    }
    assert_eq!(node.height(), 12, "height must not decide selection");
    assert_eq!(node.tip_id(), hard_tip);

    // The first block that lifts the easy branch strictly above the hard
    // work triggers the reorganisation; later blocks extend the new tip.
    let mut cum = branch_work(&easy);
    let mut blocks = easy.clone();
    let mut parent = easy.last().unwrap().id();
    let mut parent_time = steady_time(13);
    let mut replaced = false;
    for height in 14..=20 {
        let bits = net.chain.difficulty.next_bits(height - 1, parent_time).expect("difficulty");
        let time = steady_time(height);
        let block = demo::block(parent, time, bits, 2000 + height, vec![coinbase(height)]);
        parent = block.id();
        parent_time = time;
        cum = cum + Work::of_bits(bits).expect("bits");
        let outcome = node.accept_block(block.clone()).expect("accept");
        blocks.push(block);
        if !replaced && cum > hard_work {
            match outcome {
                BlockOutcome::Reorged { .. } => {}
                other => panic!("expected Reorged at height {height}, got {other:?}"),
            }
            replaced = true;
        } else if replaced {
            match outcome {
                BlockOutcome::Connected { .. } => {}
                other => panic!("expected Connected at height {height}, got {other:?}"),
            }
        } else {
            match outcome {
                BlockOutcome::Side { .. } => {}
                other => panic!("expected Side at height {height}, got {other:?}"),
            }
        }
    }
    assert!(replaced, "the easy branch never became heavier");
    assert_eq!(node.height(), 20);
    assert_eq!(node.tip_id(), blocks.last().unwrap().id());

    // equal cumulative work keeps the current tip
    let tie = build_from_genesis(&net, 20, steady_time, 4000);
    assert_eq!(branch_work(&tie), branch_work(&blocks));
    for block in &tie {
        match node.accept_block(block.clone()).expect("accept") {
            BlockOutcome::Side { .. } => {}
            other => panic!("expected Side on the tie, got {other:?}"),
        }
    }
    assert_eq!(node.tip_id(), blocks.last().unwrap().id());
    drop(node);
    fs::remove_dir_all(&dir).unwrap();
}

// A shorter branch with more cumulative work wins, and the choice is
// rebuilt identically from the stored blocks after a restart.
#[test]
fn a_shorter_heavier_branch_replaces_a_taller_lighter_one() {
    let dir = temp_dir("shorter");
    let node = open(&dir);
    let net = test_network();

    // 16 steady blocks form the current chain
    let light = build_from_genesis(&net, 16, steady_time, 5000);
    for block in &light {
        node.accept_block(block.clone()).expect("connect");
    }
    assert_eq!(node.height(), 16);

    // 12-block hard branch: shorter, more work
    let hard = build_from_genesis(&net, 12, fast_time, 6000);
    assert!(hard.len() < light.len());
    let light_work = branch_work(&light);
    assert!(branch_work(&hard) > light_work);
    let mut cum = Work::ZERO;
    for (i, block) in hard.iter().enumerate() {
        cum = cum + Work::of_bits(block.header.bits).expect("bits");
        let last = i == hard.len() - 1;
        let outcome = node.accept_block(block.clone()).expect("accept");
        if cum > light_work {
            assert!(last, "the hard branch must pass the light one only at its tip");
            assert!(matches!(outcome, BlockOutcome::Reorged { .. }), "got {outcome:?}");
        } else {
            assert!(!last, "the full hard branch must be heavier");
            assert!(matches!(outcome, BlockOutcome::Side { .. }), "got {outcome:?}");
        }
    }
    assert_eq!(node.height(), 12, "the heavier chain wins even when shorter");
    assert_eq!(node.tip_id(), hard.last().unwrap().id());

    // restart: heights, work and the best chain are rebuilt from the store
    let tip = node.tip_id();
    drop(node);
    let node = open(&dir);
    assert_eq!(node.height(), 12);
    assert_eq!(node.tip_id(), tip);
    drop(node);
    fs::remove_dir_all(&dir).unwrap();
}

/// Blocks `from..=to` on the block `parent` of time `parent_time`, on schedule.
fn build_on(
    net: &NetworkParams,
    mut parent: [u8; 32],
    mut parent_time: u32,
    heights: std::ops::RangeInclusive<u32>,
    nonce_base: u32,
    txs_of: impl Fn(u32) -> neb_chain::Transaction,
) -> Vec<Block> {
    let mut blocks = Vec::new();
    for h in heights {
        let bits = net.chain.difficulty.next_bits(h - 1, parent_time).expect("difficulty");
        let block = demo::block(parent, steady_time(h), bits, nonce_base + h, vec![txs_of(h)]);
        parent = block.id();
        parent_time = steady_time(h);
        blocks.push(block);
    }
    blocks
}

/// Waits for the state file of block `height`, which the node writes on a thread of its own.
fn wait_state_file(dir: &Path, height: u32) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if let Ok(bytes) = fs::read(dir.join("state.dat"))
            && neb_chain::state::ChainState::snapshot_height(&bytes) == Some(height)
        {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "no state file of block {height}");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

// A branch with more work that leaves the chain more than 60 blocks back stays a side branch while the node runs,
// and the node opens on the same chain when the index is rebuilt.
#[test]
fn a_rebuilt_index_keeps_the_chain_of_the_state_file_against_a_deep_branch() {
    let dir = temp_dir("deep");
    let node = open(&dir);
    let net = test_network();

    let best = build_from_genesis(&net, 70, steady_time, 7000);
    for block in &best {
        node.accept_block(block.clone()).expect("connect");
    }
    let tip = best.last().unwrap().id();
    // 66 blocks on block 5: one block more than the chain, and 65 blocks to remove
    let branch = build_on(&net, best[4].id(), steady_time(5), 6..=71, 8000, coinbase);
    for block in &branch {
        let outcome = node.accept_block(block.clone()).expect("accept");
        assert!(matches!(outcome, BlockOutcome::Side { .. }), "got {outcome:?}");
    }
    assert_eq!((node.height(), node.tip_id()), (70, tip));
    wait_state_file(&dir, 70);
    drop(node);

    fs::remove_file(dir.join("index.dat")).unwrap();
    let node = open(&dir);
    assert_eq!((node.height(), node.tip_id()), (70, tip), "the rebuilt index follows the state file");
    drop(node);
    // the index that was written is read as it is
    let node = open(&dir);
    assert_eq!((node.height(), node.tip_id()), (70, tip));
    drop(node);

    // Without the state file nothing tells which chain the node was on: the valid chain with the most work.
    fs::remove_file(dir.join("index.dat")).unwrap();
    fs::remove_file(dir.join("state.dat")).unwrap();
    let node = open(&dir);
    assert_eq!((node.height(), node.tip_id()), (71, branch.last().unwrap().id()));
    drop(node);
    fs::remove_dir_all(&dir).unwrap();
}

// A branch with more work and a block whose coinbase pays too much is refused while the node runs; its blocks stay
// in the log, and the node opens on its chain when the index is rebuilt, with the state file or without it.
#[test]
fn a_rebuilt_index_leaves_aside_a_branch_that_does_not_connect() {
    let dir = temp_dir("invalid");
    let node = open(&dir);
    let net = test_network();

    let best = build_from_genesis(&net, 20, steady_time, 7000);
    for block in &best {
        node.accept_block(block.clone()).expect("connect");
    }
    let tip = best.last().unwrap().id();
    let branch = build_on(&net, best[14].id(), steady_time(15), 16..=22, 8000, |h| {
        if h == 17 { demo::coinbase(h, demo::MINER_LOCK, demo::PROJECT_LOCK, 2 * subsidy(h)) } else { coinbase(h) }
    });
    for block in &branch[..5] {
        let outcome = node.accept_block(block.clone()).expect("accept");
        assert!(matches!(outcome, BlockOutcome::Side { .. }), "got {outcome:?}");
    }
    for block in &branch[5..] {
        let refused = node.accept_block(block.clone());
        assert!(matches!(refused, Err(neb_node::NodeError::Reorg(_))), "got {refused:?}");
    }
    assert_eq!((node.height(), node.tip_id()), (20, tip));
    wait_state_file(&dir, 20);
    drop(node);

    fs::remove_file(dir.join("index.dat")).unwrap();
    let node = open(&dir);
    assert_eq!((node.height(), node.tip_id()), (20, tip));
    drop(node);

    fs::remove_file(dir.join("index.dat")).unwrap();
    fs::remove_file(dir.join("state.dat")).unwrap();
    let node = open(&dir);
    assert_eq!((node.height(), node.tip_id()), (20, tip));
    drop(node);
    fs::remove_dir_all(&dir).unwrap();
}

// The state file is older than a reorganisation the node made: the rebuilt index takes the chain the node moved to,
// which has more work and leaves the chain of the file within the limit.
#[test]
fn a_rebuilt_index_follows_a_reorganisation_made_after_the_state_file() {
    let dir = temp_dir("stale");
    let node = open(&dir);
    let net = test_network();

    let first = build_from_genesis(&net, 20, steady_time, 7000);
    for block in &first {
        node.accept_block(block.clone()).expect("connect");
    }
    wait_state_file(&dir, 20);
    let at_20 = fs::read(dir.join("state.dat")).unwrap();
    let branch = build_on(&net, first[14].id(), steady_time(15), 16..=22, 8000, coinbase);
    for block in &branch {
        node.accept_block(block.clone()).expect("accept");
    }
    let tip = branch.last().unwrap().id();
    assert_eq!((node.height(), node.tip_id()), (22, tip));
    wait_state_file(&dir, 21);
    drop(node);

    fs::write(dir.join("state.dat"), &at_20).unwrap();
    fs::remove_file(dir.join("index.dat")).unwrap();
    let node = open(&dir);
    assert_eq!((node.height(), node.tip_id()), (22, tip));
    drop(node);
    fs::remove_dir_all(&dir).unwrap();
}

// A stored block whose parent is not stored makes `Node::open` fail with `NodeError::Store`.
#[test]
fn open_refuses_a_store_with_a_block_without_parent() {
    let dir = temp_dir("orphan");
    drop(open(&dir));
    let orphan = demo::block([7u8; 32], T0 + 60, 0xC280_0000, 1, vec![coinbase(1)]);
    neb_store::BlockStore::open(&dir).unwrap().put_block(&orphan).unwrap();
    let opened = Node::open(NodeConfig { network: test_network(), data_dir: dir.to_path_buf(), verify_pow: false });
    assert!(matches!(opened, Err(neb_node::NodeError::Store(_))));
    fs::remove_dir_all(&dir).unwrap();
}
