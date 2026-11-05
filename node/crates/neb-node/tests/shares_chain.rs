//! Nodes on the easy network with `verify_pow` enabled: blocks mined with real
//! share lists synchronise, survive a restart and reorganisations, and every
//! node ends on the model of a replay from the genesis block (spec/shares.md
//! section 11.4). The model is consensus state, so equal `weights_hash` after
//! every path is the property under test.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use neb_chain::demo;
use neb_chain::state::{ChainState, project_share, subsidy};
use neb_chain::{Block, ShareRecord};
use neb_node::params::NetworkParams;
use neb_node::{BlockOutcome, Node, NodeConfig};

const T0: u32 = 1_700_000_000;
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-shares-{tag}-{}-{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn network() -> NetworkParams {
    NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, 0x05],
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
        genesis: demo::easy_genesis(T0),
        chain: demo::easy_params(T0, demo::PROJECT_LOCK),
    }
}

fn open(dir: &Path) -> Node {
    Node::open(NodeConfig { network: network(), data_dir: dir.to_path_buf(), verify_pow: true }).expect("node")
}

fn wait_for<F: Fn() -> bool>(f: F, seconds: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    f()
}

/// A chain state that mines blocks with real shares. `round` holds the share
/// records of the block being built on the tip.
struct Probe {
    state: ChainState,
    round: Vec<ShareRecord>,
}

impl Probe {
    fn new() -> Probe {
        let state =
            ChainState::from_genesis(demo::easy_params(T0, demo::PROJECT_LOCK), demo::easy_genesis(T0)).unwrap();
        Probe { state, round: Vec::new() }
    }

    /// The probe after `blocks`, which another miner found.
    fn following(blocks: &[Block]) -> Probe {
        let mut probe = Probe::new();
        for b in blocks {
            probe.state.connect_block(b).expect("connect");
        }
        if let Some(tip) = blocks.last() {
            // Share records of the round of the tip: the headers of the tip up
            // to the winning nonce that meet the share target.
            let h = tip.header;
            probe.round = probe.state.valid_shares((0..=h.nonce).map(|nonce| ShareRecord {
                body_root: h.body_root,
                time: h.time,
                nonce,
            }));
        }
        probe
    }

    fn mine(&mut self, lock: [u8; 32]) -> Block {
        let height = self.state.height() + 1;
        let reward = subsidy(height);
        let coinbase = demo::coinbase(height, lock, demo::PROJECT_LOCK, reward - project_share(reward));
        let (block, round) = demo::mine_next(&self.state, vec![coinbase], std::mem::take(&mut self.round));
        self.state.connect_block(&block).expect("a mined block connects");
        self.round = round;
        block
    }

    fn mine_many(&mut self, lock: [u8; 32], count: u32) -> Vec<Block> {
        (0..count).map(|_| self.mine(lock)).collect()
    }
}

// Initial block download of 22 blocks whose share lists train the model: the
// downloading node reaches the model of the miner.
#[test]
fn two_nodes_synchronise_twenty_two_blocks_with_shares() {
    let (dir_a, dir_b) = (temp_dir("sync-a"), temp_dir("sync-b"));
    let mut probe = Probe::new();
    let blocks = probe.mine_many(demo::MINER_LOCK, 22);
    let listed: usize = blocks.iter().map(|b| b.shares.len()).sum();
    assert!(listed > 200, "{listed} shares listed");

    let a = open(&dir_a);
    for b in &blocks {
        assert!(matches!(a.accept_block(b.clone()), Ok(BlockOutcome::Connected { .. })));
    }
    assert_eq!(a.weights_hash(), probe.state.tip_model().weights_hash());

    let b = open(&dir_b);
    let handle = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| b.height() == 22 && b.tip_id() == a.tip_id(), 60), "B did not sync, height {}", b.height());
    assert_eq!(b.weights_hash(), a.weights_hash());
    assert_eq!(b.weights_hash(), probe.state.tip_model().weights_hash());
    assert_ne!(b.weights_hash(), Probe::new().state.tip_model().weights_hash(), "the model moved");
    drop(handle);
    drop((a, b));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// The store holds blocks only; the model is rebuilt by replay and equals the
// miner's, and the node keeps accepting blocks on it.
#[test]
fn a_restarted_node_rebuilds_the_model_by_replay() {
    let dir = temp_dir("restart");
    let mut probe = Probe::new();
    let blocks = probe.mine_many(demo::MINER_LOCK, 14);
    let before = {
        let node = open(&dir);
        for b in &blocks {
            node.accept_block(b.clone()).expect("connect");
        }
        node.weights_hash()
    };
    let node = open(&dir);
    assert_eq!(node.height(), 14);
    assert_eq!(node.weights_hash(), before);
    assert_eq!(node.weights_hash(), probe.state.tip_model().weights_hash());
    let next = probe.mine(demo::MINER_LOCK);
    assert!(matches!(node.accept_block(next), Ok(BlockOutcome::Connected { height: 15 })));
    assert_eq!(node.weights_hash(), probe.state.tip_model().weights_hash());
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

// A reorganisation to branch Y and back to branch X: the models of the
// disconnected blocks are kept as side models, so X can win again without a
// replay, and the model after each switch equals a replay of that chain.
#[test]
fn nodes_follow_reorganisations_back_and_forth_on_the_model_of_a_replay() {
    let (dir_a, dir_b) = (temp_dir("reorg-a"), temp_dir("reorg-b"));
    let mut main = Probe::new();
    let prefix = main.mine_many(demo::MINER_LOCK, 4);
    let mut branch_x = Probe::following(&prefix);
    let mut branch_y = Probe::following(&prefix);
    let x = branch_x.mine_many([0x31; 32], 6); // heights 5..10
    let y = branch_y.mine_many([0x32; 32], 4); // heights 5..8
    assert_ne!(x[0].id(), y[0].id());

    // A holds X up to height 7, B holds Y up to height 8
    let a = open(&dir_a);
    for b in prefix.iter().chain(&x[..3]) {
        a.accept_block(b.clone()).expect("a connects");
    }
    let b = open(&dir_b);
    for blk in prefix.iter().chain(&y) {
        b.accept_block(blk.clone()).expect("b connects");
    }
    assert_eq!((a.height(), b.height()), (7, 8));
    let handle = b.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    a.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| a.height() == 8 && a.tip_id() == b.tip_id(), 60), "A did not switch to Y");
    let replay_y = Probe::following(&[prefix.clone(), y.clone()].concat());
    assert_eq!(a.weights_hash(), replay_y.state.tip_model().weights_hash());
    drop(handle);

    // X grows past Y: block 8 of X is a side block (equal work), block 9
    // reorganises back, block 10 extends the tip.
    let outcomes: Vec<BlockOutcome> =
        x[3..].iter().map(|blk| a.accept_block(blk.clone()).expect("a takes the extension of X")).collect();
    assert!(matches!(outcomes[0], BlockOutcome::Side { .. }), "{outcomes:?}");
    assert!(matches!(outcomes[1], BlockOutcome::Reorged { .. }), "{outcomes:?}");
    assert!(matches!(outcomes[2], BlockOutcome::Connected { .. }), "{outcomes:?}");
    assert_eq!(a.height(), 10);
    assert_eq!(a.tip_id(), x[5].id());
    let replay_x = Probe::following(&[prefix, x].concat());
    assert_eq!(a.weights_hash(), replay_x.state.tip_model().weights_hash());
    drop((a, b));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}
