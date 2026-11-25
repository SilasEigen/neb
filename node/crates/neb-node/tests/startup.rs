//! The state file of the data directory (`startup.rs`): a restart reads it and replays the blocks after it, and
//! replays the whole chain when the file is missing, damaged or of another tip. The chain is the same every time.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use neb_chain::demo;
use neb_chain::state::{project_share, subsidy};
use neb_node::params::NetworkParams;
use neb_node::startup::Start;
use neb_node::{Node, NodeConfig};

const T0: u32 = 1_700_000_000;

fn temp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("neb-node-startup-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn open(dir: &Path) -> Node {
    let network = NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, 0x01],
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
    };
    Node::open(NodeConfig { network, data_dir: dir.to_path_buf(), verify_pow: false }).expect("node")
}

/// Connects blocks `from..=to` on the tip of `node`; `salt` makes the ids of two branches differ.
fn extend(node: &Node, from: u32, to: u32, salt: u32) {
    for height in from..=to {
        let reward = subsidy(height);
        let coinbase = demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward));
        let block = demo::block(node.tip_id(), T0 + 60 * height, 0xC280_0000, salt + height, vec![coinbase]);
        node.accept_block(block).expect("accept");
    }
}

/// The state file once it names `height` or a later block: the node writes it on a thread of its own, with the
/// state of the moment that thread runs.
fn state_file_at(dir: &Path, height: u32) -> Vec<u8> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(bytes) = std::fs::read(dir.join("state.dat"))
            && neb_chain::state::ChainState::snapshot_height(&bytes).is_some_and(|h| h >= height)
        {
            return bytes;
        }
        assert!(Instant::now() < deadline, "no state file of block {height}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_restart_reads_the_state_file_and_replays_the_blocks_after_it() {
    let dir = temp_dir("restart");
    let node = open(&dir);
    assert_eq!(node.start(), Start { loaded: None, replayed: 0 });
    extend(&node, 1, 20, 1000);
    let at_20 = state_file_at(&dir, 20);
    // five blocks that are not due for a write
    extend(&node, 21, 25, 1000);
    let tip = node.tip_id();
    drop(node);

    // the file of block 20 and five blocks after it
    let node = open(&dir);
    assert_eq!(node.start(), Start { loaded: Some(20), replayed: 5 });
    assert_eq!((node.height(), node.tip_id()), (25, tip));
    // a block connects on the state that was read, and the node wrote the state of block 25 when it started
    extend(&node, 26, 26, 1000);
    let tip = node.tip_id();
    drop(node);
    let node = open(&dir);
    assert_eq!(node.start(), Start { loaded: Some(25), replayed: 1 });
    assert_eq!((node.height(), node.tip_id()), (26, tip));
    drop(node);

    // a damaged file: the whole chain is replayed, and a good file is written again
    let mut damaged = std::fs::read(dir.join("state.dat")).unwrap();
    let middle = damaged.len() / 2;
    damaged[middle] ^= 1;
    std::fs::write(dir.join("state.dat"), &damaged).unwrap();
    let node = open(&dir);
    assert_eq!(node.start(), Start { loaded: None, replayed: 26 });
    assert_eq!((node.height(), node.tip_id()), (26, tip));
    drop(node);
    assert_eq!(open(&dir).start(), Start { loaded: Some(26), replayed: 0 });

    // no file: the whole chain again
    std::fs::remove_file(dir.join("state.dat")).unwrap();
    let node = open(&dir);
    assert_eq!(node.start(), Start { loaded: None, replayed: 26 });
    assert_eq!(node.tip_id(), tip);
    drop(node);

    // an older file of the same chain is still good: six blocks after it
    std::fs::write(dir.join("state.dat"), &at_20).unwrap();
    let node = open(&dir);
    assert_eq!(node.start(), Start { loaded: Some(20), replayed: 6 });
    assert_eq!(node.tip_id(), tip);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

// The file of a tip that a reorganisation removed is not used: the node replays its chain and lands on the best one.
#[test]
fn the_state_file_of_a_removed_tip_is_left_aside() {
    let dir = temp_dir("reorg");
    let node = open(&dir);
    extend(&node, 1, 10, 1000);
    let old = state_file_at(&dir, 10);
    drop(node);

    // another node builds a heavier branch from block 8, which this one takes
    let other_dir = temp_dir("reorg-other");
    let other = open(&other_dir);
    extend(&other, 1, 8, 1000);
    extend(&other, 9, 12, 2000);
    let node = open(&dir);
    for height in 9..=12 {
        node.accept_block(other.block_at(height).expect("block")).expect("branch block");
    }
    assert_eq!((node.height(), node.tip_id()), (12, other.tip_id()));
    let tip = node.tip_id();
    // the reorganisation happened at block 11 and asked for a write
    state_file_at(&dir, 11);
    drop(node);

    std::fs::write(dir.join("state.dat"), &old).unwrap();
    let node = open(&dir);
    assert_eq!(node.start(), Start { loaded: None, replayed: 12 });
    assert_eq!((node.height(), node.tip_id()), (12, tip));
    drop(node);
    drop(other);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&other_dir);
}
