//! Two-node tests of the peer-to-peer layer over loopback: initial block
//! download, reorganisation to a heavier branch, restart, transaction relay and
//! peer lookup by address. `verify_pow` is false so that blocks need no ticket
//! search; every other header and body rule runs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ed25519_dalek::{Signer, SigningKey};
use neb_chain::Block;
use neb_chain::demo;
use neb_chain::state::{OutPoint, project_share, subsidy};
use neb_chain::tx::{LOCK_SINGLE, Transaction, TxIn, TxOut};
use neb_node::params::NetworkParams;
use neb_node::{Node, NodeConfig};

const T0: u32 = 1_700_000_000;
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-node-{tag}-{}-{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn test_network() -> NetworkParams {
    test_network_to(demo::MINER_LOCK)
}

fn test_network_to(miner_lock: [u8; 32]) -> NetworkParams {
    NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, 0x01],
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
        genesis: demo::genesis_to(T0, miner_lock),
        chain: demo::demo_params(T0, demo::PROJECT_LOCK),
    }
}

fn lock_of(sk: &SigningKey) -> [u8; 32] {
    *blake3::hash(&sk.verifying_key().to_bytes()).as_bytes()
}

fn open(dir: &Path) -> Node {
    Node::open(NodeConfig { network: test_network(), data_dir: dir.to_path_buf(), verify_pow: false }).expect("node")
}

fn coinbase(height: u32) -> neb_chain::Transaction {
    let reward = subsidy(height);
    demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward))
}

/// Blocks 1..=count on the genesis, one per 60 s at bits 0xC280_0000 (the
/// demo pow limit, so the ASERT rule keeps it on schedule).
fn prefix_blocks(count: u32) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut parent = demo::genesis(T0).id();
    for height in 1..=count {
        let block = demo::block(parent, T0 + 60 * height, 0xC280_0000, 1000 + height, vec![coinbase(height)]);
        parent = block.id();
        blocks.push(block);
    }
    blocks
}

/// Connects `count` blocks on the tip of `node`, starting at `from_height`.
/// `salt` makes the nonces, and so the block ids, differ between branches.
fn extend(node: &Node, from_height: u32, count: u32, time_base: u32, salt: u32) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut parent = node.tip_id();
    for i in 0..count {
        let height = from_height + i;
        let block = demo::block(parent, time_base + 60 * height, 0xC280_0000, salt + height, vec![coinbase(height)]);
        node.accept_block(block.clone()).expect("accept");
        parent = block.id();
        blocks.push(block);
    }
    blocks
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

// A fresh node downloads the whole chain from its peer through
// getblocks/inv/getdata/block, in batches.
#[test]
fn node_b_converges_on_node_a() {
    let dir_a = temp_dir("sync-a");
    let dir_b = temp_dir("sync-b");
    let a = open(&dir_a);
    let prefix = prefix_blocks(4);
    for block in &prefix {
        a.accept_block(block.clone()).expect("connect");
    }
    assert_eq!(a.height(), 4);
    let b = open(&dir_b);
    assert_eq!(b.height(), 0);
    let handle = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(handle.addr).expect("connect");
    let synced = wait_for(|| b.height() == 4 && b.tip_id() == a.tip_id(), 15);
    if !synced {
        eprintln!("A height {} tip {:02x?}", a.height(), &a.tip_id()[..4]);
        eprintln!("B height {} tip {:02x?}", b.height(), &b.tip_id()[..4]);
        eprintln!("B peers {:?}", b.peer_info());
        eprintln!("A peers {:?}", a.peer_info());
        eprintln!("B headers? block1 present {}", b.block(&prefix[0].id()).is_some());
    }
    assert!(synced, "B did not sync");
    assert_eq!(b.block_at(4).unwrap().id(), a.block_at(4).unwrap().id());
    drop(handle);
    drop(b);
    drop(a);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// A node on a shorter branch switches to the heavier branch of its peer and
// keeps the choice after a restart.
#[test]
fn a_longer_branch_replaces_the_tip() {
    let dir_a = temp_dir("reorg-a");
    let dir_b = temp_dir("reorg-b");
    let a = open(&dir_a);
    let b = open(&dir_b);
    let prefix = prefix_blocks(2);
    for block in &prefix {
        a.accept_block(block.clone()).expect("a prefix");
        b.accept_block(block.clone()).expect("b prefix");
    }
    // From the same fork, A builds two blocks and B builds three.
    extend(&a, 3, 2, T0, 10_000);
    extend(&b, 3, 3, T0, 20_000);
    assert_eq!(a.height(), 4);
    assert_eq!(b.height(), 5);
    assert_ne!(a.tip_id(), b.tip_id());
    let old_tip = a.tip_id();
    let handle = b.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    a.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| a.height() == 5 && a.tip_id() == b.tip_id(), 15), "A did not reorg");
    assert_ne!(a.tip_id(), old_tip);
    assert_eq!(a.block_at(5).unwrap().id(), b.block_at(5).unwrap().id());
    assert_eq!(a.block_at(2).unwrap().id(), prefix[1].id());
    let tip = b.tip_id();
    drop(handle);
    drop(a);
    drop(b);
    let a2 = open(&dir_a);
    assert_eq!(a2.height(), 5);
    assert_eq!(a2.tip_id(), tip);
    drop(a2);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// The block log replays to the same height.
#[test]
fn the_chain_survives_a_restart() {
    let dir = temp_dir("restart");
    {
        let node = open(&dir);
        let prefix = prefix_blocks(3);
        for block in prefix {
            node.accept_block(block).expect("connect");
        }
        assert_eq!(node.height(), 3);
    }
    let node = open(&dir);
    assert_eq!(node.height(), 3);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

// A signed spend of the genesis output, accepted by A after 101 blocks (the
// coinbase has matured), is announced to B and fetched by `getdata`.
#[test]
fn a_transaction_relays_to_the_second_node() {
    let sk = SigningKey::from_bytes(&[9u8; 32]);
    let dir_a = temp_dir("tx-a");
    let dir_b = temp_dir("tx-b");
    let a =
        Node::open(NodeConfig { network: test_network_to(lock_of(&sk)), data_dir: dir_a.clone(), verify_pow: false })
            .expect("a");
    let prefix = {
        let mut blocks = Vec::new();
        let mut parent = demo::genesis_to(T0, lock_of(&sk)).id();
        for height in 1..=101 {
            let block = demo::block(parent, T0 + 60 * height, 0xC280_0000, 3000 + height, vec![coinbase(height)]);
            parent = block.id();
            blocks.push(block);
        }
        blocks
    };
    for block in &prefix {
        a.accept_block(block.clone()).expect("connect");
    }
    let b =
        Node::open(NodeConfig { network: test_network_to(lock_of(&sk)), data_dir: dir_b.clone(), verify_pow: false })
            .expect("b");
    let handle = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| b.height() == 101, 20), "B did not sync");

    let genesis_cb = demo::genesis_to(T0, lock_of(&sk)).txs[0].txid();
    let op = OutPoint { txid: genesis_cb, index: 0 };
    let spent = a.utxo(&op).expect("utxo").as_txout();
    let mut tx = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: op.txid, prev_index: op.index, witness: Vec::new() }],
        outputs: vec![TxOut {
            amount: 18 * 100_000_000,
            lock_type: LOCK_SINGLE,
            lock_data: lock_of(&SigningKey::from_bytes(&[7u8; 32])),
        }],
    };
    let msg = tx.sighash(&a.chain_id(), 0, &spent);
    let sig = sk.sign(&msg);
    let mut witness = Vec::new();
    witness.extend_from_slice(&sk.verifying_key().to_bytes());
    witness.extend_from_slice(&sig.to_bytes());
    tx.inputs[0].witness = witness;
    let txid = a.submit_tx(tx).expect("submit");
    assert!(wait_for(|| b.has_tx(&txid), 10), "transaction did not relay");
    drop(handle);
    drop(a);
    drop(b);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// The dialled address, not only the count, identifies a connected peer.
#[test]
fn a_dialled_peer_is_visible_by_address() {
    // The seed connector of the daemon polls `is_peer_connected` with the dialled address.
    let dir_a = temp_dir("addr-a");
    let dir_b = temp_dir("addr-b");
    let a = open(&dir_a);
    let b = open(&dir_b);
    let handle = b.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    assert!(!a.is_peer_connected(&handle.addr));
    a.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| a.is_peer_connected(&handle.addr), 10), "the dialled peer is not visible");
    drop(handle);
    drop(a);
    drop(b);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}
