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
        dataset_names: &[],
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

// A relayed block whose parent the node lacks: the node asks its sender for the missing blocks. After a race, A and B
// are connected on the same tip; A then takes the other branch, whose first block it did not announce (a side block
// is not), and relays only the new tip.
#[test]
fn a_relayed_block_without_its_parent_brings_the_missing_blocks() {
    let dir_a = temp_dir("orphan-a");
    let dir_b = temp_dir("orphan-b");
    let a = open(&dir_a);
    let b = open(&dir_b);
    let mut blocks = prefix_blocks(2);
    blocks.push(demo::block(blocks[1].id(), T0 + 180, 0xC280_0000, 30_003, vec![coinbase(3)]));
    for block in &blocks {
        a.accept_block(block.clone()).expect("a");
        b.accept_block(block.clone()).expect("b");
    }
    let handle = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| a.peer_info().iter().any(|p| p.handshake) && b.peer_info().iter().any(|p| p.handshake), 15));
    // the handshake asks for blocks once: let that exchange end, so that only the relay below can bring the branch
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(b.tip_id(), blocks[2].id());

    let other3 = demo::block(blocks[1].id(), T0 + 181, 0xC280_0000, 40_003, vec![coinbase(3)]);
    let other4 = demo::block(other3.id(), T0 + 240, 0xC280_0000, 40_004, vec![coinbase(4)]);
    a.accept_block(other3).expect("side block");
    assert_eq!(a.tip_id(), blocks[2].id(), "equal work keeps the tip");
    a.accept_block(other4.clone()).expect("heavier branch");
    assert_eq!(a.tip_id(), other4.id());
    assert!(wait_for(|| b.tip_id() == other4.id(), 15), "B did not fetch the parent of the relayed block");
    assert_eq!(b.height(), 4);
    drop(handle);
    drop(a);
    drop(b);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// A peer that asks for headers in a loop and reads no answer does not delay the relay of a block to another peer.
#[test]
fn a_peer_that_reads_nothing_does_not_delay_the_relay_of_a_block() {
    use neb_node::p2p::encode_message;
    use std::io::Write;
    let dir_a = temp_dir("stall-a");
    let dir_b = temp_dir("stall-b");
    let a = open(&dir_a);
    let b = open(&dir_b);
    let blocks = prefix_blocks(40);
    for block in &blocks {
        a.accept_block(block.clone()).expect("a");
        b.accept_block(block.clone()).expect("b");
    }
    let handle = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");

    // the first session of A: a version, a verack, then getheaders from the genesis block without end
    let magic = test_network().magic;
    let genesis = demo::genesis(T0).id();
    let mut version = Vec::new();
    version.extend_from_slice(&1u32.to_le_bytes());
    version.extend_from_slice(&0u32.to_le_bytes());
    version.extend_from_slice(&7u64.to_le_bytes());
    version.extend_from_slice(&genesis);
    version.extend_from_slice(&0u16.to_le_bytes());
    version.push(0);
    let mut locator = vec![1u8];
    locator.extend_from_slice(&genesis);
    let mut silent = std::net::TcpStream::connect(handle.addr).expect("dial");
    silent.write_all(&encode_message(magic, "version", &version).unwrap()).unwrap();
    silent.write_all(&encode_message(magic, "verack", &[]).unwrap()).unwrap();
    let ask = encode_message(magic, "getheaders", &locator).unwrap();
    let flood = std::thread::spawn(move || while silent.write_all(&ask).is_ok() {});
    assert!(wait_for(|| a.peer_info().iter().any(|p| p.handshake), 15));

    b.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| a.peer_info().iter().filter(|p| p.handshake).count() == 2, 15));
    // long enough for the answers to fill the socket of the silent peer and for its session to be held
    std::thread::sleep(Duration::from_secs(2));

    let next = demo::block(blocks[39].id(), T0 + 60 * 41, 0xC280_0000, 1041, vec![coinbase(41)]);
    let sent = Instant::now();
    a.accept_block(next.clone()).expect("block 41");
    assert!(sent.elapsed() < Duration::from_millis(500), "accepting the block waited {:?}", sent.elapsed());
    assert!(wait_for(|| b.tip_id() == next.id(), 2), "the block reached B after more than 2 s");
    drop(handle);
    drop(a);
    drop(b);
    let _ = flood.join();
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// A relay with 200 peers that read and 5 that ask without end and read nothing: 30 blocks in a row reach every
// reading peer, and no block makes the node wait.
#[test]
fn a_relay_with_many_peers_delivers_every_block() {
    use neb_node::p2p::encode_message;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    const READERS: usize = 200;
    const BLOCKS: u32 = 30;
    let dir = temp_dir("load");
    let mut network = test_network();
    network.max_peers = 300;
    let a = Node::open(NodeConfig { network, data_dir: dir.clone(), verify_pow: false }).expect("node");
    let blocks = prefix_blocks(40);
    for block in &blocks {
        a.accept_block(block.clone()).expect("a");
    }
    let handle = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    let magic = test_network().magic;
    let genesis = demo::genesis(T0).id();
    let hello = |nonce: u64| {
        let mut version = Vec::new();
        version.extend_from_slice(&1u32.to_le_bytes());
        version.extend_from_slice(&40u32.to_le_bytes());
        version.extend_from_slice(&nonce.to_le_bytes());
        version.extend_from_slice(&genesis);
        version.extend_from_slice(&0u16.to_le_bytes());
        version.push(0);
        let mut stream = TcpStream::connect(handle.addr).expect("dial");
        stream.write_all(&encode_message(magic, "version", &version).unwrap()).unwrap();
        stream.write_all(&encode_message(magic, "verack", &[]).unwrap()).unwrap();
        stream
    };

    let mut locator = vec![1u8];
    locator.extend_from_slice(&genesis);
    let ask = encode_message(magic, "getheaders", &locator).unwrap();
    // a second handle on every socket, to close them at the end
    let mut sockets = Vec::new();
    let floods: Vec<_> = (0..5u64)
        .map(|i| {
            let (mut silent, ask) = (hello(1000 + i), ask.clone());
            sockets.push(silent.try_clone().unwrap());
            std::thread::spawn(move || while silent.write_all(&ask).is_ok() {})
        })
        .collect();

    // each reading peer counts the block announcements it receives
    let announced: Vec<Arc<AtomicUsize>> = (0..READERS).map(|_| Arc::new(AtomicUsize::new(0))).collect();
    let readers: Vec<_> = (0..READERS)
        .map(|i| {
            let (mut stream, count) = (hello(i as u64 + 1), announced[i].clone());
            sockets.push(stream.try_clone().unwrap());
            std::thread::spawn(move || {
                let mut header = [0u8; 24];
                while stream.read_exact(&mut header).is_ok() {
                    let mut payload = vec![0u8; u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize];
                    if stream.read_exact(&mut payload).is_err() {
                        break;
                    }
                    if &header[4..8] == b"inv " && payload.len() == 34 {
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();
    assert!(wait_for(|| a.peer_info().iter().filter(|p| p.handshake).count() == READERS + 5, 30), "handshakes");
    std::thread::sleep(Duration::from_secs(2));

    let mut parent = blocks[39].id();
    let mut slowest = Duration::ZERO;
    for height in 41..41 + BLOCKS {
        let block = demo::block(parent, T0 + 60 * height, 0xC280_0000, 1000 + height, vec![coinbase(height)]);
        parent = block.id();
        let t = Instant::now();
        a.accept_block(block).expect("block");
        slowest = slowest.max(t.elapsed());
    }
    assert!(slowest < Duration::from_millis(500), "one block made the node wait {slowest:?}");
    let all = || announced.iter().all(|c| c.load(Ordering::Relaxed) == BLOCKS as usize);
    let got = wait_for(all, 10);
    let short = announced.iter().filter(|c| c.load(Ordering::Relaxed) != BLOCKS as usize).count();
    assert!(got, "{short} of {READERS} reading peers did not get the {BLOCKS} announcements");
    for socket in &sockets {
        let _ = socket.shutdown(std::net::Shutdown::Both);
    }
    drop(handle);
    drop(a);
    for t in floods.into_iter().chain(readers) {
        let _ = t.join();
    }
    let _ = std::fs::remove_dir_all(&dir);
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

/// A node whose genesis output belongs to `sk`, with 101 blocks on it so that the output is mature, the id of
/// block 101, and a signed spend of that output.
fn node_with_a_spend(tag: &str, sk: &SigningKey) -> (Node, PathBuf, [u8; 32], Transaction) {
    let dir = temp_dir(tag);
    let node =
        Node::open(NodeConfig { network: test_network_to(lock_of(sk)), data_dir: dir.clone(), verify_pow: false })
            .expect("node");
    let mut parent = demo::genesis_to(T0, lock_of(sk)).id();
    for height in 1..=101 {
        let block = demo::block(parent, T0 + 60 * height, 0xC280_0000, 3000 + height, vec![coinbase(height)]);
        parent = block.id();
        node.accept_block(block).expect("connect");
    }
    let op = OutPoint { txid: demo::genesis_to(T0, lock_of(sk)).txs[0].txid(), index: 0 };
    let spent = node.utxo(&op).expect("utxo").as_txout();
    let mut tx = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: op.txid, prev_index: op.index, witness: Vec::new() }],
        outputs: vec![TxOut {
            amount: 18 * 100_000_000,
            lock_type: LOCK_SINGLE,
            lock_data: lock_of(&SigningKey::from_bytes(&[7u8; 32])),
        }],
    };
    let sig = sk.sign(&tx.sighash(&node.chain_id(), 0, &spent));
    let mut witness = Vec::new();
    witness.extend_from_slice(&sk.verifying_key().to_bytes());
    witness.extend_from_slice(&sig.to_bytes());
    tx.inputs[0].witness = witness;
    (node, dir, parent, tx)
}

// A signed spend of the genesis output, accepted by A after 101 blocks (the
// coinbase has matured), is announced to B and fetched by `getdata`.
#[test]
fn a_transaction_relays_to_the_second_node() {
    let sk = SigningKey::from_bytes(&[9u8; 32]);
    let (a, dir_a, _, tx) = node_with_a_spend("tx-a", &sk);
    let dir_b = temp_dir("tx-b");
    let b =
        Node::open(NodeConfig { network: test_network_to(lock_of(&sk)), data_dir: dir_b.clone(), verify_pow: false })
            .expect("b");
    let handle = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| b.height() == 101, 20), "B did not sync");
    let txid = a.submit_tx(tx).expect("submit");
    assert!(wait_for(|| b.has_tx(&txid), 10), "transaction did not relay");
    drop(handle);
    drop(a);
    drop(b);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// A transaction handed to a node that has no peer is announced again, and is no longer once a peer asked for it.
#[test]
fn a_transaction_submitted_without_a_peer_is_announced_again() {
    let sk = SigningKey::from_bytes(&[9u8; 32]);
    let (a, dir_a, _, tx) = node_with_a_spend("resend-a", &sk);
    let txid = a.submit_tx(tx).expect("submit");
    let dir_b = temp_dir("resend-b");
    let b =
        Node::open(NodeConfig { network: test_network_to(lock_of(&sk)), data_dir: dir_b.clone(), verify_pow: false })
            .expect("b");
    let handle = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(handle.addr).expect("connect");
    assert!(wait_for(|| b.height() == 101, 20), "B did not sync");
    assert!(!b.has_tx(&txid), "the only announcement was made before B connected");
    assert_eq!(a.resend_unasked(), 1);
    assert!(wait_for(|| b.has_tx(&txid), 10), "the transaction was not announced again");
    assert!(wait_for(|| a.resend_unasked() == 0, 10), "B asked for it: nothing is left to announce");
    drop(handle);
    drop(a);
    drop(b);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// The transaction of a block that a heavier branch removes returns to the mempool, and leaves it again when a block
// of the new branch holds it.
#[test]
fn a_transaction_of_a_removed_block_returns_to_the_mempool() {
    let sk = SigningKey::from_bytes(&[9u8; 32]);
    let (node, dir, tip, tx) = node_with_a_spend("reorg-tx", &sk);
    let txid = tx.txid();
    let block = |parent, height: u32, salt, txs| demo::block(parent, T0 + 60 * height, 0xC280_0000, salt, txs);
    node.accept_block(block(tip, 102, 1, vec![coinbase(102), tx.clone()])).expect("block with the transaction");
    assert!(!node.has_tx(&txid));
    // two blocks without it on block 101
    let other = block(tip, 102, 2, vec![coinbase(102)]);
    let next = block(other.id(), 103, 2, vec![coinbase(103)]);
    node.accept_block(other).expect("side block");
    node.accept_block(next.clone()).expect("heavier branch");
    assert_eq!((node.height(), node.tip_id()), (103, next.id()));
    assert!(node.has_tx(&txid), "the transaction of the removed block is in the mempool again");
    node.accept_block(block(next.id(), 104, 2, vec![coinbase(104), tx])).expect("block with the transaction");
    assert!(!node.has_tx(&txid));
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
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
