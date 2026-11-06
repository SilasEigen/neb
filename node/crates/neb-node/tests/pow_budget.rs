//! The proof budget and the ban of spec/node.md section 8.1, with `verify_pow`
//! enabled on the receiving node. The demo network target is 2^217 (bits
//! 0xC280_0000), so a block with an arbitrary nonce passes the cheap checks
//! and fails the proof check, as the block of a hostile peer would. Shares are
//! tested on the easy network (target near 2^250), where blocks can be mined.

use std::io::Read;
use std::net::{IpAddr, TcpStream};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use neb_chain::demo;
use neb_chain::state::{ChainState, project_share, subsidy};
use neb_chain::{Block, ShareRecord, share_key, share_target};
use neb_node::budget::TokenBucket;
use neb_node::params::NetworkParams;
use neb_node::{BlockOutcome, Node, NodeConfig, NodeError};
use neb_pow::shares::ShareBuffers;

const T0: u32 = 1_700_000_000;
const BITS: u32 = 0xC280_0000;
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-budget-{tag}-{}-{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn network(easy: bool, magic: u8) -> NetworkParams {
    NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, magic],
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
        genesis: if easy { demo::easy_genesis(T0) } else { demo::genesis(T0) },
        chain: if easy { demo::easy_params(T0, demo::PROJECT_LOCK) } else { demo::demo_params(T0, demo::PROJECT_LOCK) },
    }
}

fn open(dir: &Path, easy: bool, verify_pow: bool) -> Node {
    Node::open(NodeConfig { network: network(easy, 3), data_dir: dir.to_path_buf(), verify_pow }).expect("node")
}

/// A block on the genesis with the right `bits` and time and an arbitrary nonce.
fn unproved_block(nonce: u32) -> Block {
    let reward = subsidy(1);
    let coinbase = demo::coinbase(1, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward));
    demo::block(demo::genesis(T0).id(), T0 + 60, BITS, nonce, vec![coinbase])
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

fn loopback() -> IpAddr {
    "127.0.0.1".parse().unwrap()
}

/// True when the node closes a fresh connection without sending a byte.
fn is_refused(addr: std::net::SocketAddr) -> bool {
    let mut stream = TcpStream::connect(addr).expect("tcp connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut byte = [0u8; 1];
    match stream.read(&mut byte) {
        Ok(n) => n == 0,
        Err(e) => matches!(e.kind(), std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted),
    }
}

// Attack scenario: a peer serves a block with a valid header and an invalid
// proof. The receiver closes the session, bans the address for inbound and
// outbound use, and does not store the block.
#[test]
fn a_peer_that_has_a_block_with_an_invalid_proof_is_banned() {
    let dir_a = temp_dir("pow-a");
    let dir_b = temp_dir("pow-b");
    let a = open(&dir_a, false, true);
    let b = open(&dir_b, false, false);
    let bad = unproved_block(1);
    b.accept_block(bad.clone()).expect("B skips the proof check");
    let a_listen = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    let b_listen = b.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");

    b.connect_peer(a_listen.addr).expect("connect");
    assert!(wait_for(|| a.is_banned(loopback()), 15), "A did not ban the sender");
    assert!(wait_for(|| a.peer_count() == 0, 5), "A kept the session");
    assert!(!a.has_block(&bad.id()));

    assert!(is_refused(a_listen.addr), "A accepted a banned address");
    let dial = a.connect_peer(b_listen.addr).expect_err("A dialled a banned address");
    assert_eq!(dial.kind(), std::io::ErrorKind::PermissionDenied);

    drop((a_listen, b_listen));
    drop((a, b));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// The same ban applies to a block relayed after the handshake, not only to
// blocks requested during synchronisation.
#[test]
fn a_block_announced_after_the_handshake_is_checked_too() {
    let dir_a = temp_dir("ann-a");
    let dir_b = temp_dir("ann-b");
    let a = open(&dir_a, false, true);
    let b = open(&dir_b, false, false);
    let a_listen = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(a_listen.addr).expect("connect");
    assert!(wait_for(|| a.peer_info().iter().any(|p| p.handshake), 15), "no handshake");

    let bad = unproved_block(2);
    b.accept_block(bad.clone()).expect("B skips the proof check");
    assert!(wait_for(|| a.is_banned(loopback()), 15), "A did not ban the sender");
    assert!(!a.has_block(&bad.id()));
    assert!(is_refused(a_listen.addr), "A accepted a banned address");

    drop(a_listen);
    drop((a, b));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// Two mined blocks on the easy network; the second lists `listed` of the
/// first round (the records are chosen by the caller from the model).
fn mined_pair(listed: impl Fn(&ChainState) -> Vec<ShareRecord>) -> (Block, Block) {
    let mut probe =
        ChainState::from_genesis(demo::easy_params(T0, demo::PROJECT_LOCK), demo::easy_genesis(T0)).unwrap();
    let (first, _) = demo::mine_plain(&probe, vec![]);
    probe.connect_block(&first).unwrap();
    let shares = listed(&probe);
    let (second, _) = demo::mine_plain(&probe, shares);
    (first, second)
}

// A block with a valid proof but one share that misses the share target is
// refused and its sender banned; the preceding valid block stays connected.
#[test]
fn a_peer_that_sends_a_block_with_an_invalid_share_is_banned() {
    let dir_a = temp_dir("share-a");
    let dir_b = temp_dir("share-b");
    let a = open(&dir_a, true, true);
    let b = open(&dir_b, true, false);
    // Height 2 lists a record of the round of block 1 whose ticket on the
    // model of the genesis misses the share target.
    let (first, second) = mined_pair(|probe| {
        let model_pp = probe.model_at(0).unwrap();
        let target = share_target(probe.tip().header.bits).unwrap();
        let mut buffers = ShareBuffers::new();
        let tip = probe.tip().header;
        let bad = (0u32..)
            .map(|nonce| ShareRecord { body_root: [0xAB; 32], time: tip.time, nonce })
            .find(|r| {
                !target.admits(&buffers.ticket(
                    &model_pp,
                    &*probe.params.datasets[0],
                    &share_key(r, tip.parent, tip.bits),
                ))
            })
            .unwrap();
        vec![bad]
    });
    b.accept_block(first.clone()).expect("B skips the proof check");
    b.accept_block(second.clone()).expect("B skips the proof check");
    let a_listen = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(a_listen.addr).expect("connect");
    assert!(wait_for(|| a.is_banned(loopback()), 20), "A did not ban the sender");
    assert_eq!(a.height(), 1, "the valid block connected, the bad one did not");
    assert!(!a.has_block(&second.id()));
    drop(a_listen);
    drop((a, b));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// Cost accounting of spec/node.md section 8.1 at 1 token per second: tokens
// are spent only after the cheap checks pass.
#[test]
fn the_session_pays_one_token_for_the_block_and_one_for_each_share() {
    let dir = temp_dir("tokens");
    let node = open(&dir, true, true);
    let (first, second) = mined_pair(|probe| {
        let model_pp = probe.model_at(0).unwrap();
        let target = share_target(probe.tip().header.bits).unwrap();
        let mut buffers = ShareBuffers::new();
        let tip = probe.tip().header;
        let mut two: Vec<ShareRecord> = (0u32..)
            .map(|nonce| ShareRecord { body_root: [0xAB; 32], time: tip.time, nonce })
            .filter(|r| {
                target.admits(&buffers.ticket(
                    &model_pp,
                    &*probe.params.datasets[0],
                    &share_key(r, tip.parent, tip.bits),
                ))
            })
            .take(2)
            .collect();
        two.sort_unstable();
        two
    });
    assert_eq!(second.shares.len(), 2);
    // The bucket starts with one token: block 1 (one evaluation) is
    // immediate; block 2 needs three evaluations (proof and two shares).
    let mut budget = TokenBucket::new(NonZeroU32::new(1).unwrap(), Instant::now());
    let start = Instant::now();
    assert_eq!(node.accept_block_from(first, Some(&mut budget)).unwrap(), BlockOutcome::Connected { height: 1 });
    assert!(start.elapsed() < Duration::from_millis(900), "the first token is free");
    // a block that fails a cheap check does not take a token
    let mut orphan = second.clone();
    orphan.header.parent = [7u8; 32];
    let cheap = Instant::now();
    assert!(matches!(node.accept_block_from(orphan, Some(&mut budget)), Err(NodeError::Orphan)));
    assert!(cheap.elapsed() < Duration::from_millis(500), "a cheap failure waited");
    let before = Instant::now();
    assert_eq!(node.accept_block_from(second, Some(&mut budget)).unwrap(), BlockOutcome::Connected { height: 2 });
    assert!(
        before.elapsed() >= Duration::from_millis(2400),
        "three evaluations at one per second: {:?}",
        before.elapsed()
    );
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}
