//! Two nodes, each with its own miner, on the easy network with `verify_pow` enabled: the shares one miner finds
//! reach the other node with the `shares` message, and the next block lists the shares of both miners
//! (spec/shares.md sections 11.2 and 11.5, spec/node.md section 8.2). The block of one node holds records of
//! headers that the other node's miner built, with another coinbase, and both nodes end on the same model.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use neb_block::Target;
use neb_chain::block::shares_root;
use neb_chain::{ShareRecord, demo};
use neb_node::params::NetworkParams;
use neb_node::{Node, NodeConfig};
use neb_pow::shares::{Dataset, Model, ShareBuffers};
use neb_work::frames::{FrameOut, Job, decode_model, parse_frame};

const T0: u32 = 1_700_000_000;
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-relay-{tag}-{}-{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn network() -> NetworkParams {
    NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, 0x06],
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

fn open(dir: &std::path::Path) -> Node {
    open_with(dir, true)
}

fn open_with(dir: &std::path::Path, verify_pow: bool) -> Node {
    Node::open(NodeConfig { network: network(), data_dir: dir.to_path_buf(), verify_pow }).expect("node")
}

fn wait_for<F: FnMut() -> bool>(mut f: F, seconds: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    f()
}

/// The model and the job a miner receives for job `id`: the model frame, then the job frame.
fn work_of(node: &Node, id: u64) -> (Model, Job) {
    let frames = node.work_frames(id).expect("the frames of the job");
    let FrameOut::Message { payload, consumed, .. } = parse_frame(&frames).unwrap() else { panic!("model frame") };
    let model = decode_model(payload).unwrap();
    let FrameOut::Message { payload, .. } = parse_frame(&frames[consumed..]).unwrap() else { panic!("job frame") };
    (model, Job::decode(payload).unwrap())
}

/// What a miner does with a job: the first `count` nonces of the reference header whose ticket meets the share
/// target, and the first nonce that meets the block target.
fn scan(model: &Model, job: &Job, ds: &dyn Dataset, count: usize) -> (Vec<u32>, u32) {
    let ex = vec![0u8; job.extranonce_size as usize];
    let (share, block) = (Target(job.share_target), Target(job.target));
    let mut buffers = ShareBuffers::new();
    let (mut shares, mut winner) = (Vec::new(), None);
    for nonce in 0u32.. {
        let ticket = buffers.ticket(model, ds, &job.header(&ex, job.time_start, nonce).id());
        if share.admits(&ticket) && shares.len() < count {
            shares.push(nonce);
        }
        if winner.is_none() && block.admits(&ticket) {
            winner = Some(nonce);
        }
        if shares.len() == count && winner.is_some() {
            break;
        }
    }
    (shares, winner.unwrap())
}

fn record_of(job: &Job, nonce: u32) -> ShareRecord {
    let base = job.header(&vec![0u8; job.extranonce_size as usize], job.time_start, 0);
    ShareRecord { body_root: base.body_root, time: base.time, nonce }
}

// The shares of node B's miner reach node A and are listed by the block A's miner finds; B accepts that block.
#[test]
fn the_block_of_one_node_lists_the_shares_that_the_miner_of_another_node_found() {
    let (dir_a, dir_b) = (temp_dir("a"), temp_dir("b"));
    let (a, b) = (open(&dir_a), open(&dir_b));
    let listen = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(listen.addr).expect("connect");
    assert!(
        wait_for(|| a.peer_info().iter().any(|p| p.handshake) && b.peer_info().iter().any(|p| p.handshake), 15),
        "no handshake"
    );
    let ds = a.network().chain.dataset.clone();
    let start = a.tip_model().hashes();

    // Two miners on the tip (the genesis block), with two coinbases.
    let (ja, jb) = (a.make_job([0x41; 32], T0 + 60).unwrap(), b.make_job([0x42; 32], T0 + 60).unwrap());
    let ((model_a, job_a), (model_b, job_b)) = (work_of(&a, ja), work_of(&b, jb));
    assert_eq!(job_a.parent, job_b.parent);
    assert_ne!(record_of(&job_a, 0).body_root, record_of(&job_b, 0).body_root, "two miners, two bodies");
    let (shares_a, winner_a) = scan(&model_a, &job_a, &*ds, 6);
    let (shares_b, _) = scan(&model_b, &job_b, &*ds, 6);

    // Each miner reports to its own node; the nodes relay to each other.
    for &n in &shares_a {
        assert_eq!(a.accept_share(ja, n), Ok(true));
    }
    assert_eq!(a.accept_share(ja, winner_a), Ok(!shares_a.contains(&winner_a)), "the winner is a share of its round");
    for &n in &shares_b {
        assert_eq!(b.accept_share(jb, n), Ok(true));
    }
    assert_eq!(b.accept_share(jb, shares_b[0]), Ok(false), "a duplicate");
    // A nonce whose ticket is above the share target is refused, not trusted.
    let share = Target(job_b.share_target);
    let mut buffers = ShareBuffers::new();
    let ex_b = vec![0u8; job_b.extranonce_size as usize];
    let bad = (0u32..)
        .find(|&n| !share.admits(&buffers.ticket(&model_b, &*ds, &job_b.header(&ex_b, job_b.time_start, n).id())))
        .unwrap();
    assert!(b.accept_share(jb, bad).is_err());

    let mut expected: Vec<ShareRecord> = shares_a
        .iter()
        .map(|&n| record_of(&job_a, n))
        .chain([record_of(&job_a, winner_a)])
        .chain(shares_b.iter().map(|&n| record_of(&job_b, n)))
        .collect();
    expected.sort_unstable();
    expected.dedup();

    // A's miner finds block 1; B follows it.
    let ex = vec![0u8; job_a.extranonce_size as usize];
    assert_eq!(a.accept_submit(ja, &ex, job_a.time_start, winner_a), Ok(1));
    assert!(wait_for(|| b.height() == 1 && b.tip_id() == a.tip_id(), 20), "B did not follow block 1");

    // The next job of each node lists the shares of both miners (the records of the round of block 1).
    let want = shares_root(&expected);
    for node in [&a, &b] {
        let listed = wait_for(|| work_of(node, node.make_job([0x41; 32], T0 + 120).unwrap()).1.shares_root == want, 20);
        assert!(listed, "a node did not list the {} records of both miners", expected.len());
    }

    // A's miner finds block 2 with that list; B checks it with the real rules and ends on the same model.
    let job2 = a.make_job([0x41; 32], T0 + 120).unwrap();
    let (model2, job_2) = work_of(&a, job2);
    let (_, winner2) = scan(&model2, &job_2, &*ds, 0);
    assert_eq!(a.accept_submit(job2, &ex, job_2.time_start, winner2), Ok(2));
    assert!(wait_for(|| b.height() == 2 && b.tip_id() == a.tip_id(), 20), "B did not accept block 2");
    assert_eq!(a.block_at(2).unwrap().shares, expected, "the block lists the records of both miners");
    assert_eq!(a.tip_model().hashes(), b.tip_model().hashes(), "the same weights and residuals");
    assert_ne!(a.tip_model().hashes(), start, "the shares moved the model");

    drop(listen);
    drop((a, b));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// A peer that relays a record whose ticket is above the share target is banned, like a peer that sends a block with an
// invalid share (spec/node.md section 8.1). The record is not kept and not relayed.
#[test]
fn a_peer_that_relays_a_record_above_the_share_target_is_banned() {
    let (dir_a, dir_b) = (temp_dir("ban-a"), temp_dir("ban-b"));
    let a = open(&dir_a);
    let b = open_with(&dir_b, false); // B does not check what its miners report
    let listen = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    b.connect_peer(listen.addr).expect("connect");
    assert!(
        wait_for(|| a.peer_info().iter().any(|p| p.handshake) && b.peer_info().iter().any(|p| p.handshake), 15),
        "no handshake"
    );
    let ds = a.network().chain.dataset.clone();

    let jb = b.make_job([0x42; 32], T0 + 60).unwrap();
    let (model, job) = work_of(&b, jb);
    let share = Target(job.share_target);
    let mut buffers = ShareBuffers::new();
    let ex = vec![0u8; job.extranonce_size as usize];
    let bad = (0u32..)
        .find(|&n| !share.admits(&buffers.ticket(&model, &*ds, &job.header(&ex, job.time_start, n).id())))
        .unwrap();
    assert_eq!(b.accept_share(jb, bad), Ok(true), "B keeps and relays what it was told");
    let loopback: std::net::IpAddr = "127.0.0.1".parse().unwrap();
    assert!(wait_for(|| a.is_banned(loopback), 15), "A did not ban the peer that relayed the record");
    assert!(wait_for(|| a.peer_count() == 0, 5), "A kept the session");
    // A kept nothing: its next job lists no record.
    let ja = a.make_job([0x41; 32], T0 + 60).unwrap();
    assert_eq!(work_of(&a, ja).1.shares_root, shares_root(&[]));

    drop(listen);
    drop((a, b));
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}
