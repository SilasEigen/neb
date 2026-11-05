//! Work service job push (spec/node.md section 7.1): when the best block
//! changes, every connected miner receives a model frame and a fresh job,
//! whether the block came from a peer or from the miner's own submission.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use neb_chain::demo;
use neb_chain::state::{project_share, subsidy};
use neb_node::params::NetworkParams;
use neb_node::style::{ColorMode, Live, Log};
use neb_node::{Node, NodeConfig};
use neb_work::frames::{FrameOut, Job, MSG_JOB, MSG_MODEL, MSG_RESULT, ResultMsg, Submit, parse_frame};

const T0: u32 = 1_700_000_000;
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-work-{tag}-{}-{}", std::process::id(), n));
    let _ = fs::remove_dir_all(&p);
    p
}

fn test_network() -> NetworkParams {
    NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, 0x04],
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
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

/// Reads one work-service frame off the miner socket. The buffer must live
/// across calls: a read may deliver more than one frame.
fn read_frame(stream: &mut TcpStream, buf: &mut Vec<u8>) -> (u8, Vec<u8>) {
    let mut tmp = [0u8; 4096];
    loop {
        match parse_frame(buf) {
            Ok(FrameOut::Message { msg_type, payload, consumed }) => {
                let payload = payload.to_vec();
                buf.drain(..consumed);
                return (msg_type, payload);
            }
            Ok(FrameOut::Need(_)) => {
                let n = stream.read(&mut tmp).expect("read a work-service frame");
                assert!(n > 0, "the work service closed the socket");
                buf.extend_from_slice(&tmp[..n]);
            }
            Err(e) => panic!("bad frame: {e}"),
        }
    }
}

/// A job is always preceded by the model frame it is mined on.
fn read_job(stream: &mut TcpStream, buf: &mut Vec<u8>) -> Job {
    let (msg_type, _) = read_frame(stream, buf);
    assert_eq!(msg_type, MSG_MODEL, "expected a model frame");
    let (msg_type, payload) = read_frame(stream, buf);
    assert_eq!(msg_type, MSG_JOB, "expected a job frame");
    Job::decode(&payload).expect("job decodes")
}

fn read_result(stream: &mut TcpStream, buf: &mut Vec<u8>) -> ResultMsg {
    let (msg_type, payload) = read_frame(stream, buf);
    assert_eq!(msg_type, MSG_RESULT, "expected a result frame");
    ResultMsg::decode(&payload).expect("result decodes")
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

// A miner that submitted nothing still gets a job on the new tip after a
// block arrives over the peer network.
#[test]
fn a_relayed_block_pushes_a_job_to_the_connected_miner() {
    let dir_a = temp_dir("a");
    let dir_b = temp_dir("b");
    let a = open(&dir_a);
    let b = open(&dir_b);

    // B holds block 1; the miner connects to A, which is still at the genesis block.
    let block = demo::block(b.tip_id(), T0 + 61, 0xC280_0000, 4242, vec![coinbase(1)]);
    b.accept_block(block.clone()).expect("B connects its block");
    assert_eq!(b.height(), 1);
    let live = Live::new();
    let log = Arc::new(Log::new(live.clone(), ColorMode::Never));
    a.set_miner_lock(demo::MINER_LOCK);
    let work = a.start_work_service("127.0.0.1:0".parse().unwrap(), log, live).expect("work service");
    let mut miner = TcpStream::connect(work.addr).expect("miner dials the work service");
    miner.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut buf = Vec::new();
    let first = read_job(&mut miner, &mut buf);
    assert_eq!(first.parent, a.tip_id());
    assert_eq!(first.height, 1);

    // A dials B and connects block 1; the next job reaches the miner
    // unsolicited and builds on it.
    let p2p = b.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    a.connect_peer(p2p.addr).expect("A dials B");
    assert!(wait_for(|| a.height() == 1 && a.tip_id() == block.id(), 15), "A did not sync the relayed block");
    let second = read_job(&mut miner, &mut buf);
    assert_eq!(second.parent, block.id(), "the fresh job follows the relayed block");
    assert_eq!(second.height, 2);
    assert_ne!(second.id, first.id, "the job must be a new one");
    assert_eq!(a.miner_count(), 1);

    drop(p2p);
    drop(work);
    drop(miner);
    drop(a);
    drop(b);
    let _ = fs::remove_dir_all(&dir_a);
    let _ = fs::remove_dir_all(&dir_b);
}

// After a submission is connected, the miner receives the new job before the
// result frame, so it never mines on the old tip after "accepted".
#[test]
fn a_miners_own_submit_leaves_it_with_a_fresh_job() {
    let dir = temp_dir("submit");
    let node = open(&dir);
    let live = Live::new();
    let log = Arc::new(Log::new(live.clone(), ColorMode::Never));
    node.set_miner_lock(demo::MINER_LOCK);
    let work = node.start_work_service("127.0.0.1:0".parse().unwrap(), log, live).expect("work service");
    let mut miner = TcpStream::connect(work.addr).expect("miner dials the work service");
    miner.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut buf = Vec::new();
    let first = read_job(&mut miner, &mut buf);

    let submit = Submit {
        id: first.id,
        extranonce: vec![0xAB; first.extranonce_size as usize],
        time: first.time_start,
        nonce: 7,
    };
    miner.write_all(&submit.encode()).expect("submit");
    // Connecting the block broadcasts the job; the result frame follows.
    let second = read_job(&mut miner, &mut buf);
    assert_eq!(second.parent, node.tip_id());
    assert_eq!(second.height, 2);
    assert_eq!(node.height(), 1);
    let result = read_result(&mut miner, &mut buf);
    assert!(result.accepted, "the submission must be accepted in this test");
    assert_eq!(result.code, 0);

    drop(work);
    drop(miner);
    drop(node);
    let _ = fs::remove_dir_all(&dir);
}

// Without --miner-lock a node gives no work: a miner that connects is closed without a job, so no block can pay a
// lock that nobody chose.
#[test]
fn a_node_without_a_miner_lock_refuses_miners() {
    let dir = temp_dir("nolock");
    let node = open(&dir);
    let live = Live::new();
    let log = Arc::new(Log::new(live.clone(), ColorMode::Never));
    let work = node.start_work_service("127.0.0.1:0".parse().unwrap(), log, live).expect("work service");
    let mut miner = TcpStream::connect(work.addr).expect("miner dials the work service");
    miner.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut buf = [0u8; 64];
    assert_eq!(miner.read(&mut buf).expect("the node closes the connection"), 0);
    assert_eq!(node.height(), 0);
    let _ = fs::remove_dir_all(&dir);
}
