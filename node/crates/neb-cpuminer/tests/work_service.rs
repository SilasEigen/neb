//! The CPU miner against a node on the easy network with `verify_pow` enabled:
//! model and job frames in, shares and solutions out, and blocks whose share
//! lists contain the shares the miner sent (spec/shares.md section 11.5).

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use neb_chain::demo;
use neb_engines::{Engine, EngineKind};
use neb_node::params::NetworkParams;
use neb_node::style::{ColorMode, Live, Log};
use neb_node::{Node, NodeConfig};
use neb_pow::shares::SyntheticDataset;

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

// Round trip over the real work service with real tickets on the synthetic
// dataset: every solution is accepted, and the winning nonce of each block is
// listed, in increasing order, by the next block.
#[test]
fn the_miner_finds_blocks_and_the_next_block_lists_its_shares() {
    mine_and_check(EngineKind::Reference);
}

// The same with the AVX-512 engine, when the processor has it: the node verifies every proof of work and every
// share, so a wrong ticket from the engine would be refused (rejected solutions), and the reference re-evaluates
// each hit before it is sent.
#[test]
fn the_avx512_miner_finds_blocks_that_the_node_accepts() {
    if !neb_engines::avx512::available() {
        assert!(std::env::var_os("NEB_REQUIRE_AVX512").is_none(), "NEB_REQUIRE_AVX512 is set");
        eprintln!("SKIPPED: no AVX-512 engine here");
        return;
    }
    let engine = mine_and_check(EngineKind::Avx512Sparse);
    let s = engine.stats();
    assert!(s.verified_hits > 0 && s.mismatches == 0 && !s.disabled, "{s:?}");
}

fn mine_and_check(kind: EngineKind) -> Arc<Engine> {
    let t0 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
    let dir = std::env::temp_dir().join(format!("neb-cpuminer-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let network = NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, 0x06],
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
        genesis: demo::easy_genesis(t0),
        chain: demo::easy_params(t0, demo::PROJECT_LOCK),
    };
    let node = Node::open(NodeConfig { network, data_dir: dir.clone(), verify_pow: true }).expect("node");
    let live = Live::new();
    let log = Arc::new(Log::new(live.clone(), ColorMode::Never));
    node.set_miner_lock(demo::MINER_LOCK);
    let work = node.start_work_service("127.0.0.1:0".parse().unwrap(), log, live).expect("work service");

    let engine = Engine::new(kind, Arc::new(SyntheticDataset::new()));
    assert_eq!(engine.active(), kind, "{:?}", engine.messages());
    let miner = neb_cpuminer::start_with_engine(work.addr, vec![engine.clone()], 2, "test-miner").expect("miner");
    assert!(wait_for(|| node.height() >= 8, 120), "the chain stopped at height {}", node.height());
    let stats = miner.stats().clone();
    assert!(wait_for(|| stats.accepted.load(Ordering::Relaxed) >= 6, 20));
    assert_eq!(stats.rejected.load(Ordering::Relaxed), 0, "no solution was refused");
    miner.stop();

    let height = node.height();
    let blocks: Vec<_> = (1..=height).map(|h| node.block_at(h).unwrap()).collect();
    assert!(blocks[0].shares.is_empty());
    let mut listed = 0;
    for pair in blocks.windows(2) {
        // The winner of a round is a share of that round and is reported
        // before the solution, so the next block lists it.
        assert!(
            pair[1].shares.contains(&neb_chain::ShareRecord::of_header(&pair[0].header)),
            "height {}",
            node.height()
        );
        assert!(pair[1].shares.windows(2).all(|w| w[0] < w[1]));
        listed += pair[1].shares.len();
    }
    assert!(listed > blocks.len(), "{listed} shares in {} blocks", blocks.len());
    assert!(stats.shares.load(Ordering::Relaxed) as usize >= listed, "every listed share was sent");
    assert!(stats.attempts.load(Ordering::Relaxed) > 0);
    drop(work);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
    engine
}
