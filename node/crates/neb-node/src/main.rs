//! Command line of the NEB node binary: `--run` starts the node, `--genesis`
//! prints the network parameters, `--model-accuracy` evaluates the chain model,
//! and the `--*-selftest` options run built-in checks. Exit code 0 is success,
//! 1 a runtime failure, 2 a usage error.
//!
//! The daemon is the `neb_node` library (spec/node.md); the share step of the
//! proof of work is the pinned C reference in `pow/ref` linked by `neb-pow`
//! (spec/shares.md).

use neb_block::{Header, Target};
use neb_difficulty::{HALF_LIFE, Params, SPACING};
use neb_node::params;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("--version") | Some("version") => {
            print_version();
            0
        }
        Some("--check-hardware") => check_hardware(),
        Some("--selftest") | Some("selftest") => selftest(),
        Some("--shares-selftest") => shares_selftest(),
        Some("--chain-selftest") => chain_selftest(),
        Some("--work-selftest") => work_selftest(),
        Some("--run") => neb_node::daemon::run(&args[1..]),
        Some("--genesis") => print_genesis(),
        Some("--dataset-hash") => dataset_hash(args.get(1)),
        Some("--model-accuracy") => model_accuracy(&args[1..]),
        Some("--help") | Some("-h") | None => {
            print_help();
            0
        }
        Some(other) => {
            eprintln!("neb-node: unknown option {other}");
            print_help();
            2
        }
    };
    std::process::exit(code);
}

fn print_version() {
    println!("neb-node {}", env!("CARGO_PKG_VERSION"));
}

fn print_help() {
    println!(
        "neb-node: the NEB testnet full node\n\n\
usage:\n\
  neb-node --version         print the version\n\
  neb-node --check-hardware  print the CPU and the proof-check path\n\
  neb-node --selftest        run the internal checks and exit\n\
  neb-node --shares-selftest replay the published training-share vectors\n\
  neb-node --chain-selftest  connect a demo chain and reorganise it\n\
  neb-node --work-selftest   build a job, submit it, connect the block\n\
  neb-node --genesis         print the testnet genesis and network parameters\n\
  neb-node --dataset-hash F  print the BLAKE3 hash of a dataset file\n\
  neb-node --model-accuracy --dataset F --datadir D [--samples N]\n\
                             accuracy of the chain model on the first N samples\n\
  neb-node --run --dataset F [options]  run the testnet node (see docs/testnet.md)"
    );
}

// The proof check is the portable C reference compiled into the binary, so no
// CPU feature detection is needed.
fn check_hardware() -> i32 {
    println!("cpu: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    println!("proof check: the pinned C reference (pow/ref), linked in");
    println!("NODE READY");
    0
}

fn selftest() -> i32 {
    let mut fail = 0;

    let h = Header { version: 1, bits: 0xBB80_0000, time: 1_800_000_000, nonce: 0xDEAD_BEEF, ..Header::default() };
    ok("header round-trip", Header::decode(&h.encode()) == h, &mut fail);

    match Target::from_bits(0xBB80_0000) {
        Ok(t) => ok("compact target round-trip", t.to_bits() == Ok(0xBB80_0000), &mut fail),
        Err(_) => ok("compact target round-trip", false, &mut fail),
    }

    let params = Params {
        spacing: SPACING,
        half_life: HALF_LIFE,
        pow_limit_bits: 0xC280_0000,
        anchor_bits: 0xBB80_0000,
        anchor_time: 1_800_000_000,
    };
    let mut steady = true;
    for h in 1u32..40 {
        steady &= params.next_bits(h - 1, 1_800_000_000 + 60 * (h - 1)).ok() == Some(0xBB80_0000);
    }
    ok("asert steady blocks keep the target", steady, &mut fail);

    if fail == 0 {
        println!("selftest: PASS");
    } else {
        println!("selftest: FAIL");
    }
    fail
}

// Replays vectors/shares.txt (spec/shares.md section 10) through the linked
// reference: every share ticket, then the state after every step. The counts
// are those of the vector file.
fn shares_selftest() -> i32 {
    const VECTORS: &str = include_str!("../../../../vectors/shares.txt");
    let mut fail = 0;
    match neb_pow::shares::replay_vectors(VECTORS) {
        Ok((steps, shares)) => ok(
            &format!("shares vectors: {steps} steps, {shares} shares bit-exact"),
            steps == 24 && shares == 543,
            &mut fail,
        ),
        Err(e) => ok(&format!("shares vectors: {e}"), false, &mut fail),
    }
    if fail == 0 {
        println!("shares selftest: PASS");
    } else {
        println!("shares selftest: FAIL");
    }
    fail
}

// Five blocks mined with real shares on the synthetic dataset and connected,
// then a three-block branch from height 2 that overtakes them (reorganisation
// of 3 blocks). The conservation invariant (spec/node.md section 6.7) is
// checked at the end. The demo targets are far easier than the launch bits.
fn chain_selftest() -> i32 {
    use neb_chain::ChainState;
    use neb_chain::demo;
    use neb_chain::state::{project_share, subsidy};

    const T0: u32 = 1_800_000_000;
    let mut fail = 0;
    let new_state = || ChainState::from_genesis(demo::easy_params(T0, demo::PROJECT_LOCK), demo::easy_genesis(T0));
    let mut state = match new_state() {
        Ok(s) => s,
        Err(e) => {
            println!("FAIL chain: genesis: {e}");
            return 1;
        }
    };
    let mut blocks = Vec::new();
    let mut connected = true;
    let mut round = Vec::new();
    for _ in 1..=5u32 {
        let (blk, next) = demo::mine_plain(&state, round);
        round = next;
        if let Err(e) = state.connect_block(&blk) {
            println!("FAIL chain: connect at height {}: {e}", state.height() + 1);
            connected = false;
            break;
        }
        blocks.push(blk);
    }
    ok("chain: five mined blocks connected, conservation holds", connected && state.conservation(), &mut fail);

    // Competing branch from height 2 with a different miner lock, so its
    // coinbases differ from the main chain.
    let mut probe = new_state().expect("genesis");
    for blk in &blocks[..2] {
        probe.connect_block(blk).expect("replay");
    }
    let mut branch = Vec::new();
    let time = probe.tip_time();
    let mut round =
        probe.valid_shares((0..200).map(|nonce| neb_chain::ShareRecord { body_root: [0xAB; 32], time, nonce }));
    for _ in 0..3u32 {
        let height = probe.height() + 1;
        let reward = subsidy(height);
        let coinbase = demo::coinbase(height, [0x33; 32], demo::PROJECT_LOCK, reward - project_share(reward));
        let (blk, next) = demo::mine_next(&probe, vec![coinbase], round);
        round = next;
        probe.connect_block(&blk).expect("branch");
        branch.push(blk);
    }
    let reorged = state.reorg(2, &branch).is_ok();
    ok(
        "chain: reorganisation from height 2, conservation holds, model equals the replay",
        reorged
            && state.height() == 5
            && state.conservation()
            && state.tip_id() == branch[2].id()
            && state.tip_model().hashes() == probe.tip_model().hashes(),
        &mut fail,
    );
    if fail == 0 {
        println!("chain selftest: PASS");
    } else {
        println!("chain selftest: FAIL");
    }
    fail
}

// Work service round trip on a demo chain: job, nonce search against the
// block target with the job's attack data, submission rebuilt into a block,
// block connected, and the job frame decoded from its wire encoding.
fn work_selftest() -> i32 {
    use neb_chain::ChainState;
    use neb_chain::demo;
    use neb_pow::shares::ShareBuffers;
    use neb_work::frames::{FrameOut, Job, parse_frame};
    use neb_work::{Mempool, Work};

    const T0: u32 = 1_800_000_000;
    let mut fail = 0;
    let mut state = match ChainState::from_genesis(demo::easy_params(T0, demo::PROJECT_LOCK), demo::easy_genesis(T0)) {
        Ok(s) => s,
        Err(e) => {
            println!("FAIL work: genesis: {e}");
            return 1;
        }
    };
    let mut round = Vec::new();
    for _ in 1..=3u32 {
        let (blk, next) = demo::mine_plain(&state, round);
        round = next;
        state.connect_block(&blk).expect("demo block");
    }
    let pool = Mempool::new();
    let mut work = Work::new();
    let id = match work.make_job(&state, &pool, demo::MINER_LOCK, T0 + 600) {
        Ok(id) => id,
        Err(e) => {
            println!("FAIL work: job: {e}");
            return 1;
        }
    };
    let template = work.job(id).expect("job");
    let (job, attack) = (template.job.clone(), template.model.clone());
    let ex = vec![0xABu8; job.extranonce_size as usize];
    let target = Target(job.target);
    let mut buffers = ShareBuffers::new();
    let nonce = (0u32..)
        .find(|&n| {
            target.admits(&buffers.ticket(&attack, &*state.params.dataset, &job.header(&ex, job.time_start, n).id()))
        })
        .expect("a nonce");
    let block = match work.submit(id, &ex, job.time_start, nonce) {
        Ok(b) => b,
        Err(e) => {
            println!("FAIL work: submit: {e}");
            return 1;
        }
    };
    ok(
        "work: template and submission rebuild one block",
        block.body_ok() && block.header == job.header(&ex, job.time_start, nonce),
        &mut fail,
    );
    let connected = state.connect_block(&block).is_ok() && state.conservation();
    ok("work: the submitted block connects, conservation holds", connected, &mut fail);
    let framed = job.encode();
    let roundtrip = matches!(
        parse_frame(&framed),
        Ok(FrameOut::Message { payload, .. }) if Job::decode(payload).ok().as_ref() == Some(&job)
    );
    ok("work: job frame round-trips", roundtrip, &mut fail);
    if fail == 0 {
        println!("work selftest: PASS");
    } else {
        println!("work selftest: FAIL");
    }
    fail
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn print_genesis() -> i32 {
    let genesis = params::genesis();
    let id = genesis.id();
    println!("network     : {}", params::NAME);
    println!("beacon      : {}", params::BEACON);
    println!("genesis id  : {}", hex(&id));
    println!("genesis time: {}", genesis.header.time);
    println!("genesis bits: 0x{:08x}", genesis.header.bits);
    println!("pow limit   : 0x{:08x}", params::POW_LIMIT_BITS);
    println!("magic       : {}", hex(&params::magic()));
    println!("p2p port    : {}", params::P2P_PORT);
    println!("work port   : {}", params::WORK_PORT);
    println!("rpc port    : {}", params::RPC_PORT);
    println!("dataset hash: {}", hex(&params::DATASET_HASH));
    println!("miner lock  : {} (no key: the genesis coins cannot be spent)", hex(&params::GENESIS_MINER_LOCK));
    println!("project key : {}", hex(&params::PROJECT_PUBLIC_KEY));
    println!("project lock: {}", hex(&params::project_share_lock()));
    0
}

// Replays the stored chain and counts how many of the first N dataset samples
// the initial model and the model of the tip classify correctly. The data
// directory must not be in use by a running node.
fn model_accuracy(args: &[String]) -> i32 {
    const USAGE: &str = "usage: neb-node --model-accuracy --dataset FILE --datadir DIR [--samples N]";
    let (mut dataset, mut datadir, mut samples) = (None, None, 2000u32);
    for pair in args.chunks(2) {
        match (pair[0].as_str(), pair.get(1)) {
            ("--dataset", Some(v)) => dataset = Some(v.clone()),
            ("--datadir", Some(v)) => datadir = Some(v.clone()),
            ("--samples", Some(v)) => match v.parse() {
                Ok(n) => samples = n,
                Err(_) => {
                    eprintln!("--samples needs a non-negative integer, got {v}");
                    eprintln!("{USAGE}");
                    return 2;
                }
            },
            _ => {
                eprintln!("{USAGE}");
                return 2;
            }
        }
    }
    let (Some(dataset), Some(datadir)) = (dataset, datadir) else {
        eprintln!("{USAGE}");
        return 2;
    };
    let ds = match params::load_dataset(std::path::Path::new(&dataset)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cannot load the dataset: {e}");
            return 1;
        }
    };
    let node = match neb_node::Node::open(neb_node::NodeConfig::new(params::testnet(ds.clone()), &datadir)) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("cannot open {datadir}: {e}");
            return 1;
        }
    };
    let initial = neb_pow::shares::Model::from_seed(&params::genesis_id());
    let tip = node.tip_model();
    let samples = samples.min(ds.n());
    let right = |m: &neb_pow::shares::Model| {
        (0..samples)
            .filter(|&i| {
                let (x, label) = ds.get(i);
                m.predict(&x) == label
            })
            .count()
    };
    println!("height {}  weights {}", node.height(), hex(&node.weights_hash()));
    println!("initial model: {} of {samples} correct", right(&initial));
    println!("chain model  : {} of {samples} correct", right(&tip));
    0
}

fn dataset_hash(arg: Option<&String>) -> i32 {
    let Some(path) = arg else {
        eprintln!("usage: neb-node --dataset-hash <file>");
        return 2;
    };
    match std::fs::read(path) {
        Ok(bytes) => {
            println!("{}", hex(blake3::hash(&bytes).as_bytes()));
            0
        }
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            1
        }
    }
}

fn ok(name: &str, cond: bool, fail: &mut i32) {
    if cond {
        println!("PASS {name}");
    } else {
        println!("FAIL {name}");
        *fail = 1;
    }
}
