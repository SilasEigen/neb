//! `neb-miner-cpu --node HOST:PORT (--dataset FILE | --synthetic) [--threads N] [--engine NAME] [--plain]` and
//! `neb-miner-cpu (--dataset FILE | --synthetic) --engine NAME --selftest`
//!
//! Command line of the CPU miner of the work service (spec/node.md section
//! 7.1). Shows the dashboard in a terminal; with `--plain`, or when the output
//! is not a terminal, prints a status line every 5 s. The dataset must be the one the node
//! uses; the miner does not check its hash, and a different dataset yields
//! tickets that the node rejects. `--synthetic` is for demo networks only.
//!
//! `--engine` chooses how the attempts are evaluated: `ref` (the C reference, the default), `avx512` (AVX-512 VNNI,
//! sparse variant), `avx512-sparse` or `avx512-dense`. An optimised engine runs a self-test first and is used only
//! if it passes and the processor has the instructions; otherwise the miner says why and uses the reference. The
//! reference re-evaluates every share and solution of an optimised engine before it is reported. `--selftest` runs
//! that self-test and exits: 0 when the requested engine passed, 1 when it did not.
//! Exit code: 0 after `q`, 1 when the node closes the connection or cannot be
//! reached, 2 on a usage error.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use neb_cpuminer::dashboard::{self, Hardware, Info, Outcome};
use neb_engines::{Engine, EngineKind};
use neb_pow::shares::{FileDataset, SharedDataset, SyntheticDataset};

fn usage() -> i32 {
    eprintln!(
        "usage: neb-miner-cpu --node HOST:PORT (--dataset FILE | --synthetic) [--threads N] [--engine ref|avx512|avx512-sparse|avx512-dense] [--plain]"
    );
    eprintln!("       neb-miner-cpu (--dataset FILE | --synthetic) --engine NAME --selftest");
    2
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&args));
}

fn run(args: &[String]) -> i32 {
    let (mut node, mut dataset_path, mut synthetic, mut plain, mut selftest) = (None, None, false, false, false);
    let mut engine_kind = EngineKind::Reference;
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1);
        match (args[i].as_str(), value) {
            ("--node", Some(v)) => node = Some(v.clone()),
            ("--dataset", Some(v)) => dataset_path = Some(v.clone()),
            ("--threads", Some(v)) => match v.parse::<usize>() {
                Ok(n) if n > 0 => threads = n,
                _ => return usage(),
            },
            ("--engine", Some(v)) => match EngineKind::parse(v) {
                Ok(k) => engine_kind = k,
                Err(e) => {
                    eprintln!("{e}");
                    return usage();
                }
            },
            ("--selftest", _) => {
                selftest = true;
                i += 1;
                continue;
            }
            ("--synthetic", _) => {
                synthetic = true;
                i += 1;
                continue;
            }
            ("--plain", _) => {
                plain = true;
                i += 1;
                continue;
            }
            _ => return usage(),
        }
        i += 2;
    }
    if node.is_none() && !selftest {
        return usage();
    }
    let dataset: SharedDataset = match (synthetic, dataset_path) {
        (true, _) => Arc::new(SyntheticDataset::new()),
        (false, Some(path)) => match FileDataset::load(std::path::Path::new(&path), None) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("cannot load the dataset: {e}");
                return 1;
            }
        },
        (false, None) => return usage(),
    };
    let engine = Engine::new(engine_kind, dataset);
    for message in engine.messages() {
        eprintln!("neb-miner-cpu: {message}");
    }
    if selftest {
        let ok = engine.active() == engine_kind;
        println!(
            "engine {}: {}",
            engine_kind.name(),
            if ok { "SELFTEST PASSED" } else { "SELFTEST FAILED or engine unavailable" }
        );
        return i32::from(!ok);
    }
    let Some(node) = node else { return usage() };
    let engine_name = engine.active().name();
    let miner = match neb_cpuminer::start_with_engine(
        node.as_str(),
        engine.clone(),
        threads,
        concat!("neb-miner-cpu ", env!("CARGO_PKG_VERSION")),
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot connect to {node}: {e}");
            return 1;
        }
    };
    if !plain && dashboard::available() {
        let info = Info {
            title: "CPU proof-of-work miner".into(),
            device: format!("{threads} threads, engine {engine_name}"),
            node: node.clone(),
        };
        let outcome = dashboard::run(&miner.session(), &info, &|| Hardware::Cpu { threads });
        miner.stop();
        return if outcome == Outcome::Quit { 0 } else { 1 };
    }
    println!("neb-miner-cpu: {threads} threads, engine {engine_name}, node {node}");
    let started = Instant::now();
    let (mut last, mut last_attempts) = (started, 0u64);
    while !miner.stopped() {
        std::thread::sleep(Duration::from_secs(5));
        let stats = miner.stats();
        let attempts = stats.attempts.load(Ordering::Relaxed);
        let now = Instant::now();
        let rate = (attempts - last_attempts) as f64 / now.duration_since(last).as_secs_f64();
        println!(
            "{:>5}s  {:>8.0} attempts/s  shares {}  blocks {} (accepted {}, rejected {})",
            started.elapsed().as_secs(),
            rate,
            stats.shares.load(Ordering::Relaxed),
            stats.blocks.load(Ordering::Relaxed),
            stats.accepted.load(Ordering::Relaxed),
            stats.rejected.load(Ordering::Relaxed),
        );
        let e = engine.stats();
        if e.disabled || e.mismatches > 0 {
            println!(
                "        engine {} switched off after {} mismatches with the reference; mining with the reference",
                engine_kind.name(),
                e.mismatches
            );
        }
        (last, last_attempts) = (now, attempts);
    }
    eprintln!("the node closed the connection");
    miner.stop();
    1
}
