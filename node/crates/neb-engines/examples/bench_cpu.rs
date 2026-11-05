//! Throughput of the ticket engines on this machine, as JSON lines.
//!
//! `cargo run --release -p neb-engines --example bench_cpu -- [options]`
//!   --engines LIST    comma list of ref, avx512-dense, avx512-sparse, checked:avx512-sparse, checked:avx512-dense
//!                     (default: all that can run here). `checked:` runs the engine as a miner does (reference
//!                     verification of hits at the share target 2^-15 and one audit in 4096).
//!   --threads LIST    comma list of thread counts (default: 1 and the number of logical processors)
//!   --attempts N      attempts per repetition (default 262144)
//!   --reps N          repetitions after one warm-up (default 5)
//!   --dataset FILE    network dataset file (default: the synthetic dataset)
//!   --model FILE      a raw model dump (508160 bytes: W1, W2, R1, R2); repeatable. Default: the initial model.
//!   --seed-model      add the initial model of the zero seed
//! Every engine evaluates the same keys on the same model; `checksum` (xor of the first 8 bytes of every ticket)
//! must be equal across engines for one model and attempt count, otherwise the run is invalid.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use neb_block::Target;
use neb_engines::avx512::{self, FlatDataset, PreparedWeights, Scratch};
use neb_engines::{Engine, EngineKind};
use neb_pow::shares::{FileDataset, Model, N1, N2, ShareBuffers, SharedDataset, SyntheticDataset};

fn key(k: u64) -> [u8; 32] {
    let mut input = b"neb-bench-key".to_vec();
    input.extend_from_slice(&k.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

fn load_model(path: &str) -> Result<Model, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() != (N1 + N2) * 5 {
        return Err(format!("{path}: {} bytes, expected {}", b.len(), (N1 + N2) * 5));
    }
    let w: Vec<i8> = b[..N1 + N2].iter().map(|&x| x as i8).collect();
    let r: Vec<i32> = b[N1 + N2..].as_chunks::<4>().0.iter().map(|&c| i32::from_le_bytes(c)).collect();
    Model::from_state(&w[..N1], &w[N1..], &r[..N1], &r[N1..])
}

fn sha256_hex(path: &str) -> String {
    // sha256 is not a dependency: the shell wrapper records the hashes of the model files; here the BLAKE3 will do
    let b = std::fs::read(path).unwrap_or_default();
    blake3::hash(&b).to_hex().to_string()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// One repetition: `attempts` keys on `threads` threads; returns (seconds, checksum).
fn run(
    engine: &str,
    model: &Arc<Model>,
    ds: &SharedDataset,
    flat: &Option<Arc<FlatDataset>>,
    threads: usize,
    attempts: u64,
) -> (f64, u64) {
    let next = AtomicU64::new(0);
    let checksum = AtomicU64::new(0);
    let idx = AtomicUsize::new(0);
    let share = {
        let mut t = [0xFFu8; 32];
        t[31] = 0;
        t[30] = 1; // about 2^-15 of the tickets
        Target(t)
    };
    let block = Target([0u8; 32]);
    let checked_engine: Option<Arc<Engine>> =
        engine.strip_prefix("checked:").map(|n| Engine::new(EngineKind::parse(n).unwrap(), ds.clone()));
    let prepared = checked_engine.as_ref().map(|e| e.prepare(model));
    let weights = if engine.starts_with("avx512") { Some(PreparedWeights::new(model).unwrap()) } else { None };
    let sparse = engine == "avx512-sparse";
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let _ = idx.fetch_add(1, Ordering::Relaxed);
                let mut local = 0u64;
                let mut buffers = ShareBuffers::new();
                let mut scratch = if weights.is_some() { Some(Scratch::new().unwrap()) } else { None };
                let mut worker = checked_engine.as_ref().map(|e| e.worker());
                loop {
                    let start = next.fetch_add(64, Ordering::Relaxed);
                    if start >= attempts {
                        break;
                    }
                    for k in start..(start + 64).min(attempts) {
                        let key = key(k);
                        let ticket = if let Some(w) = worker.as_mut() {
                            w.attempt(prepared.as_ref().unwrap(), &key, &share, &block).ticket
                        } else if let (Some(sc), Some(wt), Some(fl)) =
                            (scratch.as_mut(), weights.as_ref(), flat.as_ref())
                        {
                            sc.eval(wt, fl, &key, sparse).ticket
                        } else {
                            buffers.ticket(model, &**ds, &key)
                        };
                        local ^= u64::from_le_bytes(ticket[..8].try_into().unwrap());
                    }
                }
                checksum.fetch_xor(local, Ordering::Relaxed);
            });
        }
    });
    (t0.elapsed().as_secs_f64(), checksum.load(Ordering::Relaxed))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let all_logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    let threads: Vec<usize> =
        get("--threads").map_or(vec![1, all_logical], |s| s.split(',').map(|x| x.parse().unwrap()).collect());
    let attempts: u64 = get("--attempts").map_or(262_144, |s| s.parse().unwrap());
    let reps: usize = get("--reps").map_or(5, |s| s.parse().unwrap());
    let dataset_path = get("--dataset");
    let ds: SharedDataset = match &dataset_path {
        Some(p) => Arc::new(FileDataset::load(std::path::Path::new(p), None).expect("dataset")),
        None => Arc::new(SyntheticDataset::new()),
    };
    let dataset_name = dataset_path.clone().unwrap_or_else(|| "synthetic".into());
    let mut engines: Vec<String> = get("--engines").map_or_else(
        || {
            let mut v = vec!["ref".to_string()];
            if avx512::available() {
                v.extend(["avx512-dense", "avx512-sparse", "checked:avx512-sparse"].map(String::from));
            }
            v
        },
        |s| s.split(',').map(String::from).collect(),
    );
    engines.retain(|e| e == "ref" || avx512::available());
    let mut models: Vec<(String, Arc<Model>, String)> = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if a == "--model" {
            let p = &args[i + 1];
            models.push((p.clone(), Arc::new(load_model(p).expect("model")), sha256_hex(p)));
        }
    }
    if models.is_empty() || args.iter().any(|a| a == "--seed-model") {
        models.insert(0, ("initial(zero seed)".into(), Arc::new(Model::from_seed(&[0u8; 32])), String::new()));
    }
    let flat = if avx512::available() { Some(Arc::new(FlatDataset::new(&*ds).expect("flat dataset"))) } else { None };
    for (mname, model, mhash) in &models {
        for &t in &threads {
            let mut reference_checksum = None;
            for e in &engines {
                let _ = run(e, model, &ds, &flat, t, attempts.min(4096)); // warm-up
                let (mut rates, mut sum) = (Vec::new(), 0);
                for _ in 0..reps {
                    let (secs, c) = run(e, model, &ds, &flat, t, attempts);
                    rates.push(attempts as f64 / secs);
                    sum = c;
                }
                let valid = *reference_checksum.get_or_insert(sum) == sum;
                let best = rates.iter().cloned().fold(0.0, f64::max);
                let worst = rates.iter().cloned().fold(f64::MAX, f64::min);
                let mut sorted = rates.clone();
                println!(
                    "{{\"engine\":\"{e}\",\"threads\":{t},\"model\":\"{mname}\",\"model_blake3\":\"{mhash}\",\"weights_hash\":\"{}\",\"dataset\":\"{dataset_name}\",\"attempts\":{attempts},\"reps\":{reps},\"median_attempts_per_s\":{:.0},\"best\":{best:.0},\"worst\":{worst:.0},\"checksum\":\"{sum:016x}\",\"checksum_equal_to_first_engine\":{valid}}}",
                    model.weights_hash().iter().map(|b| format!("{b:02x}")).collect::<String>(),
                    median(&mut sorted)
                );
            }
        }
    }
}
