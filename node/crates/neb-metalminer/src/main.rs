//! `neb-miner-metal`: the NEB miner on an Apple GPU.
//!
//! ```text
//! neb-miner-metal --node 127.0.0.1:3441 --dataset neb-datasets.bin
//! neb-miner-metal --dataset neb-datasets.bin --selftest      # GPU against the C reference, every slot
//! neb-miner-metal --dataset neb-datasets.bin --bench 20000   # attempts per second, slot 0
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use neb_metalminer::MetalEngine;
use neb_pow::shares::{Arch, Dataset, Model, SharedDataset, SyntheticDataset, load_datasets, share_eval};

fn usage() -> String {
    "usage: neb-miner-metal --dataset FILE [--node HOST:PORT] [--batch N] [--name TEXT]\n\
     \x20      neb-miner-metal --dataset FILE --selftest [--seeds N] [--keys N]\n\
     \x20      neb-miner-metal --dataset FILE --bench ATTEMPTS\n\
     \x20      --syn replaces --dataset with the synthetic test dataset of spec/shares.md section 9"
        .to_string()
}

struct Args {
    node: String,
    dataset: Option<PathBuf>,
    batch: u32,
    name: String,
    selftest: bool,
    bench: Option<u32>,
    seeds: u32,
    keys: u32,
    syn: bool,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        node: "127.0.0.1:3441".into(),
        dataset: None,
        batch: 2048,
        name: "metal".into(),
        selftest: false,
        bench: None,
        seeds: 4,
        keys: 32,
        syn: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |what: &str| it.next().ok_or_else(|| format!("{what} needs a value"));
        match flag.as_str() {
            "--node" => a.node = value("--node")?,
            "--dataset" => a.dataset = Some(PathBuf::from(value("--dataset")?)),
            "--batch" => a.batch = value("--batch")?.parse().map_err(|_| "--batch must be a number".to_string())?,
            "--name" => a.name = value("--name")?,
            "--selftest" => a.selftest = true,
            "--bench" => a.bench = Some(value("--bench")?.parse().map_err(|_| "--bench must be a number".to_string())?),
            "--seeds" => a.seeds = value("--seeds")?.parse().map_err(|_| "--seeds must be a number".to_string())?,
            "--keys" => a.keys = value("--keys")?.parse().map_err(|_| "--keys must be a number".to_string())?,
            "--syn" => a.syn = true,
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown option {other:?}\n{}", usage())),
        }
    }
    if !a.syn && a.dataset.is_none() {
        return Err(usage());
    }
    Ok(a)
}

/// One dataset per model slot, in slot order: a dataset set file (`NMDS`) gives one per slot, a single dataset file one.
fn load_dataset(a: &Args) -> Result<Vec<SharedDataset>, String> {
    if a.syn {
        return Ok(vec![Arc::new(SyntheticDataset::new())]);
    }
    let path = a.dataset.clone().expect("checked by parse");
    load_datasets(&path, None)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}

fn selftest(a: &Args, datasets: &[SharedDataset]) -> Result<(), String> {
    for (slot, dataset) in datasets.iter().enumerate() {
        self_test_slot(a, slot, dataset)?;
    }
    Ok(())
}

fn self_test_slot(a: &Args, slot: usize, dataset: &Arc<dyn Dataset>) -> Result<(), String> {
    if dataset.arch() != Arch::V1 {
        // The kernel is the 784-128-10 attempt: a slot of another architecture is mined by another engine, if any.
        println!("slot {slot}: {}, not the 784-128-10 kernel, skipped", dataset.arch().name());
        return Ok(());
    }
    let mut engine = MetalEngine::new(&**dataset, a.batch)?;
    println!("device: {}", engine.device_name());
    println!(
        "slot {slot}: {} samples, {} models x {} keys",
        dataset.n(),
        a.seeds,
        a.keys
    );
    let mut checked = 0usize;
    for seed in 0..a.seeds {
        let mut s = [0u8; 32];
        s[..4].copy_from_slice(&seed.to_le_bytes());
        let model = Model::from_seed_arch(dataset.arch(), &s);
        engine.set_model(&model)?;
        for k in 0..a.keys {
            let mut input = b"neb-metal-selftest".to_vec();
            input.extend_from_slice(&s);
            input.extend_from_slice(&k.to_le_bytes());
            let key = *blake3::hash(&input).as_bytes();
            let got = engine.scan_key(&key)?;
            let want = share_eval(&model, &**dataset, &key);
            if got.digest_bytes() != want.digest {
                return Err(format!(
                    "model {seed} key {k}: digest {} != reference {}",
                    hex(&got.digest_bytes()),
                    hex(&want.digest)
                ));
            }
            if got.ticket_bytes() != want.ticket {
                return Err(format!(
                    "model {seed} key {k}: ticket {} != reference {}",
                    hex(&got.ticket_bytes()),
                    hex(&want.ticket)
                ));
            }
            if got.indices != want.indices {
                return Err(format!("model {seed} key {k}: indices {:?} != reference {:?}", got.indices, want.indices));
            }
            checked += 1;
        }
    }
    println!("self-test passed: {checked} attempts equal to the C reference (digest, ticket and sample indices)");
    Ok(())
}

fn bench(a: &Args, datasets: &[SharedDataset], total: u32) -> Result<(), String> {
    let dataset = datasets.iter().find(|d| d.arch() == Arch::V1).ok_or("no 784-128-10 dataset to benchmark")?;
    let mut engine = MetalEngine::new(&**dataset, a.batch)?;
    if datasets.len() > 1 {
        println!("slot 0 of {} (the benchmark runs one slot)", datasets.len());
    }
    let model = Model::from_seed_arch(dataset.arch(), &[0u8; 32]);
    engine.set_model(&model)?;
    let mut header = [0u8; 80];
    header[..32].copy_from_slice(&model.weights_hash());
    let batch = engine.max_batch();
    // Warm-up: the first dispatch compiles the pipeline and touches the buffers.
    engine.scan(&header, 0, batch.min(64))?;
    let started = Instant::now();
    let mut done = 0u32;
    let mut nonce = 0u32;
    while done < total {
        let count = batch.min(total - done);
        engine.scan(&header, nonce, count)?;
        nonce = nonce.wrapping_add(count);
        done += count;
        if nonce < count {
            nonce = 0; // the nonce space wrapped; the batch is a benchmark, not a job
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "device: {}\nbatch: {batch}\n{done} attempts in {seconds:.3} s: {:.0} attempts/s",
        engine.device_name(),
        done as f64 / seconds
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::from(2);
        }
    };
    let datasets = match load_dataset(&a) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let result = if a.selftest {
        selftest(&a, &datasets)
    } else if let Some(total) = a.bench {
        bench(&a, &datasets, total)
    } else {
        match neb_metalminer::start(&a.node, datasets, &a.name, a.batch) {
            Ok(handle) => {
                println!("neb-miner-metal: connected to {}, press Ctrl-C to stop", a.node);
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(60));
                    if handle.stopped() {
                        break;
                    }
                }
                let message = handle.stats().disconnect_message();
                handle.stop();
                Err(message.to_string())
            }
            Err(e) => Err(e.to_string()),
        }
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
