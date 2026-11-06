//! `chain_fixture gen-stress OUT` writes the synthetic stress chain used by the tests (steps of 1 to 256
//! shares around the thresholds of the residual bound, mined with the C reference on the synthetic dataset).
//! `chain_fixture gen-pools OUT NAME BLOCKS POOLS [DATASET_FILE HASH_HEX]` writes a chain mined by several pools in lockstep
//! (the share list of a block holds records of the headers of every pool, spec/shares.md section 11.2), on the synthetic
//! dataset or on the dataset file with the given BLAKE3 hash (a set file of K datasets gives a chain of K models).
//! `chain_fixture replay FIXTURE [--dataset FILE]` replays a fixture with the node's own functions and prints
//! a summary; a fixture on the MNIST dataset needs `--dataset FILE` (the file must have the pinned hash).
//! Run from the repository root: `cargo run --release -p neb-chain --example chain_fixture -- replay pow/tests/chains/synthetic-stress.txt`.

use std::sync::Arc;

use neb_chain::demo;
use neb_chain::fixture::{ChainFixture, generate_pools, generate_stress, replay};
use neb_pow::shares::{SharedDataset, SyntheticDataset, load_datasets};

/// Steps of the stress chain, from block 2: the sizes around 16 (layer 2) and 128 (layer 1), the extremes 1
/// and 256, and sizes in between.
pub const STRESS_SIZES: [usize; 40] = [
    3, 16, 17, 1, 32, 64, 127, 128, 129, 256, 255, 2, 100, 17, 16, 48, 200, 129, 128, 5, 256, 33, 64, 16, 17, 129, 256,
    8, 24, 160, 12, 96, 17, 128, 129, 20, 40, 256, 16, 1,
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("gen-stress") if args.len() == 2 => gen_stress(&args[1]),
        Some("gen-pools") if args.len() == 5 || args.len() == 7 => gen_pools(&args[1..]),
        Some("replay") if args.len() >= 2 => {
            replay_file(&args[1], args.get(2).map(String::as_str) == Some("--dataset"), args.get(3))
        }
        _ => {
            eprintln!(
                "usage: chain_fixture gen-stress OUT | gen-pools OUT NAME BLOCKS POOLS [DATASET_FILE HASH_HEX] | replay FIXTURE [--dataset FILE]"
            );
            2
        }
    };
    std::process::exit(code);
}

fn params_with(datasets: Vec<SharedDataset>) -> neb_chain::ChainParams {
    let mut p = demo::easy_params(1_793_318_400, demo::PROJECT_LOCK);
    p.datasets = datasets;
    p
}

fn gen_stress(out: &str) -> i32 {
    let params = params_with(vec![Arc::new(SyntheticDataset::new())]);
    match generate_stress(
        "synthetic-stress",
        "synthetic",
        "neb-synthetic-stress",
        &params,
        demo::EASY_LIMIT_BITS,
        &STRESS_SIZES,
        0x5EED_0001,
    ) {
        Ok(fx) => {
            std::fs::write(out, fx.to_text()).expect("write the fixture");
            println!("wrote {out}: {} blocks", fx.blocks.len());
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn gen_pools(args: &[String]) -> i32 {
    let (out, name) = (&args[0], &args[1]);
    let (Ok(blocks), Ok(pools)) = (args[2].parse::<u32>(), args[3].parse::<u32>()) else {
        eprintln!("BLOCKS and POOLS are numbers");
        return 2;
    };
    let (ds, tag): (Vec<SharedDataset>, String) = match args.get(4..) {
        Some([file, hash]) => {
            let mut want = [0u8; 32];
            for (i, o) in want.iter_mut().enumerate() {
                *o = u8::from_str_radix(&hash[2 * i..2 * i + 2], 16).expect("a 64-digit hex hash");
            }
            (
                load_datasets(std::path::Path::new(file), Some(&want)).expect("the dataset file with this hash"),
                format!("file {hash}"),
            )
        }
        _ => (vec![Arc::new(SyntheticDataset::new())], "synthetic".to_string()),
    };
    let label = format!("neb-pools-{name}");
    match generate_pools(name, &tag, &label, &params_with(ds), demo::EASY_BITS, blocks, pools) {
        Ok(fx) => {
            std::fs::write(out, fx.to_text()).expect("write the fixture");
            println!("wrote {out}: {} blocks, {pools} pools", fx.blocks.len());
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn replay_file(path: &str, has_dataset: bool, dataset_path: Option<&String>) -> i32 {
    let text = std::fs::read_to_string(path).expect("read the fixture");
    let fx = match ChainFixture::parse(&text) {
        Ok(fx) => fx,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 1;
        }
    };
    let ds: Vec<SharedDataset> = if fx.dataset == "synthetic" {
        (0..fx.slots).map(|_| Arc::new(SyntheticDataset::new()) as SharedDataset).collect()
    } else if let (true, Some(file)) = (has_dataset, dataset_path) {
        let want = fx.dataset.strip_prefix("file ").and_then(|h| {
            let mut out = [0u8; 32];
            (h.len() == 64).then_some(())?;
            for (i, o) in out.iter_mut().enumerate() {
                *o = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).ok()?;
            }
            Some(out)
        });
        match load_datasets(std::path::Path::new(file), want.as_ref()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    } else {
        eprintln!("{path} needs --dataset FILE ({})", fx.dataset);
        return 2;
    };
    let t0 = std::time::Instant::now();
    match replay(&fx, &params_with(ds)) {
        Ok(r) => {
            let sum = |k: usize, f: fn(&neb_chain::fixture::LayerLoss) -> u64| {
                r.losses.iter().map(|(_, l)| f(&l[k])).sum::<u64>()
            };
            let hex = |b: &[u8; 32]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
            println!(
                "REPLAY OK {}: blocks {} steps {} shares {} largest_step {} seconds {:.1}",
                fx.name,
                r.blocks,
                r.steps,
                r.shares,
                r.largest_step,
                t0.elapsed().as_secs_f64()
            );
            for k in 0..2 {
                println!(
                    "  layer {}: cut_events {} clip_events {} lr_abs {} wc_abs {}",
                    k + 1,
                    sum(k, |l| l.cut_events),
                    sum(k, |l| l.clip_events),
                    sum(k, |l| l.lr_abs),
                    sum(k, |l| l.wc_abs)
                );
            }
            println!(
                "  final weights_hash {} residual_hash {}",
                hex(&r.final_weights_hash),
                hex(&r.final_residual_hash)
            );
            0
        }
        Err(e) => {
            println!("REPLAY FAILED {}: {e}", fx.name);
            1
        }
    }
}
