//! Throughput of the CUDA kernels on this machine, as JSON lines (feature `cuda`).
//!
//! `cargo run --release --features cuda -p neb-gpuminer --example bench_gpu -- [options]`
//!   --kernels LIST   comma list of k0, k1, k2 (default: the kernels the device supports)
//!   --batch N        nonces per scan (default 65536)
//!   --scans N        scans per repetition (default 16)
//!   --reps N         repetitions after one warm-up (default 7)
//!   --dataset FILE   network dataset file (default: the synthetic dataset)
//!   --model FILE     a raw model dump (508160 bytes: W1, W2, R1, R2); repeatable. Default: the initial model.
//!   --device N       CUDA device (default 0)
//! Every kernel scans the same nonces of the same header on the same model with a bound that keeps about 1.5 % of them;
//! `hits` and `xor` (the xor of the nonces that hit) must be equal across kernels, otherwise the run is invalid.

use std::sync::Arc;
use std::time::Instant;

use neb_gpuminer::cuda::{self, CudaScanner};
use neb_gpuminer::{GpuKernel, Scanner, flatten, unpack_weights};
use neb_pow::shares::{FileDataset, Model, N1, N2, SharedDataset, SyntheticDataset};

fn load_model(path: &str) -> Result<Model, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() != (N1 + N2) * 5 {
        return Err(format!("{path}: {} bytes, expected {}", b.len(), (N1 + N2) * 5));
    }
    let w: Vec<i8> = b[..N1 + N2].iter().map(|&x| x as i8).collect();
    let r: Vec<i32> = b[N1 + N2..].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    Model::from_state(&w[..N1], &w[N1..], &r[..N1], &r[N1..])
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let device: usize = get("--device").map_or(0, |s| s.parse().unwrap());
    let batch: u32 = get("--batch").map_or(65_536, |s| s.parse().unwrap());
    let scans: u32 = get("--scans").map_or(16, |s| s.parse().unwrap());
    let reps: usize = get("--reps").map_or(7, |s| s.parse().unwrap());
    let dataset_path = get("--dataset");
    let ds: SharedDataset = match &dataset_path {
        Some(p) => Arc::new(FileDataset::load(std::path::Path::new(p), None).expect("dataset")),
        None => Arc::new(SyntheticDataset::new()),
    };
    let dataset_name = dataset_path.clone().unwrap_or_else(|| "synthetic".into());
    let kernels: Vec<GpuKernel> = get("--kernels").map_or_else(
        || {
            [GpuKernel::K0, GpuKernel::K1, GpuKernel::K2]
                .into_iter()
                .filter(|&k| cuda::kernel_supported(device, k))
                .collect()
        },
        |s| s.split(',').map(|k| GpuKernel::parse(k).expect("kernel")).collect(),
    );
    let mut models: Vec<(String, Model)> = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if a == "--model" {
            models.push((args[i + 1].clone(), load_model(&args[i + 1]).expect("model")));
        }
    }
    if models.is_empty() {
        models.push(("initial(zero seed)".into(), Model::from_seed(&[0u8; 32])));
    }
    let (px, lab) = flatten(&*ds);
    let name = cuda::device_name(device).unwrap_or_default();
    let (major, minor) = cuda::device_capability(device).unwrap_or((0, 0));
    let mut header = [0u8; 80];
    for (i, b) in header.iter_mut().enumerate().take(76) {
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    let mut bound = [0xFFu8; 32];
    bound[31] = 0x03;
    for (mname, model) in &models {
        let (w1, w2) = unpack_weights(&model.packed_weights());
        let mut reference: Option<(usize, u32)> = None;
        for &k in &kernels {
            let mut s = CudaScanner::open_kernel(device, k, &px, &lab, &w1, &w2, batch, 8192).expect("open");
            let run = |s: &mut CudaScanner| -> (f64, usize, u32) {
                let t0 = Instant::now();
                let (mut hits, mut x) = (0usize, 0u32);
                for i in 0..scans {
                    let h = s.scan(&header, i.wrapping_mul(batch), batch, &bound).expect("scan");
                    hits += h.len();
                    x ^= h.iter().fold(0, |a, n| a ^ n);
                }
                (t0.elapsed().as_secs_f64(), hits, x)
            };
            let _ = run(&mut s); // warm-up
            let mut rates = Vec::new();
            let (mut hits, mut x) = (0, 0);
            for _ in 0..reps {
                let (secs, h, xx) = run(&mut s);
                rates.push(scans as f64 * batch as f64 / secs);
                (hits, x) = (h, xx);
            }
            let valid = *reference.get_or_insert((hits, x)) == (hits, x);
            rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{{\"kernel\":\"{}\",\"device_compute_capability\":\"{major}.{minor}\",\"model\":\"{mname}\",\"weights_hash\":\"{}\",\"dataset\":\"{dataset_name}\",\"batch\":{batch},\"scans\":{scans},\"reps\":{reps},\"median_attempts_per_s\":{:.0},\"best\":{:.0},\"worst\":{:.0},\"hits\":{hits},\"xor\":\"{x:08x}\",\"equal_to_first_kernel\":{valid},\"device\":\"{}\"}}",
                k.name(),
                model.weights_hash().iter().map(|b| format!("{b:02x}")).collect::<String>(),
                rates[rates.len() / 2],
                rates[rates.len() - 1],
                rates[0],
                name
            );
        }
    }
}
