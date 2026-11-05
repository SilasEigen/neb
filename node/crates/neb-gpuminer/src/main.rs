//! `neb-miner-gpu --node HOST:PORT (--dataset FILE | --synthetic) [--devices LIST] [--gpu-kernel k0|k1|k2] [--plain] [--stats FILE]`
//! and `neb-miner-gpu --selftest [--gpu-kernel k0|k1|k2] [--dataset FILE | --synthetic]`
//!
//! GPU miner of the work service (spec/node.md section 7.1). `--devices` is a comma-separated list of CUDA device
//! indexes (default: all). Shows the dashboard in a terminal; with `--plain`, or when the output is not a terminal,
//! prints a status line every 5 s. `--stats FILE` rewrites a JSON file with the rate of each device every 2 s (the HiveOS
//! package reads it). The dataset must be the one the node uses; the miner does not
//! check its hash. `--synthetic` is for demo networks only.
//!
//! `--gpu-kernel` chooses the kernel: `k0` (dp4a, the default) or the tensor-core kernels `k1` and `k2`, which need compute
//! capability 8.0 or later. A tensor-core kernel is used on a device only if the device supports it and passes the
//! self-test against the C reference; otherwise the miner says why and uses K0 on that device. The reference checks every
//! nonce before it is sent, the model of each job is checked on a few nonces, one batch in 64 is audited, and on any
//! disagreement the device goes back to K0. `--selftest` runs the self-test of the requested kernel on each device
//! (default: K0, K1 and K2, skipping the ones a device does not support) on the given dataset (default: the synthetic one).
//! Exit code: 0 for a passed self test or after `q`, 1 when a device fails, the node closes the connection or
//! cannot be reached, 2 on a usage error.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use neb_cpuminer::dashboard::{self, Hardware, Info, Outcome};
use neb_gpuminer::cuda::{self, CudaScanner};
use neb_gpuminer::{
    BATCH, Fallback, GpuKernel, Scanner, flatten, spawn_worker_with_fallback, stats, telemetry, unpack_weights,
};
use neb_pow::shares::{Dataset, FileDataset, Model, SharedDataset, SyntheticDataset};

/// Nonces a scan can report; a round has about 32 shares per block, so this is far above the need.
const MAX_HITS: u32 = 4096;

fn usage() -> i32 {
    eprintln!(
        "usage: neb-miner-gpu --node HOST:PORT (--dataset FILE | --synthetic) [--devices 0,1,...] [--gpu-kernel k0|k1|k2] [--plain] [--stats FILE]"
    );
    eprintln!(
        "       neb-miner-gpu --selftest [--gpu-kernel k0|k1|k2] [--dataset FILE | --synthetic] [--devices 0,1,...]"
    );
    2
}

fn main() {
    // CUDA numbers the devices fastest first by default; NVML and nvidia-smi use the order of the PCI bus.
    if std::env::var_os("CUDA_DEVICE_ORDER").is_none() {
        // SAFETY: at the start of `main`, before any other thread exists.
        unsafe { std::env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID") };
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&args));
}

fn parse_devices(list: &str) -> Option<Vec<usize>> {
    list.split(',').map(|d| d.trim().parse::<usize>().ok()).collect()
}

/// The kernel to use on a device: the requested one if the device supports it and it passes the self-test, else K0.
fn choose_kernel(device: usize, requested: GpuKernel, dataset: &dyn Dataset) -> GpuKernel {
    if requested == GpuKernel::K0 {
        return requested;
    }
    let name = cuda::device_name(device).unwrap_or_default();
    if !cuda::kernel_supported(device, requested) {
        let cc = cuda::device_capability(device).map_or("?".into(), |(a, b)| format!("{a}.{b}"));
        eprintln!(
            "neb-miner-gpu: device {device} ({name}): {} needs compute capability 8.0 or later, this device has {cc}; using K0",
            requested.name()
        );
        return GpuKernel::K0;
    }
    match cuda::kernel_selftest(device, requested, dataset) {
        Ok(summary) => {
            eprintln!("neb-miner-gpu: device {device} ({name}): {} self-test passed: {summary}", requested.name());
            requested
        }
        Err(e) => {
            eprintln!("neb-miner-gpu: device {device} ({name}): {} self-test FAILED: {e}; using K0", requested.name());
            GpuKernel::K0
        }
    }
}

type DeviceSetup = (CudaScanner, Option<Fallback>, GpuKernel);

fn open_all(
    devices: &[usize],
    dataset: &dyn Dataset,
    requested: GpuKernel,
    model: &Model,
) -> Result<Vec<DeviceSetup>, String> {
    let (pixels, labels) = flatten(dataset);
    let (pixels, labels) = (Arc::new(pixels), Arc::new(labels));
    let (w1, w2) = unpack_weights(&model.packed_weights());
    let (w1, w2) = (Arc::new(w1), Arc::new(w2));
    devices
        .iter()
        .map(|&d| {
            let kernel = choose_kernel(d, requested, dataset);
            let scanner = CudaScanner::open_kernel(d, kernel, &pixels, &labels, &w1, &w2, BATCH, MAX_HITS)?;
            let fallback: Option<Fallback> = (kernel != GpuKernel::K0).then(|| {
                let (pixels, labels, w1, w2) = (pixels.clone(), labels.clone(), w1.clone(), w2.clone());
                Box::new(move || {
                    CudaScanner::open(d, &pixels, &labels, &w1, &w2, BATCH, MAX_HITS)
                        .map(|s| Box::new(s) as Box<dyn Scanner>)
                }) as Fallback
            });
            Ok((scanner, fallback, kernel))
        })
        .collect()
}

fn run(args: &[String]) -> i32 {
    let (mut node, mut dataset_path, mut synthetic, mut devices, mut selftest, mut plain) =
        (None, None, false, None, false, false);
    let mut stats_path: Option<String> = None;
    let mut requested: Option<GpuKernel> = None;
    let mut i = 0;
    while i < args.len() {
        match (args[i].as_str(), args.get(i + 1)) {
            ("--node", Some(v)) => node = Some(v.clone()),
            ("--dataset", Some(v)) => dataset_path = Some(v.clone()),
            ("--devices", Some(v)) => devices = Some(v.clone()),
            ("--stats", Some(v)) => stats_path = Some(v.clone()),
            ("--gpu-kernel", Some(v)) => match GpuKernel::parse(v) {
                Ok(k) => requested = Some(k),
                Err(e) => {
                    eprintln!("{e}");
                    return usage();
                }
            },
            ("--synthetic", _) => {
                synthetic = true;
                i += 1;
                continue;
            }
            ("--selftest", _) => {
                selftest = true;
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
    let list = match devices.as_deref().map(parse_devices) {
        None => None,
        Some(Some(list)) => Some(list),
        Some(None) => return usage(),
    };
    let count = match cuda::device_count() {
        Ok(count) => count,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let devices = match list {
        None => (0..count).collect(),
        Some(list) => {
            if let Some(d) = list.iter().find(|&&d| d >= count) {
                eprintln!("device {d} does not exist ({count} found)");
                return 2;
            }
            list
        }
    };
    if devices.is_empty() {
        eprintln!("no CUDA device found");
        return 1;
    }
    let dataset: SharedDataset = match (synthetic || (selftest && dataset_path.is_none()), dataset_path) {
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
    if selftest {
        return self_test(&devices, requested, &*dataset);
    }
    let Some(node) = node else { return usage() };
    let scanners =
        match open_all(&devices, &*dataset, requested.unwrap_or(GpuKernel::K0), &Model::from_seed(&[0u8; 32])) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        };
    let mismatches = Arc::new(AtomicU64::new(0));
    let fallbacks = Arc::new(AtomicU64::new(0));
    let counters: Vec<Arc<AtomicU64>> = devices.iter().map(|_| Arc::new(AtomicU64::new(0))).collect();
    let kernels: Vec<GpuKernel> = scanners.iter().map(|s| s.2).collect();
    let miner = {
        let (dataset, mismatches, fallbacks, counters) =
            (dataset.clone(), mismatches.clone(), fallbacks.clone(), counters.clone());
        neb_cpuminer::start_with(node.as_str(), concat!("neb-miner-gpu ", env!("CARGO_PKG_VERSION")), move |session| {
            scanners
                .into_iter()
                .zip(counters)
                .map(|((scanner, fallback, _), attempts)| {
                    spawn_worker_with_fallback(
                        session.clone(),
                        dataset.clone(),
                        Box::new(scanner),
                        fallback,
                        attempts,
                        mismatches.clone(),
                        fallbacks.clone(),
                    )
                })
                .collect()
        })
    };
    let miner = match miner {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot connect to {node}: {e}");
            return 1;
        }
    };
    let names: Vec<String> = devices
        .iter()
        .zip(&kernels)
        .map(|(&d, k)| format!("{d}: {} [{}]", cuda::device_name(d).unwrap_or_default(), k.name()))
        .collect();
    let readings = telemetry::start(&devices, names.clone());
    let _stats_thread = stats_path.map(|path| {
        let bus_numbers = devices.iter().map(|&d| cuda::device_bus(d).unwrap_or(0)).collect();
        stats::spawn(miner.session(), path.into(), counters, bus_numbers, readings.clone())
    });
    if !plain && dashboard::available() {
        let info = Info { title: "GPU proof-of-work miner".into(), device: names.join(", "), node: node.clone() };
        let outcome =
            dashboard::run(&miner.session(), &info, &|| Hardware::Gpu(readings.lock().expect("readings").clone()));
        if outcome == Outcome::Disconnected {
            eprintln!("{}", miner.stats().disconnect_message());
        }
        miner.stop();
        return if outcome == Outcome::Quit { 0 } else { 1 };
    }
    println!("neb-miner-gpu: {}, node {node}", names.join(", "));
    let started = Instant::now();
    let (mut last, mut last_attempts) = (started, 0u64);
    while !miner.stopped() {
        std::thread::sleep(Duration::from_secs(5));
        let stats = miner.stats();
        let attempts = stats.attempts.load(Ordering::Relaxed);
        let now = Instant::now();
        let rate = (attempts - last_attempts) as f64 / now.duration_since(last).as_secs_f64();
        println!(
            "{:>5}s  {:>9.0} attempts/s  shares {}  blocks {} (accepted {}, rejected {})",
            started.elapsed().as_secs(),
            rate,
            stats.shares.load(Ordering::Relaxed),
            stats.blocks.load(Ordering::Relaxed),
            stats.accepted.load(Ordering::Relaxed),
            stats.rejected.load(Ordering::Relaxed),
        );
        let bad = mismatches.load(Ordering::Relaxed);
        if bad > 0 {
            eprintln!("warning: {bad} nonces of the device were refused by the CPU check");
        }
        let back = fallbacks.load(Ordering::Relaxed);
        if back > 0 {
            eprintln!("warning: {back} device(s) went back to K0 after a disagreement with the reference");
        }
        (last, last_attempts) = (now, attempts);
    }
    if miner.stats().jobs.load(Ordering::Relaxed) == 0 {
        eprintln!("{}", miner.stats().disconnect_message());
    } else {
        eprintln!("the node closed the connection or a device failed");
    }
    miner.stop();
    1
}

/// For each device and each kernel (all three, or the requested one): the self-test of `cuda::kernel_selftest`.
fn self_test(devices: &[usize], requested: Option<GpuKernel>, dataset: &dyn Dataset) -> i32 {
    let kernels = requested.map_or(vec![GpuKernel::K0, GpuKernel::K1, GpuKernel::K2], |k| vec![k]);
    let mut failed = false;
    for &device in devices {
        let name = cuda::device_name(device).unwrap_or_default();
        let cc = cuda::device_capability(device).map_or("?".into(), |(a, b)| format!("{a}.{b}"));
        for &kernel in &kernels {
            if !cuda::kernel_supported(device, kernel) {
                println!(
                    "device {device} {name} (compute capability {cc}): {} is not supported on this device{}",
                    kernel.name(),
                    if requested.is_some() { ": FAILED" } else { " (skipped)" }
                );
                failed |= requested.is_some();
                continue;
            }
            match cuda::kernel_selftest(device, kernel, dataset) {
                Ok(summary) => {
                    println!("device {device} {name} (compute capability {cc}): {}: {summary}: PASSED", kernel.name())
                }
                Err(e) => {
                    println!(
                        "device {device} {name} (compute capability {cc}): {}: DIFFERENT from the C reference: {e}",
                        kernel.name()
                    );
                    failed = true;
                }
            }
        }
    }
    println!("{}", if failed { "SELFTEST FAILED" } else { "SELFTEST PASSED" });
    i32::from(failed)
}
