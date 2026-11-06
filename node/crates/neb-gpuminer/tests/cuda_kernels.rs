//! The CUDA kernels on a real device (feature `cuda`): self-test of every kernel, complete reports, the model upload
//! from several threads at once, and the GPU worker against a verifying node. Without a CUDA device the tests print
//! SKIPPED; with `NEB_REQUIRE_CUDA=1` that is a failure. K1 and K2 are skipped on devices below compute capability 8.0.
#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use neb_block::Target;
use neb_chain::demo;
use neb_gpuminer::cuda::{self, CudaScanner};
use neb_gpuminer::{GpuKernel, Scanner, flatten, spawn_worker_with_fallback, unpack_weights};
use neb_node::params::NetworkParams;
use neb_node::style::{ColorMode, Live, Log};
use neb_node::{Node, NodeConfig};
use neb_pow::shares::{Model, ShareBuffers, SharedDataset, SyntheticDataset};

fn devices() -> Vec<usize> {
    match cuda::device_count() {
        Ok(n) if n > 0 => (0..n).collect(),
        _ => {
            assert!(
                std::env::var_os("NEB_REQUIRE_CUDA").is_none(),
                "NEB_REQUIRE_CUDA is set but there is no CUDA device"
            );
            eprintln!("SKIPPED: no CUDA device");
            Vec::new()
        }
    }
}

fn kernels(device: usize) -> Vec<GpuKernel> {
    [GpuKernel::K0, GpuKernel::K1, GpuKernel::K2].into_iter().filter(|&k| cuda::kernel_supported(device, k)).collect()
}

fn header() -> [u8; 80] {
    let mut h = [0u8; 80];
    for (i, b) in h.iter_mut().enumerate().take(76) {
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    h
}

#[test]
fn every_supported_kernel_passes_the_self_test_on_every_device() {
    let ds = SyntheticDataset::new();
    for d in devices() {
        let ks = kernels(d);
        assert!(ks.contains(&GpuKernel::K0));
        for k in ks {
            let summary = cuda::kernel_selftest(d, k, &ds).unwrap_or_else(|e| panic!("device {d}, {}: {e}", k.name()));
            eprintln!("device {d} {}: {summary}", k.name());
        }
    }
}

/// A device with room for 64 hits scans 512 nonces whose tickets all meet the bound: every hit must come back.
#[test]
fn a_report_that_does_not_fit_is_split_not_lost() {
    let ds = SyntheticDataset::new();
    let model = Model::from_seed(&[0u8; 32]);
    let (px, lab) = flatten(&ds);
    let (w1, w2) = unpack_weights(&model.packed_weights());
    for d in devices() {
        for k in kernels(d) {
            let mut s = CudaScanner::open_kernel(d, k, &px, &lab, &w1, &w2, 512, 64).unwrap();
            let mut got = s.scan(&header(), 1000, 512, &[0xFF; 32]).unwrap();
            got.sort_unstable();
            assert_eq!(got, (1000..1512).collect::<Vec<u32>>(), "device {d}, {}", k.name());
        }
    }
}

/// Several threads upload alternating models to their own scanner on the same device at the same time and check the
/// hits after each upload. A staging buffer of the model shared by all callers would let the packed weights of one
/// thread be overwritten by another before they were copied to the device; the buffer is local to each call.
#[test]
fn models_uploaded_from_several_threads_do_not_corrupt_each_other() {
    let ds = SyntheticDataset::new();
    let models = [Model::from_seed(&[0u8; 32]), Model::from_seed(&[5u8; 32])];
    let (px, lab) = flatten(&ds);
    let mut bound = [0xFFu8; 32];
    bound[31] = 0x80;
    let target = Target(bound);
    let expected: Vec<Vec<u32>> = models
        .iter()
        .map(|m| {
            let mut b = ShareBuffers::new();
            let mut h = header();
            (0..96u32)
                .filter(|&n| {
                    h[76..80].copy_from_slice(&n.to_le_bytes());
                    target.admits(&b.ticket(m, &ds, blake3::hash(&h).as_bytes()))
                })
                .collect()
        })
        .collect();
    assert!(expected[0] != expected[1]);
    let uploaded: Vec<_> = models.iter().map(|m| unpack_weights(&m.packed_weights())).collect();
    for d in devices() {
        for k in kernels(d) {
            let failures = AtomicU64::new(0);
            std::thread::scope(|s| {
                for t in 0..8usize {
                    let (px, lab, uploaded, expected, failures) = (&px, &lab, &uploaded, &expected, &failures);
                    s.spawn(move || {
                        let (w1, w2) = &uploaded[t % 2];
                        let mut scanner = CudaScanner::open_kernel(d, k, px, lab, w1, w2, 128, 4096).unwrap();
                        for i in 0..60usize {
                            let which = (t + i) % 2;
                            let (w1, w2) = &uploaded[which];
                            scanner.set_model(w1, w2).unwrap();
                            let mut got = scanner.scan(&header(), 0, 96, &bound).unwrap();
                            got.sort_unstable();
                            if got != expected[which] {
                                failures.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    });
                }
            });
            assert_eq!(
                failures.load(Ordering::Relaxed),
                0,
                "device {d}, {}: a scan used a model other than the one just uploaded",
                k.name()
            );
        }
    }
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

/// The GPU worker with each kernel against a node that checks every proof of work and every share. On the easy network
/// one attempt in two is a share. The batches are 256 nonces (about 128 hits, each checked by the reference) and a scan
/// holds 32 hits, so every report is cut and split.
#[test]
fn the_gpu_worker_mines_blocks_that_the_node_accepts_with_every_kernel() {
    for d in devices() {
        for (n, k) in kernels(d).into_iter().enumerate() {
            let t0 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
            let dir = std::env::temp_dir().join(format!("neb-gpu-kernels-{}-{d}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let network = NetworkParams {
                name: "test",
                magic: [0xAB, 0xCD, 0xEF, 0x10 + n as u8],
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
            let dataset: SharedDataset = Arc::new(SyntheticDataset::new());
            let (px, lab) = flatten(&*dataset);
            let (w1, w2) = unpack_weights(&Model::from_seed(&[0u8; 32]).packed_weights());
            let scanner = CudaScanner::open_kernel(d, k, &px, &lab, &w1, &w2, 256, 32).unwrap();
            let (mismatches, fallbacks, attempts) =
                (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
            let miner = {
                let (dataset, mismatches, fallbacks, attempts) =
                    (dataset.clone(), mismatches.clone(), fallbacks.clone(), attempts.clone());
                let (px, lab, w1, w2) = (Arc::new(px), Arc::new(lab), Arc::new(w1), Arc::new(w2));
                neb_cpuminer::start_with(work.addr, "test-gpu-kernel", move |session| {
                    let fallback: Option<neb_gpuminer::Fallback> = (k != GpuKernel::K0).then(|| {
                        Box::new(move || {
                            CudaScanner::open(d, &px, &lab, &w1, &w2, 256, 32).map(|s| Box::new(s) as Box<dyn Scanner>)
                        }) as neb_gpuminer::Fallback
                    });
                    let lane = neb_gpuminer::Lane { dataset, scanner: Box::new(scanner), fallback };
                    vec![spawn_worker_with_fallback(session.clone(), vec![lane], attempts, mismatches, fallbacks)]
                })
                .expect("miner")
            };
            assert!(
                wait_for(|| node.height() >= 5, 120),
                "{}: the chain stopped at height {}",
                k.name(),
                node.height()
            );
            let stats = miner.stats().clone();
            assert!(wait_for(|| stats.accepted.load(Ordering::Relaxed) >= 3, 20));
            assert_eq!(stats.rejected.load(Ordering::Relaxed), 0, "{}: no solution was refused", k.name());
            assert_eq!(
                mismatches.load(Ordering::Relaxed),
                0,
                "{}: the reference refused no nonce of the device",
                k.name()
            );
            assert_eq!(fallbacks.load(Ordering::Relaxed), 0, "{}: the kernel was never replaced", k.name());
            miner.stop();
            drop(work);
            drop(node);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
