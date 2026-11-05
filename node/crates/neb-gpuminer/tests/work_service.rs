//! The GPU worker against a node on the easy network with `verify_pow` enabled, with a CPU scanner in place of the
//! device: the path of model upload, nonce ranges, CPU check, shares and solutions is the one that a CUDA device uses.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use neb_block::Target;
use neb_chain::demo;
use neb_gpuminer::{Scanner, flatten, spawn_worker, spawn_worker_with_fallback, unpack_weights};
use neb_node::params::NetworkParams;
use neb_node::style::{ColorMode, Live, Log};
use neb_node::{Node, NodeConfig};
use neb_pow::shares::{Dataset, Model, N1, N2, NIN, ShareBuffers, SharedDataset, SyntheticDataset};

/// Evaluates the attempts with the C reference on the weights that it was given, as a device would.
struct CpuScanner {
    dataset: SharedDataset,
    model: Option<Model>,
    uploads: Arc<Mutex<u32>>,
    buffers: ShareBuffers,
    /// Makes the device wrong: it reports the first nonce of a range as a hit although its ticket is above the
    /// bound, in every scan (`Always`) or only in scans of more than one nonce (`InBatches`).
    fault: Fault,
}

#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    Always,
    InBatches,
}

impl Scanner for CpuScanner {
    fn capacity(&self) -> u32 {
        64
    }

    fn set_model(&mut self, w1: &[i8], w2: &[i8]) -> Result<(), String> {
        assert_eq!((w1.len(), w2.len()), (N1, N2));
        // Pack the matrices again: the model that the worker was given must survive the unpacking.
        let nibbles: Vec<u8> = w1.iter().chain(w2).map(|&w| (w as u8) & 15).collect();
        let packed: Vec<u8> = nibbles.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
        self.model = Some(Model::from_packed_weights(&packed)?);
        *self.uploads.lock().unwrap() += 1;
        Ok(())
    }

    fn scan(&mut self, header: &[u8; 80], start: u32, count: u32, bound: &[u8; 32]) -> Result<Vec<u32>, String> {
        let model = self.model.as_ref().expect("a model was uploaded before the first scan");
        let target = Target(*bound);
        let mut header = *header;
        let mut hits = Vec::new();
        for nonce in start..start + count {
            header[76..80].copy_from_slice(&nonce.to_le_bytes());
            if target.admits(&self.buffers.ticket(model, &*self.dataset, blake3::hash(&header).as_bytes())) {
                hits.push(nonce);
            }
        }
        let wrong = match self.fault {
            Fault::None => false,
            Fault::Always => true,
            Fault::InBatches => count > 1,
        };
        if wrong && !hits.contains(&start) {
            hits.push(start);
        }
        Ok(hits)
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

#[test]
fn unpacking_returns_the_weights_of_the_model() {
    let model = Model::from_seed(&[7u8; 32]);
    let (w1, w2) = unpack_weights(&model.packed_weights());
    assert_eq!((w1.len(), w2.len()), (N1, N2));
    assert!(w1.iter().chain(&w2).all(|&w| (-8..=7).contains(&w)));
    let mut again = Vec::new();
    for pair in w1.iter().chain(&w2).collect::<Vec<_>>().chunks(2) {
        again.push(((*pair[0] as u8) & 15) | (((*pair[1] as u8) & 15) << 4));
    }
    assert_eq!(again, model.packed_weights());
}

#[test]
fn flatten_lists_pixels_then_labels() {
    let dataset = SyntheticDataset::new();
    let (pixels, labels) = flatten(&dataset);
    assert_eq!((pixels.len(), labels.len()), (dataset.n() as usize * NIN, dataset.n() as usize));
    let (x, label) = dataset.get(5);
    assert_eq!((&pixels[5 * NIN..6 * NIN], labels[5]), (&x[..], label));
}

#[test]
fn the_gpu_worker_finds_blocks_and_the_next_block_lists_its_shares() {
    let t0 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
    let dir = std::env::temp_dir().join(format!("neb-gpuminer-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let network = NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, 0x07],
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
    let uploads = Arc::new(Mutex::new(0));
    let mismatches = Arc::new(AtomicU64::new(0));
    let device_attempts = Arc::new(AtomicU64::new(0));
    let scanner = CpuScanner {
        dataset: dataset.clone(),
        model: None,
        uploads: uploads.clone(),
        buffers: ShareBuffers::new(),
        fault: Fault::None,
    };
    let miner = {
        let (dataset, mismatches, device_attempts) = (dataset.clone(), mismatches.clone(), device_attempts.clone());
        neb_cpuminer::start_with(work.addr, "test-gpu-miner", move |session| {
            vec![spawn_worker(session.clone(), dataset, Box::new(scanner), device_attempts, mismatches)]
        })
        .expect("miner")
    };
    assert!(wait_for(|| node.height() >= 6, 120), "the chain stopped at height {}", node.height());
    let stats = miner.stats().clone();
    assert!(wait_for(|| stats.accepted.load(Ordering::Relaxed) >= 4, 20));
    assert_eq!(stats.rejected.load(Ordering::Relaxed), 0, "no solution was refused");
    assert_eq!(mismatches.load(Ordering::Relaxed), 0, "the scanner and the CPU check agree");
    miner.stop();
    assert_eq!(
        device_attempts.load(Ordering::Relaxed),
        stats.attempts.load(Ordering::Relaxed),
        "one device, so its count is the total"
    );

    let height = node.height();
    let blocks: Vec<_> = (1..=height).map(|h| node.block_at(h).unwrap()).collect();
    let mut listed = 0;
    for pair in blocks.windows(2) {
        assert!(
            pair[1].shares.contains(&neb_chain::ShareRecord::of_header(&pair[0].header)),
            "the winner of a round is listed by the next block"
        );
        assert!(pair[1].shares.windows(2).all(|w| w[0] < w[1]));
        listed += pair[1].shares.len();
    }
    assert!(listed > blocks.len(), "{listed} shares in {} blocks", blocks.len());
    assert!(*uploads.lock().unwrap() as usize >= blocks.len(), "the model was uploaded for each block");
    drop(work);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

fn easy_node(magic_last: u8, name: &str) -> (Node, std::path::PathBuf) {
    let t0 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
    let dir = std::env::temp_dir().join(format!("neb-gpuminer-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let network = NetworkParams {
        name: "test",
        magic: [0xAB, 0xCD, 0xEF, magic_last],
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
    (Node::open(NodeConfig { network, data_dir: dir.clone(), verify_pow: true }).expect("node"), dir)
}

/// A device that returns a nonce the reference refuses is replaced by K0 (here a faithful scanner), the batch is
/// scanned again, and the node never sees a wrong share: no solution is refused and the chain grows.
fn a_faulty_kernel_is_replaced(fault: Fault, magic_last: u8, name: &str) {
    let (node, dir) = easy_node(magic_last, name);
    let live = Live::new();
    let log = Arc::new(Log::new(live.clone(), ColorMode::Never));
    node.set_miner_lock(demo::MINER_LOCK);
    let work = node.start_work_service("127.0.0.1:0".parse().unwrap(), log, live).expect("work service");
    let dataset: SharedDataset = Arc::new(SyntheticDataset::new());
    let uploads = Arc::new(Mutex::new(0));
    let (mismatches, fallbacks, device_attempts) =
        (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let faulty = CpuScanner {
        dataset: dataset.clone(),
        model: None,
        uploads: uploads.clone(),
        buffers: ShareBuffers::new(),
        fault,
    };
    let miner = {
        let (dataset, mismatches, fallbacks, device_attempts, uploads) =
            (dataset.clone(), mismatches.clone(), fallbacks.clone(), device_attempts.clone(), uploads.clone());
        neb_cpuminer::start_with(work.addr, "test-gpu-fault", move |session| {
            let clean = dataset.clone();
            let fallback: neb_gpuminer::Fallback = Box::new(move || {
                Ok(Box::new(CpuScanner {
                    dataset: clean,
                    model: None,
                    uploads,
                    buffers: ShareBuffers::new(),
                    fault: Fault::None,
                }) as Box<dyn Scanner>)
            });
            vec![spawn_worker_with_fallback(
                session.clone(),
                dataset,
                Box::new(faulty),
                Some(fallback),
                device_attempts,
                mismatches,
                fallbacks,
            )]
        })
        .expect("miner")
    };
    assert!(wait_for(|| node.height() >= 6, 120), "the chain stopped at height {}", node.height());
    let stats = miner.stats().clone();
    assert!(wait_for(|| stats.accepted.load(Ordering::Relaxed) >= 4, 20));
    assert_eq!(stats.rejected.load(Ordering::Relaxed), 0, "no solution was refused");
    assert_eq!(fallbacks.load(Ordering::Relaxed), 1, "the device went back to K0 once");
    miner.stop();
    drop(work);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_kernel_wrong_on_a_new_model_is_replaced_by_k0() {
    a_faulty_kernel_is_replaced(Fault::Always, 0x08, "always");
}

#[test]
fn a_kernel_that_returns_a_refused_nonce_in_a_batch_is_replaced_by_k0() {
    // the single-nonce probes of the model check pass; the batches do not
    a_faulty_kernel_is_replaced(Fault::InBatches, 0x09, "batches");
}
