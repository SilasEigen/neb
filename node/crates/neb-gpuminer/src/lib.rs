//! GPU miner client of the work service (spec/node.md section 7.1, spec/shares.md section 11.5).
//!
//! The session is that of the CPU miner (`neb_cpuminer`): the same frames, the same reference header, the same
//! rules for shares and solutions. Only the attempts differ. A worker takes ranges of nonces, has a [`Scanner`] (a
//! CUDA device) evaluate them and keep the nonces whose ticket meets the share target, and evaluates those few
//! nonces again on the CPU with the C reference before it reports them. A device that returns a wrong nonce costs
//! a counted mismatch and nothing else, and a nonce it misses is only a lost share.
//!
//! Kernels ([`GpuKernel`]): K0 (dp4a, the default) and the tensor-core kernels K1 and K2 (compute capability 8.0 or
//! later), which the caller asks for by name. Like the AVX-512 engine of the CPU miner they are used only after a
//! self-test against the reference ([`check_scanner`]); the worker checks the model it uploads and audits single
//! nonces during the run ([`probe_ticket`]), and on any disagreement it replaces the kernel by K0
//! ([`spawn_worker_with_fallback`]). A scan never returns a partial result: a range with more hits than the device
//! can report is split ([`scan_complete`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use neb_cpuminer::Session;
use neb_pow::shares::{Dataset, Model, N1, N2, NIN, ShareBuffers};

/// The kernel that evaluates the attempts on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuKernel {
    /// dp4a, one thread block per attempt (`pow/gpu/shares_cuda.cu`): the default.
    K0,
    /// Tensor cores, every entry computed (`pow/gpu/shares_tc.cu`).
    K1,
    /// Tensor cores, skipping the zero entries.
    K2,
}

impl GpuKernel {
    pub fn parse(s: &str) -> Result<GpuKernel, String> {
        match s {
            "k0" | "K0" => Ok(GpuKernel::K0),
            "k1" | "K1" => Ok(GpuKernel::K1),
            "k2" | "K2" => Ok(GpuKernel::K2),
            _ => Err(format!("unknown kernel {s:?}: expected k0, k1 or k2")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GpuKernel::K0 => "K0",
            GpuKernel::K1 => "K1",
            GpuKernel::K2 => "K2",
        }
    }

    /// The code of `neb_scan_open_kernel` (`pow/gpu/scan_cuda.h`).
    pub fn code(self) -> i32 {
        self as i32
    }
}

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod stats;
pub mod telemetry;

/// Nonces of the reference header that a scan covers at most.
pub const BATCH: u32 = 1 << 16;

/// Evaluates attempts on a device.
pub trait Scanner: Send {
    /// Most nonces one scan can cover.
    fn capacity(&self) -> u32;
    /// Sets the model the next scans are evaluated on: `w1` is 128 x 784 and `w2` 10 x 128, row-major.
    fn set_model(&mut self, w1: &[i8], w2: &[i8]) -> Result<(), String>;
    /// All the nonces in `start..start + count` whose ticket on `header` with that nonce is not above `bound`, a
    /// 256-bit little-endian value, in no particular order. A result is never partial: a scanner that cannot report
    /// every hit of a range (see [`scan_complete`]) splits the range or fails.
    fn scan(&mut self, header: &[u8; 80], start: u32, count: u32, bound: &[u8; 32]) -> Result<Vec<u32>, String>;
}

/// Scans `start..start + count` with `run`, which returns the hits it can report and the number of hits there are,
/// and splits the range in halves while the report is cut (fewer hits reported than found). The hits of a nonce do
/// not depend on the range, so the union of the halves is exact. A device with room for `max_hits` hits reports
/// everything once the ranges are small enough; a range of one nonce has at most one hit.
pub fn scan_complete<F>(start: u32, count: u32, run: &mut F) -> Result<Vec<u32>, String>
where
    F: FnMut(u32, u32) -> Result<(Vec<u32>, usize), String>,
{
    let (hits, found) = run(start, count)?;
    if hits.len() >= found {
        return Ok(hits);
    }
    if count <= 1 {
        return Err(format!("the device reports {found} hits for {count} nonce"));
    }
    let half = count / 2;
    let second = start.checked_add(half).ok_or("the nonce range passes the end of the nonce space")?;
    let mut all = scan_complete(start, half, run)?;
    all.extend(scan_complete(second, count - half, run)?);
    Ok(all)
}

/// `value - 1` for a 256-bit little-endian value; `None` for 0.
fn minus_one(value: &[u8; 32]) -> Option<[u8; 32]> {
    let mut out = *value;
    for b in out.iter_mut() {
        if *b == 0 {
            *b = 0xFF;
        } else {
            *b -= 1;
            return Some(out);
        }
    }
    None
}

/// True when the ticket of `nonce` on `header` is exactly `reference` on the device: a one-nonce scan with the
/// bound `reference` must hit, and one with the bound `reference - 1` must not.
pub fn probe_ticket(
    scanner: &mut dyn Scanner,
    header: &[u8; 80],
    nonce: u32,
    reference: &[u8; 32],
) -> Result<bool, String> {
    if scanner.scan(header, nonce, 1, reference)? != [nonce] {
        return Ok(false);
    }
    Ok(match minus_one(reference) {
        None => true,
        Some(below) => scanner.scan(header, nonce, 1, &below)?.is_empty(),
    })
}

/// Nonces whose ticket the worker checks exactly (see [`probe_ticket`]) each time it uploads a model.
pub const MODEL_PROBES: usize = 4;

/// One batch in this many, the worker checks the ticket of one nonce of the batch exactly.
pub const AUDIT_BATCHES: u64 = 64;

/// Compares a scanner with the C reference on `header`, for the model and dataset it holds: the hits of the nonces
/// `0..count` for three bounds (a ticket whose top byte is at most 0x40, 0x80 and 0xC0) must be exactly the
/// reference's, and `probes` nonces spread over the range must have exactly the reference ticket. The first
/// difference is an error.
pub fn check_scanner(
    scanner: &mut dyn Scanner,
    model: &Model,
    dataset: &dyn Dataset,
    header: &[u8; 80],
    count: u32,
    probes: usize,
) -> Result<(), String> {
    let tickets: Vec<[u8; 32]> = {
        let threads = 8usize.min(count as usize).max(1);
        let per = (count as usize).div_ceil(threads);
        let mut parts: Vec<Vec<[u8; 32]>> = Vec::new();
        thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    s.spawn(move || {
                        let mut buffers = ShareBuffers::new();
                        let mut h = *header;
                        (t * per..((t + 1) * per).min(count as usize))
                            .map(|n| {
                                h[76..80].copy_from_slice(&(n as u32).to_le_bytes());
                                buffers.ticket(model, dataset, blake3::hash(&h).as_bytes())
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            parts = handles.into_iter().map(|h| h.join().expect("reference thread")).collect();
        });
        parts.into_iter().flatten().collect()
    };
    for top in [0x40u8, 0x80, 0xC0] {
        let mut bound = [0xFFu8; 32];
        bound[31] = top;
        let want: Vec<u32> = (0..count).filter(|&n| tickets[n as usize][31] <= top).collect();
        let mut got = scanner.scan(header, 0, count, &bound)?;
        got.sort_unstable();
        if got != want {
            let first = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            return Err(format!(
                "bound with top byte {top:#04x}: {} hits, the reference has {} (first difference at index {first})",
                got.len(),
                want.len()
            ));
        }
    }
    for i in 0..probes.min(count as usize) {
        let nonce = (i * count as usize / probes.max(1)) as u32;
        if !probe_ticket(scanner, header, nonce, &tickets[nonce as usize])? {
            return Err(format!("the ticket of nonce {nonce} differs from the reference"));
        }
    }
    Ok(())
}

/// The weight matrices of packed weights (spec/shares.md section 7), as the device layout needs them.
pub fn unpack_weights(packed: &[u8]) -> (Vec<i8>, Vec<i8>) {
    let signed = |nibble: u8| ((nibble << 4) as i8) >> 4;
    let all: Vec<i8> = packed.iter().flat_map(|b| [signed(b & 15), signed(b >> 4)]).collect();
    debug_assert_eq!(all.len(), N1 + N2);
    (all[..N1].to_vec(), all[N1..].to_vec())
}

/// Samples of a dataset as the device wants them: all pixels, then all labels.
pub fn flatten(dataset: &dyn Dataset) -> (Vec<u8>, Vec<u8>) {
    let n = dataset.n() as usize;
    let (mut pixels, mut labels) = (Vec::with_capacity(n * NIN), Vec::with_capacity(n));
    for i in 0..n as u32 {
        let (x, label) = dataset.get(i);
        pixels.extend_from_slice(&x);
        labels.push(label);
    }
    (pixels, labels)
}

/// Gives a scanner on K0 for the same device, used when the kernel in use disagrees with the reference.
pub type Fallback = Box<dyn FnOnce() -> Result<Box<dyn Scanner>, String> + Send>;

/// Starts a worker that mines the jobs of `session` with `scanner`. `attempts` counts the attempts of this device and
/// `mismatches` the nonces that the scanner returned and the CPU check refused.
pub fn spawn_worker(
    session: Session,
    dataset: Arc<dyn Dataset>,
    scanner: Box<dyn Scanner>,
    attempts: Arc<AtomicU64>,
    mismatches: Arc<AtomicU64>,
) -> JoinHandle<()> {
    spawn_worker_with_fallback(session, dataset, scanner, None, attempts, mismatches, Arc::new(AtomicU64::new(0)))
}

/// As [`spawn_worker`] for a scanner that runs an optimised kernel when `fallback` is given. The worker then also
/// checks each model it uploads on [`MODEL_PROBES`] nonces, audits the ticket of one nonce every [`AUDIT_BATCHES`]
/// batches, and treats a nonce that the reference refuses as a fault. On any fault it takes the scanner of
/// `fallback` (K0), uploads the model to it, scans the batch again and counts one in `fallbacks`. The reference
/// decides every nonce that is reported, with or without `fallback`.
pub fn spawn_worker_with_fallback(
    session: Session,
    dataset: Arc<dyn Dataset>,
    mut scanner: Box<dyn Scanner>,
    mut fallback: Option<Fallback>,
    attempts: Arc<AtomicU64>,
    mismatches: Arc<AtomicU64>,
    fallbacks: Arc<AtomicU64>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut buffers = ShareBuffers::new();
        let mut loaded: Option<Arc<Model>> = None;
        let mut batches = 0u64;
        let batch = scanner.capacity().min(BATCH);
        // Replaces the scanner by the fallback and uploads `model` to it; false when that is impossible.
        let switch =
            |scanner: &mut Box<dyn Scanner>, fallback: &mut Option<Fallback>, model: &Model, why: &str| -> bool {
                let Some(make) = fallback.take() else { return false };
                eprintln!("neb-miner-gpu: {why}; switching this device to K0");
                match make().and_then(|mut s| {
                    let (w1, w2) = unpack_weights(&model.packed_weights());
                    s.set_model(&w1, &w2)?;
                    Ok(s)
                }) {
                    Ok(s) => {
                        *scanner = s;
                        fallbacks.fetch_add(1, Ordering::Relaxed);
                        true
                    }
                    Err(e) => {
                        eprintln!("neb-miner-gpu: cannot open K0 on the device: {e}");
                        false
                    }
                }
            };
        while !session.stopped() {
            let Some(unit) = session.current() else {
                thread::sleep(Duration::from_millis(20));
                continue;
            };
            let Some((start, count)) = unit.take_nonces(batch) else {
                thread::sleep(Duration::from_millis(20));
                continue;
            };
            let reference = |buffers: &mut ShareBuffers, header: &mut [u8; 80], nonce: u32| {
                header[76..80].copy_from_slice(&nonce.to_le_bytes());
                buffers.ticket(&unit.model, &*dataset, blake3::hash(header).as_bytes())
            };
            if !loaded.as_ref().is_some_and(|m| Arc::ptr_eq(m, &unit.model)) {
                let (w1, w2) = unpack_weights(&unit.model.packed_weights());
                if let Err(e) = scanner.set_model(&w1, &w2) {
                    eprintln!("neb-miner-gpu: cannot load the model on the device: {e}");
                    return;
                }
                loaded = Some(unit.model.clone());
                if fallback.is_some() {
                    let mut header = unit.header;
                    for i in 0..MODEL_PROBES as u32 {
                        let nonce = i.wrapping_mul(0x9E37_79B9);
                        let want = reference(&mut buffers, &mut header, nonce);
                        if !probe_ticket(&mut *scanner, &header, nonce, &want).unwrap_or(false) {
                            if !switch(
                                &mut scanner,
                                &mut fallback,
                                &unit.model,
                                "the kernel disagrees with the reference on a new model",
                            ) {
                                return;
                            }
                            break;
                        }
                    }
                }
            }
            let confirmed = loop {
                let hits = match scanner.scan(&unit.header, start, count, &unit.share_target.0) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("neb-miner-gpu: the device failed: {e}");
                        return;
                    }
                };
                let mut header = unit.header;
                let mut confirmed = Vec::with_capacity(hits.len());
                let mut refused = 0u64;
                for nonce in hits {
                    let ticket = reference(&mut buffers, &mut header, nonce);
                    if unit.share_target.admits(&ticket) {
                        confirmed.push((nonce, ticket));
                    } else {
                        refused += 1;
                    }
                }
                mismatches.fetch_add(refused, Ordering::Relaxed);
                batches += 1;
                let mut fault = refused > 0;
                if fallback.is_some() && !fault && batches.is_multiple_of(AUDIT_BATCHES) {
                    let nonce = start + count / 2;
                    let want = reference(&mut buffers, &mut header, nonce);
                    fault = !probe_ticket(&mut *scanner, &header, nonce, &want).unwrap_or(false);
                }
                if fault && fallback.is_some() {
                    if !switch(&mut scanner, &mut fallback, &unit.model, "the kernel disagrees with the reference") {
                        return;
                    }
                    continue; // scan the batch again with K0
                }
                break confirmed;
            };
            session.stats().attempts.fetch_add(count as u64, Ordering::Relaxed);
            attempts.fetch_add(count as u64, Ordering::Relaxed);
            if session.superseded(&unit) {
                continue;
            }
            for (nonce, ticket) in confirmed {
                session.report_share(&unit, nonce);
                if unit.target.admits(&ticket) {
                    session.report_solution(&unit, nonce);
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use neb_pow::shares::SyntheticDataset;

    #[test]
    fn a_cut_report_is_split_until_it_is_complete() {
        // a device that holds at most 4 hits per scan; every nonce below 20 is a hit
        let mut calls = 0;
        let mut run = |start: u32, count: u32| -> Result<(Vec<u32>, usize), String> {
            calls += 1;
            let all: Vec<u32> = (start..start + count).filter(|&n| n < 20).collect();
            Ok((all.iter().copied().take(4).collect(), all.len()))
        };
        let mut got = scan_complete(0, 64, &mut run).unwrap();
        got.sort_unstable();
        assert_eq!(got, (0..20).collect::<Vec<u32>>());
        assert!(calls > 1);
        // a range without hits is one call
        let mut one = 0;
        let mut none = |_: u32, _: u32| -> Result<(Vec<u32>, usize), String> {
            one += 1;
            Ok((vec![], 0))
        };
        assert_eq!(scan_complete(0, 65_536, &mut none).unwrap(), Vec::<u32>::new());
        assert_eq!(one, 1);
        // an impossible report is an error, not an infinite loop
        let mut absurd = |_: u32, _: u32| -> Result<(Vec<u32>, usize), String> { Ok((vec![], 2)) };
        assert!(scan_complete(7, 1, &mut absurd).is_err());
        // the nonces at the end of the nonce space
        let mut top = |start: u32, count: u32| -> Result<(Vec<u32>, usize), String> {
            let all: Vec<u32> = (0..count).map(|i| start + i).collect();
            Ok((all.iter().copied().take(3).collect(), all.len()))
        };
        let mut got = scan_complete(u32::MAX - 9, 10, &mut top).unwrap();
        got.sort_unstable();
        assert_eq!(got, (u32::MAX - 9..=u32::MAX).collect::<Vec<u32>>());
    }

    #[test]
    fn minus_one_borrows_across_bytes() {
        assert_eq!(minus_one(&[0u8; 32]), None);
        let mut v = [0u8; 32];
        v[0] = 1;
        assert_eq!(minus_one(&v), Some([0u8; 32]));
        let mut v = [0u8; 32];
        v[2] = 1;
        let mut want = [0xFFu8; 32];
        want[2] = 0;
        want[3..].fill(0);
        want[0] = 0xFF;
        want[1] = 0xFF;
        assert_eq!(minus_one(&v), Some(want));
    }

    /// A device made of the C reference, with a switch that makes it wrong in one way or another.
    struct RefScanner {
        dataset: SyntheticDataset,
        model: Model,
        wrong_nonce: Option<u32>,
        wrong_ticket_low_bit: bool,
        max_hits: usize,
    }

    impl RefScanner {
        fn new(model: Model) -> RefScanner {
            RefScanner {
                dataset: SyntheticDataset::new(),
                model,
                wrong_nonce: None,
                wrong_ticket_low_bit: false,
                max_hits: usize::MAX,
            }
        }
    }

    impl Scanner for RefScanner {
        fn capacity(&self) -> u32 {
            4096
        }
        fn set_model(&mut self, _: &[i8], _: &[i8]) -> Result<(), String> {
            Ok(())
        }
        fn scan(&mut self, header: &[u8; 80], start: u32, count: u32, bound: &[u8; 32]) -> Result<Vec<u32>, String> {
            let target = neb_block::Target(*bound);
            let mut buffers = ShareBuffers::new();
            let mut h = *header;
            let mut hits = Vec::new();
            for n in start..start + count {
                h[76..80].copy_from_slice(&n.to_le_bytes());
                let mut t = buffers.ticket(&self.model, &self.dataset, blake3::hash(&h).as_bytes());
                if self.wrong_ticket_low_bit {
                    t[0] ^= 1;
                }
                if target.admits(&t) || self.wrong_nonce == Some(n) {
                    hits.push(n);
                }
            }
            hits.truncate(self.max_hits.min(hits.len()));
            Ok(hits)
        }
    }

    fn header() -> [u8; 80] {
        let mut h = [0u8; 80];
        for (i, b) in h.iter_mut().enumerate().take(76) {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        h
    }

    #[test]
    fn the_exact_probe_accepts_the_reference_ticket_and_nothing_else() {
        let model = Model::from_seed(&[0u8; 32]);
        let ds = SyntheticDataset::new();
        let mut dev = RefScanner::new(Model::from_seed(&[0u8; 32]));
        let mut h = header();
        h[76..80].copy_from_slice(&5u32.to_le_bytes());
        let ticket = ShareBuffers::new().ticket(&model, &ds, blake3::hash(&h).as_bytes());
        assert!(probe_ticket(&mut dev, &header(), 5, &ticket).unwrap());
        let mut higher = ticket;
        higher[30] ^= 0x10; // another value, above or below
        assert!(!probe_ticket(&mut dev, &header(), 5, &higher).unwrap());
        let mut one_bit = ticket;
        one_bit[0] ^= 1;
        assert!(!probe_ticket(&mut dev, &header(), 5, &one_bit).unwrap());
        dev.wrong_ticket_low_bit = true; // a device that is wrong by one bit
        assert!(!probe_ticket(&mut dev, &header(), 5, &ticket).unwrap());
    }

    #[test]
    fn the_scanner_check_passes_a_faithful_device_and_catches_each_kind_of_fault() {
        let model = Model::from_seed(&[0u8; 32]);
        let ds = SyntheticDataset::new();
        let mut ok = RefScanner::new(Model::from_seed(&[0u8; 32]));
        check_scanner(&mut ok, &model, &ds, &header(), 256, 8).expect("a faithful device passes");
        let mut extra = RefScanner::new(Model::from_seed(&[0u8; 32]));
        extra.wrong_nonce = Some(3);
        let e = check_scanner(&mut extra, &model, &ds, &header(), 256, 8).unwrap_err();
        assert!(e.contains("hits"), "{e}");
        let mut low = RefScanner::new(Model::from_seed(&[0u8; 32]));
        low.wrong_ticket_low_bit = true;
        let e = check_scanner(&mut low, &model, &ds, &header(), 256, 8).unwrap_err();
        assert!(e.contains("ticket"), "{e}");
        let mut cut = RefScanner::new(Model::from_seed(&[0u8; 32]));
        cut.max_hits = 10; // a device that silently drops hits
        let e = check_scanner(&mut cut, &model, &ds, &header(), 256, 8).unwrap_err();
        assert!(e.contains("hits"), "{e}");
        let mut other_model = RefScanner::new(Model::from_seed(&[1u8; 32]));
        assert!(check_scanner(&mut other_model, &model, &ds, &header(), 256, 8).is_err());
    }

    #[test]
    fn kernel_names_parse() {
        assert_eq!(GpuKernel::parse("k2"), Ok(GpuKernel::K2));
        assert_eq!(GpuKernel::parse("K0"), Ok(GpuKernel::K0));
        assert!(GpuKernel::parse("k3").is_err());
        assert_eq!((GpuKernel::K0.code(), GpuKernel::K1.code(), GpuKernel::K2.code()), (0, 1, 2));
    }
}
