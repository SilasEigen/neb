//! Metal (Apple GPU) miner of the NEB testnet.
//!
//! The attempt of `spec/shares.md` sections 5.1 to 5.4 is evaluated by the kernel `src/neb_share.metal`, compiled from
//! source when the miner starts (no Xcode and no offline `metal` compiler is needed). One threadgroup of 256 threads
//! evaluates one nonce; the layout follows the dp4a kernel of `pow/gpu/shares_cuda.cu`.
//!
//! Under the rules of `neb-engines` the GPU is an optimisation, never the authority: every attempt that the GPU
//! reports as a share or a solution is evaluated again by the C reference (`neb-pow`, `neb_share_eval`) before it is
//! sent to the node, and a disagreement is counted and dropped. `neb-miner-metal --selftest` compares digests,
//! tickets and sample indices with the reference over several models.

/// The Metal source, compiled by the device at start.
pub const KERNEL_SOURCE: &str = include_str!("neb_share.metal");

/// Words of one result record: digest (8), ticket (8), sample indices (16).
pub const RECORD_WORDS: usize = 8 + 8 + 16;

/// One attempt as the kernel returns it, bit for bit what `neb_share_eval` gives.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Record {
    pub digest: [u32; 8],
    pub ticket: [u32; 8],
    pub indices: [u32; 16],
}

impl Record {
    /// The ticket as the bytes that `Target::admits` compares.
    pub fn ticket_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, w) in self.ticket.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// The digest as the bytes of spec section 5.3, for the self-test.
    pub fn digest_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, w) in self.digest.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

#[repr(C)]
struct NebMetal {
    _private: [u8; 0],
}

mod imp {
    use std::ffi::{CStr, c_char, c_void};
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::{Duration, Instant};

    use neb_cpuminer::{MinerHandle, Session, start_with};
    use neb_pow::shares::{Dataset, Model, NIN, ShareBuffers};

    use super::{KERNEL_SOURCE, NebMetal, Record};

    unsafe extern "C" {
        fn neb_metal_new(
            source: *const c_char,
            pixels: *const c_void,
            labels: *const c_void,
            n: u32,
            max_batch: u32,
            err: *mut c_char,
            errcap: usize,
        ) -> *mut NebMetal;
        fn neb_metal_free(g: *mut NebMetal);
        fn neb_metal_name(g: *mut NebMetal) -> *const c_char;
        fn neb_metal_set_model(g: *mut NebMetal, w1: *const i8, w2: *const i8, err: *mut c_char, errcap: usize)
        -> bool;
        fn neb_metal_scan(
            g: *mut NebMetal,
            hdr: *const u32,
            nonce_base: u32,
            count: u32,
            key_mode: u32,
            err: *mut c_char,
            errcap: usize,
        ) -> bool;
        fn neb_metal_out(g: *mut NebMetal) -> *mut u32;
    }

    fn error_text(buf: &[c_char]) -> String {
        // SAFETY: the C code writes a NUL-terminated message into `buf`.
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }

    /// The Apple GPU with the dataset of the network.
    pub struct MetalEngine {
        g: *mut NebMetal,
        max_batch: u32,
        name: String,
    }

    // SAFETY: the handle owns the device and its buffers, and every call is made from the thread that holds it.
    unsafe impl Send for MetalEngine {}

    impl MetalEngine {
        /// Uploads `dataset` (every sample, once) and compiles the kernel. `max_batch` is the largest batch of nonces of
        /// one dispatch; 2048 is about 25 ms of work on an M4 Pro.
        pub fn new(dataset: &dyn Dataset, max_batch: u32) -> Result<MetalEngine, String> {
            let n = dataset.n();
            if n == 0 {
                return Err("the dataset is empty".into());
            }
            let mut pixels = vec![0u8; n as usize * NIN];
            let mut labels = vec![0u8; n as usize];
            for i in 0..n {
                let (x, label) = dataset.get(i);
                pixels[i as usize * NIN..][..NIN].copy_from_slice(&x);
                labels[i as usize] = label;
            }
            let source = std::ffi::CString::new(KERNEL_SOURCE).map_err(|e| e.to_string())?;
            let mut err = [0 as c_char; 4096];
            // SAFETY: the buffers outlive the call, and the device copies what it needs before it returns.
            let g = unsafe {
                neb_metal_new(
                    source.as_ptr(),
                    pixels.as_ptr().cast::<c_void>(),
                    labels.as_ptr().cast::<c_void>(),
                    n,
                    max_batch.max(1),
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if g.is_null() {
                return Err(error_text(&err));
            }
            // SAFETY: `g` is a live handle.
            let name = unsafe { CStr::from_ptr(neb_metal_name(g)) }.to_string_lossy().into_owned();
            Ok(MetalEngine { g, max_batch: max_batch.max(1), name })
        }

        /// The name of the device, such as "Apple M4 Pro".
        pub fn device_name(&self) -> &str {
            &self.name
        }

        pub fn max_batch(&self) -> u32 {
            self.max_batch
        }

        /// The weights of the job that the batch is mined on.
        pub fn set_model(&mut self, model: &Model) -> Result<(), String> {
            let mut err = [0 as c_char; 4096];
            // SAFETY: `w1` and `w2` have N1 and N2 entries, the sizes the C code reads.
            let ok = unsafe {
                neb_metal_set_model(self.g, model.w1().as_ptr(), model.w2().as_ptr(), err.as_mut_ptr(), err.len())
            };
            if ok { Ok(()) } else { Err(error_text(&err)) }
        }

        /// Attempts `nonce_base .. nonce_base + count` on the 80-byte header `header`, whose nonce field must be zero.
        /// Returns one record per nonce, in order.
        pub fn scan(&mut self, header: &[u8; 80], nonce_base: u32, count: u32) -> Result<&[Record], String> {
            let mut hdr = [0u32; 20];
            for (i, w) in hdr.iter_mut().enumerate() {
                *w = u32::from_le_bytes(header[4 * i..4 * i + 4].try_into().unwrap());
            }
            self.scan_words(&hdr, nonce_base, count, 0)
        }

        /// One attempt on a bare key, as the vectors of the reference use it (`spec/shares.md` section 3.5): for the
        /// self-test, which does not have an 80-byte header to hash.
        pub fn scan_key(&mut self, key: &[u8; 32]) -> Result<Record, String> {
            let mut hdr = [0u32; 20];
            for (i, w) in hdr.iter_mut().take(8).enumerate() {
                *w = u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap());
            }
            let records = self.scan_words(&hdr, 0, 1, 1)?;
            records.first().copied().ok_or_else(|| "no record".to_string())
        }

        fn scan_words(
            &mut self,
            hdr: &[u32; 20],
            nonce_base: u32,
            count: u32,
            key_mode: u32,
        ) -> Result<&[Record], String> {
            let mut err = [0 as c_char; 4096];
            // SAFETY: `hdr` has the 20 words the kernel reads, and the record buffer holds `max_batch` records.
            let ok = unsafe {
                neb_metal_scan(self.g, hdr.as_ptr(), nonce_base, count, key_mode, err.as_mut_ptr(), err.len())
            };
            if !ok {
                return Err(error_text(&err));
            }
            // SAFETY: the handle is live and owns the record buffer.
            let words = unsafe { neb_metal_out(self.g) };
            // SAFETY: the last scan wrote `count` records of `RECORD_WORDS` words into that buffer and `count <= max_batch`,
            // which the C code enforces, so the slice covers exactly those records.
            let records = unsafe { std::slice::from_raw_parts(words.cast::<Record>(), count as usize) };
            Ok(records)
        }
    }

    impl Drop for MetalEngine {
        fn drop(&mut self) {
            // SAFETY: the handle is freed once, and no borrow of it survives.
            unsafe { neb_metal_free(self.g) }
        }
    }

    /// How a miner run turned out.
    pub struct MiningSummary {
        pub attempts: u64,
        pub verified_hits: u64,
        pub mismatches: u64,
    }

    /// Connects to the work service, mines with the GPU and reports shares and solutions.
    pub fn start(
        node: impl std::net::ToSocketAddrs,
        dataset: Arc<dyn Dataset>,
        name: &str,
        batch: u32,
    ) -> io::Result<MinerHandle> {
        start_with(node, name, move |session| {
            let session = session.clone();
            vec![thread::spawn(move || {
                if let Err(e) = run(&session, &dataset, batch) {
                    eprintln!("metal miner: {e}");
                }
            })]
        })
    }

    /// The worker loop: the model of the current job, batches of nonces, the reference on every hit.
    pub fn run(session: &Session, dataset: &Arc<dyn Dataset>, batch: u32) -> Result<MiningSummary, String> {
        let mut engine = MetalEngine::new(&**dataset, batch)?;
        let mut buffers = ShareBuffers::new();
        let mut summary = MiningSummary { attempts: 0, verified_hits: 0, mismatches: 0 };
        let mut model_hash: Option<[u8; 32]> = None;
        let mut last_report = Instant::now();

        while !session.stopped() {
            let Some(unit) = session.current() else {
                thread::sleep(Duration::from_millis(20));
                continue;
            };
            let hash = unit.model.weights_hash();
            if model_hash != Some(hash) {
                engine.set_model(&unit.model)?;
                model_hash = Some(hash);
            }
            let Some((start, len)) = unit.take_nonces(engine.max_batch()) else {
                thread::sleep(Duration::from_millis(20));
                continue;
            };
            let records = engine.scan(&unit.header, start, len)?.to_vec();
            summary.attempts += len as u64;
            session.stats().attempts.fetch_add(len as u64, Ordering::Relaxed);

            for (i, record) in records.iter().enumerate() {
                let ticket = record.ticket_bytes();
                let nonce = start.wrapping_add(i as u32);
                let hit = unit.share_target.admits(&ticket) || unit.target.admits(&ticket);
                if !hit {
                    continue;
                }
                // The reference decides: the GPU only proposes.
                let mut header = unit.header;
                header[76..80].copy_from_slice(&nonce.to_le_bytes());
                let key = *blake3::hash(&header).as_bytes();
                let truth = buffers.ticket(&unit.model, &**dataset, &key);
                if truth != ticket {
                    summary.mismatches += 1;
                    eprintln!(
                        "metal miner: the GPU ticket of nonce {nonce} differs from the reference; dropped and the hit is not sent"
                    );
                    continue;
                }
                summary.verified_hits += 1;
                if unit.share_target.admits(&truth) {
                    session.report_share(&unit, nonce);
                }
                if unit.target.admits(&truth) {
                    session.report_solution(&unit, nonce);
                }
            }

            if last_report.elapsed() >= Duration::from_secs(5) {
                let stats = session.stats();
                let per_second =
                    summary.attempts as f64 / last_report.elapsed().max(Duration::from_millis(1)).as_secs_f64();
                eprintln!(
                    "metal miner [{}]: {per_second:.0} attempts/s, {} shares, {} blocks ({} accepted, {} rejected)",
                    engine.device_name(),
                    stats.shares.load(Ordering::Relaxed),
                    stats.blocks.load(Ordering::Relaxed),
                    stats.accepted.load(Ordering::Relaxed),
                    stats.rejected.load(Ordering::Relaxed)
                );
                last_report = Instant::now();
                summary.attempts = 0;
            }
        }
        Ok(summary)
    }
}

#[cfg(target_os = "macos")]
pub use imp::{MetalEngine, MiningSummary, run, start};

/// The same interface on a platform without Metal: every entry point says why it cannot run, so that the crate and its
/// binary build and the workspace tests pass on the Linux machine of the CI.
#[cfg(not(target_os = "macos"))]
mod unsupported {
    use std::io;
    use std::sync::Arc;

    use neb_cpuminer::MinerHandle;
    use neb_pow::shares::{Dataset, Model};

    use super::Record;

    /// Why this build cannot mine: `neb-miner-metal` needs Metal, which only macOS provides.
    const WHY: &str = "neb-miner-metal needs macOS on Apple silicon: this build has no Metal";

    pub struct MetalEngine {
        _private: (),
    }

    impl MetalEngine {
        pub fn new(_dataset: &dyn Dataset, _max_batch: u32) -> Result<MetalEngine, String> {
            Err(WHY.to_string())
        }

        pub fn device_name(&self) -> &str {
            "no Metal"
        }

        pub fn max_batch(&self) -> u32 {
            0
        }

        pub fn set_model(&mut self, _model: &Model) -> Result<(), String> {
            Err(WHY.to_string())
        }

        pub fn scan(&mut self, _header: &[u8; 80], _nonce_base: u32, _count: u32) -> Result<&[Record], String> {
            Err(WHY.to_string())
        }

        pub fn scan_key(&mut self, _key: &[u8; 32]) -> Result<Record, String> {
            Err(WHY.to_string())
        }
    }

    pub struct MiningSummary {
        pub attempts: u64,
        pub verified_hits: u64,
        pub mismatches: u64,
    }

    pub fn start(
        _node: impl std::net::ToSocketAddrs,
        _dataset: Arc<dyn Dataset>,
        _name: &str,
        _batch: u32,
    ) -> io::Result<MinerHandle> {
        Err(io::Error::other(WHY))
    }

    pub fn run(
        _session: &neb_cpuminer::Session,
        _dataset: &Arc<dyn Dataset>,
        _batch: u32,
    ) -> Result<MiningSummary, String> {
        Err(WHY.to_string())
    }
}

#[cfg(not(target_os = "macos"))]
pub use unsupported::{MetalEngine, MiningSummary, run, start};
