//! Metal (Apple GPU) miner of the NEB testnet.
//!
//! The attempt of `spec/shares.md` sections 5.1 to 5.4 is evaluated by the kernel `src/neb_share.metal`, and the attempt
//! of `spec/nebhash-v2.md` section 4 (KEP1 and RET2) by `src/neb_share2.metal`; both are compiled from source when the
//! miner starts (no Xcode and no offline `metal` compiler is needed), after the shared hash code of
//! `src/neb_common.metal`. One threadgroup of 256 threads evaluates one nonce; the layout follows the dp4a kernel of
//! `pow/gpu/shares_cuda.cu` for the 784-128-10 network and `pow/gpu/nebhash2_cuda.cu` for the v2 architectures, whose
//! layer table is the one of the C reference (`neb2_arch_get`).
//!
//! Under the rules of `neb-engines` the GPU is an optimisation, never the authority: every attempt that the GPU
//! reports as a share or a solution is evaluated again by the C reference (`neb-pow`, `neb_share_eval`) before it is
//! sent to the node, and a disagreement is counted and dropped. `neb-miner-metal --selftest` compares digests,
//! tickets and sample indices with the reference over several models.

/// The Metal source of the 784-128-10 attempt, compiled by the device at start.
pub const KERNEL_SOURCE: &str = concat!(include_str!("neb_common.metal"), "\n", include_str!("neb_share.metal"));

/// The Metal source of the nebhash v2 attempt (KEP1 and RET2).
pub const KERNEL_SOURCE2: &str = concat!(include_str!("neb_common.metal"), "\n", include_str!("neb_share2.metal"));

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
#[cfg(target_os = "macos")]
struct NebMetal {
    _private: [u8; 0],
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::{CStr, c_char, c_int, c_void};
    use std::io;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::{Duration, Instant};

    use neb_cpuminer::{MinerHandle, Session, start_with};
    use neb_pow::shares::{Arch, Dataset, Model, NB, ShareBuffers, SharedDataset};

    use super::{KERNEL_SOURCE, KERNEL_SOURCE2, NebMetal, Record};

    /// Mirror of `neb2_layer` in `pow/ref/nebhash2.h`: nineteen ints.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CLayer {
        kind: i32,
        cin: i32,
        cout: i32,
        k: i32,
        stride: i32,
        pad: i32,
        pool: i32,
        s: i32,
        sb: i32,
        su: i32,
        in_len: i32,
        in_off: i32,
        nsrc: i32,
        src: [i32; 2],
        conv_len: i32,
        out_len: i32,
        nw: i32,
        w_off: i32,
    }

    /// Mirror of `neb2_arch` in `pow/ref/nebhash2.h`.
    #[repr(C)]
    struct CArch {
        id: i32,
        nin: i32,
        nlayers: i32,
        nweights: i32,
        l: [CLayer; MAX_LAYERS],
    }

    const _: () = assert!(std::mem::size_of::<CLayer>() == 19 * 4);
    const _: () = assert!(std::mem::size_of::<CArch>() == (4 + MAX_LAYERS * 19) * 4);

    /// `NEB2_MAX_LAYERS`.
    const MAX_LAYERS: usize = 6;
    /// Ints of a layer in the plan of `neb_share2.metal` (`struct Layer`), and of its header (`struct Plan`).
    const PLAN_LAYER_INTS: usize = 27;
    const PLAN_HEADER_INTS: usize = 8;

    unsafe extern "C" {
        /// The layer table of the C reference, the one the node runs: the kernel reads it, not a copy of it.
        fn neb2_arch_get(id: c_int) -> *const CArch;
        fn neb_metal_new(
            source: *const c_char,
            pixels: *const c_void,
            labels: *const c_void,
            n: u32,
            max_batch: u32,
            nin: u32,
            plan: *const c_void,
            plan_bytes: usize,
            nweights: u32,
            block_bytes: u32,
            err: *mut c_char,
            errcap: usize,
        ) -> *mut NebMetal;
        fn neb_metal_set_weights(g: *mut NebMetal, w: *const i8, err: *mut c_char, errcap: usize) -> bool;
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

    /// The plan of `neb_share2.metal` for a v2 architecture: the layer table of the C reference, the sizes derived from
    /// it and the layout of the scratch area (per sample: the input and the routed error of every layer; common to the
    /// samples: the activations, the pooled outputs and their errors), as in `make_plan` of `pow/gpu/nebhash2_cuda.cu`.
    /// Returns the ints of the plan, the number of weights and the scratch bytes of one attempt.
    fn v2_plan(arch: Arch) -> Result<(Vec<i32>, u32, u32), String> {
        let id = arch.id().ok_or("not a nebhash v2 architecture")? as c_int;
        // SAFETY: the C reference returns a pointer to a static table, or NULL for an unknown id.
        let table = unsafe { neb2_arch_get(id).as_ref() }.ok_or("the C reference has no such architecture")?;
        let up16 = |v: i32| (v + 15) & !15;
        let nl = table.nlayers as usize;
        let (mut at, mut common) = (0i32, 0i32);
        let mut layers = Vec::with_capacity(nl * PLAN_LAYER_INTS);
        for l in &table.l[..nl] {
            let sq = |n: i32| if l.kind == 2 { n * n } else { n };
            let nin_in = if l.kind == 3 { l.cin } else { l.cin * sq(l.in_len) };
            let nconv = l.cout * sq(l.conv_len);
            let nout = l.cout * sq(l.out_len);
            let (off_in, off_d) = (at, at + up16(nin_in));
            at = off_d + up16(nconv);
            let off_a = common;
            let off_out = off_a + up16(nconv);
            let off_e = off_out + up16(nout);
            common = off_e + up16(nout);
            layers.extend_from_slice(&[
                l.kind, l.cin, l.cout, l.k, l.stride, l.pad, l.pool, l.s, l.sb, l.su, l.in_len, l.in_off, l.nsrc,
                l.src[0], l.src[1], l.conv_len, l.out_len, l.nw, l.w_off, nin_in, nconv, nout, off_in, off_d, off_a,
                off_out, off_e,
            ]);
        }
        let block_bytes = (NB as i32 * at + common + 255) & !255;
        let mut plan = vec![0i32; PLAN_HEADER_INTS + MAX_LAYERS * PLAN_LAYER_INTS];
        plan[..6].copy_from_slice(&[table.nlayers, table.nweights, table.nin, at, common, block_bytes]);
        plan[PLAN_HEADER_INTS..PLAN_HEADER_INTS + layers.len()].copy_from_slice(&layers);
        Ok((plan, table.nweights as u32, block_bytes as u32))
    }

    /// The Apple GPU with the dataset of the network.
    pub struct MetalEngine {
        g: *mut NebMetal,
        max_batch: u32,
        name: String,
        arch: Arch,
    }

    // SAFETY: the handle owns the device and its buffers, and every call is made from the thread that holds it.
    unsafe impl Send for MetalEngine {}

    impl MetalEngine {
        /// Uploads `dataset` (every sample, once) and compiles the kernel of its architecture. `max_batch` is the largest
        /// batch of nonces of one scan; 2048 is about 25 ms of work on an M4 Pro for the 784-128-10 network.
        pub fn new(dataset: &dyn Dataset, max_batch: u32) -> Result<MetalEngine, String> {
            let n = dataset.n();
            if n == 0 {
                return Err("the dataset is empty".into());
            }
            let arch = dataset.arch();
            let nin = arch.nin();
            let mut pixels = vec![0u8; n as usize * nin];
            let mut labels = vec![0u8; n as usize];
            for i in 0..n {
                labels[i as usize] = dataset.sample(i, &mut pixels[i as usize * nin..][..nin]);
            }
            let (plan, nweights, block_bytes, source) = if arch == Arch::V1 {
                (Vec::new(), 0, 0, KERNEL_SOURCE)
            } else {
                let (plan, nweights, block_bytes) = v2_plan(arch)?;
                (plan, nweights, block_bytes, KERNEL_SOURCE2)
            };
            let source = std::ffi::CString::new(source).map_err(|e| e.to_string())?;
            let mut err = [0 as c_char; 4096];
            // SAFETY: the buffers outlive the call, and the device copies what it needs before it returns.
            let g = unsafe {
                neb_metal_new(
                    source.as_ptr(),
                    pixels.as_ptr().cast::<c_void>(),
                    labels.as_ptr().cast::<c_void>(),
                    n,
                    max_batch.max(1),
                    nin as u32,
                    if plan.is_empty() { std::ptr::null() } else { plan.as_ptr().cast::<c_void>() },
                    plan.len() * 4,
                    nweights,
                    block_bytes,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if g.is_null() {
                return Err(error_text(&err));
            }
            // SAFETY: `g` is a live handle.
            let name = unsafe { CStr::from_ptr(neb_metal_name(g)) }.to_string_lossy().into_owned();
            Ok(MetalEngine { g, max_batch: max_batch.max(1), name, arch })
        }

        /// The name of the device, such as "Apple M4 Pro".
        pub fn device_name(&self) -> &str {
            &self.name
        }

        pub fn max_batch(&self) -> u32 {
            self.max_batch
        }

        /// The weights of the job that the batch is mined on. The model must be of the architecture of the dataset.
        pub fn set_model(&mut self, model: &Model) -> Result<(), String> {
            if model.arch() != self.arch {
                return Err(format!("a {} model on a {} engine", model.arch().name(), self.arch.name()));
            }
            let mut err = [0 as c_char; 4096];
            if self.arch != Arch::V1 {
                // SAFETY: a v2 model has `arch.weights()` weights, the count the engine was built for.
                let ok =
                    unsafe { neb_metal_set_weights(self.g, model.weights().as_ptr(), err.as_mut_ptr(), err.len()) };
                return if ok { Ok(()) } else { Err(error_text(&err)) };
            }
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

    /// Connects to the work service, mines with the GPU and reports shares and solutions. `datasets` are the datasets of
    /// the network in slot order (one for a network of one model), as `neb-cpuminer` takes them.
    pub fn start(
        node: impl std::net::ToSocketAddrs,
        datasets: Vec<SharedDataset>,
        name: &str,
        batch: u32,
    ) -> io::Result<MinerHandle> {
        start_with(node, name, move |session| {
            let session = session.clone();
            let datasets = datasets.clone();
            vec![thread::spawn(move || {
                if let Err(e) = run(&session, &datasets, batch) {
                    eprintln!("metal miner: {e}");
                }
            })]
        })
    }

    /// The worker loop: one engine per model slot, the model of the current job, batches of nonces, the reference on
    /// every hit.
    pub fn run(session: &Session, datasets: &[SharedDataset], batch: u32) -> Result<MiningSummary, String> {
        // One engine per slot, in slot order, each with the kernel of its dataset's architecture (the 784-128-10
        // attempt of spec/shares.md or one of nebhash v2). A job of a slot without an engine waits for the next job.
        let mut engines: Vec<Option<MetalEngine>> = Vec::with_capacity(datasets.len());
        for d in datasets {
            engines.push(Some(MetalEngine::new(&**d, batch)?));
        }
        let mut models: Vec<Option<[u8; 32]>> = vec![None; engines.len()];
        let mut buffers = ShareBuffers::new();
        let mut summary = MiningSummary { attempts: 0, verified_hits: 0, mismatches: 0 };
        let mut last_report = Instant::now();

        while !session.stopped() {
            let Some(unit) = session.current() else {
                thread::sleep(Duration::from_millis(20));
                continue;
            };
            // A slot this miner has no dataset for cannot be mined, and neither can a model of another network than the
            // dataset's (an old dataset file after an upgrade): wait for the next job.
            let slot = unit.slot as usize;
            let fits = engines.get(slot).is_some_and(|e| e.is_some()) && datasets[slot].arch() == unit.model.arch();
            let Some(Some(engine)) = engines.get_mut(slot).filter(|_| fits) else {
                thread::sleep(Duration::from_millis(200));
                continue;
            };
            let hash = unit.model.weights_hash();
            if models[slot] != Some(hash) {
                engine.set_model(&unit.model)?;
                models[slot] = Some(hash);
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
                let truth = buffers.ticket(&unit.model, &*datasets[slot], &key);
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

    use neb_cpuminer::MinerHandle;
    use neb_pow::shares::{Dataset, Model, SharedDataset};

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
        _datasets: Vec<SharedDataset>,
        _name: &str,
        _batch: u32,
    ) -> io::Result<MinerHandle> {
        Err(io::Error::other(WHY))
    }

    pub fn run(
        _session: &neb_cpuminer::Session,
        _datasets: &[SharedDataset],
        _batch: u32,
    ) -> Result<MiningSummary, String> {
        Err(WHY.to_string())
    }
}

#[cfg(not(target_os = "macos"))]
pub use unsupported::{MetalEngine, MiningSummary, run, start};
