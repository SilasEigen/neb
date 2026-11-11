//! Safe wrapper of the C reference of the training-share step
//! (`pow/ref/shares.c`, normative text `spec/shares.md` sections 4 to 7).
//!
//! The C code is the only implementation of the arithmetic. This module owns
//! the memory layout shared with it, the dataset callback, reusable gradient
//! buffers and the replay of `vectors/shares.txt` (spec/shares.md section 10).
//! All arithmetic is integer, so results are bit-identical on every platform
//! and compiler. The C functions keep no global state and may run on many
//! threads at once; each thread needs its own `ShareBuffers`.
//!
//! Unsafe boundary: every `unsafe` block passes pointers to live, correctly
//! sized buffers (`N1`, `N2`, 32-byte hashes, `NebModel` of 508160 bytes). The
//! use of a share in the proof of work and in the chain state is in `neb-chain`.

use std::ffi::{c_int, c_void};
use std::sync::Arc;

fn hex_bytes<const K: usize>(s: &str) -> Result<[u8; K], String> {
    let b = s.as_bytes();
    if b.len() != 2 * K {
        return Err(format!("expected {} hex characters, got {}", 2 * K, b.len()));
    }
    let mut out = [0u8; K];
    for i in 0..K {
        let hi = (b[2 * i] as char).to_digit(16).ok_or("not hex")?;
        let lo = (b[2 * i + 1] as char).to_digit(16).ok_or("not hex")?;
        out[i] = (hi * 16 + lo) as u8;
    }
    Ok(out)
}

/// Inputs per sample. `NIN` to `NB` are constants of spec/shares.md section 2.
pub const NIN: usize = 784;
/// Hidden units.
pub const NH: usize = 128;
/// Classes.
pub const NOUT: usize = 10;
/// Samples per share.
pub const NB: usize = 16;
/// Entries of `W1`, `R1` and `G1`.
pub const N1: usize = NH * NIN;
/// Entries of `W2`, `R2` and `G2`.
pub const N2: usize = NOUT * NH;
/// Largest number of shares in one step (spec/shares.md section 6): the largest
/// `n` with `2^SU1 + n * CLIP < 2^31`, that is `2^15 + 8388479 * 256 =
/// 2147483392`, so `R1 + GS1` fits in an int32. Equal to `NEB_MAX_SHARES` in
/// `pow/ref/shares.h`.
pub const MAX_SHARES: usize = 8_388_479;
/// Samples of the synthetic test dataset (spec/shares.md section 9).
pub const SYN_N: u32 = 2048;
/// Bytes of the packed weights `pack(W1) || pack(W2)` (spec/shares.md section 7): two
/// 4-bit weights per byte, `(N1 + N2) / 2 = 50816`.
pub const PACKED_WEIGHTS: usize = (N1 + N2) / 2;
/// Update shifts of layer 1 and layer 2 (spec/shares.md section 2): the residual of an entry is
/// divided by `2^SU` to give the weight step. A consensus build uses 15 and 12.
pub const SU1: u32 = 15;
pub const SU2: u32 = 12;
/// Clip of one entry of the gradient of one share (spec/shares.md section 2).
pub const CLIP: i32 = 256;
/// Range of a weight (spec/shares.md section 4): a signed 4-bit value.
pub const WEIGHT_MIN: i8 = -8;
pub const WEIGHT_MAX: i8 = 7;

/// Mirror of `neb_model` in `pow/ref/shares.h`: field order and types must
/// match. Size: `N1 + N2` bytes of weights (one per int8) and `4 * (N1 + N2)`
/// bytes of residuals, `5 * 101632 = 508160`; a model is always heap-allocated.
#[repr(C)]
struct NebModel {
    w1: [i8; N1],
    w2: [i8; N2],
    r1: [i32; N1],
    r2: [i32; N2],
}

// The Rust mirror must have the layout of the C struct: weights, then residuals, no padding.
const _: () = {
    assert!(std::mem::size_of::<NebModel>() == (N1 + N2) + 4 * (N1 + N2));
    assert!(std::mem::offset_of!(NebModel, w1) == 0);
    assert!(std::mem::offset_of!(NebModel, w2) == N1);
    assert!(std::mem::offset_of!(NebModel, r1) == N1 + N2);
    assert!(std::mem::offset_of!(NebModel, r2) == N1 + N2 + 4 * N1);
};

/// Mirror of `neb_dataset`: sample count, sample callback and its context.
/// The callback fills `NIN` pixel bytes and returns the label.
#[repr(C)]
struct NebDataset {
    n: u32,
    get: extern "C" fn(*mut c_void, u32, *mut u8) -> c_int,
    ctx: *mut c_void,
}

// Functions of `pow/ref/shares.h` and `pow/ref/chacha8.h`.
unsafe extern "C" {
    fn neb_model_init(seed: *const u8, m: *mut NebModel);
    fn neb_share_eval(
        m: *const NebModel,
        ds: *const NebDataset,
        key: *const u8,
        g1: *mut i32,
        g2: *mut i32,
        digest: *mut u8,
        ticket: *mut u8,
        idx: *mut u32,
        nclip: *mut u32,
    );
    fn neb_model_apply(m: *mut NebModel, gs1: *const i32, gs2: *const i32);
    fn neb_model_hashes(m: *const NebModel, weights_hash: *mut u8, residual_hash: *mut u8);
    fn neb_model_predict(m: *const NebModel, x: *const u8) -> c_int;
    fn chacha8_stream(key: *const u8, out: *mut u8, n: usize);
}

/// Why a model step is refused. The C step is exact only while `|R + GS|` fits in an int32 without the value
/// `-2^31` (its `tdiv` negates the sum); the consensus never leaves that domain (a block lists at most 256
/// shares, so `|GS| <= 65536`), but the type makes the limit explicit instead of leaving it to the C code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepError {
    /// A slice of gradient sums does not have `N1` (layer 1) or `N2` (layer 2) entries.
    Length { layer: u8, expected: usize, got: usize },
    /// A step of more than [`MAX_SHARES`] shares.
    TooManyShares(usize),
    /// `residual + sum` of this entry is outside `-(2^31 - 1)..=2^31 - 1`.
    OutOfRange { layer: u8, index: usize, residual: i32, sum: i32 },
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            StepError::Length { layer, expected, got } => {
                write!(f, "layer {layer}: {got} gradient sums, expected {expected}")
            }
            StepError::TooManyShares(n) => write!(f, "a step has at most {MAX_SHARES} shares, got {n}"),
            StepError::OutOfRange { layer, index, residual, sum } => {
                write!(
                    f,
                    "layer {layer} entry {index}: residual {residual} plus sum {sum} is outside the int32 range of the step"
                )
            }
        }
    }
}

impl std::error::Error for StepError {}

/// The samples a share draws from (spec/shares.md section 4, spec/nebhash-v2.md section 2): `n() >= 1` samples of
/// the architecture's inputs in 0..=7 and a label below its class count.
///
/// `get` and `sample` are called with `index < n()` only. They must be pure functions of `index`, because every node
/// must derive the same samples. The ranges are asserted in debug builds only.
pub trait Dataset: Send + Sync + std::fmt::Debug {
    fn n(&self) -> u32;
    /// A sample of a 784-input dataset (the v1 network and KEP1). A dataset of another input count returns its first
    /// 784 values; the v2 path uses [`Dataset::sample`].
    fn get(&self, index: u32) -> ([u8; NIN], u8);
    /// The network that trains on this dataset.
    fn arch(&self) -> Arch {
        Arch::V1
    }
    /// Writes the `arch().nin()` values of sample `index` to `x` and returns its label.
    fn sample(&self, index: u32, x: &mut [u8]) -> u8 {
        let (values, label) = self.get(index);
        x[..NIN].copy_from_slice(&values);
        label
    }
}

/// A dataset shared by the chain state, the work service and the miners.
pub type SharedDataset = Arc<dyn Dataset>;

/// Sample callback of the C code. `ctx` points to a `&dyn Dataset` that lives
/// on the stack of the caller of `neb_share_eval` (a thin pointer to the fat
/// reference, because a trait object does not fit in a `*mut c_void`).
extern "C" fn dataset_get(ctx: *mut c_void, index: u32, x: *mut u8) -> c_int {
    // SAFETY: the C code passes back the `ctx` of the `NebDataset` built by `share_eval` or `ShareBuffers::eval`, a
    // pointer to a `&dyn Dataset` on their stack that outlives the call of `neb_share_eval`.
    let ds = unsafe { &*(ctx as *const &dyn Dataset) };
    let (pixels, label) = ds.get(index);
    debug_assert!(pixels.iter().all(|&p| p <= 7) && label < NOUT as u8);
    // SAFETY: `x` is the sample buffer of the C code, `NIN` bytes long (pow/ref/shares.h), and does not overlap `pixels`.
    unsafe { std::ptr::copy_nonoverlapping(pixels.as_ptr(), x, NIN) };
    label as c_int
}

enum Inner {
    V1(Box<NebModel>),
    V2(Arch, Box<v2::V2Model>),
}

/// A model of one architecture: its weights in -8..=7 and its residuals (spec/shares.md section 4,
/// spec/nebhash-v2.md section 6). A v1 model is `W1`, `W2`, `R1`, `R2`; cloning it copies 508160 bytes.
pub struct Model {
    inner: Inner,
}

impl Clone for Model {
    fn clone(&self) -> Model {
        match &self.inner {
            Inner::V1(m) => {
                // Copy between two heap blocks; a by-value copy would put 508160 bytes on the stack.
                let mut inner = Box::<NebModel>::new_uninit();
                // SAFETY: one `NebModel` is copied from an initialised box into a fresh allocation of the same type,
                // which initialises it; `NebModel` holds integers only.
                let inner = unsafe {
                    std::ptr::copy_nonoverlapping(&**m, inner.as_mut_ptr(), 1);
                    inner.assume_init()
                };
                Model { inner: Inner::V1(inner) }
            }
            Inner::V2(arch, m) => {
                let mut copy = v2::zeroed();
                copy.arch = m.arch;
                copy.w.copy_from_slice(&m.w);
                copy.r.copy_from_slice(&m.r);
                Model { inner: Inner::V2(*arch, copy) }
            }
        }
    }
}

impl Model {
    /// The network of the model.
    pub fn arch(&self) -> Arch {
        match &self.inner {
            Inner::V1(_) => Arch::V1,
            Inner::V2(a, _) => *a,
        }
    }

    /// All the weights in model order (v1: `W1` then `W2`).
    pub fn weights(&self) -> &[i8] {
        match &self.inner {
            // SAFETY: `w1` and `w2` are adjacent in `NebModel` (layout asserted above), so `N1 + N2` weights follow
            // the start of `w1` inside the same allocation.
            Inner::V1(m) => unsafe { std::slice::from_raw_parts(m.w1.as_ptr(), N1 + N2) },
            Inner::V2(a, m) => &m.w[..a.weights()],
        }
    }

    /// All the residuals in model order (v1: `R1` then `R2`).
    pub fn residuals(&self) -> &[i32] {
        match &self.inner {
            // SAFETY: as in `weights`, `r1` and `r2` are adjacent.
            Inner::V1(m) => unsafe { std::slice::from_raw_parts(m.r1.as_ptr(), N1 + N2) },
            Inner::V2(a, m) => &m.r[..a.weights()],
        }
    }

    /// The packed weights of spec/shares.md section 7: two 4-bit weights per byte, weight `e` in the low nibble of
    /// byte `e / 2` for even `e` and in the high nibble for odd `e`, over all the weights in model order.
    pub fn packed_weights(&self) -> Vec<u8> {
        self.weights().chunks(2).map(|p| ((p[0] as u8) & 15) | (((p[1] as u8) & 15) << 4)).collect()
    }

    /// A v1 model with these packed weights; see [`Model::from_packed`].
    pub fn from_packed_weights(bytes: &[u8]) -> Result<Model, String> {
        Model::from_packed(Arch::V1, bytes)
    }

    /// A model of `arch` with these packed weights and zero residuals: enough to evaluate shares, because the
    /// ticket depends on the weights only. Fails on a length other than `arch.packed_bytes()`. Every nibble is a
    /// valid weight, so the caller compares the weights hash to detect a damaged frame.
    pub fn from_packed(arch: Arch, bytes: &[u8]) -> Result<Model, String> {
        if bytes.len() != arch.packed_bytes() {
            return Err(format!("expected {} packed weight bytes, got {}", arch.packed_bytes(), bytes.len()));
        }
        // Sign extension of a 4-bit two's complement value.
        let signed = |nibble: u8| ((nibble << 4) as i8) >> 4;
        let weights: Vec<i8> = bytes.iter().flat_map(|b| [signed(b & 15), signed(b >> 4)]).collect();
        match arch {
            Arch::V1 => {
                let zeros = vec![0i32; N1 + N2];
                Model::from_state(&weights[..N1], &weights[N1..], &zeros[..N1], &zeros[N1..])
            }
            _ => {
                let mut m = v2::zeroed();
                m.arch = arch.id().unwrap() as i32;
                m.w[..weights.len()].copy_from_slice(&weights);
                Ok(Model { inner: Inner::V2(arch, m) })
            }
        }
    }

    /// Weights of layer 1 of a v1 model, `NH` rows of `NIN`, row-major; empty for a v2 model.
    pub fn w1(&self) -> &[i8] {
        match &self.inner {
            Inner::V1(m) => &m.w1,
            Inner::V2(..) => &[],
        }
    }

    /// Weights of layer 2 of a v1 model, `NOUT` rows of `NH`, row-major; empty for a v2 model.
    pub fn w2(&self) -> &[i8] {
        match &self.inner {
            Inner::V1(m) => &m.w2,
            Inner::V2(..) => &[],
        }
    }

    /// Residuals of layer 1 of a v1 model (same layout as [`Model::w1`]); empty for a v2 model.
    pub fn r1(&self) -> &[i32] {
        match &self.inner {
            Inner::V1(m) => &m.r1,
            Inner::V2(..) => &[],
        }
    }

    /// Residuals of layer 2 of a v1 model (same layout as [`Model::w2`]); empty for a v2 model.
    pub fn r2(&self) -> &[i32] {
        match &self.inner {
            Inner::V1(m) => &m.r2,
            Inner::V2(..) => &[],
        }
    }

    /// A v1 model with exactly this state. Fails on a wrong length, a weight
    /// outside `WEIGHT_MIN..=WEIGHT_MAX` or a residual above `2^SU` in absolute
    /// value (spec/shares.md section 4). Used by the tests and the tools that
    /// replay a state; a chain builds its models with [`Model::from_seed`] and
    /// the model step.
    pub fn from_state(w1: &[i8], w2: &[i8], r1: &[i32], r2: &[i32]) -> Result<Model, String> {
        if (w1.len(), w2.len(), r1.len(), r2.len()) != (N1, N2, N1, N2) {
            return Err(format!(
                "a state has {N1}, {N2}, {N1} and {N2} entries, got {}, {}, {} and {}",
                w1.len(),
                w2.len(),
                r1.len(),
                r2.len()
            ));
        }
        if w1.iter().chain(w2).any(|w| !(WEIGHT_MIN..=WEIGHT_MAX).contains(w)) {
            return Err("a weight is outside -8..=7".into());
        }
        if r1.iter().any(|r| r.unsigned_abs() > 1 << SU1) || r2.iter().any(|r| r.unsigned_abs() > 1 << SU2) {
            return Err("a residual is above 2^SU in absolute value".into());
        }
        let mut inner = Box::<NebModel>::new_uninit();
        // SAFETY: the four copies write every field of `NebModel` (`w1`, `w2`, `r1`, `r2`, whose lengths `N1` and `N2`
        // were checked above) through raw field pointers, without reading uninitialised memory.
        let inner = unsafe {
            let m = inner.as_mut_ptr();
            std::ptr::copy_nonoverlapping(w1.as_ptr(), (&raw mut (*m).w1).cast::<i8>(), N1);
            std::ptr::copy_nonoverlapping(w2.as_ptr(), (&raw mut (*m).w2).cast::<i8>(), N2);
            std::ptr::copy_nonoverlapping(r1.as_ptr(), (&raw mut (*m).r1).cast::<i32>(), N1);
            std::ptr::copy_nonoverlapping(r2.as_ptr(), (&raw mut (*m).r2).cast::<i32>(), N2);
            inner.assume_init()
        };
        Ok(Model { inner: Inner::V1(inner) })
    }

    /// The initial v1 state of a 32-byte seed (spec/shares.md section 4); on the chain the seed is the genesis block id.
    pub fn from_seed(seed: &[u8; 32]) -> Model {
        Model::from_seed_arch(Arch::V1, seed)
    }

    /// The initial state of `arch` for a 32-byte seed (spec/shares.md section 4, spec/nebhash-v2.md section 6).
    pub fn from_seed_arch(arch: Arch, seed: &[u8; 32]) -> Model {
        if arch != Arch::V1 {
            return Model { inner: Inner::V2(arch, v2::init(arch, seed)) };
        }
        let mut inner = Box::<NebModel>::new_uninit();
        // SAFETY: `neb_model_init` reads 32 bytes of `seed` and writes every field of the model.
        let inner = unsafe {
            neb_model_init(seed.as_ptr(), inner.as_mut_ptr());
            inner.assume_init()
        };
        Model { inner: Inner::V1(inner) }
    }

    /// The model step of spec/shares.md section 6 on a v1 model, the sums of each layer apart; see
    /// [`Model::try_apply_sums`].
    pub fn try_apply_step(&mut self, gs1: &[i32], gs2: &[i32]) -> Result<(), StepError> {
        if gs1.len() != N1 {
            return Err(StepError::Length { layer: 1, expected: N1, got: gs1.len() });
        }
        if gs2.len() != N2 {
            return Err(StepError::Length { layer: 2, expected: N2, got: gs2.len() });
        }
        let mut all = Vec::with_capacity(N1 + N2);
        all.extend_from_slice(gs1);
        all.extend_from_slice(gs2);
        self.try_apply_sums(&all)
    }

    /// The model step: `gs` holds the sums of the clipped gradients of the shares of the step, in model order (at
    /// most [`MAX_SHARES`] shares, which keeps the int32 sums exact: [`train_step`] and [`GradSum`] check it).
    /// Fails, without changing the model, when `gs` has the wrong length or when `residual + sum` of an entry leaves
    /// the exact domain of the C step.
    pub fn try_apply_sums(&mut self, gs: &[i32]) -> Result<(), StepError> {
        let n = self.arch().weights();
        if gs.len() != n {
            return Err(StepError::Length { layer: 0, expected: n, got: gs.len() });
        }
        for (index, (&residual, &sum)) in self.residuals().iter().zip(gs).enumerate() {
            if (residual as i64 + sum as i64).unsigned_abs() > i32::MAX as u64 {
                // v1 names the layer and the index inside it; v2 gives the index in model order (layer 0)
                let (layer, index) = match self.arch() {
                    Arch::V1 if index >= N1 => (2, index - N1),
                    Arch::V1 => (1, index),
                    _ => (0, index),
                };
                return Err(StepError::OutOfRange { layer, index, residual, sum });
            }
        }
        match &mut self.inner {
            // SAFETY: `gs` has `N1 + N2` entries (checked above): the C step reads `N1` from the start and `N2` after.
            Inner::V1(m) => unsafe { neb_model_apply(&mut **m, gs.as_ptr(), gs.as_ptr().add(N1)) },
            Inner::V2(_, m) => v2::apply(m, gs),
        }
        Ok(())
    }

    /// [`Model::try_apply_step`] for callers that hold the bound of a step by construction (a block lists at
    /// most 256 shares). Panics with the reason when the step is refused.
    pub fn apply_step(&mut self, gs1: &[i32], gs2: &[i32]) {
        if let Err(e) = self.try_apply_step(gs1, gs2) {
            panic!("model step refused: {e}");
        }
    }

    /// `(weights_hash, residual_hash)` of spec/shares.md section 7.
    pub fn hashes(&self) -> ([u8; 32], [u8; 32]) {
        match &self.inner {
            Inner::V1(m) => {
                let (mut w, mut r) = ([0u8; 32], [0u8; 32]);
                // SAFETY: the C code writes 32 bytes to each output buffer.
                unsafe { neb_model_hashes(&**m, w.as_mut_ptr(), r.as_mut_ptr()) };
                (w, r)
            }
            Inner::V2(_, m) => v2::hashes(m),
        }
    }

    pub fn weights_hash(&self) -> [u8; 32] {
        self.hashes().0
    }

    pub fn residual_hash(&self) -> [u8; 32] {
        self.hashes().1
    }

    /// The class of a 784-input sample; see [`Model::predict_sample`].
    pub fn predict(&self, x: &[u8; NIN]) -> u8 {
        self.predict_sample(x)
    }

    /// The class of a sample of the model's input count: the output with the largest sum, the smallest index on a
    /// tie. Used to report accuracy; the protocol does not use it. Panics on another input count.
    pub fn predict_sample(&self, x: &[u8]) -> u8 {
        assert_eq!(x.len(), self.arch().nin(), "a sample of the model's input count");
        match &self.inner {
            // SAFETY: the C code reads the `NIN` bytes of `x`.
            Inner::V1(m) => unsafe { neb_model_predict(&**m, x.as_ptr()) as u8 },
            Inner::V2(_, m) => v2::predict(m, x),
        }
    }
}

/// One evaluated share (spec/shares.md sections 5.1 to 5.4): the clipped
/// gradients, the gradient digest, the ticket and the sample indices. For a v1 model `g1` and `g2` hold the two
/// layers (`N1` and `N2` entries); for a v2 model `g1` holds every layer in model order and `g2` is empty. The ticket
/// is the BLAKE3 output as bytes; it is compared with a target as a 256-bit little-endian integer.
pub struct ShareResult {
    pub digest: [u8; 32],
    pub ticket: [u8; 32],
    pub indices: [u32; NB],
    pub g1: Vec<i32>,
    pub g2: Vec<i32>,
}

/// Evaluates the share of `key` on `model`. In the proof of work `key` is the
/// BLAKE3 hash of the header of the attempt. Allocates the gradient vectors; loops use [`ShareBuffers`] instead.
/// Panics when the dataset is not of the model's architecture.
pub fn share_eval(model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> ShareResult {
    let mut r = ShareResult { digest: [0; 32], ticket: [0; 32], indices: [0; NB], g1: Vec::new(), g2: Vec::new() };
    match &model.inner {
        Inner::V1(m) => {
            assert_eq!(ds.arch(), Arch::V1, "the dataset is not of the model's architecture");
            r.g1 = vec![0; N1];
            r.g2 = vec![0; N2];
            let cds = NebDataset { n: ds.n(), get: dataset_get, ctx: &ds as *const &dyn Dataset as *mut c_void };
            // SAFETY: the buffers have the sizes of pow/ref/shares.h (`N1` and `N2` gradients, 32-byte digest and
            // ticket, `NB` indices; a null `nclip` is allowed), and `cds.ctx` points to `ds`, which outlives the call.
            unsafe {
                neb_share_eval(
                    &**m,
                    &cds,
                    key.as_ptr(),
                    r.g1.as_mut_ptr(),
                    r.g2.as_mut_ptr(),
                    r.digest.as_mut_ptr(),
                    r.ticket.as_mut_ptr(),
                    r.indices.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
        }
        Inner::V2(arch, m) => {
            r.g1 = vec![0; arch.weights()];
            (r.digest, r.ticket) = v2::share_eval(m, *arch, ds, key, &mut r.g1, Some(&mut r.indices));
        }
    }
    r
}

/// Gradient scratch space that a thread allocates once and reuses for every
/// ticket it evaluates. One instance per thread: evaluation takes `&mut self`.
pub struct ShareBuffers {
    g1: Vec<i32>,
    g2: Vec<i32>,
    g: Vec<i32>,
}

impl ShareBuffers {
    pub fn new() -> ShareBuffers {
        ShareBuffers { g1: Vec::new(), g2: Vec::new(), g: Vec::new() }
    }

    /// The ticket of spec/shares.md section 5.4 for `key` on `model`.
    pub fn ticket(&mut self, model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> [u8; 32] {
        self.eval(model, ds, key).1
    }

    /// The gradient of the last evaluation in model order, as two slices for a v1 model (layers 1 and 2) and as the
    /// first slice for a v2 model.
    fn last(&self, arch: Arch) -> (&[i32], &[i32]) {
        if arch == Arch::V1 { (&self.g1, &self.g2) } else { (&self.g[..arch.weights()], &[]) }
    }

    fn eval(&mut self, model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
        match &model.inner {
            Inner::V1(m) => {
                assert_eq!(ds.arch(), Arch::V1, "the dataset is not of the model's architecture");
                if self.g1.len() != N1 {
                    self.g1 = vec![0; N1];
                    self.g2 = vec![0; N2];
                }
                let cds = NebDataset { n: ds.n(), get: dataset_get, ctx: &ds as *const &dyn Dataset as *mut c_void };
                let (mut digest, mut ticket) = ([0u8; 32], [0u8; 32]);
                // SAFETY: as in `share_eval`; the indices and `nclip` outputs are null, which the C code allows.
                unsafe {
                    neb_share_eval(
                        &**m,
                        &cds,
                        key.as_ptr(),
                        self.g1.as_mut_ptr(),
                        self.g2.as_mut_ptr(),
                        digest.as_mut_ptr(),
                        ticket.as_mut_ptr(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    )
                };
                (digest, ticket)
            }
            Inner::V2(arch, m) => {
                if self.g.len() < arch.weights() {
                    self.g = vec![0; v2::MAX_WEIGHTS];
                }
                v2::share_eval(m, *arch, ds, key, &mut self.g, None)
            }
        }
    }
}

impl Default for ShareBuffers {
    fn default() -> Self {
        ShareBuffers::new()
    }
}

/// The sum of the clipped gradients of the shares of one model step (spec/shares.md
/// section 6), all evaluated on models of one architecture. The int32 sums are exact only
/// while the number of shares is at most `MAX_SHARES`; `try_add` refuses a share beyond it
/// (a block lists at most 256, spec/node.md section 6.1).
pub struct GradSum {
    buffers: ShareBuffers,
    arch: Option<Arch>,
    gs: Vec<i32>,
    count: usize,
}

impl GradSum {
    pub fn new() -> GradSum {
        GradSum { buffers: ShareBuffers::new(), arch: None, gs: Vec::new(), count: 0 }
    }

    /// Number of shares added so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The sums in model order: for v1 the sums of layer 1 and layer 2 (`N1` and `N2` entries), for v2 every layer
    /// in the first slice. Both are empty before the first share.
    pub fn sums(&self) -> (&[i32], &[i32]) {
        match self.arch {
            Some(Arch::V1) => self.gs.split_at(N1),
            _ => (&self.gs, &[]),
        }
    }

    /// Evaluates one share on `model`, adds its gradients to the sum and returns its ticket. Fails, before
    /// evaluating anything, when the sum already holds [`MAX_SHARES`] shares, the largest count for which
    /// the int32 sums are exact. Panics when `model` is not of the architecture of the shares already added.
    pub fn try_add(&mut self, model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> Result<[u8; 32], StepError> {
        if self.count >= MAX_SHARES {
            return Err(StepError::TooManyShares(self.count + 1));
        }
        let arch = model.arch();
        match self.arch {
            None => {
                self.arch = Some(arch);
                self.gs = vec![0; arch.weights()];
            }
            Some(a) => assert_eq!(a, arch, "shares of two architectures in one step"),
        }
        let (_, ticket) = self.buffers.eval(model, ds, key);
        let (a, b) = self.buffers.last(arch);
        // `+=` cannot overflow while the share count is at most `MAX_SHARES` (bound derived at
        // `MAX_SHARES`), which the check above guarantees.
        for (s, g) in self.gs.iter_mut().zip(a.iter().chain(b)) {
            *s += g;
        }
        self.count += 1;
        Ok(ticket)
    }

    /// [`GradSum::try_add`] for callers that bound the count by construction (a block lists at most 256
    /// shares). Panics when the sum is full.
    pub fn add(&mut self, model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> [u8; 32] {
        match self.try_add(model, ds, key) {
            Ok(ticket) => ticket,
            Err(e) => panic!("gradient sum refused a share: {e}"),
        }
    }

    /// Applies the sum to `model` as one step, which may be a different model from the one the shares were
    /// evaluated on (spec/shares.md section 11.2). Without shares the model is unchanged.
    pub fn try_apply(&self, model: &mut Model) -> Result<(), StepError> {
        if self.count > 0 {
            model.try_apply_sums(&self.gs)?;
        }
        Ok(())
    }

    /// [`GradSum::try_apply`]; the sums of at most [`MAX_SHARES`] shares are always inside the exact domain
    /// of the step, so this cannot panic.
    pub fn apply(&self, model: &mut Model) {
        if let Err(e) = self.try_apply(model) {
            panic!("model step refused: {e}");
        }
    }
}

impl Default for GradSum {
    fn default() -> Self {
        GradSum::new()
    }
}

/// One model step as in the vectors: evaluates every share on `model`, then
/// applies the sum of their clipped gradients to the same `model`. The chain
/// evaluates on an older state than the one it updates and uses [`GradSum`]
/// for that. Fails unless `1 <= keys.len() <= MAX_SHARES`.
pub fn train_step(model: &mut Model, ds: &dyn Dataset, keys: &[[u8; 32]]) -> Result<Vec<ShareResult>, String> {
    if keys.is_empty() || keys.len() > MAX_SHARES {
        return Err(format!("a step has 1 to {MAX_SHARES} shares, got {}", keys.len()));
    }
    let results: Vec<ShareResult> = keys.iter().map(|k| share_eval(model, ds, k)).collect();
    let mut gs = vec![0i32; model.arch().weights()];
    for r in &results {
        for (s, g) in gs.iter_mut().zip(r.g1.iter().chain(&r.g2)) {
            *s += g;
        }
    }
    model.try_apply_sums(&gs).map_err(|e| e.to_string())?;
    Ok(results)
}

mod file;
mod v2;
mod vectors;
pub use file::{FileDataset, MAX_DATASETS, RECORD_BYTES, datasets_from_bytes, load_datasets};
pub use v2::Arch;
pub use vectors::{SyntheticDataset, SyntheticDataset2, replay_vectors, replay_vectors2};

#[cfg(test)]
mod tests;
