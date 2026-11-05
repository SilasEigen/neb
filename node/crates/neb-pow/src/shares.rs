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

/// The samples a share draws from (spec/shares.md section 4): `n() >= 1`
/// samples, each `NIN` pixels in 0..=7 and a label in 0..=9.
///
/// `get` is called with `index < n()` only. It must be a pure function of
/// `index`, because every node must derive the same samples. The ranges of
/// pixels and label are asserted in debug builds only.
pub trait Dataset: Send + Sync + std::fmt::Debug {
    fn n(&self) -> u32;
    fn get(&self, index: u32) -> ([u8; NIN], u8);
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

/// The weights `W1`, `W2` in -8..=7 and the residuals `R1`, `R2` (spec/shares.md
/// section 4). Cloning copies all 508160 bytes.
pub struct Model {
    inner: Box<NebModel>,
}

impl Clone for Model {
    fn clone(&self) -> Model {
        // Copy between two heap blocks; a by-value copy would put 508160 bytes on the stack.
        let mut inner = Box::<NebModel>::new_uninit();
        // SAFETY: one `NebModel` is copied from an initialised box into a fresh allocation of the same type, which
        // initialises it; `NebModel` holds integers only.
        let inner = unsafe {
            std::ptr::copy_nonoverlapping(&*self.inner, inner.as_mut_ptr(), 1);
            inner.assume_init()
        };
        Model { inner }
    }
}

impl Model {
    /// The packed weights of spec/shares.md section 7: `pack(W1) || pack(W2)`. Weight `e` of a
    /// matrix is the low nibble of byte `e / 2` for even `e` and the high nibble
    /// for odd `e` (spec/shares.md section 3). `N1` and `N2` are even, so packing the
    /// weights as one sequence gives the two matrices one after the other.
    pub fn packed_weights(&self) -> Vec<u8> {
        let nibbles = self.inner.w1.iter().chain(self.inner.w2.iter());
        let mut out = Vec::with_capacity(PACKED_WEIGHTS);
        let mut pending = None;
        for w in nibbles {
            let nibble = (*w as u8) & 15;
            match pending.take() {
                None => pending = Some(nibble),
                Some(low) => out.push(low | (nibble << 4)),
            }
        }
        out
    }

    /// A model with these packed weights and zero residuals: enough to
    /// evaluate shares, because the ticket depends on the weights only (the
    /// residuals matter only when a step is applied). Fails
    /// on a length other than `PACKED_WEIGHTS`. Every nibble is a valid weight,
    /// so the caller compares the weights hash to detect a damaged frame.
    pub fn from_packed_weights(bytes: &[u8]) -> Result<Model, String> {
        if bytes.len() != PACKED_WEIGHTS {
            return Err(format!("expected {PACKED_WEIGHTS} packed weight bytes, got {}", bytes.len()));
        }
        let mut inner = Box::<NebModel>::new_uninit();
        // All fields zero: the residuals stay zero, the weights are set below.
        // SAFETY: `NebModel` holds integers only, so all-zero bytes are a valid value.
        let mut inner = unsafe {
            std::ptr::write_bytes(inner.as_mut_ptr(), 0, 1);
            inner.assume_init()
        };
        // Sign extension of a 4-bit two's complement value.
        let signed = |nibble: u8| ((nibble << 4) as i8) >> 4;
        for (j, b) in bytes.iter().enumerate() {
            for (k, nibble) in [b & 15, b >> 4].into_iter().enumerate() {
                let e = 2 * j + k;
                if e < N1 {
                    inner.w1[e] = signed(nibble);
                } else {
                    inner.w2[e - N1] = signed(nibble);
                }
            }
        }
        Ok(Model { inner })
    }

    /// Weights of layer 1, `NH` rows of `NIN`, row-major.
    pub fn w1(&self) -> &[i8] {
        &self.inner.w1
    }

    /// Weights of layer 2, `NOUT` rows of `NH`, row-major.
    pub fn w2(&self) -> &[i8] {
        &self.inner.w2
    }

    /// Residuals of layer 1 (same layout as [`Model::w1`]).
    pub fn r1(&self) -> &[i32] {
        &self.inner.r1
    }

    /// Residuals of layer 2 (same layout as [`Model::w2`]).
    pub fn r2(&self) -> &[i32] {
        &self.inner.r2
    }

    /// A model with exactly this state. Fails on a wrong length, a weight
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
        Ok(Model { inner })
    }

    /// The initial state of a 32-byte seed (spec/shares.md section 4); on the
    /// chain the seed is the genesis block id.
    pub fn from_seed(seed: &[u8; 32]) -> Model {
        let mut inner = Box::<NebModel>::new_uninit();
        // SAFETY: `neb_model_init` reads 32 bytes of `seed` and writes every field of the model.
        let inner = unsafe {
            neb_model_init(seed.as_ptr(), inner.as_mut_ptr());
            inner.assume_init()
        };
        Model { inner }
    }

    /// The model step of spec/shares.md section 6. `gs1` and `gs2` are the sums of the
    /// clipped gradients of the shares of the step (at most [`MAX_SHARES`], which keeps the int32 sums
    /// exact: [`train_step`] and [`GradSum`] check it). Fails, without changing the model, when a slice has
    /// the wrong length or when `residual + sum` of an entry leaves the exact domain of the C step. The check
    /// is a pass over the 101632 entries and changes no result inside the domain.
    pub fn try_apply_step(&mut self, gs1: &[i32], gs2: &[i32]) -> Result<(), StepError> {
        if gs1.len() != N1 {
            return Err(StepError::Length { layer: 1, expected: N1, got: gs1.len() });
        }
        if gs2.len() != N2 {
            return Err(StepError::Length { layer: 2, expected: N2, got: gs2.len() });
        }
        for (layer, residuals, sums) in [(1u8, &self.inner.r1[..], gs1), (2u8, &self.inner.r2[..], gs2)] {
            for (index, (&residual, &sum)) in residuals.iter().zip(sums).enumerate() {
                if (residual as i64 + sum as i64).unsigned_abs() > i32::MAX as u64 {
                    return Err(StepError::OutOfRange { layer, index, residual, sum });
                }
            }
        }
        // SAFETY: `gs1` and `gs2` have `N1` and `N2` entries (checked above), the lengths the C step reads.
        unsafe { neb_model_apply(&mut *self.inner, gs1.as_ptr(), gs2.as_ptr()) };
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
        let (mut w, mut r) = ([0u8; 32], [0u8; 32]);
        // SAFETY: the C code writes 32 bytes to each output buffer.
        unsafe { neb_model_hashes(&*self.inner, w.as_mut_ptr(), r.as_mut_ptr()) };
        (w, r)
    }

    pub fn weights_hash(&self) -> [u8; 32] {
        self.hashes().0
    }

    pub fn residual_hash(&self) -> [u8; 32] {
        self.hashes().1
    }

    /// The class of a sample: the output with the largest sum, the smallest
    /// index on a tie. Used to report accuracy; the protocol does not use it.
    /// Integer arithmetic, so the result is the same on every platform.
    pub fn predict(&self, x: &[u8; NIN]) -> u8 {
        // SAFETY: the C code reads the `NIN` bytes of `x`.
        unsafe { neb_model_predict(&*self.inner, x.as_ptr()) as u8 }
    }
}

/// One evaluated share (spec/shares.md sections 5.1 to 5.4): the clipped
/// gradients (`N1` and `N2` entries), the gradient digest, the ticket and the
/// sample indices. The ticket is the BLAKE3 output as bytes; it is compared with
/// a target as a 256-bit little-endian integer.
pub struct ShareResult {
    pub digest: [u8; 32],
    pub ticket: [u8; 32],
    pub indices: [u32; NB],
    pub g1: Vec<i32>,
    pub g2: Vec<i32>,
}

/// Evaluates the share of `key` on `model`. In the proof of work `key` is the
/// BLAKE3 hash of the header of the attempt. Allocates the gradient vectors
/// (406528 bytes); loops use [`ShareBuffers`] instead.
pub fn share_eval(model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> ShareResult {
    let cds = NebDataset { n: ds.n(), get: dataset_get, ctx: &ds as *const &dyn Dataset as *mut c_void };
    let mut r = ShareResult { digest: [0; 32], ticket: [0; 32], indices: [0; NB], g1: vec![0; N1], g2: vec![0; N2] };
    // SAFETY: the buffers have the sizes of pow/ref/shares.h (`N1` and `N2` gradients, 32-byte digest and ticket, `NB`
    // indices; a null `nclip` is allowed), and `cds.ctx` points to `ds`, which outlives the call.
    unsafe {
        neb_share_eval(
            &*model.inner,
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
    r
}

/// Gradient scratch space that a thread allocates once and reuses for every
/// ticket it evaluates. One instance per thread: evaluation takes `&mut self`.
pub struct ShareBuffers {
    g1: Vec<i32>,
    g2: Vec<i32>,
}

impl ShareBuffers {
    pub fn new() -> ShareBuffers {
        ShareBuffers { g1: vec![0; N1], g2: vec![0; N2] }
    }

    /// The ticket of spec/shares.md section 5.4 for `key` on `model`.
    pub fn ticket(&mut self, model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> [u8; 32] {
        self.eval(model, ds, key).1
    }

    fn eval(&mut self, model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
        let cds = NebDataset { n: ds.n(), get: dataset_get, ctx: &ds as *const &dyn Dataset as *mut c_void };
        let (mut digest, mut ticket) = ([0u8; 32], [0u8; 32]);
        // SAFETY: as in `share_eval`; the indices and `nclip` outputs are null, which the C code allows.
        unsafe {
            neb_share_eval(
                &*model.inner,
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
}

impl Default for ShareBuffers {
    fn default() -> Self {
        ShareBuffers::new()
    }
}

/// The sum of the clipped gradients of the shares of one model step (spec/shares.md
/// section 6), all evaluated on the same model. The int32 sums are exact only
/// while the number of shares is at most `MAX_SHARES`; `try_add` refuses a share beyond it
/// (a block lists at most 256, spec/node.md section 6.1).
pub struct GradSum {
    buffers: ShareBuffers,
    gs1: Vec<i32>,
    gs2: Vec<i32>,
    count: usize,
}

impl GradSum {
    pub fn new() -> GradSum {
        GradSum { buffers: ShareBuffers::new(), gs1: vec![0; N1], gs2: vec![0; N2], count: 0 }
    }

    /// Number of shares added so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The sums of the clipped gradients of layer 1 and layer 2 (`N1` and `N2` entries).
    pub fn sums(&self) -> (&[i32], &[i32]) {
        (&self.gs1, &self.gs2)
    }

    /// Evaluates one share on `model`, adds its gradients to the sum and returns its ticket. Fails, before
    /// evaluating anything, when the sum already holds [`MAX_SHARES`] shares, the largest count for which
    /// the int32 sums are exact.
    pub fn try_add(&mut self, model: &Model, ds: &dyn Dataset, key: &[u8; 32]) -> Result<[u8; 32], StepError> {
        if self.count >= MAX_SHARES {
            return Err(StepError::TooManyShares(self.count + 1));
        }
        let (_, ticket) = self.buffers.eval(model, ds, key);
        // `+=` cannot overflow while the share count is at most `MAX_SHARES` (bound derived at
        // `MAX_SHARES`), which the check above guarantees.
        for (s, g) in self.gs1.iter_mut().zip(&self.buffers.g1) {
            *s += g;
        }
        for (s, g) in self.gs2.iter_mut().zip(&self.buffers.g2) {
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
            model.try_apply_step(&self.gs1, &self.gs2)?;
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
    let mut gs1 = vec![0i32; N1];
    let mut gs2 = vec![0i32; N2];
    for r in &results {
        for (s, g) in gs1.iter_mut().zip(&r.g1) {
            *s += g;
        }
        for (s, g) in gs2.iter_mut().zip(&r.g2) {
            *s += g;
        }
    }
    model.apply_step(&gs1, &gs2);
    Ok(results)
}

/// Keystream of ChaCha8 with the given key, zero IV and block counter from 0.
fn chacha8(key: &[u8; 32], out: &mut [u8]) {
    // SAFETY: the C code reads 32 bytes of `key` and writes `out.len()` bytes to `out`.
    unsafe { chacha8_stream(key.as_ptr(), out.as_mut_ptr(), out.len()) };
}

/// The test dataset of spec/shares.md section 9: `SYN_N` samples built from ten
/// class prototypes, label `i mod 10`. It exists to produce the test vectors and
/// to run the demo and test chains.
#[derive(Debug)]
pub struct SyntheticDataset {
    protos: [[u8; NIN]; 10],
}

impl Default for SyntheticDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticDataset {
    pub fn new() -> SyntheticDataset {
        let mut protos = [[0u8; NIN]; 10];
        for (c, p) in protos.iter_mut().enumerate() {
            let mut input = [0u8; 14];
            input[..13].copy_from_slice(b"neb-syn-proto");
            // The class number as one byte (spec/shares.md section 9).
            input[13] = c as u8;
            chacha8(blake3::hash(&input).as_bytes(), p);
            p.iter_mut().for_each(|v| *v &= 7);
        }
        SyntheticDataset { protos }
    }
}

impl Dataset for SyntheticDataset {
    fn n(&self) -> u32 {
        SYN_N
    }

    fn get(&self, index: u32) -> ([u8; NIN], u8) {
        let mut input = [0u8; 18];
        input[..14].copy_from_slice(b"neb-syn-sample");
        input[14..].copy_from_slice(&index.to_le_bytes());
        let mut noise = [0u8; NIN];
        chacha8(blake3::hash(&input).as_bytes(), &mut noise);
        let class = index % 10;
        let proto = &self.protos[class as usize];
        // Prototype pixel plus noise in -1..=2, clipped to 0..=7.
        let x = std::array::from_fn(|p| (proto[p] as i32 + (noise[p] & 3) as i32 - 1).clamp(0, 7) as u8);
        (x, class as u8)
    }
}

/// A dataset stored in a file (spec/shares.md section 11.6): the magic `NMD1`,
/// the sample count `N >= 1` as a little-endian u32, then `N` records of `NIN`
/// pixels (0..=7) and a label (0..=9). The network parameters pin the BLAKE3
/// hash of the file.
pub struct FileDataset {
    n: u32,
    records: Vec<u8>,
}

/// Bytes of one record of a dataset file.
pub const RECORD_BYTES: usize = NIN + 1;
const FILE_MAGIC: &[u8; 4] = b"NMD1";

impl FileDataset {
    /// Checks the format only, not the hash (see [`FileDataset::load`]): magic,
    /// `N >= 1`, an exact size of `8 + 785 * N` bytes, then every pixel and
    /// label.
    pub fn from_bytes(bytes: &[u8]) -> Result<FileDataset, String> {
        if bytes.len() < 8 || &bytes[..4] != FILE_MAGIC {
            return Err("not a dataset file (bad magic)".into());
        }
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let records = &bytes[8..];
        if n == 0 || records.len() != n as usize * RECORD_BYTES {
            return Err(format!(
                "the file has {} record bytes, {n} samples need {}",
                records.len(),
                n as usize * RECORD_BYTES
            ));
        }
        if records.chunks(RECORD_BYTES).any(|r| r[..NIN].iter().any(|&p| p > 7) || r[NIN] > 9) {
            return Err("a sample has a pixel above 7 or a label above 9".into());
        }
        Ok(FileDataset { n, records: records.to_vec() })
    }

    /// Reads a dataset file. When `pinned` is set, the BLAKE3 hash of the file
    /// must equal it; the hash is checked before the content is interpreted.
    pub fn load(path: &std::path::Path, pinned: Option<&[u8; 32]>) -> Result<FileDataset, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if let Some(want) = pinned {
            let got = blake3::hash(&bytes);
            if got.as_bytes() != want {
                return Err(format!("{} has the wrong hash {}", path.display(), got.to_hex()));
            }
        }
        FileDataset::from_bytes(&bytes)
    }
}

impl std::fmt::Debug for FileDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FileDataset({} samples)", self.n)
    }
}

impl Dataset for FileDataset {
    fn n(&self) -> u32 {
        self.n
    }

    /// Panics when `index >= n()`, which a share never requests.
    fn get(&self, index: u32) -> ([u8; NIN], u8) {
        let r = &self.records[index as usize * RECORD_BYTES..][..RECORD_BYTES];
        (r[..NIN].try_into().unwrap(), r[NIN])
    }
}

/// The key of share `j` of step `s` in the test vectors (spec/shares.md section 10):
/// `BLAKE3("neb-test-share" || LE32(s) || LE32(j))`.
fn test_share_key(s: u32, j: u32) -> [u8; 32] {
    let mut input = [0u8; 22];
    input[..14].copy_from_slice(b"neb-test-share");
    input[14..18].copy_from_slice(&s.to_le_bytes());
    input[18..].copy_from_slice(&j.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

fn num<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("bad number {s:?}"))
}

/// Replays a `neb-shares-vectors 1` file (spec/shares.md section 10) from the
/// zero seed on the synthetic dataset. The `init` line, every share (indices,
/// digest, ticket) and the state hashes after every step must match. Returns
/// the number of steps and of shares checked; the first mismatch is an error
/// that names the line, or the step and share.
pub fn replay_vectors(text: &str) -> Result<(usize, usize), String> {
    let ds = SyntheticDataset::new();
    let mut model = Model::from_seed(&[0u8; 32]);
    let mut lines = text.lines().enumerate().map(|(n, l)| (n + 1, l.trim())).filter(|(_, l)| !l.is_empty());

    let (_, first) = lines.next().ok_or("empty file")?;
    if first != "neb-shares-vectors 1" {
        return Err(format!("unknown header {first:?}"));
    }
    let (n, init) = lines.next().ok_or("missing init line")?;
    let f: Vec<&str> = init.split(' ').collect();
    if f.len() != 3 || f[0] != "init" {
        return Err(format!("line {n}: expected an init line"));
    }
    if (hex_bytes::<32>(f[1])?, hex_bytes::<32>(f[2])?) != model.hashes() {
        return Err(format!("line {n}: init hash mismatch"));
    }

    let mut keys = Vec::new();
    let mut want: Vec<([u8; 32], [u8; 32], [u32; NB])> = Vec::new();
    let (mut steps, mut shares) = (0usize, 0usize);
    for (n, line) in lines {
        let f: Vec<&str> = line.split(' ').collect();
        match f[0] {
            "share" if f.len() == 5 + NB => {
                if num::<usize>(f[1])? != steps || num::<usize>(f[2])? != keys.len() {
                    return Err(format!("line {n}: share out of order"));
                }
                let mut idx = [0u32; NB];
                for (slot, s) in idx.iter_mut().zip(&f[5..]) {
                    *slot = num(s)?;
                }
                // Keys follow from the position, so the file carries none.
                keys.push(test_share_key(steps as u32, keys.len() as u32));
                want.push((hex_bytes(f[3])?, hex_bytes(f[4])?, idx));
            }
            "state" if f.len() == 5 => {
                if num::<usize>(f[1])? != steps || num::<usize>(f[2])? != keys.len() {
                    return Err(format!("line {n}: state out of order"));
                }
                // The shares of a step are evaluated on the state left by the
                // previous step, then applied together.
                let got = train_step(&mut model, &ds, &keys)?;
                for (j, (r, w)) in got.iter().zip(&want).enumerate() {
                    if r.indices != w.2 {
                        return Err(format!("step {steps} share {j}: indices mismatch"));
                    }
                    if r.digest != w.0 {
                        return Err(format!("step {steps} share {j}: digest mismatch"));
                    }
                    if r.ticket != w.1 {
                        return Err(format!("step {steps} share {j}: ticket mismatch"));
                    }
                }
                if (hex_bytes::<32>(f[3])?, hex_bytes::<32>(f[4])?) != model.hashes() {
                    return Err(format!("line {n}: state hash mismatch after step {steps}"));
                }
                steps += 1;
                shares += keys.len();
                keys.clear();
                want.clear();
            }
            _ => return Err(format!("line {n}: unexpected line")),
        }
    }
    if !keys.is_empty() {
        return Err("the last step has no state line".into());
    }
    Ok((steps, shares))
}

#[cfg(test)]
mod tests;
