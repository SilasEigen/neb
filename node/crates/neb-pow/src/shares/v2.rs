//! nebhash v2 (`spec/nebhash-v2.md`): the convolutional architectures, wrapped from the C reference
//! `pow/ref/nebhash2.c`. The [`super::Model`] of a v2 architecture holds a [`V2Model`]; the share evaluation, the step and
//! the hashes go through the functions below.

use std::ffi::{c_int, c_void};

use super::{Dataset, NB};

/// The network a model and its dataset use: the 784-128-10 network of `spec/shares.md` or an architecture of
/// `spec/nebhash-v2.md` section 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    /// 784-128-10, `spec/shares.md`.
    V1,
    /// Kepler transit signals, 784 inputs, two outputs (section 3.1).
    Kep1,
    /// Retinal scans at 64 x 64, 4096 inputs, two outputs (section 3.2).
    Ret2,
}

impl Arch {
    /// The id of a v2 architecture in a `NMD2` dataset header (section 2), `None` for v1.
    pub fn id(self) -> Option<u32> {
        match self {
            Arch::V1 => None,
            Arch::Kep1 => Some(1),
            Arch::Ret2 => Some(2),
        }
    }

    pub fn from_id(id: u32) -> Option<Arch> {
        match id {
            1 => Some(Arch::Kep1),
            2 => Some(Arch::Ret2),
            _ => None,
        }
    }

    /// Input values of a sample.
    pub fn nin(self) -> usize {
        match self {
            Arch::V1 | Arch::Kep1 => 784,
            Arch::Ret2 => 4096,
        }
    }

    /// Classes, so the largest label is `nout() - 1`.
    pub fn nout(self) -> usize {
        match self {
            Arch::V1 => 10,
            Arch::Kep1 | Arch::Ret2 => 2,
        }
    }

    /// Weights (and residuals, and gradient entries) of a model.
    pub fn weights(self) -> usize {
        match self {
            Arch::V1 => super::N1 + super::N2,
            Arch::Kep1 => 26_000,
            Arch::Ret2 => 22_280,
        }
    }

    /// Bytes of the packed weights (`spec/shares.md` section 7, section 6 here).
    pub fn packed_bytes(self) -> usize {
        self.weights() / 2
    }

    pub fn name(self) -> &'static str {
        match self {
            Arch::V1 => "v1",
            Arch::Kep1 => "kep1",
            Arch::Ret2 => "ret2",
        }
    }
}

/// Largest weight count of a v2 architecture (`NEB2_MAX_WEIGHTS`).
pub(super) const MAX_WEIGHTS: usize = 26_000;

/// Mirror of `neb2_model` in `pow/ref/nebhash2.h`.
#[repr(C)]
pub(super) struct V2Model {
    pub(super) arch: i32,
    pub(super) w: [i8; MAX_WEIGHTS],
    pub(super) r: [i32; MAX_WEIGHTS],
}

const _: () = {
    assert!(std::mem::size_of::<V2Model>() == 4 + MAX_WEIGHTS + 4 * MAX_WEIGHTS);
    assert!(std::mem::offset_of!(V2Model, w) == 4);
    assert!(std::mem::offset_of!(V2Model, r) == 4 + MAX_WEIGHTS);
};

#[repr(C)]
struct V2Dataset {
    n: u32,
    get: extern "C" fn(*mut c_void, u32, *mut u8) -> c_int,
    ctx: *mut c_void,
}

unsafe extern "C" {
    fn neb2_model_init(arch: c_int, seed: *const u8, m: *mut V2Model) -> c_int;
    fn neb2_share_eval(
        m: *const V2Model,
        ds: *const V2Dataset,
        key: *const u8,
        g: *mut i32,
        digest: *mut u8,
        ticket: *mut u8,
        idx: *mut u32,
        nclip: *mut u32,
    ) -> c_int;
    fn neb2_model_apply(m: *mut V2Model, gs: *const i32);
    fn neb2_model_hashes(m: *const V2Model, weights_hash: *mut u8, residual_hash: *mut u8);
    fn neb2_model_predict(m: *const V2Model, x: *const u8) -> c_int;
}

/// Sample callback of the C code: `ctx` points to a `&dyn Dataset` on the caller's stack.
extern "C" fn dataset_sample(ctx: *mut c_void, index: u32, x: *mut u8) -> c_int {
    // SAFETY: `ctx` is the pointer built by `share_eval` below, to a `&dyn Dataset` that outlives the C call.
    let ds = unsafe { &*(ctx as *const &dyn Dataset) };
    let nin = ds.arch().nin();
    // SAFETY: the C code passes a buffer of `nin` bytes for the architecture of the model, which equals the dataset's
    // (checked by `share_eval`).
    let out = unsafe { std::slice::from_raw_parts_mut(x, nin) };
    let label = ds.sample(index, out);
    debug_assert!(out.iter().all(|&v| v <= 7) && (label as usize) < ds.arch().nout());
    label as c_int
}

/// A fresh boxed model with every field zero.
pub(super) fn zeroed() -> Box<V2Model> {
    let mut m = Box::<V2Model>::new_uninit();
    // SAFETY: `V2Model` holds integers only, so all-zero bytes are a valid value.
    unsafe {
        std::ptr::write_bytes(m.as_mut_ptr(), 0, 1);
        m.assume_init()
    }
}

pub(super) fn init(arch: Arch, seed: &[u8; 32]) -> Box<V2Model> {
    let mut m = zeroed();
    let id = arch.id().expect("a v2 architecture") as c_int;
    // SAFETY: `neb2_model_init` reads 32 bytes of `seed` and writes the model; the id is a known architecture.
    let rc = unsafe { neb2_model_init(id, seed.as_ptr(), &mut *m) };
    assert_eq!(rc, 0, "neb2_model_init failed");
    m
}

/// Evaluates a share: fills `g` (the architecture's weight count), returns `(digest, ticket)` and writes the
/// indices when `idx` is given. Panics when the dataset is not of the model's architecture or memory runs out.
pub(super) fn share_eval(
    m: &V2Model,
    arch: Arch,
    ds: &dyn Dataset,
    key: &[u8; 32],
    g: &mut [i32],
    idx: Option<&mut [u32; NB]>,
) -> ([u8; 32], [u8; 32]) {
    assert_eq!(ds.arch(), arch, "the dataset is not of the model's architecture");
    assert!(g.len() >= arch.weights());
    let cds = V2Dataset { n: ds.n(), get: dataset_sample, ctx: &ds as *const &dyn Dataset as *mut c_void };
    let (mut digest, mut ticket) = ([0u8; 32], [0u8; 32]);
    let idx_ptr = idx.map_or(std::ptr::null_mut(), |i| i.as_mut_ptr());
    // SAFETY: `g` holds at least the model's weight count, the digest and ticket are 32 bytes, `idx_ptr` is null or
    // `NB` entries, and `cds.ctx` points to `ds`, which outlives the call.
    let rc = unsafe {
        neb2_share_eval(
            m,
            &cds,
            key.as_ptr(),
            g.as_mut_ptr(),
            digest.as_mut_ptr(),
            ticket.as_mut_ptr(),
            idx_ptr,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0, "neb2_share_eval failed");
    (digest, ticket)
}

pub(super) fn apply(m: &mut V2Model, gs: &[i32]) {
    // SAFETY: the caller passes the sums of the model's weight count (checked in `Model::try_apply_sums`).
    unsafe { neb2_model_apply(m, gs.as_ptr()) }
}

pub(super) fn hashes(m: &V2Model) -> ([u8; 32], [u8; 32]) {
    let (mut w, mut r) = ([0u8; 32], [0u8; 32]);
    // SAFETY: the C code writes 32 bytes to each output.
    unsafe { neb2_model_hashes(m, w.as_mut_ptr(), r.as_mut_ptr()) };
    (w, r)
}

pub(super) fn predict(m: &V2Model, x: &[u8]) -> u8 {
    // SAFETY: the caller passes the architecture's `nin` values (checked in `Model::predict_sample`).
    unsafe { neb2_model_predict(m, x.as_ptr()) as u8 }
}
