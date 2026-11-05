//! Safe wrapper of the AVX-512 evaluation of training shares (`pow/cpu/shares_avx512.c`).
//!
//! The C code computes the same digest, ticket and sample indices as the reference `neb_share_eval`, and the same
//! clipped gradients, with AVX-512 VNNI instructions. A [`PreparedWeights`] and a [`Scratch`] can be built only
//! when [`available`] is true, so the evaluation functions never run on a processor without the instructions.

use neb_pow::shares::{Dataset, Model, N1, N2, NB, NIN};

/// True when this build contains the AVX-512 unit (x86-64 and a compiler that accepts the flags).
pub fn compiled_in() -> bool {
    cfg!(neb_avx512)
}

/// True when the processor and the operating system provide AVX-512 F, BW, VL, DQ and VNNI.
pub fn cpu_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The engine can run here: it was compiled in and the processor has the instructions.
pub fn available() -> bool {
    compiled_in() && cpu_supported()
}

/// Why the engine cannot run, for the messages of the miners.
pub fn unavailable_reason() -> Option<&'static str> {
    if !compiled_in() {
        Some("this build has no AVX-512 engine (not x86-64, or the compiler refused the flags)")
    } else if !cpu_supported() {
        Some("this processor lacks AVX-512 F, BW, VL, DQ or VNNI")
    } else {
        None
    }
}

/// Digest, ticket and sample indices of one share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evaluation {
    pub digest: [u8; 32],
    pub ticket: [u8; 32],
    pub indices: [u32; NB],
}

#[cfg(neb_avx512)]
mod ffi {
    use std::ffi::c_int;

    #[repr(C)]
    pub struct CDataset {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct CModel {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct CScratch {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn neb_cpu_dataset_new(pixels: *const u8, labels: *const u8, n: u32) -> *mut CDataset;
        pub fn neb_cpu_dataset_free(d: *mut CDataset);
        pub fn neb_cpu_model_new(w1: *const i8, w2: *const i8) -> *mut CModel;
        pub fn neb_cpu_model_free(m: *mut CModel);
        pub fn neb_cpu_scratch_new() -> *mut CScratch;
        pub fn neb_cpu_scratch_free(s: *mut CScratch);
        pub fn neb_cpu_share_eval(
            m: *const CModel,
            s: *mut CScratch,
            d: *const CDataset,
            key: *const u8,
            sparse: c_int,
            digest: *mut u8,
            ticket: *mut u8,
            idx: *mut u32,
            g1: *mut i32,
            g2: *mut i32,
        );
    }
}

fn check_available() -> Result<(), String> {
    match unavailable_reason() {
        None => Ok(()),
        Some(why) => Err(why.to_string()),
    }
}

/// The samples of a dataset in the flat layout of the C code. Read-only after creation.
pub struct FlatDataset {
    #[cfg(neb_avx512)]
    raw: *mut ffi::CDataset,
    n: u32,
}

// SAFETY: the C object is never written after `neb_cpu_dataset_new`, and pow/cpu/shares_avx512.h allows threads to
// share it.
unsafe impl Send for FlatDataset {}
// SAFETY: see `Send` above.
unsafe impl Sync for FlatDataset {}

impl FlatDataset {
    pub fn new(dataset: &dyn Dataset) -> Result<FlatDataset, String> {
        check_available()?;
        let n = dataset.n();
        let (mut pixels, mut labels) = (Vec::with_capacity(n as usize * NIN), Vec::with_capacity(n as usize));
        for i in 0..n {
            let (x, label) = dataset.get(i);
            pixels.extend_from_slice(&x);
            labels.push(label);
        }
        #[cfg(neb_avx512)]
        {
            // SAFETY: `check_available` passed; `pixels` holds `n * NIN` bytes and `labels` `n` bytes, which the C code
            // copies, and a `Dataset` has `n >= 1`.
            let raw = unsafe { ffi::neb_cpu_dataset_new(pixels.as_ptr(), labels.as_ptr(), n) };
            if raw.is_null() {
                return Err("out of memory for the flat dataset".into());
            }
            Ok(FlatDataset { raw, n })
        }
        #[cfg(not(neb_avx512))]
        {
            let _ = (pixels, labels, n);
            Err("no AVX-512 engine in this build".into())
        }
    }

    pub fn n(&self) -> u32 {
        self.n
    }
}

impl Drop for FlatDataset {
    fn drop(&mut self) {
        #[cfg(neb_avx512)]
        // SAFETY: `raw` came from `neb_cpu_dataset_new`, is not null and is freed once.
        unsafe {
            ffi::neb_cpu_dataset_free(self.raw)
        };
    }
}

/// The weights of a model repacked for the instructions. Read-only after creation.
pub struct PreparedWeights {
    #[cfg(neb_avx512)]
    raw: *mut ffi::CModel,
}

// SAFETY: the C object is never written after `neb_cpu_model_new`, and pow/cpu/shares_avx512.h allows threads to
// share it.
unsafe impl Send for PreparedWeights {}
// SAFETY: see `Send` above.
unsafe impl Sync for PreparedWeights {}

impl PreparedWeights {
    pub fn new(model: &Model) -> Result<PreparedWeights, String> {
        check_available()?;
        debug_assert_eq!((model.w1().len(), model.w2().len()), (N1, N2));
        #[cfg(neb_avx512)]
        {
            // SAFETY: `check_available` passed, and the weights of a `Model` have `N1` and `N2` entries.
            let raw = unsafe { ffi::neb_cpu_model_new(model.w1().as_ptr(), model.w2().as_ptr()) };
            if raw.is_null() {
                return Err("out of memory for the prepared weights".into());
            }
            Ok(PreparedWeights { raw })
        }
        #[cfg(not(neb_avx512))]
        {
            let _ = model;
            Err("no AVX-512 engine in this build".into())
        }
    }
}

impl Drop for PreparedWeights {
    fn drop(&mut self) {
        #[cfg(neb_avx512)]
        // SAFETY: `raw` came from `neb_cpu_model_new`, is not null and is freed once.
        unsafe {
            ffi::neb_cpu_model_free(self.raw)
        };
    }
}

/// Working memory of one thread.
pub struct Scratch {
    #[cfg(neb_avx512)]
    raw: *mut ffi::CScratch,
}

// SAFETY: a scratch buffer belongs to one thread at a time; it may move between threads, and it is not `Sync`.
unsafe impl Send for Scratch {}

impl Scratch {
    pub fn new() -> Result<Scratch, String> {
        check_available()?;
        #[cfg(neb_avx512)]
        {
            // SAFETY: `check_available` passed.
            let raw = unsafe { ffi::neb_cpu_scratch_new() };
            if raw.is_null() {
                return Err("out of memory for the scratch buffer".into());
            }
            Ok(Scratch { raw })
        }
        #[cfg(not(neb_avx512))]
        Err("no AVX-512 engine in this build".into())
    }

    /// One share. `sparse` selects the variant that skips the zero entries; both give the same bits.
    pub fn eval(
        &mut self,
        weights: &PreparedWeights,
        dataset: &FlatDataset,
        key: &[u8; 32],
        sparse: bool,
    ) -> Evaluation {
        self.run(weights, dataset, key, sparse, None)
    }

    /// As [`Scratch::eval`], also writing the clipped gradients (`N1` and `N2` entries).
    pub fn eval_with_gradients(
        &mut self,
        weights: &PreparedWeights,
        dataset: &FlatDataset,
        key: &[u8; 32],
        sparse: bool,
        g1: &mut [i32],
        g2: &mut [i32],
    ) -> Evaluation {
        assert!(g1.len() == N1 && g2.len() == N2, "gradient buffers have the wrong length");
        self.run(weights, dataset, key, sparse, Some((g1, g2)))
    }

    #[cfg_attr(not(neb_avx512), allow(unused_variables))]
    fn run(
        &mut self,
        weights: &PreparedWeights,
        dataset: &FlatDataset,
        key: &[u8; 32],
        sparse: bool,
        grads: Option<(&mut [i32], &mut [i32])>,
    ) -> Evaluation {
        let mut out = Evaluation { digest: [0; 32], ticket: [0; 32], indices: [0; NB] };
        #[cfg(neb_avx512)]
        {
            let (g1, g2) = match grads {
                Some((a, b)) => (a.as_mut_ptr(), b.as_mut_ptr()),
                None => (std::ptr::null_mut(), std::ptr::null_mut()),
            };
            // SAFETY: `self`, `weights` and `dataset` exist only if `available()` was true when they were built, so their
            // pointers are valid; `&mut self` gives this thread the scratch buffer; the outputs have the sizes of
            // pow/cpu/shares_avx512.h, and the gradients are `N1` and `N2` long (checked by the caller) or null.
            unsafe {
                ffi::neb_cpu_share_eval(
                    weights.raw,
                    self.raw,
                    dataset.raw,
                    key.as_ptr(),
                    sparse as std::ffi::c_int,
                    out.digest.as_mut_ptr(),
                    out.ticket.as_mut_ptr(),
                    out.indices.as_mut_ptr(),
                    g1,
                    g2,
                )
            };
        }
        out
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        #[cfg(neb_avx512)]
        // SAFETY: `raw` came from `neb_cpu_scratch_new`, is not null and is freed once.
        unsafe {
            ffi::neb_cpu_scratch_free(self.raw)
        };
    }
}
