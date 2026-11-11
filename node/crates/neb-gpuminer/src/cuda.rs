//! The CUDA device behind [`crate::Scanner`] (`pow/gpu/scan_cuda.h`).

use std::ffi::{CStr, c_char, c_int};

use crate::{GpuKernel, Scanner, check_scanner, flatten, scan_complete, unpack_weights};
use neb_pow::shares::{Arch, Dataset, Model, N1, N2, NIN, SyntheticDataset};

#[repr(C)]
struct Raw {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn neb_scan_device_count(count: *mut c_int) -> c_int;
    fn neb_scan_device_name(device: c_int, name: *mut c_char, len: c_int) -> c_int;
    fn neb_scan_device_bus(device: c_int, bus: *mut c_int) -> c_int;
    fn neb_scan_device_capability(device: c_int, major: *mut c_int, minor: *mut c_int) -> c_int;
    fn neb_scan_kernel_supported(device: c_int, kernel: c_int) -> c_int;
    fn neb_scan_open_kernel(
        device: c_int,
        kernel: c_int,
        pixels: *const u8,
        labels: *const u8,
        n: u32,
        w1: *const i8,
        w2: *const i8,
        capacity: u32,
        max_hits: u32,
        out: *mut *mut Raw,
    ) -> c_int;
    fn neb_scan_set_model(s: *mut Raw, w1: *const i8, w2: *const i8) -> c_int;
    fn neb_scan_run(
        s: *mut Raw,
        header: *const u8,
        start: u32,
        count: u32,
        bound: *const u8,
        hits: *mut u32,
        nhits: *mut u32,
    ) -> c_int;
    fn neb_scan_close(s: *mut Raw);
    fn neb_scan_error(status: c_int) -> *const c_char;
}

fn check(status: c_int, what: &str) -> Result<(), String> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: `neb_scan_error` returns `cudaGetErrorString`, a static NUL-terminated string.
    let msg = unsafe { CStr::from_ptr(neb_scan_error(status)) }.to_string_lossy().into_owned();
    Err(format!("{what}: {msg} (CUDA status {status})"))
}

/// Number of CUDA devices.
pub fn device_count() -> Result<usize, String> {
    let mut n = 0;
    // SAFETY: the call writes one int to `n`.
    check(unsafe { neb_scan_device_count(&mut n) }, "cannot list the CUDA devices")?;
    Ok(n.max(0) as usize)
}

pub fn device_name(device: usize) -> Result<String, String> {
    let mut buf = [0 as c_char; 256];
    check(
        // SAFETY: the call writes a NUL-terminated string of at most `buf.len()` bytes to `buf`.
        unsafe { neb_scan_device_name(device as c_int, buf.as_mut_ptr(), buf.len() as c_int) },
        "cannot read the device name",
    )?;
    // SAFETY: on success `buf` holds a NUL-terminated string (see above).
    Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
}

/// The PCI bus number of a device (0a:00.0 is 10), which HiveOS uses to tell the cards apart.
pub fn device_bus(device: usize) -> Result<u32, String> {
    let mut bus = 0;
    // SAFETY: the call writes one int to `bus`.
    check(unsafe { neb_scan_device_bus(device as c_int, &mut bus) }, "cannot read the PCI bus of the device")?;
    Ok(bus.max(0) as u32)
}

/// The compute capability of a device, for example `(8, 6)`.
pub fn device_capability(device: usize) -> Result<(u32, u32), String> {
    let (mut major, mut minor) = (0, 0);
    check(
        // SAFETY: the call writes one int to `major` and one to `minor`.
        unsafe { neb_scan_device_capability(device as c_int, &mut major, &mut minor) },
        "cannot read the compute capability",
    )?;
    Ok((major.max(0) as u32, minor.max(0) as u32))
}

/// True when `kernel` can run on the device (K1 and K2 need compute capability 8.0 or later).
pub fn kernel_supported(device: usize, kernel: GpuKernel) -> bool {
    // SAFETY: the call takes two integers and dereferences no pointer.
    unsafe { neb_scan_kernel_supported(device as c_int, kernel.code()) != 0 }
}

/// One CUDA device holding the dataset and the model.
pub struct CudaScanner {
    raw: *mut Raw,
    capacity: u32,
    hits: Vec<u32>,
    kernel: GpuKernel,
}

// SAFETY: pow/gpu/scan_cuda.h allows a handle to move between threads as long as one thread uses it at a time, which
// `&mut self` ensures, and every call selects the device first.
unsafe impl Send for CudaScanner {}

impl CudaScanner {
    /// Uploads the dataset (`pixels` n x 784, `labels` n) and the model to `device`, for the kernel K0. A scan covers at
    /// most `capacity` nonces; a range with more than `max_hits` hits is split, so no hit is lost.
    pub fn open(
        device: usize,
        pixels: &[u8],
        labels: &[u8],
        w1: &[i8],
        w2: &[i8],
        capacity: u32,
        max_hits: u32,
    ) -> Result<CudaScanner, String> {
        CudaScanner::open_kernel(device, GpuKernel::K0, pixels, labels, w1, w2, capacity, max_hits)
    }

    /// As [`CudaScanner::open`] with the kernel `kernel`; fails when the device cannot run it.
    #[allow(clippy::too_many_arguments)]
    pub fn open_kernel(
        device: usize,
        kernel: GpuKernel,
        pixels: &[u8],
        labels: &[u8],
        w1: &[i8],
        w2: &[i8],
        capacity: u32,
        max_hits: u32,
    ) -> Result<CudaScanner, String> {
        assert_eq!(pixels.len(), labels.len() * NIN, "pixels must hold 784 bytes per label");
        assert!(w1.len() == N1 && w2.len() == N2, "the weights must have {N1} and {N2} entries");
        let mut raw = std::ptr::null_mut();
        // SAFETY: the lengths of `pixels`, `labels`, `w1` and `w2` are those of pow/gpu/scan_cuda.h (checked above),
        // and the call writes the handle to `raw`.
        let status = unsafe {
            neb_scan_open_kernel(
                device as c_int,
                kernel.code(),
                pixels.as_ptr(),
                labels.as_ptr(),
                labels.len() as u32,
                w1.as_ptr(),
                w2.as_ptr(),
                capacity,
                max_hits,
                &mut raw,
            )
        };
        check(status, &format!("cannot open CUDA device {device} for {}", kernel.name()))?;
        Ok(CudaScanner { raw, capacity, hits: vec![0; max_hits as usize], kernel })
    }

    pub fn kernel(&self) -> GpuKernel {
        self.kernel
    }
}

impl Drop for CudaScanner {
    fn drop(&mut self) {
        // SAFETY: `raw` came from a successful `neb_scan_open_kernel` and is closed once.
        unsafe { neb_scan_close(self.raw) };
    }
}

impl Scanner for CudaScanner {
    fn capacity(&self) -> u32 {
        self.capacity
    }

    fn set_model(&mut self, w1: &[i8], w2: &[i8]) -> Result<(), String> {
        assert!(w1.len() == N1 && w2.len() == N2, "the weights must have {N1} and {N2} entries");
        // SAFETY: `raw` is an open handle used by this thread only (`&mut self`), and the weights have the lengths
        // checked above.
        check(unsafe { neb_scan_set_model(self.raw, w1.as_ptr(), w2.as_ptr()) }, "cannot upload the model")
    }

    fn scan(&mut self, header: &[u8; 80], start: u32, count: u32, bound: &[u8; 32]) -> Result<Vec<u32>, String> {
        let (raw, cap) = (self.raw, self.hits.len());
        let hits = &mut self.hits;
        scan_complete(start, count, &mut |start, count| {
            let mut found = 0;
            // SAFETY: `raw` is an open handle used by this thread only; `header` is 80 bytes and `bound` 32; `hits` has
            // `max_hits` entries, the most the call writes; the C code refuses a count above the capacity.
            let status = unsafe {
                neb_scan_run(raw, header.as_ptr(), start, count, bound.as_ptr(), hits.as_mut_ptr(), &mut found)
            };
            check(status, "scan failed")?;
            let found = found as usize;
            Ok((hits[..found.min(cap)].to_vec(), found))
        })
    }
}

/// Number of nonces and bound of the self-test cases (see [`check_scanner`]).
const SELFTEST_NONCES: u32 = 512;
const SELFTEST_PROBES: usize = 8;

/// The self-test of a kernel on a device: for each test model (the initial state, a trained model, three models with
/// extreme weights) and for the given dataset and the synthetic one, the hits and the exact tickets of the device
/// must equal the reference's. Returns a summary, or the first difference.
pub fn kernel_selftest(device: usize, kernel: GpuKernel, dataset: &dyn Dataset) -> Result<String, String> {
    let synthetic = SyntheticDataset::new();
    let mut header = [0u8; 80];
    for (i, b) in header.iter_mut().enumerate().take(76) {
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    let mut cases = 0;
    for (dname, ds) in [("dataset", dataset), ("synthetic", &synthetic as &dyn Dataset)] {
        let (pixels, labels) = flatten(ds);
        for (mname, model) in neb_engines::test_models(ds) {
            let (w1, w2) = unpack_weights(&model.packed_weights());
            let mut scanner =
                CudaScanner::open_kernel(device, kernel, &pixels, &labels, &w1, &w2, SELFTEST_NONCES, 4096)?;
            check_scanner(&mut scanner, &model, ds, &header, SELFTEST_NONCES, SELFTEST_PROBES)
                .map_err(|e| format!("model {mname}, {dname}: {e}"))?;
            cases += 1;
        }
    }
    Ok(format!(
        "{cases} cases of {SELFTEST_NONCES} nonces, {SELFTEST_PROBES} exact tickets each, equal to the reference"
    ))
}

unsafe extern "C" {
    fn neb2_scan_open(
        device: c_int,
        arch: c_int,
        values: *const u8,
        labels: *const u8,
        n: u32,
        weights: *const i8,
        capacity: u32,
        max_hits: u32,
        out: *mut *mut Raw,
    ) -> c_int;
    fn neb2_scan_set_model(s: *mut Raw, weights: *const i8) -> c_int;
    fn neb2_scan_run(
        s: *mut Raw,
        header: *const u8,
        start: u32,
        count: u32,
        bound: *const u8,
        hits: *mut u32,
        nhits: *mut u32,
    ) -> c_int;
    fn neb2_scan_close(s: *mut Raw);
}

/// One CUDA device holding a dataset of a v2 architecture and its model (`pow/gpu/nebhash2_cuda.h`).
pub struct Cuda2Scanner {
    raw: *mut Raw,
    arch: Arch,
    capacity: u32,
    hits: Vec<u32>,
}

// SAFETY: as for `CudaScanner`: one thread at a time uses the handle (`&mut self`), and every call selects the device.
unsafe impl Send for Cuda2Scanner {}

impl Cuda2Scanner {
    /// Uploads `dataset` (of a v2 architecture) and `model` to `device`.
    pub fn open(
        device: usize,
        dataset: &dyn Dataset,
        model: &Model,
        capacity: u32,
        max_hits: u32,
    ) -> Result<Cuda2Scanner, String> {
        let arch = dataset.arch();
        let id = arch.id().ok_or("the v2 kernel needs a dataset of a v2 architecture")?;
        if model.arch() != arch {
            return Err(format!("a {} model for a {} dataset", model.arch().name(), arch.name()));
        }
        let (values, labels) = flatten(dataset);
        let mut raw = std::ptr::null_mut();
        // SAFETY: `values` holds `nin` bytes per label and `weights` the architecture's weight count, the sizes of
        // nebhash2_cuda.h; the call writes the handle to `raw`.
        let status = unsafe {
            neb2_scan_open(
                device as c_int,
                id as c_int,
                values.as_ptr(),
                labels.as_ptr(),
                labels.len() as u32,
                model.weights().as_ptr(),
                capacity,
                max_hits,
                &mut raw,
            )
        };
        check(status, &format!("cannot open CUDA device {device} for {}", arch.name()))?;
        Ok(Cuda2Scanner { raw, arch, capacity, hits: vec![0; max_hits as usize] })
    }
}

impl Drop for Cuda2Scanner {
    fn drop(&mut self) {
        // SAFETY: `raw` came from a successful `neb2_scan_open` and is closed once.
        unsafe { neb2_scan_close(self.raw) };
    }
}

impl Scanner for Cuda2Scanner {
    fn capacity(&self) -> u32 {
        self.capacity
    }

    fn set_model(&mut self, _: &[i8], _: &[i8]) -> Result<(), String> {
        Err(format!("the {} kernel takes a model through set_weights", self.arch.name()))
    }

    fn set_weights(&mut self, model: &Model) -> Result<(), String> {
        if model.arch() != self.arch {
            return Err(format!("a {} model on the {} kernel", model.arch().name(), self.arch.name()));
        }
        // SAFETY: `raw` is an open handle used by this thread only, and the weights have the architecture's count.
        check(unsafe { neb2_scan_set_model(self.raw, model.weights().as_ptr()) }, "cannot upload the model")
    }

    fn scan(&mut self, header: &[u8; 80], start: u32, count: u32, bound: &[u8; 32]) -> Result<Vec<u32>, String> {
        let (raw, cap) = (self.raw, self.hits.len());
        let hits = &mut self.hits;
        scan_complete(start, count, &mut |start, count| {
            let mut found = 0;
            // SAFETY: as in `CudaScanner::scan`: an open handle, an 80-byte header, a 32-byte bound and `max_hits` slots.
            let status = unsafe {
                neb2_scan_run(raw, header.as_ptr(), start, count, bound.as_ptr(), hits.as_mut_ptr(), &mut found)
            };
            check(status, "scan failed")?;
            let found = found as usize;
            Ok((hits[..found.min(cap)].to_vec(), found))
        })
    }
}

/// The self-test of the v2 kernel on a device for a dataset of a v2 architecture: on the initial model of a seed and
/// on a model trained by a few steps, the hits and exact tickets of the device must equal the reference's.
pub fn kernel2_selftest(device: usize, dataset: &dyn Dataset) -> Result<String, String> {
    let arch = dataset.arch();
    let mut header = [0u8; 80];
    for (i, b) in header.iter_mut().enumerate().take(76) {
        *b = (i as u8).wrapping_mul(37).wrapping_add(13);
    }
    let mut model = Model::from_seed_arch(arch, &[5u8; 32]);
    let mut scanner = Cuda2Scanner::open(device, dataset, &model, SELFTEST_NONCES, 4096)?;
    for case in 0..2 {
        scanner.set_weights(&model)?;
        check_scanner(&mut scanner, &model, dataset, &header, SELFTEST_NONCES, SELFTEST_PROBES)
            .map_err(|e| format!("{} model {case}: {e}", arch.name()))?;
        let keys: Vec<[u8; 32]> = (0..32u8).map(|j| *blake3::hash(&[case as u8, j]).as_bytes()).collect();
        neb_pow::shares::train_step(&mut model, dataset, &keys)?;
    }
    Ok(format!(
        "2 {} models, {SELFTEST_NONCES} nonces, {SELFTEST_PROBES} exact tickets each, equal to the reference",
        arch.name()
    ))
}
