//! With the `cuda` feature, compiles the CUDA sources of `pow/gpu` with nvcc and links the CUDA runtime statically.
//! `NEB_CUDA_ARCHS` lists the GPU generations as compute capabilities without the dot (default `75,86,89,120`); the
//! last one also gets PTX for newer devices. dp4a needs 6.1 or later, and CUDA 13 builds for 7.5 or later.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=NEB_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let (gpu, refdir) = (root.join("pow/gpu"), root.join("pow/ref"));
    let archs = std::env::var("NEB_CUDA_ARCHS").unwrap_or_else(|_| "75,86,89,120".into());
    let archs: Vec<&str> = archs.split(',').map(str::trim).filter(|a| !a.is_empty()).collect();
    // Two libraries, each with its own device link: the tensor-core kernels (shares_tc.cu) and the dp4a kernel
    // (shares_cuda.cu) must not be device-linked together. When they were (one library, one device link), the dp4a
    // kernel returned wrong tickets although both compile and run on their own.
    let build = |files: &[&str]| {
        let mut b = cc::Build::new();
        b.cuda(true)
            .cudart("static")
            .cargo_warnings(false)
            .include(&gpu)
            .include(&refdir)
            .flag("-std=c++17")
            .flag("-O3");
        for a in &archs {
            b.flag("-gencode").flag(format!("arch=compute_{a},code=sm_{a}"));
        }
        if let Some(last) = archs.last() {
            b.flag("-gencode").flag(format!("arch=compute_{last},code=compute_{last}"));
        }
        for f in files {
            b.file(gpu.join(f));
            println!("cargo:rerun-if-changed={}", gpu.join(f).display());
        }
        b
    };
    for h in ["scan_cuda.h", "shares_cuda.h", "shares_tc.h", "tc_common.cuh"] {
        println!("cargo:rerun-if-changed={}", gpu.join(h).display());
    }
    println!("cargo:rerun-if-changed={}", refdir.join("shares.h").display());
    build(&["scan_cuda.cu", "shares_cuda.cu"]).compile("neb_gpu");
    build(&["shares_tc.cu"]).compile("neb_gpu_tc");
    if let Some(cuda) = std::env::var_os("CUDA_PATH") {
        let lib = if cfg!(windows) { "lib/x64" } else { "lib64" };
        println!("cargo:rustc-link-search=native={}", Path::new(&cuda).join(lib).display());
    }
}
