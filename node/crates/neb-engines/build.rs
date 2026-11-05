//! Compiles the AVX-512 evaluation of the training shares (`pow/cpu/shares_avx512.c`) in a unit of its own, so that
//! no other code of the program can be compiled with AVX-512 instructions. The unit is built for x86-64 only and the
//! `neb_avx512` cfg is set when the compiler accepts it; the processor is checked at run time by `avx512::available`.
//! The engine is a mining optimisation: nothing in the node depends on this crate.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(neb_avx512)");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let (dir, refdir) = (root.join("pow/cpu"), root.join("pow/ref"));
    for f in ["shares_avx512.c", "shares_avx512.h"] {
        println!("cargo:rerun-if-changed={}", dir.join(f).display());
    }
    for h in ["shares.h", "blake3_min.h"] {
        println!("cargo:rerun-if-changed={}", refdir.join(h).display());
    }
    let mut b = cc::Build::new();
    b.include(&dir).include(&refdir).file(dir.join("shares_avx512.c"));
    if b.get_compiler().is_like_msvc() {
        b.flag("/arch:AVX512");
    } else {
        for f in ["-O3", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq", "-mavx512vnni"] {
            b.flag(f);
        }
    }
    // The symbols neb_share_indices and blake3_hash come from the C reference that neb-pow compiles.
    match b.try_compile("neb_avx512") {
        Ok(()) => println!("cargo:rustc-cfg=neb_avx512"),
        Err(e) => println!("cargo:warning=the AVX-512 engine is not built: {e}"),
    }
}
