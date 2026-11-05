//! Compiles the C reference of the training-share step (`pow/ref`) into this
//! crate. The node runs the same C files that generate and replay the vectors
//! (`vectors/shares.txt`); the arithmetic has no second implementation in Rust.
//! A C compiler is a build requirement.

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let refdir = root.join("pow/ref");
    let files = ["shares.c", "chacha8.c", "blake3_min.c"];
    let mut b = cc::Build::new();
    b.include(&refdir);
    for f in files {
        b.file(refdir.join(f));
        println!("cargo:rerun-if-changed={}", refdir.join(f).display());
    }
    for h in ["shares.h", "chacha8.h", "blake3_min.h"] {
        println!("cargo:rerun-if-changed={}", refdir.join(h).display());
    }
    // The C code is integer arithmetic only, so the optimisation level cannot
    // change a result; vectors/shares.txt pins the results.
    b.flag_if_supported("-O2");
    b.compile("neb_ref");
}
