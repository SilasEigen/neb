fn main() {
    // Metal exists only on macOS. On every other platform the crate compiles without it and its entry points say why,
    // so that `cargo clippy --workspace` and `cargo test --workspace` pass on the Linux machine of the CI.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        println!("cargo:rerun-if-changed=build.rs");
        return;
    }
    cc::Build::new().file("src/metal.m").flag_if_supported("-Wno-deprecated-declarations").compile("neb_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rerun-if-changed=src/metal.m");
    println!("cargo:rerun-if-changed=src/neb_share.metal");
}
