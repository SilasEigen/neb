//! The training-share step of the NEB protocol (`spec/shares.md` sections 4 to
//! 7). The C reference in `pow/ref` is compiled by `build.rs` and wrapped by the
//! [`shares`] module, which also holds the datasets and the vector replay.

pub mod shares;
