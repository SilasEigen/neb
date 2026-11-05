//! Optimised ticket engines of the NEB miners.
//!
//! The ticket of an attempt is defined by the C reference (`pow/ref`, `spec/shares.md` sections 5.1 to 5.4); the
//! node validates blocks and shares with it and with nothing else, and does not depend on this crate. A miner may
//! evaluate its attempts faster with an engine from here, under four rules:
//!
//! 1. An optimised engine is never the default. The caller asks for it by name ([`EngineKind`]).
//! 2. [`Engine::new`] runs a self-test first (several models and datasets, digest, ticket, sample indices and
//!    clipped gradients, against the reference). If the processor lacks the instructions, or any value differs,
//!    the engine is the reference and [`Engine::messages`] says why.
//! 3. Every share and every block solution that an optimised engine finds is evaluated again by the reference
//!    before it is reported ([`Worker::attempt`]); a nonce that the reference refuses is never sent.
//! 4. Any disagreement with the reference, found when a hit is verified, in the audit of one attempt in
//!    [`AUDIT_INTERVAL`], or in the check of each new model, switches the engine off for the rest of the run. The
//!    attempts are then evaluated by the reference. A hit that an engine misses (a false negative) cannot be
//!    caught by the verification of hits; the audits and the model check bound that risk but do not remove it.
//!
//! Engines: the C reference itself, and the AVX-512 VNNI evaluation in [`avx512`] (dense and sparse variants).
//! The GPU kernels are in `neb-gpuminer`, with the same rules.

pub mod avx512;
mod engine;
mod selftest;

pub use engine::{AUDIT_INTERVAL, Attempt, Engine, EngineKind, EngineStats, MODEL_CHECK_KEYS, Prepared, Worker};
pub use selftest::{SelfTestReport, test_models};
