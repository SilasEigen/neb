//! Chain state of the node: blocks, transactions, locks, the UTXO set, the
//! model owned by the chain and reorganisations.
//!
//! Implements spec/node.md sections 4 to 6 and the chain integration of
//! spec/shares.md section 11. Module roles:
//!
//! - `block`: block encoding, share list rules, `body_root`.
//! - `tx`: transaction encoding, `txid`, `sighash`, lock verification.
//! - `work`: 256-bit cumulative work for chain selection.
//! - `model`: proof of work and share list of a block, and the model step.
//! - `state`: UTXO set, connect, disconnect, reorganisation, header rules.
//! - `demo`: block construction helpers for tests and tools.
//! - `fixture`: recorded chains that tests and tools replay (test support, not used by the node).
//!
//! The crate performs no I/O and holds no clock: `check_header` receives the
//! current time as an argument. Everything it decides is consensus-critical
//! and must give the same answer on every node. Amounts are u64 base units of
//! 10^-8 coin; sums are taken in u128 and cannot overflow. The network layer
//! and the choice between competing branches (cumulative work) belong to
//! `neb-node`; persistence belongs to `neb-store`.

pub mod block;
pub mod demo;
pub mod fixture;
pub mod model;
pub mod state;
pub mod tx;
pub mod work;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_shares;

pub use block::{Block, ShareRecord};
pub use model::{MODEL_DEPTH, ProofError, SHARE_RATIO_LOG2, share_key, share_target, verify_block};
pub use state::{ChainError, ChainParams, ChainState, OutPoint, Utxo};
pub use tx::{COIN, Transaction, TxIn, TxOut};
pub use work::Work;
