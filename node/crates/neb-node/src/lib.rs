//! The NEB node library: network parameters, the persistent chain, the
//! peer-to-peer layer (spec/node.md section 8), JSON-RPC and the work service
//! (spec/node.md section 7) and the daemon entry points.
//!
//! The consensus rules live in `neb-chain`, `neb-block`, `neb-difficulty` and
//! `neb-pow`. This crate holds no consensus rule of its own: it feeds blocks
//! to those crates and decides only local policy (peers, budgets, bans,
//! storage, ports).

pub mod budget;
pub mod daemon;
pub mod miners;
pub mod node;
pub mod p2p;
pub mod params;
pub mod rpc;
pub mod shares;
pub mod style;

pub use node::{BlockOutcome, Node, NodeConfig, NodeError};
pub use params::{NetworkParams, testnet};
