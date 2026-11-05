//! Library of the `neb-wallet` command: Ed25519 keys in a text file, `neb1...`
//! addresses, a scan of the best chain for owned outputs, largest-first coin
//! selection and transaction signing (spec/node.md sections 5.1 and 5.2).
//!
//! The wallet keeps no state besides its key file. It reads blocks and submits
//! transactions through the node's JSON-RPC (spec/node.md section 7.2) and
//! never decides validity: the node checks every submission (section 6.4). The
//! wallet trusts the node it asks and does not verify proofs of work.
//! The wallet recognises, signs and spends outputs with a type 1 (single-key)
//! lock and outputs with the 1-of-1 multisig lock of one of its keys, which is
//! the form of the project share output of the coinbase.

pub mod address;
pub mod display;
pub mod encryption;
pub mod rpc;
pub mod wallet;

pub use address::{ADDRESS_PREFIX, decode_address, encode_address};
pub use rpc::RpcClient;
pub use wallet::{
    AmountError, FEE, OwnedUtxo, Scan, SelectError, Selection, Wallet, WalletError, WalletKey, build_transaction,
    format_amount, parse_amount, scan, select,
};
