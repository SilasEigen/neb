//! Network parameters and the deterministic genesis block of the testnet
//! (spec/node.md section 6.6; values in docs/testnet.md).
//!
//! The genesis block is a pure function of the beacon string and the constants
//! below: every node derives the same bytes, so the genesis id (the chain id)
//! needs no distributed file. The coinbase carries `BLAKE3("neb-testnet-beacon-2"
//! || beacon)` as miner data (the tag ends in 2 since the share rule of protocol 2). The output of the miner is locked
//! to [`GENESIS_MINER_LOCK`], which no key can unlock, so the genesis coins can never be spent. The project share goes to
//! the 1-of-1 multisig of
//! [`PROJECT_PUBLIC_KEY`]; its secret key is held by the project and is not
//! published.
//!
//! Only launch parameters are defined here: genesis target, pow limit, ports,
//! peer limit and the dataset hash. Subsidy, halving, project share, difficulty
//! and block rules come from `neb-chain` and `neb-difficulty`. Changing any
//! value below creates a different network (different genesis id and magic).

use std::path::Path;

use ed25519_dalek::SigningKey;
use neb_block::Header;
use neb_chain::state::{ChainParams, project_share, subsidy};
use neb_chain::tx::{COIN, LOCK_MULTISIG, LOCK_SINGLE, Transaction, TxIn, TxOut};
use neb_chain::{Block, ChainState};
use neb_pow::shares::{SharedDataset, load_datasets};

/// Beacon of spec/node.md section 6.6: the hash of the first block of an external
/// public chain in the usual display order, as 64 lower-case hexadecimal characters.
pub const BEACON: &str = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f";

/// 2026-11-05 19:00:00 UTC, the start of the network: block 1 cannot be dated earlier.
pub const GENESIS_TIME: u32 = 1_793_905_200;

/// The target of the genesis block and the ASERT anchor: 2^236.
///
/// A ticket is a 256-bit value, so one attempt meets 2^236 with probability
/// 2^-20 and a block takes 2^20 = 1 048 576 attempts on average. At 0.5 ms per
/// attempt on one core and 8 cores that is 16 000 attempts per second, one
/// block per 65 s, close to the 60 s spacing. Compact form: 2^236 =
/// 0x800000 << 213 and 213 = 0xD5, so the bits are 0xD5800000. The share target
/// is 32 times easier, 2^241: about 32 shares per block.
pub const GENESIS_BITS: u32 = 0xD580_0000;

/// The easiest target ASERT may reach: the genesis target, 2^236.
/// The genesis time is fixed in advance, so blocks mined later than their
/// schedule would push the target up without bound; equal to the genesis
/// target, the limit caps the block rate of a catch-up burst at the rate the
/// genesis target gives.
pub const POW_LIMIT_BITS: u32 = 0xD580_0000;

/// Name of the single network.
pub const NAME: &str = "testnet";
/// Default TCP ports: peer listener, work service (loopback), JSON-RPC (loopback).
pub const P2P_PORT: u16 = 3440;
pub const WORK_PORT: u16 = 3441;
pub const RPC_PORT: u16 = 8432;
pub const DEFAULT_DATADIR: &str = "neb-testnet2-data";

/// The datasets of the network, one per model slot, in slot order (spec/shares.md section 11.6): the transit
/// signals of the Kepler DR24 TCE table (NASA Exoplanet Archive; label 1 = planet candidate, 0 = false positive)
/// and the retinal OCT scans of OCTMNIST (MedMNIST v2, CC BY 4.0; labels 0 to 3). `tools/make_datasets.py` builds
/// the set file.
pub const DATASET_NAMES: [&str; 2] = ["kepler", "retina"];

/// BLAKE3 of the dataset set file `neb-datasets.bin` (both datasets, spec/shares.md section 11.6). A node refuses
/// to start on another file.
pub const DATASET_HASH: [u8; 32] = [
    0xb1, 0x00, 0x8b, 0x97, 0x0f, 0x87, 0x08, 0x31, 0x79, 0xa7, 0xb2, 0x97, 0xb8, 0xf0, 0xfb, 0xec, 0x4c, 0xfc, 0x8c,
    0xc6, 0x6f, 0x60, 0x5c, 0xf2, 0xfb, 0x2c, 0x03, 0xba, 0xd8, 0xb6, 0xf3, 0x7a,
];

/// BLAKE3 of the dataset file made by `tools/make_mnist_dataset.py` from the
/// 60 000 MNIST training images, the dataset of protocol 2; the recorded
/// chain fixture `pools-mnist` uses it.
pub const MNIST_DATASET_HASH: [u8; 32] = [
    0x1a, 0x6a, 0xb4, 0x34, 0x7b, 0x5e, 0x7d, 0x81, 0x88, 0xd6, 0x65, 0x12, 0x47, 0x07, 0xdf, 0x7a, 0x1a, 0x2a, 0x28,
    0xbf, 0x60, 0xe5, 0x27, 0xe1, 0x67, 0xa8, 0x94, 0xfd, 0xec, 0x37, 0x3e, 0xd2,
];
pub const ADDRESS_PREFIX: &str = "neb";
/// Protocol number of the `version` message (spec/node.md section 8.2). 3: the chain trains one model per dataset
/// and a block of height h works on slot h mod K (spec/shares.md section 11.2); 2: a share is bound to the parent
/// of its header and the share list holds 40-byte records. A node of another protocol does not connect.
pub const PROTOCOL_VERSION: u32 = 3;
/// Maximum simultaneous peer sessions, inbound and outbound together. Local policy.
pub const MAX_PEERS: usize = 8;
/// Highest value of `--max-peers`: each session has its own threads.
pub const MAX_PEERS_LIMIT: usize = 1024;
/// Public relays that a node dials when it is given no `--connect`, no `--seed` and no `--no-seeds`, as `HOST:PORT`.
/// Local policy: it is not part of the consensus rules and does not change the genesis id.
pub const DEFAULT_SEEDS: [&str; 2] = ["seed1.neuralenginebyte.org:3440", "seed2.neuralenginebyte.org:3440"];
pub const USER_AGENT: &str = concat!("neb-node/", env!("CARGO_PKG_VERSION"));

/// Lock of the genesis miner output: 32 zero bytes. A single-key lock is `BLAKE3(public key)`, and no key is known
/// whose hash is zero, so the coins of height 0 less the project share can never be spent.
pub const GENESIS_MINER_LOCK: [u8; 32] = [0u8; 32];

/// Public key of the project share output (spec/node.md section 6.3), the only
/// key of a 1-of-1 multisig lock. The matching secret key is not in this repository.
pub const PROJECT_PUBLIC_KEY: [u8; 32] = [
    0x25, 0xda, 0xdb, 0xa8, 0x2b, 0x13, 0x72, 0xfd, 0x76, 0xc1, 0x90, 0x15, 0x24, 0xcc, 0x74, 0x22, 0x24, 0x66, 0x1d,
    0x6f, 0xc5, 0xf0, 0xba, 0x1b, 0x4e, 0x95, 0xf5, 0x26, 0xa3, 0xaa, 0x72, 0xd3,
];

/// Lock data of a single-key lock: `BLAKE3(public key)` (spec/node.md section 5.2).
pub fn single_lock(key: &SigningKey) -> [u8; 32] {
    *blake3::hash(&key.verifying_key().to_bytes()).as_bytes()
}

pub fn project_share_lock() -> [u8; 32] {
    Transaction::multisig_lock_data(1, &[PROJECT_PUBLIC_KEY])
}

/// Coinbase witness of spec/node.md section 6.6, 44 bytes:
/// `height (u32 LE, 0) || extranonce (8) || miner data (32)`. The extranonce is
/// the first 8 bytes of `BLAKE3("neb-testnet-extranonce" || beacon)`.
fn genesis_witness() -> Vec<u8> {
    let mut witness = Vec::with_capacity(44);
    witness.extend_from_slice(&0u32.to_le_bytes());
    let mut extranonce = blake3::Hasher::new();
    extranonce.update(b"neb-testnet-extranonce");
    extranonce.update(BEACON.as_bytes());
    witness.extend_from_slice(&extranonce.finalize().as_bytes()[..8]);
    let mut miner_data = blake3::Hasher::new();
    miner_data.update(b"neb-testnet-beacon-3");
    miner_data.update(BEACON.as_bytes());
    witness.extend_from_slice(miner_data.finalize().as_bytes());
    witness
}

/// The genesis block: one coinbase paying the reward of height 0 less the
/// project share to [`GENESIS_MINER_LOCK`] and the project share to the project
/// multisig; no shares; nonce 0. Its proof of work is not checked.
pub fn genesis() -> Block {
    let reward = subsidy(0);
    let share = project_share(reward);
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [0u8; 32], prev_index: 0xFFFF_FFFF, witness: genesis_witness() }],
        outputs: vec![
            TxOut { amount: reward - share, lock_type: LOCK_SINGLE, lock_data: GENESIS_MINER_LOCK },
            TxOut { amount: share, lock_type: LOCK_MULTISIG, lock_data: project_share_lock() },
        ],
    };
    let mut block = Block {
        header: Header {
            version: 1,
            parent: [0u8; 32],
            body_root: [0u8; 32],
            time: GENESIS_TIME,
            bits: GENESIS_BITS,
            nonce: 0,
        },
        txs: vec![coinbase],
        shares: Vec::new(),
    };
    block.header.body_root = block.body_root();
    block
}

pub fn chain_params(datasets: Vec<SharedDataset>) -> ChainParams {
    ChainParams {
        difficulty: neb_difficulty::Params {
            spacing: neb_difficulty::SPACING,
            half_life: neb_difficulty::HALF_LIFE,
            pow_limit_bits: POW_LIMIT_BITS,
            anchor_bits: GENESIS_BITS,
            anchor_time: GENESIS_TIME,
        },
        project_share_lock: project_share_lock(),
        datasets,
    }
}

/// Reads the dataset set file and checks it against the pinned hash and the number of slots.
pub fn load_dataset(path: &Path) -> Result<Vec<SharedDataset>, String> {
    let sets = load_datasets(path, Some(&DATASET_HASH))?;
    if sets.len() != DATASET_NAMES.len() {
        return Err(format!(
            "{} holds {} datasets, the network has {}",
            path.display(),
            sets.len(),
            DATASET_NAMES.len()
        ));
    }
    Ok(sets)
}

pub fn genesis_id() -> [u8; 32] {
    genesis().id()
}

/// The four-byte network magic, `BLAKE3("neb-testnet-magic" || genesis id)[..4]`
/// (spec/node.md section 8.2). It binds the framing to the genesis, so nodes
/// with different parameters do not parse each other's messages.
pub fn magic() -> [u8; 4] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"neb-testnet-magic");
    hasher.update(&genesis_id());
    hasher.finalize().as_bytes()[..4].try_into().unwrap()
}

#[derive(Clone, Debug)]
pub struct NetworkParams {
    pub name: &'static str,
    pub magic: [u8; 4],
    pub p2p_port: u16,
    pub work_port: u16,
    pub rpc_port: u16,
    pub address_prefix: &'static str,
    pub protocol_version: u32,
    pub max_peers: usize,
    pub user_agent: &'static str,
    pub genesis: Block,
    pub chain: ChainParams,
}

impl NetworkParams {
    /// The chain id is the genesis block id (spec/node.md section 6.6).
    pub fn chain_id(&self) -> [u8; 32] {
        self.genesis.id()
    }
}

pub fn testnet(dataset: Vec<SharedDataset>) -> NetworkParams {
    let genesis = genesis();
    NetworkParams {
        name: NAME,
        magic: magic(),
        p2p_port: P2P_PORT,
        work_port: WORK_PORT,
        rpc_port: RPC_PORT,
        address_prefix: ADDRESS_PREFIX,
        protocol_version: PROTOCOL_VERSION,
        max_peers: MAX_PEERS,
        user_agent: USER_AGENT,
        genesis,
        chain: chain_params(dataset),
    }
}

/// Builds the chain state from the genesis block.
pub fn genesis_state(dataset: Vec<SharedDataset>) -> Result<ChainState, String> {
    ChainState::from_genesis(chain_params(dataset), genesis()).map_err(|e| e.to_string())
}

/// One coin in base units (10^8).
pub const ONE_COIN: u64 = COIN;

#[cfg(test)]
mod tests {
    use super::*;
    use neb_pow::shares::SyntheticDataset;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn dataset() -> Vec<SharedDataset> {
        DATASET_NAMES.iter().map(|_| std::sync::Arc::new(SyntheticDataset::new()) as SharedDataset).collect()
    }

    // The genesis block layout of spec/node.md section 6.6.
    #[test]
    fn genesis_is_deterministic_and_fixed() {
        let a = genesis();
        let b = genesis();
        assert_eq!(a, b);
        assert_eq!(a.id(), genesis_id());
        assert_eq!(a.header.version, 1);
        assert_eq!(a.header.parent, [0u8; 32]);
        assert_eq!(a.header.time, GENESIS_TIME);
        assert_eq!(a.header.bits, GENESIS_BITS);
        assert_eq!(a.header.nonce, 0);
        assert!(a.body_ok());
        assert!(a.shares.is_empty());
        assert_eq!(a.txs.len(), 1);
        let coinbase = &a.txs[0];
        assert!(coinbase.is_coinbase());
        assert_eq!(coinbase.outputs.len(), 2);
        assert_eq!(coinbase.outputs[0].amount, subsidy(0) - project_share(subsidy(0)));
        assert_eq!(coinbase.outputs[1].amount, project_share(subsidy(0)));
        assert_eq!(coinbase.outputs[1].lock_type, LOCK_MULTISIG);
        assert_eq!(coinbase.outputs[1].lock_data, project_share_lock());
        assert_eq!(coinbase.inputs[0].witness.len(), 44);
        assert_eq!(coinbase.inputs[0].witness[4..12].len(), 8);
        assert_eq!(coinbase.inputs[0].witness[12..].len(), 32);
    }

    // Launch parameters and the genesis agree: conservation holds at height 0.
    #[test]
    fn genesis_state_accepts_the_genesis() {
        let state = genesis_state(dataset()).expect("genesis state");
        assert_eq!(state.height(), 0);
        assert_eq!(state.chain_id, genesis_id());
        assert!(state.conservation());
        assert_eq!(state.unspent_total(), subsidy(0) as u128);
    }

    // A changed port or limit must be a deliberate edit.
    #[test]
    fn network_parameters_are_stable() {
        let p = testnet(dataset());
        assert_eq!(p.name, "testnet");
        assert_eq!(p.p2p_port, 3440);
        assert_eq!(p.work_port, 3441);
        assert_eq!(p.rpc_port, 8432);
        assert_eq!(p.address_prefix, "neb");
        assert_eq!(p.max_peers, 8);
        assert_eq!(p.chain_id(), genesis_id());
        assert_eq!(p.magic, magic());
        assert_ne!(p.magic, [0u8; 4]);
        assert_eq!(p.chain.difficulty.anchor_bits, GENESIS_BITS);
        assert_eq!(p.chain.difficulty.pow_limit_bits, POW_LIMIT_BITS);
    }

    // The compact bits of the doc comment of `GENESIS_BITS` decode to the stated powers of two.
    #[test]
    fn the_genesis_target_is_2_pow_236_and_the_share_target_is_2_pow_241() {
        let target = neb_block::Target::from_bits(GENESIS_BITS).unwrap();
        let mut want = [0u8; 32];
        want[29] = 0x10; // bit 236 = byte 29, bit 4
        assert_eq!(target.0, want);
        let share = neb_chain::share_target(GENESIS_BITS).unwrap();
        let mut want = [0u8; 32];
        want[30] = 0x02; // bit 241 = byte 30, bit 1
        assert_eq!(share.0, want);
    }

    // A node refuses a dataset file that is not the pinned one.
    #[test]
    fn a_dataset_file_must_match_the_pinned_hash() {
        let dir = std::env::temp_dir().join(format!("neb-params-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.bin");
        let mut bytes = b"NMD1".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 785]);
        std::fs::write(&path, &bytes).unwrap();
        let err = load_dataset(&path).unwrap_err();
        assert!(err.contains("wrong hash"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Pins the keys, the project lock, the genesis id and the magic: any change moves the network identity.
    #[test]
    fn keys_and_network_identity_are_pinned() {
        assert_eq!(genesis().txs[0].outputs[0].lock_data, [0u8; 32]);
        assert_eq!(project_share_lock(), genesis().txs[0].outputs[1].lock_data);
        assert_eq!(hex(&PROJECT_PUBLIC_KEY), "25dadba82b1372fd76c1901524cc742224661d6fc5f0ba1b4e95f526a3aa72d3");
        assert_eq!(hex(&project_share_lock()), "270bbd6380c0c7022176e202083d5671ed6bb7bc0c27350bcbd134846ce87dd5");
        assert_eq!(hex(&genesis_id()), "4b9cdfd8a859c2cde2fc768b2264f67c5b4d3ec839cd070fcfdab478d0e55e28");
        assert_eq!(hex(&magic()), "cd16272d");
    }
}
