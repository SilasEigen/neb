//! Parameters and genesis block of the main network (spec/node.md section 6.6).
//!
//! The rules are those of the test network; what differs is listed here. The two model slots hold a KEP1 network
//! each from the genesis block (spec/nebhash-v2.md), with no dataset upgrade. The genesis coinbase pays, after
//! the reward of height 0, the carry-over of the two test networks: every lock receives a tenth of its balance on
//! each (`crate::carry`), on the same lock, so the wallet file of a tester holds its coins. These outputs are
//! coinbase outputs of height 0 and mature after 100 blocks, like any reward.
//!
//! The launch values ([`BEACON`], [`GENESIS_TIME`], [`GENESIS_BITS`] and the list of the second test network) fix
//! the genesis id; `docs/mainnet.md` lists them.

use std::collections::BTreeMap;

use neb_block::Header;
use neb_chain::Block;
use neb_chain::state::{ChainParams, project_share, subsidy};
use neb_chain::tx::{LOCK_MULTISIG, LOCK_SINGLE, Transaction, TxIn, TxOut};
use neb_pow::shares::SharedDataset;

use crate::carry::{Lock, merge, parse_lines};
use crate::params::{GENESIS_MINER_LOCK, POW_LIMIT_BITS, project_share_lock};

pub const NAME: &str = "mainnet";
pub const DEFAULT_DATADIR: &str = "neb-data";

/// Beacon of spec/node.md section 6.6: the hash of Bitcoin block 976143, the lowest block of that chain whose
/// timestamp (1794841795) is later than the time of block 12900 of the second test network (1794840900). The rule
/// was fixed before that block existed, so nobody could build on this genesis before it.
pub const BEACON: &str = "00000000000000000002038eee1b35e0033afa7f0d6f142bf07b941611b748a8";

/// 2026-11-18 14:00:00 UTC, the start of the network: block 1 cannot be dated earlier, and a node refuses a block
/// dated more than 120 s ahead of its clock.
pub const GENESIS_TIME: u32 = 1_795_010_400;

/// The target of the genesis block and the ASERT anchor: 32 times the target of block 12900 of the second test
/// network (`0xc2b38c00`), about 1.2 * 10^10 attempts per block. A network with the attempt rate of that test
/// network starts with blocks a few seconds apart and reaches the 60 s spacing after about 50 blocks; a slower one
/// waits longer for block 1, then ASERT follows the real block times. The pow limit stays the easy target of the
/// test network, 2^236.
pub const GENESIS_BITS: u32 = 0xC7B3_8C00;

/// The datasets of the network file, in slot order, both for the KEP1 network: the Kepler DR24 transit signals
/// (planet candidate or false positive) and the TESS signals of Astronet-Triage-v2 (eclipse or junk).
/// `tools/make_mainnet_datasets.py` builds the file.
pub const DATASET_NAMES: [&str; 2] = ["kepler-v2", "tess"];

/// BLAKE3 of the dataset set file of the main network.
pub const DATASET_HASH: [u8; 32] = [
    0x79, 0x0b, 0x57, 0xfb, 0xd8, 0x30, 0xf3, 0xdc, 0xbe, 0x7d, 0xe9, 0xd1, 0x14, 0x6a, 0xa8, 0x9a, 0x0a, 0x20, 0x79,
    0x91, 0x18, 0xe3, 0x0d, 0xd4, 0x8a, 0x34, 0x3a, 0x5c, 0x98, 0xf1, 0x9e, 0x72,
];

/// Public relays that a node dials when it is given no `--connect`, no `--seed` and no `--no-seeds`, as `HOST:PORT`.
/// Local policy: it is not part of the consensus rules and does not change the genesis id.
pub const DEFAULT_SEEDS: [&str; 4] = [
    "seed1.neuralenginebyte.org:3440",
    "seed2.neuralenginebyte.org:3440",
    "seed3.neuralenginebyte.org:3440",
    "seed4.neuralenginebyte.org:3440",
];

/// Carry-over of the first test network at its last block, height 3001
/// (`f2d0fe4dd450d5d56fd2a6959daf7c28cf71e6d52c00aa83146eb35c50b30f16`).
const CARRY_TESTNET1: &str = include_str!("../data/carry-testnet1.txt");

/// Carry-over of the second test network at height 12900
/// (`0b58879dafb768ca484d0a8894f8afc34b246c4813be5a77869c831f1f713106`), made by `examples/carry_over.rs`.
const CARRY_TESTNET2: &str = include_str!("../data/carry-testnet2.txt");

/// The carry-over of every lock: the sum of its lines in the lists of the two test networks.
pub fn carry_list() -> BTreeMap<Lock, u64> {
    let lists = [CARRY_TESTNET1, CARRY_TESTNET2].map(|text| parse_lines(text).expect("a built-in list is well formed"));
    merge(&lists).expect("the carry-over is below the supply")
}

/// Coinbase witness of spec/node.md section 6.6, 44 bytes: `height (u32 LE, 0) || extranonce (8) || miner data (32)`,
/// derived from the beacon with the tags of the main network. The beacon tag ends in 2: the genesis of version
/// 1.0.0, which never had a block, used `neb-mainnet-beacon-1` with another dataset in slot 1, and a node of that
/// version must not take this network for its own.
fn genesis_witness() -> Vec<u8> {
    let mut witness = Vec::with_capacity(44);
    witness.extend_from_slice(&0u32.to_le_bytes());
    let mut extranonce = blake3::Hasher::new();
    extranonce.update(b"neb-mainnet-extranonce");
    extranonce.update(BEACON.as_bytes());
    witness.extend_from_slice(&extranonce.finalize().as_bytes()[..8]);
    let mut miner_data = blake3::Hasher::new();
    miner_data.update(b"neb-mainnet-beacon-2");
    miner_data.update(BEACON.as_bytes());
    witness.extend_from_slice(miner_data.finalize().as_bytes());
    witness
}

/// The genesis block: one coinbase paying the reward of height 0 less the project share to the lock that no key
/// can unlock, the project share to the project multisig, then the carry-over of every lock by increasing lock;
/// no shares; nonce 0. Its proof of work is not checked.
pub fn genesis() -> Block {
    let reward = subsidy(0);
    let share = project_share(reward);
    let mut outputs = vec![
        TxOut { amount: reward - share, lock_type: LOCK_SINGLE, lock_data: GENESIS_MINER_LOCK },
        TxOut { amount: share, lock_type: LOCK_MULTISIG, lock_data: project_share_lock() },
    ];
    outputs.extend(carry_list().into_iter().map(|((lock_type, lock_data), amount)| TxOut {
        amount,
        lock_type,
        lock_data,
    }));
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: [0u8; 32], prev_index: 0xFFFF_FFFF, witness: genesis_witness() }],
        outputs,
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
        upgrades: Vec::new(),
    }
}

/// The four-byte network magic, `BLAKE3("neb-mainnet-magic" || genesis id)[..4]` (spec/node.md section 8.2).
pub fn magic() -> [u8; 4] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"neb-mainnet-magic");
    hasher.update(&genesis().id());
    hasher.finalize().as_bytes()[..4].try_into().expect("four bytes")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use neb_chain::ChainState;
    use neb_pow::shares::{Arch, SyntheticDataset2};

    use super::*;
    use crate::params::{self, Network};

    fn datasets() -> Vec<SharedDataset> {
        vec![Arc::new(SyntheticDataset2::new(Arch::Kep1)), Arc::new(SyntheticDataset2::new(Arch::Kep1))]
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    // The two lists as they were computed: their hashes are those printed by the tools that made them.
    #[test]
    fn the_built_in_lists_are_the_computed_ones() {
        let hash = |text: &str| blake3::hash(text.replace("\r\n", "\n").as_bytes()).to_hex().to_string();
        assert_eq!(hash(CARRY_TESTNET1), "05b0f48dc4d698530016c147e9ba56aef24f4237183fae568233fb9d42eda32c");
        assert_eq!(hash(CARRY_TESTNET2), "a0fea9e6628da361568ebce88a8d942ab68b0c667bb1355350d257f9a3049916");
        let (one, two) = (parse_lines(CARRY_TESTNET1).unwrap(), parse_lines(CARRY_TESTNET2).unwrap());
        assert_eq!((one.len(), one.values().sum::<u64>()), (69, 570_190_000_000));
        assert_eq!((two.len(), two.values().sum::<u64>()), (159, 2_450_999_999_952));
        assert_eq!(carry_list().len(), 215);
    }

    #[test]
    fn the_genesis_pays_the_reward_then_the_carry_over_of_every_lock() {
        let genesis = genesis();
        assert!(genesis.body_ok());
        assert_eq!(
            Block::decode(&genesis.encode()).as_ref(),
            Ok(&genesis),
            "a genesis of more outputs than a transaction may hold decodes"
        );
        let outputs = &genesis.txs[0].outputs;
        let list = carry_list();
        assert_eq!(outputs.len(), 2 + list.len());
        assert_eq!(
            (outputs[0].amount, outputs[0].lock_data),
            (subsidy(0) - project_share(subsidy(0)), GENESIS_MINER_LOCK)
        );
        assert_eq!((outputs[1].amount, outputs[1].lock_data), (project_share(subsidy(0)), project_share_lock()));
        for (out, (lock, amount)) in outputs[2..].iter().zip(&list) {
            assert_eq!(((out.lock_type, out.lock_data), out.amount), (*lock, *amount));
        }
        // neither the lock that no key opens nor the project receives a carry-over
        assert!(!list.contains_key(&(LOCK_SINGLE, GENESIS_MINER_LOCK)));
        assert!(!list.contains_key(&(LOCK_MULTISIG, project_share_lock())));
    }

    #[test]
    fn the_chain_starts_with_the_carry_over_unspent_and_the_convolutional_models() {
        let net = Network::Mainnet.params(datasets());
        let state = ChainState::from_genesis(net.chain.clone(), net.genesis.clone()).expect("genesis");
        let carried: u128 = carry_list().values().map(|&a| u128::from(a)).sum();
        assert!(state.conservation());
        assert_eq!(state.unspent_total(), u128::from(subsidy(0)) + carried);
        assert_eq!(state.utxos().len(), 2 + carry_list().len());
        assert!(state.utxos().values().all(|u| u.coinbase && u.height == 0), "the carry-over matures like a reward");
        let models = state.tip_models();
        assert_eq!((models[0].arch(), models[1].arch()), (Arch::Kep1, Arch::Kep1));
        assert_ne!(models[0].weights_hash(), models[1].weights_hash(), "each slot starts from its own weights");
        assert_eq!((net.chain.slots(), net.chain.check_upgrades()), (2, Ok(())));
    }

    #[test]
    fn the_two_networks_do_not_share_an_identity() {
        let (main, test) = (Network::Mainnet.params(datasets()), Network::Testnet);
        assert_eq!((main.name, main.dataset_names), ("mainnet", &DATASET_NAMES[..]));
        assert_ne!(main.chain_id(), params::genesis_id());
        assert_ne!(main.magic, params::magic());
        assert_ne!(Network::Mainnet.default_datadir(), test.default_datadir());
        assert_eq!(Network::Mainnet.default_seeds(), &DEFAULT_SEEDS[..]);
        // the values of docs/mainnet.md
        assert_eq!(hex(&main.chain_id()), "da2f5af7bb4c5e2bcc8d558b2f4f01d4c88557a943e6cff9cd035638fe9439ae");
        assert_eq!(hex(&main.magic), "ad33cae5");
        assert_eq!((GENESIS_TIME, GENESIS_BITS), (1_795_010_400, 0xC7B3_8C00));
        assert_eq!(hex(&DATASET_HASH), "790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72");
    }
}
