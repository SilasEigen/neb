//! Replays a dump of the best chain up to a height, every proof of work and share checked, and prints the
//! carry-over of every lock at that height: a tenth of its balance (`neb_node::carry`). The output of the genesis
//! miner lock, which no key can spend, and the project share carry nothing.
//!
//!     cargo run --release -p neb-node --example carry_over -- DATASET_FILE BLOCKS_FILE HEIGHT
//!
//! `BLOCKS_FILE` holds one block per line, as the hexadecimal of `getblock`, for heights 1, 2, 3 and so on. The
//! list goes to the standard output, one lock per line; the block id at the height, the number of locks, the total
//! and the BLAKE3 hash of the list go to the standard error.

use std::io::{BufRead, BufReader};
use std::path::Path;

use neb_chain::tx::{LOCK_MULTISIG, LOCK_SINGLE};
use neb_chain::{Block, ChainState};
use neb_node::carry::{carry_lines, carry_over};
use neb_node::params;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(dataset), Some(blocks), Some(height)) = (args.get(1), args.get(2), args.get(3)) else {
        eprintln!("usage: carry_over DATASET_FILE BLOCKS_FILE HEIGHT");
        std::process::exit(2);
    };
    let height: u32 = height.parse().unwrap_or_else(|_| panic!("{height}: not a height"));
    let ds = params::load_dataset(Path::new(dataset)).unwrap_or_else(|e| panic!("dataset: {e}"));
    let net = params::testnet(ds);
    let mut state = ChainState::from_genesis(net.chain.clone(), net.genesis.clone()).expect("genesis");
    let file = std::fs::File::open(blocks).unwrap_or_else(|e| panic!("{blocks}: {e}"));
    for (i, line) in BufReader::new(file).lines().enumerate().take(height as usize) {
        let line = line.expect("read");
        let block = unhex(line.trim()).and_then(|b| Block::decode(&b).ok());
        let block = block.unwrap_or_else(|| panic!("line {}: not a block", i + 1));
        if let Err(e) = state.connect_block(&block) {
            panic!("block {}: {e}", i + 1);
        }
    }
    assert_eq!(state.height(), height, "the dump stops before the height");
    let excluded = [(LOCK_SINGLE, params::GENESIS_MINER_LOCK), (LOCK_MULTISIG, params::project_share_lock())];
    let list = carry_over(state.utxos().values(), &excluded);
    let text = carry_lines(&list);
    print!("{text}");
    eprintln!("height {height}");
    eprintln!("block {}", hex(&state.tip_id()));
    eprintln!("locks {}", list.len());
    eprintln!("total {}", list.values().map(|&a| u128::from(a)).sum::<u128>());
    eprintln!("blake3 {}", blake3::hash(text.as_bytes()));
}
