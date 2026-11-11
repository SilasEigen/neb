//! Replays a dump of the best chain with the consensus rules of this build, every proof of work and share checked,
//! and prints the height, the tip id and the state hash (`weightshash` of `getinfo`). Run with the parameters of a
//! new release on a dump of the live chain, it shows that the release accepts the chain as it is and reaches the same
//! state, so a node can take the release before an upgrade height.
//!
//!     cargo run --release -p neb-node --example replay_dump -- DATASET_FILE BLOCKS_FILE
//!
//! `BLOCKS_FILE` holds one block per line, as the hexadecimal of `getblock`, for heights 1, 2, 3 and so on.

use std::io::{BufRead, BufReader};
use std::path::Path;

use neb_chain::{Block, ChainState, models_hash};
use neb_node::params;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(dataset), Some(blocks)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: replay_dump DATASET_FILE BLOCKS_FILE");
        std::process::exit(2);
    };
    let ds = params::load_dataset(Path::new(dataset)).unwrap_or_else(|e| panic!("dataset: {e}"));
    let net = params::testnet(ds);
    let mut state = ChainState::from_genesis(net.chain.clone(), net.genesis.clone()).expect("genesis");
    let file = std::fs::File::open(blocks).unwrap_or_else(|e| panic!("{blocks}: {e}"));
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.expect("read");
        let block = unhex(line.trim()).and_then(|b| Block::decode(&b).ok());
        let block = block.unwrap_or_else(|| panic!("line {}: not a block", i + 1));
        if let Err(e) = state.connect_block(&block) {
            panic!("block {}: {e}", i + 1);
        }
    }
    println!("height {}", state.height());
    println!("tip {}", hex(&state.tip_id()));
    println!("weightshash {}", hex(&models_hash(&state.tip_models())));
}
