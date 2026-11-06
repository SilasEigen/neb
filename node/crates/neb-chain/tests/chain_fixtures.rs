//! Replays of recorded chains (`pow/tests/chains/*.txt`) with the node's own functions (`verify_block`): the
//! proof of work of every block, every share, the model step, the state hashes and the losses of the bounds.
//!
//! - `synthetic-stress` (41 blocks, steps of 1 to 256 shares around the 16 and 128 thresholds, synthetic
//!   dataset) runs everywhere. It was mined with the C reference by `examples/chain_fixture.rs`, and replayed by
//!   the independent Python oracle (`pow/oracle/chain_oracle.py`).
//! - `pools-synthetic` (4 pools in lockstep) and `pools-mnist` (3 pools, MNIST): the list of a block holds the share
//!   records of the headers of every pool (spec/shares.md section 11.2). `pools-mnist` needs the dataset file of the
//!   network: set `NEB_DATASET` to it (the BLAKE3 hash is checked against the one pinned by the network
//!   parameters). Without it they are skipped with a message; with `NEB_REQUIRE_DATASET=1` the absence is a failure.

use std::sync::Arc;

use neb_chain::ShareRecord;
use neb_chain::demo;
use neb_chain::fixture::{ChainFixture, LayerLoss, replay};
use neb_pow::shares::{Dataset, FileDataset, SyntheticDataset};

fn fixture(name: &str) -> ChainFixture {
    let path = format!("{}/../../../pow/tests/chains/{name}.txt", env!("CARGO_MANIFEST_DIR"));
    ChainFixture::parse(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))).unwrap()
}

fn params_with(ds: Arc<dyn Dataset>) -> neb_chain::ChainParams {
    let mut p = demo::easy_params(1_793_318_400, demo::PROJECT_LOCK);
    p.datasets = vec![ds];
    p
}

fn hex32(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}

/// The dataset file of the network, or `None` (skip) when `NEB_DATASET` is not set.
fn mnist() -> Option<Arc<dyn Dataset>> {
    let Some(path) = std::env::var_os("NEB_DATASET") else {
        assert!(std::env::var_os("NEB_REQUIRE_DATASET").is_none(), "NEB_REQUIRE_DATASET is set but NEB_DATASET is not");
        eprintln!("SKIPPED: set NEB_DATASET to the network dataset file to replay the MNIST chains");
        return None;
    };
    let pinned = neb_node_dataset_hash();
    Some(Arc::new(
        FileDataset::load(std::path::Path::new(&path), Some(&pinned)).expect("the dataset file with the pinned hash"),
    ))
}

/// BLAKE3 of the dataset file that the network parameters pin (`DATASET_HASH` in `neb-node/src/params.rs`),
/// read from the fixture so that this crate does not depend on the node.
fn neb_node_dataset_hash() -> [u8; 32] {
    hex32(fixture("pools-mnist").dataset.strip_prefix("file ").expect("pools-mnist is a file dataset"))
}

#[test]
fn the_synthetic_stress_chain_replays() {
    let fx = fixture("synthetic-stress");
    let r = replay(&fx, &params_with(Arc::new(SyntheticDataset::new()))).unwrap();
    assert_eq!((r.blocks, r.steps, r.shares, r.largest_step), (41, 40, 3351, 256));
    // The thresholds were really crossed: no cut in the steps of at most 16 (layer 2) and 128 (layer 1)
    // shares, cuts above; the recorded losses are recomputed by `replay`.
    let sizes: Vec<usize> = fx.blocks.iter().skip(1).map(|b| b.list.len()).collect();
    for (&n, (_, loss)) in sizes.iter().zip(&r.losses) {
        if n <= 16 {
            assert_eq!(loss[1].cut_events, 0, "layer 2 cuts at {n} shares");
        }
        if n <= 128 {
            assert_eq!(loss[0].cut_events, 0, "layer 1 cuts at {n} shares");
        }
    }
    assert!(
        sizes.iter().zip(&r.losses).any(|(&n, (_, l))| n == 17 && l[1].cut_events > 0),
        "the 17-share step must cut layer 2"
    );
    assert!(
        sizes.iter().zip(&r.losses).any(|(&n, (_, l))| n == 129 && l[0].cut_events > 0),
        "the 129-share step must cut layer 1"
    );
    assert_eq!(r.final_weights_hash, hex32("0e64c4ae6cc653719c5b6d1e8c92247d1b0cda4978e1180dd1e9b770fd2b8bca"));
    assert_eq!(r.final_residual_hash, hex32("986ef1ce2ebfef456901bb11569309743910c31ea62cc2f64555111cb8b9417c"));
}

// Four pools in lockstep (spec/shares.md section 11.2): the list of a block holds records of the headers of every pool,
// whichever pool wins the block.
#[test]
fn the_multi_pool_chain_replays_with_shares_of_every_pool() {
    let fx = fixture("pools-synthetic");
    let r = replay(&fx, &params_with(Arc::new(SyntheticDataset::new()))).unwrap();
    assert_eq!((r.blocks, r.steps, r.shares, r.largest_step), (40, 39, 1285, 122));
    assert_eq!(r.final_weights_hash, hex32("0416f0f9e8fc04b3bcb1ca2b3127c097007e3d8a0ca9cc97c2282f745ead53df"));
    assert_eq!(r.final_residual_hash, hex32("eada99fb5b0ed5c943552fcf2ffd2c75bd19e93a5f0783797741577e4da19b4d"));
    let mut wins = [0u32; 4];
    for b in &fx.blocks {
        wins[b.pool as usize] += 1;
    }
    assert!(wins.iter().all(|&w| w > 0), "every pool wins a block: {wins:?}");
    let all_four = fx
        .blocks
        .iter()
        .skip(1)
        .filter(|b| b.list.iter().map(|r| r.body_root).collect::<std::collections::BTreeSet<_>>().len() == 4)
        .count();
    assert!(all_four >= 30, "{all_four} of 39 lists hold the records of the four pools");
}

#[test]
fn a_multi_pool_fixture_with_a_changed_record_is_rejected() {
    let fx = fixture("pools-synthetic");
    let params = params_with(Arc::new(SyntheticDataset::new()));
    // a record of another pool with its time changed: the list no longer matches the body root
    let mut a = fx.clone();
    a.blocks[10].list[3].time ^= 1;
    assert!(replay(&a, &params).is_err());
    // two records swapped: the list is not strictly increasing
    let mut b = fx.clone();
    b.blocks[10].list.swap(0, 1);
    assert!(replay(&b, &params).is_err(), "a list out of order");
}

#[test]
fn a_modified_fixture_is_rejected() {
    let fx = fixture("synthetic-stress");
    let params = params_with(Arc::new(SyntheticDataset::new()));
    // a share nonce changed (the list stays sorted, so the block fails on the body root first)
    let mut a = fx.clone();
    a.blocks[10].list[0].nonce ^= 1;
    assert!(replay(&a, &params).unwrap_err().contains("body_root"));
    // a share nonce changed together with the body root: the block header changes, so the next parent link fails,
    // or the proof of work of the block fails
    let mut b = fx.clone();
    b.blocks[5].weights_hash[0] ^= 1;
    assert!(replay(&b, &params).unwrap_err().contains("state hashes"));
    let mut c = fx.clone();
    c.blocks[7].header.nonce ^= 1;
    assert!(replay(&c, &params).is_err());
    // losses are checked
    let mut d = fx.clone();
    if let Some(l) = d.blocks[12].loss.as_mut() {
        l[1] = LayerLoss { cut_events: l[1].cut_events + 1, ..l[1] };
    }
    assert!(replay(&d, &params).unwrap_err().contains("losses"));
    // a block of height 1 that lists a share
    let mut e = fx.clone();
    e.blocks[0].list = vec![ShareRecord { body_root: [1; 32], time: 0, nonce: 1 }];
    assert!(replay(&e, &params).is_err());
}

#[test]
fn a_fixture_round_trips_through_its_text_form() {
    let fx = fixture("synthetic-stress");
    assert_eq!(ChainFixture::parse(&fx.to_text()).unwrap(), fx);
    assert!(ChainFixture::parse("neb-chain-fixture 1\n").is_err(), "the format of the earlier share rule");
    assert!(ChainFixture::parse("neb-chain-fixture 2\n").is_err());
}

// Three pools in lockstep on the network dataset: the final hashes are those of the Rust replay and of the Python oracle.
#[test]
fn the_recorded_three_pool_mnist_chain_replays() {
    let Some(ds) = mnist() else { return };
    let r = replay(&fixture("pools-mnist"), &params_with(ds)).unwrap();
    assert_eq!((r.blocks, r.steps, r.shares, r.largest_step), (60, 59, 2000, 155));
    assert_eq!(r.final_weights_hash, hex32("c9817031e26931ee61c8a68bcbaf078fe9afb38f8b5c5cdb9ee63d159be53d0a"));
    assert_eq!(r.final_residual_hash, hex32("f28e2498ff16dc97dbc6f6003138dc87e0b34be2e09d062124ddd80e77e513c2"));
}

// Two models (Kepler and retina) mined by three pools in lockstep on the dataset set of protocol 3: block h is mined
// on slot h mod 2 and its shares train slot (h - 1) mod 2. The final hashes are those of the Rust replay and of the
// Python oracle. Needs the set file in `NEB_DATASETS` (`neb-datasets.bin`); skipped without it.
#[test]
fn the_recorded_two_model_chain_replays() {
    let fx = fixture("pools-testnet2");
    assert_eq!(fx.slots, 2);
    let Some(path) = std::env::var_os("NEB_DATASETS") else {
        eprintln!("SKIPPED: set NEB_DATASETS to neb-datasets.bin to replay the two-model chain");
        return;
    };
    let want = hex32(fx.dataset.strip_prefix("file ").expect("a file dataset"));
    let mut params = params_with(Arc::new(SyntheticDataset::new()));
    params.datasets = neb_pow::shares::load_datasets(std::path::Path::new(&path), Some(&want)).expect("the set file");
    let r = replay(&fx, &params).unwrap();
    assert_eq!((r.blocks, r.steps, r.shares, r.largest_step), (40, 39, 1394, 155));
    assert_eq!(r.final_weights_hash, hex32("83ead1e1759aabc6d442be0cb76cb1940961dd27667ff028894d3753aecec02d"));
    assert_eq!(r.final_residual_hash, hex32("79729b78e5a1c40afd38335434e3eac566339e2ba663d81af6e97c55aa07da5c"));
}
