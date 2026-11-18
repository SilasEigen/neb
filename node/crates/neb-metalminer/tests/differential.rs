//! Metal only: on a platform without Metal the miner does not exist, so there is nothing to compare.
#![cfg(target_os = "macos")]

use neb_metalminer::MetalEngine;
use neb_pow::shares::{Model, SyntheticDataset, share_eval};

fn key(seed: &[u8; 32], k: u32) -> [u8; 32] {
    let mut input = b"neb-metal-test".to_vec();
    input.extend_from_slice(seed);
    input.extend_from_slice(&k.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

#[test]
fn bare_keys_match_the_reference_on_several_models() {
    let ds = SyntheticDataset::new();
    let mut engine = MetalEngine::new(&ds, 8).expect("a Metal device");
    for seed in 0..6u8 {
        let mut s = [0u8; 32];
        s[0] = seed;
        s[1] = seed.wrapping_mul(37);
        s[31] = 0xff - seed;
        let model = Model::from_seed(&s);
        engine.set_model(&model).expect("the model fits");
        for k in 0..8u32 {
            let key = key(&s, k);
            let got = engine.scan_key(&key).expect("the batch runs");
            let want = share_eval(&model, &ds, &key);
            assert_eq!(got.digest_bytes(), want.digest, "digest of model {seed} key {k}");
            assert_eq!(got.ticket_bytes(), want.ticket, "ticket of model {seed} key {k}");
            assert_eq!(got.indices, want.indices, "indices of model {seed} key {k}");
        }
    }
}

#[test]
fn header_mode_matches_the_reference_for_every_nonce() {
    let ds = SyntheticDataset::new();
    let mut engine = MetalEngine::new(&ds, 16).expect("a Metal device");
    let model = Model::from_seed(&[7u8; 32]);
    engine.set_model(&model).expect("the model fits");

    let mut header = [0u8; 80];
    for (i, b) in header.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(11).wrapping_add(3);
    }
    header[76..80].fill(0); // the nonce field, as the work service sends it
    let start = 1000u32;
    let records = engine.scan(&header, start, 9).expect("the batch runs").to_vec();
    assert_eq!(records.len(), 9);
    for (i, record) in records.iter().enumerate() {
        let nonce = start + i as u32;
        let mut h = header;
        h[76..80].copy_from_slice(&nonce.to_le_bytes());
        let want = share_eval(&model, &ds, &blake3::hash(&h).into());
        assert_eq!(record.ticket_bytes(), want.ticket, "ticket of nonce {nonce}");
        assert_eq!(record.digest_bytes(), want.digest, "digest of nonce {nonce}");
        assert_eq!(record.indices, want.indices, "indices of nonce {nonce}");
    }
}
