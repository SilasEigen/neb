//! Tests of the share wrapper: vectors, model steps, packed weights and dataset files.

use super::vectors::test_share_key;
use super::*;

fn vectors(name: &str) -> String {
    let p = format!("{}/../../../vectors/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {p}: {e}"))
}

// The C reference reproduces every index, digest, ticket and state hash of
// the published vectors: the consensus-critical arithmetic is pinned.
#[test]
fn vectors_replay_bit_exact() {
    assert_eq!(replay_vectors(&vectors("shares.txt")), Ok((24, 543)));
}

// Replay compares every digest, not only the state hashes: one flipped hex
// digit in share 5 of step 0 must be reported for that share.
#[test]
fn a_changed_vector_is_rejected() {
    let text = vectors("shares.txt");
    let at = text.find("share 0 5 ").unwrap() + "share 0 5 ".len();
    let mut bad = text.into_bytes();
    bad[at] = if bad[at] == b'0' { b'1' } else { b'0' };
    let err = replay_vectors(&String::from_utf8(bad).unwrap()).unwrap_err();
    assert!(err.contains("step 0 share 5: digest mismatch"), "{err}");
}

// The Rust copy of the test dataset (spec/shares.md section 9) equals the C
// generator that produced the vectors.
#[test]
fn synthetic_dataset_matches_the_c_generator() {
    // BLAKE3 over the 2048 samples, each as 784 pixels and its label byte,
    // equals the same hash over the output of pow/tests/syn_dataset.c.
    let ds = SyntheticDataset::new();
    let mut all = Vec::with_capacity(SYN_N as usize * (NIN + 1));
    for i in 0..SYN_N {
        let (x, label) = ds.get(i);
        all.extend_from_slice(&x);
        all.push(label);
    }
    let want = "5beb173c9f2206f0d5ff1493f2dbf5bebfbfab4fe3beaeace9c557b12a5eb35e";
    assert_eq!(hex_bytes::<32>(want).unwrap(), *blake3::hash(&all).as_bytes());
}

// `MAX_SHARES` bounds the int32 sums in C; the Rust copy must not drift
// from `NEB_MAX_SHARES`.
#[test]
fn the_share_bound_is_the_one_of_the_c_header() {
    let header = std::fs::read_to_string(format!("{}/../../../pow/ref/shares.h", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let line = header.lines().find(|l| l.contains("define NEB_MAX_SHARES")).unwrap();
    let value = line.split_whitespace().nth(2).unwrap().trim_end_matches('u');
    assert_eq!(value.parse::<usize>().unwrap(), MAX_SHARES);
}

// The step is defined for n >= 1 (spec/shares.md section 6); a block
// without shares leaves the state unchanged and never calls it.
#[test]
fn a_step_needs_at_least_one_share() {
    let mut model = Model::from_seed(&[0u8; 32]);
    assert!(train_step(&mut model, &SyntheticDataset::new(), &[]).is_err());
}

// A full sum refuses one more share instead of letting the int32 sums wrap.
#[test]
fn a_full_gradient_sum_refuses_one_more_share() {
    let mut sum = GradSum::new();
    sum.count = MAX_SHARES;
    let model = Model::from_seed(&[0u8; 32]);
    let r = sum.try_add(&model, &SyntheticDataset::new(), &test_share_key(0, 0));
    assert_eq!(r, Err(StepError::TooManyShares(MAX_SHARES + 1)));
    assert_eq!(sum.count(), MAX_SHARES);
}

// The vectors pin bits, not learning: 24 steps of 32 shares must raise the
// number of correctly classified synthetic samples by more than 500 of 2048.
#[test]
fn training_improves_the_classification() {
    let ds = SyntheticDataset::new();
    let right = |m: &Model| {
        (0..SYN_N)
            .filter(|&i| {
                let (x, label) = ds.get(i);
                m.predict(&x) == label
            })
            .count()
    };
    let mut model = Model::from_seed(&[0u8; 32]);
    let before = right(&model);
    for s in 0..24 {
        let keys: Vec<[u8; 32]> = (0..32).map(|j| test_share_key(s, j)).collect();
        train_step(&mut model, &ds, &keys).unwrap();
    }
    assert!(right(&model) > before + 500, "{before} before, {} after", right(&model));
}

// The weights of a `model` frame rebuild the same model, and their packing
// is the one hashed into `weights_hash` (spec/shares.md section 7).
#[test]
fn packed_weights_round_trip_and_match_the_weights_hash() {
    let model = Model::from_seed(&[7u8; 32]);
    let packed = model.packed_weights();
    assert_eq!(packed.len(), PACKED_WEIGHTS);
    let copy = Model::from_packed_weights(&packed).unwrap();
    assert_eq!(copy.weights_hash(), model.weights_hash());
    assert_eq!(copy.packed_weights(), packed);
    let mut tagged = b"neb-model-weights".to_vec();
    tagged.extend_from_slice(&packed);
    assert_eq!(*blake3::hash(&tagged).as_bytes(), model.weights_hash());
    assert!(Model::from_packed_weights(&packed[1..]).is_err());
}

// The allocation-free paths (`ShareBuffers`, `GradSum`) agree with the
// allocating ones, and `Model::clone` is a deep copy.
#[test]
fn reusing_buffers_gives_the_same_tickets_and_sums() {
    let ds = SyntheticDataset::new();
    let model = Model::from_seed(&[1u8; 32]);
    let keys: Vec<[u8; 32]> = (0..5).map(|j| test_share_key(0, j)).collect();
    let mut buffers = ShareBuffers::new();
    let mut sum = GradSum::new();
    let mut stepped = model.clone();
    for k in &keys {
        let want = share_eval(&model, &ds, k).ticket;
        assert_eq!(buffers.ticket(&model, &ds, k), want);
        assert_eq!(sum.add(&model, &ds, k), want);
    }
    sum.apply(&mut stepped);
    let mut reference = model.clone();
    train_step(&mut reference, &ds, &keys).unwrap();
    assert_eq!(stepped.hashes(), reference.hashes());
    assert_eq!(model.hashes(), Model::from_seed(&[1u8; 32]).hashes(), "clone is a deep copy");
}

// A dataset file returns the samples it was built from and is refused when
// truncated, with a pixel above 7, a label above 9 or a wrong magic
// (spec/shares.md section 11.6).
#[test]
fn a_dataset_file_round_trips_and_is_checked() {
    let ds = SyntheticDataset::new();
    let mut bytes = b"NMD1".to_vec();
    bytes.extend_from_slice(&64u32.to_le_bytes());
    for i in 0..64 {
        let (x, label) = ds.get(i);
        bytes.extend_from_slice(&x);
        bytes.push(label);
    }
    let file = FileDataset::from_bytes(&bytes).unwrap();
    assert_eq!(file.n(), 64);
    assert_eq!(file.get(5), ds.get(5));
    assert!(FileDataset::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut bad_pixel = bytes.clone();
    bad_pixel[8] = 8;
    assert!(FileDataset::from_bytes(&bad_pixel).is_err());
    let mut bad_label = bytes.clone();
    bad_label[8 + NIN] = 10;
    assert!(FileDataset::from_bytes(&bad_label).is_err());
    bytes[0] = b'X';
    assert!(FileDataset::from_bytes(&bytes).is_err());
}

#[test]
fn nebhash2_vectors_replay_bit_exact() {
    let (steps, shares) = replay_vectors2(&vectors("nebhash2.txt")).unwrap();
    assert_eq!((steps, shares), (12, 174));
    let mut bad = vectors("nebhash2.txt");
    let at = bad.find("state 2 3").unwrap() + 20;
    let c = if bad.as_bytes()[at] == b'0' { "1" } else { "0" };
    bad.replace_range(at..at + 1, c);
    assert!(replay_vectors2(&bad).is_err());
}

#[test]
fn nebhash2_packed_weights_round_trip_and_datasets_parse() {
    for arch in [Arch::Kep1, Arch::Ret2] {
        let m = Model::from_seed_arch(arch, &[7u8; 32]);
        assert_eq!(m.arch(), arch);
        assert_eq!(m.weights().len(), arch.weights());
        let back = Model::from_packed(arch, &m.packed_weights()).unwrap();
        assert_eq!(back.weights_hash(), m.weights_hash());
        assert!(Model::from_packed(arch, &m.packed_weights()[1..]).is_err());
        // a NMD2 file of three samples, alone and inside a set with a v1 file
        let ds = SyntheticDataset2::new(arch);
        let mut file = b"NMD2".to_vec();
        file.extend_from_slice(&arch.id().unwrap().to_le_bytes());
        file.extend_from_slice(&3u32.to_le_bytes());
        let mut x = vec![0u8; arch.nin()];
        for i in 0..3 {
            let label = ds.sample(i, &mut x);
            file.extend_from_slice(&x);
            file.push(label);
        }
        let parsed = FileDataset::from_bytes(&file).unwrap();
        assert_eq!((parsed.n(), parsed.arch()), (3, arch));
        let mut y = vec![0u8; arch.nin()];
        assert_eq!(parsed.sample(2, &mut y), ds.sample(2, &mut x));
        assert_eq!(x, y);
        let mut v1 = b"NMD1".to_vec();
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&[0u8; NIN + 1]);
        let mut set = b"NMDS".to_vec();
        set.extend_from_slice(&2u32.to_le_bytes());
        set.extend_from_slice(&v1);
        set.extend_from_slice(&file);
        let both = datasets_from_bytes(&set).unwrap();
        assert_eq!((both[0].arch(), both[1].arch()), (Arch::V1, arch));
        let mut label2 = file.clone();
        *label2.last_mut().unwrap() = 2;
        assert!(FileDataset::from_bytes(&label2).is_err());
    }
}
