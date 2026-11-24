//! nebhash v2 on the GPU against the vectors of `vectors/` and the C reference. Metal only.
#![cfg(target_os = "macos")]

use neb_metalminer::MetalEngine;
use neb_pow::shares::{Arch, Dataset, Model, NB, SyntheticDataset2, share_eval};

fn hex32(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

/// Weights per layer and update shift per layer (`spec/nebhash-v2.md` section 3).
fn layers(arch: Arch) -> (&'static [usize], &'static [u32]) {
    match arch {
        Arch::Kep1 => (&[40, 640, 40, 640, 24576, 64], &[15, 15, 15, 15, 15, 12]),
        Arch::Ret2 => (&[72, 1152, 4608, 16384, 64], &[16, 16, 16, 16, 13]),
        Arch::V1 => unreachable!(),
    }
}

fn pack(w: &[i8]) -> Vec<u8> {
    w.chunks(2).map(|p| (p[0] as u8 & 15) | ((p[1] as u8 & 15) << 4)).collect()
}

/// The model step of `spec/nebhash-v2.md` section 5 with every update shift lowered by `drop` (section 8).
fn step(arch: Arch, w: &mut [i8], r: &mut [i32], gs: &[i32], drop: u32) {
    let (nw, su) = layers(arch);
    let mut at = 0;
    for (n, su) in nw.iter().zip(su) {
        let su = su - drop;
        let unit = 1i32 << su;
        for f in at..at + n {
            let mut v = r[f] + gs[f];
            let q = if v >= 0 { v >> su } else { -((-v) >> su) };
            let d = q.clamp(-1, 1);
            v -= d * unit;
            r[f] = v.clamp(-unit, unit);
            w[f] = (w[f] as i32 - d).clamp(-8, 7) as i8;
        }
        at += n;
    }
}

/// Replays a vector file with the GPU: every share of every step must give the digest, the ticket and the sample indices
/// of the file, on the weights that the steps before it produced.
fn replay(text: &str, drop: u32) -> usize {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    assert_eq!(lines.next(), Some("neb-nebhash2-vectors 1"));
    let mut engine: Option<(MetalEngine, SyntheticDataset2)> = None;
    let (mut arch, mut w, mut r) = (Arch::Kep1, Vec::<i8>::new(), Vec::<i32>::new());
    let mut shares: Vec<([u8; 32], [u8; 32], [u32; NB])> = Vec::new();
    let (mut keys, mut checked) = (Vec::<[u8; 32]>::new(), 0usize);
    for line in lines {
        let f: Vec<&str> = line.split(' ').collect();
        match f[0] {
            "init" => {
                arch = Arch::from_id(f[1].parse().unwrap()).unwrap();
                let model = Model::from_seed_arch(arch, &[0u8; 32]);
                w = model.weights().to_vec();
                r = vec![0; w.len()];
                let ds = SyntheticDataset2::new(arch);
                engine = Some((MetalEngine::new(&ds, 4).expect("a Metal device"), ds));
            }
            "share" => {
                let mut input = Vec::from(&b"neb-test-share2"[..]);
                input.push(arch.id().unwrap() as u8);
                input.extend_from_slice(&f[2].parse::<u32>().unwrap().to_le_bytes());
                input.extend_from_slice(&(keys.len() as u32).to_le_bytes());
                keys.push(*blake3::hash(&input).as_bytes());
                let mut idx = [0u32; NB];
                for (slot, v) in idx.iter_mut().zip(&f[6..]) {
                    *slot = v.parse().unwrap();
                }
                shares.push((hex32(f[4]), hex32(f[5]), idx));
            }
            "state" => {
                let (engine, ds) = engine.as_mut().unwrap();
                let model = Model::from_packed(arch, &pack(&w)).expect("packed weights");
                engine.set_model(&model).expect("the weights fit");
                let mut gs = vec![0i32; w.len()];
                for (j, (key, want)) in keys.iter().zip(&shares).enumerate() {
                    let got = engine.scan_key(key).expect("the attempt runs");
                    assert_eq!(got.digest_bytes(), want.0, "{} digest of share {j} of step {}", arch.name(), f[2]);
                    assert_eq!(got.ticket_bytes(), want.1, "{} ticket of share {j} of step {}", arch.name(), f[2]);
                    assert_eq!(got.indices, want.2, "{} indices of share {j} of step {}", arch.name(), f[2]);
                    // the gradient of the share, from the C reference (the GPU returns only the digest)
                    let g = share_eval(&model, ds, key).g1;
                    gs.iter_mut().zip(&g).for_each(|(s, g)| *s += g);
                    checked += 1;
                }
                step(arch, &mut w, &mut r, &gs, drop);
                keys.clear();
                shares.clear();
            }
            other => panic!("unexpected line {other:?}"),
        }
    }
    checked
}

#[test]
fn consensus_vectors_replay_on_the_gpu() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../vectors/nebhash2.txt")).unwrap();
    let n = replay(&text, 0);
    assert!(n > 100, "{n} shares");
}

#[test]
fn moving_vectors_replay_on_the_gpu() {
    // every update shift lowered by 6: a few shares move the weights, so the later steps run on trained weights
    let text =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../vectors/nebhash2_s1.txt")).unwrap();
    let n = replay(&text, 6);
    assert!(n > 100, "{n} shares");
}

/// Header mode on both architectures, over a batch larger than the scratch of one dispatch (512 attempts), so that the
/// cut between dispatches is crossed.
#[test]
fn header_mode_matches_the_reference_across_dispatches() {
    for arch in [Arch::Kep1, Arch::Ret2] {
        let ds = SyntheticDataset2::new(arch);
        let mut engine = MetalEngine::new(&ds, 640).expect("a Metal device");
        let model = Model::from_seed_arch(arch, &[9u8; 32]);
        engine.set_model(&model).expect("the weights fit");
        let mut header = [0u8; 80];
        for (i, b) in header.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(13).wrapping_add(5);
        }
        header[76..80].fill(0);
        let start = 70_000u32;
        let records = engine.scan(&header, start, 600).expect("the batch runs").to_vec();
        assert_eq!(records.len(), 600);
        for i in [0usize, 1, 255, 510, 511, 512, 513, 599] {
            let nonce = start + i as u32;
            let mut h = header;
            h[76..80].copy_from_slice(&nonce.to_le_bytes());
            let want = share_eval(&model, &ds, &blake3::hash(&h).into());
            assert_eq!(records[i].digest_bytes(), want.digest, "{} digest of nonce {nonce}", arch.name());
            assert_eq!(records[i].ticket_bytes(), want.ticket, "{} ticket of nonce {nonce}", arch.name());
            assert_eq!(records[i].indices, want.indices, "{} indices of nonce {nonce}", arch.name());
        }
    }
}

#[test]
fn a_model_of_another_architecture_is_refused() {
    let ds = SyntheticDataset2::new(Arch::Kep1);
    let mut engine = MetalEngine::new(&ds, 4).expect("a Metal device");
    assert_eq!(ds.arch(), Arch::Kep1);
    assert!(engine.set_model(&Model::from_seed_arch(Arch::Ret2, &[0u8; 32])).is_err());
    assert!(engine.set_model(&Model::from_seed(&[0u8; 32])).is_err());
}
