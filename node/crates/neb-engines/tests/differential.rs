//! Differential tests of the AVX-512 engine against the C reference, and of the rules that the miners rely on:
//! the reference decides every hit, audits catch wrong tickets, a disagreement switches the engine off.
//! Tests that need AVX-512 print SKIPPED where the processor lacks it; with `NEB_REQUIRE_AVX512=1` that is a failure.

use std::sync::Arc;

use neb_block::Target;
use neb_engines::avx512::{self, FlatDataset, PreparedWeights, Scratch};
use neb_engines::{AUDIT_INTERVAL, Engine, EngineKind, test_models};
use neb_pow::shares::{Dataset, GradSum, Model, N1, N2, NIN, SharedDataset, SyntheticDataset, share_eval};

fn available() -> bool {
    if avx512::available() {
        return true;
    }
    assert!(
        std::env::var_os("NEB_REQUIRE_AVX512").is_none(),
        "NEB_REQUIRE_AVX512 is set: {:?}",
        avx512::unavailable_reason()
    );
    eprintln!("SKIPPED: {}", avx512::unavailable_reason().unwrap_or("?"));
    false
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 512 images in which 80 % of the pixels are 0, like MNIST: the sparse variant skips most of the work.
#[derive(Debug)]
struct MostlyZero;
impl Dataset for MostlyZero {
    fn n(&self) -> u32 {
        512
    }
    fn get(&self, index: u32) -> ([u8; NIN], u8) {
        let mut s = 0x1000_0000 + index as u64;
        let x =
            std::array::from_fn(
                |_| if splitmix(&mut s).is_multiple_of(5) { 1 + (splitmix(&mut s) % 7) as u8 } else { 0 },
            );
        (x, (splitmix(&mut s) % 10) as u8)
    }
}

/// 64 images that are all 0 or all 7, some with scattered pixels: the corners of the sparse variant.
#[derive(Debug)]
struct Extreme;
impl Dataset for Extreme {
    fn n(&self) -> u32 {
        64
    }
    fn get(&self, index: u32) -> ([u8; NIN], u8) {
        let mut x = [if index & 1 == 1 { 7 } else { 0 }; NIN];
        if index % 5 == 4 {
            let mut s = index as u64;
            for p in (0..NIN).step_by(3) {
                x[p] = (splitmix(&mut s) & 7) as u8;
            }
        }
        (x, (index % 10) as u8)
    }
}

fn key(tag: &str, a: u32, b: u32, k: u32) -> [u8; 32] {
    let mut input = tag.as_bytes().to_vec();
    for v in [a, b, k] {
        input.extend_from_slice(&v.to_le_bytes());
    }
    *blake3::hash(&input).as_bytes()
}

fn keys_per_case() -> u32 {
    std::env::var("NEB_DIFF_KEYS").ok().and_then(|v| v.parse().ok()).unwrap_or(512)
}

#[test]
fn avx512_equals_the_reference_on_every_model_and_dataset() {
    if !available() {
        return;
    }
    let synthetic = SyntheticDataset::new();
    let datasets: Vec<(&str, Box<dyn Dataset>)> = vec![
        ("synthetic", Box::new(SyntheticDataset::new())),
        ("mostly-zero", Box::new(MostlyZero)),
        ("extreme", Box::new(Extreme)),
    ];
    let models = test_models(&synthetic);
    let mut scratch = Scratch::new().unwrap();
    let (mut g1, mut g2) = (vec![0i32; N1], vec![0i32; N2]);
    let (mut evaluations, mut gradients) = (0u64, 0u64);
    for (mi, (mname, model)) in models.iter().enumerate() {
        let weights = PreparedWeights::new(model).unwrap();
        for (di, (dname, ds)) in datasets.iter().enumerate() {
            let flat = FlatDataset::new(&**ds).unwrap();
            for k in 0..keys_per_case() {
                let key = key("neb-avx512-differential", mi as u32, di as u32, k);
                let want = share_eval(model, &**ds, &key);
                for sparse in [false, true] {
                    let got = if k % 16 == 0 {
                        g1.fill(0x5a5a_5a5a);
                        g2.fill(0x5a5a_5a5a);
                        let got = scratch.eval_with_gradients(&weights, &flat, &key, sparse, &mut g1, &mut g2);
                        assert!(
                            g1 == want.g1 && g2 == want.g2,
                            "gradients: model {mname}, dataset {dname}, key {k}, sparse {sparse}"
                        );
                        gradients += 1;
                        got
                    } else {
                        scratch.eval(&weights, &flat, &key, sparse)
                    };
                    assert_eq!(
                        got.indices, want.indices,
                        "indices: model {mname}, dataset {dname}, key {k}, sparse {sparse}"
                    );
                    assert_eq!(
                        got.digest, want.digest,
                        "digest: model {mname}, dataset {dname}, key {k}, sparse {sparse}"
                    );
                    assert_eq!(
                        got.ticket, want.ticket,
                        "ticket: model {mname}, dataset {dname}, key {k}, sparse {sparse}"
                    );
                    evaluations += 1;
                }
            }
        }
    }
    eprintln!("{evaluations} evaluations ({gradients} with gradients) equal to the reference");
    assert_eq!(evaluations, 5 * 3 * u64::from(keys_per_case()) * 2);
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

/// The published vectors (`vectors/shares.txt`): every share of the 24 steps, on the model that the reference
/// leaves, evaluated by both variants of the engine.
#[test]
fn avx512_reproduces_the_published_vectors() {
    if !available() {
        return;
    }
    let text = std::fs::read_to_string(format!("{}/../../../vectors/shares.txt", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let ds = SyntheticDataset::new();
    let flat = FlatDataset::new(&ds).unwrap();
    let mut scratch = Scratch::new().unwrap();
    let mut model = Model::from_seed(&[0u8; 32]);
    let mut pending: Vec<[u8; 32]> = Vec::new();
    let mut checked = 0;
    for line in text.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        match f[0] {
            "share" => {
                let (s, j): (u32, u32) = (f[1].parse().unwrap(), f[2].parse().unwrap());
                let mut input = b"neb-test-share".to_vec();
                input.extend_from_slice(&s.to_le_bytes());
                input.extend_from_slice(&j.to_le_bytes());
                let key = *blake3::hash(&input).as_bytes();
                let weights = PreparedWeights::new(&model).unwrap();
                for sparse in [false, true] {
                    let got = scratch.eval(&weights, &flat, &key, sparse);
                    assert_eq!(got.digest.to_vec(), hex(f[3]), "step {s} share {j} digest, sparse {sparse}");
                    assert_eq!(got.ticket.to_vec(), hex(f[4]), "step {s} share {j} ticket, sparse {sparse}");
                    let idx: Vec<u32> = f[5..].iter().map(|x| x.parse().unwrap()).collect();
                    assert_eq!(got.indices.to_vec(), idx, "step {s} share {j} indices, sparse {sparse}");
                }
                pending.push(key);
                checked += 1;
            }
            "state" => {
                let mut sum = GradSum::new();
                for k in &pending {
                    sum.add(&model, &ds, k);
                }
                sum.apply(&mut model);
                pending.clear();
                let (wh, rh) = model.hashes();
                assert_eq!(wh.to_vec(), hex(f[3]));
                assert_eq!(rh.to_vec(), hex(f[4]));
            }
            _ => {}
        }
    }
    assert_eq!(checked, 543);
}

fn synthetic() -> SharedDataset {
    Arc::new(SyntheticDataset::new())
}

fn nothing() -> Target {
    Target([0u8; 32])
}

fn everything() -> Target {
    Target([0xFF; 32])
}

/// Admits a ticket whose top byte is 0 (one in 256).
fn top_byte_zero() -> Target {
    let mut t = [0xFFu8; 32];
    t[31] = 0;
    Target(t)
}

fn prepared_initial(engine: &Engine) -> Arc<neb_engines::Prepared> {
    engine.prepare(&Arc::new(Model::from_seed(&[0u8; 32])))
}

#[test]
fn the_reference_engine_decides_like_the_share_buffers() {
    let engine = Engine::reference(synthetic());
    assert_eq!((engine.requested(), engine.active()), (EngineKind::Reference, EngineKind::Reference));
    let prepared = prepared_initial(&engine);
    let mut worker = engine.worker();
    let model = prepared.model().clone();
    for k in 0..40 {
        let key = key("neb-engine-test", 1, 1, k);
        let a = worker.attempt(&prepared, &key, &top_byte_zero(), &nothing());
        let want = share_eval(&model, &SyntheticDataset::new(), &key).ticket;
        assert_eq!(a.ticket, want);
        assert_eq!(a.share, want[31] == 0);
        assert!(!a.block);
    }
    assert_eq!(engine.stats(), Default::default());
}

#[test]
fn a_requested_engine_falls_back_with_a_message_when_it_cannot_run() {
    let engine = Engine::new_pretending_unavailable(EngineKind::Avx512Sparse, synthetic());
    assert_eq!((engine.requested(), engine.active()), (EngineKind::Avx512Sparse, EngineKind::Reference));
    assert!(
        engine.messages().iter().any(|m| m.contains("not available") && m.contains("using the reference")),
        "{:?}",
        engine.messages()
    );
    // and it still mines, with the reference
    let prepared = prepared_initial(&engine);
    let a = engine.worker().attempt(&prepared, &[7u8; 32], &everything(), &nothing());
    assert!(a.share && !a.block);
}

#[test]
fn an_engine_that_passes_its_self_test_is_in_use_and_raises_no_alarm() {
    if !available() {
        return;
    }
    for kind in [EngineKind::Avx512Dense, EngineKind::Avx512Sparse] {
        let engine = Engine::new(kind, synthetic());
        assert_eq!(engine.active(), kind, "{:?}", engine.messages());
        assert!(engine.messages()[0].contains("self-test passed"));
        let prepared = prepared_initial(&engine);
        let mut worker = engine.worker();
        // every attempt is a hit: each is verified by the reference
        for k in 0..100 {
            let a = worker.attempt(&prepared, &key("neb-engine-test", 2, 2, k), &everything(), &nothing());
            assert!(a.share);
        }
        // no hit at all: the reference runs only for the audits
        for k in 0..3 * AUDIT_INTERVAL {
            worker.attempt(&prepared, &key("neb-engine-test", 3, 3, k as u32), &nothing(), &nothing());
        }
        let s = engine.stats();
        assert_eq!((s.verified_hits, s.audits, s.mismatches, s.disabled), (100, 3, 0, false), "{kind:?}");
    }
}

/// An engine that claims a hit it does not have is stopped by the reference: the attempt is reported as no hit,
/// the engine is switched off and the next attempts are evaluated by the reference.
#[test]
fn a_false_hit_is_refused_by_the_reference_and_switches_the_engine_off() {
    if !available() {
        return;
    }
    let engine = Engine::new(EngineKind::Avx512Sparse, synthetic());
    assert_eq!(engine.active(), EngineKind::Avx512Sparse);
    let prepared = prepared_initial(&engine);
    let mut worker = engine.worker();
    engine.inject_fault(1, true); // every ticket gets a zero top byte: the engine claims a hit every time
    let model = prepared.model().clone();
    let key = key("neb-engine-test", 4, 4, 0);
    let truth = share_eval(&model, &SyntheticDataset::new(), &key).ticket;
    assert_ne!(truth[31], 0, "the chosen key must not be a real hit");
    let a = worker.attempt(&prepared, &key, &top_byte_zero(), &nothing());
    assert!(!a.share, "the reference refuses the claimed hit");
    assert_eq!(a.ticket, truth);
    let s = engine.stats();
    assert_eq!((s.mismatches, s.disabled, s.verified_hits), (1, true, 0));
    assert_eq!(engine.active(), EngineKind::Reference);
    assert!(engine.messages().iter().any(|m| m.contains("switched off")), "{:?}", engine.messages());
    // later attempts come from the reference, whatever the engine would say
    let b = worker.attempt(&prepared, &key, &top_byte_zero(), &nothing());
    assert_eq!((b.ticket, b.share), (truth, false));
    assert_eq!(engine.stats().mismatches, 1);
}

/// A wrong ticket that is not a hit, with no hit anywhere, is found by the audit of one attempt in 4096.
#[test]
fn the_audit_catches_a_wrong_ticket_that_never_claims_a_hit() {
    if !available() {
        return;
    }
    let engine = Engine::new(EngineKind::Avx512Dense, synthetic());
    let prepared = prepared_initial(&engine);
    let mut worker = engine.worker();
    engine.inject_fault(1, false); // one bit of every ticket is wrong
    for k in 0..AUDIT_INTERVAL - 1 {
        worker.attempt(&prepared, &key("neb-engine-test", 5, 5, k as u32), &nothing(), &nothing());
    }
    assert!(!engine.stats().disabled, "no check ran yet");
    worker.attempt(&prepared, &key("neb-engine-test", 5, 5, 99_999), &nothing(), &nothing());
    let s = engine.stats();
    assert_eq!((s.mismatches, s.disabled), (1, true));
}

/// The check of a new model catches an engine that is wrong on it.
#[test]
fn the_check_of_a_new_model_catches_a_wrong_engine() {
    if !available() {
        return;
    }
    let engine = Engine::new(EngineKind::Avx512Sparse, synthetic());
    engine.inject_fault(1, false);
    let _ = prepared_initial(&engine);
    let s = engine.stats();
    assert_eq!((s.mismatches, s.disabled), (1, true));
    assert_eq!(engine.active(), EngineKind::Reference);
}

#[test]
fn engine_names_parse() {
    assert_eq!(EngineKind::parse("ref"), Ok(EngineKind::Reference));
    assert_eq!(EngineKind::parse("avx512"), Ok(EngineKind::Avx512Sparse));
    assert_eq!(EngineKind::parse("avx512-dense"), Ok(EngineKind::Avx512Dense));
    assert!(EngineKind::parse("avx2").is_err());
}
