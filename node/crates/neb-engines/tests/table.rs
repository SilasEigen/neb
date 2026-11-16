//! Differential tests of the table engine of nebhash v2 against the C reference, and of the engine rules with it:
//! the reference decides every hit, audits catch wrong tickets, a disagreement switches the engine off.

use std::sync::Arc;

use neb_block::Target;
use neb_engines::table::Table;
use neb_engines::{AUDIT_INTERVAL, Engine, EngineKind, test_models_v2};
use neb_pow::shares::{
    Arch, Dataset, Model, SharedDataset, SyntheticDataset, SyntheticDataset2, sample_grad, share_eval,
};

const ARCHS: [Arch; 2] = [Arch::Kep1, Arch::Ret2];

fn key(tag: &str, a: u32, k: u32) -> [u8; 32] {
    let mut input = tag.as_bytes().to_vec();
    for v in [a, k] {
        input.extend_from_slice(&v.to_le_bytes());
    }
    *blake3::hash(&input).as_bytes()
}

fn keys_per_case() -> u32 {
    std::env::var("NEB_DIFF_KEYS").ok().and_then(|v| v.parse().ok()).unwrap_or(256)
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

fn synthetic(arch: Arch) -> SharedDataset {
    Arc::new(SyntheticDataset2::new(arch))
}

fn initial(arch: Arch) -> Arc<Model> {
    Arc::new(Model::from_seed_arch(arch, &[0u8; 32]))
}

/// The gradient of a share is the sum of the gradients of its samples taken alone, clipped: the fact the table
/// rests on, checked on the reference itself.
#[test]
fn the_gradient_of_a_share_is_the_clipped_sum_of_its_samples() {
    for arch in ARCHS {
        let ds = SyntheticDataset2::new(arch);
        for (mi, (name, model)) in test_models_v2(arch, &ds).into_iter().enumerate() {
            for k in 0..8 {
                let want = share_eval(&model, &ds, &key("neb-table-sum", mi as u32, k));
                let mut sum = vec![0i32; arch.weights()];
                let (mut x, mut g) = (vec![0u8; arch.nin()], vec![0i32; arch.weights()]);
                for &i in &want.indices {
                    let label = ds.sample(i, &mut x);
                    sample_grad(&model, &x, label, &mut g, None);
                    sum.iter_mut().zip(&g).for_each(|(s, v)| *s += v);
                }
                sum.iter_mut().for_each(|v| *v = (*v).clamp(-256, 256));
                assert_eq!(sum, want.g1, "{} {name} key {k}", arch.name());
            }
        }
    }
}

#[test]
fn the_table_equals_the_reference_on_every_model() {
    for arch in ARCHS {
        let ds = SyntheticDataset2::new(arch);
        for (mi, (name, model)) in test_models_v2(arch, &ds).into_iter().enumerate() {
            // one thread, and more threads than fit evenly: the rows do not depend on the split
            let tables = [1, 7].map(|threads| Table::build(&model, &ds, threads, Vec::new()).expect("a table"));
            let mut acc = Vec::new();
            for k in 0..keys_per_case() {
                let key = key("neb-table-test", mi as u32, k);
                let want = share_eval(&model, &ds, &key);
                for table in &tables {
                    let got = table.eval(&key, &mut acc);
                    let case = format!("{} {name} key {k}", arch.name());
                    assert_eq!(got.indices, want.indices, "indices, {case}");
                    assert_eq!(acc[..arch.weights()], want.g1[..], "clipped gradient, {case}");
                    assert_eq!(got.digest, want.digest, "digest, {case}");
                    assert_eq!(got.ticket, want.ticket, "ticket, {case}");
                }
            }
        }
    }
}

#[test]
fn a_table_is_refused_for_a_v1_model_or_another_dataset() {
    let v1 = Model::from_seed(&[0u8; 32]);
    assert!(Table::build(&v1, &SyntheticDataset::new(), 1, Vec::new()).is_err());
    let kep1 = Model::from_seed_arch(Arch::Kep1, &[0u8; 32]);
    assert!(Table::build(&kep1, &SyntheticDataset2::new(Arch::Ret2), 1, Vec::new()).is_err());
}

#[test]
fn the_table_engine_on_a_v1_dataset_is_the_reference() {
    let engine = Engine::new(EngineKind::Table, Arc::new(SyntheticDataset::new()));
    assert_eq!((engine.requested(), engine.active()), (EngineKind::Table, EngineKind::Reference));
    assert!(engine.messages().iter().any(|m| m.contains("nebhash v2 only")), "{:?}", engine.messages());
    let prepared = engine.prepare(&Arc::new(Model::from_seed(&[0u8; 32])));
    assert!(engine.worker().attempt(&prepared, &[7u8; 32], &everything(), &nothing()).share);
}

#[test]
fn the_table_engine_is_in_use_and_raises_no_alarm() {
    for arch in ARCHS {
        let ds = synthetic(arch);
        let engine = Engine::new(EngineKind::Table, ds.clone());
        assert_eq!(engine.active(), EngineKind::Table, "{:?}", engine.messages());
        assert!(engine.messages()[0].contains("self-test passed"));
        let model = initial(arch);
        let prepared = engine.prepare(&model);
        let mut worker = engine.worker();
        // every attempt is a hit: each is verified by the reference
        for k in 0..100 {
            let key = key("neb-table-engine", 2, k);
            let a = worker.attempt(&prepared, &key, &everything(), &nothing());
            assert!(a.share && a.ticket == share_eval(&model, &*ds, &key).ticket);
        }
        // no hit at all: the reference runs only for the audits, and the tickets are those of the reference
        for k in 0..2 * AUDIT_INTERVAL {
            let key = key("neb-table-engine", 3, k as u32);
            let a = worker.attempt(&prepared, &key, &nothing(), &nothing());
            if k % 512 == 0 {
                assert_eq!(a.ticket, share_eval(&model, &*ds, &key).ticket);
            }
        }
        let s = engine.stats();
        assert_eq!((s.verified_hits, s.audits, s.mismatches, s.disabled), (100, 2, 0, false), "{}", arch.name());
    }
}

/// A second job on the same weights reuses the table; other weights get a new one, and both give the tickets of
/// the reference.
#[test]
fn the_table_follows_the_model() {
    let arch = Arch::Kep1;
    let ds = synthetic(arch);
    let engine = Engine::new(EngineKind::Table, ds.clone());
    let models = test_models_v2(arch, &*ds);
    let mut worker = engine.worker();
    for round in 0..2 {
        for (mi, (name, model)) in models.iter().enumerate() {
            let model = Arc::new(model.clone());
            for job in 0..2 {
                let prepared = engine.prepare(&model);
                for k in 0..20 {
                    let key = key("neb-table-follow", (round * 10 + mi) as u32, k + 100 * job);
                    let a = worker.attempt(&prepared, &key, &nothing(), &nothing());
                    assert_eq!(a.ticket, share_eval(&model, &*ds, &key).ticket, "{name}");
                }
            }
        }
    }
    assert_eq!(engine.stats().mismatches, 0);
    assert_eq!(engine.active(), EngineKind::Table);
}

/// A table that claims a hit it does not have is stopped by the reference: the attempt is reported as no hit, the
/// engine is switched off and the next attempts are evaluated by the reference.
#[test]
fn a_false_hit_is_refused_by_the_reference_and_switches_the_engine_off() {
    let arch = Arch::Kep1;
    let ds = synthetic(arch);
    let engine = Engine::new(EngineKind::Table, ds.clone());
    let model = initial(arch);
    let prepared = engine.prepare(&model);
    let mut worker = engine.worker();
    engine.inject_fault(1, true); // every ticket gets a zero top byte: the engine claims a hit every time
    let key =
        (0..).map(|k| key("neb-table-fault", 4, k)).find(|k| share_eval(&model, &*ds, k).ticket[31] != 0).unwrap();
    let truth = share_eval(&model, &*ds, &key).ticket;
    let a = worker.attempt(&prepared, &key, &top_byte_zero(), &nothing());
    assert!(!a.share, "the reference refuses the claimed hit");
    assert_eq!(a.ticket, truth);
    let s = engine.stats();
    assert_eq!((s.mismatches, s.disabled, s.verified_hits), (1, true, 0));
    assert_eq!(engine.active(), EngineKind::Reference);
    let b = worker.attempt(&prepared, &key, &top_byte_zero(), &nothing());
    assert_eq!((b.ticket, b.share), (truth, false));
}

/// A wrong ticket that is not a hit, with no hit anywhere, is found by the audit of one attempt in 4096.
#[test]
fn the_audit_catches_a_wrong_ticket_that_never_claims_a_hit() {
    let arch = Arch::Ret2;
    let engine = Engine::new(EngineKind::Table, synthetic(arch));
    let prepared = engine.prepare(&initial(arch));
    let mut worker = engine.worker();
    engine.inject_fault(1, false); // one bit of every ticket is wrong
    for k in 0..AUDIT_INTERVAL - 1 {
        worker.attempt(&prepared, &key("neb-table-fault", 5, k as u32), &nothing(), &nothing());
    }
    assert!(!engine.stats().disabled, "no check ran yet");
    worker.attempt(&prepared, &key("neb-table-fault", 5, 99_999), &nothing(), &nothing());
    let s = engine.stats();
    assert_eq!((s.mismatches, s.disabled), (1, true));
}

/// The check of a new model catches a table that is wrong on it: the model is then mined with the reference.
#[test]
fn the_check_of_a_new_model_catches_a_wrong_table() {
    let arch = Arch::Kep1;
    let ds = synthetic(arch);
    let engine = Engine::new(EngineKind::Table, ds.clone());
    engine.inject_fault(1, false);
    let model = initial(arch);
    let prepared = engine.prepare(&model);
    let s = engine.stats();
    assert_eq!((s.mismatches, s.disabled), (1, true));
    assert_eq!(engine.active(), EngineKind::Reference);
    let key = key("neb-table-fault", 6, 0);
    let a = engine.worker().attempt(&prepared, &key, &nothing(), &nothing());
    assert_eq!(a.ticket, share_eval(&model, &*ds, &key).ticket);
}

#[test]
fn the_table_name_parses() {
    assert_eq!(EngineKind::parse("table"), Ok(EngineKind::Table));
    assert_eq!(EngineKind::Table.name(), "table");
}
