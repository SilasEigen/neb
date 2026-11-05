//! The self-test of an engine: the models and datasets of the C test (`pow/tests/test_shares_avx512.c`), a smaller
//! number of keys, compared with the reference.

use neb_pow::shares::{Dataset, Model, N1, N2, SyntheticDataset, share_eval, train_step};

use crate::avx512::{FlatDataset, PreparedWeights, Scratch};
use crate::engine::EngineKind;

/// Keys compared (digest, ticket, indices) per model and dataset, and the gradients of one key in `GRADIENT_EVERY`.
const KEYS: u32 = 96;
const GRADIENT_EVERY: u32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelfTestReport {
    pub tickets: u64,
    pub gradients: u64,
}

fn key_for(tag: &str, model: usize, k: u32) -> [u8; 32] {
    let mut input = b"neb-engine-selftest-".to_vec();
    input.extend_from_slice(tag.as_bytes());
    input.extend_from_slice(&(model as u32).to_le_bytes());
    input.extend_from_slice(&k.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

/// The models of the self-test and of the differential tests: the initial state, a model trained for a few
/// steps on `dataset`, and three models with extreme weights (all 7, all -8, alternating), whose sums reach the
/// largest values of the arithmetic.
pub fn test_models(dataset: &dyn Dataset) -> Vec<(&'static str, Model)> {
    let mut trained = Model::from_seed(&[0u8; 32]);
    for s in 0..6u32 {
        let keys: Vec<[u8; 32]> = (0..16).map(|j| key_for("train", s as usize, j)).collect();
        train_step(&mut trained, dataset, &keys).expect("a step of 16 shares");
    }
    let (r1, r2) = (vec![0i32; N1], vec![0i32; N2]);
    let uniform = |w: i8| Model::from_state(&vec![w; N1], &vec![w; N2], &r1, &r2).expect("valid state");
    let alternating = {
        let w1: Vec<i8> = (0..N1).map(|e| if e & 1 == 1 { 7 } else { -8 }).collect();
        let w2: Vec<i8> = (0..N2).map(|e| if (e / 3) & 1 == 1 { -8 } else { 7 }).collect();
        Model::from_state(&w1, &w2, &r1, &r2).expect("valid state")
    };
    vec![
        ("initial", Model::from_seed(&[0u8; 32])),
        ("trained", trained),
        ("all-7", uniform(7)),
        ("all--8", uniform(-8)),
        ("alternating", alternating),
    ]
}

/// Compares the engine `kind` with the reference on the given dataset (flattened in `flat`) and on the synthetic
/// dataset, for every test model. The first difference is an error that names the case.
pub(crate) fn run(kind: EngineKind, dataset: &dyn Dataset, flat: &FlatDataset) -> Result<SelfTestReport, String> {
    let sparse = kind == EngineKind::Avx512Sparse;
    let synthetic = SyntheticDataset::new();
    let synthetic_flat = FlatDataset::new(&synthetic)?;
    let mut scratch = Scratch::new()?;
    let (mut g1, mut g2) = (vec![0i32; N1], vec![0i32; N2]);
    let mut report = SelfTestReport { tickets: 0, gradients: 0 };
    for (dname, ds, fl) in [("dataset", dataset, flat), ("synthetic", &synthetic as &dyn Dataset, &synthetic_flat)] {
        for (mi, (mname, model)) in test_models(ds).into_iter().enumerate() {
            let weights = PreparedWeights::new(&model)?;
            for k in 0..KEYS {
                let key = key_for(dname, mi, k);
                let want = share_eval(&model, ds, &key);
                let case = || format!("model {mname}, {dname}, key {k}");
                let got = if k % GRADIENT_EVERY == 0 {
                    g1.fill(0x5a5a_5a5a);
                    g2.fill(0x5a5a_5a5a);
                    let got = scratch.eval_with_gradients(&weights, fl, &key, sparse, &mut g1, &mut g2);
                    if g1 != want.g1 || g2 != want.g2 {
                        return Err(format!("clipped gradients differ ({})", case()));
                    }
                    report.gradients += 1;
                    got
                } else {
                    scratch.eval(&weights, fl, &key, sparse)
                };
                if got.indices != want.indices {
                    return Err(format!("sample indices differ ({})", case()));
                }
                if got.digest != want.digest {
                    return Err(format!("gradient digest differs ({})", case()));
                }
                if got.ticket != want.ticket {
                    return Err(format!("ticket differs ({})", case()));
                }
                report.tickets += 1;
            }
        }
    }
    Ok(report)
}
