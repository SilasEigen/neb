//! Determinism of the share arithmetic: the same bits whatever the number of threads, the partition of the work,
//! the batch size, the order of the shares in a step and the platform. The arithmetic is integer only, so these
//! tests must hold on every compiler; the pinned digests below are the same on Windows (MSVC) and Linux (gcc).

use neb_pow::shares::{GradSum, Model, N1, N2, ShareBuffers, SharedDataset, SyntheticDataset, train_step};

fn key(j: u32) -> [u8; 32] {
    let mut input = b"neb-determinism-key".to_vec();
    input.extend_from_slice(&j.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

/// A model that has taken a few steps, so that it has non-zero residuals and weights.
fn trained(ds: &SyntheticDataset) -> Model {
    let mut m = Model::from_seed(&[3u8; 32]);
    for s in 0..6 {
        let keys: Vec<[u8; 32]> = (0..16).map(|j| key(1000 + 16 * s + j)).collect();
        train_step(&mut m, ds, &keys).unwrap();
    }
    m
}

fn tickets_sequential(m: &Model, ds: &SyntheticDataset, keys: &[[u8; 32]]) -> Vec<[u8; 32]> {
    let mut b = ShareBuffers::new();
    keys.iter().map(|k| b.ticket(m, ds, k)).collect()
}

/// `threads` workers take the indexes of `order` in `chunk`-sized packets from a shared counter.
fn tickets_parallel(
    m: &Model,
    ds: &SyntheticDataset,
    keys: &[[u8; 32]],
    threads: usize,
    chunk: usize,
    order: &[usize],
) -> Vec<[u8; 32]> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(vec![[0u8; 32]; keys.len()]);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut b = ShareBuffers::new();
                loop {
                    let start = next.fetch_add(chunk, std::sync::atomic::Ordering::Relaxed);
                    if start >= order.len() {
                        break;
                    }
                    for &i in &order[start..(start + chunk).min(order.len())] {
                        let t = b.ticket(m, ds, &keys[i]);
                        out.lock().unwrap()[i] = t;
                    }
                }
            });
        }
    });
    out.into_inner().unwrap()
}

fn digest(tickets: &[[u8; 32]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for t in tickets {
        h.update(t);
    }
    *h.finalize().as_bytes()
}

#[test]
fn tickets_do_not_depend_on_threads_chunks_or_order() {
    let ds = SyntheticDataset::new();
    let m = trained(&ds);
    let keys: Vec<[u8; 32]> = (0..1024).map(key).collect();
    let want = tickets_sequential(&m, &ds, &keys);
    let forward: Vec<usize> = (0..keys.len()).collect();
    let backward: Vec<usize> = (0..keys.len()).rev().collect();
    let strided: Vec<usize> = (0..7).flat_map(|r| (r..keys.len()).step_by(7)).collect();
    for (threads, chunk, order) in [
        (1, 1, &forward),
        (4, 1, &forward),
        (4, 8, &backward),
        (32, 3, &strided),
        (32, 64, &forward),
        (7, 5, &backward),
    ] {
        assert_eq!(tickets_parallel(&m, &ds, &keys, threads, chunk, order), want, "{threads} threads, chunk {chunk}");
    }
    // pinned: the same digest on every platform and compiler
    let pinned = "d80ad45e32c55e7f1d8c70144d7b0f432a28252947c2302e153109c7164983d6";
    let got: String = digest(&want).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, pinned);
}

#[test]
fn a_step_does_not_depend_on_the_order_of_its_shares_or_on_how_they_were_summed() {
    let ds = SyntheticDataset::new();
    let m = trained(&ds);
    let keys: Vec<[u8; 32]> = (0..200).map(|j| key(5000 + j)).collect();
    let step = |order: &[usize]| {
        let mut sum = GradSum::new();
        for &i in order {
            sum.add(&m, &ds, &keys[i]);
        }
        let mut next = m.clone();
        sum.apply(&mut next);
        next.hashes()
    };
    let forward: Vec<usize> = (0..keys.len()).collect();
    let backward: Vec<usize> = forward.iter().rev().copied().collect();
    let shuffled: Vec<usize> = (0..keys.len()).map(|i| (i * 37 + 11) % keys.len()).collect();
    let want = step(&forward);
    assert_eq!(step(&backward), want);
    assert_eq!(step(&shuffled), want);

    // partial sums from several threads, added together afterwards: integer sums are associative
    let partials: Vec<(Vec<i32>, Vec<i32>)> = std::thread::scope(|s| {
        let handles: Vec<_> = keys
            .chunks(keys.len().div_ceil(5))
            .map(|part| {
                let (m, ds) = (&m, &ds);
                s.spawn(move || {
                    let mut sum = GradSum::new();
                    for k in part {
                        sum.add(m, ds, k);
                    }
                    let (a, b) = sum.sums();
                    (a.to_vec(), b.to_vec())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut gs1, mut gs2) = (vec![0i32; N1], vec![0i32; N2]);
    for (a, b) in &partials {
        gs1.iter_mut().zip(a).for_each(|(s, x)| *s += x);
        gs2.iter_mut().zip(b).for_each(|(s, x)| *s += x);
    }
    let mut combined = m.clone();
    combined.apply_step(&gs1, &gs2);
    assert_eq!(combined.hashes(), want);
    let hex: String = want.0.iter().chain(want.1.iter()).map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex,
        "1b95b23dfde2c685c70e983771073fa127c0b7bd83a1ea55437df1abb51d2b2e5e597c2d4c36daa95c24362ffeb6815f9d7f5564107e31e6b3cd6b9136719f86"
    );
}

#[test]
fn a_shared_dataset_gives_the_same_tickets_on_every_thread_and_every_repeat() {
    let ds: SharedDataset = std::sync::Arc::new(SyntheticDataset::new());
    let m = std::sync::Arc::new(Model::from_seed(&[8u8; 32]));
    let keys: Vec<[u8; 32]> = (0..64).map(key).collect();
    let first: Vec<[u8; 32]> = {
        let mut b = ShareBuffers::new();
        keys.iter().map(|k| b.ticket(&m, &*ds, k)).collect()
    };
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let (ds, m, keys) = (ds.clone(), m.clone(), keys.clone());
            std::thread::spawn(move || {
                let mut b = ShareBuffers::new();
                (0..5).map(|_| keys.iter().map(|k| b.ticket(&m, &*ds, k)).collect::<Vec<_>>()).collect::<Vec<_>>()
            })
        })
        .collect();
    for h in handles {
        for repeat in h.join().unwrap() {
            assert_eq!(repeat, first);
        }
    }
}
