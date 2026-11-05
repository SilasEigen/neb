//! Numerical boundaries of the model step (spec/shares.md section 6), checked on the C reference through the
//! public API of `neb-pow` against an independent step written here in 64-bit integers from the text of the
//! specification, entry by entry.
//!
//! What the step does, per entry, with `T = 2^SU`, `S = R + G`:
//! `D = clip(tdiv(S, T), -1, 1)`, `R2 = S - D*T`, `R' = clip(R2, -T, T)`, `W' = clip(W - D, -8, 7)`.
//! The charge that the bounds discard is `lr = R2 - R'` (residual) and `wc = W' - (W - D)` (weight), and
//! `T*W' - R' = T*W - R - G + T*wc + lr` holds for every entry (the conservation identity of the step).
//! `lr != 0` exactly when `|S| > 2T`, and `wc != 0` exactly when `W = 7, D = -1` or `W = -8, D = +1`.
//! These tests pin the behaviour of the reference; none of them may be changed to follow a code change.

use neb_pow::shares::{CLIP, MAX_SHARES, Model, N1, N2, SU1, SU2, StepError};

/// One entry of the specification step, in i64. Returns `(W', R', D, lr, wc)`.
fn step_ref(w: i8, r: i32, g: i32, su: u32) -> (i8, i32, i64, i64, i64) {
    let t = 1i64 << su;
    let s = r as i64 + g as i64;
    let d = (s / t).clamp(-1, 1); // Rust truncates toward zero: tdiv
    let r2 = s - d * t;
    let r3 = r2.clamp(-t, t);
    let w2 = w as i64 - d;
    let w3 = w2.clamp(-8, 7);
    (w3 as i8, r3 as i32, d, r2 - r3, w3 - w2)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A point of the domain: weight, residual, sum of gradients.
type Point = (i8, i32, i32);
/// Weight and residual of one entry after a step.
type WeightResidual = (i8, i32);

/// Applies one step to a model that holds `l1` in the first entries of layer 1 and `l2` in the first
/// entries of layer 2 (the other entries are 0, 0, 0). Returns the points of each layer after the step.
fn apply(l1: &[Point], l2: &[Point]) -> (Vec<WeightResidual>, Vec<WeightResidual>) {
    assert!(l1.len() <= N1 && l2.len() <= N2);
    let (mut w1, mut r1, mut g1) = (vec![0i8; N1], vec![0i32; N1], vec![0i32; N1]);
    let (mut w2, mut r2, mut g2) = (vec![0i8; N2], vec![0i32; N2], vec![0i32; N2]);
    for (i, &(w, r, g)) in l1.iter().enumerate() {
        (w1[i], r1[i], g1[i]) = (w, r, g);
    }
    for (i, &(w, r, g)) in l2.iter().enumerate() {
        (w2[i], r2[i], g2[i]) = (w, r, g);
    }
    let mut m = Model::from_state(&w1, &w2, &r1, &r2).expect("a valid state");
    m.apply_step(&g1, &g2);
    let out1 = (0..l1.len()).map(|i| (m.w1()[i], m.r1()[i])).collect();
    let out2 = (0..l2.len()).map(|i| (m.w2()[i], m.r2()[i])).collect();
    (out1, out2)
}

/// Checks the C step against the reference on both layers; returns the number of entries with a residual cut
/// and with a weight clip (so that a test can assert that the cases were really exercised).
fn check(l1: &[Point], l2: &[Point]) -> (u64, u64) {
    let (o1, o2) = apply(l1, l2);
    let (mut cuts, mut clips) = (0, 0);
    for (layer, pts, outs, su) in [(1, l1, &o1, SU1), (2, l2, &o2, SU2)] {
        for (&(w, r, g), &(wn, rn)) in pts.iter().zip(outs) {
            let (want_w, want_r, d, lr, wc) = step_ref(w, r, g, su);
            assert_eq!((wn, rn), (want_w, want_r), "layer {layer}: W={w} R={r} G={g}");
            let t = 1i64 << su;
            // Conservation with losses, and the exact conditions of the losses.
            let q_before = t * w as i64 - r as i64;
            let q_after = t * wn as i64 - rn as i64;
            assert_eq!(q_after, q_before - g as i64 + t * wc + lr, "layer {layer}: identity at W={w} R={r} G={g}");
            assert_eq!(
                lr != 0,
                (r as i64 + g as i64).abs() > 2 * t,
                "layer {layer}: residual cut condition at R={r} G={g}"
            );
            assert_eq!(
                wc != 0,
                (w == 7 && d == -1) || (w == -8 && d == 1),
                "layer {layer}: weight clip condition at W={w} D={d}"
            );
            assert_eq!(d == 0, (r as i64 + g as i64).abs() < t, "layer {layer}: D condition at R={r} G={g}");
            assert!((rn as i64).abs() <= t, "layer {layer}: residual bound");
            cuts += (lr != 0) as u64;
            clips += (wc != 0) as u64;
        }
    }
    (cuts, clips)
}

fn residuals(t: i32) -> Vec<i32> {
    let mut v = vec![
        -t,
        -t + 1,
        -t + 2,
        -t / 2 - 1,
        -t / 2,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        t / 2 - 1,
        t / 2,
        t / 2 + 1,
        t - 2,
        t - 1,
        t,
    ];
    v.sort_unstable();
    v.dedup();
    v
}

/// Sums of gradients that put `S = R + G` on and around every threshold of the step, plus the sums that `n`
/// shares of maximal gradient give (`n * CLIP` for the sizes around 16 and 128, and for 256).
fn sums_for(r: i32, t: i32) -> Vec<i32> {
    let mut v = Vec::new();
    for s in [0, 1, t - 1, t, t + 1, 2 * t - 1, 2 * t, 2 * t + 1, 3 * t, 5 * t / 2, 4 * t] {
        for sign in [1, -1] {
            let target = sign * s;
            for dx in [-1i32, 0, 1] {
                v.push(target + dx - r);
            }
        }
    }
    for n in [1i32, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256] {
        v.push(n * CLIP);
        v.push(-n * CLIP);
    }
    let cap = 256 * CLIP;
    v.retain(|g| g.abs() <= cap);
    v.sort_unstable();
    v.dedup();
    v
}

#[test]
fn the_step_matches_an_independent_i64_step_on_every_threshold() {
    let weights: [i8; 7] = [-8, -7, -1, 0, 1, 6, 7];
    let mut l1 = Vec::new();
    let mut l2 = Vec::new();
    for (su, pts) in [(SU1, &mut l1), (SU2, &mut l2)] {
        let t = 1i32 << su;
        for &w in &weights {
            for r in residuals(t) {
                for g in sums_for(r, t) {
                    pts.push((w, r, g));
                }
            }
        }
    }
    // Layer 2 has only N2 entries: its points go through in packets.
    assert!(l1.len() <= N1);
    let (mut cuts, mut clips) = check(&l1, &[]);
    for chunk in l2.chunks(N2) {
        let (c, w) = check(&[], chunk);
        cuts += c;
        clips += w;
    }
    assert!(cuts > 1000 && clips > 100, "the grid must exercise both losses: {cuts} cuts, {clips} clips");
}

#[test]
fn the_step_matches_the_i64_step_on_random_states() {
    let mut s = 0x6E65_6221u64;
    let (mut cuts, mut clips) = (0, 0);
    for _ in 0..300 {
        let mut sample = |su: u32, count: usize| -> Vec<Point> {
            let t = 1i64 << su;
            (0..count)
                .map(|_| {
                    let w = match splitmix(&mut s) % 8 {
                        0 => -8,
                        1 => 7,
                        _ => (splitmix(&mut s) % 16) as i8 - 8,
                    };
                    let r = match splitmix(&mut s) % 6 {
                        0 => t,
                        1 => -t,
                        2 => 0,
                        _ => (splitmix(&mut s) % (2 * t as u64 + 1)) as i64 - t,
                    };
                    let n = [1i64, 4, 16, 17, 64, 128, 129, 256][(splitmix(&mut s) % 8) as usize];
                    let cap = n * CLIP as i64;
                    let g = (splitmix(&mut s) % (2 * cap as u64 + 1)) as i64 - cap;
                    (w, r as i32, g as i32)
                })
                .collect()
        };
        let (a, b) = (sample(SU1, N1), sample(SU2, N2));
        let (c, w) = check(&a, &b);
        cuts += c;
        clips += w;
    }
    assert!(cuts > 300_000 && clips > 1_000_000, "{cuts} cuts, {clips} clips");
}

/// The residual is cut exactly when more than `T / CLIP` shares of maximal gradient push the same way:
/// 16 shares in layer 2 and 128 in layer 1 never cut, one more can.
#[test]
fn the_residual_is_cut_above_16_shares_in_layer_2_and_128_in_layer_1() {
    for (layer, su, limit) in [(1, SU1, 128i32), (2, SU2, 16i32)] {
        let t = 1i32 << su;
        for sign in [1i32, -1] {
            let mut pts = Vec::new();
            for n in 1..=256i32 {
                pts.push((0i8, sign * t, sign * n * CLIP));
            }
            let (o1, o2) = if layer == 1 { apply(&pts, &[]) } else { apply(&[], &pts) };
            let outs = if layer == 1 { o1 } else { o2 };
            for (n, &(_, r_after)) in (1..=256i32).zip(&outs) {
                let expected = if n <= limit { sign * n * CLIP } else { sign * t };
                assert_eq!(r_after, expected, "layer {layer}: {n} shares of maximal gradient on a full residual");
                let cut = (r_after as i64 - (sign * n * CLIP) as i64) != 0;
                assert_eq!(cut, n > limit, "layer {layer}, {n} shares");
            }
        }
    }
}

/// With the largest step the consensus bound allows in the type (`MAX_SHARES` shares of maximal gradient), the
/// int32 sum `R + GS` still fits and the C result equals the reference.
#[test]
fn the_largest_step_stays_inside_the_exact_domain() {
    let cap = MAX_SHARES as i64 * CLIP as i64;
    for (su, expect_max) in [(SU1, 2_147_483_392i64), (SU2, 2_147_454_720i64)] {
        let t = 1i64 << su;
        assert_eq!(cap + t, expect_max);
        assert!(expect_max <= i32::MAX as i64);
    }
    let mut p1 = Vec::new();
    let mut p2 = Vec::new();
    for w in [-8i8, 0, 7] {
        for sign in [1i64, -1] {
            p1.push((w, (sign * (1i64 << SU1)) as i32, (sign * cap) as i32));
            p2.push((w, (sign * (1i64 << SU2)) as i32, (sign * cap) as i32));
            p1.push((w, (-sign * (1i64 << SU1)) as i32, (sign * cap) as i32));
            p2.push((w, (-sign * (1i64 << SU2)) as i32, (sign * cap) as i32));
        }
    }
    check(&p1, &p2);
}

/// A trajectory of 200 steps with random sums, the cumulative identity `T*W - R = Q0 - sum G + L` on every entry
/// of both layers, where `L` is the sum of the losses `T*wc + lr` computed from the states the C produced.
#[test]
fn the_cumulative_identity_holds_over_a_trajectory() {
    let mut s = 0x0123_4567u64;
    let mut m = Model::from_seed(&[9u8; 32]);
    let q0: Vec<Vec<i64>> = vec![
        (0..N1).map(|i| (1i64 << SU1) * m.w1()[i] as i64 - m.r1()[i] as i64).collect(),
        (0..N2).map(|i| (1i64 << SU2) * m.w2()[i] as i64 - m.r2()[i] as i64).collect(),
    ];
    let mut sum_g = [vec![0i64; N1], vec![0i64; N2]];
    let mut lost = [vec![0i64; N1], vec![0i64; N2]];
    let mut total_cuts = 0u64;
    for step in 0..200 {
        let n = [1i64, 5, 16, 17, 32, 100, 128, 129, 256][step % 9];
        let cap = n * CLIP as i64;
        let mut sample = |len: usize| -> Vec<i32> {
            (0..len).map(|_| ((splitmix(&mut s) % (2 * cap as u64 + 1)) as i64 - cap) as i32).collect()
        };
        let (g1, g2) = (sample(N1), sample(N2));
        let before = (m.w1().to_vec(), m.w2().to_vec(), m.r1().to_vec(), m.r2().to_vec());
        m.apply_step(&g1, &g2);
        for (layer, su, w, r, g) in [(0usize, SU1, &before.0, &before.2, &g1), (1, SU2, &before.1, &before.3, &g2)] {
            let t = 1i64 << su;
            for i in 0..g.len() {
                let (_, _, _, lr, wc) = step_ref(w[i], r[i], g[i], su);
                sum_g[layer][i] += g[i] as i64;
                lost[layer][i] += t * wc + lr;
                total_cuts += (lr != 0) as u64;
            }
        }
    }
    for (layer, su, w, r) in [(0usize, SU1, m.w1(), m.r1()), (1, SU2, m.w2(), m.r2())] {
        for i in 0..w.len() {
            let q = (1i64 << su) * w[i] as i64 - r[i] as i64;
            assert_eq!(q, q0[layer][i] - sum_g[layer][i] + lost[layer][i], "layer {}: entry {i}", layer + 1);
        }
    }
    assert!(total_cuts > 0, "the trajectory must include residual cuts");
}

/// Every constant that the Rust crate mirrors equals the `#define` of the C header (the build compiles that
/// header, so a drift would change what the arithmetic means).
#[test]
fn the_constants_are_those_of_the_c_header() {
    let header = std::fs::read_to_string(format!("{}/../../../pow/ref/shares.h", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let define = |name: &str| -> i64 {
        let line = header
            .lines()
            .find(|l| l.starts_with(&format!("#define NEB_{name} ")))
            .unwrap_or_else(|| panic!("NEB_{name} not found"));
        let v = line.split_whitespace().nth(2).unwrap().trim_matches(|c| c == '(' || c == ')').trim_end_matches('u');
        v.parse().unwrap_or_else(|_| panic!("bad value {v:?} for NEB_{name}"))
    };
    use neb_pow::shares::{NB, NH, NIN, NOUT, WEIGHT_MAX, WEIGHT_MIN};
    assert_eq!(define("NIN"), NIN as i64);
    assert_eq!(define("NH"), NH as i64);
    assert_eq!(define("NOUT"), NOUT as i64);
    assert_eq!(define("NB"), NB as i64);
    assert_eq!(define("SU1"), SU1 as i64);
    assert_eq!(define("SU2"), SU2 as i64);
    assert_eq!(define("CLIP"), CLIP as i64);
    assert_eq!(define("TPOS"), WEIGHT_MAX as i64);
    assert_eq!(define("TNEG"), WEIGHT_MIN as i64);
    assert_eq!(define("MAX_SHARES"), MAX_SHARES as i64);
    assert_eq!(N1 as i64, define("NH") * define("NIN"));
    assert_eq!(N2 as i64, define("NOUT") * define("NH"));
}

/// The C step negates the sum to divide toward zero, which is undefined for `R + GS = -2^31`. The consensus
/// never reaches it (`|R + GS| <= 98304` with 256 shares), and the Rust step refuses it, and everything else
/// outside `|R + GS| <= 2^31 - 1`, before calling C. The values just inside the domain are accepted and equal
/// to the reference.
#[test]
fn the_step_refuses_what_the_c_code_cannot_do_exactly() {
    for (layer, su) in [(1u8, SU1), (2u8, SU2)] {
        let t = 1i32 << su;
        let n = if layer == 1 { N1 } else { N2 };
        let model_with = |r: i32| {
            let (mut r1, mut r2) = (vec![0i32; N1], vec![0i32; N2]);
            if layer == 1 {
                r1[0] = r
            } else {
                r2[0] = r
            }
            Model::from_state(&vec![0i8; N1], &vec![0i8; N2], &r1, &r2).unwrap()
        };
        let sums_with = |g: i32| {
            let (mut g1, mut g2) = (vec![0i32; N1], vec![0i32; N2]);
            if layer == 1 {
                g1[0] = g
            } else {
                g2[0] = g
            }
            (g1, g2)
        };
        let read = |m: &Model| if layer == 1 { (m.w1()[0], m.r1()[0]) } else { (m.w2()[0], m.r2()[0]) };
        // refused: S = -2^31, S = 2^31 (R = 1), S = -2^31 - 1 (R = -1), S = 2^31 - 1 + T (R = T)
        for (r, g) in [(0, i32::MIN), (1, i32::MAX), (-1, i32::MIN), (t, i32::MAX)] {
            let mut m = model_with(r);
            let before = m.hashes();
            let (g1, g2) = sums_with(g);
            assert_eq!(
                m.try_apply_step(&g1, &g2),
                Err(StepError::OutOfRange { layer, index: 0, residual: r, sum: g }),
                "layer {layer}: R={r} G={g}"
            );
            assert_eq!(m.hashes(), before, "a refused step leaves the model unchanged");
        }
        // accepted: S = -2^31 + 1, S = 2^31 - 1, and S = -2^31 + 1 reached with a residual
        for (r, g) in [(0, i32::MIN + 1), (0, i32::MAX), (-t, i32::MIN + 1 + t), (t, i32::MAX - t)] {
            let mut m = model_with(r);
            let (g1, g2) = sums_with(g);
            m.try_apply_step(&g1, &g2).unwrap_or_else(|e| panic!("layer {layer}: R={r} G={g}: {e}"));
            let (want_w, want_r, ..) = step_ref(0, r, g, su);
            assert_eq!(read(&m), (want_w, want_r), "layer {layer}: R={r} G={g}");
        }
        // lengths
        let mut m = model_with(0);
        let (g1, g2) = sums_with(0);
        assert_eq!(m.try_apply_step(&g1[1..], &g2), Err(StepError::Length { layer: 1, expected: N1, got: N1 - 1 }));
        assert_eq!(m.try_apply_step(&g1, &g2[1..]), Err(StepError::Length { layer: 2, expected: N2, got: N2 - 1 }));
        assert!(n > 0);
    }
}

/// `apply_step` panics with the reason instead of reaching the undefined behaviour of the C code.
#[test]
#[should_panic(expected = "model step refused")]
fn apply_step_panics_on_the_forbidden_value() {
    let mut m = Model::from_state(&vec![0i8; N1], &vec![0i8; N2], &vec![0i32; N1], &vec![0i32; N2]).unwrap();
    let mut g1 = vec![0i32; N1];
    g1[7] = i32::MIN;
    m.apply_step(&g1, &vec![0i32; N2]);
}

/// The whole consensus domain, both layers: `W` in {-8, -7, -6, -1, 0, 5, 6, 7}, `R` in `[-T, T]`, `GS` in
/// `[-65536, 65536]` (256 shares of maximal gradient). 7.7e10 entries, several minutes on 16 threads:
/// `cargo test --release -p neb-pow --test step_boundaries -- --ignored`.
#[test]
#[ignore = "long: 7.7e10 entries"]
fn the_step_is_exact_on_the_whole_consensus_domain() {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    for (layer, su) in [(1, SU1), (2, SU2)] {
        let t = 1i32 << su;
        let weights: Vec<i8> = vec![-8, -7, -6, -1, 0, 5, 6, 7];
        let next = std::sync::atomic::AtomicI32::new(-t);
        let total = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    let n_ent = if layer == 1 { N1 } else { N2 };
                    loop {
                        let r = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if r > t {
                            break;
                        }
                        // all (W, G) for this residual, packed in as many models as needed
                        let mut pts: Vec<Point> = Vec::with_capacity(n_ent);
                        let flush = |pts: &mut Vec<Point>| {
                            if pts.is_empty() {
                                return;
                            }
                            if layer == 1 {
                                check(pts, &[])
                            } else {
                                check(&[], pts)
                            };
                            total.fetch_add(pts.len() as u64, std::sync::atomic::Ordering::Relaxed);
                            pts.clear();
                        };
                        for &w in &weights {
                            for g in -65_536..=65_536 {
                                pts.push((w, r, g));
                                if pts.len() == n_ent {
                                    flush(&mut pts);
                                }
                            }
                        }
                        flush(&mut pts);
                    }
                });
            }
        });
        println!("layer {layer}: {} entries", total.load(std::sync::atomic::Ordering::Relaxed));
    }
}
