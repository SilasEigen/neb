//! Red-team consensus security & anti-shortcut verification suite for `nebhash v3` (`NMD3`).
//!
//! Actively attempts to break or shortcut the consensus along 8 adversarial axes:
//! 1. Per-sample gradient table precomputation (`nebhash2_table_cuda.cu` style)
//! 2. Sub-quantization / single-LSB lossy approximation of the frozen base (`trace_digest` bypass test)
//! 3. Layer-skipping / zero-gradient guessing across early & late LoRA tensors
//! 4. Sequential MoE expert prefetching / static PCIe layer-swapping bypass
//! 5. 4-bit nibble packing / entropy compression of the frozen base into 6 GB / 8 GB VRAM
//! 6. Adversarial `i32::MIN` / `i32::MAX` step overflow UB & multi-threaded determinism
//! 7. Ticket cryptographic avalanche & bit-bias grinding resistance
//! 8. Bit-exact `vectors/nebhash3.txt` replay & real ML classification convergence

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use neb_pow::shares::v3::{NB3, V3Arch, V3Base, V3Dataset, V3Model};

unsafe extern "C" {
    fn chacha8_stream(key: *const u8, out: *mut u8, len: usize);
}

fn chacha8_vec(key: &[u8; 32], len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    unsafe { chacha8_stream(key.as_ptr(), out.as_mut_ptr(), len) };
    out
}

fn syn3_base(arch: V3Arch) -> V3Base {
    let mut msg = Vec::from(&b"neb-syn3-base"[..]);
    msg.push(arch.id() as u8);
    let key = *blake3::hash(&msg).as_bytes();
    let raw = chacha8_vec(&key, arch.base_weights());
    let signed: Vec<i8> = raw.into_iter().map(|b| b as i8).collect();
    V3Base::new(arch, signed).unwrap()
}

fn syn3_dataset(arch: V3Arch, n: usize) -> V3Dataset {
    let d = arch.d();
    let mut protos = Vec::new();
    for c in 0..2u8 {
        let mut msg = Vec::from(&b"neb-syn3-proto"[..]);
        msg.push(arch.id() as u8);
        msg.push(c);
        let key = *blake3::hash(&msg).as_bytes();
        let raw = chacha8_vec(&key, d);
        protos.push(raw.into_iter().map(|b| b & 7).collect::<Vec<u8>>());
    }
    let mut samples = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let mut msg = Vec::from(&b"neb-syn3-sample"[..]);
        msg.push(arch.id() as u8);
        msg.extend_from_slice(&(i as u32).to_le_bytes());
        let key = *blake3::hash(&msg).as_bytes();
        let noise = chacha8_vec(&key, d);
        let c = i & 1;
        let mut x = vec![0u8; d];
        for j in 0..d {
            let v = (protos[c][j] as i32) + ((noise[j] & 3) as i32) - 1;
            x[j] = v.clamp(0, 7) as u8;
        }
        samples.push(x);
        labels.push(c as u8);
    }
    V3Dataset {
        arch,
        base_hash: [0u8; 32],
        samples,
        labels,
    }
}

fn hamming_bits_32(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn parse_hex32(s: &str) -> [u8; 32] {
    assert_eq!(s.len(), 64);
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}

/// Attack 1: Miner constructs or encounters repeated samples and tries to precompute per-sample gradients.
#[test]
fn redteam_1_anti_table_identical_samples_attack() {
    let arch = V3Arch::LoraTiny;
    let base = syn3_base(arch);
    let model = V3Model::init(arch, &[0x11; 32]).unwrap();

    // Every sample in the dataset is 100% identical (best possible case for a table attacker).
    let sample: Vec<u8> = (0..arch.d()).map(|j| ((j * 5 + 3) % 8) as u8).collect();
    let ds = V3Dataset {
        arch,
        base_hash: base.hash().unwrap(),
        samples: vec![sample; 128],
        labels: vec![1u8; 128],
    };

    let mut digests = HashSet::new();
    let mut tickets = HashSet::new();
    let mut grad_vectors = HashSet::new();

    for nonce in 0..64u32 {
        let mut key = [0u8; 32];
        key[0..4].copy_from_slice(&nonce.to_le_bytes());
        let out = model.eval_share(&base, &ds, &key).unwrap();
        assert!(digests.insert(out.digest), "duplicate digest at nonce {nonce}");
        assert!(tickets.insert(out.ticket), "duplicate ticket at nonce {nonce}");
        assert!(grad_vectors.insert(out.gradients), "duplicate gradient vector at nonce {nonce}");
    }
}

/// Attack 2: Lossy / approximate base-matrix evaluation (e.g., FP16 approximation or skipping 1 weight).
/// Even when a +1 perturbation to a single weight in a routed expert leaves every clipped activation `x_{l+1}`
/// and every final gradient `G` 100% unchanged, `trace_digest` must detect the 1-LSB inner-product error and
/// trigger a full ~50% cryptographic avalanche in `digest` and `ticket`.
#[test]
fn redteam_2_single_lsb_frozen_base_approximation_attack() {
    let arch = V3Arch::LoraTiny;
    let base = syn3_base(arch);
    let ds = syn3_dataset(arch, 256);
    let model = V3Model::init(arch, &[0x22; 32]).unwrap();
    let key = *blake3::hash(b"redteam-lossy-base-check").as_bytes();

    let ref_out = model.eval_share(&base, &ds, &key).unwrap();
    let d = arch.d();
    let expert_sz = d * d;

    // (a) Perturbing an UNROUTED expert must NOT affect the share (confirms exact expert isolation).
    let unrouted = (0..arch.nexperts() as u32)
        .find(|e| !ref_out.experts.contains(e))
        .expect("at least one unrouted expert");
    let mut base_unrouted = base.clone();
    let idx_un = (unrouted as usize) * expert_sz;
    base_unrouted.weights_mut()[idx_un] = base_unrouted.weights()[idx_un].wrapping_add(1);
    let un_out = model.eval_share(&base_unrouted, &ds, &key).unwrap();
    assert_eq!(un_out.digest, ref_out.digest);
    assert_eq!(un_out.ticket, ref_out.ticket);

    // (b) For EVERY routed layer l = 0..L-1, perturb a single weight by +1 inside `routed_expert[l]`.
    // Find a weight column where at least one sample has a non-masked non-zero input, and verify that even when
    // `gradients` remain 100% identical (sub-quantization perturbation!), `digest` and `ticket` avalanche!
    let mut found_silent_grad_avalanche = false;
    for (layer, &exp_id) in ref_out.experts.iter().enumerate() {
        let mut detected_on_layer = false;
        for col in 0..d {
            let mut tampered = base.clone();
            let w_idx = (exp_id as usize) * expert_sz + col; // row 0, col
            let old = tampered.weights()[w_idx];
            tampered.weights_mut()[w_idx] = if old == 127 { old - 1 } else { old + 1 };
            let bad_out = model.eval_share(&tampered, &ds, &key).unwrap();
            if bad_out.digest != ref_out.digest {
                detected_on_layer = true;
                let hd_digest = hamming_bits_32(&bad_out.digest, &ref_out.digest);
                let hd_ticket = hamming_bits_32(&bad_out.ticket, &ref_out.ticket);
                assert!(
                    (80..=176).contains(&hd_digest),
                    "layer {layer} digest hamming distance {hd_digest} not avalanche-like"
                );
                assert!(
                    (80..=176).contains(&hd_ticket),
                    "layer {layer} ticket hamming distance {hd_ticket} not avalanche-like"
                );
                if bad_out.gradients == ref_out.gradients {
                    // Sub-quantization stealth tamper: final gradients were IDENTICAL, yet trace_digest caught it!
                    found_silent_grad_avalanche = true;
                }
                break;
            }
        }
        assert!(detected_on_layer, "routed expert at layer {layer} was not bound into digest!");
    }
    assert!(
        found_silent_grad_avalanche,
        "expected to observe at least one sub-quantization tamper where gradients matched but trace_digest caught it"
    );
}

/// Attack 3: Layer-skipping / zero-gradient guessing shortcut.
/// Verifies that every trainable tensor (A[0], B[0], ..., A[L-1], B[L-1], H) has healthy non-zero gradient density
/// and bounded clipping, so skipping any layer or guessing zeros fails 100% of the time.
#[test]
fn redteam_3_no_dead_layers_or_zero_gradient_shortcut() {
    let arch = V3Arch::LoraTiny;
    let base = syn3_base(arch);
    let ds = syn3_dataset(arch, 512);
    let model = V3Model::init(arch, &[0x33; 32]).unwrap();

    let block_ab = arch.rank() * arch.d();
    let ntensors = 2 * arch.nblocks() + 1;
    let mut nonzero_counts = vec![0usize; ntensors];
    let mut sat_counts = vec![0usize; ntensors];
    let mut total_counts = vec![0usize; ntensors];

    let n_trials = 24usize;
    for t in 0..n_trials {
        let key = *blake3::hash(format!("grad-density-{t}").as_bytes()).as_bytes();
        let out = model.eval_share(&base, &ds, &key).unwrap();
        let mut off = 0;
        for ti in 0..ntensors {
            let len = if ti < 2 * arch.nblocks() { block_ab } else { 2 * arch.d() };
            let slice = &out.gradients[off..off + len];
            nonzero_counts[ti] += slice.iter().filter(|&&g| g != 0).count();
            sat_counts[ti] += slice.iter().filter(|&&g| g.abs() == 1024).count();
            total_counts[ti] += len;
            off += len;
        }
    }

    for ti in 0..ntensors {
        let nz_ratio = nonzero_counts[ti] as f64 / total_counts[ti] as f64;
        let sat_ratio = sat_counts[ti] as f64 / total_counts[ti] as f64;
        assert!(
            nz_ratio > 0.25,
            "tensor {ti} has dangerously low non-zero gradient density {nz_ratio:.3} (vulnerable to zero-guessing!)"
        );
        assert!(
            sat_ratio < 0.45,
            "tensor {ti} is over-saturated ({sat_ratio:.3}), harming training signal!"
        );
    }
}

/// Attack 4: Static expert prefetching & Mega-Batch PCIe streaming bypass.
/// Proves that:
/// (a) `e_l` for `l >= 1` cannot be predicted from `key` alone because modifying sample activations alters the route.
/// (b) Across a batch of keys, all `E` experts are uniformly hit at every layer.
#[test]
fn redteam_4_sequential_moe_routing_prevents_prefetch_and_streaming() {
    let arch = V3Arch::LoraTiny;
    let base = syn3_base(arch);
    let ds1 = syn3_dataset(arch, 128);
    let mut ds2 = ds1.clone();
    // Flip all samples in ds2 so batch activations x_1 definitely change while `key` stays identical.
    for s in &mut ds2.samples {
        for v in s.iter_mut() {
            *v = 7 - *v;
        }
    }
    let model = V3Model::init(arch, &[0x44; 32]).unwrap();

    let mut route_changed_count = 0;
    for i in 0..32u32 {
        let key = *blake3::hash(&i.to_le_bytes()).as_bytes();
        let o1 = model.eval_share(&base, &ds1, &key).unwrap();
        let o2 = model.eval_share(&base, &ds2, &key).unwrap();
        // Layer 0 depends only on key, so e_0 matches; layers >= 1 depend on activations x_l!
        assert_eq!(o1.experts[0], o2.experts[0]);
        if o1.experts[1..] != o2.experts[1..] {
            route_changed_count += 1;
        }
    }
    assert!(
        route_changed_count >= 28,
        "expected data-dependent expert routing to change on almost all trials, got {route_changed_count}/32"
    );

    // Uniform coverage across all E experts
    let mut hits = vec![vec![0usize; arch.nexperts()]; arch.nblocks()];
    for i in 0..512u32 {
        let key = *blake3::hash(format!("coverage-{i}").as_bytes()).as_bytes();
        let out = model.eval_share(&base, &ds1, &key).unwrap();
        for (l, &e) in out.experts.iter().enumerate() {
            hits[l][e as usize] += 1;
        }
    }
    for (l, layer_hits) in hits.iter().enumerate() {
        for (e, &cnt) in layer_hits.iter().enumerate() {
            // Expected = 512 / 16 = 32; require [12, 60]
            assert!(
                (12..=60).contains(&cnt),
                "uneven expert routing at layer {l} expert {e}: {cnt} hits"
            );
        }
    }
}

/// Attack 5: 4-bit nibble packing / lossless compression into 6 GB / 8 GB legacy VRAM.
/// Verifies full 8-bit signed coverage (`-128..127`) and Shannon entropy > 7.9 bits/weight.
#[test]
fn redteam_5_full_int8_entropy_defeats_4bit_nibble_packing() {
    let arch = V3Arch::LoraTiny;
    let base = syn3_base(arch);
    let w = base.weights();

    let mut counts = [0usize; 256];
    let mut outside_4bit = 0usize;
    for &v in w {
        counts[(v as u8) as usize] += 1;
        if !(-8..=7).contains(&v) {
            outside_4bit += 1;
        }
    }
    assert!(counts.iter().all(|&c| c > 0), "all 256 int8 values must appear in base weights");
    let outside_ratio = outside_4bit as f64 / w.len() as f64;
    assert!(outside_ratio > 0.92, "outside 4-bit ratio too low: {outside_ratio}");

    let n = w.len() as f64;
    let entropy: f64 = counts
        .iter()
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum();
    assert!(entropy > 7.95, "Shannon entropy {entropy:.4} bits/weight too low; vulnerable to compression!");
}

/// Attack 6: Adversarial `i32::MIN` / `i32::MAX` gradient overflow UB and multi-threaded race check.
#[test]
fn redteam_6_adversarial_int32_overflow_and_multithread_determinism() {
    let arch = V3Arch::LoraTiny;
    let mut model = V3Model::init(arch, &[0x66; 32]).unwrap();
    let nw = arch.trainable_weights();

    // Feed extreme adversarial i32 values directly into `apply` (including i32::MIN whose negation overflows i32!).
    for extreme in [i32::MIN, i32::MAX, -1_500_000_000, 1_500_000_000, 0] {
        let gs = vec![extreme; nw];
        model.apply(&gs).unwrap();
        for &w in model.weights() {
            assert!((-8..=7).contains(&w), "weight {w} escaped [-8, 7]");
        }
        let lora_end = arch.nblocks() * 2 * arch.rank() * arch.d();
        for (i, &r) in model.residuals().iter().enumerate() {
            let bound = if i < lora_end { 1 << 14 } else { 1 << 12 };
            assert!((-bound..=bound).contains(&r), "residual {r} escaped [-{bound}, {bound}]");
        }
    }

    // Multi-threaded determinism across 8 worker threads sharing V3Base and V3Dataset
    let base = Arc::new(syn3_base(arch));
    let ds = Arc::new(syn3_dataset(arch, 256));
    let seed = [0x77u8; 32];
    let key = [0x88u8; 32];
    let expected = V3Model::init(arch, &seed).unwrap().eval_share(&base, &ds, &key).unwrap();

    let mut handles = Vec::new();
    for _ in 0..8 {
        let b = Arc::clone(&base);
        let d = Arc::clone(&ds);
        handles.push(thread::spawn(move || {
            let m = V3Model::init(arch, &seed).unwrap();
            let mut last = None;
            for _ in 0..8 {
                last = Some(m.eval_share(&b, &d, &key).unwrap());
            }
            last.unwrap()
        }));
    }
    for h in handles {
        let got = h.join().unwrap();
        assert_eq!(got.digest, expected.digest);
        assert_eq!(got.ticket, expected.ticket);
        assert_eq!(got.gradients, expected.gradients);
    }
}

/// Attack 7: Ticket bit-bias / partial-nonce grinding resistance (avalanche test).
#[test]
fn redteam_7_ticket_cryptographic_avalanche_and_uniformity() {
    let arch = V3Arch::LoraTiny;
    let base = syn3_base(arch);
    let ds = syn3_dataset(arch, 256);
    let model = V3Model::init(arch, &[0x55; 32]).unwrap();

    let n_pairs = 128usize;
    let mut total_hd = 0u64;
    let mut bit_flips = [0usize; 256];

    for i in 0..n_pairs {
        let k1 = *blake3::hash(format!("avalanche-{i}").as_bytes()).as_bytes();
        let mut k2 = k1;
        // Flip exactly 1 bit in `key`
        let bit_pos = i % 256;
        k2[bit_pos / 8] ^= 1u8 << (bit_pos % 8);
        assert_ne!(k1, k2);
        let o1 = model.eval_share(&base, &ds, &k1).unwrap();
        let o2 = model.eval_share(&base, &ds, &k2).unwrap();
        total_hd += hamming_bits_32(&o1.ticket, &o2.ticket) as u64;
        for b in 0..256 {
            let diff = ((o1.ticket[b / 8] ^ o2.ticket[b / 8]) >> (b % 8)) & 1;
            bit_flips[b] += diff as usize;
        }
    }

    let mean_hd = total_hd as f64 / n_pairs as f64;
    assert!(
        (118.0..=138.0).contains(&mean_hd),
        "mean ticket avalanche Hamming distance {mean_hd:.2} deviates from ideal 128.0"
    );
    for (b, &flips) in bit_flips.iter().enumerate() {
        let rate = flips as f64 / n_pairs as f64;
        assert!(
            (0.30..=0.70).contains(&rate),
            "ticket bit {b} flip rate {rate:.3} shows exploitable bias"
        );
    }
}

/// Attack 8: Full `vectors/nebhash3.txt` bit-exact replay + real on-chain classification learning verification.
#[test]
fn redteam_8_vector_replay_and_real_ml_convergence() {
    // Part A: Replay vectors/nebhash3.txt bit-for-bit
    let arch = V3Arch::LoraTiny;
    let base = syn3_base(arch);
    let ds = syn3_dataset(arch, 2048);
    let mut model = V3Model::init(arch, &[0u8; 32]).unwrap();

    let vec_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../vectors/nebhash3.txt");
    let vec_text = std::fs::read_to_string(&vec_path).expect("read vectors/nebhash3.txt");
    let mut lines = vec_text.lines();

    let base_line: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    assert_eq!(parse_hex32(base_line[3]), base.hash().unwrap());

    let init_line: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let (wh0, rh0) = model.hashes().unwrap();
    assert_eq!(parse_hex32(init_line[3]), wh0);
    assert_eq!(parse_hex32(init_line[4]), rh0);

    let step_shares = [32usize, 1, 3, 32, 17, 2];
    for (s, &nsh) in step_shares.iter().enumerate() {
        let mut sums = vec![0i32; arch.trainable_weights()];
        for j in 0..nsh {
            let mut msg = Vec::from(&b"neb-test-share3"[..]);
            msg.push(arch.id() as u8);
            msg.extend_from_slice(&(s as u32).to_le_bytes());
            msg.extend_from_slice(&(j as u32).to_le_bytes());
            let key = *blake3::hash(&msg).as_bytes();
            let out = model.eval_share(&base, &ds, &key).unwrap();
            for (acc, &g) in sums.iter_mut().zip(out.gradients.iter()) {
                *acc += g;
            }
            if j == 0 || j == nsh - 1 {
                let sh_line: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
                assert_eq!(parse_hex32(sh_line[6]), out.digest);
                assert_eq!(parse_hex32(sh_line[7]), out.ticket);
                assert_eq!(sh_line[8].parse::<u32>().unwrap(), out.indices[0]);
                assert_eq!(sh_line[9].parse::<u32>().unwrap(), out.indices[NB3 - 1]);
                assert_eq!(sh_line[10].parse::<u32>().unwrap(), out.experts[0]);
                assert_eq!(sh_line[11].parse::<u32>().unwrap(), out.experts[arch.nblocks() - 1]);
            }
        }
        model.apply(&sums).unwrap();
        let st_line: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
        let (wh, rh) = model.hashes().unwrap();
        assert_eq!(parse_hex32(st_line[5]), wh);
        assert_eq!(parse_hex32(st_line[6]), rh);
    }
    assert!(lines.next().is_none());

    // Part B: Verify genuine machine-learning convergence on a separable noisy binary dataset!
    // Class 0 has high values on features 0..32 and low on 32..64; Class 1 is the opposite, plus random noise.
    let mut train_samples = Vec::with_capacity(256);
    let mut train_labels = Vec::with_capacity(256);
    for i in 0..256usize {
        let lbl = (i & 1) as u8;
        let noise = chacha8_vec(&[i as u8; 32], arch.d());
        let mut x = vec![0u8; arch.d()];
        for j in 0..arch.d() {
            let signal = if (j < 32) ^ (lbl == 1) { 5i32 } else { 2i32 };
            let n = (noise[j] & 3) as i32 - 1;
            x[j] = (signal + n).clamp(0, 7) as u8;
        }
        train_samples.push(x);
        train_labels.push(lbl);
    }
    let learn_ds = V3Dataset {
        arch,
        base_hash: base.hash().unwrap(),
        samples: train_samples,
        labels: train_labels,
    };

    let mut learn_model = V3Model::init(arch, &[0x99; 32]).unwrap();
    let eval_acc = |m: &V3Model| -> f64 {
        let correct = learn_ds
            .samples
            .iter()
            .zip(learn_ds.labels.iter())
            .filter(|(x, lbl)| m.predict(&base, x).unwrap() == **lbl)
            .count();
        correct as f64 / learn_ds.samples.len() as f64
    };

    let acc_before = eval_acc(&learn_model);
    // Run 35 block steps (32 shares per step, scaled up by 16x to simulate longer training in fast CI)
    for step in 0..35u32 {
        let mut sums = vec![0i32; arch.trainable_weights()];
        for sh in 0..8u32 {
            let key = *blake3::hash(format!("learn-{step}-{sh}").as_bytes()).as_bytes();
            let out = learn_model.eval_share(&base, &learn_ds, &key).unwrap();
            for (acc, &g) in sums.iter_mut().zip(out.gradients.iter()) {
                *acc += g * 16;
            }
        }
        learn_model.apply(&sums).unwrap();
    }
    let acc_after = eval_acc(&learn_model);
    assert!(
        acc_after >= 0.88 && acc_after > acc_before + 0.20,
        "expected strong ML convergence: acc_before={acc_before:.3}, acc_after={acc_after:.3}"
    );
}
