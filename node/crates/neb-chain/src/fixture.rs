//! Chain fixtures: recorded chains that a test, a tool or an independent implementation replays to check
//! the proof of work, the share lists, the model step and the state hashes block by block (spec/shares.md
//! section 11). Test support like [`crate::demo`]; no node code depends on it.
//!
//! A fixture holds the seed of the models (the id of block 0) and the number of model slots K, then for each
//! block its 80-byte header, the share list it commits to (40-byte share records, spec/shares.md section 11.2)
//! and the hashes of the model its shares update: the model of slot `(h - 1) mod K` in the state of block `h`
//! (with K = 1, the model of the block). The body of a block is not recorded: the generators below use a stand-in body, one coinbase
//! `"sim-coinbase" || LE32(height)` (followed by `LE32(pool)` for a pool above 0), whose root is
//! [`stand_in_body_root`]. The proof of work and the model step do not depend on the body beyond the
//! `body_root` field of the header.
//!
//! Text format (`pow/tests/chains/*.txt`, LF endings):
//!
//! ```text
//! neb-chain-fixture 2
//! name NAME
//! dataset synthetic | dataset file BLAKE3_HEX
//! [slots K]                       (only when K > 1)
//! seed HEX32
//! block H pool P header HEX160 list R1,R2,... | - wh HEX64 rh HEX64 [loss c1 k1 a1 b1 c2 k2 a2 b2]
//! ```
//!
//! `P` is the pool whose stand-in coinbase the block has, and each `R` is a share record in 80 hex characters
//! (`body_root || LE32(time) || LE32(nonce)`).
//!
//! `loss` gives, for layer 1 then layer 2, the entries whose residual was cut by the bound, the entries whose
//! weight was clipped, the sum of the absolute cut residuals and the sum of the absolute cut weight charge
//! (`2^SU` per unit): the losses of the step. A block without a step
//! (height 1, or an empty list) has none.

use neb_block::Header;
use neb_pow::shares::{GradSum, Model, SU1, SU2};

use crate::block::{
    Block, MAX_SHARES_PER_BLOCK, SHARE_RECORD_BYTES, ShareRecord, body_root_of, check_shares_list, shares_root,
};
use crate::model::{Models, initial_models, share_key, share_target, slot, verify_block};
use crate::state::ChainParams;

/// What the bounds of the step discarded in one layer during one step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerLoss {
    /// Entries whose residual was cut at `+-2^SU`.
    pub cut_events: u64,
    /// Entries whose weight was clipped to `-8..=7`.
    pub clip_events: u64,
    /// Sum of `|lr|`, the residual charge cut.
    pub lr_abs: u64,
    /// Sum of `2^SU * |wc|`, the weight charge cut.
    pub wc_abs: u64,
}

pub type StepLoss = [LayerLoss; 2];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixtureBlock {
    pub height: u32,
    /// The pool whose stand-in coinbase the block has (0 for a single miner).
    pub pool: u32,
    pub header: Header,
    pub list: Vec<ShareRecord>,
    pub weights_hash: [u8; 32],
    pub residual_hash: [u8; 32],
    pub loss: Option<StepLoss>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainFixture {
    pub name: String,
    /// `"synthetic"` or `"file BLAKE3_HEX"`.
    pub dataset: String,
    /// Model slots of the chain (spec/shares.md section 11.2).
    pub slots: u32,
    pub seed: [u8; 32],
    pub blocks: Vec<FixtureBlock>,
}

/// Summary of a successful replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayReport {
    pub blocks: usize,
    pub steps: usize,
    pub shares: usize,
    pub largest_step: usize,
    /// Per block with a step: height and losses, as recomputed.
    pub losses: Vec<(u32, StepLoss)>,
    pub final_weights_hash: [u8; 32],
    pub final_residual_hash: [u8; 32],
}

/// `H("body", LE32(1), leaf, shares_root)` with one stand-in coinbase (see the module documentation).
pub fn stand_in_body_root(height: u32, pool: u32, list: &[ShareRecord]) -> [u8; 32] {
    let mut cb = b"sim-coinbase".to_vec();
    cb.extend_from_slice(&height.to_le_bytes());
    if pool > 0 {
        cb.extend_from_slice(&pool.to_le_bytes());
    }
    let tx = *blake3::hash(&cb).as_bytes();
    let mut leaf_in = b"leaf".to_vec();
    leaf_in.extend_from_slice(&tx);
    let leaf = *blake3::hash(&leaf_in).as_bytes();
    body_root_of(1, &leaf, &shares_root(list))
}

fn to_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn from_hex<const K: usize>(s: &str) -> Result<[u8; K], String> {
    if s.len() != 2 * K || !s.is_ascii() {
        return Err(format!("expected {} hex characters, got {:?}", 2 * K, s));
    }
    let mut out = [0u8; K];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| format!("not hex: {s:?}"))?;
    }
    Ok(out)
}

impl ChainFixture {
    pub fn to_text(&self) -> String {
        let mut out = format!("neb-chain-fixture 2\nname {}\ndataset {}\n", self.name, self.dataset);
        if self.slots > 1 {
            out += &format!("slots {}\n", self.slots);
        }
        out += &format!("seed {}\n", to_hex(&self.seed));
        for b in &self.blocks {
            let list = if b.list.is_empty() {
                "-".to_string()
            } else {
                b.list.iter().map(|r| to_hex(&r.to_bytes())).collect::<Vec<_>>().join(",")
            };
            out += &format!(
                "block {} pool {} header {} list {} wh {} rh {}",
                b.height,
                b.pool,
                to_hex(&b.header.encode()),
                list,
                to_hex(&b.weights_hash),
                to_hex(&b.residual_hash)
            );
            if let Some(l) = &b.loss {
                out += &format!(
                    " loss {}",
                    l.iter()
                        .flat_map(|x| [x.cut_events, x.clip_events, x.lr_abs, x.wc_abs])
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            out.push('\n');
        }
        out
    }

    pub fn parse(text: &str) -> Result<ChainFixture, String> {
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        if lines.next() != Some("neb-chain-fixture 2") {
            return Err("not a neb-chain-fixture 2 file".into());
        }
        let field = |line: Option<&str>, key: &str| -> Result<String, String> {
            line.and_then(|l| l.strip_prefix(&format!("{key} ")))
                .map(str::to_string)
                .ok_or(format!("missing {key} line"))
        };
        let name = field(lines.next(), "name")?;
        let dataset = field(lines.next(), "dataset")?;
        let mut next = lines.next();
        let slots = match next.and_then(|l| l.strip_prefix("slots ")) {
            Some(k) => {
                next = lines.next();
                k.parse::<u32>().ok().filter(|k| *k >= 1).ok_or(format!("bad slots line {k:?}"))?
            }
            None => 1,
        };
        let seed = from_hex::<32>(&field(next, "seed")?)?;
        let mut blocks = Vec::new();
        for line in lines {
            let f: Vec<&str> = line.split(' ').collect();
            let num = |s: &str| s.parse::<u64>().map_err(|_| format!("bad number {s:?} in {line:?}"));
            if f.len() < 12
                || f[0] != "block"
                || f[2] != "pool"
                || f[4] != "header"
                || f[6] != "list"
                || f[8] != "wh"
                || f[10] != "rh"
            {
                return Err(format!("malformed block line {line:?}"));
            }
            let list: Vec<ShareRecord> = if f[7] == "-" {
                Vec::new()
            } else {
                f[7].split(',')
                    .map(|s| from_hex::<SHARE_RECORD_BYTES>(s).map(|b| ShareRecord::from_bytes(&b)))
                    .collect::<Result<_, _>>()?
            };
            let loss = match f.len() {
                12 => None,
                21 if f[12] == "loss" => {
                    let v: Vec<u64> = f[13..].iter().map(|s| num(s)).collect::<Result<_, _>>()?;
                    let l = |i: usize| LayerLoss {
                        cut_events: v[i],
                        clip_events: v[i + 1],
                        lr_abs: v[i + 2],
                        wc_abs: v[i + 3],
                    };
                    Some([l(0), l(4)])
                }
                _ => return Err(format!("malformed block line {line:?}")),
            };
            blocks.push(FixtureBlock {
                height: num(f[1])? as u32,
                pool: num(f[3])? as u32,
                header: Header::decode(&from_hex::<80>(f[5])?),
                list,
                weights_hash: from_hex(f[9])?,
                residual_hash: from_hex(f[11])?,
                loss,
            });
        }
        Ok(ChainFixture { name, dataset, slots, seed, blocks })
    }
}

/// The losses of one step, recomputed entry by entry in 64-bit integers from the model before the step and
/// the sums of gradients, which must give exactly the model `after` that the C step produced.
pub fn account_step(before: &Model, gs1: &[i32], gs2: &[i32], after: &Model) -> Result<StepLoss, String> {
    let mut out = [LayerLoss::default(); 2];
    for (k, (su, w, r, g, wn, rn)) in [
        (SU1, before.w1(), before.r1(), gs1, after.w1(), after.r1()),
        (SU2, before.w2(), before.r2(), gs2, after.w2(), after.r2()),
    ]
    .into_iter()
    .enumerate()
    {
        let t = 1i64 << su;
        for i in 0..w.len() {
            let s = r[i] as i64 + g[i] as i64;
            let d = (s / t).clamp(-1, 1);
            let r2 = s - d * t;
            let r3 = r2.clamp(-t, t);
            let w2 = w[i] as i64 - d;
            let w3 = w2.clamp(-8, 7);
            if (w3, r3) != (wn[i] as i64, rn[i] as i64) {
                return Err(format!(
                    "layer {}: entry {i}: the C step gives W={} R={}, the 64-bit step gives W={w3} R={r3}",
                    k + 1,
                    wn[i],
                    rn[i]
                ));
            }
            let (lr, wc) = (r2 - r3, w3 - w2);
            out[k].cut_events += (lr != 0) as u64;
            out[k].clip_events += (wc != 0) as u64;
            out[k].lr_abs += lr.unsigned_abs();
            out[k].wc_abs += (t * wc).unsigned_abs();
        }
    }
    Ok(out)
}

/// Replays `fx` with the node's own functions: the proof of work of every block and the validity of every
/// share (`verify_block`), then the model step; checks the headers, the stand-in bodies, the state hashes and
/// (when recorded) the losses, and recomputes the step in 64-bit integers.
pub fn replay(fx: &ChainFixture, params: &ChainParams) -> Result<ReplayReport, String> {
    let k = params.slots();
    if fx.slots as usize != k {
        return Err(format!("the fixture has {} slots, the parameters {k}", fx.slots));
    }
    let mut models: Vec<Models> = vec![initial_models(&fx.seed, k)];
    let mut prev_id = fx.seed;
    let mut prev_header: Option<Header> = None;
    let mut report = ReplayReport {
        blocks: 0,
        steps: 0,
        shares: 0,
        largest_step: 0,
        losses: Vec::new(),
        final_weights_hash: [0; 32],
        final_residual_hash: [0; 32],
    };
    for (i, fb) in fx.blocks.iter().enumerate() {
        let h = i as u32 + 1;
        let tag = format!("block {h}");
        if fb.height != h {
            return Err(format!("{tag}: the fixture says height {}", fb.height));
        }
        if fb.header.parent != prev_id {
            return Err(format!("{tag}: the parent is not the previous block"));
        }
        if fb.header.body_root != stand_in_body_root(h, fb.pool, &fb.list) {
            return Err(format!("{tag}: body_root is not the stand-in root of the list"));
        }
        check_shares_list(&fb.list).map_err(|e| format!("{tag}: {e}"))?;
        let models_p = models[i].clone();
        let models_pp = if h >= 2 { models[i - 1].clone() } else { models_p.clone() };
        let t = slot(h - 1, k);
        let parent_header = prev_header.unwrap_or(Header {
            version: 0,
            parent: [0; 32],
            body_root: [0; 32],
            time: 0,
            bits: 0,
            nonce: 0,
        });
        let block = Block { header: fb.header, txs: Vec::new(), shares: fb.list.clone() };
        let new = verify_block(params, true, &block, &parent_header, h, &models_p, &models_pp)
            .map_err(|e| format!("{tag}: {e}"))?;
        if new[t].hashes() != (fb.weights_hash, fb.residual_hash) {
            return Err(format!("{tag}: the state hashes differ from the fixture"));
        }
        if !fb.list.is_empty() {
            let mut sum = GradSum::new();
            for r in &fb.list {
                sum.add(
                    &models_pp[t],
                    &**params.dataset_at(h - 1),
                    &share_key(r, parent_header.parent, parent_header.bits),
                );
            }
            let (gs1, gs2) = sum.sums();
            let loss = account_step(&models_p[t], gs1, gs2, &new[t]).map_err(|e| format!("{tag}: {e}"))?;
            if let Some(want) = fb.loss
                && want != loss
            {
                return Err(format!(
                    "{tag}: the losses differ from the fixture: recorded {want:?}, recomputed {loss:?}"
                ));
            }
            report.steps += 1;
            report.shares += fb.list.len();
            report.largest_step = report.largest_step.max(fb.list.len());
            report.losses.push((h, loss));
        } else if fb.loss.is_some() {
            return Err(format!("{tag}: losses recorded for a block without shares"));
        }
        prev_id = fb.header.id();
        prev_header = Some(fb.header);
        (report.final_weights_hash, report.final_residual_hash) = new[t].hashes();
        models.push(new);
        report.blocks += 1;
    }
    if fx.blocks.is_empty() {
        (report.final_weights_hash, report.final_residual_hash) = models[0][0].hashes();
    }
    Ok(report)
}

/// A deterministic chain whose steps have prescribed sizes, for the tests that need steps around the
/// thresholds of the residual bound (16 and 128 shares per step): block `h >= 2` lists `sizes[h - 2]`
/// share records with random `body_root`, `time` and nonce (block 1 lists none). `bits` must give a share
/// target of `2^256 - 1` (so that every record is a valid share) and a block target that a nonce scan finds
/// quickly. Mines each block with the C reference on `params.dataset`.
pub fn generate_stress(
    name: &str,
    dataset_tag: &str,
    label: &str,
    params: &ChainParams,
    bits: u32,
    sizes: &[usize],
    rng_seed: u64,
) -> Result<ChainFixture, String> {
    if share_target(bits).map_err(|e| e.to_string())?.0 != [0xFF; 32] {
        return Err("the stress chain needs a saturated share target".into());
    }
    if sizes.iter().any(|&n| n == 0 || n > MAX_SHARES_PER_BLOCK) {
        return Err("a step has 1 to 256 shares".into());
    }
    let seed = *blake3::hash(label.as_bytes()).as_bytes();
    let k = params.slots();
    let mut fx =
        ChainFixture { name: name.into(), dataset: dataset_tag.into(), slots: k as u32, seed, blocks: Vec::new() };
    let mut state = rng_seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let target = neb_block::Target::from_bits(bits).map_err(|e| e.to_string())?;
    let mut models: Vec<Models> = vec![initial_models(&seed, k)];
    let mut prev = Header { version: 0, parent: [0; 32], body_root: [0; 32], time: 0, bits: 0, nonce: 0 };
    let mut prev_id = seed;
    let mut buffers = neb_pow::shares::ShareBuffers::new();
    for h in 1..=sizes.len() as u32 + 1 {
        let list: Vec<ShareRecord> = if h == 1 {
            Vec::new()
        } else {
            let want = sizes[h as usize - 2];
            let mut set = std::collections::BTreeSet::new();
            while set.len() < want {
                let mut body_root = [0u8; 32];
                for chunk in body_root.chunks_mut(8) {
                    chunk.copy_from_slice(&next().to_le_bytes());
                }
                set.insert(ShareRecord { body_root, time: next() as u32, nonce: next() as u32 });
            }
            set.into_iter().collect()
        };
        let mut header = Header {
            version: 1,
            parent: prev_id,
            body_root: stand_in_body_root(h, 0, &list),
            time: 1_793_318_400 + 60 * h,
            bits,
            nonce: 0,
        };
        let models_p = models[h as usize - 1].clone();
        let models_pp = if h >= 2 { models[h as usize - 2].clone() } else { models_p.clone() };
        let (s, t) = (slot(h, k), slot(h - 1, k));
        for nonce in 0u32.. {
            header.nonce = nonce;
            if target.admits(&buffers.ticket(&models_p[s], &**params.dataset_at(h), &header.id())) {
                break;
            }
        }
        let block = Block { header, txs: Vec::new(), shares: list.clone() };
        let new = verify_block(params, true, &block, &prev, h, &models_p, &models_pp)
            .map_err(|e| format!("block {h}: {e}"))?;
        let loss = if list.is_empty() {
            None
        } else {
            let mut sum = GradSum::new();
            for r in &list {
                sum.add(&models_pp[t], &**params.dataset_at(h - 1), &share_key(r, prev.parent, prev.bits));
            }
            let (gs1, gs2) = sum.sums();
            Some(account_step(&models_p[t], gs1, gs2, &new[t])?)
        };
        let (weights_hash, residual_hash) = new[t].hashes();
        fx.blocks.push(FixtureBlock { height: h, pool: 0, header, list, weights_hash, residual_hash, loss });
        prev_id = header.id();
        prev = header;
        models.push(new);
    }
    Ok(fx)
}

/// A chain mined by `pools` miners in lockstep (spec/shares.md section 11.2). At each height every pool has
/// its own stand-in coinbase, hence its own `body_root`, and its own header time; they scan the nonces
/// 0, 1, 2, ... in turn, pool 0 first, until one attempt meets the block target. That header is the block.
/// Every attempt of every pool up to that one that meets the share target is a share of the round, so the
/// next block lists records of several headers (the winner and the pools that lost the race). `bits` is fixed
/// for the whole chain. Mines on `params.dataset` with the C reference; a block lists at most 256 records
/// (the lowest in byte order when the round has more).
pub fn generate_pools(
    name: &str,
    dataset_tag: &str,
    label: &str,
    params: &ChainParams,
    bits: u32,
    blocks: u32,
    pools: u32,
) -> Result<ChainFixture, String> {
    let seed = *blake3::hash(label.as_bytes()).as_bytes();
    let k = params.slots();
    let mut fx =
        ChainFixture { name: name.into(), dataset: dataset_tag.into(), slots: k as u32, seed, blocks: Vec::new() };
    let target = neb_block::Target::from_bits(bits).map_err(|e| e.to_string())?;
    let round_target = share_target(bits).map_err(|e| e.to_string())?;
    let mut models: Vec<Models> = vec![initial_models(&seed, k)];
    let mut prev = Header { version: 0, parent: [0; 32], body_root: [0; 32], time: 0, bits: 0, nonce: 0 };
    let mut prev_id = seed;
    let mut buffers = neb_pow::shares::ShareBuffers::new();
    let mut round: std::collections::BTreeSet<ShareRecord> = std::collections::BTreeSet::new();
    for h in 1..=blocks {
        let list: Vec<ShareRecord> = round.iter().take(MAX_SHARES_PER_BLOCK).copied().collect();
        let models_p = models[h as usize - 1].clone();
        let models_pp = if h >= 2 { models[h as usize - 2].clone() } else { models_p.clone() };
        let (s, t) = (slot(h, k), slot(h - 1, k));
        let time = 1_793_318_400 + 60 * h;
        let bodies: Vec<[u8; 32]> = (0..pools).map(|j| stand_in_body_root(h, j, &list)).collect();
        let mut next_round = std::collections::BTreeSet::new();
        let mut won = None;
        'scan: for nonce in 0u32.. {
            for j in 0..pools {
                let header =
                    Header { version: 1, parent: prev_id, body_root: bodies[j as usize], time: time + j, bits, nonce };
                let ticket = buffers.ticket(&models_p[s], &**params.dataset_at(h), &header.id());
                if round_target.admits(&ticket) {
                    next_round.insert(ShareRecord::of_header(&header));
                }
                if target.admits(&ticket) {
                    won = Some((j, header));
                    break 'scan;
                }
            }
        }
        let (pool, header) = won.expect("the scan runs until a solution");
        let block = Block { header, txs: Vec::new(), shares: list.clone() };
        let new = verify_block(params, true, &block, &prev, h, &models_p, &models_pp)
            .map_err(|e| format!("block {h}: {e}"))?;
        let loss = if list.is_empty() {
            None
        } else {
            let mut sum = GradSum::new();
            for r in &list {
                sum.add(&models_pp[t], &**params.dataset_at(h - 1), &share_key(r, prev.parent, prev.bits));
            }
            let (gs1, gs2) = sum.sums();
            Some(account_step(&models_p[t], gs1, gs2, &new[t])?)
        };
        let (weights_hash, residual_hash) = new[t].hashes();
        fx.blocks.push(FixtureBlock { height: h, pool, header, list, weights_hash, residual_hash, loss });
        prev_id = header.id();
        prev = header;
        models.push(new);
        round = next_round;
    }
    Ok(fx)
}
