//! Proof of work and model step of a block (spec/shares.md section 11).
//!
//! A chain trains K models, one per dataset of the network (K = 1 on a network of one dataset). The state of a block
//! is the list of the K models; a block of height `h` works on the model of slot `h mod K`. The ticket of a block is
//! an attempt on that model of its parent's state. The shares it lists belong to the round of its parent (height
//! `h - 1`, slot `(h - 1) mod K`) and are computed on that slot's model in the grandparent's state; their summed
//! gradients update that slot's model in the parent's state, so a share trains the model it was mined on. With K = 1
//! this is the rule of protocol 2. Everything here is integer arithmetic done by `neb_pow` (the C reference), and
//! every node must obtain the same models.

use std::sync::Arc;

use neb_block::{Header, Target, TargetError};
use neb_pow::shares::{Arch, Dataset, GradSum, Model, ShareBuffers};

use crate::block::{Block, ShareRecord, check_shares_list};
use crate::state::{ChainParams, Upgrade};

/// The state of a block: one model per slot (spec/shares.md section 11.2). A model that a block does not update is
/// shared with the parent's state.
pub type Models = Arc<[Arc<Model>]>;

/// The slot a block of height `height` works on: `height mod k`.
pub fn slot(height: u32, k: usize) -> usize {
    height as usize % k
}

/// The hash of a state (spec/shares.md section 7): with one slot the weights hash of its model, as in protocol 2;
/// with more, `BLAKE3("neb-models" || weights_hash_0 || ... || weights_hash_{K-1})`.
pub fn models_hash(models: &Models) -> [u8; 32] {
    if models.len() == 1 {
        return models[0].weights_hash();
    }
    let mut h = blake3::Hasher::new();
    h.update(b"neb-models");
    for m in models.iter() {
        h.update(&m.weights_hash());
    }
    *h.finalize().as_bytes()
}

/// The state of the genesis block for `k` slots (spec/shares.md section 11.2): with one slot the model of the seed
/// equal to the genesis id, as in protocol 2; with more, slot `i` has the seed
/// `BLAKE3("neb-model-slot" || genesis id || LE32(i))`.
pub fn initial_models(chain_id: &[u8; 32], k: usize) -> Models {
    if k == 1 {
        return Arc::from(vec![Arc::new(Model::from_seed(chain_id))]);
    }
    (0..k as u32)
        .map(|i| {
            let mut h = blake3::Hasher::new();
            h.update(b"neb-model-slot");
            h.update(chain_id);
            h.update(&i.to_le_bytes());
            Arc::new(Model::from_seed(h.finalize().as_bytes()))
        })
        .collect()
}

/// The initial model of an upgrade (spec/shares.md section 11.7): the model of the architecture of the upgrade's
/// dataset for the seed `BLAKE3("neb-model-upgrade" || LE32(height) || LE32(slot))`.
pub fn upgrade_model(u: &Upgrade, arch: Arch) -> Model {
    let mut h = blake3::Hasher::new();
    h.update(b"neb-model-upgrade");
    h.update(&u.height.to_le_bytes());
    h.update(&(u.slot as u32).to_le_bytes());
    Model::from_seed_arch(arch, h.finalize().as_bytes())
}

/// The share target is the block target times 2^5, so a round yields about 32
/// shares (spec/shares.md section 11.2).
pub const SHARE_RATIO_LOG2: u32 = 5;

/// A node keeps the models of the tip and of this many levels below it
/// (spec/shares.md section 11.4); 100 exceeds the longest reorganisation (60).
pub const MODEL_DEPTH: u32 = 100;

#[derive(Debug, PartialEq, Eq)]
/// Why a proof of work or a share list is rejected. All three make the block
/// invalid.
pub enum ProofError {
    /// The share list or the bits of the block are malformed.
    Shape(&'static str),
    /// The ticket of the block is above its target.
    BadPow,
    /// This share record has a ticket above the share target.
    BadShare(ShareRecord),
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofError::Shape(why) => write!(f, "{why}"),
            ProofError::BadPow => write!(f, "proof of work does not meet the target"),
            ProofError::BadShare(r) => {
                write!(f, "share record (time {}, nonce {}) does not meet the share target", r.time, r.nonce)
            }
        }
    }
}

impl std::error::Error for ProofError {}

/// `min(target(bits) * 2^5, 2^256 - 1)` (spec/shares.md section 11.2).
pub fn share_target(bits: u32) -> Result<Target, TargetError> {
    let target = Target::from_bits(bits)?;
    Ok(target.shifted_left(SHARE_RATIO_LOG2).unwrap_or(Target([0xFF; 32])))
}

/// Key of a share record of the round whose headers have the parent `parent`
/// and the bits `bits`: the id of its share header (spec/shares.md section
/// 11.2).
pub fn share_key(record: &ShareRecord, parent: [u8; 32], bits: u32) -> [u8; 32] {
    record.header(parent, bits).id()
}

/// Checks the proof of work of `block` (spec/shares.md section 11.1) and its
/// share list, and returns the state of the block (section 11.2).
///
/// `height` is the height of `block`; `models_p` is the state of the parent
/// `parent`; `models_pp` is the state of the grandparent, equal to `models_p`
/// when the parent is the genesis block (the block has height 1). Order: list
/// shape, then the one ticket of the block, then one ticket per share (at most
/// 256), so an invalid block is rejected as early and as cheaply as possible.
/// With `check` false the targets are not compared (tests of the other rules);
/// the gradients are still summed.
pub fn verify_block(
    params: &ChainParams,
    check: bool,
    block: &Block,
    parent: &Header,
    height: u32,
    models_p: &Models,
    models_pp: &Models,
) -> Result<Models, ProofError> {
    check_shares_list(&block.shares).map_err(ProofError::Shape)?;
    if height <= 1 && !block.shares.is_empty() {
        return Err(ProofError::Shape("the block of height 1 lists no share"));
    }
    let k = params.slots();
    if models_p.len() != k || models_pp.len() != k {
        return Err(ProofError::Shape("the state does not have one model per slot"));
    }
    // The key of the block is its id; its ticket uses the model of its slot in the state of the parent, and the
    // dataset of its height (an upgrade changes the dataset of a slot from a height on).
    if check {
        let s = slot(height, k);
        let target = Target::from_bits(block.header.bits).map_err(|_| ProofError::Shape("bits"))?;
        if !target.admits(&ShareBuffers::new().ticket(&models_p[s], &**params.dataset_at(height), &block.id())) {
            return Err(ProofError::BadPow);
        }
    }
    if block.shares.is_empty() {
        return Ok(params.start_upgrades(height, models_p.clone()));
    }
    // The shares were mined for the parent's height, on the model of the parent's slot and the dataset of that height.
    let t = slot(height - 1, k);
    let ds: &dyn Dataset = &**params.dataset_at(height - 1);
    let model_pp = &models_pp[t];
    // Shares belong to the round of the parent: headers whose parent is the
    // parent of `parent`, with its bits, evaluated on M(PP). The gradients are
    // summed in int32, so the order of the list does not matter.
    let target = share_target(parent.bits).map_err(|_| ProofError::Shape("bits of the parent"))?;
    let mut sum = GradSum::new();
    for record in &block.shares {
        let ticket = sum.add(model_pp, ds, &share_key(record, parent.parent, parent.bits));
        if check && !target.admits(&ticket) {
            return Err(ProofError::BadShare(*record));
        }
    }
    // The step applies to the model of the parent's state, not the grandparent's.
    let mut next = (*models_p[t]).clone();
    sum.apply(&mut next);
    let mut models: Vec<Arc<Model>> = models_p.to_vec();
    models[t] = Arc::new(next);
    Ok(params.start_upgrades(height, models.into()))
}
