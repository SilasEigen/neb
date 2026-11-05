//! Proof of work and model step of a block (spec/shares.md section 11).
//!
//! The ticket of a block is an attempt on the model of its parent. The shares
//! it lists belong to the round of its parent and are computed on the model of
//! the grandparent; their summed gradients update the model of the parent, so
//! a share trains the model one block after it was found. Everything here is
//! integer arithmetic done by `neb_pow` (the C reference), and every node
//! must obtain the same model.

use std::sync::Arc;

use neb_block::{Header, Target, TargetError};
use neb_pow::shares::{Dataset, GradSum, Model, ShareBuffers};

use crate::block::{Block, ShareRecord, check_shares_list};
use crate::state::ChainParams;

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
/// share list, and returns the model of the block (section 11.2).
///
/// `model_p` is the model of the parent `parent`; `model_pp` is the model of
/// the grandparent, equal to `model_p` when the parent is the genesis block
/// (`parent_is_genesis`, the block has height 1). Order: list shape, then the
/// one ticket of the block, then one ticket per share (at most 256), so an
/// invalid block is rejected as early and as cheaply as possible. With `check`
/// false the targets are not compared (tests of the other rules); the
/// gradients are still summed.
pub fn verify_block(
    params: &ChainParams,
    check: bool,
    block: &Block,
    parent: &Header,
    parent_is_genesis: bool,
    model_p: &Arc<Model>,
    model_pp: &Arc<Model>,
) -> Result<Arc<Model>, ProofError> {
    check_shares_list(&block.shares).map_err(ProofError::Shape)?;
    if parent_is_genesis && !block.shares.is_empty() {
        return Err(ProofError::Shape("the block of height 1 lists no share"));
    }
    let ds: &dyn Dataset = &*params.dataset;
    // The key of the block is its id; its ticket uses the model of the parent.
    if check {
        let target = Target::from_bits(block.header.bits).map_err(|_| ProofError::Shape("bits"))?;
        if !target.admits(&ShareBuffers::new().ticket(model_p, ds, &block.id())) {
            return Err(ProofError::BadPow);
        }
    }
    if block.shares.is_empty() {
        return Ok(model_p.clone());
    }
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
    // The step applies to M(P), not M(PP): a delay of one block.
    let mut next = (**model_p).clone();
    sum.apply(&mut next);
    Ok(Arc::new(next))
}
