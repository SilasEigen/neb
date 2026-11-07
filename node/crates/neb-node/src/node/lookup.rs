//! Lookups over the blocks the node knows: the fork point of a branch, the start of a locator, and the kept states of
//! side-branch blocks.

use neb_chain::MODEL_DEPTH;

use super::{MAX_SIDE_MODELS, NodeInner};

/// Height of the best-chain ancestor of `id` (the block itself when it is on
/// the best chain); `None` when an ancestor is unknown.
pub(super) fn fork_height(inner: &NodeInner, id: &[u8; 32]) -> Option<u32> {
    let mut cur = *id;
    loop {
        if let Some(h) = inner.best_index.get(&cur) {
            return Some(*h);
        }
        let header = inner.blocks.get(&cur)?.header;
        if header.parent == [0u8; 32] {
            return Some(0);
        }
        cur = header.parent;
    }
}

pub(super) fn locator_start(inner: &NodeInner, locator: &[[u8; 32]]) -> u32 {
    for id in locator {
        if let Some(h) = inner.best_index.get(id) {
            return h + 1;
        }
    }
    0
}

/// The state of a known block: from the best chain when it is among the last
/// `MODEL_DEPTH + 1` levels, else from the side-branch states. `None` when the
/// state is not kept (spec/shares.md section 11.4).
pub(super) fn models_of(inner: &NodeInner, id: &[u8; 32]) -> Option<neb_chain::Models> {
    match inner.best_index.get(id) {
        Some(height) => inner.state.models_at(*height),
        None => inner.side_models.get(id).cloned(),
    }
}

/// Drops side models older than `MODEL_DEPTH` levels below the tip, then the
/// lowest blocks until at most `MAX_SIDE_MODELS` remain. The bound limits the
/// memory an attacker can pin with valid side-branch blocks.
pub(super) fn prune_side_models(inner: &mut NodeInner) {
    let floor = inner.state.height().saturating_sub(MODEL_DEPTH);
    let heights = &inner.heights;
    inner.side_models.retain(|id, _| heights.get(id).is_some_and(|h| *h >= floor));
    while inner.side_models.len() > MAX_SIDE_MODELS {
        let lowest = *inner.side_models.keys().min_by_key(|id| heights[*id]).expect("not empty");
        inner.side_models.remove(&lowest);
    }
}
