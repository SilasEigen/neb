//! Share records at the node (spec/shares.md section 11.5, spec/node.md section
//! 8): the ones its miners find and the ones its peers relay with `shares`.
//!
//! A share is bound to the parent of its header, so a record is valid for a
//! parent whatever the block that wins the round. The node keeps the records of
//! two parents, the tip and the parent of the tip; the job for a child of the
//! tip lists the records of the second. Every record is evaluated once (one
//! ticket) and relayed once. Local policy only: no rule here affects consensus.

use neb_chain::block::{MAX_SHARES_PER_BLOCK, ShareRecord};
use neb_chain::share_key;
use neb_pow::shares::ShareBuffers;

use crate::budget::TokenBucket;
use crate::node::{Node, NodeError, NodeInner};

impl Node {
    /// Records a share a miner found on job `id` for the share list of a later
    /// block and relays it to the peers. True when the record is new.
    pub fn accept_share(&self, id: u64, nonce: u32) -> Result<bool, String> {
        let new = {
            let mut inner = self.inner.lock().expect("node lock");
            let NodeInner { state, work, .. } = &mut *inner;
            work.add_share(id, nonce, &*state.params.dataset, state.verify_pow)?
        };
        match new {
            Some((parent, record)) => {
                self.announce_shares(&parent, &[record], None);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Records the share records of a `shares` message of a peer, whose headers
    /// have the parent `parent`. A parent that is neither the tip nor the parent
    /// of the tip is ignored. A record already kept costs nothing; every other
    /// one costs a ticket evaluation and a token of `budget` (spec/node.md
    /// section 8.1). The records that are new are relayed to the other peers
    /// (all except the session `from`). Returns how many were new. A record
    /// whose ticket is above the share target is an `Err(BadProof)`: the caller
    /// bans the peer, as for a block with an invalid share.
    pub fn accept_peer_shares(
        &self,
        parent: [u8; 32],
        records: &[ShareRecord],
        mut budget: Option<&mut TokenBucket>,
        from: Option<u64>,
    ) -> Result<usize, NodeError> {
        let (model, bits, target, dataset, fresh) = {
            let inner = self.inner.lock().expect("node lock");
            let Some((model, bits, target)) = inner.state.share_round(&parent) else {
                return Ok(0);
            };
            let mut fresh: Vec<ShareRecord> =
                records.iter().filter(|r| !inner.work.has_record(&parent, r)).copied().collect();
            fresh.sort_unstable();
            fresh.dedup();
            (model, bits, target, inner.state.params.dataset.clone(), fresh)
        };
        if fresh.is_empty() {
            return Ok(0);
        }
        // The node lock is released: waiting for tokens may take seconds.
        let mut buffers = ShareBuffers::new();
        for record in &fresh {
            if !self.verify_pow {
                continue;
            }
            if let Some(budget) = budget.as_deref_mut() {
                budget.wait();
            }
            let ticket = buffers.ticket(&model, &*dataset, &share_key(record, parent, bits));
            if !target.admits(&ticket) {
                return Err(NodeError::BadProof(format!(
                    "share record (time {}, nonce {}) does not meet the share target",
                    record.time, record.nonce
                )));
            }
        }
        let new = self.inner.lock().expect("node lock").work.add_records(parent, &fresh);
        self.announce_shares(&parent, &new, from);
        Ok(new.len())
    }

    /// Sends the records, sorted and at most 256 per message, to every peer
    /// except the session `except` (spec/node.md section 8.2).
    pub fn announce_shares(&self, parent: &[u8; 32], records: &[ShareRecord], except: Option<u64>) {
        let mut sorted = records.to_vec();
        sorted.sort_unstable();
        for chunk in sorted.chunks(MAX_SHARES_PER_BLOCK) {
            if let Ok(frame) = crate::p2p::shares_frame(self.network.magic, parent, chunk) {
                self.peers.broadcast(&frame, except);
            }
        }
    }
}
