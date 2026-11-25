//! Transactions that would otherwise be lost: those of the blocks a reorganisation removed return to the mempool,
//! and a transaction handed to this node is announced again until a peer asks for it. Local policy.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use neb_chain::tx::Transaction;

use crate::node::{InvKind, MAX_INV, Node, NodeError, NodeInner};
use crate::p2p::{PeerSet, inv_frame};

/// The transactions handed to this node that no peer has asked for yet.
#[derive(Default)]
pub(crate) struct Unasked {
    ids: BTreeSet<[u8; 32]>,
    /// Whether the thread that announces them again is running.
    resending: bool,
}

/// Time between two announcements of the transactions no peer has asked for.
const RESEND_EVERY: Duration = Duration::from_secs(60);

/// Puts back in the mempool the transactions of the blocks `old_ids` (newest first) that left the best chain. One
/// that the new branch holds, that conflicts with it or that the pool refuses for any other reason is dropped, as
/// is one that spends an output of another of them: the pool takes confirmed outputs only.
pub(crate) fn return_txs(inner: &mut NodeInner, old_ids: &[[u8; 32]]) {
    let NodeInner { state, mempool, blocks, .. } = inner;
    for tx in old_ids.iter().rev().filter_map(|id| blocks.get(id)).flat_map(|b| b.txs.iter().skip(1)) {
        let _ = mempool.accept(state, tx);
    }
}

/// Announces the transactions of `unasked` that are still in the mempool and forgets the others. Returns how many
/// were announced.
fn resend(inner: &Mutex<NodeInner>, peers: &PeerSet, unasked: &Mutex<Unasked>, magic: [u8; 4]) -> usize {
    let ids: Vec<[u8; 32]> = {
        let inner = inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut unasked = unasked.lock().unwrap_or_else(|e| e.into_inner());
        unasked.ids.retain(|id| inner.mempool.tx(id).is_some());
        unasked.ids.iter().copied().collect()
    };
    for part in ids.chunks(MAX_INV) {
        if let Ok(frame) = inv_frame(magic, InvKind::Tx, part) {
            peers.broadcast(&frame, None);
        }
    }
    ids.len()
}

impl Node {
    /// Adds a transaction of a peer to the mempool and announces it. A transaction already in the mempool is
    /// returned as accepted.
    pub fn accept_tx(&self, tx: Transaction) -> Result<[u8; 32], NodeError> {
        let id = tx.txid();
        let mut inner = self.inner.lock().expect("node lock");
        if inner.mempool.tx(&id).is_some() {
            return Ok(id);
        }
        {
            let NodeInner { state, mempool, .. } = &mut *inner;
            mempool.accept(state, &tx).map_err(|e| NodeError::Connect(e.to_string()))?;
        }
        drop(inner);
        self.announce(InvKind::Tx, &[id], None);
        Ok(id)
    }

    /// As `accept_tx`, for a transaction handed to this node by its user: it is announced again every minute
    /// until a peer asks for it or it leaves the mempool, so a node without a peer at that moment does not lose it.
    pub fn submit_tx(&self, tx: Transaction) -> Result<[u8; 32], NodeError> {
        let id = self.accept_tx(tx)?;
        let mut unasked = self.unasked.lock().unwrap_or_else(|e| e.into_inner());
        unasked.ids.insert(id);
        // One thread while the list is not empty; it ends with the node, of which it holds no part.
        if !unasked.resending {
            unasked.resending = true;
            let (inner, peers) = (Arc::downgrade(&self.inner), Arc::downgrade(&self.peers));
            let (unasked, magic) = (Arc::downgrade(&self.unasked), self.network.magic);
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(RESEND_EVERY);
                    let (Some(inner), Some(peers), Some(unasked)) =
                        (inner.upgrade(), peers.upgrade(), unasked.upgrade())
                    else {
                        return;
                    };
                    resend(&inner, &peers, &unasked, magic);
                    let mut unasked = unasked.lock().unwrap_or_else(|e| e.into_inner());
                    if unasked.ids.is_empty() {
                        unasked.resending = false;
                        return;
                    }
                }
            });
        }
        Ok(id)
    }

    /// Announces now the transactions handed to this node that no peer has asked for. Returns how many.
    pub fn resend_unasked(&self) -> usize {
        resend(&self.inner, &self.peers, &self.unasked, self.network.magic)
    }

    /// The mempool transaction `id`, for a peer that asks for it.
    pub fn mempool_tx(&self, id: &[u8; 32]) -> Option<Transaction> {
        let tx = self.inner.lock().expect("node lock").mempool.tx(id).cloned();
        if tx.is_some() {
            self.unasked.lock().unwrap_or_else(|e| e.into_inner()).ids.remove(id);
        }
        tx
    }
}
