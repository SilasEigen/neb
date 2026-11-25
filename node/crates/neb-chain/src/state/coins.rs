//! The unspent outputs of the best chain, with the list of the outpoints of each lock: the outputs of one lock are
//! read without a walk of the whole set. The list is derived from the outputs, so the state file does not hold it.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;

use super::{ChainState, OutPoint, Utxo};

#[derive(Default)]
pub(super) struct Coins {
    map: BTreeMap<OutPoint, Utxo>,
    /// `(lock data, outpoint)` of every output of `map`.
    owned: BTreeSet<([u8; 32], OutPoint)>,
}

impl Coins {
    pub(super) fn map(&self) -> &BTreeMap<OutPoint, Utxo> {
        &self.map
    }

    pub(super) fn get(&self, op: &OutPoint) -> Option<&Utxo> {
        self.map.get(op)
    }

    pub(super) fn contains_key(&self, op: &OutPoint) -> bool {
        self.map.contains_key(op)
    }

    pub(super) fn len(&self) -> usize {
        self.map.len()
    }

    /// As `BTreeMap::insert`: the output that `op` named before, if any.
    pub(super) fn insert(&mut self, op: OutPoint, utxo: Utxo) -> Option<Utxo> {
        self.owned.insert((utxo.lock_data, op));
        let old = self.map.insert(op, utxo.clone());
        if let Some(old) = &old
            && old.lock_data != utxo.lock_data
        {
            self.owned.remove(&(old.lock_data, op));
        }
        old
    }

    pub(super) fn remove(&mut self, op: &OutPoint) -> Option<Utxo> {
        let old = self.map.remove(op)?;
        self.owned.remove(&(old.lock_data, *op));
        Some(old)
    }

    /// The first `limit` outputs of `lock` that `keep` accepts, in outpoint order, after `after` when one is given.
    pub(super) fn of(
        &self,
        lock: &[u8; 32],
        after: Option<OutPoint>,
        limit: usize,
        keep: impl Fn(&Utxo) -> bool,
    ) -> Vec<(OutPoint, Utxo)> {
        let start = after
            .map_or(Bound::Included((*lock, OutPoint { txid: [0; 32], index: 0 })), |op| Bound::Excluded((*lock, op)));
        let end = Bound::Included((*lock, OutPoint { txid: [0xFF; 32], index: u32::MAX }));
        let owned = self.owned.range((start, end)).filter_map(|(_, op)| Some((*op, self.map.get(op)?.clone())));
        owned.filter(|(_, u)| keep(u)).take(limit).collect()
    }
}

impl<'a> IntoIterator for &'a Coins {
    type Item = (&'a OutPoint, &'a Utxo);
    type IntoIter = std::collections::btree_map::Iter<'a, OutPoint, Utxo>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

impl ChainState {
    /// The first `limit` unspent outputs whose lock data is `lock`, in outpoint order, after the outpoint `after`
    /// when one is given. Reads the outputs of that lock only.
    pub fn utxos_of(&self, lock: &[u8; 32], after: Option<OutPoint>, limit: usize) -> Vec<(OutPoint, Utxo)> {
        self.utxos.of(lock, after, limit, |_| true)
    }

    /// As `utxos_of`, for the outputs of the lock type `lock_type`, if one is given, that hold `min_amount` or
    /// more.
    pub fn utxos_where(
        &self,
        lock: &[u8; 32],
        after: Option<OutPoint>,
        limit: usize,
        (lock_type, min_amount): (Option<u8>, u64),
    ) -> Vec<(OutPoint, Utxo)> {
        self.utxos.of(lock, after, limit, |u| u.amount >= min_amount && lock_type.is_none_or(|t| t == u.lock_type))
    }
}
