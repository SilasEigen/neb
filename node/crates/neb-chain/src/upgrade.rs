//! Dataset upgrades at a height (spec/shares.md section 11.7): the datasets and the models of the slots that an
//! upgrade moves, as the chain parameters give them.

use std::sync::Arc;

use neb_pow::shares::{Model, SharedDataset};

use crate::model::{self, Models, slot};
use crate::state::ChainParams;

/// From block `height` on, model slot `slot` works on a new dataset, starting from a new model (spec/shares.md
/// section 11.7). `height mod K` equals `slot`, so the first block on the new dataset is the first block of the
/// slot at or above `height`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Upgrade {
    pub height: u32,
    pub slot: usize,
}

impl ChainParams {
    /// K, the number of model slots: the datasets that no upgrade brought.
    pub fn slots(&self) -> usize {
        self.datasets.len() - self.upgrades.len()
    }

    /// Whether the upgrades are well formed: fewer than the datasets, increasing heights of at least 2, each at a
    /// height of its own slot.
    pub fn check_upgrades(&self) -> Result<(), String> {
        if self.upgrades.len() >= self.datasets.len() {
            return Err("the network file has no dataset left for a model slot".into());
        }
        let k = self.slots();
        let mut last = 1;
        for u in &self.upgrades {
            if u.height <= last || u.slot >= k || slot(u.height, k) != u.slot {
                return Err(format!("upgrade at height {} of slot {} is not valid for {k} slots", u.height, u.slot));
            }
            last = u.height;
        }
        Ok(())
    }

    /// The index in `datasets` of the dataset a block of height `height` works on: the dataset of the last upgrade
    /// of its slot at or below `height`, otherwise the dataset of its slot (spec/shares.md section 11.7).
    pub fn dataset_index_at(&self, height: u32) -> usize {
        let k = self.slots();
        let s = slot(height, k);
        let mut index = s;
        for (i, u) in self.upgrades.iter().enumerate() {
            if u.slot == s && u.height <= height {
                index = k + i;
            }
        }
        index
    }

    /// The dataset of the model a block of height `height` works on.
    pub fn dataset_at(&self, height: u32) -> &SharedDataset {
        &self.datasets[self.dataset_index_at(height)]
    }

    /// The last upgrade of `slot` whose model is in the state of the block at `height`: an upgrade starting at `H`
    /// replaces the model in the state of the block at `H - 1` (see `start_upgrades`).
    fn upgrade_in_state(&self, slot: usize, height: u32) -> Option<(usize, &Upgrade)> {
        self.upgrades.iter().enumerate().rev().find(|(_, u)| u.slot == slot && u.height <= height.saturating_add(1))
    }

    /// The index of the dataset that the model of `slot` in the state of the block at `height` belongs to: the
    /// dataset of the last upgrade of that slot already in that state, otherwise the dataset of the slot.
    pub fn dataset_index_of(&self, slot: usize, height: u32) -> usize {
        self.upgrade_in_state(slot, height).map_or(slot, |(i, _)| self.slots() + i)
    }

    /// The model the model of `slot` in the state of the block at `height` started from: the genesis model of the
    /// slot, or the initial model of its last upgrade already in that state (spec/shares.md sections 11.2 and 11.7).
    pub fn initial_model_of(&self, chain_id: &[u8; 32], slot: usize, height: u32) -> Arc<Model> {
        match self.upgrade_in_state(slot, height) {
            Some((i, u)) => Arc::new(model::upgrade_model(u, self.datasets[self.slots() + i].arch())),
            None => model::initial_models(chain_id, self.slots())[slot].clone(),
        }
    }

    /// The state of the block at `height` with the upgrades that start at `height + 1` applied: the slot of each
    /// gets the initial model of the upgrade (spec/shares.md section 11.7).
    pub fn start_upgrades(&self, height: u32, models: Models) -> Models {
        let mut next: Option<Vec<Arc<Model>>> = None;
        for (i, u) in self.upgrades.iter().enumerate().filter(|(_, u)| u.height == height.wrapping_add(1)) {
            let arch = self.datasets[self.slots() + i].arch();
            next.get_or_insert_with(|| models.to_vec())[u.slot] = Arc::new(model::upgrade_model(u, arch));
        }
        next.map_or(models, Models::from)
    }
}
