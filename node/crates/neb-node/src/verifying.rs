//! The blocks whose proof is being checked.
//!
//! Every peer announces a new block and each session that receives it would check its proof and its shares, at
//! the same time and for the same result. A session claims the block before the check: the first one checks, the
//! others wait for it and then find the block known. When the check fails the next session checks in its turn, so
//! each peer still answers for the block it sent.

use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex};

#[derive(Default)]
pub(crate) struct Verifying {
    ids: Mutex<BTreeSet<[u8; 32]>>,
    released: Condvar,
}

/// One block held by the thread that checks it; dropping it lets the next thread in.
pub(crate) struct Claim<'a> {
    set: &'a Verifying,
    id: [u8; 32],
}

impl Verifying {
    /// Waits until no other thread holds `id`, then holds it for the caller.
    pub(crate) fn claim(&self, id: [u8; 32]) -> Claim<'_> {
        let mut ids = self.ids.lock().unwrap_or_else(|e| e.into_inner());
        while ids.contains(&id) {
            ids = self.released.wait(ids).unwrap_or_else(|e| e.into_inner());
        }
        ids.insert(id);
        Claim { set: self, id }
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.set.ids.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.id);
        self.set.released.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    // Sixteen threads bring the same block: never two inside the check at once, and another block is not delayed.
    #[test]
    fn one_thread_at_a_time_holds_a_block() {
        let set = Arc::new(Verifying::default());
        let (inside, most) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let other = set.claim([2; 32]);
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let (set, inside, most) = (set.clone(), inside.clone(), most.clone());
                thread::spawn(move || {
                    let _claim = set.claim([1; 32]);
                    most.fetch_max(inside.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(2));
                    inside.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(most.load(Ordering::SeqCst), 1);
        drop(other);
        assert!(set.ids.lock().unwrap().is_empty());
    }
}
