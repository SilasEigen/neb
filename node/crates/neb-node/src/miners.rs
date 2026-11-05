//! The miner sessions of the work service (spec/node.md section 7.1):
//! the write half of every connected session, so that a block connecting can
//! push a fresh job to all of them at once, not only to the miner that
//! submitted it.

use std::collections::BTreeMap;
use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// One miner session's write half.
pub type MinerWriter = Arc<Mutex<TcpStream>>;

/// Registry of the open work-service sessions. The set lock is released
/// before any writer lock is taken, so `broadcast` cannot deadlock with a
/// session thread that registers or unregisters.
pub struct MinerSet {
    next_id: AtomicU64,
    sessions: Mutex<BTreeMap<u64, MinerWriter>>,
}

impl Default for MinerSet {
    fn default() -> Self {
        Self::new()
    }
}

impl MinerSet {
    pub fn new() -> MinerSet {
        MinerSet { next_id: AtomicU64::new(1), sessions: Mutex::new(BTreeMap::new()) }
    }

    /// Registers one session; the id unregisters it when the session ends.
    pub fn add(&self, writer: MinerWriter) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.sessions.lock().expect("miner lock").insert(id, writer);
        id
    }

    pub fn remove(&self, id: u64) {
        self.sessions.lock().expect("miner lock").remove(&id);
    }

    pub fn len(&self) -> usize {
        self.sessions.lock().expect("miner lock").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writes one frame to every connected session. A failed write is ignored:
    /// the session's own thread sees the closed socket and unregisters.
    pub fn broadcast(&self, frame: &[u8]) {
        let writers: Vec<MinerWriter> = self.sessions.lock().expect("miner lock").values().cloned().collect();
        for writer in writers {
            let mut stream = writer.lock().expect("miner writer");
            let _ = stream.write_all(frame);
            let _ = stream.flush();
        }
    }
}
