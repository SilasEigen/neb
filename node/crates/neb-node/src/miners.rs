//! The miner sessions of the work service (spec/node.md section 7.1):
//! the write half of every connected session, so that a block connecting can
//! push a fresh job to all of them at once, not only to the miner that
//! submitted it.

use std::collections::BTreeMap;
use std::io::{self, Write};
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
            let _ = write(&writer, frame);
        }
    }
}

/// Writes one frame to a session. A write that fails or exceeds `WRITE_TIMEOUT` shuts the socket down, which ends
/// the session, so a session that stopped reading holds up the others once and not at every job.
pub fn write(writer: &MinerWriter, frame: &[u8]) -> io::Result<()> {
    let mut stream = writer.lock().expect("miner writer");
    let result = stream.write_all(frame).and_then(|()| stream.flush());
    if result.is_err() {
        let _ = stream.shutdown(std::net::Shutdown::Both);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;
    use std::time::{Duration, Instant};

    // A session that stopped reading is closed at its first write that times out, and the sessions after it in
    // the set still get the frame.
    #[test]
    fn a_session_that_stopped_reading_does_not_hold_the_others() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let pair = || (TcpStream::connect(listener.local_addr().unwrap()).unwrap(), listener.accept().unwrap().0);
        let (stalled, stalled_far) = pair();
        stalled.set_write_timeout(Some(Duration::from_millis(200))).unwrap();
        let (reading, mut reading_far) = pair();
        let set = MinerSet::new();
        let stalled = Arc::new(Mutex::new(stalled));
        set.add(stalled.clone());
        set.add(Arc::new(Mutex::new(reading)));
        let reader = std::thread::spawn(move || {
            let mut total = 0usize;
            let mut tmp = vec![0u8; 1 << 16];
            while let Ok(n) = reading_far.read(&mut tmp) {
                if n == 0 {
                    break;
                }
                total += n;
            }
            total
        });
        // the first session never reads: its socket buffers fill up and a write to it times out
        let chunk = vec![0u8; 1 << 20];
        let mut sent = 0usize;
        let closed = (0..1024).any(|_| {
            set.broadcast(&chunk);
            sent += 1;
            write(&stalled, &[0u8; 1]).is_err()
        });
        assert!(closed, "1 GiB written to a session that never reads");
        let t = Instant::now();
        set.broadcast(&chunk);
        assert!(t.elapsed() < Duration::from_millis(150), "the closed session held the push for {:?}", t.elapsed());
        drop(set);
        assert_eq!(reader.join().unwrap(), (sent + 1) << 20, "the other session got every frame");
        drop(stalled_far);
    }
}
