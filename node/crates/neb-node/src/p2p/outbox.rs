//! The frames waiting to be written to one peer, and the thread that writes them.
//!
//! A relay only adds its frame to the queue of each peer, so a peer that reads slowly or not at all delays no
//! other peer. The session of the peer waits before it adds an answer while more than `PAUSE_BYTES` are queued:
//! the node then reads nothing more from that peer, so its requests cannot grow the queue without bound. Only the
//! writer decides that a peer is stalled: a write that makes no progress for the write limit closes the peer, so
//! a peer on a slow link that keeps taking bytes is kept.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use super::MAX_PAYLOAD;

/// Above this many queued bytes the session of the peer waits before it queues an answer.
const PAUSE_BYTES: usize = MAX_PAYLOAD;
/// Above this many queued bytes the peer is closed. Answers stop at `PAUSE_BYTES` plus one frame, so only relayed
/// frames that the peer does not read can reach it.
const CLOSE_BYTES: usize = 4 * MAX_PAYLOAD;

struct State {
    frames: VecDeque<Arc<[u8]>>,
    bytes: usize,
    /// No frame is accepted any more; the writer ends once the queue is empty.
    ending: bool,
    /// The socket is shut down and the queue dropped.
    closed: bool,
}

pub(crate) struct Outbox {
    state: Mutex<State>,
    changed: Condvar,
    stream: TcpStream,
}

impl Outbox {
    /// Takes the write half of a session and starts its writer thread. A single write that makes no progress
    /// for `limit` closes the peer.
    pub(crate) fn start(stream: TcpStream, limit: Duration) -> io::Result<Arc<Outbox>> {
        stream.set_write_timeout(Some(limit))?;
        let outbox = Arc::new(Outbox {
            state: Mutex::new(State { frames: VecDeque::new(), bytes: 0, ending: false, closed: false }),
            changed: Condvar::new(),
            stream,
        });
        let own = outbox.clone();
        // The writer only copies bytes to a socket: a small stack, so a relay with a thousand peers stays light.
        // When the system gives no thread the session does not start.
        thread::Builder::new().stack_size(128 * 1024).spawn(move || own.write_loop())?;
        Ok(outbox)
    }

    fn write_loop(&self) {
        let mut stream = &self.stream;
        loop {
            let frame = {
                let mut state = self.state.lock().expect("outbox lock");
                loop {
                    if state.closed {
                        return;
                    }
                    if let Some(frame) = state.frames.pop_front() {
                        break frame;
                    }
                    if state.ending {
                        drop(state);
                        self.close();
                        return;
                    }
                    state = self.changed.wait(state).expect("outbox lock");
                }
            };
            // The lock is not held during the write: other threads keep queueing.
            if stream.write_all(&frame).and_then(|()| stream.flush()).is_err() {
                self.close();
                return;
            }
            let mut state = self.state.lock().expect("outbox lock");
            if state.closed {
                return;
            }
            state.bytes -= frame.len();
            drop(state);
            self.changed.notify_all();
        }
    }

    fn queue(&self, state: &mut State, frame: Arc<[u8]>) {
        state.bytes += frame.len();
        state.frames.push_back(frame);
        self.changed.notify_all();
    }

    /// Queues a relayed frame and returns at once; the bytes are shared between the peers of one relay. A peer
    /// whose queue is over `CLOSE_BYTES` is closed.
    pub(crate) fn relay(&self, frame: &Arc<[u8]>) -> io::Result<()> {
        let mut state = self.state.lock().expect("outbox lock");
        if state.closed || state.ending {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        if state.bytes > CLOSE_BYTES {
            drop(state);
            self.close();
            return Err(io::ErrorKind::TimedOut.into());
        }
        self.queue(&mut state, frame.clone());
        Ok(())
    }

    /// Queues a frame of the session of this peer. Waits while more than `PAUSE_BYTES` are queued, until the
    /// writer wrote some of them or closed the peer.
    pub(crate) fn send(&self, frame: &[u8]) -> io::Result<()> {
        let mut state = self.state.lock().expect("outbox lock");
        loop {
            if state.closed || state.ending {
                return Err(io::ErrorKind::BrokenPipe.into());
            }
            if state.bytes <= PAUSE_BYTES {
                break;
            }
            state = self.changed.wait(state).expect("outbox lock");
        }
        self.queue(&mut state, frame.into());
        Ok(())
    }

    /// Ends the session: the frames already queued are written, then the socket is shut down.
    pub(crate) fn finish(&self) {
        self.state.lock().expect("outbox lock").ending = true;
        self.changed.notify_all();
    }

    /// Shuts the socket down and drops the queue, which ends the session: its thread reads the closed socket.
    fn close(&self) {
        let mut state = self.state.lock().expect("outbox lock");
        state.closed = true;
        state.frames.clear();
        state.bytes = 0;
        drop(state);
        let _ = self.stream.shutdown(Shutdown::Both);
        self.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;
    use std::time::Instant;

    fn pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let near = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        (near, listener.accept().unwrap().0)
    }

    fn read_all(mut far: TcpStream) -> thread::JoinHandle<usize> {
        thread::spawn(move || {
            let (mut total, mut tmp) = (0usize, vec![0u8; 1 << 16]);
            while let Ok(n) = far.read(&mut tmp) {
                if n == 0 {
                    break;
                }
                total += n;
            }
            total
        })
    }

    // A peer that asks and never reads: its own session is paused, then closed, while a relay to it and to a
    // peer that reads returns at once.
    #[test]
    fn a_peer_that_stopped_reading_delays_no_relay() {
        let (stalled_near, stalled_far) = pair();
        let (reading_near, reading_far) = pair();
        let stalled = Outbox::start(stalled_near, Duration::from_millis(300)).unwrap();
        let reading = Outbox::start(reading_near, Duration::from_millis(300)).unwrap();
        let reader = read_all(reading_far);

        // the session of the stalled peer queues answers until it is paused
        let answers = {
            let stalled = stalled.clone();
            thread::spawn(move || {
                let chunk = vec![0u8; 1 << 20];
                (0..1024).position(|_| stalled.send(&chunk).is_err())
            })
        };
        thread::sleep(Duration::from_millis(100));
        let inv: Arc<[u8]> = Arc::from([7u8; 37]);
        let t = Instant::now();
        let _ = stalled.relay(&inv);
        reading.relay(&inv).unwrap();
        assert!(t.elapsed() < Duration::from_millis(50), "the relay waited {:?}", t.elapsed());

        let closed_after = answers.join().unwrap();
        assert!(closed_after.is_some(), "1 GiB queued for a peer that never reads");
        let t = Instant::now();
        assert!(stalled.relay(&inv).is_err() && stalled.send(&inv).is_err());
        assert!(t.elapsed() < Duration::from_millis(50), "a closed peer made a write wait {:?}", t.elapsed());

        reading.finish();
        assert_eq!(reader.join().unwrap(), inv.len());
        drop(stalled_far);
    }

    // A peer on a slow link that keeps reading is kept, however long its queue stays over the pause level.
    #[test]
    fn a_slow_peer_that_keeps_reading_is_kept() {
        let (near, mut far) = pair();
        let outbox = Outbox::start(near, Duration::from_millis(200)).unwrap();
        let reader = thread::spawn(move || {
            let (mut total, mut tmp) = (0usize, vec![0u8; 128 * 1024]);
            while let Ok(n) = far.read(&mut tmp) {
                if n == 0 {
                    break;
                }
                total += n;
                thread::sleep(Duration::from_millis(20));
            }
            total
        });
        let chunk = vec![3u8; 1 << 20];
        let t = Instant::now();
        for i in 0..10 {
            outbox.send(&chunk).unwrap_or_else(|e| panic!("frame {i} refused after {:?}: {e}", t.elapsed()));
        }
        outbox.finish();
        assert_eq!(reader.join().unwrap(), 10 << 20);
        assert!(t.elapsed() > Duration::from_millis(600), "the link was not slow: {:?}", t.elapsed());
    }

    // Relays alone, never read, are bounded: past `CLOSE_BYTES` the peer is closed.
    #[test]
    fn relays_to_a_peer_that_reads_nothing_are_bounded() {
        let (near, far) = pair();
        let outbox = Outbox::start(near, Duration::from_secs(30)).unwrap();
        let chunk: Arc<[u8]> = Arc::from(vec![0u8; 1 << 20]);
        let refused = (0..1024).position(|_| outbox.relay(&chunk).is_err());
        assert!(refused.is_some(), "1 GiB of relays queued");
        drop(far);
    }

    // `finish` writes what is queued before the socket closes.
    #[test]
    fn finish_writes_the_queued_frames() {
        let (near, far) = pair();
        let outbox = Outbox::start(near, Duration::from_secs(10)).unwrap();
        let reader = read_all(far);
        for _ in 0..8 {
            outbox.send(&vec![1u8; 300_000]).unwrap();
        }
        outbox.finish();
        assert!(outbox.send(&[0u8; 1]).is_err());
        assert_eq!(reader.join().unwrap(), 8 * 300_000);
    }
}
