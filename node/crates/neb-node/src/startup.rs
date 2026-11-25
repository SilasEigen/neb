//! The chain state at the start of the node: read from the state file of the data directory when that file fits
//! the stored chain, replayed from the genesis block otherwise.
//!
//! The file is `state.dat`, the bytes of `ChainState::snapshot`. It is written again every `STATE_EVERY` blocks,
//! after every reorganisation and, on Unix, when the node is asked to stop, so a start replays a few blocks or none. Nothing depends on it: without the file, with a
//! damaged one, or when a block stored after it does not connect on it, the node replays its whole chain as it
//! did before the file existed.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use neb_chain::Block;
use neb_chain::state::{ChainParams, ChainState};

use crate::node::{Node, NodeError};

const STATE_FILE: &str = "state.dat";
/// Blocks between two writes of the state file.
pub(crate) const STATE_EVERY: u32 = 10;
/// One write at a time, as two would share the temporary file; the value is the order number of the state that was
/// written last, so a thread that comes late does not put an older state over a newer one.
static WRITING: Mutex<u64> = Mutex::new(0);
/// Order numbers of the states taken for a write, in the order they were taken.
static TAKEN: AtomicU64 = AtomicU64::new(0);

/// How the state of a start was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Start {
    /// Height of the state file that was used; `None` when the chain was replayed from the genesis block.
    pub loaded: Option<u32>,
    /// Blocks that were checked again.
    pub replayed: u32,
}

fn connect_from(state: &mut ChainState, chain: &[Block]) -> Result<(), NodeError> {
    for block in &chain[state.height() as usize + 1..] {
        state.connect_block(block).map_err(|e| NodeError::Connect(e.to_string()))?;
    }
    Ok(())
}

/// The state of the best chain `chain`, whose first block is the genesis block.
pub(crate) fn open_state(
    params: &ChainParams,
    chain: &[Block],
    dir: &Path,
    verify_pow: bool,
) -> Result<(ChainState, Start), NodeError> {
    // The genesis block and the parameters are checked on every path.
    let mut replayed =
        ChainState::from_genesis(params.clone(), chain[0].clone()).map_err(|e| NodeError::Connect(e.to_string()))?;
    replayed.verify_pow = verify_pow;
    if let Ok(bytes) = fs::read(dir.join(STATE_FILE))
        && let Some(height) = ChainState::snapshot_height(&bytes)
        && (height as usize) < chain.len()
        && let Ok(mut state) = ChainState::from_snapshot(params.clone(), chain[..=height as usize].to_vec(), &bytes)
    {
        state.verify_pow = verify_pow;
        // A stored block that does not connect on the state of the file: the file is left aside and the chain
        // replayed, which also tells whether the stored chain itself is at fault.
        if connect_from(&mut state, chain).is_ok() {
            let replayed = state.height() - height;
            return Ok((state, Start { loaded: Some(height), replayed }));
        }
    }
    connect_from(&mut replayed, chain)?;
    let height = replayed.height();
    Ok((replayed, Start { loaded: None, replayed: height }))
}

impl Node {
    /// How the state of this run was obtained.
    pub fn start(&self) -> Start {
        self.start
    }
}

/// Writes the state file: a temporary file that is synced, then renamed over the old one, so a crash leaves either
/// the old file or the new one. A failure is ignored: the next start replays more blocks.
pub(crate) fn write_state(dir: &Path, bytes: &[u8]) {
    write_in_order(dir, bytes, TAKEN.fetch_add(1, Ordering::SeqCst) + 1);
}

fn write_in_order(dir: &Path, bytes: &[u8], order: u64) {
    let mut last = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    if order <= *last {
        return;
    }
    *last = order;
    let tmp = dir.join("state.dat.tmp");
    let written = fs::File::create(&tmp).and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()));
    if written.is_ok() {
        let _ = fs::rename(&tmp, dir.join(STATE_FILE));
    } else {
        let _ = fs::remove_file(&tmp);
    }
}

/// Writes the state file on a thread of its own when the tip is at a height that is due, or after a
/// reorganisation (`always`). Called once the block was relayed and the miners have their job: the thread takes
/// the node lock for the time of the copy, never for the disk.
pub(crate) fn save_state(node: &Node, height: u32, always: bool) {
    if always || height.is_multiple_of(STATE_EVERY) {
        let node = node.clone();
        std::thread::spawn(move || {
            let (bytes, dir, order) = {
                let inner = node.inner.lock().expect("node lock");
                (inner.state.snapshot(), inner.store.dir().to_path_buf(), TAKEN.fetch_add(1, Ordering::SeqCst) + 1)
            };
            write_in_order(&dir, &bytes, order);
        });
    }
}

/// Set by the handler of the stop signals.
#[cfg(unix)]
static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// On Unix, a stop asked with SIGTERM or SIGINT writes the state of the tip before the process ends, so the next
/// start replays no block. The handler only sets a flag; a thread reads it, takes the node lock, which keeps any
/// later block out, writes the file and ends the process. On other systems, and for a stop that sends no signal,
/// the file of the last write is used: the next start replays fewer than `STATE_EVERY` blocks.
#[cfg(unix)]
pub(crate) fn save_on_stop(node: &Node) {
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;
    extern "C" fn on_stop(_signal: i32) {
        STOP.store(true, Ordering::SeqCst);
    }
    unsafe extern "C" {
        fn signal(signum: i32, handler: extern "C" fn(i32)) -> usize;
    }
    // SAFETY: `signal` is the C library call of that name; `on_stop` has the type of a signal handler and does
    // one atomic store, which a handler may do.
    unsafe {
        signal(SIGTERM, on_stop);
        signal(SIGINT, on_stop);
    }
    let node = node.clone();
    std::thread::spawn(move || {
        while !STOP.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let inner = node.inner.lock().unwrap_or_else(|e| e.into_inner());
        write_state(inner.store.dir(), &inner.state.snapshot());
        std::process::exit(0);
    });
}

#[cfg(not(unix))]
pub(crate) fn save_on_stop(_node: &Node) {}
