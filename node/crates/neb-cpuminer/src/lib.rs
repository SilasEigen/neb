//! CPU miner client of the work service (spec/node.md section 7.1,
//! spec/shares.md section 11.5).
//!
//! The node sends a model frame, then a job, and repeats both whenever the best
//! block changes. An attempt is one nonce of the reference header of the job
//! (all-zero extranonce, `time_start`, nonce in bytes 76..80 little-endian):
//! `key = BLAKE3(header)` and `ticket = ticket(model, dataset, key)`. A ticket
//! that meets the share target is reported with a share frame; one that meets
//! the block target is also submitted. Because the share target is easier, a
//! solution is always reported as a share first, so the next block lists it.
//!
//! The miner never rolls the extranonce or the time: every share must belong to
//! the one reference header, or the round would be lost (spec/shares.md section
//! 11.5). One reader thread parses frames; worker threads take chunks of the
//! 2^32 nonces from a shared counter, so no nonce is tried twice per job. A
//! worker abandons a job as soon as a newer one is installed.
//!
//! The session (connection, frames, current job, counters) is separate from the
//! workers: [`start_with`] gives a [`Session`] to the caller, who supplies the
//! workers. `neb-miner-gpu` uses it to mine with CUDA devices. The session also
//! keeps the blocks that the miner found, for the [`dashboard`].
//!
//! The attempts are evaluated by an [`Engine`] (`neb-engines`). The default is the C reference, as in the
//! node. An optimised engine (AVX-512) is used only when the caller asks for it by name and it passes its
//! self-test; the reference then re-evaluates every share and solution it finds before it is reported.

use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use neb_block::Target;
use neb_engines::{Engine, Prepared};
use neb_pow::shares::{Model, SharedDataset};
use neb_work::frames::{self, FrameOut, Hello, Job, MSG_JOB, MSG_MODEL, MSG_RESULT, ResultMsg, ShareMsg, Submit};

/// Nonces a thread takes from the shared counter at a time. About 4 ms of
/// work at 0.5 ms per attempt: small enough to leave a stale job quickly,
/// large enough that the atomic counter is not contended.
const CHUNK: u64 = 8;

pub mod dashboard;
mod term;

/// A block that this miner submitted.
#[derive(Clone, Debug)]
pub struct FoundBlock {
    pub height: u32,
    pub nonce: u32,
    pub id: [u8; 32],
    /// Unix time of the submission.
    pub found_at: u64,
    /// Seconds from the arrival of the job to the solution.
    pub seconds: f64,
    /// The node accepted the block.
    pub accepted: bool,
}

/// Counters shown by the status line.
#[derive(Default)]
pub struct Stats {
    pub attempts: AtomicU64,
    pub shares: AtomicU64,
    pub blocks: AtomicU64,
    pub accepted: AtomicU64,
    pub rejected: AtomicU64,
    pub jobs: AtomicU64,
}

/// A job with the model it is mined on and its reference header. `generation`
/// numbers the installed jobs; a worker compares it with `Shared::generation`
/// to detect a newer job.
pub struct Unit {
    generation: u64,
    pub job: Job,
    pub model: Arc<Model>,
    /// The reference header with the nonce field set to 0; the nonce is
    /// bytes 76..80, little-endian.
    pub header: [u8; 80],
    extranonce: Vec<u8>,
    pub target: Target,
    pub share_target: Target,
    next_nonce: AtomicU64,
    /// A solution was submitted; the workers stop until the node sends the
    /// next job (also after a rejection).
    found: AtomicBool,
    installed: Instant,
    /// The model prepared for the engine, made by the first worker that needs it.
    prepared: OnceLock<Arc<Prepared>>,
}

impl Unit {
    /// The model of the job prepared for `engine` (and checked against the reference when the engine is
    /// optimised). Made once per job.
    pub fn prepared(&self, engine: &Engine) -> &Arc<Prepared> {
        self.prepared.get_or_init(|| engine.prepare(&self.model))
    }

    /// Takes up to `count` nonces that no worker has tried on this job:
    /// `(first, length)`. `None` when a solution was submitted or the 2^32
    /// nonces are used up.
    pub fn take_nonces(&self, count: u32) -> Option<(u32, u32)> {
        if self.found.load(Ordering::Relaxed) {
            return None;
        }
        // The counter is u64 so that the nonce space ends without wrapping.
        let start = self.next_nonce.fetch_add(count as u64, Ordering::Relaxed);
        if start > u32::MAX as u64 {
            return None;
        }
        let end = (start + count as u64).min(u32::MAX as u64 + 1);
        Some((start as u32, (end - start) as u32))
    }
}

struct Shared {
    current: Mutex<Option<Arc<Unit>>>,
    generation: AtomicU64,
    stop: AtomicBool,
    writer: Mutex<TcpStream>,
    stats: Arc<Stats>,
    /// Blocks submitted by this miner, oldest first.
    blocks: Mutex<Vec<FoundBlock>>,
    /// Height of the tip of the last job plus one; 0 before the first job.
    tip: AtomicU64,
    bits: AtomicU32,
}

/// The connection to the work service as seen by a worker.
#[derive(Clone)]
pub struct Session {
    shared: Arc<Shared>,
}

impl Session {
    /// The job to mine, once the node has sent a model and a job.
    pub fn current(&self) -> Option<Arc<Unit>> {
        self.shared.current.lock().expect("current job").clone()
    }

    /// True when the connection ended or the miner is stopping.
    pub fn stopped(&self) -> bool {
        self.shared.stop.load(Ordering::Relaxed)
    }

    /// True when the node has sent a job newer than `unit`.
    pub fn superseded(&self, unit: &Unit) -> bool {
        self.shared.generation.load(Ordering::Relaxed) != unit.generation
    }

    pub fn stats(&self) -> &Arc<Stats> {
        &self.shared.stats
    }

    /// The blocks this miner submitted, oldest first. A block that the node refused is removed.
    pub fn found_blocks(&self) -> Vec<FoundBlock> {
        self.shared.blocks.lock().expect("blocks").clone()
    }

    /// Height of the tip that the last job builds on.
    pub fn tip_height(&self) -> Option<u32> {
        self.shared.tip.load(Ordering::Relaxed).checked_sub(1).map(|h| h as u32)
    }

    /// Compact target of the last job.
    pub fn bits(&self) -> Option<u32> {
        self.tip_height().map(|_| self.shared.bits.load(Ordering::Relaxed))
    }

    /// Reports a nonce whose ticket meets the share target of `unit`.
    pub fn report_share(&self, unit: &Unit, nonce: u32) {
        self.shared.stats.shares.fetch_add(1, Ordering::Relaxed);
        send(&self.shared, &ShareMsg { job_id: unit.job.id, nonce }.encode());
    }

    /// Submits a nonce whose ticket meets the block target of `unit`. Only the
    /// first solution of a job is submitted; the workers stop on this job.
    pub fn report_solution(&self, unit: &Unit, nonce: u32) {
        if unit.found.swap(true, Ordering::Relaxed) {
            return;
        }
        self.shared.stats.blocks.fetch_add(1, Ordering::Relaxed);
        let id = unit.job.header(&unit.extranonce, unit.job.time_start, nonce).id();
        let found_at = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        self.shared.blocks.lock().expect("blocks").push(FoundBlock {
            height: unit.job.height,
            nonce,
            id,
            found_at,
            seconds: unit.installed.elapsed().as_secs_f64(),
            accepted: false,
        });
        let submit = Submit { id: unit.job.id, extranonce: unit.extranonce.clone(), time: unit.job.time_start, nonce };
        send(&self.shared, &submit.encode());
    }
}

pub struct MinerHandle {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl MinerHandle {
    pub fn stats(&self) -> &Arc<Stats> {
        &self.shared.stats
    }

    pub fn session(&self) -> Session {
        Session { shared: self.shared.clone() }
    }

    /// True once the connection to the node has ended.
    pub fn stopped(&self) -> bool {
        self.shared.stop.load(Ordering::Relaxed)
    }

    pub fn stop(self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        let _ = self.shared.writer.lock().expect("writer").shutdown(std::net::Shutdown::Both);
        for t in self.threads {
            let _ = t.join();
        }
    }
}

/// Connects to the work service, sends the hello frame, and starts the reader
/// and `threads` CPU workers (at least one). The miner stops when the
/// connection closes or on a malformed frame; see `MinerHandle::stopped`.
pub fn start(node: impl ToSocketAddrs, dataset: SharedDataset, threads: usize, name: &str) -> io::Result<MinerHandle> {
    start_with_engine(node, Engine::reference(dataset), threads, name)
}

/// As [`start`], with the attempts evaluated by `engine` (see [`Engine::new`]).
pub fn start_with_engine(
    node: impl ToSocketAddrs,
    engine: Arc<Engine>,
    threads: usize,
    name: &str,
) -> io::Result<MinerHandle> {
    start_with(node, name, |session| {
        (0..threads.max(1))
            .map(|_| {
                let (session, engine) = (session.clone(), engine.clone());
                thread::spawn(move || work(&session, &engine))
            })
            .collect()
    })
}

/// Connects to the work service, sends the hello frame, starts the reader and
/// the worker threads that `spawn` returns for the session.
pub fn start_with(
    node: impl ToSocketAddrs,
    name: &str,
    spawn: impl FnOnce(&Session) -> Vec<JoinHandle<()>>,
) -> io::Result<MinerHandle> {
    let mut stream = TcpStream::connect(node)?;
    stream.set_nodelay(true)?;
    stream.write_all(&frame_hello(name))?;
    let reader = stream.try_clone()?;
    let shared = Arc::new(Shared {
        current: Mutex::new(None),
        generation: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        writer: Mutex::new(stream),
        stats: Arc::new(Stats::default()),
        blocks: Mutex::new(Vec::new()),
        tip: AtomicU64::new(0),
        bits: AtomicU32::new(0),
    });
    let mut handles = Vec::new();
    {
        let shared = shared.clone();
        handles.push(thread::spawn(move || {
            read_frames(reader, &shared);
            shared.stop.store(true, Ordering::Relaxed);
        }));
    }
    handles.extend(spawn(&Session { shared: shared.clone() }));
    Ok(MinerHandle { shared, threads: handles })
}

fn frame_hello(name: &str) -> Vec<u8> {
    frames::frame(frames::MSG_HELLO, &Hello { version: 1, name: name.to_string() }.encode())
}

/// Reader loop: a job is installed only after a model frame was received, and
/// is mined on the latest model. A bad frame, a model that does not decode, or
/// a job without a model ends the session.
fn read_frames(mut stream: TcpStream, shared: &Shared) {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 65536];
    let mut model: Option<Arc<Model>> = None;
    loop {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
        loop {
            match frames::parse_frame(&buf) {
                Ok(FrameOut::Need(_)) => break,
                Ok(FrameOut::Message { msg_type, payload, consumed }) => {
                    match msg_type {
                        MSG_MODEL => match frames::decode_model(payload) {
                            Ok(m) => model = Some(Arc::new(m)),
                            Err(_) => return,
                        },
                        MSG_JOB => {
                            let (Ok(job), Some(model)) = (Job::decode(payload), model.clone()) else { return };
                            install(shared, job, model);
                        }
                        MSG_RESULT => {
                            if let Ok(result) = ResultMsg::decode(payload) {
                                let counter =
                                    if result.accepted { &shared.stats.accepted } else { &shared.stats.rejected };
                                counter.fetch_add(1, Ordering::Relaxed);
                                answer_latest(shared, result.accepted);
                            }
                        }
                        _ => {}
                    }
                    buf.drain(..consumed);
                }
                Err(_) => return,
            }
        }
    }
}

/// A result frame answers the latest submitted block that has no answer yet; a miner submits one solution per job.
fn answer_latest(shared: &Shared, accepted: bool) {
    let mut blocks = shared.blocks.lock().expect("blocks");
    if let Some(i) = blocks.iter().rposition(|b| !b.accepted) {
        if accepted {
            blocks[i].accepted = true;
        } else {
            blocks.remove(i);
        }
    }
}

/// Makes `job` current. Workers notice the new generation between attempts.
fn install(shared: &Shared, job: Job, model: Arc<Model>) {
    let extranonce = vec![0u8; job.extranonce_size as usize];
    let header = job.header(&extranonce, job.time_start, 0).encode();
    let generation = shared.generation.fetch_add(1, Ordering::SeqCst) + 1;
    shared.bits.store(job.bits, Ordering::Relaxed);
    shared.tip.store(job.height as u64, Ordering::Relaxed);
    let unit = Unit {
        generation,
        target: Target(job.target),
        share_target: Target(job.share_target),
        job,
        model,
        header,
        extranonce,
        next_nonce: AtomicU64::new(0),
        found: AtomicBool::new(false),
        installed: Instant::now(),
        prepared: OnceLock::new(),
    };
    *shared.current.lock().expect("current job") = Some(Arc::new(unit));
    shared.stats.jobs.fetch_add(1, Ordering::Relaxed);
}

fn send(shared: &Shared, frame: &[u8]) {
    let _ = shared.writer.lock().expect("writer").write_all(frame);
}

/// CPU worker loop. The job is shared; each worker owns its scratch buffers. The engine decides every hit with the
/// reference (see [`neb_engines::Worker::attempt`]).
fn work(session: &Session, engine: &Arc<Engine>) {
    let mut worker = engine.worker();
    while !session.stopped() {
        let Some(unit) = session.current() else {
            thread::sleep(Duration::from_millis(20));
            continue;
        };
        let Some((start, len)) = unit.take_nonces(CHUNK as u32) else {
            thread::sleep(Duration::from_millis(20));
            continue;
        };
        let prepared = unit.prepared(engine).clone();
        let mut header = unit.header;
        for nonce in (start as u64..start as u64 + len as u64).map(|n| n as u32) {
            if session.superseded(&unit) {
                break;
            }
            header[76..80].copy_from_slice(&nonce.to_le_bytes());
            let key = *blake3::hash(&header).as_bytes();
            let attempt = worker.attempt(&prepared, &key, &unit.share_target, &unit.target);
            session.stats().attempts.fetch_add(1, Ordering::Relaxed);
            if attempt.share {
                session.report_share(&unit, nonce);
            }
            if attempt.block {
                session.report_solution(&unit, nonce);
            }
        }
    }
}
