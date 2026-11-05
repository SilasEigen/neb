//! The engine selected by a miner, its per-model preparation, and the per-thread worker that applies the rules of
//! the crate (see the crate documentation).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use neb_block::Target;
use neb_pow::shares::{Model, ShareBuffers, SharedDataset, share_eval};

use crate::avx512::{self, FlatDataset, PreparedWeights, Scratch};
use crate::selftest;

/// One attempt in this many (per worker) is evaluated by the reference even when the engine reports no hit, to
/// catch an engine that returns wrong tickets without ever claiming a hit.
pub const AUDIT_INTERVAL: u64 = 4096;

/// Keys evaluated by both the engine and the reference when a model is prepared.
pub const MODEL_CHECK_KEYS: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    /// The C reference (`neb_share_eval`): the default.
    Reference,
    /// AVX-512 VNNI, every entry computed.
    Avx512Dense,
    /// AVX-512 VNNI, skipping the zero entries (the fastest on sparse inputs such as MNIST).
    Avx512Sparse,
}

impl EngineKind {
    pub fn name(self) -> &'static str {
        match self {
            EngineKind::Reference => "reference",
            EngineKind::Avx512Dense => "avx512-dense",
            EngineKind::Avx512Sparse => "avx512-sparse",
        }
    }

    /// `ref`, `reference`, `avx512` (the sparse variant), `avx512-sparse`, `avx512-dense`.
    pub fn parse(s: &str) -> Result<EngineKind, String> {
        match s {
            "ref" | "reference" => Ok(EngineKind::Reference),
            "avx512" | "avx512-sparse" => Ok(EngineKind::Avx512Sparse),
            "avx512-dense" => Ok(EngineKind::Avx512Dense),
            _ => Err(format!("unknown engine {s:?}: expected ref, avx512, avx512-sparse or avx512-dense")),
        }
    }

    fn sparse(self) -> bool {
        self == EngineKind::Avx512Sparse
    }
}

/// Counters of the checks of an engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineStats {
    /// Hits (shares and solutions) that the reference confirmed.
    pub verified_hits: u64,
    /// Attempts without a hit that the reference evaluated again.
    pub audits: u64,
    /// Tickets, digests or indices that differed from the reference.
    pub mismatches: u64,
    /// The engine was switched off.
    pub disabled: bool,
}

/// The result of one attempt, decided by the reference whenever the engine claimed a hit or the attempt was
/// audited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attempt {
    pub ticket: [u8; 32],
    /// The ticket meets the share target.
    pub share: bool,
    /// The ticket meets the block target.
    pub block: bool,
}

pub struct Engine {
    requested: EngineKind,
    kind: EngineKind,
    dataset: SharedDataset,
    flat: Option<FlatDataset>,
    disabled: AtomicBool,
    verified_hits: AtomicU64,
    audits: AtomicU64,
    mismatches: AtomicU64,
    messages: Mutex<Vec<String>>,
    #[cfg(feature = "fault-injection")]
    fault: AtomicU64,
}

impl Engine {
    /// The reference engine.
    pub fn reference(dataset: SharedDataset) -> Arc<Engine> {
        Engine::build(EngineKind::Reference, dataset, false)
    }

    /// The engine `requested`, if it can run here and passes its self-test; otherwise the reference, with the
    /// reason in [`Engine::messages`].
    pub fn new(requested: EngineKind, dataset: SharedDataset) -> Arc<Engine> {
        Engine::build(requested, dataset, false)
    }

    /// As [`Engine::new`] on a processor that lacks the instructions (for the tests of the fallback).
    #[doc(hidden)]
    pub fn new_pretending_unavailable(requested: EngineKind, dataset: SharedDataset) -> Arc<Engine> {
        Engine::build(requested, dataset, true)
    }

    fn build(requested: EngineKind, dataset: SharedDataset, pretend_unavailable: bool) -> Arc<Engine> {
        let mut engine = Engine {
            requested,
            kind: EngineKind::Reference,
            dataset,
            flat: None,
            disabled: AtomicBool::new(false),
            verified_hits: AtomicU64::new(0),
            audits: AtomicU64::new(0),
            mismatches: AtomicU64::new(0),
            messages: Mutex::new(Vec::new()),
            #[cfg(feature = "fault-injection")]
            fault: AtomicU64::new(0),
        };
        if requested != EngineKind::Reference {
            let name = requested.name();
            let unavailable =
                if pretend_unavailable { Some("pretended: no AVX-512 here") } else { avx512::unavailable_reason() };
            if let Some(why) = unavailable {
                engine.note(format!("{name} is not available: {why}; using the reference"));
            } else {
                let t0 = std::time::Instant::now();
                let outcome = FlatDataset::new(&*engine.dataset)
                    .and_then(|flat| selftest::run(requested, &*engine.dataset, &flat).map(|report| (flat, report)));
                match outcome {
                    Ok((flat, report)) => {
                        engine.note(format!(
                            "{name} self-test passed: {} tickets and {} gradients equal to the reference ({:.1} s)",
                            report.tickets,
                            report.gradients,
                            t0.elapsed().as_secs_f64()
                        ));
                        engine.flat = Some(flat);
                        engine.kind = requested;
                    }
                    Err(e) => engine.note(format!("{name} self-test FAILED: {e}; using the reference")),
                }
            }
        }
        Arc::new(engine)
    }

    fn note(&self, message: String) {
        self.messages.lock().expect("messages").push(message);
    }

    /// The engine that was asked for.
    pub fn requested(&self) -> EngineKind {
        self.requested
    }

    /// The engine in use now: the reference when the request failed or the engine was switched off.
    pub fn active(&self) -> EngineKind {
        if self.disabled.load(Ordering::Relaxed) { EngineKind::Reference } else { self.kind }
    }

    /// Messages about the self-test, the fallback and the mismatches, oldest first.
    pub fn messages(&self) -> Vec<String> {
        self.messages.lock().expect("messages").clone()
    }

    pub fn stats(&self) -> EngineStats {
        EngineStats {
            verified_hits: self.verified_hits.load(Ordering::Relaxed),
            audits: self.audits.load(Ordering::Relaxed),
            mismatches: self.mismatches.load(Ordering::Relaxed),
            disabled: self.disabled.load(Ordering::Relaxed),
        }
    }

    fn optimised(&self) -> bool {
        self.kind != EngineKind::Reference && !self.disabled.load(Ordering::Relaxed)
    }

    fn mismatch(&self, what: String) {
        self.mismatches.fetch_add(1, Ordering::Relaxed);
        if !self.disabled.swap(true, Ordering::Relaxed) {
            self.note(format!(
                "{} returned a value that differs from the reference ({what}); switched off, using the reference",
                self.kind.name()
            ));
        }
    }

    /// Prepares `model` for the engine and checks it against the reference on [`MODEL_CHECK_KEYS`] keys derived from
    /// the model. Cheap (a few milliseconds); a miner calls it once per job.
    pub fn prepare(&self, model: &Arc<Model>) -> Arc<Prepared> {
        let mut weights = None;
        if self.optimised() {
            match PreparedWeights::new(model) {
                Ok(w) => {
                    if let (Some(flat), Ok(mut scratch)) = (&self.flat, Scratch::new()) {
                        let wh = model.weights_hash();
                        for i in 0..MODEL_CHECK_KEYS {
                            let mut input = b"neb-engine-model-check".to_vec();
                            input.extend_from_slice(&wh);
                            input.extend_from_slice(&i.to_le_bytes());
                            let key = *blake3::hash(&input).as_bytes();
                            let want = share_eval(model, &*self.dataset, &key);
                            let got = scratch.eval(&w, flat, &key, self.kind.sparse());
                            if (got.digest, self.corrupt(got.ticket, u64::from(i) + 1), got.indices)
                                != (want.digest, want.ticket, want.indices)
                            {
                                self.mismatch(format!("model check, key {i}"));
                                break;
                            }
                        }
                    }
                    weights = Some(w);
                }
                Err(e) => self.mismatch(format!("cannot prepare the model: {e}")),
            }
        }
        Arc::new(Prepared { model: model.clone(), weights })
    }

    /// A worker for one thread.
    pub fn worker(self: &Arc<Self>) -> Worker {
        Worker {
            engine: self.clone(),
            buffers: ShareBuffers::new(),
            scratch: Scratch::new().ok().filter(|_| self.optimised()),
            attempts: 0,
        }
    }

    /// The ticket as the engine reports it: the fault of the tests is applied here.
    #[cfg(feature = "fault-injection")]
    fn corrupt(&self, mut ticket: [u8; 32], attempts: u64) -> [u8; 32] {
        let fault = self.fault.load(Ordering::Relaxed);
        let every = fault & !(1 << 63);
        if every != 0 && attempts.is_multiple_of(every) {
            if fault >> 63 == 1 {
                ticket[31] = 0;
            } else {
                ticket[0] ^= 1;
            }
        }
        ticket
    }

    #[cfg(not(feature = "fault-injection"))]
    fn corrupt(&self, ticket: [u8; 32], _attempts: u64) -> [u8; 32] {
        ticket
    }

    /// Makes the engine return a wrong ticket on one attempt in `every` (0 stops): bit 0 of byte 0 is flipped, or,
    /// when `claim_hit` is set, the top byte of the ticket is cleared so that the engine claims a hit.
    #[cfg(feature = "fault-injection")]
    pub fn inject_fault(&self, every: u64, claim_hit: bool) {
        self.fault.store(every | (u64::from(claim_hit) << 63), Ordering::Relaxed);
    }
}

/// A model prepared for an engine.
pub struct Prepared {
    model: Arc<Model>,
    weights: Option<PreparedWeights>,
}

impl Prepared {
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
}

/// The attempts of one thread.
pub struct Worker {
    engine: Arc<Engine>,
    buffers: ShareBuffers,
    scratch: Option<Scratch>,
    attempts: u64,
}

impl Worker {
    /// Evaluates the attempt `key` on the model of `prepared` and decides whether it meets the targets. With an
    /// optimised engine, the reference decides every attempt that the engine claims as a hit and every
    /// [`AUDIT_INTERVAL`]-th attempt; the engine alone decides the others, which it reports as no hit.
    pub fn attempt(
        &mut self,
        prepared: &Prepared,
        key: &[u8; 32],
        share_target: &Target,
        block_target: &Target,
    ) -> Attempt {
        self.attempts += 1;
        let verdict = |ticket: [u8; 32]| Attempt {
            ticket,
            share: share_target.admits(&ticket),
            block: block_target.admits(&ticket),
        };
        let engine = &*self.engine;
        if engine.optimised()
            && let (Some(scratch), Some(weights), Some(flat)) =
                (self.scratch.as_mut(), prepared.weights.as_ref(), engine.flat.as_ref())
        {
            let claimed = engine.corrupt(scratch.eval(weights, flat, key, engine.kind.sparse()).ticket, self.attempts);
            let hit = share_target.admits(&claimed) || block_target.admits(&claimed);
            let audit = self.attempts.is_multiple_of(AUDIT_INTERVAL);
            if !hit && !audit {
                return verdict(claimed);
            }
            let reference = self.buffers.ticket(&prepared.model, &*engine.dataset, key);
            if reference != claimed {
                engine.mismatch(format!(
                    "ticket of an {}",
                    if hit { "attempt claimed as a hit" } else { "audited attempt" }
                ));
            } else if hit {
                engine.verified_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                engine.audits.fetch_add(1, Ordering::Relaxed);
            }
            return verdict(reference);
        }
        verdict(self.buffers.ticket(&prepared.model, &*engine.dataset, key))
    }
}
