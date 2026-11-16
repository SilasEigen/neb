//! The table engine of nebhash v2 (`spec/nebhash-v2.md` section 4).
//!
//! The gradient of an attempt is the sum of the gradients of its 16 samples, clipped afterwards, and the gradient of
//! a sample depends on the model and on that sample only. So for one model state the engine evaluates every sample of
//! the dataset once ([`Table::build`], with `neb2_sample_grad` of the C reference) and keeps the result; an attempt
//! then reads 16 rows, adds them, clips, and computes the digest and the ticket ([`Table::eval`]). Nothing of the
//! forward and backward passes is left in an attempt.
//!
//! A row holds, per layer in table order: for a convolution, every gradient entry (16 bits when the bound of one
//! sample fits, 32 bits otherwise); for a linear layer, its `cout` errors and its `cin` inputs, whose outer product
//! is the gradient of the layer. A row of KEP1 takes 3 714 bytes and a row of RET2 12 386 bytes, so the tables of the
//! two datasets of the network take about 50 MB and 640 MB.
//!
//! The table follows the model: it is built again for every model state, which costs one sample evaluation per
//! sample of the dataset, spread over the threads of the machine.

use neb_pow::shares::{Arch, CLIP, Dataset, Layer, Model, NB, sample_grad, share_indices};

/// How the entries of one layer are kept in a row.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Wide,
    Narrow,
    Linear,
}

#[derive(Clone, Copy, Debug)]
struct Part {
    kind: Kind,
    layer: Layer,
    /// Offset of the layer in a row.
    at: usize,
}

/// Bytes of the layer in a row.
fn part_bytes(kind: Kind, l: &Layer) -> usize {
    match kind {
        Kind::Wide => 4 * l.weights,
        Kind::Narrow => 2 * l.weights,
        Kind::Linear => l.cout + l.cin,
    }
}

fn parts(arch: Arch) -> (Vec<Part>, usize) {
    let mut at = 0;
    let parts = arch
        .layers()
        .into_iter()
        .map(|layer| {
            // one sample adds at most `positions` products of an error (at most 8) and an input (at most 8)
            let kind = match (layer.linear, layer.positions * 64 <= i16::MAX as usize) {
                (true, _) => Kind::Linear,
                (false, true) => Kind::Narrow,
                (false, false) => Kind::Wide,
            };
            let part = Part { kind, layer, at };
            at += part_bytes(kind, &layer);
            part
        })
        .collect();
    (parts, at)
}

/// The per-sample gradients of one model state on one dataset.
pub struct Table {
    arch: Arch,
    n: u32,
    parts: Vec<Part>,
    row: usize,
    rows: Vec<u8>,
    /// Whether the gradient of the sample has a non-zero entry: a sample that the model already answers exactly adds
    /// nothing to an attempt.
    live: Vec<bool>,
}

/// The result of an attempt read from a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableEval {
    pub digest: [u8; 32],
    pub ticket: [u8; 32],
    pub indices: [u32; NB],
}

impl Table {
    /// Bytes of the rows of a table of `arch` over `n` samples.
    pub fn bytes(arch: Arch, n: u32) -> usize {
        parts(arch).1 * n as usize
    }

    /// Evaluates every sample of `dataset` on `model` with `threads` threads. `spare` is a buffer to reuse (the rows
    /// of a table that is no longer needed), or an empty vector. Fails on a v1 model, on a dataset of another
    /// architecture and when a value does not fit its field, which the bounds of the arithmetic exclude.
    pub fn build(model: &Model, dataset: &dyn Dataset, threads: usize, mut spare: Vec<u8>) -> Result<Table, String> {
        let arch = model.arch();
        if arch == Arch::V1 || dataset.arch() != arch {
            return Err("the table engine needs a v2 model and a dataset of its architecture".into());
        }
        let (parts, row) = parts(arch);
        let n = dataset.n();
        spare.clear();
        spare.resize(row * n as usize, 0);
        let mut live = vec![false; n as usize];
        let chunk = (n as usize).div_ceil(threads.max(1)).max(1);
        let result: Result<(), String> = std::thread::scope(|scope| {
            let handles: Vec<_> = spare
                .chunks_mut(row * chunk)
                .zip(live.chunks_mut(chunk))
                .enumerate()
                .map(|(c, (rows, live))| {
                    let parts = &parts;
                    scope.spawn(move || fill(model, dataset, parts, row, (c * chunk) as u32, rows, live))
                })
                .collect();
            handles.into_iter().try_for_each(|h| h.join().map_err(|_| "a table thread panicked".to_string())?)
        });
        result?;
        Ok(Table { arch, n, parts, row, rows: spare, live })
    }

    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// Samples whose gradient is zero on this model.
    pub fn idle_rows(&self) -> usize {
        self.live.iter().filter(|&&l| !l).count()
    }

    /// The rows, to build the next table in the same memory.
    pub fn into_spare(self) -> Vec<u8> {
        self.rows
    }

    /// The attempt `key`: its digest, its ticket and its sample indices. `acc` is the scratch of the caller; after
    /// the call its first `arch.weights()` entries are the clipped gradient of the attempt.
    pub fn eval(&self, key: &[u8; 32], acc: &mut Vec<i32>) -> TableEval {
        let weights = self.arch.weights();
        if acc.len() < weights {
            acc.resize(weights, 0);
        }
        let acc = &mut acc[..weights];
        acc.fill(0);
        let indices = share_indices(key, self.n);
        for &i in &indices {
            if !self.live[i as usize] {
                continue;
            }
            let row = &self.rows[i as usize * self.row..][..self.row];
            for p in &self.parts {
                let out = &mut acc[p.layer.offset..][..p.layer.weights];
                let bytes = &row[p.at..][..part_bytes(p.kind, &p.layer)];
                match p.kind {
                    Kind::Wide => {
                        for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                            *o += i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        }
                    }
                    Kind::Narrow => {
                        for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                            *o += i32::from(i16::from_le_bytes([b[0], b[1]]));
                        }
                    }
                    Kind::Linear => {
                        let (d, input) = bytes.split_at(p.layer.cout);
                        for (out, &d) in out.chunks_exact_mut(p.layer.cin).zip(d) {
                            let d = i32::from(d as i8);
                            if d == 0 {
                                continue;
                            }
                            for (o, &x) in out.iter_mut().zip(input) {
                                *o += d * i32::from(x as i8);
                            }
                        }
                    }
                }
            }
        }
        // clip, then the digest of spec/shares.md section 5.3 over the layers l = 1 .. L
        let mut lanes = [0u32; 8];
        for (l, p) in self.parts.iter().enumerate() {
            let base = 7919 * (l as u32 + 1);
            let layer = &mut acc[p.layer.offset..][..p.layer.weights];
            for (j, chunk) in layer.chunks_mut(8).enumerate() {
                for (k, v) in chunk.iter_mut().enumerate() {
                    *v = (*v).clamp(-CLIP, CLIP);
                    lanes[k] = lanes[k].wrapping_add((*v as u32) ^ (base + (8 * j + k) as u32));
                }
            }
        }
        let mut input = [0u8; 64];
        for (k, lane) in lanes.iter().enumerate() {
            input[4 * k..4 * k + 4].copy_from_slice(&lane.to_le_bytes());
        }
        input[32..].copy_from_slice(key);
        let digest = input[..32].try_into().expect("32 bytes");
        TableEval { digest, ticket: *blake3::hash(&input).as_bytes(), indices }
    }
}

/// Fills the rows of the samples `first ..`, one per entry of `live`.
fn fill(
    model: &Model,
    dataset: &dyn Dataset,
    parts: &[Part],
    row: usize,
    first: u32,
    rows: &mut [u8],
    live: &mut [bool],
) -> Result<(), String> {
    let arch = model.arch();
    let mut x = vec![0u8; arch.nin()];
    let mut g = vec![0i32; arch.weights()];
    let mut fac = vec![0i8; arch.factor_bytes()];
    for (i, (out, live)) in rows.chunks_exact_mut(row).zip(live.iter_mut()).enumerate() {
        let label = dataset.sample(first + i as u32, &mut x);
        sample_grad(model, &x, label, &mut g, Some(&mut fac));
        *live = g.iter().any(|&v| v != 0);
        let mut f = 0;
        for p in parts {
            let values = &g[p.layer.offset..][..p.layer.weights];
            let bytes = &mut out[p.at..][..part_bytes(p.kind, &p.layer)];
            match p.kind {
                Kind::Wide => {
                    for (b, v) in bytes.chunks_exact_mut(4).zip(values) {
                        b.copy_from_slice(&v.to_le_bytes());
                    }
                }
                Kind::Narrow => {
                    for (b, &v) in bytes.chunks_exact_mut(2).zip(values) {
                        let v = i16::try_from(v).map_err(|_| "a gradient entry does not fit 16 bits".to_string())?;
                        b.copy_from_slice(&v.to_le_bytes());
                    }
                }
                Kind::Linear => {
                    let len = p.layer.cout + p.layer.cin;
                    for (b, &v) in bytes.iter_mut().zip(&fac[f..f + len]) {
                        *b = v as u8;
                    }
                    f += len;
                }
            }
        }
    }
    Ok(())
}
