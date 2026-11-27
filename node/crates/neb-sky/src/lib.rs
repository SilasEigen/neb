//! The sky search job, version 1 (`spec/sky-search.md`): a box search for a repeating dip in light curves.
//!
//! A light curve is a row of hourly slots, each an `i8`: the flux in sixteenths of its scatter, -128 for no data.
//! Every step is integer arithmetic, so every implementation returns the same bytes for the same batch.

/// Time units in one slot (one hour).
pub const UNIT: i64 = 4096;
/// Time units in one phase bin (a quarter of an hour).
pub const BIN: i64 = 1024;
/// Box widths in phase bins: 1, 1.5, 2, 3, 4 and 6 hours.
pub const WIDTHS: [usize; 6] = [4, 6, 8, 12, 16, 24];
/// Shortest trial period: half a day.
pub const P_MIN: i64 = 12 * UNIT;
/// Longest trial period: nine days.
pub const P_MAX: i64 = 216 * UNIT;
/// Consecutive trial periods that compete for one entry of the result.
pub const GROUP: usize = 64;
/// Entries of the result per light curve.
pub const KEEP: usize = 32;
/// Fewest slots inside a box, and outside it.
pub const MIN_IN: i64 = 3;
/// Longest light curve; keeps every product below 2^63.
pub const MAX_SLOTS: usize = 2048;
/// The slot value that means no data.
pub const NO_DATA: i8 = -128;
/// Bytes of one entry in a result file.
pub const ENTRY_BYTES: usize = 16;

pub mod client;

const VERSION: u16 = 1;

/// One box: its score, its period in time units, its first phase bin and the index of its width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    /// Section 5 of the specification; zero when the entry holds no box.
    pub score: u64,
    /// The trial period, in time units.
    pub period: u32,
    /// The first phase bin of the box.
    pub start: u16,
    /// The index of the box width in [`WIDTHS`].
    pub width: u8,
}

/// Why a batch file was refused.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchError {
    /// The magic or the version is not that of a version 1 batch.
    Header,
    /// A light curve is longer than [`MAX_SLOTS`], runs past the end of the file, or bytes follow the last one.
    Length,
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BatchError::Header => "not a sky search batch of version 1",
            BatchError::Length => "a light curve length does not fit the file",
        })
    }
}

impl std::error::Error for BatchError {}

/// The trial periods in time units: each is the one before plus its 8192nd part, rounded down.
#[must_use]
pub fn periods() -> Vec<i64> {
    let mut out = Vec::with_capacity(24_400);
    let mut p = P_MIN;
    while p <= P_MAX {
        out.push(p);
        p += p >> 13;
    }
    out
}

/// The best box of one period: ties go to the lower width, then to the lower start.
///
/// The casts cannot truncate: a period is at most `P_MAX`, a start is below 865 phase bins, a width index below 6,
/// and with at most `MAX_SLOTS` slots `num * num` stays below 2^63.
fn best_box(slots: &[(i64, i64)], total: i64, period: i64, sums: &mut [i64], counts: &mut [i64]) -> Entry {
    let m = ((period + BIN - 1) / BIN) as usize;
    let n = slots.len() as i64;
    sums[..m].fill(0);
    counts[..m].fill(0);
    for &(at, value) in slots {
        let b = (at % period / BIN) as usize;
        sums[b] += value;
        counts[b] += 1;
    }
    let mut best = Entry { period: period as u32, ..Entry::default() };
    for (qi, &q) in WIDTHS.iter().enumerate() {
        let (mut s, mut r) = (0i64, 0i64);
        for j in 0..q {
            s += sums[j];
            r += counts[j];
        }
        for a in 0..m {
            let num = r * total - n * s; // positive when the box is below the rest
            if num > 0 && r >= MIN_IN && n - r >= MIN_IN {
                let score = (num * num / (r * (n - r))) as u64;
                if score > best.score {
                    best = Entry { score, period: period as u32, start: a as u16, width: qi as u8 };
                }
            }
            let (leave, enter) = (a, (a + q) % m);
            s += sums[enter] - sums[leave];
            r += counts[enter] - counts[leave];
        }
    }
    best
}

/// The `KEEP` entries of one light curve, best first; entries without a box are zero.
///
/// # Panics
/// When the light curve is longer than `MAX_SLOTS`.
#[must_use]
pub fn search(x: &[i8]) -> [Entry; KEEP] {
    assert!(x.len() <= MAX_SLOTS, "light curve longer than MAX_SLOTS");
    let slots: Vec<(i64, i64)> = x
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v != NO_DATA)
        .map(|(i, &v)| (i as i64 * UNIT + UNIT / 2, i64::from(v)))
        .collect();
    let total: i64 = slots.iter().map(|s| s.1).sum();
    let mut out = [Entry::default(); KEEP];
    if (slots.len() as i64) < 2 * MIN_IN {
        return out;
    }
    let bins = ((P_MAX + BIN - 1) / BIN) as usize;
    let (mut sums, mut counts) = (vec![0i64; bins], vec![0i64; bins]);
    let mut winners: Vec<Entry> = periods()
        .chunks(GROUP)
        .map(|group| {
            let mut top = Entry::default();
            for &p in group {
                let e = best_box(&slots, total, p, &mut sums, &mut counts);
                if e.score > top.score {
                    top = e; // ties go to the shorter period
                }
            }
            top
        })
        .collect();
    winners.sort_by_key(|w| std::cmp::Reverse(w.score)); // stable: ties go to the group of shorter periods
    for (o, w) in out.iter_mut().zip(winners) {
        if w.score > 0 {
            *o = w;
        }
    }
    out
}

/// The light curves of a batch file.
///
/// # Errors
/// When the header is not a version 1 batch or a length does not fit the file.
pub fn read_batch(raw: &[u8]) -> Result<Vec<&[u8]>, BatchError> {
    if raw.len() < 12 || &raw[..4] != b"NEBS" || raw[4..6] != VERSION.to_le_bytes() {
        return Err(BatchError::Header);
    }
    let count = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let mut curves = Vec::new();
    let mut at = 12;
    for _ in 0..count {
        let head = raw.get(at..at + 2).ok_or(BatchError::Length)?;
        let n = usize::from(u16::from_le_bytes([head[0], head[1]]));
        if n > MAX_SLOTS {
            return Err(BatchError::Length);
        }
        curves.push(raw.get(at + 2..at + 2 + n).ok_or(BatchError::Length)?);
        at += 2 + n;
    }
    if at != raw.len() {
        return Err(BatchError::Length);
    }
    Ok(curves)
}

/// The result file of a batch file.
///
/// # Errors
/// When the batch file is refused.
pub fn run(raw: &[u8]) -> Result<Vec<u8>, BatchError> {
    run_threads(raw, 1)
}

/// The result file of a batch file, the light curves shared between `threads` threads.
///
/// # Errors
/// When the batch file is refused.
pub fn run_threads(raw: &[u8], threads: usize) -> Result<Vec<u8>, BatchError> {
    let curves = read_batch(raw)?;
    let mut found = vec![[Entry::default(); KEEP]; curves.len()];
    let per = curves.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (some, out) in curves.chunks(per).zip(found.chunks_mut(per)) {
            scope.spawn(move || {
                for (c, o) in some.iter().zip(out) {
                    let x: Vec<i8> = c.iter().map(|&b| b as i8).collect();
                    *o = search(&x);
                }
            });
        }
    });
    let mut out = Vec::with_capacity(44 + curves.len() * KEEP * ENTRY_BYTES);
    out.extend_from_slice(b"NEBR");
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(blake3::hash(raw).as_bytes());
    out.extend_from_slice(&(curves.len() as u32).to_le_bytes());
    for e in found.iter().flatten() {
        out.extend_from_slice(&e.score.to_le_bytes());
        out.extend_from_slice(&e.period.to_le_bytes());
        out.extend_from_slice(&e.start.to_le_bytes());
        out.extend_from_slice(&[e.width, 0]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// The shared vectors: every light curve gives the entries of the reference, and the batch gives its result.
    #[test]
    fn vectors() {
        let text = include_str!("../../../../vectors/sky-search.txt");
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("neb-sky-search-vectors 1"));
        let (mut batch, mut curves) = (Vec::new(), 0u32);
        let mut entries = Vec::new();
        for line in lines {
            let f: Vec<&str> = line.split(' ').collect();
            match f[0] {
                "curve" => {
                    let x = unhex(f.get(1).copied().unwrap_or(""));
                    batch.extend_from_slice(&(x.len() as u16).to_le_bytes());
                    batch.extend_from_slice(&x);
                    entries.extend_from_slice(&unhex(f[2]));
                    curves += 1;
                }
                "batch" => {
                    let mut raw = b"NEBS\x01\0\0\0".to_vec();
                    raw.extend_from_slice(&curves.to_le_bytes());
                    raw.extend_from_slice(&batch);
                    let result = run(&raw).unwrap();
                    assert_eq!(blake3::hash(&raw).to_hex().as_str(), f[1]);
                    assert_eq!(&result[44..], &entries[..], "entries differ from the reference");
                    assert_eq!(blake3::hash(&result).to_hex().as_str(), f[2]);
                }
                other => panic!("unknown line {other}"),
            }
        }
        assert!(curves > 0);
    }

    #[test]
    fn refuses_bad_batches() {
        assert_eq!(read_batch(b"NEBS"), Err(BatchError::Header));
        assert_eq!(read_batch(b"NEBS\x02\0\0\0\0\0\0\0"), Err(BatchError::Header));
        assert_eq!(read_batch(b"NEBS\x01\0\0\0\x01\0\0\0\x05\0ab"), Err(BatchError::Length));
        assert_eq!(read_batch(b"NEBS\x01\0\0\0\0\0\0\0x"), Err(BatchError::Length));
        assert_eq!(read_batch(b"NEBS\x01\0\0\0\0\0\0\0").map(|c| c.len()), Ok(0));
    }
}
