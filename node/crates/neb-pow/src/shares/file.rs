//! Dataset files (spec/shares.md section 11.6): one dataset (`NMD1`) or a set of datasets, one per model slot
//! (`NMDS`).

use std::sync::Arc;

use super::{Dataset, NIN, SharedDataset};

/// A dataset stored in a file (spec/shares.md section 11.6): the magic `NMD1`,
/// the sample count `N >= 1` as a little-endian u32, then `N` records of `NIN`
/// pixels (0..=7) and a label (0..=9). The network parameters pin the BLAKE3
/// hash of the file.
pub struct FileDataset {
    n: u32,
    records: Vec<u8>,
}

/// Bytes of one record of a dataset file.
pub const RECORD_BYTES: usize = NIN + 1;
const FILE_MAGIC: &[u8; 4] = b"NMD1";

impl FileDataset {
    /// Checks the format only, not the hash (see [`FileDataset::load`]): magic,
    /// `N >= 1`, an exact size of `8 + 785 * N` bytes, then every pixel and
    /// label.
    pub fn from_bytes(bytes: &[u8]) -> Result<FileDataset, String> {
        if bytes.len() < 8 || &bytes[..4] != FILE_MAGIC {
            return Err("not a dataset file (bad magic)".into());
        }
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let records = &bytes[8..];
        if n == 0 || records.len() != n as usize * RECORD_BYTES {
            return Err(format!(
                "the file has {} record bytes, {n} samples need {}",
                records.len(),
                n as usize * RECORD_BYTES
            ));
        }
        if records.chunks(RECORD_BYTES).any(|r| r[..NIN].iter().any(|&p| p > 7) || r[NIN] > 9) {
            return Err("a sample has a pixel above 7 or a label above 9".into());
        }
        Ok(FileDataset { n, records: records.to_vec() })
    }

    /// Reads a dataset file. When `pinned` is set, the BLAKE3 hash of the file
    /// must equal it; the hash is checked before the content is interpreted.
    pub fn load(path: &std::path::Path, pinned: Option<&[u8; 32]>) -> Result<FileDataset, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if let Some(want) = pinned {
            let got = blake3::hash(&bytes);
            if got.as_bytes() != want {
                return Err(format!("{} has the wrong hash {}", path.display(), got.to_hex()));
            }
        }
        FileDataset::from_bytes(&bytes)
    }
}

const SET_MAGIC: &[u8; 4] = b"NMDS";
/// Most datasets one file may hold (spec/shares.md section 11.6).
pub const MAX_DATASETS: usize = 16;

/// The datasets of a network, one per model slot (spec/shares.md section 11.6): a file of one dataset (`NMD1`) gives
/// one slot; a set file is `NMDS || LE32(K) || K dataset files back to back`, `1 <= K <= MAX_DATASETS`, and gives K
/// slots in that order. Each part is checked as [`FileDataset::from_bytes`] checks a file.
pub fn datasets_from_bytes(bytes: &[u8]) -> Result<Vec<FileDataset>, String> {
    if bytes.len() < 8 || &bytes[..4] != SET_MAGIC {
        return Ok(vec![FileDataset::from_bytes(bytes)?]);
    }
    let k = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    if k == 0 || k > MAX_DATASETS {
        return Err(format!("a dataset set holds 1 to {MAX_DATASETS} datasets, not {k}"));
    }
    let mut out = Vec::with_capacity(k);
    let mut at = 8;
    for i in 0..k {
        let head = bytes.get(at..at + 8).ok_or_else(|| format!("dataset {i} of the set is cut short"))?;
        let n = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
        let end = n.checked_mul(RECORD_BYTES).and_then(|r| r.checked_add(at + 8)).filter(|e| *e <= bytes.len());
        let end = end.ok_or_else(|| format!("dataset {i} of the set is cut short"))?;
        out.push(FileDataset::from_bytes(&bytes[at..end]).map_err(|e| format!("dataset {i} of the set: {e}"))?);
        at = end;
    }
    if at != bytes.len() {
        return Err("bytes after the last dataset of the set".into());
    }
    Ok(out)
}

/// Reads a dataset file or a dataset set file; when `pinned` is set the BLAKE3 hash of the whole file must equal it,
/// checked before the content is interpreted.
pub fn load_datasets(path: &std::path::Path, pinned: Option<&[u8; 32]>) -> Result<Vec<SharedDataset>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if let Some(want) = pinned {
        let got = blake3::hash(&bytes);
        if got.as_bytes() != want {
            return Err(format!("{} has the wrong hash {}", path.display(), got.to_hex()));
        }
    }
    Ok(datasets_from_bytes(&bytes)?.into_iter().map(|d| Arc::new(d) as SharedDataset).collect())
}

impl std::fmt::Debug for FileDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FileDataset({} samples)", self.n)
    }
}

impl Dataset for FileDataset {
    fn n(&self) -> u32 {
        self.n
    }

    /// Panics when `index >= n()`, which a share never requests.
    fn get(&self, index: u32) -> ([u8; NIN], u8) {
        let r = &self.records[index as usize * RECORD_BYTES..][..RECORD_BYTES];
        (r[..NIN].try_into().unwrap(), r[NIN])
    }
}
