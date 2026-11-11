//! Dataset files (spec/shares.md section 11.6, spec/nebhash-v2.md section 2): one dataset (`NMD1` or `NMD2`) or a set
//! of datasets (`NMDS`).

use std::sync::Arc;

use super::{Arch, Dataset, NIN, SharedDataset};

/// A dataset stored in a file. `NMD1` (spec/shares.md section 11.6): the magic, the sample count `N >= 1` as a
/// little-endian u32, then `N` records of `NIN` pixels (0..=7) and a label (0..=9), for the v1 network. `NMD2`
/// (spec/nebhash-v2.md section 2): the magic, the architecture id and `N` as little-endian u32, then `N` records of the
/// architecture's inputs (0..=7) and a label below its class count. The network parameters pin the BLAKE3 hash of the
/// file.
pub struct FileDataset {
    n: u32,
    arch: Arch,
    records: Vec<u8>,
}

/// Bytes of one record of a v1 dataset file.
pub const RECORD_BYTES: usize = NIN + 1;
const FILE_MAGIC: &[u8; 4] = b"NMD1";
const FILE_MAGIC_V2: &[u8; 4] = b"NMD2";

/// `(architecture, header bytes)` of a dataset that starts at `bytes`, or an error for an unknown magic or id.
fn header(bytes: &[u8]) -> Result<(Arch, usize), String> {
    match bytes.get(..4) {
        Some(m) if m == FILE_MAGIC => Ok((Arch::V1, 8)),
        Some(m) if m == FILE_MAGIC_V2 => {
            let id = bytes.get(4..8).ok_or("a NMD2 header is cut short")?;
            let id = u32::from_le_bytes(id.try_into().unwrap());
            Ok((Arch::from_id(id).ok_or_else(|| format!("unknown architecture id {id}"))?, 12))
        }
        _ => Err("not a dataset file (bad magic)".into()),
    }
}

impl FileDataset {
    /// Checks the format only, not the hash (see [`FileDataset::load`]): magic,
    /// `N >= 1`, an exact size, then every value and label.
    pub fn from_bytes(bytes: &[u8]) -> Result<FileDataset, String> {
        let (arch, head) = header(bytes)?;
        let n = bytes.get(head - 4..head).ok_or("a dataset header is cut short")?;
        let n = u32::from_le_bytes(n.try_into().unwrap());
        let records = &bytes[head..];
        let (nin, rec) = (arch.nin(), arch.nin() + 1);
        if n == 0 || records.len() != n as usize * rec {
            return Err(format!("the file has {} record bytes, {n} samples need {}", records.len(), n as usize * rec));
        }
        if records.chunks(rec).any(|r| r[..nin].iter().any(|&p| p > 7) || r[nin] as usize >= arch.nout()) {
            return Err(format!("a sample has a value above 7 or a label above {}", arch.nout() - 1));
        }
        Ok(FileDataset { n, arch, records: records.to_vec() })
    }

    /// Bytes of a dataset of `arch` with `n` samples, header included.
    fn size(arch: Arch, head: usize, n: usize) -> Option<usize> {
        n.checked_mul(arch.nin() + 1)?.checked_add(head)
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
        let (arch, head) = header(&bytes[at..]).map_err(|e| format!("dataset {i} of the set: {e}"))?;
        let nb = bytes.get(at + head - 4..at + head).ok_or_else(|| format!("dataset {i} of the set is cut short"))?;
        let n = u32::from_le_bytes(nb.try_into().unwrap()) as usize;
        let end = FileDataset::size(arch, head, n).and_then(|s| s.checked_add(at)).filter(|e| *e <= bytes.len());
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
        write!(f, "FileDataset({} samples, {})", self.n, self.arch.name())
    }
}

impl Dataset for FileDataset {
    fn n(&self) -> u32 {
        self.n
    }

    fn arch(&self) -> Arch {
        self.arch
    }

    /// Panics when `index >= n()`, which a share never requests.
    fn get(&self, index: u32) -> ([u8; NIN], u8) {
        let nin = self.arch.nin();
        let r = &self.records[index as usize * (nin + 1)..][..nin + 1];
        (r[..NIN].try_into().unwrap(), r[nin])
    }

    /// Panics when `index >= n()` or `x` is shorter than the architecture's inputs.
    fn sample(&self, index: u32, x: &mut [u8]) -> u8 {
        let nin = self.arch.nin();
        let r = &self.records[index as usize * (nin + 1)..][..nin + 1];
        x[..nin].copy_from_slice(&r[..nin]);
        r[nin]
    }
}
