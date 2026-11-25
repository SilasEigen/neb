//! Safe Rust wrapper for `nebhash v3` (`spec/nebhash-v3.md`, `pow/ref/nebhash3.c`):
//! frozen quantized INT8 base backbones with on-chain integer LoRA adapters and key-derived deterministic dropout.

use std::ffi::{c_int, c_void};
use std::ptr;

/// Batch size per `nebhash v3` share (`NEB3_NB`), aligned to `sm_80+` m16 INT8 Tensor Core tiles.
pub const NB3: usize = 32;

/// Architecture profiles defined in `spec/nebhash-v3.md` section 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum V3Arch {
    /// `D = 64, R = 8, L = 2` deterministic test & CI profile.
    LoraTiny,
    /// `D = 4096, R = 8, L = 4` Apache-2.0 Qwen3-8B INT8 frozen base profile (~132 KB packed LoRA weights).
    Qwen3_8bLora,
}

impl V3Arch {
    pub fn id(self) -> u32 {
        match self {
            V3Arch::LoraTiny => 1,
            V3Arch::Qwen3_8bLora => 2,
        }
    }

    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            1 => Some(V3Arch::LoraTiny),
            2 => Some(V3Arch::Qwen3_8bLora),
            _ => None,
        }
    }

    pub fn d(self) -> usize {
        match self {
            V3Arch::LoraTiny => 64,
            V3Arch::Qwen3_8bLora => 4096,
        }
    }

    pub fn rank(self) -> usize {
        8
    }

    pub fn nblocks(self) -> usize {
        match self {
            V3Arch::LoraTiny => 2,
            V3Arch::Qwen3_8bLora => 4,
        }
    }

    /// Number of trainable LoRA + head weights.
    pub fn trainable_weights(self) -> usize {
        self.nblocks() * 2 * self.d() * self.rank() + 2 * self.d()
    }

    /// Packed bytes of trainable weights (4-bit nibbles, 2 weights per byte).
    pub fn packed_trainable_bytes(self) -> usize {
        self.trainable_weights() / 2
    }

    /// Number of frozen INT8 base projection weights across the adapted blocks.
    pub fn base_weights(self) -> usize {
        self.nblocks() * self.d() * self.d()
    }
}

#[repr(C)]
struct Neb3ModelRaw {
    arch: i32,
    nweights: usize,
    w: *mut i8,
    r: *mut i32,
}

#[repr(C)]
struct Neb3BaseRaw {
    arch: i32,
    nweights: usize,
    w_base: *const i8,
}

#[repr(C)]
struct Neb3DatasetRaw {
    n: u32,
    get: extern "C" fn(*mut c_void, u32, *mut u8) -> c_int,
    ctx: *mut c_void,
}

unsafe extern "C" {
    fn neb3_model_init(arch: c_int, seed: *const u8, m: *mut Neb3ModelRaw) -> c_int;
    fn neb3_model_free(m: *mut Neb3ModelRaw);
    fn neb3_base_hash(base: *const Neb3BaseRaw, out_hash: *mut u8) -> c_int;
    fn neb3_share_eval(
        m: *const Neb3ModelRaw,
        base: *const Neb3BaseRaw,
        ds: *const Neb3DatasetRaw,
        key: *const u8,
        g: *mut i32,
        digest: *mut u8,
        ticket: *mut u8,
        idx_out: *mut u32,
        nclip: *mut u32,
    ) -> c_int;
    fn neb3_model_apply(m: *mut Neb3ModelRaw, gs: *const i32);
    fn neb3_model_hashes(m: *const Neb3ModelRaw, weights_hash: *mut u8, residual_hash: *mut u8) -> c_int;
}

/// Owned frozen INT8 base weights for a `V3Arch`.
pub struct V3Base {
    arch: V3Arch,
    weights: Vec<i8>,
}

impl V3Base {
    pub fn new(arch: V3Arch, weights: Vec<i8>) -> Result<Self, String> {
        if weights.len() != arch.base_weights() {
            return Err(format!(
                "expected {} base weights, got {}",
                arch.base_weights(),
                weights.len()
            ));
        }
        Ok(Self { arch, weights })
    }

    pub fn hash(&self) -> Result<[u8; 32], String> {
        let raw = Neb3BaseRaw {
            arch: self.arch.id() as i32,
            nweights: self.weights.len(),
            w_base: self.weights.as_ptr(),
        };
        let mut out = [0u8; 32];
        let rc = unsafe { neb3_base_hash(&raw, out.as_mut_ptr()) };
        if rc != 0 {
            return Err("neb3_base_hash failed".into());
        }
        Ok(out)
    }
}

/// Owned trainable LoRA + head model state (`neb3_model`).
pub struct V3Model {
    arch: V3Arch,
    raw: Neb3ModelRaw,
}

// SAFETY: `V3Model` owns its C heap buffers exclusively and does not alias shared mutable state.
unsafe impl Send for V3Model {}
unsafe impl Sync for V3Model {}

impl Drop for V3Model {
    fn drop(&mut self) {
        unsafe { neb3_model_free(&mut self.raw) };
    }
}

/// Result of evaluating one `nebhash v3` share.
pub struct V3ShareOutput {
    pub gradients: Vec<i32>,
    pub digest: [u8; 32],
    pub ticket: [u8; 32],
    pub indices: [u32; NB3],
    pub clipped: u32,
}

/// Parsed `NMD3` dataset header and sample records (`spec/nebhash-v3.md` section 2).
pub struct V3Dataset {
    pub arch: V3Arch,
    pub base_hash: [u8; 32],
    pub samples: Vec<Vec<u8>>,
    pub labels: Vec<u8>,
}

impl V3Dataset {
    /// Parse a binary `NMD3` dataset buffer.
    pub fn parse_nmd3(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 44 {
            return Err("NMD3 buffer too short".into());
        }
        if &bytes[0..4] != b"NMD3" {
            return Err("invalid NMD3 magic".into());
        }
        let arch_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let arch = V3Arch::from_id(arch_id).ok_or_else(|| format!("unknown NMD3 arch id {arch_id}"))?;
        let mut base_hash = [0u8; 32];
        base_hash.copy_from_slice(&bytes[8..40]);
        let n = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
        if n == 0 {
            return Err("NMD3 dataset must have at least 1 sample".into());
        }
        let rec_len = arch.d() + 1;
        let expected = 44usize
            .checked_add(n.checked_mul(rec_len).ok_or("NMD3 size overflow")?)
            .ok_or("NMD3 size overflow")?;
        if bytes.len() != expected {
            return Err(format!("NMD3 length mismatch: expected {expected}, got {}", bytes.len()));
        }
        let mut samples = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        let mut off = 44;
        for _ in 0..n {
            let feat = &bytes[off..off + arch.d()];
            if feat.iter().any(|&v| v > 7) {
                return Err("NMD3 feature value > 7".into());
            }
            let label = bytes[off + arch.d()];
            if label > 1 {
                return Err("NMD3 label > 1".into());
            }
            samples.push(feat.to_vec());
            labels.push(label);
            off += rec_len;
        }
        Ok(Self {
            arch,
            base_hash,
            samples,
            labels,
        })
    }
}

extern "C" fn v3_ds_get(ctx: *mut c_void, index: u32, x_out: *mut u8) -> c_int {
    let ds = unsafe { &*(ctx as *const V3Dataset) };
    let i = index as usize;
    let feat = &ds.samples[i];
    unsafe {
        ptr::copy_nonoverlapping(feat.as_ptr(), x_out, feat.len());
    }
    ds.labels[i] as c_int
}

impl V3Model {
    pub fn init(arch: V3Arch, seed: &[u8; 32]) -> Result<Self, String> {
        let mut raw = Neb3ModelRaw {
            arch: 0,
            nweights: 0,
            w: ptr::null_mut(),
            r: ptr::null_mut(),
        };
        let rc = unsafe { neb3_model_init(arch.id() as c_int, seed.as_ptr(), &mut raw) };
        if rc != 0 {
            return Err("neb3_model_init failed".into());
        }
        Ok(Self { arch, raw })
    }

    pub fn eval_share(&self, base: &V3Base, ds: &V3Dataset, key: &[u8; 32]) -> Result<V3ShareOutput, String> {
        if base.arch != self.arch || ds.arch != self.arch {
            return Err("architecture mismatch".into());
        }
        let raw_base = Neb3BaseRaw {
            arch: base.arch.id() as i32,
            nweights: base.weights.len(),
            w_base: base.weights.as_ptr(),
        };
        let raw_ds = Neb3DatasetRaw {
            n: ds.samples.len() as u32,
            get: v3_ds_get,
            ctx: (ds as *const V3Dataset) as *mut c_void,
        };
        let mut gradients = vec![0i32; self.arch.trainable_weights()];
        let mut digest = [0u8; 32];
        let mut ticket = [0u8; 32];
        let mut indices = [0u32; NB3];
        let mut clipped = 0u32;
        let rc = unsafe {
            neb3_share_eval(
                &self.raw,
                &raw_base,
                &raw_ds,
                key.as_ptr(),
                gradients.as_mut_ptr(),
                digest.as_mut_ptr(),
                ticket.as_mut_ptr(),
                indices.as_mut_ptr(),
                &mut clipped,
            )
        };
        if rc != 0 {
            return Err("neb3_share_eval failed".into());
        }
        Ok(V3ShareOutput {
            gradients,
            digest,
            ticket,
            indices,
            clipped,
        })
    }

    pub fn apply(&mut self, summed_gradients: &[i32]) -> Result<(), String> {
        if summed_gradients.len() != self.arch.trainable_weights() {
            return Err("gradient length mismatch".into());
        }
        unsafe { neb3_model_apply(&mut self.raw, summed_gradients.as_ptr()) };
        Ok(())
    }

    pub fn hashes(&self) -> Result<([u8; 32], [u8; 32]), String> {
        let mut wh = [0u8; 32];
        let mut rh = [0u8; 32];
        let rc = unsafe { neb3_model_hashes(&self.raw, wh.as_mut_ptr(), rh.as_mut_ptr()) };
        if rc != 0 {
            return Err("neb3_model_hashes failed".into());
        }
        Ok((wh, rh))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_8b_lora_dimensions_match_spec() {
        let arch = V3Arch::Qwen3_8bLora;
        assert_eq!(arch.d(), 4096);
        assert_eq!(arch.rank(), 8);
        assert_eq!(arch.nblocks(), 4);
        assert_eq!(arch.trainable_weights(), 270_336);
        assert_eq!(arch.packed_trainable_bytes(), 135_168);
        assert_eq!(arch.base_weights(), 67_108_864);
    }

    #[test]
    fn key_derived_dropout_prevents_per_sample_table_precomputation() {
        let arch = V3Arch::LoraTiny;
        // Identical sample repeated N times so every key selects the exact same sample contents.
        let mut sample = vec![0u8; arch.d()];
        for (i, v) in sample.iter_mut().enumerate() {
            *v = ((i * 3 + 1) % 8) as u8;
        }
        let ds = V3Dataset {
            arch,
            base_hash: [0u8; 32],
            samples: vec![sample; 64],
            labels: vec![1u8; 64],
        };
        let base_w: Vec<i8> = (0..arch.base_weights()).map(|i| ((i % 15) as i8) - 7).collect();
        let base = V3Base::new(arch, base_w).unwrap();
        let model = V3Model::init(arch, &[7u8; 32]).unwrap();

        let out_a = model.eval_share(&base, &ds, &[1u8; 32]).unwrap();
        let out_b = model.eval_share(&base, &ds, &[2u8; 32]).unwrap();

        // Even with identical samples for every batch slot, distinct keys produce distinct dropout masks -> distinct gradients.
        assert_ne!(out_a.gradients, out_b.gradients);
        assert_ne!(out_a.digest, out_b.digest);
    }

    #[test]
    fn parse_nmd3_roundtrip_and_validation() {
        let arch = V3Arch::LoraTiny;
        let base_hash = [0xABu8; 32];
        let mut buf = Vec::new();
        buf.extend_from_slice(b"NMD3");
        buf.extend_from_slice(&arch.id().to_le_bytes());
        buf.extend_from_slice(&base_hash);
        buf.extend_from_slice(&2u32.to_le_bytes());
        for lbl in [0u8, 1u8] {
            buf.extend(std::iter::repeat_n(3u8, arch.d()));
            buf.push(lbl);
        }
        let ds = V3Dataset::parse_nmd3(&buf).unwrap();
        assert_eq!(ds.arch, arch);
        assert_eq!(ds.base_hash, base_hash);
        assert_eq!(ds.samples.len(), 2);
        assert_eq!(ds.labels, vec![0, 1]);

        // Feature > 7 must be rejected
        buf[44] = 8;
        assert!(V3Dataset::parse_nmd3(&buf).is_err());
    }
}
