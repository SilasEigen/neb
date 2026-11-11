//! The synthetic test datasets and the replay of the vector files (spec/shares.md sections 9 and 10,
//! spec/nebhash-v2.md section 8).

use super::{Arch, Dataset, Model, NB, NIN, SYN_N, chacha8_stream, hex_bytes, train_step};

/// Keystream of ChaCha8 with the given key, zero IV and block counter from 0.
fn chacha8(key: &[u8; 32], out: &mut [u8]) {
    // SAFETY: the C code reads 32 bytes of `key` and writes `out.len()` bytes to `out`.
    unsafe { chacha8_stream(key.as_ptr(), out.as_mut_ptr(), out.len()) };
}

/// The test dataset of spec/shares.md section 9: `SYN_N` samples built from ten
/// class prototypes, label `i mod 10`. It exists to produce the test vectors and
/// to run the demo and test chains.
#[derive(Debug)]
pub struct SyntheticDataset {
    protos: [[u8; NIN]; 10],
}

impl Default for SyntheticDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticDataset {
    pub fn new() -> SyntheticDataset {
        let mut protos = [[0u8; NIN]; 10];
        for (c, p) in protos.iter_mut().enumerate() {
            let mut input = [0u8; 14];
            input[..13].copy_from_slice(b"neb-syn-proto");
            // The class number as one byte (spec/shares.md section 9).
            input[13] = c as u8;
            chacha8(blake3::hash(&input).as_bytes(), p);
            p.iter_mut().for_each(|v| *v &= 7);
        }
        SyntheticDataset { protos }
    }
}

impl Dataset for SyntheticDataset {
    fn n(&self) -> u32 {
        SYN_N
    }

    fn get(&self, index: u32) -> ([u8; NIN], u8) {
        let mut input = [0u8; 18];
        input[..14].copy_from_slice(b"neb-syn-sample");
        input[14..].copy_from_slice(&index.to_le_bytes());
        let mut noise = [0u8; NIN];
        chacha8(blake3::hash(&input).as_bytes(), &mut noise);
        let class = index % 10;
        let proto = &self.protos[class as usize];
        // Prototype pixel plus noise in -1..=2, clipped to 0..=7.
        let x = std::array::from_fn(|p| (proto[p] as i32 + (noise[p] & 3) as i32 - 1).clamp(0, 7) as u8);
        (x, class as u8)
    }
}

/// The key of share `j` of step `s` in the test vectors (spec/shares.md section 10):
/// `BLAKE3("neb-test-share" || LE32(s) || LE32(j))`.
pub(super) fn test_share_key(s: u32, j: u32) -> [u8; 32] {
    let mut input = [0u8; 22];
    input[..14].copy_from_slice(b"neb-test-share");
    input[14..18].copy_from_slice(&s.to_le_bytes());
    input[18..].copy_from_slice(&j.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

fn num<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("bad number {s:?}"))
}

/// Replays a `neb-shares-vectors 1` file (spec/shares.md section 10) from the
/// zero seed on the synthetic dataset. The `init` line, every share (indices,
/// digest, ticket) and the state hashes after every step must match. Returns
/// the number of steps and of shares checked; the first mismatch is an error
/// that names the line, or the step and share.
pub fn replay_vectors(text: &str) -> Result<(usize, usize), String> {
    let ds = SyntheticDataset::new();
    let mut model = Model::from_seed(&[0u8; 32]);
    let mut lines = text.lines().enumerate().map(|(n, l)| (n + 1, l.trim())).filter(|(_, l)| !l.is_empty());

    let (_, first) = lines.next().ok_or("empty file")?;
    if first != "neb-shares-vectors 1" {
        return Err(format!("unknown header {first:?}"));
    }
    let (n, init) = lines.next().ok_or("missing init line")?;
    let f: Vec<&str> = init.split(' ').collect();
    if f.len() != 3 || f[0] != "init" {
        return Err(format!("line {n}: expected an init line"));
    }
    if (hex_bytes::<32>(f[1])?, hex_bytes::<32>(f[2])?) != model.hashes() {
        return Err(format!("line {n}: init hash mismatch"));
    }

    let mut keys = Vec::new();
    let mut want: Vec<([u8; 32], [u8; 32], [u32; NB])> = Vec::new();
    let (mut steps, mut shares) = (0usize, 0usize);
    for (n, line) in lines {
        let f: Vec<&str> = line.split(' ').collect();
        match f[0] {
            "share" if f.len() == 5 + NB => {
                if num::<usize>(f[1])? != steps || num::<usize>(f[2])? != keys.len() {
                    return Err(format!("line {n}: share out of order"));
                }
                let mut idx = [0u32; NB];
                for (slot, s) in idx.iter_mut().zip(&f[5..]) {
                    *slot = num(s)?;
                }
                // Keys follow from the position, so the file carries none.
                keys.push(test_share_key(steps as u32, keys.len() as u32));
                want.push((hex_bytes(f[3])?, hex_bytes(f[4])?, idx));
            }
            "state" if f.len() == 5 => {
                if num::<usize>(f[1])? != steps || num::<usize>(f[2])? != keys.len() {
                    return Err(format!("line {n}: state out of order"));
                }
                // The shares of a step are evaluated on the state left by the
                // previous step, then applied together.
                let got = train_step(&mut model, &ds, &keys)?;
                for (j, (r, w)) in got.iter().zip(&want).enumerate() {
                    if r.indices != w.2 {
                        return Err(format!("step {steps} share {j}: indices mismatch"));
                    }
                    if r.digest != w.0 {
                        return Err(format!("step {steps} share {j}: digest mismatch"));
                    }
                    if r.ticket != w.1 {
                        return Err(format!("step {steps} share {j}: ticket mismatch"));
                    }
                }
                if (hex_bytes::<32>(f[3])?, hex_bytes::<32>(f[4])?) != model.hashes() {
                    return Err(format!("line {n}: state hash mismatch after step {steps}"));
                }
                steps += 1;
                shares += keys.len();
                keys.clear();
                want.clear();
            }
            _ => return Err(format!("line {n}: unexpected line")),
        }
    }
    if !keys.is_empty() {
        return Err("the last step has no state line".into());
    }
    Ok((steps, shares))
}

/// The test dataset of spec/nebhash-v2.md section 8 for one v2 architecture: two class prototypes
/// `ChaCha8(BLAKE3("neb-syn2-proto" || arch || c)) & 7`, sample noise `ChaCha8(BLAKE3("neb-syn2-sample" || arch ||
/// LE32(i)))`, value `clip(P + (noise & 3) - 1, 0, 7)`, label `i mod 2`, `SYN_N` samples.
#[derive(Debug)]
pub struct SyntheticDataset2 {
    arch: Arch,
    protos: [Vec<u8>; 2],
}

impl SyntheticDataset2 {
    /// Panics for `Arch::V1`, whose test dataset is [`SyntheticDataset`].
    pub fn new(arch: Arch) -> SyntheticDataset2 {
        let id = arch.id().expect("a v2 architecture") as u8;
        let protos = [0u8, 1].map(|c| {
            let mut input = [0u8; 16];
            input[..14].copy_from_slice(b"neb-syn2-proto");
            input[14] = id;
            input[15] = c;
            let mut p = vec![0u8; arch.nin()];
            chacha8(blake3::hash(&input).as_bytes(), &mut p);
            p.iter_mut().for_each(|v| *v &= 7);
            p
        });
        SyntheticDataset2 { arch, protos }
    }
}

impl Dataset for SyntheticDataset2 {
    fn n(&self) -> u32 {
        SYN_N
    }

    fn arch(&self) -> Arch {
        self.arch
    }

    fn get(&self, index: u32) -> ([u8; NIN], u8) {
        let mut x = vec![0u8; self.arch.nin()];
        let label = self.sample(index, &mut x);
        (x[..NIN].try_into().unwrap(), label)
    }

    fn sample(&self, index: u32, x: &mut [u8]) -> u8 {
        let mut input = [0u8; 20];
        input[..15].copy_from_slice(b"neb-syn2-sample");
        input[15] = self.arch.id().unwrap() as u8;
        input[16..].copy_from_slice(&index.to_le_bytes());
        let nin = self.arch.nin();
        let mut noise = vec![0u8; nin];
        chacha8(blake3::hash(&input).as_bytes(), &mut noise);
        let proto = &self.protos[(index % 2) as usize];
        for p in 0..nin {
            x[p] = (proto[p] as i32 + (noise[p] & 3) as i32 - 1).clamp(0, 7) as u8;
        }
        (index % 2) as u8
    }
}

/// Replays a `neb-nebhash2-vectors 1` file (spec/nebhash-v2.md section 8) with the consensus update shifts. Returns
/// the number of steps and of shares checked.
pub fn replay_vectors2(text: &str) -> Result<(usize, usize), String> {
    let mut lines = text.lines().enumerate().map(|(n, l)| (n + 1, l.trim())).filter(|(_, l)| !l.is_empty());
    let (_, first) = lines.next().ok_or("empty file")?;
    if first != "neb-nebhash2-vectors 1" {
        return Err(format!("unknown header {first:?}"));
    }
    let mut current: Option<(Arch, Model, SyntheticDataset2)> = None;
    let (mut keys, mut want) = (Vec::new(), Vec::<([u8; 32], [u8; 32], [u32; NB])>::new());
    let (mut steps, mut shares) = (0usize, 0usize);
    for (n, line) in lines {
        let f: Vec<&str> = line.split(' ').collect();
        match f[0] {
            "init" if f.len() == 4 => {
                let arch = Arch::from_id(num(f[1])?).ok_or_else(|| format!("line {n}: unknown architecture"))?;
                let model = Model::from_seed_arch(arch, &[0u8; 32]);
                if (hex_bytes::<32>(f[2])?, hex_bytes::<32>(f[3])?) != model.hashes() {
                    return Err(format!("line {n}: init hash mismatch"));
                }
                current = Some((arch, model, SyntheticDataset2::new(arch)));
            }
            "share" if f.len() == 6 + NB => {
                let (arch, _, _) = current.as_ref().ok_or("share before init")?;
                let s: u32 = num(f[2])?;
                let mut idx = [0u32; NB];
                for (slot, v) in idx.iter_mut().zip(&f[6..]) {
                    *slot = num(v)?;
                }
                let mut input = [0u8; 24];
                input[..15].copy_from_slice(b"neb-test-share2");
                input[15] = arch.id().unwrap() as u8;
                input[16..20].copy_from_slice(&s.to_le_bytes());
                input[20..].copy_from_slice(&(keys.len() as u32).to_le_bytes());
                keys.push(*blake3::hash(&input).as_bytes());
                want.push((hex_bytes(f[4])?, hex_bytes(f[5])?, idx));
            }
            "state" if f.len() == 6 => {
                let (_, model, ds) = current.as_mut().ok_or("state before init")?;
                let got = train_step(model, ds, &keys)?;
                for (j, (r, w)) in got.iter().zip(&want).enumerate() {
                    if (r.digest, r.ticket, r.indices) != *w {
                        return Err(format!("line {n}: share {j} of the step differs"));
                    }
                }
                if (hex_bytes::<32>(f[4])?, hex_bytes::<32>(f[5])?) != model.hashes() {
                    return Err(format!("line {n}: state hash mismatch"));
                }
                steps += 1;
                shares += keys.len();
                keys.clear();
                want.clear();
            }
            _ => return Err(format!("line {n}: unexpected line")),
        }
    }
    Ok((steps, shares))
}
