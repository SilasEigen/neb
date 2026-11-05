//! Block store: an append-only log of blocks and an append-only index of
//! chain events (spec/node.md section 9). Consensus does not depend on the
//! storage format; the store only has to return what it was given.
//!
//! Both files hold records
//! `magic u32 | length u32 | payload | checksum u32`, all little-endian, where
//! the checksum is the first four bytes of BLAKE3(payload). It detects torn
//! and damaged writes; it is not an authentication code. Every append is
//! synced before the call returns, so a crash leaves at most one torn record
//! at the end of a file. Opening the store drops that tail and keeps every
//! complete record before it. A bad record that is not the last one is
//! reported as corruption for blocks; for the index it triggers a rebuild.
//!
//! The log holds every accepted block, side branches included. The index
//! records the connect and disconnect events that produced the best chain.
//! A connect event also carries the cumulative work of the new tip (spec/node.md
//! section 6.5). The value is recomputed from the block headers on every read
//! and must match, so a damaged value cannot change the chain. If the index is
//! missing or inconsistent, it is rebuilt from the log by following parent
//! links and choosing the tip with the most cumulative work, never the one
//! with the most blocks.
//!
//! `BlockStore` has no internal locking; callers serialise access.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use neb_chain::{Block, Work};

/// Record magic for a block: the bytes "BLK1" read as a little-endian u32.
const BLOCK_MAGIC: u32 = u32::from_le_bytes(*b"BLK1");
/// Record magic for an index event: the bytes "IDX1".
const INDEX_MAGIC: u32 = u32::from_le_bytes(*b"IDX1");
/// Upper bound of a block record payload in bytes. A block is at most
/// 1 000 000 bytes (spec/node.md section 5.4); the margin leaves room for
/// a larger limit without a format change.
const MAX_BLOCK_PAYLOAD: u32 = 4 * 1024 * 1024;
/// A connect event in bytes: event byte, block id (32), cumulative work (32,
/// little-endian 256-bit).
const CONNECT_PAYLOAD: u32 = 65;
/// A disconnect event: the event byte and the block id.
const DISCONNECT_PAYLOAD: u32 = 33;
/// Magic and length fields before the payload, in bytes.
const RECORD_HEADER: u64 = 8;

/// First payload byte of a connect event in the index.
pub const EVENT_CONNECT: u8 = 1;
/// First payload byte of a disconnect event in the index.
pub const EVENT_DISCONNECT: u8 = 2;

#[derive(Debug)]
pub enum StoreError {
    Io(io::Error),
    /// A record or event violates the format or the chain rules of the store.
    Corrupt(String),
    /// The index or a parent link names a block that is not in the log.
    MissingBlock([u8; 32]),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "store io: {e}"),
            StoreError::Corrupt(e) => write!(f, "store corrupt: {e}"),
            StoreError::MissingBlock(id) => {
                write!(f, "store: block {} is indexed but absent", hex(id))
            }
        }
    }
}

impl std::error::Error for StoreError {}

impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        StoreError::Io(e)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn checksum(payload: &[u8]) -> u32 {
    let h = blake3::hash(payload);
    u32::from_le_bytes(h.as_bytes()[..4].try_into().unwrap())
}

/// Appends one record at the end and syncs it. The record is built in one
/// buffer and written with one call so that a crash tears at most this record.
fn write_record(file: &mut File, magic: u32, payload: &[u8]) -> io::Result<()> {
    file.seek(SeekFrom::End(0))?;
    let mut out = Vec::with_capacity(12 + payload.len());
    out.extend_from_slice(&magic.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&checksum(payload).to_le_bytes());
    file.write_all(&out)?;
    file.sync_data()
}

/// Reads one record header. `None` means a clean end of file.
fn read_header(file: &mut File) -> io::Result<Option<(u32, u32)>> {
    let mut buf = [0u8; 8];
    match file.read_exact(&mut buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    Ok(Some((u32::from_le_bytes(buf[..4].try_into().unwrap()), u32::from_le_bytes(buf[4..].try_into().unwrap()))))
}

/// What a scan of a record file found.
struct Scan {
    /// Offset just past the last complete record.
    good_end: u64,
    /// True when the file ended in a partial record that was ignored.
    truncated: bool,
}

/// Cuts the file at `offset` (the start of a torn record) and syncs.
fn truncate_to(file: &mut File, offset: u64) -> io::Result<()> {
    file.set_len(offset)?;
    file.seek(SeekFrom::Start(offset))?;
    file.sync_all()
}

/// The block log and the chain index of one data directory (`blocks.dat`,
/// `index.dat`). All blocks are also held in memory.
pub struct BlockStore {
    dir: PathBuf,
    blocks_file: File,
    index_file: File,
    blocks: BTreeMap<[u8; 32], Block>,
    /// Byte offset of each block's record in `blocks.dat`.
    offsets: BTreeMap<[u8; 32], u64>,
    /// Ids of the best chain, genesis first.
    chain: Vec<[u8; 32]>,
    /// Cumulative work per best-chain block, parallel to `chain`.
    chain_work: Vec<Work>,
}

impl BlockStore {
    /// Opens or creates the store in `dir`: repairs torn tails and rebuilds
    /// the index when it is missing or inconsistent. Errors on a damaged
    /// block record that is not the last one, and on blocks whose parent is
    /// absent from the log.
    pub fn open(dir: impl AsRef<Path>) -> Result<BlockStore, StoreError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let block_path = dir.join("blocks.dat");
        let index_path = dir.join("index.dat");

        let mut blocks_file =
            OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&block_path)?;
        let mut index_file =
            OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&index_path)?;

        let mut blocks = BTreeMap::new();
        let mut offsets = BTreeMap::new();
        let block_scan = scan_blocks(&mut blocks_file, &mut blocks, &mut offsets)?;
        if block_scan.truncated {
            blocks_file.set_len(block_scan.good_end)?;
            blocks_file.seek(SeekFrom::End(0))?;
            blocks_file.sync_all()?;
        }

        let mut chain = Vec::new();
        let mut chain_work = Vec::new();
        // Any index error, including I/O, selects the rebuild: the log is the
        // source of truth.
        let index_ok = scan_index(&mut index_file, &blocks, &mut chain, &mut chain_work).unwrap_or(false);
        // An empty index next to a non-empty log (index deleted) also rebuilds.
        let needs_rebuild = !index_ok || chain.is_empty() && !blocks.is_empty();
        if needs_rebuild {
            let rebuilt = rebuild_chain(&blocks)?;
            drop(index_file);
            write_index(&index_path, &rebuilt)?;
            index_file = OpenOptions::new().read(true).write(true).open(&index_path)?;
            chain = rebuilt.iter().map(|(id, _)| *id).collect();
            chain_work = rebuilt.iter().map(|(_, work)| *work).collect();
        }

        Ok(BlockStore { dir, blocks_file, index_file, blocks, offsets, chain, chain_work })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Height of the best tip; 0 for both an empty store and a genesis-only chain.
    pub fn height(&self) -> u32 {
        self.chain.len().saturating_sub(1) as u32
    }

    pub fn chain_ids(&self) -> &[[u8; 32]] {
        &self.chain
    }

    /// The best chain in order, genesis first.
    pub fn chain_blocks(&self) -> Result<Vec<Block>, StoreError> {
        let mut out = Vec::with_capacity(self.chain.len());
        for id in &self.chain {
            out.push(self.blocks.get(id).cloned().ok_or(StoreError::MissingBlock(*id))?);
        }
        Ok(out)
    }

    pub fn block(&self, id: &[u8; 32]) -> Option<&Block> {
        self.blocks.get(id)
    }

    pub fn contains(&self, id: &[u8; 32]) -> bool {
        self.blocks.contains_key(id)
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Every block in the log, connected or not, in id order.
    pub fn all_blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.values()
    }

    /// Byte offset of the block's record in `blocks.dat`.
    pub fn offset(&self, id: &[u8; 32]) -> Option<u64> {
        self.offsets.get(id).copied()
    }

    /// Appends a block to the log without touching the index (a side branch
    /// block). Returns true when it was written, false when already stored.
    pub fn put_block(&mut self, block: &Block) -> Result<bool, StoreError> {
        let id = block.id();
        if self.blocks.contains_key(&id) {
            return Ok(false);
        }
        let bytes = block.encode();
        let offset = self.blocks_file.metadata()?.len();
        write_record(&mut self.blocks_file, BLOCK_MAGIC, &bytes)?;
        self.offsets.insert(id, offset);
        self.blocks.insert(id, block.clone());
        Ok(true)
    }

    /// Writes the block, then the connect event that extends the best chain
    /// with the new tip's cumulative work. The order matters: a crash between
    /// the two leaves a side block, never an event for a missing block. The
    /// block must extend the current tip (the genesis block the empty chain);
    /// consensus validity is checked by the caller.
    pub fn connect(&mut self, block: &Block) -> Result<(), StoreError> {
        let id = block.id();
        let parent_ok = match self.chain.last() {
            Some(tip) => block.header.parent == *tip,
            None => block.header.parent == [0u8; 32],
        };
        if !parent_ok {
            return Err(StoreError::Corrupt("connect event does not extend the chain".into()));
        }
        let work = self.chain_work.last().copied().unwrap_or(Work::ZERO) + block_work(block)?;
        self.put_block(block)?;
        self.append_connect(&id, work)?;
        self.chain.push(id);
        self.chain_work.push(work);
        Ok(())
    }

    /// Records that the tip `id` left the best chain. The block stays in the
    /// log; anything other than the tip is an error.
    pub fn disconnect(&mut self, id: &[u8; 32]) -> Result<(), StoreError> {
        if self.chain.last() != Some(id) {
            return Err(StoreError::Corrupt("disconnect event is not the tip".into()));
        }
        self.append_disconnect(id)?;
        self.chain.pop();
        self.chain_work.pop();
        Ok(())
    }

    fn append_connect(&mut self, id: &[u8; 32], work: Work) -> Result<(), StoreError> {
        write_record(&mut self.index_file, INDEX_MAGIC, &connect_payload(id, work))?;
        Ok(())
    }

    fn append_disconnect(&mut self, id: &[u8; 32]) -> Result<(), StoreError> {
        let mut payload = [0u8; DISCONNECT_PAYLOAD as usize];
        payload[0] = EVENT_DISCONNECT;
        payload[1..].copy_from_slice(id);
        write_record(&mut self.index_file, INDEX_MAGIC, &payload)?;
        Ok(())
    }
}

/// Work of one block, `floor(2^256 / (target + 1))` at its `bits` (spec/node.md
/// section 6.5).
fn block_work(block: &Block) -> Result<Work, StoreError> {
    Work::of_bits(block.header.bits).map_err(|e| StoreError::Corrupt(format!("block bits: {e}")))
}

/// Connect event payload: event byte, block id, cumulative work.
fn connect_payload(id: &[u8; 32], work: Work) -> [u8; CONNECT_PAYLOAD as usize] {
    let mut payload = [0u8; CONNECT_PAYLOAD as usize];
    payload[0] = EVENT_CONNECT;
    payload[1..33].copy_from_slice(id);
    payload[33..].copy_from_slice(&work.to_le_bytes());
    payload
}

/// Reads every complete block record into memory. An incomplete record at the
/// end of the file, or one whose checksum fails and that ends the file, is a
/// torn write (`truncated`). A bad magic, an oversize length that fits in the
/// file, a bad checksum in the middle, or an undecodable payload is an error.
/// A block that appears twice keeps its first copy.
fn scan_blocks(
    file: &mut File,
    blocks: &mut BTreeMap<[u8; 32], Block>,
    offsets: &mut BTreeMap<[u8; 32], u64>,
) -> Result<Scan, StoreError> {
    file.seek(SeekFrom::Start(0))?;
    let len = file.metadata()?.len();
    let mut offset = 0u64;
    loop {
        if offset == len {
            return Ok(Scan { good_end: offset, truncated: false });
        }
        if len - offset < 8 {
            return Ok(Scan { good_end: offset, truncated: true });
        }
        file.seek(SeekFrom::Start(offset))?;
        let Some((magic, payload_len)) = read_header(file)? else {
            return Ok(Scan { good_end: offset, truncated: true });
        };
        if magic != BLOCK_MAGIC {
            return Err(StoreError::Corrupt(format!("bad block magic at offset {offset}")));
        }
        // Check the length before allocating: it is read from the file.
        if payload_len > MAX_BLOCK_PAYLOAD {
            if offset + RECORD_HEADER + payload_len as u64 + 4 > len {
                return Ok(Scan { good_end: offset, truncated: true });
            }
            return Err(StoreError::Corrupt(format!("block length {payload_len} out of range")));
        }
        let record_end = offset + RECORD_HEADER + payload_len as u64 + 4;
        if record_end > len {
            return Ok(Scan { good_end: offset, truncated: true });
        }
        let mut payload = vec![0u8; payload_len as usize];
        file.read_exact(&mut payload).map_err(|e| StoreError::Corrupt(format!("short block payload: {e}")))?;
        let mut sum = [0u8; 4];
        file.read_exact(&mut sum).map_err(|e| StoreError::Corrupt(format!("short block checksum: {e}")))?;
        if u32::from_le_bytes(sum) != checksum(&payload) {
            if record_end == len {
                return Ok(Scan { good_end: offset, truncated: true });
            }
            return Err(StoreError::Corrupt(format!("block checksum at offset {offset}")));
        }
        let block =
            Block::decode(&payload).map_err(|e| StoreError::Corrupt(format!("block at offset {offset}: {e}")))?;
        let id = block.id();
        if offsets.insert(id, offset).is_none() {
            blocks.insert(id, block);
        }
        offset = record_end;
    }
}

/// Replays the index events, recomputing the cumulative work of every connect
/// from the block headers. Returns false when the file is not a valid event
/// log for these blocks (bad magic or length, damaged record that is not the
/// last, unknown block, wrong parent or work, disconnect of a non-tip), so
/// the caller rebuilds it. A torn last record is cut off and is not an error.
fn scan_index(
    file: &mut File,
    blocks: &BTreeMap<[u8; 32], Block>,
    chain: &mut Vec<[u8; 32]>,
    chain_work: &mut Vec<Work>,
) -> Result<bool, StoreError> {
    file.seek(SeekFrom::Start(0))?;
    let len = file.metadata()?.len();
    let mut offset = 0u64;
    loop {
        if offset == len {
            return Ok(true);
        }
        if len - offset < 8 {
            truncate_to(file, offset)?;
            return Ok(true);
        }
        file.seek(SeekFrom::Start(offset))?;
        let Some((magic, payload_len)) = read_header(file)? else {
            truncate_to(file, offset)?;
            return Ok(true);
        };
        if magic != INDEX_MAGIC {
            return Ok(false);
        }
        if payload_len != CONNECT_PAYLOAD && payload_len != DISCONNECT_PAYLOAD {
            return Ok(false);
        }
        let record_end = offset + RECORD_HEADER + payload_len as u64 + 4;
        if record_end > len {
            truncate_to(file, offset)?;
            return Ok(true);
        }
        let mut payload = vec![0u8; payload_len as usize];
        file.read_exact(&mut payload)?;
        let mut sum = [0u8; 4];
        file.read_exact(&mut sum)?;
        if u32::from_le_bytes(sum) != checksum(&payload) {
            if record_end == len {
                truncate_to(file, offset)?;
                return Ok(true);
            }
            return Ok(false);
        }
        let id: [u8; 32] = payload[1..33].try_into().unwrap();
        match payload[0] {
            EVENT_CONNECT if payload_len == CONNECT_PAYLOAD => {
                let Some(block) = blocks.get(&id) else {
                    return Ok(false);
                };
                let parent_ok = match chain.last() {
                    Some(tip) => block.header.parent == *tip,
                    None => block.header.parent == [0u8; 32],
                };
                if !parent_ok {
                    return Ok(false);
                }
                let expected = chain_work.last().copied().unwrap_or(Work::ZERO) + block_work(block)?;
                let recorded = Work::from_le_bytes(payload[33..65].try_into().unwrap());
                if recorded != expected {
                    return Ok(false);
                }
                chain.push(id);
                chain_work.push(expected);
            }
            EVENT_DISCONNECT if payload_len == DISCONNECT_PAYLOAD => {
                if chain.pop() != Some(id) || chain_work.pop().is_none() {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        }
        offset = record_end;
    }
}

/// Rebuilds the best chain by following parent links and choosing the tip
/// with the most cumulative work (spec/node.md section 6.5, 9). Ties break on
/// the smaller block id, so the result does not depend on the file order.
/// Blocks whose ancestry does not reach a genesis block are ignored.
/// Returns the chain, genesis first, with the cumulative work per block.
/// Memoised: each block's work is computed once from the nearest known
/// ancestor, so the pass is linear in the number of blocks.
fn rebuild_chain(blocks: &BTreeMap<[u8; 32], Block>) -> Result<Vec<([u8; 32], Work)>, StoreError> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    let mut work: BTreeMap<[u8; 32], Work> = BTreeMap::new();
    'blocks: for id in blocks.keys() {
        if work.contains_key(id) {
            continue;
        }
        let mut path = Vec::new();
        let mut cur = *id;
        let base;
        loop {
            if let Some(w) = work.get(&cur) {
                base = *w;
                break;
            }
            // A branch whose ancestry does not reach a genesis block has no
            // cumulative work and is never a candidate tip.
            let Some(block) = blocks.get(&cur) else { continue 'blocks };
            path.push(cur);
            if block.header.parent == [0u8; 32] {
                base = Work::ZERO;
                break;
            }
            cur = block.header.parent;
            // A hash cycle cannot occur; the bound only stops a corrupt log.
            if path.len() > 10_000_000 {
                return Err(StoreError::Corrupt("parent cycle".into()));
            }
        }
        let mut w = base;
        for p in path.iter().rev() {
            let block = blocks.get(p).ok_or(StoreError::MissingBlock(*p))?;
            w = w + block_work(block)?;
            work.insert(*p, w);
        }
    }
    // Tips are the blocks that no other block names as parent.
    let children: std::collections::BTreeSet<[u8; 32]> =
        blocks.values().filter(|b| b.header.parent != [0u8; 32]).map(|b| b.header.parent).collect();
    let mut best: Option<([u8; 32], Work)> = None;
    for id in blocks.keys() {
        if children.contains(id) {
            continue;
        }
        let Some(&w) = work.get(id) else { continue };
        let better = match best {
            None => true,
            Some((bid, bw)) => w > bw || (w == bw && *id < bid),
        };
        if better {
            best = Some((*id, w));
        }
    }
    let (mut cur, _) = best.ok_or(StoreError::Corrupt("no tip".into()))?;
    let mut chain = Vec::new();
    loop {
        chain.push((cur, work[&cur]));
        let block = blocks.get(&cur).ok_or(StoreError::MissingBlock(cur))?;
        if block.header.parent == [0u8; 32] {
            break;
        }
        cur = block.header.parent;
        if chain.len() > 10_000_000 {
            return Err(StoreError::Corrupt("parent cycle".into()));
        }
    }
    chain.reverse();
    Ok(chain)
}

/// Replaces the index by one connect event per chain block. Written to a
/// temporary file and renamed, so a crash leaves the old or the new index.
fn write_index(path: &Path, chain: &[([u8; 32], Work)]) -> Result<(), StoreError> {
    let tmp = path.with_extension("tmp");
    let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(&tmp)?;
    for (id, work) in chain {
        write_record(&mut file, INDEX_MAGIC, &connect_payload(id, *work))?;
    }
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests;
