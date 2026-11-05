//! Tests of the block store: records, torn tails, checksums and the connect/disconnect index.

use super::*;
use neb_chain::demo;

fn temp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "neb-store-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let _ = fs::remove_dir_all(&p);
    p
}

fn chain(n: u32) -> Vec<Block> {
    let t0 = 1_800_000_000u32;
    let mut blocks = vec![demo::genesis(t0)];
    let mut tip = blocks[0].id();
    for i in 1..=n {
        let block = demo::block(
            tip,
            t0 + 60 * i,
            0xC280_0000,
            100 + i,
            vec![demo::coinbase(i, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000)],
        );
        tip = block.id();
        blocks.push(block);
    }
    blocks
}

// Blocks and the chain survive a close and reopen.
#[test]
fn append_and_reload() {
    let dir = temp_dir("reload");
    let blocks = chain(3);
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for b in &blocks {
            store.connect(b).unwrap();
        }
        assert_eq!(store.height(), 3);
        assert_eq!(store.block_count(), 4);
    }
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), blocks);
    assert_eq!(store.height(), 3);
    assert_eq!(store.block(&blocks[2].id()).unwrap(), &blocks[2]);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// Re-delivery of a block (peers send duplicates) must not grow the log.
#[test]
fn duplicate_blocks_are_stored_once() {
    let dir = temp_dir("dupe");
    let blocks = chain(1);
    let mut store = BlockStore::open(&dir).unwrap();
    assert!(store.put_block(&blocks[1]).unwrap());
    assert!(!store.put_block(&blocks[1]).unwrap());
    assert_eq!(store.block_count(), 1);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// Crash during an append: the partial record is cut off and all earlier blocks stay.
#[test]
fn torn_tail_is_truncated() {
    let dir = temp_dir("torn");
    let blocks = chain(2);
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for b in &blocks {
            store.connect(b).unwrap();
        }
    }
    let path = dir.join("blocks.dat");
    let len = fs::metadata(&path).unwrap().len();
    {
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[0xAA, 0xBB, 0xCC]).unwrap();
    }
    assert!(fs::metadata(&path).unwrap().len() > len);
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), blocks);
    assert_eq!(fs::metadata(&path).unwrap().len(), len);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// A damaged final record is treated as a torn write and dropped.
#[test]
fn a_corrupt_last_checksum_is_dropped() {
    let dir = temp_dir("checksum");
    let blocks = chain(1);
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for b in &blocks {
            store.connect(b).unwrap();
        }
    }
    let path = dir.join("blocks.dat");
    let len = fs::metadata(&path).unwrap().len();
    {
        let mut f = OpenOptions::new().read(true).write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(len - 1)).unwrap();
        let mut last = [0u8; 1];
        f.read_exact(&mut last).unwrap();
        f.seek(SeekFrom::Start(len - 1)).unwrap();
        f.write_all(&[last[0] ^ 0xFF]).unwrap();
    }
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), blocks[..1].to_vec());
    assert!(fs::metadata(&path).unwrap().len() < len);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// Losing the index loses no chain: it is rebuilt from the log.
#[test]
fn the_index_rebuilds_from_the_block_log() {
    let dir = temp_dir("rebuild");
    let blocks = chain(3);
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for b in &blocks {
            store.connect(b).unwrap();
        }
    }
    fs::remove_file(dir.join("index.dat")).unwrap();
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), blocks);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// Byte layout of the connect events: magic, length, event byte, id, work, checksum.
#[test]
fn connect_events_carry_the_cumulative_work() {
    let dir = temp_dir("work");
    let blocks = chain(3);
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for b in &blocks {
            store.connect(b).unwrap();
        }
    }
    let data = fs::read(dir.join("index.dat")).unwrap();
    let mut offset = 0usize;
    let mut expected = Work::ZERO;
    for block in &blocks {
        let magic = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        assert_eq!(magic, INDEX_MAGIC);
        let len = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        assert_eq!(len, CONNECT_PAYLOAD as usize);
        let payload = &data[offset + 8..offset + 8 + len];
        let sum = u32::from_le_bytes(data[offset + 8 + len..offset + 12 + len].try_into().unwrap());
        assert_eq!(sum, checksum(payload));
        assert_eq!(payload[0], EVENT_CONNECT);
        assert_eq!(&payload[1..33], &block.id());
        expected = expected + Work::of_bits(block.header.bits).unwrap();
        assert_eq!(&payload[33..65], &expected.to_le_bytes());
        offset += 8 + len + 4;
    }
    assert_eq!(offset, data.len());
    fs::remove_dir_all(&dir).unwrap();
}

// A well-formed record with a wrong work value must not be trusted: the index is rebuilt.
#[test]
fn reload_recomputes_the_work_and_repairs_a_wrong_value() {
    let dir = temp_dir("recompute");
    let blocks = chain(2);
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for b in &blocks {
            store.connect(b).unwrap();
        }
    }
    let path = dir.join("index.dat");
    let clean = fs::read(&path).unwrap();
    // Flip one byte of the second connect's recorded work and fix the
    // checksum, so that only the work value is wrong.
    let mut dirty = clean.clone();
    let record = 8 + CONNECT_PAYLOAD as usize + 4;
    let work_byte = record + 8 + 33;
    dirty[work_byte] ^= 0xFF;
    let sum = checksum(&dirty[record + 8..record + 8 + CONNECT_PAYLOAD as usize]);
    dirty[record + 8 + CONNECT_PAYLOAD as usize..record + 12 + CONNECT_PAYLOAD as usize]
        .copy_from_slice(&sum.to_le_bytes());
    fs::write(&path, &dirty).unwrap();
    assert_ne!(fs::read(&path).unwrap(), clean);
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), blocks);
    drop(store);
    // The rebuild rewrote the index with the correct values.
    assert_eq!(fs::read(&path).unwrap(), clean);
    fs::remove_dir_all(&dir).unwrap();
}

// Chain selection is by cumulative work: a shorter chain of harder blocks beats a taller one.
#[test]
fn rebuild_chooses_the_heaviest_tip_not_the_tallest() {
    let dir = temp_dir("heaviest");
    let t0 = 1_800_000_000u32;
    // A tall chain of light blocks: 9 blocks above genesis, each worth
    // floor(2^256 / (2^217 + 1)) = 2^39 - 1.
    let mut light = vec![demo::genesis(t0)];
    let mut parent = light[0].id();
    for i in 1..=9u32 {
        let block = demo::block(
            parent,
            t0 + 60 * i,
            0xC280_0000,
            100 + i,
            vec![demo::coinbase(i, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000)],
        );
        parent = block.id();
        light.push(block);
    }
    // A short chain of heavy blocks: 5 blocks above genesis, each worth
    // floor(2^256 / (2^216 + 1)) = 2^40 - 1.
    let mut heavy = vec![demo::genesis(t0)];
    let mut parent = heavy[0].id();
    for i in 1..=5u32 {
        let block = demo::block(
            parent,
            t0 + 5000 + 60 * i,
            0xC180_0000,
            900 + i,
            vec![demo::coinbase(i, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000)],
        );
        parent = block.id();
        heavy.push(block);
    }
    let unit_light = Work::of_bits(0xC280_0000).unwrap();
    let unit_heavy = Work::of_bits(0xC180_0000).unwrap();
    let sum = |unit: Work, n: u32| (0..n).fold(Work::ZERO, |acc, _| acc + unit);
    assert!(sum(unit_heavy, 6) > sum(unit_light, 10), "the short chain is the heavy one");
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for block in &light {
            store.connect(block).unwrap();
        }
        for block in heavy.iter().skip(1) {
            store.put_block(block).unwrap();
        }
        assert_eq!(store.height(), 9);
    }
    fs::remove_file(dir.join("index.dat")).unwrap();
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.height(), 5, "the heavier chain must win the rebuild");
    assert_eq!(store.chain_blocks().unwrap(), heavy);
    drop(store);
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), heavy);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// A block whose parent is not stored is ignored when the index is rebuilt.
#[test]
fn rebuild_ignores_a_branch_without_a_genesis_block() {
    let dir = temp_dir("dangling");
    let blocks = chain(2);
    let orphan = demo::block(
        [7u8; 32],
        1_800_000_000 + 9000,
        0xC180_0000,
        5,
        vec![demo::coinbase(9, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000)],
    );
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for b in &blocks {
            store.connect(b).unwrap();
        }
        store.put_block(&orphan).unwrap();
    }
    fs::remove_file(dir.join("index.dat")).unwrap();
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), blocks);
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// Disconnect events are replayed in order on open.
#[test]
fn disconnect_events_replay() {
    let dir = temp_dir("disconnect");
    let blocks = chain(3);
    {
        let mut store = BlockStore::open(&dir).unwrap();
        for b in &blocks {
            store.connect(b).unwrap();
        }
        store.disconnect(&blocks[3].id()).unwrap();
        store.disconnect(&blocks[2].id()).unwrap();
        assert_eq!(store.height(), 1);
    }
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), blocks[..2].to_vec());
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// Side branch blocks are kept (spec/node.md section 9) but are not on the best chain.
#[test]
fn side_blocks_stay_in_the_log_but_not_the_chain() {
    let dir = temp_dir("side");
    let blocks = chain(2);
    let mut store = BlockStore::open(&dir).unwrap();
    for b in &blocks {
        store.connect(b).unwrap();
    }
    let side = demo::block(
        blocks[1].id(),
        1_800_000_000 + 500,
        0xC280_0000,
        77,
        vec![demo::coinbase(2, demo::MINER_LOCK, demo::PROJECT_LOCK, 19 * 100_000_000)],
    );
    store.put_block(&side).unwrap();
    assert_eq!(store.block_count(), 4);
    assert_eq!(store.height(), 2);
    drop(store);
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.chain_blocks().unwrap(), blocks);
    assert!(store.contains(&side.id()));
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}

// A fresh directory opens as an empty store.
#[test]
fn missing_index_on_an_empty_store_is_fine() {
    let dir = temp_dir("empty");
    let store = BlockStore::open(&dir).unwrap();
    assert_eq!(store.height(), 0);
    assert!(store.chain_blocks().unwrap().is_empty());
    drop(store);
    fs::remove_dir_all(&dir).unwrap();
}
