//! The node object: the chain state, the block store, the mempool, the work
//! service jobs, the peer set and the ban list behind one mutex.
//!
//! The consensus rules are in `neb-chain` (spec/node.md section 6.1, spec/shares.md
//! section 11). This module adds what a rule cannot know: the index of known
//! blocks across side branches, the cheap header checks of blocks that do not
//! extend the tip, the proof budget of a peer, and the choice of the best
//! chain by cumulative work (spec/node.md section 6.5).
//!
//! Lock discipline: `NodeInner` is behind one mutex. It is never held while
//! waiting on the proof budget, writing to a socket, or verifying a block;
//! `accept_block_from` takes it twice and re-checks `blocks` in between.

use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use neb_block::Header;
use neb_chain::block::check_shares_list;
use neb_chain::state::{ChainState, HeaderError, MAX_REORG};
use neb_chain::tx::Transaction;
use neb_chain::{Block, Work as ChainWork};
use neb_pow::shares::Model;
use neb_store::BlockStore;
use neb_work::{Mempool, Work};

use crate::budget::{BanList, DEFAULT_POW_BUDGET, TokenBucket};
use crate::miners::{MinerSet, MinerWriter};
use crate::p2p::{PeerInfo, PeerSet};
use crate::params::NetworkParams;

mod lookup;
use lookup::{fork_height, locator_start, models_of, prune_side_models};

/// Maximum entries of an `inv`, `getdata` or `notfound` message (spec/node.md section 8.2).
pub const MAX_INV: usize = 5_000;
/// Maximum headers of a `headers` message (spec/node.md section 8.2).
pub const HEADERS_BATCH: usize = 2_000;
/// Blocks announced per `getblocks` answer. Local policy.
pub const BLOCKS_BATCH: usize = 500;
/// Upper bound on the stored models of side-branch blocks: 64 * 508160 bytes
/// (one model: weights and residuals, spec/shares.md section 4) is about 32 MB.
const MAX_SIDE_MODELS: usize = 64;

#[derive(Debug)]
pub enum NodeError {
    Io(String),
    Store(String),
    Decode(String),
    Header(String),
    /// The block passed the cheap checks and its proof of work or one of its
    /// shares is invalid (spec/shares.md section 11). The sending peer is banned.
    BadProof(String),
    Connect(String),
    Reorg(String),
    Orphan,
    Known,
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeError::Io(e) => write!(f, "io: {e}"),
            NodeError::Store(e) => write!(f, "store: {e}"),
            NodeError::Decode(e) => write!(f, "decode: {e}"),
            NodeError::Header(e) => write!(f, "header: {e}"),
            NodeError::BadProof(e) => write!(f, "proof: {e}"),
            NodeError::Connect(e) => write!(f, "connect: {e}"),
            NodeError::Reorg(e) => write!(f, "reorg: {e}"),
            NodeError::Orphan => write!(f, "orphan block"),
            NodeError::Known => write!(f, "already known"),
        }
    }
}

impl std::error::Error for NodeError {}

impl From<std::io::Error> for NodeError {
    fn from(e: std::io::Error) -> Self {
        NodeError::Io(e.to_string())
    }
}

impl From<neb_store::StoreError> for NodeError {
    fn from(e: neb_store::StoreError) -> Self {
        NodeError::Store(e.to_string())
    }
}

#[derive(Clone, Debug)]
pub struct NodeConfig {
    pub network: NetworkParams,
    pub data_dir: PathBuf,
    /// Verifies the proof of work and the shares of every block. Must be true
    /// outside tests. When false, the ticket checks are skipped but the share
    /// lists still train the model and every other check runs.
    pub verify_pow: bool,
}

impl NodeConfig {
    pub fn new(network: NetworkParams, data_dir: impl AsRef<Path>) -> NodeConfig {
        NodeConfig { network, data_dir: data_dir.as_ref().to_path_buf(), verify_pow: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvKind {
    Block = 1,
    Tx = 2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockOutcome {
    Connected { height: u32 },
    Side { height: u32 },
    Reorged { height: u32, disconnected: u32, connected: u32 },
    AlreadyKnown,
}

pub struct NodeInner {
    pub state: ChainState,
    pub mempool: Mempool,
    pub work: Work,
    pub store: BlockStore,
    /// Every known block, on the best chain or on a side branch.
    pub blocks: BTreeMap<[u8; 32], Block>,
    pub heights: BTreeMap<[u8; 32], u32>,
    /// Cumulative work of every known block: the sum of the work of the blocks
    /// from the genesis to it, both included (spec/node.md section 6.5).
    /// Recomputed from the headers on load; chain selection compares this
    /// value, never the height.
    pub block_work: BTreeMap<[u8; 32], ChainWork>,
    /// Models of the side-branch blocks whose proof and shares were checked,
    /// for the last `MODEL_DEPTH` levels (spec/shares.md section 11.4). The
    /// models of the best chain are in `state`.
    pub side_models: BTreeMap<[u8; 32], neb_chain::Models>,
    /// Block id to height for the blocks of the current best chain.
    pub best_index: BTreeMap<[u8; 32], u32>,
    /// Lock that the jobs pay (`--miner-lock`); without it the node gives no work to miners.
    pub miner_lock: Option<[u8; 32]>,
    pub last_block_at: u32,
}

#[derive(Clone)]
pub struct Node {
    pub(crate) inner: Arc<Mutex<NodeInner>>,
    pub(crate) peers: Arc<PeerSet>,
    pub(crate) miners: Arc<MinerSet>,
    pub(crate) network: NetworkParams,
    pub(crate) verify_pow: bool,
    pub(crate) pow_budget: NonZeroU32,
    pub(crate) bans: Arc<BanList>,
    pub(crate) nonce: u64,
}

/// Unix time in seconds. Used for the clock rules of spec/node.md section 6.1
/// (local policy: the result depends on the node's clock).
pub fn now() -> u32 {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_secs() as u32
}

/// Per-process value of the `version` message, to detect a connection to self
/// (spec/node.md section 8.2). Not secret, not random.
fn local_nonce() -> u64 {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_nanos() as u64;
    nanos ^ ((std::process::id() as u64) << 32)
}

impl Node {
    /// Opens the store, writing the genesis block into an empty one, and replays
    /// the best chain from the genesis block to rebuild the models. Heights and
    /// cumulative work of all stored blocks are recomputed from the headers
    /// (nothing derived is read from the store). Fails when the store belongs to
    /// another genesis, a stored block does not connect, or a stored block has
    /// no parent in the store (`NodeError::Store`).
    pub fn open(config: NodeConfig) -> Result<Node, NodeError> {
        let mut store = BlockStore::open(&config.data_dir)?;
        let genesis = config.network.genesis.clone();
        if store.chain_blocks()?.is_empty() {
            store.connect(&genesis)?;
        }
        let chain = store.chain_blocks()?;
        if chain.first().map(|b| b.id()) != Some(genesis.id()) {
            return Err(NodeError::Store("the store belongs to a different genesis".into()));
        }
        let mut state = ChainState::from_genesis(config.network.chain.clone(), genesis.clone())
            .map_err(|e| NodeError::Connect(e.to_string()))?;
        state.verify_pow = config.verify_pow;
        for block in &chain[1..] {
            state.connect_block(block).map_err(|e| NodeError::Connect(e.to_string()))?;
        }
        let missing_parent = || NodeError::Store("a stored block has no parent in the store".into());
        let mut blocks = BTreeMap::new();
        let mut heights = BTreeMap::new();
        for block in store.all_blocks() {
            blocks.insert(block.id(), block.clone());
        }
        let ids: Vec<[u8; 32]> = blocks.keys().copied().collect();
        for id in ids {
            if heights.contains_key(&id) {
                continue;
            }
            let mut path = Vec::new();
            let mut cur = id;
            let base;
            loop {
                if let Some(h) = heights.get(&cur) {
                    base = *h;
                    break;
                }
                let block = blocks.get(&cur).ok_or_else(missing_parent)?;
                path.push(cur);
                if block.header.parent == [0u8; 32] {
                    base = 0;
                    break;
                }
                cur = block.header.parent;
            }
            let deepest = *path.last().expect("at least one block");
            let mut h: u32 = if blocks[&deepest].header.parent == [0u8; 32] { 0 } else { base + 1 };
            for p in path.iter().rev() {
                heights.insert(*p, h);
                h = h.saturating_add(1);
            }
        }
        let mut block_work: BTreeMap<[u8; 32], ChainWork> = BTreeMap::new();
        let ids: Vec<[u8; 32]> = blocks.keys().copied().collect();
        for id in ids {
            if block_work.contains_key(&id) {
                continue;
            }
            let mut path = Vec::new();
            let mut cur = id;
            let base;
            loop {
                if let Some(w) = block_work.get(&cur) {
                    base = *w;
                    break;
                }
                let block = blocks.get(&cur).ok_or_else(missing_parent)?;
                path.push(cur);
                if block.header.parent == [0u8; 32] {
                    base = ChainWork::ZERO;
                    break;
                }
                cur = block.header.parent;
            }
            let mut w = base;
            for p in path.iter().rev() {
                let block = &blocks[p];
                let one = ChainWork::of_bits(block.header.bits)
                    .map_err(|e| NodeError::Header(format!("stored block bits: {e}")))?;
                w = w + one;
                block_work.insert(*p, w);
            }
        }
        let mut best_index = BTreeMap::new();
        for height in 0..=state.height() {
            best_index.insert(state.id_at(height), height);
        }
        let now = now();
        let inner = NodeInner {
            state,
            mempool: Mempool::new(),
            work: Work::new(),
            store,
            blocks,
            heights,
            block_work,
            side_models: BTreeMap::new(),
            best_index,
            miner_lock: None,
            last_block_at: now,
        };
        let peers = Arc::new(PeerSet::new(config.network.max_peers));
        let miners = Arc::new(MinerSet::new());
        Ok(Node {
            inner: Arc::new(Mutex::new(inner)),
            peers,
            miners,
            network: config.network,
            verify_pow: config.verify_pow,
            pow_budget: NonZeroU32::new(DEFAULT_POW_BUDGET).expect("nonzero"),
            bans: Arc::new(BanList::default()),
            nonce: local_nonce(),
        })
    }

    /// Sets the ticket evaluations per second that one peer session may cost
    /// the node (spec/node.md section 8.1).
    pub fn with_pow_budget(mut self, per_second: NonZeroU32) -> Node {
        self.pow_budget = per_second;
        self
    }

    /// True while this address is banned for sending an invalid proof of work
    /// or share (spec/node.md section 8.1).
    pub fn is_banned(&self, ip: IpAddr) -> bool {
        self.bans.is_banned(ip)
    }

    pub fn network(&self) -> &NetworkParams {
        &self.network
    }

    pub fn chain_id(&self) -> [u8; 32] {
        self.network.genesis.id()
    }

    pub fn height(&self) -> u32 {
        self.inner.lock().expect("node lock").state.height()
    }

    pub fn tip_id(&self) -> [u8; 32] {
        self.inner.lock().expect("node lock").state.tip_id()
    }

    pub fn header_at(&self, height: u32) -> Option<Header> {
        let inner = self.inner.lock().expect("node lock");
        if height > inner.state.height() {
            return None;
        }
        let id = inner.state.id_at(height);
        inner.blocks.get(&id).map(|b| b.header)
    }

    pub fn block(&self, id: &[u8; 32]) -> Option<Block> {
        self.inner.lock().expect("node lock").blocks.get(id).cloned()
    }

    pub fn block_at(&self, height: u32) -> Option<Block> {
        let inner = self.inner.lock().expect("node lock");
        if height > inner.state.height() {
            return None;
        }
        let id = inner.state.id_at(height);
        inner.blocks.get(&id).cloned()
    }

    pub fn has_block(&self, id: &[u8; 32]) -> bool {
        let inner = self.inner.lock().expect("node lock");
        inner.blocks.contains_key(id)
    }

    pub fn utxo(&self, op: &neb_chain::OutPoint) -> Option<neb_chain::Utxo> {
        self.inner.lock().expect("node lock").state.utxo(op).cloned()
    }

    pub fn has_tx(&self, id: &[u8; 32]) -> bool {
        self.inner.lock().expect("node lock").mempool.tx(id).is_some()
    }

    pub fn mempool_ids(&self) -> Vec<[u8; 32]> {
        self.inner.lock().expect("node lock").mempool.txids()
    }

    pub fn peer_info(&self) -> Vec<PeerInfo> {
        self.peers.info()
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// True when a session with this exact address is connected.
    pub fn is_peer_connected(&self, addr: &SocketAddr) -> bool {
        self.peers.is_connected(addr)
    }

    pub fn nonce(&self) -> u64 {
        self.nonce
    }

    pub fn set_miner_lock(&self, lock: [u8; 32]) {
        self.inner.lock().expect("node lock").miner_lock = Some(lock);
    }

    pub fn miner_lock(&self) -> Option<[u8; 32]> {
        self.inner.lock().expect("node lock").miner_lock
    }

    /// Registers one miner session of the work service; the id unregisters
    /// it when the session ends.
    pub fn register_miner(&self, writer: MinerWriter) -> u64 {
        self.miners.add(writer)
    }

    pub fn unregister_miner(&self, id: u64) {
        self.miners.remove(id);
    }

    /// The number of connected miner sessions.
    pub fn miner_count(&self) -> usize {
        self.miners.len()
    }

    /// Header steps 1, 2, 4 and 5 of spec/node.md section 6.1, in that order:
    /// version, parent, expected `bits`, time window. They need no model, so an
    /// attacker pays nothing to be rejected here but must still name a known
    /// parent and guess `bits`. The proof of work and the shares are checked
    /// later, on the model of the parent (spec/shares.md section 11.1).
    fn check_header_cheap(
        &self,
        inner: &NodeInner,
        header: &Header,
        parent: &Header,
        parent_height: u32,
        now: u32,
    ) -> Result<(), NodeError> {
        if header.version != 1 {
            return Err(NodeError::Header(HeaderError::Version.to_string()));
        }
        if header.parent != parent.id() {
            return Err(NodeError::Header("parent does not match".into()));
        }
        let want = inner
            .state
            .params
            .difficulty
            .next_bits(parent_height, parent.time)
            .map_err(|e| NodeError::Header(format!("difficulty: {e}")))?;
        if header.bits != want {
            return Err(NodeError::Header(HeaderError::BadBits.to_string()));
        }
        // Median of the parent and up to ten predecessors; stops early near the
        // genesis block.
        let mut times = vec![parent.time];
        let mut cur = *parent;
        let mut cur_height = parent_height;
        while times.len() < 11 && cur_height > 0 {
            let Some(next) = inner.blocks.get(&cur.parent).map(|b| b.header) else {
                break;
            };
            times.push(next.time);
            cur = next;
            cur_height -= 1;
        }
        times.sort_unstable();
        let median = times[times.len() / 2];
        if header.time <= median {
            return Err(NodeError::Header(HeaderError::StaleTime.to_string()));
        }
        if header.time > now + 120 {
            return Err(NodeError::Header(HeaderError::FutureTime.to_string()));
        }
        Ok(())
    }

    /// Accepts a block: connects it when it extends the tip, keeps it as a
    /// side branch otherwise, and reorganises when its branch carries strictly
    /// more cumulative work and removes at most `MAX_REORG` blocks (spec/node.md
    /// section 6.5). A local caller is not subject to a proof budget.
    pub fn accept_block(&self, block: Block) -> Result<BlockOutcome, NodeError> {
        self.accept_block_from(block, None)
    }

    /// As `accept_block`. A peer session passes its budget, which is spent
    /// only when the block reaches the proof check: one token for the proof
    /// of the block and one for each share (spec/node.md section 8.1). The
    /// proof and the shares are checked on the models of the parent and the
    /// grandparent, so a block whose parent is unknown, or whose parent model
    /// is not kept, is refused as an orphan.
    ///
    /// Order of checks, cheap first: duplicate, known parent, header rules,
    /// share list shape, body commitment, models, budget, then `verify_block`
    /// (proof of work, shares, step). The transactions are checked when the
    /// block connects.
    pub fn accept_block_from(&self, block: Block, budget: Option<&mut TokenBucket>) -> Result<BlockOutcome, NodeError> {
        let id = block.id();
        let (height, parent, models) = {
            let inner = self.inner.lock().expect("node lock");
            if inner.blocks.contains_key(&id) {
                return Ok(BlockOutcome::AlreadyKnown);
            }
            let Some(parent) = inner.blocks.get(&block.header.parent).map(|b| b.header) else {
                return Err(NodeError::Orphan);
            };
            let parent_height = *inner.heights.get(&block.header.parent).ok_or(NodeError::Orphan)?;
            self.check_header_cheap(&inner, &block.header, &parent, parent_height, now())?;
            check_shares_list(&block.shares).map_err(|e| NodeError::BadProof(e.into()))?;
            if !block.body_ok() {
                return Err(NodeError::Connect("body_root does not commit to the body".into()));
            }
            let model_p = models_of(&inner, &block.header.parent).ok_or(NodeError::Orphan)?;
            let model_pp = if parent_height == 0 {
                model_p.clone()
            } else {
                models_of(&inner, &parent.parent).ok_or(NodeError::Orphan)?
            };
            (parent_height + 1, parent, (model_p, model_pp))
        };
        // The node lock is released: waiting for tokens may take seconds.
        if let (true, Some(budget)) = (self.verify_pow, budget) {
            for _ in 0..1 + block.shares.len() {
                budget.wait();
            }
        }
        let model = neb_chain::verify_block(
            &self.network.chain,
            self.verify_pow,
            &block,
            &parent,
            height,
            &models.0,
            &models.1,
        )
        .map_err(|e| NodeError::BadProof(e.to_string()))?;

        // Another thread may have accepted the same block while the lock was free.
        let mut inner = self.inner.lock().expect("node lock");
        if inner.blocks.contains_key(&id) {
            return Ok(BlockOutcome::AlreadyKnown);
        }
        let parent_work = *inner.block_work.get(&block.header.parent).ok_or(NodeError::Orphan)?;
        let one = ChainWork::of_bits(block.header.bits).map_err(|e| NodeError::Header(format!("target: {e}")))?;
        let cumulative = parent_work + one;

        if inner.state.tip_id() == block.header.parent {
            inner.state.connect_verified(&block, model).map_err(|e| NodeError::Connect(e.to_string()))?;
            inner.store.connect(&block)?;
            inner.blocks.insert(id, block.clone());
            inner.heights.insert(id, height);
            inner.block_work.insert(id, cumulative);
            inner.best_index.insert(id, height);
            inner.mempool.remove_confirmed(&block.txs);
            inner.last_block_at = now();
            let h = inner.state.height();
            drop(inner);
            self.push_job_to_miners();
            self.announce(InvKind::Block, &[id], None);
            return Ok(BlockOutcome::Connected { height: h });
        }

        // Side branch: keep the block and its model, then compare cumulative
        // work with the best chain. Equal work keeps the tip. A branch that
        // would remove more than 60 blocks stays a side branch whatever its
        // work (spec/node.md section 6.5).
        inner.store.put_block(&block)?;
        inner.blocks.insert(id, block.clone());
        inner.heights.insert(id, height);
        inner.block_work.insert(id, cumulative);
        inner.side_models.insert(id, model);
        prune_side_models(&mut inner);

        let best_work = *inner.block_work.get(&inner.state.tip_id()).expect("the tip has work");
        if cumulative <= best_work {
            drop(inner);
            return Ok(BlockOutcome::Side { height });
        }
        let Some(fork_height) = fork_height(&inner, &block.header.parent) else {
            drop(inner);
            return Ok(BlockOutcome::Side { height });
        };
        let depth = inner.state.height().saturating_sub(fork_height);
        if depth > MAX_REORG {
            drop(inner);
            return Ok(BlockOutcome::Side { height });
        }
        let mut branch = Vec::new();
        let mut cur = block.clone();
        loop {
            branch.push(cur.clone());
            if cur.header.parent == inner.state.id_at(fork_height) {
                break;
            }
            let Some(parent) = inner.blocks.get(&cur.header.parent).cloned() else {
                drop(inner);
                return Ok(BlockOutcome::Side { height });
            };
            cur = parent;
        }
        branch.reverse();
        let old_height = inner.state.height();
        let old_ids: Vec<[u8; 32]> = (fork_height + 1..=old_height).rev().map(|h| inner.state.id_at(h)).collect();
        let removed = inner.state.reorg(fork_height, &branch).map_err(|e| NodeError::Reorg(e.to_string()))?;
        // Disconnected blocks stay known, with their models, as side blocks.
        for old in &old_ids {
            inner.store.disconnect(old)?;
            inner.best_index.remove(old);
        }
        for (offset, b) in branch.iter().enumerate() {
            inner.store.connect(b)?;
            inner.best_index.insert(b.id(), fork_height + 1 + offset as u32);
            inner.side_models.remove(&b.id());
            inner.mempool.remove_confirmed(&b.txs);
        }
        for (old_id, old_model) in removed {
            inner.side_models.insert(old_id, old_model);
        }
        prune_side_models(&mut inner);
        inner.last_block_at = now();
        let new_height = inner.state.height();
        let connected = branch.len() as u32;
        let disconnected = old_ids.len() as u32;
        let tip = inner.state.tip_id();
        drop(inner);
        self.push_job_to_miners();
        self.announce(InvKind::Block, &[tip], None);
        Ok(BlockOutcome::Reorged { height: new_height, disconnected, connected })
    }

    /// Builds a work-service job on the tip for the given miner lock; `now` is
    /// Unix seconds and sets `time_start`. Returns the job id.
    pub fn make_job(&self, miner_lock: [u8; 32], now: u32) -> Result<u64, String> {
        let mut inner = self.inner.lock().expect("node lock");
        let NodeInner { state, mempool, work, .. } = &mut *inner;
        work.make_job(state, mempool, miner_lock, now)
    }

    /// The frames a miner needs to work on job `id`: the model frame of the
    /// parent model, then the job (spec/shares.md section 11.5). One buffer,
    /// so that a session never holds a job without its model.
    pub fn work_frames(&self, id: u64) -> Option<Vec<u8>> {
        let inner = self.inner.lock().expect("node lock");
        let template = inner.work.job(id)?;
        let mut frames = neb_work::model_frame(template.slot, &template.model);
        frames.extend_from_slice(&template.job.encode());
        Some(frames)
    }

    /// The model the next block works on.
    pub fn tip_model(&self) -> Arc<Model> {
        self.inner.lock().expect("node lock").state.tip_model()
    }

    /// The state of the tip: one model per slot.
    pub fn tip_models(&self) -> neb_chain::Models {
        self.inner.lock().expect("node lock").state.tip_models()
    }

    /// The hash of the state of the tip (spec/shares.md section 7): with one slot the weights hash of its model.
    pub fn weights_hash(&self) -> [u8; 32] {
        neb_chain::models_hash(&self.inner.lock().expect("node lock").state.tip_models())
    }

    /// Pushes one fresh job to every connected miner after the best block
    /// changes, whatever its origin (spec/node.md section 7.1).
    fn push_job_to_miners(&self) {
        if self.miners.is_empty() {
            return;
        }
        let Some(lock) = self.miner_lock() else {
            return;
        };
        let Ok(id) = self.make_job(lock, now()) else {
            return;
        };
        let Some(frames) = self.work_frames(id) else {
            return;
        };
        self.miners.broadcast(&frames);
    }

    /// Rebuilds a block from a work submission and accepts it. The error is a
    /// result code of spec/node.md section 7.1: 1 bad submission, 2 bad header
    /// (header rule, proof of work or share), 3 bad state. A block that connects
    /// as a side block is code 3.
    pub fn accept_submit(&self, id: u64, extranonce: &[u8], time: u32, nonce: u32) -> Result<u32, u8> {
        let block = {
            let inner = self.inner.lock().expect("node lock");
            inner.work.submit(id, extranonce, time, nonce).map_err(|_| 1u8)?
        };
        match self.accept_block(block) {
            Ok(BlockOutcome::Connected { height }) => Ok(height),
            Ok(BlockOutcome::Reorged { height, .. }) => Ok(height),
            Ok(_) => Err(3),
            Err(NodeError::Header(_) | NodeError::BadProof(_)) => Err(2),
            Err(_) => Err(3),
        }
    }

    /// Adds a transaction to the mempool and announces it. A transaction
    /// already in the mempool is returned as accepted.
    pub fn submit_tx(&self, tx: Transaction) -> Result<[u8; 32], NodeError> {
        let id = tx.txid();
        let mut inner = self.inner.lock().expect("node lock");
        if inner.mempool.tx(&id).is_some() {
            return Ok(id);
        }
        {
            let NodeInner { state, mempool, .. } = &mut *inner;
            mempool.accept(state, &tx).map_err(|e| NodeError::Connect(e.to_string()))?;
        }
        drop(inner);
        self.announce(InvKind::Tx, &[id], None);
        Ok(id)
    }

    /// Sends an `inv` of `ids` to every peer except the session `except`.
    pub fn announce(&self, kind: InvKind, ids: &[[u8; 32]], except: Option<u64>) {
        if ids.is_empty() {
            return;
        }
        if let Ok(frame) = crate::p2p::inv_frame(self.network.magic, kind, ids) {
            self.peers.broadcast(&frame, except);
        }
    }

    /// The locator of the best chain, newest first: the last 10 heights one by
    /// one, then the gap doubles (2, 4, 8, ...), then height 0. At most 64 ids
    /// (spec/node.md section 8.2) for any chain shorter than 2^54 blocks.
    pub fn locator(&self) -> Vec<[u8; 32]> {
        let inner = self.inner.lock().expect("node lock");
        let mut heights = Vec::new();
        let mut h = inner.state.height();
        for _ in 0..10 {
            heights.push(h);
            if h == 0 {
                break;
            }
            h -= 1;
        }
        let mut step = 2u32;
        while h > 0 {
            heights.push(h);
            h = h.saturating_sub(step);
            step = step.saturating_mul(2);
        }
        if !heights.contains(&0) {
            heights.push(0);
        }
        heights.iter().map(|h| inner.state.id_at(*h)).collect()
    }

    /// Headers of the best chain after the first locator entry that is on it
    /// (the genesis block when none is), at most `max`.
    pub fn headers_after(&self, locator: &[[u8; 32]], max: usize) -> Vec<Header> {
        let inner = self.inner.lock().expect("node lock");
        let start = locator_start(&inner, locator);
        let mut out = Vec::new();
        let mut h = start;
        while h <= inner.state.height() && out.len() < max {
            let id = inner.state.id_at(h);
            if let Some(block) = inner.blocks.get(&id) {
                out.push(block.header);
            }
            h += 1;
        }
        out
    }

    pub fn block_ids_after(&self, locator: &[[u8; 32]], max: usize) -> Vec<[u8; 32]> {
        let inner = self.inner.lock().expect("node lock");
        let start = locator_start(&inner, locator);
        let mut out = Vec::new();
        let mut h = start;
        while h <= inner.state.height() && out.len() < max {
            out.push(inner.state.id_at(h));
            h += 1;
        }
        out
    }

    pub fn height_of(&self, id: &[u8; 32]) -> Option<u32> {
        self.inner.lock().expect("node lock").heights.get(id).copied()
    }

    pub fn now(&self) -> u32 {
        now()
    }
}
