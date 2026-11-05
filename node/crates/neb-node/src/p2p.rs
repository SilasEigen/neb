//! Peer-to-peer layer (spec/node.md section 8) on `std::net`, one thread per
//! session, blocking sockets.
//!
//! Framing (section 8.2): `magic (4) || command (12) || length (u32 LE) ||
//! checksum (4) || payload`, the checksum being the first 4 bytes of
//! `BLAKE3(payload)`. A framing, checksum or decode error closes the session.
//! The handshake is `version` then `verack`; the session closes on a different
//! genesis id or protocol, or on the node's own nonce.
//!
//! Synchronisation is sequential: `getblocks` -> `inv` -> `getdata` for at most
//! `GETDATA_BLOCKS` blocks -> `block`, then `getblocks` again when the last
//! block of the batch was validated. A block needs the models of its two
//! ancestors, so blocks cannot be validated out of order (spec/node.md
//! section 6.1). New blocks and transactions are relayed with `inv`, new share
//! records with `shares`.
//!
//! Local policy only: no rule here affects consensus. A peer is banned for
//! 24 hours only for a block with an invalid proof of work or share, or a
//! share record above the share target (spec/node.md section 8.1); the budget
//! of ticket evaluations is per session.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use neb_block::Header;
use neb_chain::Block;
use neb_chain::block::{MAX_SHARES_PER_BLOCK, SHARE_RECORD_BYTES, ShareRecord, check_shares_list};
use neb_chain::tx::Transaction;
use neb_codec::{Reader, write_varint};

use crate::budget::TokenBucket;
use crate::node::{BlockOutcome, InvKind, Node, NodeError};

/// Maximum payload of a message in bytes (spec/node.md section 8.2).
pub const MAX_PAYLOAD: usize = 2 * 1024 * 1024;
/// 4 magic + 12 command + 4 length + 4 checksum.
const HEADER_BYTES: usize = 24;

const CMD_VERSION: &str = "version";
const CMD_VERACK: &str = "verack";
const CMD_PING: &str = "ping";
const CMD_PONG: &str = "pong";
const CMD_GETADDR: &str = "getaddr";
const CMD_ADDR: &str = "addr";
const CMD_GETHEADERS: &str = "getheaders";
const CMD_HEADERS: &str = "headers";
/// Blocks requested per `getdata`; the next batch is requested after the last
/// block of this one was validated.
const GETDATA_BLOCKS: usize = 16;
const CMD_GETBLOCKS: &str = "getblocks";
const CMD_INV: &str = "inv";
const CMD_GETDATA: &str = "getdata";
const CMD_NOTFOUND: &str = "notfound";
const CMD_BLOCK: &str = "block";
const CMD_TX: &str = "tx";
const CMD_SHARES: &str = "shares";

#[derive(Clone, Debug)]
pub struct PeerInfo {
    pub id: u64,
    pub address: String,
    pub inbound: bool,
    pub version: u32,
    pub user_agent: String,
    pub height: u32,
    pub connected_secs: u64,
    pub handshake: bool,
}

/// One connected session. The writer is shared so that relays from other
/// threads and the session thread do not interleave frames.
struct PeerEntry {
    info: PeerInfo,
    writer: Arc<Mutex<TcpStream>>,
    started: Instant,
}

/// The connected sessions, at most `limit`; a session beyond the limit is
/// closed at once. Never held while writing: `broadcast` copies the writers out.
pub struct PeerSet {
    limit: usize,
    next_id: AtomicU64,
    peers: Mutex<BTreeMap<u64, PeerEntry>>,
}

impl PeerSet {
    pub fn new(limit: usize) -> PeerSet {
        PeerSet { limit, next_id: AtomicU64::new(1), peers: Mutex::new(BTreeMap::new()) }
    }

    fn add(&self, address: String, inbound: bool, writer: Arc<Mutex<TcpStream>>) -> Option<u64> {
        let mut peers = self.peers.lock().expect("peer lock");
        if peers.len() >= self.limit {
            return None;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        peers.insert(
            id,
            PeerEntry {
                info: PeerInfo {
                    id,
                    address,
                    inbound,
                    version: 0,
                    user_agent: String::new(),
                    height: 0,
                    connected_secs: 0,
                    handshake: false,
                },
                writer,
                started: Instant::now(),
            },
        );
        Some(id)
    }

    fn remove(&self, id: u64) {
        self.peers.lock().expect("peer lock").remove(&id);
    }

    fn update<F: FnOnce(&mut PeerInfo)>(&self, id: u64, f: F) {
        if let Some(entry) = self.peers.lock().expect("peer lock").get_mut(&id) {
            f(&mut entry.info);
        }
    }

    pub fn len(&self) -> usize {
        self.peers.lock().expect("peer lock").len()
    }

    /// True when a session with this exact address is connected.
    pub fn is_connected(&self, addr: &SocketAddr) -> bool {
        let text = addr.to_string();
        self.peers.lock().expect("peer lock").values().any(|e| e.info.address == text)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn info(&self) -> Vec<PeerInfo> {
        let peers = self.peers.lock().expect("peer lock");
        let mut out = Vec::with_capacity(peers.len());
        for entry in peers.values() {
            let mut info = entry.info.clone();
            info.connected_secs = entry.started.elapsed().as_secs();
            out.push(info);
        }
        out
    }

    pub(crate) fn broadcast(&self, frame: &[u8], except: Option<u64>) {
        let writers: Vec<Arc<Mutex<TcpStream>>> = {
            let peers = self.peers.lock().expect("peer lock");
            peers.iter().filter(|(id, _)| Some(**id) != except).map(|(_, e)| e.writer.clone()).collect()
        };
        for writer in writers {
            let _ = write_all(&writer, frame);
        }
    }
}

fn write_all(writer: &Arc<Mutex<TcpStream>>, frame: &[u8]) -> io::Result<()> {
    let mut stream = writer.lock().expect("peer writer");
    stream.write_all(frame)?;
    stream.flush()
}

fn command_bytes(command: &str) -> Result<[u8; 12], String> {
    let bytes = command.as_bytes();
    if bytes.len() > 12 || !bytes.is_ascii() {
        return Err("command must be at most 12 ASCII bytes".into());
    }
    let mut out = [0u8; 12];
    out[..bytes.len()].copy_from_slice(bytes);
    Ok(out)
}

/// Frames `payload` as a message of `command` (spec/node.md section 8.2).
/// Fails when the command exceeds 12 ASCII bytes or the payload 2 MiB.
pub fn encode_message(magic: [u8; 4], command: &str, payload: &[u8]) -> Result<Vec<u8>, String> {
    if payload.len() > MAX_PAYLOAD {
        return Err("payload over 2 MiB".into());
    }
    let mut out = Vec::with_capacity(HEADER_BYTES + payload.len());
    out.extend_from_slice(&magic);
    out.extend_from_slice(&command_bytes(command)?);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&blake3::hash(payload).as_bytes()[..4]);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Fills `buf`. Returns `Ok(false)` when the peer closed the connection. With
/// `idle_ok` a read timeout before the first byte is returned as the I/O
/// error; any other timeout is `InvalidData`.
fn read_full(stream: &mut TcpStream, buf: &mut [u8], idle_ok: bool) -> io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match stream.read(&mut buf[filled..]) {
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut => {
                if idle_ok && filled == 0 {
                    return Err(e);
                }
                return Err(io::Error::new(io::ErrorKind::InvalidData, "read timeout inside a message"));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

pub struct Message {
    pub command: String,
    pub payload: Vec<u8>,
}

/// Reads one message. `Ok(None)` is a clean close between messages. The order
/// of checks bounds the allocation: the length is checked against
/// `MAX_PAYLOAD` before the payload buffer exists, and magic before length.
/// A read timeout before the first byte of a message is returned as is (the
/// session is idle); a timeout after part of a header or payload was read is
/// an `InvalidData` error, because the stream is then in the middle of a frame.
fn read_message(stream: &mut TcpStream, magic: [u8; 4]) -> io::Result<Option<Message>> {
    let mut header = [0u8; HEADER_BYTES];
    if !read_full(stream, &mut header, true)? {
        return Ok(None);
    }
    if header[..4] != magic {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "wrong network magic"));
    }
    let command_end = header[4..16].iter().position(|b| *b == 0).unwrap_or(12);
    if !header[4..4 + command_end].is_ascii() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "command is not ASCII"));
    }
    let command = String::from_utf8(header[4..4 + command_end].to_vec()).unwrap();
    let len = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
    if len > MAX_PAYLOAD {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "payload over 2 MiB"));
    }
    let mut payload = vec![0u8; len];
    if !read_full(stream, &mut payload, false)? {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    let checksum = u32::from_le_bytes(header[20..24].try_into().unwrap());
    if checksum != u32::from_le_bytes(blake3::hash(&payload).as_bytes()[..4].try_into().unwrap()) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "payload checksum"));
    }
    Ok(Some(Message { command, payload }))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionMsg {
    pub protocol: u32,
    pub height: u32,
    pub nonce: u64,
    pub genesis: [u8; 32],
    pub port: u16,
    pub user_agent: String,
}

impl VersionMsg {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.protocol.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.nonce.to_le_bytes());
        out.extend_from_slice(&self.genesis);
        out.extend_from_slice(&self.port.to_le_bytes());
        write_varint(&mut out, self.user_agent.len() as u64);
        out.extend_from_slice(self.user_agent.as_bytes());
        out
    }

    fn decode(payload: &[u8]) -> Result<VersionMsg, String> {
        let mut r = Reader::new(payload);
        let shape = |e: neb_codec::CodecError| e.to_string();
        let protocol = r.read_u32().map_err(shape)?;
        let height = r.read_u32().map_err(shape)?;
        let nonce = r.read_u64().map_err(shape)?;
        let genesis: [u8; 32] = r.read_exact(32).map_err(shape)?.try_into().unwrap();
        let port = r.read_u16().map_err(shape)?;
        let n = r.read_varint().map_err(shape)?;
        // The length is checked before it is used to read, so it cannot
        // request an oversized read.
        if n > 256 {
            return Err("user agent too long".into());
        }
        let user_agent = String::from_utf8(r.read_exact(n as usize).map_err(shape)?.to_vec())
            .map_err(|_| "user agent is not utf8".to_string())?;
        r.expect_done().map_err(shape)?;
        Ok(VersionMsg { protocol, height, nonce, genesis, port, user_agent })
    }
}

/// Locator payload: `count (varint, at most 64) || ids (32 bytes each)`.
fn locator_payload(locator: &[[u8; 32]]) -> Vec<u8> {
    let mut out = Vec::new();
    write_varint(&mut out, locator.len() as u64);
    for id in locator.iter().take(64) {
        out.extend_from_slice(id);
    }
    out
}

fn parse_locator(payload: &[u8]) -> Result<Vec<[u8; 32]>, String> {
    let mut r = Reader::new(payload);
    let n = r.read_varint().map_err(|e| e.to_string())?;
    if n > 64 {
        return Err("locator too long".into());
    }
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        out.push(r.read_exact(32).map_err(|e| e.to_string())?.try_into().unwrap());
    }
    r.expect_done().map_err(|e| e.to_string())?;
    Ok(out)
}

fn headers_payload(headers: &[Header]) -> Vec<u8> {
    let mut out = Vec::new();
    write_varint(&mut out, headers.len() as u64);
    for h in headers {
        out.extend_from_slice(&h.encode());
    }
    out
}

/// `inv`, `getdata` and `notfound` payload: `count (varint) || (kind (u8), id (32))*`.
fn inv_payload(entries: &[(u8, [u8; 32])]) -> Vec<u8> {
    let mut out = Vec::new();
    write_varint(&mut out, entries.len() as u64);
    for (kind, id) in entries {
        out.push(*kind);
        out.extend_from_slice(id);
    }
    out
}

fn parse_inv(payload: &[u8]) -> Result<Vec<(u8, [u8; 32])>, String> {
    let mut r = Reader::new(payload);
    let n = r.read_varint().map_err(|e| e.to_string())?;
    // 5000 entries of 33 bytes fit in `MAX_PAYLOAD`.
    if n > 5000 {
        return Err("inventory list over 5000".into());
    }
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let kind = r.read_u8().map_err(|e| e.to_string())?;
        if kind != InvKind::Block as u8 && kind != InvKind::Tx as u8 {
            return Err("unknown inventory kind".into());
        }
        let id: [u8; 32] = r.read_exact(32).map_err(|e| e.to_string())?.try_into().unwrap();
        out.push((kind, id));
    }
    r.expect_done().map_err(|e| e.to_string())?;
    Ok(out)
}

/// One `inv` message for the node's relay path.
pub fn inv_frame(magic: [u8; 4], kind: InvKind, ids: &[[u8; 32]]) -> Result<Vec<u8>, String> {
    let entries: Vec<(u8, [u8; 32])> = ids.iter().take(5000).map(|id| (kind as u8, *id)).collect();
    encode_message(magic, CMD_INV, &inv_payload(&entries))
}

/// `shares` payload: `parent (32) || count (varint) || record (40)*`, the
/// records in strictly increasing order (spec/node.md section 8.2).
fn shares_payload(parent: &[u8; 32], records: &[ShareRecord]) -> Vec<u8> {
    let mut out = parent.to_vec();
    write_varint(&mut out, records.len() as u64);
    for r in records {
        out.extend_from_slice(&r.to_bytes());
    }
    out
}

/// Decodes a `shares` payload. Refuses more than 256 records, records that are
/// not strictly increasing and any trailing byte.
fn parse_shares(payload: &[u8]) -> Result<([u8; 32], Vec<ShareRecord>), String> {
    let mut r = Reader::new(payload);
    let parent: [u8; 32] = r.read_exact(32).map_err(|e| e.to_string())?.try_into().unwrap();
    let n = r.read_varint().map_err(|e| e.to_string())?;
    if n > MAX_SHARES_PER_BLOCK as u64 {
        return Err("more than 256 share records".into());
    }
    let mut records = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let bytes: [u8; SHARE_RECORD_BYTES] =
            r.read_exact(SHARE_RECORD_BYTES).map_err(|e| e.to_string())?.try_into().unwrap();
        records.push(ShareRecord::from_bytes(&bytes));
    }
    r.expect_done().map_err(|e| e.to_string())?;
    check_shares_list(&records)?;
    Ok((parent, records))
}

/// One `shares` message for the relay path: at most 256 records, sorted.
pub fn shares_frame(magic: [u8; 4], parent: &[u8; 32], records: &[ShareRecord]) -> Result<Vec<u8>, String> {
    let records = &records[..records.len().min(MAX_SHARES_PER_BLOCK)];
    encode_message(magic, CMD_SHARES, &shares_payload(parent, records))
}

/// The node keeps no address book: `getaddr` is answered with zero entries.
fn empty_addr() -> Vec<u8> {
    let mut out = Vec::new();
    write_varint(&mut out, 0);
    out
}

pub struct P2PHandle {
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Drop for P2PHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Node {
    /// Starts the inbound listener. The returned handle stops the accept
    /// thread when dropped; peer threads end when their sockets close. A banned
    /// address is closed before any byte is read.
    pub fn start_p2p(&self, addr: SocketAddr) -> io::Result<P2PHandle> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let local = listener.local_addr()?;
        let node = self.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let join = thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, peer)) => {
                        if node.is_banned(peer.ip()) {
                            continue;
                        }
                        let _ = stream.set_nonblocking(false);
                        let n = node.clone();
                        thread::spawn(move || serve_peer(n, stream, true, peer));
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(P2PHandle { addr: local, stop, join: Some(join) })
    }

    /// Connects to one peer synchronously (the socket is opened here) and
    /// hands it to a peer thread. Returns an error when the connection fails
    /// or the peer is banned.
    pub fn connect_peer(&self, addr: SocketAddr) -> io::Result<()> {
        if self.is_banned(addr.ip()) {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "the peer is banned"));
        }
        let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
        let node = self.clone();
        thread::spawn(move || serve_peer(node, stream, false, addr));
        Ok(())
    }

    pub fn mempool_tx(&self, id: &[u8; 32]) -> Option<Transaction> {
        self.inner.lock().expect("node lock").mempool.tx(id).cloned()
    }
}

/// Sends `getblocks` with the locator of the best chain.
fn send_getblocks(node: &Node, writer: &Arc<Mutex<TcpStream>>) {
    let magic = node.network().magic;
    if let Ok(frame) = encode_message(magic, CMD_GETBLOCKS, &locator_payload(&node.locator())) {
        let _ = write_all(writer, &frame);
    }
}

fn serve_peer(node: Node, mut stream: TcpStream, inbound: bool, peer_addr: SocketAddr) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let writer = match stream.try_clone() {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(_) => return,
    };
    let address = peer_addr.to_string();
    let Some(peer_id) = node.peers.add(address, inbound, writer.clone()) else {
        return;
    };
    let magic = node.network().magic;
    let version = VersionMsg {
        protocol: node.network().protocol_version,
        height: node.height(),
        nonce: node.nonce(),
        genesis: node.chain_id(),
        port: node.network().p2p_port,
        user_agent: node.network().user_agent.to_string(),
    };
    if let Ok(frame) = encode_message(magic, CMD_VERSION, &version.encode())
        && write_all(&writer, &frame).is_err()
    {
        node.peers.remove(peer_id);
        return;
    }
    // `batch_last` is the id of the last block requested in the current
    // `getdata`; its arrival triggers the next `getblocks`.
    let mut handshake_done = false;
    let mut batch_last: Option<[u8; 32]> = None;
    let mut their_version: Option<VersionMsg> = None;
    let mut budget = TokenBucket::new(node.pow_budget, Instant::now());
    // True from a keep-alive ping until the next message of any kind.
    let mut ping_unanswered = false;
    'session: loop {
        let message = match read_message(&mut stream, magic) {
            Ok(Some(m)) => {
                ping_unanswered = false;
                m
            }
            Ok(None) => break,
            // Idle for 60 s: keep-alive ping; idle for 60 s more: disconnect.
            Err(e) if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut => {
                if ping_unanswered {
                    break;
                }
                ping_unanswered = true;
                if let Ok(frame) = encode_message(magic, CMD_PING, &0u64.to_le_bytes()) {
                    let _ = write_all(&writer, &frame);
                }
                continue;
            }
            Err(_) => break,
        };
        match message.command.as_str() {
            CMD_VERSION => {
                let Ok(v) = VersionMsg::decode(&message.payload) else { break };
                if v.genesis != node.chain_id()
                    || v.protocol != node.network().protocol_version
                    || v.nonce == node.nonce()
                {
                    break;
                }
                their_version = Some(v.clone());
                node.peers.update(peer_id, |info| {
                    info.version = v.protocol;
                    info.height = v.height;
                    info.user_agent = v.user_agent.clone();
                });
                if let Ok(frame) = encode_message(magic, CMD_VERACK, &[]) {
                    let _ = write_all(&writer, &frame);
                }
            }
            CMD_VERACK => {
                handshake_done = true;
                node.peers.update(peer_id, |info| info.handshake = true);
                send_getblocks(&node, &writer);
            }
            CMD_PING => {
                if message.payload.len() == 8
                    && let Ok(frame) = encode_message(magic, CMD_PONG, &message.payload)
                {
                    let _ = write_all(&writer, &frame);
                }
            }
            CMD_PONG => {}
            // Before `verack`, only version, verack, ping and pong are processed.
            _ if !handshake_done => {}
            CMD_GETHEADERS => {
                let Ok(locator) = parse_locator(&message.payload) else { break };
                let headers = node.headers_after(&locator, 2000);
                if let Ok(frame) = encode_message(magic, CMD_HEADERS, &headers_payload(&headers)) {
                    let _ = write_all(&writer, &frame);
                }
            }
            CMD_GETBLOCKS => {
                let Ok(locator) = parse_locator(&message.payload) else { break };
                let ids = node.block_ids_after(&locator, 500);
                let entries: Vec<(u8, [u8; 32])> = ids.iter().map(|id| (InvKind::Block as u8, *id)).collect();
                if let Ok(frame) = encode_message(magic, CMD_INV, &inv_payload(&entries)) {
                    let _ = write_all(&writer, &frame);
                }
            }
            CMD_INV => {
                let Ok(entries) = parse_inv(&message.payload) else { break };
                // At most 500 entries are requested, of which at most
                // `GETDATA_BLOCKS` blocks.
                let mut wanted = Vec::new();
                let mut blocks_wanted = 0;
                for (kind, id) in entries {
                    if wanted.len() >= 500 {
                        break;
                    }
                    if kind == InvKind::Block as u8 {
                        if blocks_wanted < GETDATA_BLOCKS && !node.has_block(&id) {
                            wanted.push((kind, id));
                            blocks_wanted += 1;
                            batch_last = Some(id);
                        }
                    } else if !node.has_tx(&id) {
                        wanted.push((kind, id));
                    }
                }
                if !wanted.is_empty()
                    && let Ok(frame) = encode_message(magic, CMD_GETDATA, &inv_payload(&wanted))
                {
                    let _ = write_all(&writer, &frame);
                }
            }
            CMD_GETDATA => {
                let Ok(entries) = parse_inv(&message.payload) else { break };
                let mut missing = Vec::new();
                for (kind, id) in entries {
                    if kind == InvKind::Block as u8 {
                        if let Some(block) = node.block(&id) {
                            if let Ok(frame) = encode_message(magic, CMD_BLOCK, &block.encode()) {
                                let _ = write_all(&writer, &frame);
                            }
                        } else {
                            missing.push((kind, id));
                        }
                    } else if let Some(tx) = node.mempool_tx(&id) {
                        if let Ok(frame) = encode_message(magic, CMD_TX, &tx.encode()) {
                            let _ = write_all(&writer, &frame);
                        }
                    } else {
                        missing.push((kind, id));
                    }
                }
                if !missing.is_empty()
                    && let Ok(frame) = encode_message(magic, CMD_NOTFOUND, &inv_payload(&missing))
                {
                    let _ = write_all(&writer, &frame);
                }
            }
            CMD_BLOCK => {
                let Ok(block) = Block::decode(&message.payload) else { break };
                // `solicited`: the block is the last one of the requested batch.
                let solicited = batch_last == Some(block.id());
                if solicited {
                    batch_last = None;
                }
                let mut ask_more = false;
                match node.accept_block_from(block, Some(&mut budget)) {
                    Ok(BlockOutcome::Connected { .. }) | Ok(BlockOutcome::Reorged { .. }) => {
                        let height = node.height();
                        node.peers.update(peer_id, |info| info.height = info.height.max(height));
                        node.announce(InvKind::Block, &[node.tip_id()], Some(peer_id));
                        ask_more = solicited;
                    }
                    Ok(_) => ask_more = solicited,
                    // Invalid proof or share: ban the address (spec/node.md section 8.1).
                    Err(NodeError::BadProof(_)) => {
                        node.bans.ban(peer_addr.ip());
                        break 'session;
                    }
                    // An unsolicited orphan means the peer is ahead: resync once.
                    Err(NodeError::Orphan) => ask_more = !solicited && batch_last.is_none(),
                    Err(_) => {}
                }
                if ask_more {
                    send_getblocks(&node, &writer);
                }
            }
            CMD_TX => {
                let Ok(tx) = Transaction::decode(&message.payload) else { break };
                let _ = node.submit_tx(tx);
            }
            CMD_SHARES => {
                let Ok((parent, records)) = parse_shares(&message.payload) else { break };
                // A record with a ticket above the share target is as bad as a
                // block with an invalid share: ban the address (spec/node.md
                // section 8.1).
                if let Err(NodeError::BadProof(_)) =
                    node.accept_peer_shares(parent, &records, Some(&mut budget), Some(peer_id))
                {
                    node.bans.ban(peer_addr.ip());
                    break 'session;
                }
            }
            CMD_GETADDR => {
                if let Ok(frame) = encode_message(magic, CMD_ADDR, &empty_addr()) {
                    let _ = write_all(&writer, &frame);
                }
            }
            CMD_ADDR => {}
            CMD_NOTFOUND => {
                let _ = parse_inv(&message.payload);
                // The batch will not complete; stop waiting for its last block.
                batch_last = None;
            }
            _ => {}
        }
        if let Some(v) = &their_version
            && handshake_done
        {
            node.peers.update(peer_id, |info| info.height = info.height.max(v.height));
        }
    }
    node.peers.remove(peer_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        server.set_read_timeout(Some(Duration::from_millis(200))).unwrap();
        (client, server)
    }

    // A timeout before the first byte is idle; a timeout inside a header or payload is a framing error.
    #[test]
    fn a_timeout_inside_a_message_is_not_idle() {
        let magic = [1, 2, 3, 4];
        let (_quiet, mut server) = pair();
        let err = read_message(&mut server, magic).err().expect("timeout");
        assert!(matches!(err.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut));

        let frame = encode_message(magic, CMD_PING, &[0u8; 8]).unwrap();
        for cut in [10, HEADER_BYTES + 3] {
            let (mut client, mut server) = pair();
            client.write_all(&frame[..cut]).unwrap();
            let err = read_message(&mut server, magic).err().expect("timeout");
            assert_eq!(err.kind(), io::ErrorKind::InvalidData, "cut at {cut}");
        }

        let (mut client, mut server) = pair();
        client.write_all(&frame).unwrap();
        assert_eq!(read_message(&mut server, magic).unwrap().unwrap().command, CMD_PING);
    }
}
