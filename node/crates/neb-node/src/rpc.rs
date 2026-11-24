//! JSON-RPC over HTTP POST (spec/node.md section 7.2) on `std::net` and the
//! `neb-json` reader, one thread per connection, one request per connection.
//! Responses carry both `result` and `error` (the "1.0" shape); the `jsonrpc`
//! member of a request is ignored and batch requests are not supported.
//!
//! The methods are the read path of an explorer or wallet plus
//! `sendrawtransaction`, which is validated as a mempool transaction
//! (spec/node.md section 6.4). The server has no authentication: the caller
//! binds it to loopback. Hex strings are lower-case on output and either case
//! on input.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use neb_chain::tx::Transaction;
use neb_json::Value;

use crate::node::Node;
use crate::rpc_chain;

/// Largest accepted request body and header block, in bytes. The body holds a block of the largest size as
/// hexadecimal text for `submitblock`, and the rest of the call.
const MAX_BODY: usize = 2 * neb_chain::block::MAX_BLOCK_BYTES + 64 * 1024;
const MAX_HEADER: usize = 64 * 1024;

pub struct RpcHandle {
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Drop for RpcHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // The accept thread blocks in `accept`: one connection to the listener wakes it, and it then reads
        // the flag. When that connection cannot be made the thread is left to end with the process.
        let woken = TcpStream::connect_timeout(&wake_addr(self.addr), Duration::from_secs(2)).is_ok();
        if let Some(join) = self.join.take()
            && woken
        {
            let _ = join.join();
        }
    }
}

/// The address that reaches a listener bound to `addr` from this machine: the loopback address of its
/// family when it is bound to every interface.
fn wake_addr(addr: SocketAddr) -> SocketAddr {
    let ip = match addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    SocketAddr::new(ip, addr.port())
}

impl Node {
    /// Starts the RPC listener; the handle stops the accept thread when dropped.
    pub fn start_rpc(&self, addr: SocketAddr) -> io::Result<RpcHandle> {
        let listener = TcpListener::bind(addr)?;
        let local = listener.local_addr()?;
        let node = self.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let join = thread::spawn(move || {
            // A blocking accept: a call is served as soon as it arrives.
            loop {
                let accepted = listener.accept();
                if stop2.load(Ordering::Relaxed) {
                    break;
                }
                match accepted {
                    Ok((stream, _)) => {
                        let node = node.clone();
                        thread::spawn(move || {
                            let _ = handle_connection(stream, node);
                        });
                    }
                    // A connection that ended before it was accepted, or no descriptor left for it:
                    // the listener still works, so wait a little and accept again.
                    Err(_) => thread::sleep(Duration::from_millis(50)),
                }
            }
        });
        Ok(RpcHandle { addr: local, stop, join: Some(join) })
    }
}

/// Serves one request. A malformed HTTP request is dropped without an answer;
/// a JSON-RPC error is an HTTP 200 with an `error` member. The 10 s timeouts bound a stalled client.
fn handle_connection(mut stream: TcpStream, node: Node) -> io::Result<()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let (request_line, body) = match read_request(&mut stream) {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    if !request_line.starts_with("POST ") {
        let response = "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        stream.write_all(response.as_bytes())?;
        return Ok(());
    }
    let (result, id) = match neb_json::parse(std::str::from_utf8(&body).unwrap_or("")) {
        Ok(request) => {
            let id = request.get("id").cloned().unwrap_or(Value::Null);
            (dispatch(&node, &request), id)
        }
        Err(e) => (Err(RpcError::new(-32700, format!("parse error: {e}"))), Value::Null),
    };
    let response = jsonrpc_response(result, id);
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

/// Reads the request line and the body. Sizes are bounded before any
/// allocation that depends on the client: headers at `MAX_HEADER` while
/// scanning, then `Content-Length` against `MAX_BODY`.
fn read_request(stream: &mut TcpStream) -> io::Result<(String, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = find_bytes(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > MAX_HEADER {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "request headers too large"));
        }
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short request"));
        }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let request_line = head.lines().next().unwrap_or("").trim().to_string();
    let mut content_length = 0usize;
    for line in head.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else { continue };
        if name.eq_ignore_ascii_case("content-length") {
            content_length =
                value.trim().parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad content length"))?;
        }
    }
    if content_length > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "request body too large"));
    }
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short body"));
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_length);
    Ok((request_line, body))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// JSON-RPC error. Codes: -32700 parse error, -32600 invalid request, -32601
/// unknown method, -32602 invalid parameters (JSON-RPC specification); -5 not found, -8
/// out of range, -22 undecodable transaction, -26 transaction rejected (a
/// common numbering of these conditions).
#[derive(Debug, Clone)]
pub(crate) struct RpcError {
    code: i64,
    message: String,
}

impl RpcError {
    pub(crate) fn new(code: i64, message: impl Into<String>) -> RpcError {
        RpcError { code, message: message.into() }
    }
}

fn jsonrpc_response(result: Result<Value, RpcError>, id: Value) -> String {
    let mut out = BTreeMap::new();
    match result {
        Ok(value) => {
            out.insert("result".to_string(), value);
            out.insert("error".to_string(), Value::Null);
        }
        Err(e) => {
            out.insert("result".to_string(), Value::Null);
            let mut error = BTreeMap::new();
            error.insert("code".to_string(), Value::Number(e.code));
            error.insert("message".to_string(), Value::String(e.message));
            out.insert("error".to_string(), Value::Object(error));
        }
    }
    out.insert("id".to_string(), id);
    Value::Object(out).to_string()
}

fn params(request: &Value) -> &[Value] {
    request.get("params").and_then(Value::as_array).unwrap_or(&[])
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn parse_hex32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

pub(crate) fn parse_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(text.len() / 2);
    for i in 0..text.len() / 2 {
        out.push(u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?);
    }
    Some(out)
}

pub(crate) fn required_str<'a>(params: &'a [Value], index: usize, name: &str) -> Result<&'a str, RpcError> {
    params.get(index).and_then(Value::as_str).ok_or_else(|| RpcError::new(-32602, format!("{name} must be a string")))
}

fn required_height(params: &[Value]) -> Result<u32, RpcError> {
    let value =
        params.first().and_then(Value::as_i64).ok_or_else(|| RpcError::new(-32602, "height must be an integer"))?;
    u32::try_from(value).map_err(|_| RpcError::new(-8, "block height out of range"))
}

fn dispatch(node: &Node, request: &Value) -> Result<Value, RpcError> {
    let method =
        request.get("method").and_then(Value::as_str).ok_or_else(|| RpcError::new(-32600, "method is missing"))?;
    let params = params(request);
    match method {
        "getblockcount" => Ok(Value::Number(node.height() as i64)),
        "getbestblockhash" => Ok(Value::String(hex(&node.tip_id()))),
        // Height 0..=tip of the best chain.
        "getblockhash" => {
            let height = required_height(params)?;
            if height > node.height() {
                return Err(RpcError::new(-8, "block height out of range"));
            }
            let block = node.block_at(height).ok_or_else(|| RpcError::new(-8, "block height out of range"))?;
            Ok(Value::String(hex(&block.id())))
        }
        // Any known block by id, side branches included; the result is the block
        // encoding of spec/node.md section 4 in hex, or with a true second parameter
        // an object with its fields, where it stands and its transactions.
        "getblock" => {
            let text = required_str(params, 0, "block hash")?;
            let id = parse_hex32(text).ok_or_else(|| RpcError::new(-8, "block hash must be 64 hex characters"))?;
            let block = node.block(&id).ok_or_else(|| RpcError::new(-5, "block not found"))?;
            match params.get(1) {
                None | Some(Value::Bool(false)) => Ok(Value::String(hex(&block.encode()))),
                Some(Value::Bool(true)) => Ok(rpc_chain::block_json(node, &block)),
                Some(_) => Err(RpcError::new(-32602, "the second parameter of getblock is true or false")),
            }
        }
        "gettransaction" => rpc_chain::gettransaction(node, params),
        "getjob" => rpc_chain::getjob(node, params),
        "submitblock" => rpc_chain::submitblock(node, params),
        "submitshares" => rpc_chain::submitshares(node, params),
        "getrawmempool" => Ok(Value::Array(node.mempool_ids().iter().map(|id| Value::String(hex(id))).collect())),
        "getpeerinfo" => Ok(Value::Array(
            node.peer_info()
                .into_iter()
                .map(|p| {
                    let mut map = BTreeMap::new();
                    map.insert("id".to_string(), Value::Number(p.id as i64));
                    map.insert("addr".to_string(), Value::String(p.address));
                    map.insert("inbound".to_string(), Value::Bool(p.inbound));
                    map.insert("version".to_string(), Value::Number(p.version as i64));
                    map.insert("subver".to_string(), Value::String(p.user_agent));
                    map.insert("height".to_string(), Value::Number(p.height as i64));
                    map.insert("handshake".to_string(), Value::Bool(p.handshake));
                    map.insert("connected".to_string(), Value::Number(p.connected_secs as i64));
                    Value::Object(map)
                })
                .collect(),
        )),
        // `bits` is the compact target of the tip; `weightshash` is the hash of
        // the state of the tip (spec/shares.md section 7), `slots` its models.
        "getinfo" => {
            let net = node.network();
            let mut map = BTreeMap::new();
            map.insert("version".to_string(), Value::String(env!("CARGO_PKG_VERSION").into()));
            map.insert("protocolversion".to_string(), Value::Number(net.protocol_version as i64));
            map.insert("network".to_string(), Value::String(net.name.into()));
            map.insert("testnet".to_string(), Value::Bool(net.name == crate::params::NAME));
            map.insert("blocks".to_string(), Value::Number(node.height() as i64));
            map.insert("bestblockhash".to_string(), Value::String(hex(&node.tip_id())));
            map.insert("chainid".to_string(), Value::String(hex(&node.chain_id())));
            map.insert("connections".to_string(), Value::Number(node.peer_count() as i64));
            map.insert("miners".to_string(), Value::Number(node.miner_count() as i64));
            let bits = node.header_at(node.height()).map(|h| h.bits).unwrap_or(net.genesis.header.bits);
            map.insert("bits".to_string(), Value::Number(bits as i64));
            map.insert("weightshash".to_string(), Value::String(hex(&node.weights_hash())));
            map.insert("slots".to_string(), Value::Number(net.chain.slots() as i64));
            map.insert("workport".to_string(), Value::Number(net.work_port as i64));
            map.insert("p2pport".to_string(), Value::Number(net.p2p_port as i64));
            Ok(Value::Object(map))
        }
        // The packed weights (spec/shares.md section 7) of the model of one slot in the state of the tip, or with
        // "genesis" in the initial state; the residuals are not included. Parameters: none (slot 0), a slot number,
        // "genesis" (slot 0), or a slot number and "genesis". Read-only, for explorers.
        "getmodel" => {
            let chain = &node.network().chain;
            let k = chain.slots();
            let (slot, genesis) = match (params.first(), params.get(1)) {
                (None, None) => (0, false),
                (Some(Value::String(g)), None) if g == "genesis" => (0, true),
                (Some(Value::Number(n)), None) => (*n, false),
                (Some(Value::Number(n)), Some(Value::String(g))) if g == "genesis" => (*n, true),
                _ => return Err(RpcError::new(-32602, "getmodel takes [slot] and optionally \"genesis\"")),
            };
            if slot < 0 || slot as usize >= k {
                return Err(RpcError::new(-32602, format!("slot must be 0 to {}", k - 1)));
            }
            let slot = slot as usize;
            let (height, id, models) = if genesis {
                (0, node.chain_id(), chain.genesis_models(&node.chain_id()))
            } else {
                loop {
                    // the tip is read again after its state, so the three values belong to the same block
                    let id = node.tip_id();
                    let models = node.tip_models();
                    if node.tip_id() == id {
                        break (node.height_of(&id).unwrap_or(0), id, models);
                    }
                }
            };
            let model = &models[slot];
            let mut map = BTreeMap::new();
            map.insert("height".to_string(), Value::Number(height as i64));
            map.insert("blockhash".to_string(), Value::String(hex(&id)));
            map.insert("slot".to_string(), Value::Number(slot as i64));
            map.insert("slots".to_string(), Value::Number(k as i64));
            // the dataset this model is mined and trained on (an upgrade changes it from a height on)
            let name = node.network().dataset_names.get(chain.dataset_index_of(slot, height)).copied().unwrap_or("");
            map.insert("dataset".to_string(), Value::String(name.into()));
            map.insert("weightshash".to_string(), Value::String(hex(&model.weights_hash())));
            map.insert("weights".to_string(), Value::String(hex(&model.packed_weights())));
            Ok(Value::Object(map))
        }
        // The unspent outputs of the best chain with that lock data, for a wallet that keeps no copy of the chain.
        "getutxos" => rpc_chain::getutxos(node, params),
        // Mempool admission only (spec/node.md section 6.4); the node announces
        // the transaction to its peers.
        "sendrawtransaction" => {
            let text = required_str(params, 0, "transaction hex")?;
            let bytes = parse_hex(text).ok_or_else(|| RpcError::new(-22, "transaction is not hex"))?;
            let tx = Transaction::decode(&bytes).map_err(|e| RpcError::new(-22, format!("decode: {e}")))?;
            let id = node.submit_tx(tx).map_err(|e| RpcError::new(-26, format!("rejected: {e}")))?;
            Ok(Value::String(hex(&id)))
        }
        _ => Err(RpcError::new(-32601, format!("method not found: {method}"))),
    }
}
