//! JSON-RPC server (spec/node.md section 7.2): the read methods, error codes,
//! and `sendrawtransaction` with a real Ed25519 signature. `verify_pow` is
//! false so that blocks need no ticket search.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ed25519_dalek::{Signer, SigningKey};
use neb_chain::Block;
use neb_chain::demo;
use neb_chain::state::{OutPoint, project_share, subsidy};
use neb_chain::tx::{LOCK_SINGLE, Transaction, TxIn, TxOut};
use neb_json::Value;
use neb_node::params::NetworkParams;
use neb_node::{Node, NodeConfig};

const T0: u32 = 1_700_000_000;
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-rpc-{tag}-{}-{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn lock_of(sk: &SigningKey) -> [u8; 32] {
    *blake3::hash(&sk.verifying_key().to_bytes()).as_bytes()
}

fn network(miner_lock: [u8; 32]) -> NetworkParams {
    NetworkParams {
        name: "test",
        magic: [1, 2, 3, 4],
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
        genesis: demo::genesis_to(T0, miner_lock),
        chain: demo::demo_params(T0, demo::PROJECT_LOCK),
    }
}

fn coinbase(height: u32) -> Transaction {
    let reward = subsidy(height);
    demo::coinbase(height, demo::MINER_LOCK, demo::PROJECT_LOCK, reward - project_share(reward))
}

/// Connects `count` blocks, one per 60 s at the demo pow limit.
fn add_chain(node: &Node, count: u32) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut parent = node.tip_id();
    for height in 1..=count {
        let block = demo::block(parent, T0 + 60 * height, 0xC280_0000, 4000 + height, vec![coinbase(height)]);
        parent = block.id();
        node.accept_block(block.clone()).expect("connect");
        blocks.push(block);
    }
    blocks
}

/// One request per connection, as the server expects (`Connection: close`).
fn call(addr: SocketAddr, method: &str, params: Vec<Value>) -> Value {
    let mut req = BTreeMap::new();
    req.insert("jsonrpc".to_string(), Value::String("1.0".into()));
    req.insert("id".to_string(), Value::Number(7));
    req.insert("method".to_string(), Value::String(method.into()));
    req.insert("params".to_string(), Value::Array(params));
    let body = Value::Object(req).to_string();
    let mut stream = TcpStream::connect(addr).expect("connect");
    let head = format!(
        "POST / HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).unwrap();
    stream.write_all(body.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let text = String::from_utf8(response).unwrap();
    let (_, body) = text.split_once("\r\n\r\n").expect("http body");
    neb_json::parse(body).expect("json response")
}

fn result(response: &Value) -> &Value {
    response.get("result").expect("result")
}

fn error_code(response: &Value) -> i64 {
    response.get("error").and_then(|e| e.get("code")).and_then(Value::as_i64).expect("error code")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// Every read method on a two-block chain, and the error codes for an
// out-of-range height and an unknown method.
#[test]
fn read_methods_and_errors() {
    let dir = temp_dir("read");
    let node = Node::open(NodeConfig { network: network(demo::MINER_LOCK), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    let blocks = add_chain(&node, 2);
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    assert_eq!(result(&call(rpc.addr, "getblockcount", vec![])).as_i64(), Some(2));
    assert_eq!(result(&call(rpc.addr, "getbestblockhash", vec![])).as_str().unwrap(), hex(&blocks[1].id()));
    assert_eq!(result(&call(rpc.addr, "getblockhash", vec![Value::Number(1)])).as_str().unwrap(), hex(&blocks[0].id()));
    let block_response = call(rpc.addr, "getblock", vec![Value::String(hex(&blocks[0].id()))]);
    let block_hex = result(&block_response);
    assert_eq!(Block::decode(&decode_hex(block_hex.as_str().unwrap())).unwrap(), blocks[0]);
    let info_response = call(rpc.addr, "getinfo", vec![]);
    let info = result(&info_response);
    assert_eq!(info.get("blocks").unwrap().as_i64(), Some(2));
    assert_eq!(info.get("network").unwrap().as_str(), Some("test"));
    assert_eq!(info.get("chainid").unwrap().as_str().unwrap(), hex(&node.chain_id()));
    assert_eq!(info.get("miners").unwrap().as_i64(), Some(0));
    assert_eq!(result(&call(rpc.addr, "getrawmempool", vec![])).as_array().unwrap().len(), 0);
    assert_eq!(result(&call(rpc.addr, "getpeerinfo", vec![])).as_array().unwrap().len(), 0);

    let missing = call(rpc.addr, "getblockhash", vec![Value::Number(99)]);
    assert_eq!(error_code(&missing), -8);
    let unknown = call(rpc.addr, "nope", vec![]);
    assert_eq!(error_code(&unknown), -32601);
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

// getmodel gives the packed weights of the tip model (the weights hash of getinfo) and, with "genesis", those of the
// initial model; any other parameter is refused.
#[test]
fn getmodel_gives_the_weights_of_the_tip_and_of_the_genesis() {
    let dir = temp_dir("model");
    let node = Node::open(NodeConfig { network: network(demo::MINER_LOCK), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    let blocks = add_chain(&node, 2);
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    let response = call(rpc.addr, "getmodel", vec![]);
    let tip = result(&response);
    assert_eq!(tip.get("height").unwrap().as_i64(), Some(2));
    assert_eq!(tip.get("blockhash").unwrap().as_str().unwrap(), hex(&blocks[1].id()));
    assert_eq!(tip.get("weightshash").unwrap().as_str().unwrap(), hex(&node.weights_hash()));
    let weights = decode_hex(tip.get("weights").unwrap().as_str().unwrap());
    assert_eq!(weights, node.tip_model().packed_weights());
    assert_eq!(weights.len(), 50_816);

    let response = call(rpc.addr, "getmodel", vec![Value::String("genesis".into())]);
    let genesis = result(&response);
    let initial = neb_pow::shares::Model::from_seed(&node.chain_id());
    assert_eq!(genesis.get("height").unwrap().as_i64(), Some(0));
    assert_eq!(genesis.get("weightshash").unwrap().as_str().unwrap(), hex(&initial.weights_hash()));
    assert_eq!(error_code(&call(rpc.addr, "getmodel", vec![Value::Number(1)])), -32602);
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

fn decode_hex(text: &str) -> Vec<u8> {
    (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap()).collect()
}

// A transaction signed over the sighash of spec/node.md section 5 enters the
// mempool through the RPC; 101 blocks are connected first so the genesis
// coinbase has matured.
#[test]
fn sendrawtransaction_reaches_the_mempool() {
    let sk = SigningKey::from_bytes(&[9u8; 32]);
    let dir = temp_dir("send");
    let node = Node::open(NodeConfig { network: network(lock_of(&sk)), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    add_chain(&node, 101);
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");

    let genesis_cb = demo::genesis_to(T0, lock_of(&sk)).txs[0].txid();
    let op = OutPoint { txid: genesis_cb, index: 0 };
    let spent = node.utxo(&op).expect("utxo").as_txout();
    let mut tx = Transaction {
        version: 1,
        inputs: vec![TxIn { prev_txid: op.txid, prev_index: op.index, witness: Vec::new() }],
        outputs: vec![TxOut { amount: 18 * 100_000_000, lock_type: LOCK_SINGLE, lock_data: [0x55; 32] }],
    };
    let msg = tx.sighash(&node.chain_id(), 0, &spent);
    let sig = sk.sign(&msg);
    let mut witness = Vec::new();
    witness.extend_from_slice(&sk.verifying_key().to_bytes());
    witness.extend_from_slice(&sig.to_bytes());
    tx.inputs[0].witness = witness;
    let txid = tx.txid();
    let sent = call(rpc.addr, "sendrawtransaction", vec![Value::String(hex(&tx.encode()))]);
    assert_eq!(result(&sent).as_str().unwrap(), hex(&txid));
    let pool_response = call(rpc.addr, "getrawmempool", vec![]);
    let pool = result(&pool_response);
    assert_eq!(pool.as_array().unwrap(), &[Value::String(hex(&txid))]);
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}
