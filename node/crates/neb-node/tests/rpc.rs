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
        dataset_names: &[],
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

// getutxos lists the unspent outputs of a lock on the best chain, and nothing for a lock that holds none.
#[test]
fn getutxos_lists_the_outputs_of_a_lock() {
    let dir = temp_dir("utxos");
    let node = Node::open(NodeConfig { network: network(demo::MINER_LOCK), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    add_chain(&node, 2);
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    let response = call(rpc.addr, "getutxos", vec![Value::String(hex(&demo::MINER_LOCK))]);
    let outputs = result(&response).as_array().unwrap();
    let mut heights: Vec<i64> = outputs.iter().map(|o| o.get("height").unwrap().as_i64().unwrap()).collect();
    heights.sort_unstable();
    assert_eq!(heights, vec![0, 1, 2]);
    for output in outputs {
        let txid = output.get("txid").unwrap().as_str().unwrap();
        let op = OutPoint {
            txid: decode_hex(txid).try_into().unwrap(),
            index: output.get("index").unwrap().as_i64().unwrap() as u32,
        };
        let utxo = node.utxo(&op).expect("the output is unspent");
        assert_eq!(output.get("amount").unwrap().as_i64(), Some(utxo.amount as i64));
        assert_eq!(output.get("locktype").unwrap().as_i64(), Some(utxo.lock_type as i64));
        assert_eq!(output.get("coinbase").unwrap().as_bool(), Some(true));
        assert_eq!(utxo.lock_data, demo::MINER_LOCK);
    }
    // the node stops reading at the limit it is given
    assert_eq!(node.utxos_of(&demo::MINER_LOCK, None, 2).len(), 2);
    assert_eq!(node.utxos_of(&demo::MINER_LOCK, None, 20_000).len(), 3);
    let none = call(rpc.addr, "getutxos", vec![Value::String(hex(&[0x77; 32]))]);
    assert_eq!(result(&none).as_array().unwrap().len(), 0);
    assert_eq!(error_code(&call(rpc.addr, "getutxos", vec![Value::String("12".into())])), -8);
    assert_eq!(error_code(&call(rpc.addr, "getutxos", vec![])), -32602);
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

// getblock with `true` gives the block as an object: where it stands, its header fields and its transactions.
// gettransaction reads a transaction in the block of the height it is given.
#[test]
fn getblock_verbose_and_gettransaction() {
    let dir = temp_dir("verbose");
    let node = Node::open(NodeConfig { network: network(demo::MINER_LOCK), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    let blocks = add_chain(&node, 3);
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    let id = Value::String(hex(&blocks[0].id()));

    let response = call(rpc.addr, "getblock", vec![id.clone(), Value::Bool(true)]);
    let block = result(&response);
    assert_eq!(block.get("hash").unwrap().as_str().unwrap(), hex(&blocks[0].id()));
    assert_eq!(block.get("height").unwrap().as_i64(), Some(1));
    assert_eq!(block.get("mainchain").unwrap().as_bool(), Some(true));
    assert_eq!(block.get("confirmations").unwrap().as_i64(), Some(3));
    assert_eq!(block.get("parent").unwrap().as_str().unwrap(), hex(&blocks[0].header.parent));
    assert_eq!(block.get("time").unwrap().as_i64(), Some(i64::from(blocks[0].header.time)));
    assert_eq!(block.get("bits").unwrap().as_i64(), Some(i64::from(blocks[0].header.bits)));
    let txs = block.get("tx").unwrap().as_array().unwrap();
    assert_eq!(txs.len(), 1);
    let coinbase = &blocks[0].txs[0];
    assert_eq!(txs[0].get("txid").unwrap().as_str().unwrap(), hex(&coinbase.txid()));
    assert_eq!(txs[0].get("coinbase").unwrap().as_bool(), Some(true));
    assert_eq!(txs[0].get("inputs").unwrap().as_array().unwrap().len(), 0);
    let outputs = txs[0].get("outputs").unwrap().as_array().unwrap();
    assert_eq!(outputs.len(), coinbase.outputs.len());
    for (n, output) in outputs.iter().enumerate() {
        assert_eq!(output.get("index").unwrap().as_i64(), Some(n as i64));
        assert_eq!(output.get("amount").unwrap().as_i64(), Some(coinbase.outputs[n].amount as i64));
        assert_eq!(output.get("lock").unwrap().as_str().unwrap(), hex(&coinbase.outputs[n].lock_data));
    }
    // without the flag, or with false, the answer is the hex of 1.0.2
    let plain = call(rpc.addr, "getblock", vec![id.clone(), Value::Bool(false)]);
    assert_eq!(Block::decode(&decode_hex(result(&plain).as_str().unwrap())).unwrap(), blocks[0]);
    assert_eq!(error_code(&call(rpc.addr, "getblock", vec![id, Value::Number(1)])), -32602);

    let txid = Value::String(hex(&coinbase.txid()));
    let response = call(rpc.addr, "gettransaction", vec![txid.clone(), Value::Number(1)]);
    let tx = result(&response);
    assert_eq!(tx.get("height").unwrap().as_i64(), Some(1));
    assert_eq!(tx.get("blockhash").unwrap().as_str().unwrap(), hex(&blocks[0].id()));
    assert_eq!(tx.get("confirmations").unwrap().as_i64(), Some(3));
    assert_eq!(decode_hex(tx.get("hex").unwrap().as_str().unwrap()), coinbase.encode());
    // the wrong block, a height above the tip, and no height for a transaction that is not in the mempool
    assert_eq!(error_code(&call(rpc.addr, "gettransaction", vec![txid.clone(), Value::Number(2)])), -5);
    assert_eq!(error_code(&call(rpc.addr, "gettransaction", vec![txid.clone(), Value::Number(9)])), -8);
    assert_eq!(error_code(&call(rpc.addr, "gettransaction", vec![txid])), -5);
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

// getutxos with a limit reads a lock by pages: each page starts after the last outpoint of the page before.
#[test]
fn getutxos_reads_a_lock_by_pages() {
    let dir = temp_dir("pages");
    let node = Node::open(NodeConfig { network: network(demo::MINER_LOCK), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    add_chain(&node, 6);
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    let lock = Value::String(hex(&demo::MINER_LOCK));
    let all_response = call(rpc.addr, "getutxos", vec![lock.clone()]);
    let all = result(&all_response).as_array().unwrap().to_vec();
    assert_eq!(all.len(), 7);

    let mut paged = Vec::new();
    let mut params = vec![lock.clone(), Value::Number(3)];
    loop {
        let response = call(rpc.addr, "getutxos", params.clone());
        let page = result(&response).as_array().unwrap().to_vec();
        assert!(page.len() <= 3);
        let short = page.len() < 3;
        if let Some(last) = page.last() {
            params = vec![
                lock.clone(),
                Value::Number(3),
                last.get("txid").unwrap().clone(),
                last.get("index").unwrap().clone(),
            ];
        }
        paged.extend(page);
        if short {
            break;
        }
    }
    assert_eq!(paged, all, "the pages are the whole list, in the same order");
    assert_eq!(error_code(&call(rpc.addr, "getutxos", vec![lock.clone(), Value::Number(0)])), -32602);
    assert_eq!(error_code(&call(rpc.addr, "getutxos", vec![lock.clone(), Value::Number(20_001)])), -32602);
    assert_eq!(
        error_code(&call(rpc.addr, "getutxos", vec![lock, Value::Number(3), Value::String("12".into())])),
        -32602
    );
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

// submitshares hands share records to the node without a peer session: a record of the round of the tip is kept
// once, and a record for a parent that is not recent is ignored.
#[test]
fn submitshares_keeps_the_records_of_a_recent_parent() {
    let dir = temp_dir("shares");
    let node = Node::open(NodeConfig { network: network(demo::MINER_LOCK), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    let blocks = add_chain(&node, 3);
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    let tip = Value::String(hex(&blocks[2].id()));
    let record = |n: u8| {
        let mut bytes = [n; 40];
        bytes[32..36].copy_from_slice(&(T0 + 200).to_le_bytes());
        Value::String(hex(&bytes))
    };
    let two = Value::Array(vec![record(1), record(2)]);
    assert_eq!(result(&call(rpc.addr, "submitshares", vec![tip.clone(), two.clone()])).as_i64(), Some(2));
    assert_eq!(
        result(&call(rpc.addr, "submitshares", vec![tip.clone(), two.clone()])).as_i64(),
        Some(0),
        "known records"
    );
    let old = Value::String(hex(&blocks[0].id()));
    assert_eq!(result(&call(rpc.addr, "submitshares", vec![old, two])).as_i64(), Some(0), "not a recent parent");

    assert_eq!(error_code(&call(rpc.addr, "submitshares", vec![tip.clone()])), -32602);
    assert_eq!(error_code(&call(rpc.addr, "submitshares", vec![tip.clone(), Value::Array(vec![])])), -32602);
    let short = Value::Array(vec![Value::String("00".repeat(39))]);
    assert_eq!(error_code(&call(rpc.addr, "submitshares", vec![tip, short])), -22);
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

// submitblock connects a whole block handed over JSON-RPC, and getjob gives the transactions and the share list of a
// job of the work service. A submission on a job the node does not keep has its own result code.
#[test]
fn submitblock_and_getjob() {
    let dir = temp_dir("submitblock");
    let node = Node::open(NodeConfig { network: network(demo::MINER_LOCK), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    let blocks = add_chain(&node, 2);
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");

    let next = demo::block(blocks[1].id(), T0 + 60 * 3, 0xC280_0000, 4003, vec![coinbase(3)]);
    let sent = call(rpc.addr, "submitblock", vec![Value::String(hex(&next.encode()))]);
    assert_eq!(result(&sent).as_str().unwrap(), hex(&next.id()));
    assert_eq!(node.height(), 3);
    assert_eq!(node.tip_id(), next.id());
    // the same block again is known, not an error
    let again = call(rpc.addr, "submitblock", vec![Value::String(hex(&next.encode()))]);
    assert_eq!(result(&again).as_str().unwrap(), hex(&next.id()));
    assert_eq!(error_code(&call(rpc.addr, "submitblock", vec![Value::String("zz".into())])), -22);
    assert_eq!(error_code(&call(rpc.addr, "submitblock", vec![Value::String("00".into())])), -22);
    // a call that carries a block of the largest size is read: this one is answered, with the error of its content
    assert_eq!(error_code(&call(rpc.addr, "submitblock", vec![Value::String("00".repeat(1_000_000))])), -22);
    // a block whose parent the node does not have
    let orphan = demo::block([0x33; 32], T0 + 60 * 9, 0xC280_0000, 4009, vec![coinbase(9)]);
    assert_eq!(error_code(&call(rpc.addr, "submitblock", vec![Value::String(hex(&orphan.encode()))])), -26);

    let id = node.make_job(demo::MINER_LOCK, T0 + 60 * 4).expect("job");
    let response = call(rpc.addr, "getjob", vec![Value::Number(id as i64)]);
    let job = result(&response);
    assert_eq!(job.get("id").unwrap().as_i64(), Some(id as i64));
    assert_eq!(job.get("height").unwrap().as_i64(), Some(4));
    assert_eq!(job.get("parent").unwrap().as_str().unwrap(), hex(&next.id()));
    assert_eq!(job.get("transactions").unwrap().as_array().unwrap().len(), 0);
    assert_eq!(job.get("shares").unwrap().as_array().unwrap().len(), 0);
    assert_eq!(error_code(&call(rpc.addr, "getjob", vec![Value::Number(id as i64 + 1000)])), -5);
    assert_eq!(error_code(&call(rpc.addr, "getjob", vec![])), -32602);

    // result code 4 for a job the node does not keep, 1 for a bad submission on a known job
    assert_eq!(node.accept_submit(id + 1000, &[0u8; 16], T0 + 60 * 4, 1), Err(4));
    assert_eq!(node.accept_submit(id, &[0u8; 3], T0 + 60 * 4, 1), Err(1));
    // a session that connects takes the newest job, and ids do not start at 1
    assert_eq!(node.current_job(demo::MINER_LOCK, T0 + 60 * 4 + 5), Ok(id));
    assert_ne!(node.current_job(demo::MINER_LOCK, T0 + 60 * 4 + 31), Ok(id), "an old job is replaced");
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}

// A call is served when it arrives: 100 calls in a row, each on its own connection, take well under the
// 5 s that a 50 ms wait before each accept would cost. Dropping the handle then ends the accept thread.
#[test]
fn calls_in_a_row_are_served_without_a_wait() {
    let sk = SigningKey::from_bytes(&[5u8; 32]);
    let dir = temp_dir("fast");
    let node =
        Node::open(NodeConfig { network: network(lock_of(&sk)), data_dir: dir.clone(), verify_pow: false }).unwrap();
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    let start = std::time::Instant::now();
    for _ in 0..100 {
        assert_eq!(result(&call(rpc.addr, "getblockcount", vec![])).as_i64(), Some(0));
    }
    let took = start.elapsed();
    assert!(took < std::time::Duration::from_millis(2500), "100 calls took {took:?}");
    let addr = rpc.addr;
    drop(rpc);
    assert!(TcpStream::connect(addr).is_err(), "the listener is closed once the handle is dropped");
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
}
