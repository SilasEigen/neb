//! JSON-RPC methods for software that follows the chain without a copy of it, a pool or a payout system in
//! particular (spec/node.md section 7.2): a block and a transaction as JSON objects, the unspent outputs of a lock
//! by pages, and share records handed to the node without a peer session. None of them changes a consensus rule.

use std::collections::BTreeMap;

use neb_chain::block::{MAX_SHARES_PER_BLOCK, SHARE_RECORD_BYTES};
use neb_chain::{Block, OutPoint, ShareRecord, Transaction, Utxo};
use neb_json::Value;

use crate::node::{Node, NodeError};
use crate::rpc::{RpcError, hex, parse_hex, parse_hex32, required_str};

/// The most outputs that `getutxos` gives in one call.
pub(crate) const MAX_UTXOS: usize = 20_000;

impl Node {
    /// The first `limit` unspent outputs of the best chain whose lock data is `lock`, in outpoint order, after
    /// `after` when one is given.
    pub fn utxos_of(&self, lock: &[u8; 32], after: Option<OutPoint>, limit: usize) -> Vec<(OutPoint, Utxo)> {
        self.inner.lock().expect("node lock").state.utxos_of(lock, after, limit)
    }

    /// As `utxos_of`, for the outputs of the lock type of `only`, if one is given, that hold its amount or more.
    pub fn utxos_where(
        &self,
        lock: &[u8; 32],
        after: Option<OutPoint>,
        limit: usize,
        only: (Option<u8>, u64),
    ) -> Vec<(OutPoint, Utxo)> {
        self.inner.lock().expect("node lock").state.utxos_where(lock, after, limit, only)
    }
}

fn object(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect::<BTreeMap<_, _>>())
}

fn number(n: impl Into<i64>) -> Value {
    Value::Number(n.into())
}

/// A transaction as an object: its id, its inputs and its outputs. Amounts are in base units.
fn tx_json(tx: &Transaction) -> Vec<(&'static str, Value)> {
    let coinbase = tx.is_coinbase();
    // the single input of a coinbase names no output
    let inputs = if coinbase { &[][..] } else { &tx.inputs[..] };
    let inputs = inputs
        .iter()
        .map(|i| object(vec![("txid", Value::String(hex(&i.prev_txid))), ("index", number(i.prev_index))]));
    let outputs = tx.outputs.iter().enumerate().map(|(n, o)| {
        object(vec![
            ("index", number(n as u32)),
            ("amount", Value::Number(o.amount as i64)),
            ("locktype", number(o.lock_type)),
            ("lock", Value::String(hex(&o.lock_data))),
        ])
    });
    vec![
        ("txid", Value::String(hex(&tx.txid()))),
        ("coinbase", Value::Bool(coinbase)),
        ("size", Value::Number(tx.encode().len() as i64)),
        ("inputs", Value::Array(inputs.collect())),
        ("outputs", Value::Array(outputs.collect())),
    ]
}

/// Where a known block stands: its height when the node knows it, whether it is on the best chain, and its
/// confirmations (1 for the tip, -1 on a side branch).
fn standing(node: &Node, id: &[u8; 32]) -> (Option<u32>, bool, i64) {
    let height = node.height_of(id);
    let best = height.is_some_and(|h| node.block_at(h).is_some_and(|b| b.id() == *id));
    let confirmations = match height {
        Some(h) if best => i64::from(node.height().saturating_sub(h)) + 1,
        _ => -1,
    };
    (height, best, confirmations)
}

/// `getblock` with a true second parameter: the header fields, where the block stands and its transactions.
pub(crate) fn block_json(node: &Node, block: &Block) -> Value {
    let id = block.id();
    let (height, best, confirmations) = standing(node, &id);
    let header = &block.header;
    object(vec![
        ("hash", Value::String(hex(&id))),
        ("height", height.map_or(Value::Null, number)),
        ("mainchain", Value::Bool(best)),
        ("confirmations", Value::Number(confirmations)),
        ("version", number(header.version)),
        ("parent", Value::String(hex(&header.parent))),
        ("bodyroot", Value::String(hex(&header.body_root))),
        ("time", number(header.time)),
        ("bits", number(header.bits)),
        ("nonce", number(header.nonce)),
        ("size", Value::Number(block.encode().len() as i64)),
        ("shares", Value::Number(block.shares.len() as i64)),
        ("tx", Value::Array(block.txs.iter().map(|tx| object(tx_json(tx))).collect())),
    ])
}

/// `gettransaction txid [height]`. The node keeps no index of transactions: with a height the transaction is read
/// in the block of that height on the best chain, and without one only the mempool is searched.
pub(crate) fn gettransaction(node: &Node, params: &[Value]) -> Result<Value, RpcError> {
    let text = required_str(params, 0, "transaction id")?;
    let txid = parse_hex32(text).ok_or_else(|| RpcError::new(-8, "transaction id must be 64 hex characters"))?;
    let Some(height) = params.get(1) else {
        let tx = node
            .mempool_tx(&txid)
            .ok_or_else(|| RpcError::new(-5, "transaction not in the mempool; give the height of its block"))?;
        let mut fields = tx_json(&tx);
        fields.push(("confirmations", Value::Number(0)));
        fields.push(("hex", Value::String(hex(&tx.encode()))));
        return Ok(object(fields));
    };
    let height = height
        .as_i64()
        .and_then(|h| u32::try_from(h).ok())
        .ok_or_else(|| RpcError::new(-32602, "height must be an integer"))?;
    let block = node.block_at(height).ok_or_else(|| RpcError::new(-8, "block height out of range"))?;
    let tx = block
        .txs
        .iter()
        .find(|tx| tx.txid() == txid)
        .ok_or_else(|| RpcError::new(-5, "transaction not found in the block of that height"))?;
    let mut fields = tx_json(tx);
    fields.push(("height", number(height)));
    fields.push(("blockhash", Value::String(hex(&block.id()))));
    fields.push(("confirmations", Value::Number(i64::from(node.height().saturating_sub(height)) + 1)));
    fields.push(("hex", Value::String(hex(&tx.encode()))));
    Ok(object(fields))
}

/// `getutxos lock [limit] [after txid, after index] [locktype] [minamount]`. Without a limit the call gives every output of the lock and
/// fails above `MAX_UTXOS`, as before. With a limit it gives at most that many, after the outpoint that is named:
/// a caller reads a large lock by pages, giving the last outpoint of a page to get the next, until a page is
/// shorter than the limit. Pages are read one call at a time, so a block that connects between two calls can move
/// outputs in or out: read again from the start for an exact view. With a lock type, a minimum amount or both,
/// the pages hold only the outputs of that lock type that hold that amount or more; `null` stands for a page
/// start or a lock type that is not given.
pub(crate) fn getutxos(node: &Node, params: &[Value]) -> Result<Value, RpcError> {
    let text = required_str(params, 0, "lock")?;
    let lock = parse_hex32(text).ok_or_else(|| RpcError::new(-8, "lock must be 64 hex characters"))?;
    let found = match params.get(1) {
        None => {
            let found = node.utxos_of(&lock, None, MAX_UTXOS + 1);
            if found.len() > MAX_UTXOS {
                return Err(RpcError::new(-8, "too many outputs for one call: give a limit and read by pages"));
            }
            found
        }
        Some(limit) => {
            let limit = limit
                .as_i64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| (1..=MAX_UTXOS).contains(n))
                .ok_or_else(|| RpcError::new(-32602, format!("limit must be 1 to {MAX_UTXOS}")))?;
            let after = match (params.get(2), params.get(3)) {
                (None | Some(Value::Null), None | Some(Value::Null)) => None,
                (Some(Value::String(txid)), Some(Value::Number(index))) => Some(OutPoint {
                    txid: parse_hex32(txid).ok_or_else(|| RpcError::new(-8, "after txid must be 64 hex characters"))?,
                    index: u32::try_from(*index).map_err(|_| RpcError::new(-8, "after index out of range"))?,
                }),
                _ => return Err(RpcError::new(-32602, "the page start is a txid and an index")),
            };
            let lock_type = match params.get(4) {
                None | Some(Value::Null) => None,
                Some(t) => Some(
                    t.as_i64()
                        .and_then(|t| u8::try_from(t).ok())
                        .ok_or_else(|| RpcError::new(-32602, "locktype must be 0 to 255"))?,
                ),
            };
            let min_amount = match params.get(5) {
                None | Some(Value::Null) => 0,
                Some(a) => a
                    .as_i64()
                    .and_then(|a| u64::try_from(a).ok())
                    .ok_or_else(|| RpcError::new(-32602, "minamount must be an amount in base units"))?,
            };
            node.utxos_where(&lock, after, limit, (lock_type, min_amount))
        }
    };
    let outputs = found.into_iter().map(|(op, utxo)| {
        object(vec![
            ("txid", Value::String(hex(&op.txid))),
            ("index", number(op.index)),
            ("amount", Value::Number(utxo.amount as i64)),
            ("locktype", number(utxo.lock_type)),
            ("height", number(utxo.height)),
            ("coinbase", Value::Bool(utxo.coinbase)),
        ])
    });
    Ok(Value::Array(outputs.collect()))
}

/// `submitshares parent [record, ...]`: share records of the round of `parent` (spec/shares.md section 11.2), each
/// the 40 bytes `body_root || LE32(time) || LE32(nonce)` in hex. The node checks each new record (one ticket
/// evaluation), keeps it for the share list of a later block and relays it, as it does for a `shares` message of a
/// peer. The result is the number of records that were new; it is 0 when the parent is neither the tip nor the
/// parent of the tip. A record above the share target is an error and nothing of the call is kept.
pub(crate) fn submitshares(node: &Node, params: &[Value]) -> Result<Value, RpcError> {
    let text = required_str(params, 0, "parent")?;
    let parent = parse_hex32(text).ok_or_else(|| RpcError::new(-8, "parent must be 64 hex characters"))?;
    let list =
        params.get(1).and_then(Value::as_array).ok_or_else(|| RpcError::new(-32602, "records must be an array"))?;
    if list.is_empty() || list.len() > MAX_SHARES_PER_BLOCK {
        return Err(RpcError::new(-32602, format!("a call carries 1 to {MAX_SHARES_PER_BLOCK} records")));
    }
    let mut records = Vec::with_capacity(list.len());
    for item in list {
        let bytes = item.as_str().and_then(parse_hex).ok_or_else(|| RpcError::new(-22, "a record is not hex"))?;
        let bytes: [u8; SHARE_RECORD_BYTES] =
            bytes.try_into().map_err(|_| RpcError::new(-22, format!("a record has {SHARE_RECORD_BYTES} bytes")))?;
        records.push(ShareRecord::from_bytes(&bytes));
    }
    match node.accept_peer_shares(parent, &records, None, None) {
        Ok(new) => Ok(Value::Number(new as i64)),
        Err(NodeError::BadProof(why)) => Err(RpcError::new(-26, format!("rejected: {why}"))),
        Err(e) => Err(RpcError::new(-1, e.to_string())),
    }
}

/// `getjob id`: what a job of the work service holds besides its Job frame, so that a pool can rebuild the whole
/// block of a solution and hand it to any node with `submitblock`: the transactions after the coinbase and the share
/// list, in block order. The coinbase is the one the pool built from the prefix and the suffix of the job. An id that
/// is not among the jobs the node keeps is an error.
pub(crate) fn getjob(node: &Node, params: &[Value]) -> Result<Value, RpcError> {
    let id = params
        .first()
        .and_then(Value::as_i64)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| RpcError::new(-32602, "job id must be an integer"))?;
    let inner = node.inner.lock().expect("node lock");
    let template = inner.work.job(id).ok_or_else(|| RpcError::new(-5, "unknown job"))?;
    let txs = template.txs.iter().skip(1).map(|tx| Value::String(hex(&tx.encode())));
    let shares = template.shares.iter().map(|r| Value::String(hex(&r.to_bytes())));
    Ok(object(vec![
        ("id", Value::Number(id as i64)),
        ("height", number(template.job.height)),
        ("parent", Value::String(hex(&template.job.parent))),
        ("transactions", Value::Array(txs.collect())),
        ("shares", Value::Array(shares.collect())),
    ]))
}

/// `submitblock hex`: a whole block (spec/node.md section 4), checked and connected like a block from a peer, then
/// announced. The result is the block id. A block the node already has, or that it keeps on a side branch, is not an
/// error: the result is still its id.
pub(crate) fn submitblock(node: &Node, params: &[Value]) -> Result<Value, RpcError> {
    let text = required_str(params, 0, "block hex")?;
    let bytes = parse_hex(text).ok_or_else(|| RpcError::new(-22, "block is not hex"))?;
    let block = Block::decode(&bytes).map_err(|e| RpcError::new(-22, format!("decode: {e}")))?;
    let id = block.id();
    match node.accept_block(block) {
        Ok(_) => Ok(Value::String(hex(&id))),
        Err(e) => Err(RpcError::new(-26, format!("rejected: {e}"))),
    }
}
