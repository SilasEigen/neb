//! The wallet's unspent outputs, read with the node's `getutxos` call (spec/node.md section 7.2) instead of a
//! replay of the chain: a few calls for each key, whatever the height. A node that does not know the call
//! (before 1.0.3) is read block by block, as before.

use std::collections::BTreeMap;

use neb_chain::OutPoint;
use neb_chain::tx::{LOCK_MULTISIG, LOCK_SINGLE};
use neb_json::Value;

use crate::rpc::RpcClient;
use crate::wallet::{OwnedUtxo, Scan, Wallet, WalletError, hex, hex32, replay};

/// Outputs asked for in one `getutxos` call.
const PAGE: usize = 1000;
/// Reads of the whole set before giving up, when the tip keeps moving between two reads.
const ATTEMPTS: usize = 4;
/// The error text of a node that does not know a method.
const UNKNOWN_METHOD: &str = "(code -32601)";

fn rpc(text: impl Into<String>) -> WalletError {
    WalletError::Rpc(text.into())
}

fn tip(client: &RpcClient) -> Result<u32, WalletError> {
    let count = client.block_count().map_err(WalletError::Rpc)?;
    u32::try_from(count).map_err(|_| rpc("bad block count"))
}

fn field<'a>(entry: &'a Value, name: &str) -> Result<&'a Value, WalletError> {
    entry.get(name).ok_or_else(|| rpc(format!("an output has no {name}")))
}

fn number<T: TryFrom<i64>>(entry: &Value, name: &str) -> Result<T, WalletError> {
    let n = field(entry, name)?.as_i64().ok_or_else(|| rpc(format!("{name} is not an integer")))?;
    T::try_from(n).map_err(|_| rpc(format!("{name} is out of range")))
}

/// One entry of a `getutxos` answer for the lock `lock_data`.
fn parse(entry: &Value, lock_data: [u8; 32]) -> Result<OwnedUtxo, WalletError> {
    let txid = field(entry, "txid")?.as_str().and_then(hex32).ok_or_else(|| rpc("txid is not 64 hex characters"))?;
    let coinbase = field(entry, "coinbase")?.as_bool().ok_or_else(|| rpc("coinbase is not a boolean"))?;
    Ok(OwnedUtxo {
        outpoint: OutPoint { txid, index: number(entry, "index")? },
        amount: number(entry, "amount")?,
        lock_type: number(entry, "locktype")?,
        lock_data,
        height: number(entry, "height")?,
        coinbase,
    })
}

/// Every unspent output whose lock data is `lock_data` and whose lock type is `lock_type`, read by pages.
fn outputs_of(
    client: &RpcClient,
    lock_type: u8,
    lock_data: [u8; 32],
    into: &mut BTreeMap<OutPoint, OwnedUtxo>,
) -> Result<(), WalletError> {
    let mut after: Option<OutPoint> = None;
    loop {
        let mut params = vec![Value::String(hex(&lock_data)), Value::Number(PAGE as i64)];
        if let Some(last) = after {
            params.push(Value::String(hex(&last.txid)));
            params.push(Value::Number(i64::from(last.index)));
        }
        let answer = client.call("getutxos", params).map_err(WalletError::Rpc)?;
        let page = answer.as_array().ok_or_else(|| rpc("getutxos did not give a list"))?;
        for entry in page {
            let utxo = parse(entry, lock_data)?;
            // The node gives the page in outpoint order: anything else would loop for ever.
            if after.is_some_and(|last| utxo.outpoint <= last) {
                return Err(rpc("getutxos pages are not in order"));
            }
            after = Some(utxo.outpoint);
            // The call selects by lock data alone: an output of another lock type is not the wallet's.
            if utxo.lock_type == lock_type {
                into.insert(utxo.outpoint, utxo);
            }
        }
        if page.len() < PAGE {
            return Ok(());
        }
    }
}

/// The unspent outputs of the wallet keys from `getutxos`. The pages are separate calls, so the set is read
/// again when the tip moved meanwhile.
fn from_utxo_set(client: &RpcClient, wallet: &Wallet) -> Result<Scan, WalletError> {
    let mut tip_height = tip(client)?;
    for _ in 0..ATTEMPTS {
        let mut utxos = BTreeMap::new();
        for key in &wallet.keys {
            outputs_of(client, LOCK_SINGLE, key.lock_data, &mut utxos)?;
            outputs_of(client, LOCK_MULTISIG, key.multisig_lock_data, &mut utxos)?;
        }
        let after = tip(client)?;
        if after == tip_height {
            return Ok(Scan { utxos: utxos.into_values().collect(), tip_height });
        }
        tip_height = after;
    }
    Err(rpc("the chain moved during every read of the outputs: try again"))
}

/// The wallet's unspent outputs on the best chain, in outpoint order. Blocks only: the mempool is not read.
pub fn scan(client: &RpcClient, wallet: &Wallet) -> Result<Scan, WalletError> {
    match from_utxo_set(client, wallet) {
        Err(WalletError::Rpc(text)) if text.ends_with(UNKNOWN_METHOD) => replay(client, wallet),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(json: &str) -> Value {
        neb_json::parse(json).unwrap()
    }

    #[test]
    fn an_entry_becomes_an_owned_output() {
        let txid = "ab".repeat(32);
        let e = entry(&format!(
            r#"{{"txid":"{txid}","index":3,"amount":1900000000,"locktype":1,"height":42,"coinbase":true}}"#
        ));
        let u = parse(&e, [7u8; 32]).unwrap();
        assert_eq!((u.outpoint.txid, u.outpoint.index), ([0xab; 32], 3));
        assert_eq!((u.amount, u.lock_type, u.lock_data, u.height, u.coinbase), (1_900_000_000, 1, [7u8; 32], 42, true));
    }

    #[test]
    fn a_bad_entry_is_refused() {
        let txid = "ab".repeat(32);
        for bad in [
            r#"{"index":0,"amount":1,"locktype":1,"height":1,"coinbase":false}"#.to_string(),
            format!(r#"{{"txid":"{txid}","index":-1,"amount":1,"locktype":1,"height":1,"coinbase":false}}"#),
            format!(r#"{{"txid":"{txid}","index":0,"amount":-5,"locktype":1,"height":1,"coinbase":false}}"#),
            format!(r#"{{"txid":"{txid}","index":0,"amount":1,"locktype":300,"height":1,"coinbase":false}}"#),
            format!(r#"{{"txid":"{txid}","index":0,"amount":1,"locktype":1,"height":1,"coinbase":1}}"#),
            r#"{"txid":"zz","index":0,"amount":1,"locktype":1,"height":1,"coinbase":false}"#.to_string(),
        ] {
            assert!(parse(&entry(&bad), [0u8; 32]).is_err(), "{bad}");
        }
    }
}
