//! A payment to one or several recipients in one transaction, with a fee that covers its size.
//!
//! The node relays a transaction that pays at least one base unit per byte (spec/node.md section 6.4). The fee of
//! the wallet, `FEE`, covers 10,000 bytes: about 74 inputs. A payment that needs more inputs, or that has many
//! outputs, pays its size instead, so it is not refused.

use neb_chain::Transaction;
use neb_chain::tx::MAX_OUTPUTS;

use crate::wallet::{FEE, OwnedUtxo, SelectError, Selection, Wallet, WalletError, build_payment, select};

/// Builds and signs the transaction that pays each (lock, amount) of `payments` from `utxos`, with the change to
/// `change_lock`. The fee is `FEE`, or the size of the transaction in bytes when that is larger. At most 254
/// recipients, which leaves an output for the change.
pub fn pay(
    wallet: &Wallet,
    utxos: &[OwnedUtxo],
    payments: &[([u8; 32], u64)],
    chain_id: &[u8; 32],
    change_lock: [u8; 32],
) -> Result<(Transaction, Selection), WalletError> {
    if payments.is_empty() || payments.len() >= MAX_OUTPUTS || payments.iter().any(|p| p.1 == 0) {
        return Err(WalletError::Select(SelectError::ZeroTarget));
    }
    let target = payments
        .iter()
        .try_fold(0u64, |sum, p| sum.checked_add(p.1))
        .ok_or(WalletError::Select(SelectError::Insufficient { available: 0, needed: u64::MAX }))?;
    let mut fee = FEE;
    // A higher fee can take one more input, which makes the transaction larger: the size settles in a few rounds.
    for _ in 0..8 {
        let selection = select(utxos, target, fee).map_err(WalletError::Select)?;
        let tx = build_payment(wallet, &selection, payments, chain_id, change_lock)?;
        let size = tx.encode().len() as u64;
        if selection.fee >= size {
            return Ok((tx, selection));
        }
        fee = size;
    }
    Err(WalletError::Sign("the fee of the payment did not settle".into()))
}

#[cfg(test)]
mod tests {
    use neb_chain::OutPoint;
    use neb_chain::tx::LOCK_SINGLE;

    use super::*;
    use crate::wallet::WalletKey;

    fn wallet() -> Wallet {
        let mut wallet = Wallet::empty("");
        wallet.keys = vec![WalletKey::from_secret([5u8; 32])];
        wallet
    }

    fn owned(wallet: &Wallet, n: u32, amount: u64) -> OwnedUtxo {
        OwnedUtxo {
            outpoint: OutPoint { txid: [4u8; 32], index: n },
            amount,
            lock_type: LOCK_SINGLE,
            lock_data: wallet.keys[0].lock_data,
            height: 1,
            coinbase: false,
        }
    }

    #[test]
    fn one_transaction_pays_several_recipients_in_order() {
        let wallet = wallet();
        let change = wallet.keys[0].lock_data;
        let utxos = [owned(&wallet, 0, 1_000_000_000)];
        let payments = [([1u8; 32], 100_000_000), ([2u8; 32], 250_000_000), ([3u8; 32], 1)];
        let (tx, selection) = pay(&wallet, &utxos, &payments, &[7u8; 32], change).unwrap();
        assert_eq!(tx.outputs.len(), 4);
        for (output, payment) in tx.outputs.iter().zip(&payments) {
            assert_eq!((output.lock_data, output.amount), *payment);
        }
        assert_eq!(selection.fee, FEE);
        assert_eq!(tx.outputs[3].lock_data, change);
        assert_eq!(tx.outputs[3].amount, 1_000_000_000 - 350_000_001 - FEE);
        assert!(tx.check_witness(&[7u8; 32], 0, &selection.inputs[0].as_txout()).is_ok());
    }

    // 120 inputs make a transaction of about 16,000 bytes: the fixed fee of 10,000 base units would be refused.
    #[test]
    fn a_payment_with_many_inputs_pays_its_size() {
        let wallet = wallet();
        let utxos: Vec<OwnedUtxo> = (0..150).map(|n| owned(&wallet, n, 100_000_000)).collect();
        let payments = [([1u8; 32], 120 * 100_000_000 - 50_000)];
        let (tx, selection) = pay(&wallet, &utxos, &payments, &[7u8; 32], wallet.keys[0].lock_data).unwrap();
        let size = tx.encode().len() as u64;
        assert!(size > FEE, "the test needs a transaction larger than the fixed fee, got {size}");
        assert!(selection.fee >= size, "fee {} below one base unit per byte of {size}", selection.fee);
        let paid: u64 = tx.outputs.iter().map(|o| o.amount).sum();
        assert_eq!(paid + selection.fee, selection.inputs.iter().map(|u| u.amount).sum::<u64>());
    }

    #[test]
    fn a_payment_needs_recipients_and_amounts() {
        let wallet = wallet();
        let change = wallet.keys[0].lock_data;
        let utxos = [owned(&wallet, 0, 1_000_000_000)];
        assert!(pay(&wallet, &utxos, &[], &[7u8; 32], change).is_err());
        assert!(pay(&wallet, &utxos, &[([1u8; 32], 0)], &[7u8; 32], change).is_err());
        assert!(pay(&wallet, &utxos, &vec![([1u8; 32], 1); 255], &[7u8; 32], change).is_err());
        assert!(pay(&wallet, &utxos, &[([1u8; 32], 2_000_000_000)], &[7u8; 32], change).is_err());
    }
}
