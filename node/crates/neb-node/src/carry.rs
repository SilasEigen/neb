//! Carry-over of the balances of one chain into the genesis of another.
//!
//! The balance of a lock is the sum of its unspent outputs at a height, coinbase outputs of any age included. The
//! carry-over of a lock is `floor(balance / CARRY_DIVISOR)` in base units. The list is a pure function of the
//! unspent outputs, so anyone who replays the chain to the same height obtains the same list.

use std::collections::BTreeMap;

use neb_chain::Utxo;

/// A lock: its type and its 32 bytes of lock data (spec/node.md section 5.2).
pub type Lock = (u8, [u8; 32]);

/// A lock carries one base unit for every `CARRY_DIVISOR` base units of its balance.
pub const CARRY_DIVISOR: u128 = 10;

/// The carry-over of every lock that holds at least `CARRY_DIVISOR` base units, by increasing lock. The locks of
/// `excluded` carry nothing.
pub fn carry_over<'a>(utxos: impl IntoIterator<Item = &'a Utxo>, excluded: &[Lock]) -> BTreeMap<Lock, u64> {
    let mut balances: BTreeMap<Lock, u128> = BTreeMap::new();
    for utxo in utxos {
        let lock = (utxo.lock_type, utxo.lock_data);
        if !excluded.contains(&lock) {
            *balances.entry(lock).or_default() += u128::from(utxo.amount);
        }
    }
    balances
        .into_iter()
        .filter_map(|(lock, balance)| {
            // The whole supply stays below 2^64 base units, so a tenth of a balance fits in a u64.
            let carried = u64::try_from(balance / CARRY_DIVISOR).expect("a balance is below the supply");
            (carried > 0).then_some((lock, carried))
        })
        .collect()
}

/// The list as text, one line per lock: `lock type, lock data in hexadecimal, base units`.
pub fn carry_lines(list: &BTreeMap<Lock, u64>) -> String {
    let mut text = String::new();
    for ((lock_type, lock_data), amount) in list {
        let hex: String = lock_data.iter().map(|b| format!("{b:02x}")).collect();
        text.push_str(&format!("{lock_type} {hex} {amount}\n"));
    }
    text
}

/// Reads a list written by [`carry_lines`]. A line that is not `lock type, 64 hexadecimal characters, base units`,
/// an amount of zero or a lock listed twice is an error.
pub fn parse_lines(text: &str) -> Result<BTreeMap<Lock, u64>, String> {
    let mut list = BTreeMap::new();
    for (n, line) in text.lines().enumerate() {
        let bad = || format!("line {}: not a carry-over line", n + 1);
        let mut fields = line.split(' ');
        let (Some(lock_type), Some(hex), Some(amount), None) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            return Err(bad());
        };
        let lock_type: u8 = lock_type.parse().map_err(|_| bad())?;
        let amount: u64 = amount.parse().map_err(|_| bad())?;
        let mut lock_data = [0u8; 32];
        if hex.len() != 64 || amount == 0 {
            return Err(bad());
        }
        for (byte, pair) in lock_data.iter_mut().zip(hex.as_bytes().chunks(2)) {
            let pair = std::str::from_utf8(pair).map_err(|_| bad())?;
            *byte = u8::from_str_radix(pair, 16).map_err(|_| bad())?;
        }
        if list.insert((lock_type, lock_data), amount).is_some() {
            return Err(format!("line {}: the lock is listed twice", n + 1));
        }
    }
    Ok(list)
}

/// The sum of several lists, lock by lock. `None` when a sum passes 2^64.
pub fn merge(lists: &[BTreeMap<Lock, u64>]) -> Option<BTreeMap<Lock, u64>> {
    let mut sum: BTreeMap<Lock, u64> = BTreeMap::new();
    for list in lists {
        for (lock, amount) in list {
            let total = sum.entry(*lock).or_default();
            *total = total.checked_add(*amount)?;
        }
    }
    Some(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utxo(amount: u64, lock_type: u8, lock: u8, coinbase: bool) -> Utxo {
        Utxo { amount, lock_type, lock_data: [lock; 32], height: 1, coinbase }
    }

    #[test]
    fn a_lock_carries_a_tenth_of_the_sum_of_its_outputs() {
        let utxos = [utxo(1_000, 1, 7, true), utxo(999, 1, 7, false), utxo(25, 1, 8, false), utxo(9, 1, 9, false)];
        let list = carry_over(&utxos, &[]);
        // 1,999 base units carry 199, 25 carry 2, and 9 carry nothing.
        assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![((1, [7; 32]), 199), ((1, [8; 32]), 2)]);
    }

    #[test]
    fn the_same_data_under_two_lock_types_is_two_locks_and_excluded_locks_carry_nothing() {
        let utxos = [utxo(100, 1, 7, false), utxo(200, 2, 7, false), utxo(5_000, 1, 0, true)];
        let list = carry_over(&utxos, &[(1, [0; 32])]);
        assert_eq!(list.into_iter().collect::<Vec<_>>(), vec![((1, [7; 32]), 10), ((2, [7; 32]), 20)]);
    }

    #[test]
    fn a_list_is_read_back_from_its_text_and_bad_lines_are_refused() {
        let list = carry_over(&[utxo(100, 1, 0xab, false), utxo(70, 2, 0xcd, true)], &[]);
        assert_eq!(parse_lines(&carry_lines(&list)), Ok(list));
        let lock = "ab".repeat(32);
        for bad in [
            format!("1 {lock}"),
            format!("1 {lock} 0"),
            format!("1 {lock} 5 6"),
            format!("x {lock} 5"),
            "1 abcd 5".into(),
        ] {
            assert!(parse_lines(&bad).is_err(), "{bad}");
        }
        assert!(
            parse_lines(&format!(
                "1 {lock} 5
1 {lock} 6
"
            ))
            .is_err(),
            "a lock listed twice"
        );
    }

    #[test]
    fn merged_lists_add_the_amounts_of_a_lock() {
        let a = carry_over(&[utxo(100, 1, 7, false), utxo(50, 1, 8, false)], &[]);
        let b = carry_over(&[utxo(300, 1, 7, false), utxo(90, 1, 9, false)], &[]);
        let sum = merge(&[a, b]).expect("no overflow");
        assert_eq!(sum.into_iter().collect::<Vec<_>>(), vec![((1, [7; 32]), 40), ((1, [8; 32]), 5), ((1, [9; 32]), 9)]);
        let big = BTreeMap::from([((1, [1; 32]), u64::MAX)]);
        assert_eq!(merge(&[big.clone(), big]), None);
    }

    #[test]
    fn the_text_has_one_line_per_lock() {
        let list = carry_over(&[utxo(100, 1, 0xab, false)], &[]);
        assert_eq!(carry_lines(&list), format!("1 {} 10\n", "ab".repeat(32)));
    }
}
