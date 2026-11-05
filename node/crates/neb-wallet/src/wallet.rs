//! Key file, coin scan, coin selection and transaction building.
//!
//! Key file: UTF-8 text with one Ed25519 secret per line, 64 hexadecimal
//! characters (the 32-byte RFC 8032 seed). Blank lines and lines starting with
//! `#` are skipped and a repeated key is read once. The file can be encrypted
//! with a password (`encryption`). Each key controls one lock of type 1 with
//! `lock_data = BLAKE3(public key)` (spec/node.md section 5.2).
//!
//! Amounts are u64 base units of 10^-8 coin (spec/node.md section 3).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use ed25519_dalek::{Signer, SigningKey};
use neb_chain::state::COINBASE_MATURITY;
use neb_chain::tx::{LOCK_MULTISIG, LOCK_SINGLE, Transaction, TxIn, TxOut};
use neb_chain::{Block, OutPoint};
use rand_core::{OsRng, RngCore};

use crate::address::{ADDRESS_PREFIX, AddressError, decode_address, encode_address};
use crate::encryption::{self, EncryptionError};
use crate::rpc::RpcClient;

/// Fee of every payment: 0.0001 coin. Wallet policy; consensus only requires
/// that outputs do not exceed inputs (spec/node.md section 5.4), and a template
/// takes transactions highest fee first (section 7.1).
pub const FEE: u64 = 10_000;
/// Change below this amount (0.00001 coin) is added to the fee instead of
/// creating a dust output. Wallet policy; consensus accepts any output amount
/// above zero (spec/node.md section 5.4).
pub const DUST: u64 = 1_000;

#[derive(Debug)]
pub enum WalletError {
    Io(String),
    Key(String),
    Address(AddressError),
    Rpc(String),
    Decode(String),
    Select(SelectError),
    NoKeys,
    MissingKey([u8; 32]),
    Sign(String),
    Amount(AmountError),
    /// The key file is encrypted and no password was given.
    Encrypted,
    Encryption(EncryptionError),
}

impl std::fmt::Display for WalletError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WalletError::Io(e) => write!(f, "wallet io: {e}"),
            WalletError::Key(e) => write!(f, "key: {e}"),
            WalletError::Address(e) => write!(f, "address: {e}"),
            WalletError::Rpc(e) => write!(f, "rpc: {e}"),
            WalletError::Decode(e) => write!(f, "decode: {e}"),
            WalletError::Select(e) => write!(f, "selection: {e}"),
            WalletError::NoKeys => write!(f, "the wallet has no keys"),
            WalletError::MissingKey(lock) => write!(f, "no key for lock {}", hex(lock)),
            WalletError::Sign(e) => write!(f, "sign: {e}"),
            WalletError::Amount(e) => write!(f, "amount: {e}"),
            WalletError::Encrypted => write!(f, "the wallet is encrypted: its password is needed"),
            WalletError::Encryption(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WalletError {}

impl From<std::io::Error> for WalletError {
    fn from(e: std::io::Error) -> Self {
        WalletError::Io(e.to_string())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn parse_secret(text: &str) -> Result<[u8; 32], String> {
    if text.len() != 64 {
        return Err("a secret is 64 hex characters".into());
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).map_err(|_| "secret is not hex")?;
    }
    Ok(out)
}

/// One Ed25519 key and the locks it controls: the type 1 lock `lock_data`, whose
/// bech32m form is `address`, and the type 2 lock `multisig_lock_data` of the
/// 1-of-1 multisig of the key (the form of the project share output).
pub struct WalletKey {
    pub secret: [u8; 32],
    pub signing: SigningKey,
    pub lock_data: [u8; 32],
    pub multisig_lock_data: [u8; 32],
    pub address: String,
}

impl WalletKey {
    /// Derives the public key and the lock `BLAKE3(public key)` from the seed.
    pub fn from_secret(secret: [u8; 32]) -> WalletKey {
        let signing = SigningKey::from_bytes(&secret);
        let public = signing.verifying_key().to_bytes();
        let lock_data = *blake3::hash(&public).as_bytes();
        let multisig_lock_data = Transaction::multisig_lock_data(1, &[public]);
        WalletKey {
            secret,
            signing,
            lock_data,
            multisig_lock_data,
            address: encode_address(ADDRESS_PREFIX, &lock_data),
        }
    }

    pub fn from_text(text: &str) -> Result<WalletKey, String> {
        Ok(WalletKey::from_secret(parse_secret(text)?))
    }

    /// A key from the operating system's random number generator.
    pub fn generate() -> WalletKey {
        let mut secret = [0u8; 32];
        OsRng.fill_bytes(&mut secret);
        WalletKey::from_secret(secret)
    }

    /// The lock of the key as 64 lowercase hexadecimal characters, the form
    /// that `neb-node --miner-lock` takes.
    pub fn lock_hex(&self) -> String {
        hex(&self.lock_data)
    }

    pub fn secret_hex(&self) -> String {
        hex(&self.secret)
    }

    /// The Ed25519 public key as 64 lowercase hexadecimal characters.
    pub fn public_hex(&self) -> String {
        hex(&self.signing.verifying_key().to_bytes())
    }

    /// The lock of the 1-of-1 multisig of the key as 64 lowercase hexadecimal characters.
    pub fn multisig_lock_hex(&self) -> String {
        hex(&self.multisig_lock_data)
    }

    /// True when the key alone can spend an output with this lock type and lock data.
    pub fn controls(&self, lock_type: u8, lock_data: &[u8; 32]) -> bool {
        match lock_type {
            LOCK_SINGLE => &self.lock_data == lock_data,
            LOCK_MULTISIG => &self.multisig_lock_data == lock_data,
            _ => false,
        }
    }
}

/// The keys of one key file, in file order. The first key is the default
/// address and receives the change of a payment.
pub struct Wallet {
    pub path: String,
    pub keys: Vec<WalletKey>,
    /// Password of an encrypted key file; `save` encrypts again with it.
    password: Option<String>,
}

impl Wallet {
    pub fn empty(path: impl Into<String>) -> Wallet {
        Wallet { path: path.into(), keys: Vec::new(), password: None }
    }

    /// True when the key file at `path` is encrypted.
    pub fn is_encrypted(path: impl AsRef<Path>) -> Result<bool, WalletError> {
        Ok(encryption::is_encrypted(&fs::read_to_string(path)?))
    }

    /// Reads a plain key file. An encrypted one is `WalletError::Encrypted`. A line that is not a 64-hex secret is an
    /// error that names its 1-based line number.
    pub fn load(path: impl AsRef<Path>) -> Result<Wallet, WalletError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        if encryption::is_encrypted(&text) {
            return Err(WalletError::Encrypted);
        }
        Wallet::parse(path, &text, None)
    }

    /// Reads a key file, plain or encrypted with `password`.
    pub fn load_with_password(path: impl AsRef<Path>, password: &str) -> Result<Wallet, WalletError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        if !encryption::is_encrypted(&text) {
            return Wallet::parse(path, &text, None);
        }
        let plain = encryption::decrypt(&text, password).map_err(WalletError::Encryption)?;
        Wallet::parse(path, &plain, Some(password.to_string()))
    }

    fn parse(path: &Path, text: &str, password: Option<String>) -> Result<Wallet, WalletError> {
        let mut keys = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let key = WalletKey::from_text(line).map_err(|e| WalletError::Key(format!("line {}: {e}", n + 1)))?;
            if keys.iter().any(|k: &WalletKey| k.lock_data == key.lock_data) {
                continue;
            }
            keys.push(key);
        }
        Ok(Wallet { path: path.display().to_string(), keys, password })
    }

    /// Writes a new file with one generated key. Refuses to replace a file
    /// that is not empty.
    pub fn create(path: impl AsRef<Path>) -> Result<Wallet, WalletError> {
        Wallet::create_with(path, None)
    }

    /// As `create`, with a file encrypted with `password`.
    pub fn create_encrypted(path: impl AsRef<Path>, password: &str) -> Result<Wallet, WalletError> {
        Wallet::create_with(path, Some(password.to_string()))
    }

    fn create_with(path: impl AsRef<Path>, password: Option<String>) -> Result<Wallet, WalletError> {
        let path = path.as_ref();
        if path.exists() && fs::metadata(path)?.len() > 0 {
            return Err(WalletError::Io(format!("{} already exists", path.display())));
        }
        let wallet = Wallet { password, ..Wallet::empty(path.display().to_string()) };
        let key = WalletKey::generate();
        wallet.save_with(&[key])
    }

    /// Encrypts the key file with `password`; later saves keep it encrypted.
    pub fn encrypt(&mut self, password: &str) -> Result<(), WalletError> {
        self.password = Some(password.to_string());
        self.save()
    }

    /// Writes `keys` to the key file. A file created here has mode 0600 on
    /// Unix; the mode of an existing file is kept.
    fn write_keys(&self, keys: &[WalletKey]) -> Result<(), WalletError> {
        let mut text = String::from("# neb testnet wallet: one 64-hex secret per line\n");
        for key in keys {
            text.push_str(&key.secret_hex());
            text.push('\n');
        }
        if let Some(password) = &self.password {
            text = encryption::encrypt(&text, password);
        }
        let mut options = fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        std::io::Write::write_all(&mut options.open(&self.path)?, text.as_bytes())?;
        Ok(())
    }

    fn save_with(&self, keys: &[WalletKey]) -> Result<Wallet, WalletError> {
        self.write_keys(keys)?;
        Ok(Wallet {
            path: self.path.clone(),
            keys: keys.iter().map(|k| WalletKey::from_secret(k.secret)).collect(),
            password: self.password.clone(),
        })
    }

    pub fn save(&self) -> Result<(), WalletError> {
        self.write_keys(&self.keys)
    }

    pub fn key_for_lock(&self, lock: &[u8; 32]) -> Option<&WalletKey> {
        self.keys.iter().find(|k| &k.lock_data == lock)
    }

    /// The key that can spend an output with this lock type and lock data.
    pub fn key_for_output(&self, lock_type: u8, lock_data: &[u8; 32]) -> Option<&WalletKey> {
        self.keys.iter().find(|k| k.controls(lock_type, lock_data))
    }
}

/// An unspent output of the best chain that a wallet key can spend. `height`
/// is the height of the block that created it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedUtxo {
    pub outpoint: OutPoint,
    pub amount: u64,
    pub lock_type: u8,
    pub lock_data: [u8; 32],
    pub height: u32,
    pub coinbase: bool,
}

impl OwnedUtxo {
    /// True when a transaction in the block of height `next_height` may spend
    /// the output: always for an ordinary output, from `height + 100` for a
    /// coinbase output (spec/node.md section 5.3).
    pub fn mature(&self, next_height: u32) -> bool {
        !self.coinbase || next_height >= self.height + COINBASE_MATURITY
    }

    /// The output as the signature message binds it (spec/node.md section 5.2).
    pub fn as_txout(&self) -> TxOut {
        TxOut { amount: self.amount, lock_type: self.lock_type, lock_data: self.lock_data }
    }
}

/// The wallet's unspent outputs on the best chain up to `tip_height`, in
/// outpoint order.
#[derive(Debug, Clone)]
pub struct Scan {
    pub utxos: Vec<OwnedUtxo>,
    pub tip_height: u32,
}

impl Scan {
    /// Sum of all outputs, immature coinbase outputs included.
    pub fn balance(&self) -> u64 {
        self.utxos.iter().map(|u| u.amount).sum()
    }

    /// Sum of the outputs that the next block may spend.
    pub fn mature_balance(&self) -> u64 {
        let next = self.tip_height + 1;
        self.utxos.iter().filter(|u| u.mature(next)).map(|u| u.amount).sum()
    }

    pub fn mature(&self) -> Vec<OwnedUtxo> {
        let next = self.tip_height + 1;
        self.utxos.iter().filter(|u| u.mature(next)).cloned().collect()
    }
}

fn hex32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

fn block_hash(value: &neb_json::Value) -> Result<[u8; 32], WalletError> {
    let text = value.as_str().ok_or_else(|| WalletError::Rpc("block hash is not a string".into()))?;
    hex32(text).ok_or_else(|| WalletError::Rpc("block hash is not 64 hex characters".into()))
}

/// Replays the best chain from height 0 to the tip and returns the unspent
/// outputs of the wallet keys. Costs one `getblockhash` and one `getblock` call
/// per block. The calls are not atomic: a reorganisation during the scan can
/// mix blocks of two chains, and the node may then refuse a payment built from
/// the result.
pub fn scan(client: &RpcClient, wallet: &Wallet) -> Result<Scan, WalletError> {
    let tip_value = client.call("getblockcount", vec![]).map_err(WalletError::Rpc)?;
    let tip_height = tip_value.as_u64().ok_or_else(|| WalletError::Rpc("bad block count".into()))? as u32;
    let mut utxos: BTreeMap<OutPoint, OwnedUtxo> = BTreeMap::new();
    // Blocks are applied in height order, so a spend removes the output an
    // earlier block (or an earlier transaction of the same block) created.
    for height in 0..=tip_height {
        let hash = block_hash(
            &client.call("getblockhash", vec![neb_json::Value::Number(height as i64)]).map_err(WalletError::Rpc)?,
        )?;
        let text = client.call("getblock", vec![neb_json::Value::String(hex(&hash))]).map_err(WalletError::Rpc)?;
        let text = text.as_str().ok_or_else(|| WalletError::Rpc("block is not a string".into()))?;
        let bytes = decode_hex(text).ok_or_else(|| WalletError::Rpc("block is not hex".into()))?;
        let block = Block::decode(&bytes).map_err(|e| WalletError::Decode(e.to_string()))?;
        for tx in &block.txs {
            for input in &tx.inputs {
                // A coinbase input (prev_txid all zero) names no output: no-op.
                utxos.remove(&OutPoint { txid: input.prev_txid, index: input.prev_index });
            }
            for (index, output) in tx.outputs.iter().enumerate() {
                // Type 1 locks of a wallet key, and the 1-of-1 multisig locks of the wallet keys
                // (the project share output). Other multisig locks are not owned.
                if wallet.key_for_output(output.lock_type, &output.lock_data).is_some() {
                    utxos.insert(
                        OutPoint { txid: tx.txid(), index: index as u32 },
                        OwnedUtxo {
                            outpoint: OutPoint { txid: tx.txid(), index: index as u32 },
                            amount: output.amount,
                            lock_type: output.lock_type,
                            lock_data: output.lock_data,
                            height,
                            coinbase: tx.is_coinbase(),
                        },
                    );
                }
            }
        }
    }
    Ok(Scan { utxos: utxos.into_values().collect(), tip_height })
}

fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(text.len() / 2);
    for i in 0..text.len() / 2 {
        out.push(u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?);
    }
    Some(out)
}

/// Inputs chosen for a payment. `fee` is the fee actually paid: the requested
/// fee plus any change below `DUST`. The inputs sum to `target + fee + change`,
/// and `change` is zero or at least `DUST`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub inputs: Vec<OwnedUtxo>,
    pub target: u64,
    pub fee: u64,
    pub change: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectError {
    ZeroTarget,
    /// More than 255 inputs are needed (spec/node.md section 5.4).
    TooManyInputs,
    Insufficient {
        available: u64,
        needed: u64,
    },
}

impl std::fmt::Display for SelectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectError::ZeroTarget => write!(f, "target is zero"),
            SelectError::TooManyInputs => write!(f, "over 255 inputs"),
            SelectError::Insufficient { available, needed } => {
                write!(f, "available {available} below needed {needed}")
            }
        }
    }
}

impl std::error::Error for SelectError {}

/// Takes the largest outputs first (equal amounts in outpoint order) until
/// their sum reaches `target + fee`; the remainder is the change, or goes to
/// the fee when it is below `DUST`. Largest-first uses the fewest inputs, so
/// `TooManyInputs` means that no selection fits the limit of 255.
pub fn select(utxos: &[OwnedUtxo], target: u64, fee: u64) -> Result<Selection, SelectError> {
    if target == 0 {
        return Err(SelectError::ZeroTarget);
    }
    let needed = target.checked_add(fee).ok_or(SelectError::Insufficient { available: 0, needed: u64::MAX })?;
    let mut sorted: Vec<OwnedUtxo> = utxos.to_vec();
    // The outpoint tie-break makes the selection reproducible.
    sorted.sort_by(|a, b| b.amount.cmp(&a.amount).then(a.outpoint.cmp(&b.outpoint)));
    let mut chosen = Vec::new();
    let mut total = 0u64;
    for utxo in sorted {
        chosen.push(utxo.clone());
        total = total.saturating_add(utxo.amount);
        if total >= needed {
            let mut change = total - needed;
            let mut actual_fee = fee;
            if change < DUST {
                actual_fee += change;
                change = 0;
            }
            if chosen.len() > 255 {
                return Err(SelectError::TooManyInputs);
            }
            return Ok(Selection { inputs: chosen, target, fee: actual_fee, change });
        }
    }
    Err(SelectError::Insufficient { available: total, needed })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AmountError {
    Empty,
    BadCharacter,
    TooManyDecimals,
    Overflow,
}

impl std::fmt::Display for AmountError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AmountError::Empty => write!(f, "empty amount"),
            AmountError::BadCharacter => write!(f, "amount has a non-digit"),
            AmountError::TooManyDecimals => write!(f, "more than 8 decimals"),
            AmountError::Overflow => write!(f, "amount is too large"),
        }
    }
}

impl std::error::Error for AmountError {}

/// Parses a decimal coin amount (`1`, `0.5`, `.5`, `1.`) into base units: at
/// most 8 decimals, no sign, no exponent. A value above `u64::MAX` base units
/// is `Overflow`.
pub fn parse_amount(text: &str) -> Result<u64, AmountError> {
    if text.is_empty() {
        return Err(AmountError::Empty);
    }
    let (whole, fraction) = match text.split_once('.') {
        Some((a, b)) => (a, b),
        None => (text, ""),
    };
    if whole.is_empty() && fraction.is_empty() {
        return Err(AmountError::Empty);
    }
    if !whole.chars().all(|c| c.is_ascii_digit()) || !fraction.chars().all(|c| c.is_ascii_digit()) {
        return Err(AmountError::BadCharacter);
    }
    if fraction.len() > 8 {
        return Err(AmountError::TooManyDecimals);
    }
    let whole: u64 = if whole.is_empty() { 0 } else { whole.parse().map_err(|_| AmountError::Overflow)? };
    // Right-pad to 8 digits so that "5" after the point is 50_000_000 base units.
    let mut frac = fraction.to_string();
    while frac.len() < 8 {
        frac.push('0');
    }
    let frac: u64 = if frac.is_empty() { 0 } else { frac.parse().map_err(|_| AmountError::Overflow)? };
    whole.checked_mul(neb_chain::COIN).and_then(|v| v.checked_add(frac)).ok_or(AmountError::Overflow)
}

/// The inverse of `parse_amount`: decimal coins without trailing zeros, such
/// as `1.00000001` or `1`.
pub fn format_amount(base_units: u64) -> String {
    let whole = base_units / neb_chain::COIN;
    let fraction = base_units % neb_chain::COIN;
    if fraction == 0 {
        format!("{whole}")
    } else {
        let mut text = format!("{fraction:08}");
        while text.ends_with('0') {
            text.pop();
        }
        format!("{whole}.{text}")
    }
}

/// Builds the version 1 transaction of a selection from `select`: the target
/// to `recipient_lock`, then the change, if any, to `change_lock`; both are
/// type 1 locks. Each input is signed with the wallet key of the lock it spends
/// (spec/node.md section 5.2). `chain_id` is the genesis block id. Fails with
/// `MissingKey` when the wallet has no key for an input.
pub fn build_transaction(
    wallet: &Wallet,
    selection: &Selection,
    recipient_lock: [u8; 32],
    chain_id: &[u8; 32],
    change_lock: [u8; 32],
) -> Result<Transaction, WalletError> {
    if selection.inputs.is_empty() {
        return Err(WalletError::Select(SelectError::ZeroTarget));
    }
    let mut inputs = Vec::with_capacity(selection.inputs.len());
    for utxo in &selection.inputs {
        inputs.push(TxIn { prev_txid: utxo.outpoint.txid, prev_index: utxo.outpoint.index, witness: Vec::new() });
    }
    let mut outputs = vec![TxOut { amount: selection.target, lock_type: LOCK_SINGLE, lock_data: recipient_lock }];
    if selection.change > 0 {
        outputs.push(TxOut { amount: selection.change, lock_type: LOCK_SINGLE, lock_data: change_lock });
    }
    let mut tx = Transaction { version: 1, inputs, outputs };
    // The sighash commits to the txid, which excludes witnesses (spec/node.md
    // section 5.1), so inputs can be signed one after another.
    for (index, utxo) in selection.inputs.iter().enumerate() {
        let key =
            wallet.key_for_output(utxo.lock_type, &utxo.lock_data).ok_or(WalletError::MissingKey(utxo.lock_data))?;
        let spent = utxo.as_txout();
        let message = tx.sighash(chain_id, index as u32, &spent);
        let signature = key.signing.sign(&message);
        let public = key.signing.verifying_key().to_bytes();
        let witness = if utxo.lock_type == LOCK_MULTISIG {
            // Witness of a 1-of-1 multisig lock: m (1) || n (1) || public key (32) || key index (0) || signature (64).
            let mut witness = Vec::with_capacity(99);
            witness.extend_from_slice(&[1, 1]);
            witness.extend_from_slice(&public);
            witness.push(0);
            witness.extend_from_slice(&signature.to_bytes());
            witness
        } else {
            // Witness of a type 1 lock: public key (32) || signature (64), 96 bytes.
            let mut witness = Vec::with_capacity(96);
            witness.extend_from_slice(&public);
            witness.extend_from_slice(&signature.to_bytes());
            witness
        };
        tx.inputs[index].witness = witness;
        // The node's verification routine, so an invalid witness is never sent.
        tx.check_witness(chain_id, index as u32, &spent).map_err(|e| WalletError::Sign(e.to_string()))?;
    }
    Ok(tx)
}

/// Decodes a recipient address for the wallet's network.
pub fn recipient_lock(address: &str) -> Result<[u8; 32], WalletError> {
    decode_address(ADDRESS_PREFIX, address).map_err(WalletError::Address)
}

/// The default change lock: the first key.
pub fn change_lock(wallet: &Wallet) -> Result<[u8; 32], WalletError> {
    wallet.keys.first().map(|k| k.lock_data).ok_or(WalletError::NoKeys)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utxo(amount: u64, seed: u8) -> OwnedUtxo {
        OwnedUtxo {
            outpoint: OutPoint { txid: [seed; 32], index: 0 },
            amount,
            lock_type: LOCK_SINGLE,
            lock_data: [seed; 32],
            height: 1,
            coinbase: false,
        }
    }

    // Decimal text converts exactly to base units and back. A ninth decimal, a
    // sign and a second point are refused.
    #[test]
    fn amount_parsing_and_formatting() {
        assert_eq!(parse_amount("1").unwrap(), 100_000_000);
        assert_eq!(parse_amount("0.5").unwrap(), 50_000_000);
        assert_eq!(parse_amount("1.00000001").unwrap(), 100_000_001);
        assert_eq!(parse_amount(".5").unwrap(), 50_000_000);
        assert!(parse_amount("1.000000001").is_err());
        assert!(parse_amount("-1").is_err());
        assert!(parse_amount("1.2.3").is_err());
        assert_eq!(format_amount(100_000_001), "1.00000001");
        assert_eq!(format_amount(100_000_000), "1");
    }

    // 0.9 coin is paid from the single 1-coin output; 1.2 coin needs the 1-coin
    // and the 0.5-coin outputs. Change = inputs - target - fee.
    #[test]
    fn selection_is_largest_first_and_returns_change() {
        let utxos = vec![utxo(10_000_000, 1), utxo(100_000_000, 2), utxo(50_000_000, 3)];
        let selection = select(&utxos, 90_000_000, 10_000).unwrap();
        assert_eq!(selection.inputs.len(), 1);
        assert_eq!(selection.inputs[0].amount, 100_000_000);
        assert_eq!(selection.change, 9_990_000);
        assert_eq!(selection.fee, 10_000);
        let selection = select(&utxos, 120_000_000, 10_000).unwrap();
        assert_eq!(selection.inputs.len(), 2);
        assert_eq!(selection.change, 29_990_000);
    }

    // Change of 100 base units (below DUST) is added to the fee: 400 + 100 = 500
    // and no change output. Missing funds and a zero target are errors.
    #[test]
    fn selection_handles_dust_and_shortfall() {
        let utxos = vec![utxo(100_000_000, 1)];
        let selection = select(&utxos, 90_000_000, 10_000).unwrap();
        assert_eq!(selection.change, 9_990_000);
        let selection = select(&utxos, 99_999_500, 400).unwrap();
        assert_eq!(selection.change, 0);
        assert_eq!(selection.fee, 500);
        assert!(matches!(select(&utxos, 200_000_000, 0), Err(SelectError::Insufficient { .. })));
        assert_eq!(select(&utxos, 0, 0), Err(SelectError::ZeroTarget));
    }

    // A created key file reloads with the same lock and is not overwritten. The
    // payment has the recipient output and the change (input - 0.5 coin - FEE);
    // its witness verifies for the chain id it was signed for and for no other
    // (a signature is bound to one chain, spec/node.md section 5.2).
    #[test]
    fn wallet_file_roundtrip_and_signing() {
        let dir = std::env::temp_dir().join(format!("neb-wallet-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wallet.txt");
        let wallet = Wallet::create(&path).unwrap();
        assert_eq!(wallet.keys.len(), 1);
        let loaded = Wallet::load(&path).unwrap();
        assert_eq!(loaded.keys[0].lock_data, wallet.keys[0].lock_data);
        assert!(Wallet::create(&path).is_err());

        let recipient = WalletKey::from_secret([3u8; 32]);
        let owned = OwnedUtxo {
            outpoint: OutPoint { txid: [4u8; 32], index: 0 },
            amount: 100_000_000,
            lock_type: LOCK_SINGLE,
            lock_data: wallet.keys[0].lock_data,
            height: 1,
            coinbase: false,
        };
        let selection = select(&[owned], 50_000_000, FEE).unwrap();
        let chain_id = [7u8; 32];
        let tx =
            build_transaction(&wallet, &selection, recipient.lock_data, &chain_id, wallet.keys[0].lock_data).unwrap();
        assert_eq!(tx.outputs.len(), 2);
        assert_eq!(tx.outputs[0].amount, 50_000_000);
        assert_eq!(tx.outputs[1].amount, 49_990_000);
        let spent = selection.inputs[0].as_txout();
        assert!(tx.check_witness(&chain_id, 0, &spent).is_ok());
        assert!(tx.check_witness(&[9u8; 32], 0, &spent).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // An output of the 1-of-1 multisig lock of a wallet key (the form of the project share output) is
    // owned by that wallet only, and it is spent with a 99-byte witness that the node's routine accepts.
    #[test]
    fn a_one_of_one_multisig_output_is_owned_and_spent() {
        let key = WalletKey::from_secret([5u8; 32]);
        let wallet = Wallet { keys: vec![WalletKey::from_secret([5u8; 32])], ..Wallet::empty("") };
        let other = Wallet { keys: vec![WalletKey::from_secret([6u8; 32])], ..Wallet::empty("") };
        assert!(wallet.key_for_output(LOCK_MULTISIG, &key.multisig_lock_data).is_some());
        assert!(other.key_for_output(LOCK_MULTISIG, &key.multisig_lock_data).is_none());
        assert!(wallet.key_for_output(LOCK_SINGLE, &key.multisig_lock_data).is_none());
        let owned = OwnedUtxo {
            outpoint: OutPoint { txid: [4u8; 32], index: 1 },
            amount: 100_000_000,
            lock_type: LOCK_MULTISIG,
            lock_data: key.multisig_lock_data,
            height: 1,
            coinbase: true,
        };
        let selection = select(&[owned], 50_000_000, FEE).unwrap();
        let chain_id = [7u8; 32];
        let tx = build_transaction(&wallet, &selection, [3u8; 32], &chain_id, key.lock_data).unwrap();
        assert_eq!(tx.inputs[0].witness.len(), 99);
        let spent = selection.inputs[0].as_txout();
        assert_eq!(spent.lock_type, LOCK_MULTISIG);
        assert!(tx.check_witness(&chain_id, 0, &spent).is_ok());
        assert!(tx.check_witness(&[9u8; 32], 0, &spent).is_err());
    }
}
