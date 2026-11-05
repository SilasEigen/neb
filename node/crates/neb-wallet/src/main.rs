//! `neb-wallet` command line: argument parsing and output only. Keys, scanning,
//! selection and signing are in the library.
//!
//! `--wallet` and `--rpc` come before the command. An error prints
//! `neb-wallet: <error>` on stderr and exits with status 1.

use std::path::{Path, PathBuf};

use neb_wallet::display::{Look, Row};
use neb_wallet::wallet::WalletKey;
use neb_wallet::{
    ADDRESS_PREFIX, FEE, Wallet, build_transaction, decode_address, format_amount, parse_amount, scan, select,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("neb-wallet: {e}");
            std::process::exit(1);
        }
    }
}

fn usage() -> String {
    "usage: neb-wallet [--wallet FILE] [--rpc URL] COMMAND\n\
     \n\
     commands:\n\
       new                 create a wallet with one new key, encrypted with a password\n\
       encrypt             encrypt a plain wallet file with a password\n\
       import SECRET       add a 64-hex secret to the wallet\n\
       list                print the addresses\n\
       address             print the first address\n\
       lock                print the lock of the first key, for neb-node --miner-lock\n\
       lock-of ADDRESS     print the lock of a neb1 address, for neb-node --miner-lock\n\
       public-key          print the Ed25519 public key of the first key (64 hex)\n\
       balance             scan the node and print the balance\n\
       send ADDRESS AMOUNT send coins to a neb1 address\n\
     \n\
     defaults: --wallet wallet.txt --rpc http://127.0.0.1:8432\n\
     The password is asked on the terminal, or read from NEB_WALLET_PASSWORD."
        .to_string()
}

const PASSWORD_ENV: &str = "NEB_WALLET_PASSWORD";

/// Opens the wallet file, asking for the password when the file is encrypted.
fn open(path: &Path) -> Result<Wallet, String> {
    if !Wallet::is_encrypted(path).map_err(|e| e.to_string())? {
        return Wallet::load(path).map_err(|e| e.to_string());
    }
    let password = match std::env::var(PASSWORD_ENV) {
        Ok(p) => p,
        Err(_) => rpassword::prompt_password("wallet password: ").map_err(|e| format!("password: {e}"))?,
    };
    Wallet::load_with_password(path, &password).map_err(|e| e.to_string())
}

/// A new password, typed twice. Empty means no encryption when `allow_empty`.
fn new_password(allow_empty: bool) -> Result<Option<String>, String> {
    let password = match std::env::var(PASSWORD_ENV) {
        Ok(p) => p,
        Err(_) => {
            let hint = if allow_empty { " (empty: no encryption)" } else { "" };
            let first = rpassword::prompt_password(format!("new wallet password{hint}: "))
                .map_err(|e| format!("password: {e}"))?;
            let again = rpassword::prompt_password("again: ").map_err(|e| format!("password: {e}"))?;
            if first != again {
                return Err("the two passwords differ".into());
            }
            first
        }
    };
    match (password.is_empty(), allow_empty) {
        (false, _) => Ok(Some(password)),
        (true, true) => Ok(None),
        (true, false) => Err("the password is empty".into()),
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut wallet_path = PathBuf::from("wallet.txt");
    let mut rpc_url = "http://127.0.0.1:8432".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--wallet" => {
                wallet_path = PathBuf::from(args.get(i + 1).ok_or("--wallet needs a value")?);
                i += 2;
            }
            "--rpc" => {
                rpc_url = args.get(i + 1).ok_or("--rpc needs a value")?.clone();
                i += 2;
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ => break,
        }
    }
    let command = args.get(i).map(String::as_str).unwrap_or("");
    match command {
        "new" => {
            let wallet = match new_password(true)? {
                Some(password) => Wallet::create_encrypted(&wallet_path, &password),
                None => Wallet::create(&wallet_path),
            }
            .map_err(|e| e.to_string())?;
            let addresses: Vec<String> = wallet.keys.iter().map(|k| k.address.clone()).collect();
            print!("{}", Look::detect().created(&wallet.path, &addresses));
            Ok(())
        }
        "encrypt" => {
            if Wallet::is_encrypted(&wallet_path).map_err(|e| e.to_string())? {
                return Err("the wallet is already encrypted".into());
            }
            let mut wallet = Wallet::load(&wallet_path).map_err(|e| e.to_string())?;
            let password = new_password(false)?.expect("a password");
            wallet.encrypt(&password).map_err(|e| e.to_string())?;
            println!("encrypted {}", wallet.path);
            Ok(())
        }
        "import" => {
            let secret = args.get(i + 1).ok_or("import needs a 64-hex secret")?;
            let key = WalletKey::from_text(secret).map_err(|e| e.to_string())?;
            let mut wallet = if wallet_path.exists() {
                open(&wallet_path)?
            } else {
                Wallet::empty(wallet_path.display().to_string())
            };
            if wallet.key_for_lock(&key.lock_data).is_none() {
                wallet.keys.push(key);
                wallet.save().map_err(|e| e.to_string())?;
            }
            println!("address {}", wallet.keys.last().expect("one key").address);
            Ok(())
        }
        "list" => {
            let wallet = open(&wallet_path)?;
            for key in &wallet.keys {
                println!("{}", key.address);
            }
            Ok(())
        }
        "address" => {
            let wallet = open(&wallet_path)?;
            let key = wallet.keys.first().ok_or("wallet has no keys")?;
            println!("{}", key.address);
            Ok(())
        }
        "lock" => {
            let wallet = open(&wallet_path)?;
            let key = wallet.keys.first().ok_or("wallet has no keys")?;
            println!("{}", key.lock_hex());
            Ok(())
        }
        "public-key" => {
            let wallet = open(&wallet_path)?;
            let key = wallet.keys.first().ok_or("wallet has no keys")?;
            println!("{}", key.public_hex());
            Ok(())
        }
        "lock-of" => {
            let address = args.get(i + 1).ok_or("lock-of needs an address")?;
            let lock = decode_address(ADDRESS_PREFIX, address).map_err(|e| e.to_string())?;
            println!("{}", lock.iter().map(|b| format!("{b:02x}")).collect::<String>());
            Ok(())
        }
        "balance" => {
            let wallet = open(&wallet_path)?;
            let client = neb_wallet::RpcClient::new(rpc_url.clone())?;
            let scan = scan(&client, &wallet).map_err(|e| e.to_string())?;
            let next = scan.tip_height + 1;
            let mut rows = Vec::new();
            for key in &wallet.keys {
                let confirmed: u64 = scan
                    .utxos
                    .iter()
                    .filter(|u| key.controls(u.lock_type, &u.lock_data) && u.mature(next))
                    .map(|u| u.amount)
                    .sum();
                // coinbase outputs younger than 100 blocks (spec/node.md section 5.3)
                let pending: u64 = scan
                    .utxos
                    .iter()
                    .filter(|u| key.controls(u.lock_type, &u.lock_data) && !u.mature(next))
                    .map(|u| u.amount)
                    .sum();
                rows.push(Row {
                    address: key.address.clone(),
                    confirmed: format_amount(confirmed),
                    immature: (pending > 0).then(|| format_amount(pending)),
                });
            }
            print!("{}", Look::detect().balance(scan.tip_height, &rows, &format_amount(scan.mature_balance())));
            Ok(())
        }
        "send" => {
            let address = args.get(i + 1).ok_or("send needs an address")?;
            let amount_text = args.get(i + 2).ok_or("send needs an amount")?;
            let amount = parse_amount(amount_text).map_err(|e| e.to_string())?;
            let recipient = decode_address(ADDRESS_PREFIX, address).map_err(|e| e.to_string())?;
            let wallet = open(&wallet_path)?;
            let change = wallet.keys.first().ok_or("wallet has no keys")?.lock_data;
            let client = neb_wallet::RpcClient::new(rpc_url.clone())?;
            let scan = scan(&client, &wallet).map_err(|e| e.to_string())?;
            // The scan reads blocks only, not the mempool: an output spent by a
            // pending transaction can be selected again, and the node refuses a
            // payment that spends it a second time.
            let selection = select(&scan.mature(), amount, FEE).map_err(|e| e.to_string())?;
            // the chain id is the genesis block id (spec/node.md section 5.2)
            let chain_id_text = client.block_hash(0)?;
            let chain_id = decode_hex32(&chain_id_text).ok_or("bad genesis hash")?;
            let tx = build_transaction(&wallet, &selection, recipient, &chain_id, change).map_err(|e| e.to_string())?;
            let txid = client.send_raw_transaction(&hex(&tx.encode()))?;
            println!("inputs   {}", selection.inputs.len());
            println!("change   {} NEB", format_amount(selection.change));
            println!("fee      {} NEB", format_amount(selection.fee));
            println!("txid     {txid}");
            Ok(())
        }
        "" => {
            println!("{}", usage());
            Ok(())
        }
        other => Err(format!("unknown command {other}")),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn decode_hex32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}
