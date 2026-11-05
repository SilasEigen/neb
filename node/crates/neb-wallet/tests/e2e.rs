//! End-to-end wallet test against a real node and its JSON-RPC server
//! (spec/node.md section 7.2). `verify_pow` is false so that blocks need no
//! ticket search.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ed25519_dalek::SigningKey;
use neb_chain::demo;
use neb_chain::state::{project_share, subsidy};
use neb_chain::tx::Transaction;
use neb_node::params::NetworkParams;
use neb_node::{Node, NodeConfig};
use neb_wallet::{FEE, RpcClient, Wallet, build_transaction, scan, select};

const T0: u32 = 1_700_000_000;
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-wallet-e2e-{tag}-{}-{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn lock_of(sk: &SigningKey) -> [u8; 32] {
    *blake3::hash(&sk.verifying_key().to_bytes()).as_bytes()
}

fn network(miner_lock: [u8; 32]) -> NetworkParams {
    network_with_project(miner_lock, demo::PROJECT_LOCK)
}

fn network_with_project(miner_lock: [u8; 32], project_lock: [u8; 32]) -> NetworkParams {
    NetworkParams {
        name: "test",
        magic: [9, 9, 9, 9],
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
        genesis: demo::genesis_to(T0, miner_lock),
        chain: demo::demo_params(T0, project_lock),
    }
}

fn coinbase(height: u32) -> Transaction {
    coinbase_to(height, demo::PROJECT_LOCK)
}

fn coinbase_to(height: u32, project_lock: [u8; 32]) -> Transaction {
    let reward = subsidy(height);
    demo::coinbase(height, demo::MINER_LOCK, project_lock, reward - project_share(reward))
}

// The wallet finds its genesis output through getblockhash/getblock, spends
// part of it with a signature bound to the chain id, and the node accepts the
// transaction. 101 blocks make the genesis coinbase mature (spec/node.md
// section 5.3); the balance is the genesis reward less the 5 percent project
// share (spec/node.md section 6.3): 19 coins.
#[test]
fn wallet_scans_signs_and_submits() {
    let sk = SigningKey::from_bytes(&[9u8; 32]);
    let dir = temp_dir("node");
    let wallet_dir = temp_dir("wallet");
    std::fs::create_dir_all(&wallet_dir).unwrap();
    let wallet_path = wallet_dir.join("wallet.txt");
    std::fs::write(&wallet_path, format!("{}\n", sk.to_bytes().iter().map(|b| format!("{b:02x}")).collect::<String>()))
        .unwrap();
    let wallet = Wallet::load(&wallet_path).expect("wallet");
    assert_eq!(wallet.keys[0].lock_data, lock_of(&sk));

    let node = Node::open(NodeConfig { network: network(lock_of(&sk)), data_dir: dir.clone(), verify_pow: false })
        .expect("node");
    let mut parent = node.tip_id();
    for height in 1..=101 {
        let block = demo::block(parent, T0 + 60 * height, 0xC280_0000, 5000 + height, vec![coinbase(height)]);
        parent = block.id();
        node.accept_block(block).expect("connect");
    }
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    let client = RpcClient::new(format!("http://{}", rpc.addr)).expect("client");

    let scan = scan(&client, &wallet).expect("scan");
    assert_eq!(scan.tip_height, 101);
    assert_eq!(scan.mature_balance(), 19 * 100_000_000);
    let recipient = neb_wallet::address::encode_address("neb", &[0x33; 32]);
    let recipient_lock = neb_wallet::decode_address("neb", &recipient).unwrap();
    let selection = select(&scan.mature(), 5 * 100_000_000, FEE).expect("select");
    let chain_id = node.chain_id();
    let tx = build_transaction(&wallet, &selection, recipient_lock, &chain_id, wallet.keys[0].lock_data).expect("tx");
    let txid =
        client.send_raw_transaction(&tx.encode().iter().map(|b| format!("{b:02x}")).collect::<String>()).expect("send");
    assert_eq!(txid.len(), 64);
    assert!(node.has_tx(&tx.txid()));
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&wallet_dir);
}

// The project share is a 1-of-1 multisig output of the coinbase. A wallet that holds the project key
// sees the share once it is mature (100 blocks), and spends it with a witness that the node accepts:
// blocks 1 and 2 of a 101-block chain give two mature shares of 1 coin each.
#[test]
fn wallet_with_the_project_key_sees_and_spends_the_project_share() {
    let sk = SigningKey::from_bytes(&[11u8; 32]);
    let project_lock = Transaction::multisig_lock_data(1, &[sk.verifying_key().to_bytes()]);
    let dir = temp_dir("project-node");
    let wallet_dir = temp_dir("project-wallet");
    std::fs::create_dir_all(&wallet_dir).unwrap();
    let wallet_path = wallet_dir.join("wallet.txt");
    std::fs::write(&wallet_path, format!("{}\n", sk.to_bytes().iter().map(|b| format!("{b:02x}")).collect::<String>()))
        .unwrap();
    let wallet = Wallet::load(&wallet_path).expect("wallet");
    assert_eq!(wallet.keys[0].multisig_lock_data, project_lock);

    let node = Node::open(NodeConfig {
        network: network_with_project(demo::MINER_LOCK, project_lock),
        data_dir: dir.clone(),
        verify_pow: false,
    })
    .expect("node");
    let mut parent = node.tip_id();
    for height in 1..=101 {
        let block =
            demo::block(parent, T0 + 60 * height, 0xC280_0000, 5000 + height, vec![coinbase_to(height, project_lock)]);
        parent = block.id();
        node.accept_block(block).expect("connect");
    }
    let rpc = node.start_rpc("127.0.0.1:0".parse().unwrap()).expect("rpc");
    let client = RpcClient::new(format!("http://{}", rpc.addr)).expect("client");

    let scan = scan(&client, &wallet).expect("scan");
    assert_eq!(scan.utxos.len(), 101);
    assert_eq!(scan.balance(), 101 * 100_000_000);
    assert_eq!(scan.mature_balance(), 2 * 100_000_000);
    let selection = select(&scan.mature(), 150_000_000, FEE).expect("select");
    assert_eq!(selection.inputs.len(), 2);
    let recipient_lock =
        neb_wallet::decode_address("neb", &neb_wallet::address::encode_address("neb", &[0x44; 32])).unwrap();
    let tx =
        build_transaction(&wallet, &selection, recipient_lock, &node.chain_id(), wallet.keys[0].lock_data).expect("tx");
    let txid =
        client.send_raw_transaction(&tx.encode().iter().map(|b| format!("{b:02x}")).collect::<String>()).expect("send");
    assert_eq!(txid.len(), 64);
    assert!(node.has_tx(&tx.txid()));
    drop(rpc);
    drop(node);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&wallet_dir);
}

// A created key file reads back with its key; on Unix it is readable by its owner only.
#[test]
fn a_created_key_file_reads_back_and_is_private() {
    let dir = temp_dir("keyfile");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wallet.txt");
    let created = Wallet::create(&path).expect("create");
    let loaded = Wallet::load(&path).expect("load");
    assert_eq!(loaded.keys.len(), 1);
    assert_eq!(loaded.keys[0].lock_data, created.keys[0].lock_data);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

// The lock hex printed by `neb-wallet lock` is the lock behind the address: 64 hex characters.
#[test]
fn the_lock_hex_matches_the_address() {
    let key = neb_wallet::wallet::WalletKey::generate();
    let text = key.lock_hex();
    assert_eq!(text.len(), 64);
    assert!(text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    let decoded = neb_wallet::decode_address(neb_wallet::ADDRESS_PREFIX, &key.address).unwrap();
    let bytes: Vec<u8> = (0..32).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap()).collect();
    assert_eq!(bytes, decoded.to_vec());
}

// An encrypted key file holds no secret in clear, opens with its password only, and stays encrypted when a key is
// added. A plain file can be encrypted afterwards.
#[test]
fn an_encrypted_key_file_needs_its_password() {
    let dir = temp_dir("encrypted");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wallet.txt");
    let created = Wallet::create_encrypted(&path, "pw one").expect("create");
    let secret = created.keys[0].secret_hex();
    assert!(!std::fs::read_to_string(&path).unwrap().contains(&secret));
    assert!(Wallet::is_encrypted(&path).unwrap());
    assert!(matches!(Wallet::load(&path), Err(neb_wallet::WalletError::Encrypted)));
    assert!(Wallet::load_with_password(&path, "pw two").is_err());
    let mut wallet = Wallet::load_with_password(&path, "pw one").expect("open");
    assert_eq!(wallet.keys[0].lock_data, created.keys[0].lock_data);
    wallet.keys.push(neb_wallet::WalletKey::generate());
    wallet.save().unwrap();
    assert!(Wallet::is_encrypted(&path).unwrap());
    assert_eq!(Wallet::load_with_password(&path, "pw one").unwrap().keys.len(), 2);

    let plain = dir.join("plain.txt");
    let mut wallet = Wallet::create(&plain).expect("create plain");
    wallet.encrypt("pw three").unwrap();
    assert!(Wallet::is_encrypted(&plain).unwrap());
    assert_eq!(Wallet::load_with_password(&plain, "pw three").unwrap().keys[0].lock_data, wallet.keys[0].lock_data);
    std::fs::remove_dir_all(&dir).unwrap();
}

// The command line: `new` with NEB_WALLET_PASSWORD writes an encrypted file, `lock` opens it with the same variable
// and refuses a wrong password.
#[test]
fn the_command_line_encrypts_a_new_wallet() {
    let dir = temp_dir("cli");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wallet.txt");
    let run = |password: &str, command: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_neb-wallet"))
            .args(["--wallet", path.to_str().unwrap(), command])
            .env("NEB_WALLET_PASSWORD", password)
            .output()
            .expect("neb-wallet runs")
    };
    assert!(run("secret words", "new").status.success());
    assert!(Wallet::is_encrypted(&path).unwrap());
    let lock = run("secret words", "lock");
    assert!(lock.status.success());
    assert_eq!(String::from_utf8(lock.stdout).unwrap().trim().len(), 64);
    let wrong = run("other words", "lock");
    assert!(!wrong.status.success());
    assert!(String::from_utf8(wrong.stderr).unwrap().contains("wrong password"));
    std::fs::remove_dir_all(&dir).unwrap();
}
