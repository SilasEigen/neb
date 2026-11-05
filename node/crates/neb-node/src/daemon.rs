//! `neb-node --run`: opens the block store, replays the chain, starts the peer
//! listener (spec/node.md section 8), the JSON-RPC server (section 7.2) and the
//! work service (section 7.1), and dials the configured peers and seeds. A peer or a seed is `HOST:PORT`, where HOST
//! is an IP address or a name; a name is resolved at every dial, so a seed that moves to another address is found again.
//!
//! The work service and JSON-RPC bind to 127.0.0.1 only: they carry no
//! authentication. Only the peer listener address is configurable (`--listen`).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use neb_work::frames::{self, FrameOut, Hello, MSG_HELLO, MSG_SHARE, MSG_SUBMIT, ResultMsg, ShareMsg, Submit};

use crate::budget::DEFAULT_POW_BUDGET;
use crate::miners::MinerWriter;
use crate::node::{Node, NodeConfig, now};
use crate::params;
use crate::style::{ColorMode, Live, Log};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

struct Options {
    dataset: Option<String>,
    datadir: String,
    listen: String,
    p2p_port: u16,
    work_port: u16,
    rpc_port: u16,
    connects: Vec<String>,
    seeds: Vec<String>,
    no_seeds: bool,
    max_peers: usize,
    miner_lock: Option<[u8; 32]>,
    pow_budget: NonZeroU32,
    color: ColorMode,
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        dataset: None,
        datadir: params::DEFAULT_DATADIR.into(),
        listen: "127.0.0.1".into(),
        p2p_port: params::P2P_PORT,
        work_port: params::WORK_PORT,
        rpc_port: params::RPC_PORT,
        connects: Vec::new(),
        seeds: Vec::new(),
        no_seeds: false,
        max_peers: params::MAX_PEERS,
        miner_lock: None,
        pow_budget: NonZeroU32::new(DEFAULT_POW_BUDGET).expect("nonzero"),
        color: ColorMode::Auto,
    };
    let mut i = 0;
    while i < args.len() {
        let value = |i: usize| args.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", args[i]));
        match args[i].as_str() {
            "--dataset" => {
                options.dataset = Some(value(i)?);
                i += 2;
            }
            "--datadir" => {
                options.datadir = value(i)?;
                i += 2;
            }
            "--listen" => {
                options.listen = value(i)?;
                i += 2;
            }
            "--port" => {
                options.p2p_port = value(i)?.parse().map_err(|_| "bad port".to_string())?;
                i += 2;
            }
            "--work-port" => {
                options.work_port = value(i)?.parse().map_err(|_| "bad work port".to_string())?;
                i += 2;
            }
            "--rpc-port" => {
                options.rpc_port = value(i)?.parse().map_err(|_| "bad rpc port".to_string())?;
                i += 2;
            }
            "--connect" => {
                options.connects.push(value(i)?);
                i += 2;
            }
            "--seed" | "--addnode" => {
                options.seeds.push(value(i)?);
                i += 2;
            }
            "--no-seeds" => {
                options.no_seeds = true;
                i += 1;
            }
            "--max-peers" => {
                options.max_peers = value(i)?
                    .parse()
                    .ok()
                    .filter(|n| (1..=params::MAX_PEERS_LIMIT).contains(n))
                    .ok_or_else(|| format!("--max-peers takes a number from 1 to {}", params::MAX_PEERS_LIMIT))?;
                i += 2;
            }
            "--miner-lock" => {
                let text = value(i)?;
                options.miner_lock = Some(parse_hex32(&text).ok_or("--miner-lock takes 64 hex characters")?);
                i += 2;
            }
            "--pow-budget" => {
                options.pow_budget =
                    value(i)?.parse().map_err(|_| "--pow-budget takes a positive number".to_string())?;
                i += 2;
            }
            "--color" => {
                options.color = match value(i)?.as_str() {
                    "auto" => ColorMode::Auto,
                    "always" => ColorMode::Always,
                    "never" => ColorMode::Never,
                    _ => return Err("--color takes auto, always or never".into()),
                };
                i += 2;
            }
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(options)
}

pub struct WorkHandle {
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Drop for WorkHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Node {
    /// Starts the work-service listener: one thread accepts miner sessions
    /// and one thread per session serves it (spec/node.md section 7.1). The
    /// returned handle stops the accept loop when dropped; session threads end
    /// when their sockets close. The listener is non-blocking and polled every
    /// 50 ms so that the handle can stop it.
    pub fn start_work_service(&self, addr: SocketAddr, log: Arc<Log>, live: Arc<Live>) -> std::io::Result<WorkHandle> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let local = listener.local_addr()?;
        let node = self.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let join = std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        let node = node.clone();
                        let log = log.clone();
                        let live = live.clone();
                        std::thread::spawn(move || serve_miner(stream, node, log, live));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(WorkHandle { addr: local, stop, join: Some(join) })
    }
}

pub fn run(args: &[String]) -> i32 {
    let options = match parse_options(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("neb-node --run: {e}");
            return 2;
        }
    };
    let Some(dataset_path) = &options.dataset else {
        eprintln!("neb-node --run: --dataset FILE is required (see tools/make_mnist_dataset.py)");
        return 2;
    };
    let dataset = match params::load_dataset(std::path::Path::new(dataset_path)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cannot load the dataset: {e}");
            return 1;
        }
    };
    let mut configured = params::testnet(dataset);
    configured.p2p_port = options.p2p_port;
    configured.work_port = options.work_port;
    configured.rpc_port = options.rpc_port;
    configured.max_peers = options.max_peers;
    let node = match Node::open(NodeConfig {
        network: configured.clone(),
        data_dir: options.datadir.clone().into(),
        verify_pow: true,
    }) {
        Ok(n) => n.with_pow_budget(options.pow_budget),
        Err(e) => {
            eprintln!("cannot open the chain store: {e}");
            return 1;
        }
    };
    if let Some(lock) = options.miner_lock {
        node.set_miner_lock(lock);
    }
    let height = node.height();
    let genesis_id = node.chain_id();
    let live = Live::new();
    live.set_height(height);
    let log = Arc::new(Log::new(live.clone(), options.color));
    log.startup(&configured, &hex(&genesis_id), height, &options.datadir, options.work_port);
    match options.miner_lock {
        Some(lock) => log.note(&format!("miner coins go to lock {}", hex(&lock))),
        None => log.note("no --miner-lock: the work service refuses miners (relay mode)"),
    }

    let bind = format!("{}:{}", options.listen, options.p2p_port);
    let p2p = match node.start_p2p(bind.parse().expect("listen address")) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("cannot listen for peers on {bind}: {e}");
            return 1;
        }
    };
    log.note(&format!("peer listener {}", p2p.addr));
    let rpc_bind = format!("127.0.0.1:{}", options.rpc_port);
    let rpc = match node.start_rpc(rpc_bind.parse().expect("rpc address")) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("cannot listen for json-rpc on {rpc_bind}: {e}");
            return 1;
        }
    };
    log.note(&format!("json-rpc {}", rpc.addr));
    for target in &options.connects {
        if !is_target(target) {
            eprintln!("neb-node --run: bad peer address {target}, expected HOST:PORT");
            return 2;
        }
        spawn_connector(node.clone(), target.clone());
    }
    let seeds: Vec<String> = if options.no_seeds || !options.connects.is_empty() || !options.seeds.is_empty() {
        options.seeds.clone()
    } else {
        params::DEFAULT_SEEDS.iter().map(|s| s.to_string()).collect()
    };
    for target in &seeds {
        if !is_target(target) {
            eprintln!("neb-node --run: bad seed address {target}, expected HOST:PORT");
            return 2;
        }
        spawn_seed_connector(node.clone(), target.clone());
    }
    log.spawn_status();

    let work_bind = format!("127.0.0.1:{}", options.work_port);
    let work = match node.start_work_service(work_bind.parse().expect("work address"), log.clone(), live.clone()) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("cannot listen for miners on {work_bind}: {e}");
            return 1;
        }
    };
    log.note(&format!("work service {}", work.addr));

    // The handles of the peer, rpc and work listeners stop their threads when
    // dropped, so this thread must keep them alive.
    loop {
        std::thread::park();
    }
}

/// True when `target` is `HOST:PORT` with a non-empty host (an IP address, a bracketed IPv6 address or a name) and a
/// port. Whether the name resolves is checked at every dial, not here: a seed may be published after the node starts.
fn is_target(target: &str) -> bool {
    match target.rsplit_once(':') {
        Some((host, port)) => {
            !host.is_empty() && !host.contains(char::is_whitespace) && port.parse::<u16>().is_ok_and(|p| p > 0)
        }
        None => false,
    }
}

/// The addresses that `target` names now; empty when the name does not resolve.
fn resolve(target: &str) -> Vec<SocketAddr> {
    target.to_socket_addrs().map(|addrs| addrs.collect()).unwrap_or_default()
}

/// Dials a `--connect` peer every 5 s until one connection succeeds; it is
/// not redialled afterwards.
fn spawn_connector(node: Node, target: String) {
    std::thread::spawn(move || {
        loop {
            if resolve(&target).into_iter().any(|addr| node.connect_peer(addr).is_ok()) {
                return;
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    });
}

/// Keeps one seed peer dialled: a lost connection is re-dialled, and a dial
/// that fails (or a connection the peer cap refuses) backs off from 1 s,
/// doubling to 30 s. The delay resets while the peer is connected. The name is
/// resolved again at every dial.
fn spawn_seed_connector(node: Node, target: String) {
    std::thread::spawn(move || {
        let mut delay = Duration::from_secs(1);
        loop {
            let addrs = resolve(&target);
            if addrs.iter().any(|addr| node.is_peer_connected(addr)) {
                delay = Duration::from_secs(1);
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
            if addrs
                .iter()
                .any(|addr| node.connect_peer(*addr).is_ok() && wait_registered(&node, addr, Duration::from_secs(5)))
            {
                continue;
            }
            std::thread::sleep(delay);
            delay = (delay * 2).min(Duration::from_secs(30));
        }
    });
}

/// Waits up to `timeout` for a dialled address to appear in the peer set.
fn wait_registered(node: &Node, addr: &std::net::SocketAddr, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if node.is_peer_connected(addr) {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Sends the model frames and a fresh job to one session in one write, so the
/// miner never holds a job without the model it refers to.
fn send_job(node: &Node, writer: &MinerWriter, miner_lock: [u8; 32]) -> std::io::Result<()> {
    let id = node.make_job(miner_lock, now()).map_err(std::io::Error::other)?;
    let frames = node.work_frames(id).ok_or_else(|| std::io::Error::other("job disappeared"))?;
    send(writer, &frames)
}

fn send(writer: &MinerWriter, frame: &[u8]) -> std::io::Result<()> {
    writer.lock().expect("miner writer").write_all(frame)
}

fn serve_miner(mut stream: TcpStream, node: Node, log: Arc<Log>, live: Arc<Live>) {
    let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    let Some(lock) = node.miner_lock() else {
        log.note(&format!(
            "refused the miner {peer}: start the node with --miner-lock (the lock printed by `neb-wallet lock`)"
        ));
        return;
    };
    live.miner_joined();
    log.miner_joined(&peer);
    let writer = match stream.try_clone() {
        Ok(w) => Arc::new(Mutex::new(w)),
        Err(_) => {
            live.miner_left();
            log.miner_left(&peer);
            return;
        }
    };
    let session = node.register_miner(writer.clone());
    if let Err(e) = send_job(&node, &writer, lock) {
        log.frame_error(&format!("send: {e}"));
        node.unregister_miner(session);
        live.miner_left();
        log.miner_left(&peer);
        return;
    }
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    'read: loop {
        let n = match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        buf.extend_from_slice(&tmp[..n]);
        loop {
            match frames::parse_frame(&buf) {
                Ok(FrameOut::Need(_)) => break,
                Ok(FrameOut::Message { msg_type, payload, consumed }) => {
                    let payload = payload.to_vec();
                    buf.drain(..consumed);
                    match msg_type {
                        MSG_HELLO => {
                            if let Ok(h) = Hello::decode(&payload) {
                                log.miner_named(&h.name, h.version);
                            }
                        }
                        MSG_SHARE => {
                            if let Ok(share) = ShareMsg::decode(&payload) {
                                let _ = node.accept_share(share.job_id, share.nonce);
                            }
                        }
                        MSG_SUBMIT => {
                            let sub = match Submit::decode(&payload) {
                                Ok(s) => s,
                                Err(e) => {
                                    log.frame_error(&format!("bad submission frame: {e}"));
                                    continue;
                                }
                            };
                            match node.accept_submit(sub.id, &sub.extranonce, sub.time, sub.nonce) {
                                Ok(height) => {
                                    let tip = hex(&node.tip_id());
                                    live.set_height(height);
                                    let seconds = live.block_seen();
                                    log.block(height, &tip, seconds);
                                    // Connecting the block broadcast a fresh job
                                    // to every session, this one included.
                                    let _ = send(&writer, &ResultMsg { accepted: true, code: 0 }.encode());
                                }
                                Err(code) => {
                                    log.rejected(code);
                                    let _ = send(&writer, &ResultMsg { accepted: false, code }.encode());
                                    if let Err(e) = send_job(&node, &writer, lock) {
                                        log.frame_error(&format!("send: {e}"));
                                        break 'read;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Err(e) => {
                    log.frame_error(&format!("{e}"));
                    break 'read;
                }
            }
        }
    }
    node.unregister_miner(session);
    live.miner_left();
    log.miner_left(&peer);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_peer_is_host_and_port() {
        for good in ["seed.example.org:3440", "127.0.0.1:3440", "[::1]:3440", "localhost:1"] {
            assert!(is_target(good), "{good}");
        }
        for bad in ["seed.example.org", "seed:", ":3440", "seed:port", "seed:70000", "seed:0", "see d:3440", ""] {
            assert!(!is_target(bad), "{bad}");
        }
        assert!(params::DEFAULT_SEEDS.iter().all(|s| is_target(s)), "the default seeds are HOST:PORT");
    }

    #[test]
    fn a_name_is_resolved_at_the_dial() {
        assert!(resolve("localhost:3440").iter().all(|a| a.port() == 3440 && a.ip().is_loopback()));
        assert!(!resolve("localhost:3440").is_empty());
        assert!(resolve("no-such-host.invalid:3440").is_empty());
    }

    #[test]
    fn the_peer_cap_and_the_seed_switch_are_options() {
        let o = parse_options(&args(&["--max-peers", "256", "--no-seeds", "--seed", "a.example:3440"])).unwrap();
        assert_eq!((o.max_peers, o.no_seeds, o.seeds.len()), (256, true, 1));
        assert_eq!(parse_options(&[]).unwrap().max_peers, params::MAX_PEERS);
        for bad in ["0", "1025", "-1", "many"] {
            assert!(parse_options(&args(&["--max-peers", bad])).is_err(), "{bad}");
        }
    }
}
