//! Address exchange between nodes over loopback (spec/node.md section 8.4): a node learns the peers of its
//! peer with `getaddr`, proves an address with a handshake before it gives it to others, and dials the book when
//! told to. The books accept local addresses here; a public node refuses them.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use neb_chain::demo;
use neb_node::addrbook::{dial_from_book, encode_addr};
use neb_node::p2p::encode_message;
use neb_node::params::NetworkParams;
use neb_node::{Node, NodeConfig};

const T0: u32 = 1_700_000_000;
const MAGIC: [u8; 4] = [0xAB, 0xCD, 0xEF, 0x01];
static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("neb-node-{tag}-{}-{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn open(dir: &Path) -> Node {
    let network = NetworkParams {
        name: "test",
        magic: MAGIC,
        p2p_port: 0,
        work_port: 0,
        rpc_port: 0,
        address_prefix: "neb",
        protocol_version: 1,
        max_peers: 8,
        user_agent: "neb-test",
        dataset_names: &[],
        genesis: demo::genesis_to(T0, demo::MINER_LOCK),
        chain: demo::demo_params(T0, demo::PROJECT_LOCK),
    };
    let node = Node::open(NodeConfig { network, data_dir: dir.to_path_buf(), verify_pow: false }).expect("node");
    node.address_book().allow_local();
    node
}

fn wait_for<F: Fn() -> bool>(f: F, seconds: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    f()
}

fn has(node: &Node, addr: SocketAddr, good: bool) -> bool {
    node.address_book().addresses().contains(&(addr, good))
}

// A listens on another address than B and C, since a node is never given the addresses of its own IP.
#[test]
fn a_node_learns_the_peer_of_its_peer_proves_it_and_dials_it() {
    let dirs = [temp_dir("book-a"), temp_dir("book-b"), temp_dir("book-c")];
    let (a, b, c) = (open(&dirs[0]), open(&dirs[1]), open(&dirs[2]));
    let listen_a = a.start_p2p("127.0.0.2:0".parse().unwrap()).expect("listen");
    let listen_b = b.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");

    // B dials A: A is good for B, and A is told the listening port of B, which it has not proved.
    b.connect_peer(listen_a.addr).expect("connect");
    assert!(wait_for(|| has(&b, listen_a.addr, true), 10), "B did not keep the peer it dialled");
    assert!(wait_for(|| has(&a, listen_b.addr, false), 10), "A did not keep the announced port of B");
    assert!(a.address_book().sample("127.0.0.9".parse().unwrap()).is_empty(), "A gives an address it never reached");

    // C dials B and asks for its peers: it learns A, not good yet.
    c.connect_peer(listen_b.addr).expect("connect");
    assert!(wait_for(|| has(&c, listen_a.addr, false), 10), "C did not learn A from B");
    assert!(!c.is_peer_connected(&listen_a.addr));

    // C has no good address left to dial (B is connected), so it probes A: a handshake, no session kept.
    let mut dialled = Vec::new();
    assert!(dial_from_book(&c, &mut dialled, true), "C did not probe");
    assert!(wait_for(|| has(&c, listen_a.addr, true), 10), "the probe did not prove A");
    assert!(wait_for(|| !c.is_peer_connected(&listen_a.addr), 10), "a probe keeps no session");

    // A is good now: the next round dials it for a session.
    assert!(!dial_from_book(&c, &mut dialled, true));
    assert_eq!(dialled, [listen_a.addr]);
    assert!(wait_for(|| c.is_peer_connected(&listen_a.addr), 10), "C did not connect to A");

    // The book of C is in its data directory (the unit tests of the book read such a file back).
    let text = std::fs::read_to_string(dirs[2].join("peers.txt")).expect("peers file");
    assert!(
        text.contains(&format!(
            "{} good
",
            listen_a.addr
        )),
        "{text}"
    );
    assert!(
        text.contains(&format!(
            "{} good
",
            listen_b.addr
        )),
        "{text}"
    );
    drop((listen_a, listen_b, a, b, c));
    for dir in &dirs {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Reads one frame and returns its command.
fn read_command(stream: &mut TcpStream) -> Option<String> {
    let mut header = [0u8; 24];
    stream.read_exact(&mut header).ok()?;
    let length = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
    let mut payload = vec![0u8; length];
    stream.read_exact(&mut payload).ok()?;
    Some(String::from_utf8_lossy(&header[4..16]).trim_end_matches('\0').to_string())
}

// A peer that was not asked cannot write into the book, and `getaddr` is answered once per session.
#[test]
fn an_addr_nobody_asked_for_is_ignored_and_getaddr_is_answered_once() {
    let dir = temp_dir("book-raw");
    let a = open(&dir);
    let listen = a.start_p2p("127.0.0.1:0".parse().unwrap()).expect("listen");
    let mut peer = TcpStream::connect(listen.addr).expect("connect");
    peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();

    // version (protocol, height, nonce, genesis id, port 0, user agent) then verack
    let mut version = Vec::new();
    version.extend_from_slice(&1u32.to_le_bytes());
    version.extend_from_slice(&0u32.to_le_bytes());
    version.extend_from_slice(&7u64.to_le_bytes());
    version.extend_from_slice(&a.chain_id());
    version.extend_from_slice(&0u16.to_le_bytes());
    version.extend_from_slice(&[4, b't', b'e', b's', b't']);
    let told: SocketAddr = "127.0.0.9:3440".parse().unwrap();
    for (command, payload) in [
        ("version", version),
        ("verack", Vec::new()),
        ("addr", encode_addr(&[told])),
        ("getaddr", Vec::new()),
        ("getaddr", Vec::new()),
        ("ping", vec![1u8; 8]),
    ] {
        peer.write_all(&encode_message(MAGIC, command, &payload).unwrap()).expect("write");
    }
    // everything the node sends up to the pong that answers the last message
    let mut commands = Vec::new();
    while let Some(command) = read_command(&mut peer) {
        let done = command == "pong";
        commands.push(command);
        if done {
            break;
        }
    }
    assert_eq!(commands.last().map(String::as_str), Some("pong"), "{commands:?}");
    assert_eq!(commands.iter().filter(|c| *c == "addr").count(), 1, "{commands:?}");
    assert!(!commands.iter().any(|c| c == "getaddr"), "a node asks only the peers it dialled: {commands:?}");
    assert!(a.address_book().addresses().is_empty(), "the book took an address nobody asked for");
    drop((peer, listen, a));
    let _ = std::fs::remove_dir_all(&dir);
}
