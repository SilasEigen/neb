//! Address book of the peer-to-peer layer (spec/node.md section 8.4): local policy, no consensus rule.
//!
//! The node learns addresses from three places: the peers it dialled itself, the listening port an inbound peer
//! announces in `version`, and the `addr` message it asked for with `getaddr`. An address is `good` once this node
//! has completed a handshake with it on a connection it opened. Only good addresses are given to other nodes, so an
//! address that nobody could reach does not travel further than the node that was told about it.
//!
//! The book is kept in `peers.txt` in the data directory, one `ADDRESS good|new` per line.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use neb_codec::{Reader, write_varint};

use crate::node::Node;

/// Entries of one `addr` message, at most.
pub const MAX_ADDR: usize = 100;
/// Addresses the book holds, at most.
const MAX_BOOK: usize = 1000;
/// Addresses of one group, at most: a /16 for IPv4 and a /32 for IPv6, so that one network cannot fill the book.
const MAX_PER_GROUP: usize = 8;
/// An address is dropped after this many attempts in a row without a handshake, when the node has other peers.
const MAX_FAILS: u32 = 10;
/// Connections the node keeps towards addresses of the book, besides its seeds.
const BOOK_OUTBOUND: usize = 2;
const TICK: Duration = Duration::from_secs(10);
/// One address that is not good yet is probed at most this often once the connections of the book are up.
const PROBE_EVERY: Duration = Duration::from_secs(60);

struct Entry {
    addr: SocketAddr,
    good: bool,
    fails: u32,
    retry_at: Instant,
}

pub struct AddrBook {
    path: Option<PathBuf>,
    entries: Mutex<Vec<Entry>>,
    listen: Mutex<Option<SocketAddr>>,
    local_ok: AtomicBool,
}

fn group(ip: IpAddr) -> Vec<u8> {
    match ip {
        IpAddr::V4(v4) => v4.octets()[..2].to_vec(),
        IpAddr::V6(v6) => v6.octets()[..4].to_vec(),
    }
}

/// True for an address another machine on the internet could dial.
fn routable(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast())
        }
        IpAddr::V6(v6) => {
            let first = v6.segments()[0];
            // fc00::/7 is local to a site and fe80::/10 to a link.
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || first & 0xfe00 == 0xfc00
                || first & 0xffc0 == 0xfe80)
        }
    }
}

fn canonical(addr: SocketAddr) -> SocketAddr {
    SocketAddr::new(addr.ip().to_canonical(), addr.port())
}

impl AddrBook {
    /// Opens the book kept in `path`, or an empty book in memory. A line that does not parse is skipped.
    pub fn open(path: Option<PathBuf>) -> AddrBook {
        let book = AddrBook {
            path,
            entries: Mutex::new(Vec::new()),
            listen: Mutex::new(None),
            local_ok: AtomicBool::new(false),
        };
        let text = book.path.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
        for line in text.lines() {
            let mut fields = line.split(' ');
            if let (Some(Ok(addr)), state) = (fields.next().map(str::parse::<SocketAddr>), fields.next()) {
                book.insert(addr, state == Some("good"));
            }
        }
        book
    }

    /// Accepts loopback and private addresses. For tests and private networks: a public node never does.
    pub fn allow_local(&self) {
        self.local_ok.store(true, Ordering::Relaxed);
    }

    pub fn set_listen(&self, addr: SocketAddr) {
        *self.listen.lock().expect("book lock") = Some(addr);
    }

    /// The port announced in `version`: 0 when the node does not listen or listens on an address that other
    /// machines cannot dial, so that it is not offered to anyone.
    pub fn advertised_port(&self) -> u16 {
        match *self.listen.lock().expect("book lock") {
            Some(addr) if !addr.ip().is_loopback() || self.local_ok.load(Ordering::Relaxed) => addr.port(),
            _ => 0,
        }
    }

    fn usable(&self, addr: &SocketAddr) -> bool {
        addr.port() != 0 && (self.local_ok.load(Ordering::Relaxed) || routable(addr.ip()))
    }

    /// Adds one address or marks a known one good. A new address is refused when the book or its group is full,
    /// unless it is good and an address that is not good can leave in its place. True when the book changed.
    fn insert(&self, addr: SocketAddr, good: bool) -> bool {
        let addr = canonical(addr);
        if !self.usable(&addr) {
            return false;
        }
        let mut entries = self.entries.lock().expect("book lock");
        if let Some(entry) = entries.iter_mut().find(|e| e.addr == addr) {
            let changed = good && !entry.good;
            if good {
                (entry.good, entry.fails, entry.retry_at) = (true, 0, Instant::now());
            }
            return changed;
        }
        let same_group = |e: &Entry| group(e.addr.ip()) == group(addr.ip());
        let group_full = entries.iter().filter(|e| same_group(e)).count() >= MAX_PER_GROUP;
        if group_full || entries.len() >= MAX_BOOK {
            let leaving = entries.iter().position(|e| !e.good && (!group_full || same_group(e)));
            match leaving {
                Some(i) if good => entries.swap_remove(i),
                _ => return false,
            };
        }
        entries.push(Entry { addr, good, fails: 0, retry_at: Instant::now() });
        true
    }

    /// Keeps addresses this node was told about, at most `MAX_ADDR` of them. They are not good yet.
    pub fn learn(&self, addrs: &[SocketAddr]) {
        let added = addrs.iter().take(MAX_ADDR).filter(|a| self.insert(**a, false)).count();
        if added > 0 {
            self.save();
        }
    }

    /// Records a handshake completed with an address this node dialled.
    pub fn mark_good(&self, addr: SocketAddr) {
        if self.insert(addr, true) {
            self.save();
        }
    }

    /// Counts one attempt on an address: it is not tried again before a delay that doubles from 2 minutes to
    /// about an hour. A handshake clears the count (`mark_good`). After `MAX_FAILS` attempts without one the
    /// address is dropped, unless the node is alone: an offline node must not empty its book.
    pub fn attempt(&self, addr: SocketAddr, online: bool) {
        let addr = canonical(addr);
        let mut entries = self.entries.lock().expect("book lock");
        let Some(i) = entries.iter().position(|e| e.addr == addr) else { return };
        entries[i].fails += 1;
        if online && entries[i].fails >= MAX_FAILS {
            entries.swap_remove(i);
            drop(entries);
            self.save();
            return;
        }
        let delay = Duration::from_secs(60) * 2u32.pow(entries[i].fails.min(6));
        entries[i].retry_at = Instant::now() + delay;
    }

    /// An address to dial now: good or not as asked, past its delay and not refused by `skip`.
    pub fn due(&self, good: bool, skip: impl Fn(&SocketAddr) -> bool) -> Option<SocketAddr> {
        let now = Instant::now();
        let entries = self.entries.lock().expect("book lock");
        let due = entries.iter().filter(|e| e.good == good && e.retry_at <= now && !skip(&e.addr));
        due.map(|e| e.addr).min_by_key(order_key())
    }

    /// At most `MAX_ADDR` good addresses in an order that changes at every call, without those of `asker`.
    pub fn sample(&self, asker: IpAddr) -> Vec<SocketAddr> {
        let asker = asker.to_canonical();
        let entries = self.entries.lock().expect("book lock");
        let mut out: Vec<SocketAddr> =
            entries.iter().filter(|e| e.good && e.addr.ip() != asker).map(|e| e.addr).collect();
        out.sort_by_cached_key(order_key());
        out.truncate(MAX_ADDR);
        out
    }

    /// The addresses of the book and whether each is good.
    pub fn addresses(&self) -> Vec<(SocketAddr, bool)> {
        self.entries.lock().expect("book lock").iter().map(|e| (e.addr, e.good)).collect()
    }

    fn save(&self) {
        let Some(path) = &self.path else { return };
        let mut text = String::new();
        for (addr, good) in self.addresses() {
            text.push_str(&format!("{addr} {}\n", if good { "good" } else { "new" }));
        }
        // A book that cannot be written is only lost at the next start.
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

/// A sort key that differs at every call. Not secret: it only spreads the dials between addresses.
fn order_key() -> impl FnMut(&SocketAddr) -> [u8; 8] {
    let salt = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    move |addr| {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&salt.to_le_bytes());
        hasher.update(addr.to_string().as_bytes());
        hasher.finalize().as_bytes()[..8].try_into().expect("eight bytes")
    }
}

/// Payload of `addr` (spec/node.md section 8.2): `count (varint) || entries`, an entry being the 16 bytes of the
/// IPv6 address, an IPv4 address in its mapped form, then the port (u16).
pub fn encode_addr(addrs: &[SocketAddr]) -> Vec<u8> {
    let addrs = &addrs[..addrs.len().min(MAX_ADDR)];
    let mut out = Vec::with_capacity(1 + addrs.len() * 18);
    write_varint(&mut out, addrs.len() as u64);
    for addr in addrs {
        let v6 = match addr.ip() {
            IpAddr::V4(v4) => v4.to_ipv6_mapped(),
            IpAddr::V6(v6) => v6,
        };
        out.extend_from_slice(&v6.octets());
        out.extend_from_slice(&addr.port().to_le_bytes());
    }
    out
}

pub fn parse_addr(payload: &[u8]) -> Result<Vec<SocketAddr>, String> {
    let mut r = Reader::new(payload);
    let n = r.read_varint().map_err(|e| e.to_string())?;
    if n > MAX_ADDR as u64 {
        return Err(format!("address list over {MAX_ADDR}"));
    }
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let ip: [u8; 16] = r.read_exact(16).map_err(|e| e.to_string())?.try_into().expect("sixteen bytes");
        let port = r.read_u16().map_err(|e| e.to_string())?;
        out.push(canonical(SocketAddr::new(IpAddr::from(ip), port)));
    }
    r.expect_done().map_err(|e| e.to_string())?;
    Ok(out)
}

/// One round of the connector: with fewer than `BOOK_OUTBOUND` of its own connections up, the node dials a good
/// address; otherwise, when `may_probe`, it probes one address that is not good yet (a handshake, then the
/// connection is closed and the address becomes good). True when a probe was sent.
pub fn dial_from_book(node: &Node, dialled: &mut Vec<SocketAddr>, may_probe: bool) -> bool {
    let book = node.address_book();
    dialled.retain(|addr| node.is_peer_connected(addr));
    let connected = node.peer_count();
    if connected >= node.network().max_peers {
        return false;
    }
    let skip = |addr: &SocketAddr| node.is_peer_connected(addr) || node.is_banned(addr.ip());
    if dialled.len() < BOOK_OUTBOUND
        && let Some(addr) = book.due(true, skip)
    {
        book.attempt(addr, connected > 0);
        if node.connect_peer(addr).is_ok() {
            dialled.push(addr);
        }
    } else if may_probe && let Some(addr) = book.due(false, skip) {
        book.attempt(addr, connected > 0);
        let _ = node.probe_peer(addr);
        return true;
    }
    false
}

/// Keeps `BOOK_OUTBOUND` connections towards good addresses of the book, so that the node stays on the network
/// when its seeds are down, and probes one address that is not good yet per `PROBE_EVERY`.
pub fn spawn_book_connector(node: Node) {
    thread::spawn(move || {
        let mut dialled = Vec::new();
        let mut probed = Instant::now();
        loop {
            thread::sleep(TICK);
            if dial_from_book(&node, &mut dialled, probed.elapsed() >= PROBE_EVERY) {
                probed = Instant::now();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(text: &str) -> SocketAddr {
        text.parse().unwrap()
    }

    #[test]
    fn only_addresses_that_can_be_dialled_from_outside_are_kept() {
        let book = AddrBook::open(None);
        book.learn(&[
            addr("127.0.0.1:3440"),
            addr("10.1.2.3:3440"),
            addr("192.168.1.9:3440"),
            addr("169.254.0.1:3440"),
            addr("0.0.0.0:3440"),
            addr("[::1]:3440"),
            addr("[fe80::1]:3440"),
            addr("[fd00::1]:3440"),
            addr("8.8.8.8:0"),
            addr("8.8.8.8:3440"),
            addr("[::ffff:8.8.4.4]:3440"),
            addr("[2001:db8::1]:3440"),
        ]);
        let mut kept: Vec<String> = book.addresses().iter().map(|(a, _)| a.to_string()).collect();
        kept.sort();
        assert_eq!(kept, ["8.8.4.4:3440", "8.8.8.8:3440", "[2001:db8::1]:3440"]);
        // a private network asks for local addresses
        book.allow_local();
        book.learn(&[addr("127.0.0.1:3440")]);
        assert_eq!(book.addresses().len(), 4);
    }

    #[test]
    fn one_network_and_one_message_cannot_fill_the_book() {
        let book = AddrBook::open(None);
        let same_16: Vec<SocketAddr> = (0..50).map(|i| addr(&format!("9.9.{i}.1:3440"))).collect();
        book.learn(&same_16);
        assert_eq!(book.addresses().len(), MAX_PER_GROUP);
        // a message is read up to MAX_ADDR entries
        let book = AddrBook::open(None);
        let many: Vec<SocketAddr> = (0..150).map(|i| addr(&format!("{}.1.1.1:3440", 11 + i))).collect();
        book.learn(&many);
        assert_eq!(book.addresses().len(), MAX_ADDR);
        // a good address takes the place of one that is not good in a full group
        let book = AddrBook::open(None);
        book.learn(&same_16);
        book.mark_good(addr("9.9.200.1:3440"));
        assert_eq!(book.addresses().len(), MAX_PER_GROUP);
        assert!(book.addresses().contains(&(addr("9.9.200.1:3440"), true)));
    }

    #[test]
    fn only_good_addresses_are_given_and_never_to_their_own_node() {
        let book = AddrBook::open(None);
        book.learn(&[addr("8.8.8.8:3440"), addr("9.9.9.9:3440")]);
        assert!(book.sample("1.1.1.1".parse().unwrap()).is_empty(), "nothing is good yet");
        book.mark_good(addr("8.8.8.8:3440"));
        book.mark_good(addr("7.7.7.7:3440"));
        let mut given = book.sample("7.7.7.7".parse().unwrap());
        assert_eq!(given, [addr("8.8.8.8:3440")]);
        given = book.sample("1.1.1.1".parse().unwrap());
        given.sort();
        assert_eq!(given, [addr("7.7.7.7:3440"), addr("8.8.8.8:3440")]);
    }

    #[test]
    fn an_address_waits_after_an_attempt_and_leaves_after_too_many() {
        let book = AddrBook::open(None);
        book.learn(&[addr("8.8.8.8:3440")]);
        assert_eq!(book.due(false, |_| false), Some(addr("8.8.8.8:3440")));
        assert_eq!(book.due(true, |_| false), None);
        assert_eq!(book.due(false, |_| true), None, "a connected address is skipped");
        book.attempt(addr("8.8.8.8:3440"), true);
        assert_eq!(book.due(false, |_| false), None, "not before its delay");
        // a handshake clears the count
        book.mark_good(addr("8.8.8.8:3440"));
        assert_eq!(book.due(true, |_| false), Some(addr("8.8.8.8:3440")));
        // a node without any peer keeps its book
        for _ in 0..MAX_FAILS + 5 {
            book.attempt(addr("8.8.8.8:3440"), false);
        }
        assert_eq!(book.addresses().len(), 1);
        book.mark_good(addr("8.8.8.8:3440"));
        for _ in 0..MAX_FAILS {
            book.attempt(addr("8.8.8.8:3440"), true);
        }
        assert!(book.addresses().is_empty());
    }

    #[test]
    fn the_book_is_read_back_from_its_file() {
        let dir = std::env::temp_dir().join(format!("neb-addrbook-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("peers.txt");
        let _ = std::fs::remove_file(&path);
        let book = AddrBook::open(Some(path.clone()));
        book.learn(&[addr("8.8.8.8:3440")]);
        book.mark_good(addr("[2001:db8::1]:3441"));
        std::fs::write(&path, std::fs::read_to_string(&path).unwrap() + "not an address\n").unwrap();
        let mut back = AddrBook::open(Some(path)).addresses();
        back.sort();
        assert_eq!(back, [(addr("8.8.8.8:3440"), false), (addr("[2001:db8::1]:3441"), true)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_addr_payload_round_trips_and_a_long_or_cut_one_is_refused() {
        let list = [addr("8.8.8.8:3440"), addr("[2001:db8::1]:65535")];
        let payload = encode_addr(&list);
        assert_eq!(payload.len(), 1 + 2 * 18);
        assert_eq!(parse_addr(&payload).unwrap(), list);
        assert_eq!(parse_addr(&[0]).unwrap(), [], "the answer of a node without a book");
        assert!(parse_addr(&payload[..payload.len() - 1]).is_err());
        let mut long = Vec::new();
        write_varint(&mut long, MAX_ADDR as u64 + 1);
        long.resize(long.len() + (MAX_ADDR + 1) * 18, 0);
        assert!(parse_addr(&long).is_err());
    }

    #[test]
    fn a_node_that_listens_locally_announces_no_port() {
        let book = AddrBook::open(None);
        assert_eq!(book.advertised_port(), 0);
        book.set_listen(addr("127.0.0.1:3440"));
        assert_eq!(book.advertised_port(), 0);
        book.set_listen(addr("0.0.0.0:3440"));
        assert_eq!(book.advertised_port(), 3440);
    }
}
