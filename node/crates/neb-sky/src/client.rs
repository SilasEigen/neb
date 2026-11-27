//! The miner side of the transport (`spec/sky-search.md` section 10): ask the batch server for a batch, run the
//! search, post the result. Plain HTTP over the standard library; nothing secret travels and the server checks
//! every result.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Longest reply read from the server.
const MOST_BYTES: u64 = 64 << 20;
/// Seconds between two tries when the server has no batch or does not answer.
const WAIT: u64 = 60;

/// Where the batches come from and what the miner says of itself. The three text fields are optional.
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// The batch server, `http://host:port`.
    pub url: String,
    /// Name and version of the mining software.
    pub software: String,
    /// Model of the card or processor.
    pub device: String,
    /// A NEB address.
    pub address: String,
    /// Threads that run the search.
    pub threads: usize,
}

/// Counters of a running worker.
#[derive(Debug, Default)]
pub struct Stats {
    /// Results the server took.
    pub batches: AtomicU64,
    /// Light curves in those results.
    pub curves: AtomicU64,
    /// Results the server refused.
    pub refused: AtomicU64,
    /// Requests that failed.
    pub errors: AtomicU64,
}

/// Why a request failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The address is not `http://host[:port]`.
    Url(String),
    /// The connection failed or the reply was cut.
    Io(std::io::Error),
    /// The reply is not an HTTP reply.
    Reply,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Url(u) => write!(f, "the sky server address must be http://host:port, got {u}"),
            Error::Io(e) => write!(f, "sky server: {e}"),
            Error::Reply => f.write_str("the sky server sent a reply that is not HTTP"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// `host:port` and the path prefix of an `http://` address.
fn split(url: &str) -> Result<(String, String), Error> {
    let rest = url.strip_prefix("http://").ok_or_else(|| Error::Url(url.to_string()))?;
    let (host, prefix) = rest.split_once('/').map_or((rest, ""), |(h, p)| (h, p.trim_end_matches('/')));
    if host.is_empty() {
        return Err(Error::Url(url.to_string()));
    }
    let host = if host.contains(':') { host.to_string() } else { format!("{host}:80") };
    Ok((host, if prefix.is_empty() { String::new() } else { format!("/{prefix}") }))
}

/// A query value with every byte outside letters, digits, `-`, `.`, `_` and `~` percent-encoded.
fn encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// One request; the status code and the body of the reply.
fn request(config: &Config, method: &str, route: &str, body: &[u8]) -> Result<(u16, Vec<u8>), Error> {
    let (host, prefix) = split(&config.url)?;
    let query = format!(
        "software={}&device={}&address={}",
        encode(&config.software),
        encode(&config.device),
        encode(&config.address)
    );
    let addr = host.to_socket_addrs()?.next().ok_or_else(|| Error::Url(config.url.clone()))?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(20))?;
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    stream.set_write_timeout(Some(Duration::from_secs(120)))?;
    let head = format!(
        "{method} {prefix}{route}?{query} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    let mut reply = Vec::new();
    stream.take(MOST_BYTES).read_to_end(&mut reply)?;
    let end = reply.windows(4).position(|w| w == b"\r\n\r\n").ok_or(Error::Reply)?;
    let status =
        std::str::from_utf8(&reply[..end]).ok().and_then(|h| h.split(' ').nth(1)?.parse().ok()).ok_or(Error::Reply)?;
    Ok((status, reply.split_off(end + 4)))
}

/// One round: a batch asked, searched and posted. `Ok(true)` when the server had a batch.
///
/// # Errors
/// When a request fails. A result the server refuses is counted, not an error.
pub fn round(config: &Config, stats: &Stats) -> Result<bool, Error> {
    let (status, raw) = request(config, "GET", "/sky/v1/batch", &[])?;
    if status != 200 {
        return Ok(false);
    }
    let Ok(result) = crate::run_threads(&raw, config.threads) else {
        return Err(Error::Reply);
    };
    let (status, _) = request(config, "POST", "/sky/v1/result", &result)?;
    if status == 200 {
        stats.batches.fetch_add(1, Ordering::Relaxed);
        stats.curves.fetch_add(((result.len() - 44) / (crate::KEEP * crate::ENTRY_BYTES)) as u64, Ordering::Relaxed);
    } else if status != 409 {
        stats.refused.fetch_add(1, Ordering::Relaxed); // 409: the batch was settled while this miner searched it
    }
    Ok(true)
}

/// Starts a thread that runs rounds until `stop` is set. It waits a minute when the server has no batch or does
/// not answer: a server that is away never slows mining down.
pub fn spawn(config: Config, stop: Arc<AtomicBool>) -> (JoinHandle<()>, Arc<Stats>) {
    let stats = Arc::new(Stats::default());
    let shared = stats.clone();
    let handle = thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            let busy = round(&config, &shared).unwrap_or_else(|_| {
                shared.errors.fetch_add(1, Ordering::Relaxed);
                false
            });
            for _ in 0..if busy { 0 } else { WAIT } {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                thread::sleep(Duration::from_secs(1));
            }
        }
    });
    (handle, stats)
}

/// The options of the sky search on a command line: `--sky URL`, `--sky-threads N`, `--sky-address ADDRESS` and
/// `--sky-device TEXT`.
#[derive(Clone, Debug, Default)]
pub struct Options {
    url: Option<String>,
    threads: usize,
    address: String,
    device: Option<String>,
}

impl Options {
    /// Takes one option and its value: `true` when it is a valid option of the sky search.
    pub fn take(&mut self, flag: &str, value: &str) -> bool {
        match flag {
            "--sky" => self.url = Some(value.to_string()),
            "--sky-address" => self.address = value.to_string(),
            "--sky-device" => self.device = Some(value.to_string()),
            "--sky-threads" => match value.parse::<usize>() {
                Ok(n) if n > 0 => self.threads = n,
                _ => return false,
            },
            _ => return false,
        }
        true
    }

    /// The settings of the worker, or `None` when `--sky` was not given. `device` is what the server is told when
    /// `--sky-device` is absent, `threads` how many threads search when `--sky-threads` is absent.
    ///
    /// # Errors
    /// When the address is not `http://host[:port]`.
    pub fn config(self, software: &str, device: &str, threads: usize) -> Result<Option<Config>, Error> {
        let Some(url) = self.url else { return Ok(None) };
        split(&url)?;
        Ok(Some(Config {
            url,
            software: software.to_string(),
            device: self.device.unwrap_or_else(|| device.to_string()),
            address: self.address,
            threads: if self.threads > 0 { self.threads } else { threads.max(1) },
        }))
    }

    /// Starts the worker when `--sky` was given; one thread unless `--sky-threads` says otherwise.
    ///
    /// # Errors
    /// When the address is not `http://host[:port]`.
    pub fn start(self, software: &str, device: &str, stop: Arc<AtomicBool>) -> Result<Option<Arc<Stats>>, Error> {
        Ok(self.config(software, device, 1)?.map(|config| spawn(config, stop).1))
    }
}

/// One line for the status output of a miner.
#[must_use]
pub fn status(stats: &Stats) -> String {
    format!(
        "sky search: {} light curves in {} batches, {} refused, {} failed requests",
        stats.curves.load(Ordering::Relaxed),
        stats.batches.load(Ordering::Relaxed),
        stats.refused.load(Ordering::Relaxed),
        stats.errors.load(Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn options_make_a_config() {
        let mut o = Options::default();
        assert!(o.clone().config("s", "d", 3).unwrap().is_none(), "without --sky there is no worker");
        assert!(o.take("--sky", "http://sky.example:8787") && o.take("--sky-address", "neb1abc"));
        assert!(!o.take("--sky-threads", "0") && !o.take("--sky-threads", "x") && !o.take("--other", "1"));
        let c = o.clone().config("neb-sky 1", "x86_64", 3).unwrap().unwrap();
        assert_eq!((c.threads, c.device.as_str(), c.address.as_str()), (3, "x86_64", "neb1abc"));
        assert!(o.take("--sky-threads", "5") && o.take("--sky-device", "card 1"));
        let c = o.config("neb-sky 1", "x86_64", 3).unwrap().unwrap();
        assert_eq!((c.threads, c.device.as_str()), (5, "card 1"));
        let mut bad = Options::default();
        assert!(bad.take("--sky", "https://sky.example"));
        assert!(bad.config("s", "d", 1).is_err());
    }

    #[test]
    fn splits_addresses() {
        assert_eq!(split("http://sky.example:8787").unwrap(), ("sky.example:8787".into(), String::new()));
        assert_eq!(split("http://sky.example/api/").unwrap(), ("sky.example:80".into(), "/api".into()));
        assert!(split("https://sky.example").is_err());
        assert!(split("http://").is_err());
        assert_eq!(encode("neb miner/1.0 \u{b5}"), "neb%20miner%2F1.0%20%C2%B5");
    }

    /// A server of two requests: hands out one batch, then keeps what is posted.
    #[test]
    fn asks_searches_and_posts() {
        let mut batch = b"NEBS\x01\0\0\0\x02\0\0\0".to_vec();
        for n in [40u16, 0] {
            batch.extend_from_slice(&n.to_le_bytes());
            batch.extend((0..n).map(|i| if i % 9 < 2 { 0xB0 } else { 3 }));
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let served = batch.clone();
        let server = thread::spawn(move || {
            let mut posted = Vec::new();
            for (i, stream) in listener.incoming().take(2).enumerate() {
                let mut stream = stream.unwrap();
                let mut seen = Vec::new();
                let mut buf = [0u8; 4096];
                // read the head, then the body its content-length announces
                let (end, length) = loop {
                    let n = stream.read(&mut buf).unwrap();
                    seen.extend_from_slice(&buf[..n]);
                    if let Some(end) = seen.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&seen[..end]).to_string();
                        let length: usize = head
                            .split("Content-Length: ")
                            .nth(1)
                            .unwrap()
                            .split("\r\n")
                            .next()
                            .unwrap()
                            .parse()
                            .unwrap();
                        break (end + 4, length);
                    }
                };
                while seen.len() < end + length {
                    let n = stream.read(&mut buf).unwrap();
                    seen.extend_from_slice(&buf[..n]);
                }
                let head = String::from_utf8_lossy(&seen[..end]).to_string();
                assert!(head.contains("device=card%201"), "{head}");
                let body = if i == 0 { served.clone() } else { b"ok".to_vec() };
                posted = seen.split_off(end);
                stream
                    .write_all(format!("HTTP/1.0 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).as_bytes())
                    .unwrap();
                stream.write_all(&body).unwrap();
            }
            posted
        });
        let config = Config { url, device: "card 1".into(), threads: 2, ..Config::default() };
        let stats = Stats::default();
        assert!(round(&config, &stats).unwrap());
        assert_eq!(server.join().unwrap(), crate::run(&batch).unwrap());
        assert_eq!((stats.batches.load(Ordering::Relaxed), stats.curves.load(Ordering::Relaxed)), (1, 2));
    }
}
