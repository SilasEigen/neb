//! Console output of the daemon: event lines, and one status line redrawn in
//! place every 200 ms when stdout is a terminal. Output only; no node state
//! depends on it.
//!
//! Colours are ANSI 256-colour escapes: 120 (green) for accents, 244 and 237
//! (greys) for labels, 231 (white) for values. Without a terminal, or with
//! `--color never`, no escape sequence is written.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[1;38;5;120m";
const WHITE: &str = "\x1b[1;38;5;231m";
const DIM: &str = "\x1b[38;5;244m";
const DIM2: &str = "\x1b[38;5;237m";

/// Counters shown by the status line, updated by the session threads.
pub struct Live {
    started: Instant,
    miners: AtomicI32,
    last_block: Mutex<Option<Instant>>,
}

impl Live {
    pub fn new() -> Arc<Live> {
        Arc::new(Live { started: Instant::now(), miners: AtomicI32::new(0), last_block: Mutex::new(None) })
    }

    /// Records a connected block and returns the seconds since the previous
    /// one, or since the start of the process for the first.
    pub fn block_seen(&self) -> f32 {
        let now = Instant::now();
        let mut guard = self.last_block.lock().unwrap_or_else(|e| e.into_inner());
        let since = guard.unwrap_or(self.started);
        *guard = Some(now);
        now.duration_since(since).as_secs_f32()
    }

    pub fn miner_joined(&self) {
        self.miners.fetch_add(1, Ordering::Relaxed);
    }

    pub fn miner_left(&self) {
        self.miners.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy)]
pub enum ColorMode {
    Auto,
    Always,
    Never,
}

/// Serialised console writer. `lock` makes event lines and status redraws
/// atomic with respect to each other.
#[derive(Clone)]
pub struct Log {
    color: bool,
    status: bool,
    live: Arc<Live>,
    lock: Arc<Mutex<()>>,
}

impl Log {
    pub fn new(live: Arc<Live>, mode: ColorMode) -> Log {
        let tty = std::io::stdout().is_terminal();
        let color = match mode {
            ColorMode::Always => true,
            ColorMode::Never => false,
            ColorMode::Auto => tty,
        };
        Log { color, status: tty, live, lock: Arc::new(Mutex::new(())) }
    }

    fn c<'a>(&self, code: &'a str) -> &'a str {
        if self.color { code } else { "" }
    }

    fn stamp(&self) -> String {
        let secs = self.live.started.elapsed().as_secs();
        format!("{}{:>4}s{}", self.c(DIM2), secs, self.c(RESET))
    }

    fn line(&self, body: &str) {
        let _guard = self.lock.lock().expect("log lock");
        let mut out = std::io::stdout();
        if self.status {
            let _ = write!(out, "\r\x1b[K");
        }
        let _ = writeln!(out, "{}", body);
        let _ = out.flush();
    }

    pub fn startup(&self, net: &crate::params::NetworkParams, genesis: &str, height: u32, datadir: &str, port: u16) {
        self.line(&format!(
            "{}neb{} {}{}{}  genesis {}{}{}  height {}{}{}",
            self.c(GREEN),
            self.c(RESET),
            self.c(DIM),
            net.name,
            self.c(RESET),
            self.c(DIM),
            genesis,
            self.c(RESET),
            self.c(WHITE),
            height,
            self.c(RESET)
        ));
        self.line(&format!(
            "{}    {}data{} {}  {}work service{} 127.0.0.1:{}",
            "",
            self.c(DIM),
            self.c(RESET),
            datadir,
            self.c(DIM),
            self.c(RESET),
            port
        ));
    }

    pub fn miner_joined(&self, peer: &str) {
        let (g, r, d) = (self.c(GREEN), self.c(RESET), self.c(DIM));
        self.line(&format!("{}{}\u{25cf}{} {}miner connected{} {}{}{}", self.stamp(), g, r, d, r, d, peer, r));
    }

    pub fn miner_named(&self, name: &str, version: u16) {
        let (d, d2, r) = (self.c(DIM), self.c(DIM2), self.c(RESET));
        self.line(&format!("{}{}\u{00b7}{} {}{}{} {}v{}{}", self.stamp(), d2, r, d, name, r, d2, version, r));
    }

    pub fn miner_left(&self, peer: &str) {
        let (d, d2, r) = (self.c(DIM), self.c(DIM2), self.c(RESET));
        self.line(&format!("{}{}\u{25cb}{} {}miner disconnected{} {}{}{}", self.stamp(), d, r, d2, r, d2, peer, r));
    }

    pub fn block(&self, height: u32, hash: &str, seconds: f32) {
        let (g, r, d, d2, w) = (self.c(GREEN), self.c(RESET), self.c(DIM), self.c(DIM2), self.c(WHITE));
        self.line(&format!(
            "{}{}\u{25c6}{} {}block{} {}{}{}  {}{}\u{2026}{}  {}{:.1} s{}",
            self.stamp(),
            g,
            r,
            d,
            r,
            w,
            height,
            r,
            d2,
            &hash[..hash.len().min(12)],
            r,
            w,
            seconds,
            r
        ));
    }

    pub fn note(&self, text: &str) {
        self.line(&format!("{}    {}{}{}", "", self.c(DIM2), text, self.c(RESET)));
    }

    pub fn rejected(&self, code: u8) {
        self.line(&format!("{} {}submission rejected{} code {}", self.stamp(), self.c(DIM), self.c(RESET), code));
    }

    pub fn frame_error(&self, error: &str) {
        self.line(&format!("{} {}frame error{} {}", self.stamp(), self.c(DIM), self.c(RESET), error));
    }

    /// Starts the status line thread (terminal only): spinner, uptime, height
    /// of the best chain as `height` reads it (blocks from peers included),
    /// connected miners and time since the last block seen by the work service.
    pub fn spawn_status(self: &Arc<Self>, height: impl Fn() -> u32 + Send + 'static) {
        if !self.status {
            return;
        }
        let me = self.clone();
        std::thread::spawn(move || {
            let frames = [
                '\u{280b}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283c}', '\u{2834}', '\u{2826}', '\u{2827}',
                '\u{2807}', '\u{280f}',
            ];
            let mut i = 0usize;
            loop {
                std::thread::sleep(Duration::from_millis(200));
                i = (i + 1) % frames.len();
                let up = me.live.started.elapsed().as_secs();
                let height = height();
                let miners = me.live.miners.load(Ordering::Relaxed);
                let ago = me.live.last_block.lock().ok().and_then(|g| *g).map(|t| t.elapsed().as_secs_f32());
                let ago_text = match ago {
                    Some(secs) => format!("{:.0} s", secs),
                    None => "none yet".to_string(),
                };
                let body = format!(
                    "{}{} {}{}neb{} {}\u{b7} up {}m {:02}s \u{b7} height {}{} \u{b7} miners {}{} \u{b7} last block {}{}{}",
                    me.c(GREEN),
                    frames[i],
                    me.c(RESET),
                    me.c(WHITE),
                    me.c(RESET),
                    me.c(DIM),
                    up / 60,
                    up % 60,
                    height,
                    me.c(RESET),
                    miners,
                    me.c(RESET),
                    me.c(DIM),
                    ago_text,
                    me.c(RESET)
                );
                let _guard = me.lock.lock().expect("log lock");
                let mut out = std::io::stdout();
                let _ = write!(out, "\r\x1b[K{}", body);
                let _ = out.flush();
            }
        });
    }
}
