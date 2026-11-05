//! The dashboard of the miners: a terminal view of the chain, the miner and the hardware.
//!
//! It is drawn on the alternate screen about ten times a second. Each block that the miner found is a cube that fills
//! with confirmations (60 confirmations: final, the reorganisation bound; 100: spendable, the coinbase maturity), with a
//! loading bar next to it. Keys (Windows consoles): `q` quits, `b` replays the flash of a new block, the arrow keys, page
//! keys, `j` and `k` scroll the list of blocks.

use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::term::{self, Key};
use crate::{FoundBlock, Session};

const BG: &str = "\x1b[48;5;233m";
const E: &str = "\x1b[0m\x1b[48;5;233m";
const G: &str = "\x1b[1;38;5;120m";
const W: &str = "\x1b[1;38;5;231m";
const N: &str = "\x1b[38;5;231m";
const M: &str = "\x1b[38;5;250m";
const D: &str = "\x1b[38;5;244m";
const D2: &str = "\x1b[38;5;237m";

/// Width and height that a frame needs; a smaller terminal shows a message instead.
const MIN_WIDTH: usize = 120;
/// Lines of a frame: the header, the panels and the footer.
const FRAME_ROWS: usize = 30;
const CHAIN_ROWS: usize = 5;
const LEFT_WIDTH: usize = 62;
const FINAL_DEPTH: i64 = 60;
const SPENDABLE_DEPTH: i64 = 100;

/// A block cube is 12 x 12 pixels drawn as 6 x 3 braille cells.
const CUBE_EDGE: [u16; 12] = [0x060, 0x198, 0x606, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x606, 0x198, 0x060];
const CUBE_INNER: [u16; 12] = [0x000, 0x000, 0x000, 0xC03, 0x30C, 0x0F0, 0x060, 0x060, 0x060, 0x060, 0x060, 0x060];
const CUBE_TOP: [u16; 12] = [0x000, 0x060, 0x1F8, 0x3FC, 0x4F2, 0x000, 0x000, 0x000, 0x000, 0x000, 0x000, 0x000];
/// The pixels of the fill, in the order in which a cube fills from the bottom: `(x, y)`.
const CUBE_FILL: [(u8, u8); 56] = [
    (4, 9),
    (7, 9),
    (8, 9),
    (3, 9),
    (4, 8),
    (7, 8),
    (8, 8),
    (3, 8),
    (9, 8),
    (2, 8),
    (1, 8),
    (10, 8),
    (7, 7),
    (4, 7),
    (3, 7),
    (8, 7),
    (9, 7),
    (2, 7),
    (10, 7),
    (1, 7),
    (4, 6),
    (7, 6),
    (8, 6),
    (3, 6),
    (9, 6),
    (2, 6),
    (10, 6),
    (1, 6),
    (8, 5),
    (3, 5),
    (9, 5),
    (2, 5),
    (10, 5),
    (1, 5),
    (5, 4),
    (6, 4),
    (7, 4),
    (4, 4),
    (1, 4),
    (10, 4),
    (5, 3),
    (6, 3),
    (7, 3),
    (4, 3),
    (8, 3),
    (3, 3),
    (9, 3),
    (2, 3),
    (6, 2),
    (5, 2),
    (4, 2),
    (7, 2),
    (8, 2),
    (3, 2),
    (5, 1),
    (6, 1),
];
const ICON_CHAIN: &str = "\u{279e}\u{28b2}\u{28b5}\u{2857}";
const ICON_MINER: &str = "\u{28b7}\u{28ad}\u{282d}\u{287e}";
const LOGO: [&str; 5] = [
    "            \u{2588}     ",
    "            \u{2588}     ",
    "\u{2588}\u{2580}\u{2580}\u{2580}\u{2584} \u{2584}\u{2588}\u{2580}\u{2580}\u{2584} \u{2588}\u{2588}\u{2588}\u{2588}\u{2584} ",
    "\u{2588}   \u{2588} \u{2588}\u{2588}\u{2580}\u{2580}\u{2580}\u{2580}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588} ",
    "\u{2580}   \u{2580}  \u{2580}\u{2580}\u{2580}  \u{2580}\u{2580}\u{2580}\u{2580}  ",
];

/// One GPU as the sensors report it.
#[derive(Clone, Debug, Default)]
pub struct GpuReading {
    pub name: String,
    pub temp_c: Option<u32>,
    pub power_w: Option<f64>,
    pub sm_mhz: Option<u32>,
    pub util: Option<u32>,
    /// Fan speed in percent.
    pub fan: Option<u32>,
}

pub enum Hardware {
    Cpu { threads: usize },
    Gpu(Vec<GpuReading>),
}

/// The texts of the header.
pub struct Info {
    /// For example `GPU proof-of-work miner`.
    pub title: String,
    pub device: String,
    pub node: String,
}

/// Why the dashboard returned.
#[derive(PartialEq, Debug)]
pub enum Outcome {
    Quit,
    Disconnected,
}

/// Everything that one frame shows.
struct View<'a> {
    info: &'a Info,
    hardware: &'a Hardware,
    online: bool,
    uptime: f64,
    rate: f64,
    peak: f64,
    attempts: u64,
    shares: u64,
    bits: Option<u32>,
    tip: Option<u32>,
    blocks: &'a [FoundBlock],
    utc_offset: i64,
}

/// Visible width: the characters outside the colour sequences.
fn vw(s: &str) -> usize {
    let mut n = 0;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            n += 1;
        }
    }
    n
}

fn pad_to(s: &str, width: usize) -> String {
    let n = vw(s);
    if n >= width { s.to_string() } else { format!("{s}{D2}{}", " ".repeat(width - n)) }
}

fn commas(v: u64) -> String {
    let digits = v.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn hms(unix: u64, utc_offset: i64) -> String {
    let day = (unix as i64 + utc_offset).rem_euclid(86_400);
    format!("{:02}:{:02}:{:02}", day / 3600, day % 3600 / 60, day % 60)
}

fn rate_text(rate: f64) -> String {
    if rate >= 1e6 { format!("{:.2} MH/s", rate / 1e6) } else { format!("{:.1} kH/s", rate / 1e3) }
}

/// The three text rows of a block's cube. A pending cube fills from the bottom with the confirmations; a gap travels up
/// through the fill and the next pixel blinks. A final or spendable cube is solid, with a glint that crosses it. `t` is
/// the animation clock in seconds, `seed` shifts the motion of each block.
fn cube_rows(depth: i64, t: f64, seed: u32) -> [String; 3] {
    let mut px = [[false; 12]; 12];
    let mature = depth >= SPENDABLE_DEPTH;
    let fin = depth >= FINAL_DEPTH;
    for (y, row) in px.iter_mut().enumerate() {
        for (x, p) in row.iter_mut().enumerate() {
            let edge = (CUBE_EDGE[y] >> x) & 1 == 1;
            let inner = (CUBE_INNER[y] >> x) & 1 == 1;
            let top = (CUBE_TOP[y] >> x) & 1 == 1;
            *p = edge || (!mature && inner) || (fin && !mature && top);
        }
    }
    let n = if fin {
        CUBE_FILL.len()
    } else {
        (depth.max(0) as usize * CUBE_FILL.len() / FINAL_DEPTH as usize).min(CUBE_FILL.len() - 1)
    };
    for &(x, y) in &CUBE_FILL[..n] {
        px[y as usize][x as usize] = true;
    }
    if fin {
        let period = if mature { 5.0 } else { 3.5 };
        let k = (((t + seed as f64 * 0.37) % period / period * 1.6 - 0.3) * 22.0).floor() as i64;
        for (y, row) in px.iter_mut().enumerate() {
            for (x, p) in row.iter_mut().enumerate() {
                let edge = (CUBE_EDGE[y] >> x) & 1 == 1;
                let s = (x + y) as i64;
                if *p && !edge && (s == k || s == k + 1) {
                    *p = false;
                }
            }
        }
    } else {
        if ((t * 4.0) as i64 + seed as i64) % 2 == 0 {
            let (x, y) = CUBE_FILL[n];
            px[y as usize][x as usize] = true;
        }
        if n > 3 {
            let g = ((t * 7.0 + seed as f64 * 5.0) % (n as f64 + 8.0)) as i64;
            for j in 0..2 {
                if g - j >= 0 && g - j < n as i64 {
                    let (x, y) = CUBE_FILL[(g - j) as usize];
                    px[y as usize][x as usize] = false;
                }
            }
        }
    }
    const BIT: [[u32; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
    let mut rows = [String::new(), String::new(), String::new()];
    for (r, row) in rows.iter_mut().enumerate() {
        for c in 0..6 {
            let mut bits = 0x2800u32;
            for dx in 0..2 {
                for dy in 0..4 {
                    if px[4 * r + dy][2 * c + dx] {
                        bits |= BIT[dx][dy];
                    }
                }
            }
            row.push(char::from_u32(bits).expect("a braille pattern"));
        }
    }
    rows
}

/// A loading bar of `n` cells, `on` of them full. A glint sweeps over the full cells (`speed` cells per second, then a
/// short rest) and the next cell pulses dark, medium, bright. `sweep` is false for a bar that stays still.
fn bar_str(on: usize, n: usize, t: f64, speed: f64, seed: f64, sweep: bool) -> String {
    const FULL: &str = "\x1b[1;38;5;120m";
    const GLOW: &str = "\x1b[1;38;5;194m";
    const GLINT: &str = "\x1b[1;38;5;231m";
    const EMPTY: &str = "\x1b[22;38;5;238m";
    const PULSE: [&str; 3] = ["\x1b[22;38;5;28m", "\x1b[22;38;5;71m", "\x1b[1;38;5;120m"];
    const PHASE: [usize; 4] = [0, 1, 2, 1];
    let level = PHASE[(((t / 1.2 + seed * 0.13) % 1.0) * 4.0) as usize % 4];
    let head: i64 = if sweep && on > 0 { ((t * speed + seed * 3.7) % (on as f64 + 10.0)).floor() as i64 } else { -100 };
    let mut out = String::new();
    let mut current = "";
    for i in 0..n {
        let (color, glyph) = if i < on {
            let c = if i as i64 == head {
                GLINT
            } else if (i as i64 - head).abs() == 1 {
                GLOW
            } else {
                FULL
            };
            (c, '\u{25ae}')
        } else if i == on {
            (PULSE[level], if level == 0 { '\u{25af}' } else { '\u{25ae}' })
        } else {
            (EMPTY, '\u{25af}')
        };
        if color != current {
            out.push_str(color);
            current = color;
        }
        out.push(glyph);
    }
    out + E
}

fn confirmations(tip: Option<u32>, height: u32) -> i64 {
    tip.map_or(0, |t| (t as i64 - height as i64 + 1).max(0))
}

fn chain_column(v: &View, t: f64, scroll: &mut usize) -> Vec<String> {
    let width = LEFT_WIDTH;
    let mut lines = vec![format!("{D2}\u{256d}{}\u{256e}", "\u{2500}".repeat(width - 2))];
    let (mut pending, mut fin, mut spendable) = (0, 0, 0);
    for b in v.blocks {
        let depth = confirmations(v.tip, b.height);
        if depth >= SPENDABLE_DEPTH {
            spendable += 1;
        } else if depth >= FINAL_DEPTH {
            fin += 1;
        } else {
            pending += 1;
        }
    }
    lines.push(format!(
        " {G}{ICON_CHAIN} {W}CHAIN{E}{D}  \u{2502} pending {M}{pending}{E}{D}  final {M}{fin}{E}{D}  spendable {M}{spendable}{E}"
    ));
    let total = v.blocks.len();
    let max_scroll = total.saturating_sub(CHAIN_ROWS);
    *scroll = (*scroll).min(max_scroll);
    let older = max_scroll - *scroll;
    let (full, brief) = if *scroll > 0 {
        (format!("\u{2191} {} newer  \u{2193} {older} older", *scroll), format!("\u{2191}{} \u{2193}{older}", *scroll))
    } else if older > 0 {
        (format!("{older} older not shown"), format!("{older} older"))
    } else {
        (String::new(), String::new())
    };
    let mut head = format!(" {D}HEIGHT \u{25b2}    CONFIRMATIONS \u{25bc}    STATUS{E}");
    let indicator = if vw(&head) + vw(&full) + 3 <= width - 2 { full } else { brief };
    if !indicator.is_empty() {
        let gap = (width - 2).saturating_sub(1 + vw(&head) + vw(&indicator));
        head.push_str(&format!("{}{D}{indicator}{E}", " ".repeat(gap)));
    }
    lines.push(head);
    let mut shown = 0;
    for b in v.blocks.iter().rev().skip(*scroll).take(CHAIN_ROWS) {
        shown += 1;
        let depth = confirmations(v.tip, b.height);
        let cube = cube_rows(depth, t, b.height);
        let (status, detail) = if depth >= SPENDABLE_DEPTH {
            (format!("{G}SPENDABLE{E}"), format!("{D}\u{b7}{E}"))
        } else if depth >= FINAL_DEPTH {
            (format!("{G}FINAL{E}"), format!("{D}spendable in {}{E}", SPENDABLE_DEPTH - depth))
        } else {
            (format!("{G}PENDING {depth} / {SPENDABLE_DEPTH}{E}"), format!("{D}final in {}{E}", FINAL_DEPTH - depth))
        };
        let on = if depth >= FINAL_DEPTH { 34 } else { depth as usize * 34 / FINAL_DEPTH as usize };
        let speed = if depth >= FINAL_DEPTH { 6.0 } else { 12.0 };
        let bar = bar_str(on, 34, t, speed, b.height as f64, depth < SPENDABLE_DEPTH);
        lines.push(format!(" {W}{}  {W}#{}{E}            {D}{}{E}", cube[0], b.height, hms(b.found_at, v.utc_offset)));
        lines.push(format!(" {G}{}  {bar}{E} {status}", cube[1]));
        lines.push(format!(" {G}{}   {detail}", cube[2]));
    }
    if shown == 0 {
        lines.push(format!(" {D}waiting for the first block\u{2026}{E}"));
        lines.push(String::new());
        lines.push(String::new());
    }
    while lines.len() < 18 {
        lines.push(String::new());
    }
    lines.push(format!("{D2}\u{2570}{}\u{256f}", "\u{2500}".repeat(width - 2)));
    lines
}

fn miner_column(v: &View, t: f64, flash: Option<u32>, width: usize) -> Vec<String> {
    let rule = format!("{D2}{}{E}", "\u{2500}".repeat(width - 2));
    let mut lines = vec![format!("{D2}\u{256d}{}\u{256e}", "\u{2500}".repeat(width - 2))];
    lines.push(format!(" {G}{ICON_MINER} {W}MINER{E}"));
    let avg = if v.uptime > 0.0 { v.attempts as f64 / v.uptime } else { 0.0 };
    lines.push(format!(
        " {D}HASHRATE{E}   {W}{}{E} attempts/s   {D}AVG{E}  {M}{}{E}",
        commas(v.rate as u64),
        rate_text(avg)
    ));
    let on = if v.peak > 0.0 { ((v.rate / v.peak * 10.0).round() as usize).min(10) } else { 0 };
    lines.push(format!(" {}", bar_str(on, 10, t, 5.0, 0.0, true)));
    lines.push(format!(" {D}ATTEMPTS{E}  {N}{}{E}", commas(v.attempts)));
    let diff = v.bits.map_or("n/a".to_string(), |b| format!("0x{b:08x}"));
    lines.push(format!(" {D}BLOCKS{E}    {N}{}{E}        {D}DIFF{E}  {M}{diff}{E}", v.blocks.len()));
    let tip = v.tip.map_or("n/a".to_string(), |h| format!("#{h}"));
    lines.push(format!(" {D}SHARES{E}    {N}{}{E}        {D}TIP{E}   {M}{tip}{E}", commas(v.shares)));
    lines.push(rule.clone());
    let label = if flash.is_some_and(|k| k % 2 == 1) { W } else { G };
    lines.push(format!(" {label}LAST BLOCK{E}"));
    if let Some(b) = v.blocks.last() {
        let id: String = b.id.iter().map(|x| format!("{x:02x}")).collect();
        lines.push(format!(" {D}NONCE{E}      {N}{}{E}", b.nonce));
        lines.push(format!(" {D}HASH{E}       {W}{}{E}{D}{}\u{2026}{E}", &id[..10], &id[10..28]));
        let (dot, word) = if b.accepted { (format!("{G}\u{25cf} "), "yes") } else { (format!("{D}\u{25cb} "), "no") };
        lines.push(format!(" {D}VERIFIED{E}   {dot}{N}{word}{E}        {D}IN{E} {M}{:.1} s{E}", b.seconds));
    } else {
        lines.push(String::new());
        lines.push(format!(" {D}waiting for the first block\u{2026}{E}"));
        lines.push(String::new());
    }
    lines.push(rule);
    match v.hardware {
        Hardware::Cpu { threads } => {
            lines.push(format!(" {G}CPU{E}"));
            lines.push(format!(" {D}THREADS{E}    {N}{threads}{E}"));
        }
        Hardware::Gpu(gpus) => {
            lines.push(format!(" {G}GPU{E}"));
            let f = |x: Option<String>| x.unwrap_or_else(|| "n/a".to_string());
            let reading = |g: &GpuReading| {
                (
                    f(g.temp_c.map(|c| format!("{c} \u{b0}C"))),
                    f(g.power_w.map(|w| format!("{w:.0} W"))),
                    f(g.sm_mhz.map(|c| format!("{c} MHz"))),
                    f(g.util.map(|u| format!("{u} %"))),
                )
            };
            if let [g] = gpus.as_slice() {
                let (temp, power, clock, util) = reading(g);
                lines.push(format!(" {D}TEMP{E}       {N}{temp}{E}        {D}POWER{E}  {M}{power}{E}"));
                lines.push(format!(" {D}SM CLOCK{E}   {N}{clock}{E}     {D}UTIL{E}  {M}{util}{E}"));
                let eff = g
                    .power_w
                    .filter(|w| *w > 0.0)
                    .map_or("n/a".to_string(), |w| format!("{:.1} kH/J", v.rate / w / 1000.0));
                lines.push(format!(" {D}EFFICIENCY{E} {N}{eff}{E}"));
            } else {
                for (i, g) in gpus.iter().enumerate().take(3) {
                    let (temp, power, clock, util) = reading(g);
                    lines.push(format!(" {D}GPU {i}{E}  {N}{temp}{E}  {M}{power}{E}  {M}{clock}{E}  {M}{util}{E}"));
                }
            }
        }
    }
    while lines.len() < 18 {
        lines.push(String::new());
    }
    lines.push(format!("{D2}\u{2570}{}\u{256f}", "\u{2500}".repeat(width - 2)));
    lines
}

/// One frame of the dashboard as a string of text and colour sequences.
fn render(v: &View, t: f64, flash: Option<u32>, scroll: &mut usize, width: usize, clock: &str) -> String {
    let w = width.clamp(MIN_WIDTH, 140);
    let right_width = w - LEFT_WIDTH - 3;
    let left = chain_column(v, t, scroll);
    let right = miner_column(v, t, flash, right_width);
    let mut out = String::from("\x1b[H");
    out.push_str(BG);
    let subs =
        ["", v.info.title.as_str(), "training steps as proof of work", v.info.node.as_str(), v.info.device.as_str()];
    for (i, logo) in LOGO.iter().enumerate() {
        // the solid letter is the live dot, in the accent colour
        let cut: usize = logo.char_indices().nth(12).map_or(logo.len(), |(b, _)| b);
        let mut line = format!("  {W}{}{G}{}{E}", &logo[..cut], &logo[cut..]);
        let pad = 32usize.saturating_sub(vw(&line));
        line.push_str(&" ".repeat(pad));
        if !subs[i].is_empty() {
            let color = match i {
                1 => W,
                2 => M,
                _ => D,
            };
            line.push_str(&format!("{D2} \u{2502} {E}{color}{}{E}", subs[i]));
        }
        if (1..=3).contains(&i) {
            let badge = match i {
                1 => format!("{D2}\u{256d}{}\u{256e}", "\u{2500}".repeat(16)),
                2 => {
                    let (dot, word) = if v.online {
                        (format!("{G}\u{25cf} "), "MINER ONLINE ")
                    } else {
                        (format!("{D}\u{25cb} "), "MINER OFFLINE")
                    };
                    format!("{D2}\u{2502}{E} {dot}{W}{}{E}{D2}\u{2502}", pad_to(word, 13))
                }
                _ => format!("{D2}\u{2570}{}\u{256f}", "\u{2500}".repeat(16)),
            };
            if let Some(room) = w.checked_sub(20 + vw(&line)).filter(|r| *r > 0) {
                line.push_str(&format!("{}{BG}{badge}", " ".repeat(room)));
            }
        }
        out.push_str(&pad_to(&line, w));
        out.push('\n');
    }
    for i in 0..left.len() {
        let border = i == 0 || i == left.len() - 1;
        let line = if border {
            format!("  {BG}{} {}", left[i], right[i])
        } else {
            format!(
                "  {BG}{D2}\u{2502}{E}{}{D2}\u{2502}{E} {D2}\u{2502}{E}{}{D2}\u{2502}{E}",
                pad_to(&left[i], LEFT_WIDTH - 2),
                pad_to(&right[i], right_width - 2)
            )
        };
        out.push_str(&pad_to(&line, w));
        out.push('\n');
    }
    out.push('\n');
    out.push_str(&format!("{BG}  {D2}\u{256d}{}\u{256e}{E}\n", "\u{2500}".repeat(w - 4)));
    let keys = if term::HAS_KEYS {
        format!("   {G}q{E}{D} quit{E}   {G}b{E}{D} block preview{E}   {G}\u{2191}\u{2193}{E}{D} scroll{E}")
    } else {
        format!("   {G}ctrl-c{E}{D} quit{E}")
    };
    let gap = (w - 4).saturating_sub(vw(&keys) + 9);
    let footer = format!("{keys}{}{G}{clock}{E} ", " ".repeat(gap));
    out.push_str(&format!("{BG}  {D2}\u{2502}{E}{}{D2}\u{2502}{E}\n", pad_to(&footer, w - 4)));
    out.push_str(&format!("{BG}  {D2}\u{2570}{}\u{256f}{E}", "\u{2500}".repeat(w - 4)));
    out.push_str("\x1b[0m");
    out
}

/// True when the output is a terminal that can show the dashboard.
pub fn available() -> bool {
    term::is_tty()
}

/// Leaves the alternate screen when the dashboard ends, however it ends.
struct Screen;

impl Screen {
    fn enter() -> Screen {
        term::setup();
        print!("\x1b[?1049h\x1b[?25l");
        let _ = std::io::stdout().flush();
        Screen
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        print!("{}", term::LEAVE);
        let _ = std::io::stdout().flush();
    }
}

/// Shows the dashboard until the user quits or the node closes the connection. `hardware` is called for every frame.
pub fn run(session: &Session, info: &Info, hardware: &dyn Fn() -> Hardware) -> Outcome {
    let _screen = Screen::enter();
    let start = Instant::now();
    let utc_offset = term::utc_offset_secs();
    let (mut scroll, mut peak) = (0usize, 0.0f64);
    let mut samples: Vec<(Instant, u64)> = Vec::new();
    let (mut flash_since, mut accepted_seen): (Option<Instant>, usize) = (None, 0);
    let mut size = term::size().unwrap_or((MIN_WIDTH, FRAME_ROWS));
    let mut size_read = Instant::now();
    loop {
        let now = Instant::now();
        for key in term::keys() {
            match key {
                Key::Quit => return Outcome::Quit,
                Key::Preview => flash_since = Some(now),
                Key::Scroll(step) => scroll = (scroll as i64 + step).max(0) as usize,
            }
        }
        if session.stopped() {
            return Outcome::Disconnected;
        }
        let stats = session.stats();
        let attempts = stats.attempts.load(std::sync::atomic::Ordering::Relaxed);
        samples.push((now, attempts));
        samples.retain(|(t, _)| now.duration_since(*t) <= Duration::from_secs(2));
        let rate = match samples.first() {
            Some((t0, a0)) if now > *t0 => (attempts - a0) as f64 / now.duration_since(*t0).as_secs_f64(),
            _ => 0.0,
        };
        peak = peak.max(rate);
        let blocks = session.found_blocks();
        let accepted = blocks.iter().filter(|b| b.accepted).count();
        if accepted > accepted_seen && flash_since.is_none() {
            flash_since = Some(now);
        }
        accepted_seen = accepted;
        let flash = flash_since.and_then(|s| {
            let k = (now.duration_since(s).as_millis() / 80) as u32;
            (k < 8).then_some(k)
        });
        if flash.is_none() {
            flash_since = None;
        }
        if now.duration_since(size_read) > Duration::from_millis(500) {
            size = term::size().unwrap_or(size);
            size_read = now;
        }
        let hw = hardware();
        let view = View {
            info,
            hardware: &hw,
            online: true,
            uptime: start.elapsed().as_secs_f64(),
            rate,
            peak,
            attempts,
            shares: stats.shares.load(std::sync::atomic::Ordering::Relaxed),
            bits: session.bits(),
            tip: session.tip_height(),
            blocks: &blocks,
            utc_offset,
        };
        let unix = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let frame = if size.0 < MIN_WIDTH || size.1 < FRAME_ROWS {
            format!(
                "\x1b[2J\x1b[H{M}Enlarge the terminal to at least {MIN_WIDTH} x {FRAME_ROWS} (now {} x {}).\x1b[0m",
                size.0, size.1
            )
        } else {
            render(&view, start.elapsed().as_secs_f64(), flash, &mut scroll, size.0, &hms(unix, utc_offset))
        };
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(frame.as_bytes());
        let _ = out.flush();
        drop(out);
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(height: u32) -> FoundBlock {
        FoundBlock { height, nonce: 12345, id: [0xab; 32], found_at: 1_793_360_000, seconds: 6.5, accepted: true }
    }

    fn view<'a>(info: &'a Info, hardware: &'a Hardware, blocks: &'a [FoundBlock]) -> View<'a> {
        View {
            info,
            hardware,
            online: true,
            uptime: 100.0,
            rate: 750_000.0,
            peak: 760_000.0,
            attempts: 75_000_000,
            shares: 1904,
            bits: Some(0xd580_0000),
            tip: Some(130),
            blocks,
            utc_offset: 3600,
        }
    }

    #[test]
    fn every_line_of_a_frame_fits_the_terminal() {
        let info = Info {
            title: "GPU proof-of-work miner".into(),
            device: "0: a device".into(),
            node: "127.0.0.1:3441".into(),
        };
        let gpu = Hardware::Gpu(vec![GpuReading {
            name: "a device".into(),
            temp_c: Some(61),
            power_w: Some(107.0),
            sm_mhz: Some(1785),
            util: Some(97),
            fan: Some(70),
        }]);
        let blocks: Vec<FoundBlock> = [20, 50, 80, 100, 120, 125].into_iter().map(block).collect();
        for width in [120, 130, 140] {
            let mut scroll = 1;
            let frame = render(&view(&info, &gpu, &blocks), 12.3, Some(3), &mut scroll, width, "17:00:00");
            for line in frame.split('\n') {
                assert!(vw(line) <= width, "{} columns in a width of {width}: {line:?}", vw(line));
            }
            assert!(frame.contains("PENDING") && frame.contains("SPENDABLE") && frame.contains("FINAL"));
            assert!(frame.contains("1,904") && frame.contains("0xd5800000") && frame.contains("MINER ONLINE"));
        }
    }

    #[test]
    fn a_cube_has_three_rows_of_six_cells_and_a_bar_has_its_cells() {
        for depth in [0, 30, 59, 60, 99, 100, 250] {
            let rows = cube_rows(depth, 1.7, 9);
            assert!(
                rows.iter()
                    .all(|r| r.chars().count() == 6 && r.chars().all(|c| ('\u{2800}'..='\u{28ff}').contains(&c)))
            );
        }
        for on in [0, 10, 34] {
            assert_eq!(vw(&bar_str(on, 34, 2.2, 12.0, 5.0, true)), 34);
        }
    }

    #[test]
    fn the_cpu_panel_replaces_the_gpu_panel_and_the_list_scrolls() {
        let info =
            Info { title: "CPU proof-of-work miner".into(), device: "8 threads".into(), node: "127.0.0.1:3441".into() };
        let cpu = Hardware::Cpu { threads: 8 };
        let blocks: Vec<FoundBlock> = (1..=20).map(block).collect();
        let mut scroll = 1000;
        let frame = render(&view(&info, &cpu, &blocks), 0.0, None, &mut scroll, 120, "00:00:00");
        assert_eq!(scroll, 15);
        assert!(frame.contains("THREADS") && !frame.contains("EFFICIENCY") && frame.contains("\u{2191}15"));
    }

    #[test]
    fn numbers_and_times_are_formatted() {
        assert_eq!(
            (commas(0), commas(999), commas(1000), commas(1234567)),
            ("0".into(), "999".into(), "1,000".into(), "1,234,567".into())
        );
        assert_eq!(hms(0, 3600), "01:00:00");
        assert_eq!(hms(86_399, 0), "23:59:59");
    }
}
