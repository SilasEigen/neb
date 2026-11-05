//! The stats file of the GPU miner, in the form that the stats script of the HiveOS package reads: one JSON object that is
//! rewritten every two seconds, with the rate of each device in kH/s, the temperatures, the fans, the PCI bus numbers, the
//! uptime, the version and the blocks that the node accepted and refused.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use neb_cpuminer::Session;

use crate::telemetry::Readings;

/// What one file shows.
pub struct Snapshot<'a> {
    pub khs: &'a [f64],
    pub temp: &'a [u32],
    pub fan: &'a [u32],
    pub bus_numbers: &'a [u32],
    pub uptime: u64,
    pub accepted: u64,
    pub rejected: u64,
}

fn array<T: std::fmt::Display>(values: &[T]) -> String {
    let items: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    format!("[{}]", items.join(","))
}

/// The JSON text of a snapshot. `hs` is in kH/s as `hs_units` says, and `khs` is the total.
pub fn render(s: &Snapshot) -> String {
    let hs: Vec<String> = s.khs.iter().map(|v| format!("{v:.2}")).collect();
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\"khs\":{:.2},\"hs\":[{}],\"hs_units\":\"khs\",\"temp\":{},\"fan\":{},\"uptime\":{},\"ver\":\"{}\",\"ar\":[{},{}],\"algo\":\"neb\",\"bus_numbers\":{}}}",
        s.khs.iter().sum::<f64>(),
        hs.join(","),
        array(s.temp),
        array(s.fan),
        s.uptime,
        env!("CARGO_PKG_VERSION"),
        s.accepted,
        s.rejected,
        array(s.bus_numbers),
    );
    out
}

/// Rewrites `path` every two seconds until the session ends. `counters` hold the attempts of each device.
pub fn spawn(
    session: Session,
    path: PathBuf,
    counters: Vec<Arc<AtomicU64>>,
    bus_numbers: Vec<u32>,
    readings: Readings,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let start = Instant::now();
        let (mut last, mut last_counts) = (start, vec![0u64; counters.len()]);
        while !session.stopped() {
            thread::sleep(Duration::from_secs(2));
            let now = Instant::now();
            let seconds = now.duration_since(last).as_secs_f64();
            let counts: Vec<u64> = counters.iter().map(|c| c.load(Ordering::Relaxed)).collect();
            let khs: Vec<f64> =
                counts.iter().zip(&last_counts).map(|(c, l)| (c - l) as f64 / seconds / 1000.0).collect();
            (last, last_counts) = (now, counts);
            let sensors = readings.lock().expect("readings").clone();
            let temp: Vec<u32> =
                (0..counters.len()).map(|i| sensors.get(i).and_then(|r| r.temp_c).unwrap_or(0)).collect();
            let fan: Vec<u32> = (0..counters.len()).map(|i| sensors.get(i).and_then(|r| r.fan).unwrap_or(0)).collect();
            let stats = session.stats();
            let text = render(&Snapshot {
                khs: &khs,
                temp: &temp,
                fan: &fan,
                bus_numbers: &bus_numbers,
                uptime: start.elapsed().as_secs(),
                accepted: stats.accepted.load(Ordering::Relaxed),
                rejected: stats.rejected.load(Ordering::Relaxed),
            });
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, text).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_has_the_fields_that_hiveos_reads() {
        let text = render(&Snapshot {
            khs: &[750.5, 745.25],
            temp: &[65, 61],
            fan: &[70, 0],
            bus_numbers: &[1, 10],
            uptime: 120,
            accepted: 7,
            rejected: 1,
        });
        for part in [
            "\"khs\":1495.75",
            "\"hs\":[750.50,745.25]",
            "\"hs_units\":\"khs\"",
            "\"temp\":[65,61]",
            "\"fan\":[70,0]",
            "\"uptime\":120",
            "\"ar\":[7,1]",
            "\"bus_numbers\":[1,10]",
        ] {
            assert!(text.contains(part), "{part} in {text}");
        }
        assert!(text.starts_with('{') && text.ends_with('}'));
    }
}
