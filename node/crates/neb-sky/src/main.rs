//! `neb-sky --sky http://HOST:PORT [--sky-threads N] [--sky-address ADDRESS] [--sky-device TEXT] [--once]`
//!
//! The sky search alone (spec/sky-search.md): it asks the batch server for a batch, searches it and posts the result,
//! next to any miner. It does not mine. `--sky-threads` is the number of search threads (a quarter of the processor
//! threads, at least one, when absent: the search should not take the machine from the miner). `--sky-address` gives the server a NEB address, `--sky-device` names the cards
//! or the processor. With `--once` it stops when the server has no batch left; without it, it asks again every
//! minute, and it keeps going while the server is away.
//! Exit code: 0 after `--once` has emptied the server, 1 when a request failed in that mode, 2 on a usage error.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use neb_sky::client::{Options, Stats, round, spawn, status};

fn usage() -> i32 {
    eprintln!(
        "usage: neb-sky --sky http://HOST:PORT [--sky-threads N] [--sky-address ADDRESS] [--sky-device TEXT] [--once]"
    );
    2
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&args));
}

fn run(args: &[String]) -> i32 {
    let mut options = Options::default();
    let mut once = false;
    let mut i = 0;
    while i < args.len() {
        match (args[i].as_str(), args.get(i + 1)) {
            ("--once", _) => {
                once = true;
                i += 1;
                continue;
            }
            (flag, Some(value)) if options.take(flag, value) => {}
            _ => return usage(),
        }
        i += 2;
    }
    let software = concat!("neb-sky ", env!("CARGO_PKG_VERSION"));
    let quarter = std::thread::available_parallelism().map_or(1, |n| n.get() / 4);
    let config = match options.config(software, std::env::consts::ARCH, quarter) {
        Ok(Some(config)) => config,
        Ok(None) => return usage(),
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    println!("{software}: {} search thread(s), server {}", config.threads, config.url);
    if once {
        let stats = Stats::default();
        let outcome = loop {
            match round(&config, &stats) {
                Ok(true) => {}
                Ok(false) => break 0,
                Err(e) => {
                    eprintln!("{e}");
                    stats.errors.fetch_add(1, Ordering::Relaxed);
                    break 1;
                }
            }
        };
        println!("{}", status(&stats));
        return outcome;
    }
    let (_worker, stats) = spawn(config, Arc::new(AtomicBool::new(false)));
    let mut last = String::new();
    loop {
        std::thread::sleep(Duration::from_secs(10));
        let now = status(&stats);
        if now != last {
            println!("{now}");
            last = now;
        }
    }
}
