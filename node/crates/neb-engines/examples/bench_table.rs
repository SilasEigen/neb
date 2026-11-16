//! Measures the table engine of nebhash v2 against the C reference on a dataset file.
//!
//!   bench_table DATASETS INDEX [--weights FILE] [--keys N] [--threads N]
//!
//! DATASETS is a network dataset file and INDEX the dataset in it (a v2 dataset). `--weights` is a file with the packed
//! weights of a model in hex (the `weights` field of the `getmodel` RPC); without it the model is the initial state of
//! a zero seed. Prints the time to build the table, and the attempts per second of the reference on one thread and of
//! the table on one thread and on `--threads` threads (default: all). Every ticket of the table that is also
//! evaluated by the reference is compared with it; a difference is an error.

use std::sync::Arc;
use std::time::Instant;

use neb_engines::table::Table;
use neb_pow::shares::{Model, ShareBuffers, load_datasets};

fn key(k: u64) -> [u8; 32] {
    let mut input = b"neb-bench-table".to_vec();
    input.extend_from_slice(&k.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: bench_table DATASETS INDEX [--weights FILE] [--keys N] [--threads N]".into());
    }
    let index: usize = args[1].parse().map_err(|_| "INDEX must be a number")?;
    let (mut weights, mut keys) = (None, 200_000u64);
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        let v = it.next().ok_or(format!("{a} needs a value"))?;
        match a.as_str() {
            "--weights" => weights = Some(v.clone()),
            "--keys" => keys = v.parse().map_err(|_| "--keys must be a number")?,
            "--threads" => threads = v.parse().map_err(|_| "--threads must be a number")?,
            _ => return Err(format!("unknown option {a}")),
        }
    }
    let datasets = load_datasets(std::path::Path::new(&args[0]), None)?;
    let ds = datasets.get(index).ok_or(format!("the file has {} datasets", datasets.len()))?.clone();
    let arch = ds.arch();
    let model = match weights {
        None => Model::from_seed_arch(arch, &[0u8; 32]),
        Some(path) => {
            let hex = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
            let hex = hex.trim().as_bytes();
            let digit = |c: u8| (c as char).to_digit(16).map(|d| d as u8).ok_or("the weights file is not hex");
            let bytes = hex.chunks(2).map(|p| Ok(digit(p[0])? << 4 | digit(*p.get(1).unwrap_or(&b'0'))?));
            Model::from_packed(arch, &bytes.collect::<Result<Vec<u8>, &str>>()?)?
        }
    };
    println!("dataset {index}: {} samples, architecture {}, {} threads", ds.n(), arch.name(), threads);

    let t = Instant::now();
    let table = Arc::new(Table::build(&model, &*ds, threads, Vec::new())?);
    println!(
        "table: {:.2} s to build, {:.0} MB, {} of {} samples with a zero gradient",
        t.elapsed().as_secs_f64(),
        Table::bytes(arch, ds.n()) as f64 / 1e6,
        table.idle_rows(),
        ds.n()
    );

    let ref_keys = keys.min(400);
    let mut buffers = ShareBuffers::new();
    let t = Instant::now();
    let reference: Vec<[u8; 32]> = (0..ref_keys).map(|k| buffers.ticket(&model, &*ds, &key(k))).collect();
    let ref_rate = ref_keys as f64 / t.elapsed().as_secs_f64();

    let mut acc = Vec::new();
    let t = Instant::now();
    for k in 0..keys {
        let ticket = table.eval(&key(k), &mut acc).ticket;
        if k < ref_keys && ticket != reference[k as usize] {
            return Err(format!("the ticket of key {k} differs from the reference"));
        }
    }
    let one = keys as f64 / t.elapsed().as_secs_f64();

    let t = Instant::now();
    std::thread::scope(|scope| {
        for w in 0..threads as u64 {
            let table = table.clone();
            scope.spawn(move || {
                let mut acc = Vec::new();
                let mut sink = 0u8;
                for k in 0..keys {
                    sink ^= table.eval(&key(k + (w + 1) * keys), &mut acc).ticket[0];
                }
                std::hint::black_box(sink);
            });
        }
    });
    let many = (keys * threads as u64) as f64 / t.elapsed().as_secs_f64();
    println!("reference, 1 thread: {ref_rate:.0} attempts/s ({ref_keys} tickets, all equal to the table)");
    println!("table, 1 thread: {one:.0} attempts/s ({:.0} x the reference)", one / ref_rate);
    println!("table, {threads} threads: {many:.0} attempts/s ({:.0} x one reference thread)", many / ref_rate);
    Ok(())
}
