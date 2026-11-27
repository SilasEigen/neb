//! Writes the result file of a batch file: `cargo run --release -p neb-sky --example run -- BATCH RESULT`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, batch, result] = args.as_slice() else {
        return Err("usage: run BATCH RESULT".into());
    };
    std::fs::write(result, neb_sky::run(&std::fs::read(batch)?)?)?;
    Ok(())
}
