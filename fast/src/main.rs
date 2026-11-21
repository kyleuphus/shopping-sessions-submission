//! Entry point. The scorer runs the image's CMD with NO arguments and reads the
//! env vars, so env is the source of truth; argv is accepted too because
//! `scripts/run_fixtures.py` passes DATA_DIR and OUTPUT_DIR as the last two
//! arguments, which makes local runs convenient.
//!
//! Exit code 0 means "done, check my output".

use std::path::PathBuf;

use fast::pipeline;

// Use mimalloc for pipeline allocations, including those on Rayon workers.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn dir_from(env_key: &str, argv_index: usize, default: &str) -> PathBuf {
    if let Ok(value) = std::env::var(env_key) {
        return PathBuf::from(value);
    }
    if let Some(value) = std::env::args().nth(argv_index) {
        return PathBuf::from(value);
    }
    PathBuf::from(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let started = std::time::Instant::now();

    let data = dir_from("DATA_DIR", 1, "/data");
    let output = dir_from("OUTPUT_DIR", 2, "/output");

    pipeline::run(&data, &output)?;

    eprintln!("done in {:.3}s", started.elapsed().as_secs_f64());
    Ok(())
}
