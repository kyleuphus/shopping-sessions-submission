//! Entry point. The scorer runs the image's CMD with NO arguments and reads the
//! env vars, so env is the source of truth; argv is accepted too because
//! `scripts/run_fixtures.py` passes DATA_DIR and OUTPUT_DIR as the last two
//! arguments, which makes local runs convenient.
//!
//! Exit code 0 means "done, check my output".

use std::path::PathBuf;

use fast::{pipeline, pipeline2, pipeline3};

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

    let flag = |key: &str| std::env::var_os(key).as_deref() == Some(std::ffi::OsStr::new("1"));
    // v3 is the default (and is what the Dockerfile's PGO run trains).
    // FAST_V1=1 / FAST_V2=1 select the older pipelines for comparison. v3
    // itself falls back to v1 when a width guard fails (pipeline3.rs).
    if flag("FAST_V1") {
        pipeline::run(&data, &output)?;
    } else if flag("FAST_V2") {
        pipeline2::run(&data, &output)?;
    } else {
        pipeline3::run(&data, &output)?;
    }

    eprintln!("done in {:.3}s", started.elapsed().as_secs_f64());
    // Exit straight away rather than returning: the OS reclaims everything, so
    // skip any remaining destructors. Output files are already closed and
    // flushed (see the end of pipeline::run). Errors above still return Err,
    // which exits nonzero.
    std::process::exit(0)
}
