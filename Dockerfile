# Shopping Sessions submission image.
#
# THIS is the file the graders build: they clone the repo and build from its
# root. It compiles the Rust pipeline in `fast/`. Keep it in sync with
# `fast/Dockerfile`, which is the same build scoped to that directory for local
# iteration.
#
# Build time is NOT scored, only the run, so the build stage optimises without
# restraint (see fast/Cargo.toml's release profile: opt-level 3, fat LTO,
# codegen-units 1).
#
# No FROM line pins a platform, so this builds for linux/amd64 (the scorers,
# Intel Xeon) or linux/arm64 (fast native loops on an Apple-silicon Mac).

FROM rust:1-slim-bookworm AS build
WORKDIR /src

# Fetch dependencies against a dummy source tree first, so editing the real
# sources does not re-download the crate graph on every rebuild.
COPY fast/Cargo.toml fast/Cargo.lock ./
RUN mkdir -p src \
 && echo 'fn main() {}' > src/main.rs \
 && echo '' > src/lib.rs \
 && cargo fetch

COPY fast/src/ ./src/
# Touch the real sources so cargo does not reuse the dummy build above.
RUN touch src/main.rs src/lib.rs \
 && cargo build --release --locked --offline --bin pipeline

FROM debian:bookworm-slim
# The scorer runs the container as uid 1000 with a read-only root filesystem,
# no network, /data read-only, and only /output and /scratch writable.
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /src/target/release/pipeline /app/pipeline

ENV RUST_BACKTRACE=1
WORKDIR /app
# Runs with no arguments; the pipeline reads DATA_DIR / OUTPUT_DIR from the env.
# Exit 0 means "done, check my output".
CMD ["/app/pipeline"]
