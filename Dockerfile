# Shopping Sessions submission image.
#
# THIS is the file the graders build: they clone the repo and build from its
# root. It compiles the Rust pipeline in `fast/`. Keep it in sync with
# `fast/Dockerfile`, which is the same build scoped to that directory for local
# iteration (it differs only in where the generator source comes from).
#
# Build time is NOT scored, only the run, so the build stage optimises without
# restraint (see fast/Cargo.toml's release profile: opt-level 3, fat LTO,
# codegen-units 1) and adds profile-guided optimisation (PGO) on top.
#
# PGO IS BEST-EFFORT. A failed build is a failed submission, so no PGO step can
# fail this build: if anything in it goes wrong (llvm-profdata missing, the
# generator does not build, the instrumented binary crashes, the profile is
# empty, the profile-use build fails) we build the plain binary exactly as
# before. The build log always says which path was taken, on a line starting
# "PGO: used" or "PGO: fallback because ...". (Use --progress=plain to see it.)
#
# CPU flags: on x86_64 the final binary must carry x86-64-v3 + AES. Those
# normally come from fast/.cargo/config.toml, but a RUSTFLAGS env var REPLACES
# that list entirely, so every RUSTFLAGS below that builds the shipped binary
# repeats them explicitly. On aarch64 there are no special flags.
#
# No FROM line pins a platform, so this builds for linux/amd64 (the scorers,
# Intel Xeon) or linux/arm64 (fast native loops on an Apple-silicon Mac).

FROM rust:1-slim-bookworm AS build

# llvm-profdata, matching this rustc's LLVM. Optional: without it we fall back.
RUN rustup component add llvm-tools-preview \
 || echo "PGO: warning: could not install llvm-tools-preview"

# PGO training data: a 2M-event dataset from the official generator, with a
# seed that is neither the dev set's (1) nor anything the scorer uses. The
# generator has no dependencies, so it builds offline. It is built from /gen,
# outside /src, so fast/.cargo/config.toml's CPU flags do not apply to it.
# Failure here only removes the data, which makes the PGO step fall back.
COPY generator/Cargo.toml generator/Cargo.lock /gen/
COPY generator/src/ /gen/src/
RUN cd /gen \
 && cargo build --release --locked --offline \
 && ./target/release/generate --seed 4242 --events 2000000 --clicks 500000 --output /pgo-data \
 || { echo "PGO: warning: generator failed; no training data"; rm -rf /pgo-data; }

WORKDIR /src

# Fetch dependencies against a dummy source tree first, so editing the real
# sources does not re-download the crate graph on every rebuild.
COPY fast/Cargo.toml fast/Cargo.lock ./
RUN mkdir -p src \
 && echo 'fn main() {}' > src/main.rs \
 && echo '' > src/lib.rs \
 && cargo fetch

COPY fast/src/ ./src/
COPY fast/.cargo/ ./.cargo/

# The build. Steps, each of which may fail into the plain build:
#   1. instrumented build (-Cprofile-generate) into a separate target dir;
#   2. run it on /pgo-data (profiles from build scripts are discarded first);
#      on x86_64, if that fails (e.g. an emulator without AVX2 SIGILLs), retry
#      1-2 once without the CPU flags. Profiles match functions by CFG hash, so
#      a mismatch only costs those functions their profile (warnings, not
#      errors; -pgo-warn-missing-function is deliberately NOT set);
#   3. merge the .profraw files with llvm-profdata;
#   4. build with -Cprofile-use plus the CPU flags;
#   5. if the training ran with the same flags, smoke-run the PGO binary on the
#      training data and require exit 0.
# The plain build is the pre-PGO build: no RUSTFLAGS, so .cargo/config.toml
# supplies the CPU flags.
#
# Every cargo build passes an explicit --target (the host triple). That keeps
# RUSTFLAGS and the config.toml CPU flags off build scripts and proc-macros,
# which cargo otherwise compiles with them too: built for x86-64-v3, they crash
# (SIGSEGV/SIGILL) in a build host without AVX2, such as QEMU amd64 emulation
# on an Apple-silicon Mac, failing the whole build. The pipeline binary itself
# is compiled exactly as before; it lands in target/<triple>/release/ and is
# copied to /out/pipeline.
RUN touch src/main.rs src/lib.rs; \
    if [ "$(uname -m)" = "x86_64" ]; then \
      CPU_FLAGS="-C target-cpu=x86-64-v3 -C target-feature=+aes,+pclmulqdq"; \
    else \
      CPU_FLAGS=""; \
    fi; \
    TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"; \
    BUILD="cargo build --release --locked --offline --target $TRIPLE --bin pipeline"; \
    PROFDATA="$(find "$(rustc --print sysroot)" -name llvm-profdata -type f 2>/dev/null | head -n 1)"; \
    PGO=""; WHY=""; TRAIN_FLAGS=""; \
    train() { \
      rm -rf /tmp/pgo-prof /tmp/pgo-out \
      && RUSTFLAGS="$1 -C profile-generate=/tmp/pgo-prof" \
           $BUILD --target-dir /tmp/pgo-target \
      && rm -rf /tmp/pgo-prof && mkdir -p /tmp/pgo-prof /tmp/pgo-out \
      && DATA_DIR=/pgo-data OUTPUT_DIR=/tmp/pgo-out /tmp/pgo-target/$TRIPLE/release/pipeline \
      && [ -n "$(find /tmp/pgo-prof -name '*.profraw' -size +0 2>/dev/null)" ]; \
    }; \
    if [ -z "$PROFDATA" ]; then WHY="llvm-profdata not found"; \
    elif [ ! -f /pgo-data/manifest.json ]; then WHY="no training data"; \
    elif train "$CPU_FLAGS"; then TRAIN_FLAGS="same"; \
    elif [ -n "$CPU_FLAGS" ] && echo "PGO: training with CPU flags failed; retrying without them" \
         && train ""; then TRAIN_FLAGS="portable"; \
    else WHY="instrumented build or training run failed"; fi; \
    if [ -z "$WHY" ]; then \
      if "$PROFDATA" merge -o /tmp/pgo.profdata /tmp/pgo-prof && [ -s /tmp/pgo.profdata ]; then \
        if RUSTFLAGS="$CPU_FLAGS -C profile-use=/tmp/pgo.profdata" \
             $BUILD; then \
          if [ "$TRAIN_FLAGS" = "same" ]; then \
            rm -rf /tmp/pgo-out && mkdir -p /tmp/pgo-out \
            && DATA_DIR=/pgo-data OUTPUT_DIR=/tmp/pgo-out ./target/$TRIPLE/release/pipeline \
            && PGO="used (trained with the shipped CPU flags; smoke run passed)" \
            || WHY="PGO binary failed its smoke run"; \
          else \
            PGO="used (profile from a build without CPU flags; smoke run skipped)"; \
          fi; \
        else WHY="profile-use build failed"; fi; \
      else WHY="llvm-profdata merge failed or empty profile"; fi; \
    fi; \
    if [ -n "$PGO" ]; then \
      echo "PGO: $PGO"; \
    else \
      echo "PGO: fallback because $WHY"; \
      rm -f target/$TRIPLE/release/pipeline; \
      env -u RUSTFLAGS $BUILD || exit 1; \
      echo "PGO: fallback build done (plain build, no PGO)"; \
    fi; \
    mkdir -p /out && cp target/$TRIPLE/release/pipeline /out/pipeline || exit 1; \
    rm -rf /tmp/pgo-target /tmp/pgo-prof /tmp/pgo-out /tmp/pgo.profdata /pgo-data

FROM debian:bookworm-slim
# The scorer runs the container as uid 1000 with a read-only root filesystem,
# no network, /data read-only, and only /output and /scratch writable.
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /out/pipeline /app/pipeline

ENV RUST_BACKTRACE=1
WORKDIR /app
# Runs with no arguments; the pipeline reads DATA_DIR / OUTPUT_DIR from the env.
# Exit 0 means "done, check my output".
CMD ["/app/pipeline"]
