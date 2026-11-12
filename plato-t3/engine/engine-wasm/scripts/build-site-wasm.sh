#!/bin/bash
# Build of the browser engine (engine_wasm.wasm) from the source in plato-t3/engine.
#
# Cargo's crate disambiguator (-C metadata) hashes the absolute path of the checkout, and panic locations embed
# source and registry paths, so the bytes depend on WHERE the source is built. This script always builds a clean
# copy of the engine tree at one fixed physical path, with every path remapped to a constant, so builds with the
# same toolchain (rustc/cargo 1.99.1, target wasm32-unknown-unknown) and the same crates (Cargo.lock; serde_json
# 1.0.154, itoa 1.0.18 from crates.io) do not depend on the checkout's location.
#
#   usage: plato-t3/engine/engine-wasm/scripts/build-site-wasm.sh OUT.wasm
#
# The crates must already be in the local registry (the build is --offline):
#   (cd plato-t3/engine/engine-wasm && cargo fetch --locked --target wasm32-unknown-unknown)
# ALLOW_OTHER_RUSTC=1 builds with whatever rustc is installed (a working engine, different bytes).
set -euo pipefail
OUT=${1:?output path}
ENGINE=$(cd "$(dirname "$0")/../.." && pwd -P)              # plato-t3/engine
ROOT=/private/tmp/polytopia-bot-wasm-build          # fixed physical path (no symlinks: /tmp is /private/tmp on macOS)
REG=$(cd "${CARGO_HOME:-$HOME/.cargo}" && pwd -P)/registry/src
[[ -e $ROOT ]] && { echo "$ROOT exists; remove it first (it must be a fresh extraction)"; exit 1; }
mkdir -p $ROOT
tar -C "$ENGINE" --exclude target -c engine-wasm engine-rs src/polytopia_bot/gpu/rules.json | tar -x -C $ROOT
cd $ROOT/engine-wasm
V=$("$HOME/.cargo/bin/rustc" -V)
[[ $V == "rustc 1.99.1 (58cf8f9c3 2026-10-13)" || -n ${ALLOW_OTHER_RUSTC:-} ]] || { echo "needs rustc 1.99.1 (58cf8f9c3), found $V"; exit 1; }
env -i PATH="$HOME/.cargo/bin:/usr/bin:/bin" HOME="$HOME" CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" \
  SOURCE_DATE_EPOCH=0 CARGO_TARGET_DIR=$ROOT/target CARGO_INCREMENTAL=0 \
  RUSTFLAGS="-C target-feature=+simd128 --remap-path-prefix=$ROOT=/polytopia-bot --remap-path-prefix=$REG=/cargo-registry" \
  cargo build --release --offline --locked -j1 --target wasm32-unknown-unknown
cp $ROOT/target/wasm32-unknown-unknown/release/engine_wasm.wasm "$OUT"
shasum -a 256 "$OUT" 2>/dev/null || sha256sum "$OUT"
