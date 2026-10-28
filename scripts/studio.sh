#!/bin/sh
set -eu
STUDIO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$STUDIO_ROOT"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.14 .venv
  uv pip install -e .
fi
# The Rust engine plays every game; building it needs a Rust toolchain (https://rustup.rs).
if ! .venv/bin/python -c 'import polytopia_engine_rs' 2>/dev/null; then
  uv pip install 'maturin>=1.7,<2'
  (cd engine-rs && VIRTUAL_ENV="$STUDIO_ROOT/.venv" "$STUDIO_ROOT/.venv/bin/maturin" develop --release)
fi
exec .venv/bin/python -m polytopia_bot.studio.server "$@"
