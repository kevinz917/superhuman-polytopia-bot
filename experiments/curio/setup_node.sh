#!/usr/bin/env bash
# Provision one Curio box for the polytopia RL stack, as the login user, nothing system-wide (no sudo, no services):
#   setup_node.sh head   ROOT      # uv + Python 3.12 venv, jax[cuda12], rustup (user), maturin, engine-rs wheel
#   setup_node.sh worker ROOT      # uv + Python 3.12 venv, CPU jax, the wheel copied into ROOT/wheels by `publish`
# Idempotent: re-running only installs what is missing. The code tree is expected at ROOT/code/current (curio.py
# publish puts it there); the wheel is built from ROOT/code/current/engine-rs into ROOT/wheels/<engine tree sha>/.
# Pins follow experiments/modal_distributed_rl.py: jax 0.11.2, numpy>=2.1, zstandard 0.23.0, maturin>=1.7,<2.
set -euo pipefail
role=$1; root=${2:?root}
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
mkdir -p "$root"/{ckpt,runs,logs,pids,jax-cache,ladder,arena,wheels,code}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
if [ ! -x "$root/venv/bin/python" ]; then
  uv venv -q -p 3.12 "$root/venv"
fi
py="$root/venv/bin/python"
if [ "$role" = head ]; then
  uv pip install -q -p "$py" 'jax[cuda12]==0.11.2' 'numpy>=2.1' 'zstandard==0.23.0' 'maturin>=1.7,<2' 'pytest>=9,<10'
  if ! command -v cargo >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
  fi
else
  uv pip install -q -p "$py" 'jax==0.11.2' 'numpy>=2.1' 'zstandard==0.23.0' 'pytest>=9,<10'
fi

# Engine wheel: build on the head (once per engine-rs tree), install wherever a wheel for the current tree exists.
if [ -d "$root/code/current/engine-rs" ]; then
  tree=$(cat "$root/code/current/ENGINE_TREE" 2>/dev/null || echo unknown)
  if [ "$role" = head ] && ! ls "$root/wheels/$tree"/*.whl >/dev/null 2>&1; then
    (cd "$root/code/current/engine-rs" && CARGO_TARGET_DIR="$root/cargo-target" \
      CARGO_BUILD_JOBS=32 "$root/venv/bin/maturin" build --release -q -o "$root/wheels/$tree")
  fi
  if ls "$root/wheels/$tree"/*.whl >/dev/null 2>&1; then
    uv pip install -q -p "$py" --force-reinstall --no-deps "$root/wheels/$tree"/*.whl
  else
    echo "no wheel for engine tree $tree in $root/wheels (publish builds it on the head)" >&2
  fi
fi

echo "setup_ok role=$role host=$(hostname) python=$($py -c 'import sys;print(sys.version.split()[0])') \
jax=$($py -c 'import jax;print(jax.__version__)' 2>/dev/null || echo missing) \
engine=$($py -c 'import polytopia_engine_rs;print("ok")' 2>/dev/null || echo missing) cpus=$(nproc)"
