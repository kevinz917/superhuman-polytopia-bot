#!/usr/bin/env bash
# Provision a fresh Verda Ubuntu instance: firewall, Python 3.12 venv, JAX (CUDA on the learner), engine-rs wheel
# on actors, and the node's systemd service (installed, not started).
# Usage: setup_node.sh learner [ALLOWED_IP ...]   (IPs allowed to reach the learner port 7070)
#        setup_node.sh actor LEARNER_IP INDEX [GAMES] [PROCESSES]
# Actors install pilot/wheels/*.whl when the bundle carries one (built once, on the first actor), else build it.
set -euo pipefail
role=$1; shift
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -yq build-essential pkg-config curl ufw tmux >/dev/null

# Firewall first: SSH (key-only) from anywhere, the learner port only from the actor nodes.
ufw --force reset >/dev/null
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp >/dev/null
if [ "$role" = learner ]; then
  for ip in "$@"; do ufw allow from "$ip" to any port 7070 proto tcp >/dev/null; done
fi
ufw --force enable >/dev/null
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && (sshd -t && systemctl reload ssh || true)

curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=/root/.local/bin:$PATH
uv venv -q -p 3.12 /opt/venv
env='PATH=/opt/venv/bin:/usr/bin:/bin PYTHONPATH=/workspace/src XLA_PYTHON_CLIENT_PREALLOCATE=false'
if [ "$role" = learner ]; then
  uv pip install -q -p /opt/venv/bin/python 'jax[cuda12]==0.11.2' 'numpy>=2.1' 'zstandard==0.23.0'
  cat > /etc/systemd/system/polytopia-learner.service <<EOF
[Unit]
Description=polytopia learner
After=network-online.target
[Service]
WorkingDirectory=/workspace/pilot
Environment=$env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=.97 POLYTOPIA_TRUNK_DTYPE=bfloat16
ExecStart=/opt/venv/bin/python node_learner.py run
Restart=on-failure
RestartSec=30
StandardOutput=append:/workspace/learner.log
StandardError=append:/workspace/learner.log
EOF
else
  learner_ip=$1; index=$2; games=${3:-24}; processes=${4:-0}
  uv pip install -q -p /opt/venv/bin/python 'jax==0.11.2' 'numpy>=2.1' 'zstandard==0.23.0'
  if ! ls /workspace/pilot/wheels/*.whl >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
    uv pip install -q -p /opt/venv/bin/python 'maturin>=1.7,<2'
    (cd /workspace/engine-rs && PATH=/root/.cargo/bin:/opt/venv/bin:$PATH maturin build --release -q -o /workspace/pilot/wheels)
  fi
  uv pip install -q -p /opt/venv/bin/python /workspace/pilot/wheels/*.whl
  cat > /etc/systemd/system/polytopia-actor.service <<EOF
[Unit]
Description=polytopia actor
After=network-online.target
[Service]
WorkingDirectory=/workspace/pilot
Environment=$env JAX_PLATFORMS=cpu
ExecStart=/opt/venv/bin/python node_actor.py $learner_ip $index $games $processes
Restart=always
RestartSec=15
StandardOutput=append:/workspace/actor.log
StandardError=append:/workspace/actor.log
EOF
fi
systemctl daemon-reload
echo "setup_ok role=$role cpus=$(nproc) $(grep -m1 'model name' /proc/cpuinfo)"
