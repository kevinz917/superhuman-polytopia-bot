# Dependencies only, for every Daytona sandbox (CPU actors, GPU actors with the in-node inference server, learner):
# the stack of experiments/modal_distributed_rl.py, i.e. JAX 0.11.2 with the CUDA 12 plugin (JAX_PLATFORMS picks the
# platform per process) and the Rust toolchain for the engine-rs wheel. The code is not baked in: `launch.py push`
# sends a git archive of src/, engine-rs/ and scripts/ and builds the wheel in the sandbox (the CLI's build-context
# upload dropped directories, 2026-10-26), so a code change needs no rebuild.
#   python experiments/daytona/launch.py snapshot
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential pkg-config procps openssh-client \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/venv/bin:/opt/cargo/bin:$PATH
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
RUN python -m venv /opt/venv && pip install --no-cache-dir 'jax[cuda12]==0.11.2' 'numpy>=2.1' 'zstandard==0.23.0' \
    'pytest>=9,<10' 'maturin>=1.7,<2'
RUN mkdir -p /workspace && chmod -R a+rwX /workspace /opt/venv /opt/cargo /opt/rustup
ENV PYTHONPATH=/workspace/src XLA_PYTHON_CLIENT_PREALLOCATE=false
WORKDIR /workspace
