"""Build credential-free local coder and trusted referee images from explicit files."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from polytopia_bot.skirmish.docker_sandbox import docker

ROOT = Path(__file__).resolve().parents[1]
CODER = 'polytopia-skirmish-coder:local-v1'
REFEREE = 'polytopia-skirmish-referee:local-v1'
SDK = ['__init__.py', 'environment.py', 'ruleset.py',
    'learning/__init__.py', 'learning/adapter.py', 'learning/interface.py', 'learning/memory.py',
    'rl/__init__.py', 'rl/actions.py', 'rl/cpu_engine.py', 'rl/observation.py', 'rl/rust_engine.py',
    'studio/__init__.py', 'studio/environment.py', 'maps/__init__.py', 'maps/variants.py', 'maps/state.py',
    'skirmish/__init__.py', 'skirmish/common.py', 'skirmish/authoring_tools.py', 'skirmish/wasi.py', 'skirmish/wasi_worker.py']

DOCKERFILE = '''FROM python:3.12-slim-bookworm AS base
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config procps util-linux passwd && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain 1.90.0
ENV PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin
RUN rustup target add wasm32-wasip1
RUN pip install --no-cache-dir jax==0.11.2 numpy==2.4.4 maturin==1.9.6 wasmtime==38.0.0
COPY engine-rs /build/engine-rs
RUN cd /build/engine-rs && maturin build --release --locked -o /tmp/wheels && pip install /tmp/wheels/*.whl && rm -rf /build /tmp/wheels
ENV JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

FROM base AS coder
COPY bot /bot
RUN cd /bot && cargo fetch --locked && cargo build --release --offline --locked --target wasm32-wasip1
COPY engine-rs/src /reference/engine-rs/src
COPY rules.json /reference/rules.json
COPY mini /reference
COPY sdk /opt/skirmish-sdk/polytopia_bot
RUN /usr/sbin/useradd --create-home --shell /bin/bash coder && cp -a /root/.cargo /opt/cargo && cp -a /root/.rustup /opt/rustup && cp -a /root/.cargo /home/coder/.cargo && chown -R coder:coder /home/coder/.cargo && cp -a /bot /verify && chown -R coder:coder /bot && chmod -R a+rX /opt/cargo /opt/rustup && rm -rf /root/.cargo /root/.rustup
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:/usr/local/bin:/usr/bin:/bin PYTHONPATH=/opt/skirmish-sdk
RUN cp /reference/skirmish.sh /usr/local/bin/skirmish && chmod 755 /usr/local/bin/skirmish && cp /bot/src/main.rs /reference/STARTER.rs && python -c "from polytopia_bot.skirmish.authoring_tools import build_examples;build_examples('/reference/examples')"
WORKDIR /bot

FROM base AS referee
COPY runtime /workspace/src/polytopia_bot
ENV PYTHONPATH=/workspace/src POLYTOPIA_EXECUTION_BACKEND=local-docker
WORKDIR /workspace
'''


def build():
    with tempfile.TemporaryDirectory(prefix='polytopia-local-build-') as directory:
        context = Path(directory)
        shutil.copytree(ROOT/'engine-rs', context/'engine-rs', ignore=shutil.ignore_patterns('target', '__pycache__'))
        shutil.copytree(ROOT/'skirmish/bot-template', context/'bot', ignore=shutil.ignore_patterns('target'))
        shutil.copytree(ROOT/'skirmish/mini', context/'mini')
        shutil.copyfile(ROOT/'src/polytopia_bot/gpu/rules.json', context/'rules.json')
        files = SDK + [str(p.relative_to(ROOT/'src/polytopia_bot')) for p in (ROOT/'src/polytopia_bot/gpu').glob('*.py')] + ['gpu/rules.json']
        for name in files:
            target = context/'sdk'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/'src/polytopia_bot'/name, target)
        shutil.copytree(ROOT/'src/polytopia_bot', context/'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'static'))
        (context/'Dockerfile').write_text(DOCKERFILE)
        for stage, tag in [('coder', CODER), ('referee', REFEREE)]:
            subprocess.run(docker('build', '--target', stage, '-t', tag, context), check=True)
    print(json.dumps(dict(coder=CODER, referee=REFEREE)))


if __name__ == '__main__':
    build()
