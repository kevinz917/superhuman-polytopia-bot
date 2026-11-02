FROM python:3.12-slim-bookworm AS base
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config procps util-linux passwd && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain 1.90.0
ENV PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin
RUN rustup target add wasm32-wasip1
RUN pip install --no-cache-dir jax==0.11.2 numpy==2.4.4 maturin==1.9.6 wasmtime==38.0.0
COPY public.tar.gz /tmp/public.tar.gz
RUN mkdir /build-context && tar -xzf /tmp/public.tar.gz -C /build-context && rm /tmp/public.tar.gz
RUN mkdir -p /build/engine-rs && cp -a /build-context/engine-rs/. /build/engine-rs/
RUN cd /build/engine-rs && maturin build --release --locked -o /tmp/wheels && pip install /tmp/wheels/*.whl && rm -rf /build /tmp/wheels
ENV JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

FROM base AS coder
LABEL polytopia.skirmish.role=coder
RUN mkdir -p /bot && cp -a /build-context/bot/. /bot/
RUN cd /bot && cargo fetch --locked && cargo build --release --offline --locked --target wasm32-wasip1
RUN mkdir -p /reference/engine-rs/src && cp -a /build-context/engine-rs/src/. /reference/engine-rs/src/
RUN mkdir -p $(dirname /reference/rules.json) && cp -a /build-context/rules.json /reference/rules.json
RUN mkdir -p /reference && cp -a /build-context/mini/. /reference/
RUN mkdir -p /opt/skirmish-sdk/polytopia_bot && cp -a /build-context/sdk/. /opt/skirmish-sdk/polytopia_bot/
RUN /usr/sbin/useradd --create-home --shell /bin/bash coder && cp -a /root/.cargo /opt/cargo && cp -a /root/.rustup /opt/rustup && cp -a /root/.cargo /home/coder/.cargo && chown -R coder:coder /home/coder/.cargo && cp -a /bot /verify && chown -R coder:coder /bot && chmod -R a+rX /opt/cargo /opt/rustup && rm -rf /root/.cargo /root/.rustup
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:/usr/local/bin:/usr/bin:/bin PYTHONPATH=/opt/skirmish-sdk
RUN cp /reference/skirmish.sh /usr/local/bin/skirmish && chmod 755 /usr/local/bin/skirmish && cp /bot/src/main.rs /reference/STARTER.rs && python -c "from polytopia_bot.skirmish.authoring_tools import build_examples;build_examples('/reference/examples')"
RUN mkdir -p /bot/practice && chown coder:coder /bot/practice
ENV POLYTOPIA_SKIRMISH_CONTINUOUS=1 POLYTOPIA_EXECUTION_BACKEND=local-docker POLYTOPIA_SKIRMISH_CHECK_FUEL=4000000000000
WORKDIR /bot

RUN rm -rf /build-context
USER root
FROM coder AS verifier
COPY private.tar.gz /tmp/private.tar.gz
RUN mkdir /private-context && tar -xzf /tmp/private.tar.gz -C /private-context && rm /tmp/private.tar.gz && mkdir -p /workspace/src /suite && cp -a /private-context/runtime /workspace/src/polytopia_bot && cp -a /private-context/refs /refs && cp -a /private-context/tests /tests && cp /private-context/manifest.json /private-context/maps.json /suite/ && rm -rf /private-context
ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
USER root
