ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG RUST_VERSION=1.85.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG RUST_VERSION
ARG CARGO_FEATURES="cuda-ep"

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"

WORKDIR /app

COPY . .

RUN cargo build --locked --release --features "${CARGO_FEATURES}" --bin doc2agent \
    && mkdir -p /opt/onnxruntime \
    && ort_dir="$(find "${HOME}/.cache/ort.pyke.io" -type f \( -name 'libonnxruntime.so*' -o -name 'libonnxruntime_providers_shared.so' \) -print -quit 2>/dev/null | xargs -r dirname)" \
    && if [[ -n "${ort_dir}" ]]; then cp -a "${ort_dir}/." /opt/onnxruntime/; fi

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgcc-s1 \
    libgomp1 \
    libssl3 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/doc2agent /usr/local/bin/doc2agent
COPY --from=builder /opt/onnxruntime /opt/onnxruntime
COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /models /opt/pdfium

ENV DOC2AGENT_HOST=0.0.0.0 \
    DOC2AGENT_PORT=3000 \
    DOC2AGENT_REQUEST_TIMEOUT=60 \
    DOC2AGENT_MAX_BODY_SIZE=52428800 \
    DOC2AGENT_EXTRACTION_CONCURRENCY=16 \
    DOC2AGENT_OCR_CONCURRENCY=4 \
    DOC2AGENT_SESSION_POOL_SIZE=4 \
    DOC2AGENT_MAX_BATCH=32 \
    DOC2AGENT_INTRA_THREADS=1 \
    DOC2AGENT_INTER_THREADS=1 \
    DOC2AGENT_DEVICE_ID=0 \
    RUST_LOG=doc2agent=info,tower_http=info

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD bash -ec 'curl -fsS "http://127.0.0.1:${DOC2AGENT_PORT:-3000}/health" >/dev/null'

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["doc2agent"]

