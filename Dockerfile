# StructCheck — Docker image matching cluster `structcheck-rl.sqsh` (conda env `llm`)
# ---------------------------------------------------------------------------
# Python deps are a **full lock** parsed from the sqsh METADATA (108 dist-infos → 107 pins),
# same as a `pip freeze` of the environment you actually run under Slurm.
#
# Regenerate pins after rebuilding sqsh (overwrites requirements.txt):
#   python3 tools/extract_sqsh_requirements_lock.py /path/to/structcheck-rl.sqsh
#
# Optional: copy site `.whl` files into `docker/wheels/` before build; they install **after**
# `pip install -r requirements.txt`.
#
# Build:
#   docker build -t structcheck-rl:$(date +%Y%m%d) -f Dockerfile .
#
# Run (GPU):
#   docker run --gpus all -it -v /work:/work structcheck-rl:TAG bash
# ---------------------------------------------------------------------------

FROM --platform=linux/amd64 nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential pkg-config \
    libjpeg-turbo8 libpng16-16 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm -f miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/envs/llm/bin:/opt/conda/bin:${PATH}"

RUN conda create -y -n llm python=3.10 pip \
    && conda clean -afy

# Match PyTorch / torch-* / nvidia-* / xformers wheels used on the cluster
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# `pip` inside conda `llm` (3.10) — same layout as Slurm: /opt/conda/envs/llm/bin/python3.10
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt \
        --extra-index-url "${TORCH_INDEX_URL}"

# Optional: extra wheels (e.g. mirrors of what was `pip install *.whl` on the build host for sqsh)
COPY docker/wheels /opt/wheelhouse
RUN bash -ce 'shopt -s nullglob; W=(/opt/wheelhouse/*.whl); \
    if [ ${#W[@]} -gt 0 ]; then pip install --no-cache-dir "${W[@]}"; fi; \
    rm -rf /opt/wheelhouse'

RUN echo 'export PATH="/opt/conda/envs/llm/bin:/opt/conda/bin:$PATH"' >> /root/.bashrc

LABEL org.opencontainers.image.title="structcheck-rl"
LABEL org.opencontainers.image.description="StructCheck: full pip lock from structcheck-rl.sqsh llm env"
LABEL structcheck.base_image="nvidia/cuda:12.8.0-runtime-ubuntu22.04"
LABEL structcheck.conda_env="llm"
LABEL structcheck.lock_source="requirements.txt"

CMD ["/bin/bash"]
