FROM --platform=linux/amd64 nvidia/cuda:12.1.1-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential \
    python3 python3-pip python3-venv \
    ca-certificates libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Miniconda
# -----------------------------
WORKDIR /tmp
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# -----------------------------
# Accept Anaconda Terms of Service
# -----------------------------
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# -----------------------------
# Create conda environment
# -----------------------------
RUN conda create -y -n llm python=3.10

# -----------------------------
# Install PyTorch (CPU fallback)
# HPC nodes will override with CUDA version
# -----------------------------
RUN conda run -n llm pip install torch --index-url https://download.pytorch.org/whl/cu121


# -----------------------------
# Install HuggingFace + TRL stack
# -----------------------------
RUN conda run -n llm pip install \
        transformers \
        datasets \
        accelerate \
        peft \
        trl \
        bitsandbytes \
        sentencepiece \
        protobuf \
        openpyxl \
        mergekit \
        llm_blender \
        weave

# -----------------------------
# Install Unsloth + speedups
# -----------------------------
RUN conda run -n llm pip install unsloth xformers

# -----------------------------
# Auto-activate env for interactive shells
# -----------------------------
RUN echo "source /opt/conda/bin/activate llm" >> ~/.bashrc

# -----------------------------
# Working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Default command
# -----------------------------
CMD ["/bin/bash"]
