# =============================================================================
# SAM 2.1 Fine-Tuning Docker Image
# GPU-accelerated training for floor plan segmentation
# =============================================================================
# Base: NVIDIA CUDA 12.4 with cuDNN (Ubuntu 22.04, Python 3.10)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------- Clone SAM 2 repo ----------
WORKDIR /app/fine_tuning_SAM2
RUN git clone https://github.com/facebookresearch/sam2.git

# ---------- Install SAM 2 with dev/training extras ----------
WORKDIR /app/fine_tuning_SAM2/sam2
RUN pip install --no-cache-dir -e ".[dev]"

# ---------- Install Roboflow (for data download) ----------
RUN pip install --no-cache-dir roboflow

# ---------- Download pre-trained checkpoint (~309 MB) ----------
RUN mkdir -p ./checkpoints && \
    curl -L -o ./checkpoints/sam2.1_hiera_base_plus.pt \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"

# ---------- Create data directory ----------
RUN mkdir -p /app/fine_tuning_SAM2/data/train

# ---------- Copy scripts ----------
COPY scripts/download_data.py  /app/scripts/download_data.py
COPY scripts/entrypoint.sh     /app/scripts/entrypoint.sh
RUN chmod +x /app/scripts/entrypoint.sh

# ---------- Copy custom training config and script ----------
COPY configs/train.yaml  /app/fine_tuning_SAM2/sam2/sam2/configs/train.yaml
COPY scripts/train.py    /app/fine_tuning_SAM2/sam2/training/train.py

# ---------- Environment ----------
ENV PYTHONPATH="/app/fine_tuning_SAM2/sam2:${PYTHONPATH}"
ENV NCCL_DEBUG=INFO

# ---------- TensorBoard port ----------
EXPOSE 6006

# ---------- Entrypoint: download data → start training ----------
WORKDIR /app/fine_tuning_SAM2/sam2
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["python", "training/train.py", "-c", "train", "--use-cluster", "0", "--num-gpus", "1"]
