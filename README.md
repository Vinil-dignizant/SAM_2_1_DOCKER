# SAM 2.1 Fine-Tuning — Floor Plan Segmentation

Fine-tune [SAM 2.1](https://github.com/facebookresearch/sam2) (Hiera Base+) on custom floor plan data using Docker.

---

## 📁 Project Structure

```
sam_comt/
├── Dockerfile                  # CUDA 12.4 container for training
├── docker-compose.yml          # GPU passthrough + volume mounts
├── .dockerignore
├── configs/
│   └── train.yaml              # Training hyperparameters (container paths)
├── scripts/
│   └── train.py                # Training launcher script
├── data/
│   └── train/                  # Training images + annotations (.jpg + .json)
├── sam2_logs/                   # Checkpoints & TensorBoard logs (auto-created)
├── data_download.ipynb         # ⬇️  Notebook to download training data
├── quick_training.md           # Detailed step-by-step training guide
└── README.md
```

---

## ⬇️ Step 1 — Download Training Data

Open and run the **[data_download.ipynb](data_download.ipynb)** notebook. It does the following:

1. Installs the `roboflow` package
2. Downloads the **cortex_floor_plan_roomonly** dataset in SAM 2 format
3. Renames the output folder to `data/`

```python
# What the notebook runs:
!pip install roboflow
import os
from roboflow import Roboflow

rf = Roboflow(api_key="tzmNp3NOHOyu1fb0hOVo")
project = rf.workspace("vinil-grlb1").project("cortex_floor_plan_roomonly")
version = project.version(1)
dataset = version.download("sam2")
os.rename(dataset.location, "data")
```

After running, you should have:

```
data/
└── train/
    ├── image_001.jpg
    ├── image_001.json
    ├── image_002.jpg
    ├── image_002.json
    └── ...
```

> **Note:** If filenames contain extra dots, run the renaming fix from [quick_training.md](quick_training.md#step-6-prepare-dataset) (Step 6.3).

---

## 🐳 Step 2 — Docker Setup

### Prerequisites

| Requirement | Minimum |
|---|---|
| Docker Engine | v20+ |
| Docker Compose | v2+ |
| NVIDIA GPU | ≥ 16 GB VRAM recommended (32 GB ideal) |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | Installed & configured |

Verify GPU access in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Build & Run

```bash
# Build the image (~15-20 min first time — clones SAM2 & downloads checkpoint)
docker compose build

# Start training
docker compose up

# Also

docker compose up --build
```

### What happens during build

1. Installs Python 3.10 + system deps on CUDA 12.4 Ubuntu base
2. Clones the official SAM 2 repo from GitHub
3. Installs SAM 2 with training extras (`pip install -e ".[dev]"`)
4. Downloads the `sam2.1_hiera_base_plus.pt` checkpoint (~309 MB)
5. Copies in the custom `train.yaml` and `train.py`

---

## 🚀 Step 3 — Training

Training starts automatically when you run `docker compose up`. To run manually:

```bash
docker run --gpus all --shm-size=8g \
  -v ./data/train:/app/fine_tuning_SAM2/data/train \
  -v ./sam2_logs:/app/fine_tuning_SAM2/sam2/sam2_logs \
  -p 6006:6006 \
  sam2-training
```

---

## 📊 Step 4 — Monitor with TensorBoard

Docker Compose starts a TensorBoard sidecar automatically.

| Service | URL |
|---|---|
| Training TensorBoard | `http://localhost:6006` |
| Sidecar TensorBoard | `http://localhost:6007` |

Or run TensorBoard manually:

```bash
tensorboard --bind_all --logdir ./sam2_logs
```

### Key Metrics

| Metric | Target |
|---|---|
| `total_loss` | < 0.35 |
| `loss_mask` | Pixel-level accuracy |
| `loss_dice` | Boundary quality |
| `loss_iou` | Overlap score |

---

## 💾 Checkpoints

Checkpoints are saved every **5 epochs** to:

```
sam2_logs/train/checkpoints/checkpoint_XX.pt
```

This folder is volume-mounted, so checkpoints persist on the host even if the container stops.

---

## ⚙️ Key Hyperparameters

Edit [`configs/train.yaml`](configs/train.yaml) to tune training:

| Parameter | Default | Notes |
|---|---|---|
| `scratch.train_batch_size` | `2` | Lower if OOM, increase with more VRAM |
| `scratch.base_lr` | `5e-6` | Don't exceed `1e-5` |
| `scratch.vision_lr` | `1e-6` | Lower LR for backbone |
| `scratch.num_epochs` | `150` | Reduce to `100` for faster runs |
| `scratch.num_train_workers` | `4` | DataLoader parallelism |
| `trainer.checkpoint.save_freq` | `5` | Checkpoint every N epochs |

After editing, rebuild: `docker compose build --no-cache`

---

## 🔧 Troubleshooting

| Issue | Fix |
|---|---|
| CUDA out of memory | Set `train_batch_size: 1` in `configs/train.yaml` |
| Shared memory error | Increase `shm_size` in `docker-compose.yml` |
| Checkpoint not found | Verify the build downloaded `sam2.1_hiera_base_plus.pt` |
| GPU not visible in Docker | Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| DataLoader deadlock | Set `num_train_workers: 0` |

---

## 📚 References

- [SAM 2 GitHub](https://github.com/facebookresearch/sam2)
- [Detailed Training Guide](quick_training.md)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
