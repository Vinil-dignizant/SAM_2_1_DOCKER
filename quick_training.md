# SAM 2.1 Fine-Tuning — Complete Training Guide

> A step-by-step guide to set up and run SAM 2.1 fine-tuning for **floor plan segmentation**.

---

## Prerequisites

| Requirement         | Details                                                         |
|---------------------|-----------------------------------------------------------------|
| **OS**              | Windows (Git Bash) or Linux                                     |
| **Python**          | 3.10+                                                           |
| **GPU**             | NVIDIA GPU with CUDA support (VRAM ≥ 32 GB recommended)         |
| **CUDA Toolkit**    | Compatible with PyTorch ≥ 2.5.1                                 |
| **Git**             | Installed and available in terminal                             |
| **Internet**        | Required for cloning repo, downloading checkpoints, and dataset |

---

## Project Structure Overview

After setup, your project directory should look like this:

```
<YOUR_PROJECT_ROOT>/
├── fine_tuning_SAM2/
│   ├── data/
│   │   └── train/                     ← Training images + ground truth masks
│   └── sam2/                          ← Cloned SAM 2 repository (repo root)
│       ├── checkpoints/
│       │   └── sam2.1_hiera_base_plus.pt   ← Pre-trained weights
│       ├── sam2/
│       │   └── configs/
│       │       └── train.yaml         ← ⚙️ TRAINING CONFIG (you modify this)
│       ├── training/
│       │   └── train.py               ← 🚀 TRAINING SCRIPT (you modify this)
│       ├── sam2_logs/                  ← Created automatically during training
│       └── setup.py
└── sam_env/                           ← Python virtual environment
```

> **Key Point:** You need to modify **two files** to set up training:
> 1. `fine_tuning_SAM2/sam2/sam2/configs/train.yaml` — training configuration
> 2. `fine_tuning_SAM2/sam2/training/train.py` — training launcher script

---

## Step 1: Clone the SAM 2 Repository

Create your project folder and clone the official SAM 2 repo inside it:

```bash
mkdir -p "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2"
cd "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2"

git clone https://github.com/facebookresearch/sam2.git
```

This creates the `sam2/` directory under `fine_tuning_SAM2/`.

---

## Step 2: Create & Activate Virtual Environment

### Create a new virtual environment

```bash
python -m venv "<YOUR_PROJECT_ROOT>/sam_env"
```

### Activate it

**Windows (Git Bash):**
```bash
source "<YOUR_PROJECT_ROOT>/sam_env/Scripts/activate"
```

**Linux / macOS:**
```bash
source "<YOUR_PROJECT_ROOT>/sam_env/bin/activate"
```

### Verify Python version

```bash
python -V
# Should show Python 3.10+
```

---

## Step 3: Install Dependencies

Navigate to the cloned repo root and install SAM 2 with development extras:

```bash
cd "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/sam2"

pip install --upgrade pip
pip install -e ".[dev]"
```

This installs all core + training dependencies:
- **Core:** `torch`, `torchvision`, `numpy`, `hydra-core`, `iopath`, `pillow`
- **Training (dev):** `fvcore`, `tensorboard`, `submitit`, `opencv-python`, `tensordict`, etc.

---

## Step 4: Verify PyTorch + CUDA

Run this in Python to confirm GPU access:

```python
import torch, sys
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
```

> **Expected output:** `CUDA available: True` and your GPU name shown. If CUDA is not available, reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Step 5: Download Model Checkpoint

The base model weights need to be in `fine_tuning_SAM2/sam2/checkpoints/`.

### Direct download (specific checkpoint)

```bash
mkdir -p ./checkpoints
curl -L -o ./checkpoints/sam2.1_hiera_base_plus.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
```

### Verify download (~309 MB)

```bash
ls -lh ./checkpoints/sam2.1_hiera_base_plus.pt
test -f ./checkpoints/sam2.1_hiera_base_plus.pt && echo "✅ Checkpoint OK" || echo "❌ Checkpoint MISSING"
```

---

## Step 6: Prepare Dataset

The training expects images and corresponding ground truth masks in the **same folder**, following SAM 2's naming convention.

### 6.1 — Download from Roboflow (example)

```python
pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
version = project.version(1)
dataset = version.download("sam2")
```

### 6.1.1 Use the below for dataset download

```python

!pip install roboflow
import os
from roboflow import Roboflow
rf = Roboflow(api_key="tzmNp3NOHOyu1fb0hOVo")
project = rf.workspace("vinil-grlb1").project("cortex_floor_plan_roomonly")
version = project.version(1)
dataset = version.download("sam2")
os.rename(dataset.location, "data")
```

### 6.2 — Place data in the correct folder

Move or rename the downloaded data folder to:

```
<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/data/train/
```

```python
import os
os.rename(dataset.location, "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/data/train")
```

### 6.3 — Fix filenames (if needed)

Roboflow-exported filenames may use dots (`.`) in the name. SAM 2 expects a clean `name_<number>.ext` format. Run this to fix:

```python
import os, re

FOLDER = "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/data/train"

for filename in os.listdir(FOLDER):
    new_filename = filename.replace(".", "_", filename.count(".") - 1)
    if not re.search(r"_\d+\.\w+$", new_filename):
        new_filename = new_filename.replace(".", "_1.")
    os.rename(os.path.join(FOLDER, filename), os.path.join(FOLDER, new_filename))

print("✅ Renamed files in:", FOLDER)
```

### Expected data structure

When exported from Roboflow in `sam2` format, the data folder will contain flat `.jpg` + `.json` pairs:

```
fine_tuning_SAM2/data/train/
├── 10000_png_rf_a4cfd3d2...._1.jpg     ← Source image
├── 10000_png_rf_a4cfd3d2...._1.json    ← Annotation (polygons/masks)
├── 10001_png_rf_0bf4a84d...._1.jpg
├── 10001_png_rf_0bf4a84d...._1.json
└── ...
```

Each `.json` file contains the segmentation mask annotations for the corresponding `.jpg` image.

---

## Step 7: Configure Training — `train.yaml`

> **File to modify:** `fine_tuning_SAM2/sam2/sam2/configs/train.yaml`

Replace the **entire contents** of this file with the configuration below. Update the paths marked with `# ← UPDATE THIS` to match your system.

<details>
<summary><strong>Click to expand full train.yaml</strong></summary>

```yaml
# @package _global_
#
# OPTIMIZED TRAINING CONFIG FOR FLOOR PLAN SEGMENTATION
# Based on SAM 2.1 Hiera Base+ model
#
# Key settings:
# - 150 epochs with warmup + cosine decay
# - Conservative learning rates (proven to work)
# - Batch size 2, AMP bfloat16

scratch:
  resolution: 1024
  train_batch_size: 2
  num_train_workers: 4               # Set to 0 if you hit multiprocessing issues on Windows
  num_frames: 1
  max_num_objects: 5
  base_lr: 5.0e-6                    # Conservative LR — proven stable
  vision_lr: 1.0e-6                  # Lower backbone LR for fine details
  phases_per_epoch: 1
  num_epochs: 150                    # Longer training for lower loss

dataset:
  # ← UPDATE THESE PATHS to your actual data location
  img_folder: "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/data/train"
  gt_folder: "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/data/train"
  multiplier: 2

# Data augmentation transforms (optimized for floor plans)
vos:
  train_transforms:
    - _target_: training.dataset.transforms.ComposeAPI
      transforms:
        - _target_: training.dataset.transforms.RandomHorizontalFlip
          consistent_transform: True
        - _target_: training.dataset.transforms.RandomAffine
          degrees: 5                 # Conservative rotation for floor plans
          shear: 3                   # Preserve wall angles
          image_interpolation: bilinear
          consistent_transform: True
        - _target_: training.dataset.transforms.RandomResizeAPI
          sizes: ${scratch.resolution}
          square: true
          consistent_transform: True
        - _target_: training.dataset.transforms.ColorJitter
          consistent_transform: True
          brightness: 0.2
          contrast: 0.15
          saturation: 0.05
          hue: null
        - _target_: training.dataset.transforms.RandomGrayscale
          p: 0.15                    # Floor plans are often grayscale
          consistent_transform: True
        - _target_: training.dataset.transforms.ColorJitter
          consistent_transform: False
          brightness: 0.1
          contrast: 0.05
          saturation: 0.05
          hue: null
        - _target_: training.dataset.transforms.ToTensorAPI
        - _target_: training.dataset.transforms.NormalizeAPI
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]

trainer:
  _target_: training.trainer.Trainer
  mode: train_only
  max_epochs: ${times:${scratch.num_epochs},${scratch.phases_per_epoch}}
  accelerator: cuda
  seed_value: 42

  model:
    _target_: training.model.sam2.SAM2Train
    image_encoder:
      _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
      scalp: 1
      trunk:
        _target_: sam2.modeling.backbones.hieradet.Hiera
        embed_dim: 112
        num_heads: 2
        drop_path_rate: 0.1
      neck:
        _target_: sam2.modeling.backbones.image_encoder.FpnNeck
        position_encoding:
          _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
          num_pos_feats: 256
          normalize: true
          scale: null
          temperature: 10000
        d_model: 256
        backbone_channel_list: [896, 448, 224, 112]
        fpn_top_down_levels: [2, 3]
        fpn_interp_model: nearest

    memory_attention:
      _target_: sam2.modeling.memory_attention.MemoryAttention
      d_model: 256
      pos_enc_at_input: true
      layer:
        _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
        activation: relu
        dim_feedforward: 2048
        dropout: 0.1
        pos_enc_at_attn: false
        self_attention:
          _target_: sam2.modeling.sam.transformer.RoPEAttention
          rope_theta: 10000.0
          feat_sizes: [32, 32]
          embedding_dim: 256
          num_heads: 1
          downsample_rate: 1
          dropout: 0.1
        d_model: 256
        pos_enc_at_cross_attn_keys: true
        pos_enc_at_cross_attn_queries: false
        cross_attention:
          _target_: sam2.modeling.sam.transformer.RoPEAttention
          rope_theta: 10000.0
          feat_sizes: [32, 32]
          rope_k_repeat: True
          embedding_dim: 256
          num_heads: 1
          downsample_rate: 1
          dropout: 0.1
          kv_in_dim: 64
      num_layers: 4

    memory_encoder:
        _target_: sam2.modeling.memory_encoder.MemoryEncoder
        out_dim: 64
        position_encoding:
          _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
          num_pos_feats: 64
          normalize: true
          scale: null
          temperature: 10000
        mask_downsampler:
          _target_: sam2.modeling.memory_encoder.MaskDownSampler
          kernel_size: 3
          stride: 2
          padding: 1
        fuser:
          _target_: sam2.modeling.memory_encoder.Fuser
          layer:
            _target_: sam2.modeling.memory_encoder.CXBlock
            dim: 256
            kernel_size: 7
            padding: 3
            layer_scale_init_value: 1e-6
            use_dwconv: True
          num_layers: 2

    num_maskmem: 7
    image_size: ${scratch.resolution}
    sigmoid_scale_for_mem_enc: 20.0
    sigmoid_bias_for_mem_enc: -10.0
    use_mask_input_as_output_without_sam: true
    directly_add_no_mem_embed: true
    use_high_res_features_in_sam: true
    multimask_output_in_sam: true
    iou_prediction_use_sigmoid: True
    use_obj_ptrs_in_encoder: true
    add_tpos_enc_to_obj_ptrs: true
    proj_tpos_enc_in_obj_ptrs: true
    use_signed_tpos_enc_to_obj_ptrs: true
    only_obj_ptrs_in_the_past_for_eval: true
    pred_obj_scores: true
    pred_obj_scores_mlp: true
    fixed_no_obj_ptr: true
    multimask_output_for_tracking: true
    use_multimask_token_for_obj_ptr: true
    multimask_min_pt_num: 0
    multimask_max_pt_num: 1
    use_mlp_for_obj_ptr_proj: true
    no_obj_embed_spatial: true

    compile_image_encoder: False

    prob_to_use_pt_input_for_train: 0.5
    prob_to_use_pt_input_for_eval: 0.0
    prob_to_use_box_input_for_train: 0.5
    prob_to_use_box_input_for_eval: 0.0
    prob_to_sample_from_gt_for_train: 0.1

    num_frames_to_correct_for_train: 2
    num_frames_to_correct_for_eval: 1
    rand_frames_to_correct_for_train: True
    add_all_frames_to_correct_as_cond: True

    num_init_cond_frames_for_train: 2
    rand_init_cond_frames_for_train: True
    num_correction_pt_per_frame: 7

  data:
    train:
      _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset
      phases_per_epoch: ${scratch.phases_per_epoch}
      batch_sizes:
        - ${scratch.train_batch_size}

      datasets:
        - _target_: training.dataset.vos_dataset.VOSDataset
          transforms: ${vos.train_transforms}
          training: true
          video_dataset:
            _target_: training.dataset.vos_raw_dataset.SA1BRawDataset
            img_folder: ${dataset.img_folder}
            gt_folder: ${dataset.gt_folder}
          multiplier: ${dataset.multiplier}
          sampler:
            _target_: training.dataset.vos_sampler.RandomUniformSampler
            num_frames: 1
            max_num_objects: ${scratch.max_num_objects}
      shuffle: True
      num_workers: ${scratch.num_train_workers}
      pin_memory: True
      drop_last: True
      collate_fn:
        _target_: training.utils.data_utils.collate_fn
        _partial_: true
        dict_key: all

  optim:
    amp:
      enabled: True
      amp_dtype: bfloat16

    optimizer:
      _target_: torch.optim.AdamW

    gradient_clip:
      _target_: training.optimizer.GradientClipper
      max_norm: 0.1
      norm_type: 2

    options:
      lr:
        - scheduler:
            _target_: fvcore.common.param_scheduler.CompositeParamScheduler
            schedulers:
              # WARMUP: ~5 epochs of linear warmup
              - _target_: fvcore.common.param_scheduler.LinearParamScheduler
                start_value: 1.0e-7
                end_value: ${scratch.base_lr}
              # MAIN: Cosine decay to base_lr / 100
              - _target_: fvcore.common.param_scheduler.CosineParamScheduler
                start_value: ${scratch.base_lr}
                end_value: ${divide:${scratch.base_lr},100}
            lengths: [0.033, 0.967]    # 5/150 epochs warmup, rest cosine
            interval_scaling: ['rescaled', 'rescaled']
        - scheduler:
            _target_: fvcore.common.param_scheduler.CompositeParamScheduler
            schedulers:
              - _target_: fvcore.common.param_scheduler.LinearParamScheduler
                start_value: 1.0e-8
                end_value: ${scratch.vision_lr}
              - _target_: fvcore.common.param_scheduler.CosineParamScheduler
                start_value: ${scratch.vision_lr}
                end_value: ${divide:${scratch.vision_lr},100}
            lengths: [0.033, 0.967]
            interval_scaling: ['rescaled', 'rescaled']
          param_names:
            - 'image_encoder.*'
      weight_decay:
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.1
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.0
          param_names:
            - '*bias*'
          module_cls_names: ['torch.nn.LayerNorm']

  loss:
    all:
      _target_: training.loss_fns.MultiStepMultiMasksAndIous
      weight_dict:
        loss_mask: 20
        loss_dice: 3                   # Higher = better boundary learning
        loss_iou: 2                    # Higher = better overlap optimization
        loss_class: 1
      supervise_all_iou: true
      iou_use_l1_loss: true
      pred_obj_scores: true
      focal_gamma_obj_score: 2.0
      focal_alpha_obj_score: 0.25

  distributed:
    backend: gloo
    find_unused_parameters: True

  logging:
    tensorboard_writer:
      _target_: training.utils.logger.make_tensorboard_logger
      log_dir: ${launcher.experiment_log_dir}/tensorboard
      flush_secs: 60
    log_dir: ${launcher.experiment_log_dir}/logs
    log_freq: 5                        # Log every 5 steps

  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 5                       # Save checkpoint every 5 epochs
    model_weight_initializer:
      _partial_: True
      _target_: training.utils.checkpoint_utils.load_state_dict_into_model
      strict: True
      ignore_unexpected_keys: null
      ignore_missing_keys: null

      state_dict:
        _target_: training.utils.checkpoint_utils.load_checkpoint_and_apply_kernels
        checkpoint_path: ./checkpoints/sam2.1_hiera_base_plus.pt
        ckpt_state_dict_keys: ['model']

launcher:
  num_nodes: 1
  gpus_per_node: 1
  experiment_log_dir: sam2_logs

submitit:
  partition: null
  account: null
  qos: null
  cpus_per_task: 10
  use_cluster: false
  timeout_hour: 24
  name: null
  port_range: [10000, 65000]
```

</details>

### What to change in `train.yaml`

| Setting | Where | What to update |
|---------|-------|----------------|
| **Dataset paths** | `dataset.img_folder` and `dataset.gt_folder` | Absolute path to your `data/train/` folder |
| **Batch size** | `scratch.train_batch_size` | Increase if you have more VRAM (e.g., `4` for 24 GB GPUs) |
| **Workers** | `scratch.num_train_workers` | Set to `0` if you get multiprocessing errors on Windows |
| **Epochs** | `scratch.num_epochs` | `150` is recommended; reduce to `100` for a quicker run |
| **Learning rate** | `scratch.base_lr` | `5.0e-6` is proven stable. Don't increase above `1e-5` |
| **Distributed backend** | `trainer.distributed.backend` | Use `gloo` on Windows, `nccl` on Linux |

---

## Step 8: Set Up Training Script — `train.py`

> **File to modify:** `fine_tuning_SAM2/sam2/training/train.py`

Replace the **entire contents** of the original `train.py` with the version below. This is a modified version of the official training script with the following fixes:
- **Hydra config path fix** — points to `../sam2/configs` relative to the script
- **Auto CWD** — automatically changes working directory to the repo root so relative checkpoint paths resolve correctly
- **PYTHONPATH auto-insert** — adds the repo root to `sys.path` so `sam2` and `training` modules import correctly
- **Verbose config logging** — prints the full resolved config at startup for debugging

<details>
<summary><strong>Click to expand full train.py</strong></summary>

```python
# Replace the original sam2/training/train.py with this file
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import submitit
import torch

from hydra import compose, initialize
from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA runtime does not support `fork`
    if num_proc == 1:
        # directly call single_proc so we can easily set breakpoints
        # mp.spawn does not let us set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self):
        job_env = submitit.JobEnvironment()
        add_pythonpath_to_sys_path()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        register_omegaconf_resolvers()
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        trainer.run()

    def __call__(self):
        job_env = submitit.JobEnvironment()
        self.setup_job_info(job_env.job_id, job_env.global_rank)
        try:
            self.run_trainer()
        except Exception as e:
            message = format_exception(e)
            logging.error(message)
            raise e

    def setup_job_info(self, job_id, rank):
        """Set up slurm job info"""
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None),
            "experiment_log_dir": self.cfg.launcher.experiment_log_dir,
        }
        self.has_setup = True


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def main(args, overrides) -> None:
    cfg = compose(config_name=args.config, overrides=overrides)
    print("--- Loaded Configuration (cfg) ---")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------")
    print("----------------------------------")
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = cfg.launcher.experiment_log_dir
    submitit_dir = os.path.join(submitit_dir, "submitit_logs")
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
    if submitit_conf.use_cluster:
        executor = submitit.AutoExecutor(folder=submitit_dir)
        submitit_conf.partition = (
            args.partition
            if args.partition is not None
            else submitit_conf.get("partition", None)
        )
        submitit_conf.account = (
            args.account
            if args.account is not None
            else submitit_conf.get("account", None)
        )
        submitit_conf.qos = (
            args.qos if args.qos is not None else submitit_conf.get("qos", None)
        )
        job_kwargs = {
            "timeout_min": 60 * submitit_conf.timeout_hour,
            "name": (
                submitit_conf.name if hasattr(submitit_conf, "name") else args.config
            ),
            "slurm_partition": submitit_conf.partition,
            "gpus_per_node": cfg.launcher.gpus_per_node,
            "tasks_per_node": cfg.launcher.gpus_per_node,
            "cpus_per_task": submitit_conf.cpus_per_task,
            "nodes": cfg.launcher.num_nodes,
            "slurm_additional_parameters": {
                "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
            },
        }
        if "include_nodes" in submitit_conf:
            assert (
                len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes
            ), "Not enough nodes"
            job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(
                submitit_conf["include_nodes"]
            )
        if submitit_conf.account is not None:
            job_kwargs["slurm_additional_parameters"]["account"] = submitit_conf.account
        if submitit_conf.qos is not None:
            job_kwargs["slurm_additional_parameters"]["qos"] = submitit_conf.qos

        if submitit_conf.get("mem_gb", None) is not None:
            job_kwargs["mem_gb"] = submitit_conf.mem_gb
        elif submitit_conf.get("mem", None) is not None:
            job_kwargs["slurm_mem"] = submitit_conf.mem

        if submitit_conf.get("constraints", None) is not None:
            job_kwargs["slurm_constraint"] = submitit_conf.constraints

        if submitit_conf.get("comment", None) is not None:
            job_kwargs["slurm_comment"] = submitit_conf.comment

        if submitit_conf.get("srun_args", None) is not None:
            job_kwargs["slurm_srun_args"] = []
            if submitit_conf.srun_args.get("cpu_bind", None) is not None:
                job_kwargs["slurm_srun_args"].extend(
                    ["--cpu-bind", submitit_conf.srun_args.cpu_bind]
                )

        print("###################### SLURM Config ####################")
        print(job_kwargs)
        print("##########################################")
        executor.update_parameters(**job_kwargs)

        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        runner = SubmititRunner(main_port, cfg)
        job = executor.submit(runner)
        print(f"Submitit Job ID: {job.job_id}")
        runner.setup_job_info(job.job_id, rank=0)
    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":
    # Auto-detect repo root and set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../sam2/training
    package_root = os.path.dirname(script_dir)               # .../sam2 (repo root)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)

    # Change CWD to repo root so relative paths in configs resolve correctly
    try:
        os.chdir(package_root)
    except Exception:
        pass

    # Initialize Hydra with relative path from train.py to the config directory
    initialize(config_path="../sam2/configs", version_base="1.2")

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="config file name (e.g. train)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="0: run locally, 1: run on SLURM cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUs per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")

    args, overrides = parser.parse_known_args()

    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()

    main(args, overrides)
```

</details>

---

## Step 9: Run Training

All commands must be run from the **repo root** directory:

```bash
cd "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/sam2"
```

### Run training (single GPU, local)

**Windows (Git Bash):**
```bash
USE_LIBUV=0 python training/train.py -c train --use-cluster 0 --num-gpus 1
```

**Linux:**
```bash
python training/train.py -c train --use-cluster 0 --num-gpus 1
```

> **Note:** The `-c train` refers to the config filename `train.yaml` (without the `.yaml` extension) located in `sam2/configs/`.

### Alternative: Set environment variable first (Windows)

```bash
export USE_LIBUV=0
python training/train.py -c train --use-cluster 0 --num-gpus 1
```

---

## Step 10: Monitor Training with TensorBoard

Training logs are saved to `sam2_logs/train/tensorboard/` by default.

### Start TensorBoard

```bash
tensorboard --bind_all --logdir "<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/sam2/sam2_logs"
```

Then open `http://localhost:6006` in your browser.

### Key metrics to watch

| Metric | What it tells you |
|--------|-------------------|
| `total_loss` | Overall training loss (target: < 0.35) |
| `loss_mask` | Pixel-level mask accuracy |
| `loss_dice` | Boundary quality (Dice coefficient) |
| `loss_iou` | Intersection-over-Union overlap |

---

## Step 11: Use Trained Checkpoints

Checkpoints are saved every **5 epochs** to:

```
<YOUR_PROJECT_ROOT>/fine_tuning_SAM2/sam2/sam2_logs/train/checkpoints/
```

The checkpoint files are named `checkpoint_XX.pt` (e.g., `checkpoint_07.pt`).

To load a fine-tuned checkpoint for inference, use it in place of the base `sam2.1_hiera_base_plus.pt` checkpoint in your inference scripts.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **`CUDA out of memory`** | Reduce `scratch.train_batch_size` to `1` in `train.yaml` |
| **`Checkpoint assertion error`** | Verify `./checkpoints/sam2.1_hiera_base_plus.pt` exists and the path in `train.yaml` is correct |
| **Windows multiprocessing deadlock** | Set `scratch.num_train_workers: 0` in `train.yaml` |
| **`ModuleNotFoundError: No module named 'sam2'`** | Run from the repo root (`fine_tuning_SAM2/sam2/`) or ensure the `train.py` auto-paths are working |
| **`USE_LIBUV` error on Windows** | Prefix the command with `USE_LIBUV=0` |
| **`gloo` backend errors on Linux** | Change `trainer.distributed.backend` from `gloo` to `nccl` in `train.yaml` |
| **Training loss plateau** | Try reducing learning rate or increasing `num_epochs`. The proven config achieves ~0.35 loss |

---

## Quick Reference — Key Hyperparameters

| Parameter | Location in `train.yaml` | Default | Notes |
|-----------|--------------------------|---------|-------|
| Batch size | `scratch.train_batch_size` | `2` | Lower = more stable, higher = faster |
| Learning rate | `scratch.base_lr` | `5e-6` | Conservative; don't go above `1e-5` |
| Vision encoder LR | `scratch.vision_lr` | `1e-6` | Lower LR for pre-trained backbone |
| Epochs | `scratch.num_epochs` | `150` | More epochs = lower loss |
| Data workers | `scratch.num_train_workers` | `4` | Set `0` for Windows debugging |
| Max objects per image | `scratch.max_num_objects` | `5` | Increase for dense annotations |
| Checkpoint frequency | `trainer.checkpoint.save_freq` | `5` | Save every N epochs |
| AMP precision | `trainer.optim.amp.amp_dtype` | `bfloat16` | Mixed precision training |
| Gradient clipping | `trainer.optim.gradient_clip.max_norm` | `0.1` | Prevents gradient explosion |

---

## Tips

- **Always run commands from the repo root:** `fine_tuning_SAM2/sam2/`
- **Use absolute paths** in `train.yaml` for dataset folders
- **Match CUDA & PyTorch versions** — check at [pytorch.org](https://pytorch.org/get-started/locally/)
- **Windows users:** Use Git Bash, set `USE_LIBUV=0`, and use `gloo` backend
- **Linux users:** Use `nccl` backend for best GPU communication performance