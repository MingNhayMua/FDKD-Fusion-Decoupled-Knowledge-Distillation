# Training — FDKD on Tiny ImageNet

Training uses **MMPretrain** (classification models) + **MMRazor** (knowledge distillation).

## Environment Setup

```bash
conda create -n fdkd python=3.10 -y
conda activate fdkd

# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OpenMMLab ecosystem
pip install -U openmim
mim install mmengine
mim install "mmpretrain>=1.0.0"
mim install "mmrazor>=1.0.0"
```

## Dataset

Download Tiny ImageNet (200 classes, 64×64 images):

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/
```

## Training Pipeline

The FDKD pipeline consists of **3 training stages**. Run all commands from the **project root**.

### Step 0: Train Teacher (Swin-B, fully supervised)

Train Swin-B on Tiny ImageNet to get the teacher model:

```bash
python tools/train.py training/configs/swinb_tinyimagenet.py \
    --work-dir work_dirs/swinb_fully
```

### Step 1: FDKD Stage 1 — Swin-B → ResNet-152 (DKD)

Freeze teacher (Swin-B), distill to assistant (ResNet-152) using Decoupled Knowledge Distillation:

```bash
python tools/train.py training/configs/distill_dkd/dkd_swin-base_resnet152_tiny_imagenet.py \
    --work-dir work_dirs/dkd_swinb_r152
```

### Step 2: FDKD Stage 2 — Distilled R152 → R18 (DKD)

Freeze distilled assistant (R152), distill to student (R18):

```bash
python tools/train.py training/configs/distill_dkd/dkd_resnet152_resnet18_tiny_imagenet.py \
    --work-dir work_dirs/dkd_r152_r18
```

### Baselines (for comparison)

```bash
# Direct distillation: Swin-B → R18 (DKD)
python tools/train.py training/configs/distill_dkd/dkd_swin-base_resnet18_tiny_imagenet.py

# FitNets: Swin-B → R18
python tools/train.py training/configs/distill_fitnets/fitnets_swin-base_resnet18_tiny_imagenet.py

# CRD: Swin-B → R18
python tools/train.py training/configs/distill_crd/crd_swin-base_resnet18_tiny_imagenet.py

# OFD: Swin-B → R18
python tools/train.py training/configs/distill_ofd/ofd_swin-base_resnet18_tiny_imagenet.py
```

## Config Structure

```
training/configs/
├── _base_/                                          ← Shared MMPretrain configs
│   ├── datasets/
│   │   └── tinyimagenet_bs64_224.py                 ← Tiny ImageNet (200 classes)
│   ├── models/
│   │   ├── resnet18.py                              ← ResNet-18 architecture
│   │   └── swin_transformer_base.py                 ← Swin-B architecture
│   ├── schedules/
│   │   ├── imagenet_bs256.py                        ← SGD schedule
│   │   └── imagenet_bs1024_adamw_swin.py            ← AdamW for Swin
│   └── default_runtime.py                          ← Logging, checkpointing
│
├── distill_dkd/                                     ← DKD configs (MMRazor)
│   ├── dkd_swin-base_resnet152_tiny_imagenet.py     ← FDKD Stage 1
│   ├── dkd_resnet152_resnet18_tiny_imagenet.py      ← FDKD Stage 2
│   ├── dkd_swin-base_resnet18_tiny_imagenet.py      ← Direct DKD baseline
│   ├── dkd_swin-base_resnet101_tiny_imagenet.py     ← R101 variant
│   └── dkd_resnet101_resnet18_8xb32_tiny_imagenet.py
│
├── distill_fitnets/                                 ← FitNets baseline
│   └── fitnets_swin-base_resnet18_tiny_imagenet.py
│
├── distill_crd/                                     ← CRD baseline
│   └── crd_swin-base_resnet18_tiny_imagenet.py
│
└── distill_ofd/                                     ← OFD baseline
    └── ofd_swin-base_resnet18_tiny_imagenet.py
```

## DKD Hyperparameters

From the actual config used in training:

| Parameter | Value | Description |
|---|---|---|
| τ (tau) | 1 | Temperature for KL divergence |
| β (beta) | 1.0 | NCKD weight |
| loss_weight | 1 | Overall DKD loss weight |
| LR | 0.001 | SGD learning rate |
| Momentum | 0.9 | SGD momentum |
| Weight Decay | 1e-4 | L2 regularization |
| Grad Clip | 5.0 | Gradient clipping max norm |
| LR Warmup | 5 epochs | Linear warmup (0.01 → 1.0) |
| LR Milestones | [30, 60, 90] | MultiStepLR decay |
| LR Gamma | 0.1 | Decay factor |

## Results (Tiny ImageNet)

| Method | Teacher → Student | Top-1 | Top-5 |
|---|---|---|---|
| Swin-B (Teacher) | — | 90.81% | 98.42% |
| ResNet-18 (supervised) | — | 68.87% | 87.72% |
| KD (Swin-B → R18) | Swin-B → R18 | — | — |
| DKD (Swin-B → R18, direct) | Swin-B → R18 | 63.06% | 83.85% |
| FitNets (Swin-B → R18) | Swin-B → R18 | 73.30% | 91.03% |
| CRD (Swin-B → R18) | Swin-B → R18 | — | — |
| OFD (Swin-B → R18) | Swin-B → R18 | — | — |
| **FDKD (ours, 2-stage)** | Swin-B → R152 → R18 | **75.85%** | **92.16%** |

## Checkpoint Mapping

After training, copy `work_dirs/` to Google Drive for the demo backend:

| work_dir | Demo Role | Description |
|---|---|---|
| `swinb_fully/` | Teacher | Swin-B fully supervised |
| `dkd_swinb_r152/` | Assistant | Stage 1 distilled R152 |
| `dkd_r152_r18/` | Student | Stage 2 distilled R18 (FDKD) |
| `dkd_swinb_r18/` | — | Direct DKD baseline |
| `fitnets_swinb_r18/` | — | FitNets baseline |
