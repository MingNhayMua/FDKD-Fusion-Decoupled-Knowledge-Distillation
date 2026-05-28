# FDKD: Fusion Decoupled Knowledge Distillation for Progressive Knowledge Transfer

Authors: Quang-Minh Tran and Viet-Hoang Nguyen.

> **Abstract:** Knowledge distillation is a standard approach for model compression, but its effectiveness degrades under large teacher–student capacity gaps, resulting in optimization instability or collapse and poor feature alignment. While **Decoupled Knowledge Distillation (DKD)** improves supervision by separating target and non-target components, it does not address capacity mismatch. **Teacher Assistant Knowledge Distillation (TAKD)** alleviates this gap via intermediate models but relies on conventional objectives that entangle signals and propagate noisy logit distributions, limiting training stability under heterogeneous settings. To address this, we propose **Fusion Decoupled Knowledge Distillation (FDKD)**, a unified framework that integrates TAKD with stage-wise decoupled supervision to enable progressive knowledge transfer. Unlike traditional multi-stage methods that rely on conventional KD objectives and propagate noisy logit distributions, FDKD exploits a decoupled objective across the teacher–assistant–student hierarchy, preserving both global category relationships and local sample-specific information as model capacity scales down. Extensive experiments on standard benchmarks demonstrate that the proposed FDKD achieves superior performance under large capacity gaps, **outperforming vanilla TAKD and standalone DKD by 7.53% and 12.58% in top-1 accuracy**, respectively.

![FDKD Framework](image/FDKD.jpg)

---

## Overview

FDKD combines **TAKD** (Teacher Assistant Knowledge Distillation) with **DKD** (Decoupled Knowledge Distillation) to enable progressive knowledge transfer across large capacity gaps:

```
Stage 1: Teacher (Swin-B, frozen) ──DKD──► Assistant (ResNet-152)
Stage 2: Assistant (frozen)       ──DKD──► Student   (ResNet-18)
```

The interactive demo compares **4 models** side-by-side for visual analysis:
- **Teacher** — Swin-B (86.95M), trained fully
- **DKD (Direct)** — ResNet-18 (11.28M), distilled directly from Swin-B
- **TAKD** — ResNet-18 (11.28M), FDKD result: Swin-B → ResNet-152 → ResNet-18
- **Baseline** — ResNet-18 (11.28M), trained with hard labels (no KD)

---

## Project Structure

```
├── run_colab.py                        ← Colab launcher
├── run_local.py                        ← Local launcher
├── fdkd_colab.ipynb                    ← Colab notebook (Miniforge + conda)
│
├── utils/                              ← Shared modules
│   ├── config.py                       ← Global config (single source of truth)
│   ├── distributions.py                ← Softmax, logits_to_probs, top-k
│   ├── math_utils.py                   ← KL divergence, cosine, entropy, rank_corr
│   ├── image.py                        ← Image preprocessing
│   ├── labels.py                       ← Tiny ImageNet class labels
│   └── tiny_imagenet_labels.json       ← 200-class label mapping
│
├── backend/                            ← FastAPI inference backend
│   ├── main.py                         ← FastAPI app + API routes
│   ├── requirements.txt
│   ├── models/
│   │   ├── loader.py                   ← Traced model loading
│   │   └── swin_pure.py                ← Pure PyTorch Swin-B implementation
│   ├── inference/
│   │   ├── pipeline.py                 ← Multi-model inference
│   │   ├── dkd.py                      ← TCKD / NCKD decomposition
│   │   └── metrics.py                  ← Distribution-level metrics
│   └── visualization/
│       └── gradcam.py                  ← FC-weight GradCAM heatmaps
│
├── frontend/                           ← Next.js interactive demo
│   ├── src/app/                        ← Pages + global styles
│   ├── src/components/                 ← UI components
│   ├── src/hooks/                      ← State management (Zustand)
│   ├── src/services/                   ← API client (Axios)
│   └── src/types/                      ← TypeScript definitions
│
├── training/                           ← MMPretrain + MMRazor configs
│   ├── README.md                       ← Training instructions
│   └── configs/
│       ├── _base/                      ← Shared base configs
│       ├── distill_dkd/                ← DKD distillation configs
│       ├── distill_fitnets/            ← FitNets baseline
│       ├── distill_crd/                ← CRD baseline
│       ├── distill_kd/                 ← Vanilla KD baseline
│       └── distill_ofd/                ← OFD baseline
│
└── checkpoints/                        ← Place .pth files here (gitignored)
```

---

## Quick Start

### 1. Backend — Google Colab (T4 GPU)

Open [`fdkd_colab.ipynb`](fdkd_colab.ipynb) in Colab and run all cells, or:

```bash
!git clone https://github.com/MingNhayMua/FDKD-Fusion-Decoupled-Knowledge-Distillation.git
%cd FDKD-Fusion-Decoupled-Knowledge-Distillation

# Install miniforge + conda env (see notebook for full setup)
# Export models from checkpoints:
!conda run -n openmmlab python export_models.py --checkpoint-dir "/content/drive/MyDrive/CS338-checkpoint"

# Start server:
!conda run --no-capture-output -n openmmlab python run_colab.py --checkpoint-dir "/content/drive/MyDrive/CS338-checkpoint" --token "YOUR_TOKEN" --port 8000
```

Copy the printed ngrok URL.

### 2. Frontend — Local

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000 → paste ngrok URL → upload image
```

### 3. Deploy Frontend to Vercel

```bash
cd frontend
npx vercel
```

---

## Architecture

```
Google Colab (T4 GPU)                    Frontend (Vercel / localhost:3000)
┌────────────────────────┐              ┌──────────────────────────┐
│  FastAPI Backend        │   ngrok     │  Next.js Frontend         │
│  ├── 4-model inference  │◄──────────►│  ├── Model Comparison     │
│  ├── DKD decomposition  │   HTTPS    │  ├── GradCAM Heatmaps     │
│  ├── GradCAM heatmaps   │            │  ├── Distribution Charts  │
│  └── Pairwise metrics   │            │  └── DKD Analysis (TCKD/  │
│                         │            │      NCKD/Dark Knowledge)  │
│  Google Drive mounted   │            │                           │
│  └── checkpoints/       │            │                           │
└────────────────────────┘              └──────────────────────────┘
```

## Demo Models

The demo loads 4 models from Google Drive checkpoints:

| File | Architecture | Role |
|---|---|---|
| `swinb_fully.pth` | Swin-B (86.95M) | **Teacher** |
| `swinb_r18_clean.pth` | ResNet-18 (11.28M) | **DKD (Direct)** — distilled from Swin-B |
| `disilledr152_r18_clean.pth` | ResNet-18 (11.28M) | **TAKD** — FDKD result (Swin→R152→R18) |
| `r18_fully.pth` | ResNet-18 (11.28M) | **Baseline** — hard labels, no KD |

The backend auto-detects checkpoints by name (see `utils/config.py`).

> **Download checkpoints:** [Google Drive](https://drive.google.com/drive/folders/1FOVw29_-ZqeAijI36_Gd_asOcbNqtdIR?usp=sharing)

---

## Results (Tiny ImageNet)

| Method | Top-1 | Top-5 | Params |
|---|---|---|---|
| Swin-B (Teacher) | 90.81% | 98.42% | 86.95M |
| ResNet-18 (Supervised) | 68.87% | 87.72% | 11.28M |
| DKD (direct) | 63.06% | 83.85% | 11.28M |
| FitNets | 73.30% | 91.03% | 11.28M |
| **FDKD (ours)** | **75.85%** | **92.16%** | **11.28M** |

---

## Tech Stack

**Backend:** Python, FastAPI, PyTorch, torchvision, scipy  
**Frontend:** Next.js 14, TypeScript, TailwindCSS, Framer Motion, Recharts  
**Training:** MMPretrain, MMRazor  
**Deployment:** Google Colab (GPU), Vercel (frontend), ngrok (tunnel)

---

## Citation

```bibtex
@article{tran2025fdkd,
  title={Fusion Decoupled Knowledge Distillation: Knowledge Transfer via Decoupling Teaching Assistant},
  author={Tran, Q.-M. and Nguyen, V.-H.},
  year={2025}
}
```
