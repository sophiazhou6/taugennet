# TauGenNet

A 3D latent diffusion model for synthesizing tau PET brain scans conditioned on structural MRI and plasma biomarkers. Trained and evaluated on the ADNI dataset (187 subjects, AD and MCI cohorts).

---

## Overview

TauGenNet generates tau PET scans without requiring an actual PET scan. Given a subject's structural MRI and one of three conditioning signals, the model synthesizes a realistic tau PET volume at 96×112×96 voxels.

**Conditioning modes:**

| Mode | Input | Description |
|------|-------|-------------|
| `ptau217` | Scalar plasma p-tau217 (pg/mL) | Encoded via frozen CLIP as `"Plasma is X.XXX."` |
| `atrophy` | 86-dim FreeSurfer atrophy z-scores | Encoded via a learned MLP conditioner |
| `combined` | Both of the above | Atrophy MLP + CLIP embeddings concatenated |

---

## Architecture

```
MRI ──► Encoder3D ──► z_m ─────────────────────────────────────┐
                                                                 ▼
PET ──► Encoder3D ──► z_0 ──► Forward Diffusion ──► z_t ──► [z_t ‖ z_m] ──► DenoisingUNet3D ──► ε̂
                                                                 ▲
Conditioning ──► Conditioner ──► CLIP / MLP tokens ─────────────┘

At inference: z_T ~ N(0,I) ──► DDIM reverse (50 steps) ──► z_0 ──► Decoder3D ──► Synthesized PET
```

- **Autoencoder** (`src/models.py`): VAE with 3 stride-2 downsampling stages, 8× spatial compression, 3 latent channels. Shared weights for PET and MRI.
- **Denoising U-Net** (`src/models.py`): 3-level U-Net with channel widths [256, 512, 768], 6 transformer blocks per level (self-attention + cross-attention + FFN), GroupNorm(32).
- **Diffusion schedule** (`src/diffusion.py`): Linear beta schedule, T=1000 steps. Supports DDPM (full 500 steps) and DDIM (fast 50 steps).

---

## Repository Structure

```
taugennet/
├── src/                        # Core library
│   ├── config.py               # All hyperparameters and paths
│   ├── models.py               # Autoencoder3D, DenoisingUNet3D
│   ├── diffusion.py            # DiffusionSchedule, q_sample, p_sample, ddim_sample
│   ├── conditioning.py         # PTau217Conditioner, AtrophyConditioner, CombinedConditioner
│   ├── dataset.py              # v1 dataset loader (72/28 split)
│   ├── dataset_v2.py           # v2 dataset loader (configurable split)
│   ├── dataset_combined.py     # Combined-mode dataset loader
│   └── inference.py            # synthesize_tau_pet(), load_models()
│
├── scripts/                    # Training, evaluation, and utility scripts
│   ├── train.py                # Unified training entry point (all modes)
│   ├── evaluate.py             # Evaluation: MSE, SSIM, ROI metrics, ablations
│   ├── metrics_table.py        # Parse eval log → markdown metrics table
│   ├── run_experiment.py       # End-to-end orchestrator (submit → eval → commit)
│   ├── roi_table.py            # Region-of-interest MSE table
│   └── notebooks/              # Jupyter notebooks
│       └── tables_and_figures.ipynb
│
├── slurm/                      # SLURM job scripts (Princeton HPC)
│   ├── train/                  # Training jobs
│   │   └── train_paper.slurm  # Full paper pipeline (AE + 3 diffusion modes)
│   ├── eval/                   # Evaluation jobs
│   │   └── eval_paper.slurm
│   └── jobs/                   # Per-mode submission scripts
│
├── data/                       # Data (not committed — lives on HPC only)
│   ├── raw/                    # ADNI tau PET + MRI volumes, CSVs
│   ├── processed/              # Preprocessed intermediate files
│   └── generated/              # Synthesized PET outputs
│
├── results/                    # Outputs (not committed)
│   ├── checkpoints/            # Model checkpoints (.pt)
│   ├── figures/                # Generated figures
│   └── logs/                   # SLURM job logs
│
├── docs/
│   └── verification_report.md  # Paper vs. code audit
└── clip_model/                 # CLIP ViT-B/32 weights (not committed)
```

---

## Setup

**Environment** (Princeton HPC):

```bash
conda activate taugennet
# or use the full path:
/home/sz3962/.conda/envs/taugennet/bin/python3
```

**Register Jupyter kernel:**

```bash
python -m ipykernel install --user --name=taugennet --display-name "TauGenNet"
```

**Data paths** are configured in `src/config.py`. The raw ADNI data lives at `/scratch/network/sz3962/taugennet/data/raw/` and is not tracked by git.

---

## Training

### Full paper pipeline (recommended)

Trains the shared autoencoder + all three diffusion modes sequentially (~2 days on A100):

```bash
sbatch slurm/train/train_paper.slurm
```

### Custom training

```bash
# AE pretraining only
python scripts/train.py --mode atrophy --dataset v2 --ae-epochs 100 --diff-epochs 0 \
    --ae-checkpoint results/checkpoints/ae.pt

# Diffusion training (skip AE, load pretrained checkpoint)
python scripts/train.py --mode atrophy --dataset v2 --skip-ae \
    --ae-checkpoint results/checkpoints/ae.pt \
    --diff-epochs 2000 --patience 500

# Custom split (80/10/10)
python scripts/train.py --mode atrophy --dataset v2 --split 80/10/10 --skip-ae \
    --ae-checkpoint results/checkpoints/ae.pt
```

**Modes:** `atrophy` | `ptau217` | `combined`  
**Datasets:** `v2` (configurable split) | `combined` (combined mode only)

### Monitor jobs

```bash
squeue -u sz3962
tail -f results/logs/taugennet_<JOBID>.out
```

---

## Evaluation

```bash
# Submit evaluation job
sbatch slurm/eval/eval_paper.slurm

# Or run directly
python scripts/evaluate.py --mode atrophy

# Parse log into a metrics table
python scripts/metrics_table.py --log-file results/logs/taugennet_<JOBID>.out
```

**Metrics computed:**
- Global: MAE, MSE, PSNR, SSIM
- Per-ROI MSE across 6 Alzheimer's-relevant regions (parahippocampal, fusiform, inferior temporal, hippocampus, posterior cingulate, entorhinal) — Table II
- Ablation: MRI+Atrophy vs. Atrophy-only — Table I

---

## Inference

```python
from src.inference import load_models, synthesize_tau_pet
from src.diffusion import DiffusionSchedule
import torch

ae, unet, conditioner, latent_std, _ = load_models(
    "results/checkpoints/atrophy_paper/diff_atrophy.pt", mode="atrophy"
)
schedule = DiffusionSchedule()

# mri_vol: (1, 1, 96, 112, 96) normalised tensor
# cond:    (86,) atrophy z-score vector
generated_pet = synthesize_tau_pet(
    mri_vol, cond, ae, unet, schedule,
    conditioner.encode, latent_std,
    n_steps=50, sampler='ddim'
)
# → (1, 1, 96, 112, 96) synthesized tau PET
```

---

## Data

- **Dataset:** ADNI tau PET (cerebellar SUVR-normalized) + T1 MRI (registered to MNI space)
- **Subjects:** 187 (AD + MCI cohorts, mixed in all splits)
- **Default split:** 70% train / 10% val / 20% test (~131/19/37 subjects)
- **Volume shape:** 96×112×96 voxels (2 mm MNI space)
- **Conditioning CSVs:** `ADNI34Tau_withFluidBiomarkers.csv` (p-tau217), `regional_atrophy_zscores.csv` (86 FreeSurfer regions)

Raw data is not committed to this repository and lives on the Princeton HPC scratch filesystem only.

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Volume shape | 96×112×96 |
| Latent channels | 3 |
| Latent scale | 8× |
| Diffusion steps (T) | 1000 |
| Beta schedule | Linear (1e-4 → 0.02) |
| DDIM inference steps | 50 |
| Batch size | 8 |
| AE epochs | 100 |
| Diffusion epochs | 2000 (+ early stopping, patience 500) |
| Learning rate | 1e-4 |
| CLIP model | ViT-B/32 (frozen) |
