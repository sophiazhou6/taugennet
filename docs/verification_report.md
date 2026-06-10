# TauGenNet — Paper vs. Code Verification Report

**Paper**: "Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D Diffusion Models"  
DOI: 10.1109/TRPMS.2026.3688162  
Audited: 2026-06-09  
Modules checked: `src/config.py`, `src/models.py`, `src/diffusion.py`, `src/conditioning.py`, `src/dataset.py`, `src/inference.py`, `src/train.py` (via `scripts/train.py`), `scripts/evaluate.py`

---

## Legend
| Symbol | Meaning |
|--------|---------|
| ✅ | Code matches paper |
| ⚠️ | Known intentional deviation (documented) |
| 🔴 | Bug or unintentional mismatch — fix recommended |

---

## 1. Input / Volume Shape

| Claim | Paper | Code | Status |
|-------|-------|------|--------|
| Training volume shape | 160×160×96 | `VOL_SHAPE=(96,112,96)` (`config.py:21`) | ⚠️ Intentional: user's ADNI data is 91×109×91 @ 2 mm; 96×112×96 is the correct native shape |
| Latent spatial dims | 20×20×12 | 12×14×12 (`LAT_H/W/D`, `config.py:25-27`) | ⚠️ Intentional: follows from 96×112×96 ÷ 8 |
| Latent downscale factor | 8× | `LATENT_SCALE=8` (`config.py:23`) | ✅ |

---

## 2. Autoencoder Architecture

| Claim | Paper | Code | Status |
|-------|-------|------|--------|
| Latent channels | 3 | `LATENT_CH=3` (`config.py:22`) | ✅ |
| Channel stages | [64, 128, 128, 128] — 4 stages, 3 stride-2 | `ch_mult=(64,128,128)` (`models.py:33`) — 3 stride-2 + 1 final ResBlock stage | ✅ (4th stage = final ResBlock before projection) |
| Spatial downscale ops | 3× stride-2 → 8× total | `n_down=int(log2(8))=3` (`models.py:35`) | ✅ |
| GroupNorm groups | 32 | `GROUPNORM_GROUPS=32` (`config.py:33`); used everywhere with `eps=1e-6` | ✅ |
| GroupNorm eps | 1e-6 | `eps=1e-6` throughout `models.py` | ✅ |
| Activation | ReLU | `nn.SiLU()` (`models.py:21,23,49,77`) | ⚠️ Intentional: SiLU kept; paper specifies ReLU |
| Attention in AE | None | None — only `ResBlock3D`, `Conv3d`, `ConvTranspose3d` | ✅ |
| Skip connection in ResBlock | Yes | `self.skip1 = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else Identity()` | ✅ |

---

## 3. Denoising U-Net Architecture

| Claim | Paper | Code | Status |
|-------|-------|------|--------|
| Resolution levels | 3 | 3 encoder blocks (enc1/enc2/enc3) + mid | ✅ |
| Channel widths | [256, 512, 768] | `ch_list=(256,512,768)` (`models.py:308`) | ✅ |
| ResBlocks per level | 2 | 2 per `UNetBlock3D` (norm1/conv1+conv2 + norm3/conv3+conv4) (`models.py:239-258`) | ✅ |
| TransformerBlocks per level | 6 (3 after each ResBlock) | `n_transformer=3` twice per `UNetBlock3D` → 6 total (`models.py:246,256`) | ✅ |
| TransformerBlock structure | SA + CA + FFN | `TransformerBlock3D` = `SelfAttention3D` + `CrossAttention3D` + `FeedForward3D` (`models.py:194-206`) | ✅ |
| Attention at all levels | Yes | All enc1–enc3, mid, dec1–dec3 use `UNetBlock3D` with transformers | ✅ |
| Input channels | 6 (z_t:3 + z_m:3) | `in_ch=LATENT_CH*2=6` (`models.py:308`) | ✅ |
| Output channels | 3 | `nn.Conv3d(c1, LATENT_CH, 1)` → 3 (`models.py:341`) | ✅ |
| GroupNorm / eps | 32 / 1e-6 | Same as AE (`models.py:239-254`) | ✅ |

### ⚠️ Architectural Concern: UNet Bottleneck Resolution

The 3-level UNet downsamples the latent volume three times. For the user's 12×14×12 latent, each stride-2 Conv(k=4,p=1) halves (rounding down for odd dims):

| After level | User's latents (12×14×12) | Paper's latents (20×20×12) |
|-------------|--------------------------|---------------------------|
| enc1 | 6×7×6 | 10×10×6 |
| enc2 | 3×4×3 | 5×5×3 |
| enc3 (bottleneck) | **1×2×1 = 2 tokens** | 2×3×1 = 6 tokens |

The user's bottleneck collapses to **2 spatial tokens** — 3× fewer than the paper. This does not prevent convergence (the code handles it via `F.interpolate` in the upsample chain at `models.py:357-359`) but is a capacity bottleneck worth monitoring.

**Recommended fix**: if performance is limited, reduce to 2 UNet downsampling levels by removing `enc3`'s `downsample_ch` argument, giving a 3×4×3 = 36-token bottleneck.

---

## 4. Diffusion Schedule

| Claim | Paper | Code | Status |
|-------|-------|------|--------|
| Schedule type | Linear | `torch.linspace(beta_start, beta_end, T)` (`diffusion.py:19`) | ✅ |
| T steps | 1000 | `T_STEPS=1000` (`config.py:30`) | ✅ |
| β_start | 0.0015 | `BETA_START=0.0015` (`config.py:31`) | ✅ |
| β_end | 0.0205 | `BETA_END=0.0205` (`config.py:32`) | ✅ |
| Training loss | MSE on ε-prediction | `F.mse_loss(eps_hat, eps)` (`train.py:219`) | ✅ |
| Inference sampler | DDPM, 500 steps | Default in `synthesize_tau_pet`: **`sampler='ddim'`, `n_steps=50`** (`inference.py:69`) | ⚠️ Default misleads; full eval correctly uses `sampler='ddpm', n_steps=500` in `evaluate.py:87` |

---

## 5. Conditioning

| Claim | Paper | Code | Status |
|-------|-------|------|--------|
| CLIP model | ViT-B/32, frozen | `CLIPTextModel.from_pretrained(...)`, all params frozen (`conditioning.py:42-44`) | ✅ |
| CLIP output dim | 512 | `COND_DIM=512`, assertion at `conditioning.py:45` | ✅ |
| Prompt format | `"Plasma is X.XXX."` | `f"Plasma is {v.item():.3f}."` (`conditioning.py:57`) | ✅ |
| Conditioning input to UNet | Cross-attention at every level | `ctx` passed to every `UNetBlock3D.forward` → every `TransformerBlock3D` | ✅ |

---

## 6. Training

| Claim | Paper | Code | Status |
|-------|-------|------|--------|
| Batch size | 8 | `BATCH_SIZE=8` (`config.py:37`) | ✅ |
| Learning rate | 1e-4 | `LR=1e-4` (`config.py:39`) | ✅ |
| Diffusion epochs | 500–600 | `DIFF_EPOCHS=2000` + early stopping (`config.py:36`) | ⚠️ Intentional: early stopping with `patience=100` terminates similarly |
| Optimizer | Adam | `torch.optim.Adam` (`train.py:188`) | ✅ |
| Gradient clipping | Not specified | `clip_grad_norm_(unet.parameters(), 1.0)` (`train.py:223`) | ✅ (common practice) |
| Mixed precision | Not specified | AE training uses `autocast`+`GradScaler` (`train.py:91`); diffusion loop does **not** | ⚠️ Inconsistency: diffusion training lacks AMP — slows training significantly at 256-512-768ch |

---

## 7. Dataset

| Claim | Paper | Code | Status |
|-------|-------|------|--------|
| Total subjects | 360 (357 usable) | 183 (ptau217 matched) / 249 (atrophy matched) | ⚠️ Different cohort subset; see CLAUDE.md |
| Split | 247 / 10 / 100 | 72% train+val / 28% test (`dataset.py:234`) | ⚠️ Proportional split on smaller dataset |
| Volume normalisation | [0, 1] per-subject | `(vol - vmin) / (vmax - vmin + 1e-8)` (`dataset.py:127`) | ✅ |
| Resampling | trilinear to VOL_SHAPE | `F.interpolate(..., mode='trilinear')` (`dataset.py:121-123`) | ✅ |
| Brain mask | DK atlas (labels 1–86) | `T1_seg_in_MNI.nii.gz`, labels 1–86 (`dataset.py:111`) | ✅ |

---

## 8. Evaluation

| Claim | Paper | Code | Status |
|-------|-------|------|--------|
| Metric: 3D SSIM | Yes | `skimage.metrics.structural_similarity` with `data_range=1.0` (`evaluate.py:94,150`) | ✅ |
| Metric: NRMSE | Yes — `√MSE / mean(real)` | **`abs(mean_real - mean_gen) / mean_real`** (`evaluate.py:204`) | 🔴 **BUG**: computes normalised mean bias, not NRMSE |
| Metric: MSE | Yes | `np.mean((gen-real)**2)` (`evaluate.py:201`) | ✅ |
| ROIs | Entorhinal, Parahippocampal, Hippocampus, Fusiform, Inferior Temporal, Post. Cingulate | All 6 present in `ROI_DEFS` (`evaluate.py:18-25`) | ✅ (names match) |
| ROI extraction | Atlas-based masks | **Approximate bounding boxes as fractions of VOL_SHAPE** (`evaluate.py:32-34`) | ⚠️ Approximation — not atlas-derived voxel masks |
| Plasma bin stratification | NRMSE per bin × ROI | **Skipped** — bins defined but binning commented out (`evaluate.py:220-221`) | ⚠️ Intentional: dataset updated to atrophy mode; ptau217 values needed for stratification |

---

## 9. Bug Summary

### 🔴 BUG 1 — `inference.py:52`: hardcoded `4` in `latent_std` fallback

```python
# CURRENT (broken for old checkpoints):
latent_std = ckpt.get("latent_std", torch.ones(1, 4, 1, 1, 1)).to(device)
# CORRECT:
latent_std = ckpt.get("latent_std", torch.ones(1, LATENT_CH, 1, 1, 1)).to(device)
```

Shape (1,4,1,1,1) cannot broadcast against (B,3,H,W,D) latents → `RuntimeError` when loading old checkpoints.  
`train.py:425,428` was already corrected; `inference.py:52` was not.

**Fix**: replace `4` with `LATENT_CH` and import it.

---

### 🔴 BUG 2 — `evaluate.py:204`: incorrect NRMSE formula

```python
# CURRENT (normalised mean bias, NOT NRMSE):
nrmses.append(float(abs(real.mean() - gen.mean()) / (real.mean() + 1e-8)))

# CORRECT (paper: NRMSE = √MSE / mean_real):
nrmses.append(float(np.sqrt(np.mean((gen - real)**2)) / (real.mean() + 1e-8)))
```

The paper reports NRMSE (root-mean-square error normalised by the mean SUVR).  
The current formula reports the normalised mean absolute difference in regional means — a bias metric, not an error metric.

---

## 10. Summary Table

| # | Check | Status | File:Line |
|---|-------|--------|-----------|
| 1 | VOL_SHAPE = 96×112×96 | ⚠️ Intentional adaptation | `config.py:21` |
| 2 | LATENT_CH = 3 | ✅ | `config.py:22` |
| 3 | LATENT_SCALE = 8 | ✅ | `config.py:23` |
| 4 | AE channels [64,128,128]+final | ✅ | `models.py:33` |
| 5 | No attention in AE | ✅ | `models.py:15-81` |
| 6 | GroupNorm 32 groups, eps=1e-6 | ✅ | `config.py:33`; `models.py:21` |
| 7 | Activation SiLU (paper: ReLU) | ⚠️ Intentional | `models.py:21` |
| 8 | UNet channels [256,512,768] | ✅ | `models.py:308` |
| 9 | 2 ResBlocks per UNet level | ✅ | `models.py:234-258` |
| 10 | 6 TransformerBlocks per level | ✅ | `models.py:246,256` |
| 11 | SA+CA+FFN per TransformerBlock | ✅ | `models.py:194-206` |
| 12 | Attention at all UNet levels | ✅ | `models.py:317-338` |
| 13 | UNet in=6ch, out=3ch | ✅ | `models.py:308,341` |
| 14 | T=1000 linear schedule | ✅ | `config.py:30-32`; `diffusion.py:19` |
| 15 | β_start=0.0015, β_end=0.0205 | ✅ | `config.py:31-32` |
| 16 | MSE ε-prediction loss | ✅ | `train.py:219` |
| 17 | DDPM 500-step inference | ⚠️ Default is DDIM/50; eval script uses DDPM/500 | `inference.py:69`; `evaluate.py:87` |
| 18 | CLIP ViT-B/32, frozen | ✅ | `conditioning.py:40-43` |
| 19 | Prompt "Plasma is X.XXX." | ✅ | `conditioning.py:57` |
| 20 | COND_DIM = 512 | ✅ | `config.py:52` |
| 21 | Batch size 8 | ✅ | `config.py:37` |
| 22 | LR 1e-4, Adam | ✅ | `config.py:39`; `train.py:188` |
| 23 | Per-subject [0,1] normalisation | ✅ | `dataset.py:126-127` |
| 24 | DK atlas mask labels 1–86 | ✅ | `dataset.py:111` |
| 25 | 6 evaluation ROIs | ✅ | `evaluate.py:18-25` |
| 26 | 3D SSIM | ✅ | `evaluate.py:94` |
| 27 | NRMSE formula | 🔴 **BUG** | `evaluate.py:204` |
| 28 | latent_std fallback shape | 🔴 **BUG** | `inference.py:52` |
| 29 | UNet bottleneck 2 tokens vs 6 | ⚠️ Architectural concern | `models.py:319` |
| 30 | AMP for diffusion training | ⚠️ Missing | `train.py:208-226` |

---

## 11. Recommended Fixes (priority order)

1. **`inference.py:52`** — replace `4` with `LATENT_CH` (import it from config); prevents crash on legacy checkpoints.
2. **`evaluate.py:204`** — fix NRMSE: `np.sqrt(np.mean((gen-real)**2)) / (real.mean()+1e-8)`.
3. **`inference.py:69`** — change default `sampler='ddpm'`, `n_steps=500` so that calling `synthesize_tau_pet` without explicit args reproduces the paper.
4. **`train.py:208-226`** — wrap diffusion forward+backward in `autocast('cuda')` for 2-3× speedup (safe with `GradScaler` already in scope for AE training; just extend pattern to diffusion loop).
5. **`evaluate.py`** — re-enable plasma bin stratification when running `ptau217` mode to reproduce paper's Table II exactly.
