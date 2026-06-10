"""
scripts/roi_table.py

Per-ROI NRMSE and 3D SSIM for TauGenNet SiLU runs, plus per-subject box plots.

Outputs:
  - TABLE III: NRMSE and 3D SSIM across 6 brain regions, 3 conditioning modes
  - Box plot: per-subject whole-brain NRMSE and SSIM across modes
    saved to results/figures/silu/boxplot_nrmse_ssim.png
"""

import os, sys
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, VOL_SHAPE
from src import dataset_v2       as _dataset_v2
from src import dataset_combined as _dataset_combined
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet

# ── Bilateral DK atlas label IDs for each ROI ─────────────────────────────────
ROI_LABELS = {
    "Entorhinal":      [5,  39],
    "Parahippocampal": [15, 49],
    "Hippocampus":     [74, 83],
    "Fusiform":        [6,  40],
    "Inf. Temporal":   [8,  42],
    "Post. Cingulate": [22, 56],
}
ROI_ORDER = ["Entorhinal", "Parahippocampal", "Hippocampus",
             "Fusiform", "Inf. Temporal", "Post. Cingulate"]

CONFIGS = [
    dict(mode="atrophy",  label="Atrophy",
         ckpt_dir="results/checkpoints/silu/atrophy",  dataset="v2"),
    dict(mode="ptau217",  label="p-tau217",
         ckpt_dir="results/checkpoints/silu/ptau217",  dataset="v2"),
    dict(mode="combined", label="Combined",
         ckpt_dir="results/checkpoints/silu/combined", dataset="combined"),
]

BOXPLOT_OUT = "results/figures/silu/boxplot_nrmse_ssim.png"


# ── ROI helpers ───────────────────────────────────────────────────────────────

def load_roi_mask(mri_path, label_ids, vol_shape):
    seg_path = os.path.join(os.path.dirname(mri_path), "T1_seg_in_MNI.nii.gz")
    seg  = nib.load(seg_path).get_fdata().astype(np.uint8)
    mask = np.zeros(seg.shape, dtype=bool)
    for lid in label_ids:
        mask |= (seg == lid)
    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_t = F.interpolate(mask_t, size=vol_shape, mode="nearest")
    return mask_t.squeeze().numpy().astype(bool)


def roi_nrmse(real, gen, mask):
    r, g = real[mask], gen[mask]
    return float(np.sqrt(np.mean((g - r) ** 2)) / (r.mean() + 1e-8))


def roi_ssim(real, gen, mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return float("nan")
    mn, mx = coords.min(0), coords.max(0)
    r = real[mn[0]:mx[0]+1, mn[1]:mx[1]+1, mn[2]:mx[2]+1]
    g = gen [mn[0]:mx[0]+1, mn[1]:mx[1]+1, mn[2]:mx[2]+1]
    min_dim  = min(r.shape)
    win_size = min(7, min_dim) if min(min_dim, 7) % 2 == 1 else min(7, min_dim) - 1
    win_size = max(win_size, 3)
    return float(ssim3d(r, g, data_range=1.0, win_size=win_size))


def whole_brain_nrmse(real, gen):
    return float(np.sqrt(np.mean((gen - real) ** 2)) / (real.mean() + 1e-8))


def whole_brain_ssim(real, gen):
    return float(ssim3d(real, gen, data_range=1.0))


# ── Per-mode evaluation ───────────────────────────────────────────────────────

def evaluate_mode(cfg):
    print(f"\n{'='*64}")
    print(f"  {cfg['label']}")
    print(f"{'='*64}")

    ckpt_path = os.path.join(cfg["ckpt_dir"], f"diff_{cfg['mode']}_best.pt")
    ae, unet, conditioner, latent_std, _ = load_models(
        ckpt_path, cfg["mode"], device=DEVICE, arch="silu"
    )
    schedule    = DiffusionSchedule(device=DEVICE)
    encode_cond = conditioner.encode

    if cfg["dataset"] == "combined":
        _, _, test_ds, _, _, _ = _dataset_combined.build_dataloaders(
            mode="combined", batch_size=1, use_mask=True
        )
    else:
        _, _, test_ds, _, _, _ = _dataset_v2.build_dataloaders(
            mode=cfg["mode"], batch_size=1, use_dk_mask=True
        )

    print(f"Test subjects: {len(test_ds)}")

    nrmse_roi = {r: [] for r in ROI_ORDER}
    ssim_roi  = {r: [] for r in ROI_ORDER}
    wb_nrmses, wb_ssims = [], []

    for i in range(len(test_ds)):
        pet, mri, cond = test_ds[i]
        real_np = pet.squeeze().numpy()
        gen_np  = synthesize_tau_pet(
            mri.unsqueeze(0), cond, ae, unet, schedule, encode_cond,
            latent_std, device=DEVICE, n_steps=500, sampler="ddpm"
        ).squeeze().numpy()

        # whole-brain metrics for box plot
        wb_nrmses.append(whole_brain_nrmse(real_np, gen_np))
        wb_ssims.append(whole_brain_ssim(real_np, gen_np))

        # per-ROI metrics for table
        mri_path = test_ds.mri_paths[i]
        for rname in ROI_ORDER:
            mask = load_roi_mask(mri_path, ROI_LABELS[rname], VOL_SHAPE)
            nrmse_roi[rname].append(roi_nrmse(real_np, gen_np, mask))
            ssim_roi [rname].append(roi_ssim (real_np, gen_np, mask))

        if (i + 1) % 10 == 0 or (i + 1) == len(test_ds):
            print(f"  subject {i+1}/{len(test_ds)}  "
                  f"wb_nrmse={wb_nrmses[-1]:.4f}  wb_ssim={wb_ssims[-1]:.4f}")

    nrmse_stats = {r: (np.nanmean(nrmse_roi[r]), np.nanstd(nrmse_roi[r])) for r in ROI_ORDER}
    ssim_stats  = {r: (np.nanmean(ssim_roi [r]), np.nanstd(ssim_roi [r])) for r in ROI_ORDER}

    return nrmse_stats, ssim_stats, wb_nrmses, wb_ssims


# ── Box plot ──────────────────────────────────────────────────────────────────

def make_boxplot(all_wb):
    """all_wb: list of (label, nrmse_list, ssim_list)"""
    labels   = [x[0] for x in all_wb]
    nrmse_data = [x[1] for x in all_wb]
    ssim_data  = [x[2] for x in all_wb]

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("TauGenNet (SiLU) — Per-Subject Distribution\nAcross Conditioning Modes",
                 fontsize=13, fontweight="bold")

    for ax, data, ylabel, title in [
        (axes[0], nrmse_data, "NRMSE",  "Whole-Brain NRMSE (lower is better)"),
        (axes[1], ssim_data,  "3D SSIM","Whole-Brain 3D SSIM (higher is better)"),
    ]:
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        # overlay individual points
        for i, (d, color) in enumerate(zip(data, colors), start=1):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
            ax.scatter(i + jitter, d, color=color, alpha=0.5, s=18, zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(BOXPLOT_OUT), exist_ok=True)
    plt.savefig(BOXPLOT_OUT, dpi=150, bbox_inches="tight")
    print(f"\nBox plot saved -> {BOXPLOT_OUT}")


# ── Table printer ─────────────────────────────────────────────────────────────

def print_table(results):
    col_w   = 17
    label_w = 24
    divider = "-" * (label_w + col_w * len(ROI_ORDER))

    print("\n")
    print("=" * (label_w + col_w * len(ROI_ORDER)))
    print("TABLE III")
    print("COMPARISON EXPERIMENT OF NRMSE AND 3D SSIM FOR TAUGENNET")
    print("ACROSS DIFFERENT BRAIN REGIONS")
    print("=" * (label_w + col_w * len(ROI_ORDER)))
    print(f"{'Method':<{label_w}}" + "".join(f"{r:>{col_w}}" for r in ROI_ORDER))
    print(divider)

    for cfg, (nrmse_d, ssim_d, _, _) in results:
        print(f"\n{cfg['label']}")
        nrow = f"  {'NRMSE (mean+/-std)':<{label_w-2}}" + \
               "".join(f"{nrmse_d[r][0]:>{col_w-6}.4f}+/-{nrmse_d[r][1]:.3f} " for r in ROI_ORDER)
        srow = f"  {'SSIM  (mean+/-std)':<{label_w-2}}" + \
               "".join(f"{ssim_d[r][0]:>{col_w-6}.4f}+/-{ssim_d[r][1]:.3f} "  for r in ROI_ORDER)
        print(nrow)
        print(srow)
        print(divider)


if __name__ == "__main__":
    results  = []
    all_wb   = []

    for cfg in CONFIGS:
        nrmse_d, ssim_d, wb_nrmses, wb_ssims = evaluate_mode(cfg)
        results.append((cfg, (nrmse_d, ssim_d, wb_nrmses, wb_ssims)))
        all_wb.append((cfg["label"], wb_nrmses, wb_ssims))

    print_table(results)
    make_boxplot(all_wb)
    print("\nDone.")
