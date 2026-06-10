import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from src.config import DEVICE, VOL_SHAPE, FIGURES_DIR
from src.dataset_v2 import build_dataloaders, unnormalize
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet, synthesize_no_mri


# ── ROI definitions (approximate bounding boxes as fractions of VOL_SHAPE) ───
ROI_DEFS = {
    "Parahippocampal":   (0.40, 0.55, 0.35, 0.65, 0.30, 0.55),
    "Fusiform":          (0.45, 0.60, 0.30, 0.70, 0.25, 0.50),
    "Inferior Temporal": (0.30, 0.55, 0.25, 0.75, 0.20, 0.55),
    "Hippocampus":       (0.42, 0.52, 0.40, 0.60, 0.35, 0.50),
    "Post. Cingulate":   (0.40, 0.55, 0.38, 0.62, 0.50, 0.70),
    "Entorhinal":        (0.43, 0.55, 0.38, 0.62, 0.28, 0.45),
}
PLASMA_BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (10, float("inf"))]
BIN_LABELS  = ["0-2", "2-4", "4-6", "6-8", "10+"]

H, W, D = VOL_SHAPE


def roi_mean(vol_np, roi):
    y0, y1, x0, x1, z0, z1 = roi
    return vol_np[int(y0*H):int(y1*H), int(x0*W):int(x1*W), int(z0*D):int(z1*D)].mean()


def plot_loss_curves(ae_losses, diff_losses, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/loss_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ae_losses)
    axes[0].set_title(f"AE Loss (100 epochs, final={ae_losses[-1]:.4f})")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(diff_losses)
    axes[1].set_title(f"Diffusion Loss (600 epochs, final={diff_losses[-1]:.4f})")
    axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Epoch 1 loss:   {diff_losses[0]:.4f}")
    if len(diff_losses) > 99:
        print(f"Epoch 100 loss: {diff_losses[99]:.4f}")
    print(f"Final loss:     {diff_losses[-1]:.4f}")


def plot_ae_reconstruction(ae, test_ds, device=DEVICE, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/ae_recon.png"
    ae.eval()
    with torch.no_grad():
        pet, mri, _ = test_ds[0]
        recon, _, _ = ae(pet.unsqueeze(0).to(device))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    mid = 48
    axes[0].imshow(pet[0, :, :, mid].numpy(), cmap="hot")
    axes[0].set_title("Real PET")
    axes[1].imshow(recon[0, 0, :, :, mid].cpu().numpy(), cmap="hot")
    axes[1].set_title("AE Reconstruction")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()


def compare_generated_vs_real(ae, unet, schedule, encode_cond, latent_std, test_ds,
                               device=DEVICE, n_steps=500, save_path=None, sample_idx=0):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/taugennet_comparison.png"
    sample_pet, sample_mri, sample_atrophy = test_ds[sample_idx]
    mri_input = sample_mri.unsqueeze(0)
    real_pet  = sample_pet.squeeze().numpy()

    print("Synthesising tau PET...")
    generated = synthesize_tau_pet(mri_input, sample_atrophy, ae, unet, schedule,
                                   encode_cond, latent_std, device=device,
                                   n_steps=n_steps, sampler='ddpm')
    gen_pet = generated.squeeze().numpy()

    # ── Sanity check: generated image should not be re-normalized ────────────
    print(f"Generated PET — min: {gen_pet.min():.4f}  max: {gen_pet.max():.4f}  "
          f"(expected ~[0, 1] for normalized output)")
    if gen_pet.min() < -0.1 or gen_pet.max() > 1.1:
        print(f"WARNING: generated values outside expected [-0.1, 1.1] range. "
              f"Check that no extra normalization step is applied post-decode.")

    # ── Unnormalize to real SUVR space for metrics ────────────────────────────
    pet_min, pet_max = test_ds.get_pet_norms(sample_idx)
    real_suvr = unnormalize(real_pet, pet_min, pet_max)
    gen_suvr  = unnormalize(gen_pet,  pet_min, pet_max)
    suvr_range = pet_max - pet_min

    # ── Quantitative metrics in SUVR space ────────────────────────────────────
    mae  = np.mean(np.abs(gen_suvr - real_suvr))
    mse  = np.mean((gen_suvr - real_suvr) ** 2)
    psnr = 10 * np.log10(suvr_range ** 2 / mse) if mse > 0 else float('inf')
    ssim_score = ssim(real_suvr, gen_suvr, data_range=suvr_range)
    print(f"MAE  (SUVR): {mae:.4f}")
    print(f"MSE  (SUVR²): {mse:.6f}")
    print(f"PSNR (dB):  {psnr:.2f}")
    print(f"SSIM:        {ssim_score:.4f}")

    # ── Visual comparison ─────────────────────────────────────────────────────
    views = {"Axial": 2, "Sagittal": 0, "Coronal": 1}
    fig, axes = plt.subplots(len(views), 3, figsize=(15, 4 * len(views)))

    for row_idx, (view_name, axis_dim) in enumerate(views.items()):
        mid       = real_suvr.shape[axis_dim] // 2
        real_slc  = np.take(real_suvr, mid, axis=axis_dim)
        gen_slc   = np.take(gen_suvr,  mid, axis=axis_dim)
        diff_slc  = np.abs(real_slc - gen_slc)

        # Shared intensity scale anchored on the real image
        vmin_pet  = real_slc.min()
        vmax_pet  = real_slc.max()

        for col_idx, (slc, title, cmap, vmin, vmax, cbar_label) in enumerate([
            (real_slc, "Real PET",       "hot",      vmin_pet, vmax_pet, "SUVR"),
            (gen_slc,  "Generated PET",  "hot",      vmin_pet, vmax_pet, "SUVR"),
            (diff_slc, "Abs Difference", "coolwarm", 0,        diff_slc.max(), "SUVR"),
        ]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
            ax.set_xlabel("x (vox)")
            ax.set_ylabel(f"{view_name}\ny (vox)" if col_idx == 0 else "y (vox)")
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")

    plt.suptitle(
        f"Real vs Generated Tau PET (SUVR space)\n"
        f"MAE={mae:.4f} SUVR  |  MSE={mse:.6f} SUVR²  |  SSIM={ssim_score:.4f}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return mae, mse, psnr, ssim_score


def run_steps_comparison(ae, unet, schedule, encode_cond, latent_std, test_ds,
                         device=DEVICE, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/steps_comparison.png"
    ae.eval(); unet.eval()
    pet, mri, atrophy = test_ds[0]

    results = {}
    for n_steps in [1, 5, 10, 50]:
        gen = synthesize_tau_pet(mri.unsqueeze(0), atrophy, ae, unet, schedule,
                                 encode_cond, latent_std, device=device, n_steps=n_steps)
        results[n_steps] = gen.squeeze().numpy()

    # Unnormalize for display
    pet_min, pet_max = test_ds.get_pet_norms(0)
    pet_suvr = unnormalize(pet[0].numpy(), pet_min, pet_max)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
    mid = 48
    im0 = axes[0].imshow(pet_suvr[:, :, mid], cmap="hot")
    axes[0].set_title("Real PET")
    axes[0].set_xlabel("x (vox)"); axes[0].set_ylabel("y (vox)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="SUVR")
    for idx, n_steps in enumerate([1, 5, 10, 50]):
        gen_suvr = unnormalize(results[n_steps], pet_min, pet_max)
        im = axes[idx+1].imshow(gen_suvr[:, :, mid], cmap="hot",
                                vmin=pet_suvr[:, :, mid].min(),
                                vmax=pet_suvr[:, :, mid].max())
        axes[idx+1].set_title(f"{n_steps} steps")
        axes[idx+1].set_xlabel("x (vox)"); axes[idx+1].set_ylabel("y (vox)")
        plt.colorbar(im, ax=axes[idx+1], fraction=0.046, pad=0.04, label="SUVR")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()


def run_mri_ablation_visual(ae, unet, schedule, encode_cond, latent_std, test_ds,
                             device=DEVICE, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/mri_ablation.png"
    """Generate with real MRI vs zeroed MRI to visualise structural guidance."""
    pet, mri, atrophy = test_ds[0]

    gen_normal = synthesize_tau_pet(mri.unsqueeze(0), atrophy, ae, unet, schedule,
                                    encode_cond, latent_std, device=device,
                                    n_steps=500, sampler='ddpm').squeeze().numpy()

    mri_zero   = torch.zeros_like(mri)
    gen_no_mri = synthesize_no_mri(mri_zero.unsqueeze(0), atrophy, ae, unet, schedule,
                                   encode_cond, latent_std, device=device,
                                   n_steps=500).squeeze().numpy()

    pet_min, pet_max = test_ds.get_pet_norms(0)
    pet_suvr        = unnormalize(pet[0].numpy(), pet_min, pet_max)
    gen_normal_suvr = unnormalize(gen_normal,     pet_min, pet_max)
    gen_nomri_suvr  = unnormalize(gen_no_mri,     pet_min, pet_max)

    mid    = 48
    vmin_s = pet_suvr[:, :, mid].min()
    vmax_s = pet_suvr[:, :, mid].max()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, slc, title in [
        (axes[0], pet_suvr[:, :, mid],        "Real PET"),
        (axes[1], gen_normal_suvr[:, :, mid],  "Generated (real MRI)"),
        (axes[2], gen_nomri_suvr[:, :, mid],   "Generated (zero MRI)"),
    ]:
        im = ax.imshow(slc, cmap="hot", vmin=vmin_s, vmax=vmax_s)
        ax.set_title(title)
        ax.set_xlabel("x (vox)"); ax.set_ylabel("y (vox)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="SUVR")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()

    diff = np.mean(np.abs(gen_normal_suvr - gen_nomri_suvr))
    print(f"Mean difference with/without MRI: {diff:.4f} SUVR")


def run_multi_subject_metrics(ae, unet, schedule, encode_cond, latent_std, test_ds,
                               device=DEVICE, n_subjects=10, n_steps=500):
    """Compute MAE/MSE/SSIM over multiple test subjects in SUVR space."""
    maes, mses, ssims = [], [], []
    for i in range(min(n_subjects, len(test_ds))):
        pet, mri, atrophy = test_ds[i]
        real_norm = pet.squeeze().numpy()
        gen_norm  = synthesize_tau_pet(mri.unsqueeze(0), atrophy, ae, unet, schedule,
                                       encode_cond, latent_std, device=device,
                                       n_steps=n_steps, sampler='ddpm').squeeze().numpy()

        pet_min, pet_max = test_ds.get_pet_norms(i)
        real = unnormalize(real_norm, pet_min, pet_max)
        gen  = unnormalize(gen_norm,  pet_min, pet_max)
        suvr_range = pet_max - pet_min

        maes.append(np.mean(np.abs(gen - real)))
        mses.append(np.mean((gen - real) ** 2))
        ssims.append(ssim(real, gen, data_range=suvr_range))

    print(f"Mean MAE  (SUVR):  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"Mean MSE  (SUVR²): {np.mean(mses):.6f} ± {np.std(mses):.6f}")
    print(f"Mean SSIM:         {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    return maes, mses, ssims


def run_roi_mse_eval(ae, unet, schedule, encode_cond, latent_std, test_ds, test_loader,
                     device=DEVICE, n_steps=20):
    """Region-wise MSE across all test subjects in SUVR space (cf. Table II).

    NOTE: The original notebook stratified by scalar p-tau217 plasma bins, but
    the dataset was later updated to return 86-dim atrophy vectors instead of a
    scalar. Binning is therefore skipped here; all subjects are pooled.
    """
    rec_real = {r: [] for r in ROI_DEFS}
    rec_gen  = {r: [] for r in ROI_DEFS}

    print("Running evaluation on test set...")
    subject_idx = 0
    for pet, mri, ptau in tqdm(test_loader):
        for i in range(pet.shape[0]):
            mri_i    = mri[i:i+1]
            pet_norm = pet[i].squeeze().numpy()
            gen_norm = synthesize_tau_pet(mri_i, ptau[i], ae, unet, schedule,
                                          encode_cond, latent_std, device=device,
                                          n_steps=n_steps).squeeze().numpy()

            pet_min, pet_max = test_ds.get_pet_norms(subject_idx)
            pet_np = unnormalize(pet_norm, pet_min, pet_max)
            gen_np = unnormalize(gen_norm, pet_min, pet_max)
            subject_idx += 1

            for rname, roi in ROI_DEFS.items():
                rec_real[rname].append(roi_mean(pet_np, roi))
                rec_gen[rname].append(roi_mean(gen_np, roi))

    mse_table = pd.DataFrame(index=["All subjects"], columns=list(ROI_DEFS.keys()), dtype=float)
    for rname in ROI_DEFS:
        r = rec_real[rname]
        g = rec_gen[rname]
        mse_table.loc["All subjects", rname] = round((np.mean(r) - np.mean(g)) ** 2, 6) if r and g else float("nan")

    print("\nRegion-wise MSE in SUVR² (pooled across all test subjects, cf. Table II):")
    print(mse_table.to_string())
    return mse_table


def run_ablation_study(ae, unet, schedule, encode_cond, latent_std, test_ds, test_loader,
                       device=DEVICE, n_steps=20):
    """MRI+Atrophy vs Atrophy-only ablation in SUVR space (cf. Table I)."""
    results = {k: {r: [] for r in ROI_DEFS} for k in ["Real", "MRI+Atrophy", "Atrophy-only"]}

    print("Running ablation study...")
    subject_idx = 0
    for pet, mri, ptau in tqdm(test_loader):
        for i in range(pet.shape[0]):
            mri_i    = mri[i:i+1]
            pet_norm = pet[i].squeeze().numpy()
            gen_full_norm  = synthesize_tau_pet(mri_i, ptau[i], ae, unet, schedule,
                                                encode_cond, latent_std, device=device,
                                                n_steps=n_steps).squeeze().numpy()
            gen_ablat_norm = synthesize_no_mri(mri_i, ptau[i], ae, unet, schedule,
                                               encode_cond, latent_std, device=device,
                                               n_steps=n_steps).squeeze().numpy()

            pet_min, pet_max = test_ds.get_pet_norms(subject_idx)
            pet_np       = unnormalize(pet_norm,       pet_min, pet_max)
            gen_full_np  = unnormalize(gen_full_norm,  pet_min, pet_max)
            gen_ablat_np = unnormalize(gen_ablat_norm, pet_min, pet_max)
            subject_idx += 1

            for rname, roi in ROI_DEFS.items():
                results["Real"][rname].append(roi_mean(pet_np, roi))
                results["MRI+Atrophy"][rname].append(roi_mean(gen_full_np, roi))
                results["Atrophy-only"][rname].append(roi_mean(gen_ablat_np, roi))

    ablation = pd.DataFrame(index=["Atrophy-only", "MRI+Atrophy"],
                            columns=list(ROI_DEFS.keys()), dtype=float)
    for cond in ["Atrophy-only", "MRI+Atrophy"]:
        for rname in ROI_DEFS:
            mse = (np.mean(results["Real"][rname]) - np.mean(results[cond][rname])) ** 2
            ablation.loc[cond, rname] = round(mse, 6)

    print("\nAblation MSE in SUVR² (group-level) – cf. Table I:")
    print(ablation.to_string())
    return ablation


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["atrophy", "ptau217"], default="atrophy")
    args = p.parse_args()

    COND_MODE = args.mode
    CKPT_PATH = f"results/checkpoints/{COND_MODE}/diff_{COND_MODE}.pt"

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(mode=COND_MODE)
    ae, unet, conditioner, latent_std, diff_losses = load_models(CKPT_PATH, COND_MODE)
    encode_cond = conditioner.encode
    schedule    = DiffusionSchedule()
    print(f"latent_std per channel: {latent_std.squeeze().tolist()}")

    # Loss curves
    print(f"Final diffusion loss: {diff_losses[-1]:.6f}")
    plt.figure(figsize=(7, 3))
    plt.plot(diff_losses); plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"Diffusion Loss ({COND_MODE} mode)"); plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{FIGURES_DIR}/diff_loss_{COND_MODE}.png", dpi=150)
    plt.show()

    # AE reconstruction quality
    plot_ae_reconstruction(ae, test_ds)

    # Real vs generated comparison (3-view, SUVR metrics)
    compare_generated_vs_real(ae, unet, schedule, encode_cond, latent_std, test_ds,
                              save_path=f"{FIGURES_DIR}/comparison_{COND_MODE}.png")

    # Denoising steps comparison
    run_steps_comparison(ae, unet, schedule, encode_cond, latent_std, test_ds,
                         save_path=f"{FIGURES_DIR}/steps_{COND_MODE}.png")

    # MRI ablation visualisation
    run_mri_ablation_visual(ae, unet, schedule, encode_cond, latent_std, test_ds,
                            save_path=f"{FIGURES_DIR}/mri_ablation_{COND_MODE}.png")

    # Multi-subject metrics (SUVR)
    run_multi_subject_metrics(ae, unet, schedule, encode_cond, latent_std, test_ds)

    # ROI MSE table (Table II) — now in SUVR²
    run_roi_mse_eval(ae, unet, schedule, encode_cond, latent_std, test_ds, test_loader)

    # Ablation study (Table I) — now in SUVR²
    run_ablation_study(ae, unet, schedule, encode_cond, latent_std, test_ds, test_loader)
