import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from src.config import DEVICE, VOL_SHAPE, FIGURES_DIR
from src.dataset import build_dataloaders
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
                               device=DEVICE, n_steps=500, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/taugennet_comparison.png"
    sample_pet, sample_mri, sample_atrophy = test_ds[0]
    mri_input = sample_mri.unsqueeze(0)
    real_pet  = sample_pet.squeeze().numpy()

    print("Synthesising tau PET...")
    generated = synthesize_tau_pet(mri_input, sample_atrophy, ae, unet, schedule,
                                   encode_cond, latent_std, device=device,
                                   n_steps=n_steps, sampler='ddpm')
    gen_pet = generated.squeeze().numpy()

    # ── Quantitative metrics ──────────────────────────────────────────────────
    mae  = np.mean(np.abs(gen_pet - real_pet))
    mse  = np.mean((gen_pet - real_pet) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    ssim_score = ssim(real_pet, gen_pet, data_range=1.0)
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_score:.4f}")

    # ── Visual comparison ─────────────────────────────────────────────────────
    views = {"Axial": 2, "Sagittal": 0, "Coronal": 1}
    fig, axes = plt.subplots(len(views), 3, figsize=(12, 3 * len(views)))

    for row_idx, (view_name, axis_dim) in enumerate(views.items()):
        mid      = real_pet.shape[axis_dim] // 2
        real_slc = np.take(real_pet, mid, axis=axis_dim)
        gen_slc  = np.take(gen_pet,  mid, axis=axis_dim)
        diff_slc = np.abs(real_slc - gen_slc)

        for col_idx, (slc, title, cmap) in enumerate([
            (real_slc, "Real PET",       "hot"),
            (gen_slc,  "Generated PET",  "hot"),
            (diff_slc, "Abs Difference", "coolwarm"),
        ]):
            ax = axes[row_idx, col_idx]
            ax.imshow(slc, cmap=cmap, vmin=0, vmax=1)
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(view_name, fontsize=10)
            ax.axis("off")

    plt.suptitle(f"Real vs Generated Tau PET\nMAE={mae:.4f}  MSE={mse:.4f}  SSIM={ssim_score:.4f}",
                 fontsize=12, y=1.01)
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

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    mid = 48
    axes[0].imshow(pet[0, :, :, mid].numpy(), cmap="hot")
    axes[0].set_title("Real PET")
    for idx, n_steps in enumerate([1, 5, 10, 50]):
        axes[idx+1].imshow(results[n_steps][:, :, mid], cmap="hot")
        axes[idx+1].set_title(f"{n_steps} steps")
    for ax in axes: ax.axis("off")
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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    mid = 48
    axes[0].imshow(pet[0, :, :, mid].numpy(), cmap="hot")
    axes[0].set_title("Real PET")
    axes[1].imshow(gen_normal[:, :, mid], cmap="hot")
    axes[1].set_title("Generated (real MRI)")
    axes[2].imshow(gen_no_mri[:, :, mid], cmap="hot")
    axes[2].set_title("Generated (zero MRI)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()

    diff = np.mean(np.abs(gen_normal - gen_no_mri))
    print(f"Mean difference with/without MRI: {diff:.4f}")


def run_multi_subject_metrics(ae, unet, schedule, encode_cond, latent_std, test_ds,
                               device=DEVICE, n_subjects=10, n_steps=500):
    """Compute MAE/MSE/SSIM over multiple test subjects."""
    maes, mses, ssims = [], [], []
    for i in range(min(n_subjects, len(test_ds))):
        pet, mri, atrophy = test_ds[i]
        real = pet.squeeze().numpy()
        gen  = synthesize_tau_pet(mri.unsqueeze(0), atrophy, ae, unet, schedule,
                                  encode_cond, latent_std, device=device,
                                  n_steps=n_steps, sampler='ddpm').squeeze().numpy()
        maes.append(np.mean(np.abs(gen - real)))
        mses.append(np.mean((gen - real) ** 2))
        ssims.append(ssim(real, gen, data_range=1.0))

    print(f"Mean MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"Mean MSE:  {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"Mean SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    return maes, mses, ssims


def run_roi_mse_eval(ae, unet, schedule, encode_cond, latent_std, test_loader,
                     device=DEVICE, n_steps=20):
    """Region-wise MSE across all test subjects (reproduces Table II structure).

    NOTE: The original notebook stratified by scalar p-tau217 plasma bins, but
    the dataset was later updated to return 86-dim atrophy vectors instead of a
    scalar. Binning is therefore skipped here; all subjects are pooled.
    """
    rec_real = {r: [] for r in ROI_DEFS}
    rec_gen  = {r: [] for r in ROI_DEFS}

    print("Running evaluation on test set...")
    for pet, mri, ptau in tqdm(test_loader):
        for i in range(pet.shape[0]):
            mri_i  = mri[i:i+1]
            pet_np = pet[i].squeeze().numpy()
            gen_np = synthesize_tau_pet(mri_i, ptau[i], ae, unet, schedule,
                                        encode_cond, latent_std, device=device,
                                        n_steps=n_steps).squeeze().numpy()
            for rname, roi in ROI_DEFS.items():
                rec_real[rname].append(roi_mean(pet_np, roi))
                rec_gen[rname].append(roi_mean(gen_np, roi))

    mse_table = pd.DataFrame(index=["All subjects"], columns=list(ROI_DEFS.keys()), dtype=float)
    for rname in ROI_DEFS:
        r = rec_real[rname]
        g = rec_gen[rname]
        mse_table.loc["All subjects", rname] = round((np.mean(r) - np.mean(g)) ** 2, 6) if r and g else float("nan")

    print("\nRegion-wise MSE (pooled across all test subjects, cf. Table II):")
    print(mse_table.to_string())
    return mse_table


def run_ablation_study(ae, unet, schedule, encode_cond, latent_std, test_loader,
                       device=DEVICE, n_steps=20):
    """MRI+Atrophy vs Atrophy-only ablation (reproduces Table I)."""
    results = {k: {r: [] for r in ROI_DEFS} for k in ["Real", "MRI+Atrophy", "Atrophy-only"]}

    print("Running ablation study...")
    for pet, mri, ptau in tqdm(test_loader):
        for i in range(pet.shape[0]):
            mri_i  = mri[i:i+1]
            pet_np = pet[i].squeeze().numpy()
            gen_full  = synthesize_tau_pet(mri_i, ptau[i], ae, unet, schedule,
                                           encode_cond, latent_std, device=device,
                                           n_steps=n_steps).squeeze().numpy()
            gen_ablat = synthesize_no_mri(mri_i, ptau[i], ae, unet, schedule,
                                          encode_cond, latent_std, device=device,
                                          n_steps=n_steps).squeeze().numpy()
            for rname, roi in ROI_DEFS.items():
                results["Real"][rname].append(roi_mean(pet_np, roi))
                results["MRI+Atrophy"][rname].append(roi_mean(gen_full, roi))
                results["Atrophy-only"][rname].append(roi_mean(gen_ablat, roi))

    ablation = pd.DataFrame(index=["Atrophy-only", "MRI+Atrophy"],
                            columns=list(ROI_DEFS.keys()), dtype=float)
    for cond in ["Atrophy-only", "MRI+Atrophy"]:
        for rname in ROI_DEFS:
            mse = (np.mean(results["Real"][rname]) - np.mean(results[cond][rname])) ** 2
            ablation.loc[cond, rname] = round(mse, 6)

    print("\nAblation MSE (group-level) – cf. Table I:")
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

    # Real vs generated comparison (3-view, metrics)
    compare_generated_vs_real(ae, unet, schedule, encode_cond, latent_std, test_ds,
                              save_path=f"{FIGURES_DIR}/comparison_{COND_MODE}.png")

    # Denoising steps comparison
    run_steps_comparison(ae, unet, schedule, encode_cond, latent_std, test_ds,
                         save_path=f"{FIGURES_DIR}/steps_{COND_MODE}.png")

    # MRI ablation visualisation
    run_mri_ablation_visual(ae, unet, schedule, encode_cond, latent_std, test_ds,
                            save_path=f"{FIGURES_DIR}/mri_ablation_{COND_MODE}.png")

    # Multi-subject metrics
    run_multi_subject_metrics(ae, unet, schedule, encode_cond, latent_std, test_ds)

    # ROI MSE table (Table II)
    run_roi_mse_eval(ae, unet, schedule, encode_cond, latent_std, test_loader)

    # Ablation study (Table I)
    run_ablation_study(ae, unet, schedule, encode_cond, latent_std, test_loader)
