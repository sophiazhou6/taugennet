import os, time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.config import (
    DEVICE, T_STEPS, DIFF_EPOCHS, LR,
    AE_CHECKPOINT_PATH, DIFF_CHECKPOINT_PATH, CLIP_MODEL_PATH,
)
from src.dataset import build_dataloaders
from src.diffusion import DiffusionSchedule
from src.models import (
    Autoencoder3D, DenoisingUNet3D,
    load_clip_encoder, make_encode_atrophy, make_encode_ptau,
)

# Conditioning mode: 'atrophy' uses 86-dim z-scores, 'ptau217' uses scalar plasma value.
# Both are encoded into text embeddings via the frozen CLIP model.
COND_MODE = "atrophy"


def train_diffusion(ae, unet, schedule, encode_atrophy, train_loader,
                    diff_epochs=DIFF_EPOCHS, lr=LR, device=DEVICE,
                    checkpoint_path=DIFF_CHECKPOINT_PATH,
                    ae_losses=None, start_epoch=0):
    diff_opt       = torch.optim.Adam(unet.parameters(), lr=lr)
    diff_losses    = []
    ae.eval()
    training_start = time.time()

    print("=== Diffusion Model Training ===", flush=True)
    for epoch in range(start_epoch, diff_epochs):
        epoch_start = time.time()
        unet.train()
        epoch_loss = 0

        for pet, mri, ptau in train_loader:
            pet, mri, ptau = pet.to(device), mri.to(device), ptau.to(device)
            with torch.no_grad():
                z0 = ae.encode_mean(pet)
                zm = ae.encode_mean(mri)
                c  = encode_atrophy(ptau)  # CLIP frozen, inside no_grad

            B = pet.shape[0]
            t = torch.randint(0, T_STEPS, (B,), device=device)  # uniform, paper Eq. 14

            zt, eps = schedule.q_sample(z0, t)
            ht      = torch.cat([zt, zm], dim=1)
            eps_hat = unet(ht, t, c)
            loss    = F.mse_loss(eps_hat, eps)

            diff_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            diff_opt.step()
            epoch_loss += loss.item()

        avg        = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - training_start
        remaining  = epoch_time * (diff_epochs - epoch - 1)
        diff_losses.append(avg)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{diff_epochs}  loss={avg:.4f}  "
                  f"epoch_time={epoch_time:.1f}s  "
                  f"elapsed={elapsed/60:.1f}m  "
                  f"remaining={remaining/60:.1f}m", flush=True)
            torch.save({
                "ae":          ae.state_dict(),
                "unet":        unet.state_dict(),
                "ae_losses":   ae_losses or [],
                "diff_losses": diff_losses,
                "epoch":       len(diff_losses),
            }, checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}", flush=True)

    total_time = time.time() - training_start
    plt.figure(figsize=(7, 3))
    plt.plot(diff_losses); plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Diffusion Training Loss"); plt.tight_layout()
    plt.savefig("/scratch/network/sz3962/taugennet/diff_loss.png", dpi=150)
    plt.show()
    print(f"Diffusion training complete. Total time: {total_time/60:.1f} minutes", flush=True)

    return unet, diff_losses


if __name__ == "__main__":
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(mode=COND_MODE)

    clip_tokenizer, clip_text_enc, clip_dim = load_clip_encoder(CLIP_MODEL_PATH)
    if COND_MODE == "atrophy":
        encode_cond = make_encode_atrophy(clip_tokenizer, clip_text_enc)
    else:
        encode_cond = make_encode_ptau(clip_tokenizer, clip_text_enc)
    encode_atrophy = encode_cond  # alias used throughout the training function

    ae       = Autoencoder3D().to(DEVICE)
    unet     = DenoisingUNet3D().to(DEVICE)
    schedule = DiffusionSchedule()
    print(f"Denoising U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # Resume from diffusion checkpoint if available; otherwise load pretrained AE
    ae_losses   = []
    start_epoch = 0
    diff_losses = []

    if os.path.exists(DIFF_CHECKPOINT_PATH):
        ckpt = torch.load(DIFF_CHECKPOINT_PATH, map_location=DEVICE)
        ae.load_state_dict(ckpt["ae"])
        unet.load_state_dict(ckpt["unet"])
        ae_losses   = ckpt["ae_losses"]
        diff_losses = ckpt["diff_losses"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}", flush=True)
    elif os.path.exists(AE_CHECKPOINT_PATH):
        ckpt = torch.load(AE_CHECKPOINT_PATH, map_location=DEVICE)
        ae.load_state_dict(ckpt["ae"])
        ae_losses = ckpt["ae_losses"]
        print(f"Loaded pretrained AE ({len(ae_losses)} epochs). Starting diffusion training.")
    else:
        print("Warning: no AE checkpoint found. Train the autoencoder first with train_ae.py")

    # ── U-Net forward-pass sanity check ──────────────────────────────────────
    ae.eval(); unet.eval()
    with torch.no_grad():
        pet, mri, ptau = next(iter(train_loader))
        pet, mri, ptau = pet.to(DEVICE), mri.to(DEVICE), ptau.to(DEVICE)
        z0, _, _ = ae.encode(pet)
        zm, _, _ = ae.encode(mri)
        c = encode_atrophy(ptau)
        t = torch.randint(0, T_STEPS, (pet.shape[0],), device=DEVICE)
        zt, eps = schedule.q_sample(z0, t)
        ht = torch.cat([zt, zm], dim=1)

        t_emb = unet.t_embed(t)
        x  = unet.in_conv(ht)
        e1 = unet.enc1(x, t_emb, c)
        e2 = unet.enc2(unet.down1(e1), t_emb, c)
        m  = unet.mid(unet.down2(e2), t_emb, c)
        up2 = unet.up2(m)
        print(f"e1:  {e1.shape}")
        print(f"e2:  {e2.shape}")
        print(f"m:   {m.shape}")
        print(f"up2 before crop: {up2.shape}")
        up2 = F.interpolate(up2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        print(f"up2 after crop:  {up2.shape}")
        print(f"cat: {torch.cat([up2, e2], dim=1).shape}")

    unet, diff_losses = train_diffusion(
        ae, unet, schedule, encode_atrophy, train_loader,
        ae_losses=ae_losses, start_epoch=start_epoch,
    )

    # Load final checkpoint and print summary
    ckpt = torch.load(DIFF_CHECKPOINT_PATH, map_location=DEVICE)
    ae.load_state_dict(ckpt["ae"])
    unet.load_state_dict(ckpt["unet"])
    ae_losses   = ckpt["ae_losses"]
    diff_losses = ckpt["diff_losses"]
    print(f"Checkpoint loaded. AE epochs: {len(ae_losses)}, Diffusion epochs: {len(diff_losses)}")
