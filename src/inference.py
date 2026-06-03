import torch
from tqdm import tqdm

from .config import DEVICE, T_STEPS, DIFF_CHECKPOINT_PATH, AE_CHECKPOINT_PATH
from .diffusion import DiffusionSchedule
from .models import Autoencoder3D, DenoisingUNet3D
from .conditioning import build_conditioner


def load_models(checkpoint_path, mode, device=DEVICE):
    """Load ae, unet, conditioner, and latent_std from a checkpoint.

    checkpoint_path : path to the diffusion checkpoint
    mode            : 'atrophy' or 'ptau217'
    Returns         : (ae, unet, conditioner, latent_std, diff_losses)

    latent_std defaults to ones if absent (old checkpoints without scaling).
    """
    ae          = Autoencoder3D().to(device)
    unet        = DenoisingUNet3D().to(device)
    conditioner = build_conditioner(mode, device=device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    ae.load_state_dict(ckpt["ae"])
    unet.load_state_dict(ckpt["unet"])
    if ckpt.get("conditioner"):
        conditioner.load_state_dict(ckpt["conditioner"])

    # latent_std: (1, LATENT_CH, 1, 1, 1) — ones for backwards-compat with old ckpts
    latent_std  = ckpt.get("latent_std", torch.ones(1, 4, 1, 1, 1)).to(device)
    diff_losses = ckpt.get("diff_losses", [])
    print(f"Loaded {mode} model: {len(diff_losses)} diffusion epochs")
    return ae, unet, conditioner, latent_std, diff_losses


def _prepare_cond(cond_data, device):
    """Ensure cond_data is a (1, N) tensor on device."""
    if not isinstance(cond_data, torch.Tensor):
        cond_data = torch.tensor(cond_data, dtype=torch.float32)
    if cond_data.dim() == 1:
        cond_data = cond_data.unsqueeze(0)  # (N,) → (1, N)
    return cond_data.to(device)


@torch.no_grad()
def synthesize_tau_pet(mri_vol, cond_data, ae, unet, schedule, encode_cond,
                       latent_std, device=DEVICE, n_steps=50, sampler='ddim'):
    """
    mri_vol    : (1, 1, H, W, D) normalised MRI tensor
    cond_data  : (86,) atrophy z-scores  [atrophy mode]
                 (1,)  p-tau217 value    [ptau217 mode]
    latent_std : (1, 4, 1, 1, 1) per-channel std used to normalise latents
    encode_cond: conditioner.encode callable
    sampler    : 'ddim' (fast, good at 50 steps) or 'ddpm' (best at 500+ steps)
    Returns    : (1, 1, H, W, D) synthesised tau PET
    """
    ae.eval(); unet.eval()
    zm   = ae.encode_mean(mri_vol.to(device)) / latent_std
    c    = encode_cond(_prepare_cond(cond_data, device))
    zt   = torch.randn_like(zm)
    step = T_STEPS // n_steps
    ts   = list(reversed(range(0, T_STEPS, step)))
    for i, t_idx in enumerate(tqdm(ts, desc="Denoising", leave=False)):
        if sampler == 'ddim':
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            zt = schedule.ddim_sample(unet, zt, t_idx, t_prev, zm, c)
        else:
            zt = schedule.p_sample(unet, zt, t_idx, zm, c)
    return ae.decode(zt * latent_std).cpu()


@torch.no_grad()
def synthesize_no_mri(mri_vol, cond_data, ae, unet, schedule, encode_cond,
                      latent_std, device=DEVICE, n_steps=500):
    """Ablation: MRI latent zeroed out — conditioning only, no structural guidance."""
    ae.eval(); unet.eval()
    zm_zero = torch.zeros_like(ae.encode_mean(mri_vol.to(device)))
    c       = encode_cond(_prepare_cond(cond_data, device))
    zt      = torch.randn_like(zm_zero)
    step    = T_STEPS // n_steps
    ts      = list(reversed(range(0, T_STEPS, step)))
    for i, t_idx in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        zt = schedule.ddim_sample(unet, zt, t_idx, t_prev, zm_zero, c)
    return ae.decode(zt * latent_std).cpu()
