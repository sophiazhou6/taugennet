import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from .config import LATENT_CH, LATENT_SCALE, COND_DIM, DEVICE, CLIP_MODEL_PATH


# ── 3D Autoencoder ────────────────────────────────────────────────────────────
# Shared encoder/decoder for tau PET and MRI (single-channel), paper Sec. II-A (Eq. 1-3).
# PET and MRI use the same network weights to minimise parameters and ensure
# both modalities are compressed into the same latent space.

class ResBlock3D(nn.Module):
    # residual block: two conv layers with a skip connection
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
        )
    def forward(self, x): return x + self.net(x)
# ResBlock3D - two conv layers with GroupNorm and SiLu, plus a skip connection.
# GroupNorm is used because batch sizes are too small for BatchNorm to work reliably


class Encoder3D(nn.Module):
    # compresses a 3d volume into a compact latent rep
    # channel progression: 1 --> 32 --> 64 --> 128 --> latent_ch * 2
    def __init__(self, in_ch=1, base_ch=32, latent_ch=LATENT_CH, scale=LATENT_SCALE):
        super().__init__()

        n_down = int(math.log2(scale))  # number of downsampling stages
        # conv lifts from 1 to 32 channels
        # kernel size 3, padding = 1 to keep spatial dims unchanged
        layers = [nn.Conv3d(in_ch, base_ch, 3, padding=1)]
        ch = base_ch  # track current channel count

        for _ in range(n_down):
            layers += [ResBlock3D(ch), nn.Conv3d(ch, ch * 2, 4, stride=2, padding=1)]
            ch *= 2
        layers += [ResBlock3D(ch), nn.GroupNorm(8, ch), nn.SiLU(),
                   nn.Conv3d(ch, latent_ch * 2, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.net(x)
        # split 8 channels into two 4-channel tensors: mean and logvar
        # these parametrize the latent Gaussian distribution
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar
# Encoder3D
# starts at 1 channel, lifts to 32, then 3 downsampling stages doubling channels
# each time while halving spatial dimensions


class Decoder3D(nn.Module):
    # mirror of the encoder, reconstructs volume from latent
    # channel progression: latent_ch --> .... --> 1
    def __init__(self, latent_ch=LATENT_CH, base_ch=32, out_ch=1, scale=LATENT_SCALE):
        super().__init__()
        n_up = int(math.log2(scale))
        ch = base_ch * (2 ** n_up)
        layers = [nn.Conv3d(latent_ch, ch, 3, padding=1), ResBlock3D(ch)]
        for _ in range(n_up):
            layers += [nn.ConvTranspose3d(ch, ch // 2, 4, stride=2, padding=1), ResBlock3D(ch // 2)]
            ch //= 2
        layers += [nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, out_ch, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z): return self.net(z)
# starts at 256 channels and uses ConvTranspose3d to upsample back to original resolution


class Autoencoder3D(nn.Module):
    """VAE-style autoencoder shared by PET and MRI (paper Eq. 1-3).

    encode()      – reparameterised sample; used during AE pretraining.
    encode_mean() – deterministic mean; used for z0/zm during diffusion training
                    and inference so conditioning is stable across calls.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D()
        self.decoder = Decoder3D()

    def encode(self, x):
        mean, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z   = mean + std * torch.randn_like(std)
        return z, mean, logvar

    def encode_mean(self, x):
        """Deterministic encoding — returns the distribution mean, no sampling."""
        mean, _ = self.encoder(x)
        return mean

    def decode(self, z): return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z), mean, logvar


# ── Denoising U-Net with Cross-Attention ─────────────────────────────────────
# 3D U-Net accepting h_t = [z_t, z_m] with layer-wise cross-attention on CLIP
# embeddings (paper §II-D, Eq. 8-12).

class CrossAttention3D(nn.Module):
    """Spatial-token × context cross-attention (Eq. 8–12)."""
    def __init__(self, feat_dim, context_dim=COND_DIM, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = feat_dim // n_heads
        self.scale   = self.d_head ** -0.5
        self.norm    = nn.LayerNorm(feat_dim)
        self.to_q    = nn.Linear(feat_dim,    feat_dim, bias=False)
        self.to_k    = nn.Linear(context_dim, feat_dim, bias=False)
        self.to_v    = nn.Linear(context_dim, feat_dim, bias=False)
        self.to_out  = nn.Linear(feat_dim,    feat_dim)

    def forward(self, x, context):
        B, C, H, W, D = x.shape
        N = H * W * D
        x_flat = x.view(B, C, N).permute(0, 2, 1)
        x_norm = self.norm(x_flat)

        Q = self.to_q(x_norm)
        K = self.to_k(context)
        V = self.to_v(context)

        def split(t, s):
            return t.view(B, s, self.n_heads, self.d_head).transpose(1, 2)

        Q = split(Q, N); K = split(K, context.size(1)); V = split(V, context.size(1))
        A = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        O = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, N, C)
        O = self.to_out(O)
        return (x_flat + O).permute(0, 2, 1).view(B, C, H, W, D)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb   = t[:, None].float() * freqs[None]
        emb   = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class UNetBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, context_dim=COND_DIM, use_attn=True):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, in_ch)
        self.conv1  = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.conv2  = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn   = CrossAttention3D(out_ch, context_dim=context_dim) if use_attn else None

    def forward(self, x, t_emb, ctx):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h))) + self.skip(x)
        if self.attn is not None:
            h = self.attn(h, ctx)
        return h


class DenoisingUNet3D(nn.Module):
    """Text-guided denoising U-Net (paper §II-D).
    Input : h_t = cat(z_t, z_m)  shape (B, 2*LATENT_CH, lH, lW, lD)
    Output: predicted noise        shape (B,   LATENT_CH, lH, lW, lD)
    context_dim must match the output dimension of the conditioner used.
    """
    def __init__(self, in_ch=LATENT_CH * 2, base_ch=64, t_dim=256, context_dim=COND_DIM):
        super().__init__()
        self.t_embed = TimestepEmbedding(t_dim)
        self.in_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.enc1    = UNetBlock3D(base_ch,     base_ch,     t_dim, context_dim)
        self.down1   = nn.Conv3d(base_ch,       base_ch * 2, 4, stride=2, padding=1)
        self.enc2    = UNetBlock3D(base_ch * 2, base_ch * 2, t_dim, context_dim)
        self.down2   = nn.Conv3d(base_ch * 2,   base_ch * 4, 4, stride=2, padding=1)
        self.mid     = UNetBlock3D(base_ch * 4, base_ch * 4, t_dim, context_dim)
        self.up2     = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2    = UNetBlock3D(base_ch * 4, base_ch * 2, t_dim, context_dim)
        self.up1     = nn.ConvTranspose3d(base_ch * 2, base_ch,     4, stride=2, padding=1)
        self.dec1    = UNetBlock3D(base_ch * 2, base_ch,     t_dim, context_dim)
        self.out     = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv3d(base_ch, LATENT_CH, 1)
        )

    def forward(self, ht, t, ctx):
        t_emb = self.t_embed(t)
        x  = self.in_conv(ht)
        e1 = self.enc1(x,              t_emb, ctx)
        e2 = self.enc2(self.down1(e1), t_emb, ctx)
        m  = self.mid(self.down2(e2),  t_emb, ctx)

        up2 = self.up2(m)
        up2 = F.interpolate(up2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2  = self.dec2(torch.cat([up2, e2], dim=1), t_emb, ctx)

        up1 = self.up1(d2)
        up1 = F.interpolate(up1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1  = self.dec1(torch.cat([up1, e1], dim=1), t_emb, ctx)

        return self.out(d1)


# ── CLIP Text Encoder ─────────────────────────────────────────────────────────
# Encodes conditioning inputs as text prompts via frozen CLIP (paper §II-C, Eq. 7).

def load_clip_encoder(clip_model_path=CLIP_MODEL_PATH, device=DEVICE):
    """Load frozen CLIP text encoder. Returns (tokenizer, text_enc, clip_dim)."""
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    clip_text_enc  = CLIPTextModel.from_pretrained(clip_model_path).to(device)
    clip_text_enc.eval()
    for p in clip_text_enc.parameters():
        p.requires_grad_(False)
    clip_dim = clip_text_enc.config.hidden_size
    print(f"CLIP hidden dim: {clip_dim}")
    return clip_tokenizer, clip_text_enc, clip_dim


def make_encode_atrophy(clip_tokenizer, clip_text_enc, device=DEVICE):
    """Return encode_atrophy(atrophy_vals) for atrophy mode.

    atrophy_vals : (B, 86) float tensor
    Encodes the first 20 atrophy z-scores as a text prompt via CLIP.
    Uses only 20 values to stay within CLIP's 77-token limit.
    Returns      : (B, seq_len, clip_dim)
    """
    @torch.no_grad()
    def encode_atrophy(atrophy_vals):
        prompts = []
        for i in range(atrophy_vals.shape[0]):
            scores     = atrophy_vals[i, :20].tolist()
            scores_str = " ".join([f"{v:.1f}" for v in scores])
            prompts.append(f"Atrophy: {scores_str}")
        tokens = clip_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        return clip_text_enc(**tokens).last_hidden_state

    return encode_atrophy


def make_encode_ptau(clip_tokenizer, clip_text_enc, device=DEVICE):
    """Return encode_ptau(ptau_vals) for ptau217 mode (paper Eq. 7).

    ptau_vals : (B, 1) float tensor of plasma p-tau217 values
    Prompt format: "Plasma is [value]."  (matches paper exactly)
    Returns   : (B, seq_len, clip_dim)
    """
    @torch.no_grad()
    def encode_ptau(ptau_vals):
        prompts = [f"Plasma is {ptau_vals[i, 0].item():.3f}."
                   for i in range(ptau_vals.shape[0])]
        tokens = clip_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        return clip_text_enc(**tokens).last_hidden_state

    return encode_ptau


if __name__ == "__main__":
    from .diffusion import DiffusionSchedule

    ae   = Autoencoder3D().to(DEVICE)
    unet = DenoisingUNet3D().to(DEVICE)
    print(f"Autoencoder parameters:     {sum(p.numel() for p in ae.parameters()):,}")
    print(f"Denoising U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")

    schedule = DiffusionSchedule()
    print(f"Diffusion schedule: T={schedule.T} steps")

    clip_tokenizer, clip_text_enc, clip_dim = load_clip_encoder()
    encode_atrophy = make_encode_atrophy(clip_tokenizer, clip_text_enc)
    encode_ptau    = make_encode_ptau(clip_tokenizer, clip_text_enc)
    dummy_atrophy  = torch.zeros(2, 86).to(DEVICE)
    dummy_ptau     = torch.zeros(2, 1).to(DEVICE)
    print(f"Atrophy embedding shape: {encode_atrophy(dummy_atrophy).shape}")
    print(f"pTau   embedding shape:  {encode_ptau(dummy_ptau).shape}")
