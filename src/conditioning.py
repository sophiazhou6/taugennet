"""
conditioning.py – two interchangeable conditioning strategies.

PTau217Conditioner  (mode='ptau217')
    Frozen CLIP text encoder. Encodes a scalar plasma p-tau217 value
    as the prompt "Plasma is X.XXX." matching the paper (§II-C, Eq. 7).
    No trainable parameters; nothing added to the optimizer.

AtrophyConditioner  (mode='atrophy')
    Small learned MLP. Projects an 86-dim regional atrophy z-score vector
    into a sequence of COND_DIM tokens for cross-attention.
    Uses all 86 regions (not just 20) and avoids CLIP's poor numeric encoding.
    Parameters are added to the diffusion optimizer and saved in the checkpoint.

Both expose the same interface:
    conditioner.encode(data)  → (B, n_tokens, COND_DIM)
    conditioner.out_dim       → int  (= COND_DIM = 512)
    conditioner.trainable_parameters() → list[Parameter]
    conditioner.state_dict() / load_state_dict()
    conditioner.train() / eval()
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

from .config import CLIP_MODEL_PATH, COND_DIM, DEVICE


# ── p-tau217 conditioner ──────────────────────────────────────────────────────

class PTau217Conditioner:
    """Frozen CLIP text encoder conditioner for scalar plasma p-tau217."""

    out_dim = COND_DIM

    def __init__(self, model_path=CLIP_MODEL_PATH, device=DEVICE):
        self.device    = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.model     = CLIPTextModel.from_pretrained(model_path).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # CLIP hidden size is 512 for ViT-B/32; confirm it matches COND_DIM
        assert self.model.config.hidden_size == COND_DIM, (
            f"CLIP hidden size {self.model.config.hidden_size} ≠ COND_DIM {COND_DIM}"
        )

    @torch.no_grad()
    def encode(self, ptau_vals):
        """
        ptau_vals : (B,) or (B, 1) float tensor of p-tau217 values
        Returns   : (B, seq_len, COND_DIM)
        """
        if ptau_vals.dim() == 2:
            ptau_vals = ptau_vals.squeeze(1)
        prompts = [f"Plasma is {v.item():.3f}." for v in ptau_vals]
        tokens  = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        return self.model(**tokens).last_hidden_state  # (B, seq_len, 512)

    def trainable_parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        pass  # frozen; nothing to load

    def train(self): pass
    def eval(self):  pass

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self


# ── atrophy conditioner ───────────────────────────────────────────────────────

class AtrophyConditioner(nn.Module):
    """Learned MLP conditioner for 86-dim regional atrophy z-score vectors.

    Produces a sequence of n_tokens context vectors for cross-attention,
    using all 86 regions rather than the 20-region CLIP-text truncation.
    """

    out_dim = COND_DIM

    def __init__(self, n_regions=86, out_dim=COND_DIM, n_tokens=16):
        super().__init__()
        self.n_tokens = n_tokens
        hidden = 256
        self.mlp = nn.Sequential(
            nn.LayerNorm(n_regions),
            nn.Linear(n_regions, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim * n_tokens),
        )

    def encode(self, atrophy_vals):
        """
        atrophy_vals : (B, 86) float tensor of regional atrophy z-scores
        Returns      : (B, n_tokens, COND_DIM)
        """
        return self.mlp(atrophy_vals).view(
            atrophy_vals.shape[0], self.n_tokens, self.out_dim
        )

    def forward(self, x):
        return self.encode(x)

    def trainable_parameters(self):
        return list(self.parameters())


# ── factory ───────────────────────────────────────────────────────────────────

def build_conditioner(mode: str, device=DEVICE):
    """
    mode : 'ptau217' or 'atrophy'
    Returns an initialised conditioner on the given device.
    """
    if mode == "ptau217":
        return PTau217Conditioner(device=device)
    if mode == "atrophy":
        return AtrophyConditioner().to(device)
    raise ValueError(f"Unknown conditioning mode {mode!r}. Use 'ptau217' or 'atrophy'.")
