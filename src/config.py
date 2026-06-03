import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Volume dimensions
# (96, 112, 96) covers the full MNI brain at ~2 mm (MRI native resolution),
# divisible by LATENT_SCALE=8 in every dimension, no cropping.
# Paper used (160, 160, 96) which crops ~47% of the Z axis; kept as a comment
# for reference: VOL_SHAPE = (160, 160, 96)
VOL_SHAPE    = (96, 112, 96)
LATENT_CH    = 4
LATENT_SCALE = 8              # spatial downscale factor of autoencoder

LAT_H = VOL_SHAPE[0] // LATENT_SCALE
LAT_W = VOL_SHAPE[1] // LATENT_SCALE
LAT_D = VOL_SHAPE[2] // LATENT_SCALE

# Diffusion
T_STEPS    = 1000
BETA_START = 1e-4   # values used in DDPM paper
BETA_END   = 0.02

# Training  (paper: batch=8, epochs=600 on A100)
AE_EPOCHS   = 100
DIFF_EPOCHS = 2000
BATCH_SIZE  = 8
LR          = 1e-4

# Data paths — leave empty to use synthetic data
PET_DIR  = ""   # directory of .nii.gz tau PET volumes
MRI_DIR  = ""   # directory of registered T1 MRI volumes
CSV_PATH = ""   # CSV with columns: subject_id, ptau217

#USE_SYNTHETIC  = PET_DIR == ""
USE_SYNTHETIC  = False
N_SYNTH_TRAIN  = 40
N_SYNTH_TEST   = 10

CLIP_DIM  = 512   # must match CLIP ViT-B/32 hidden size
COND_DIM  = 512   # conditioning embedding dimension (both conditioners output this)
N_REGIONS = 86

BASE_DIR          = "/scratch/network/sz3962/taugennet/data/raw"
CLIP_MODEL_PATH   = "/scratch/network/sz3962/taugennet/clip_model"
ADNI_FLUID_CSV    = "/scratch/network/sz3962/taugennet/data/raw/ADNI34Tau_withFluidBiomarkers.csv"
CHECKPOINT_DIR    = "/scratch/network/sz3962/taugennet/results/checkpoints"
FIGURES_DIR       = "/scratch/network/sz3962/taugennet/results/figures"
GENERATED_DIR     = "/scratch/network/sz3962/taugennet/data/generated"

AE_CHECKPOINT_PATH   = "/scratch/network/sz3962/taugennet/results/checkpoints/taugennet_checkpoint.pt"
DIFF_CHECKPOINT_PATH = "/scratch/network/sz3962/taugennet/results/checkpoints/taugennet_checkpoint_8_epochs.pt"

