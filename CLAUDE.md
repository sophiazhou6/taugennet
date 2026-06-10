# TauGenNet

## Project Context

This is a medical imaging research project implementing TauGenNet — a 3D diffusion model for generating tau PET brain scans conditioned on MRI. When implementing or reviewing code, always cross-check against the reference paper's architecture, resolutions, and sampling steps. Key paper specs:
- Input resolution: 96×112×96
- Sampler: DDPM (500 steps), not DDIM
- Conditioning: MRI (structural) + optional demographic/amyloid variables

## Environment

- Default environment: VS Code with a project venv on the Princeton HPC scratch filesystem
- Working directory: `/scratch/network/sz3962/taugennet`
- Do not assume Colab or a cloud notebook environment
- Resolve imports and kernels against the project venv

## Data & Reproducibility

Before stating any data statistics (split sizes, subject counts, ratios), read the config and dataset files directly — do not infer or estimate. Cite the file and line when reporting these numbers. Key facts to verify from source:
- Dataset: ADNI tau PET + MRI, 187 subjects, 96×112×96 voxels
- Always distinguish per-subject vs. global statistics explicitly

## Evaluation & Figures

When generating evaluation figures or tables:
- Include all standard metrics: MSE, SSIM (and others used in the paper)
- Clearly label whether statistics are global or per-subject
- Match the paper's reported metrics for direct comparison
