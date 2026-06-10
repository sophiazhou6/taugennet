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

## Canonical Pipeline

### Environment Setup
- Python interpreter: `/home/sz3962/.conda/envs/taugennet/bin/python3`
- Register Jupyter kernel: `python -m ipykernel install --user --name=taugennet --display-name "TauGenNet"`
- Freeze dependencies: `pip freeze > requirements.txt`

### Data Splits
Configurable via `--split TRAIN/VAL/TEST` passed to `scripts/train.py` (only applies with `--dataset v2`).
Default is `70/10/20` (~131/19/37 subjects with 187 total). Use `--split 80/10/10` for ~150/19/18.
The v1 dataset (`src/dataset.py`) retains its hardcoded 72/11/28 ratio and ignores `--split`.
Split fractions are forwarded to `build_dataloaders(train_frac=, val_frac=)` in `src/dataset_v2.py` and `src/dataset_combined.py`.

### Training Launch
```bash
# Full paper pipeline (AE + 3 diffusion modes, ~2 days on A100)
sbatch train_paper.slurm

# Custom split
python scripts/train.py --mode atrophy --dataset v2 --split 80/10/10 \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt --skip-ae \
    --diff-epochs 2000 --patience 500

# End-to-end orchestrator (submits, polls, evaluates, updates README, stages commit)
python scripts/run_experiment.py
python scripts/run_experiment.py --split 80/10/10          # custom split
python scripts/run_experiment.py --from-stage generate_tables  # resume/rerun from a stage
```

### Job Monitoring
```bash
squeue -u sz3962
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
tail -f results/logs/taugennet_<JOBID>.out
```

### Evaluation & Metrics Table
```bash
sbatch eval_paper.slurm
# Parse eval log → results/records/metrics.md + print markdown table
python scripts/metrics_table.py --log-file results/logs/taugennet_<EVALJOBID>.out
```

### Git Commit Conventions
- Prefix: `feat:` / `fix:` / `eval:` / `chore:` / `exp:`
- Include mode and split when relevant: `eval: paper metrics atrophy/ptau217/combined (70/10/20)`
- **Never commit:** `results/checkpoints/`, `data/`, `venv/`, `results/figures/`
- **Always commit after eval:** `results/records/metrics.md`, `README.md`
