# Aqua Vision Experiments & Research Archive

This directory houses all experimental runs, pilot training checkpoints, ablation studies, diagnostic scripts, baseline evaluations, and intermediate artifacts developed during the Smart India Hackathon (SIH) research lifecycle.

---

## Directory Overview

```
experiments/
├── analysis/                     # Diagnostic analysis and failure mode inspection scripts
├── backend_training/             # Legacy model training pipelines and early sonar training runs
├── computer_vision_experiments/  # CLAHE, Bilateral filtering, Lee/Frost speckle filter comparisons
├── computer_vision_results/      # Output imagery and intermediate spatial filters from CV benchmarks
├── databases/                    # Standalone development database snapshots and backups
├── legacy_yolo/                  # Initial YOLOv8/YOLO11 baseline weights, test configs, and error analyses
├── model_evaluation/             # Comparative benchmarks, PR/F1 curves, and 100-epoch summary metrics
├── runs/                         # Full YOLO detection training runs and validation logs (including 100ep checkpoints)
├── scratch/                      # Ad-hoc diagnostic and exploratory analysis scripts
├── scripts/                      # Targeted training experiments, tiled inference tests, and dataset prep
└── tmp/                          # Validation threshold sweeps and parameter tuning outputs
```

---

## Note on Production Artifacts

The production-ready model weights (`aqua_vision_100ep_best.pt`) are actively maintained in [`model/`](../model/) and loaded by [`backend/`](../backend/).
All training runs, checkpoints, and temporary logs in this folder are ignored by git (via `.gitignore`) to ensure a lightweight and clean repository.
