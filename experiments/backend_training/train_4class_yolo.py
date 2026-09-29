"""
train_4class_yolo.py
====================
4-Class Marine Debris YOLO11n Training Pipeline
Project: AI-Powered Automated Underwater Marine Debris and Anomaly Detection
         using Side-Scan Sonar Imagery
Author : Member 2 (Computer Vision & Sonar Processing)

Classes
-------
  0 = shipwreck
  1 = aircraft
  2 = mine
  3 = fishing_gear

Preprocessing
-------------
Applies the existing 4-stage CV pipeline (computer_vision/final_preprocessing_pipeline.py)
OFFLINE to every image before YOLO sees it:

  Raw SSS image (any size, BGR uint8)
      │
      ▼  Stage 1: Cross-Track Swath Illumination Normalization (LAB L* channel)
      │           axis=horizontal, method=median, smooth_sigma=35, kernel=71
      │           gain clamped [0.5, 2.5], min_intensity_floor=8
      │
      ▼  Stage 2: Edge-Preserving Bilateral Denoising (BGR)
      │           d=7, sigmaColor=50, sigmaSpace=50
      │
      ▼  Stage 3: Robust Percentile Normalization (global)
      │           p_low=1%, p_high=99%, output range [0, 255]
      │
      ▼  Stage 4: CLAHE Contrast Enhancement (LAB L* channel)
      │           clipLimit=2.0, tileGridSize=(8, 8)
      │
  Preprocessed image (same HxW, BGR uint8)
      │
      ▼  YOLO11n training
         imgsz=640 (YOLO resizes internally)
         Bounding-box labels: unchanged (YOLO normalized coords, geometry-safe)

Bounding-box safety
-------------------
All four pipeline stages are pixel-intensity operations ONLY.
No geometric transformations are applied (no crop, resize, warp, flip, rotate).
YOLO label coordinates remain exactly valid after preprocessing.

Checkpoint
----------
Uses the official yolo11n.pt pretrained checkpoint.
Does NOT modify the existing 3-class .pt files.

Output
------
Preprocessed dataset : runs/detect/4class_training/preprocessed_dataset/
Training artefacts   : runs/detect/4class_training/train/
"""

import os
import sys
import shutil
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Project root: one level above this script (SIHproject/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # backend/
PROJECT_ROOT = SCRIPT_DIR.parent                       # SIHproject/
CV_DIR = PROJECT_ROOT / "computer_vision"              # computer_vision/

# Make sure computer_vision/ is importable
if str(CV_DIR) not in sys.path:
    sys.path.insert(0, str(CV_DIR))

from final_preprocessing_pipeline import process_sonar_image, get_default_config  # noqa: E402

# ---------------------------------------------------------------------------
# 4-Class configuration — ONLY these four classes are used
# ---------------------------------------------------------------------------
NC = 4
CLASS_NAMES = {
    0: "shipwreck",
    1: "aircraft",
    2: "mine",
    3: "fishing_gear",
}
VALID_CLASS_IDS = set(CLASS_NAMES.keys())   # {0, 1, 2, 3}

# ---------------------------------------------------------------------------
# Baseline training hyperparameters
# ---------------------------------------------------------------------------
BASELINE_TRAIN_KWARGS = dict(
    epochs=100,
    patience=20,
    batch=-1,          # auto-detect maximum stable batch size
    imgsz=640,
    amp=True,
    # Moderate SSS-appropriate augmentation (same proven settings as 3-class run)
    fliplr=0.5,        # horizontal flip — valid for symmetric sonar strips
    flipud=0.0,        # vertical flip — sonar has directional meaning
    degrees=0.0,       # no rotation — sonar strips are axis-aligned
    shear=0.0,         # no shear — preserves sonar geometry
    perspective=0.0,   # no perspective warp
    mosaic=0.5,        # moderate mosaic — improves multi-object diversity
    mixup=0.0,         # no mixup — avoids label blending artefacts
    scale=0.2,         # ±20% scale jitter — handles target size variation
    translate=0.1,     # small positional jitter
    hsv_h=0.015,       # minor hue shift (sonar imagery is near-grayscale)
    hsv_s=0.0,         # no saturation shift — sonar has limited colour
    hsv_v=0.4,         # brightness variation — models range attenuation
    cos_lr=True,
    warmup_epochs=3.0,
    cls=1.0,
    box=7.5,
    dfl=1.5,
    val=True,
    save=True,
    save_period=5,
    verbose=True,
)


# ===========================================================================
# STEP 0: Readiness validation (runs before any preprocessing or training)
# ===========================================================================

def validate_readiness(data_yaml_path: Path, output_dir: Path) -> bool:
    """
    Check all 10 pre-training readiness criteria.
    Returns True if all pass, False otherwise.
    """
    import yaml

    print("\n" + "=" * 70)
    print("PRE-TRAINING READINESS VALIDATION (10 CHECKS)")
    print("=" * 70)

    passes = []

    # ------------------------------------------------------------------
    # 1. data.yaml has exactly 4 classes
    # ------------------------------------------------------------------
    with open(data_yaml_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    names = cfg.get("names", {})
    nc_yaml = len(names)
    ok1 = nc_yaml == 4 and set(names.values()) == {"shipwreck", "aircraft", "mine", "fishing_gear"}
    passes.append(ok1)
    status = "PASS" if ok1 else "FAIL"
    print(f"[{status}] 1. data.yaml has exactly 4 classes: {names}")

    # ------------------------------------------------------------------
    # 2. All labels contain only class IDs 0, 1, 2, 3
    # ------------------------------------------------------------------
    dataset_root = Path(cfg["path"])
    bad_labels: list[tuple] = []
    all_found_ids: set[int] = set()
    for split in ["train", "val", "test"]:
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            continue
        for lf in label_dir.glob("*.txt"):
            with open(lf) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(line.split()[0])
                    all_found_ids.add(cid)
                    if cid not in VALID_CLASS_IDS:
                        bad_labels.append((split, lf.name, cid))
    ok2 = len(bad_labels) == 0
    passes.append(ok2)
    status = "PASS" if ok2 else "FAIL"
    print(f"[{status}] 2. All label class IDs ∈ {{0,1,2,3}}. Found IDs: {sorted(all_found_ids)}")
    if bad_labels:
        print(f"       BAD LABELS (first 5): {bad_labels[:5]}")

    # ------------------------------------------------------------------
    # 3. No class IDs accidentally remapped — 0,1,2 must be present
    # ------------------------------------------------------------------
    expected_base = {0, 1, 2}
    ok3 = expected_base.issubset(all_found_ids)
    passes.append(ok3)
    status = "PASS" if ok3 else "FAIL"
    print(f"[{status}] 3. Classes 0 (shipwreck), 1 (aircraft), 2 (mine) present in labels")

    # ------------------------------------------------------------------
    # 4. shipwreck/aircraft/mine label IDs unchanged from 3-class
    # ------------------------------------------------------------------
    ok4 = (0 in all_found_ids and names.get(0) == "shipwreck" and
           1 in all_found_ids and names.get(1) == "aircraft" and
           2 in all_found_ids and names.get(2) == "mine")
    passes.append(ok4)
    status = "PASS" if ok4 else "FAIL"
    print(f"[{status}] 4. Existing shipwreck(0)/aircraft(1)/mine(2) labels unchanged")

    # ------------------------------------------------------------------
    # 5. fishing_gear labels use class ID 3
    # ------------------------------------------------------------------
    ok5 = 3 in all_found_ids and names.get(3) == "fishing_gear"
    passes.append(ok5)
    status = "PASS" if ok5 else "FAIL"
    print(f"[{status}] 5. fishing_gear labels use class ID 3")

    # ------------------------------------------------------------------
    # 6. Preprocessing module importable and returns correct output
    # ------------------------------------------------------------------
    try:
        test_img = np.random.randint(20, 200, (128, 128, 3), dtype=np.uint8)
        result = process_sonar_image(test_img)
        ok6 = (isinstance(result, np.ndarray) and
               result.dtype == np.uint8 and
               result.shape == test_img.shape)
    except Exception as exc:
        ok6 = False
        print(f"       Preprocessing error: {exc}")
    passes.append(ok6)
    status = "PASS" if ok6 else "FAIL"
    print(f"[{status}] 6. final_preprocessing_pipeline.process_sonar_image() works correctly")

    # ------------------------------------------------------------------
    # 7. No old 6-class class names in active config
    # ------------------------------------------------------------------
    OLD_6_CLASS_NAMES = {"drowning_victim", "seafloor", "crab_pot"}
    active_names = set(names.values()) if isinstance(names, dict) else set(names)
    ok7 = len(active_names & OLD_6_CLASS_NAMES) == 0
    passes.append(ok7)
    status = "PASS" if ok7 else "FAIL"
    print(f"[{status}] 7. No old 6-class names in active data.yaml. Active: {active_names}")

    # ------------------------------------------------------------------
    # 8. Existing 3-class .pt files exist and are untouched (read-only check)
    # ------------------------------------------------------------------
    model_dir = SCRIPT_DIR / "app" / "model"
    pt_new = model_dir / "new_marine_debris_model.pt"
    pt_old = model_dir / "improved_yolov8n_best.pt"
    ok8 = pt_new.exists() and pt_old.exists()
    passes.append(ok8)
    status = "PASS" if ok8 else "FAIL"
    print(f"[{status}] 8. Existing 3-class .pt files present:")
    print(f"       new_marine_debris_model.pt   : {pt_new.exists()} ({pt_new.stat().st_size // 1024} KB)")
    print(f"       improved_yolov8n_best.pt     : {pt_old.exists()} ({pt_old.stat().st_size // 1024} KB)")

    # ------------------------------------------------------------------
    # 9. New training output directory is separate
    # ------------------------------------------------------------------
    ok9 = not output_dir.is_relative_to(model_dir)
    passes.append(ok9)
    status = "PASS" if ok9 else "FAIL"
    print(f"[{status}] 9. New output dir is separate from 3-class model dir: {output_dir}")

    # ------------------------------------------------------------------
    # 10. Dataset splits present on disk
    # ------------------------------------------------------------------
    missing_splits = []
    for split in ["train", "val", "test"]:
        img_split = dataset_root / "images" / split
        lbl_split = dataset_root / "labels" / split
        if not img_split.exists():
            missing_splits.append(f"images/{split}")
        if not lbl_split.exists():
            missing_splits.append(f"labels/{split}")
    ok10 = len(missing_splits) == 0
    passes.append(ok10)
    status = "PASS" if ok10 else "FAIL"
    print(f"[{status}] 10. All dataset splits present on disk. Missing: {missing_splits if missing_splits else 'None'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(passes)
    passed = sum(passes)
    print("\n" + "-" * 70)
    print(f"READINESS RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("✓ ALL CHECKS PASSED — pipeline is ready for training")
    else:
        failed_idx = [i + 1 for i, p in enumerate(passes) if not p]
        print(f"✗ FAILED CHECKS: {failed_idx}")
        print("  Resolve all failures before starting training.")
    print("=" * 70 + "\n")

    return passed == total


# ===========================================================================
# STEP 1: Offline preprocessing — apply CV pipeline to all splits
# ===========================================================================

def preprocess_split(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    split_name: str,
) -> int:
    """
    Apply the 4-stage sonar preprocessing pipeline to all images in one split.
    Labels are hard-linked (zero-copy) because they are geometry-invariant.

    Returns the number of images processed.
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in img_exts]

    cfg = get_default_config()  # default 4-stage config from the CV module
    processed = 0
    errors = 0

    print(f"\n  [{split_name}] Preprocessing {len(img_files)} images …")
    t0 = time.time()

    for img_path in img_files:
        dst_img_path = dst_img_dir / img_path.name

        # Skip if already preprocessed (allows resuming interrupted runs)
        if dst_img_path.exists():
            processed += 1
            continue

        # --- Load raw image ---
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback: try grayscale
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"    [WARN] Cannot read image: {img_path.name}")
                errors += 1
                continue
            # Convert grayscale to BGR so the pipeline always gets a consistent format
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # --- Apply 4-stage CV preprocessing pipeline ---
        # Stage 1: Cross-Track Swath Illumination Normalization
        # Stage 2: Edge-Preserving Bilateral Denoising (d=7, σ=50)
        # Stage 3: Robust Percentile Normalization (1%–99%)
        # Stage 4: CLAHE Contrast Enhancement (LAB L*, clipLimit=2.0, tile=8×8)
        #
        # All stages are pixel-intensity operations ONLY.
        # No geometric transformation is applied.
        # Bounding-box coordinates are unaffected.
        preprocessed = process_sonar_image(img_bgr, config=cfg)

        # --- Save preprocessed image (lossless PNG) ---
        out_name = img_path.stem + ".png"
        cv2.imwrite(str(dst_img_dir / out_name), preprocessed)

        # --- Copy / link the label file unchanged ---
        src_lbl = src_lbl_dir / (img_path.stem + ".txt")
        dst_lbl = dst_lbl_dir / (img_path.stem + ".txt")
        if src_lbl.exists() and not dst_lbl.exists():
            shutil.copy2(str(src_lbl), str(dst_lbl))

        processed += 1

    elapsed = time.time() - t0
    rate = processed / max(elapsed, 1e-6)
    print(f"  [{split_name}] Done: {processed} preprocessed, {errors} errors "
          f"in {elapsed:.1f}s ({rate:.1f} img/s)")
    return processed


def preprocess_dataset(
    src_data_yaml: Path,
    dst_root: Path,
) -> Path:
    """
    Preprocess all three splits and write a new data.yaml pointing to the
    preprocessed directory.

    Original images and labels are NEVER modified.

    Returns the path to the new data.yaml for YOLO training.
    """
    import yaml

    with open(src_data_yaml, "r") as fh:
        cfg = yaml.safe_load(fh)

    src_root = Path(cfg["path"])

    print("\n" + "=" * 70)
    print("OFFLINE PREPROCESSING  (final_preprocessing_pipeline.py)")
    print(f"  Source  : {src_root}")
    print(f"  Output  : {dst_root}")
    print(f"  Pipeline: 4-stage (swath_norm → bilateral → robust_norm → CLAHE)")
    print(f"  Classes : {cfg['names']}")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        preprocess_split(
            src_img_dir=src_root / "images" / split,
            src_lbl_dir=src_root / "labels" / split,
            dst_img_dir=dst_root / "images" / split,
            dst_lbl_dir=dst_root / "labels" / split,
            split_name=split,
        )

    # Write preprocessed data.yaml
    preprocessed_yaml = dst_root / "data_preprocessed.yaml"
    new_cfg = {
        "path": str(dst_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NC,
        "names": cfg["names"],
    }
    with open(preprocessed_yaml, "w") as fh:
        yaml.dump(new_cfg, fh, default_flow_style=False, allow_unicode=True)

    print(f"\n  Preprocessed data.yaml written: {preprocessed_yaml}")
    return preprocessed_yaml


# ===========================================================================
# STEP 2: YOLO training
# ===========================================================================

def train_model(
    data_yaml: Path,
    base_model: str,
    output_dir: Path,
    run_name: str,
    extra_kwargs: dict,
) -> tuple[Path, YOLO]:
    """
    Train YOLO11n on the preprocessed 4-class dataset.
    """
    print("\n" + "=" * 70)
    print("STARTING YOLO11n 4-CLASS TRAINING")
    print(f"  Data YAML   : {data_yaml}")
    print(f"  Checkpoint  : {base_model}")
    print(f"  Output      : {output_dir / run_name}")
    print(f"  Classes     : {CLASS_NAMES}")
    print("=" * 70)

    model = YOLO(base_model)
    t0 = time.time()

    results = model.train(
        data=str(data_yaml),
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        **{**BASELINE_TRAIN_KWARGS, **extra_kwargs},
    )

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    best_weights = output_dir / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = output_dir / run_name / "weights" / "last.pt"

    print(f"Best weights  : {best_weights}")
    return best_weights, model


# ===========================================================================
# STEP 3: Evaluation
# ===========================================================================

def run_evaluation(weights_path: Path, data_yaml: Path, output_dir: Path):
    """
    Run full evaluation on val and test splits.
    Reports per-class precision, recall, mAP50, mAP50-95, confusion matrix,
    PR curve, and F1 curve.
    """
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION  (4-class)")
    print("=" * 70)

    model = YOLO(str(weights_path))

    metrics_all = {}
    for split in ["val", "test"]:
        print(f"\n--- {split.upper()} SET ---")
        m = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=640,
            batch=16,
            verbose=True,
            plots=True,
            save_json=True,
            project=str(output_dir),
            name=f"eval_{split}",
            exist_ok=True,
        )
        metrics_all[split] = m

    return metrics_all


def print_per_class_metrics(metrics_all: dict):
    """Print per-class breakdown for all evaluation splits."""
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS SUMMARY")
    print("=" * 70)

    for split, m in metrics_all.items():
        print(f"\n[{split.upper()}]")
        print(f"  Overall  mAP50    : {m.box.map50:.4f}")
        print(f"  Overall  mAP50-95 : {m.box.map:.4f}")
        print(f"  Overall  Precision : {m.box.mp:.4f}")
        print(f"  Overall  Recall    : {m.box.mr:.4f}")

        if hasattr(m.box, "ap_class_index") and m.box.ap_class_index is not None:
            print(f"  {'Class':<20} {'P':>8} {'R':>8} {'mAP50':>10} {'mAP50-95':>12}")
            print("  " + "-" * 62)
            for i, cid in enumerate(m.box.ap_class_index):
                cname = CLASS_NAMES.get(int(cid), f"class_{cid}")
                p = m.box.p[i] if hasattr(m.box, "p") else float("nan")
                r = m.box.r[i] if hasattr(m.box, "r") else float("nan")
                ap50 = m.box.ap50[i] if hasattr(m.box, "ap50") else float("nan")
                ap = m.box.ap[i] if hasattr(m.box, "ap") else float("nan")
                print(f"  {cname:<20} {p:>8.4f} {r:>8.4f} {ap50:>10.4f} {ap:>12.4f}")


# ===========================================================================
# CLI argument parser
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="4-Class Marine Debris YOLO11n Training Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced" / "data.yaml"),
        help="Path to the 4-class data.yaml (default: SIH_Combined_4Class_Balanced/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained checkpoint to start from (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "detect" / "4class_training"),
        help="Root directory for all training outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="YOLO run name (subfolder inside --output)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: '0' for CUDA GPU, 'cpu' for CPU (auto-detected if omitted)",
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip offline preprocessing if preprocessed images already exist",
    )
    parser.add_argument(
        "--eval_only",
        type=str,
        default=None,
        help="Skip training; run evaluation only on the given weights path",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run readiness checks only; do not preprocess or train",
    )
    return parser.parse_args()


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    args = parse_args()

    data_yaml = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    preprocessed_dir = output_dir / "preprocessed_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "0"
        print(f"[HW] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[HW] No CUDA GPU detected — using CPU")

    # ------------------------------------------------------------------
    # Step 0: Readiness validation (always runs first)
    # ------------------------------------------------------------------
    all_ok = validate_readiness(data_yaml, output_dir)

    if args.validate_only:
        sys.exit(0 if all_ok else 1)

    if not all_ok:
        print("[ABORT] Readiness validation failed. Fix the issues above before training.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Evaluation-only mode
    # ------------------------------------------------------------------
    if args.eval_only:
        weights = Path(args.eval_only).resolve()
        preprocessed_yaml = preprocessed_dir / "data_preprocessed.yaml"
        if not preprocessed_yaml.exists():
            print(f"[ERROR] Preprocessed data.yaml not found at {preprocessed_yaml}")
            print("        Run without --eval_only first to preprocess the dataset.")
            sys.exit(1)
        metrics = run_evaluation(weights, preprocessed_yaml, output_dir)
        print_per_class_metrics(metrics)
        return

    # ------------------------------------------------------------------
    # Step 1: Offline preprocessing (skippable if already done)
    # ------------------------------------------------------------------
    preprocessed_yaml = preprocessed_dir / "data_preprocessed.yaml"
    if args.skip_preprocess and preprocessed_yaml.exists():
        print(f"\n[SKIP] Preprocessing already done. Using: {preprocessed_yaml}")
    else:
        preprocessed_yaml = preprocess_dataset(data_yaml, preprocessed_dir)

    # ------------------------------------------------------------------
    # Step 2: Training
    # ------------------------------------------------------------------
    extra = {"device": device, "epochs": args.epochs}
    best_weights, model = train_model(
        data_yaml=preprocessed_yaml,
        base_model=args.model,
        output_dir=output_dir,
        run_name=args.name,
        extra_kwargs=extra,
    )

    # ------------------------------------------------------------------
    # Step 3: Evaluation
    # ------------------------------------------------------------------
    metrics = run_evaluation(best_weights, preprocessed_yaml, output_dir)
    print_per_class_metrics(metrics)

    print("\n" + "=" * 70)
    print("4-CLASS PIPELINE COMPLETED")
    print(f"  Best model  : {best_weights}")
    print(f"  Output root : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
