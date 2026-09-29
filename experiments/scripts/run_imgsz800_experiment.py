#!/usr/bin/env python
"""Orchestrate the YOLO11n imgsz=800 experiment.

Steps:
1. Validate dataset and preprocessing (reuse existing validation).
2. Preprocess the dataset (offline 4‑stage pipeline).
3. Train YOLO11n with imgsz=800 in a new output directory.
4. Evaluate on the validation split using the same imgsz.
5. Measure inference time on a handful of validation images.
6. Load baseline metrics if available and generate a comparison report.
"""

import time
import json
from pathlib import Path
import sys

# Ensure the project root (SIHproject) is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # SIHproject/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import functions from the existing training script
from backend.train_4class_yolo import (
    validate_readiness,
    preprocess_dataset,
    train_model,
    run_evaluation,
)
from ultralytics import YOLO
import numpy as np

def main():
    # ------------------------------------------------------------
    # 0. Paths & constants
    # ------------------------------------------------------------
    data_yaml = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced" / "data.yaml"
    output_root = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "imgsz800_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------
    # 1. Readiness validation (same 10 checks as original script)
    # ------------------------------------------------------------
    print("\n=== READINESS VALIDATION ===")
    if not validate_readiness(data_yaml, output_root):
        sys.exit("Readiness validation failed – aborting experiment.")

    # ------------------------------------------------------------
    # 2. Offline preprocessing (reuse existing pipeline)
    # ------------------------------------------------------------
    preprocessed_yaml = preprocess_dataset(data_yaml, output_root / "preprocessed_dataset")
    print(f"Preprocessed YAML written to: {preprocessed_yaml}\n")

    # ------------------------------------------------------------
    # 3. Training with imgsz=800 (override default 640)
    # ------------------------------------------------------------
    print("\n=== TRAINING (imgsz=800) ===")
    extra_kwargs = {"imgsz": 800}
    start = time.time()
    best_weights, model = train_model(
        data_yaml=preprocessed_yaml,
        base_model="yolo11n.pt",
        output_dir=output_root,
        run_name="imgsz800_v1",
        extra_kwargs=extra_kwargs,
    )
    training_time = time.time() - start
    print(f"Training elapsed: {training_time:.1f}s")

    # ------------------------------------------------------------
    # 4. Evaluation on validation set using imgsz=800
    # ------------------------------------------------------------
    print("\n=== EVALUATION (val, imgsz=800) ===")
    # Use YOLO directly to control imgsz
    model = YOLO(str(best_weights))
    eval_metrics = model.val(
        data=str(preprocessed_yaml),
        split="val",
        imgsz=800,
        batch=16,
        verbose=False,
        plots=False,
        save_json=True,
        project=str(output_root),
        name="eval_val",
        exist_ok=True,
    )
    # Extract overall metrics
    overall = {
        "precision": getattr(eval_metrics.box, "mp", None),
        "recall": getattr(eval_metrics.box, "mr", None),
        "map50": getattr(eval_metrics.box, "map50", None),
        "map": getattr(eval_metrics.box, "map", None),
    }
    # Per‑class metrics (order follows CLASS_NAMES in training script)
    class_names = {0: "shipwreck", 1: "aircraft", 2: "mine", 3: "fishing_gear"}
    per_class = []
    if hasattr(eval_metrics.box, "ap_class_index") and eval_metrics.box.ap_class_index is not None:
        for i, cid in enumerate(eval_metrics.box.ap_class_index):
            cname = class_names.get(int(cid), f"class_{cid}")
            per_class.append({
                "class": cname,
                "precision": getattr(eval_metrics.box, "p", [None])[i],
                "recall": getattr(eval_metrics.box, "r", [None])[i],
                "map50": getattr(eval_metrics.box, "ap50", [None])[i],
                "map": getattr(eval_metrics.box, "ap", [None])[i],
            })
    else:
        print("[WARN] Per‑class results not available in eval output.")

    # ------------------------------------------------------------
    # 5. Simple inference‑time benchmark (20 random val images)
    # ------------------------------------------------------------
    print("\n=== INFERENCE TIME BENCHMARK ===")
    # Load list of validation images
    import glob, random
    val_img_dir = preprocessed_yaml.parent / "images" / "val"
    img_paths = list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpg"))
    sample_paths = random.sample(img_paths, min(20, len(img_paths)))
    start_inf = time.time()
    for p in sample_paths:
        _ = model.predict(source=str(p), imgsz=800, verbose=False)
    inference_time = (time.time() - start_inf) / max(len(sample_paths), 1) * 1000  # ms per image
    print(f"Average inference time per image: {inference_time:.1f} ms")

    # ------------------------------------------------------------
    # 6. Load baseline metrics (if they exist)
    # ------------------------------------------------------------
    baseline_path = (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "4class_training"
        / "baseline_v1"
        / "eval_val.json"
    )
    baseline = None
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text())
        except Exception as e:
            print(f"[WARN] Could not parse baseline JSON: {e}")
    else:
        print("[INFO] Baseline eval_val.json not found – report will contain placeholders.")

    # ------------------------------------------------------------
    # 7. Generate markdown report
    # ------------------------------------------------------------
    report_path = output_root / "IMG_SZ800_EXPERIMENT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# imgsz=800 Experiment Report\n\n")
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Baseline (640) | imgsz=800 | Δ (%) |\n")
        f.write("|---|---|---|---|\n")
        for key, label in [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("map50", "mAP50"),
            ("map", "mAP50‑95"),
        ]:
            base_val = baseline.get(key) if baseline else "N/A"
            new_val = overall.get(key, "N/A")
            try:
                delta = (
                    (float(new_val) - float(base_val)) / float(base_val) * 100
                    if base_val not in (None, "N/A") and float(base_val) != 0
                    else "—"
                )
                delta_fmt = f"{delta:.1f}%" if isinstance(delta, float) else delta
            except Exception:
                delta_fmt = "—"
            f.write(f"| {label} | {base_val} | {new_val:.4f} | {delta_fmt} |\n")
        f.write("\n## Per‑Class Metrics (Recall focus)\n\n")
        f.write(
            "| Class | Precision (Δ%) | Recall (Δ%) | mAP50 (Δ%) | mAP50‑95 (Δ%) |\n"
        )
        f.write("|---|---|---|---|---|\n")
        for pc in per_class:
            cname = pc["class"]
            # Find baseline entry if available
            base_entry = None
            if baseline and isinstance(baseline.get("per_class"), list):
                for be in baseline["per_class"]:
                    if be.get("class") == cname:
                        base_entry = be
                        break
            def fmt(val):
                return f"{val:.4f}" if isinstance(val, (float, int)) else "N/A"
            def delta(new, old):
                try:
                    d = (float(new) - float(old)) / float(old) * 100
                    return f"{d:.1f}%"
                except Exception:
                    return "—"
            row = [
                cname,
                fmt(pc.get("precision")),
                fmt(pc.get("recall")),
                fmt(pc.get("map50")),
                fmt(pc.get("map")),
            ]
            # Append delta columns
            if base_entry:
                row.append(delta(pc.get("precision"), base_entry.get("precision")))
                row.append(delta(pc.get("recall"), base_entry.get("recall")))
                row.append(delta(pc.get("map50"), base_entry.get("map50")))
                row.append(delta(pc.get("map"), base_entry.get("map")))
                f.write(
                    f"| {cname} | {row[1]} ({row[5]}) | {row[2]} ({row[6]}) | {row[3]} ({row[7]}) | {row[4]} ({row[8]}) |\n"
                )
            else:
                f.write(
                    f"| {cname} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"
                )
        f.write("\n## Timing\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Training time (s) | {training_time:.1f} |\n")
        f.write(f"| Inference time per image (ms) | {inference_time:.1f} |\n")
        f.write("\n---\n")
        f.write(
            "**Verdict:** If Δ% for recall (or any metric) is positive, the imgsz=800 configuration improved that aspect. "
            "If Δ% is negative or ‘—’, no improvement was observed.\n"
        )
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
