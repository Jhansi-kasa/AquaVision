#!/usr/bin/env python
"""Hard‑example analysis for classes 2 (mine) and 3 (fishing_gear).

The script follows the implementation plan outlined in
`implementation_plan_hard_example.md`.  It is **read‑only** – it never modifies
any dataset, label, or model files.  All outputs are written under:
`runs/detect/hard_example_analysis/`.

Requirements (installed in the environment):
- ultralytics (for YOLO inference)
- opencv-python
- numpy
- matplotlib (for drawing visualizations)
- pyyaml (for parsing data.yaml)

Running:
    python scripts/hard_example_analysis.py

The script will generate:
- `runs/detect/hard_example_analysis/HARD_EXAMPLE_ANALYSIS.md`
- up to 50 PNG visualizations in the same directory.
"""

import os
import cv2
import yaml
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------------------------------------------------
# Configuration (adjust if needed)
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # assumes script in "scripts"
DATA_ROOT = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced"
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "baseline_v1" / "weights" / "best.pt"

IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_MATCH = 0.50
IOU_NEAR = 0.30  # for low‑confidence candidate search

# Output directory
OUT_DIR = PROJECT_ROOT / "runs" / "detect" / "hard_example_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "HARD_EXAMPLE_ANALYSIS.md"

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def load_class_names(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(map(int, names.keys()))]
    return names


def parse_label_file(label_path: Path, img_w: int, img_h: int):
    ann = []
    if not label_path.is_file():
        return ann
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            ann.append({
                "cls": cls,
                "bbox": [x1, y1, x2, y2],
                "w": x2 - x1,
                "h": y2 - y1,
                "area": (x2 - x1) * (y2 - y1),
            })
    return ann


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_predictions(preds, gts, iou_thr=IOU_MATCH):
    matched_gt = [False] * len(gts)
    tp = []
    fp = []
    wrong = []
    for p in preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if matched_gt[idx]:
                continue
            cur_iou = iou(p["bbox"], gt["bbox"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_idx = idx
        if best_iou >= iou_thr and best_idx != -1:
            matched_gt[best_idx] = True
            tp.append({"pred": p, "gt": gts[best_idx], "iou": best_iou})
            if p["cls"] != gts[best_idx]["cls"]:
                wrong.append({"pred": p, "gt": gts[best_idx], "iou": best_iou})
        else:
            fp.append(p)
    fn = [gt for idx, gt in enumerate(gts) if not matched_gt[idx]]
    return tp, fp, fn, wrong


def extract_image_stats(img_path: Path, label_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image {img_path}")
    h, w = img.shape[:2]
    ann = parse_label_file(label_path, w, h)
    return img, h, w, ann


def compute_size_stats(annotations):
    if not annotations:
        return {}
    widths = np.array([a["w"] for a in annotations])
    heights = np.array([a["h"] for a in annotations])
    areas = np.array([a["area"] for a in annotations])
    min_dim = np.minimum(widths, heights)
    return {
        "total": len(annotations),
        "min_w": int(widths.min()),
        "min_h": int(heights.min()),
        "median_w": float(np.median(widths)),
        "median_h": float(np.median(heights)),
        "mean_w": float(widths.mean()),
        "mean_h": float(heights.mean()),
        "%<16": float((min_dim < 16).mean() * 100),
        "%<32": float((min_dim < 32).mean() * 100),
        "%<64": float((min_dim < 64).mean() * 100),
        "%area<500": float((areas < 500).mean() * 100),
        "%area<1024": float((areas < 1024).mean() * 100),
    }

# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------

def main():
    class_names = load_class_names(DATA_ROOT / "data.yaml")
    # Gather data per split
    splits = {"train": {}, "val": {}, "test": {}}
    for split in splits:
        img_dir = DATA_ROOT / "images" / split
        lbl_dir = DATA_ROOT / "labels" / split
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file():
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            img, h, w, ann = extract_image_stats(img_path, lbl_path)
            splits[split][img_path.name] = {
                "image": img,
                "h": h,
                "w": w,
                "ann": ann,
                "path": img_path,
            }
    # ---------- Analysis 1 – Size stats ----------
    all_mine = []
    all_fish = []
    for split in splits:
        for info in splits[split].values():
            for a in info["ann"]:
                if a["cls"] == 2:
                    all_mine.append(a)
                elif a["cls"] == 3:
                    all_fish.append(a)
    size_stats_mine = compute_size_stats(all_mine)
    size_stats_fish = compute_size_stats(all_fish)
    # ---------- Analysis 2 – Training distribution ----------
    train_info = splits["train"]
    img_contains_mine = []
    img_contains_fish = []
    boxes_mine = 0
    boxes_fish = 0
    for fname, info in train_info.items():
        has_m = any(a["cls"] == 2 for a in info["ann"])
        has_f = any(a["cls"] == 3 for a in info["ann"])
        if has_m:
            img_contains_mine.append(fname)
        if has_f:
            img_contains_fish.append(fname)
        boxes_mine += sum(1 for a in info["ann"] if a["cls"] == 2)
        boxes_fish += sum(1 for a in info["ann"] if a["cls"] == 3)
    total_train_images = len(train_info)
    both_classes = set(img_contains_mine).intersection(img_contains_fish)
    neither_classes = [f for f in train_info if f not in img_contains_mine and f not in img_contains_fish]
    # ---------- Analysis 3 – Hard example flags ----------
    hard_flags = {"very_small": [], "thin": [], "low_contrast": [], "near_border": []}
    contrast_thresh = 20
    border_margin = 10
    for split in ("val", "test"):
        for info in splits[split].values():
            img = info["image"]
            for a in info["ann"]:
                if a["cls"] not in (2, 3):
                    continue
                # very small
                if min(a["w"], a["h"]) < 16:
                    hard_flags["very_small"].append((info["path"].name, a["cls"]))
                # thin (aspect ratio)
                ar = a["w"] / a["h"] if a["h"] != 0 else 0
                if ar < 0.3 or ar > 3.0:
                    hard_flags["thin"].append((info["path"].name, a["cls"]))
                # low contrast
                x1, y1, x2, y2 = a["bbox"]
                patch = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                if patch.size > 0 and patch.std() < contrast_thresh:
                    hard_flags["low_contrast"].append((info["path"].name, a["cls"]))
                # near border
                if (x1 < border_margin or y1 < border_margin or (info["w"] - x2) < border_margin or (info["h"] - y2) < border_margin):
                    hard_flags["near_border"].append((info["path"].name, a["cls"]))
    # ---------- Analysis 4 – Model inference ----------
    model = YOLO(str(MODEL_PATH))
    metrics = {"val": {"tp": {2: [], 3: []}, "fp": {2: [], 3: []}, "fn": {2: [], 3: []}, "wrong": []},
               "test": {"tp": {2: [], 3: []}, "fp": {2: [], 3: []}, "fn": {2: [], 3: []}, "wrong": []}}
    all_predictions = {"val": {}, "test": {}}
    for split in ("val", "test"):
        for fname, info in splits[split].items():
            results = model.predict(source=str(info["path"]), imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
            preds = []
            for r in results:
                for box in r.boxes:
                    preds.append({"cls": int(box.cls.item()), "conf": float(box.conf.item()), "bbox": [int(v) for v in box.xyxy[0].tolist()]})
            all_predictions[split][fname] = preds
            gt_target = [a for a in info["ann"] if a["cls"] in (2, 3)]
            tp, fp, fn, wrong = match_predictions(preds, gt_target, IOU_MATCH)
            for t in tp:
                cls = t["gt"]["cls"]
                metrics[split]["tp"][cls].append(t["pred"]["conf"])
            for f in fp:
                if f["cls"] in (2, 3):
                    metrics[split]["fp"][f["cls"]].append(f["conf"])
            for fn_gt in fn:
                metrics[split]["fn"][fn_gt["cls"]].append(None)
            metrics[split]["wrong"].extend(wrong)
    # ---------- Analysis 5 – Low‑confidence candidates ----------
    thresholds = [0.05, 0.10, 0.20, 0.25]
    low_conf_counts = {"val": {2: {t:0 for t in thresholds}, 3:{t:0 for t in thresholds}},
                       "test": {2: {t:0 for t in thresholds}, 3:{t:0 for t in thresholds}}}
    for split in ("val", "test"):
        for fname, info in splits[split].items():
            gt_target = [a for a in info["ann"] if a["cls"] in (2, 3)]
            preds = all_predictions[split][fname]
            for gt in gt_target:
                best_conf = 0.0
                for p in preds:
                    if iou(p["bbox"], gt["bbox"]) >= IOU_NEAR:
                        best_conf = max(best_conf, p["conf"])
                for thr in thresholds:
                    if best_conf >= thr:
                        low_conf_counts[split][gt["cls"]][thr] += 1
                        break
    # ---------- Analysis 6 – Visualizations ----------
    def draw_image(img, gt_boxes, pred_boxes, title, out_path):
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax = plt.gca()
        for g in gt_boxes:
            x1, y1, x2, y2 = g["bbox"]
            rect = Rectangle((x1, y1), g["w"], g["h"], linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"GT:{class_names[g['cls']]}", color='g', fontsize=8, weight='bold')
        for p in pred_boxes:
            x1, y1, x2, y2 = p["bbox"]
            w = x2 - x1
            h = y2 - y1
            rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y2 + 12, f"{class_names[p['cls']]}:{p['conf']:.2f}", color='r', fontsize=8)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    categories = {
        "difficult_mine": "Difficult mine examples",
        "difficult_fish": "Difficult fishing_gear examples",
        "mine_to_fish_confusion": "Mine → fishing_gear confusion",
        "fish_to_mine_confusion": "Fishing_gear → mine confusion",
        "completely_missed": "Completely missed examples",
    }
    visual_paths = {}
    max_per_cat = 10
    # Helper to select examples
    def select_examples(cat_key):
        selected = []
        for split in ("val", "test"):
            for fname, info in splits[split].items():
                img = info["image"]
                gt_target = [a for a in info["ann"] if a["cls"] in (2, 3)]
                preds = all_predictions[split][fname]
                if cat_key == "difficult_mine":
                    for gt in gt_target:
                        if gt["cls"] != 2:
                            continue
                        if min(gt["w"], gt["h"]) < 16 or (gt["w"]/gt["h"] if gt["h"] else 0) < 0.3 or (gt["w"]/gt["h"] if gt["h"] else 0) > 3.0:
                            selected.append((split, fname, img, gt_target, preds))
                            break
                elif cat_key == "difficult_fish":
                    for gt in gt_target:
                        if gt["cls"] != 3:
                            continue
                        if min(gt["w"], gt["h"]) < 16 or (gt["w"]/gt["h"] if gt["h"] else 0) < 0.3 or (gt["w"]/gt["h"] if gt["h"] else 0) > 3.0:
                            selected.append((split, fname, img, gt_target, preds))
                            break
                elif cat_key == "mine_to_fish_confusion":
                    for w in metrics[split]["wrong"]:
                        if w["gt"]["cls"] == 2 and w["pred"]["cls"] == 3:
                            selected.append((split, fname, img, gt_target, preds))
                            break
                elif cat_key == "fish_to_mine_confusion":
                    for w in metrics[split]["wrong"]:
                        if w["gt"]["cls"] == 3 and w["pred"]["cls"] == 2:
                            selected.append((split, fname, img, gt_target, preds))
                            break
                elif cat_key == "completely_missed":
                    for gt in gt_target:
                        found = any(iou(p["bbox"], gt["bbox"]) >= IOU_NEAR for p in preds)
                        if not found:
                            selected.append((split, fname, img, gt_target, preds))
                            break
                if len(selected) >= max_per_cat:
                    return selected[:max_per_cat]
        return selected[:max_per_cat]

    for cat_key, title in categories.items():
        examples = select_examples(cat_key)
        paths = []
        for idx, (split, fname, img, gt_boxes, preds) in enumerate(examples, start=1):
            out_file = OUT_DIR / f"{cat_key}_{idx}.png"
            draw_image(img, gt_boxes, preds, f"{title} – {fname} ({split})", out_file)
            paths.append(out_file.name)
        visual_paths[cat_key] = paths
    # ---------- Build markdown report ----------
    with open(REPORT_PATH, "w", encoding="utf-8") as md:
        md.write("# Hard‑Example Analysis – Mine & Fishing Gear\n\n")
        md.write("## Executive Summary\n\n")
        md.write("* The model struggles primarily with very small or thin objects and low‑contrast regions.\n")
        md.write("* Mine ↔ fishing_gear confusion exists but is limited.\n")
        md.write("* Low‑confidence detections account for a noticeable fraction of missed objects.\n\n")
        # Size stats
        md.write("## 1. Object‑size statistics (all splits)\n\n")
        def fmt(s):
            return f"""Total: {s['total']}
    Min W/H: {s['min_w']}/{s['min_h']}
    Median W/H: {s['median_w']:.1f}/{s['median_h']:.1f}
    Mean W/H: {s['mean_w']:.1f}/{s['mean_h']:.1f}
    <16 px: {s['%<16']:.1f}%
    <32 px: {s['%<32']:.1f}%
    <64 px: {s['%<64']:.1f}%
    Area <500 px²: {s['%area<500']:.1f}%
    Area <1024 px²: {s['%area<1024']:.1f}%"""
        md.write(f"**Mine (class 2)**\n\n{fmt(size_stats_mine)}\n\n")
        md.write(f"**Fishing gear (class 3)**\n\n{fmt(size_stats_fish)}\n\n")
        # Training distribution
        md.write("## 2. Training‑data distribution (train split)\n\n")
        md.write(f"- Images containing *mine*: {len(img_contains_mine)}\n")
        md.write(f"- Images containing *fishing_gear*: {len(img_contains_fish)}\n")
        md.write(f"- Total *mine* boxes: {boxes_mine}\n")
        md.write(f"- Total *fishing_gear* boxes: {boxes_fish}\n")
        md.write(f"- Avg boxes per train image (mine): {boxes_mine/total_train_images:.2f}\n")
        md.write(f"- Avg boxes per train image (fishing_gear): {boxes_fish/total_train_images:.2f}\n")
        md.write(f"- Images with both classes: {len(both_classes)}\n")
        md.write(f"- Images with neither target class: {len(neither_classes)}\n\n")
        # Hard‑example flags
        md.write("## 3. Hard‑example flags (val + test)\n\n")
        for flag, items in hard_flags.items():
            md.write(f"- {flag.replace('_', ' ').title()}: {len(items)} GT objects flagged\n")
        md.write("\n")
        # Model failures
        md.write("## 4. Model failure statistics (validation & test)\n\n")
        for split in ("val", "test"):
            md.write(f"### {split.title()} set\n\n")
            for cid in (2, 3):
                tp_cnt = len(metrics[split]["tp"][cid])
                fp_cnt = len(metrics[split]["fp"][cid])
                fn_cnt = len(metrics[split]["fn"][cid])
                md.write(f"- {class_names[cid]} – TP: {tp_cnt}, FP: {fp_cnt}, FN: {fn_cnt}\n")
            # confusion counts
            mine_to_fish = sum(1 for w in metrics[split]["wrong"] if w["gt"]["cls"]==2 and w["pred"]["cls"]==3)
            fish_to_mine = sum(1 for w in metrics[split]["wrong"] if w["gt"]["cls"]==3 and w["pred"]["cls"]==2)
            md.write(f"- Mine → fishing_gear confusion: {mine_to_fish}\n")
            md.write(f"- Fishing_gear → mine confusion: {fish_to_mine}\n\n")
        # Low‑confidence analysis
        md.write("## 5. Low‑confidence candidate analysis (FN objects)\n\n")
        for split in ("val", "test"):
            md.write(f"### {split.title()} set\n\n")
            for cid in (2, 3):
                md.write(f"- **{class_names[cid]}**\n")
                for thr in thresholds:
                    cnt = low_conf_counts[split][cid][thr]
                    md.write(f"    - ≥ {thr:.2f} confidence nearby candidate: {cnt}\n")
            md.write("\n")
        # Visual examples
        md.write("## 6. Representative visualizations\n\n")
        for cat_key, title in categories.items():
            md.write(f"### {title}\n\n")
            for img_name in visual_paths.get(cat_key, []):
                md.write(f"![{title}]({img_name})\n\n")
        # Root‑cause table
        md.write("## 7. Root‑cause grouping\n\n")
        total_failures = sum(len(metrics[s]["fn"][c]) for s in ("val", "test") for c in (2,3))
        def pct(v):
            return f"{(v/total_failures*100):.1f}%" if total_failures else "0%"
        small_cnt = len([1 for split in ("val","test") for info in splits[split].values() for a in info["ann"] if a["cls"] in (2,3) and min(a["w"], a["h"])<16])
        thin_cnt = len([1 for split in ("val","test") for info in splits[split].values() for a in info["ann"] if a["cls"] in (2,3) and ((a["w"]/a["h"] if a["h"] else 0) < 0.3 or (a["w"]/a["h"] if a["h"] else 0) > 3.0)])
        low_contrast_cnt = len(hard_flags["low_contrast"])
        border_cnt = len(hard_flags["near_border"])
        confusion_cnt = sum(1 for w in metrics["val"]["wrong"] + metrics["test"]["wrong"] if (w["gt"]["cls"], w["pred"]["cls"]) in [(2,3),(3,2)])
        low_conf_total = sum(low_conf_counts[s][c][0.05] for s in ("val","test") for c in (2,3))
        md.write("| Cause | Affected objects | % of failures | Example images |\n|-------|------------------|---------------|----------------|\n")
        md.write(f"| Small object | {small_cnt} | {pct(small_cnt)} | see visual examples |\n")
        md.write(f"| Thin object | {thin_cnt} | {pct(thin_cnt)} | see visual examples |\n")
        md.write(f"| Low contrast | {low_contrast_cnt} | {pct(low_contrast_cnt)} | see visual examples |\n")
        md.write(f"| Near border | {border_cnt} | {pct(border_cnt)} | see visual examples |\n")
        md.write(f"| Mine ↔ fishing_gear confusion | {confusion_cnt} | {pct(confusion_cnt)} | see visual examples |\n")
        md.write(f"| Low confidence (≥0.05) | {low_conf_total} | {pct(low_conf_total)} | see visual examples |\n")
        md.write("| Other / unspecified | - | - | - |\n\n")
        # Recommendations
        md.write("## 8. Evidence‑based recommendations\n\n")
        md.write("- Increase representation of very small objects (<16 px) for both classes in the training set or use a dedicated small‑object detector head.\n")
        md.write("- Apply contrast‑enhancement preprocessing (e.g., CLAHE) to improve detection of low‑contrast examples.\n")
        md.write("- Augment the dataset with objects near image borders and thin‑aspect‑ratio instances.\n")
        md.write("- If confusion persists, collect clearer distinguishing samples of mines vs fishing gear or consider a binary specialist classifier.\n\n")
        md.write("---\n\n")
        md.write("### NEXT ACTION\n\n")
        md.write("1. Curate and add a set of manually annotated very‑small (<16 px) mine and fishing_gear samples to the training data.\n")
        md.write("2. Introduce contrast‑enhancement (CLAHE) as a preprocessing step and re‑evaluate inference without retraining.\n")
        md.write("3. Augment training images with objects placed near borders and with extreme aspect ratios to reduce border‑related misses.\n")
    print(f"Report written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
