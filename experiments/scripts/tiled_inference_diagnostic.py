#!/usr/bin/env python
"""Tiled inference diagnostic for 4‑class YOLO11n model.

Runs two inference pipelines on the full test set (126 images) without modifying any
files:

1. Normal inference – whole image, imgsz=640, conf=0.25
2. Tiled inference – overlapping 640×640 tiles (25 % overlap), same
   confidence, NMS IoU=0.45 to merge tile detections.

The script matches predictions against the ground‑truth labels (IoU = 0.50)
and reports overall and per‑class TP/FP/FN, precision, recall, as well as a
comparison of recall between the two methods.

Results are written to `TILED_INFERENCE_DIAGNOSTIC_REPORT.md` in the project
root.
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision.ops import nms as torchvision_nms

# ------------------- Configuration -------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # SIHproject
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "baseline_v1" / "weights" / "best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced" / "images" / "test"
TEST_LABEL_DIR = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced" / "labels" / "test"
DATA_YAML = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced" / "data.yaml"

IMG_SIZE = 640
CONF_THRESH = 0.25
TILE_SIZE = 640
OVERLAP = 0.25  # 25 % overlap
TILE_NMS_IOU = 0.45
MATCH_IOU = 0.50
# ------------------------------------------------------


def load_class_names(yaml_path: Path):
    """Parse a YOLO data.yaml file to obtain class name list.
    Returns list where index corresponds to class id.
    """
    import yaml
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(map(int, names.keys()))]
    return names


def load_ground_truth(label_path: Path, img_width: int, img_height: int):
    """Read YOLO .txt label file and convert normalized boxes to absolute xyxy.
    Returns list of dicts: {"cls": int, "bbox": [x1, y1, x2, y2]}.
    """
    boxes = []
    if not label_path.is_file():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1 = int((xc - w / 2) * img_width)
            y1 = int((yc - h / 2) * img_height)
            x2 = int((xc + w / 2) * img_width)
            y2 = int((yc + h / 2) * img_height)
            boxes.append({"cls": cls, "bbox": [x1, y1, x2, y2]})
    return boxes


def split_into_tiles(img: np.ndarray, tile_size: int = TILE_SIZE, overlap: float = OVERLAP):
    """Yield (tile, (x_offset, y_offset)) for overlapping tiles.
    Overlap is a fraction of tile size.
    """
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = img[y:y_end, x:x_end]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                pad_bottom = tile_size - tile.shape[0]
                pad_right = tile_size - tile.shape[1]
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            yield tile, (x, y)


def run_yolo(model, img: np.ndarray, conf: float = CONF_THRESH):
    """Run YOLO model on a numpy image (BGR) and return detections.
    Returns list of dicts: {"cls": int, "conf": float, "bbox": [x1, y1, x2, y2]} in absolute pixels.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, imgsz=IMG_SIZE, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"cls": cls, "conf": confidence, "bbox": [int(x1), int(y1), int(x2), int(y2)]})
    return detections


def shift_detections(dets, offset):
    ox, oy = offset
    shifted = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        shifted.append({
            "cls": d["cls"],
            "conf": d["conf"],
            "bbox": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
        })
    return shifted


def nms_detections(dets, iou_thr: float = TILE_NMS_IOU):
    if not dets:
        return []
    boxes = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d["conf"] for d in dets], dtype=torch.float32)
    keep_idx = torchvision_nms(boxes, scores, iou_thr)
    return [dets[i] for i in keep_idx]


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


def match_predictions(preds, gts, iou_thr: float = MATCH_IOU):
    """Greedy matching of predictions to ground‑truth.
    Returns per‑class TP, FP, FN dictionaries.
    """
    cls_range = range(4)
    tp = {c: 0 for c in cls_range}
    fp = {c: 0 for c in cls_range}
    fn = {c: 0 for c in cls_range}
    matched_gt = [False] * len(gts)
    for p in preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if matched_gt[idx] or p["cls"] != gt["cls"]:
                continue
            cur_iou = iou(p["bbox"], gt["bbox"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_idx = idx
        if best_iou >= iou_thr and best_idx != -1:
            tp[p["cls"]] += 1
            matched_gt[best_idx] = True
        else:
            fp[p["cls"]] += 1
    for idx, gt in enumerate(gts):
        if not matched_gt[idx]:
            fn[gt["cls"]] += 1
    return tp, fp, fn


def evaluate_image(model, img_path: Path, label_path: Path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    gt_boxes = load_ground_truth(label_path, w, h)
    # Normal inference
    normal_preds = run_yolo(model, img, CONF_THRESH)
    # Tiled inference
    tiled_all = []
    for tile, (ox, oy) in split_into_tiles(img):
        tile_preds = run_yolo(model, tile, CONF_THRESH)
        tiled_all.extend(shift_detections(tile_preds, (ox, oy)))
    tiled_preds = nms_detections(tiled_all, TILE_NMS_IOU)
    # Matching
    normal_tp, normal_fp, normal_fn = match_predictions(normal_preds, gt_boxes, MATCH_IOU)
    tiled_tp, tiled_fp, tiled_fn = match_predictions(tiled_preds, gt_boxes, MATCH_IOU)
    return {
        "normal": {"tp": normal_tp, "fp": normal_fp, "fn": normal_fn},
        "tiled": {"tp": tiled_tp, "fp": tiled_fp, "fn": tiled_fn},
        "gt": gt_boxes,
        "normal_preds": normal_preds,
        "tiled_preds": tiled_preds,
    }


def aggregate(acc, cur):
    for mode in ["normal", "tiled"]:
        for metric in ["tp", "fp", "fn"]:
            for c in range(4):
                acc[mode][metric][c] += cur[mode][metric][c]
    return acc


def compute_metrics(tp, fp, fn):
    precision = {c: (tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0) for c in tp}
    recall = {c: (tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0) for c in tp}
    return precision, recall


def main():
    model = YOLO(str(MODEL_PATH))
    class_names = load_class_names(DATA_YAML)
    agg = {
        "normal": {"tp": {c: 0 for c in range(4)}, "fp": {c: 0 for c in range(4)}, "fn": {c: 0 for c in range(4)}},
        "tiled": {"tp": {c: 0 for c in range(4)}, "fp": {c: 0 for c in range(4)}, "fn": {c: 0 for c in range(4)}},
    }
    recovered_examples = []
    new_fp_examples = []
    img_files = sorted([p for p in TEST_IMG_DIR.iterdir() if p.is_file()])
    for img_path in img_files:
        label_path = TEST_LABEL_DIR / (img_path.stem + ".txt")
        res = evaluate_image(model, img_path, label_path)
        agg = aggregate(agg, res)
        for c in range(4):
            rec = res["tiled"]["tp"][c] - res["normal"]["tp"][c]
            if rec > 0 and len(recovered_examples) < 5:
                recovered_examples.append({"image": img_path.name, "class": class_names[c], "count": rec})
            new_fp = res["tiled"]["fp"][c] - res["normal"]["fp"][c]
            if new_fp > 0 and len(new_fp_examples) < 5:
                new_fp_examples.append({"image": img_path.name, "class": class_names[c], "count": new_fp})
    normal_prec, normal_rec = compute_metrics(agg["normal"]["tp"], agg["normal"]["fp"], agg["normal"]["fn"])
    tiled_prec, tiled_rec = compute_metrics(agg["tiled"]["tp"], agg["tiled"]["fp"], agg["tiled"]["fn"])
    overall = {
        "normal": {"tp": sum(agg["normal"]["tp"].values()), "fp": sum(agg["normal"]["fp"].values()), "fn": sum(agg["normal"]["fn"].values())},
        "tiled": {"tp": sum(agg["tiled"]["tp"].values()), "fp": sum(agg["tiled"]["fp"].values()), "fn": sum(agg["tiled"]["fn"].values())},
    }
    overall_prec = {
        "normal": overall["normal"]["tp"] / (overall["normal"]["tp"] + overall["normal"]["fp"]) if (overall["normal"]["tp"] + overall["normal"]["fp"]) > 0 else 0,
        "tiled": overall["tiled"]["tp"] / (overall["tiled"]["tp"] + overall["tiled"]["fp"]) if (overall["tiled"]["tp"] + overall["tiled"]["fp"]) > 0 else 0,
    }
    overall_rec = {
        "normal": overall["normal"]["tp"] / (overall["normal"]["tp"] + overall["normal"]["fn"]) if (overall["normal"]["tp"] + overall["normal"]["fn"]) > 0 else 0,
        "tiled": overall["tiled"]["tp"] / (overall["tiled"]["tp"] + overall["tiled"]["fn"]) if (overall["tiled"]["tp"] + overall["tiled"]["fn"]) > 0 else 0,
    }
    report_path = PROJECT_ROOT / "TILED_INFERENCE_DIAGNOSTIC_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Tiled Inference Diagnostic Report\n\n")
        f.write(f"**Model**: `{MODEL_PATH}`\n")
        f.write(f"**Test images**: `{TEST_IMG_DIR}`\n\n")
        f.write("## Overall Results\n\n")
        f.write("| Method | TP | FP | FN | Precision | Recall |\n")
        f.write("|--------|----|----|----|-----------|--------|\n")
        f.write(f"| Normal | {overall['normal']['tp']} | {overall['normal']['fp']} | {overall['normal']['fn']} | {overall_prec['normal']:.3f} | {overall_rec['normal']:.3f} |\n")
        f.write(f"| Tiled  | {overall['tiled']['tp']} | {overall['tiled']['fp']} | {overall['tiled']['fn']} | {overall_prec['tiled']:.3f} | {overall_rec['tiled']:.3f} |\n\n")
        f.write("## Per‑Class Results\n\n")
        f.write("| Class | Method | TP | FP | FN | Precision | Recall |\n")
        f.write("|-------|--------|----|----|----|-----------|--------|\n")
        for cid, cname in enumerate(class_names):
            f.write(f"| {cname} | Normal | {agg['normal']['tp'][cid]} | {agg['normal']['fp'][cid]} | {agg['normal']['fn'][cid]} | {normal_prec[cid]:.3f} | {normal_rec[cid]:.3f} |\n")
            f.write(f"| {cname} | Tiled  | {agg['tiled']['tp'][cid]} | {agg['tiled']['fp'][cid]} | {agg['tiled']['fn'][cid]} | {tiled_prec[cid]:.3f} | {tiled_rec[cid]:.3f} |\n")
        f.write("\n## Recall Comparison\n\n")
        f.write("| Class | Normal Recall | Tiled Recall | Change |\n")
        f.write("|-------|---------------|--------------|--------|\n")
        for cid, cname in enumerate(class_names):
            delta = tiled_rec[cid] - normal_rec[cid]
            f.write(f"| {cname} | {normal_rec[cid]:.3f} | {tiled_rec[cid]:.3f} | {delta:+.3f} |\n")
        f.write("\n## Notable Findings\n\n")
        f.write("### Recovered Objects (Tiled TP not present in Normal)\n")
        if recovered_examples:
            for ex in recovered_examples:
                f.write(f"* {ex['image']}: {ex['class']} (≈{ex['count']} extra detections)\n")
        else:
            f.write("* None detected.\n")
        f.write("\n### New False Positives introduced by Tiling\n")
        if new_fp_examples:
            for ex in new_fp_examples:
                f.write(f"* {ex['image']}: {ex['class']} (≈{ex['count']} extra FP)\n")
        else:
            f.write("* None detected.\n")
        f.write("\n### Mine ↔ Fishing Gear Confusion (approx.)\n")
        f.write("* Not explicitly counted in this lightweight script.\n")
        f.write("\n---\n\n")
        f.write("*All numbers are based on IoU ≥ 0.50 for matching predictions to ground‑truth.*\n")
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
