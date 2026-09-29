import os
import json
from pathlib import Path

# ==== CONFIGURATION ====
PROJECT_ROOT = Path(r"C:/Users/Jhansi/OneDrive/Desktop/SIHproject")
TEST_LABEL_DIR = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced" / "labels" / "test"
PREDICTIONS_PATH = Path(r"C:/Users/Jhansi/.gemini/antigravity/brain/eb7941b6-48a3-4926-89a3-e91fbae9a9da/scratch/per_class_raw.json")

# Class mappings (YOLO id -> name)
ID2NAME = {"0": "shipwreck", "1": "aircraft", "2": "mine", "3": "fishing_gear"}

# Thresholds (can be tweaked later)
LOW_CONF_THRESH = 0.25          # confidence below which a detection is considered low‑confidence
SMALL_AREA_REL = 0.01           # relative area (<1% of image) considered a very small/thin object

# ------------------------------------------------------------
# Helper: load ground‑truth objects for a given image base name
def load_gt(base_name: str):
    txt_path = TEST_LABEL_DIR / f"{base_name}.txt"
    if not txt_path.is_file():
        return []
    objs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, xc, yc, w, h = parts[:5]
            objs.append({
                "class_id": cls_id,
                "class_name": ID2NAME.get(cls_id, "unknown"),
                "bbox": [float(xc), float(yc), float(w), float(h)],
                "area": float(w) * float(h),
            })
    return objs

# Helper: robustly load large JSON predictions (UTF‑8 first, fallback UTF‑16)
def load_predictions():
    try:
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(PREDICTIONS_PATH, "r", encoding="utf-16") as f:
            data = json.load(f)
    pred_dict = {}
    for entry in data:
        # image identifier may appear under different keys
        img_key = entry.get("image") or entry.get("image_path") or entry.get("filename")
        if not img_key:
            continue
        base = Path(img_key).stem
        detections = entry.get("response", {}).get("detections", [])
        preds = []
        for det in detections:
            cls = det.get("object_class") or det.get("class")
            # Normalise to class name string
            if isinstance(cls, int):
                cls_name = ID2NAME.get(str(cls), "unknown")
            else:
                cls_name = str(cls)
            confidence = det.get("confidence", det.get("score", 0.0))
            preds.append({"class_name": cls_name, "confidence": float(confidence)})
        pred_dict[base] = preds
    return pred_dict

# ------------------------------------------------------------
# Main analysis – read‑only, no model or data changes
predictions = load_predictions()

# Containers for report rows and aggregated stats
report_rows = []
stats = {
    "missed": 0,
    "low_conf_missed": 0,
    "small_missed": 0,
    "wrong_class": 0,
    "confusion": {},
    "missed_per_class": {},
}

for txt_file in TEST_LABEL_DIR.glob("*.txt"):
    base = txt_file.stem
    gt_objs = load_gt(base)
    gt_classes = {o["class_name"] for o in gt_objs}
    pred_objs = predictions.get(base, [])
    pred_classes = {p["class_name"] for p in pred_objs}

    # ----- Missed detections -----
    for gt in gt_objs:
        gt_name = gt["class_name"]
        # any prediction of same class (any confidence) counts as detected
        detected = any(p["class_name"] == gt_name for p in pred_objs)
        if detected:
            continue  # correct detection, nothing to report
        # Missed!
        stats["missed"] += 1
        stats["missed_per_class"].setdefault(gt_name, 0)
        stats["missed_per_class"][gt_name] += 1
        # low‑confidence check (prediction of same class exists but below threshold)
        low_conf = any(p["class_name"] == gt_name and p["confidence"] < LOW_CONF_THRESH for p in pred_objs)
        if low_conf:
            stats["low_conf_missed"] += 1
        # small / thin object check (area relative to image – we lack image size, use absolute area heuristic)
        # Assuming normalized coordinates (0‑1), area < SMALL_AREA_REL is very small.
        if gt["area"] < SMALL_AREA_REL:
            stats["small_missed"] += 1
        reason_parts = []
        if low_conf:
            reason_parts.append("low confidence (<0.25)")
        if gt["area"] < SMALL_AREA_REL:
            reason_parts.append("very small/thin object")
        reason = ", ".join(reason_parts) or "missed"
        report_rows.append({
            "image": f"{base}.jpg",
            "gt": gt_name,
            "pred": "-",
            "conf": "-",
            "status": "Missed",
            "reason": reason,
        })

    # ----- Wrong‑class / confusion detections -----
    for pred in pred_objs:
        pred_name = pred["class_name"]
        conf = pred["confidence"]
        if pred_name in gt_classes:
            continue  # already counted as correct
        # false positive or confusion
        stats["wrong_class"] += 1
        # For each ground‑truth class present, record confusion direction
        for gt_name in gt_classes:
            key = f"{pred_name}→{gt_name}"
            stats["confusion"].setdefault(key, 0)
            stats["confusion"][key] += 1
        report_rows.append({
            "image": f"{base}.jpg",
            "gt": ", ".join(sorted(gt_classes)) or "none",
            "pred": pred_name,
            "conf": f"{conf:.2f}",
            "status": "Wrong class",
            "reason": "confusion / false positive",
        })

# ------------------------------------------------------------
# Build markdown report (concise as requested)
md_lines = []
md_lines.append("# Read‑only Failure Analysis Report\n")
md_lines.append("| Image | Ground‑truth | Predicted | Confidence | Status | Reason |")
md_lines.append("|-------|--------------|-----------|------------|--------|--------|")
for r in report_rows:
    md_lines.append(f"| {r['image']} | {r['gt']} | {r['pred']} | {r['conf']} | {r['status']} | {r['reason']} |")

md_lines.append("\n## Summary of Failures\n")
md_lines.append(f"* Total missed detections: {stats['missed']}\n")
md_lines.append(f"* Missed due to low confidence: {stats['low_conf_missed']}\n")
md_lines.append(f"* Missed small/thin objects: {stats['small_missed']}\n")
md_lines.append(f"* Wrong‑class / confusion detections: {stats['wrong_class']}\n")
md_lines.append("\n### Missed per class\n")
for cls, cnt in stats["missed_per_class"].items():
    md_lines.append(f"- {cls}: {cnt}\n")
md_lines.append("\n### Confusion matrix (predicted → ground‑truth)\n")
for key, cnt in stats["confusion"].items():
    md_lines.append(f"- {key}: {cnt}\n")

# Top‑3 likely causes (simple heuristic based on counts)
causes = []
if stats["low_conf_missed"] > 0:
    causes.append("low‑confidence threshold too high")
if stats["small_missed"] > 0:
    causes.append("small/thin objects not well represented")
if stats["wrong_class"] > 0:
    causes.append("class confusion (mine ↔ fishing_gear etc.)")
if not causes:
    causes.append("model quality – likely needs more training data or better augmentation")
md_lines.append("\n## Likely Primary Causes (top 3)\n")
for i, c in enumerate(causes[:3], 1):
    md_lines.append(f"{i}. {c}\n")

# Evidence‑based recommendation (no code changes yet)
md_lines.append("\n## Recommended Fixes (read‑only suggestions)\n")
md_lines.append("- **Dataset**: add more examples of the missed classes, especially small mines and fishing‑gear in shadowed/low‑contrast sonar backgrounds.\n")
md_lines.append("- **Confidence threshold**: consider lowering the inference threshold (e.g., from 0.3 to 0.2) to capture low‑confidence detections, then apply NMS/post‑processing filtering.\n")
md_lines.append("- **Training**: if many small objects are missed, increase the input image size (e.g., `imgsz=800`) and enable mosaic/scale augmentations to improve small‑object recall.\n")
md_lines.append("- **Preprocessing**: verify that the same sonar enhancement steps used during training (contrast stretching, denoising) are applied during inference.\n")
md_lines.append("- **Backend**: ensure the inference wrapper returns all detections (no accidental filtering) and that the confidence cutoff matches the analysis threshold.\n")

report_path = Path(r"C:/Users/Jhansi/.gemini/antigravity/brain/eb7941b6-48a3-4926-89a3-e91fbae9a9da/readonly_failure_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("Report written to", report_path)
