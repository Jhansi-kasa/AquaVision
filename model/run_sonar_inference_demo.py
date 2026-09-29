import os
import sys
import argparse
import cv2
import numpy as np

# CRITICAL for Windows + PyTorch: Load torchvision before ultralytics
import torch
import torchvision
from ultralytics import YOLO

CLASSES = ["Shipwreck", "Aircraft", "Mine", "Fishing Gear"]
CLASS_COLORS = {
    0: (0, 0, 255),      # Shipwreck: Red
    1: (255, 165, 0),    # Aircraft: Orange
    2: (0, 255, 255),    # Mine: Yellow (Acoustic Hazard)
    3: (0, 255, 0)       # Fishing Gear: Green
}

DEFAULT_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = (
    os.path.join(DEFAULT_PKG_DIR, "aqua_vision_100ep_best.pt")
    if os.path.exists(os.path.join(DEFAULT_PKG_DIR, "aqua_vision_100ep_best.pt"))
    else os.path.join(DEFAULT_PKG_DIR, "best.pt")
)
DEFAULT_WEIGHTS_ENS = (
    os.path.join(DEFAULT_PKG_DIR, "best_100epochs.pt")
    if os.path.exists(os.path.join(DEFAULT_PKG_DIR, "best_100epochs.pt"))
    else DEFAULT_WEIGHTS
)

OPTIMAL_THRESHOLDS = {0: 0.25, 1: 0.21, 2: 0.10, 3: 0.16}
OPTIMAL_THRESHOLDS_ENSEMBLE = {0: 0.27, 1: 0.43, 2: 0.09, 3: 0.19}

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union if union > 0 else 0.0

def apply_sonar_clahe(img_bgr):
    """Enhance acoustic shadow-highlight contrast via CLAHE in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def ensemble_wbf(b1, b2, iou_thresh=0.50, agreement_boost=1.20):
    final_boxes = []
    for cid in [0, 1, 2, 3]:
        c_b1 = [b for b in b1 if b[0] == cid]
        c_b2 = [b for b in b2 if b[0] == cid]
        matched2 = [False] * len(c_b2)
        for box1 in c_b1:
            score1, coord1 = box1[1], box1[2:]
            best_iou = 0.0
            best_idx2 = -1
            for idx2, box2 in enumerate(c_b2):
                if not matched2[idx2]:
                    iou = compute_iou(coord1, box2[2:])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx2 = idx2
            if best_iou >= iou_thresh and best_idx2 >= 0:
                matched2[best_idx2] = True
                box2 = c_b2[best_idx2]
                score2, coord2 = box2[1], box2[2:]
                w1 = score1 / (score1 + score2)
                w2 = score2 / (score1 + score2)
                fused_box = [w1*c1 + w2*c2 for c1, c2 in zip(coord1, coord2)]
                fused_score = min(1.0, max(score1, score2) * agreement_boost)
                final_boxes.append([cid, fused_score] + fused_box)
            else:
                final_boxes.append(box1)
        for idx2, box2 in enumerate(c_b2):
            if not matched2[idx2]:
                final_boxes.append(box2)
    return final_boxes

def run_demo(source_path, weights_path=DEFAULT_WEIGHTS, weights2_path=DEFAULT_WEIGHTS_ENS,
             mode="optimal", conf_thresh=0.15, iou_thresh=0.45, use_clahe=False, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_PKG_DIR, "inference_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file not found: {weights_path}")
        return

    print(f"Loading Primary Model from: {weights_path}")
    model1 = YOLO(weights_path)
    
    model2 = None
    if mode == "ensemble":
        if os.path.exists(weights2_path):
            print(f"Loading Ensemble Secondary Model from: {weights2_path}")
            model2 = YOLO(weights2_path)
        else:
            print(f"[WARNING] Ensemble secondary weights not found at {weights2_path}. Falling back to optimal single mode.")
            mode = "optimal"
            
    if os.path.isdir(source_path):
        image_files = [os.path.join(source_path, f) for f in os.listdir(source_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        image_files = [source_path]

    print(f"Running Inference on {len(image_files)} image(s) [Mode: {mode.upper()}, CLAHE={use_clahe}]...")
    
    total_detections = 0
    for img_path in image_files:
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        
        infer_img = apply_sonar_clahe(orig_img) if use_clahe else orig_img
        
        # Inference
        if mode == "ensemble":
            r1 = model1.predict(source=infer_img, conf=0.01, imgsz=512, verbose=False)[0]
            b1 = [[int(b.cls[0].item()), float(b.conf[0].item())] + b.xyxy[0].cpu().numpy().tolist() for b in r1.boxes]
            r2 = model2.predict(source=infer_img, conf=0.01, imgsz=512, verbose=False)[0]
            b2 = [[int(b.cls[0].item()), float(b.conf[0].item())] + b.xyxy[0].cpu().numpy().tolist() for b in r2.boxes]
            fused = ensemble_wbf(b1, b2, iou_thresh=0.50, agreement_boost=1.20)
            preds = [p for p in fused if p[1] >= OPTIMAL_THRESHOLDS_ENSEMBLE[p[0]]]
        elif mode == "optimal":
            r1 = model1.predict(source=infer_img, conf=0.01, imgsz=512, verbose=False)[0]
            raw = [[int(b.cls[0].item()), float(b.conf[0].item())] + b.xyxy[0].cpu().numpy().tolist() for b in r1.boxes]
            preds = [p for p in raw if p[1] >= OPTIMAL_THRESHOLDS[p[0]]]
        else: # standard
            r1 = model1.predict(source=infer_img, conf=conf_thresh, iou=iou_thresh, imgsz=512, verbose=False)[0]
            preds = [[int(b.cls[0].item()), float(b.conf[0].item())] + b.xyxy[0].cpu().numpy().tolist() for b in r1.boxes]
            
        annotated_img = orig_img.copy()
        for p in preds:
            cls_id, conf = p[0], p[1]
            xyxy = [int(v) for v in p[2:]]
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class_{cls_id}"
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            
            cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (xyxy[0], max(0, xyxy[1] - 20)), (xyxy[0] + lw, xyxy[1]), color, -1)
            cv2.putText(annotated_img, label, (xyxy[0], max(0, xyxy[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            total_detections += 1

        out_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"detected_{out_name}")
        cv2.imwrite(save_path, annotated_img)
        print(f"Saved: {save_path} ({len(preds)} anomalies detected)")

    print(f"\n[DONE] Finished processing! Total anomalies detected: {total_detections}")
    print(f"Results saved in folder: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aqua Vision Side-Scan Sonar Object Detection Demo")
    parser.add_argument("--source", type=str, 
                        default=os.path.join(DEFAULT_PKG_DIR, "sample_predictions"),
                        help="Path to an image or directory of sonar images")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--weights2", type=str, default=DEFAULT_WEIGHTS_ENS,
                        help="Path to secondary weights for ensemble (.pt)")
    parser.add_argument("--mode", type=str, default="optimal", choices=["optimal", "standard", "ensemble"],
                        help="Inference mode: 'optimal' (adaptive thresholds), 'standard' (fixed conf), 'ensemble' (dual WBF)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Fixed confidence threshold (for standard mode)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--clahe", action="store_true",
                        help="Apply CLAHE contrast enhancement")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save visual detection results")
    args = parser.parse_args()

    run_demo(args.source, args.weights, args.weights2, args.mode, args.conf, args.iou, args.clahe, args.output)
