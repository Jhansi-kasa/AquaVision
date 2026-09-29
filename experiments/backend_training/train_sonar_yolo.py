# Marine Debris Sonar YOLOv8 Retraining Pipeline
import os
import sys
import time
import json
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

CLASSES = {
    0: "shipwreck",
    1: "drowning_victim",
    2: "aircraft",
    3: "mine",
    4: "seafloor",
    5: "crab_pot"
}

BASELINE_METRICS = {
    "overall": {"precision": 0.0619, "recall": 0.1247, "mAP50": 0.0264, "mAP50_95": 0.0079},
    "shipwreck": {"precision": 0.0072, "recall": 0.0169, "mAP50": 0.0003},
    "drowning_victim": {"precision": 0.0000, "recall": 0.0000, "mAP50": 0.0000},
    "aircraft": {"precision": 0.0118, "recall": 0.2000, "mAP50": 0.0033},
    "mine": {"precision": 1.0000, "recall": 0.0000, "mAP50": 0.0000},
    "seafloor": {"precision": 1.0000, "recall": 0.0000, "mAP50": 0.0050},
    "crab_pot": {"precision": 0.0000, "recall": 0.0000, "mAP50": 0.0000},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Marine Sonar Dataset")
    parser.add_argument("--data", type=str, default=r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject\final_ai_ready_dataset\data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights or architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default 16 for CPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device ('0' for CUDA or 'cpu')")
    parser.add_argument("--project", type=str, default=r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject\backend\yolo\marine_sonar_v2", help="Save project directory")
    parser.add_argument("--name", type=str, default="train", help="Run name")
    parser.add_argument("--eval_only", type=str, default=None, help="Skip training and evaluate given weights path")
    return parser.parse_args()


def get_device(requested_device):
    if requested_device is not None:
        return requested_device
    if torch.cuda.is_available():
        print(f"[HARDWARE] NVIDIA CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        return "0"
    print("[HARDWARE] CUDA unavailable. Utilizing CPU execution.")
    return "cpu"


def train_model(args):
    device = get_device(args.device)
    data_path = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING YOLOv8 MARINE SONAR RETRAINING PIPELINE")
    print("=" * 70)
    print(f"Dataset config : {data_path}")
    print(f"Base model     : {args.model}")
    print(f"Target epochs  : {args.epochs}")
    print(f"Patience       : {args.patience}")
    print(f"Batch size     : {args.batch}")
    print(f"Resolution     : {args.imgsz}")
    print(f"Device         : {device}")
    print(f"Project output : {project_dir}")
    print("=" * 70)

    model = YOLO(args.model)

    start_time = time.time()
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.5,
        mixup=0.0,
        scale=0.2,
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.0,
        hsv_v=0.4,
        cls=1.0,
        box=7.5,
        dfl=1.5,
        cos_lr=True,
        warmup_epochs=3.0,
        val=True,
        save=True,
        save_period=5,
        verbose=True
    )
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f}s ({elapsed_time / 60:.2f} minutes).")

    best_weights = project_dir / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = project_dir / args.name / "weights" / "last.pt"

    print(f"[CHECKPOINT] Best weights saved at: {best_weights}")
    return best_weights, model


def run_full_validation(best_weights_path, data_yaml_path):
    print("\n" + "=" * 70)
    print("RUNNING POST-TRAINING VALIDATION")
    print("=" * 70)
    model = YOLO(str(best_weights_path))

    print("\n--- VALIDATION SET EVALUATION ---")
    val_metrics = model.val(data=str(data_yaml_path), split="val", imgsz=640, batch=16, verbose=True)
    
    print("\n--- TEST SET EVALUATION ---")
    test_metrics = model.val(data=str(data_yaml_path), split="test", imgsz=640, batch=16, verbose=True)

    return val_metrics, test_metrics


def test_problem_images(best_weights_path):
    print("\n" + "=" * 70)
    print("TESTING SPECIFIC PROBLEM IMAGES")
    print("=" * 70)
    model = YOLO(str(best_weights_path))

    problem_images = [
        {
            "name": "seabed_000003 (Problem Aircraft)",
            "path": r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject\final_ai_ready_dataset\images\test\seabed_000003_jpg.rf.c2e711409404577d12e062dab5b5c4d9.jpg",
            "expected": "aircraft"
        },
        {
            "name": "seabed_000199 (Problem Mine)",
            "path": r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject\final_ai_ready_dataset\images\test\seabed_000199_jpg.rf.618c41b7d1c1585e0f1ea946038224f8.jpg",
            "expected": "mine"
        },
        {
            "name": "raw_sonar_test (1).jpg (Known Positive Aircraft)",
            "path": r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject\backend\uploads\raw_sonar_test (1).jpg",
            "expected": "aircraft"
        }
    ]

    results_table = []
    for item in problem_images:
        img_path = item["path"]
        if not os.path.exists(img_path):
            print(f"[WARN] File not found: {img_path}")
            continue

        for conf in [0.10, 0.25, 0.50]:
            preds = model.predict(source=img_path, conf=conf, imgsz=640, verbose=False)
            boxes = preds[0].boxes
            detections = []
            matched = False
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls.item())
                    cls_name = CLASSES.get(cls_id, f"class_{cls_id}")
                    score = float(b.conf.item())
                    xyxy = [round(x, 1) for x in b.xyxy[0].tolist()]
                    detections.append(f"{cls_name} ({score:.2f}) {xyxy}")
                    if cls_name == item["expected"]:
                        matched = True
            
            results_table.append({
                "sample": item["name"],
                "expected": item["expected"],
                "conf_thresh": conf,
                "num_detections": len(detections),
                "detections": ", ".join(detections) if detections else "None",
                "matched_expected": matched
            })
            print(f"[{item['name']}] conf={conf:.2f} -> {len(detections)} boxes: {', '.join(detections) if detections else 'None'} (Target matched: {matched})")

    return results_table


def evaluate_mine_test_set(best_weights_path):
    print("\n" + "=" * 70)
    print("EXTENSIVE MINE TEST-SET EVALUATION (74 TEST MINE SAMPLES)")
    print("=" * 70)
    model = YOLO(str(best_weights_path))

    labels_dir = Path(r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject\final_ai_ready_dataset\labels\test")
    images_dir = Path(r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject\final_ai_ready_dataset\images\test")
    
    mine_images = []
    total_mine_gt_boxes = 0
    for lbl_file in labels_dir.glob("*.txt"):
        with open(lbl_file, "r") as f:
            lines = f.readlines()
        mine_count = sum(1 for line in lines if line.strip().startswith("3 "))
        if mine_count > 0:
            img_file = images_dir / f"{lbl_file.stem}.jpg"
            if img_file.exists():
                mine_images.append((img_file, mine_count))
                total_mine_gt_boxes += mine_count

    print(f"Total test images containing mine: {len(mine_images)}")
    print(f"Total ground-truth mine bounding boxes: {total_mine_gt_boxes}")

    correct_mine_images = 0
    detected_mine_boxes = 0
    false_positives = 0

    for img_path, gt_count in mine_images:
        preds = model.predict(source=str(img_path), conf=0.25, imgsz=640, verbose=False)
        boxes = preds[0].boxes
        img_mine_detections = 0
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls.item())
                if cls_id == 3:
                    img_mine_detections += 1
                else:
                    false_positives += 1
        if img_mine_detections > 0:
            correct_mine_images += 1
            detected_mine_boxes += img_mine_detections

    recall = detected_mine_boxes / total_mine_gt_boxes if total_mine_gt_boxes > 0 else 0.0
    print(f"Test Mine Recall @ conf=0.25: {recall:.4f} ({detected_mine_boxes}/{total_mine_gt_boxes} boxes across {correct_mine_images}/{len(mine_images)} images)")
    return {
        "total_mine_images": len(mine_images),
        "correctly_detected_images": correct_mine_images,
        "missed_images": len(mine_images) - correct_mine_images,
        "total_gt_boxes": total_mine_gt_boxes,
        "detected_mine_boxes": detected_mine_boxes,
        "recall": recall,
        "non_mine_fps": false_positives
    }


def main():
    args = parse_args()
    if args.eval_only:
        best_weights = Path(args.eval_only)
    else:
        best_weights, _ = train_model(args)
    
    val_results, test_results = run_full_validation(best_weights, args.data)
    problem_results = test_problem_images(best_weights)
    mine_results = evaluate_mine_test_set(best_weights)

    print("\n" + "=" * 70)
    print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
