import os
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject")
DATA_YAML = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "fishing_gear_targeted_v1" / "preprocessed_dataset" / "data_preprocessed.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "4class_training"
RUN_NAME = "fishing_gear_targeted_v1"

# Hyperparameters identical to baseline_v1
TRAIN_KWARGS = dict(
    data=str(DATA_YAML),
    project=str(OUTPUT_DIR),
    name=RUN_NAME,
    exist_ok=True,
    epochs=5,
    patience=20,
    batch=16,
    imgsz=640,
    device="cpu",
    workers=0,
    amp=True,
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
    cos_lr=True,
    warmup_epochs=3.0,
    cls=1.0,
    box=7.5,
    dfl=1.5,
    val=True,
    save=True,
    save_period=1,
    verbose=True,
)

def main():
    print("=" * 70)
    print("STARTING 5-EPOCH TARGETED TRAINING PILOT")
    print(f"Data YAML : {DATA_YAML}")
    print(f"Output    : {OUTPUT_DIR / RUN_NAME}")
    print(f"Device    : CPU (threads={torch.get_num_threads()})")
    print("=" * 70)
    
    # Load pretrained YOLO11n
    model = YOLO("yolo11n.pt")
    
    t0 = time.time()
    results = model.train(**TRAIN_KWARGS)
    elapsed = time.time() - t0
    
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

if __name__ == "__main__":
    main()
