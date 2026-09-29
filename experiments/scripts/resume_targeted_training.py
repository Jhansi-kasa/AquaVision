import os
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject")
LAST_PT = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "fishing_gear_targeted_v1" / "weights" / "last.pt"

def main():
    print("=" * 70)
    print("RESUMING TARGETED TRAINING RUN: fishing_gear_targeted_v1")
    print(f"Checkpoint : {LAST_PT}")
    print(f"Target     : 100 epochs (patience=20)")
    print(f"Device     : CPU (threads={torch.get_num_threads()})")
    print("=" * 70)
    
    if not LAST_PT.exists():
        sys.exit(f"Error: checkpoint not found at {LAST_PT}")
        
    model = YOLO(str(LAST_PT))
    t0 = time.time()
    results = model.train(resume=True)
    elapsed = time.time() - t0
    
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

if __name__ == "__main__":
    main()
