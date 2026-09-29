"""
Train fishing_gear_targeted_v1 for exactly 15 more epochs from the 5-epoch pilot best.pt,
producing epoch 6–20 (cumulative). Report results at the end.

Strategy: Start fresh from best.pt of pilot (epoch5), train for 15 epochs,
save under the same project/name so results.csv accumulates.
"""
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject")
PILOT_BEST  = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "fishing_gear_targeted_v1" / "weights" / "epoch4.pt"
DATA_YAML   = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "fishing_gear_targeted_v1" / "preprocessed_dataset" / "data_preprocessed.yaml"
OUTPUT_DIR  = PROJECT_ROOT / "runs" / "detect" / "4class_training"
RUN_NAME    = "fishing_gear_targeted_v1"
EPOCHS      = 15          # 15 more = cumulative epoch 20 from scratch

TRAIN_KWARGS = dict(
    data        = str(DATA_YAML),
    project     = str(OUTPUT_DIR),
    name        = RUN_NAME,
    exist_ok    = True,
    epochs      = EPOCHS,
    patience    = 20,
    batch       = 16,
    imgsz       = 640,
    device      = "cpu",
    workers     = 0,
    amp         = True,
    fliplr      = 0.5,
    flipud      = 0.0,
    degrees     = 0.0,
    shear       = 0.0,
    perspective = 0.0,
    mosaic      = 0.5,
    mixup       = 0.0,
    scale       = 0.2,
    translate   = 0.1,
    hsv_h       = 0.015,
    hsv_s       = 0.0,
    hsv_v       = 0.4,
    cos_lr      = True,
    warmup_epochs = 3.0,
    cls         = 1.0,
    box         = 7.5,
    dfl         = 1.5,
    val         = True,
    save        = True,
    save_period = 5,
    verbose     = True,
    seed        = 0,
    deterministic = True,
)

def main():
    print("=" * 70)
    print(f"TARGETED TRAINING — EPOCH 6-20 (15 more epochs)")
    print(f"Start checkpoint : {PILOT_BEST}")
    print(f"Data YAML        : {DATA_YAML}")
    print(f"Output           : {OUTPUT_DIR / RUN_NAME}")
    print(f"Epochs this run  : {EPOCHS}")
    print(f"Device           : CPU (threads={torch.get_num_threads()})")
    print("=" * 70)

    if not PILOT_BEST.exists():
        sys.exit(f"ERROR: checkpoint not found → {PILOT_BEST}")

    model = YOLO(str(PILOT_BEST))
    t0 = time.time()
    model.train(**TRAIN_KWARGS)
    elapsed = time.time() - t0
    print(f"\nCompleted {EPOCHS} epochs in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Final validation report
    print("\n" + "=" * 70)
    print("FINAL VALIDATION REPORT (epoch-20 checkpoint)")
    print("=" * 70)
    best_pt = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    m = YOLO(str(best_pt)).val(
        data    = str(DATA_YAML),
        split   = "val",
        imgsz   = 640,
        batch   = 16,
        device  = "cpu",
        verbose = False,
        plots   = False,
    )
    cnames = {0:"shipwreck", 1:"aircraft", 2:"mine", 3:"fishing_gear"}
    print(f"Overall  P={m.box.mp:.4f}  R={m.box.mr:.4f}  mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")
    for i, cid in enumerate(m.box.ap_class_index):
        cn = cnames[int(cid)]
        print(f"  {cn:<15}  P={m.box.p[i]:.4f}  R={m.box.r[i]:.4f}  mAP50={m.box.ap50[i]:.4f}  mAP50-95={m.box.ap[i]:.4f}")

if __name__ == "__main__":
    main()
