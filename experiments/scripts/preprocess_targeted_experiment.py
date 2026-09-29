import sys
import shutil
import time
from pathlib import Path
import cv2
import yaml

PROJECT_ROOT = Path(r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject")
CV_DIR = PROJECT_ROOT / "computer_vision"
if str(CV_DIR) not in sys.path:
    sys.path.insert(0, str(CV_DIR))

from final_preprocessing_pipeline import process_sonar_image, get_default_config

SRC_RAW_DATASET = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced_FishingTargeted"
BASELINE_PREPROCESSED = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "preprocessed_dataset"
EXP_ROOT = PROJECT_ROOT / "runs" / "detect" / "4class_training" / "fishing_gear_targeted_v1"
DST_PREPROCESSED = EXP_ROOT / "preprocessed_dataset"

def main():
    print("=== PREPROCESSING TARGETED DATASET (4-STAGE SONAR PIPELINE) ===")
    
    if DST_PREPROCESSED.exists():
        print(f"Cleaning {DST_PREPROCESSED}...")
        shutil.rmtree(DST_PREPROCESSED)
        
    for split in ["train", "val", "test"]:
        (DST_PREPROCESSED / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_PREPROCESSED / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    # Copy val and test directly from baseline preprocessed (guarantees 100% bit-exact consistency)
    for split in ["val", "test"]:
        print(f"Copying preprocessed {split} split from baseline...")
        for img in (BASELINE_PREPROCESSED / "images" / split).glob("*.*"):
            shutil.copy2(img, DST_PREPROCESSED / "images" / split / img.name)
        for lbl in (BASELINE_PREPROCESSED / "labels" / split).glob("*.txt"):
            shutil.copy2(lbl, DST_PREPROCESSED / "labels" / split / lbl.name)
            
    # Copy existing baseline preprocessed train images & labels
    print("Copying baseline preprocessed train images...")
    for img in (BASELINE_PREPROCESSED / "images" / "train").glob("*.*"):
        shutil.copy2(img, DST_PREPROCESSED / "images" / "train" / img.name)
    for lbl in (BASELINE_PREPROCESSED / "labels" / "train").glob("*.txt"):
        shutil.copy2(lbl, DST_PREPROCESSED / "labels" / "train" / lbl.name)
        
    # Find new augmented training images
    existing_stems = {p.stem for p in (BASELINE_PREPROCESSED / "images" / "train").glob("*.*")}
    new_images = [p for p in (SRC_RAW_DATASET / "images" / "train").glob("*.*") if p.stem not in existing_stems]
    print(f"Found {len(new_images)} new augmented training images to preprocess with 4-stage pipeline...")
    
    cfg = get_default_config()
    t0 = time.time()
    for idx, img_path in enumerate(new_images, 1):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            
        preprocessed = process_sonar_image(img_bgr, config=cfg)
        out_path = DST_PREPROCESSED / "images" / "train" / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), preprocessed)
        
        # Copy corresponding label
        lbl_path = SRC_RAW_DATASET / "labels" / "train" / (img_path.stem + ".txt")
        shutil.copy2(lbl_path, DST_PREPROCESSED / "labels" / "train" / (img_path.stem + ".txt"))
        
        if idx % 25 == 0 or idx == len(new_images):
            print(f"  Preprocessed {idx}/{len(new_images)} images ({time.time() - t0:.1f}s)")
            
    print("Writing data_preprocessed.yaml...")
    yaml_content = {
        "path": str(DST_PREPROCESSED),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": {
            0: "shipwreck",
            1: "aircraft",
            2: "mine",
            3: "fishing_gear"
        }
    }
    yaml_path = DST_PREPROCESSED / "data_preprocessed.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"Preprocessed dataset ready at: {DST_PREPROCESSED}")
    print(f"Preprocessed YAML written to: {yaml_path}")

if __name__ == "__main__":
    main()
