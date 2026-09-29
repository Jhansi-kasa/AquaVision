import os
import shutil
import random
import cv2
import numpy as np
import yaml
from pathlib import Path

# Set seeds for deterministic reproducibility
random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(r"C:\Users\Jhansi\OneDrive\Desktop\SIHproject")
SRC_DATASET = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced"
DST_DATASET = PROJECT_ROOT / "dataset" / "SIH_Combined_4Class_Balanced_FishingTargeted"

def count_boxes(lbl_dir):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    img_with_cls = {0: 0, 1: 0, 2: 0, 3: 0}
    total_boxes = 0
    total_files = 0
    for lf in lbl_dir.glob("*.txt"):
        total_files += 1
        with open(lf) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        present = set()
        for parts in lines:
            cid = int(parts[0])
            counts[cid] = counts.get(cid, 0) + 1
            present.add(cid)
            total_boxes += 1
        for cid in present:
            img_with_cls[cid] = img_with_cls.get(cid, 0) + 1
    return counts, img_with_cls, total_boxes, total_files

def main():
    print("=== PREPARING TARGETED EXPERIMENTAL DATASET ===")
    
    # Clean destination if exists
    if DST_DATASET.exists():
        print(f"Removing existing {DST_DATASET}...")
        shutil.rmtree(DST_DATASET)
    
    for split in ["train", "val", "test"]:
        (DST_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Copy val and test splits 100% UNCHANGED
    for split in ["val", "test"]:
        print(f"Copying {split} split unchanged...")
        for img_file in (SRC_DATASET / "images" / split).glob("*.*"):
            shutil.copy2(img_file, DST_DATASET / "images" / split / img_file.name)
        for lbl_file in (SRC_DATASET / "labels" / split).glob("*.txt"):
            shutil.copy2(lbl_file, DST_DATASET / "labels" / split / lbl_file.name)
            
    # Step 2: Copy baseline train split
    print("Copying baseline train split...")
    for img_file in (SRC_DATASET / "images" / "train").glob("*.*"):
        shutil.copy2(img_file, DST_DATASET / "images" / "train" / img_file.name)
    for lbl_file in (SRC_DATASET / "labels" / "train").glob("*.txt"):
        shutil.copy2(lbl_file, DST_DATASET / "labels" / "train" / lbl_file.name)
        
    counts_before, _, boxes_before, imgs_before = count_boxes(DST_DATASET / "labels" / "train")
    print(f"Train boxes before augmentation: {counts_before}")
    print(f"Total train images before: {imgs_before}, total boxes: {boxes_before}")
    fg_before = counts_before[3]
    
    # Target 10-20% increase in fishing_gear (+88 to +175 boxes)
    TARGET_FG_INCREASE = 115 # ~13.1% increase
    
    # Step 3: Identify eligible hard training images
    # Hard criteria: contains fishing_gear, with small dimensions (<32px or area < 1024 in 640x640)
    train_lbl_dir = DST_DATASET / "labels" / "train"
    train_img_dir = DST_DATASET / "images" / "train"
    
    eligible_candidates = []
    for lf in sorted(train_lbl_dir.glob("*.txt")):
        with open(lf) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        fg_boxes = [l for l in lines if int(l[0]) == 3]
        if not fg_boxes:
            continue
        
        # Check if contains small or thin fishing gear
        has_small_fg = False
        for parts in fg_boxes:
            w, h = float(parts[3]) * 640, float(parts[4]) * 640
            if w < 32 or h < 32 or (w * h) < 1024:
                has_small_fg = True
                break
        
        # Prefer images with pure or mostly fishing_gear to avoid unintended multi-class oversampling
        other_classes = [int(l[0]) for l in lines if int(l[0]) != 3]
        
        if has_small_fg:
            img_path = train_img_dir / (lf.stem + ".jpg")
            if not img_path.exists():
                img_path = train_img_dir / (lf.stem + ".png")
            if img_path.exists():
                eligible_candidates.append({
                    "stem": lf.stem,
                    "img_path": img_path,
                    "lbl_path": lf,
                    "num_fg": len(fg_boxes),
                    "other_classes": other_classes
                })
                
    print(f"Found {len(eligible_candidates)} eligible hard training candidates.")
    
    # Sort candidates by fewest other classes to minimize side-effect on other classes
    eligible_candidates.sort(key=lambda c: (len(c["other_classes"]), c["num_fg"]))
    
    # Select images to reach target FG count
    selected = []
    accum_fg = 0
    for cand in eligible_candidates:
        selected.append(cand)
        accum_fg += cand["num_fg"]
        if accum_fg >= TARGET_FG_INCREASE:
            break
            
    print(f"Selected {len(selected)} unique training images to augment, yielding {accum_fg} new FG instances.")
    
    # Step 4: Apply conservative SSS-valid augmentations
    # Alternating between:
    # 1. Horizontal flip (valid for symmetric sonar)
    # 2. Mild contrast / brightness adjustment
    # 3. Mild Gaussian noise
    # 4. Mild CLAHE
    
    augmented_count = 0
    for idx, cand in enumerate(selected):
        img = cv2.imread(str(cand["img_path"]))
        if img is None:
            continue
            
        with open(cand["lbl_path"]) as f:
            lines = [l.strip().split() for l in f if l.strip()]
            
        aug_type = idx % 4
        if aug_type == 0:
            # Horizontal flip
            aug_img = cv2.flip(img, 1)
            aug_name = f"aug_hflip_{cand['stem']}"
            aug_lines = []
            for parts in lines:
                cid = parts[0]
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                new_cx = 1.0 - cx
                aug_lines.append(f"{cid} {new_cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        elif aug_type == 1:
            # Subtle contrast / brightness (alpha=1.1, beta=8)
            aug_img = cv2.convertScaleAbs(img, alpha=1.1, beta=8)
            aug_name = f"aug_contrast_{cand['stem']}"
            aug_lines = [f"{' '.join(p)}\n" for p in lines]
        elif aug_type == 2:
            # Mild Gaussian noise
            noise = np.random.normal(0, 3, img.shape).astype(np.float32)
            aug_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            aug_name = f"aug_noise_{cand['stem']}"
            aug_lines = [f"{' '.join(p)}\n" for p in lines]
        else:
            # Mild CLAHE on luminance
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            aug_img = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
            aug_name = f"aug_clahe_{cand['stem']}"
            aug_lines = [f"{' '.join(p)}\n" for p in lines]
            
        # Save augmented image and label
        dst_img_path = train_img_dir / f"{aug_name}.jpg"
        dst_lbl_path = train_lbl_dir / f"{aug_name}.txt"
        
        cv2.imwrite(str(dst_img_path), aug_img)
        with open(dst_lbl_path, "w") as f:
            f.writelines(aug_lines)
            
        augmented_count += 1
        
    print(f"Created {augmented_count} augmented training images.")
    
    # Step 5: Recalculate statistics
    counts_after, _, boxes_after, imgs_after = count_boxes(train_lbl_dir)
    fg_after = counts_after[3]
    pct_increase = ((fg_after - fg_before) / fg_before) * 100.0
    
    print("\n=== TRAINING DATASET STATISTICS BEFORE / AFTER ===")
    print(f"Total training images: {imgs_before} -> {imgs_after} (+{imgs_after - imgs_before})")
    print(f"Total training boxes : {boxes_before} -> {boxes_after} (+{boxes_after - boxes_before})")
    print(f"shipwreck boxes      : {counts_before[0]} -> {counts_after[0]} (+{counts_after[0] - counts_before[0]})")
    print(f"aircraft boxes       : {counts_before[1]} -> {counts_after[1]} (+{counts_after[1] - counts_before[1]})")
    print(f"mine boxes           : {counts_before[2]} -> {counts_after[2]} (+{counts_after[2] - counts_before[2]})")
    print(f"fishing_gear boxes   : {fg_before} -> {fg_after} (+{fg_after - fg_before}, +{pct_increase:.2f}%)")
    
    # Step 6: Dataset validation checks
    print("\n=== DATASET INTEGRITY VERIFICATION ===")
    errors = 0
    for split in ["train", "val", "test"]:
        s_lbl = DST_DATASET / "labels" / split
        s_img = DST_DATASET / "images" / split
        for lf in s_lbl.glob("*.txt"):
            stem = lf.stem
            img_p = s_img / (stem + ".jpg")
            if not img_p.exists():
                img_p = s_img / (stem + ".png")
            if not img_p.exists():
                print(f"[ERROR] Orphan label: {lf.name}")
                errors += 1
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"[ERROR] Syntax error in {lf.name}: {line}")
                        errors += 1
                        continue
                    cid = int(parts[0])
                    if cid not in {0, 1, 2, 3}:
                        print(f"[ERROR] Invalid class {cid} in {lf.name}")
                        errors += 1
                    cx, cy, w, h = map(float, parts[1:])
                    if cx <= 0 or cx >= 1 or cy <= 0 or cy >= 1 or w <= 0 or h <= 0:
                        print(f"[ERROR] Out of bounds box in {lf.name}: {line}")
                        errors += 1
    if errors == 0:
        print("[PASS] All YOLO labels valid, 0 syntax errors, 0 zero-area boxes, 0 orphan labels.")
        
    # Write data.yaml
    data_yaml_content = {
        "path": str(DST_DATASET),
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
    with open(DST_DATASET / "data.yaml", "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
        
    print(f"Dataset YAML written to {DST_DATASET / 'data.yaml'}")

if __name__ == "__main__":
    main()
