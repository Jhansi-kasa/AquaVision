# 4-Class YOLO Training -- Readiness Report

> **Generated:** 2026-09-20 11:29:14  
> **Validator:** `backend/validate_4class_pipeline.py`  
> **Elapsed:** 0.3s

## Overall Result: [PASS] ALL CHECKS PASSED

## Dataset Configuration

| Property | Value |
|---|---|
| Dataset path | `C:\Users\Jhansi\OneDrive\Desktop\SIHproject\dataset\SIH_Combined_4Class_Balanced` |
| data.yaml | `C:\Users\Jhansi\OneDrive\Desktop\SIHproject\dataset\SIH_Combined_4Class_Balanced\data.yaml` |
| Classes (nc) | 4 |
| Class mapping | `0=shipwreck, 1=aircraft, 2=mine, 3=fishing_gear` |
| Train images | images/train |
| Val images | images/val |
| Test images | images/test |

## Preprocessing Pipeline

**Source module:** `computer_vision/final_preprocessing_pipeline.py`  
**Applied by:** `backend/train_4class_yolo.py` (offline, before YOLO training)  
**Original images:** NEVER modified -- preprocessing writes to a separate directory  

| Stage | Operation | Parameters |
|---|---|---|
| 1 | Cross-Track Swath Illumination Normalization | axis=horizontal, method=median, sigma=35, kernel=71, gain=[0.5,2.5] |
| 2 | Edge-Preserving Bilateral Denoising | d=7, sigmaColor=50, sigmaSpace=50 |
| 3 | Robust Percentile Normalization | p_low=1%, p_high=99%, out=[0,255] |
| 4 | CLAHE Contrast Enhancement | clipLimit=2.0, tileGrid=(8,8), LAB L* channel |

> **Bounding-box safety:** All pipeline stages are pixel-intensity operations only.
> No geometric transformations are applied. YOLO normalized coordinates remain valid.
> The same pipeline is applied identically to all 4 classes (class-agnostic).

## Model Configuration

| Property | Value |
|---|---|
| Starting checkpoint | `yolo11n.pt` (official pretrained, auto-downloaded) |
| Architecture | YOLOv11n |
| Number of classes | 4 |
| Class names | `[shipwreck, aircraft, mine, fishing_gear]` |
| 3-class model (new_marine_debris_model.pt) | **NOT modified** |
| 3-class model (improved_yolov8n_best.pt) | **NOT modified** |

## Training Configuration

| Parameter | Value |
|---|---|
| imgsz | 640 |
| epochs | 100 |
| patience | 20 |
| batch | -1 (auto) |
| amp | True |
| cos_lr | True |
| warmup_epochs | 3 |
| fliplr | 0.5 |
| flipud | 0.0 |
| degrees | 0.0 |
| mosaic | 0.5 |
| mixup | 0.0 |
| scale | 0.2 |
| hsv_v | 0.4 |
| Output directory | `runs/detect/4class_training/` |

## Readiness Checks

| # | Check | Status | Detail |
|---|---|---|---|
| 1 | data.yaml has exactly 4 classes with correct mapping | [PASS] PASS | {'nc_found': 4, 'names_found': {0: 'shipwreck', 1: 'aircraft', 2: 'mine', 3: 'fishing_gear'}, 'expected': {0: 'shipwreck |
| 2 | All label class IDs in {0, 1, 2, 3} | [PASS] PASS | {'all_class_ids_found': [0, 1, 2, 3], 'invalid_labels': []} |
| 3 | Classes 0/1/2 present in labels (no accidental remap) | [PASS] PASS | {'ids_present': [0, 1, 2, 3]} |
| 4 | Existing shipwreck/aircraft/mine labels unchanged | [PASS] PASS | Class IDs 0,1,2 map identically to shipwreck/aircraft/mine |
| 5 | fishing_gear labels use class ID 3 | [PASS] PASS | {'class_3_found_in_labels': True} |
| 6 | CV preprocessing pipeline works (bbox-safe, class-agnostic) | [PASS] PASS | {'pipeline': 'final_preprocessing_pipeline.py', 'stages': ['swath_normalization', 'bilateral_denoising', 'robust_normali |
| 7 | No old 6-class names in active config | [PASS] PASS | {'data_yaml_names': ['shipwreck', 'fishing_gear', 'aircraft', 'mine'], 'old_6_class_contamination_in_yaml': [], 'train_s |
| 8 | Existing 3-class .pt files are untouched | [PASS] PASS | {'new_marine_debris_model.pt': {'exists': True, 'size_bytes': 21245859, 'path': 'C:\\Users\\Jhansi\\OneDrive\\Desktop\\S |
| 9 | New output directory is separate from 3-class model directory | [PASS] PASS | {'output_dir': 'C:\\Users\\Jhansi\\OneDrive\\Desktop\\SIHproject\\runs\\detect\\4class_training', 'model_dir': 'C:\\User |
| 10 | All dataset splits present on disk | [PASS] PASS | {'dataset_root': 'C:\\Users\\Jhansi\\OneDrive\\Desktop\\SIHproject\\dataset\\SIH_Combined_4Class_Balanced', 'missing_dir |

## Files Modified

| File | Action |
|---|---|
| `backend/train_4class_yolo.py` | **CREATED** -- new 4-class training script |
| `backend/validate_4class_pipeline.py` | **CREATED** -- this validator |
| `4CLASS_TRAINING_READINESS_REPORT.md` | **CREATED** -- this report |

## Files Intentionally NOT Modified

| File | Reason |
|---|---|
| `backend/train_sonar_yolo.py` | Old 3-class script -- preserved unchanged |
| `backend/app/main.py` | Production inference -- not in scope |
| `backend/app/model/new_marine_debris_model.pt` | Active 3-class model -- untouched |
| `backend/app/model/improved_yolov8n_best.pt` | Old 3-class model -- untouched |
| `final_ai_ready_dataset/` | 3-class dataset -- untouched |
| `dataset/SIH_Combined_4Class_Balanced/` | Source dataset -- read-only |
| `computer_vision/*.py` | CV research scripts -- untouched |
| `backend/app/preprocessing/__init__.py` | Inference preprocessing -- untouched |
| Frontend / GIS code | Out of scope |

## Confirmation: Old 6-Class Logic Not Active

> The class names `drowning_victim`, `seafloor`, and `crab_pot` do NOT appear
> in any active training or evaluation configuration.
> The `train_sonar_yolo.py` file retains its stale 6-class `CLASSES` dict
> but is **not used** -- the new `train_4class_yolo.py` is the active training script.
> `data.yaml` for the 4-class dataset contains only: shipwreck, aircraft, mine, fishing_gear.

---

**4-class training pipeline is ready.**

Run training with:
```bash
cd SIHproject
python backend/train_4class_yolo.py
```