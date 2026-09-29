# Fishing Targeted Dataset Audit Report

**Date:** September 24, 2026  
**Project:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection  
**Experiment:** Controlled Targeted-Training Experiment for `fishing_gear` Recall  
**Audited File:** `scratch/fishing_gear_28_missed_cases.csv`  
**Dataset:** `dataset/SIH_Combined_4Class_Balanced/`  

---

## 1. Executive Summary & Anti-Leakage Audit

A strict audit of all failure cases in `scratch/fishing_gear_28_missed_cases.csv` was conducted to determine their split membership across `train`, `val`, and `test` in `dataset/SIH_Combined_4Class_Balanced/`.

Under **Experiment Rule 7** and **Rule 8**:
> *“Only use hard examples whose source images belong to the TRAINING set.*  
> *IMPORTANT: NEVER copy test images or test labels into the training set. NEVER use test-set images for training.*  
> *Before creating augmented data, verify for every CSV image whether it belongs to: train, val, test.”*

### Split Membership Verification:
- **Total CSV rows (missed bounding box instances):** 28
- **Unique image files in CSV:** 21
- **Number from `images/train`:** **0** (0.0%)
- **Number from `images/val`:** **0** (0.0%)
- **Number from `images/test`:** **28 instances across 21 images (100.0%)**
- **Number excluded from training augmentation:** **28 (100.0% EXCLUDED)**

---

## 2. Complete Inventory of CSV Cases & Audit Findings

Every single entry in `scratch/fishing_gear_28_missed_cases.csv` originates from the baseline evaluation on the **test split** (`dataset/SIH_Combined_4Class_Balanced/images/test`):

| # | Image Filename | Split | GT BBox [x1, y1, x2, y2] | W | H | Area | Size Category | Failure Type | Pred Class | Conf | Audit Decision |
|---|----------------|:-----:|--------------------------|---|---|------|:-------------:|:------------:|:----------:|:----:|:--------------:|
| 1 | baycove_07_06_png_jpg.rf.5a2e74f04f707fe66f050a3b61fcd7c2.jpg | **test** | [284.0, 19.0, 314.0, 49.0] | 30.0 | 30.0 | 900.0 | small | wrong_class | 2 (mine) | 0.0209 | **EXCLUDED (Test Set)** |
| 2 | baycove_07_07_png_jpg.rf.4c2d86c1becfaf0c5471fca9f6e3dbfb.jpg | **test** | [493.0, 20.0, 507.5, 35.0] | 14.5 | 15.0 | 217.5 | small | wrong_class | 2 (mine) | 0.0949 | **EXCLUDED (Test Set)** |
| 3 | baycove_07_10_png_jpg.rf.e3b62b44fe8e5e4363b52351ba782b8f.jpg | **test** | [525.0, 450.0, 549.5, 468.0] | 24.5 | 18.0 | 441.0 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 4 | baycove_07_14_png_jpg.rf.3f8cedd2d50ef473a9c8ee632ec7f990.jpg | **test** | [107.0, 604.0, 136.0, 626.0] | 29.0 | 22.0 | 638.0 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 5 | baycove_07_15_png_jpg.rf.7f81b82c41fae460608733f4c11ab936.jpg | **test** | [154.0, 457.0, 180.5, 478.0] | 26.5 | 21.0 | 556.5 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 6 | baycove_07_15_png_jpg.rf.7f81b82c41fae460608733f4c11ab936.jpg | **test** | [342.0, 559.0, 361.0, 574.5] | 19.0 | 15.5 | 294.5 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 7 | baycove_07_15_png_jpg.rf.7f81b82c41fae460608733f4c11ab936.jpg | **test** | [446.0, 339.0, 473.5, 364.5] | 27.5 | 25.5 | 701.3 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 8 | baycove_07_16_png_jpg.rf.a0248faa15ab56af1a1d1935580b1d21.jpg | **test** | [211.0, 412.0, 235.0, 429.0] | 24.0 | 17.0 | 408.0 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 9 | baycove_07_16_png_jpg.rf.a0248faa15ab56af1a1d1935580b1d21.jpg | **test** | [111.0, 388.0, 130.5, 406.5] | 19.5 | 18.5 | 360.7 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 10 | baycove_07_16_png_jpg.rf.a0248faa15ab56af1a1d1935580b1d21.jpg | **test** | [523.0, 186.0, 546.5, 207.5] | 23.5 | 21.5 | 505.3 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 11 | baycove_07_16_png_jpg.rf.a0248faa15ab56af1a1d1935580b1d21.jpg | **test** | [361.0, 96.0, 386.5, 119.5] | 25.5 | 23.5 | 599.3 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 12 | Rec09_Sensor_Depth_wcp_ss_port_00005_png_jpg.rf.6b14987e39c7311b09f59a5783251412.jpg | **test** | [369.0, 595.0, 385.0, 611.5] | 16.0 | 16.5 | 264.0 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 13 | Rec09_Sensor_Depth_wcp_ss_port_00008_png_jpg.rf.20764ab424e805778f77c0ad43d8df09.jpg | **test** | [335.0, 303.0, 362.5, 340.0] | 27.5 | 37.0 | 1017.5 | small | low_conf | 3 | 0.0252 | **EXCLUDED (Test Set)** |
| 14 | Rec09_Sensor_Depth_wcp_ss_port_00015_jpg.rf.cce6ba9121e2326ded051da43c9da480.jpg | **test** | [62.0, 391.0, 130.5, 443.5] | 68.5 | 52.5 | 3596.2 | medium | low_conf | 3 | 0.0755 | **EXCLUDED (Test Set)** |
| 15 | Rec09_Sensor_Depth_wcp_ss_port_00021_png_jpg.rf.1366c8bdfa2acdd12d723ec1ba51f6e3.jpg | **test** | [215.0, 594.0, 251.5, 605.5] | 36.5 | 11.5 | 419.8 | small | no_pred | - | - | **EXCLUDED (Test Set)** |
| 16 | Rec09_Sensor_Depth_wcp_ss_star_00034_png_jpg.rf.77e883d4b2a8d51fbe6779226bf1768f.jpg | **test** | [-0.0, 306.0, 71.5, 359.0] | 71.5 | 53.0 | 3789.5 | medium | no_pred | - | - | **EXCLUDED (Test Set)** |
| 17 | Rec14_wcp_ss_port_00000_png_jpg.rf.0d0be191ad69b06ca3737be3bfb20d60.jpg | **test** | [90.0, 523.0, 190.0, 574.0] | 100.0 | 51.0 | 5100.0 | medium | poor_iou | 3 | 0.3548 | **EXCLUDED (Test Set)** |
| 18 | Rec14_wcp_ss_port_00009_png_jpg.rf.a8e161fa80aeb064f5a658aabe981a80.jpg | **test** | [304.0, 549.0, 372.0, 606.0] | 68.0 | 57.0 | 3876.0 | medium | low_conf | 3 | 0.0197 | **EXCLUDED (Test Set)** |
| 19 | Rec14_wcp_ss_port_00019_png_jpg.rf.242bd204c337a93f0a080e722cfbf23a.jpg | **test** | [311.0, 168.0, 381.0, 211.0] | 70.0 | 43.0 | 3010.0 | medium | no_pred | - | - | **EXCLUDED (Test Set)** |
| 20 | Rec14_wcp_ss_port_00036_png_jpg.rf.905f5eac4f698ad98958fb2865ecb031.jpg | **test** | [459.0, 165.0, 530.0, 242.0] | 71.0 | 77.0 | 5467.0 | medium | low_conf | 3 | 0.1421 | **EXCLUDED (Test Set)** |
| 21 | Rec14_wcp_ss_star_00000_png_jpg.rf.169f0cbcb0df31840dfb6196a94bb577.jpg | **test** | [513.0, 234.0, 627.0, 329.0] | 114.0 | 95.0 | 10830.0 | medium | low_conf | 3 | 0.0188 | **EXCLUDED (Test Set)** |
| 22 | Rec14_wcp_ss_star_00012_png_jpg.rf.00e1f1d5a607a4ba8c948daf1623186b.jpg | **test** | [345.0, 303.0, 404.0, 361.0] | 59.0 | 58.0 | 3422.0 | medium | low_conf | 3 | 0.1938 | **EXCLUDED (Test Set)** |
| 23 | Rec14_wcp_ss_star_00012_png_jpg.rf.00e1f1d5a607a4ba8c948daf1623186b.jpg | **test** | [227.0, 403.0, 287.0, 442.0] | 60.0 | 39.0 | 2340.0 | medium | no_pred | - | - | **EXCLUDED (Test Set)** |
| 24 | Rec14_wcp_ss_star_00012_png_jpg.rf.00e1f1d5a607a4ba8c948daf1623186b.jpg | **test** | [328.0, 353.0, 374.0, 402.0] | 46.0 | 49.0 | 2254.0 | medium | low_conf | 3 | 0.0118 | **EXCLUDED (Test Set)** |
| 25 | Rec8_wcp_ss_port_00006_png_jpg.rf.60bc525ec0403980d9eea458a9cae03f.jpg | **test** | [351.0, 263.0, 419.5, 304.0] | 68.5 | 41.0 | 2808.5 | medium | low_conf | 3 | 0.1507 | **EXCLUDED (Test Set)** |
| 26 | Rec8_wcp_ss_port_00023_png_jpg.rf.13db4e62bc0f8fb1ee275725869468d4.jpg | **test** | [250.0, 582.0, 319.0, 607.5] | 69.0 | 25.5 | 1759.5 | small | wrong_class | 0 (shipwreck) | 0.0595 | **EXCLUDED (Test Set)** |
| 27 | Rec9_wcp_ss_port_00027_png_jpg.rf.c2b5bea5988412959b055d43ea6caffc.jpg | **test** | [-0.0, 562.0, 39.5, 598.0] | 39.5 | 36.0 | 1422.0 | medium | no_pred | - | - | **EXCLUDED (Test Set)** |
| 28 | Rec9_wcp_ss_port_00046_png_jpg.rf.2560848aef152a8545e9063bd1886fba.jpg | **test** | [353.0, 144.0, 425.0, 298.0] | 72.0 | 154.0 | 11088.0 | medium | poor_iou | 3 | 0.5334 | **EXCLUDED (Test Set)** |

---

## 3. Data Integrity & Scientific Protocol

### Anti-Contamination Guarantee:
1. **0 test-set images** are added to the training set.
2. **0 test-set labels** are exposed to training.
3. Test split remains **100% frozen and pristine** for unbiased evaluation.

### Eligible Training Hard Examples Selection:
To test whether targeted augmentation improves `fishing_gear` recall (aiming for the mandated **10–20% increase in total fishing_gear training instances**) without committing data leakage:
- We selected **eligible hard training examples** from `images/train` that exhibit the exact same diagnostic failure modes documented in the diagnostic report:
  - Small bounding-box dimensions ($W < 32$ px, $H < 32$ px, or Area $< 1024$ px²)
  - Subtle acoustic highlight/shadow structure prone to false negatives and mine confusion.
- In `images/train`, 225 images contain 321 small `fishing_gear` objects.
- A controlled subset of these images is augmented using side-scan sonar-preserving transformations (horizontal flip, subtle contrast/brightness, mild Gaussian noise) to add **+115 fishing_gear instances (+13.1% increase)**, strictly within the 10–20% target band.
