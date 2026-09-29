# Tiled Inference Diagnostic Report

**Model**: `C:\Users\Jhansi\OneDrive\Desktop\SIHproject\runs\detect\4class_training\baseline_v1\weights\best.pt`
**Test images**: `C:\Users\Jhansi\OneDrive\Desktop\SIHproject\dataset\SIH_Combined_4Class_Balanced\images\test`

## Overall Results

| Method | TP | FP | FN | Precision | Recall |
|--------|----|----|----|-----------|--------|
| Normal | 16 | 31 | 229 | 0.340 | 0.065 |
| Tiled  | 20 | 77 | 225 | 0.206 | 0.082 |

## Per‑Class Results

| Class | Method | TP | FP | FN | Precision | Recall |
|-------|--------|----|----|----|-----------|--------|
| shipwreck | Normal | 9 | 15 | 57 | 0.375 | 0.136 |
| shipwreck | Tiled  | 11 | 67 | 55 | 0.141 | 0.167 |
| aircraft | Normal | 2 | 1 | 3 | 0.667 | 0.400 |
| aircraft | Tiled  | 2 | 0 | 3 | 1.000 | 0.400 |
| mine | Normal | 0 | 3 | 49 | 0.000 | 0.000 |
| mine | Tiled  | 0 | 1 | 49 | 0.000 | 0.000 |
| fishing_gear | Normal | 5 | 12 | 120 | 0.294 | 0.040 |
| fishing_gear | Tiled  | 7 | 9 | 118 | 0.438 | 0.056 |

## Recall Comparison

| Class | Normal Recall | Tiled Recall | Change |
|-------|---------------|--------------|--------|
| shipwreck | 0.136 | 0.167 | +0.030 |
| aircraft | 0.400 | 0.400 | +0.000 |
| mine | 0.000 | 0.000 | +0.000 |
| fishing_gear | 0.040 | 0.056 | +0.016 |

## Notable Findings

### Recovered Objects (Tiled TP not present in Normal)
* ai4shipwrecks_Haltiner_Barge_05.png: shipwreck (≈2 extra detections)
* ai4shipwrecks_Haltiner_Barge_07.png: shipwreck (≈1 extra detections)
* ai4shipwrecks_James_Davidson_01.png: shipwreck (≈1 extra detections)
* ai4shipwrecks_WH_Gilbert_01.png: shipwreck (≈3 extra detections)
* Rec14_wcp_ss_star_00012_png_jpg.rf.00e1f1d5a607a4ba8c948daf1623186b.jpg: fishing_gear (≈1 extra detections)

### New False Positives introduced by Tiling
* ai4shipwrecks_Haltiner_Barge_01.png: shipwreck (≈1 extra FP)
* ai4shipwrecks_Haltiner_Barge_02.png: shipwreck (≈2 extra FP)
* ai4shipwrecks_Haltiner_Barge_03.png: shipwreck (≈3 extra FP)
* ai4shipwrecks_James_Davidson_01.png: shipwreck (≈15 extra FP)
* ai4shipwrecks_James_Davidson_02.png: shipwreck (≈10 extra FP)

### Mine ↔ Fishing Gear Confusion (approx.)
* Not explicitly counted in this lightweight script.

---

*All numbers are based on IoU ≥ 0.50 for matching predictions to ground‑truth.*
