# Aqua Vision Detection Model
## Physics-Constrained 4-Class Marine Debris & Anomaly Detector

This directory contains the production-ready YOLO model weights and inference/evaluation scripts for the Aqua Vision underwater side-scan sonar detection system.

---

### Target Classes & Calibrated Thresholds

The detector is calibrated to operate across 4 distinct sonar object classes with physics-tuned confidence thresholds:

| Class ID | Class Name | Optimal Confidence Threshold | Rationale & Sonic Signature |
|:---:|:---|:---:|:---|
| **0** | `Shipwreck` | **0.25** | Suppresses diffuse seafloor reverberation while maintaining high precision on large structural hulls. |
| **1** | `Aircraft` | **0.21** | Captures distinct aerodynamic acoustic shadows and wing structures. |
| **2** | `Mine` | **0.10** | High-sensitivity recall for micro-targets (<20px) exhibiting high acoustic reflectivity and sharp shadows. |
| **3** | `Fishing Gear` | **0.16** | Tuned to resolve diffuse, low-contrast acoustic reflections from submerged ghost nets and traps. |

---

### File Overview

- **`aqua_vision_100ep_best.pt`**: Primary trained YOLO model weights from the 100-epoch physics-constrained training curriculum.
- **`best.pt`**: Symlink/alias to `aqua_vision_100ep_best.pt`.
- **`evaluate_sonar_model.py`**: Automated evaluation suite with confusion matrices, class-specific PR curves, F1 analysis, and IoU matching.
- **`run_sonar_inference_demo.py`**: Standalone command-line inference demo applying CLAHE contrast enhancement and thresholded detection with bounding box visualizations.

---

### Quick Start Inference

Run inference on any raw side-scan sonar image:

```bash
# Standalone inference demo
python run_sonar_inference_demo.py --source path/to/sonar_image.jpg --weights aqua_vision_100ep_best.pt --output inference_outputs/
```

Run evaluation on a dataset:

```bash
python evaluate_sonar_model.py --dataset path/to/dataset --weights aqua_vision_100ep_best.pt
```
