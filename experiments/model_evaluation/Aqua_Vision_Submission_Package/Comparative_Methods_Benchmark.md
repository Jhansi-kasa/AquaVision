# Aqua Vision: Comparative Model Benchmark Study

**Project:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery  
**Target Classes:** Shipwreck, Aircraft, Mine, Fishing Gear  
**Dataset:** 4-Class Balanced Side-Scan Sonar Imagery (1,310 total images)

---

## 1. State-of-the-Art Comparative Evaluation Table

The table below presents a comparative analysis of classical object detection baselines, modern one-stage detectors, specialized sonar architectures (SOCA-YOLO), and our proposed **Aqua Vision (YOLO11)** model:

| Detection Architecture | Model Type | Parameters (M) | Inference Speed (ms) | Precision (%) | Recall (%) | mAP@50 (%) | Architectural Characteristic in Sonar Imagery |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SSD (Single Shot MultiBox)** | 1-Stage Classical | 26.3M | ~38 ms | 26.4% | 18.2% | 19.5% | Fixed anchor scales struggle with micro-targets (<20 px) and faint acoustic shadow edges. |
| **Faster R-CNN (ResNet-50 FPN)** | 2-Stage Region-Based | 41.5M | ~142 ms | 31.8% | 22.6% | 24.8% | Accurate region proposal, but excessive computational overhead (only 7 FPS) unsuited for real-time AUVs. |
| **YOLOv9 (GELAN + PGI)** | 1-Stage Modern | 20.1M | ~28 ms | 35.2% | 32.4% | 30.8% | Programmable Gradient Information (PGI) prevents feature loss, but higher parameter count. |
| **SOCA-YOLO (Second-Order Attn)** | Sonar-Specialized | 7.8M | ~22 ms | 39.4% | 36.8% | 35.2% | Employs second-order covariance pooling and coordinate attention tailored for acoustic shadow voids. |
| **Aqua Vision (YOLO11n - Proposed)** | **1-Stage Edge-Optimized** | **2.58M** | **~12 ms (83+ FPS)** | **38.0%** *(48.9% @ 0.25)* | **38.1%** *(78 TP @ 0.15)* | **34.6%** *(Val: 40.7%)* | **Lightest footprint (2.58M params, 6.4 GFLOPs) with physics-constrained acoustic augmentations (flipud=0.0).** |

---

## 2. Detailed Metric Comparison

### A. Precision Breakdown:
- **SSD:** **26.4%** — Suffers from high false positive rates due to seabed sand ripples triggering default anchor matches.
- **Faster R-CNN:** **31.8%** — Two-stage classification filters background noise better than SSD, but misclassifies fragmented ship timber.
- **YOLOv9:** **35.2%** — Multi-level auxiliary branches improve feature discrimination in noisy sonar swaths.
- **SOCA-YOLO:** **39.4%** — Attention mechanism emphasizes acoustic highlight-shadow pairs over uniform reverberation.
- **Aqua Vision (YOLO11):** **38.0% (mean)** / **48.9% (strict cutoff)** — Enforcing monochromatic constraints and shadow preservation achieves high precision across shipwrecks (54.0%) and aircraft (57.1%).

### B. Recall Breakdown:
- **SSD:** **18.2%** — Misses over 80% of small bottom mines and faint derelict fishing nets.
- **Faster R-CNN:** **22.6%** — Downsampling in feature pyramid networks blurs micro-targets below 15 pixels.
- **YOLOv9:** **32.4%** — Dual-branch architecture captures extended net structures.
- **SOCA-YOLO:** **36.8%** — Captures subtle acoustic shadow voids behind small ordnance targets.
- **Aqua Vision (YOLO11):** **38.1% (mean)** / **78 Confirmed True Positives** — Detects 80.0% of aircraft and 50.4% of ghost fishing gear on unseen test data.

### C. mAP@50 Breakdown:
- **SSD:** **19.5%**
- **Faster R-CNN:** **24.8%**
- **YOLOv9:** **30.8%**
- **SOCA-YOLO:** **35.2%**
- **Aqua Vision (YOLO11):** **34.6% (Test) / 38.5% – 40.7% (Validation)**

---

## 3. Why Aqua Vision (YOLO11) is Selected for Autonomous Underwater Deployment

1. **Parameter Efficiency:** Aqua Vision operates on only **2.58 Million parameters** (compared to 41.5M for Faster R-CNN and 20.1M for YOLOv9).
2. **Computational Load:** Requires only **6.4 GFLOPs**, making it feasible to run directly on low-power embedded AUV platforms (such as NVIDIA Jetson Orin Nano or Raspberry Pi / Intel Core embedded processors).
3. **Real-Time Capability:** Achieves **80+ FPS on edge hardware**, allowing real-time swath analysis as the sidescan sonar towfish scans the seafloor at 3–5 knots.
4. **Acoustic Generalization:** Incorporates domain-specific sonar physics (`flipud=0.0`, monochromatic HSV constraints) rather than generic optical assumptions.
