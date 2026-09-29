# Computer Vision & Sonar Processing Subsystem

**Project Title:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery  
**Subsystem Responsibility:** Member 2 (Computer Vision & Sonar Processing)  
**Dataset Analyzed:** `SIH_Anomaly_V1` (Active 3-Class Target Detection)

---

## Preprocessing Pipeline Architecture

$$\text{Raw Side-Scan Sonar Image} \longrightarrow \mathbf{\text{Denoising}} \longrightarrow \mathbf{\text{Normalization}} \longrightarrow \mathbf{\text{CLAHE}} \longrightarrow \text{AI-Ready Image}$$

Currently implemented stages:
- **Phase 1:** Automated Dataset Inspection & Diagnostic Auditing
- **Phase 2:** Sonar Denoising Algorithm Evaluation & Benchmark (Bilateral Filter Selected)
- **Phase 3:** Intensity Normalization & Dynamic Range Conditioning (Robust Percentile Selected)
- **Phase 4:** Contrast Limited Adaptive Histogram Equalization (CLAHE: clipLimit=2.0, tileGridSize=(8, 8) Selected)

---

## Directory Structure

```text
computer_vision/
├── dataset_inspection.py        # Automated dataset inspection and integrity auditing
├── dataset_report.md            # Full diagnostic report with dataset statistics
├── denoising.py                 # Reusable Gaussian, Median, and Bilateral filtering module
├── denoising_comparison.py      # Benchmark script evaluating filters on 6 representative samples
├── denoising_report.md          # Empirical report comparing boundary retention & speckle reduction
├── denoising_results/           # Generated 2x2 comparison images (Raw, Gaussian, Median, Bilateral)
├── normalization.py             # Reusable Min-Max and Robust Percentile normalization module
├── normalization_comparison.py  # Benchmark script comparing normalization on representative samples
├── normalization_report.md      # Quantitative report before/after normalization & safety handling
├── normalization_results/       # Generated 2x2 comparison images (Raw, Denoised, MinMax, Robust)
├── clahe.py                     # Reusable CLAHE module with LAB color preservation
├── clahe_comparison.py          # Benchmark script testing 6 CLAHE configurations (clip 1-3, grid 8/16)
├── clahe_report.md              # Detailed parameter evaluation, contrast metrics, and shadow analysis
├── clahe_results/               # Generated 8-panel comparison images across parameter grid
├── speckle_filter.py            # Reusable Lee & Frost speckle filter module (5x5 window)
├── speckle_filter_comparison.py # Benchmark script evaluating Lee vs Frost across representative samples
├── speckle_filter_report.md     # Empirical evaluation, EPI, shadow retention, and latency analysis
├── speckle_filter_results/      # Generated triplet comparison images [Raw | Lee 5x5 | Frost 5x5]
├── swath_normalization.py       # Reusable cross-track swath illumination normalization module (AVG proxy)
├── swath_normalization_comparison.py # Benchmark script evaluating swath normalization across configurations
├── swath_normalization_report.md     # Full 18-section technical report, radiometric uniformity, & ablation
├── swath_normalization_results/      # Generated triplet panels, multi-config panels, and profile plots
├── morphological_highlight_shadow.py # Reusable White Top-Hat & Black-Hat separation module
├── morphological_comparison.py       # Benchmark script testing kernel sizes (5, 9, 15) and shapes (ellipse, rect)
├── morphological_report.md           # Full 22-section technical report, highlight/shadow metrics, seabed clutter analysis
├── morphological_results/            # Generated 4-panel quad comparisons and 5-panel kernel ablation panels
└── README.md                    # Subsystem documentation and execution instructions
```

---

## Execution Instructions

### Prerequisites
Install the required dependencies:
```bash
pip install opencv-python pillow pyyaml numpy
```

### 1. Dataset Inspection
```bash
python computer_vision/dataset_inspection.py
```

### 2. Denoising Benchmark
```bash
python computer_vision/denoising_comparison.py
```

### 3. Normalization Benchmark
```bash
python computer_vision/normalization_comparison.py
```

### 4. CLAHE Benchmark
```bash
python computer_vision/clahe_comparison.py
```

### 5. Speckle Filter Benchmark (Task P2: Lee vs Frost 5x5)
```bash
python computer_vision/speckle_filter_comparison.py
```

### 6. Swath Illumination Normalization Benchmark (Task P3: Cross-Track AVG Proxy)
```bash
python computer_vision/swath_normalization_comparison.py
```

### 7. Morphological Highlight–Shadow Separation Benchmark (Task P4: White Top-Hat & Black-Hat)
```bash
python computer_vision/morphological_comparison.py
```
This will:
1. Load 6 representative benchmark samples.
2. Evaluate White Top-Hat (WTH) and Black-Hat (BTH) across 4 kernel configurations (5x5 Ellipse, 9x9 Ellipse, 15x15 Ellipse, 9x9 Rect).
3. Save 4-panel quad comparisons (`[Raw | WTH | BTH | Combined Fusion]`) and 5-panel kernel ablation panels to `computer_vision/morphological_results/`.
4. Measure highlight/shadow response strength, activation %, edge energy, seabed clutter false-positive risk, and latency.
5. Generate `computer_vision/morphological_report.md`.

---

## Morphological Highlight–Shadow Benchmark Summary (Task P4)

Evaluated across 6 representative sonar samples ($640 \times 640 \times 3$):

| Configuration | Structuring Element | WTH Mean (Highlight) | WTH Act % (>20 DN) | BTH Mean (Shadow) | BTH Act % (>20 DN) | Seabed Clutter Act % | Latency (ms) | Status / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Config 1: 5×5 Ellipse** | Small Disk ($25\text{ px}$) | 8.74 | 14.0% | 8.18 | 13.1% | 4.21% | 2.23 ms | Sensitive to speckle noise; weak shadow capture |
| **Config 2: 9×9 Ellipse** | **Medium Disk ($81\text{ px}$)** | **15.12** | **25.0%** | **14.18** | **25.1%** | **12.74%** | **3.34 ms** | **RECOMMENDED: Optimal target/clutter balance** |
| **Config 3: 15×15 Ellipse** | Large Disk ($225\text{ px}$) | 21.46 | 34.7% | 20.82 | 37.4% | 26.79% | 8.49 ms | High false-positive risk on natural sand ripples |
| **Config 4: 9×9 Rectangular**| Box ($81\text{ px}$) | 17.19 | 28.3% | 16.27 | 29.2% | 15.62% | 1.48 ms | Anisotropic corner artifacts on circular targets |

### Key Morphological Takeaways
1. **Decoupled Structural Response:** White Top-Hat successfully extracts specular acoustic highlights while Black-Hat isolates acoustic shadow voids.
2. **Seabed Clutter Trade-Off:** While large kernels ($15 \times 15$) capture broad shadow extents, they double false-positive activation on natural sand ripples ($26.8\%$ vs $12.7\%$). The $9 \times 9$ elliptical structuring element provides the optimal balance between target saliency and clutter rejection.
3. **High-Speed Execution:** Morphology operations run in **~3.3 ms** per $640 \times 640$ frame, suitable for real-time feature generation.

---

## Swath Illumination Normalization Benchmark Summary (Task P3)

Evaluated across 6 representative sonar samples ($640 \times 640 \times 3$):

| Configuration | Smoothing $\sigma$ | Gain Bounds | Profile Std (Swath Non-Uniformity) | Uniformity Gain (%) | Near-to-Far Ratio | Shadow Area (<25) | Latency (ms) | Status / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Baseline** | N/A | N/A | 30.4 | 0.0% | 5.10 | 21.4% | 0.0 ms | Baseline reference |
| **Config A (Conservative)** | $50\text{ px}$ | $[0.6, 1.8]$ | 20.8 | +27.6% | 3.13 | 20.5% | 39.7 ms | Soft gain; leaves far range slightly dark |
| **Config B (Balanced)** | **$35\text{ px}$** | **$[0.5, 2.5]$** | **19.7** | **+30.8%** | **2.96** | **19.2%** | **23.0 ms** | **RECOMMENDED: Optimal leveling & shadow fidelity** |
| **Config C (Aggressive)** | $20\text{ px}$ | $[0.4, 3.5]$ | 18.6 | +34.8% | 2.81 | 18.9% | 66.9 ms | Strong boost, but elevates sediment noise |

### Key Swath Normalization Takeaways
1. **Flattens Cross-Track Attenuation:** Config B improves swath illumination uniformity by **+30.8%**, compressing profile std from $30.4$ down to $19.7$ and reducing near-to-far ratio from $5.10$ to $2.96$.
2. **Preserves Acoustic Shadow Geometry:** Deep acoustic shadows ($I < 25\text{ DN}$) are maintained ($19.2\%$ vs $21.4\%$ raw). The nadir water-column safeguard prevents zero-backscatter zones from blowing up into gray noise.
3. **High-Speed Execution:** Processing runs in only **~23.0 ms** per $640 \times 640$ frame, fully compatible with real-time sonar feeds.

---

## Speckle Filter Benchmark Summary (Task P2)

Evaluated across 6 representative sonar samples ($640 \times 640 \times 3$):

| Method / Configuration | Window | Key Parameter | EPI (Edge Retention) | Speckle Red. (%) | PSNR (dB) | Shadow Area (<25) | Latency (ms) | Status / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Sonar Baseline** | N/A | N/A | 1.0000 | 0.00% | $\infty$ | 21.4% | 0.00 ms | Baseline reference |
| **Lee Speckle Filter** | **$5 \times 5$** | **$C_u = 0.25$** | **0.5417** | **27.10%** | **30.33 dB** | **20.4%** | **47.7 ms** | **RECOMMENDED: High edge preservation & shadow void retention** |
| **Frost Speckle Filter** | $5 \times 5$ | $K = 1.0$ | 0.4563 | 31.83% | 27.88 dB | 19.4% | 231.7 ms | Over-smooths small debris; 4.9× slower |

### Key Speckle Filter Takeaways
1. **Adaptive Edge Weighting:** The Lee filter's weighting factor $W_L = 1 - \frac{C_u^2}{C_i^2}$ approaches 1.0 at abrupt highlight/shadow boundaries, leaving critical acoustic features unsmoothed while smoothing flat sediment.
2. **Shadow Integrity:** Lee preserves $95.3\%$ of the raw shadow void ($20.4\%$ vs $21.4\%$), whereas Frost's spatial kernel dilates intensity into shadow pockets ($19.4\%$).
3. **Execution Efficiency:** Lee executes in $\sim 47.7\text{ ms}$, suitable for real-time edge processing, whereas Frost requires $\sim 231.7\text{ ms}$.

---

## CLAHE Parameter Benchmark Summary

Evaluated across representative side-scan sonar samples ($640 \times 640 \times 3$):

| Configuration | clipLimit | tileGridSize | Mean Std (Contrast) | Edge Energy (Sobel) | Shadow Retention (%) | Status / Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pre-CLAHE Input** | N/A | N/A | 48.0 | 51.4 | 19.8% | Denoised + Normalized baseline |
| **`c1.0_g8x8`** | 1.0 | `(8, 8)` | 51.7 | 72.1 | 18.6% | Subtle enhancement; conservative |
| **`c1.0_g16x16`** | 1.0 | `(16, 16)` | 50.9 | 73.2 | 18.5% | Subtle enhancement; conservative |
| **`c2.0_g8x8`** | **2.0** | **`(8, 8)`** | **55.8** | **91.2** | **17.9%** | **RECOMMENDED: Optimal contrast & shadow integrity** |
| **`c2.0_g16x16`** | 2.0 | `(16, 16)` | 54.0 | 93.0 | 17.9% | Slight tile boundary risk on flat sediment |
| **`c3.0_g8x8`** | 3.0 | `(8, 8)` | 59.3 | 106.5 | 17.5% | Over-enhancement; amplifies background speckle |
| **`c3.0_g16x16`** | 3.0 | `(16, 16)` | 56.9 | 109.4 | 17.5% | Over-enhancement; amplifies background speckle |

### Key CLAHE Takeaways
1. **Luminance Channel Enhancement:** CLAHE is applied strictly to the $L^*$ channel in CIELAB color space, boosting local acoustic backscatter gradients without altering the amber/copper false-color palette.
2. **Boundary Sharpening:** `clipLimit=2.0, tileGridSize=(8, 8)` yields a **+77.4% boost in edge gradient energy** ($51.4 \to 91.2$), rendering weak target boundaries distinct.
3. **Shadow Preservation:** Unlike global histogram equalization (which destroys acoustic shadows, reducing shadow area to near 0%), CLAHE maintains deep shadow voids ($17.9\%$).

---

---

## Final AI-Ready Preprocessing Pipeline (Complete System)

The final preprocessing pipeline for Member 1's downstream YOLO training synthesizes the empirical findings of P1, P2, P3, and P4 into a robust, 4-stage sequential chain:

```text
Raw Side-Scan Sonar (640×640×3 uint8)
               │
               ▼
[Stage 1: Swath Illumination Normalization] (P3 Config B)
  - Equalizes cross-track range attenuation fall-off in CIELAB L*
  - smooth_sigma=35, kernel_size=71, gain=[0.5, 2.5], floor=8.0 DN
               │
               ▼
   [Stage 2: Bilateral Denoising] (P1 Bilateral)
  - Edge-preserving speckle suppression in homogeneous sediment
  - d=7, sigma_color=50.0, sigma_space=50.0 (PSNR 34.4 dB)
               │
               ▼
[Stage 3: Robust Dynamic Range Normalization] (P1 Robust)
  - 1% - 99% Percentile Stretch to [0, 255] uint8
  - Safe zero-division handling for low-contrast images
               │
               ▼
       [Stage 4: CIELAB CLAHE] (P1 CLAHE)
  - Local contextual contrast enhancement in 80×80 tiles
  - clipLimit=2.0, tileGridSize=(8, 8), CIELAB L* channel
               │
               ▼
Final AI-Ready Sonar Image (640×640×3 uint8)
```

### Excluded Methods and Rationale
- **Gaussian Blur ($5\times5$):** Rejected due to excessive edge blurring (EPI 0.322 vs 0.500 for Bilateral).
- **Median Blur ($5\times5$):** Rejected due to geometric corner erosion (EPI 0.284).
- **Frost Speckle Filter ($5\times5$):** Rejected due to acoustic shadow dilution and 4.9× higher processing latency.
- **Lee Speckle Filter ($5\times5$):** Excluded from primary pipeline because Bilateral filter already provides edge-preserving spatial/range smoothing with higher signal fidelity (PSNR 34.4 dB vs 30.3 dB); chaining both causes redundant over-smoothing.
- **Morphological Top-Hat / Black-Hat (P4):** Excluded from direct image enhancement because it strips ambient seafloor contextual texture and activates natural sand ripple crests/troughs (12.7% clutter on 9×9 ellipse), creating severe false-positive risks for object detectors. Reserved as an auxiliary multi-spectral research channel.

### 8. Final AI-Ready Dataset Generation
To run the automated builder and integrity verification engine:
```bash
python computer_vision/final_dataset_builder.py
```
This script:
1. Validates `SIH_Anomaly_V1` source dataset without modifying source files.
2. Clones directory structure and copies all labels byte-for-byte into `final_ai_ready_dataset/`.
3. Processes all 564 images across `train` (420), `val` (96), and `test` (48) splits using multi-threaded execution.
4. Generates visual before/after comparison panels in `computer_vision/final_pipeline_results/`.
5. Conducts automated mathematical integrity checks (dimensions, data types, label identity, corruption checks).
6. Compiles the comprehensive dataset verification report at `final_dataset_builder_report.md`.

---

## Compliance and Constraints Observed
- **Strictly non-destructive:** Original dataset files in `SIH_Anomaly_V1` remain untouched.
- **Resolution and channels preserved:** All operations maintain native side-scan sonar resolutions in 3-channel format.
- **Active classes maintained:** Strictly 3 classes maintained (`0: shipwreck`, `1: aircraft`, `2: mine`).
- **Zero downstream claims:** No claims regarding YOLO detection improvements (mAP, precision, recall) are made. Those belong strictly to Member 1's upcoming detection training experiments.
