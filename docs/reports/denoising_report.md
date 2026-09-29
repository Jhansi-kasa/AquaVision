# Sonar Image Denoising Evaluation & Comparison Report
**Project:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery  
**Subsystem:** Member 2 (Computer Vision & Sonar Processing)  
**Stage:** Denoising Stage (Pipeline: Raw Sonar → **Denoising** → Normalization → CLAHE → AI-Ready Image)  

---

## 1. Executive Summary
Sonar imagery poses unique denoising challenges due to multiplicative acoustic speckle noise, low contrast, and weak target boundaries. Smoothing must reduce granular speckle in homogeneous seafloor regions **without eroding subtle highlights or blurring acoustic shadows** (which are essential visual cues for underwater hazard detection).

In this stage, three primary OpenCV filtering algorithms were implemented, configured, and benchmarked across **6 representative real-world side-scan sonar images** from the dataset:
1. **Gaussian Blur** (Linear isotropic spatial smoothing)
2. **Median Blur** (Non-linear rank-order median filtering)
3. **Bilateral Filter** (Non-linear edge-preserving spatial & radiometric filtering)

### Quantitative Benchmark Summary
| Method | Configured Parameters | Mean EPI (Edge Preservation) | Mean Speckle Reduction (%) | Mean PSNR (dB) | Mean MSE | Mean Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gaussian Blur** | `ksize=(5, 5), sigmaX=1.2` | **0.3187** | **27.80%** | **29.43** | **120.73** | **1.91 ms** |
| **Median Blur** | `ksize=5` | **0.2806** | **30.11%** | **27.75** | **182.68** | **3.11 ms** |
| **Bilateral Filter** | `d=7, sigmaColor=50, sigmaSpace=50` | **0.4998** | **16.22%** | **34.40** | **27.01** | **23.17 ms** |
> **Primary Recommendation:** The **Bilateral Filter** is empirically superior for the sonar preprocessing pipeline. It provides a **0.4998 Edge Preservation Index (EPI)** (nearly 1.6× to 1.8× higher than Gaussian or Median) and the lowest distortion (**PSNR 34.40 dB**, **MSE 27.01**), ensuring that weak acoustic boundaries and shadow penumbras are preserved for downstream YOLO feature extraction.

---

## 2. Methods Tested and Algorithmic Principles
### 2.1 Gaussian Blur (`cv2.GaussianBlur`)
- **Mathematical Principle:** Convolves the 2D image with an isotropic 2D Gaussian kernel:
  $$G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
- **Parameters Used:** `ksize = (5, 5)`, `sigma_x = 1.2`.
- **Behavior on Sonar:** Effective at attenuating high-frequency speckle variations, but operates blindly across edge boundaries. It uniformly attenuates high-frequency energy, blurring sharp target-shadow interfaces and washing out faint debris outlines.

### 2.2 Median Blur (`cv2.medianBlur`)
- **Mathematical Principle:** Slides an aperture window over the image and replaces the center pixel with the statistical median value of the local neighborhood:
  $$I_{median}(x, y) = \text{median}\{I(x+i, y+j) \mid (i, j) \in W\}$$
- **Parameters Used:** `ksize = 5`.
- **Behavior on Sonar:** Outstanding at removing extreme acoustic outliers and high-amplitude backscatter spikes. However, non-linear median ranking distorts fine geometric structures (such as thin cables, mine tethers, and crab pot mesh) into blocky staircases and erodes corner geometry.

### 2.3 Bilateral Filter (`cv2.bilateralFilter`)
- **Mathematical Principle:** Combines a geometric spatial domain Gaussian weight with a photometric radiometric range Gaussian weight:
  $$I_{bilat}(p) = \frac{1}{W_p} \sum_{q \in S} I(q) \cdot \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{|I(p) - I(q)|^2}{2\sigma_r^2}\right)$$
- **Parameters Used:** `d = 7`, `sigma_color = 50.0`, `sigma_space = 50.0`.
- **Behavior on Sonar:** Averages neighboring pixels only if their acoustic intensities are similar (homogeneous seafloor background). Across the boundary between a bright acoustic highlight ($I \approx 180-250$) and its deep acoustic shadow ($I \approx 0-25$), the range Gaussian term decays to zero, preventing cross-boundary blurring.

---

## 3. Sample Images Tested from Dataset
Six representative sonar samples were selected across different anomaly classes and background terrain:

| Sample ID | Target Class | File Name | Image Dimensions | Acoustic Scene Description |
| :---: | :--- | :--- | :---: | :--- |
| 0 | `shipwreck` | `seabed_000226_jpg.rf.5bb9d7223f1e9b658d10a4a3ff3fc024.jpg` | 640 × 640 × 3 | Large shipwreck structure with complex acoustic shadows and internal textural hull lines. |
| 1 | `drowning_victim` | `seabed_000119_jpg.rf.8fd92291e86636da4cc4ea4033b7c8e6.jpg` | 640 × 640 × 3 | Small submerged profile with low contrast and faint, weak object boundaries. |
| 2 | `aircraft` | `seabed_000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg` | 640 × 640 × 3 | Submerged aircraft fuselage exhibiting sharp high-return reflections and elongated shadow. |
| 3 | `mine` | `seabed_000016_jpg.rf.059f50f2c5b3f0a73a433734847c1c30.jpg` | 640 × 640 × 3 | Compact spherical/cylindrical mine hazard requiring boundary retention to avoid false negatives. |
| 5 | `crab_pot` | `gv_BC_POST_T2_00_00_2_8_png_jpg.rf.a666cb2470c54c8f05a79f4fb70b2f95.jpg` | 640 × 640 × 3 | Tiny rectangular debris trap easily blurred or erased by excessive isotropic filtering. |
| 4 | `seafloor_background` | `gv_Contact_101_sslo_png_jpg.rf.129d97ac37fee40e19e9fdd547125845.jpg` | 640 × 640 × 3 | Negative background image displaying natural sedimentary sand ripples and speckle field. |

---

## 4. Per-Sample Experimental Results

### Sample 1: `shipwreck` (seabed_000226_jpg.rf.5bb9d7223f1e9b658d10a4a3ff3fc024.jpg)
- **Scene Characteristics:** Large shipwreck structure with complex acoustic shadows and internal textural hull lines.

| Metric | Raw Image | Gaussian Blur (5x5) | Median Blur (k=5) | Bilateral Filter (d=7) |
| :--- | :---: | :---: | :---: | :---: |
| **EPI (Edge Retention)** | 1.0000 | 0.2541 | 0.2262 | **0.4690** |
| **PSNR (dB)** | $\infty$ | 27.04 dB | 26.26 dB | **33.06 dB** |
| **MSE** | 0.0 | 128.47 | 153.99 | **32.12** |
| **Speckle Index (Local $C = \sigma/\mu$)** | 0.5843 | 0.5269 | 0.5408 | 0.5652 |
| **Speckle Reduction (%)** | 0.0% | -9.83% | -7.45% | -3.27% |
| **Computation Time** | 0.0 ms | 5.35 ms | 4.43 ms | 20.47 ms |

*Visual Comparison Image Saved:* `computer_vision/denoising_results/sample_1_shipwreck_comparison.jpg`

### Sample 2: `drowning_victim` (seabed_000119_jpg.rf.8fd92291e86636da4cc4ea4033b7c8e6.jpg)
- **Scene Characteristics:** Small submerged profile with low contrast and faint, weak object boundaries.

| Metric | Raw Image | Gaussian Blur (5x5) | Median Blur (k=5) | Bilateral Filter (d=7) |
| :--- | :---: | :---: | :---: | :---: |
| **EPI (Edge Retention)** | 1.0000 | 0.2380 | 0.1970 | **0.5658** |
| **PSNR (dB)** | $\infty$ | 24.51 dB | 22.42 dB | **31.98 dB** |
| **MSE** | 0.0 | 230.32 | 372.08 | **41.25** |
| **Speckle Index (Local $C = \sigma/\mu$)** | 0.4337 | 0.2751 | 0.2704 | 0.3104 |
| **Speckle Reduction (%)** | 0.0% | -36.58% | -37.64% | -28.43% |
| **Computation Time** | 0.0 ms | 1.13 ms | 2.72 ms | 33.93 ms |

*Visual Comparison Image Saved:* `computer_vision/denoising_results/sample_2_drowning_victim_comparison.jpg`

### Sample 3: `aircraft` (seabed_000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg)
- **Scene Characteristics:** Submerged aircraft fuselage exhibiting sharp high-return reflections and elongated shadow.

| Metric | Raw Image | Gaussian Blur (5x5) | Median Blur (k=5) | Bilateral Filter (d=7) |
| :--- | :---: | :---: | :---: | :---: |
| **EPI (Edge Retention)** | 1.0000 | 0.2728 | 0.2125 | **0.7012** |
| **PSNR (dB)** | $\infty$ | 23.69 dB | 21.66 dB | **31.60 dB** |
| **MSE** | 0.0 | 278.31 | 443.92 | **44.97** |
| **Speckle Index (Local $C = \sigma/\mu$)** | 0.2124 | 0.1166 | 0.0976 | 0.1728 |
| **Speckle Reduction (%)** | 0.0% | -45.09% | -54.05% | -18.62% |
| **Computation Time** | 0.0 ms | 1.15 ms | 2.42 ms | 20.10 ms |

*Visual Comparison Image Saved:* `computer_vision/denoising_results/sample_3_aircraft_comparison.jpg`

### Sample 4: `mine` (seabed_000016_jpg.rf.059f50f2c5b3f0a73a433734847c1c30.jpg)
- **Scene Characteristics:** Compact spherical/cylindrical mine hazard requiring boundary retention to avoid false negatives.

| Metric | Raw Image | Gaussian Blur (5x5) | Median Blur (k=5) | Bilateral Filter (d=7) |
| :--- | :---: | :---: | :---: | :---: |
| **EPI (Edge Retention)** | 1.0000 | 0.4947 | 0.5035 | **0.5351** |
| **PSNR (dB)** | $\infty$ | 33.18 dB | 31.10 dB | **35.65 dB** |
| **MSE** | 0.0 | 31.24 | 50.48 | **17.71** |
| **Speckle Index (Local $C = \sigma/\mu$)** | 0.2315 | 0.1932 | 0.1870 | 0.1914 |
| **Speckle Reduction (%)** | 0.0% | -16.55% | -19.24% | -17.35% |
| **Computation Time** | 0.0 ms | 0.86 ms | 2.53 ms | 20.16 ms |

*Visual Comparison Image Saved:* `computer_vision/denoising_results/sample_4_mine_comparison.jpg`

### Sample 5: `crab_pot` (gv_BC_POST_T2_00_00_2_8_png_jpg.rf.a666cb2470c54c8f05a79f4fb70b2f95.jpg)
- **Scene Characteristics:** Tiny rectangular debris trap easily blurred or erased by excessive isotropic filtering.

| Metric | Raw Image | Gaussian Blur (5x5) | Median Blur (k=5) | Bilateral Filter (d=7) |
| :--- | :---: | :---: | :---: | :---: |
| **EPI (Edge Retention)** | 1.0000 | 0.3762 | 0.3341 | **0.4620** |
| **PSNR (dB)** | $\infty$ | 32.14 dB | 31.28 dB | **37.83 dB** |
| **MSE** | 0.0 | 39.72 | 48.40 | **10.73** |
| **Speckle Index (Local $C = \sigma/\mu$)** | 0.1136 | 0.0645 | 0.0589 | 0.0970 |
| **Speckle Reduction (%)** | 0.0% | -43.17% | -48.13% | -14.55% |
| **Computation Time** | 0.0 ms | 1.10 ms | 4.17 ms | 24.06 ms |

*Visual Comparison Image Saved:* `computer_vision/denoising_results/sample_5_crab_pot_comparison.jpg`

### Sample 6: `seafloor_background` (gv_Contact_101_sslo_png_jpg.rf.129d97ac37fee40e19e9fdd547125845.jpg)
- **Scene Characteristics:** Negative background image displaying natural sedimentary sand ripples and speckle field.

| Metric | Raw Image | Gaussian Blur (5x5) | Median Blur (k=5) | Bilateral Filter (d=7) |
| :--- | :---: | :---: | :---: | :---: |
| **EPI (Edge Retention)** | 1.0000 | 0.2762 | 0.2101 | **0.2658** |
| **PSNR (dB)** | $\infty$ | 36.01 dB | 33.78 dB | **36.30 dB** |
| **MSE** | 0.0 | 16.29 | 27.22 | **15.25** |
| **Speckle Index (Local $C = \sigma/\mu$)** | 0.3924 | 0.3313 | 0.3369 | 0.3333 |
| **Speckle Reduction (%)** | 0.0% | -15.56% | -14.13% | -15.07% |
| **Computation Time** | 0.0 ms | 1.84 ms | 2.41 ms | 20.33 ms |

*Visual Comparison Image Saved:* `computer_vision/denoising_results/sample_6_seafloor_background_comparison.jpg`

---

## 5. Visual Observations and Qualitative Assessment
1. **Raw Sonar Images:**
   - Contain widespread granular speckle noise across sandy and silty seafloors.
   - Small targets (crab pots, drowning victims, mines) exhibit subtle acoustic highlights accompanied by narrow acoustic shadows.
   - High-contrast isolated noise spikes mimic false-positive micro-debris.

2. **Gaussian Blur Visual Observations:**
   - Successfully attenuates high-frequency speckle grains.
   - **Major Flaw:** Weak object boundaries are severely smeared into surrounding seafloor texture. Small targets lose their sharp highlight peaks, and acoustic shadow penumbras become blurred and dilated.

3. **Median Blur Visual Observations:**
   - Highly effective at suppressing impulsive salt-and-pepper noise spikes.
   - **Major Flaw:** Produces piecewise constant, cartoonish blocky patches in textured seabed regions. Sharp corners of man-made structures (aircraft wings, crab pot cages) are rounded or partially erased.

4. **Bilateral Filter Visual Observations:**
   - Background seafloor sediment is noticeably smoothed, significantly suppressing granular noise.
   - **Key Advantage:** Sharp transitions between acoustic highlight and acoustic shadow remain crisp and visually prominent.
   - Weak object boundaries are preserved far better than in Gaussian or Median results, maintaining critical structural signatures needed for YOLO bounding-box detection.

---

## 6. Detailed Comparison: Boundaries, Noise, and Disadvantages

### 6.1 Which Method Preserves Object Boundaries Best?
- **Winner: Bilateral Filter.**
- Across all 6 representative samples, the Bilateral Filter recorded an average **EPI of 0.4998**, compared to **0.3187** for Gaussian Blur and **0.2806** for Median Blur.
- The range-weighting term in Bilateral filtering acts as an adaptive boundary protector: when the intensity gradient exceeds $\sigma_{color}$, filtering stops across the boundary, preserving highlight-shadow interfaces.

### 6.2 Which Method Removes Noise Best?
- **Winner for Speckle Reduction: Gaussian Blur and Median Blur.**
- Gaussian Blur achieved **27.8% speckle reduction**, and Median Blur achieved **30.1% speckle reduction**.
- However, in sonar target detection, **maximum smoothing is detrimental**. Over-smoothing destroys weak acoustic returns from small targets (such as crab pots, which make up 73% of the dataset) and blurs acoustic shadow geometry.
- Bilateral filtering achieved a balanced **16.2% speckle reduction** in homogeneous patches while maintaining superior boundary integrity, providing the optimal trade-off.

### 6.3 Disadvantages Observed per Method
| Method | Key Disadvantages & Failure Modes |
| :--- | :--- |
| **Gaussian Blur** | Blurs weak target boundaries; reduces peak backscatter contrast; dilates shadow penumbras; cannot distinguish between noise and genuine target edges. |
| **Median Blur** | Rounds off geometric corners of man-made debris; introduces blocky staircase artifacts; erodes thin elongated structures (e.g. chains, tethers, victim limbs). |
| **Bilateral Filter** | Higher computational cost (approx. 23.2 ms vs 1.9 ms per 640x640 frame); requires careful tuning of $\sigma_{color}$ to avoid over-smoothing weak targets. |

---

## 7. Pipeline Recommendation for Member 2
For the next stage of the preprocessing pipeline:
$$\text{Raw Sonar Image} \longrightarrow \mathbf{\text{Bilateral Denoising (d=7, } \sigma_c=50, \sigma_s=50\text{)}} \longrightarrow \text{Normalization} \longrightarrow \text{CLAHE} \longrightarrow \text{AI Model}$$

The Bilateral filter is selected as the recommended baseline denoising operator because it protects weak target boundaries while conditioning the image for contrast enhancement (CLAHE) without amplifying high-frequency speckle.

*(Report generated automatically via `denoising_comparison.py`)*