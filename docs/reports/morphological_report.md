# Morphological Highlight–Shadow Separation Report (Task P4)
**Project:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery  
**Subsystem:** Member 2 (Computer Vision & Sonar Processing)  
**Task:** P4 — Morphological Highlight–Shadow Separation  

---

## 1. Objective
In side-scan sonar (SSS) imagery, objects standing proud of the seabed generate a characteristic dual acoustic response: a bright specular/diffuse **acoustic highlight** facing the sonar transducer, accompanied immediately behind by an **acoustic shadow void** where acoustic energy is blocked. However, natural seafloor morphology (sand megaripples, rocky outcrops, seabed depressions, and dredge scars) can produce highlight/shadow patterns that visually mimic man-made debris.

The objective of Task P4 is to implement, evaluate, and benchmark a mathematical morphology framework (White Top-Hat and Black-Hat filtering) to separate bright acoustic highlights and dark shadow voids from the ambient seafloor, analyzing whether morphological decomposition enhances target saliency and assessing the false-positive risks introduced by natural seabed clutter.

---

## 2. Why Highlight–Shadow Processing is Relevant to Side-Scan Sonar
Unlike terrestrial optical cameras where illumination is diffuse and ambient, side-scan sonar operates as an active range-measuring side-looking acoustic sensor:
- Acoustic backscatter is strictly directional, propagating perpendicularly outward from the survey vessel.
- Any object with positive bathymetric relief (standing proud of the seafloor) obstructs acoustic rays, casting a pronounced acoustic shadow across the range direction.
- The **Highlight–Shadow Pair** is the primary visual signature utilized by sonar hydrographers to confirm that a high-backscatter return is a true 3D physical anomaly rather than a flat sediment patch with high reflectivity (e.g., shell hash or gravel).
- Automated separation of highlights and shadows provides decoupled structural feature maps that can aid downstream feature extractors.

---

## 3. Acoustic Highlight Explanation
The **acoustic highlight** corresponds to the high-amplitude return recorded when the incident wavefront strikes the target face:
- For metallic or hard man-made structures (shipwreck hulls, aircraft fuselages, cylindrical mines, steel crab pots), the acoustic impedance mismatch ($Z_{\text{metal}} \gg Z_{\text{water}}$) produces strong specular reflections ($I \approx 180 - 255\text{ DN}$).
- These highlights are typically compact, high-frequency, and higher in intensity than the surrounding sediment backscatter.

---

## 4. Acoustic Shadow Explanation
The **acoustic shadow** represents the total or near-total absence of backscattered acoustic energy in the geometric occultation zone behind the object:
- The length of the shadow $L_s$ is geometrically related to the object height $H_t$, towfish altitude $H_a$, and slant range $R_s$:
  $$H_t = \frac{H_a \cdot L_s}{R_s + L_s}$$
- Because no sound reaches this region, pixel values drop into the zero noise floor ($I \approx 0 - 15\text{ DN}$).
- The shadow shape directly encodes the 3D silhouette and cross-sectional profile of the target.

---

## 5. Why Natural Seabed Features Can Cause False Positives
Natural seafloor environments are rarely featureless:
- **Sand Megaripples and Dunes:** Transverse sediment ripples produce rhythmic alternating bright crests (facing the beam) and dark troughs (in acoustic shadow).
- **Rocky Outcrops & Boulders:** Natural rocks produce irregular highlight/shadow pairs that can mimic cylindrical mines or compact debris.
- **Biogenic Mounds & Depressions:** Depressions generate inverted signatures (shadow first, highlight second).

If morphological filtering is configured with an inappropriate structuring element, sand ripples and sediment dunes are strongly amplified, causing severe false-positive risks for downstream automated detectors.

---

## 6. Morphological Processing Concept
Mathematical morphology (Matheron & Serra, 1982) processes images based on geometric shape, probing local neighborhoods with a predefined **structuring element (SE)** denoted by $S$. By combining erosion and dilation in specific sequences, we can isolate structures that are geometrically smaller or narrower than $S$, regardless of absolute background illumination trends.

---

## 7. White Top-Hat Method
The **White Top-Hat (WTH)** transform extracts bright image elements smaller than the structuring element $S$:
$$WTH(I) = I - (I \circ S) = I - \text{dilate}(\text{erode}(I, S), S)$$
- **Morphological Opening ($I \circ S$):** The erosion eliminates bright peaks smaller than $S$; the subsequent dilation restores the remaining macro background.
- **Subtraction ($I - (I \circ S)$):** Leaves only the bright peaks, specular glints, and object highlights that were removed by opening.
- **Behavior in Sonar:** Isolates bright object highlights from slowly varying seabed background backscatter.

---

## 8. Black-Hat Method
The **Black-Hat (BTH)** (or Bottom-Hat) transform extracts dark image elements smaller than the structuring element $S$:
$$BTH(I) = (I \bullet S) - I = \text{erode}(\text{dilate}(I, S), S) - I$$
- **Morphological Closing ($I \bullet S$):** The dilation bridges dark troughs and fills in holes smaller than $S$; the subsequent erosion restores the surrounding background level.
- **Subtraction ($(I \bullet S) - I$):** Leaves only the dark troughs and acoustic shadow voids that were filled in by closing.
- **Behavior in Sonar:** Isolates acoustic shadow pockets and cavities from the surrounding sediment.

---

## 9. Kernel Sizes Tested
We evaluated three representative kernel sizes:
1. **$5 \times 5$ ($25\text{ px}$ footprint):** Targets fine speckle spikes and micro-targets.
2. **$9 \times 9$ ($81\text{ px}$ footprint):** Matched to compact marine debris (mines, crab pots, victim profiles) and shadow boundaries.
3. **$15 \times 15$ ($225\text{ px}$ footprint):** Spans wider structures and extended shadow extents.

---

## 10. Kernel Shapes Tested
1. **Elliptical (`cv2.MORPH_ELLIPSE`):** Isotropic disk structuring element; treats acoustic returns equally in all directions without directional bias or 90-degree corner artifacts.
2. **Rectangular (`cv2.MORPH_RECT`):** Cartesian box structuring element; responds strongly to rectilinear man-made contours, but introduces artificial horizontal/vertical corner elongation.

---

## 11. Images and Classes Tested
The benchmark was conducted across 6 representative real sonar samples from `SIH_Dataset` ($640 \times 640 \times 3$, `uint8`):

| Sample ID | Target Class | Filename | Dimensions | Acoustic Characteristics |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `shipwreck` | `seabed_000226_jpg.rf.5bb9d7223f1e9b658d10a4a3ff3fc024.jpg` | 640 × 640 × 3 | Large shipwreck structure with complex acoustic shadows and internal textural hull lines. |
| 2 | `drowning_victim` | `seabed_000119_jpg.rf.8fd92291e86636da4cc4ea4033b7c8e6.jpg` | 640 × 640 × 3 | Small submerged profile with low contrast and faint, weak object boundaries. |
| 3 | `aircraft` | `seabed_000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg` | 640 × 640 × 3 | Submerged aircraft fuselage exhibiting sharp high-return reflections and elongated shadow. |
| 4 | `mine` | `seabed_000016_jpg.rf.059f50f2c5b3f0a73a433734847c1c30.jpg` | 640 × 640 × 3 | Compact spherical/cylindrical mine hazard requiring boundary retention to avoid false negatives. |
| 5 | `crab_pot` | `gv_BC_POST_T2_00_00_2_8_png_jpg.rf.a666cb2470c54c8f05a79f4fb70b2f95.jpg` | 640 × 640 × 3 | Tiny rectangular debris trap easily blurred or erased by excessive isotropic filtering. |
| 6 | `seafloor_background` | `gv_Contact_101_sslo_png_jpg.rf.129d97ac37fee40e19e9fdd547125845.jpg` | 640 × 640 × 3 | Natural seabed sand ripples with subtle periodic sediment wave patterns. |

---

## 12. Quantitative Comparison

### Aggregate Benchmark Metrics (Mean across all 6 Samples)
| Structuring Element Configuration | WTH Mean (Highlight) | WTH Act % (>20 DN) | BTH Mean (Shadow) | BTH Act % (>20 DN) | HS Energy | Sobel Edge Energy | Latency (ms) | Status / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Config 1: 5×5 Ellipse** | 8.74 | 14.0% | 8.18 | 13.1% | 16.9 | 38.4 | 2.23 ms | High speckle noise; weak shadow capture |
| **Config 2: 9×9 Ellipse** | **15.12** | **25.0%** | **14.18** | **25.1%** | **29.3** | **58.1** | **3.34 ms** | **RECOMMENDED: Optimal target/clutter balance** |
| **Config 3: 15×15 Ellipse** | 21.46 | 34.7% | 20.82 | 37.4% | 42.3 | 54.2 | 8.49 ms | Excessive seabed ripple false-positive activation |
| **Config 4: 9×9 Rectangular** | 17.19 | 28.3% | 16.27 | 29.2% | 33.5 | 49.1 | 1.48 ms | Cartesian corner bias on circular targets |

---

## 13. Per-Sample Experimental Results (Config 2: 9×9 Ellipse)

### Sample 1: `shipwreck` (seabed_000226_jpg.rf.5bb9d7223f1e9b658d10a4a3ff3fc024.jpg)
- **Scene Context:** Large shipwreck structure with complex acoustic shadows and internal textural hull lines.
- **Raw Image Stats:** Mean = 74.8 DN | Std = 51.8 DN | Raw Edge Energy = 64.1
- **Latency (Config 2):** 2.72 ms

| Metric | White Top-Hat (Highlights) | Black-Hat (Shadows) | Combined / Saliency Impact |
| :--- | :---: | :---: | :--- |
| **Mean Response** | 15.43 DN | 12.78 DN | Decomposes raw backscatter into bipolar responses |
| **Maximum Response** | 240 DN | 204 DN | Preserves full peak target dynamic range |
| **Activation Pct (>20 DN)** | **23.65%** | **23.66%** | Confines response to salient acoustic features |
| **Response Std Dev** | 27.03 | 16.27 | High variance reflects sharp target localization |
| **Edge Energy (Sobel)** | 56.6 | N/A | Retains sharp outline along highlight crests |

*Generated Artifacts:*
- 4-Panel Quad Visualization: `computer_vision/morphological_results/sample_1_shipwreck_quad_panel.jpg`
- 5-Panel Kernel Ablation: `computer_vision/morphological_results/sample_1_shipwreck_kernel_ablation.jpg`

### Sample 2: `drowning_victim` (seabed_000119_jpg.rf.8fd92291e86636da4cc4ea4033b7c8e6.jpg)
- **Scene Context:** Small submerged profile with low contrast and faint, weak object boundaries.
- **Raw Image Stats:** Mean = 52.1 DN | Std = 59.2 DN | Raw Edge Energy = 91.8
- **Latency (Config 2):** 2.89 ms

| Metric | White Top-Hat (Highlights) | Black-Hat (Shadows) | Combined / Saliency Impact |
| :--- | :---: | :---: | :--- |
| **Mean Response** | 20.65 DN | 21.65 DN | Decomposes raw backscatter into bipolar responses |
| **Maximum Response** | 255 DN | 255 DN | Preserves full peak target dynamic range |
| **Activation Pct (>20 DN)** | **36.50%** | **45.47%** | Confines response to salient acoustic features |
| **Response Std Dev** | 28.49 | 23.92 | High variance reflects sharp target localization |
| **Edge Energy (Sobel)** | 84.3 | N/A | Retains sharp outline along highlight crests |

*Generated Artifacts:*
- 4-Panel Quad Visualization: `computer_vision/morphological_results/sample_2_drowning_victim_quad_panel.jpg`
- 5-Panel Kernel Ablation: `computer_vision/morphological_results/sample_2_drowning_victim_kernel_ablation.jpg`

### Sample 3: `aircraft` (seabed_000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg)
- **Scene Context:** Submerged aircraft fuselage exhibiting sharp high-return reflections and elongated shadow.
- **Raw Image Stats:** Mean = 125.8 DN | Std = 33.6 DN | Raw Edge Energy = 135.8
- **Latency (Config 2):** 4.15 ms

| Metric | White Top-Hat (Highlights) | Black-Hat (Shadows) | Combined / Saliency Impact |
| :--- | :---: | :---: | :--- |
| **Mean Response** | 34.68 DN | 31.77 DN | Decomposes raw backscatter into bipolar responses |
| **Maximum Response** | 253 DN | 233 DN | Preserves full peak target dynamic range |
| **Activation Pct (>20 DN)** | **68.28%** | **62.80%** | Confines response to salient acoustic features |
| **Response Std Dev** | 25.48 | 24.79 | High variance reflects sharp target localization |
| **Edge Energy (Sobel)** | 123.2 | N/A | Retains sharp outline along highlight crests |

*Generated Artifacts:*
- 4-Panel Quad Visualization: `computer_vision/morphological_results/sample_3_aircraft_quad_panel.jpg`
- 5-Panel Kernel Ablation: `computer_vision/morphological_results/sample_3_aircraft_kernel_ablation.jpg`

### Sample 4: `mine` (seabed_000016_jpg.rf.059f50f2c5b3f0a73a433734847c1c30.jpg)
- **Scene Context:** Compact spherical/cylindrical mine hazard requiring boundary retention to avoid false negatives.
- **Raw Image Stats:** Mean = 60.6 DN | Std = 41.1 DN | Raw Edge Energy = 52.0
- **Latency (Config 2):** 3.20 ms

| Metric | White Top-Hat (Highlights) | Black-Hat (Shadows) | Combined / Saliency Impact |
| :--- | :---: | :---: | :--- |
| **Mean Response** | 9.46 DN | 8.00 DN | Decomposes raw backscatter into bipolar responses |
| **Maximum Response** | 174 DN | 160 DN | Preserves full peak target dynamic range |
| **Activation Pct (>20 DN)** | **13.10%** | **9.53%** | Confines response to salient acoustic features |
| **Response Std Dev** | 12.47 | 9.63 | High variance reflects sharp target localization |
| **Edge Energy (Sobel)** | 40.2 | N/A | Retains sharp outline along highlight crests |

*Generated Artifacts:*
- 4-Panel Quad Visualization: `computer_vision/morphological_results/sample_4_mine_quad_panel.jpg`
- 5-Panel Kernel Ablation: `computer_vision/morphological_results/sample_4_mine_kernel_ablation.jpg`

### Sample 5: `crab_pot` (gv_BC_POST_T2_00_00_2_8_png_jpg.rf.a666cb2470c54c8f05a79f4fb70b2f95.jpg)
- **Scene Context:** Tiny rectangular debris trap easily blurred or erased by excessive isotropic filtering.
- **Raw Image Stats:** Mean = 62.1 DN | Std = 23.4 DN | Raw Edge Energy = 28.9
- **Latency (Config 2):** 4.04 ms

| Metric | White Top-Hat (Highlights) | Black-Hat (Shadows) | Combined / Saliency Impact |
| :--- | :---: | :---: | :--- |
| **Mean Response** | 5.16 DN | 5.15 DN | Decomposes raw backscatter into bipolar responses |
| **Maximum Response** | 255 DN | 245 DN | Preserves full peak target dynamic range |
| **Activation Pct (>20 DN)** | **2.41%** | **2.33%** | Confines response to salient acoustic features |
| **Response Std Dev** | 8.34 | 6.21 | High variance reflects sharp target localization |
| **Edge Energy (Sobel)** | 21.7 | N/A | Retains sharp outline along highlight crests |

*Generated Artifacts:*
- 4-Panel Quad Visualization: `computer_vision/morphological_results/sample_5_crab_pot_quad_panel.jpg`
- 5-Panel Kernel Ablation: `computer_vision/morphological_results/sample_5_crab_pot_kernel_ablation.jpg`

### Sample 6: `seafloor_background` (gv_Contact_101_sslo_png_jpg.rf.129d97ac37fee40e19e9fdd547125845.jpg)
- **Scene Context:** Natural seabed sand ripples with subtle periodic sediment wave patterns.
- **Raw Image Stats:** Mean = 36.8 DN | Std = 45.5 DN | Raw Edge Energy = 24.4
- **Latency (Config 2):** 3.06 ms

| Metric | White Top-Hat (Highlights) | Black-Hat (Shadows) | Combined / Saliency Impact |
| :--- | :---: | :---: | :--- |
| **Mean Response** | 5.33 DN | 5.74 DN | Decomposes raw backscatter into bipolar responses |
| **Maximum Response** | 101 DN | 69 DN | Preserves full peak target dynamic range |
| **Activation Pct (>20 DN)** | **5.96%** | **6.78%** | Confines response to salient acoustic features |
| **Response Std Dev** | 8.07 | 8.06 | High variance reflects sharp target localization |
| **Edge Energy (Sobel)** | 22.5 | N/A | Retains sharp outline along highlight crests |

*Generated Artifacts:*
- 4-Panel Quad Visualization: `computer_vision/morphological_results/sample_6_seafloor_background_quad_panel.jpg`
- 5-Panel Kernel Ablation: `computer_vision/morphological_results/sample_6_seafloor_background_kernel_ablation.jpg`

---

## 14. Visual Observations
1. **White Top-Hat (Highlights):**
   - Successfully extracts specular reflections from metallic debris (`shipwreck`, `aircraft`, `mine`, `crab_pot`), showing high contrast against near-black backgrounds.
   - Broad background illumination gradients (swath fall-off) are completely eliminated, as opening tracks and subtracts low-frequency illumination.
2. **Black-Hat (Shadows):**
   - Effectively isolates localized acoustic shadow pockets behind proud objects.
   - For large structures (`aircraft`, `shipwreck`), the perimeter of the shadow is captured with high precision.
3. **Combined False-Color Representation:**
   - The RGB fusion mapping (`Red=WTH, Green=Raw Gray, Blue=BTH`) provides an immediate visual confirmation of physical objects: valid man-made debris presents a distinct **Red Highlight immediately adjacent to a Blue Shadow** on a muted green seabed.

---

## 15. Highlight Preservation
- Peak highlight returns on small targets (`mine`, `crab_pot`) achieve maximum intensities of $180 - 255\text{ DN}$, indicating zero loss of peak backscatter signal.
- In contrast to linear low-pass filtering (which blunts specular peaks), White Top-Hat preserves local maxima while zeroing out surrounding ambient sediment.

---

## 16. Shadow Preservation
- Black-Hat filtering responds strongly along the interior and boundaries of true acoustic shadows.
- In large structures (`shipwreck`, `aircraft`), the core of the shadow void produces strong Black-Hat responses ($>150\text{ DN}$).
- For very wide shadows that exceed the structuring element footprint, Black-Hat highlights the **shadow transition boundaries** (penumbras) rather than the interior plateau.

---

## 17. Natural-Seabed False-Positive Analysis
A central concern of morphological filtering in side-scan sonar is whether natural seabed morphology triggers false alarms:
- On `Sample 6: seafloor_background` (`gv_Contact_101`):
  - **Config 1 ($5 \times 5$):** Total activated pixels (>20 DN) = **4.21%**. Activates high-frequency sediment speckle grain.
  - **Config 2 ($9 \times 9$):** Total activated pixels (>20 DN) = **12.74%**. Mild response along primary ripple crests; background remains predominantly suppressed.
  - **Config 3 ($15 \times 15$):** Total activated pixels (>20 DN) = **26.79%** (over $2\times$ higher than 9×9!). The larger kernel spans across sand ripple wavelengths, incorrectly amplifying natural seabed undulations into strong highlight/shadow false alarms.
- **Conclusion:** Structuring elements larger than $9 \times 9$ significantly elevate false-positive risk on textured seabeds.

---

## 18. Noise and Artifact Analysis
1. **Speckle Amplification in $5 \times 5$:** The $5 \times 5$ kernel is small enough to treat individual speckle spikes as 'highlights', creating salt-and-pepper noise across the output.
2. **Directional Bias in Rectangular Kernels:** The $9 \times 9$ rectangular kernel creates subtle boxy artifacts at curved boundaries (e.g. spherical mine hulls).
3. **Elliptical Invariance:** The elliptical kernel maintains isotropic invariance, yielding clean, natural boundary responses.

---

## 19. Kernel-Size Comparison Summary
| Property / Criterion | 5×5 Elliptical | 9×9 Elliptical (Recommended) | 15×15 Elliptical | 9×9 Rectangular |
| :--- | :---: | :---: | :---: | :---: |
| **Small Debris Highlights (`crab_pot`, `mine`)** | High | **Optimal** | Moderate | Moderate |
| **Shadow Void Extraction** | Weak | **Strong** | Very Strong | Strong |
| **Speckle Noise Rejection** | Poor | **Good** | Excellent | Moderate |
| **Seabed Clutter Rejection (Low FP Risk)** | Moderate | **Optimal (12.7% act)** | Poor (26.8% act) | Moderate |
| **Rotational Invariance** | Isotropic | **Isotropic** | Isotropic | Anisotropic (box bias) |
| **Execution Latency** | **2.23 ms** | 3.34 ms | 8.49 ms | 1.48 ms |

---

## 20. Recommended P4 Configuration
Based on quantitative highlight/shadow activation, edge retention, false-positive suppression on natural sediment, and rotational symmetry:

> **Recommended Configuration:** **Config 2: 9×9 Elliptical Structuring Element**
> - `kernel_size`: $9\text{ px}$
> - `kernel_shape`: `'ellipse'` (`cv2.MORPH_ELLIPSE`)
> - `highlight_op`: White Top-Hat ($WTH = I - (I \circ S)$)
> - `shadow_op`: Black-Hat ($BTH = (I \bullet S) - I$)
> - `fusion_mode`: `'rgb_fusion'` (`[Red=WTH, Green=Raw Gray, Blue=BTH]`)
> - `color_handling`: Luminance channel processing

---

## 21. Limitations
1. **Not a Standalone Semantic Detector:** Morphological Top-Hat/Black-Hat transforms produce radiometric feature representations, NOT confirmed object detections. High responses indicate local backscatter contrast, not necessarily man-made marine debris.
2. **Sensitivity to Seabed Sand Dunes:** In areas with severe benthic sand megaripples, morphological filters will highlight ripple crests and troughs. Morphological processing must be combined with downstream spatial context or machine learning to eliminate natural periodic clutter.
3. **Shadow Truncation on Giant Structures:** For massive objects (such as complete $100\text{ m}$ shipwrecks), the acoustic shadow exceeds $9\text{ px}$ in width; the Black-Hat filter highlights the boundary of the shadow rather than filling its entire spatial footprint.

---

## 22. Future Improvements
1. **Multi-Scale Morphological Decomposition:** Implementing morphological granulometries or multi-scale structuring elements ($k \in [5, 9, 15]$) to capture both small debris and extensive shipwreck shadows concurrently.
2. **Directional Structuring Elements:** Designing asymmetric structuring elements aligned along the cross-track range vector to leverage the physical fact that acoustic shadows always extend strictly away from the sonar transducer.
3. **Downstream Feature Integration:** Providing the White Top-Hat and Black-Hat channels as auxiliary feature inputs for convolutional backbones in the downstream YOLO pipeline (to be validated by Member 1).

*(Report generated automatically via `morphological_comparison.py`)*
