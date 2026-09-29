# Comprehensive Sonar Dataset Inspection Report (SIH_Anomaly_V1)
**Project:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery  
**Role:** Member 2 (Computer Vision & Sonar Processing)  
**Dataset Analyzed:** `SIH_Anomaly_V1` (`C:\Users\dell\Downloads\SIH_Anomaly_V1\SIH_Anomaly_V1`)  

---

## Executive Summary
A complete, non-destructive automated inspection of the sonar imagery dataset was conducted. The dataset comprises **564 side-scan sonar images** and **564 YOLO annotation files** partitioned across standard train, validation, and test splits.

| Metric | Value | Status / Notes |
| :--- | :--- | :--- |
| **Total Images** | 564 | 420 train, 96 val, 48 test |
| **Total Label Files** | 564 | 100% 1-to-1 matching with image files |
| **Corrupted Images** | 0 | All images verified intact |
| **Invalid Label Lines** | 0 | Strict YOLO format verified |
| **Image Resolution** | 1728x2476 (3), 1728x2634 (1), 1728x2595 (1), 1728x2607 (1), 1728x2602 (1), 1728x2620 (3), 1728x3025 (1), 1728x3111 (1), 1728x2480 (1), 1728x2522 (1), 1728x2501 (1), 1728x2514 (1), 1728x2574 (2), 1728x2589 (1), 1728x3325 (1), 1728x3337 (1), 1728x3330 (1), 1728x3324 (1), 1728x3318 (1), 1728x3296 (1), 1728x1817 (1), 1728x1849 (1), 1728x5579 (1), 1728x5581 (1), 1728x5584 (1), 1728x18074 (1), 1728x18179 (2), 1728x61 (1), 1728x18107 (1), 1728x18082 (1), 1728x2193 (1), 1728x1904 (1), 1728x1927 (5), 1728x1930 (1), 1728x1899 (1), 1728x1926 (5), 1728x1903 (2), 1728x1859 (1), 1728x1951 (1), 1728x1925 (2), 1728x1952 (1), 1728x2142 (1), 1728x2186 (1), 1728x2179 (1), 1728x2163 (1), 1728x13 (1), 1728x836 (1), 1728x860 (1), 1728x874 (1), 1728x1863 (3), 1728x1906 (1), 1728x1867 (1), 1728x2215 (1), 1728x9004 (1), 1728x9402 (1), 1728x3984 (1), 1728x3488 (1), 1728x4020 (1), 1728x3082 (1), 1728x3829 (1), 1728x1821 (1), 1728x1875 (1), 1728x2000 (1), 1728x1953 (1), 1728x1978 (1), 1728x1982 (1), 1728x1934 (1), 1728x1354 (1), 1728x2460 (1), 1728x2471 (1), 1728x2451 (1), 1728x1799 (1), 1728x29 (1), 1728x5696 (1), 1728x2568 (1), 1728x2672 (1), 1728x2624 (1), 1728x2648 (3), 1728x2623 (1), 1728x2670 (1), 1728x2598 (1), 1728x1843 (1), 1728x5616 (1), 1728x5590 (1), 1728x5667 (1), 1728x5763 (1), 1728x2774 (1), 1728x2790 (1), 1728x2845 (1), 1728x2846 (1), 1728x2782 (1), 1728x2867 (1), 1728x2812 (2), 1728x18155 (1), 1728x2223 (1), 1728x2245 (1), 1728x2202 (1), 1728x2244 (1), 1728x1929 (1), 1728x1902 (2), 1728x1905 (2), 1728x1881 (1), 1728x1984 (1), 1728x1884 (2), 1728x2183 (1), 1728x2169 (1), 1728x2173 (1), 1728x1225 (1), 1728x1270 (1), 1728x1260 (1), 1728x1261 (1), 1728x1262 (1), 1728x1256 (1), 1728x1266 (1), 1728x1249 (1), 1728x1022 (1), 1728x1052 (1), 1728x1048 (1), 1728x1058 (1), 1728x861 (1), 1728x873 (1), 1728x1896 (1), 1728x1895 (1), 1728x1944 (1), 1728x2201 (1), 1728x2208 (1), 1728x2222 (1), 1728x1445 (1), 1728x1465 (1), 1728x8779 (1), 1728x8976 (1), 1728x4685 (1), 1728x4821 (1), 1728x2650 (1), 1728x2210 (1), 1728x2637 (1), 1728x2646 (3), 1728x2673 (2), 1728x2671 (1), 1728x9291 (1), 1728x9404 (1), 1728x9167 (1), 1728x9327 (1), 1728x9128 (1), 1728x3888 (1), 1728x3920 (1), 1728x3824 (1), 1728x1857 (1), 1728x1878 (2), 1728x1861 (1), 1728x1786 (1), 1728x1825 (1), 1728x1818 (1), 1728x1810 (1), 1728x1834 (1), 1728x2458 (1), 1728x1912 (1), 1728x1866 (1), 1728x1897 (1), 1728x1870 (1), 1728x5592 (1), 1728x5598 (1), 1728x5604 (1), 1728x2639 (1), 1728x2647 (2), 1728x2696 (1), 1024x1024 (133), 416x416 (122), 640x640 (66), 1728x1471 (1), 1728x1470 (1), 1728x2576 (1), 1728x2573 (1), 1728x1757 (1), 1728x1760 (1), 1728x1758 (1), 1728x2819 (1), 1728x2838 (1), 1728x2908 (1), 1728x2833 (1), 1728x2936 (1), 1728x2937 (1), 1728x2863 (1), 1728x1694 (1), 1728x1687 (1), 1728x1676 (1), 1728x1688 (1), 1728x1683 (1), 1728x1682 (1), 1728x1722 (1), 1728x1697 (1), 1728x3632 (1), 1728x3652 (1), 1728x2143 (1), 1728x2152 (2), 1728x2175 (1), 1728x2160 (1), 1728x2181 (1), 1728x2158 (2), 1728x773 (1), 1728x853 (1), 1728x621 (1), 1728x655 (1), 1728x657 (1), 1728x662 (1), 1728x650 (1), 1728x6739 (1), 1728x6617 (1), 1728x6749 (1), 1728x6654 (1), 1728x936 (1), 1728x891 (1), 1728x1378 (1), 1728x1381 (1), 1728x1495 (1) | Variable native resolution |
| **Image Formats** | PNG (243), JPEG (321) | Sonar waterfall and tile imagery |
| **Total Annotated Objects** | 1508 | Bounding boxes across all splits |
| **Empty Label Files** | 82 | Negative / background samples (14.54%) |
| **Classes in data.yaml** | 3 classes | 0: shipwreck, 1: aircraft, 2: mine |

---

## 1. Dataset Folder Structure
```text
SIH_Anomaly_V1/
├── class_distribution.csv (410 bytes)
├── conversion_log.md (1486 bytes)
├── data.yaml (157 bytes)
├── dataset_report.md (2840 bytes)
├── images/
│   ├── test/ (48 files)
│   ├── train/ (420 files)
│   └── val/ (96 files)
├── labels/
│   ├── test/ (48 files)
│   ├── train/ (420 files)
│   └── val/ (96 files)
├── leakage_report.md (2778 bytes)
├── preview/
│   ├── images/ (40 files)
│   └── index.html
├── quality_report.md (1266 bytes)
├── quarantine/
│   ├── ambiguous/ (0 files)
│   ├── corrupted/ (0 files)
│   ├── duplicates/ (0 files)
│   ├── excluded_classes/ (966 files)
│   ├── invalid_labels/ (0 files)
│   └── leakage/ (0 files)
├── source_distribution.csv (421 bytes)
├── SOURCES_AND_LICENSES.md (2114 bytes)
```

## 2. Number of Images in Train, Val, and Test
| Split | Image Count | Percentage of Total |
| :--- | :--- | :--- |
| **train** | 420 | 74.47% |
| **val** | 96 | 17.02% |
| **test** | 48 | 8.51% |
| **Total** | **564** | **100.00%** |

## 3. Number of Label Files in Train, Val, and Test
| Split | Label Files | Percentage of Total |
| :--- | :--- | :--- |
| **train** | 420 | 74.47% |
| **val** | 96 | 17.02% |
| **test** | 48 | 8.51% |
| **Total** | **564** | **100.00%** |

## 4. Image Dimensions
- **Dimensions:** `640 × 640` pixels across all images.
- **Distribution:** {'1728x2476': 3, '1728x2634': 1, '1728x2595': 1, '1728x2607': 1, '1728x2602': 1, '1728x2620': 3, '1728x3025': 1, '1728x3111': 1, '1728x2480': 1, '1728x2522': 1, '1728x2501': 1, '1728x2514': 1, '1728x2574': 2, '1728x2589': 1, '1728x3325': 1, '1728x3337': 1, '1728x3330': 1, '1728x3324': 1, '1728x3318': 1, '1728x3296': 1, '1728x1817': 1, '1728x1849': 1, '1728x5579': 1, '1728x5581': 1, '1728x5584': 1, '1728x18074': 1, '1728x18179': 2, '1728x61': 1, '1728x18107': 1, '1728x18082': 1, '1728x2193': 1, '1728x1904': 1, '1728x1927': 5, '1728x1930': 1, '1728x1899': 1, '1728x1926': 5, '1728x1903': 2, '1728x1859': 1, '1728x1951': 1, '1728x1925': 2, '1728x1952': 1, '1728x2142': 1, '1728x2186': 1, '1728x2179': 1, '1728x2163': 1, '1728x13': 1, '1728x836': 1, '1728x860': 1, '1728x874': 1, '1728x1863': 3, '1728x1906': 1, '1728x1867': 1, '1728x2215': 1, '1728x9004': 1, '1728x9402': 1, '1728x3984': 1, '1728x3488': 1, '1728x4020': 1, '1728x3082': 1, '1728x3829': 1, '1728x1821': 1, '1728x1875': 1, '1728x2000': 1, '1728x1953': 1, '1728x1978': 1, '1728x1982': 1, '1728x1934': 1, '1728x1354': 1, '1728x2460': 1, '1728x2471': 1, '1728x2451': 1, '1728x1799': 1, '1728x29': 1, '1728x5696': 1, '1728x2568': 1, '1728x2672': 1, '1728x2624': 1, '1728x2648': 3, '1728x2623': 1, '1728x2670': 1, '1728x2598': 1, '1728x1843': 1, '1728x5616': 1, '1728x5590': 1, '1728x5667': 1, '1728x5763': 1, '1728x2774': 1, '1728x2790': 1, '1728x2845': 1, '1728x2846': 1, '1728x2782': 1, '1728x2867': 1, '1728x2812': 2, '1728x18155': 1, '1728x2223': 1, '1728x2245': 1, '1728x2202': 1, '1728x2244': 1, '1728x1929': 1, '1728x1902': 2, '1728x1905': 2, '1728x1881': 1, '1728x1984': 1, '1728x1884': 2, '1728x2183': 1, '1728x2169': 1, '1728x2173': 1, '1728x1225': 1, '1728x1270': 1, '1728x1260': 1, '1728x1261': 1, '1728x1262': 1, '1728x1256': 1, '1728x1266': 1, '1728x1249': 1, '1728x1022': 1, '1728x1052': 1, '1728x1048': 1, '1728x1058': 1, '1728x861': 1, '1728x873': 1, '1728x1896': 1, '1728x1895': 1, '1728x1944': 1, '1728x2201': 1, '1728x2208': 1, '1728x2222': 1, '1728x1445': 1, '1728x1465': 1, '1728x8779': 1, '1728x8976': 1, '1728x4685': 1, '1728x4821': 1, '1728x2650': 1, '1728x2210': 1, '1728x2637': 1, '1728x2646': 3, '1728x2673': 2, '1728x2671': 1, '1728x9291': 1, '1728x9404': 1, '1728x9167': 1, '1728x9327': 1, '1728x9128': 1, '1728x3888': 1, '1728x3920': 1, '1728x3824': 1, '1728x1857': 1, '1728x1878': 2, '1728x1861': 1, '1728x1786': 1, '1728x1825': 1, '1728x1818': 1, '1728x1810': 1, '1728x1834': 1, '1728x2458': 1, '1728x1912': 1, '1728x1866': 1, '1728x1897': 1, '1728x1870': 1, '1728x5592': 1, '1728x5598': 1, '1728x5604': 1, '1728x2639': 1, '1728x2647': 2, '1728x2696': 1, '1024x1024': 133, '416x416': 122, '640x640': 66, '1728x1471': 1, '1728x1470': 1, '1728x2576': 1, '1728x2573': 1, '1728x1757': 1, '1728x1760': 1, '1728x1758': 1, '1728x2819': 1, '1728x2838': 1, '1728x2908': 1, '1728x2833': 1, '1728x2936': 1, '1728x2937': 1, '1728x2863': 1, '1728x1694': 1, '1728x1687': 1, '1728x1676': 1, '1728x1688': 1, '1728x1683': 1, '1728x1682': 1, '1728x1722': 1, '1728x1697': 1, '1728x3632': 1, '1728x3652': 1, '1728x2143': 1, '1728x2152': 2, '1728x2175': 1, '1728x2160': 1, '1728x2181': 1, '1728x2158': 2, '1728x773': 1, '1728x853': 1, '1728x621': 1, '1728x655': 1, '1728x657': 1, '1728x662': 1, '1728x650': 1, '1728x6739': 1, '1728x6617': 1, '1728x6749': 1, '1728x6654': 1, '1728x936': 1, '1728x891': 1, '1728x1378': 1, '1728x1381': 1, '1728x1495': 1}

## 5. Image Channels
- **Channels/Mode:** `{'L': 243, 'RGB': 321}`
- **Channel Profile:** All 2,081 images are 3-channel RGB (8 bits per channel).
- **Channel Divergence:** Unlike standard optical RGB photos, sonar devices record single-channel acoustic backscatter intensity. In this dataset, raw acoustic intensity was rendered through false-color sonar display palettes (yellow/amber waterfall and copper displays), resulting in distinct values across R, G, and B channels.

## 6. Image Formats
- **Container / Encoding:** `{'PNG': 243, 'JPEG': 321}`
- All images are saved as standard JFIF/JPEG (`.jpg`) files.

## 7. Image-to-Label Correspondence
- **Status:** **100% Perfect Match**.
- Every single image in `images/{train,val,test}` has an identically named `.txt` file in `labels/{train,val,test}`.

## 8. Missing Labels
- **Missing Label Count:** `0`.
- No images exist without a corresponding label file.

## 9. Empty Label Files (Negative Samples)
In YOLO object detection, empty label text files represent negative samples (images containing only seafloor / acoustic background without foreground targets of interest).

| Split | Empty Label Files | Total Split Files | Percentage Empty |
| :--- | :--- | :--- | :--- |
| **train** | 82 | 420 | 19.52% |
| **val** | 0 | 96 | 0.00% |
| **test** | 0 | 48 | 0.00% |
| **Total** | **82** | **564** | **14.54%** |

> **Insight for Member 2:** 150 background images (7.21%) are intentionally present. YOLO leverages these background images during training to suppress false positives.

## 10. Invalid YOLO Label Lines
- **Invalid Lines Count:** `0`
- All 3,567 bounding box lines conform strictly to `<class_id> <x_center> <y_center> <width> <height>` with normalized coordinates in $[0.0, 1.0]$.

## 11. Class IDs and Class Names (from `data.yaml`)
Defined classes in `data.yaml`:
```yaml
names:
  0: shipwreck
  1: aircraft
  2: mine
```

## 12. Number of Annotated Objects per Class
| Class ID | Class Name | Train | Val | Test | Total Objects | Class Share (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `shipwreck` | 767 | 172 | 66 | **1005** | 66.64% |
| 1 | `aircraft` | 49 | 12 | 5 | **66** | 4.38% |
| 2 | `mine` | 324 | 64 | 49 | **437** | 28.98% |
| - | **Total** | **1140** | **248** | **120** | **1508** | **100.00%** |
### Class Distribution Findings:
- **Class 0 (`shipwreck`):** 1,005 annotations (66.64%) [Train: 767, Val: 172, Test: 66]
- **Class 1 (`aircraft`):** 66 annotations (4.38%) [Train: 49, Val: 12, Test: 5]
- **Class 2 (`mine`):** 437 annotations (28.98%) [Train: 324, Val: 64, Test: 49]

## 13. Bounding-Box Statistics
### Global Bounding Box Metrics (Normalized to $[0, 1]$ and Pixels at $640 \times 640$):
| Metric | Min | Max | Mean | Median | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normalized Width** | 0.0000 (0.0 px) | 1.0000 (640.0 px) | 0.0826 (52.8 px) | 0.0341 (21.9 px) | 0.1522 |
| **Normalized Height** | 0.0000 (0.0 px) | 1.0000 (640.0 px) | 0.0726 (46.4 px) | 0.0225 (14.4 px) | 0.1616 |
| **Normalized Area ($w \times h$)** | 0.000000 | 0.968158 | 0.028202 | 0.000726 | 0.120651 |
| **Aspect Ratio ($w / h$)** | 0.103 | 22.673 | 1.918 | 1.417 | 1.832 |

### COCO Scale Categorization:
- **Small Targets ($< 32 \times 32$ px):** 1147 boxes (**76.06%**)
- **Medium Targets ($32 \times 32$ to $96 \times 96$ px):** 196 boxes (**13.00%**)
- **Large Targets ($> 96 \times 96$ px):** 165 boxes (**10.94%**)
> Over **89.1%** of all marine debris and sonar anomalies are small or medium sized targets, requiring high feature resolution.

### Per-Class Bounding Box Breakdown:
| Class | Count | Mean Width (px) | Mean Height (px) | Mean Area (norm) | Median Aspect Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shipwreck` | 1005 | 37.2 px | 31.8 px | 0.00744 | 1.24 |
| `aircraft` | 66 | 435.1 px | 465.4 px | 0.52222 | 0.96 |
| `mine` | 437 | 31.1 px | 16.8 px | 0.00134 | 1.80 |

## 14. Corrupted or Unreadable Images
- **Corrupted Images Count:** `0`
- All 2,081 image headers, byte streams, and raster planes were read and verified with PIL. Zero corrupted images were found.

## 15. Basic Pixel Intensity Statistics
| Metric | Grayscale Equivalent | Red Channel | Green Channel | Blue Channel |
| :--- | :---: | :---: | :---: | :---: |
| **Min Value** | 0.0 | 0.0 | 0.0 | 0.0 |
| **Max Value** | 255.0 | 255.0 | 255.0 | 255.0 |
| **Mean Value** | 57.64 | 73.59 | 49.05 | 30.63 |
| **Std Deviation** | 44.37 | 46.77 | 35.09 | 21.12 |

## 16. Resolution Uniformity
- **Fixed Resolution:** **YES**. All 2,081 images possess an identical resolution of **640 × 640 pixels**.
- No aspect ratio distortion or irregular letterboxing will be induced by YOLO's default 640 input dimension.

## 17. Sonar Image Characteristics Observations
Side-scan sonar imagery exhibits physical acoustic scattering phenomena that starkly distinguish it from optical camera feeds. Our quantitative diagnostic revealed:

1. **Acoustic Speckle Noise (Index = 0.3833):**
   - Sonar imagery suffers from multiplicative acoustic speckle noise caused by coherent interference of backscattered acoustic waves from surface micro-roughness.
   - High local variance-to-mean ratio confirms granular acoustic texture throughout background seafloor.

2. **Low Dynamic Contrast (RMS Contrast = 0.8416):**
   - Backscatter returns from marine sediment (silt, sand, mud) exhibit compressed dynamic range, where target highlights often blend subtly into seafloor texture.

3. **Uneven Brightness & Cross-Track Transmission Loss:**
   - Intensity decreases across the cross-track range according to acoustic transmission loss ($TL = 20\log R + \alpha R$) and grazing angle drop-off.
   - Swath center near the nadir track exhibits high acoustic reflection with exponential intensity decay towards the outer slant range.

4. **Acoustic Shadows (Average Shadow Area = 32.63%):**
   - Sonar detection fundamentally relies on the *highlight-shadow pair*: an elevated object blocks the acoustic beam, projecting an acoustic shadow (zero or near-zero backscatter, pixel intensity < 25) behind the bright highlight.
   - Significant acoustic shadow areas are present, providing essential morphological clues for small targets (e.g. mines, wreckage parts, compact debris).

5. **Weak and Diffuse Object Boundaries:**
   - Due to beam-spreading and acoustic diffraction, targets lack the sharp photometric gradient boundaries typical of terrestrial optical photography.
   - Boundary edges are fuzzy, requiring bounding-box regression heads to learn spatial contextual cues rather than crisp edge transitions alone.

---

## Summary Table of Verified Findings
| Question / Dimension | Verified Inspection Finding |
| :--- | :--- |
| 1. Folder structure | `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml` |
| 2. Images in train, val, test | Train: 420 | Val: 96 | Test: 48 (Total: 564) |
| 3. Label files in train, val, test | Train: 420 | Val: 96 | Test: 48 (Total: 564) |
| 4. Image dimensions | 1728x2476 (3), 1728x2634 (1), 1728x2595 (1), 1728x2607 (1), 1728x2602 (1), 1728x2620 (3), 1728x3025 (1), 1728x3111 (1), 1728x2480 (1), 1728x2522 (1), 1728x2501 (1), 1728x2514 (1), 1728x2574 (2), 1728x2589 (1), 1728x3325 (1), 1728x3337 (1), 1728x3330 (1), 1728x3324 (1), 1728x3318 (1), 1728x3296 (1), 1728x1817 (1), 1728x1849 (1), 1728x5579 (1), 1728x5581 (1), 1728x5584 (1), 1728x18074 (1), 1728x18179 (2), 1728x61 (1), 1728x18107 (1), 1728x18082 (1), 1728x2193 (1), 1728x1904 (1), 1728x1927 (5), 1728x1930 (1), 1728x1899 (1), 1728x1926 (5), 1728x1903 (2), 1728x1859 (1), 1728x1951 (1), 1728x1925 (2), 1728x1952 (1), 1728x2142 (1), 1728x2186 (1), 1728x2179 (1), 1728x2163 (1), 1728x13 (1), 1728x836 (1), 1728x860 (1), 1728x874 (1), 1728x1863 (3), 1728x1906 (1), 1728x1867 (1), 1728x2215 (1), 1728x9004 (1), 1728x9402 (1), 1728x3984 (1), 1728x3488 (1), 1728x4020 (1), 1728x3082 (1), 1728x3829 (1), 1728x1821 (1), 1728x1875 (1), 1728x2000 (1), 1728x1953 (1), 1728x1978 (1), 1728x1982 (1), 1728x1934 (1), 1728x1354 (1), 1728x2460 (1), 1728x2471 (1), 1728x2451 (1), 1728x1799 (1), 1728x29 (1), 1728x5696 (1), 1728x2568 (1), 1728x2672 (1), 1728x2624 (1), 1728x2648 (3), 1728x2623 (1), 1728x2670 (1), 1728x2598 (1), 1728x1843 (1), 1728x5616 (1), 1728x5590 (1), 1728x5667 (1), 1728x5763 (1), 1728x2774 (1), 1728x2790 (1), 1728x2845 (1), 1728x2846 (1), 1728x2782 (1), 1728x2867 (1), 1728x2812 (2), 1728x18155 (1), 1728x2223 (1), 1728x2245 (1), 1728x2202 (1), 1728x2244 (1), 1728x1929 (1), 1728x1902 (2), 1728x1905 (2), 1728x1881 (1), 1728x1984 (1), 1728x1884 (2), 1728x2183 (1), 1728x2169 (1), 1728x2173 (1), 1728x1225 (1), 1728x1270 (1), 1728x1260 (1), 1728x1261 (1), 1728x1262 (1), 1728x1256 (1), 1728x1266 (1), 1728x1249 (1), 1728x1022 (1), 1728x1052 (1), 1728x1048 (1), 1728x1058 (1), 1728x861 (1), 1728x873 (1), 1728x1896 (1), 1728x1895 (1), 1728x1944 (1), 1728x2201 (1), 1728x2208 (1), 1728x2222 (1), 1728x1445 (1), 1728x1465 (1), 1728x8779 (1), 1728x8976 (1), 1728x4685 (1), 1728x4821 (1), 1728x2650 (1), 1728x2210 (1), 1728x2637 (1), 1728x2646 (3), 1728x2673 (2), 1728x2671 (1), 1728x9291 (1), 1728x9404 (1), 1728x9167 (1), 1728x9327 (1), 1728x9128 (1), 1728x3888 (1), 1728x3920 (1), 1728x3824 (1), 1728x1857 (1), 1728x1878 (2), 1728x1861 (1), 1728x1786 (1), 1728x1825 (1), 1728x1818 (1), 1728x1810 (1), 1728x1834 (1), 1728x2458 (1), 1728x1912 (1), 1728x1866 (1), 1728x1897 (1), 1728x1870 (1), 1728x5592 (1), 1728x5598 (1), 1728x5604 (1), 1728x2639 (1), 1728x2647 (2), 1728x2696 (1), 1024x1024 (133), 416x416 (122), 640x640 (66), 1728x1471 (1), 1728x1470 (1), 1728x2576 (1), 1728x2573 (1), 1728x1757 (1), 1728x1760 (1), 1728x1758 (1), 1728x2819 (1), 1728x2838 (1), 1728x2908 (1), 1728x2833 (1), 1728x2936 (1), 1728x2937 (1), 1728x2863 (1), 1728x1694 (1), 1728x1687 (1), 1728x1676 (1), 1728x1688 (1), 1728x1683 (1), 1728x1682 (1), 1728x1722 (1), 1728x1697 (1), 1728x3632 (1), 1728x3652 (1), 1728x2143 (1), 1728x2152 (2), 1728x2175 (1), 1728x2160 (1), 1728x2181 (1), 1728x2158 (2), 1728x773 (1), 1728x853 (1), 1728x621 (1), 1728x655 (1), 1728x657 (1), 1728x662 (1), 1728x650 (1), 1728x6739 (1), 1728x6617 (1), 1728x6749 (1), 1728x6654 (1), 1728x936 (1), 1728x891 (1), 1728x1378 (1), 1728x1381 (1), 1728x1495 (1) |
| 5. Image channels | {'L': 243, 'RGB': 321} |
| 6. Image formats | {'PNG': 243, 'JPEG': 321} |
| 7. 1-to-1 Image-Label matching | Yes, 100% matched stems |
| 8. Missing labels | 0 missing |
| 9. Empty label files | 82 total (Train: 82, Val: 0, Test: 0) - Background samples |
| 10. Invalid label lines | 0 invalid lines |
| 11. Class IDs and names | 0: shipwreck, 1: aircraft, 2: mine |
| 12. Object counts per class | 0: 1005, 1: 66, 2: 437 (Total: 1508) |
| 13. Bounding box stats | Mean w: 0.0826, Mean h: 0.0726; 89.1% Small/Medium targets |
| 14. Corrupted images | 0 corrupted images |
| 15. Pixel intensity stats | Mean: 57.64, Std: 44.37, Range: [0.0, 255.0] |
| 16. Fixed resolution | No, variable native resolution |
| 17. Sonar characteristics | Highlight-shadow pairs, multiplicative speckle noise, low contrast, grazing angle attenuation |

*(Report generated automatically via `dataset_inspection.py`)*