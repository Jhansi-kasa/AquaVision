r"""
final_preprocessing_pipeline.py
===============================
Final AI-Ready Sonar Image Preprocessing Pipeline Module
Member 2: Computer Vision & Sonar Processing
Project: AI-Powered Automated Underwater Marine Debris and Anomaly Detection System
         using Side-Scan Sonar Imagery
Dataset: SIH_Anomaly_V1 (C:/Users/dell/Downloads/SIH_Anomaly_V1/SIH_Anomaly_V1)

Scientific Architecture & Order of Operations:
----------------------------------------------
Based on empirical evaluations and quantitative findings in P1, P2, P3, and P4:

1. Stage 1: Cross-Track Swath Illumination Normalization (P3 Config B)
   - Eliminates macro geometric spreading loss (1/R^2) and range attenuation fall-off.
   - Operating first prevents downstream algorithms from suffering from near-range
     saturation or far-range signal starvation.
   - Parameters: axis='horizontal', method='median', smooth_sigma=35.0,
                 kernel_size=71, gain=[0.5, 2.5], min_intensity_floor=8.0.
   - Evaluated in CIELAB L* (Luminance) to strictly preserve false-color colormaps.

2. Stage 2: Edge-Preserving Denoising (P1 Bilateral Filter)
   - Suppresses high-frequency coherent acoustic speckle noise in homogeneous sediments
     while strictly preserving sharp highlight-to-shadow acoustic boundaries.
   - Selected over Gaussian (which erodes target boundaries) and Median (which erodes corners).
   - Parameters: d=7, sigma_color=50.0, sigma_space=50.0 (EPI=0.4998, PSNR=34.4 dB).

3. Stage 3: Robust Dynamic Range Percentile Normalization (P1 Robust Percentile)
   - Calibrates pixel intensity into the full [0, 255] uint8 radiometric range.
   - Uses 1%–99% percentile clipping to ignore isolated single-pixel speckle spikes
     that would otherwise compress the dynamic range in linear Min-Max.
   - Includes division-by-zero safeguard for constant / near-constant images.

4. Stage 4: Contrast-Limited Adaptive Histogram Equalization (P1 CLAHE)
   - Amplifies subtle local acoustic backscatter gradients and shadow penumbras
     in 80x80 contextual tiles without blowing out background noise.
   - Parameters: clip_limit=2.0, tile_grid_size=(8, 8), CIELAB L* Luminance domain.

Exclusion of Other Methods:
---------------------------
- Lee & Frost Speckle Filters (P2): Lee filter (5x5, Cu=0.25) is an effective speckle filter,
  but chaining both Lee and Bilateral causes redundant double-smoothing and destroys micro-textures.
  Bilateral filter already provides superior spatial/range smoothing with higher PSNR (34.4 dB vs 30.3 dB).
- Morphological Top-Hat / Black-Hat (P4): Excluded from primary image transformation because
  it strips essential background seafloor context and activates natural sand ripple crests/troughs
  (12.7% false clutter for 9x9), which risks elevated false positives in downstream YOLO detection.
  P4 is reserved strictly as an auxiliary multi-spectral research channel.

Integrity Guarantees:
---------------------
- Strictly non-destructive (source dataset is never modified).
- Output dimensions strictly preserved (native HxW matching source).
- 3-channel BGR format strictly preserved in uint8 [0..255] dynamic range.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


def get_default_config() -> Dict[str, Any]:
    """
    Return the recommended configuration for the final AI-ready preprocessing pipeline.
    """
    return {
        "pipeline_name": "Final_AI_Ready_Sonar_Pipeline_v1.0",
        "author": "Member 2 (Computer Vision & Sonar Processing)",
        "project": "SIH_Marine_Debris",
        "order": [
            "swath_normalization",
            "bilateral_denoising",
            "robust_normalization",
            "clahe",
        ],
        "swath_normalization": {
            "axis": "horizontal",
            "method": "median",
            "smooth_method": "gaussian",
            "smooth_sigma": 35.0,
            "smooth_kernel_size": 71,
            "min_gain": 0.5,
            "max_gain": 2.5,
            "target_level": "mean",
            "min_intensity_floor": 8.0,
            "enabled": True,
        },
        "bilateral_denoising": {
            "d": 7,
            "sigma_color": 50.0,
            "sigma_space": 50.0,
            "enabled": True,
        },
        "robust_normalization": {
            "p_low": 1.0,
            "p_high": 99.0,
            "min_out": 0.0,
            "max_out": 255.0,
            "eps": 1e-5,
            "enabled": True,
        },
        "clahe": {
            "clip_limit": 2.0,
            "tile_grid_size": [8, 8],
            "color_space": "LAB",
            "enabled": True,
        },
    }


def validate_image(image: np.ndarray) -> np.ndarray:
    """Validate input image dimensions, data type, and structure."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Input image is empty.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
    return image


def apply_swath_normalization(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Stage 1: Empirical Cross-Track Swath Illumination Normalization.
    Equalizes horizontal range fall-off in CIELAB L* channel.
    """
    if not params.get("enabled", True):
        return image

    is_color = image.ndim == 3 and image.shape[2] == 3
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
    else:
        L = image.astype(np.float32)

    # Constant / zero image safeguard
    if np.std(L) < 1e-3:
        return image.copy()

    axis = params.get("axis", "horizontal")
    method = params.get("method", "median")
    smooth_sigma = float(params.get("smooth_sigma", 35.0))
    smooth_ksize = int(params.get("smooth_kernel_size", 71))
    min_gain = float(params.get("min_gain", 0.5))
    max_gain = float(params.get("max_gain", 2.5))
    target_level = params.get("target_level", "mean")
    min_floor = float(params.get("min_intensity_floor", 8.0))

    if smooth_ksize % 2 == 0:
        smooth_ksize += 1

    # 1. Estimate 1D cross-track profile
    if axis == "horizontal":
        profile = np.median(L, axis=0) if method == "median" else np.mean(L, axis=0)
    else:
        profile = np.median(L, axis=1) if method == "median" else np.mean(L, axis=1)

    profile_len = len(profile)
    ksize = min(smooth_ksize, profile_len if profile_len % 2 == 1 else profile_len - 1)

    # 2. Smooth profile using 1D Gaussian kernel
    prof_2d = profile.reshape(1, -1)
    smoothed = cv2.GaussianBlur(
        prof_2d,
        ksize=(ksize, 1),
        sigmaX=smooth_sigma,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT,
    ).flatten()

    # 3. Compute gain profile with safeguards
    valid_pts = smoothed[smoothed > min_floor]
    if valid_pts.size > 0:
        target_val = float(np.mean(valid_pts)) if target_level == "mean" else float(np.median(valid_pts))
    else:
        target_val = float(np.mean(smoothed))

    denom = np.maximum(smoothed, min_floor)
    raw_gain = target_val / denom
    clamped_gain = np.clip(raw_gain, min_gain, max_gain)

    # Water column / deep shadow taper
    alpha = np.clip(smoothed / min_floor, 0.0, 1.0)
    final_gain = 1.0 + alpha * (clamped_gain - 1.0)

    # 4. Broadcast and apply gain
    if axis == "horizontal":
        gain_2d = final_gain.reshape(1, -1)
    else:
        gain_2d = final_gain.reshape(-1, 1)

    L_norm = np.clip(L * gain_2d, 0.0, 255.0).astype(np.uint8)

    if is_color:
        lab[:, :, 0] = L_norm
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return L_norm


def apply_bilateral_denoising(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Stage 2: Edge-Preserving Bilateral Denoising.
    Smooths acoustic sediment speckle while preserving highlight-shadow boundaries.
    """
    if not params.get("enabled", True):
        return image

    d = int(params.get("d", 7))
    sigma_color = float(params.get("sigma_color", 50.0))
    sigma_space = float(params.get("sigma_space", 50.0))

    return cv2.bilateralFilter(
        image,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


def apply_robust_normalization(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Stage 3: Robust Dynamic Range Percentile Normalization.
    Stretches radiometric baseline to [0, 255] uint8, clipping extreme 1% and 99% outliers.
    """
    if not params.get("enabled", True):
        return image

    p_low = float(params.get("p_low", 1.0))
    p_high = float(params.get("p_high", 99.0))
    min_out = float(params.get("min_out", 0.0))
    max_out = float(params.get("max_out", 255.0))
    eps = float(params.get("eps", 1e-5))

    img_float = image.astype(np.float64)
    v_low = float(np.percentile(img_float, p_low))
    v_high = float(np.percentile(img_float, p_high))
    spread = v_high - v_low

    if spread < eps:
        # Constant / near-constant safety fallback
        return image.copy()

    normalized = (img_float - v_low) / spread * (max_out - min_out) + min_out
    clipped = np.clip(normalized, min_out, max_out)
    return clipped.astype(np.uint8)


def apply_clahe_enhancement(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Stage 4: Contrast-Limited Adaptive Histogram Equalization.
    Enhances local acoustic target-to-shadow contrast in CIELAB L* channel.
    """
    if not params.get("enabled", True):
        return image

    clip_limit = float(params.get("clip_limit", 2.0))
    grid_size = params.get("tile_grid_size", [8, 8])
    tile_grid = (int(grid_size[0]), int(grid_size[1]))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    if image.ndim == 2:
        return clahe.apply(image)

    # 3-channel image: enhance Luminance (L*) exclusively to protect chromaticity
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_sonar_image(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    return_intermediates: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Execute the full 4-stage final AI-ready preprocessing pipeline on a side-scan sonar image.

    Parameters:
    -----------
    image : np.ndarray
        Raw input image (640x640x3 or 640x640 uint8).
    config : Optional[Dict[str, Any]]
        Configuration overrides. If None, default recommended settings are used.
    return_intermediates : bool
        If True, returns tuple of (final_image, intermediates_dict).

    Returns:
    --------
    Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]
        Processed AI-ready image, matching input dimensions (640x640) and format (uint8).
    """
    validate_image(image)
    cfg = get_default_config()
    if config is not None:
        for k, v in config.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    intermediates: Dict[str, np.ndarray] = {"raw": image.copy()}

    # Stage 1: Swath Illumination Normalization
    stage1_out = apply_swath_normalization(image, cfg["swath_normalization"])
    intermediates["stage1_swath"] = stage1_out

    # Stage 2: Bilateral Denoising
    stage2_out = apply_bilateral_denoising(stage1_out, cfg["bilateral_denoising"])
    intermediates["stage2_denoised"] = stage2_out

    # Stage 3: Robust Percentile Normalization
    stage3_out = apply_robust_normalization(stage2_out, cfg["robust_normalization"])
    intermediates["stage3_normalized"] = stage3_out

    # Stage 4: CLAHE Contrast Enhancement
    stage4_out = apply_clahe_enhancement(stage3_out, cfg["clahe"])
    intermediates["stage4_clahe"] = stage4_out

    final_image = stage4_out

    if return_intermediates:
        return final_image, intermediates
    return final_image


if __name__ == "__main__":
    print("final_preprocessing_pipeline.py: Running module self-test...")
    # Synthetic test image
    test_img = np.random.randint(20, 200, (640, 640, 3), dtype=np.uint8)
    out, inters = process_sonar_image(test_img, return_intermediates=True)
    assert out.shape == (640, 640, 3), f"Shape mismatch: {out.shape}"
    assert out.dtype == np.uint8, f"Dtype mismatch: {out.dtype}"
    assert len(inters) == 5, f"Intermediates count mismatch: {len(inters)}"
    print("[PASS] Self-test verified successfully: output is (640, 640, 3) uint8.")
