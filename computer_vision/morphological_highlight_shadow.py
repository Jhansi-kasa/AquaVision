"""
Side-Scan Sonar Morphological Highlight–Shadow Separation Module (Task P4)
===========================================================================
Project: AI-Powered Automated Underwater Marine Debris and Anomaly Detection
Subsystem: Member 2 (Computer Vision & Sonar Processing)
Active Dataset: SIH_Anomaly_V1 (3 Classes: shipwreck, aircraft, mine)

Purpose & Physical Context:
---------------------------
In side-scan sonar (SSS) imagery, objects resting on the seabed create a dual
acoustic response known as the "Highlight–Shadow Pair":
1. Acoustic Highlight: Incident acoustic wavefront strikes the proud structure,
   generating strong specular or diffuse backscatter (high pixel intensity).
2. Acoustic Shadow: Immediately behind the object (away from the transducer),
   acoustic rays are obstructed, creating a zero-backscatter void (dark pixels).

However, natural seafloor morphology (sand megaripples, rocky outcrops, seabed
depressions, dredge marks) also generates alternating bright ridges and dark
shadow troughs that can mimic man-made debris and cause false alarms.

Mathematical Morphology:
------------------------
This module implements mathematical morphology (Matheron & Serra, 1982) to
decompose side-scan sonar imagery into separate highlight and shadow components:

1. White Top-Hat (WTH):
   WTH(I) = I - (I ∘ S) = I - dilate(erode(I, S), S)
   Morphological opening (I ∘ S) eliminates bright structures smaller than
   structuring element S. Subtracting the opened image from the original isolates
   local acoustic highlights.

2. Black-Hat / Bottom-Hat (BTH):
   BTH(I) = (I ● S) - I = erode(dilate(I, S), S) - I
   Morphological closing (I ● S) bridges dark troughs smaller than structuring
   element S. Subtracting the original from the closed image isolates localized
   acoustic shadow voids.

Scientific & Algorithmic Boundary:
----------------------------------
- Top-Hat and Black-Hat filtering alone do NOT perform automated semantic
  segmentation or object detection; they provide radiometric feature maps.
- Enhanced highlight/shadow responses are not guaranteed to be marine debris;
  natural seabed clutter also produces responses.
- Evaluated independently from earlier preprocessing stages directly on raw
  sonar imagery.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


def get_structuring_element(
    kernel_size: int = 9,
    kernel_shape: str = "ellipse",
) -> np.ndarray:
    """
    Generate an OpenCV structuring element with guaranteed odd dimensions.

    Parameters
    ----------
    kernel_size : int
        Size of the structuring element (e.g., 5, 9, 15). If even, incremented by 1.
    kernel_shape : str
        Geometric shape: 'ellipse' (isotropic disk), 'rect' (box), 'cross' (plus).

    Returns
    -------
    np.ndarray
        2D structuring element array.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    shape_map = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect": cv2.MORPH_RECT,
        "rectangle": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
    }
    cv_shape = shape_map.get(kernel_shape.lower(), cv2.MORPH_ELLIPSE)
    return cv2.getStructuringElement(cv_shape, (kernel_size, kernel_size))


def extract_luminance(image: np.ndarray) -> np.ndarray:
    """
    Extract single-channel 8-bit luminance from grayscale or BGR images.
    For 3-channel images, uses perceived luminance (Y from YCrCb / L from LAB).
    """
    if image.ndim == 2:
        return image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        # Convert BGR to Grayscale via standard Rec.601 luma
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def apply_white_tophat(
    image: np.ndarray,
    kernel_size: int = 9,
    kernel_shape: str = "ellipse",
) -> np.ndarray:
    """
    Apply White Top-Hat transform to isolate bright acoustic highlights:
        WTH(I) = I - (I ∘ S)

    Parameters
    ----------
    image : np.ndarray
        Input sonar image (uint8, 2D grayscale or 3D BGR).
    kernel_size : int
        Structuring element size (e.g., 5, 9, 15).
    kernel_shape : str
        'ellipse', 'rect', or 'cross'.

    Returns
    -------
    np.ndarray
        2D uint8 array of shape (H, W) containing highlight response.
    """
    gray = extract_luminance(image)
    se = get_structuring_element(kernel_size, kernel_shape)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)


def apply_black_hat(
    image: np.ndarray,
    kernel_size: int = 9,
    kernel_shape: str = "ellipse",
) -> np.ndarray:
    """
    Apply Black-Hat transform to isolate dark acoustic shadow voids:
        BTH(I) = (I ● S) - I

    Parameters
    ----------
    image : np.ndarray
        Input sonar image (uint8, 2D grayscale or 3D BGR).
    kernel_size : int
        Structuring element size (e.g., 5, 9, 15).
    kernel_shape : str
        'ellipse', 'rect', or 'cross'.

    Returns
    -------
    np.ndarray
        2D uint8 array of shape (H, W) containing shadow response.
    """
    gray = extract_luminance(image)
    se = get_structuring_element(kernel_size, kernel_shape)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)


def compute_highlight_shadow_pair(
    image: np.ndarray,
    kernel_size: int = 9,
    kernel_shape: str = "ellipse",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simultaneously compute White Top-Hat and Black-Hat responses.

    Parameters
    ----------
    image : np.ndarray
        Input sonar image (uint8).
    kernel_size : int
        Structuring element size.
    kernel_shape : str
        'ellipse' or 'rect'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (white_tophat_highlights, black_hat_shadows)
    """
    gray = extract_luminance(image)
    se = get_structuring_element(kernel_size, kernel_shape)
    wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    return wth, bth


def create_combined_representation(
    raw_image: np.ndarray,
    wth: np.ndarray,
    bth: np.ndarray,
    mode: str = "rgb_fusion",
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Construct a combined highlight–shadow representation for visualization and analysis.

    Supported Modes:
    ----------------
    1. 'rgb_fusion' (Recommended for feature analysis):
       Maps responses into a 3-channel false-color image:
       - Channel R (Red): White Top-Hat (Acoustic Highlights)
       - Channel G (Green): Raw Grayscale Sonar (Seabed Context)
       - Channel B (Blue): Black-Hat (Acoustic Shadows)
       Visually highlights appear red/amber, shadows appear cyan/blue, and neutral
       seafloor appears green/gray.

    2. 'bipolar' (Differential response):
       Centered at mid-gray (128):
       Diff = 128 + 0.5 * (WTH - BTH)
       Highlights appear bright (>128), shadows appear dark (<128), and neutral
       ambient seabed appears 128.

    3. 'overlay' (Contextual annotation):
       Blends color-coded highlight mask (lime-green) and shadow mask (magenta)
       directly onto the original sonar image with transparency `alpha`.

    Parameters
    ----------
    raw_image : np.ndarray
        Original sonar image (uint8, BGR or Gray).
    wth : np.ndarray
        White Top-Hat response (uint8).
    bth : np.ndarray
        Black-Hat response (uint8).
    mode : str
        'rgb_fusion', 'bipolar', or 'overlay'.
    alpha : float
        Blending transparency for 'overlay' mode.

    Returns
    -------
    np.ndarray
        Combined representation image (H, W, 3) in BGR format for OpenCV display/saving.
    """
    gray = extract_luminance(raw_image)

    if mode == "rgb_fusion":
        # OpenCV uses BGR order:
        # B = Shadows (BTH), G = Raw context (Gray), R = Highlights (WTH)
        # Scale WTH and BTH slightly for balanced visualization
        wth_boost = np.clip(wth.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        bth_boost = np.clip(bth.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        fusion = cv2.merge([bth_boost, gray, wth_boost])
        return fusion

    elif mode == "bipolar":
        # Highlights positive, shadows negative, centered at 128
        diff = 128.0 + 0.5 * (wth.astype(np.float32) - bth.astype(np.float32))
        diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
        # Apply colormap: COOLWARM / JET / BGR
        diff_bgr = cv2.applyColorMap(diff_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)
        return diff_bgr

    elif mode == "overlay":
        base = raw_image.copy() if raw_image.ndim == 3 else cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        overlay = base.copy()

        # Threshold top 10% responses for overlay visualization
        th_wth = max(20, int(np.percentile(wth, 95)))
        th_bth = max(20, int(np.percentile(bth, 95)))

        # Highlight overlay in bright yellow/green (B=0, G=255, R=220)
        overlay[wth > th_wth] = [0, 255, 220]
        # Shadow overlay in deep magenta/blue (B=255, G=40, R=180)
        overlay[bth > th_bth] = [255, 40, 180]

        return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_morphological_metrics(
    raw_img: np.ndarray,
    wth: np.ndarray,
    bth: np.ndarray,
    noise_thresh: float = 20.0,
) -> Dict[str, float]:
    """
    Compute quantitative morphological response, contrast, and edge metrics.

    Parameters
    ----------
    raw_img : np.ndarray
        Raw sonar image (uint8).
    wth : np.ndarray
        White Top-Hat response.
    bth : np.ndarray
        Black-Hat response.
    noise_thresh : float
        Intensity threshold above which morphological response is considered
        a salient feature rather than ambient speckle noise.

    Returns
    -------
    Dict[str, float]
        Dictionary of morphological metrics.
    """
    gray = extract_luminance(raw_img).astype(np.float32)

    # Basic stats
    raw_mean = float(np.mean(gray))
    raw_std = float(np.std(gray))

    # Highlight metrics (WTH)
    wth_f = wth.astype(np.float32)
    wth_mean = float(np.mean(wth_f))
    wth_max = float(np.max(wth_f))
    wth_std = float(np.std(wth_f))
    wth_act_pct = float(np.mean(wth_f > noise_thresh) * 100.0)

    # Shadow metrics (BTH)
    bth_f = bth.astype(np.float32)
    bth_mean = float(np.mean(bth_f))
    bth_max = float(np.max(bth_f))
    bth_std = float(np.std(bth_f))
    bth_act_pct = float(np.mean(bth_f > noise_thresh) * 100.0)

    # Combined highlight-shadow energy
    hs_energy = float(np.mean(wth_f + bth_f))

    # Sobel Edge Energy
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    raw_edge_energy = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

    # WTH edge energy
    sobel_wx = cv2.Sobel(wth_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_wy = cv2.Sobel(wth_f, cv2.CV_32F, 0, 1, ksize=3)
    wth_edge_energy = float(np.mean(np.sqrt(sobel_wx**2 + sobel_wy**2)))

    return {
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "wth_mean": wth_mean,
        "wth_max": wth_max,
        "wth_std": wth_std,
        "wth_act_pct": wth_act_pct,
        "bth_mean": bth_mean,
        "bth_max": bth_max,
        "bth_std": bth_std,
        "bth_act_pct": bth_act_pct,
        "hs_energy": hs_energy,
        "raw_edge_energy": raw_edge_energy,
        "wth_edge_energy": wth_edge_energy,
    }


# Standard P4 parameter configurations for evaluation
CONFIG_P4_K5_ELLIPSE = {
    "name": "Config 1: 5x5 Ellipse",
    "kernel_size": 5,
    "kernel_shape": "ellipse",
    "description": "Small 5x5 disk kernel; sensitive to micro-targets, prone to high-frequency speckle.",
}

CONFIG_P4_K9_ELLIPSE = {
    "name": "Config 2: 9x9 Ellipse (Recommended)",
    "kernel_size": 9,
    "kernel_shape": "ellipse",
    "description": "Medium 9x9 disk kernel; optimal balance between target highlight/shadow extraction and seabed clutter suppression.",
}

CONFIG_P4_K15_ELLIPSE = {
    "name": "Config 3: 15x15 Ellipse",
    "kernel_size": 15,
    "kernel_shape": "ellipse",
    "description": "Large 15x15 disk kernel; extracts broad shadow extents, but heavily activates natural seabed sand ripples.",
}

CONFIG_P4_K9_RECT = {
    "name": "Config 4: 9x9 Rectangular",
    "kernel_size": 9,
    "kernel_shape": "rect",
    "description": "Medium 9x9 rectangular kernel; enhances rectilinear man-made geometry (crab pots), slight Cartesian corner bias.",
}
