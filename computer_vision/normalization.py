"""
normalization.py
Reusable Sonar Image Normalization Module
Member 2: Computer Vision & Sonar Processing

Pipeline context:
Raw Sonar Image -> Denoising -> NORMALIZATION -> Later: CLAHE -> AI-Ready Image

Implements robust, configurable normalization methods:
1. Standard Min-Max Normalization (0 - 255)
2. Robust Percentile-based Normalization (p_low to p_high, default 1% - 99%)
3. Z-Score (Standardization) Normalization scaled to 0 - 255

Features:
- Handles low-contrast and compressed dynamic range images.
- Safe handling for constant and near-constant images (avoids division by zero / NaN).
- Preserves exact 640x640 resolution, 3 channels, and uint8 format.
- Strictly non-destructive.
"""

from typing import Tuple, Union, Optional
import numpy as np


def validate_image_input(image: np.ndarray) -> np.ndarray:
    """Validate image array format and dimensions."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Input image array is empty.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D (grayscale) or 3D (color) image, got shape {image.shape}")
    return image


def normalize_min_max(
    image: np.ndarray,
    min_out: float = 0.0,
    max_out: float = 255.0,
    eps: float = 1e-5,
    channel_wise: bool = False,
    dtype: np.dtype = np.uint8,
) -> Tuple[np.ndarray, bool]:
    """
    Apply standard Linear Min-Max scaling to map pixel intensities to [min_out, max_out].
    
    Formula:
        I_norm = (I - min) / (max - min) * (max_out - min_out) + min_out

    Parameters:
    -----------
    image : np.ndarray
        Input image array.
    min_out : float, default 0.0
        Minimum target intensity.
    max_out : float, default 255.0
        Maximum target intensity.
    eps : float, default 1e-5
        Threshold to detect constant or near-constant images.
    channel_wise : bool, default False
        If True, normalizes each channel independently.
        If False, normalizes across all channels jointly (preserves false-color hue balance).
    dtype : np.dtype, default np.uint8
        Output array data type.

    Returns:
    --------
    normalized_image : np.ndarray
        Normalized image matching input shape and specified dtype.
    is_constant : bool
        True if the image was detected as constant/near-constant and handled safely.
    """
    validate_image_input(image)
    img_float = image.astype(np.float64)

    if not channel_wise or img_float.ndim == 2:
        img_min = float(np.min(img_float))
        img_max = float(np.max(img_float))
        dynamic_range = img_max - img_min

        # Safe handling for constant or near-constant images
        if dynamic_range < eps:
            # Return unchanged or mid-gray without dividing by zero
            return image.copy().astype(dtype), True

        norm = (img_float - img_min) / dynamic_range * (max_out - min_out) + min_out
        clipped = np.clip(norm, min_out, max_out)
        return clipped.astype(dtype), False
    else:
        # Per-channel normalization
        out = np.zeros_like(img_float)
        any_constant = False
        for c in range(img_float.shape[2]):
            ch = img_float[:, :, c]
            ch_min = float(np.min(ch))
            ch_max = float(np.max(ch))
            dr = ch_max - ch_min
            if dr < eps:
                out[:, :, c] = ch
                any_constant = True
            else:
                out[:, :, c] = (ch - ch_min) / dr * (max_out - min_out) + min_out
        clipped = np.clip(out, min_out, max_out)
        return clipped.astype(dtype), any_constant


def normalize_percentile(
    image: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    min_out: float = 0.0,
    max_out: float = 255.0,
    eps: float = 1e-5,
    channel_wise: bool = False,
    dtype: np.dtype = np.uint8,
) -> Tuple[np.ndarray, bool]:
    """
    Apply Robust Percentile-based Normalization with outlier clipping.
    
    In side-scan sonar, isolated speckle spikes or sensor artifacts can stretch
    the extrema without improving contrast in the actual backscatter field.
    Percentile clipping prevents single-pixel noise from compressing the dynamic range.

    Formula:
        I_low = percentile(I, p_low)
        I_high = percentile(I, p_high)
        I_norm = clip((I - I_low) / (I_high - I_low), 0, 1) * (max_out - min_out) + min_out

    Parameters:
    -----------
    image : np.ndarray
        Input image array.
    p_low : float, default 1.0
        Lower percentile threshold (e.g., 1.0% or 2.0%).
    p_high : float, default 99.0
        Upper percentile threshold (e.g., 99.0% or 98.0%).
    min_out : float, default 0.0
        Minimum target intensity.
    max_out : float, default 255.0
        Maximum target intensity.
    eps : float, default 1e-5
        Threshold to detect constant or near-constant images.
    channel_wise : bool, default False
        Normalize jointly or per-channel.
    dtype : np.dtype, default np.uint8
        Output array data type.

    Returns:
    --------
    normalized_image : np.ndarray
        Normalized image matching input shape and specified dtype.
    is_constant : bool
        True if the image was detected as constant/near-constant and handled safely.
    """
    validate_image_input(image)
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError(f"Invalid percentiles: p_low ({p_low}) must be < p_high ({p_high}) in [0, 100]")

    img_float = image.astype(np.float64)

    if not channel_wise or img_float.ndim == 2:
        val_low = float(np.percentile(img_float, p_low))
        val_high = float(np.percentile(img_float, p_high))
        spread = val_high - val_low

        # Safe handling for constant or near-constant images
        if spread < eps:
            return image.copy().astype(dtype), True

        norm = (img_float - val_low) / spread * (max_out - min_out) + min_out
        clipped = np.clip(norm, min_out, max_out)
        return clipped.astype(dtype), False
    else:
        out = np.zeros_like(img_float)
        any_constant = False
        for c in range(img_float.shape[2]):
            ch = img_float[:, :, c]
            v_l = float(np.percentile(ch, p_low))
            v_h = float(np.percentile(ch, p_high))
            sp = v_h - v_l
            if sp < eps:
                out[:, :, c] = ch
                any_constant = True
            else:
                out[:, :, c] = (ch - v_l) / sp * (max_out - min_out) + min_out
        clipped = np.clip(out, min_out, max_out)
        return clipped.astype(dtype), any_constant


def normalize_z_score(
    image: np.ndarray,
    n_std: float = 3.0,
    min_out: float = 0.0,
    max_out: float = 255.0,
    eps: float = 1e-5,
    dtype: np.dtype = np.uint8,
) -> Tuple[np.ndarray, bool]:
    """
    Apply Z-score standardization scaled to [min_out, max_out].
    Maps [mean - n_std*std, mean + n_std*std] to [min_out, max_out].
    """
    validate_image_input(image)
    img_float = image.astype(np.float64)
    mean_val = float(np.mean(img_float))
    std_val = float(np.std(img_float))

    if std_val < eps:
        return image.copy().astype(dtype), True

    val_low = mean_val - n_std * std_val
    val_high = mean_val + n_std * std_val
    spread = val_high - val_low

    norm = (img_float - val_low) / spread * (max_out - min_out) + min_out
    clipped = np.clip(norm, min_out, max_out)
    return clipped.astype(dtype), False


def normalize_sonar_image(
    image: np.ndarray,
    method: str = "percentile",
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for sonar image normalization.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image.
    method : str
        One of ['min_max', 'percentile', 'z_score'].
    kwargs : dict
        Additional parameters passed to specific normalization function.

    Returns:
    --------
    np.ndarray : Normalized image in uint8 [0, 255] range.
    """
    m = method.lower().strip()
    if m == "min_max":
        result, _ = normalize_min_max(image, **kwargs)
        return result
    elif m in ("percentile", "robust"):
        result, _ = normalize_percentile(image, **kwargs)
        return result
    elif m == "z_score":
        result, _ = normalize_z_score(image, **kwargs)
        return result
    else:
        raise ValueError(f"Unknown normalization method '{method}'. Choose 'percentile', 'min_max', or 'z_score'.")
