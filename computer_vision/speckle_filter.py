"""
speckle_filter.py
Speckle Noise Filtering Module for Side-Scan Sonar Imagery
Member 2: Computer Vision & Sonar Processing

Implements adaptive speckle filters specifically designed for coherent acoustic imagery:
1. Lee Speckle Filter (5x5 default)
2. Frost Speckle Filter (5x5 default)

Mathematical Foundations:
- Assumes a multiplicative speckle noise model: I = R * u, where I is observed intensity,
  R is true acoustic reflectance, and u is speckle noise with mean 1 and variance cu^2.
- Lee Filter uses local statistics (mean & variance) to compute an adaptive weight:
  WL = max(0, (var - cu^2 * mean^2) / (var + eps)), output = mean + WL * (I - mean).
- Frost Filter uses an adaptive exponential distance-damping kernel:
  m(i, j) = exp(-K * Ci * d(i, j)), where Ci = std/mean and d is Euclidean distance.

Key Features:
- Fully vectorized NumPy/OpenCV implementations for high throughput.
- Supports 2D grayscale and 3-channel RGB/BGR sonar imagery.
- Preserves false-color sonar colormaps via LAB luminance processing.
- Preserves 640x640 resolution, dimensions, and uint8 format.
- Independent of YOLO; strictly non-destructive.
"""

from typing import Tuple, Union, Optional
import cv2
import numpy as np


def validate_sonar_image(image: np.ndarray) -> np.ndarray:
    """Validate image array type, shape, and properties."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Input image array is empty.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
    return image


def _lee_filter_2d(
    channel: np.ndarray,
    ksize: int = 5,
    cu: float = 0.25,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Core 2D Lee filter implementation using fast local box filters.
    """
    ch_float = channel.astype(np.float64)
    
    # Local mean and squared mean over window of size (ksize, ksize)
    mean = cv2.boxFilter(ch_float, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
    sqr_mean = cv2.boxFilter(ch_float**2, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
    
    # Local variance: Var(X) = E[X^2] - (E[X])^2
    var = np.maximum(sqr_mean - mean**2, 0.0)
    
    # Multiplicative noise variance estimate: sigma_u^2 * mean^2
    noise_var = (mean**2) * (cu**2)
    
    # Adaptive weighting factor WL in [0, 1]
    # WL -> 0 in homogeneous areas (smooth backscatter), WL -> 1 near sharp edges
    weight = np.clip((var - noise_var) / (var + eps), 0.0, 1.0)
    
    # Filtered response: R_hat = mean + WL * (I - mean)
    filtered = mean + weight * (ch_float - mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def _frost_filter_2d(
    channel: np.ndarray,
    ksize: int = 5,
    damping_factor: float = 1.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Core 2D Frost filter implementation using vectorized distance damping.
    """
    ch_float = channel.astype(np.float64)
    pad = ksize // 2
    h, w = ch_float.shape

    # Local mean and variance for coefficient of variation Ci = std / mean
    mean = cv2.boxFilter(ch_float, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
    sqr_mean = cv2.boxFilter(ch_float**2, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
    var = np.maximum(sqr_mean - mean**2, 0.0)
    ci = np.sqrt(var) / (mean + eps)

    # Precompute Euclidean distances from window center (shape: ksize, ksize)
    y, x = np.mgrid[-pad : pad + 1, -pad : pad + 1]
    dist_matrix = np.sqrt(x**2 + y**2)

    # Reflection border padding to prevent edge artifacts
    padded = cv2.copyMakeBorder(ch_float, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    numerator = np.zeros_like(ch_float)
    denominator = np.zeros_like(ch_float)

    # Vectorized loop over (ksize x ksize) spatial neighbors
    for dy in range(ksize):
        for dx in range(ksize):
            d = dist_matrix[dy, dx]
            neighbor = padded[dy : dy + h, dx : dx + w]
            # Exponential weight: m(i,j) = exp(-K * Ci * d)
            w_ij = np.exp(-damping_factor * ci * d)
            numerator += w_ij * neighbor
            denominator += w_ij

    filtered = numerator / (denominator + eps)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_lee_filter(
    image: np.ndarray,
    ksize: int = 5,
    cu: float = 0.25,
    color_handling: str = "luminance",
) -> np.ndarray:
    """
    Apply the Lee Speckle Filter to a sonar image.

    Parameters:
    -----------
    image : np.ndarray
        Input sonar image (H, W) or (H, W, 3) of dtype uint8.
    ksize : int, default 5
        Window/kernel size. Must be an odd positive integer (typically 5 for 5x5).
    cu : float, default 0.25
        Estimated noise coefficient of variation (sigma_u / mean_u).
        For single-look/multi-look sonar imagery, typically 0.20 - 0.35.
    color_handling : str, default 'luminance'
        For 3-channel color images:
        - 'luminance': converts to LAB color space, filters only L channel, merges back.
                       Strictly preserves the acoustic false-color colormap.
        - 'channel_wise': filters each channel independently.

    Returns:
    --------
    np.ndarray : Lee filtered image matching input shape and uint8 dtype.
    """
    validate_sonar_image(image)
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}")
    if cu <= 0:
        raise ValueError(f"Noise coefficient cu must be positive, got {cu}")

    # Grayscale image
    if image.ndim == 2:
        return _lee_filter_2d(image, ksize=ksize, cu=cu)

    # 3-Channel Color image
    mode = color_handling.lower().strip()
    if mode == "luminance":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_filtered = _lee_filter_2d(l, ksize=ksize, cu=cu)
        return cv2.cvtColor(cv2.merge([l_filtered, a, b]), cv2.COLOR_LAB2BGR)
    elif mode == "channel_wise":
        chans = cv2.split(image)
        filtered_chans = [_lee_filter_2d(c, ksize=ksize, cu=cu) for c in chans]
        return cv2.merge(filtered_chans)
    else:
        raise ValueError(f"Unknown color_handling '{color_handling}'. Choose 'luminance' or 'channel_wise'.")


def apply_frost_filter(
    image: np.ndarray,
    ksize: int = 5,
    damping_factor: float = 1.0,
    color_handling: str = "luminance",
) -> np.ndarray:
    """
    Apply the Frost Speckle Filter to a sonar image.

    Parameters:
    -----------
    image : np.ndarray
        Input sonar image (H, W) or (H, W, 3) of dtype uint8.
    ksize : int, default 5
        Window/kernel size. Must be an odd positive integer (typically 5 for 5x5).
    damping_factor : float, default 1.0
        Exponential decay factor K. Higher values preserve edges more aggressively;
        lower values produce smoother homogeneous regions.
    color_handling : str, default 'luminance'
        For 3-channel color images:
        - 'luminance': converts to LAB color space, filters only L channel, merges back.
                       Strictly preserves the acoustic false-color colormap.
        - 'channel_wise': filters each channel independently.

    Returns:
    --------
    np.ndarray : Frost filtered image matching input shape and uint8 dtype.
    """
    validate_sonar_image(image)
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}")
    if damping_factor <= 0:
        raise ValueError(f"Damping factor must be positive, got {damping_factor}")

    # Grayscale image
    if image.ndim == 2:
        return _frost_filter_2d(image, ksize=ksize, damping_factor=damping_factor)

    # 3-Channel Color image
    mode = color_handling.lower().strip()
    if mode == "luminance":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_filtered = _frost_filter_2d(l, ksize=ksize, damping_factor=damping_factor)
        return cv2.cvtColor(cv2.merge([l_filtered, a, b]), cv2.COLOR_LAB2BGR)
    elif mode == "channel_wise":
        chans = cv2.split(image)
        filtered_chans = [_frost_filter_2d(c, ksize=ksize, damping_factor=damping_factor) for c in chans]
        return cv2.merge(filtered_chans)
    else:
        raise ValueError(f"Unknown color_handling '{color_handling}'. Choose 'luminance' or 'channel_wise'.")


def apply_speckle_filter(
    image: np.ndarray,
    method: str = "lee",
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for sonar speckle filtering.
    """
    m = method.lower().strip()
    if m == "lee":
        return apply_lee_filter(image, **kwargs)
    elif m == "frost":
        return apply_frost_filter(image, **kwargs)
    else:
        raise ValueError(f"Unknown speckle filter '{method}'. Choose 'lee' or 'frost'.")
