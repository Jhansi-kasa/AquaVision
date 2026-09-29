"""
clahe.py
Reusable Contrast Limited Adaptive Histogram Equalization (CLAHE) Module
Member 2: Computer Vision & Sonar Processing

Pipeline context:
Raw Sonar Image -> Denoising -> Normalization -> CLAHE -> AI-Ready Image

Implements robust, configurable CLAHE for side-scan sonar imagery:
- Safely processes 2D grayscale and 3-channel color imagery.
- Uses LAB Luminance-channel equalization for 3-channel sonar imagery to enhance
  local acoustic contrast while strictly preserving false-color palette integrity.
- Configurable clipLimit and tileGridSize.
- Preserves 640x640 resolution, 3 channels, and uint8 format.
- Strictly non-destructive.
"""

from typing import Tuple, Union, Optional
import cv2
import numpy as np


def validate_image_format(image: np.ndarray) -> np.ndarray:
    """Validate image array format and dimensions."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Input image array is empty.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
    return image


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = "LAB",
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters:
    -----------
    image : np.ndarray
        Input sonar image (H, W) or (H, W, 3) of dtype uint8.
    clip_limit : float, default 2.0
        Threshold for contrast limiting. Higher values increase local contrast
        but may amplify acoustic background noise.
    tile_grid_size : Tuple[int, int], default (8, 8)
        Size of grid for local histogram equalization (rows, columns).
        For 640x640:
          (8, 8) yields 80x80 pixel contextual tiles.
          (16, 16) yields 40x40 pixel contextual tiles.
    color_space : str, default 'LAB'
        Color space used when processing 3-channel imagery:
        - 'LAB': Applies CLAHE to L (Luminance) channel. Preserves false-color hue ratios.
        - 'YCrCb': Applies CLAHE to Y (Luma) channel.
        - 'HSV': Applies CLAHE to V (Value) channel.
        - 'channel_wise': Applies CLAHE independently to B, G, R channels (causes color shift).

    Returns:
    --------
    np.ndarray : CLAHE-enhanced image matching input shape and uint8 dtype.
    """
    validate_image_format(image)
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}")
    gw, gh = tile_grid_size
    if gw <= 0 or gh <= 0:
        raise ValueError(f"tile_grid_size dimensions must be positive, got {tile_grid_size}")

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(gw), int(gh)))

    # 1. Grayscale (2D) image handling
    if image.ndim == 2:
        return clahe.apply(image)

    # 2. 3-Channel image handling
    cs = color_space.upper().strip()

    if cs == "LAB":
        # Convert BGR to LAB, apply CLAHE to Luminance, and convert back
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        merged = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced

    elif cs == "YCRCB":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_clahe = clahe.apply(y)
        merged = cv2.merge([y_clahe, cr, cb])
        enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        return enhanced

    elif cs == "HSV":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_clahe = clahe.apply(v)
        merged = cv2.merge([h, s, v_clahe])
        enhanced = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
        return enhanced

    elif cs == "CHANNEL_WISE":
        # Independent per-channel equalization
        channels = cv2.split(image)
        eq_channels = [clahe.apply(ch) for ch in channels]
        return cv2.merge(eq_channels)

    else:
        raise ValueError(f"Unknown color_space '{color_space}'. Choose 'LAB', 'YCrCb', 'HSV', or 'CHANNEL_WISE'.")


def enhance_sonar_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    **kwargs,
) -> np.ndarray:
    """
    Convenience wrapper for CLAHE contrast enhancement on sonar imagery.
    """
    return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size, **kwargs)
