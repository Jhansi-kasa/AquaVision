"""
denoising.py
Reusable Sonar Image Denoising Module
Member 2: Computer Vision & Sonar Processing

Implements and configures three primary OpenCV denoising filters:
1. Gaussian Blur
2. Median Blur
3. Bilateral Filter

Strictly preserves 640x640 resolution, 3-channel RGB depth, and uint8 format.
"""

from typing import Tuple, Union, Optional
import cv2
import numpy as np


def validate_image(image: np.ndarray) -> np.ndarray:
    """Validate image array properties."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
    return image


def apply_gaussian_blur(
    image: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma_x: float = 1.2,
    sigma_y: Optional[float] = None,
) -> np.ndarray:
    """
    Apply Gaussian Blur to reduce high-frequency acoustic speckle noise.
    
    Parameters:
    -----------
    image : np.ndarray (H, W, C) or (H, W) of dtype uint8.
    ksize : Tuple[int, int], Gaussian kernel size (width, height). Must be positive odd integers.
    sigma_x : float, Gaussian kernel standard deviation in X direction.
    sigma_y : Optional[float], Gaussian kernel standard deviation in Y direction.
              If None or 0, sigma_y is set equal to sigma_x.

    Returns:
    --------
    np.ndarray : Denoised image matching input shape and dtype.
    """
    validate_image(image)
    kw, kh = ksize
    if kw % 2 == 0 or kh % 2 == 0 or kw < 1 or kh < 1:
        raise ValueError(f"Kernel dimensions must be positive odd integers, got {ksize}")
    
    sy = sigma_x if sigma_y is None else sigma_y
    denoised = cv2.GaussianBlur(image, ksize, sigmaX=sigma_x, sigmaY=sy)
    return denoised


def apply_median_blur(
    image: np.ndarray,
    ksize: int = 5,
) -> np.ndarray:
    """
    Apply Median Blur to suppress impulsive acoustic spikes and salt-and-pepper noise.
    
    Parameters:
    -----------
    image : np.ndarray (H, W, C) or (H, W) of dtype uint8.
    ksize : int, Aperture linear size; must be an odd integer greater than 1 (e.g. 3, 5, 7).

    Returns:
    --------
    np.ndarray : Denoised image matching input shape and dtype.
    """
    validate_image(image)
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}")
    
    denoised = cv2.medianBlur(image, ksize)
    return denoised


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """
    Apply Bilateral Filter to smooth speckle in homogeneous seafloor regions while
    preserving sharp acoustic highlight-shadow boundaries.
    
    Parameters:
    -----------
    image : np.ndarray (H, W, C) or (H, W) of dtype uint8.
    d : int, Diameter of each pixel neighborhood used during filtering.
        If non-positive, it is computed from sigma_space.
    sigma_color : float, Filter sigma in the color/intensity space. Larger values
                  mean farther colors will be mixed together.
    sigma_space : float, Filter sigma in the coordinate space. Larger values mean
                  farther pixels will influence each other.

    Returns:
    --------
    np.ndarray : Denoised image matching input shape and dtype.
    """
    validate_image(image)
    if d < 1:
        raise ValueError(f"Diameter d must be >= 1, got {d}")
    
    denoised = cv2.bilateralFilter(
        image,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )
    return denoised


def denoise_sonar_image(
    image: np.ndarray,
    method: str = "bilateral",
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for sonar denoising.
    
    Parameters:
    -----------
    image : np.ndarray
    method : str, one of ['gaussian', 'median', 'bilateral']
    kwargs : parameters passed to specific filter function.

    Returns:
    --------
    np.ndarray : Denoised image.
    """
    m = method.lower().strip()
    if m == "gaussian":
        return apply_gaussian_blur(image, **kwargs)
    elif m == "median":
        return apply_median_blur(image, **kwargs)
    elif m == "bilateral":
        return apply_bilateral_filter(image, **kwargs)
    else:
        raise ValueError(f"Unknown denoising method '{method}'. Choose 'gaussian', 'median', or 'bilateral'.")
