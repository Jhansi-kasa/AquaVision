"""
Side-Scan Sonar Cross-Track Swath Illumination Normalization Module (Task P3)
=============================================================================
Project: AI-Powered Automated Underwater Marine Debris and Anomaly Detection
Subsystem: Member 2 (Computer Vision & Sonar Processing)
Active Dataset: SIH_Anomaly_V1 (3 Classes: shipwreck, aircraft, mine)

Purpose & Physical Context:
---------------------------
Side-scan sonar imagery routinely exhibits severe cross-track radiometric
non-uniformity (intensity fall-off) across the swath width (range direction).
This is caused by:
1. Acoustic spherical / cylindrical geometric spreading loss (proportional to 1/R^2 or 1/R).
2. Frequency-dependent acoustic seawater attenuation (absorption and scattering).
3. Sonar beam directivity pattern (vertical beam roll-off at nadir and grazing angles).
4. Angular backscatter variation of benthic sediments (Lambertian backscatter roll-off).

Consequently, near-range regions often appear excessively bright / saturated,
while far-range regions decay into dark, low-contrast noise floors where faint
debris targets (crab pots, small mines, submerged anomalies) are obscured.

Scientific & Algorithmic Boundary:
----------------------------------
Because cropped 640x640 sonar imagery from public / competitive benchmarks
lacks explicit transducer geometry (towfish altitude, tilt angle, slant-range
lookup tables, beam pattern calibrations), an exact analytical physical 1/R^2
correction is physically impossible without making unverified assumptions.

This module implements an empirical, image-based cross-track illumination
normalization (an image-based proxy for Angle-Varying Gain / Time-Varying Gain).
It estimates the macroscopic backscatter profile across the swath, smooths the
profile to avoid target-induced bias, computes a safely clamped gain profile,
and normalizes the imagery while strictly preserving:
- Original 640x640 resolution and uint8 dynamic range.
- CIELAB chromaticity (zero false-color distortion).
- Acoustic shadow voids (near-zero returns remain dark).
- Small target boundaries without artificial banding or noise blow-up.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


def estimate_cross_track_profile(
    intensity: np.ndarray,
    axis: str = "horizontal",
    method: str = "median",
    trim_ratio: float = 0.15,
    percentile: float = 50.0,
) -> np.ndarray:
    """
    Estimate the 1D intensity profile across the cross-track (swath) direction.

    Parameters
    ----------
    intensity : np.ndarray
        2D single-channel image (H, W) representing luminance or grayscale intensity.
    axis : str
        Cross-track direction along which the profile varies:
        - 'horizontal': Cross-track is along columns (width, X-axis); statistics
          computed across rows for each column (length W). Standard for waterfall strips.
        - 'vertical': Cross-track is along rows (height, Y-axis); statistics
          computed across columns for each row (length H).
    method : str
        Statistical estimator for background profile:
        - 'median': 50th percentile along the orthogonal axis. Robust against
          localized target highlights and acoustic shadows. (Default)
        - 'trimmed_mean': Discards upper and lower `trim_ratio` percentiles to
          eliminate extreme outliers before computing mean.
        - 'mean': Standard arithmetic mean (sensitive to large objects/shadows).
        - 'percentile': User-specified percentile (e.g., 40.0 to 60.0).
    trim_ratio : float
        Fraction of extreme pixels to trim from each tail when method='trimmed_mean'.
    percentile : float
        Percentile value in [0, 100] when method='percentile'.

    Returns
    -------
    np.ndarray
        1D floating-point array of shape (W,) or (H,) representing the estimated profile.
    """
    if intensity.ndim != 2:
        raise ValueError(f"Expected 2D array for intensity, got shape {intensity.shape}")

    h, w = intensity.shape

    if axis == "horizontal":
        # Cross-track is horizontal (vary with column x, aggregate across rows y)
        if method == "median":
            profile = np.median(intensity, axis=0)
        elif method == "mean":
            profile = np.mean(intensity, axis=0)
        elif method == "percentile":
            profile = np.percentile(intensity, percentile, axis=0)
        elif method == "trimmed_mean":
            low_p = trim_ratio * 100.0
            high_p = (1.0 - trim_ratio) * 100.0
            profile = np.zeros(w, dtype=np.float32)
            for c in range(w):
                col = intensity[:, c]
                v_low = np.percentile(col, low_p)
                v_high = np.percentile(col, high_p)
                valid = col[(col >= v_low) & (col <= v_high)]
                profile[c] = np.mean(valid) if valid.size > 0 else np.median(col)
        else:
            raise ValueError(f"Unknown estimation method: {method}")
    elif axis == "vertical":
        # Cross-track is vertical (vary with row y, aggregate across columns x)
        if method == "median":
            profile = np.median(intensity, axis=1)
        elif method == "mean":
            profile = np.mean(intensity, axis=1)
        elif method == "percentile":
            profile = np.percentile(intensity, percentile, axis=1)
        elif method == "trimmed_mean":
            low_p = trim_ratio * 100.0
            high_p = (1.0 - trim_ratio) * 100.0
            profile = np.zeros(h, dtype=np.float32)
            for r in range(h):
                row = intensity[r, :]
                v_low = np.percentile(row, low_p)
                v_high = np.percentile(row, high_p)
                valid = row[(row >= v_low) & (row <= v_high)]
                profile[r] = np.mean(valid) if valid.size > 0 else np.median(row)
        else:
            raise ValueError(f"Unknown estimation method: {method}")
    else:
        raise ValueError(f"Invalid axis: {axis}. Expected 'horizontal' or 'vertical'.")

    return profile.astype(np.float32)


def smooth_profile(
    profile: np.ndarray,
    method: str = "gaussian",
    sigma: float = 35.0,
    kernel_size: int = 71,
    poly_order: int = 2,
) -> np.ndarray:
    """
    Smooth the 1D cross-track profile to capture macro swath fall-off while
    filtering out localized high-frequency texture and seabed sand ripples.

    Parameters
    ----------
    profile : np.ndarray
        1D raw estimated intensity profile.
    method : str
        Smoothing algorithm: 'gaussian', 'moving_average', or 'polynomial'.
    sigma : float
        Gaussian standard deviation in pixels.
    kernel_size : int
        Odd integer kernel size for Gaussian or moving average.
    poly_order : int
        Order of polynomial fit if method='polynomial'.

    Returns
    -------
    np.ndarray
        Smoothed 1D floating-point profile with same length as input.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    length = len(profile)
    # Clamp kernel size to length
    if kernel_size > length:
        kernel_size = length if length % 2 == 1 else length - 1

    if method == "gaussian":
        # Vectorized 1D Gaussian smoothing via cv2.GaussianBlur on (1, N)
        profile_2d = profile.reshape(1, -1)
        smoothed = cv2.GaussianBlur(
            profile_2d,
            ksize=(kernel_size, 1),
            sigmaX=sigma,
            sigmaY=0,
            borderType=cv2.BORDER_REFLECT,
        ).flatten()
    elif method == "moving_average":
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        pad_size = kernel_size // 2
        padded = np.pad(profile, pad_size, mode="reflect")
        smoothed = np.convolve(padded, kernel, mode="valid")[:length]
    elif method == "polynomial":
        x = np.linspace(-1.0, 1.0, length)
        coeffs = np.polyfit(x, profile, deg=poly_order)
        smoothed = np.polyval(coeffs, x)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return smoothed.astype(np.float32)


def compute_gain_profile(
    smoothed_profile: np.ndarray,
    target_level: Union[str, float] = "mean",
    min_gain: float = 0.5,
    max_gain: float = 2.5,
    min_intensity_floor: float = 8.0,
) -> Tuple[np.ndarray, float]:
    """
    Construct a safe, bounded multiplicative correction gain profile:
        G(x) = Target_Intensity / Smoothed_Profile(x)

    Includes sonar-specific safeguards:
    1. Gain Clamping: Prevents excessive noise amplification in deep fall-off zones.
    2. Nadir / Shadow Safeguard: If the background profile is below `min_intensity_floor`
       (e.g., nadir water column or true acoustic shadow void), the gain smoothly
       tapers toward 1.0 to prevent amplifying zero-signal water column into gray noise.

    Parameters
    ----------
    smoothed_profile : np.ndarray
        1D smoothed background intensity profile.
    target_level : str or float
        Target radiometric level to normalize toward:
        - 'mean': Mean of smoothed profile (excluding extreme floor).
        - 'median': Median of smoothed profile.
        - float: Explicit numerical target (e.g. 80.0).
    min_gain : float
        Minimum allowed gain (prevents over-darkening near-range highlights).
    max_gain : float
        Maximum allowed gain (prevents speckle explosion at far range).
    min_intensity_floor : float
        Threshold below which background is considered water column or deep shadow.

    Returns
    -------
    Tuple[np.ndarray, float]
        (clamped_gain_profile, resolved_target_intensity)
    """
    valid_profile = smoothed_profile[smoothed_profile > min_intensity_floor]
    if valid_profile.size == 0:
        valid_profile = smoothed_profile

    if isinstance(target_level, str):
        if target_level == "mean":
            target_val = float(np.mean(valid_profile))
        elif target_level == "median":
            target_val = float(np.median(valid_profile))
        else:
            raise ValueError(f"Unknown target_level string: {target_level}")
    else:
        target_val = float(target_level)

    # Safe denominator with minimum intensity floor
    denom = np.maximum(smoothed_profile, min_intensity_floor)
    raw_gain = target_val / denom

    # Standard clamping
    clamped_gain = np.clip(raw_gain, min_gain, max_gain)

    # Sonar water column safeguard:
    # If smoothed profile is below min_intensity_floor (nadir water column),
    # smoothly taper the gain back to 1.0 (neutral) so water column remains black.
    alpha = np.clip(smoothed_profile / min_intensity_floor, 0.0, 1.0)
    final_gain = 1.0 + alpha * (clamped_gain - 1.0)

    return final_gain.astype(np.float32), target_val


def detect_dominant_swath_axis(image_gray: np.ndarray) -> str:
    """
    Detect whether the cross-track swath fall-off is predominantly
    along the horizontal axis (columns) or vertical axis (rows).

    Returns
    -------
    str: 'horizontal' or 'vertical'
    """
    col_prof = np.median(image_gray, axis=0)
    row_prof = np.median(image_gray, axis=1)

    # Compute variation (span and std)
    col_var = float(np.std(col_prof)) * (float(np.max(col_prof)) - float(np.min(col_prof)))
    row_var = float(np.std(row_prof)) * (float(np.max(row_prof)) - float(np.min(row_prof)))

    return "horizontal" if col_var >= row_var else "vertical"


def normalize_cross_track_illumination(
    image: np.ndarray,
    axis: str = "horizontal",
    method: str = "median",
    smooth_method: str = "gaussian",
    smooth_sigma: float = 35.0,
    smooth_kernel_size: int = 71,
    min_gain: float = 0.5,
    max_gain: float = 2.5,
    target_level: Union[str, float] = "mean",
    min_intensity_floor: float = 8.0,
    output_range: Tuple[int, int] = (0, 255),
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply cross-track swath illumination normalization to a sonar image.

    Features:
    - Supports Grayscale (H, W) and 3-channel RGB/BGR (H, W, 3).
    - Preserves color fidelity by processing CIELAB Luminance (L*) exclusively.
    - Resolves constant or near-constant inputs safely without zero-division.
    - Preserves original 640x640 resolution and uint8 dynamic range.
    - Configurable smoothing, background estimator, and safe gain clamping.

    Parameters
    ----------
    image : np.ndarray
        Input sonar image (uint8, 2D grayscale or 3D BGR).
    axis : str
        Swath direction: 'horizontal' (columns), 'vertical' (rows), or 'auto'.
    method : str
        Profile estimator: 'median', 'trimmed_mean', 'mean', or 'percentile'.
    smooth_method : str
        Filter for 1D profile: 'gaussian', 'moving_average', or 'polynomial'.
    smooth_sigma : float
        Gaussian sigma for smoothing.
    smooth_kernel_size : int
        Filter kernel size for smoothing.
    min_gain : float
        Lower bound on multiplicative gain.
    max_gain : float
        Upper bound on multiplicative gain.
    target_level : str or float
        Target intensity level ('mean', 'median', or numeric float).
    min_intensity_floor : float
        Intensity floor for nadir / shadow protection.
    output_range : Tuple[int, int]
        Output pixel range, default (0, 255).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (normalized_image, diagnostic_metadata)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy.ndarray")

    # Safety check: constant or near-constant image
    if np.std(image) < 1e-3:
        meta = {
            "axis": axis,
            "status": "constant_image_bypassed",
            "raw_profile": np.zeros(image.shape[1] if axis != "vertical" else image.shape[0]),
            "smoothed_profile": np.zeros(image.shape[1] if axis != "vertical" else image.shape[0]),
            "gain_profile": np.ones(image.shape[1] if axis != "vertical" else image.shape[0]),
            "target_val": float(np.mean(image)),
        }
        return image.copy(), meta

    is_color = image.ndim == 3 and image.shape[2] == 3

    # Extract single-channel luminance
    if is_color:
        # Convert BGR to CIELAB; L* represents perceived luminance (0..255 in OpenCV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        intensity_plane = lab[:, :, 0].astype(np.float32)
    else:
        intensity_plane = image.astype(np.float32)

    # Determine axis
    resolved_axis = axis
    if axis == "auto":
        resolved_axis = detect_dominant_swath_axis(intensity_plane)

    # 1. Estimate 1D raw profile
    raw_profile = estimate_cross_track_profile(
        intensity=intensity_plane,
        axis=resolved_axis,
        method=method,
    )

    # 2. Smooth the profile
    smoothed_profile = smooth_profile(
        profile=raw_profile,
        method=smooth_method,
        sigma=smooth_sigma,
        kernel_size=smooth_kernel_size,
    )

    # 3. Compute safe clamped gain profile
    gain_profile, target_val = compute_gain_profile(
        smoothed_profile=smoothed_profile,
        target_level=target_level,
        min_gain=min_gain,
        max_gain=max_gain,
        min_intensity_floor=min_intensity_floor,
    )

    # 4. Apply gain along the cross-track direction
    if resolved_axis == "horizontal":
        # 1D gain of shape (W,) broadcast across rows (H, W)
        gain_2d = gain_profile.reshape(1, -1)
    else:
        # 1D gain of shape (H,) broadcast across cols (H, W)
        gain_2d = gain_profile.reshape(-1, 1)

    norm_intensity = intensity_plane * gain_2d
    norm_intensity = np.clip(norm_intensity, output_range[0], output_range[1])

    # 5. Reconstruct image
    if is_color:
        lab[:, :, 0] = norm_intensity.astype(np.uint8)
        norm_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        norm_image = norm_intensity.astype(np.uint8)

    meta = {
        "axis": resolved_axis,
        "status": "success",
        "raw_profile": raw_profile,
        "smoothed_profile": smoothed_profile,
        "gain_profile": gain_profile,
        "target_val": target_val,
        "min_gain_used": float(np.min(gain_profile)),
        "max_gain_used": float(np.max(gain_profile)),
        "mean_gain_used": float(np.mean(gain_profile)),
    }

    return norm_image, meta


# Pre-packaged configurations for evaluation and comparison
CONFIG_A_CONSERVATIVE = {
    "name": "Config A (Conservative)",
    "smooth_sigma": 50.0,
    "smooth_kernel_size": 101,
    "min_gain": 0.6,
    "max_gain": 1.8,
    "target_level": "median",
    "method": "median",
    "description": "Soft gain range [0.6, 1.8], heavy smoothing (sigma=50). Zero noise blow-up risk.",
}

CONFIG_B_BALANCED = {
    "name": "Config B (Balanced / Recommended)",
    "smooth_sigma": 35.0,
    "smooth_kernel_size": 71,
    "min_gain": 0.5,
    "max_gain": 2.5,
    "target_level": "mean",
    "method": "median",
    "description": "Balanced gain [0.5, 2.5], medium smoothing (sigma=35). Optimal fall-off correction & edge retention.",
}

CONFIG_C_AGGRESSIVE = {
    "name": "Config C (Aggressive)",
    "smooth_sigma": 20.0,
    "smooth_kernel_size": 45,
    "min_gain": 0.4,
    "max_gain": 3.5,
    "target_level": "mean",
    "method": "trimmed_mean",
    "description": "Wide gain range [0.4, 3.5], tight smoothing (sigma=20). Strongest far-range boost, higher sediment noise.",
}
