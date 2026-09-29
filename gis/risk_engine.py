# ============================================================
# RISK ENGINE
# 4-Class Marine Sonar Detection
# ============================================================

# Risk score is a prototype decision-support score from 0-100.
# It is NOT a scientifically validated probability of hazard.

DEFAULT_WEIGHTS = {
    "object_type": 40.0,
    "confidence": 30.0,
    "depth": 15.0,
    "data_quality": 15.0,
}

# Current 4-class YOLO model
OBJECT_TYPE_FACTORS = {
    "mine": 1.00,
    "shipwreck": 0.90,
    "aircraft": 0.85,
    "fishing_gear": 0.80,
}

DEPTH_REFERENCE_M = 100.0


def clamp(value, minimum=0.0, maximum=1.0):
    """Keep a value inside the supplied range."""
    return max(minimum, min(maximum, value))


def get_object_class(detection):
    """Read the object class from a detection dictionary."""
    return (
        detection.get("object_class")
        or detection.get("class")
        or detection.get("object_type")
        or "unknown"
    )


def get_detection_id(detection):
    """Read the detection ID."""
    return (
        detection.get("detection_id")
        or detection.get("id")
        or "UNKNOWN"
    )


def calculate_risk(detection, weights=None):
    """
    Calculate a prototype risk score from 0-100.

    Size/bounding-box dimensions are intentionally NOT used because
    pixel bounding-box dimensions cannot reliably represent real-world
    object size without sonar calibration / scale information.

    Factors:
        - Object type       : 40%
        - Detection confidence : 30%
        - Depth             : 15%
        - Data quality      : 15%
    """

    if weights is None:
        weights = DEFAULT_WEIGHTS

    detection_id = get_detection_id(detection)
    object_class = get_object_class(detection)

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    try:
        confidence = float(
            detection.get("confidence", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = clamp(confidence)

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    try:
        depth = float(
            detection.get("depth", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        depth = 0.0

    depth = max(0.0, depth)

    # Deeper objects require more difficult recovery/inspection.
    depth_factor = clamp(
        depth / DEPTH_REFERENCE_M
    )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    try:
        data_quality = float(
            detection.get("data_quality", 1.0) or 0.0
        )
    except (TypeError, ValueError):
        data_quality = 0.0

    data_quality = clamp(data_quality)

    # --------------------------------------------------------
    # OBJECT TYPE
    # --------------------------------------------------------

    object_factor = OBJECT_TYPE_FACTORS.get(
        object_class,
        0.50
    )

    # --------------------------------------------------------
    # CONVERT TO 0-100
    # --------------------------------------------------------

    object_score = object_factor * 100.0
    confidence_score = confidence * 100.0
    depth_score = depth_factor * 100.0
    quality_score = data_quality * 100.0

    # --------------------------------------------------------
    # WEIGHTED RISK
    # --------------------------------------------------------

    total_weight = sum(weights.values())

    if total_weight <= 0:
        total_weight = 100.0

    risk_score = (
        object_score * weights["object_type"]
        + confidence_score * weights["confidence"]
        + depth_score * weights["depth"]
        + quality_score * weights["data_quality"]
    ) / total_weight

    risk_score = round(
        clamp(risk_score, 0.0, 100.0),
        2
    )

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "detection_id": detection_id,
        "risk_score": risk_score,
        "risk_level": risk_level,

        "risk_factors": {
            "object_type": object_class,
            "confidence": confidence,
            "depth": depth,
            "data_quality": data_quality,
        },

        "factor_scores": {
            "object_type": round(object_score, 2),
            "confidence": round(confidence_score, 2),
            "depth": round(depth_score, 2),
            "data_quality": round(quality_score, 2),
        },
    }