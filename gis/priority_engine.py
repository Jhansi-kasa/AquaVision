# ============================================================
# PRIORITY ENGINE
# Member 5 - GIS, Risk and Mission Planning
# ============================================================

from typing import Any, Dict, List, Optional


def calculate_priority_score(detection: Dict[str, Any], risk_result: Any) -> float:
    """
    Calculate composite priority score (0-100) based on Member 5 specification:
    - Risk score: 50%
    - Confidence: 15%
    - Object type: 10%
    - Size: 10%
    - Depth: 5%
    - Data quality: 10%
    """
    if isinstance(risk_result, dict):
        risk_score = float(risk_result.get("risk_score", 0.0))
    else:
        try:
            risk_score = float(risk_result)
        except (TypeError, ValueError):
            risk_score = 0.0

    try:
        confidence = float(detection.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        confidence = 0.5

    obj_class = str(
        detection.get("object_class")
        or detection.get("objectClass")
        or detection.get("class")
        or ""
    ).lower()

    try:
        size = float(detection.get("estimated_size", 1.0) or 1.0)
    except (TypeError, ValueError):
        size = 1.0

    try:
        depth = float(detection.get("depth", 10.0) or 10.0)
    except (TypeError, ValueError):
        depth = 10.0

    try:
        data_quality = float(detection.get("data_quality", 0.8) or 0.8)
    except (TypeError, ValueError):
        data_quality = 0.8

    w_risk = 0.50 * risk_score
    w_conf = 0.15 * (min(1.0, max(0.0, confidence)) * 100.0)

    type_factors = {
        "mine": 100.0,
        "shipwreck": 90.0,
        "aircraft": 85.0,
        "fishing_gear": 80.0,
        "drowning_victim": 100.0,
        "crab_pot": 50.0,
        "seafloor": 0.0,
        "background": 0.0,
    }
    w_obj = 0.10 * type_factors.get(obj_class, 50.0)

    size_norm = min(100.0, max(0.0, size * 5.0))
    w_size = 0.10 * size_norm

    depth_norm = max(0.0, min(100.0, 100.0 - (depth / 2.0)))
    w_depth = 0.05 * depth_norm

    w_dq = 0.10 * (min(1.0, max(0.0, data_quality)) * 100.0)

    score = round(w_risk + w_conf + w_obj + w_size + w_depth + w_dq, 2)
    return max(0.0, min(100.0, score))


def get_mission_type(obj_class: str) -> str:
    c = str(obj_class).lower()
    if c in ["drowning_victim"]:
        return "REVIEW_RESPONSE"
    elif c in ["seafloor", "background"]:
        return "NO_ACTION"
    return "CLEANUP_REVIEW"


def calculate_priority(detection: Dict[str, Any], risk_result: Any) -> Dict[str, Any]:
    det_id = (
        detection.get("detection_id")
        or detection.get("detectionId")
        or detection.get("id")
    )
    score = calculate_priority_score(detection, risk_result)

    if score >= 70.0:
        p_tier = "HIGH"
    elif score >= 40.0:
        p_tier = "MEDIUM"
    else:
        p_tier = "LOW"

    obj_class = (
        detection.get("object_class")
        or detection.get("objectClass")
        or detection.get("class")
        or ""
    )

    return {
        "detection_id": det_id,
        "priority_score": score,
        "priority": p_tier,
        "priority_label": p_tier,
        "mission_type": get_mission_type(obj_class),
        "priority_rank": None,
    }


def rank_priorities(priority_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_results = sorted(
        priority_results,
        key=lambda x: x.get("priority_score", 0),
        reverse=True,
    )
    for rank, item in enumerate(sorted_results, start=1):
        item["priority_rank"] = rank
    return sorted_results


def calculate_priority_result(risk_score: Any, risk_level: Optional[str] = None) -> Dict[str, Any]:
    """Compatibility helper for backend priority queries."""
    try:
        score = float(risk_score)
    except (TypeError, ValueError):
        score = 0.0

    score = max(0.0, min(100.0, score))

    if risk_level:
        lvl = str(risk_level).strip().upper()
        if lvl in ["HIGH", "MEDIUM", "LOW"]:
            return {
                "priority": lvl.lower(),
                "priority_label": lvl,
                "risk_score": round(score, 2),
            }

    if score >= 70:
        p = "high"
    elif score >= 40:
        p = "medium"
    else:
        p = "low"

    return {
        "priority": p,
        "priority_label": p.upper(),
        "risk_score": round(score, 2),
    }