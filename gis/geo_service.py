# ============================================================
# GEO SERVICE
# Member 5 - GIS, Risk and Mission Planning
# ============================================================

from math import radians, sin, cos, sqrt, atan2


def validate_coordinates(latitude, longitude):
    if latitude is None or longitude is None:
        return False

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (ValueError, TypeError):
        return False

    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def location_available(detection):
    latitude = detection.get("latitude", detection.get("lat"))
    longitude = detection.get("longitude", detection.get("lon"))
    return validate_coordinates(latitude, longitude)


def haversine_distance_km(lat1, lon1, lat2, lon2):
    if not (
        validate_coordinates(lat1, lon1)
        and validate_coordinates(lat2, lon2)
    ):
        return None

    radius_km = 6371.0
    lat1_rad = radians(float(lat1))
    lat2_rad = radians(float(lat2))
    delta_lat = radians(float(lat2) - float(lat1))
    delta_lon = radians(float(lon2) - float(lon1))

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius_km * c


def find_missing_coordinates(detections):
    return [detection for detection in detections if not location_available(detection)]


def normalize_backend_detection(detection, survey=None):
    """
    Convert backend data into the common format expected by Member 5.

    Backend/YOLO fields:
        id or detection_id
        object_class / class
        confidence
        bbox
        latitude / longitude (if available)
        status

    Survey metadata can provide:
        latitude, longitude, depth, survey_id

    estimated_size and data_quality are preserved when supplied by the
    backend/integration layer. They are NOT invented here.
    """
    survey = survey or {}

    detection_id = detection.get("detection_id") or detection.get("id")
    object_class = (
        detection.get("object_class")
        or detection.get("class")
        or detection.get("object_type")
        or "unknown"
    )

    latitude = detection.get("latitude")
    if latitude is None:
        latitude = detection.get("lat")
    if latitude is None:
        latitude = survey.get("latitude")

    longitude = detection.get("longitude")
    if longitude is None:
        longitude = detection.get("lon")
    if longitude is None:
        longitude = survey.get("longitude")

    depth = detection.get("depth")
    if depth is None:
        depth = survey.get("depth")

    survey_id = detection.get("survey_id")
    if survey_id is None:
        survey_id = survey.get("survey_id", survey.get("id"))

    return {
        "detection_id": detection_id,
        "survey_id": survey_id,
        "object_class": object_class,
        "confidence": float(detection.get("confidence", 0.0) or 0.0),
        "bbox": detection.get("bbox"),
        "latitude": latitude,
        "longitude": longitude,
        "depth": depth,
        "estimated_size": detection.get(
            "estimated_size", detection.get("size")
        ),
        "data_quality": detection.get("data_quality", 1.0),
        "cleanup_status": detection.get(
            "cleanup_status", detection.get("status", "pending")
        ),
    }


def normalize_backend_data(detections, surveys=None):
    """
    Normalize a list of backend detections.

    `surveys` is optional. If supplied, it may be a list of survey objects.
    The adapter uses survey_id when available; otherwise the detection is
    normalized without survey metadata.
    """
    if isinstance(detections, dict):
        detections = detections.get("detections", [])

    if not isinstance(detections, list):
        raise ValueError("Backend detections must be a list or a dict containing 'detections'.")

    survey_lookup = {}
    if surveys:
        if isinstance(surveys, dict):
            surveys = surveys.get("surveys", [])
        for survey in surveys:
            survey_id = survey.get("survey_id", survey.get("id"))
            if survey_id is not None:
                survey_lookup[str(survey_id)] = survey

    normalized = []
    for detection in detections:
        survey_id = detection.get("survey_id")
        survey = survey_lookup.get(str(survey_id), {}) if survey_id is not None else {}
        normalized.append(normalize_backend_detection(detection, survey))

    return normalized


def map_ready_detection(detection):
    detection_id = detection.get("detection_id") or detection.get("id")
    object_class = (
        detection.get("object_class")
        or detection.get("class")
        or detection.get("object_type")
        or "unknown"
    )

    latitude = detection.get("latitude", detection.get("lat"))
    longitude = detection.get("longitude", detection.get("lon"))
    valid_location = validate_coordinates(latitude, longitude)

    if not valid_location:
        latitude = None
        longitude = None

    return {
        "detection_id": detection_id,
        "survey_id": detection.get("survey_id"),
        "object_class": object_class,
        "confidence": detection.get("confidence", 0),
        "bbox": detection.get("bbox"),
        "latitude": latitude,
        "longitude": longitude,
        "location_available": valid_location,
        "depth": detection.get("depth"),
        "estimated_size": detection.get("estimated_size", detection.get("size")),
        "data_quality": detection.get("data_quality", 1.0),
        "cleanup_status": detection.get(
            "cleanup_status", detection.get("status", "pending")
        ),
    }


def merge_ai_with_survey_metadata(ai_detection, survey_metadata):
    """Merge one AI detection with survey metadata."""
    detection_id = ai_detection.get("detection_id") or ai_detection.get("id")
    metadata = survey_metadata.get(detection_id, {})

    return normalize_backend_detection(ai_detection, metadata)
