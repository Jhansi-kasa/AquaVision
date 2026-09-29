import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_service import normalize_backend_data, validate_coordinates, haversine_distance_km
from main import generate_mission_plan


def test_coordinate_validation():
    assert validate_coordinates(17.6860, 83.2180)
    assert not validate_coordinates(100, 83.2180)


def test_haversine_distance():
    distance = haversine_distance_km(17.6900, 83.2200, 17.6900, 83.2200)
    assert distance == 0


def test_backend_normalization():
    backend = {
        "detections": [
            {
                "id": 1,
                "object_class": "shipwreck",
                "confidence": 0.93,
                "bbox": [120, 80, 340, 220],
                "status": "pending",
                "survey_id": 10,
            }
        ]
    }
    surveys = [
        {
            "id": 10,
            "latitude": 17.6860,
            "longitude": 83.2180,
            "depth": 42.5,
        }
    ]

    normalized = normalize_backend_data(backend, surveys)
    assert normalized[0]["detection_id"] == 1
    assert normalized[0]["object_class"] == "shipwreck"
    assert normalized[0]["latitude"] == 17.6860
    assert normalized[0]["longitude"] == 83.2180
    assert normalized[0]["depth"] == 42.5


def test_mission_plan_generation():
    backend = {
        "detections": [
            {
                "id": 1,
                "object_class": "shipwreck",
                "confidence": 0.93,
                "bbox": [120, 80, 340, 220],
                "status": "pending",
                "survey_id": 10,
                "estimated_size": 12.4,
                "data_quality": 0.95,
            }
        ]
    }
    surveys = [
        {
            "id": 10,
            "latitude": 17.6860,
            "longitude": 83.2180,
            "depth": 42.5,
        }
    ]

    plan = generate_mission_plan(backend, surveys)
    assert len(plan["detections"]) == 1
    assert plan["risk_results"][0]["risk_score"] > 0
    assert plan["priority_results"][0]["priority_score"] > 0
    assert plan["route"]["total_stops"] == 1


if __name__ == "__main__":
    test_coordinate_validation()
    test_haversine_distance()
    test_backend_normalization()
    test_mission_plan_generation()
    print("All GIS Member 5 tests passed successfully!")
