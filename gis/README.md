# Member 5 - GIS, Risk and Mission Planning

## Purpose

This module receives AI/backend detection data and produces:

1. GPS/location validation
2. Prototype risk score (0-100)
3. Mission priority score and ranking
4. Map-ready detection data
5. Priority-weighted mission route
6. `mission_plan.json`

This is a prototype decision-support system. The risk weights and route method are configurable and are **not scientifically validated navigation or hazard-probability models**.

## Current YOLO classes

The current YOLO model uses:

- `shipwreck`
- `drowning_victim`
- `aircraft`
- `mine`
- `seafloor`
- `crab_pot`

## Files

```text
gis/
├── main.py
├── risk_engine.py
├── priority_engine.py
├── geo_service.py
├── route_planner.py
├── detections.json          # optional mock data for local testing
├── README.md
└── tests/
    └── test_member5.py
```

## Processing flow

```text
YOLO / Backend API
        ↓
Data normalization
        ↓
Geo Service
        ↓
Risk Engine
        ↓
Priority Engine
        ↓
Route Planner
        ↓
Mission Plan / Frontend Map
```

## Backend integration

Member 6 should use the integration function in `main.py`:

```python
from main import generate_mission_plan

mission_plan = generate_mission_plan(
    backend_detections,
    surveys=backend_surveys,
    vessel_start={"latitude": 17.6900, "longitude": 83.2200}
)
```

`backend_detections` can be either:

```python
[
    {
        "id": 1,
        "object_class": "shipwreck",
        "confidence": 0.93,
        "bbox": [120, 80, 340, 220],
        "status": "pending"
    }
]
```

or:

```python
{"detections": [...]}
```

## Required data

For the most meaningful risk score, the integration layer should provide:

- `id` / `detection_id`
- `object_class`
- `confidence`
- `bbox`
- `latitude`
- `longitude`
- `depth`
- `estimated_size`
- `data_quality`
- `status` / `cleanup_status`

### Important GPS rule

GPS coordinates must come from valid survey/georeferencing metadata. A YOLO bounding box does **not** provide geographic coordinates.

If detection-level coordinates are missing, the adapter can use survey `latitude`, `longitude`, and `depth` when the detection contains a matching `survey_id`.

If GPS remains unavailable, the detection stays in the results but is excluded from route planning.

## Risk score

Default weights:

| Factor | Weight |
|---|---:|
| Object type | 30% |
| AI confidence | 20% |
| Estimated size | 20% |
| Depth | 15% |
| Data quality | 15% |

Risk level:

- `HIGH`: 70-100
- `MEDIUM`: 40-69.99
- `LOW`: below 40

## Priority score

Default weights:

| Factor | Weight |
|---|---:|
| Risk score | 50% |
| Confidence | 15% |
| Object type | 10% |
| Size | 10% |
| Depth | 5% |
| Data quality | 10% |

Mission types:

- `CLEANUP_REVIEW` for cleanup-related targets
- `REVIEW_RESPONSE` for `drowning_victim`
- `NO_ACTION` for `seafloor`

## Route planning

The prototype route planner combines:

- 70% mission priority
- 30% geographic proximity

It uses the Haversine formula for approximate distance and excludes:

- completed/cleaned/verified/cancelled/rejected detections
- detections without valid GPS
- `NO_ACTION` detections

The route is for prototype decision support, not professional marine navigation.

## Local mock test

Put `detections.json` in this folder and run:

```bash
python main.py
```

This uses mock data only. In the integrated system, Member 6 should call `generate_mission_plan()` with backend data instead of relying on the mock file.
