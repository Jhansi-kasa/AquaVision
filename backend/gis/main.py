# ============================================================
# AI MARINE DETECTION GIS SYSTEM
# Member 5 - GIS, Risk and Mission Planning
# ============================================================

import json
from pathlib import Path


try:
    from .risk_engine import calculate_risk
    from .priority_engine import (
        calculate_priority,
        rank_priorities
    )
    from .geo_service import (
        map_ready_detection,
        normalize_backend_data
    )
    from .route_planner import (
        plan_route
    )
except ImportError:
    from risk_engine import calculate_risk
    from priority_engine import (
        calculate_priority,
        rank_priorities
    )
    from geo_service import (
        map_ready_detection,
        normalize_backend_data
    )
    from route_planner import (
        plan_route
    )


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DETECTIONS_FILE = BASE_DIR / "detections.json"

MISSION_PLAN_FILE = BASE_DIR / "mission_plan.json"


# ============================================================
# LOAD DATA
# ============================================================

def load_detections():

    if not DETECTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n{DETECTIONS_FILE}"
        )

    with open(
        DETECTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if isinstance(data, dict):

        detections = data.get(
            "detections",
            []
        )

    else:

        detections = data

    if not isinstance(
        detections,
        list
    ):

        raise ValueError(
            "detections.json must contain a list."
        )

    return detections


# ============================================================
# BACKEND PLACEHOLDER
# ============================================================

def process_backend_data(detections):

    """
    Temporary local-data integration.

    Later, Member 6 can replace this function with
    actual FastAPI backend data.
    """

    return detections


# ============================================================
# RISK
# ============================================================

def calculate_all_risks(detections):

    results = []

    for detection in detections:

        result = calculate_risk(
            detection
        )

        results.append(
            result
        )

    return results


# ============================================================
# PRIORITY
# ============================================================

def calculate_all_priorities(
    detections,
    risk_results
):

    priority_results = []

    risk_lookup = {}

    for risk in risk_results:

        detection_id = risk.get(
            "detection_id"
        )

        risk_lookup[
            detection_id
        ] = risk

    for detection in detections:

        detection_id = (
            detection.get(
                "detection_id"
            )
            or detection.get(
                "id"
            )
        )

        risk_result = risk_lookup.get(
            detection_id,
            {
                "detection_id": detection_id,
                "risk_score": 0,
                "risk_level": "LOW"
            }
        )

        result = calculate_priority(
            detection,
            risk_result
        )

        priority_results.append(
            result
        )

    return rank_priorities(
        priority_results
    )


# ============================================================
# MAP DATA
# ============================================================

def create_map_data(detections):

    results = []

    for detection in detections:

        result = map_ready_detection(
            detection
        )

        results.append(
            result
        )

    return results


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    detections,
    risk_results,
    priority_results,
    route
):

    high_risk = 0
    medium_risk = 0
    low_risk = 0

    for result in risk_results:

        level = result.get(
            "risk_level",
            "LOW"
        )

        if level == "HIGH":
            high_risk += 1

        elif level == "MEDIUM":
            medium_risk += 1

        else:
            low_risk += 1

    high_priority = 0
    medium_priority = 0
    low_priority = 0

    for result in priority_results:

        priority = result.get(
            "priority",
            "LOW"
        )

        if priority == "HIGH":
            high_priority += 1

        elif priority == "MEDIUM":
            medium_priority += 1

        else:
            low_priority += 1

    return {

        "total_detections":
            len(detections),

        "risk_summary": {

            "high": high_risk,

            "medium": medium_risk,

            "low": low_risk
        },

        "priority_summary": {

            "high": high_priority,

            "medium": medium_priority,

            "low": low_priority
        },

        "route_stops":
            route.get(
                "total_stops",
                0
            ),

        "estimated_route_distance_km":
            route.get(
                "estimated_total_distance_km",
                0
            ),

        "total_clusters":
            route.get(
                "total_clusters",
                0
            )
    }


# ============================================================
# BUILD MISSION PLAN
# ============================================================

def build_mission_plan(
    detections,
    risk_results,
    priority_results,
    map_data,
    route,
    summary
):

    return {

        "project":
            "AI Marine Detection GIS System",

        "module":
            "Member 5 - GIS, Risk and Mission Planning",

        "data_source":
            "Detection data",

        "detections":
            detections,

        "risk_results":
            risk_results,

        "priority_results":
            priority_results,

        "map_data":
            map_data,

        "route":
            route,

        "summary":
            summary
    }


def generate_mission_plan(
    backend_data,
    surveys=None,
    vessel_start=None
):
    """
    Integration entrypoint for generating a complete mission plan.
    Accepts backend detection lists/dicts, optional survey metadata,
    and an optional initial vessel position.
    """
    if vessel_start is None:
        vessel_start = {"latitude": 17.6900, "longitude": 83.2200}

    if surveys is not None:
        detections = normalize_backend_data(backend_data, surveys)
    else:
        detections = process_backend_data(backend_data)

    risk_results = calculate_all_risks(detections)
    priority_results = calculate_all_priorities(detections, risk_results)
    map_data = create_map_data(detections)
    route = plan_route(detections, priority_results, vessel_start=vessel_start)
    summary = create_summary(
        detections,
        risk_results,
        priority_results,
        route
    )

    return build_mission_plan(
        detections,
        risk_results,
        priority_results,
        map_data,
        route,
        summary
    )


# ============================================================
# SAVE
# ============================================================

def save_mission_plan(mission_plan):

    with open(
        MISSION_PLAN_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            mission_plan,
            file,
            indent=4
        )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    detections,
    risk_results,
    priority_results
):

    risk_lookup = {}

    for result in risk_results:

        risk_lookup[
            result.get("detection_id")
        ] = result

    priority_lookup = {}

    for result in priority_results:

        priority_lookup[
            result.get("detection_id")
        ] = result

    print("\n===================================")
    print(" DETECTION RESULTS")
    print("===================================\n")

    for detection in detections:

        detection_id = (
            detection.get(
                "detection_id"
            )
            or detection.get(
                "id"
            )
        )

        object_class = (
            detection.get(
                "object_class"
            )
            or detection.get(
                "class"
            )
            or "unknown"
        )

        confidence = detection.get(
            "confidence",
            0
        )

        risk = risk_lookup.get(
            detection_id,
            {}
        )

        priority = priority_lookup.get(
            detection_id,
            {}
        )

        print("-----------------------------------")

        print(
            f"Detection ID   : {detection_id}"
        )

        print(
            f"Class          : {object_class}"
        )

        print(
            f"Confidence     : {confidence}"
        )

        print(
            f"Risk Score     : "
            f"{risk.get('risk_score', 0)}"
        )

        print(
            f"Risk Level     : "
            f"{risk.get('risk_level', 'LOW')}"
        )

        print(
            f"Priority Score : "
            f"{priority.get('priority_score', 0)}"
        )

        print(
            f"Priority       : "
            f"{priority.get('priority', 'LOW')}"
        )

        print(
            f"Priority Rank  : "
            f"{priority.get('priority_rank', '-')}"
        )

        print(
            f"Mission Type   : "
            f"{priority.get('mission_type', '-')}"
        )

    print("-----------------------------------")


# ============================================================
# PRINT ROUTE
# ============================================================

def print_route(route):

    print("\n===================================")
    print(" RECOMMENDED MISSION ROUTE")
    print("===================================\n")

    route_items = route.get(
        "route",
        []
    )

    if not route_items:

        print(
            "No valid route stops available."
        )

        return

    print(
        "Routing strategy : "
        f"{route.get('routing_strategy', '-')}"
    )

    print(
        "Cluster distance : "
        f"{route.get('cluster_distance_km', '-')} km\n"
    )

    print(
        "Vessel Start -> "
        f"({route['vessel_start']['latitude']}, "
        f"{route['vessel_start']['longitude']})"
    )

    for stop in route_items:

        print(
            f"{stop['sequence']}. "
            f"{stop['detection_id']} -> "
            f"{stop['object_class']} -> "
            f"({stop['latitude']}, "
            f"{stop['longitude']}) -> "
            f"{stop['priority']} -> "
            f"{stop['cluster_id']} -> "
            f"{stop['route_reason']}"
        )

    print(
        f"\nTotal route stops: "
        f"{route.get('total_stops', 0)}"
    )

    print(
        f"Total cleanup areas: "
        f"{route.get('total_clusters', 0)}"
    )

    print(
        f"Estimated distance: "
        f"{route.get('estimated_total_distance_km', 0)} km"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===================================")
    print(" AI MARINE DETECTION GIS SYSTEM")
    print("===================================\n")


    # --------------------------------------------------------
    # STEP 1
    # Load detections
    # --------------------------------------------------------

    detections = load_detections()

    print(
        f"Loaded {len(detections)} detections."
    )


    # --------------------------------------------------------
    # STEP 2
    # Backend data
    # --------------------------------------------------------

    detections = process_backend_data(
        detections
    )


    # --------------------------------------------------------
    # STEP 3
    # Risk calculation
    # --------------------------------------------------------

    risk_results = calculate_all_risks(
        detections
    )


    # --------------------------------------------------------
    # STEP 4
    # Priority calculation
    # --------------------------------------------------------

    priority_results = calculate_all_priorities(
        detections,
        risk_results
    )


    # --------------------------------------------------------
    # STEP 5
    # GIS map data
    # --------------------------------------------------------

    map_data = create_map_data(
        detections
    )


    # --------------------------------------------------------
    # STEP 6
    # Vessel start
    # --------------------------------------------------------

    vessel_start = {

        "latitude": 17.6900,

        "longitude": 83.2200
    }


    # --------------------------------------------------------
    # STEP 7
    # DISTANCE-FIRST ROUTE
    #
    # Nearest area is selected first.
    # Priority is applied inside nearby areas.
    # --------------------------------------------------------

    route = plan_route(

        detections,

        priority_results,

        vessel_start
    )


    # --------------------------------------------------------
    # STEP 8
    # Summary
    # --------------------------------------------------------

    summary = create_summary(

        detections,

        risk_results,

        priority_results,

        route
    )


    # --------------------------------------------------------
    # STEP 9
    # Mission plan
    # --------------------------------------------------------

    mission_plan = build_mission_plan(

        detections,

        risk_results,

        priority_results,

        map_data,

        route,

        summary
    )


    # --------------------------------------------------------
    # STEP 10
    # Save mission plan
    # --------------------------------------------------------

    save_mission_plan(
        mission_plan
    )


    # --------------------------------------------------------
    # STEP 11
    # Print detection results
    # --------------------------------------------------------

    print_results(

        detections,

        risk_results,

        priority_results
    )


    # --------------------------------------------------------
    # STEP 12
    # Print summary
    # --------------------------------------------------------

    print("\n===================================")
    print(" SUMMARY")
    print("===================================\n")

    print(
        f"Total detections : "
        f"{summary['total_detections']}"
    )

    print(
        f"High risk        : "
        f"{summary['risk_summary']['high']}"
    )

    print(
        f"Medium risk      : "
        f"{summary['risk_summary']['medium']}"
    )

    print(
        f"Low risk         : "
        f"{summary['risk_summary']['low']}"
    )

    print(
        f"High priority    : "
        f"{summary['priority_summary']['high']}"
    )

    print(
        f"Medium priority  : "
        f"{summary['priority_summary']['medium']}"
    )

    print(
        f"Low priority     : "
        f"{summary['priority_summary']['low']}"
    )

    print(
        f"Route stops      : "
        f"{summary['route_stops']}"
    )

    print(
        f"Cleanup areas    : "
        f"{summary['total_clusters']}"
    )

    print(
        f"Route distance   : "
        f"{summary['estimated_route_distance_km']} km"
    )


    # --------------------------------------------------------
    # STEP 13
    # Print route
    # --------------------------------------------------------

    print_route(
        route
    )


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n===================================")
    print(" PROCESSING COMPLETED")
    print("===================================")

    print(
        f"\nMission plan saved at:\n"
        f"{MISSION_PLAN_FILE}"
    )

    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()