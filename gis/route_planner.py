# ============================================================
# ROUTE PLANNER
#
# 1. If a vessel start exists, the route begins there (object or no object).
#    If not, the best starting detection is chosen automatically by trying
#    candidate starts and keeping the shortest total route.
# 2. Distance decides which area to visit next; priority breaks near-ties.
# 3. Detections close together form a cleanup cluster.
# 4. Inside a cluster: priority tier first (HIGH > MEDIUM > LOW),
#    nearest-neighbour inside each tier.
# ============================================================

import math

CLUSTER_DISTANCE_KM = 0.5
AREA_TIE_KM = 0.10
INSIDE_TIE_KM = 0.01
MAX_START_CANDIDATES = 50   # cap for the auto-start search


def haversine_distance_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    except (TypeError, ValueError):
        return float("inf")

    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_detection_id(detection):
    return (
        detection.get("detection_id")
        or detection.get("detectionId")
        or detection.get("id")
    )


def get_object_class(detection):
    return (
        detection.get("object_class")
        or detection.get("objectClass")
        or detection.get("class")
        or "unknown"
    )


def get_coordinates(detection):
    lat = detection.get("latitude")
    if lat is None:
        lat = detection.get("lat")
    lng = detection.get("longitude")
    if lng is None:
        lng = detection.get("lng")

    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return None, None

    if math.isnan(lat) or math.isnan(lng) or (lat == 0 and lng == 0):
        return None, None

    return lat, lng


def get_priority_rank(priority):
    try:
        return int(priority.get("priority_rank"))
    except (TypeError, ValueError):
        pass

    level = str(priority.get("priority", "LOW")).upper()
    return {"HIGH": 1, "MEDIUM": 2}.get(level, 3)


def get_priority_score(priority):
    try:
        return float(priority.get("priority_score", 0))
    except (TypeError, ValueError):
        return 0.0


def create_priority_lookup(priority_results):
    lookup = {}
    for p in priority_results:
        if not isinstance(p, dict):
            continue
        pid = p.get("detection_id") or p.get("detectionId") or p.get("id")
        if pid is not None:
            lookup[str(pid)] = p
    return lookup


def build_stops(detections, priority_lookup):
    stops = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue

        lat, lng = get_coordinates(detection)
        if lat is None:
            continue

        det_id = get_detection_id(detection)
        priority = priority_lookup.get(str(det_id), {})

        stops.append({
            "detection": detection,
            "id": det_id,
            "lat": lat,
            "lng": lng,
            "priority": priority,
            "rank": get_priority_rank(priority),
            "score": get_priority_score(priority),
        })
    return stops


def _dist(lat, lng, stop):
    return haversine_distance_km(lat, lng, stop["lat"], stop["lng"])


def select_next_seed(cur_lat, cur_lng, remaining):
    measured = [(_dist(cur_lat, cur_lng, s), s) for s in remaining]
    nearest = min(d for d, _ in measured)

    candidates = [(d, s) for d, s in measured if d <= nearest + AREA_TIE_KM]
    candidates.sort(key=lambda x: (x[1]["rank"], -x[1]["score"], x[0]))

    d, seed = candidates[0]
    tie_break = d > nearest + 1e-9
    return seed, tie_break


def create_cluster(seed, remaining):
    return [
        s for s in remaining
        if s is seed
        or haversine_distance_km(seed["lat"], seed["lng"], s["lat"], s["lng"])
        <= CLUSTER_DISTANCE_KM
    ]


def order_cluster(cluster, cur_lat, cur_lng):
    pending = list(cluster)
    ordered = []

    while pending:
        best_rank = min(s["rank"] for s in pending)
        tier = [s for s in pending if s["rank"] == best_rank]

        nxt = min(
            tier,
            key=lambda s: (
                round(_dist(cur_lat, cur_lng, s) / INSIDE_TIE_KM),
                -s["score"],
                _dist(cur_lat, cur_lng, s),
            ),
        )

        ordered.append(nxt)
        pending.remove(nxt)
        cur_lat, cur_lng = nxt["lat"], nxt["lng"]

    return ordered


# ---------------------------------------------------------------
# Run the planner once from a given origin.
#   vessel = (lat, lng) -> first leg is measured from the vessel
#   vessel = None       -> route starts AT the first stop (leg = 0)
# ---------------------------------------------------------------
def _run_route(stops, origin_lat, origin_lng, vessel):
    remaining = list(stops)
    route = []
    total = 0.0
    cluster_number = 0
    sequence = 1

    cur_lat, cur_lng = origin_lat, origin_lng
    prev = vessel  # None means "no previous point yet"

    while remaining:
        seed, tie_break = select_next_seed(cur_lat, cur_lng, remaining)
        cluster = create_cluster(seed, remaining)
        cluster_number += 1

        cluster_ids = {id(s) for s in cluster}
        remaining = [s for s in remaining if id(s) not in cluster_ids]

        ordered = order_cluster(cluster, cur_lat, cur_lng)

        for i, stop in enumerate(ordered):
            segment = _dist(prev[0], prev[1], stop) if prev else 0.0
            from_start = _dist(vessel[0], vessel[1], stop) if vessel else 0.0
            total += segment

            if sequence == 1 and vessel is None:
                reason = "START_POINT"
            elif i == 0:
                reason = "AREA_TIE_PRIORITY" if tie_break else "NEAREST_AREA"
            else:
                reason = "CLUSTER_PRIORITY"

            priority = stop["priority"]

            route.append({
                "sequence": sequence,
                "detection_id": stop["id"],
                "object_class": get_object_class(stop["detection"]),
                "latitude": stop["lat"],
                "longitude": stop["lng"],
                "priority": priority.get("priority", "LOW"),
                "priority_score": priority.get("priority_score", 0),
                "priority_rank": priority.get("priority_rank", stop["rank"]),
                "distance_from_vessel_km": round(from_start, 3),
                "segment_distance_km": round(segment, 3),
                "cluster_id": f"AREA-{cluster_number:02d}",
                "cluster_size": len(ordered),
                "route_reason": reason,
            })

            sequence += 1
            prev = (stop["lat"], stop["lng"])
            cur_lat, cur_lng = prev

    return route, total, cluster_number


# ---------------------------------------------------------------
# No vessel start: try candidate starting detections, keep the shortest
# route (ties -> higher priority first stop). Extreme points of the
# layout are the most likely good endpoints, so they are tried first.
# ---------------------------------------------------------------
def _best_auto_route(stops):
    c_lat = sum(s["lat"] for s in stops) / len(stops)
    c_lng = sum(s["lng"] for s in stops) / len(stops)

    candidates = sorted(
        stops,
        key=lambda s: -haversine_distance_km(c_lat, c_lng, s["lat"], s["lng"]),
    )[:MAX_START_CANDIDATES]

    best = None
    for cand in candidates:
        route, total, clusters = _run_route(stops, cand["lat"], cand["lng"], None)
        key = (round(total, 6), stops_rank(route, stops))
        if best is None or key < best[0]:
            best = (key, route, total, clusters)

    return best[1], best[2], best[3]


def stops_rank(route, stops):
    first_id = route[0]["detection_id"] if route else None
    for s in stops:
        if s["id"] == first_id:
            return s["rank"]
    return 3


def _parse_vessel(vessel_start):
    if not isinstance(vessel_start, dict):
        return None
    try:
        lat = float(vessel_start.get("latitude", vessel_start.get("lat")))
        lng = float(vessel_start.get("longitude", vessel_start.get("lng")))
    except (TypeError, ValueError):
        return None
    if math.isnan(lat) or math.isnan(lng) or (lat == 0 and lng == 0):
        return None
    return lat, lng


# ---------------------------------------------------------------
# MAIN  (vessel_start is now optional)
# ---------------------------------------------------------------
def plan_route(detections, priority_results, vessel_start=None):
    if not isinstance(detections, list):
        detections = []
    if not isinstance(priority_results, list):
        priority_results = []

    vessel = _parse_vessel(vessel_start)

    priority_lookup = create_priority_lookup(priority_results)
    stops = build_stops(detections, priority_lookup)

    if not stops:
        route, total, clusters = [], 0.0, 0
    elif vessel:
        route, total, clusters = _run_route(stops, vessel[0], vessel[1], vessel)
    else:
        route, total, clusters = _best_auto_route(stops)

    return {
        "vessel_start": (
            {"latitude": vessel[0], "longitude": vessel[1]} if vessel else None
        ),
        "start_mode": "VESSEL" if vessel else "AUTO_OPTIMIZED",
        "routing_strategy": "DISTANCE_FIRST_PRIORITY_ON_NEAR_TIES_AND_WITHIN_CLUSTER",
        "cluster_distance_km": CLUSTER_DISTANCE_KM,
        "area_tie_km": AREA_TIE_KM,
        "route": route,
        "total_stops": len(route),
        "total_clusters": clusters,
        "estimated_total_distance_km": round(total, 2),
    }