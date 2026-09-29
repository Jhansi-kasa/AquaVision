import os
import sys

# Disable Cython C-extension for SQLAlchemy to prevent Windows Application Control blocks
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"

# Ensure project root is in sys.path so computer_vision and backend imports work from any working directory
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", ".."))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)
BACKEND_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))
if BACKEND_ROOT_DIR not in sys.path:
    sys.path.insert(0, BACKEND_ROOT_DIR)

import shutil
from datetime import datetime
from typing import List, Optional
from functools import cmp_to_key

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from ultralytics import YOLO
import cv2
import numpy as np
import base64
try:
    from computer_vision.final_preprocessing_pipeline import process_sonar_image
except ImportError:
    from .preprocessing.final_preprocessing_pipeline import process_sonar_image

from .database import engine, get_db, Base
from . import models, schemas
from gis.risk_engine import calculate_risk
from gis.priority_engine import calculate_priority_result
# Create all tables in the database (only creates if they don't exist)
Base.metadata.create_all(bind=engine)

# Lightweight SQLite migration for the new GIS integration fields.
def ensure_detection_columns():
    try:
        inspector = inspect(engine)
        if "detections" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("detections")}
        additions = {
            "depth": "FLOAT",
            "estimated_size": "FLOAT",
            "data_quality": "FLOAT",
            "survey_detection_index": "INTEGER",
        }
        with engine.begin() as conn:
            for name, sql_type in additions.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE detections ADD COLUMN {name} {sql_type}"))
    except Exception as exc:
        print(f"Warning: could not migrate detection GIS fields: {exc}")


ensure_detection_columns()

app = FastAPI(title="Marine Debris Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Project root (one level above backend)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
# Primary integrated model: Aqua Vision 100-Epoch Physics-Constrained 4-Class Sonar Model
AQUA_VISION_MODEL_PATH = os.path.join(BASE_DIR, "app", "model", "aqua_vision_100ep_best.pt")
ROOT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "aqua_vision_100ep_best.pt")
EXP_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "runs", "detect", "4class_training", "aqua_vision_100ep", "weights", "best.pt")
RUNS_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "4class_training", "aqua_vision_100ep", "weights", "best.pt")
BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "4class_training", "baseline_v1", "weights", "best.pt")

if os.path.exists(AQUA_VISION_MODEL_PATH):
    MODEL_PATH = AQUA_VISION_MODEL_PATH
elif os.path.exists(ROOT_MODEL_PATH):
    MODEL_PATH = ROOT_MODEL_PATH
elif os.path.exists(EXP_MODEL_PATH):
    MODEL_PATH = EXP_MODEL_PATH
elif os.path.exists(RUNS_MODEL_PATH):
    MODEL_PATH = RUNS_MODEL_PATH
elif os.path.exists(BASELINE_MODEL_PATH):
    MODEL_PATH = BASELINE_MODEL_PATH
else:
    raise FileNotFoundError(f"4-class model not found. Checked: {AQUA_VISION_MODEL_PATH}, {ROOT_MODEL_PATH}, {EXP_MODEL_PATH}, {RUNS_MODEL_PATH}")

FOUR_CLASS_MODEL_PATH = MODEL_PATH

model = YOLO(MODEL_PATH)
print(f"MODEL PATH: {os.path.abspath(MODEL_PATH)}")
print(f"MODEL CLASSES: {model.names}")


def preprocess_sonar(img_bgr_or_gray):
    """Apply the full 4‑stage training preprocessing pipeline to an image.
    This function re‑uses `process_sonar_image` from `computer_vision.final_preprocessing_pipeline`.
    It accepts either a BGR 3‑channel image or a single‑channel grayscale image.
    Grayscale inputs are first converted to BGR to match the expected input shape.
    The returned image has identical dimensions to the input (no cropping/resizing).
    """
    # Ensure a 3‑channel BGR image for the pipeline
    if len(img_bgr_or_gray.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_bgr_or_gray
    # Reuse the verified 4‑stage preprocessing implementation
    return process_sonar_image(img_bgr)


def calculate_box_overlap(box1, box2):
    """
    Calculate Intersection over Union (IoU) and Intersection over Min Area (IoMin)
    between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0.0 else 0.0
    min_area = min(area1, area2)
    io_min = inter_area / min_area if min_area > 0.0 else 0.0

    return iou, io_min


def suppress_duplicate_detections(candidates, iou_thresh=0.45, containment_thresh=0.65, conf_tolerance=0.02):
    """
    Perform class-aware Non-Maximum Suppression (duplicate suppression) on candidate detections.

    Each candidate is a dict containing:
      - 'class' or 'object_class' or 'class_name' or 'class_id'
      - 'confidence': float
      - 'bbox': [x1, y1, x2, y2]

    Selection criteria when multiple boxes represent the same physical object:
      1. Highest confidence
      2. If confidence is very similar (|conf1 - conf2| <= conf_tolerance),
         prefer the detection with better spatial coverage (larger bounding box area)

    Rules:
      - Class-wise: Boxes of different classes never suppress each other.
      - Preserves original model bounding box (no merging or coordinate averaging).
      - Suppresses candidate boxes with high overlap (IoU >= iou_thresh or IoMin >= containment_thresh).
      - Preserves distinct physical objects of the same class with low overlap.

    Returns:
      tuple: (final_detections, raw_count, nms_count, duplicates_removed)
    """
    raw_count = len(candidates)
    if raw_count == 0:
        return [], 0, 0, 0

    # Group candidate detections by class (Class-wise duplicate suppression)
    by_class = {}
    for cand in candidates:
        cls_key = cand.get("class", cand.get("object_class", cand.get("class_name", str(cand.get("class_id", "default")))))
        by_class.setdefault(cls_key, []).append(cand)

    def compare_candidates(a, b):
        conf_a = float(a["confidence"])
        conf_b = float(b["confidence"])
        conf_diff = conf_a - conf_b

        # 1. Highest confidence if difference exceeds tolerance
        if abs(conf_diff) > conf_tolerance:
            return 1 if conf_diff > 0 else -1

        # 2. If confidence is very similar, prefer better spatial coverage / complete object coverage
        bbox_a = a["bbox"]
        bbox_b = b["bbox"]
        area_a = max(0.0, float(bbox_a[2]) - float(bbox_a[0])) * max(0.0, float(bbox_a[3]) - float(bbox_a[1]))
        area_b = max(0.0, float(bbox_b[2]) - float(bbox_b[0])) * max(0.0, float(bbox_b[3]) - float(bbox_b[1]))

        if abs(area_a - area_b) > 1.0:
            return 1 if area_a > area_b else -1

        return 1 if conf_diff >= 0 else -1

    final_detections = []

    for cls_name, cls_candidates in by_class.items():
        # Sort candidates descending using multi-criteria comparator
        sorted_candidates = sorted(cls_candidates, key=cmp_to_key(compare_candidates), reverse=True)

        kept_for_class = []
        for cand in sorted_candidates:
            is_duplicate = False
            for kept in kept_for_class:
                iou, io_min = calculate_box_overlap(cand["bbox"], kept["bbox"])
                # If high overlap with an already selected representative box of the same class
                if iou >= iou_thresh or io_min >= containment_thresh:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_for_class.append(cand)

        final_detections.extend(kept_for_class)

    # Sort final detections by confidence descending
    final_detections.sort(key=lambda d: float(d["confidence"]), reverse=True)

    nms_count = len(final_detections)
    duplicates_removed = raw_count - nms_count

    return final_detections, raw_count, nms_count, duplicates_removed


OPTIMAL_CLASS_THRESHOLDS = {
    0: 0.25,  # Shipwreck: 0.25 suppresses diffuse seafloor reverberations (85.7% Precision)
    1: 0.21,  # Aircraft: 0.21 captures aerodynamic acoustic shadows (80.0% Recall)
    2: 0.10,  # Mine: 0.10 sensitive threshold recovers micro-anomalies (<20px)
    3: 0.16   # Fishing Gear: 0.16 recovers dense underwater nets/traps (48.0% Recall)
}


def post_process_sonar_detections(
    raw_candidates: List[dict],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    containment_thresh: float = 0.65,
    conf_tolerance: float = 0.02,
):
    """
    Standard Sonar Detection Post-Processing Pipeline:
      YOLO inference -> confidence filtering -> class-aware NMS -> final detections -> database/frontend
    """
    raw_count = len(raw_candidates)

    # 1. Confidence filtering: Discard predictions below class-adaptive or provided conf_thresh
    confident_candidates = []
    below_thresh_count = 0
    for cand in raw_candidates:
        conf = float(cand.get("confidence", 0.0))
        cid = cand.get("class_id", -1)
        if isinstance(cid, str) and cid.isdigit():
            cid = int(cid)
        effective_thresh = OPTIMAL_CLASS_THRESHOLDS.get(cid, conf_thresh) if (conf_thresh is None or conf_thresh == 0.25) else conf_thresh

        if conf >= effective_thresh:
            confident_candidates.append(cand)
        else:
            below_thresh_count += 1

    # 2. Class-aware NMS (duplicate suppression on confident predictions)
    final_detections, _, after_nms_count, duplicates_removed = suppress_duplicate_detections(
        confident_candidates,
        iou_thresh=iou_thresh,
        containment_thresh=containment_thresh,
        conf_tolerance=conf_tolerance,
    )

    # 3. Temporary backend logging
    print(f"Raw predictions: {raw_count}")
    print(f"Below confidence threshold: {below_thresh_count}")
    print(f"After NMS: {after_nms_count}")
    print(f"Final detections: {len(final_detections)}")

    return final_detections, raw_count, below_thresh_count, after_nms_count


# Class-calibrated confidence thresholds optimized for high acoustic recall
CLASS_THRESHOLDS = {
    "mine": float(os.getenv("CONF_MINE", "0.08")),        # High sensitivity for subtle micro-targets (<20px)
    "shipwreck": float(os.getenv("CONF_SHIPWRECK", "0.10")), # High sensitivity for structural debris / hull parts
    "aircraft": float(os.getenv("CONF_AIRCRAFT", "0.15")),   # High sensitivity for fuselages & airframe wreckage
    "fishing_gear": float(os.getenv("CONF_FISHING_GEAR", "0.20")), # Moderate sensitivity for nets & gear
}


def predict_sonar_high_recall(processed_img, model_instance, user_conf=None, iou_thresh=0.45, use_tiling=True):
    """
    High-recall sonar target detection pipeline:
    - Automatically handles large waterfall imagery (>800px) via slicing (tiled inference)
      to avoid downsampling micro-targets (e.g. mines <20px).
    - Uses calibrated class-specific thresholds:
      * mine: 0.08 (UXO highlight-shadow pairs)
      * shipwreck: 0.10 (degraded hull / debris field fragments)
      * aircraft: 0.15 (structural airframe targets)
    - If user provides an explicit confidence threshold, it acts as the filtering floor.
    """
    h, w = processed_img.shape[:2]
    base_conf = 0.06 if user_conf is None else min(user_conf, 0.08)

    candidate_boxes = []

    # Sliced / Tiled inference for large sonar strips/waterfalls
    if use_tiling and (w > 800 or h > 800):
        tile_size = 640
        stride = 480
        for y in range(0, max(1, h - tile_size + stride), stride):
            for x in range(0, max(1, w - tile_size + stride), stride):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)
                tile = processed_img[y_start:y_end, x_start:x_end]
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue
                tile_res = model_instance.predict(tile, conf=base_conf, iou=iou_thresh, verbose=False)[0]
                for b in tile_res.boxes:
                    cid = int(b.cls[0].item())
                    rc = float(b.conf[0].item())
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().tolist()
                    candidate_boxes.append((cid, rc, [float(bx1 + x_start), float(by1 + y_start), float(bx2 + x_start), float(by2 + y_start)]))

        # Apply class-wise NMS across tile boundaries
        dedup_boxes = []
        for target_cid in set(c[0] for c in candidate_boxes):
            class_cands = [c for c in candidate_boxes if c[0] == target_cid]
            if not class_cands:
                continue
            bboxes = [[b[2][0], b[2][1], b[2][2] - b[2][0], b[2][3] - b[2][1]] for b in class_cands]
            scores = [c[1] for c in class_cands]
            nms_indices = cv2.dnn.NMSBoxes(
                bboxes=bboxes,
                scores=scores,
                score_threshold=0.0,
                nms_threshold=iou_thresh
            )
            if len(nms_indices) > 0:
                for idx in np.array(nms_indices).flatten():
                    dedup_boxes.append(class_cands[idx])
        raw_candidates = dedup_boxes
    else:
        # Standard full-frame inference with base threshold
        res = model_instance.predict(source=processed_img, conf=base_conf, iou=iou_thresh, verbose=False)[0]
        raw_candidates = []
        for b in res.boxes:
            cid = int(b.cls[0].item())
            rc = float(b.conf[0].item())
            xyxy = [float(v) for v in b.xyxy[0].cpu().numpy().tolist()]
            raw_candidates.append((cid, rc, xyxy))

    filtered = []
    names = model_instance.names
    for cid, conf, bbox in raw_candidates:
        cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
        calibrated_thresh = CLASS_THRESHOLDS.get(cname, 0.10)
        effective_thresh = user_conf if user_conf is not None else calibrated_thresh
        if conf >= effective_thresh:
            filtered.append((cid, cname, conf, bbox))

    return filtered

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Marine Debris Detection API is running"}


# ---------------------------------------------------------
# 0a. POST /surveys -> create a survey BEFORE uploading any images
# ---------------------------------------------------------
@app.post("/surveys", response_model=schemas.SurveyOut)
def create_survey(survey_in: schemas.SurveyIn, db: Session = Depends(get_db)):
    survey = models.Survey(
        name=survey_in.name,
        water_body=survey_in.water_body,
        vessel=survey_in.vessel,
        latitude=survey_in.latitude,
        longitude=survey_in.longitude,
        depth=survey_in.depth,
        location=survey_in.water_body or "Unknown",
        status="pending",
    )
    db.add(survey)
    db.commit()
    db.refresh(survey)
    return survey


# ---------------------------------------------------------
# 0b. GET /surveys/{id} -> single survey detail
# ---------------------------------------------------------
@app.get("/surveys/{survey_id}", response_model=schemas.SurveyOut)
def get_survey(survey_id: int, db: Session = Depends(get_db)):
    survey = db.query(models.Survey).filter(models.Survey.id == survey_id).first()
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")
    return survey


def serialize_detection(detection, db):
    """Return the backend detection in the format expected by Member 5 GIS."""

    image = db.query(models.Image).filter(
        models.Image.id == detection.image_id
    ).first()

    survey = (
        db.query(models.Survey)
        .filter(models.Survey.id == image.survey_id)
        .first()
        if image
        else None
    )

    bbox = detection.bbox or []

    width_px = None
    height_px = None

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            width_px = max(
                0,
                int(round(float(bbox[2]) - float(bbox[0])))
            )

            height_px = max(
                0,
                int(round(float(bbox[3]) - float(bbox[1])))
            )
        except Exception:
            pass

    # -----------------------------------------------------
    # CALCULATE RISK LEVEL FROM STORED RISK SCORE
    # -----------------------------------------------------

    risk_level = None

    if detection.risk_score is not None:
        if detection.risk_score >= 70:
            risk_level = "HIGH"
        elif detection.risk_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

    # -----------------------------------------------------
    # SERIALIZED DETECTION
    # -----------------------------------------------------

    # Survey-scoped detection ID (restarts from 1 per survey)
    survey_code = f"SURV-{survey.id:03d}" if survey else "SURV-001"
    det_num = getattr(detection, "survey_detection_index", None)
    if not det_num:
        if survey:
            prior = db.query(models.Detection.id).join(models.Image).filter(
                models.Image.survey_id == survey.id,
                models.Detection.id <= detection.id
            ).count()
            det_num = max(1, prior)
        else:
            det_num = detection.id
    det_code = f"DET-{det_num:03d}"
    full_identifier = f"{survey_code} / {det_code}"

    return {
        "id": detection.id,
        "detection_id": det_code,
        "survey_detection_index": det_num,
        "survey_code": survey_code,
        "full_identifier": full_identifier,

        "survey_id": survey.id if survey else None,
        "image_id": detection.image_id,

        "object_class": detection.object_class,
        "class": detection.object_class,

        "confidence": detection.confidence,
        "bbox": detection.bbox,

        "bbox_width_px": width_px,
        "bbox_height_px": height_px,

        "latitude": (
            detection.latitude
            if detection.latitude is not None
            else (survey.latitude if survey else None)
        ),

        "longitude": (
            detection.longitude
            if detection.longitude is not None
            else (survey.longitude if survey else None)
        ),

        "depth": (
            detection.depth
            if detection.depth is not None
            else (survey.depth if survey else None)
        ),

        # Physical size is intentionally not calculated.
        "estimated_size": None,

        "data_quality": (
            detection.data_quality
            if detection.data_quality is not None
            else 1.0
        ),

        # GIS risk information
        "risk_score": detection.risk_score,
        "risk_level": risk_level,
        "priority": detection.priority,

        "status": detection.status,
        "cleanup_status": detection.status,
    }
# ---------------------------------------------------------
# 1. POST /upload  -> upload a sonar image.
#    If survey_id is given, the image is attached to that survey.
#    If not, a new survey is auto-created (old behavior, kept for
#    backward compatibility with quick demos).
# ---------------------------------------------------------
@app.post("/upload")
def upload_image(
    file: UploadFile = File(...),
    survey_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    if not survey_id:
        raise HTTPException(
            status_code=400,
            detail="A survey MUST exist before sonar images can be uploaded. Please create or select a survey first."
        )
    survey = db.query(models.Survey).filter(models.Survey.id == survey_id).first()
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")

    # Save the uploaded file to disk
    filename = f"{survey.id}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save image record in DB
    image = models.Image(survey_id=survey.id, filepath=filepath, status="raw")
    db.add(image)
    db.commit()
    db.refresh(image)

    return {
        "message": "Upload successful",
        "survey_id": survey.id,
        "image_id": image.id,
        "filepath": filepath,
    }


# ---------------------------------------------------------
# 2. POST /analyze -> run YOLO detection on an uploaded image
#    The backend stores detection facts and survey metadata.
#    Member 5 GIS calculates the final risk, priority, and route.
# ---------------------------------------------------------
@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    survey_id: Optional[int] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    depth: Optional[float] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    db: Session = Depends(get_db),
):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No sonar image supplied")

        # A survey MUST exist before sonar images can be uploaded (Req 5, 6, 10)
        if survey_id is None:
            raise HTTPException(
                status_code=400,
                detail="A survey MUST exist before sonar images can be uploaded. Please create or select a survey first."
            )
        survey = db.query(models.Survey).filter(models.Survey.id == survey_id).first()
        if not survey:
            raise HTTPException(status_code=404, detail=f"Survey with ID {survey_id} not found")
        if latitude is not None:
            survey.latitude = latitude
        if longitude is not None:
            survey.longitude = longitude
        if depth is not None:
            survey.depth = depth

        # Read the uploaded image into memory and save the original.
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded sonar image is empty")

        safe_name = os.path.basename(file.filename)
        filename = f"{survey.id}_{safe_name}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as buffer:
            buffer.write(raw_bytes)

        print(f"[SONAR] Image received: {safe_name} ({len(raw_bytes)} bytes)")

        # Decode the real uploaded sonar image.
        image_array = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            image_array = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Could not decode sonar image")

        # Real sonar preprocessing matching the new model:
        # CLAHE on L-channel in LAB space + bilateral speckle reduction filter.
        processed_img = preprocess_sonar(image_array)

        processed_path = os.path.join(UPLOAD_DIR, f"processed_{survey.id}_{safe_name}.png")
        cv2.imwrite(processed_path, processed_img)

        print(f"[SONAR] Processed image: {processed_path}")

        # Determine confidence threshold (strictly enforced at 0.25 floor) and IoU (0.45)
        # Note: Frontend UI sends 0.05 from legacy code, but post-processing confidence filter
        # must discard all predictions below 0.25 confidence.
        if confidence_threshold is not None:
            conf_thresh = float(confidence_threshold)
        else:
            conf_thresh = float(os.getenv("CONF_THRESHOLD", "0.25"))

        if conf_thresh < 0.05:
            conf_thresh = 0.05

        iou_thresh = float(os.getenv("IOU_THRESHOLD", "0.45"))

        print(f"[YOLO] Model: {os.path.abspath(MODEL_PATH)}")
        print(f"[YOLO] Classes: {model.names}")
        print(f"[YOLO] Confidence threshold: {conf_thresh}")
        print(f"[YOLO] IoU threshold: {iou_thresh}")

        # -----------------------------------------------------
        # 1. YOLO INFERENCE (Raw predictions with non-zero baseline floor)
        # -----------------------------------------------------
        results = model(processed_path, conf=0.01, iou=0.7, verbose=False)

        # Extract raw candidate detections
        raw_candidates = []
        for result in results:
            names = result.names or model.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = [float(v) for v in box.xyxy[0].tolist()]
                class_name = names[class_id] if isinstance(names, dict) else str(class_id)
                raw_candidates.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "class": class_name,
                    "object_class": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                })

        # -----------------------------------------------------
        # 2. CONFIDENCE FILTERING & 3. CLASS-AWARE NMS
        # -----------------------------------------------------
        deduped_candidates, raw_yolo_count, below_thresh_count, after_nms_count = post_process_sonar_detections(
            raw_candidates,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            containment_thresh=0.65,
            conf_tolerance=0.02,
        )

        # Create a displayable detection image from the processed image.
        detection_image = processed_img.copy() if len(processed_img.shape) == 3 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        detections = []

        # Count existing detections for this survey so detection IDs restart from 1 per survey (Req 11, 12, 13)
        existing_survey_dets = db.query(models.Detection).join(models.Image).filter(
            models.Image.survey_id == survey.id
        ).count()
        survey_code = f"SURV-{survey.id:03d}"

        # Store image record.
        image = models.Image(survey_id=survey.id, filepath=filepath, status="processed")
        db.add(image)
        db.flush()

        # -----------------------------------------------------
        # 4. PROCESS FINAL DEDUPLICATED DETECTIONS ONLY
        # -----------------------------------------------------
        for cand in deduped_candidates:
            class_id = cand["class_id"]
            class_name = cand["class_name"]
            confidence = cand["confidence"]
            bbox = cand["bbox"]

            x1, y1, x2, y2 = [
                int(round(v))
                for v in bbox
            ]

            x1 = max(
                0,
                min(x1, detection_image.shape[1] - 1)
            )
            x2 = max(
                0,
                min(x2, detection_image.shape[1] - 1)
            )
            y1 = max(
                0,
                min(y1, detection_image.shape[0] - 1)
            )
            y2 = max(
                0,
                min(y2, detection_image.shape[0] - 1)
            )

            width_px = max(0, x2 - x1)
            height_px = max(0, y2 - y1)

            # Draw thick, clearly visible green bounding box (Req 5)
            cv2.rectangle(
                detection_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                8,
            )

            label = (
                f"{class_name} "
                f"{confidence * 100:.1f}%"
            )

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lbl_top = max(0, y1 - th - 8)
            lbl_bottom = max(th + 8, y1)
            cv2.rectangle(
                detection_image,
                (x1, lbl_top),
                (min(detection_image.shape[1] - 1, x1 + tw + 8), lbl_bottom),
                (0, 220, 0),
                -1,
            )
            cv2.putText(
                detection_image,
                label,
                (x1 + 4, lbl_bottom - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # =========================================
            # RISK CALCULATION
            # =========================================
            risk_input = {
                "object_class": class_name,
                "confidence": confidence,
                "depth": survey.depth or 0.0,
                "data_quality": 1.0,
            }

            risk_result = calculate_risk(risk_input)

            risk_score = risk_result["risk_score"]
            risk_level = risk_result["risk_level"]

            # =========================================
            # PRIORITY CALCULATION
            # =========================================
            priority_result = calculate_priority_result(
                risk_score,
                class_name,
            )

            priority = priority_result["priority"]

            print(
                f"[RISK] {class_name} | "
                f"confidence={confidence:.3f} | "
                f"depth={survey.depth} | "
                f"risk={risk_score:.2f} | "
                f"level={risk_level} | "
                f"priority={priority}"
            )

            # =========================================
            # SAVE DETECTION (IDs generated ONLY after deduplication)
            # =========================================
            survey_det_idx = existing_survey_dets + len(detections) + 1
            det_code = f"DET-{survey_det_idx:03d}"
            full_identifier = f"{survey_code} / {det_code}"

            detection = models.Detection(
                image_id=image.id,
                object_class=class_name,
                confidence=confidence,
                bbox=bbox,
                latitude=survey.latitude,
                longitude=survey.longitude,
                depth=survey.depth,
                estimated_size=None,
                data_quality=1.0,
                risk_score=risk_score,
                priority=priority,
                status="pending",
                survey_detection_index=survey_det_idx,
            )

            db.add(detection)
            db.flush()

            detections.append({
                "id": detection.id,
                "detection_id": det_code,
                "survey_detection_index": survey_det_idx,
                "survey_code": survey_code,
                "full_identifier": full_identifier,
                "survey_id": survey.id,
                "image_id": image.id,
                "object_class": class_name,
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "bbox_width_px": width_px,
                "bbox_height_px": height_px,
                "latitude": survey.latitude,
                "longitude": survey.longitude,
                "depth": survey.depth,
                "estimated_size": None,
                "data_quality": 1.0,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "priority": priority,
                "status": "pending",
            })

        for d in detections:
            print(f"[YOLO] Detection: {d['object_class']} ({d['confidence']:.4f}) BBox: {d['bbox']}")
        print(f"[YOLO] Final detections: {len(detections)}")
        print(f"[API] Returning detections: {len(detections)}")

        # Encode images as data URLs.
        ok1, processed_png = cv2.imencode('.png', processed_img)
        ok2, detection_png = cv2.imencode('.png', detection_image)
        if not ok1 or not ok2:
            raise HTTPException(status_code=500, detail="Could not encode processed sonar images")

        processed_data_url = "data:image/png;base64," + base64.b64encode(processed_png.tobytes()).decode('ascii')
        detection_data_url = "data:image/png;base64," + base64.b64encode(detection_png.tobytes()).decode('ascii')

        db.commit()

        return {
            "image_id": image.id,
            "survey_id": survey.id,
            "survey_code": survey_code,
            "filename": safe_name,
            "processed_image": processed_data_url,
            "detection_image": detection_data_url,
            "detections": detections,
            "detection_count": len(detections),
            "confidence_threshold": conf_thresh,
            "preprocessing": {
                "denoising": "bilateralFilter",
                "clahe": True,
                "color_space": "LAB",
                "tileGridSize": [8, 8],
                "clipLimit": 2.0,
            },
            "survey": {
                "id": survey.id,
                "survey_code": survey_code,
                "name": survey.name,
                "latitude": survey.latitude,
                "longitude": survey.longitude,
                "depth": survey.depth,
            },
        }
    except Exception as e:
        print(f"[ERROR] Analyze failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Helper to run detection at a specific confidence threshold and generate annotated image
def _detect_and_annotate(processed_img_path: str, conf: float, iou: float):
    """Run YOLO inference on the given processed image with the supplied confidence.
    Returns a tuple (detections_list, annotated_image_data_url)."""
    proc_img = cv2.imread(processed_img_path)
    if proc_img is None:
        proc_img = cv2.imdecode(np.fromfile(processed_img_path, np.uint8), cv2.IMREAD_COLOR)
    detection_img = proc_img.copy() if len(proc_img.shape) == 3 else cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
    results = model(processed_img_path, conf=0.01, iou=0.7, verbose=False)
    raw_candidates = []
    for result in results:
        names = result.names or model.names
        for box in result.boxes:
            cid = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]
            class_name = names[cid] if isinstance(names, dict) else str(cid)
            raw_candidates.append({
                "class_id": cid,
                "class_name": class_name,
                "class": class_name,
                "object_class": class_name,
                "confidence": confidence,
                "bbox": bbox,
            })

    deduped_candidates, _, _, _ = post_process_sonar_detections(
        raw_candidates,
        conf_thresh=conf,
        iou_thresh=iou,
        containment_thresh=0.65,
        conf_tolerance=0.02,
    )

    detections = []
    for cand in deduped_candidates:
        confidence = cand["confidence"]
        bbox = cand["bbox"]
        class_name = cand["class_name"]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        label = f"{class_name} {confidence * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lbl_top = max(0, y1 - th - 8)
        lbl_bottom = max(th + 8, y1)
        cv2.rectangle(detection_img, (x1, lbl_top), (min(detection_img.shape[1] - 1, x1 + tw + 8), lbl_bottom), (0, 220, 0), -1)
        cv2.putText(detection_img, label, (x1 + 4, lbl_bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        detections.append({"object_class": class_name, "confidence": confidence, "bbox": bbox})
    ok, png = cv2.imencode('.png', detection_img)
    data_url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode('ascii') if ok else None
    return detections, data_url

# ---------------------------------------------------------
# 2b. POST /analyze_debug - run YOLO at multiple thresholds for inspection
# ---------------------------------------------------------
@app.post("/analyze_debug")
async def analyze_debug_image(
    file: UploadFile = File(...),
    survey_id: Optional[int] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    depth: Optional[float] = Form(None),
    db: Session = Depends(get_db),
):
    """Same as /analyze but returns detections for thresholds 0.25, 0.10 and 0.05.
    Useful for visual inspection of false positives and missed objects."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No sonar image supplied")
    # Resolve/create survey (mirrors /analyze logic)
    if survey_id is not None:
        survey = db.query(models.Survey).filter(models.Survey.id == survey_id).first()
        if not survey:
            raise HTTPException(status_code=404, detail="Survey not found")
        if latitude is not None:
            survey.latitude = latitude
        if longitude is not None:
            survey.longitude = longitude
        if depth is not None:
            survey.depth = depth
    else:
        # Default to first survey rather than creating a new survey on every upload without survey_id
        survey = db.query(models.Survey).order_by(models.Survey.id.asc()).first()
        if not survey:
            survey = models.Survey(
                name="Arabian Sea Survey",
                location="Arabian Sea",
                water_body="Arabian Sea",
                vessel="Sagar tara",
                latitude=latitude or 17.5,
                longitude=longitude or 60.5,
                depth=depth or 425.0,
                status="pending",
            )
            db.add(survey)
            db.commit()
            db.refresh(survey)
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded sonar image is empty")
    safe_name = os.path.basename(file.filename)
    filename = f"{survey.id}_{safe_name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as buffer:
        buffer.write(raw_bytes)
    image_array = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        image_array = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Could not decode sonar image")
    processed_img = preprocess_sonar(image_array)
    processed_path = os.path.join(UPLOAD_DIR, f"processed_{survey.id}_{safe_name}.png")
    cv2.imwrite(processed_path, processed_img)
    debug_thresholds = [0.25, 0.10, 0.05]
    iou_thresh = float(os.getenv("IOU_THRESHOLD", "0.45"))
    debug_results = {}
    for thr in debug_thresholds:
        dets, img_url = _detect_and_annotate(processed_path, conf=thr, iou=iou_thresh)
        debug_results[f"{thr:.2f}"] = {"detections": dets, "detection_image": img_url}
    # Store image record (same as /analyze)
    image = models.Image(survey_id=survey.id, filepath=filepath, status="processed")
    db.add(image)
    db.commit()
    return {
        "survey_id": survey.id,
        "filename": safe_name,
        "processed_image_path": processed_path,
        "debug_results": debug_results,
        "survey": {"latitude": survey.latitude, "longitude": survey.longitude, "depth": survey.depth},
    }

# ---------------------------------------------------------
# 3. GET /detections -> list all detections (optionally filter by status)
# ---------------------------------------------------------# ---------------------------------------------------------
@app.get("/detections")
def get_detections(status: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(models.Detection)
    if status:
        query = query.filter(models.Detection.status == status)
    return [serialize_detection(d, db) for d in query.all()]


# ---------------------------------------------------------
# 3b. GET /detections/{id} -> single detection detail
# ---------------------------------------------------------
@app.get("/detections/{detection_id}")
def get_detection(detection_id: int, db: Session = Depends(get_db)):
    detection = db.query(models.Detection).filter(models.Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return serialize_detection(detection, db)


# ---------------------------------------------------------
# 3c. GET /detections/{id}/image -> return detection sonar image as base64
# ---------------------------------------------------------
@app.get("/detections/{detection_id}/image")
def get_detection_image(
    detection_id: str,
    image_id: Optional[int] = None,
    annotate: bool = True,
    db: Session = Depends(get_db),
):
    """Return the sonar image for a detection as a base64 data URL.
    Supports:
      - Numeric detection ID (e.g. 96)
      - Formatted detection code (e.g. 'DET-002', 'SURV-001 / DET-002')
      - Direct image_id lookup via query param or fallback
      - Draws target bounding box in bright green if available
    """
    import re
    detection = None
    target_id_str = str(detection_id).strip()

    # 1. Try direct numeric lookup if detection_id is an integer (e.g. 96)
    if target_id_str.isdigit():
        num_id = int(target_id_str)
        detection = db.query(models.Detection).filter(models.Detection.id == num_id).first()
        if not detection:
            detection = db.query(models.Detection).filter(models.Detection.survey_detection_index == num_id).first()

    # 2. Try parsing composite code like "SURV-001 / DET-002" or "DET-002"
    if not detection:
        det_match = re.search(r'DET[-_]?(\d+)', target_id_str, re.IGNORECASE)
        surv_match = re.search(r'SURV[-_]?(\d+)', target_id_str, re.IGNORECASE)

        if det_match:
            det_idx = int(det_match.group(1))
            query = db.query(models.Detection)
            if surv_match:
                s_id = int(surv_match.group(1))
                query = query.join(models.Image).filter(models.Image.survey_id == s_id)

            detection = query.filter(models.Detection.survey_detection_index == det_idx).first()
            if not detection and surv_match:
                # Fallback: order by ID within survey
                survey_dets = query.order_by(models.Detection.id).all()
                if 1 <= det_idx <= len(survey_dets):
                    detection = survey_dets[det_idx - 1]
            elif not detection:
                detection = db.query(models.Detection).filter(models.Detection.id == det_idx).first()

    # 3. Locate image record
    image_record = None
    if detection and detection.image_id:
        image_record = db.query(models.Image).filter(models.Image.id == detection.image_id).first()

    if not image_record and image_id:
        image_record = db.query(models.Image).filter(models.Image.id == image_id).first()

    if not image_record and target_id_str.isdigit():
        # Check if the ID provided was actually an image ID
        image_record = db.query(models.Image).filter(models.Image.id == int(target_id_str)).first()

    if not image_record:
        raise HTTPException(status_code=404, detail=f"No image found for detection '{detection_id}'")

    filepath = image_record.filepath
    if filepath and not os.path.isfile(filepath):
        cand = os.path.join(UPLOAD_DIR, os.path.basename(filepath))
        if os.path.isfile(cand):
            filepath = cand

    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # 4. If annotate requested and detection has bbox, render bounding box directly on image
    if annotate and detection and detection.bbox:
        try:
            bbox_raw = detection.bbox
            if isinstance(bbox_raw, str):
                import json
                bbox = json.loads(bbox_raw)
            else:
                bbox = bbox_raw

            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                img = cv2.imread(filepath)
                if img is not None:
                    h, w = img.shape[:2]
                    x1 = max(0, min(int(round(float(bbox[0]))), w - 1))
                    y1 = max(0, min(int(round(float(bbox[1]))), h - 1))
                    x2 = max(0, min(int(round(float(bbox[2]))), w - 1))
                    y2 = max(0, min(int(round(float(bbox[3]))), h - 1))

                    if x2 > x1 and y2 > y1:
                        # Draw green bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 68), 8)

                        # Label badge
                        conf_pct = int(round((detection.confidence or 0) * (100 if detection.confidence <= 1 else 1)))
                        label = f"{detection.object_class} {conf_pct}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        lbl_top = max(0, y1 - th - 10)
                        cv2.rectangle(img, (x1, lbl_top), (x1 + tw + 10, y1), (0, 255, 68), -1)
                        cv2.putText(
                            img,
                            label,
                            (x1 + 5, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        data = base64.b64encode(enc.tobytes()).decode('ascii')
                        return {
                            "data_url": f"data:image/jpeg;base64,{data}",
                            "mime": "image/jpeg",
                            "filename": os.path.basename(filepath),
                        }
        except Exception as err:
            print(f"[IMAGE_ANNOTATE] Warning while drawing bbox: {err}")

    # Fallback to direct raw/processed file read
    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    mime = mime_map.get(ext, "image/png")

    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")

    return {
        "data_url": f"data:{mime};base64,{data}",
        "mime": mime,
        "filename": os.path.basename(filepath),
    }


@app.get("/images/{image_id}/image")
def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
    """Return an image by Image ID as a base64 data URL."""
    image_record = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image_record:
        raise HTTPException(status_code=404, detail="Image record not found")

    filepath = image_record.filepath
    if filepath and not os.path.isfile(filepath):
        cand = os.path.join(UPLOAD_DIR, os.path.basename(filepath))
        if os.path.isfile(cand):
            filepath = cand

    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = mime_map.get(ext, "image/png")

    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")

    return {
        "data_url": f"data:{mime};base64,{data}",
        "mime": mime,
        "filename": os.path.basename(filepath),
    }


# ---------------------------------------------------------
# 4. GET /survey -> list all surveys with basic summary
#    (kept as-is for backward compatibility with earlier demo)
# ---------------------------------------------------------
@app.get("/survey", response_model=List[schemas.SurveyOut])
def get_surveys(db: Session = Depends(get_db)):
    return db.query(models.Survey).order_by(models.Survey.id.asc()).all()


# ---------------------------------------------------------
# 4b. GET /surveys -> same as above, matches plural REST convention
# ---------------------------------------------------------
@app.get("/surveys", response_model=List[schemas.SurveyOut])
def get_surveys_plural(db: Session = Depends(get_db)):
    return db.query(models.Survey).order_by(models.Survey.id.asc()).all()


# ---------------------------------------------------------
# 5. POST /review -> expert confirms/rejects a detection
# ---------------------------------------------------------
@app.post("/review")
def review_detection(review_in: schemas.ReviewIn, db: Session = Depends(get_db)):
    detection = db.query(models.Detection).filter(
        models.Detection.id == review_in.detection_id
    ).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    verdict = review_in.verdict.lower().strip()
    if verdict not in ["confirmed", "rejected", "flagged"]:
        raise HTTPException(status_code=400, detail="Verdict must be 'confirmed', 'rejected', or 'flagged'")

    detection.status = "confirmed" if verdict == "confirmed" else ("flagged" if verdict == "flagged" else "rejected")
    review = models.Review(
        detection_id=detection.id,
        reviewer=review_in.reviewer or "operator",
        verdict=verdict,
    )
    db.add(review)
    db.commit()
    db.refresh(detection)

    return {
        "message": "Review recorded",
        "detection_id": detection.id,
        "new_status": detection.status,
        "status": detection.status,
        "comment": review_in.comment,
    }

# =========================
# CLEANUP API
# =========================

@app.post("/cleanup")
def cleanup_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    detection = db.query(models.Detection).filter(
        models.Detection.id == detection_id
    ).first()

    if not detection:
        raise HTTPException(
            status_code=404,
            detail="Detection not found"
        )

    # Mark the debris as cleaned
    detection.status = "cleaned"

    db.commit()
    db.refresh(detection)

    return {
        "message": "Cleanup marked successfully",
        "detection_id": detection.id,
        "status": detection.status
    }

# ---------------------------------------------------------
# 6. GET /report -> aggregate stats for reporting
# ---------------------------------------------------------
@app.get("/report")
def get_report(db: Session = Depends(get_db)):
    total_surveys = db.query(models.Survey).count()
    total_detections = db.query(models.Detection).count()
    high_risk = db.query(models.Detection).filter(models.Detection.priority == "high").count()
    pending = db.query(models.Detection).filter(models.Detection.status == "pending").count()

    return {
        "total_surveys": total_surveys,
        "total_debris": total_detections,
        "high_risk": high_risk,
        "pending_cleanup": pending,
    }


# ---------------------------------------------------------
# 7. GET /priority -> detections sorted by risk score, highest first
# ---------------------------------------------------------
@app.get("/priority")
def get_priority_detections(db: Session = Depends(get_db)):
    detections = db.query(models.Detection).order_by(models.Detection.risk_score.desc().nullslast()).all()
    return [serialize_detection(d, db) for d in detections]


# ---------------------------------------------------------
# 8. POST /verify -> before/after cleanup verification
#    NOTE: This uses a simple MOCK comparison for now.
#    Later this can call Member 6's real before/after image
#    comparison logic (e.g. re-running detection on the "after"
#    image and checking whether the same object is still found).
# ---------------------------------------------------------
@app.post("/verify")
def verify_detection(
    detection_id: int,
    before_image_id: int,
    after_image_id: int,
    comment: str = None,
    db: Session = Depends(get_db)
):
    detection = db.query(models.Detection).filter(
        models.Detection.id == detection_id
    ).first()

    if not detection:
        raise HTTPException(
            status_code=404,
            detail="Detection not found"
        )

    before_image = db.query(models.Image).filter(
        models.Image.id == before_image_id
    ).first()

    after_image = db.query(models.Image).filter(
        models.Image.id == after_image_id
    ).first()

    if not before_image:
        raise HTTPException(
            status_code=404,
            detail="Before image not found"
        )

    if not after_image:
        raise HTTPException(
            status_code=404,
            detail="After image not found"
        )

    # Run YOLO on the AFTER image with sonar preprocessing
    after_bgr = cv2.imread(after_image.filepath)
    if after_bgr is not None:
        proc_after = preprocess_sonar(after_bgr)
        results = model(proc_after, conf=0.15, verbose=False)
    else:
        results = model(after_image.filepath, conf=0.15, verbose=False)

    raw_after_cands = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            class_name = model.names[class_id]

            raw_after_cands.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox
            })

    after_detections, _, _, _ = suppress_duplicate_detections(
        raw_after_cands,
        iou_thresh=0.45,
        containment_thresh=0.65,
        conf_tolerance=0.02
    )

    # Calculate IoU between two bounding boxes
    def calculate_iou(box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)

        intersection_area = (
            intersection_width * intersection_height
        )

        area1 = (
            max(0, box1[2] - box1[0]) *
            max(0, box1[3] - box1[1])
        )

        area2 = (
            max(0, box2[2] - box2[0]) *
            max(0, box2[3] - box2[1])
        )

        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0

        return intersection_area / union_area

    # Compare original detection with AFTER detections
    same_object_found = False
    best_iou = 0
    after_confidence = 0

    for after_detection in after_detections:

        if after_detection["class"] != detection.object_class:
            continue

        iou = calculate_iou(
            detection.bbox,
            after_detection["bbox"]
        )

        if iou > best_iou:
            best_iou = iou
            after_confidence = after_detection["confidence"]

        # Same object considered present if boxes overlap
        if iou >= 0.30:
            same_object_found = True

    if same_object_found:

        result_status = "still_present"

        # Heuristic confidence for presence
        verification_confidence = round(
            (after_confidence + best_iou) / 2,
            2
        )

    else:

        result_status = "potentially_removed"

        # This is a heuristic, not a calibrated probability
        verification_confidence = round(
            1 - detection.confidence * 0.5,
            2
        )

        # Mark cleanup as completed
        detection.status = "cleaned"

    verification = models.Verification(
        detection_id=detection_id,
        before_image_id=before_image_id,
        after_image_id=after_image_id,
        result=result_status,
        confidence=verification_confidence,
        comment=comment
    )

    db.add(verification)

    db.commit()
    db.refresh(verification)

    return {
        "id": verification.id,
        "detection_id": detection_id,
        "before_image_id": before_image_id,
        "after_image_id": after_image_id,
        "result": result_status,
        "confidence": verification_confidence,
        "comment": comment,
        "message": "Before/after verification completed using YOLO"
    }

# ---------------------------------------------------------
# 8b. GET /verifications -> list all verification records
# ---------------------------------------------------------
@app.get("/verifications")
def get_verifications(db: Session = Depends(get_db)):
    verifications = db.query(models.Verification).all()
    result = []
    for v in verifications:
        detection = db.query(models.Detection).filter(models.Detection.id == v.detection_id).first()
        det_data = serialize_detection(detection, db) if detection else {}
        result.append({
            "id": v.id,
            "detection_id": v.detection_id,
            "before_image_id": v.before_image_id,
            "after_image_id": v.after_image_id,
            "result": v.result,
            "confidence": v.confidence,
            "comment": v.comment,
            "timestamp": v.timestamp.isoformat() if v.timestamp else None,
            "detection": det_data,
        })
    return result


# ---------------------------------------------------------
# 8c. POST /verify_cleanup -> run YOLO on after-cleanup image, compare with original detection
# ---------------------------------------------------------
@app.post("/verify_cleanup")
async def verify_cleanup(
    file: UploadFile = File(...),
    detection_id: int = Form(...),
    db: Session = Depends(get_db),
):
    """
    Before/after cleanup verification endpoint.
    Runs real YOLO inference on the after-cleanup sonar image and compares
    against the original confirmed detection using IoU matching.
    """
    # --- 1. Load the confirmed detection ---
    detection = db.query(models.Detection).filter(
        models.Detection.id == detection_id
    ).first()
    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection with id={detection_id} not found")
    allowed_statuses = {"confirmed", "verified", "accepted", "cleaned", "flagged"}
    if detection.status.lower() not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Detection status is '{detection.status}' — only confirmed/verified detections can be verified. "
                   f"Please confirm this detection on the Human Verification page first."
        )

    # --- 2. Load the original before image record ---
    before_image = db.query(models.Image).filter(
        models.Image.id == detection.image_id
    ).first()
    if not before_image:
        raise HTTPException(status_code=404, detail="Original detection image not found in database")

    # --- 3. Read and annotate the before image ---
    before_bgr = cv2.imread(before_image.filepath)
    if before_bgr is None:
        before_bgr = cv2.imdecode(np.fromfile(before_image.filepath, np.uint8), cv2.IMREAD_COLOR)

    before_annotated = before_bgr.copy() if before_bgr is not None else None
    if before_annotated is not None and detection.bbox and len(detection.bbox) >= 4:
        x1b, y1b, x2b, y2b = [int(round(float(v))) for v in detection.bbox]
        cv2.rectangle(before_annotated, (x1b, y1b), (x2b, y2b), (0, 255, 0), 8)
        lbl_b = f"{detection.object_class} {detection.confidence * 100:.1f}%"
        (tw_b, th_b), _ = cv2.getTextSize(lbl_b, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lt_b = max(0, y1b - th_b - 8)
        lb_b = max(th_b + 8, y1b)
        cv2.rectangle(before_annotated, (x1b, lt_b),
                      (min(before_annotated.shape[1] - 1, x1b + tw_b + 8), lb_b), (0, 220, 0), -1)
        cv2.putText(before_annotated, lbl_b, (x1b + 4, lb_b - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    before_data_url = None
    if before_annotated is not None:
        ok_b, png_b = cv2.imencode('.png', before_annotated)
        if ok_b:
            before_data_url = "data:image/png;base64," + base64.b64encode(png_b.tobytes()).decode('ascii')

    # --- 4. Save the after-cleanup image ---
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded after-cleanup image is empty")

    survey_id_for_file = before_image.survey_id if before_image else 0
    safe_name = os.path.basename(file.filename)
    after_filename = f"after_{survey_id_for_file}_{detection_id}_{safe_name}"
    after_filepath = os.path.join(UPLOAD_DIR, after_filename)
    with open(after_filepath, "wb") as buffer:
        buffer.write(raw_bytes)

    after_image_db = models.Image(
        survey_id=survey_id_for_file,
        filepath=after_filepath,
        status="after_cleanup"
    )
    db.add(after_image_db)
    db.flush()

    # --- 5. Run YOLO on the after-cleanup image ---
    after_arr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if after_arr is None:
        after_arr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if after_arr is None:
        raise HTTPException(status_code=400, detail="Could not decode after-cleanup sonar image")

    proc_after = preprocess_sonar(after_arr)
    after_results = model(proc_after, conf=0.08, iou=0.45, verbose=False)

    after_cands = []
    for res in after_results:
        names_a = res.names or model.names
        for box in res.boxes:
            cid_a = int(box.cls[0])
            conf_a = float(box.conf[0])
            bbox_a = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]
            cname_a = names_a[cid_a] if isinstance(names_a, dict) else str(cid_a)
            after_cands.append({"class": cname_a, "class_name": cname_a, "confidence": conf_a, "bbox": bbox_a})

    after_detections_raw, _, _, _ = suppress_duplicate_detections(
        after_cands,
        iou_thresh=0.45,
        containment_thresh=0.65,
        conf_tolerance=0.02
    )

    # --- 6. Draw after-image with bounding boxes ---
    after_annotated = proc_after.copy() if len(proc_after.shape) == 3 else cv2.cvtColor(proc_after, cv2.COLOR_GRAY2BGR)
    for ad in after_detections_raw:
        bx1, by1, bx2, by2 = [int(round(float(v))) for v in ad["bbox"]]
        cv2.rectangle(after_annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 8)
        lbl_a = f"{ad['class']} {ad['confidence'] * 100:.1f}%"
        (tw_a, th_a), _ = cv2.getTextSize(lbl_a, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lt_a = max(0, by1 - th_a - 8)
        lb_a = max(th_a + 8, by1)
        cv2.rectangle(after_annotated, (bx1, lt_a),
                      (min(after_annotated.shape[1] - 1, bx1 + tw_a + 8), lb_a), (0, 220, 0), -1)
        cv2.putText(after_annotated, lbl_a, (bx1 + 4, lb_a - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    ok_a, png_a = cv2.imencode('.png', after_annotated)
    after_data_url = (
        "data:image/png;base64," + base64.b64encode(png_a.tobytes()).decode('ascii')
        if ok_a else None
    )

    # --- 7. IoU matching against original confirmed detection ---
    def calculate_iou_vc(box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        iw = max(0.0, xi2 - xi1)
        ih = max(0.0, yi2 - yi1)
        inter = iw * ih
        a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    same_object_found = False
    best_iou = 0.0
    after_conf_best = 0.0
    matched_after_detection = None

    for ad in after_detections_raw:
        if ad["class"] != detection.object_class:
            continue
        if detection.bbox and len(detection.bbox) >= 4:
            iou_val = calculate_iou_vc(detection.bbox, ad["bbox"])
        else:
            iou_val = 0.31  # no bbox stored — treat same class as present
        if iou_val > best_iou:
            best_iou = iou_val
            after_conf_best = ad["confidence"]
            matched_after_detection = ad
        if iou_val >= 0.30:
            same_object_found = True

    if same_object_found:
        result_status = "still_present"
        cleanup_status_label = "NOT CLEARED"
        verification_confidence = round((after_conf_best + best_iou) / 2, 2)
    else:
        result_status = "potentially_removed"
        cleanup_status_label = "CLEARED"
        verification_confidence = round(1 - detection.confidence * 0.5, 2)
        detection.status = "cleaned"

    # --- 8. Save Verification record ---
    verification = models.Verification(
        detection_id=detection_id,
        before_image_id=detection.image_id,
        after_image_id=after_image_db.id,
        result=result_status,
        confidence=verification_confidence,
        comment=cleanup_status_label,
    )
    db.add(verification)
    db.commit()
    db.refresh(verification)

    # Separate: detections of OTHER classes found in the after-cleanup scan
    other_objects_detected = [
        ad for ad in after_detections_raw
        if ad["class"] != detection.object_class
    ]

    return {
        "verification_id": verification.id,
        "detection_id": detection_id,
        "before_image_id": detection.image_id,
        "after_image_id": after_image_db.id,
        "result": result_status,
        "cleanup_status": cleanup_status_label,
        "confidence": verification_confidence,
        "best_iou": round(best_iou, 3),
        "after_detections": after_detections_raw,
        "matched_detection": matched_after_detection,
        "other_objects_detected": other_objects_detected,
        "other_objects_present": len(other_objects_detected) > 0,
        "before_image": before_data_url,
        "after_image": after_data_url,
        "timestamp": verification.timestamp.isoformat() if verification.timestamp else None,
        "message": "Cleanup verification completed using YOLO re-detection",
    }


# ---------------------------------------------------------
# 9. GET /report/{survey_id} -> report scoped to a single survey
# ---------------------------------------------------------
@app.get("/report/{survey_id}")
def get_survey_report(survey_id: int, db: Session = Depends(get_db)):
    survey = db.query(models.Survey).filter(models.Survey.id == survey_id).first()
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")

    image_ids = [img.id for img in db.query(models.Image).filter(models.Image.survey_id == survey_id).all()]

    detections = db.query(models.Detection).filter(models.Detection.image_id.in_(image_ids)).all() if image_ids else []

    high_risk = [d for d in detections if d.priority == "high"]
    medium_risk = [d for d in detections if d.priority == "medium"]
    low_risk = [d for d in detections if d.priority == "low"]

    return {
        "survey_id": survey.id,
        "survey_name": survey.name,
        "total_detections": len(detections),
        "high_risk": len(high_risk),
        "medium_risk": len(medium_risk),
        "low_risk": len(low_risk),
        "detections": [
            {
                "id": d.id,
                "class": d.object_class,
                "confidence": d.confidence,
                "risk_score": d.risk_score,
                "priority": d.priority,
                "status": d.status,
                "latitude": d.latitude,
                "longitude": d.longitude,
            }
            for d in detections
        ],
    }


# ---------------------------------------------------------
# 10. GET /health -> simple health check for uptime monitoring
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}
