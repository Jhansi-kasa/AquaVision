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

# FIX: backend/gis/main.py uses top-level imports such as `from risk_engine import calculate_risk`.
# That only works if the gis folder itself is on sys.path. Append (not insert at 0) so that
# files inside gis/ can never shadow modules from the backend package or project root.
GIS_DIR = os.path.join(BACKEND_ROOT_DIR, "gis")
if os.path.isdir(GIS_DIR) and GIS_DIR not in sys.path:
    sys.path.append(GIS_DIR)

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

from gis.risk_engine import calculate_risk
from gis.priority_engine import calculate_priority_result
from gis.main import generate_mission_plan
try:
    from computer_vision.final_preprocessing_pipeline import process_sonar_image
except ImportError:
    from .preprocessing.final_preprocessing_pipeline import process_sonar_image

from .database import engine, get_db, Base
from . import models, schemas
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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
    raise FileNotFoundError(
        f"4-class model not found. Checked: {AQUA_VISION_MODEL_PATH}, {ROOT_MODEL_PATH}, "
        f"{EXP_MODEL_PATH}, {RUNS_MODEL_PATH}, {BASELINE_MODEL_PATH}"
    )

FOUR_CLASS_MODEL_PATH = MODEL_PATH

model = YOLO(MODEL_PATH)
print(f"MODEL PATH: {os.path.abspath(MODEL_PATH)}")
print(f"MODEL CLASSES: {model.names}")


def preprocess_sonar(img_bgr_or_gray):
    """Apply the full 4-stage training preprocessing pipeline to an image.
    Re-uses `process_sonar_image` from `computer_vision.final_preprocessing_pipeline`.
    Accepts either a BGR 3-channel image or a single-channel grayscale image.
    Grayscale inputs are first converted to BGR to match the expected input shape.
    The returned image has identical dimensions to the input (no cropping/resizing).
    """
    if len(img_bgr_or_gray.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_bgr_or_gray
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

        # 2. If confidence is very similar, prefer better spatial coverage
        bbox_a = a["bbox"]
        bbox_b = b["bbox"]
        area_a = max(0.0, float(bbox_a[2]) - float(bbox_a[0])) * max(0.0, float(bbox_a[3]) - float(bbox_a[1]))
        area_b = max(0.0, float(bbox_b[2]) - float(bbox_b[0])) * max(0.0, float(bbox_b[3]) - float(bbox_b[1]))

        if abs(area_a - area_b) > 1.0:
            return 1 if area_a > area_b else -1

        return 1 if conf_diff >= 0 else -1

    final_detections = []

    for cls_name, cls_candidates in by_class.items():
        sorted_candidates = sorted(cls_candidates, key=cmp_to_key(compare_candidates), reverse=True)

        kept_for_class = []
        for cand in sorted_candidates:
            is_duplicate = False
            for kept in kept_for_class:
                iou, io_min = calculate_box_overlap(cand["bbox"], kept["bbox"])
                if iou >= iou_thresh or io_min >= containment_thresh:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_for_class.append(cand)

        final_detections.extend(kept_for_class)

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

    # 1. Confidence filtering: class-adaptive thresholds when default 0.25 is used, otherwise user value
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

    # 2. Class-aware NMS
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
    "mine": float(os.getenv("CONF_MINE", "0.08")),
    "shipwreck": float(os.getenv("CONF_SHIPWRECK", "0.10")),
    "aircraft": float(os.getenv("CONF_AIRCRAFT", "0.15")),
    "fishing_gear": float(os.getenv("CONF_FISHING_GEAR", "0.20")),
}


def predict_sonar_high_recall(processed_img, model_instance, user_conf=None, iou_thresh=0.45, use_tiling=True):
    """
    High-recall sonar target detection pipeline:
    - Handles large waterfall imagery (>800px) via tiled inference to avoid
      downsampling micro-targets (e.g. mines <20px).
    - Uses calibrated class-specific thresholds.
    - If user provides an explicit confidence threshold, it acts as the filtering floor.
    """
    h, w = processed_img.shape[:2]
    base_conf = 0.06 if user_conf is None else min(user_conf, 0.08)

    candidate_boxes = []

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
            width_px = max(0, int(round(float(bbox[2]) - float(bbox[0]))))
            height_px = max(0, int(round(float(bbox[3]) - float(bbox[1]))))
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
# 1. POST /upload  -> upload a sonar image to an existing survey.
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
    filename = f"{survey.id}_{os.path.basename(file.filename)}"
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

        # A survey MUST exist before sonar images can be uploaded
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

        # Real sonar preprocessing matching the model:
        # CLAHE on L-channel in LAB space + bilateral speckle reduction filter.
        processed_img = preprocess_sonar(image_array)

        processed_path = os.path.join(UPLOAD_DIR, f"processed_{survey.id}_{safe_name}.png")
        cv2.imwrite(processed_path, processed_img)

        print(f"[SONAR] Processed image: {processed_path}")

        # Determine confidence threshold (0.25 default) and IoU (0.45)
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
        # 1. YOLO INFERENCE (raw predictions with low baseline floor)
        # -----------------------------------------------------
        results = model(processed_path, conf=0.01, iou=0.7, verbose=False)

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

        # Count existing detections for this survey so detection IDs restart from 1 per survey
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
            class_name = cand["class_name"]
            confidence = cand["confidence"]
            bbox = cand["bbox"]

            x1, y1, x2, y2 = [int(round(v)) for v in bbox]

            x1 = max(0, min(x1, detection_image.shape[1] - 1))
            x2 = max(0, min(x2, detection_image.shape[1] - 1))
            y1 = max(0, min(y1, detection_image.shape[0] - 1))
            y2 = max(0, min(y2, detection_image.shape[0] - 1))

            width_px = max(0, x2 - x1)
            height_px = max(0, y2 - y1)

            # Draw thick, clearly visible green bounding box
            cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 8)

            label = f"{class_name} {confidence * 100:.1f}%"

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
            priority_result = calculate_priority_result(risk_score, risk_level)
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
    except HTTPException:
        # FIX: keep intentional 400/404 responses instead of converting them to 500
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
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
    # Resolve/create survey
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
    debug_filename = f"debug_{survey.id}_{safe_name}"
    debug_filepath = os.path.join(UPLOAD_DIR, debug_filename)
    with open(debug_filepath, "wb") as buffer:
        buffer.write(raw_bytes)

    image_array = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        image_array = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Could not decode sonar image")

    processed_img = preprocess_sonar(image_array)
    processed_path = os.path.join(UPLOAD_DIR, f"debug_processed_{survey.id}_{safe_name}.png")
    if not cv2.imwrite(processed_path, processed_img):
        raise HTTPException(status_code=500, detail="Could not save processed debug image")

    iou_thresh = float(os.getenv("IOU_THRESHOLD", "0.45"))
    threshold_results = {}
    for threshold in (0.25, 0.10, 0.05):
        detections, data_url = _detect_and_annotate(processed_path, threshold, iou_thresh)
        threshold_results[str(threshold)] = {
            "confidence_threshold": threshold,
            "detections": detections,
            "detection_count": len(detections),
            "detection_image": data_url,
        }

    db.commit()

    return {
        "survey_id": survey.id,
        "survey_code": f"SURV-{survey.id:03d}",
        "filename": safe_name,
        "thresholds": threshold_results,
    }