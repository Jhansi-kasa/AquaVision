from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class DetectionOut(BaseModel):
    id: int
    object_class: str
    confidence: float
    bbox: list
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    survey_id: Optional[int] = None
    image_id: Optional[int] = None
    depth: Optional[float] = None
    estimated_size: Optional[float] = None
    data_quality: Optional[float] = None
    risk_score: Optional[float] = None
    priority: Optional[str] = None
    status: str

    class Config:
        from_attributes = True


class ImageOut(BaseModel):
    id: int
    filepath: str
    uploaded_at: datetime
    status: str

    class Config:
        from_attributes = True


class SurveyIn(BaseModel):
    name: Optional[str] = None
    water_body: Optional[str] = None
    vessel: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None


class SurveyOut(BaseModel):
    id: int
    name: Optional[str] = None
    date: datetime
    location: Optional[str] = None
    water_body: Optional[str] = None
    vessel: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    status: str

    class Config:
        from_attributes = True


class ReviewIn(BaseModel):
    detection_id: int
    reviewer: Optional[str] = "operator"
    verdict: str  # "confirmed", "rejected", or "flagged"
    comment: Optional[str] = None


class VerifyIn(BaseModel):
    detection_id: int
    before_image_id: int
    after_image_id: int
    comment: Optional[str] = None


class VerifyOut(BaseModel):
    id: int
    detection_id: int
    before_image_id: int
    after_image_id: int
    result: str
    confidence: Optional[float] = None
    comment: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True
