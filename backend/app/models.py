from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Survey(Base):
    __tablename__ = "surveys"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    date = Column(DateTime, default=datetime.utcnow)
    location = Column(String, nullable=True)       # kept for backward compatibility
    water_body = Column(String, nullable=True)
    vessel = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    depth = Column(Float, nullable=True)
    status = Column(String, default="pending")  # pending, completed

    images = relationship("Image", back_populates="survey")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    survey_id = Column(Integer, ForeignKey("surveys.id"))
    filepath = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="raw")  # raw, processed

    survey = relationship("Survey", back_populates="images")
    detections = relationship("Detection", back_populates="image")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    object_class = Column(String, nullable=False)   # e.g. "ghost_net"
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=False)              # [x1, y1, x2, y2]
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    depth = Column(Float, nullable=True)
    estimated_size = Column(Float, nullable=True)
    data_quality = Column(Float, nullable=True, default=1.0)
    risk_score = Column(Float, nullable=True)
    priority = Column(String, nullable=True)          # high, medium, low
    status = Column(String, default="pending")         # pending, verified, rejected
    survey_detection_index = Column(Integer, nullable=True) # 1-based index per survey (e.g. DET-001)

    image = relationship("Image", back_populates="detections")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"))
    reviewer = Column(String, nullable=True)
    verdict = Column(String, nullable=False)  # confirmed, rejected
    timestamp = Column(DateTime, default=datetime.utcnow)


class Verification(Base):
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"))
    before_image_id = Column(Integer, ForeignKey("images.id"))
    after_image_id = Column(Integer, ForeignKey("images.id"))
    result = Column(String, nullable=False)   # potentially_removed, still_present
    confidence = Column(Float, nullable=True)
    comment = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
