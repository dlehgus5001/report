"""
SQLAlchemy ORM models for Sensor Detections DB, Object Pairings DB, and Reports DB.

Sensor DB:  stores every detected object from current-time satellite/drone imagery.
Pairing DB: stores temporal pairs (current detection ↔ past detection at same coordinates).
Reports DB: stores every generated military intelligence report with saved time and file path.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()
PairingBase = declarative_base()
ReportBase = declarative_base()


# ---------------------------------------------------------------------------
# Sensor DB Models
# ---------------------------------------------------------------------------

class ImageRecord(Base):
    """Metadata for each ingested satellite/drone image."""

    __tablename__ = "image_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ingestion_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    capture_time = Column(DateTime, nullable=False)
    source_type = Column(String(32), nullable=False)   # "satellite" | "drone"
    image_path = Column(Text, nullable=False)
    # Geographic bounding box of the image
    lat_center = Column(Float, nullable=False)
    lon_center = Column(Float, nullable=False)
    lat_min = Column(Float, nullable=True)
    lat_max = Column(Float, nullable=True)
    lon_min = Column(Float, nullable=True)
    lon_max = Column(Float, nullable=True)
    resolution_m = Column(Float, nullable=True)        # GSD in metres/pixel
    sensor_platform = Column(String(64), nullable=True)  # e.g. "WorldView-3", "MQ-9"

    # Dimensions of the image array actually used during detection (after SR preprocessing).
    # bbox coordinates in DetectionRecord are in this pixel space.
    det_width  = Column(Integer, nullable=True)
    det_height = Column(Integer, nullable=True)

    # Pipeline session that produced the detections on this image.
    # Links image records back to the report generation session.
    session_id = Column(String(36), nullable=True)

    detections = relationship("DetectionRecord", back_populates="image", cascade="all, delete-orphan")


class DetectionRecord(Base):
    """
    A single object detected by SAM2 + CLIP classifier in a satellite/drone image.
    Corresponds to the sensor-layer output in the Project Maven pipeline.
    """

    __tablename__ = "detection_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey("image_records.id"), nullable=False)
    detection_time = Column(DateTime, nullable=False)  # must be set to ImageRecord.capture_time

    # Object classification
    object_class = Column(String(128), nullable=False)   # e.g. "military tank"
    object_class_index = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=False)

    # Pixel-space bounding box within image
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    # Geographic coordinates of the detected object (projected from image metadata)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    # SAM2 mask summary (run-length encoded or polygon as JSON string)
    mask_rle = Column(Text, nullable=True)
    mask_area_px = Column(Float, nullable=True)

    # Extra attributes
    source_type = Column(String(32), nullable=True)  # "satellite" | "drone"
    extra = Column(JSON, nullable=True)

    # Pipeline session that generated this detection (same as parent ImageRecord.session_id)
    session_id = Column(String(36), nullable=True)

    image = relationship("ImageRecord", back_populates="detections")


# ---------------------------------------------------------------------------
# Pairing DB Models
# ---------------------------------------------------------------------------

class PairingRecord(PairingBase):
    """
    A temporal pairing between a current detection and the most recent past
    detection at the same geographic region.

    Status values:
        "new"        – object appears for the first time (no past match)
        "matched"    – object present in both frames
        "disappeared"– object was in the past frame but absent in current
    """

    __tablename__ = "pairing_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pairing_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Region identification
    lat_center = Column(Float, nullable=False)
    lon_center = Column(Float, nullable=False)

    # Current-frame detection (always populated for "new" and "matched")
    current_detection_id = Column(String(36), nullable=True)
    current_object_class = Column(String(128), nullable=True)
    current_confidence = Column(Float, nullable=True)
    current_lat = Column(Float, nullable=True)
    current_lon = Column(Float, nullable=True)
    current_capture_time = Column(DateTime, nullable=True)
    current_bbox = Column(JSON, nullable=True)   # {"x1":…, "y1":…, "x2":…, "y2":…}

    # Past-frame detection (None for "new")
    past_detection_id = Column(String(36), nullable=True)
    past_object_class = Column(String(128), nullable=True)
    past_confidence = Column(Float, nullable=True)
    past_lat = Column(Float, nullable=True)
    past_lon = Column(Float, nullable=True)
    past_capture_time = Column(DateTime, nullable=True)
    past_bbox = Column(JSON, nullable=True)

    status = Column(String(32), nullable=False)  # "new" | "matched" | "disappeared"
    source_type = Column(String(32), nullable=True)

    # Analysis session tag (groups all pairs from one pipeline run)
    session_id = Column(String(36), nullable=True)


# ---------------------------------------------------------------------------
# Reports DB Model
# ---------------------------------------------------------------------------

class ReportRecord(ReportBase):
    """
    A single generated military intelligence report.

    Saved to the Reports DB every time MilitaryReporter.generate_report() is called.
    """

    __tablename__ = "report_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # When this row was inserted into the DB
    saved_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Timestamp embedded in the report header (moment of LLM generation)
    report_time = Column(DateTime, nullable=False)

    # Pipeline session that produced this report
    session_id = Column(String(36), nullable=True)

    # Which LLM and backend were used
    llm_model = Column(String(128), nullable=False)
    llm_backend = Column(String(32), nullable=False)   # "huggingface" | "ollama"

    # How many pairing records were analysed
    pairing_count = Column(Integer, nullable=False, default=0)

    # Path to the .txt file on disk (None if --report-output was not specified)
    file_path = Column(Text, nullable=True)

    # Full report text content
    report_content = Column(Text, nullable=False)


# ---------------------------------------------------------------------------
# DB factory helpers
# ---------------------------------------------------------------------------

# race-condition 없이 테이블을 안전하게 생성
def _safe_create_all(metadata, engine) -> None:
    """
    create_all() 을 race-condition 에 안전하게 호출한다.

    두 프로세스(web_api + main.py 서브프로세스)가 동시에 빈 DB를 초기화할 때
    두 번째 프로세스가 'table X already exists' OperationalError를 일으키는
    SQLite race condition을 조용히 무시한다.
    테이블이 이미 존재하면 정상 상태이므로 계속 진행해도 안전하다.
    """
    from sqlalchemy.exc import OperationalError
    try:
        metadata.create_all(engine)
    except OperationalError as exc:
        if "already exists" in str(exc).lower():
            pass  # 다른 프로세스가 먼저 생성 완료 — 정상
        else:
            raise


# 센서 DB용 SQLAlchemy 엔진 생성 및 반환
def create_sensor_engine(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    _safe_create_all(Base.metadata, engine)
    return engine


# 페어링 DB용 SQLAlchemy 엔진 생성 및 반환
def create_pairing_engine(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    _safe_create_all(PairingBase.metadata, engine)
    return engine


# 보고서 DB용 SQLAlchemy 엔진 생성 및 반환
def create_report_engine(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    _safe_create_all(ReportBase.metadata, engine)
    return engine
