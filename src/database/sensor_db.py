"""
Sensor DB operations – insert and query detected objects.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from .models import DetectionRecord, ImageRecord, create_sensor_engine
from src.config import SENSOR_DB_PATH

logger = logging.getLogger(__name__)

_engine = None


# 센서 DB 엔진을 반환하고 마이그레이션 수행
def get_engine():
    global _engine
    if _engine is None:
        _engine = create_sensor_engine(SENSOR_DB_PATH)
        # 기존 DB에 없는 컬럼을 마이그레이션으로 추가 (이미 존재하면 무시)
        _migrations = [
            ("image_records",     "det_width INTEGER"),
            ("image_records",     "det_height INTEGER"),
            ("image_records",     "session_id VARCHAR(36)"),
            ("detection_records", "session_id VARCHAR(36)"),
            ("image_records",     "lat_min FLOAT"),
            ("image_records",     "lat_max FLOAT"),
            ("image_records",     "lon_min FLOAT"),
            ("image_records",     "lon_max FLOAT"),
        ]
        with _engine.connect() as conn:
            for table, col_def in _migrations:
                try:
                    conn.execute(
                        __import__("sqlalchemy").text(
                            f"ALTER TABLE {table} ADD COLUMN {col_def}"
                        )
                    )
                    conn.commit()
                except Exception:
                    pass  # 이미 존재하면 무시
    return _engine


# ---------------------------------------------------------------------------
# Image record CRUD
# ---------------------------------------------------------------------------

# 이미지 메타데이터를 DB에 삽입하고 레코드 반환
def insert_image_record(
    capture_time: datetime,
    source_type: str,
    image_path: str,
    lat_center: float,
    lon_center: float,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    resolution_m: Optional[float] = None,
    sensor_platform: Optional[str] = None,
    det_width: Optional[int] = None,
    det_height: Optional[int] = None,
    session_id: Optional[str] = None,
) -> ImageRecord:
    engine = get_engine()
    with Session(engine) as session:
        record = ImageRecord(
            capture_time=capture_time,
            source_type=source_type,
            image_path=image_path,
            lat_center=lat_center,
            lon_center=lon_center,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            resolution_m=resolution_m,
            sensor_platform=sensor_platform,
            det_width=det_width,
            det_height=det_height,
            session_id=session_id,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(f"[SensorDB] Inserted ImageRecord id={record.id}")
        return record


def update_image_record_meta(
    image_id: str,
    session_id: Optional[str] = None,
    det_width: Optional[int] = None,
    det_height: Optional[int] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
) -> Optional[ImageRecord]:
    """
    Update mutable fields of an existing ImageRecord and return the updated
    detached object.  Only non-None arguments overwrite existing values.
    """
    engine = get_engine()
    with Session(engine) as sess:
        rec = sess.get(ImageRecord, image_id)
        if rec is None:
            return None
        if session_id is not None:
            rec.session_id = session_id
        if det_width is not None:
            rec.det_width = det_width
        if det_height is not None:
            rec.det_height = det_height
        if lat_min is not None:
            rec.lat_min = lat_min
            rec.lat_max = lat_max
            rec.lon_min = lon_min
            rec.lon_max = lon_max
        sess.commit()
        sess.refresh(rec)
        sess.expunge(rec)
        logger.debug(f"[SensorDB] Updated ImageRecord id={image_id}")
        return rec


# ---------------------------------------------------------------------------
# Detection record CRUD
# ---------------------------------------------------------------------------

# 탐지 레코드 한 건을 DB에 삽입
def insert_detection(detection: DetectionRecord) -> str:
    engine = get_engine()
    with Session(engine) as session:
        session.add(detection)
        session.commit()
        session.refresh(detection)
        logger.debug(
            f"[SensorDB] Inserted detection id={detection.id} "
            f"class={detection.object_class} lat={detection.lat:.4f} lon={detection.lon:.4f}"
        )
        return detection.id


# 탐지 레코드 목록을 일괄 삽입하고 ID 목록 반환
def insert_detections_bulk(detections: List[DetectionRecord]) -> List[str]:
    engine = get_engine()
    ids = []
    with Session(engine) as session:
        for d in detections:
            session.add(d)
        session.commit()
        for d in detections:
            session.refresh(d)
            ids.append(d.id)
    logger.info(f"[SensorDB] Bulk inserted {len(ids)} detections.")
    return ids


# SQLite 비교를 위해 타임존 정보를 제거
def _naive(dt: datetime) -> datetime:
    """Strip timezone info so SQLite (which stores naive datetimes) comparisons work."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


# 가장 최근 과거 탐지 결과와 캡처 시각을 반환
def get_most_recent_past_detections(
    lat_center: float,
    lon_center: float,
    radius_deg: float,
    before_time: datetime,
    limit: int = 100,
    prefer_session_id: str = None,
    session_only: bool = False,
) -> tuple:
    """
    Return (detections, past_capture_time) where detections are within
    `radius_deg` of (lat_center, lon_center) captured strictly BEFORE
    `before_time`, limited to the single most recent capture_time batch.

    prefer_session_id: when set, first try to find past data within this
    session (same-session temporal pairing). Only falls back to the global
    cross-session search when no same-session past data exists AND
    session_only=False.

    session_only: when True, never fall back to cross-session search.
    Use this when a same-session past ImageRecord is known to exist —
    prevents previous-session images contaminating pairing even when the
    same-session past image has zero detections.

    past_capture_time is the capture_time of that past batch (None if no records).
    """
    engine = get_engine()
    before_time_naive = _naive(before_time)

    from sqlalchemy import func
    from .models import ImageRecord

    # 추가 필터 조건으로 최신 캡처 배치와 레코드 조회
    def _query_latest_and_records(session, extra_filters):
        """Find most-recent capture_time batch matching extra_filters."""
        latest_time_row = (
            session.query(func.max(ImageRecord.capture_time))
            .join(DetectionRecord, DetectionRecord.image_id == ImageRecord.id)
            .filter(
                ImageRecord.capture_time < before_time_naive,
                DetectionRecord.lat.between(lat_center - radius_deg, lat_center + radius_deg),
                DetectionRecord.lon.between(lon_center - radius_deg, lon_center + radius_deg),
                *extra_filters,
            )
            .scalar()
        )
        if latest_time_row is None:
            return None, []

        records = (
            session.query(DetectionRecord)
            .join(ImageRecord, DetectionRecord.image_id == ImageRecord.id)
            .filter(
                ImageRecord.capture_time == latest_time_row,
                DetectionRecord.lat.between(lat_center - radius_deg, lat_center + radius_deg),
                DetectionRecord.lon.between(lon_center - radius_deg, lon_center + radius_deg),
                *extra_filters,
            )
            .limit(limit)
            .all()
        )
        return latest_time_row, records

    with Session(engine) as session:
        # 1. 동일 세션 내 과거 데이터 우선 검색 (crop 모드에서 세션 오염 방지)
        if prefer_session_id:
            ts, records = _query_latest_and_records(
                session, [ImageRecord.session_id == prefer_session_id]
            )
            if records:
                for r in records:
                    session.expunge(r)
                logger.info(
                    f"[SensorDB] Found {len(records)} past detections (same session) near "
                    f"({lat_center:.4f}, {lon_center:.4f}) at {ts}"
                )
                return records, ts

            # session_only=True: 동일 세션에 과거 이미지가 존재하지만 탐지 결과가 없는 경우
            # cross-session 폴백 없이 빈 결과 반환 (이전 세션 이미지와 잘못 페어링되는 현상 방지)
            if session_only:
                logger.info(
                    f"[SensorDB] No same-session past detections near "
                    f"({lat_center:.4f}, {lon_center:.4f}) — session_only, skip global fallback"
                )
                return [], None

        # 2. 전체 DB 폴백 (cross-session) — session_only=False 일 때만 실행
        ts, records = _query_latest_and_records(session, [])
        for r in records:
            session.expunge(r)

        logger.info(
            f"[SensorDB] Found {len(records)} past detections near "
            f"({lat_center:.4f}, {lon_center:.4f}) at {ts}"
        )
        return records, ts


# detection_id로 탐지 레코드 조회
def get_detection_by_id(detection_id: str) -> Optional[DetectionRecord]:
    """Return the DetectionRecord for the given id, or None if not found."""
    engine = get_engine()
    with Session(engine) as session:
        record = session.query(DetectionRecord).filter(DetectionRecord.id == detection_id).first()
        if record is not None:
            session.expunge(record)
        return record


# 세션의 합성 탐지 레코드 전체 삭제 (재페어링 전 호출)
def delete_synthetic_detections_by_session(session_id: str) -> int:
    """Delete all synthetic DetectionRecords (source_type='synthetic') for the session."""
    engine = get_engine()
    with Session(engine) as session:
        deleted = (
            session.query(DetectionRecord)
            .filter(
                DetectionRecord.session_id == session_id,
                DetectionRecord.source_type == "synthetic",
            )
            .delete(synchronize_session=False)
        )
        session.commit()
    logger.info(f"[SensorDB] Deleted {deleted} synthetic detections for session {session_id}")
    return deleted


# detection_id로 탐지 레코드 삭제
def delete_detection_by_id(detection_id: str) -> bool:
    """Delete a single DetectionRecord. Returns True if deleted, False if not found."""
    engine = get_engine()
    with Session(engine) as session:
        record = session.query(DetectionRecord).filter(DetectionRecord.id == detection_id).first()
        if record is None:
            return False
        session.delete(record)
        session.commit()
        logger.info(f"[SensorDB] Deleted DetectionRecord id={detection_id}")
        return True


# 캡처 시각 ±허용범위 내 이미지 레코드 목록 반환
def get_images_by_capture_time(capture_time: datetime, tolerance_sec: int = 5) -> List[ImageRecord]:
    """capture_time ± tolerance_sec 범위의 ImageRecord 목록 반환."""
    from datetime import timedelta
    engine = get_engine()
    lo = capture_time - timedelta(seconds=tolerance_sec)
    hi = capture_time + timedelta(seconds=tolerance_sec)
    with Session(engine) as session:
        records = (
            session.query(ImageRecord)
            .filter(ImageRecord.capture_time >= lo, ImageRecord.capture_time <= hi)
            .order_by(ImageRecord.capture_time)
            .all()
        )
        for r in records:
            session.expunge(r)
        return records


# image_path로 이미지 레코드 조회 (중복 삽입 방지용)
def get_image_record_by_path(image_path: str) -> Optional[ImageRecord]:
    """Return the ImageRecord for the given image_path, or None if not found."""
    engine = get_engine()
    with Session(engine) as session:
        record = (
            session.query(ImageRecord)
            .filter(ImageRecord.image_path == image_path)
            .first()
        )
        if record is not None:
            session.expunge(record)
        return record


# image_id로 이미지 레코드 조회
def get_image_record_by_id(image_id: str) -> Optional[ImageRecord]:
    """Return the ImageRecord for the given id, or None if not found."""
    engine = get_engine()
    with Session(engine) as session:
        record = session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
        if record is not None:
            session.expunge(record)
        return record


# 이미지 ID에 속한 탐지 레코드 전체 반환
def get_detections_by_image(image_id: str) -> List[DetectionRecord]:
    engine = get_engine()
    with Session(engine) as session:
        records = (
            session.query(DetectionRecord)
            .filter(DetectionRecord.image_id == image_id)
            .all()
        )
        for r in records:
            session.expunge(r)
        return records


# 이미지 탐지 결과를 전부 삭제 후 새 목록으로 교체
def replace_detections_for_image(image_id: str, new_detections: List[DetectionRecord]) -> int:
    """이미지의 모든 탐지 결과를 삭제하고 새 목록으로 교체. 교체된 건수 반환."""
    engine = get_engine()
    with Session(engine) as session:
        session.query(DetectionRecord).filter(DetectionRecord.image_id == image_id).delete()
        for d in new_detections:
            session.add(d)
        session.commit()
        for d in new_detections:
            session.refresh(d)
            session.expunge(d)
        logger.info(f"[SensorDB] replace_detections image_id={image_id} count={len(new_detections)}")
        return len(new_detections)


# 지정 반경 내 가장 최근 촬영 이미지 레코드 반환
def get_latest_image_near(
    lat: float,
    lon: float,
    radius_deg: float = 0.1,
) -> Optional[ImageRecord]:
    """Return the most recently captured ImageRecord within radius_deg of (lat, lon)."""
    engine = get_engine()
    from sqlalchemy import func
    with Session(engine) as session:
        record = (
            session.query(ImageRecord)
            .filter(
                ImageRecord.lat_center.between(lat - radius_deg, lat + radius_deg),
                ImageRecord.lon_center.between(lon - radius_deg, lon + radius_deg),
            )
            .order_by(ImageRecord.capture_time.desc())
            .first()
        )
        if record:
            session.expunge(record)
        return record


# 이미지 레코드와 탐지 수를 최신순으로 반환
def get_all_images_with_count(limit: int = None):
    """Return (ImageRecord, detection_count) tuples ordered by capture_time desc."""
    engine = get_engine()
    from sqlalchemy import func
    with Session(engine) as session:
        q = (
            session.query(ImageRecord, func.count(DetectionRecord.id).label("det_count"))
            .outerjoin(
                DetectionRecord,
                (DetectionRecord.image_id == ImageRecord.id)
                & (DetectionRecord.source_type != "synthetic"),
            )
            .group_by(ImageRecord.id)
            .order_by(ImageRecord.capture_time.desc())
        )
        if limit is not None:
            q = q.limit(limit)
        rows = q.all()
        result = []
        for img, count in rows:
            session.expunge(img)
            result.append((img, count))
        return result


# 세션 ID에 해당하는 이미지 레코드 전체 반환
def get_images_by_session(session_id: str) -> List[ImageRecord]:
    """Return all ImageRecords tagged with the given pipeline session_id."""
    engine = get_engine()
    with Session(engine) as session:
        records = (
            session.query(ImageRecord)
            .filter(ImageRecord.session_id == session_id)
            .order_by(ImageRecord.capture_time)
            .all()
        )
        for r in records:
            session.expunge(r)
        return records


# 지정 반경 내 최신 이미지 배치의 탐지 결과 반환
def get_latest_detections_near(
    lat: float,
    lon: float,
    radius_deg: float = 0.1,
    limit: int = 200,
) -> List[DetectionRecord]:
    """Return detections from the most recent image batch within radius_deg of (lat, lon)."""
    engine = get_engine()
    from sqlalchemy import func
    with Session(engine) as session:
        # Find the latest capture_time near this region
        latest_time = (
            session.query(func.max(ImageRecord.capture_time))
            .filter(
                ImageRecord.lat_center.between(lat - radius_deg, lat + radius_deg),
                ImageRecord.lon_center.between(lon - radius_deg, lon + radius_deg),
            )
            .scalar()
        )
        if latest_time is None:
            return []
        records = (
            session.query(DetectionRecord)
            .join(ImageRecord, DetectionRecord.image_id == ImageRecord.id)
            .filter(
                ImageRecord.capture_time == latest_time,
                ImageRecord.lat_center.between(lat - radius_deg, lat + radius_deg),
                ImageRecord.lon_center.between(lon - radius_deg, lon + radius_deg),
            )
            .limit(limit)
            .all()
        )
        for r in records:
            session.expunge(r)
        return records
