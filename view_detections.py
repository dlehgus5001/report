#!/usr/bin/env python3
"""
view_detections.py – SAM3 탐지 결과 DB 조회 도구

사용법:
  python view_detections.py                        # 최근 탐지 20건
  python view_detections.py --limit 50             # 최근 50건
  python view_detections.py --class "military tank" # 특정 클래스 필터
  python view_detections.py --image-id <uuid>      # 특정 이미지 탐지 결과
  python view_detections.py --session <uuid>       # 특정 세션 페어링 결과
  python view_detections.py --summary              # 클래스별 통계 요약
  python view_detections.py --pairings             # 페어링 결과 조회
  python view_detections.py --pairings --status matched
"""

import argparse
import sys
from datetime import datetime

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from src.config import SENSOR_DB_PATH, PAIRING_DB_PATH
from src.database.models import (
    Base, PairingBase,
    ImageRecord, DetectionRecord, PairingRecord,
)


# ---------------------------------------------------------------------------
# DB 연결
# ---------------------------------------------------------------------------

def get_sensor_session() -> Session:
    engine = create_engine(f"sqlite:///{SENSOR_DB_PATH}", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def get_pairing_session() -> Session:
    engine = create_engine(f"sqlite:///{PAIRING_DB_PATH}", echo=False)
    PairingBase.metadata.create_all(engine)
    return Session(engine)


# ---------------------------------------------------------------------------
# 출력 헬퍼
# ---------------------------------------------------------------------------

SEP = "─" * 90

def fmt_dt(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "N/A"

def fmt_f(v, decimals=4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"

def print_header(title: str):
    print(f"\n{'═' * 90}")
    print(f"  {title}")
    print(f"{'═' * 90}")


# ---------------------------------------------------------------------------
# 탐지 결과 조회
# ---------------------------------------------------------------------------

def show_detections(
    limit: int = 20,
    object_class: str = None,
    image_id: str = None,
):
    print_header(f"SAM3 탐지 결과  (sensor_detections.db  →  detection_records)")

    with get_sensor_session() as sess:
        q = (
            sess.query(DetectionRecord, ImageRecord)
            .join(ImageRecord, DetectionRecord.image_id == ImageRecord.id)
        )
        if object_class:
            q = q.filter(DetectionRecord.object_class.ilike(f"%{object_class}%"))
        if image_id:
            q = q.filter(DetectionRecord.image_id == image_id)

        q = q.order_by(DetectionRecord.detection_time.desc()).limit(limit)
        rows = q.all()

    if not rows:
        print("  조회 결과 없음.")
        return

    print(f"  총 {len(rows)}건 표시 (최신순)\n")
    print(f"  {'ID':8}  {'탐지시각':19}  {'클래스':20}  {'신뢰도':6}  "
          f"{'위도':10}  {'경도':10}  {'BBox W×H':12}  {'소스':10}  {'이미지 캡처':19}")
    print(SEP)

    for det, img in rows:
        w = int(det.bbox_x2 - det.bbox_x1)
        h = int(det.bbox_y2 - det.bbox_y1)
        print(
            f"  {det.id[:8]}  "
            f"{fmt_dt(det.detection_time):19}  "
            f"{det.object_class:20}  "
            f"{det.confidence:6.3f}  "
            f"{fmt_f(det.lat):10}  "
            f"{fmt_f(det.lon):10}  "
            f"{w:5}×{h:<5}  "
            f"{(det.source_type or img.source_type or '?'):10}  "
            f"{fmt_dt(img.capture_time):19}"
        )

    print(SEP)


# ---------------------------------------------------------------------------
# 클래스별 통계 요약
# ---------------------------------------------------------------------------

def show_summary():
    print_header("SAM3 탐지 결과 – 클래스별 통계 요약")

    with get_sensor_session() as sess:
        rows = (
            sess.query(
                DetectionRecord.object_class,
                func.count(DetectionRecord.id).label("count"),
                func.avg(DetectionRecord.confidence).label("avg_conf"),
                func.min(DetectionRecord.confidence).label("min_conf"),
                func.max(DetectionRecord.confidence).label("max_conf"),
            )
            .group_by(DetectionRecord.object_class)
            .order_by(func.count(DetectionRecord.id).desc())
            .all()
        )

        total = sess.query(func.count(DetectionRecord.id)).scalar()
        img_count = sess.query(func.count(ImageRecord.id)).scalar()

    if not rows:
        print("  탐지 결과 없음.")
        return

    print(f"  이미지 {img_count}장 / 탐지 객체 총 {total}건\n")
    print(f"  {'클래스':25}  {'건수':>6}  {'평균신뢰도':>10}  {'최소':>7}  {'최대':>7}")
    print(SEP)
    for r in rows:
        print(
            f"  {r.object_class:25}  {r.count:6}  "
            f"{r.avg_conf:10.3f}  {r.min_conf:7.3f}  {r.max_conf:7.3f}"
        )
    print(SEP)


# ---------------------------------------------------------------------------
# 이미지별 탐지 결과
# ---------------------------------------------------------------------------

def show_images(limit: int = 10):
    print_header("이미지별 탐지 수 (image_records)")

    with get_sensor_session() as sess:
        rows = (
            sess.query(
                ImageRecord,
                func.count(DetectionRecord.id).label("det_count"),
            )
            .outerjoin(DetectionRecord, DetectionRecord.image_id == ImageRecord.id)
            .group_by(ImageRecord.id)
            .order_by(ImageRecord.capture_time.desc())
            .limit(limit)
            .all()
        )

    if not rows:
        print("  이미지 없음.")
        return

    print(f"  {'ID':8}  {'캡처시각':19}  {'소스':10}  {'탐지수':>6}  "
          f"{'위도':10}  {'경도':10}  {'플랫폼':15}")
    print(SEP)
    for img, cnt in rows:
        print(
            f"  {img.id[:8]}  {fmt_dt(img.capture_time):19}  "
            f"{img.source_type:10}  {cnt:6}  "
            f"{fmt_f(img.lat_center):10}  {fmt_f(img.lon_center):10}  "
            f"{(img.sensor_platform or 'N/A'):15}"
        )
    print(SEP)


# ---------------------------------------------------------------------------
# 페어링 결과 조회
# ---------------------------------------------------------------------------

STATUS_LABEL = {
    "matched":     "✓ MATCHED    ",
    "new":         "★ NEW        ",
    "disappeared": "✗ DISAPPEARED",
}

def show_pairings(
    limit: int = 20,
    status: str = None,
    session_id: str = None,
):
    print_header(f"Sam3Tracker 페어링 결과  (object_pairings.db  →  pairing_records)")

    with get_pairing_session() as sess:
        q = sess.query(PairingRecord)
        if status:
            q = q.filter(PairingRecord.status == status)
        if session_id:
            q = q.filter(PairingRecord.session_id == session_id)
        q = q.order_by(PairingRecord.pairing_time.desc()).limit(limit)
        rows = q.all()

        # 상태별 전체 집계
        counts = dict(
            sess.query(PairingRecord.status, func.count(PairingRecord.id))
            .group_by(PairingRecord.status)
            .all()
        )

    total = sum(counts.values())
    print(f"  전체 {total}건 │ "
          f"matched {counts.get('matched', 0)}  "
          f"new {counts.get('new', 0)}  "
          f"disappeared {counts.get('disappeared', 0)}\n")

    if not rows:
        print("  조회 결과 없음.")
        return

    print(f"  표시: {len(rows)}건\n")

    for p in rows:
        label = STATUS_LABEL.get(p.status, p.status)
        print(f"  {label}  [{fmt_dt(p.pairing_time)}]")

        if p.status in ("matched", "new"):
            print(f"    현재  ID={p.current_detection_id[:8] if p.current_detection_id else 'N/A':8}"
                  f"  CLASS={p.current_object_class or 'N/A':20}"
                  f"  CONF={fmt_f(p.current_confidence, 3):6}"
                  f"  LAT={fmt_f(p.current_lat):10}  LON={fmt_f(p.current_lon):10}")

        if p.status in ("matched", "disappeared"):
            print(f"    과거  ID={p.past_detection_id[:8] if p.past_detection_id else 'N/A':8}"
                  f"  CLASS={p.past_object_class or 'N/A':20}"
                  f"  CONF={fmt_f(p.past_confidence, 3):6}"
                  f"  LAT={fmt_f(p.past_lat):10}  LON={fmt_f(p.past_lon):10}")

        src = p.source_type or "N/A"
        sess_str = p.session_id[:8] if p.session_id else "N/A"
        print(f"    소스={src:10}  세션={sess_str}")
        print(f"  {SEP[:70]}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 탐지 결과 DB 조회 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--limit",    type=int,  default=20,   help="출력 최대 건수 (기본 20)")
    parser.add_argument("--class",    dest="obj_class", default=None, help="클래스 이름 필터 (부분 일치)")
    parser.add_argument("--image-id", default=None, help="특정 이미지 UUID")
    parser.add_argument("--session",  default=None, help="파이프라인 세션 UUID")
    parser.add_argument("--summary",  action="store_true", help="클래스별 통계 요약")
    parser.add_argument("--images",   action="store_true", help="이미지별 탐지 수")
    parser.add_argument("--pairings", action="store_true", help="Sam3Tracker 페어링 결과 조회")
    parser.add_argument("--status",   choices=["matched", "new", "disappeared"], default=None,
                        help="페어링 status 필터 (--pairings 와 함께 사용)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.summary:
        show_summary()
        show_images(limit=args.limit)
        return

    if args.pairings:
        show_pairings(limit=args.limit, status=args.status, session_id=args.session)
        return

    # 기본: 탐지 결과
    show_detections(
        limit=args.limit,
        object_class=args.obj_class,
        image_id=args.image_id,
    )


if __name__ == "__main__":
    main()
