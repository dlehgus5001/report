"""
Multi-Source Intelligent System (MSIS) – Main Pipeline

Implements the Project Maven-inspired processing pipeline:

  ┌─────────────────────────────────────────────────────────────────┐
  │  SENSORS (Satellite / Drone cameras)                            │
  │  → Images with GPS coordinates & timestamps                     │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  INGESTION LAYER                                                │
  │  ImageLoader: reads image files + metadata.json index          │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  DETECTION LAYER (AI/ML)                                        │
  │  SAM3 (facebook/sam3) text-prompted concept segmentation        │
  │  → masks + bounding boxes + confidence scores in one pass       │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  SENSOR DB (SQLite)                                             │
  │  Tables: image_records, detection_records                       │
  │  Stores: object_id, class, confidence, lat/lon, time            │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  TEMPORAL PAIRING LAYER                                         │
  │  Matches current detections ↔ most-recent-past detections       │
  │  at same coordinates → status: new / matched / disappeared      │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  PAIRING DB (SQLite)                                            │
  │  Table: pairing_records                                         │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  REPORTING LAYER (EXAONE4-32b LLM)                              │
  │  Input:  latest pairing records                                 │
  │  Output: Military change-detection intelligence report          │
  └─────────────────────────────────────────────────────────────────┘
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.config import IMAGES_DIR, COORDINATE_MATCH_RADIUS_DEG, TRACKING_MODE, GRAPHRAG_CONTEXT_ENABLED
from src.database.models import DetectionRecord
from src.database.sensor_db import (
    insert_image_record,
    insert_detections_bulk,
    replace_detections_for_image,
    get_image_record_by_path,
    delete_synthetic_detections_by_session,
)
from src.database.pairing_db import (
    insert_pairings_bulk,
    get_pairings_by_session,
    delete_pairings_by_session,
)
from src.database.reports_db import delete_reports_by_session
from src.detection.image_loader import load_metadata_index, iter_images
from src.detection.sam3_detector import SAM3Detector, DetectionResult
from src.pairing.temporal_pairing import pair_by_tracking, pair_by_similarity
from src.reporting.military_reporter import MilitaryReporter
from src.graph.graph_indexer import GraphIndexer
from src.pipeline_status import (
    write_status as _write_ps, clear_status as _clear_ps,
    STAGE_SR, STAGE_DETECTION, STAGE_PAIRING, STAGE_REPORTING,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MavenPipeline:
    """
    End-to-end Project Maven pipeline orchestrator.

    Usage:
        pipeline = MavenPipeline()
        report = pipeline.run(
            metadata_json="data/images/metadata.json",
            report_output_path="data/reports/latest_report.txt",
        )
        print(report)
    """

    def __init__(self):
        self.detector = SAM3Detector()
        self.reporter = MilitaryReporter()
        self.graph_indexer = GraphIndexer()

    # ------------------------------------------------------------------
    # Step 1: Ingest + Detect
    # ------------------------------------------------------------------

    # 이미지 탐지 및 센서 DB 저장
    def _detect_and_store(self, metadata_json: str, session_id: str) -> List[str]:
        """
        Load all images from metadata index, run SAM3 detection,
        save image records and detection records to Sensor DB.

        Returns list of image_ids processed.
        """
        metas = sorted(load_metadata_index(metadata_json), key=lambda m: m.capture_time)
        image_ids = []
        _lat = metas[-1].lat_center if metas else None
        _lon = metas[-1].lon_center if metas else None
        _detection_stage_set = False

        for loaded in iter_images(metas):
            # SR은 iter_images 내부에서 완료됨 → 루프 첫 진입 시 탐지 단계로 전환
            if not _detection_stage_set:
                _write_ps(STAGE_DETECTION, _lat, _lon, session_id)
                _detection_stage_set = True
            meta = loaded.meta

            # 동일 image_path 레코드가 이미 있으면 재사용 — 파이프라인 재실행 시 중복 방지.
            # 단, session_id·det_width/height·geo bounds 는 현재 실행 값으로 항상 갱신한다.
            # geo bounds 가 없으면 pixel_to_geo 가 모든 탐지를 lat_center 로 폴백하므로
            # FOV 판단(past_not_included / current_not_included)이 불가해진다.
            from src.database.sensor_db import update_image_record_meta
            det_h, det_w = loaded.array.shape[:2]
            existing_record = get_image_record_by_path(meta.image_path)
            if existing_record is not None:
                updated = update_image_record_meta(
                    image_id=existing_record.id,
                    session_id=session_id,
                    det_width=det_w,
                    det_height=det_h,
                    lat_min=meta.lat_min,
                    lat_max=meta.lat_max,
                    lon_min=meta.lon_min,
                    lon_max=meta.lon_max,
                )
                img_record = updated if updated is not None else existing_record
                logger.info(
                    f"  [reuse] ImageRecord updated for {meta.image_path[:40]}… "
                    f"id={img_record.id[:8]}  session={session_id[:8]}"
                )
            else:
                img_record = insert_image_record(
                    capture_time=meta.capture_time,
                    source_type=meta.source_type,
                    image_path=meta.image_path,
                    lat_center=meta.lat_center,
                    lon_center=meta.lon_center,
                    lat_min=meta.lat_min,
                    lat_max=meta.lat_max,
                    lon_min=meta.lon_min,
                    lon_max=meta.lon_max,
                    resolution_m=meta.resolution_m,
                    sensor_platform=meta.sensor_platform,
                    det_width=det_w,
                    det_height=det_h,
                    session_id=session_id,
                )
            image_id = img_record.id
            image_ids.append(image_id)

            # Run SAM3 detection
            det_results: List[DetectionResult] = self.detector.detect(loaded, image_id)

            if not det_results:
                logger.info(f"  No detections for image {image_id}. Skipping.")
                continue

            # Convert to ORM records
            orm_detections = [
                DetectionRecord(
                    id=det.detection_id,
                    image_id=image_id,
                    detection_time=meta.capture_time.replace(tzinfo=None),  # naive UTC for SQLite
                    object_class=det.object_class,
                    object_class_index=det.object_class_index,
                    confidence=det.confidence,
                    bbox_x1=det.bbox_x1,
                    bbox_y1=det.bbox_y1,
                    bbox_x2=det.bbox_x2,
                    bbox_y2=det.bbox_y2,
                    lat=det.lat,
                    lon=det.lon,
                    mask_rle=det.mask_rle,
                    mask_area_px=det.mask_area_px,
                    source_type=det.source_type,
                    session_id=session_id,
                )
                for det in det_results
            ]

            # 기존 레코드이면 교체, 신규이면 bulk insert
            if existing_record is not None:
                replace_detections_for_image(image_id, orm_detections)
            else:
                insert_detections_bulk(orm_detections)
            logger.info(
                f"  [{img_record.id[:8]}] Stored {len(orm_detections)} detections "
                f"for image at ({meta.lat_center:.4f}, {meta.lon_center:.4f})"
            )

        return image_ids

    # ------------------------------------------------------------------
    # Step 2: Temporal Pairing
    # ------------------------------------------------------------------

    # 시간적 페어링 수행 및 DB 저장
    def _pair_and_store(
        self,
        metadata_json: str,
        session_id: str,
    ) -> int:
        """
        For each current-session image:
          1. Load image from disk (needed for Sam3Tracker).
          2. Fetch current detections from Sensor DB (stored by _detect_and_store).
          3. Fetch most-recent past detections from Sensor DB.
          4. Run Sam3Tracker on current image with past bboxes → TrackedObject list.
          5. pair_by_tracking() assigns status by object ID (matched/new/disappeared).
          6. Bulk-insert PairingRecord list into Pairing DB.

        Returns total number of pairing records inserted.
        """
        from PIL import Image as PILImage
        from src.database.sensor_db import (
            get_engine,
            get_most_recent_past_detections,
        )
        from src.database.models import ImageRecord, DetectionRecord as DR
        from sqlalchemy.orm import Session

        # Sort by capture_time ascending so older frames are always stored before
        # newer ones — ensuring get_most_recent_past_detections finds them correctly.
        metas = sorted(load_metadata_index(metadata_json), key=lambda m: m.capture_time)
        total_pairings = 0

        for loaded in iter_images(metas):
            meta = loaded.meta
            engine = get_engine()

            # --- Fetch current detections from Sensor DB ---
            # Normalise capture_time to naive for SQLite comparison
            capture_time_naive = meta.capture_time.replace(tzinfo=None)
            with Session(engine) as sess:
                img_rec = (
                    sess.query(ImageRecord)
                    .filter(
                        ImageRecord.image_path == meta.image_path,
                        ImageRecord.capture_time == capture_time_naive,
                    )
                    .order_by(ImageRecord.ingestion_time.desc())
                    .first()
                )
                if img_rec is None:
                    continue
                image_id = img_rec.id

                # 세션 내 동일 session_id 로 capture_time이 더 이른 ImageRecord 존재 여부 확인.
                # 없으면 이 이미지가 세션의 "과거 기준 이미지"이므로 페어링 대상이 아님 — 건너뜀.
                has_session_past = (
                    sess.query(ImageRecord.id)
                    .filter(
                        ImageRecord.session_id == session_id,
                        ImageRecord.capture_time < capture_time_naive,
                    )
                    .first() is not None
                )
                if not has_session_past:
                    logger.info(
                        f"[Pair] Skip {Path(meta.image_path).name} "
                        f"— oldest image in session (reference frame, not current)"
                    )
                    continue

                current_orm = (
                    sess.query(DR).filter(DR.image_id == image_id).all()
                )
                current_dets = [
                    DetectionResult(
                        detection_id=d.id,
                        detection_time=d.detection_time,
                        image_id=d.image_id,
                        object_class=d.object_class,
                        object_class_index=d.object_class_index or 0,
                        confidence=d.confidence,
                        bbox_x1=d.bbox_x1, bbox_y1=d.bbox_y1,
                        bbox_x2=d.bbox_x2, bbox_y2=d.bbox_y2,
                        lat=d.lat, lon=d.lon,
                        source_type=d.source_type or meta.source_type,
                    )
                    for d in current_orm
                    if (d.source_type or "") != "synthetic"
                ]

            # --- Fetch past detections (same session only) ---
            # 동일 세션의 과거 이미지가 항상 존재함이 위에서 확인됐으므로
            # session_only=True 로 cross-session 폴백을 완전히 차단.
            past_records, past_capture_time = get_most_recent_past_detections(
                lat_center=meta.lat_center,
                lon_center=meta.lon_center,
                radius_deg=COORDINATE_MATCH_RADIUS_DEG,
                before_time=meta.capture_time,
                prefer_session_id=session_id,
                session_only=True,
            )
            # 합성 탐지 레코드(cross-check 부산물)를 과거 목록에서 제거한다.
            # 세션 내 이미지가 여러 장일 때 이전 페어링에서 생성된 synthetic 이
            # 다음 페어링의 past_records 로 흘러들어 bbox 중복을 유발할 수 있음.
            past_records = [r for r in past_records if (r.source_type or "") != "synthetic"]

            # 과거 이미지 ID: 탐지 결과가 없어도 FOV 판단에 사용
            with Session(engine) as _s:
                _past_img = (
                    _s.query(ImageRecord)
                    .filter(
                        ImageRecord.session_id == session_id,
                        ImageRecord.capture_time < capture_time_naive,
                    )
                    .order_by(ImageRecord.capture_time.desc())
                    .first()
                )
                past_image_id_for_fov = _past_img.id if _past_img else None

            # --- Build pairing records ---
            orig_h, orig_w = loaded.array.shape[:2]

            if TRACKING_MODE == "similarity":
                # Strategy B: SAM mask crop + CLIP embedding similarity
                pil_image = PILImage.fromarray(loaded.array).convert("RGB")
                pairing_records = pair_by_similarity(
                    current_detections=current_dets,
                    past_detections=past_records,
                    current_image=pil_image,
                    current_capture_time=meta.capture_time,
                    past_capture_time=past_capture_time,
                    region_lat=meta.lat_center,
                    region_lon=meta.lon_center,
                    session_id=session_id,
                    source_type=meta.source_type,
                    current_image_id=image_id,
                    past_image_id=past_image_id_for_fov,
                )
            else:
                # Strategy A (default): SAM3 video tracker
                pil_image = PILImage.fromarray(loaded.array).convert("RGB")
                tracked_objects = (
                    self.detector.track_objects(pil_image, past_records, orig_w, orig_h)
                    if past_records else []
                )
                pairing_records = pair_by_tracking(
                    tracked_objects=tracked_objects,
                    current_detections=current_dets,
                    past_detections=past_records,
                    current_capture_time=meta.capture_time,
                    past_capture_time=past_capture_time,
                    region_lat=meta.lat_center,
                    region_lon=meta.lon_center,
                    session_id=session_id,
                    source_type=meta.source_type,
                    current_image_id=image_id,
                    past_image_id=past_image_id_for_fov,
                )

            if pairing_records:
                insert_pairings_bulk(pairing_records)
                total_pairings += len(pairing_records)

        return total_pairings

    # ------------------------------------------------------------------
    # Step 3: Report Generation
    # ------------------------------------------------------------------

    # GraphRAG 컨텍스트 기반 군사 보고서 생성
    def _generate_report(
        self,
        session_id: str,
        output_path: Optional[str] = None,
        target_description: str = "",
    ) -> str:
        """
        Retrieve latest pairings, build GraphRAG historical context, and
        generate the military report with context-enriched prompting.
        Graph RAG indexing is always performed here so context is available
        from the very first run (not only after explicit regeneration).
        """
        from src.database.sensor_db import get_images_by_session, get_detections_by_image

        pairings = get_pairings_by_session(session_id)
        if pairings:
            latest_pt = max(p.pairing_time for p in pairings)
            pairings = [p for p in pairings if p.pairing_time == latest_pt]

        # 현재 이미지의 실제 탐지 건수 (synthetic 제외) — 보고서 헤더 정확도를 위해 DB에서 직접 산출
        n_real_current = None
        imgs = get_images_by_session(session_id)
        if imgs:
            latest_img = max(imgs, key=lambda i: i.capture_time)
            dets = get_detections_by_image(latest_img.id)
            n_real_current = sum(1 for d in dets if (d.source_type or "") != "synthetic")

        # --- GraphRAG: 이번 세션 데이터만 반영 (이전 누적 삭제 후 재인덱싱) ---
        if pairings:
            self.graph_indexer.clear()   # 이전 실행 누적 데이터 제거
            self.graph_indexer.index_pairings(pairings, session_id)
            graph_stats = self.graph_indexer.stats()
            logger.info(
                "[Pipeline] GraphRAG indexed: entities=%d, assets=%d, "
                "relations=%d, communities=%d",
                graph_stats.get("entities", 0),
                graph_stats.get("assets", 0),
                graph_stats.get("relations", 0),
                graph_stats.get("communities", 0),
            )

        historical_context = ""
        if GRAPHRAG_CONTEXT_ENABLED:
            historical_context = self.graph_indexer.get_historical_context(pairings)
            if historical_context:
                logger.info("[Pipeline] GraphRAG historical context retrieved (%d chars).",
                            len(historical_context))

        report = self.reporter.generate_report(
            pairings,
            output_path=output_path,
            session_id=session_id,
            historical_context=historical_context,
            target_description=target_description,
            n_real_detections=n_real_current,
        )
        return report

    # ------------------------------------------------------------------
    # Re-run from existing detections (사용자 편집 후 재처리)
    # ------------------------------------------------------------------

    # 수정된 탐지 기반 페어링·보고서 재실행
    def rerun_from_detections(self, image_id: str, target_description: str = "") -> str:
        """
        이미지 1장에 대해 SensorDB에 저장된 탐지 결과(사용자 수정 포함)를 기반으로
        Temporal Pairing → Report Generation을 재실행한다.

        - session_id는 원본 파이프라인 세션을 그대로 재사용한다.
        - 기존 pairing_records / report_records(동일 session)는 삭제 후 재삽입된다.

        Args:
            image_id:           현재 프레임 이미지 ID.
            target_description: 사용자가 입력한 표적 설명 (선택). 보고서 생성 시 LLM 프롬프트에 반영.

        Returns:
            새로 생성된 보고서 텍스트.
        """
        import numpy as np
        from pathlib import Path
        from PIL import Image as PILImage

        from src.config import IMAGES_DIR
        from src.database.sensor_db import (
            get_engine,
            get_image_record_by_id,
            get_detections_by_image,
            get_most_recent_past_detections,
        )
        from src.database.pairing_db import delete_pairings_by_session
        from src.database.reports_db import delete_reports_by_session
        from src.detection.super_resolution import super_resolve

        # ── 1. 이미지 레코드 조회 ────────────────────────────────────────────
        img_rec = get_image_record_by_id(image_id)
        if img_rec is None:
            raise ValueError(f"Image not found: {image_id}")

        session_id   = img_rec.session_id
        capture_time = img_rec.capture_time

        logger.info(
            f"[Rerun] image={image_id[:8]}  session={session_id}  "
            f"capture={capture_time}"
        )

        # ── 2. 이미지 파일 로드 + SR ─────────────────────────────────────────
        img_path = Path(img_rec.image_path)
        if not img_path.is_absolute():
            img_path = Path(IMAGES_DIR) / img_rec.image_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        pil_raw  = PILImage.open(img_path).convert("RGB")
        arr      = np.array(pil_raw, dtype=np.uint8)
        arr      = super_resolve(arr)
        orig_h, orig_w = arr.shape[:2]
        pil_image = PILImage.fromarray(arr)

        # ── 3. 합성 탐지 선제 삭제 → 실제 탐지 결과만 읽기 ────────────────────
        # 반드시 read 이전에 삭제해야 synthetic이 current_dets에 포함되지 않음
        delete_synthetic_detections_by_session(session_id)

        orm_dets     = get_detections_by_image(image_id)
        current_dets = [
            DetectionResult(
                detection_id=d.id,
                detection_time=d.detection_time,
                image_id=d.image_id,
                object_class=d.object_class,
                object_class_index=d.object_class_index or 0,
                confidence=d.confidence,
                bbox_x1=d.bbox_x1, bbox_y1=d.bbox_y1,
                bbox_x2=d.bbox_x2, bbox_y2=d.bbox_y2,
                lat=d.lat, lon=d.lon,
                source_type=d.source_type or img_rec.source_type,
            )
            for d in orm_dets
            if (d.source_type or "") != "synthetic"  # 방어적 필터 — 삭제 후에도 혹시 남은 경우 제외
        ]
        logger.info(f"[Rerun] current detections: {len(current_dets)}")

        # ── 4. 과거 탐지 결과 ────────────────────────────────────────────────
        past_records, past_capture_time = get_most_recent_past_detections(
            lat_center=img_rec.lat_center,
            lon_center=img_rec.lon_center,
            radius_deg=COORDINATE_MATCH_RADIUS_DEG,
            before_time=capture_time,
            prefer_session_id=session_id,
        )
        # 과거 합성 탐지도 제외 (다른 세션에서 생성된 synthetic 포함 방지)
        past_records = [p for p in past_records if (p.source_type or "") != "synthetic"]
        logger.info(f"[Rerun] past detections: {len(past_records)}")

        # 과거 이미지 ID: 탐지 결과가 없어도 FOV 판단에 사용
        capture_naive_rerun = capture_time.replace(tzinfo=None) if getattr(capture_time, "tzinfo", None) else capture_time
        from sqlalchemy.orm import Session as _OrmSess2
        from src.database.models import ImageRecord as _IRec2
        with _OrmSess2(get_engine()) as _s2:
            _past_img2 = (
                _s2.query(_IRec2)
                .filter(
                    _IRec2.session_id == session_id,
                    _IRec2.capture_time < capture_naive_rerun,
                )
                .order_by(_IRec2.capture_time.desc())
                .first()
            )
            past_image_id_rerun = _past_img2.id if _past_img2 else None

        # ── 5. 기존 pairing 삭제 (합성 탐지는 step 3에서 이미 삭제함) ──────────
        delete_pairings_by_session(session_id)

        # ── 6. Temporal Pairing ──────────────────────────────────────────────
        if TRACKING_MODE == "similarity":
            pairing_records = pair_by_similarity(
                current_detections=current_dets,
                past_detections=past_records,
                current_image=pil_image,
                current_capture_time=capture_time,
                past_capture_time=past_capture_time,
                region_lat=img_rec.lat_center,
                region_lon=img_rec.lon_center,
                session_id=session_id,
                source_type=img_rec.source_type,
                current_image_id=img_rec.id,
                past_image_id=past_image_id_rerun,
            )
        else:
            tracked_objects = (
                self.detector.track_objects(pil_image, past_records, orig_w, orig_h)
                if past_records else []
            )
            pairing_records = pair_by_tracking(
                tracked_objects=tracked_objects,
                current_detections=current_dets,
                past_detections=past_records,
                current_capture_time=capture_time,
                past_capture_time=past_capture_time,
                region_lat=img_rec.lat_center,
                region_lon=img_rec.lon_center,
                session_id=session_id,
                source_type=img_rec.source_type,
                current_image_id=img_rec.id,
                past_image_id=past_image_id_rerun,
            )

        if pairing_records:
            insert_pairings_bulk(pairing_records)
            logger.info(f"[Rerun] Inserted {len(pairing_records)} pairing records.")

        # SAM3 모델을 GPU에서 해제 — vLLM 로딩 전 VRAM 확보
        self.detector.unload()

        # ── 7. 기존 보고서 삭제 ──────────────────────────────────────────────
        delete_reports_by_session(session_id)

        # ── 8. 보고서 재생성 ─────────────────────────────────────────────────
        report = self._generate_report(session_id, target_description=target_description)
        logger.info(f"[Rerun] Report regenerated for session={session_id}")
        return report

    # ------------------------------------------------------------------
    # Step-based API (단계별 실행)
    # ------------------------------------------------------------------

    # 이미지 적재만 수행하고 ID 목록 반환
    def ingest_only(self, metadata_json: str, session_id: str) -> List[str]:
        """
        Phase 1: 이미지 적재만 수행 (SAM3 탐지 없음).
        ImageRecord를 Sensor DB에 저장하고 image_id 목록을 반환한다.
        """
        metas = sorted(load_metadata_index(metadata_json), key=lambda m: m.capture_time)
        if not metas:
            logger.warning("[Ingest] metadata.json에 항목 없음")
            return []

        _lat = metas[-1].lat_center
        _lon = metas[-1].lon_center
        _write_ps(STAGE_SR, _lat, _lon, session_id)

        image_ids: List[str] = []
        for loaded in iter_images(metas):
            meta = loaded.meta
            det_h, det_w = loaded.array.shape[:2]
            img_record = insert_image_record(
                capture_time=meta.capture_time,
                source_type=meta.source_type,
                image_path=meta.image_path,
                lat_center=meta.lat_center,
                lon_center=meta.lon_center,
                lat_min=meta.lat_min,
                lat_max=meta.lat_max,
                lon_min=meta.lon_min,
                lon_max=meta.lon_max,
                resolution_m=meta.resolution_m,
                sensor_platform=meta.sensor_platform,
                det_width=det_w,
                det_height=det_h,
                session_id=session_id,
            )
            image_ids.append(img_record.id)
            logger.info(
                "[Ingest] ImageRecord 저장 완료 %s (탐지 없음)", img_record.id[:8]
            )

        _clear_ps()
        return image_ids

    # 단일 이미지에 SAM3 탐지 실행 후 DB 저장
    def detect_sam3_for_image(self, image_id: str) -> List[DetectionRecord]:
        """
        Phase 2a (SAM3 경로): 저장된 이미지 1장에 SAM3 탐지를 실행하고
        DetectionRecord를 Sensor DB에 저장 후 반환한다.
        """
        import numpy as np
        from PIL import Image as PILImage
        from src.database.sensor_db import get_image_record_by_id
        from src.detection.super_resolution import super_resolve

        img_rec = get_image_record_by_id(image_id)
        if img_rec is None:
            raise ValueError(f"Image not found: {image_id}")

        img_path = Path(img_rec.image_path)
        if not img_path.is_absolute():
            img_path = Path(IMAGES_DIR) / img_rec.image_path
        if not img_path.exists():
            raise FileNotFoundError(f"이미지 파일 없음: {img_path}")

        pil_raw = PILImage.open(img_path).convert("RGB")
        arr = np.array(pil_raw, dtype=np.uint8)
        arr = super_resolve(arr)
        det_h, det_w = arr.shape[:2]

        # LoadedImage-like duck type
        class _FakeMeta:
            image_path = img_rec.image_path
            capture_time = img_rec.capture_time
            source_type = img_rec.source_type
            lat_center = img_rec.lat_center
            lon_center = img_rec.lon_center
            lat_min = img_rec.lat_min
            lat_max = img_rec.lat_max
            lon_min = img_rec.lon_min
            lon_max = img_rec.lon_max
            resolution_m = img_rec.resolution_m or 0.5
            sensor_platform = img_rec.sensor_platform or "unknown"

        class _FakeLoaded:
            array = arr
            meta = _FakeMeta()

        det_results: List[DetectionResult] = self.detector.detect(_FakeLoaded(), image_id)

        if not det_results:
            logger.info("[Detect/SAM3] 탐지 결과 없음 (image=%s)", image_id[:8])
            return []

        capture_naive = img_rec.capture_time
        if hasattr(capture_naive, "tzinfo") and capture_naive.tzinfo is not None:
            capture_naive = capture_naive.replace(tzinfo=None)

        orm_detections = [
            DetectionRecord(
                id=det.detection_id,
                image_id=image_id,
                detection_time=capture_naive,
                object_class=det.object_class,
                object_class_index=det.object_class_index,
                confidence=det.confidence,
                bbox_x1=det.bbox_x1,
                bbox_y1=det.bbox_y1,
                bbox_x2=det.bbox_x2,
                bbox_y2=det.bbox_y2,
                lat=det.lat,
                lon=det.lon,
                mask_rle=det.mask_rle,
                mask_area_px=det.mask_area_px,
                source_type=det.source_type,
                session_id=img_rec.session_id,
            )
            for det in det_results
        ]
        replace_detections_for_image(image_id, orm_detections)
        logger.info(
            "[Detect/SAM3] %d건 저장 완료 (image=%s)", len(orm_detections), image_id[:8]
        )
        return orm_detections

    # 탐지 완료 세션의 페어링 및 보고서 생성
    def run_pairing_and_report(
        self,
        session_id: str,
        target_description: str = "",
    ) -> str:
        """
        Phase 2b: 탐지 완료 후 Temporal Pairing → Graph RAG → 보고서 생성.
        Sensor DB에 저장된 탐지 결과(SAM3 또는 수동)를 기반으로 실행한다.
        """
        import numpy as np
        from datetime import timezone
        from pathlib import Path as _Path
        from PIL import Image as PILImage
        from src.database.sensor_db import get_engine, get_most_recent_past_detections
        from src.database.models import ImageRecord, DetectionRecord as DR
        from src.detection.super_resolution import super_resolve
        from sqlalchemy.orm import Session as OrmSession

        engine = get_engine()

        # 세션의 모든 이미지 조회 (capture_time 오름차순)
        with OrmSession(engine) as sess:
            img_recs = (
                sess.query(ImageRecord)
                .filter(ImageRecord.session_id == session_id)
                .order_by(ImageRecord.capture_time.asc())
                .all()
            )
            img_list = [
                {
                    "id": r.id,
                    "image_path": r.image_path,
                    "capture_time": r.capture_time,
                    "source_type": r.source_type or "satellite",
                    "lat_center": r.lat_center,
                    "lon_center": r.lon_center,
                    "lat_min": r.lat_min,
                    "lat_max": r.lat_max,
                    "lon_min": r.lon_min,
                    "lon_max": r.lon_max,
                    "resolution_m": r.resolution_m or 0.5,
                    "sensor_platform": r.sensor_platform or "unknown",
                    "session_id": r.session_id,
                }
                for r in img_recs
            ]

        if not img_list:
            logger.warning("[Analyze] 세션 %s 에 이미지 없음", session_id)
            return ""

        _lat = img_list[-1]["lat_center"]
        _lon = img_list[-1]["lon_center"]
        _write_ps(STAGE_PAIRING, _lat, _lon, session_id)

        total_pairings = 0
        for img_info in img_list:
            ct = img_info["capture_time"]
            ct_naive = ct.replace(tzinfo=None) if getattr(ct, "tzinfo", None) else ct

            # 세션 내 더 오래된 이미지가 있는지 확인
            with OrmSession(engine) as sess:
                has_past = (
                    sess.query(ImageRecord.id)
                    .filter(
                        ImageRecord.session_id == session_id,
                        ImageRecord.capture_time < ct_naive,
                    )
                    .first()
                    is not None
                )
            if not has_past:
                logger.info(
                    "[Analyze] Skip %s — 세션 내 가장 오래된 이미지(기준 프레임)",
                    _Path(img_info["image_path"]).name,
                )
                continue

            # 현재 탐지 결과 조회
            with OrmSession(engine) as sess:
                current_orm = (
                    sess.query(DR).filter(DR.image_id == img_info["id"]).all()
                )
                current_dets = [
                    DetectionResult(
                        detection_id=d.id,
                        detection_time=d.detection_time,
                        image_id=d.image_id,
                        object_class=d.object_class,
                        object_class_index=d.object_class_index or 0,
                        confidence=d.confidence,
                        bbox_x1=d.bbox_x1,
                        bbox_y1=d.bbox_y1,
                        bbox_x2=d.bbox_x2,
                        bbox_y2=d.bbox_y2,
                        lat=d.lat,
                        lon=d.lon,
                        source_type=d.source_type or img_info["source_type"],
                    )
                    for d in current_orm
                ]

            ct_aware = ct_naive.replace(tzinfo=timezone.utc)

            past_records, past_capture_time = get_most_recent_past_detections(
                lat_center=img_info["lat_center"],
                lon_center=img_info["lon_center"],
                radius_deg=COORDINATE_MATCH_RADIUS_DEG,
                before_time=ct_aware,
                prefer_session_id=session_id,
                session_only=True,
            )

            # 과거 이미지 ID: 탐지 결과가 없어도 FOV 판단에 사용
            with OrmSession(engine) as _fov_sess:
                _past_img_fov = (
                    _fov_sess.query(ImageRecord)
                    .filter(
                        ImageRecord.session_id == session_id,
                        ImageRecord.capture_time < ct_naive,
                    )
                    .order_by(ImageRecord.capture_time.desc())
                    .first()
                )
                past_image_id_analyze = _past_img_fov.id if _past_img_fov else None

            # 이미지 파일 로드 (SAM3 추적기용)
            img_path = _Path(img_info["image_path"])
            if not img_path.is_absolute():
                img_path = _Path(IMAGES_DIR) / img_info["image_path"]

            if not img_path.exists():
                logger.warning("[Analyze] 이미지 파일 없음: %s", img_path)
                continue

            pil_raw = PILImage.open(img_path).convert("RGB")
            arr = np.array(pil_raw, dtype=np.uint8)
            arr = super_resolve(arr)
            orig_h, orig_w = arr.shape[:2]
            pil_image = PILImage.fromarray(arr)

            if TRACKING_MODE == "similarity":
                pairing_records = pair_by_similarity(
                    current_detections=current_dets,
                    past_detections=past_records,
                    current_image=pil_image,
                    current_capture_time=ct_aware,
                    past_capture_time=past_capture_time,
                    region_lat=img_info["lat_center"],
                    region_lon=img_info["lon_center"],
                    session_id=session_id,
                    source_type=img_info["source_type"],
                    current_image_id=img_info["id"],
                    past_image_id=past_image_id_analyze,
                )
            else:
                tracked_objects = (
                    self.detector.track_objects(pil_image, past_records, orig_w, orig_h)
                    if past_records
                    else []
                )
                pairing_records = pair_by_tracking(
                    tracked_objects=tracked_objects,
                    current_detections=current_dets,
                    past_detections=past_records,
                    current_capture_time=ct_aware,
                    past_capture_time=past_capture_time,
                    region_lat=img_info["lat_center"],
                    region_lon=img_info["lon_center"],
                    session_id=session_id,
                    source_type=img_info["source_type"],
                    current_image_id=img_info["id"],
                    past_image_id=past_image_id_analyze,
                )

            if pairing_records:
                insert_pairings_bulk(pairing_records)
                total_pairings += len(pairing_records)

        logger.info(
            "[Analyze] 세션 %s: %d pairing records 생성", session_id, total_pairings
        )

        # SAM3 모델 해제 후 LLM 로딩
        self.detector.unload()

        _write_ps(STAGE_REPORTING, _lat, _lon, session_id)
        try:
            report = self._generate_report(session_id, target_description=target_description)
        finally:
            _clear_ps()

        logger.info("[Analyze] 보고서 생성 완료 session=%s", session_id)
        return report

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # 전체 파이프라인 엔드투엔드 실행
    def run(
        self,
        metadata_json: str,
        report_output_path: Optional[str] = None,
        target_description: str = "",
    ) -> str:
        """
        Execute the full pipeline end-to-end.

        Args:
            metadata_json:       Path to the image metadata index JSON.
            report_output_path:  Optional file path to save the generated report.
            target_description:  Optional mission target description injected into
                                 the LLM prompt (from simulator/web_api).

        Returns:
            Military intelligence report as a string.
        """
        session_id = str(uuid.uuid4())
        logger.info(f"{'='*60}")
        logger.info(f"MSIS Pipeline started  session={session_id}")
        logger.info(f"{'='*60}")

        # Auto-generate sample data if metadata.json is empty
        metas_check = load_metadata_index(metadata_json)
        if not metas_check:
            logger.info(
                "[Pipeline] metadata.json has no entries – "
                "auto-generating synthetic sample data for testing."
            )
            from scripts.generate_sample_data import generate_sample_data
            generate_sample_data(metadata_json)
            metas_check = load_metadata_index(metadata_json)

        # 상태 표시용 대표 좌표 (마지막 메타의 lat/lon)
        _lat = metas_check[-1].lat_center if metas_check else None
        _lon = metas_check[-1].lon_center if metas_check else None

        try:
            # --- Step 1: SR + Detection ---
            logger.info("[Pipeline] Step 1/3 – Image ingestion & SAM3 detection")
            _write_ps(STAGE_SR, _lat, _lon, session_id)
            image_ids = self._detect_and_store(metadata_json, session_id)
            logger.info(f"[Pipeline] Processed {len(image_ids)} images.")

            # --- Step 2: Pairing ---
            logger.info("[Pipeline] Step 2/3 – Temporal object pairing")
            _write_ps(STAGE_PAIRING, _lat, _lon, session_id)
            n_pairings = self._pair_and_store(metadata_json, session_id)
            logger.info(f"[Pipeline] Created {n_pairings} pairing records.")

            # SAM3 모델을 GPU에서 해제 — vLLM 로딩 전 VRAM 확보
            self.detector.unload()

            # --- Step 3: Reporting ---
            logger.info("[Pipeline] Step 3/3 – Military report generation (EXAONE4-32b)")
            _write_ps(STAGE_REPORTING, _lat, _lon, session_id)
            report = self._generate_report(session_id, report_output_path,
                                            target_description=target_description)
        finally:
            _clear_ps()

        logger.info(f"[Pipeline] Pipeline complete.  session={session_id}")
        return report
