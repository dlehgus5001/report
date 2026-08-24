"""
Temporal object pairing – two selectable strategies.

Strategy A – "sam3_tracker" (default):
  1. Sam3Tracker runs on the current frame with past-frame bboxes as prompts
     → returns TrackedObject list (past_detection_id + updated bbox + score)
  2. pair_by_tracking() assigns status for every object:
     ┌───────────────┬───────────────────────────────────────────────────┐
     │ status        │ condition                                         │
     ├───────────────┼───────────────────────────────────────────────────┤
     │ "matched"     │ Sam3Tracker found the past object in current frame │
     │ "new"         │ current detection has no matching tracked past obj │
     │ "disappeared" │ past object was NOT returned by Sam3Tracker        │
     └───────────────┴───────────────────────────────────────────────────┘

Strategy B – "similarity":
  No video tracker session needed.
  pair_by_similarity() directly matches current DetectionResult objects to
  past DetectionRecord objects using a weighted score:
    score = (1 - geo_dist / COORDINATE_MATCH_RADIUS_DEG)
            + SIMILARITY_CLASS_BONUS  (if classes match)
  Greedy assignment (best-score pair first).  Status rules:
     ┌───────────────┬───────────────────────────────────────────────────┐
     │ "matched"     │ current ↔ past pair found within radius           │
     │ "new"         │ current detection – no past match within radius    │
     │ "disappeared" │ past detection – no current match within radius    │
     └───────────────┴───────────────────────────────────────────────────┘

Select strategy via env var:  TRACKING_MODE=sam3_tracker | similarity
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from src.config import (
    CLIP_MODEL_NAME,
    COORDINATE_MATCH_RADIUS_DEG,
    MOVE_DISTANCE_THRESHOLD_DEG,
    STATIC_EXACT_MATCH_DEG,
    SIMILARITY_CLIP_WEIGHT,
    SIMILARITY_MATCH_THRESHOLD,
    SIMILARITY_SIZE_WEIGHT,
)
from src.database.models import DetectionRecord, PairingRecord
from src.database.sensor_db import (
    get_image_record_by_id,
    get_most_recent_past_detections,
    insert_detections_bulk,
)
from src.detection.sam3_detector import DetectionResult, TrackedObject

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Minimum bbox IoU to accept a TrackedObject ↔ DetectionResult match
IOU_MATCH_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Static (fixed-position) object classes – physically immovable between frames.
# These include all permanent ground infrastructure whose lat/lon cannot change.
#
# Pairing rules:
#   1. ONLY pair two detections whose geo distance ≤ STATIC_EXACT_MATCH_DEG
#      (≈11 m – same-point constraint; allows for satellite-image projection
#      errors while rejecting clearly different structures).
#      Any pair beyond this threshold is treated as two distinct objects.
#   2. When one side is missing a detection, attempt cross-image similarity:
#      crop the same geo region from both images and compare with CLIP.
#      If similarity ≥ _STATIC_SIM_THRESHOLD → synthesise a DetectionRecord
#      for the missing frame, insert into sensor DB, and pair as "matched".
# ---------------------------------------------------------------------------
_STATIC_CLASSES: frozenset = frozenset({
    "civilian building",
    "command post",
    "fuel storage",
    "supply depot",
    "military building",
    "radar installation",   # 고정 레이더 시설 – 이동 불가
})
_STATIC_SIM_THRESHOLD: float = 0.5   # CLIP cosine similarity threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 타임존 정보를 제거해 naive datetime으로 변환
def _naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Strip timezone info so naive/aware datetimes can be compared safely."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None)


# 두 위경도 좌표 간 유클리드 거리 계산
def _geo_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Simple Euclidean distance in degrees between two lat/lon points."""
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


# 두 탐지 객체의 bbox 크기 유사도 반환
def _size_similarity(a: "DetectionResult", b: "DetectionResult") -> float:
    """두 탐지 객체의 크기 유사도 (0~1).

    mask_area_px 가 유효하면 세그멘테이션 면적 사용, 없으면 bbox 면적 사용.
    min/max 비율이므로 크기가 같을수록 1.0, 차이가 클수록 0에 가까워진다.
    """
    # 탐지 객체의 면적(픽셀)을 반환
    def _area(det: "DetectionResult") -> float:
        if det.mask_area_px and det.mask_area_px > 0:
            return det.mask_area_px
        return max((det.bbox_x2 - det.bbox_x1) * (det.bbox_y2 - det.bbox_y1), 1e-6)

    area_a = _area(a)
    area_b = _area(b)
    return min(area_a, area_b) / max(area_a, area_b)


# 두 bbox의 IoU(교집합/합집합 비율) 계산
def _bbox_iou(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


# ---------------------------------------------------------------------------
# FOV (Field-of-View) coverage helpers
#
# Status semantics:
#   "new"                  – current detection IS within past image FOV but no past match
#                            (truly new object in the overlap zone)
#   "disappeared"          – past detection IS within current image FOV but no current match
#                            (object was there and is now gone in the overlap zone)
#   "past_not_included"    – current detection is OUTSIDE the past image FOV
#                            (past imagery did not cover this area; cannot determine if new)
#   "current_not_included" – past detection is OUTSIDE the current image FOV
#                            (current imagery does not cover this area; cannot determine if gone)
# ---------------------------------------------------------------------------

# 이미지 ID로 FOV 위경도 범위 조회
def _get_image_fov_bounds(image_id: Optional[str]) -> Optional[tuple]:
    """
    Return (lat_min, lat_max, lon_min, lon_max) for a single ImageRecord.
    Preferred over _collect_fov_bounds() when the image ID is known directly —
    works even when the detection list for that image is empty.
    Returns None when image_id is None or the record has no valid bounds.
    """
    if not image_id:
        return None
    rec = get_image_record_by_id(image_id)
    if rec is None or None in (rec.lat_min, rec.lat_max, rec.lon_min, rec.lon_max):
        return None
    return (rec.lat_min, rec.lat_max, rec.lon_min, rec.lon_max)


# 탐지 목록에서 이미지 FOV 통합 범위 반환
def _collect_fov_bounds(detections: list) -> Optional[tuple]:
    """
    Return the union of geographic bounds (lat_min, lat_max, lon_min, lon_max)
    across all ImageRecords referenced by the detections.
    Returns None when no valid bounds can be determined.
    """
    lat_min_all, lat_max_all = float("inf"), float("-inf")
    lon_min_all, lon_max_all = float("inf"), float("-inf")
    found = False
    seen: set = set()
    for det in detections:
        iid = getattr(det, "image_id", None)
        if not iid or iid in seen:
            continue
        seen.add(iid)
        rec = get_image_record_by_id(iid)
        if rec is None or None in (rec.lat_min, rec.lat_max, rec.lon_min, rec.lon_max):
            continue
        lat_min_all = min(lat_min_all, rec.lat_min)
        lat_max_all = max(lat_max_all, rec.lat_max)
        lon_min_all = min(lon_min_all, rec.lon_min)
        lon_max_all = max(lon_max_all, rec.lon_max)
        found = True
    return (lat_min_all, lat_max_all, lon_min_all, lon_max_all) if found else None


# 좌표가 FOV 범위 내에 있는지 확인
def _in_fov(lat: float, lon: float, bounds: Optional[tuple]) -> bool:
    """
    Return True if (lat, lon) lies within the given bounds.
    When bounds is None (ImageRecord metadata unavailable), returns True
    conservatively so objects are not silently dropped.
    """
    if bounds is None:
        return True
    lat_min, lat_max, lon_min, lon_max = bounds
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


# ---------------------------------------------------------------------------
# Geo-bounds validity guard
# ---------------------------------------------------------------------------

# 이미지의 geo bounds가 있어야 lat/lon이 신뢰 가능 (없으면 모두 lat_center 폴백)
def _image_has_geo_bounds(image_id: Optional[str]) -> bool:
    """이미지 레코드에 lat_min/lat_max/lon_min/lon_max 가 모두 있으면 True.
    없으면 pixel_to_geo 가 lat_center/lon_center 를 반환하므로 Step 0 geo-match 불가.
    """
    if not image_id:
        return False
    rec = get_image_record_by_id(image_id)
    return (
        rec is not None
        and None not in (rec.lat_min, rec.lat_max, rec.lon_min, rec.lon_max)
    )


# ---------------------------------------------------------------------------
# Duplicate-pairing guard
# ---------------------------------------------------------------------------

# Status priority: matched (양쪽 프레임 모두 확인) > changed (같은 위치, 구조 변화)
# > disappeared (현재 FOV 내 소실) > current_not_included / past_not_included (FOV 밖) > new (단독 출현)
_STATUS_PRIORITY: dict = {
    "matched":              0,
    "changed":              1,   # 고정 시설물 같은 위치, CLIP 유사도 < 임계값 → 구조 변화
    "disappeared":          2,
    "past_not_included":    3,
    "current_not_included": 4,
    "new":                  5,
}


# 중복 페어링 레코드를 우선순위 기준으로 제거
def _dedup_pairing_records(records: List[PairingRecord]) -> List[PairingRecord]:
    """동일 past_detection_id(또는 current_detection_id)에 대해 페어링 레코드가 중복으로
    생성된 경우 우선순위가 높은(더 정보량이 많은) 레코드 하나만 남긴다.

    한 객체에 두 개의 상태가 붙는 버그를 최종 방어선으로 차단한다.
    """
    # 레코드 상태의 우선순위 값 반환
    def _pri(r: PairingRecord) -> int:
        return _STATUS_PRIORITY.get(r.status, 99)

    # past_detection_id 기준 중복 제거
    past_seen: dict = {}   # past_id → best PairingRecord
    for r in records:
        pid = r.past_detection_id
        if pid is None:
            continue
        if pid not in past_seen or _pri(r) < _pri(past_seen[pid]):
            if pid in past_seen:
                logger.warning(
                    "[Pairing] 중복 past_detection_id=%s: '%s' 를 '%s' 로 교체",
                    pid[:8], past_seen[pid].status, r.status,
                )
            past_seen[pid] = r

    # current_detection_id 기준 중복 제거 (new / past_not_included 에서 발생 가능)
    cur_seen: dict = {}    # cur_id → best PairingRecord
    for r in records:
        cid = r.current_detection_id
        if cid is None:
            continue
        if cid not in cur_seen or _pri(r) < _pri(cur_seen[cid]):
            if cid in cur_seen:
                logger.warning(
                    "[Pairing] 중복 current_detection_id=%s: '%s' 를 '%s' 로 교체",
                    cid[:8], cur_seen[cid].status, r.status,
                )
            cur_seen[cid] = r

    # 두 기준 모두 통과한 레코드만 유지
    kept = []
    for r in records:
        pid = r.past_detection_id
        cid = r.current_detection_id
        keep = True
        if pid is not None and past_seen.get(pid) is not r:
            keep = False
        if cid is not None and cur_seen.get(cid) is not r:
            keep = False
        if keep:
            kept.append(r)

    return kept


# ---------------------------------------------------------------------------
# Static-object cross-image similarity helpers
# ---------------------------------------------------------------------------

# 지리 bbox를 이미지 픽셀 좌표로 변환
def _geo_bbox_on_image(
    lat_top: float, lon_left: float, lat_bottom: float, lon_right: float,
    img_w: int, img_h: int,
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    padding: int = 4,
) -> Optional[tuple]:
    """
    Convert geographic bbox (lat_top, lon_left, lat_bottom, lon_right) to
    pixel bbox (x1, y1, x2, y2) on an image defined by its geo bounds.

    pixel_to_geo formula (from image_loader.py):
        lat = lat_max - (cy / h) * (lat_max - lat_min)
        lon = lon_min + (cx / w) * (lon_max - lon_min)
    Inverse:
        cy = (lat_max - lat) / (lat_max - lat_min) * h
        cx = (lon - lon_min) / (lon_max - lon_min) * w

    Returns None if the bbox falls entirely outside the image bounds.
    """
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span <= 0 or lon_span <= 0:
        return None

    x1 = int((lon_left  - lon_min) / lon_span * img_w) - padding
    x2 = int((lon_right - lon_min) / lon_span * img_w) + padding
    y1 = int((lat_max - lat_top)    / lat_span * img_h) - padding
    y2 = int((lat_max - lat_bottom) / lat_span * img_h) + padding

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# 탐지 결과의 지리 bbox 좌표 반환
def _det_geo_bbox(det, img_record) -> Optional[tuple]:
    """
    Return the geographic bounding box of a detection as
    (lat_top, lon_left, lat_bottom, lon_right).

    Works with both DetectionResult (dataclass) and DetectionRecord (ORM)
    because both expose the same field names.
    """
    w = img_record.det_width
    h = img_record.det_height
    if not w or not h:
        return None
    lat_min = img_record.lat_min
    lat_max = img_record.lat_max
    lon_min = img_record.lon_min
    lon_max = img_record.lon_max
    if None in (lat_min, lat_max, lon_min, lon_max):
        return None

    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min

    lat_top    = lat_max - (det.bbox_y1 / h) * lat_span
    lat_bottom = lat_max - (det.bbox_y2 / h) * lat_span
    lon_left   = lon_min + (det.bbox_x1 / w) * lon_span
    lon_right  = lon_min + (det.bbox_x2 / w) * lon_span
    return lat_top, lon_left, lat_bottom, lon_right


# 이미지 ID로 PIL 이미지를 초해상도 적용 후 로드
def _load_pil_for_image_id(image_id: str) -> Optional["PILImage"]:
    """Load PIL image (with super-resolution) for a given image_id."""
    from PIL import Image as PILImage
    from src.detection.super_resolution import super_resolve

    record = get_image_record_by_id(image_id)
    if record is None:
        return None
    from pathlib import Path as _Path
    p = _Path(record.image_path)
    if not p.exists():
        return None
    img = PILImage.open(p).convert("RGB")
    arr = super_resolve(np.array(img, dtype=np.uint8))
    return PILImage.fromarray(arr)


# 두 이미지 크롭 간 CLIP 코사인 유사도 반환
def _clip_similarity_crops(crop_a: "PILImage", crop_b: "PILImage") -> float:
    """Return CLIP cosine similarity between two PIL crops. Falls back to 0.0 on error."""
    try:
        embeds = _clip_embedder.embed([crop_a, crop_b])   # (2, D)
        return float(embeds[0] @ embeds[1])
    except Exception as exc:
        logger.warning(f"[StaticCrossCheck] CLIP similarity failed: {exc}")
        return 0.0


# 정적 객체의 교차 이미지 유사도 비교 후 합성 탐지 반환
def _static_cross_check(
    det,                    # DetectionResult or DetectionRecord – the side WITH a bbox
    img_rec_with: "ImageRecord",
    pil_with: "PILImage",
    img_rec_without: "ImageRecord",
    pil_without: "PILImage",
    capture_time_without,   # datetime for the "missing" frame
    session_id: str,
    source_type: str,
) -> Optional[DetectionRecord]:
    """
    For a static-class object detected in one frame but absent in the other:
    1. Crop the detected bbox from pil_with.
    2. Crop the same geographic region from pil_without.
    3. Compute CLIP cosine similarity.
    4. If similarity ≥ _STATIC_SIM_THRESHOLD, create and insert a synthetic
       DetectionRecord for the "missing" frame, then return it.
    5. Otherwise return None.
    """
    geo_box = _det_geo_bbox(det, img_rec_with)
    if geo_box is None:
        return None
    lat_top, lon_left, lat_bottom, lon_right = geo_box

    # Crop from the "with bbox" image
    w_with = img_rec_with.det_width or pil_with.width
    h_with = img_rec_with.det_height or pil_with.height
    coords_with = _geo_bbox_on_image(
        lat_top, lon_left, lat_bottom, lon_right,
        w_with, h_with,
        img_rec_with.lat_min, img_rec_with.lat_max,
        img_rec_with.lon_min, img_rec_with.lon_max,
    )
    if coords_with is None:
        return None
    cx1, cy1, cx2, cy2 = coords_with
    crop_with = pil_with.crop((cx1, cy1, cx2, cy2))

    # Crop the same geo region from the "without bbox" image
    w_wo = img_rec_without.det_width or pil_without.width
    h_wo = img_rec_without.det_height or pil_without.height
    coords_wo = _geo_bbox_on_image(
        lat_top, lon_left, lat_bottom, lon_right,
        w_wo, h_wo,
        img_rec_without.lat_min, img_rec_without.lat_max,
        img_rec_without.lon_min, img_rec_without.lon_max,
    )
    if coords_wo is None:
        return None
    wx1, wy1, wx2, wy2 = coords_wo
    crop_wo = pil_without.crop((wx1, wy1, wx2, wy2))

    sim = _clip_similarity_crops(crop_with, crop_wo)
    logger.info(
        f"[StaticCrossCheck] {det.object_class}  sim={sim:.3f}  "
        f"threshold={_STATIC_SIM_THRESHOLD}"
    )
    if sim < _STATIC_SIM_THRESHOLD:
        return None

    # Synthesise a DetectionRecord for the "without" frame
    from src.detection.image_loader import pixel_to_geo
    from src.database.models import ImageRecord as _IRModel

    # Pixel-space bbox on the "without" image
    cx_px = (wx1 + wx2) / 2.0
    cy_px = (wy1 + wy2) / 2.0
    # Geo centre from the known detection (more accurate than re-projecting)
    lat_c = det.lat
    lon_c = det.lon

    synthetic = DetectionRecord(
        image_id=img_rec_without.id,
        detection_time=capture_time_without,
        object_class=det.object_class,
        object_class_index=getattr(det, "object_class_index", None),
        confidence=det.confidence,
        bbox_x1=float(wx1),
        bbox_y1=float(wy1),
        bbox_x2=float(wx2),
        bbox_y2=float(wy2),
        lat=lat_c,
        lon=lon_c,
        mask_rle=None,
        mask_area_px=None,
        source_type="synthetic",   # 재페어링 시 식별·삭제 가능하도록 고정 마커
        session_id=session_id,
    )
    ids = insert_detections_bulk([synthetic])
    synthetic.id = ids[0]
    logger.info(
        f"[StaticCrossCheck] Synthetic DetectionRecord inserted  "
        f"id={synthetic.id}  class={synthetic.object_class}  "
        f"sim={sim:.3f}  image_id={img_rec_without.id[:8]}"
    )
    return synthetic


# ---------------------------------------------------------------------------
# Main pairing function
# ---------------------------------------------------------------------------

# SAM3 트래커 결과로 페어링 레코드 목록 생성
def pair_by_tracking(
    tracked_objects: List[TrackedObject],
    current_detections: List[DetectionResult],
    past_detections: List[DetectionRecord],
    current_capture_time: datetime,
    past_capture_time: Optional[datetime],
    region_lat: float,
    region_lon: float,
    session_id: str,
    source_type: str = "satellite",
    current_image_id: Optional[str] = None,
    past_image_id: Optional[str] = None,
) -> List[PairingRecord]:
    """
    Build PairingRecord list using Sam3Tracker object IDs.

    Status assignment is explicitly based on metadata.json capture times:
      "new"         – object only in the LATER frame (current_capture_time)
      "matched"     – object in BOTH the earlier frame (past_capture_time) AND
                      the later frame (current_capture_time)
      "disappeared" – object only in the EARLIER frame (past_capture_time)

    Args:
        tracked_objects:      Output of SAM3Detector.track_objects() –
                              past objects confirmed present in current frame.
        current_detections:   Sam3Model detections for the current frame
                              (from sensor DB, already stored).
        past_detections:      DetectionRecord list for the most-recent past frame.
        current_capture_time: Capture timestamp of the current (later) image.
        past_capture_time:    Capture timestamp of the past (earlier) image batch.
        region_lat/lon:       Geographic centre of the region.
        session_id:           Pipeline run UUID.
        source_type:          "satellite" | "drone"

    Returns:
        List of PairingRecord ORM objects ready for bulk insertion.
    """
    now = datetime.utcnow()

    # Guard: current frame must be strictly later than past frame.
    # Use naive datetimes to avoid offset-naive vs offset-aware comparison errors.
    if past_capture_time is not None and _naive(current_capture_time) <= _naive(past_capture_time):
        logger.warning(
            f"[Pairing/tracker] Skipping region ({region_lat:.4f}, {region_lon:.4f}): "
            f"current_capture_time ({current_capture_time}) is not later than "
            f"past_capture_time ({past_capture_time}). Check metadata.json ordering."
        )
        return []

    # Only pair against past detections that truly belong to the earlier frame.
    if past_capture_time is not None:
        past_detections = [
            d for d in past_detections
            if _naive(d.detection_time) <= _naive(past_capture_time)
        ]

    # ── 중복 제거: 동일 detection_id가 리스트에 두 번 이상 있으면 다중 pairing 발생 ──
    _seen_cur: set = set()
    current_detections = [
        d for d in current_detections
        if d.detection_id not in _seen_cur and not _seen_cur.add(d.detection_id)
    ]
    _seen_past: set = set()
    past_detections = [
        d for d in past_detections
        if d.id not in _seen_past and not _seen_past.add(d.id)
    ]

    past_by_id = {d.id: d for d in past_detections}

    # IDs that Sam3Tracker confirmed are present in the current frame
    tracked_past_ids = {t.past_detection_id for t in tracked_objects}

    matched_current_ids: set = set()
    matched_past_ids: set = set()   # 동일 과거 객체가 여러 TrackedObject에 중복 등장 방지
    pairing_records: List[PairingRecord] = []

    # ------------------------------------------------------------------
    # 0. Static class geo-based pre-matching
    #    고정 시설물(건물·레이더 등)은 물리적으로 이동 불가.
    #    SAM3 tracker(픽셀 IoU)보다 먼저 위경도 근접도 기준으로 매칭한다.
    #    같은 클래스이고 geo 거리 ≤ STATIC_EXACT_MATCH_DEG(≈111m) 인 쌍 중 가장
    #    가까운 쌍부터 greedy 할당. 임계값 초과 쌍은 절대 매칭하지 않는다.
    #
    #    ※ 이미지에 lat_min/lat_max/lon_min/lon_max 가 없으면 pixel_to_geo 가
    #      lat_center/lon_center 를 반환 → 모든 탐지 객체가 동일 좌표로 보여
    #      무관한 건물들까지 오매칭된다. bounds 확인 후 신뢰할 수 없으면 건너뜀.
    # ------------------------------------------------------------------
    static_cur_list  = [d for d in current_detections if d.object_class.lower() in _STATIC_CLASSES]
    static_past_list = [d for d in past_detections    if d.object_class.lower() in _STATIC_CLASSES]

    # 과거 이미지 ID: past_image_id 파라미터 우선, 없으면 첫 번째 past 탐지에서 취득
    _past_iid_t = past_image_id or next(
        (getattr(d, "image_id", None) for d in past_detections if getattr(d, "image_id", None)),
        None,
    )
    _step0_t_geo_ok = _image_has_geo_bounds(current_image_id) and _image_has_geo_bounds(_past_iid_t)
    if not _step0_t_geo_ok:
        logger.debug(
            "[Pairing/tracker] Step 0 건너뜀: 이미지 geo bounds 없음 "
            "(cur=%s, past=%s)", current_image_id, _past_iid_t
        )

    if static_cur_list and static_past_list and _step0_t_geo_ok:
        geo_candidates = []
        for ci, cur in enumerate(static_cur_list):
            for pi, past in enumerate(static_past_list):
                if cur.object_class.lower() != past.object_class.lower():
                    continue
                dist = _geo_distance(cur.lat, cur.lon, past.lat, past.lon)
                if dist <= STATIC_EXACT_MATCH_DEG:
                    geo_candidates.append((dist, ci, pi))
        geo_candidates.sort()          # 가장 가까운 쌍부터
        geo_matched_ci: set = set()
        geo_matched_pi: set = set()
        _s0t_cur_pil = None                          # 현재 프레임 PIL (1회 로드)
        _s0t_past_pil_cache: Dict[str, Optional] = {}  # image_id → PIL
        for dist, ci, pi in geo_candidates:
            if ci in geo_matched_ci or pi in geo_matched_pi:
                continue
            cur  = static_cur_list[ci]
            past = static_past_list[pi]
            geo_matched_ci.add(ci)
            geo_matched_pi.add(pi)
            matched_current_ids.add(cur.detection_id)
            matched_past_ids.add(past.id)
            # 현재·과거 크롭 PIL 로드 (캐시 활용)
            if _s0t_cur_pil is None:
                _ciid = getattr(cur, "image_id", None) or current_image_id or ""
                _s0t_cur_pil = _load_pil_image(_ciid) if _ciid else None
            _piid = getattr(past, "image_id", None) or ""
            if _piid not in _s0t_past_pil_cache:
                _s0t_past_pil_cache[_piid] = _load_pil_image(_piid) if _piid else None
            _s0t_past_pil = _s0t_past_pil_cache.get(_piid)
            # CLIP 유사도로 변화 여부 판단: "matched" (변화 없음) or "changed" (구조 변화)
            _st = _static_pair_status(cur, past, cur_pil=_s0t_cur_pil, past_pil=_s0t_past_pil)
            pairing_records.append(PairingRecord(
                pairing_time=now,
                lat_center=region_lat, lon_center=region_lon,
                current_detection_id=cur.detection_id,
                current_object_class=cur.object_class,
                current_confidence=cur.confidence,
                current_lat=cur.lat, current_lon=cur.lon,
                current_capture_time=current_capture_time,
                current_bbox={"x1": cur.bbox_x1, "y1": cur.bbox_y1,
                              "x2": cur.bbox_x2, "y2": cur.bbox_y2},
                past_detection_id=past.id,
                past_object_class=past.object_class,
                past_confidence=past.confidence,
                past_lat=past.lat, past_lon=past.lon,
                past_capture_time=past_capture_time,
                past_bbox={"x1": past.bbox_x1, "y1": past.bbox_y1,
                           "x2": past.bbox_x2, "y2": past.bbox_y2},
                status=_st,
                source_type=source_type,
                session_id=session_id,
            ))


    for tracked in tracked_objects:
        # 트래커가 동일 past_detection_id를 중복 반환하는 경우 첫 번째만 처리
        if tracked.past_detection_id in matched_past_ids:
            continue
        past = past_by_id.get(tracked.past_detection_id)

        # 비교 기준 클래스: past DB 레코드가 있으면 그 클래스, 없으면 tracker가 기억한 클래스
        expected_cls = (
            past.object_class if past else tracked.past_object_class or ""
        ).lower()

        # 건물류(static)는 Step 0에서 geo 기반으로 이미 매칭 완료.
        # 트래커의 픽셀 IoU 매칭을 적용하지 않는다.
        if expected_cls in _STATIC_CLASSES:
            continue

        matched_past_ids.add(tracked.past_detection_id)

        # Find the current Sam3Model detection with highest IoU overlap
        best_current: Optional[DetectionResult] = None
        best_iou = IOU_MATCH_THRESHOLD

        for cur in current_detections:
            if cur.detection_id in matched_current_ids:
                continue
            # 동일 클래스인 경우만 같은 객체 후보로 허용
            if cur.object_class.lower() != expected_cls:
                continue
            iou = _bbox_iou(
                tracked.bbox_x1, tracked.bbox_y1,
                tracked.bbox_x2, tracked.bbox_y2,
                cur.bbox_x1, cur.bbox_y1,
                cur.bbox_x2, cur.bbox_y2,
            )
            if iou > best_iou:
                best_iou = iou
                best_current = cur

        # IoU fallback 1: bbox 중심점 픽셀 거리 (트래커 예측 위치와 실제 탐지 위치 차이)
        # 이동체가 프레임 간 bbox 크기의 2배 이상 이동한 경우 IoU 가 0으로 떨어질 수 있음.
        if best_current is None:
            tracked_cx = (tracked.bbox_x1 + tracked.bbox_x2) / 2
            tracked_cy = (tracked.bbox_y1 + tracked.bbox_y2) / 2
            tracked_w  = max(tracked.bbox_x2 - tracked.bbox_x1, 1.0)
            tracked_h  = max(tracked.bbox_y2 - tracked.bbox_y1, 1.0)
            max_px_dist = max(tracked_w, tracked_h) * 2.0
            best_px_dist = max_px_dist
            for cur in current_detections:
                if cur.detection_id in matched_current_ids:
                    continue
                if cur.object_class.lower() != expected_cls:
                    continue
                cur_cx = (cur.bbox_x1 + cur.bbox_x2) / 2
                cur_cy = (cur.bbox_y1 + cur.bbox_y2) / 2
                dist = ((cur_cx - tracked_cx) ** 2 + (cur_cy - tracked_cy) ** 2) ** 0.5
                if dist < best_px_dist:
                    best_px_dist = dist
                    best_current = cur
            if best_current is not None:
                logger.debug(
                    "[Pairing/tracker] IoU 매칭 실패 → 픽셀 중심점 폴백 "
                    "cls=%s dist=%.1f", expected_cls, best_px_dist,
                )

        # IoU fallback 2: 위경도 근접도 (픽셀 폴백도 실패한 경우)
        if best_current is None and past is not None and past.lat is not None:
            best_geo_dist = COORDINATE_MATCH_RADIUS_DEG
            for cur in current_detections:
                if cur.detection_id in matched_current_ids:
                    continue
                if cur.object_class.lower() != expected_cls:
                    continue
                if cur.lat is None or cur.lon is None:
                    continue
                geo_dist = _geo_distance(cur.lat, cur.lon, past.lat, past.lon)
                if geo_dist < best_geo_dist:
                    best_geo_dist = geo_dist
                    best_current = cur
            if best_current is not None:
                logger.debug(
                    "[Pairing/tracker] 픽셀 폴백 실패 → 위경도 폴백 "
                    "cls=%s geo_dist=%.5f", expected_cls, best_geo_dist,
                )

        if best_current is not None:
            matched_current_ids.add(best_current.detection_id)

        status = "matched"

        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            # current side – may be None if Sam3Model missed this object
            current_detection_id=best_current.detection_id if best_current else None,
            current_object_class=best_current.object_class if best_current else tracked.past_object_class,
            current_confidence=best_current.confidence if best_current else tracked.score,
            current_lat=best_current.lat if best_current else None,
            current_lon=best_current.lon if best_current else None,
            current_capture_time=current_capture_time,
            current_bbox={
                "x1": best_current.bbox_x1, "y1": best_current.bbox_y1,
                "x2": best_current.bbox_x2, "y2": best_current.bbox_y2,
            } if best_current else {
                "x1": tracked.bbox_x1, "y1": tracked.bbox_y1,
                "x2": tracked.bbox_x2, "y2": tracked.bbox_y2,
            },
            # past side
            past_detection_id=tracked.past_detection_id,
            past_object_class=past.object_class if past else tracked.past_object_class,
            past_confidence=past.confidence if past else None,
            past_lat=past.lat if past else None,
            past_lon=past.lon if past else None,
            past_capture_time=past_capture_time,
            past_bbox={
                "x1": past.bbox_x1, "y1": past.bbox_y1,
                "x2": past.bbox_x2, "y2": past.bbox_y2,
            } if past else None,
            status=status,
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    # ------------------------------------------------------------------
    # 2. "new" / "past_not_included"
    #    현재 탐지 객체가 과거 이미지 FOV 밖이면 "past_not_included"
    # ------------------------------------------------------------------
    # 과거 이미지 ID가 직접 제공된 경우 ImageRecord에서 FOV를 바로 조회
    # (past_detections가 비어 있어도 정확한 FOV 판단 가능)
    past_fov = _get_image_fov_bounds(past_image_id) or _collect_fov_bounds(past_detections)
    for cur in current_detections:
        if cur.detection_id in matched_current_ids:
            continue
        cur_status = "new" if _in_fov(cur.lat, cur.lon, past_fov) else "past_not_included"
        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            current_detection_id=cur.detection_id,
            current_object_class=cur.object_class,
            current_confidence=cur.confidence,
            current_lat=cur.lat,
            current_lon=cur.lon,
            current_capture_time=current_capture_time,
            current_bbox={
                "x1": cur.bbox_x1, "y1": cur.bbox_y1,
                "x2": cur.bbox_x2, "y2": cur.bbox_y2,
            },
            status=cur_status,
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    # ------------------------------------------------------------------
    # 3. "disappeared" / "current_not_included"
    #    과거 탐지 객체가 현재 이미지 FOV 밖이면 "current_not_included"
    # ------------------------------------------------------------------
    # 현재 이미지 ID가 직접 제공된 경우 ImageRecord에서 FOV를 바로 조회
    # (current_detections가 비어 있어도 정확한 FOV 판단 가능)
    cur_fov = _get_image_fov_bounds(current_image_id) or _collect_fov_bounds(current_detections)
    for past in past_detections:
        # tracked_past_ids 가 아닌 matched_past_ids 로 guard:
        # Step 0 에서 geo-matched 된 객체(matched_past_ids 에 추가, tracked_past_ids 에는 없을 수 있음)도
        # 제대로 스킵해야 중복 레코드 생성을 막을 수 있다.
        if past.id in matched_past_ids:
            continue
        past_status = "disappeared" if _in_fov(past.lat, past.lon, cur_fov) else "current_not_included"
        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            past_detection_id=past.id,
            past_object_class=past.object_class,
            past_confidence=past.confidence,
            past_lat=past.lat,
            past_lon=past.lon,
            past_capture_time=past_capture_time,
            past_bbox={
                "x1": past.bbox_x1, "y1": past.bbox_y1,
                "x2": past.bbox_x2, "y2": past.bbox_y2,
            },
            status=past_status,
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    # ------------------------------------------------------------------
    # 4. Static-class cross-image check for "new" and "disappeared"
    #    When a building is unmatched (new or disappeared), compare crops
    #    from both images.  If similarity ≥ threshold → synthetic detection
    #    inserted and pair recorded as "matched".
    # ------------------------------------------------------------------
    # Static-class objects that are "new" (within past FOV, no geo-match) or
    # "disappeared" (within current FOV, no geo-match): attempt cross-image
    # similarity check.  A synthetic DetectionRecord is inserted for the
    # missing frame only when CLIP cosine similarity ≥ _STATIC_SIM_THRESHOLD.
    static_new = [p for p in pairing_records
                  if p.status == "new"
                  and (p.current_object_class or "").lower() in _STATIC_CLASSES]
    static_disappeared = [p for p in pairing_records
                          if p.status == "disappeared"
                          and (p.past_object_class or "").lower() in _STATIC_CLASSES]

    if static_new or static_disappeared:
        # Collect distinct image_ids from current detections to load PILs
        cur_image_ids = list({d.image_id for d in current_detections if d.image_id})
        past_image_ids = list({d.image_id for d in past_detections if d.image_id})
        cur_img_cache: Dict[str, Optional] = {}
        past_img_cache: Dict[str, Optional] = {}

        # 이미지 레코드와 PIL을 캐시에서 조회하거나 로드
        def _get_img_rec_pil(iid: str, cache: dict):
            if iid not in cache:
                rec = get_image_record_by_id(iid)
                pil = _load_pil_for_image_id(iid) if rec else None
                cache[iid] = (rec, pil)
            return cache[iid]

        # For "new" static: past side is missing → load current PIL, check past image
        for pr in static_new:
            if pr.current_detection_id is None:
                continue
            # find the DetectionResult
            cur_det = next(
                (d for d in current_detections if d.detection_id == pr.current_detection_id),
                None,
            )
            if cur_det is None or not cur_det.image_id:
                continue
            cur_rec, cur_pil = _get_img_rec_pil(cur_det.image_id, cur_img_cache)
            if cur_rec is None or cur_pil is None:
                continue
            # Pick the past image that covers the same region.
            # Fall back to the function-level past_image_id parameter when the
            # area has no past detections at all (that is why the object is "new").
            past_iid = past_image_ids[0] if past_image_ids else past_image_id
            if not past_iid:
                continue
            past_rec, past_pil = _get_img_rec_pil(past_iid, past_img_cache)
            if past_rec is None or past_pil is None:
                continue
            synth = _static_cross_check(
                det=cur_det,
                img_rec_with=cur_rec,
                pil_with=cur_pil,
                img_rec_without=past_rec,
                pil_without=past_pil,
                capture_time_without=past_capture_time or current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.past_detection_id = synth.id
                pr.past_object_class = synth.object_class
                pr.past_confidence = synth.confidence
                pr.past_lat = synth.lat
                pr.past_lon = synth.lon
                pr.past_capture_time = synth.detection_time
                pr.past_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

        # For "disappeared" static: current side is missing → load past PIL, check current image
        for pr in static_disappeared:
            if pr.past_detection_id is None:
                continue
            past_det = next(
                (d for d in past_detections if d.id == pr.past_detection_id),
                None,
            )
            if past_det is None or not past_det.image_id:
                continue
            past_rec, past_pil = _get_img_rec_pil(past_det.image_id, past_img_cache)
            if past_rec is None or past_pil is None:
                continue
            # Fall back to the function-level current_image_id parameter when the
            # area has no current detections at all (that is why the object is "disappeared").
            cur_iid = cur_image_ids[0] if cur_image_ids else current_image_id
            if not cur_iid:
                continue
            cur_rec, cur_pil = _get_img_rec_pil(cur_iid, cur_img_cache)
            if cur_rec is None or cur_pil is None:
                continue
            synth = _static_cross_check(
                det=past_det,
                img_rec_with=past_rec,
                pil_with=past_pil,
                img_rec_without=cur_rec,
                pil_without=cur_pil,
                capture_time_without=current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.current_detection_id = synth.id
                pr.current_object_class = synth.object_class
                pr.current_confidence = synth.confidence
                pr.current_lat = synth.lat
                pr.current_lon = synth.lon
                pr.current_capture_time = synth.detection_time
                pr.current_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

    # ------------------------------------------------------------------
    # 5. Movable-class cross-image check for "new" and "disappeared"
    #    Same crop-and-compare logic as static cross-check, but applied to
    #    non-building classes (tanks, vehicles, aircraft, etc.).
    # ------------------------------------------------------------------
    movable_new        = [p for p in pairing_records
                          if p.status == "new"
                          and (p.current_object_class or "").lower() not in _STATIC_CLASSES]
    movable_disappeared = [p for p in pairing_records
                           if p.status == "disappeared"
                           and (p.past_object_class or "").lower() not in _STATIC_CLASSES]

    if movable_new or movable_disappeared:
        mov_cur_img_cache: Dict[str, Optional] = {}
        mov_past_img_cache: Dict[str, Optional] = {}

        # 현재 이미지 레코드·PIL을 캐시에서 조회하거나 로드
        def _get_mov_img_rec_pil_cur(iid: str):
            if iid not in mov_cur_img_cache:
                rec = get_image_record_by_id(iid)
                pil = _load_pil_for_image_id(iid) if rec else None
                mov_cur_img_cache[iid] = (rec, pil)
            return mov_cur_img_cache[iid]

        # 과거 이미지 레코드·PIL을 캐시에서 조회하거나 로드
        def _get_mov_img_rec_pil_past(iid: str):
            if iid not in mov_past_img_cache:
                rec = get_image_record_by_id(iid)
                pil = _load_pil_for_image_id(iid) if rec else None
                mov_past_img_cache[iid] = (rec, pil)
            return mov_past_img_cache[iid]

        mov_cur_image_ids  = list({d.image_id for d in current_detections if d.image_id})
        mov_past_image_ids = list({d.image_id for d in past_detections if d.image_id})

        for pr in movable_new:
            if pr.current_detection_id is None:
                continue
            cur_det = next(
                (d for d in current_detections if d.detection_id == pr.current_detection_id),
                None,
            )
            if cur_det is None or not cur_det.image_id:
                continue
            cur_rec, cur_pil = _get_mov_img_rec_pil_cur(cur_det.image_id)
            if cur_rec is None or cur_pil is None:
                continue
            if not mov_past_image_ids:
                continue
            past_iid = mov_past_image_ids[0]
            past_rec, past_pil = _get_mov_img_rec_pil_past(past_iid)
            if past_rec is None or past_pil is None:
                continue
            synth = _static_cross_check(
                det=cur_det,
                img_rec_with=cur_rec,
                pil_with=cur_pil,
                img_rec_without=past_rec,
                pil_without=past_pil,
                capture_time_without=past_capture_time or current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.past_detection_id = synth.id
                pr.past_object_class = synth.object_class
                pr.past_confidence = synth.confidence
                pr.past_lat = synth.lat
                pr.past_lon = synth.lon
                pr.past_capture_time = synth.detection_time
                pr.past_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

        for pr in movable_disappeared:
            if pr.past_detection_id is None:
                continue
            past_det = next(
                (d for d in past_detections if d.id == pr.past_detection_id),
                None,
            )
            if past_det is None or not past_det.image_id:
                continue
            past_rec, past_pil = _get_mov_img_rec_pil_past(past_det.image_id)
            if past_rec is None or past_pil is None:
                continue
            if not mov_cur_image_ids:
                continue
            cur_iid = mov_cur_image_ids[0]
            cur_rec, cur_pil = _get_mov_img_rec_pil_cur(cur_iid)
            if cur_rec is None or cur_pil is None:
                continue
            synth = _static_cross_check(
                det=past_det,
                img_rec_with=past_rec,
                pil_with=past_pil,
                img_rec_without=cur_rec,
                pil_without=cur_pil,
                capture_time_without=current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.current_detection_id = synth.id
                pr.current_object_class = synth.object_class
                pr.current_confidence = synth.confidence
                pr.current_lat = synth.lat
                pr.current_lon = synth.lon
                pr.current_capture_time = synth.detection_time
                pr.current_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

    pairing_records = _dedup_pairing_records(pairing_records)

    logger.info(
        f"[Pairing/tracker] ({region_lat:.4f}, {region_lon:.4f}): "
        f"{sum(1 for p in pairing_records if p.status == 'matched')} matched  "
        f"{sum(1 for p in pairing_records if p.status == 'new')} new  "
        f"{sum(1 for p in pairing_records if p.status == 'disappeared')} disappeared"
    )
    return pairing_records


# ---------------------------------------------------------------------------
# Strategy B: SAM mask crop + CLIP embedding similarity
# ---------------------------------------------------------------------------

class _CLIPEmbedder:
    """Lazy-loaded CLIP image encoder (module-level singleton)."""

    def __init__(self):
        self._model = None
        self._processor = None

    # CLIP 모델과 프로세서를 지연 로드
    def _load(self):
        from transformers import AutoModel, AutoProcessor
        logger.info(f"[CLIPEmbedder] Loading {CLIP_MODEL_NAME} ...")
        self._processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
        self._model = AutoModel.from_pretrained(CLIP_MODEL_NAME)
        self._model.eval()
        logger.info("[CLIPEmbedder] Ready.")

    # PIL 크롭 리스트의 L2 정규화 임베딩 반환
    def embed(self, crops: list) -> np.ndarray:
        """
        Compute L2-normalised image embeddings for a list of PIL crops.
        Supports both CLIPModel (get_image_features) and plain ViTModel
        (pooler_output / CLS token from last_hidden_state).
        Returns float32 array of shape (N, D).
        """
        import torch

        if self._model is None:
            self._load()

        inputs = self._processor(images=crops, return_tensors="pt", padding=True)
        # CLIPProcessor may include text keys – keep only vision inputs for ViT
        vision_inputs = {k: v for k, v in inputs.items() if k == "pixel_values"}

        with torch.no_grad():
            if hasattr(self._model, "get_image_features"):
                # CLIPModel path — vision inputs only to avoid text-side interference
                feats = self._model.get_image_features(**vision_inputs)
                # newer transformers may return a model output object instead of tensor
                if not isinstance(feats, torch.Tensor):
                    if hasattr(feats, "image_embeds"):
                        feats = feats.image_embeds
                    elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                        feats = feats.pooler_output
                    else:
                        feats = feats.last_hidden_state[:, 0, :]
            else:
                # ViTModel / CLIPVisionModel path
                out = self._model(**vision_inputs)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    feats = out.pooler_output                       # (N, D)
                else:
                    feats = out.last_hidden_state[:, 0, :]          # CLS token

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


_clip_embedder = _CLIPEmbedder()


# JSON RLE 문자열을 마스크 배열로 디코드
def _decode_rle(mask_rle_json: str) -> Optional[np.ndarray]:
    """Decode a JSON RLE string produced by sam2_detector._encode_rle()."""
    try:
        data = json.loads(mask_rle_json)
        h, w = data["shape"]
        flat: list = []
        for val, count in data["rle"]:
            flat.extend([val] * count)
        return np.array(flat, dtype=bool).reshape(h, w)
    except Exception:
        return None


# SAM 마스크로 객체 영역만 크롭하여 반환
def _mask_crop(image: "PILImage", x1: float, y1: float, x2: float, y2: float,
               mask_rle: Optional[str], padding: int = 4) -> Optional["PILImage"]:
    """
    Crop the object region from *image* using the SAM segmentation mask.

    If mask_rle is available the background is zeroed out before cropping,
    so CLIP sees only the object pixels.  Falls back to a plain bbox crop
    when mask_rle is absent or cannot be decoded.
    Returns None when the bounding box is degenerate (zero or negative area).
    """
    from PIL import Image as PILImage

    iw, ih = image.size
    bx1 = max(0, int(x1) - padding)
    by1 = max(0, int(y1) - padding)
    bx2 = min(iw, int(x2) + padding)
    by2 = min(ih, int(y2) + padding)

    if bx2 <= bx1 or by2 <= by1:
        logger.warning(f"[MaskCrop] Degenerate bbox ({x1},{y1},{x2},{y2}); skipping crop.")
        return None

    if mask_rle:
        mask = _decode_rle(mask_rle)
        if mask is not None and mask.shape == (ih, iw):
            arr = np.array(image)
            arr[~mask] = 0          # zero out background
            crop = arr[by1:by2, bx1:bx2]
            return PILImage.fromarray(crop)

    # Fallback: plain bbox crop
    return image.crop((bx1, by1, bx2, by2))


# 센서DB image_id로 디스크에서 PIL 이미지 로드
def _load_pil_image(image_id: str) -> Optional["PILImage"]:
    """Load a PIL image from disk given a sensor-DB image_id.

    Applies super_resolve so that the returned image dimensions match the
    resolution at which bbox coordinates were originally computed.
    """
    from PIL import Image as PILImage
    from src.detection.super_resolution import super_resolve

    record = get_image_record_by_id(image_id)
    if record is None:
        logger.warning(f"[CLIPSim] ImageRecord not found for image_id={image_id}")
        return None
    path = Path(record.image_path)
    if not path.exists():
        logger.warning(f"[CLIPSim] Image file missing: {path}")
        return None
    img = PILImage.open(path).convert("RGB")
    arr = super_resolve(np.array(img, dtype=np.uint8))
    return PILImage.fromarray(arr)


# 같은 위치 고정 시설물 쌍의 CLIP 유사도로 변화 여부를 "matched"/"changed"로 판단
def _static_pair_status(
    cur,                                           # DetectionResult – 현재 프레임
    past,                                          # DetectionRecord  – 과거 프레임
    cur_pil: Optional["PILImage"] = None,
    past_pil: Optional["PILImage"] = None,
) -> str:
    """
    Compute CLIP cosine similarity between same-location static facility crops.
    Returns "matched" (no structural change) when sim >= _STATIC_SIM_THRESHOLD,
    "changed" (structural alteration detected) when sim < threshold.
    Falls back conservatively to "matched" when images cannot be loaded.
    """
    try:
        if cur_pil is None:
            _cur_iid = getattr(cur, "image_id", None) or ""
            cur_pil = _load_pil_image(_cur_iid) if _cur_iid else None
        if past_pil is None:
            _past_iid = getattr(past, "image_id", None) or ""
            past_pil = _load_pil_image(_past_iid) if _past_iid else None
        if cur_pil is None or past_pil is None:
            logger.warning("[StaticPairStatus] PIL 이미지 로드 실패 → 'matched' 기본값 사용")
            return "matched"
        crop_cur = _mask_crop(
            cur_pil, cur.bbox_x1, cur.bbox_y1, cur.bbox_x2, cur.bbox_y2,
            getattr(cur, "mask_rle", None),
        )
        crop_past = _mask_crop(
            past_pil, past.bbox_x1, past.bbox_y1, past.bbox_x2, past.bbox_y2,
            getattr(past, "mask_rle", None),
        )
        if crop_cur is None or crop_past is None:
            return "matched"
        sim = _clip_similarity_crops(crop_cur, crop_past)
        status = "matched" if sim >= _STATIC_SIM_THRESHOLD else "changed"
        logger.info(
            f"[StaticPair] {cur.object_class}  sim={sim:.3f}  "
            f"threshold={_STATIC_SIM_THRESHOLD}  → {status}"
        )
        return status
    except Exception as exc:
        logger.warning(f"[StaticPairStatus] 유사도 계산 오류: {exc} → 'matched' 기본값 사용")
        return "matched"


# 탐지 목록에 대한 CLIP 임베딩 배열 생성
def _compute_embeddings(
    detections: list,
    image: Optional["PILImage"],
    image_cache: Dict[str, Optional["PILImage"]],
    is_current: bool,
) -> np.ndarray:
    """
    Produce CLIP embeddings for *detections*.

    For current-frame detections the caller supplies *image* directly.
    For past-frame detections the image is looked up by image_id and cached
    in *image_cache*.  Objects whose source image cannot be loaded receive a
    zero vector (they will never win the greedy assignment).

    Returns float32 array of shape (N, D).
    """
    crops = []
    valid_indices = []

    for i, det in enumerate(detections):
        if is_current:
            src_image = image
        else:
            iid = det.image_id
            if iid not in image_cache:
                image_cache[iid] = _load_pil_image(iid)
            src_image = image_cache[iid]

        if src_image is None:
            continue

        crop = _mask_crop(
            src_image,
            det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2,
            det.mask_rle,
        )
        if crop is None:
            continue
        crops.append(crop)
        valid_indices.append(i)

    if not crops:
        raise ValueError(f"No valid crops for any of the {len(detections)} detections; cannot compute embeddings.")

    embeds_valid = _clip_embedder.embed(crops)          # (len(valid), D)
    D = embeds_valid.shape[1]
    embeds = np.zeros((len(detections), D), dtype=np.float32)
    for out_idx, orig_idx in enumerate(valid_indices):
        embeds[orig_idx] = embeds_valid[out_idx]
    return embeds


# CLIP 유사도 기반 현재·과거 탐지 페어링 수행
def pair_by_similarity(
    current_detections: List[DetectionResult],
    past_detections: List[DetectionRecord],
    current_image: "PILImage",
    current_capture_time: datetime,
    past_capture_time: Optional[datetime],
    region_lat: float,
    region_lon: float,
    session_id: str,
    source_type: str = "satellite",
    current_image_id: Optional[str] = None,
    past_image_id: Optional[str] = None,
) -> List[PairingRecord]:
    """
    Build PairingRecord list by matching current detections to past detections
    using SAM mask crop + CLIP embedding cosine similarity.

    Status assignment is explicitly based on metadata.json capture times:
      "new"         – object only in the LATER frame (current_capture_time):
                      unmatched current detections
      "matched"     – object in BOTH the earlier frame (past_capture_time) AND
                      the later frame (current_capture_time)
      "disappeared" – object only in the EARLIER frame (past_capture_time):
                      unmatched past detections

    Scoring (CLIP cosine similarity; geo proximity as fallback):
        score = cosine_similarity(cur_embed, past_embed)   ∈ [-1, 1]
    Greedy assignment by descending score; pair accepted only when
    score > SIMILARITY_MATCH_THRESHOLD.

    Args:
        current_detections:  DetectionResult list for the current (later) frame.
        past_detections:     DetectionRecord list for the most-recent past frame.
        current_image:       PIL image of the current frame (for crop + embed).
        current_capture_time: Capture timestamp of the current (later) image.
        past_capture_time:   Capture timestamp of the past (earlier) image batch.
        region_lat/lon:      Geographic centre of the region.
        session_id:          Pipeline run UUID.
        source_type:         "satellite" | "drone"

    Returns:
        List of PairingRecord ORM objects ready for bulk insertion.
    """
    now = datetime.utcnow()
    pairing_records: List[PairingRecord] = []

    # Guard: current frame must be strictly later than past frame.
    # Use naive datetimes to avoid offset-naive vs offset-aware comparison errors.
    if past_capture_time is not None and _naive(current_capture_time) <= _naive(past_capture_time):
        logger.warning(
            f"[Pairing/similarity] Skipping region ({region_lat:.4f}, {region_lon:.4f}): "
            f"current_capture_time ({current_capture_time}) is not later than "
            f"past_capture_time ({past_capture_time}). Check metadata.json ordering."
        )
        return []

    # Only pair against past detections that truly belong to the earlier frame.
    # detection_time is set to meta.capture_time at ingestion time.
    if past_capture_time is not None:
        past_detections = [
            d for d in past_detections
            if _naive(d.detection_time) <= _naive(past_capture_time)
        ]

    # ── 중복 제거: 동일 detection_id가 두 번 이상 있으면 다중 pairing 발생 ──
    _seen_cur_s: set = set()
    current_detections = [
        d for d in current_detections
        if d.detection_id not in _seen_cur_s and not _seen_cur_s.add(d.detection_id)
    ]
    _seen_past_s: set = set()
    past_detections = [
        d for d in past_detections
        if d.id not in _seen_past_s and not _seen_past_s.add(d.id)
    ]

    # ------------------------------------------------------------------
    # Step 0. Static class geo-based pre-matching
    # CLIP 점수와 무관하게, 같은 클래스 고정 시설물이 STATIC_EXACT_MATCH_DEG(≈111m) 이내이면
    # 가장 가까운 쌍부터 greedy로 matched 페어링 생성.
    # 임계값 초과 쌍은 절대 매칭하지 않는다 (다른 위치의 건물을 동일 건물로 식별하지 않음).
    #
    # ※ 이미지에 lat_min/lat_max/lon_min/lon_max 가 없으면 pixel_to_geo 가
    #   lat_center/lon_center 를 반환 → 모든 탐지 객체가 동일 좌표로 보여
    #   무관한 건물들까지 오매칭된다. bounds 확인 후 신뢰할 수 없으면 건너뜀.
    # ------------------------------------------------------------------
    _s_cur  = [d for d in current_detections if d.object_class.lower() in _STATIC_CLASSES]
    _s_past = [d for d in past_detections    if d.object_class.lower() in _STATIC_CLASSES]
    _geo_pre_cur_ids:  set = set()
    _geo_pre_past_ids: set = set()

    # 과거 이미지 ID: past_image_id 파라미터 우선, 없으면 첫 번째 past 탐지에서 취득
    _past_iid_s = past_image_id or next(
        (getattr(d, "image_id", None) for d in past_detections if getattr(d, "image_id", None)),
        None,
    )
    _step0_s_geo_ok = _image_has_geo_bounds(current_image_id) and _image_has_geo_bounds(_past_iid_s)
    if not _step0_s_geo_ok:
        logger.debug(
            "[Pairing/similarity] Step 0 건너뜀: 이미지 geo bounds 없음 "
            "(cur=%s, past=%s)", current_image_id, _past_iid_s
        )

    if _s_cur and _s_past and _step0_s_geo_ok:
        _gcands = sorted(
            (_geo_distance(c.lat, c.lon, p.lat, p.lon), ci, pi)
            for ci, c in enumerate(_s_cur)
            for pi, p in enumerate(_s_past)
            if c.object_class.lower() == p.object_class.lower()
            and _geo_distance(c.lat, c.lon, p.lat, p.lon) <= STATIC_EXACT_MATCH_DEG
        )
        _gc_seen: set = set()
        _gp_seen: set = set()
        _s0s_past_pil_cache: Dict[str, Optional] = {}  # image_id → PIL
        for _dist, _ci, _pi in _gcands:
            if _ci in _gc_seen or _pi in _gp_seen:
                continue
            _c = _s_cur[_ci]
            _p = _s_past[_pi]
            _gc_seen.add(_ci)
            _gp_seen.add(_pi)
            _geo_pre_cur_ids.add(_c.detection_id)
            _geo_pre_past_ids.add(_p.id)
            # 과거 프레임 PIL 로드 (캐시 활용); 현재 PIL은 함수 인자로 제공됨
            _piid = getattr(_p, "image_id", None) or ""
            if _piid not in _s0s_past_pil_cache:
                _s0s_past_pil_cache[_piid] = _load_pil_image(_piid) if _piid else None
            _s0s_past_pil = _s0s_past_pil_cache.get(_piid)
            # CLIP 유사도로 변화 여부 판단: "matched" (변화 없음) or "changed" (구조 변화)
            _st = _static_pair_status(_c, _p, cur_pil=current_image, past_pil=_s0s_past_pil)
            pairing_records.append(PairingRecord(
                pairing_time=now, lat_center=region_lat, lon_center=region_lon,
                current_detection_id=_c.detection_id,
                current_object_class=_c.object_class,
                current_confidence=_c.confidence,
                current_lat=_c.lat, current_lon=_c.lon,
                current_capture_time=current_capture_time,
                current_bbox={"x1": _c.bbox_x1, "y1": _c.bbox_y1,
                              "x2": _c.bbox_x2, "y2": _c.bbox_y2},
                past_detection_id=_p.id,
                past_object_class=_p.object_class,
                past_confidence=_p.confidence,
                past_lat=_p.lat, past_lon=_p.lon,
                past_capture_time=past_capture_time,
                past_bbox={"x1": _p.bbox_x1, "y1": _p.bbox_y1,
                           "x2": _p.bbox_x2, "y2": _p.bbox_y2},
                status=_st, source_type=source_type, session_id=session_id,
            ))
    # 이미 geo-matched된 탐지는 CLIP 처리에서 제외
    current_detections = [d for d in current_detections if d.detection_id not in _geo_pre_cur_ids]
    past_detections    = [d for d in past_detections    if d.id not in _geo_pre_past_ids]

    if not current_detections or not past_detections:
        # Nothing to match – apply FOV check for each unmatched detection
        _past_fov_early = _get_image_fov_bounds(past_image_id) or _collect_fov_bounds(past_detections)
        _cur_fov_early  = _get_image_fov_bounds(current_image_id) or _collect_fov_bounds(current_detections)
        for cur in current_detections:
            s = "new" if _in_fov(cur.lat, cur.lon, _past_fov_early) else "past_not_included"
            pairing_records.append(PairingRecord(
                pairing_time=now, lat_center=region_lat, lon_center=region_lon,
                current_detection_id=cur.detection_id,
                current_object_class=cur.object_class,
                current_confidence=cur.confidence,
                current_lat=cur.lat, current_lon=cur.lon,
                current_capture_time=current_capture_time,
                current_bbox={"x1": cur.bbox_x1, "y1": cur.bbox_y1,
                              "x2": cur.bbox_x2, "y2": cur.bbox_y2},
                status=s, source_type=source_type, session_id=session_id,
            ))
        for past in past_detections:
            s = "disappeared" if _in_fov(past.lat, past.lon, _cur_fov_early) else "current_not_included"
            pairing_records.append(PairingRecord(
                pairing_time=now, lat_center=region_lat, lon_center=region_lon,
                past_detection_id=past.id,
                past_object_class=past.object_class,
                past_confidence=past.confidence,
                past_lat=past.lat, past_lon=past.lon,
                past_capture_time=past_capture_time,
                past_bbox={"x1": past.bbox_x1, "y1": past.bbox_y1,
                           "x2": past.bbox_x2, "y2": past.bbox_y2},
                status=s, source_type=source_type, session_id=session_id,
            ))
        return pairing_records

    # ------------------------------------------------------------------
    # Compute CLIP embeddings (batch per frame)
    # ------------------------------------------------------------------
    past_image_cache: Dict[str, Optional] = {}
    try:
        cur_embeds = _compute_embeddings(
            current_detections, current_image, {}, is_current=True,
        )                                                    # (N, D)
        past_embeds = _compute_embeddings(
            past_detections, None, past_image_cache, is_current=False,
        )                                                    # (M, D)
        # Cosine similarity matrix (N, M); embeddings are already L2-normalised
        clip_sim_matrix = cur_embeds @ past_embeds.T        # (N, M)
    except Exception as exc:
        logger.warning(f"[CLIPSim] Embedding failed ({exc}); falling back to geo-only scoring.")
        clip_sim_matrix = None

    # ------------------------------------------------------------------
    # Build scored candidate pairs (all combinations; no geo hard filter)
    # Score = CLIP cosine similarity when available, otherwise geo score.
    # ------------------------------------------------------------------
    candidates = []   # (score, cur_idx, past_idx)
    for ci, cur in enumerate(current_detections):
        for pi, past in enumerate(past_detections):
            # 동일 클래스인 경우만 같은 객체 후보로 허용
            if cur.object_class.lower() != past.object_class.lower():
                continue
            # 고정 시설물은 Step 0 에서 정확 위치 쌍을 처리하고 리스트에서 제거한다.
            # candidate loop 에 남은 static 객체는 Step 0 임계값 초과(≈111m+) 또는
            # 형상 변화로 centroid 가 크게 이동한 경우이므로, geo 가드 없이 CLIP 으로만
            # 최선 쌍을 선택한다. Gale-Shapley 가 유사도 최고 쌍부터 greedy 할당한다.
            if clip_sim_matrix is not None:
                base_score = float(clip_sim_matrix[ci, pi])
            else:
                # Fallback: geo proximity score when CLIP is unavailable
                geo_dist = _geo_distance(cur.lat, cur.lon, past.lat, past.lon)
                base_score = max(0.0, 1.0 - geo_dist / COORDINATE_MATCH_RADIUS_DEG)
            size_sim = _size_similarity(cur, past)
            score = (1.0 - SIMILARITY_SIZE_WEIGHT) * base_score + SIMILARITY_SIZE_WEIGHT * size_sim
            candidates.append((score, ci, pi))

    # Iterative per-ci best-pi assignment (Gale-Shapley / deferred-acceptance):
    #
    #  1. ci는 threshold 초과 pi 중 최고 점수 pi에 제안한다.
    #  2. pi가 비어 있으면 → 잠정 매칭.
    #  3. pi에 이미 다른 ci가 매칭돼 있으면:
    #       - 새 ci의 점수가 더 높으면  → 새 ci가 pi를 획득, 기존 ci는 차선 pi로 재시도.
    #       - 새 ci의 점수가 낮으면     → 새 ci는 차선 pi로 재시도.
    #  4. 시도할 pi가 더 없으면 → ci는 "new" 처리.
    #
    # ci_ptr 이 항상 단조 증가하므로 유한 횟수 안에 반드시 종료된다.
    # ------------------------------------------------------------------

    # 각 ci의 후보 목록 (점수 내림차순, threshold 초과만)
    ci_candidates: dict[int, list] = {}
    for score, ci, pi in candidates:
        if score > SIMILARITY_MATCH_THRESHOLD:
            ci_candidates.setdefault(ci, []).append((score, pi))
    for ci in ci_candidates:
        ci_candidates[ci].sort(key=lambda x: x[0], reverse=True)

    ci_ptr:  dict[int, int]             = {ci: 0 for ci in ci_candidates}
    pi_match: dict[int, tuple[float, int]] = {}   # pi -> (score, ci)

    unmatched: set = set(ci_candidates)
    while unmatched:
        ci    = unmatched.pop()
        ptr   = ci_ptr[ci]
        if ptr >= len(ci_candidates[ci]):
            continue                              # 후보 소진 → "new"
        score, pi  = ci_candidates[ci][ptr]
        ci_ptr[ci] = ptr + 1

        if pi not in pi_match:
            pi_match[pi] = (score, ci)            # pi 비어있음 → 즉시 매칭
        elif score > pi_match[pi][0]:
            _, displaced = pi_match[pi]
            pi_match[pi] = (score, ci)            # 더 높은 점수 → ci가 pi 획득
            unmatched.add(displaced)              # 밀려난 ci 재시도
        else:
            unmatched.add(ci)                     # 낮은 점수 → ci 차선으로 재시도

    matched_cur_ids: set = set()
    matched_past_ids: set = set()
    pairs: list = []   # [(DetectionResult, DetectionRecord)]
    for pi, (_, ci) in pi_match.items():
        cur  = current_detections[ci]
        past = past_detections[pi]
        matched_cur_ids.add(cur.detection_id)
        matched_past_ids.add(past.id)
        pairs.append((cur, past))

    # ------------------------------------------------------------------
    # 1. "matched" / "changed"
    #    CLIP 후보 루프로 매칭된 static 쌍도 _static_pair_status() 로 변화 판정.
    # ------------------------------------------------------------------
    _s1_past_pil_cache: Dict[str, Optional] = {}
    for cur, past in pairs:
        if cur.object_class.lower() in _STATIC_CLASSES:
            _piid = getattr(past, "image_id", None) or ""
            if _piid not in _s1_past_pil_cache:
                _s1_past_pil_cache[_piid] = _load_pil_image(_piid) if _piid else None
            status = _static_pair_status(
                cur, past,
                cur_pil=current_image,
                past_pil=_s1_past_pil_cache.get(_piid),
            )
        else:
            status = "matched"
        pairing_records.append(PairingRecord(
            pairing_time=now, lat_center=region_lat, lon_center=region_lon,
            current_detection_id=cur.detection_id,
            current_object_class=cur.object_class,
            current_confidence=cur.confidence,
            current_lat=cur.lat, current_lon=cur.lon,
            current_capture_time=current_capture_time,
            current_bbox={"x1": cur.bbox_x1, "y1": cur.bbox_y1,
                          "x2": cur.bbox_x2, "y2": cur.bbox_y2},
            past_detection_id=past.id,
            past_object_class=past.object_class,
            past_confidence=past.confidence,
            past_lat=past.lat, past_lon=past.lon,
            past_capture_time=past_capture_time,
            past_bbox={"x1": past.bbox_x1, "y1": past.bbox_y1,
                       "x2": past.bbox_x2, "y2": past.bbox_y2},
            status=status, source_type=source_type, session_id=session_id,
        ))

    # ------------------------------------------------------------------
    # 2. "new" / "past_not_included"
    # ------------------------------------------------------------------
    past_fov_sim = _get_image_fov_bounds(past_image_id) or _collect_fov_bounds(past_detections)
    for cur in current_detections:
        if cur.detection_id in matched_cur_ids:
            continue
        s = "new" if _in_fov(cur.lat, cur.lon, past_fov_sim) else "past_not_included"
        pairing_records.append(PairingRecord(
            pairing_time=now, lat_center=region_lat, lon_center=region_lon,
            current_detection_id=cur.detection_id,
            current_object_class=cur.object_class,
            current_confidence=cur.confidence,
            current_lat=cur.lat, current_lon=cur.lon,
            current_capture_time=current_capture_time,
            current_bbox={"x1": cur.bbox_x1, "y1": cur.bbox_y1,
                          "x2": cur.bbox_x2, "y2": cur.bbox_y2},
            status=s, source_type=source_type, session_id=session_id,
        ))

    # ------------------------------------------------------------------
    # 3. "disappeared" / "current_not_included"
    # ------------------------------------------------------------------
    cur_fov_sim = _get_image_fov_bounds(current_image_id) or _collect_fov_bounds(current_detections)
    for past in past_detections:
        if past.id in matched_past_ids:
            continue
        s = "disappeared" if _in_fov(past.lat, past.lon, cur_fov_sim) else "current_not_included"
        pairing_records.append(PairingRecord(
            pairing_time=now, lat_center=region_lat, lon_center=region_lon,
            past_detection_id=past.id,
            past_object_class=past.object_class,
            past_confidence=past.confidence,
            past_lat=past.lat, past_lon=past.lon,
            past_capture_time=past_capture_time,
            past_bbox={"x1": past.bbox_x1, "y1": past.bbox_y1,
                       "x2": past.bbox_x2, "y2": past.bbox_y2},
            status=s, source_type=source_type, session_id=session_id,
        ))

    # ------------------------------------------------------------------
    # 4. Static-class cross-image check for unmatched "new" / "disappeared"
    # ------------------------------------------------------------------
    # Static-class objects that are "new" (within past FOV, no geo-match) or
    # "disappeared" (within current FOV, no geo-match): attempt cross-image
    # similarity check.  A synthetic DetectionRecord is inserted for the
    # missing frame only when CLIP cosine similarity ≥ _STATIC_SIM_THRESHOLD.
    static_new_sim = [p for p in pairing_records
                      if p.status == "new"
                      and (p.current_object_class or "").lower() in _STATIC_CLASSES]
    static_dis_sim = [p for p in pairing_records
                      if p.status == "disappeared"
                      and (p.past_object_class or "").lower() in _STATIC_CLASSES]

    if static_new_sim or static_dis_sim:
        cur_img_cache2: Dict[str, Optional] = {}
        past_img_cache2: Dict[str, Optional] = {}
        past_image_ids2 = list({d.image_id for d in past_detections if d.image_id})
        cur_image_ids2  = list({d.image_id for d in current_detections if d.image_id})

        # 이미지 레코드·PIL을 캐시에서 조회하거나 로드
        def _rec_pil(iid, cache):
            if iid not in cache:
                rec = get_image_record_by_id(iid)
                pil = _load_pil_for_image_id(iid) if rec else None
                cache[iid] = (rec, pil)
            return cache[iid]

        # "new" static: past side missing → synthesise in past image
        for pr in static_new_sim:
            if pr.current_detection_id is None:
                continue
            cur_det = next(
                (d for d in current_detections if d.detection_id == pr.current_detection_id),
                None,
            )
            if cur_det is None or not cur_det.image_id:
                continue
            cur_rec, cur_pil = _rec_pil(cur_det.image_id, cur_img_cache2)
            if cur_rec is None or cur_pil is None:
                continue
            # Fall back to function-level past_image_id when area has no past detections.
            past_iid = past_image_ids2[0] if past_image_ids2 else past_image_id
            if not past_iid:
                continue
            past_rec, past_pil = _rec_pil(past_iid, past_img_cache2)
            if past_rec is None or past_pil is None:
                continue
            synth = _static_cross_check(
                det=cur_det,
                img_rec_with=cur_rec,
                pil_with=cur_pil,
                img_rec_without=past_rec,
                pil_without=past_pil,
                capture_time_without=past_capture_time or current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.past_detection_id = synth.id
                pr.past_object_class = synth.object_class
                pr.past_confidence = synth.confidence
                pr.past_lat = synth.lat
                pr.past_lon = synth.lon
                pr.past_capture_time = synth.detection_time
                pr.past_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

        # "disappeared" static: current side missing → synthesise in current image
        for pr in static_dis_sim:
            if pr.past_detection_id is None:
                continue
            past_det = next(
                (d for d in past_detections if d.id == pr.past_detection_id),
                None,
            )
            if past_det is None or not past_det.image_id:
                continue
            past_rec, past_pil = _rec_pil(past_det.image_id, past_img_cache2)
            if past_rec is None or past_pil is None:
                continue
            # current image: get image_id from any current detection or function parameter.
            # current_detections may be empty when all current static detections were
            # geo-matched in Step 0, so fall back to current_image_id parameter.
            cur_iid2 = next(
                (d.image_id for d in current_detections if d.image_id), None
            ) or current_image_id
            if cur_iid2 is None:
                continue
            cur_rec, cur_pil = _rec_pil(cur_iid2, cur_img_cache2)
            if cur_rec is None or cur_pil is None:
                continue
            synth = _static_cross_check(
                det=past_det,
                img_rec_with=past_rec,
                pil_with=past_pil,
                img_rec_without=cur_rec,
                pil_without=cur_pil,
                capture_time_without=current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.current_detection_id = synth.id
                pr.current_object_class = synth.object_class
                pr.current_confidence = synth.confidence
                pr.current_lat = synth.lat
                pr.current_lon = synth.lon
                pr.current_capture_time = synth.detection_time
                pr.current_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

    # ------------------------------------------------------------------
    # 5. Movable-class cross-image check for "new" and "disappeared"
    #    Same crop-and-compare logic as static cross-check, but applied to
    #    non-building classes (tanks, vehicles, aircraft, etc.).
    # ------------------------------------------------------------------
    mov_new2        = [p for p in pairing_records
                       if p.status == "new"
                       and (p.current_object_class or "").lower() not in _STATIC_CLASSES]
    mov_disappeared2 = [p for p in pairing_records
                        if p.status == "disappeared"
                        and (p.past_object_class or "").lower() not in _STATIC_CLASSES]

    if mov_new2 or mov_disappeared2:
        mov_cur_img_cache2: Dict[str, Optional] = {}
        mov_past_img_cache2: Dict[str, Optional] = {}

        # 현재 이미지 레코드·PIL을 캐시에서 조회하거나 로드
        def _rec_pil_mov_cur(iid: str):
            if iid not in mov_cur_img_cache2:
                rec = get_image_record_by_id(iid)
                pil = _load_pil_for_image_id(iid) if rec else None
                mov_cur_img_cache2[iid] = (rec, pil)
            return mov_cur_img_cache2[iid]

        # 과거 이미지 레코드·PIL을 캐시에서 조회하거나 로드
        def _rec_pil_mov_past(iid: str):
            if iid not in mov_past_img_cache2:
                rec = get_image_record_by_id(iid)
                pil = _load_pil_for_image_id(iid) if rec else None
                mov_past_img_cache2[iid] = (rec, pil)
            return mov_past_img_cache2[iid]

        mov2_cur_image_ids  = list({d.image_id for d in current_detections if d.image_id})
        mov2_past_image_ids = list({d.image_id for d in past_detections if d.image_id})

        for pr in mov_new2:
            if pr.current_detection_id is None:
                continue
            cur_det = next(
                (d for d in current_detections if d.detection_id == pr.current_detection_id),
                None,
            )
            if cur_det is None or not cur_det.image_id:
                continue
            cur_rec, cur_pil = _rec_pil_mov_cur(cur_det.image_id)
            if cur_rec is None or cur_pil is None:
                continue
            if not mov2_past_image_ids:
                continue
            past_iid = mov2_past_image_ids[0]
            past_rec, past_pil = _rec_pil_mov_past(past_iid)
            if past_rec is None or past_pil is None:
                continue
            synth = _static_cross_check(
                det=cur_det,
                img_rec_with=cur_rec,
                pil_with=cur_pil,
                img_rec_without=past_rec,
                pil_without=past_pil,
                capture_time_without=past_capture_time or current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.past_detection_id = synth.id
                pr.past_object_class = synth.object_class
                pr.past_confidence = synth.confidence
                pr.past_lat = synth.lat
                pr.past_lon = synth.lon
                pr.past_capture_time = synth.detection_time
                pr.past_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

        for pr in mov_disappeared2:
            if pr.past_detection_id is None:
                continue
            past_det = next(
                (d for d in past_detections if d.id == pr.past_detection_id),
                None,
            )
            if past_det is None or not past_det.image_id:
                continue
            past_rec, past_pil = _rec_pil_mov_past(past_det.image_id)
            if past_rec is None or past_pil is None:
                continue
            if not mov2_cur_image_ids:
                continue
            cur_iid2 = mov2_cur_image_ids[0]
            cur_rec, cur_pil = _rec_pil_mov_cur(cur_iid2)
            if cur_rec is None or cur_pil is None:
                continue
            synth = _static_cross_check(
                det=past_det,
                img_rec_with=past_rec,
                pil_with=past_pil,
                img_rec_without=cur_rec,
                pil_without=cur_pil,
                capture_time_without=current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.current_detection_id = synth.id
                pr.current_object_class = synth.object_class
                pr.current_confidence = synth.confidence
                pr.current_lat = synth.lat
                pr.current_lon = synth.lon
                pr.current_capture_time = synth.detection_time
                pr.current_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

    pairing_records = _dedup_pairing_records(pairing_records)

    logger.info(
        f"[Pairing/similarity] ({region_lat:.4f}, {region_lon:.4f}): "
        f"{sum(1 for p in pairing_records if p.status == 'matched')} matched  "
        f"{sum(1 for p in pairing_records if p.status == 'new')} new  "
        f"{sum(1 for p in pairing_records if p.status == 'disappeared')} disappeared"
    )
    return pairing_records
