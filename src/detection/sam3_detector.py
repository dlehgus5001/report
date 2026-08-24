"""
SAM3 object detector and tracker for aerial imagery.

Two modes:
  SAM3Detector.detect()           - sam3 image model: text-prompted segmentation
                                    Used for initial detection on each frame.
  SAM3Detector.track_objects()    - sam3 video predictor: session-based tracking
                                    Takes past-frame bboxes + class as prompts,
                                    returns which past objects are still present.

Pairing flow (in temporal_pairing.py):
  1. track_objects() → TrackedObject list (past_detection_id + new bbox + score)
  2. pair_by_tracking() matches TrackedObjects to current detections by IoU
     → status = matched / new / disappeared  (ID-based, not coordinate-based)

SAM3 Installation:
  git clone https://github.com/facebookresearch/sam3.git
  cd sam3 && pip install -e .
  hf auth login  # HuggingFace 접근 토큰 필요
"""

import gc
import json
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.config import (
    DETECTION_CONFIDENCE_THRESHOLD,
    LARGE_OBJECT_CLASSES,
    MAX_BBOX_AREA_RATIO,
    MAX_BBOX_AREA_RATIO_LARGE,
    MILITARY_OBJECT_CLASSES,
    NMS_IOU_THRESHOLD,
    SAM3_CHECKPOINT,
    SAM3_DEVICE,
    SAM3_MASK_SCORE_THRESHOLD,
    SAM3_MODEL_NAME,
    TILE_ENABLED,
    TILE_SIZE,
    TILE_OVERLAP,
    TILE_NMS_IOU,
    TILE_NMS_IOMIN,
    TILE_MERGE_GAP,
    TILE_MULTISCALE,
    TILE_MEDIUM_SCALE,
    TILE_MEDIUM_SIZE,
    TILE_MEDIUM_OVERLAP,
)
from src.detection.image_loader import ImageMeta, LoadedImage, pixel_to_geo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Output of SAM3 detection for a single object in an image."""
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detection_time: Optional[datetime] = None
    image_id: str = ""

    object_class: str = ""
    object_class_index: int = 0
    confidence: float = 0.0

    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0

    lat: float = 0.0
    lon: float = 0.0

    mask_rle: Optional[str] = None
    mask_area_px: float = 0.0

    source_type: str = "satellite"


@dataclass
class TrackedObject:
    """
    Result of SAM3 video predictor tracking for one past detection.

    score reflects how confidently the tracker found the object in the
    current frame; objects below DETECTION_CONFIDENCE_THRESHOLD are
    considered disappeared.
    """
    past_detection_id: str      # ID of the DetectionRecord in the sensor DB
    past_object_class: str
    bbox_x1: float              # updated bbox in current-frame pixel coords
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    score: float                # SAM3 mask IOU / presence score


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

# 마스크를 RLE 형식 JSON 문자열로 인코딩
def _encode_rle(mask: np.ndarray) -> str:
    flat = mask.flatten().astype(np.uint8).tolist()
    rle: list = []
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1]:
            count += 1
        else:
            rle.append([flat[i - 1], count])
            count = 1
    rle.append([flat[-1], count])
    return json.dumps({"shape": list(mask.shape), "rle": rle})


# 이진 마스크에서 바운딩 박스 좌표 추출
def _mask_to_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return float(x_min), float(y_min), float(x_max), float(y_max)


# 마스크에서 최대 연결 성분만 남겨 정밀도 향상
def _tighten_mask(mask: np.ndarray) -> np.ndarray:
    """
    마스크에서 가장 큰 연결 성분(connected component)만 남긴다.
    scipy가 없을 경우 원본 마스크를 그대로 반환.
    """
    if not mask.any():
        return mask
    try:
        from scipy.ndimage import label as ndimage_label
        labeled, n = ndimage_label(mask)
        if n == 0:
            return mask
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        return labeled == sizes.argmax()
    except ImportError:
        return mask


# 슬라이딩 윈도우 타일 좌표 목록 생성
def _tile_coords(img_w: int, img_h: int,
                 tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """슬라이딩 윈도우 타일 좌표 목록 (x1, y1, x2, y2) 반환.

    SAM3 입력 비율(1:1 / 1008×1008) 유지를 위해 모든 타일을
    tile_size × tile_size 정방형으로 고정한다.
    경계에 걸리는 마지막 타일은 클리핑하지 않고 시작점을 안쪽으로
    당겨(shift-back) 정방형을 유지한다.
    이미지 자체가 tile_size 보다 작으면 전체 이미지를 단일 타일로 반환.
    """
    # 이미지가 타일보다 작은 경우 전체를 단일 타일로
    if img_w <= tile_size and img_h <= tile_size:
        return [(0, 0, img_w, img_h)]

    stride = max(tile_size - overlap, 1)
    seen: set = set()
    tiles: List[Tuple[int, int, int, int]] = []

    y = 0
    while True:
        # shift-back: 하단 경계를 넘지 않도록 y1 조정 → 항상 tile_size 높이 유지
        y1 = min(y, max(0, img_h - tile_size))
        y2 = y1 + tile_size

        x = 0
        while True:
            # shift-back: 우측 경계를 넘지 않도록 x1 조정 → 항상 tile_size 너비 유지
            x1 = min(x, max(0, img_w - tile_size))
            x2 = x1 + tile_size

            coord = (x1, y1, x2, y2)
            if coord not in seen:
                seen.add(coord)
                tiles.append(coord)

            if x + tile_size >= img_w:
                break
            x += stride

        if y + tile_size >= img_h:
            break
        y += stride

    return tiles


# IoU와 IoMin 기반 NMS로 중복 탐지 제거
def _nms_detections(
    results: List[DetectionResult],
    iou_threshold: float,
    iomin_threshold: float = 0.6,
) -> List[DetectionResult]:
    """NMS with IoU + IoMin suppression.

    IoU 만으로는 크기가 크게 다른 두 박스(예: 스케일1 대형 vs 스케일3 소형)가
    동일 객체를 중복 탐지할 때 제거가 안 된다.
    IoMin = intersection / min(area_a, area_b) 를 함께 사용해
    한 박스가 다른 박스 안에 충분히 포함되면 낮은 신뢰도 쪽을 억제한다.
    """
    if not results:
        return results
    results = sorted(results, key=lambda r: r.confidence, reverse=True)
    kept: List[DetectionResult] = []
    for candidate in results:
        suppress = False
        for kept_r in kept:
            ix1 = max(candidate.bbox_x1, kept_r.bbox_x1)
            iy1 = max(candidate.bbox_y1, kept_r.bbox_y1)
            ix2 = min(candidate.bbox_x2, kept_r.bbox_x2)
            iy2 = min(candidate.bbox_y2, kept_r.bbox_y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter == 0.0:
                continue
            area_c = (candidate.bbox_x2 - candidate.bbox_x1) * (candidate.bbox_y2 - candidate.bbox_y1)
            area_k = (kept_r.bbox_x2 - kept_r.bbox_x1) * (kept_r.bbox_y2 - kept_r.bbox_y1)
            iou   = inter / (area_c + area_k - inter + 1e-6)
            iomin = inter / (min(area_c, area_k) + 1e-6)
            if iou > iou_threshold or iomin > iomin_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)
    return kept


# 동일 클래스 내 완전 포함 박스를 큰 박스로 병합
def _merge_contained_detections(results: List[DetectionResult]) -> List[DetectionResult]:
    """동일 클래스에서 한 박스가 다른 박스에 완전히 포함된 경우 큰 박스 하나로 병합한다.

    완전 포함 판정: IoMin = intersection / min(area_a, area_b) >= 0.99
    (부동소수점 오차를 허용하기 위해 1.0 이 아닌 0.99 사용)

    결과 박스는 두 박스 중 넓이가 큰 (포함하는) 박스이며, 신뢰도는 두 박스의 최대값.
    NMS 이후에도 남아 있는 동일 클래스 중첩 박스를 최종 정리하는 용도로 사용된다.
    """
    if not results:
        return results

    merged = True
    detections = list(results)
    while merged:
        merged = False
        used = [False] * len(detections)
        kept: List[DetectionResult] = []
        for i, a in enumerate(detections):
            if used[i]:
                continue
            for j in range(i + 1, len(detections)):
                b = detections[j]
                if used[j]:
                    continue
                if a.object_class.lower() != b.object_class.lower():
                    continue
                ix1 = max(a.bbox_x1, b.bbox_x1)
                iy1 = max(a.bbox_y1, b.bbox_y1)
                ix2 = min(a.bbox_x2, b.bbox_x2)
                iy2 = min(a.bbox_y2, b.bbox_y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter == 0.0:
                    continue
                area_a = (a.bbox_x2 - a.bbox_x1) * (a.bbox_y2 - a.bbox_y1)
                area_b = (b.bbox_x2 - b.bbox_x1) * (b.bbox_y2 - b.bbox_y1)
                iomin = inter / (min(area_a, area_b) + 1e-6)
                if iomin < 0.99:
                    continue
                # 완전 포함 확인 → 더 큰 박스로 흡수
                if area_a >= area_b:
                    a.confidence = max(a.confidence, b.confidence)
                else:
                    # b가 더 크므로 a의 bbox를 b로 교체
                    a.bbox_x1, a.bbox_y1 = b.bbox_x1, b.bbox_y1
                    a.bbox_x2, a.bbox_y2 = b.bbox_x2, b.bbox_y2
                    a.confidence = max(a.confidence, b.confidence)
                used[j] = True
                merged = True
            kept.append(a)
        detections = kept
    return detections


# 타일 경계에서 인접한 동일 클래스 박스를 union으로 병합
def _merge_adjacent_detections(
    results: List[DetectionResult],
    gap_threshold: int,
) -> List[DetectionResult]:
    """타일 경계에서 쪼개진 동일 클래스 박스를 union bbox로 병합한다.

    두 박스가 같은 클래스이고 수평/수직 방향 간격이 gap_threshold 이하면
    하나의 union bbox로 합친다(신뢰도는 두 박스 중 최대값 사용).
    gap_threshold=0이면 맞닿은 박스만, 양수면 약간의 간격도 허용.
    """
    if gap_threshold <= 0 or not results:
        return results

    merged = True
    detections = list(results)
    while merged:
        merged = False
        kept: List[DetectionResult] = []
        used = [False] * len(detections)
        for i, a in enumerate(detections):
            if used[i]:
                continue
            for j in range(i + 1, len(detections)):
                b = detections[j]
                if used[j]:
                    continue
                if a.object_class.lower() != b.object_class.lower():
                    continue
                # 두 박스가 gap_threshold 이내로 인접하는지 확인
                gap_x = max(0.0, max(a.bbox_x1, b.bbox_x1) - min(a.bbox_x2, b.bbox_x2))
                gap_y = max(0.0, max(a.bbox_y1, b.bbox_y1) - min(a.bbox_y2, b.bbox_y2))
                if gap_x <= gap_threshold and gap_y <= gap_threshold:
                    # union bbox로 병합, 신뢰도는 최대값
                    a.bbox_x1 = min(a.bbox_x1, b.bbox_x1)
                    a.bbox_y1 = min(a.bbox_y1, b.bbox_y1)
                    a.bbox_x2 = max(a.bbox_x2, b.bbox_x2)
                    a.bbox_y2 = max(a.bbox_y2, b.bbox_y2)
                    a.confidence = max(a.confidence, b.confidence)
                    used[j] = True
                    merged = True
            kept.append(a)
        detections = kept
    return detections


# 두 바운딩 박스의 IoU 값 계산
def _iou(box_a: Tuple, box_b: Tuple) -> float:
    """두 bbox (x1,y1,x2,y2) 의 IoU."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ---------------------------------------------------------------------------
# SAM3 Detector + Tracker
# ---------------------------------------------------------------------------

class SAM3Detector:
    """
    facebook/sam3 패키지를 직접 사용하는 탐지기.

    - _model / _processor  : sam3 image model (텍스트 프롬프트 세그멘테이션)
    - _video_predictor     : sam3 video predictor (세션 기반 객체 추적)

    두 모델 모두 첫 호출 시 lazy-load.

    SAM3 설치:
        git clone https://github.com/facebookresearch/sam3.git
        cd sam3 && pip install -e .
        hf auth login
    """

    def __init__(self):
        self._model = None            # build_sam3_image_model() 반환값
        self._processor = None        # Sam3Processor(model)
        self._video_predictor = None  # build_sam3_video_predictor() 반환값
        self._tracker_load_attempted = False
        self._device = SAM3_DEVICE if torch.cuda.is_available() else "cpu"

    # GPU 메모리에서 모델 가중치 해제 및 VRAM 반환
    def unload(self) -> None:
        """모델 가중치를 GPU에서 해제하고 VRAM을 반환한다."""
        self._model = None
        self._processor = None
        self._video_predictor = None
        self._tracker_load_attempted = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[SAM3Detector] 모델 언로드 완료 — VRAM 반환")

    # ------------------------------------------------------------------
    # Model loaders
    # ------------------------------------------------------------------

    # SAM3 이미지 모델을 lazy-load 방식으로 로드
    def _load_model(self) -> None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info(
                f"[SAM3Detector] Loading sam3 image model "
                f"(checkpoint={SAM3_CHECKPOINT}) on {self._device} …"
            )
            self._model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
            self._model = self._model.to(self._device).eval()
            self._processor = Sam3Processor(self._model)
            logger.info("[SAM3Detector] sam3 image model loaded.")
        except (ImportError, OSError, Exception) as exc:
            logger.warning(
                f"[SAM3Detector] Could not load sam3 image model ({exc}). "
                "Using fallback grid-detector. "
                "Install: git clone https://github.com/facebookresearch/sam3.git && pip install -e ."
            )
            self._model = None

    # SAM3 비디오 예측기를 lazy-load 방식으로 로드
    def _load_tracker(self) -> None:
        self._tracker_load_attempted = True
        try:
            from sam3.model_builder import build_sam3_video_predictor

            logger.info(
                f"[SAM3Detector] Loading sam3 video predictor "
                f"(checkpoint={SAM3_CHECKPOINT}) on {self._device} …"
            )
            self._video_predictor = build_sam3_video_predictor(checkpoint_path=SAM3_CHECKPOINT)
            logger.info("[SAM3Detector] sam3 video predictor loaded.")
        except (ImportError, OSError, Exception) as exc:
            logger.warning(
                f"[SAM3Detector] Could not load sam3 video predictor ({exc}). "
                "Tracking will fall back to image-model re-detection."
            )
            self._video_predictor = None

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    # SAM3 미설치 시 빈 탐지 결과 반환
    def _fallback_detect(
        self, image: np.ndarray, image_id: str, meta: ImageMeta
    ) -> List[DetectionResult]:
        """SAM3 모델 미설치 시 빈 결과를 반환한다."""
        logger.warning(
            "[SAM3Detector] SAM3 모델 없음 — 탐지 결과 없음. "
            "SAM3 설치: git clone https://github.com/facebookresearch/sam3.git && pip install -e ."
        )
        return []

    # 비디오 예측기 불가 시 과거 위치를 그대로 유지하여 반환
    def _fallback_track(
        self, past_detections: list
    ) -> List[TrackedObject]:
        """video predictor 사용 불가 시 과거 위치를 그대로 유지."""
        logger.warning(
            "[SAM3Detector] video predictor unavailable – "
            "treating all past detections as tracked (positions unchanged)."
        )
        return [
            TrackedObject(
                past_detection_id=d.id,
                past_object_class=d.object_class,
                bbox_x1=d.bbox_x1, bbox_y1=d.bbox_y1,
                bbox_x2=d.bbox_x2, bbox_y2=d.bbox_y2,
                score=d.confidence,
            )
            for d in past_detections
        ]

    # ------------------------------------------------------------------
    # SAM3 image model: text-prompted detection (class별 1회 forward pass)
    # ------------------------------------------------------------------

    # SAM3 텍스트 프롬프트로 단일 클래스 객체 탐지
    def _detect_class(
        self,
        pil_image,
        class_name: str,
        class_index: int,
        orig_w: int,
        orig_h: int,
        now: datetime,
        image_id: str,
        meta: ImageMeta,
    ) -> List[DetectionResult]:
        """
        SAM3 텍스트 프롬프트로 class_name 에 해당하는 객체를 탐지.

        sam3 API:
            inference_state = processor.set_image(pil_image)
            output = processor.set_text_prompt(state=inference_state, prompt=class_name)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        """
        with torch.no_grad():
            inference_state = self._processor.set_image(pil_image)
            output = self._processor.set_text_prompt(
                state=inference_state,
                prompt=class_name,
            )
        # inference_state는 GPU 텐서를 보유하므로 즉시 해제
        del inference_state

        masks_out  = output.get("masks",  [])   # list[np.ndarray] | ndarray (N, H, W)
        boxes_out  = output.get("boxes",  [])   # (N, 4)  xyxy float
        scores_out = output.get("scores", [])   # (N,)    float

        # numpy 배열로 통일
        if hasattr(masks_out, "cpu"):
            masks_out = masks_out.cpu().numpy()
        if hasattr(boxes_out, "cpu"):
            boxes_out = boxes_out.cpu().numpy()
        if hasattr(scores_out, "cpu"):
            scores_out = scores_out.cpu().numpy()

        masks_out  = np.asarray(masks_out)
        boxes_out  = np.asarray(boxes_out)
        scores_out = np.asarray(scores_out).flatten()

        ratio    = MAX_BBOX_AREA_RATIO_LARGE if class_name in LARGE_OBJECT_CLASSES else MAX_BBOX_AREA_RATIO
        max_area = ratio * orig_w * orig_h
        detections: List[DetectionResult] = []

        for i, confidence in enumerate(scores_out):
            confidence = float(confidence)
            if confidence < SAM3_MASK_SCORE_THRESHOLD:
                continue

            # 마스크 처리
            if masks_out.ndim >= 3 and i < len(masks_out):
                raw_mask = np.squeeze(masks_out[i]).astype(bool)
                if raw_mask.shape != (orig_h, orig_w):
                    # 출력 해상도가 다를 경우 원본 크기로 리사이즈
                    from PIL import Image as PILImage
                    pil_mask = PILImage.fromarray(raw_mask.astype(np.uint8) * 255, "L")
                    pil_mask = pil_mask.resize((orig_w, orig_h), PILImage.NEAREST)
                    raw_mask = np.array(pil_mask) > 127
                mask_np = _tighten_mask(raw_mask)
            else:
                mask_np = None

            # bbox 결정: 마스크 기반 → boxes 폴백
            if mask_np is not None and mask_np.any():
                try:
                    x1, y1, x2, y2 = _mask_to_bbox(mask_np)
                except (IndexError, ValueError):
                    x1, y1, x2, y2 = (float(v) for v in boxes_out[i])
            elif i < len(boxes_out):
                x1, y1, x2, y2 = (float(v) for v in boxes_out[i])
                mask_np = np.zeros((orig_h, orig_w), dtype=bool)
                mask_np[int(y1):int(y2), int(x1):int(x2)] = True
            else:
                continue

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            if (x2 - x1) * (y2 - y1) > max_area:
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            lat, lon = pixel_to_geo(cx, cy, orig_w, orig_h, meta)
            detections.append(DetectionResult(
                detection_time=now, image_id=image_id,
                object_class=class_name, object_class_index=class_index,
                confidence=confidence,
                bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                lat=lat, lon=lon,
                mask_rle=_encode_rle(mask_np),
                mask_area_px=float(mask_np.sum()),
                source_type=meta.source_type,
            ))
        return detections

    # 단일 타일에서 전체 클래스 탐지 후 원본 좌표로 변환
    def _detect_on_tile(
        self,
        pil_image,
        tile_x1: int, tile_y1: int,
        tile_w: int, tile_h: int,
        orig_w: int, orig_h: int,
        now, image_id: str, meta: ImageMeta,
    ) -> List[DetectionResult]:
        """타일 한 장에 대해 모든 클래스 탐지 후 원본 좌표로 변환."""
        from PIL import Image as PILImage
        tile_img = pil_image.crop((tile_x1, tile_y1,
                                   tile_x1 + tile_w, tile_y1 + tile_h))
        results: List[DetectionResult] = []
        for class_index, class_name in enumerate(MILITARY_OBJECT_CLASSES):
            try:
                dets = self._detect_class(
                    tile_img, class_name, class_index,
                    tile_w, tile_h, now, image_id, meta,
                )
                # 타일 내 좌표 → 원본 이미지 좌표로 역변환
                for d in dets:
                    d.bbox_x1 += tile_x1
                    d.bbox_y1 += tile_y1
                    d.bbox_x2 += tile_x1
                    d.bbox_y2 += tile_y1
                    # 위경도도 원본 이미지 기준으로 재계산
                    from src.detection.image_loader import pixel_to_geo
                    cx = (d.bbox_x1 + d.bbox_x2) / 2.0
                    cy = (d.bbox_y1 + d.bbox_y2) / 2.0
                    d.lat, d.lon = pixel_to_geo(cx, cy, orig_w, orig_h, meta)
                results.extend(dets)
            except Exception as exc:
                logger.warning(
                    f"[SAM3Detector] tile({tile_x1},{tile_y1}) "
                    f"class '{class_name}': {exc}"
                )
        return results

    # SAM3로 이미지에서 군사 객체를 탐지하고 결과 반환
    def detect(self, loaded_image: LoadedImage, image_id: str) -> List[DetectionResult]:
        """
        SAM3 텍스트 프롬프트 세그멘테이션으로 군사 객체를 탐지한다.
        TILE_ENABLED=True 이면 슬라이딩 윈도우로 타일 분할 탐지,
        False 이면 전체 이미지를 한 번에 처리 (기존 방식).
        """
        if self._model is None:
            self._load_model()

        image_np = loaded_image.array
        meta     = loaded_image.meta
        orig_h, orig_w = image_np.shape[:2]

        logger.info(
            f"[SAM3Detector] Detecting: {meta.image_path} "
            f"({orig_w}×{orig_h}) tile={TILE_ENABLED}"
        )

        if self._model is None:
            results = self._fallback_detect(image_np, image_id, meta)
            return _merge_contained_detections(_nms_detections(results, NMS_IOU_THRESHOLD))

        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image_np).convert("RGB")
        now = meta.capture_time
        all_results: List[DetectionResult] = []

        if TILE_ENABLED:
            # ── [스케일 1] 전체 이미지 탐지 — 대형 객체 ─────────────────────
            if TILE_MULTISCALE:
                logger.info("[SAM3Detector] 스케일1(대형): 전체 이미지 탐지")
                for class_index, class_name in enumerate(MILITARY_OBJECT_CLASSES):
                    # 클래스마다 캐시 해제 — 전체 이미지 forward는 VRAM 소모가 크다
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        all_results.extend(
                            self._detect_class(
                                pil_image, class_name, class_index,
                                orig_w, orig_h, now, image_id, meta,
                            )
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            f"[SAM3Detector] 스케일1 OOM on class '{class_name}' — skipping"
                        )
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as exc:
                        logger.warning(
                            f"[SAM3Detector] 스케일1 class '{class_name}': {exc}"
                        )
                # 스케일1 완료 후 GPU 캐시 해제 — 다음 스케일 실행 전 VRAM 확보
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("[SAM3Detector] 스케일1 완료: GPU 캐시 해제")

            # ── [스케일 2] 중간 크기 타일 탐지 — 중형 객체 ──────────────────
            if TILE_MULTISCALE and TILE_MEDIUM_SCALE:
                med_tiles = _tile_coords(orig_w, orig_h, TILE_MEDIUM_SIZE, TILE_MEDIUM_OVERLAP)
                logger.info(
                    f"[SAM3Detector] 스케일2(중형): {len(med_tiles)}개 타일 "
                    f"(size={TILE_MEDIUM_SIZE}, overlap={TILE_MEDIUM_OVERLAP})"
                )
                for idx, (tx1, ty1, tx2, ty2) in enumerate(med_tiles):
                    tw, th = tx2 - tx1, ty2 - ty1
                    logger.debug(f"  중형타일 {idx+1}/{len(med_tiles)}  "
                                 f"({tx1},{ty1})→({tx2},{ty2})")
                    all_results.extend(
                        self._detect_on_tile(
                            pil_image, tx1, ty1, tw, th,
                            orig_w, orig_h, now, image_id, meta,
                        )
                    )
                    # 타일마다 GPU 캐시 해제 — 중형 타일은 개수가 많아 점진적으로 해제
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                logger.debug("[SAM3Detector] 스케일2 완료: GPU 캐시 해제")

            # ── [스케일 3] 소형 타일 탐지 — 소형 객체 ───────────────────────
            tiles = _tile_coords(orig_w, orig_h, TILE_SIZE, TILE_OVERLAP)
            logger.info(f"[SAM3Detector] 스케일3(소형): {len(tiles)}개 타일 "
                        f"(size={TILE_SIZE}, overlap={TILE_OVERLAP})")
            for idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                tw, th = tx2 - tx1, ty2 - ty1
                logger.debug(f"  소형타일 {idx+1}/{len(tiles)}  "
                             f"({tx1},{ty1})→({tx2},{ty2})")
                all_results.extend(
                    self._detect_on_tile(
                        pil_image, tx1, ty1, tw, th,
                        orig_w, orig_h, now, image_id, meta,
                    )
                )
                # 타일마다 GPU 캐시 해제 — 소형 타일은 개수가 가장 많으므로 타일 단위 해제
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.debug("[SAM3Detector] 스케일3 완료: GPU 캐시 해제")
        else:
            for class_index, class_name in enumerate(MILITARY_OBJECT_CLASSES):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    all_results.extend(
                        self._detect_class(
                            pil_image, class_name, class_index,
                            orig_w, orig_h, now, image_id, meta,
                        )
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"[SAM3Detector] OOM on class '{class_name}' — skipping"
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    logger.warning(
                        f"[SAM3Detector] Error on class '{class_name}': {exc}"
                    )

        final = _nms_detections(
            all_results,
            iou_threshold=TILE_NMS_IOU if TILE_ENABLED else NMS_IOU_THRESHOLD,
            iomin_threshold=TILE_NMS_IOMIN if TILE_ENABLED else 0.6,
        )
        if TILE_ENABLED and TILE_MERGE_GAP > 0:
            final = _merge_adjacent_detections(final, TILE_MERGE_GAP)
        final = _merge_contained_detections(final)
        logger.info(
            f"[SAM3Detector] {len(all_results)} raw → {len(final)} after NMS "
            f"(image_id={image_id[:8]})"
        )
        return final

    # ------------------------------------------------------------------
    # SAM3 video predictor: session-based object tracking across frames
    # ------------------------------------------------------------------

    # SAM3 비디오 예측기로 현재 프레임에서 과거 객체 추적
    def track_objects(
        self,
        pil_image,
        past_detections: list,   # List[DetectionRecord] from sensor DB
        orig_w: int,
        orig_h: int,
    ) -> List[TrackedObject]:
        """
        SAM3 video predictor로 현재 프레임에서 과거 객체를 추적한다.

        sam3 video predictor API:
            response = video_predictor.handle_request(
                request=dict(type="start_session", resource_path="<image_path>")
            )
            session_id = response["session_id"]
            response = video_predictor.handle_request(
                request=dict(type="add_prompt", session_id=session_id,
                             frame_index=0, text="<class_name>")
            )
            output = response["outputs"]

        현재 프레임을 임시 파일로 저장 → 세션 시작 → 각 과거 객체 클래스로
        텍스트 프롬프트 추가 → 출력 bbox를 과거 bbox와 IoU로 매칭.

        Args:
            pil_image:        Current-frame PIL image (RGB).
            past_detections:  DetectionRecord list from the most-recent past frame.
            orig_w / orig_h:  Original image dimensions.

        Returns:
            List of TrackedObject – one entry per successfully tracked past object.
        """
        if not past_detections:
            return []

        if not self._tracker_load_attempted:
            self._load_tracker()

        if self._video_predictor is None:
            return self._fallback_track(past_detections)

        tracked: List[TrackedObject] = []

        try:
            # video predictor 는 파일 경로를 받으므로 현재 프레임을 임시 저장
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            pil_image.save(tmp_path, format="JPEG", quality=95)

            # 세션 시작
            response = self._video_predictor.handle_request(
                request=dict(type="start_session", resource_path=tmp_path)
            )
            session_id = response["session_id"]

            # 과거 객체 클래스별로 텍스트 프롬프트 추가
            # 같은 클래스가 여러 개 있을 수 있으므로 클래스별로 묶어서 처리
            class_to_past: dict = {}
            for d in past_detections:
                class_to_past.setdefault(d.object_class, []).append(d)

            for class_name, dets in class_to_past.items():
                try:
                    response = self._video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            text=class_name,
                        )
                    )
                    output = response.get("outputs", {})
                    new_boxes  = output.get("boxes",  [])
                    new_scores = output.get("scores", [])
                    new_masks  = output.get("masks",  [])

                    if hasattr(new_boxes, "cpu"):
                        new_boxes = new_boxes.cpu().numpy()
                    if hasattr(new_scores, "cpu"):
                        new_scores = new_scores.cpu().numpy()

                    new_boxes  = np.asarray(new_boxes)
                    new_scores = np.asarray(new_scores).flatten()

                    # 각 과거 객체 → IoU가 가장 높은 현재 탐지 결과와 매칭
                    for past_det in dets:
                        past_box = (
                            past_det.bbox_x1, past_det.bbox_y1,
                            past_det.bbox_x2, past_det.bbox_y2,
                        )
                        best_iou, best_idx = 0.0, -1
                        for j in range(len(new_scores)):
                            if float(new_scores[j]) < DETECTION_CONFIDENCE_THRESHOLD:
                                continue
                            nb = new_boxes[j]
                            iou = _iou(past_box, (nb[0], nb[1], nb[2], nb[3]))
                            if iou > best_iou:
                                best_iou, best_idx = iou, j

                        if best_idx < 0 or best_iou < 0.1:
                            logger.debug(
                                f"[SAM3Tracker] {past_det.id[:8]} "
                                f"({class_name}) → disappeared (best_iou={best_iou:.3f})"
                            )
                            continue

                        score = float(new_scores[best_idx])
                        nb = new_boxes[best_idx]

                        # 마스크가 있으면 tight bbox 재계산
                        if (hasattr(new_masks, "__len__") and len(new_masks) > best_idx
                                and new_masks[best_idx] is not None):
                            raw_mask = np.squeeze(np.asarray(new_masks[best_idx])).astype(bool)
                            if raw_mask.shape != (orig_h, orig_w):
                                from PIL import Image as PILImage
                                pm = PILImage.fromarray(raw_mask.astype(np.uint8) * 255, "L")
                                pm = pm.resize((orig_w, orig_h), PILImage.NEAREST)
                                raw_mask = np.array(pm) > 127
                            mask_np = _tighten_mask(raw_mask)
                            if mask_np.any():
                                try:
                                    x1, y1, x2, y2 = _mask_to_bbox(mask_np)
                                except (IndexError, ValueError):
                                    x1, y1, x2, y2 = nb[0], nb[1], nb[2], nb[3]
                            else:
                                x1, y1, x2, y2 = nb[0], nb[1], nb[2], nb[3]
                        else:
                            x1, y1, x2, y2 = nb[0], nb[1], nb[2], nb[3]

                        tracked.append(TrackedObject(
                            past_detection_id=past_det.id,
                            past_object_class=class_name,
                            bbox_x1=float(x1), bbox_y1=float(y1),
                            bbox_x2=float(x2), bbox_y2=float(y2),
                            score=score,
                        ))
                        logger.debug(
                            f"[SAM3Tracker] {past_det.id[:8]} ({class_name}) "
                            f"tracked  score={score:.3f}  iou={best_iou:.3f}"
                        )

                except Exception as exc:
                    logger.warning(
                        f"[SAM3Tracker] Error tracking class '{class_name}': {exc}"
                    )

            # 세션 정리
            try:
                self._video_predictor.handle_request(
                    request=dict(type="end_session", session_id=session_id)
                )
            except Exception:
                pass

        except Exception as exc:
            logger.warning(f"[SAM3Tracker] Session error: {exc}. Falling back.")
            return self._fallback_track(past_detections)
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        logger.info(
            f"[SAM3Tracker] {len(tracked)}/{len(past_detections)} "
            "past objects successfully tracked."
        )
        return tracked
