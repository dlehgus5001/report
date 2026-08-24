"""
Image loader for satellite and drone imagery.

Each image must have an associated metadata entry (JSON sidecar or metadata.json index)
containing:
  - capture_time (ISO 8601)
  - source_type: "satellite" | "drone"
  - lat_center, lon_center
  - lat_min, lat_max, lon_min, lon_max  (optional bounding box)
  - resolution_m  (optional, GSD in metres/pixel)
  - sensor_platform  (optional, e.g. "WorldView-3", "MQ-9 Reaper")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from PIL import Image

from src.detection.super_resolution import super_resolve

logger = logging.getLogger(__name__)


@dataclass
class ImageMeta:
    """All metadata associated with one sensor image."""
    image_path: str
    capture_time: datetime
    source_type: str             # "satellite" | "drone"
    lat_center: float
    lon_center: float
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    resolution_m: Optional[float] = None
    sensor_platform: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class LoadedImage:
    meta: ImageMeta
    array: np.ndarray            # H × W × 3  uint8 RGB


# 메타데이터 딕셔너리를 ImageMeta 객체로 변환
def _parse_meta_entry(entry: dict, base_dir: Path) -> Optional[ImageMeta]:
    try:
        image_path = str(base_dir / entry["image_file"])
        capture_time = datetime.fromisoformat(entry["capture_time"])
        return ImageMeta(
            image_path=image_path,
            capture_time=capture_time,
            source_type=entry.get("source_type", "satellite"),
            lat_center=float(entry["lat_center"]),
            lon_center=float(entry["lon_center"]),
            lat_min=entry.get("lat_min"),
            lat_max=entry.get("lat_max"),
            lon_min=entry.get("lon_min"),
            lon_max=entry.get("lon_max"),
            resolution_m=entry.get("resolution_m"),
            sensor_platform=entry.get("sensor_platform"),
            extra={k: v for k, v in entry.items() if k not in {
                "image_file", "capture_time", "source_type",
                "lat_center", "lon_center", "lat_min", "lat_max",
                "lon_min", "lon_max", "resolution_m", "sensor_platform",
            }},
        )
    except (KeyError, ValueError) as e:
        logger.warning(f"[ImageLoader] Skipping malformed metadata entry: {e}")
        return None


# JSON 메타데이터 인덱스 파일을 로드하여 ImageMeta 목록 반환
def load_metadata_index(metadata_json: str) -> List[ImageMeta]:
    """
    Load the image metadata index file.
    Expected format: JSON array of image metadata objects.
    """
    path = Path(metadata_json)
    if not path.exists():
        raise FileNotFoundError(f"Metadata index not found: {path}")
    with open(path) as f:
        entries = json.load(f)
    base_dir = path.parent
    metas = [_parse_meta_entry(e, base_dir) for e in entries]
    metas = [m for m in metas if m is not None]
    logger.info(f"[ImageLoader] Loaded {len(metas)} image metadata entries from {path}")
    return metas


# 메타데이터 목록에서 이미지를 읽어 LoadedImage로 순차 반환
def iter_images(metas: List[ImageMeta]) -> Iterator[LoadedImage]:
    """
    Yield LoadedImage objects (metadata + numpy array) for each valid image file.
    Images are converted to RGB uint8, then Super-Resolution으로 업스케일
    (최대 SR_TARGET_W×SR_TARGET_H, 비율 유지).
    Real-ESRGAN 미설치 시 PIL LANCZOS로 폴백.
    """
    for meta in metas:
        p = Path(meta.image_path)
        if not p.exists():
            logger.warning(f"[ImageLoader] Image file not found: {p}. Skipping.")
            continue
        try:
            img = Image.open(p).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            arr = super_resolve(arr)
            logger.debug(f"[ImageLoader] Loaded {p.name}  shape={arr.shape}")
            yield LoadedImage(meta=meta, array=arr)
        except Exception as e:
            logger.error(f"[ImageLoader] Failed to load {p}: {e}")


# 픽셀 좌표를 위경도 지리 좌표로 변환
def pixel_to_geo(
    px: float, py: float,
    img_w: int, img_h: int,
    meta: ImageMeta,
) -> tuple:
    """
    Convert pixel coordinates (px, py) within an image to (lat, lon)
    using linear interpolation over the image's geographic bounding box.
    Falls back to centre coordinates if bounding box is absent.
    """
    if None in (meta.lat_min, meta.lat_max, meta.lon_min, meta.lon_max):
        return meta.lat_center, meta.lon_center
    lon = meta.lon_min + (px / img_w) * (meta.lon_max - meta.lon_min)
    # Image y-axis is inverted relative to latitude
    lat = meta.lat_max - (py / img_h) * (meta.lat_max - meta.lat_min)
    return lat, lon
