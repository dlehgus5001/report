#!/usr/bin/env python3
"""
visualize_detections.py – SAM3 탐지 결과를 이미지에 렌더링하는 도구

DB에서 DetectionRecord를 불러와 원본 이미지 위에
바운딩박스 / 마스크 / 레이블을 그려 PNG로 저장합니다.

사용법:
  python visualize_detections.py                          # 최근 이미지 1장
  python visualize_detections.py --limit 5               # 최근 이미지 5장
  python visualize_detections.py --image-id <uuid>       # 특정 이미지
  python visualize_detections.py --class "military tank" # 특정 클래스 필터
  python visualize_detections.py --no-mask               # 마스크 없이 bbox만
  python visualize_detections.py --out-dir ./my_output   # 저장 디렉터리 지정
  python visualize_detections.py --show                  # 저장 후 바로 열기
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config import SENSOR_DB_PATH, IMAGES_DIR
from src.database.models import Base, ImageRecord, DetectionRecord


# ---------------------------------------------------------------------------
# 클래스별 색상 팔레트 (BGR 아닌 RGB)
# ---------------------------------------------------------------------------

PALETTE = [
    (255,  60,  60),  # 빨강
    ( 60, 180, 255),  # 파랑
    ( 60, 220,  60),  # 초록
    (255, 180,   0),  # 주황
    (180,  60, 255),  # 보라
    (  0, 210, 180),  # 청록
    (255,  60, 200),  # 핑크
    (200, 200,   0),  # 노랑
    (255, 140,  60),  # 연주황
    (100, 100, 255),  # 연파랑
]

def class_color(class_index: int) -> Tuple[int, int, int]:
    return PALETTE[class_index % len(PALETTE)]


# ---------------------------------------------------------------------------
# RLE 디코더 (sam2_detector._encode_rle 역변환)
# ---------------------------------------------------------------------------

def decode_rle(rle_json: str, fallback_shape: Tuple[int, int] = None) -> Optional[np.ndarray]:
    """
    JSON {"shape": [H, W], "rle": [[value, count], ...]} → bool ndarray (H, W)
    디코딩 실패 시 None 반환.
    """
    try:
        data = json.loads(rle_json)
        h, w = data["shape"]
        flat = []
        for value, count in data["rle"]:
            flat.extend([value] * count)
        arr = np.array(flat, dtype=np.uint8).reshape(h, w)
        return arr.astype(bool)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 렌더링 함수
# ---------------------------------------------------------------------------

def _try_load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_detections(
    image: Image.Image,
    detections: List[DetectionRecord],
    draw_mask: bool = True,
    alpha: float = 0.35,
) -> Image.Image:
    """
    원본 PIL 이미지(RGB)에 탐지 결과를 오버레이한 새 이미지를 반환.

    - 반투명 마스크 오버레이 (draw_mask=True 일 때)
    - 바운딩박스 (두꺼운 테두리)
    - 클래스 + 신뢰도 레이블
    """
    orig_w, orig_h = image.size
    base = image.convert("RGBA")

    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_draw  = ImageDraw.Draw(mask_layer)

    box_layer  = Image.new("RGBA", base.size, (0, 0, 0, 0))
    box_draw   = ImageDraw.Draw(box_layer)

    font_large = _try_load_font(18)
    font_small = _try_load_font(13)

    for det in detections:
        color = class_color(det.object_class_index or 0)
        r, g, b = color

        x1, y1 = int(det.bbox_x1), int(det.bbox_y1)
        x2, y2 = int(det.bbox_x2), int(det.bbox_y2)

        # ── 마스크 오버레이 ──────────────────────────────────────────────
        if draw_mask and det.mask_rle:
            mask_np = decode_rle(det.mask_rle, fallback_shape=(orig_h, orig_w))
            if mask_np is not None and mask_np.shape == (orig_h, orig_w):
                # 마스크 픽셀만 색칠 (RGBA)
                overlay_arr = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
                overlay_arr[mask_np, 0] = r
                overlay_arr[mask_np, 1] = g
                overlay_arr[mask_np, 2] = b
                overlay_arr[mask_np, 3] = int(255 * alpha)
                mask_pil = Image.fromarray(overlay_arr, mode="RGBA")
                mask_layer = Image.alpha_composite(mask_layer, mask_pil)

        # ── 바운딩박스 ───────────────────────────────────────────────────
        for lw in range(3):  # 3px 두께
            box_draw.rectangle(
                [x1 - lw, y1 - lw, x2 + lw, y2 + lw],
                outline=(r, g, b, 230),
            )

        # ── 레이블 배경 + 텍스트 ─────────────────────────────────────────
        label = f"{det.object_class}  {det.confidence:.2f}"
        try:
            bbox_txt = font_large.getbbox(label)
            tw, th = bbox_txt[2] - bbox_txt[0], bbox_txt[3] - bbox_txt[1]
        except AttributeError:
            tw, th = len(label) * 9, 16

        pad = 3
        lx1, ly1 = x1, max(0, y1 - th - pad * 2)
        lx2, ly2 = x1 + tw + pad * 2, y1

        box_draw.rectangle([lx1, ly1, lx2, ly2], fill=(r, g, b, 210))
        box_draw.text((lx1 + pad, ly1 + pad), label,
                      fill=(255, 255, 255, 255), font=font_large)

    # ── 합성 ────────────────────────────────────────────────────────────
    composite = Image.alpha_composite(base, mask_layer)
    composite = Image.alpha_composite(composite, box_layer)
    return composite.convert("RGB")


def render_pairings_legend(
    image: Image.Image,
    detections: List[DetectionRecord],
    pairing_status: Optional[str],   # "matched" | "new" | "disappeared" | None
) -> Image.Image:
    """이미지 하단에 페어링 상태 범례 바를 추가."""
    if pairing_status is None:
        return image
    status_color = {
        "matched":     (60, 220, 60),
        "new":         (60, 180, 255),
        "disappeared": (255, 60, 60),
    }
    color = status_color.get(pairing_status, (200, 200, 200))
    bar_h = 28
    orig_w, orig_h = image.size
    new_img = Image.new("RGB", (orig_w, orig_h + bar_h), color)
    new_img.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_img)
    font = _try_load_font(16)
    label = f"  Status: {pairing_status.upper()}  |  탐지 {len(detections)}건"
    draw.text((8, orig_h + 5), label, fill=(255, 255, 255), font=font)
    return new_img


# ---------------------------------------------------------------------------
# DB 조회
# ---------------------------------------------------------------------------

def get_sensor_session() -> Session:
    engine = create_engine(f"sqlite:///{SENSOR_DB_PATH}", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def load_image_records(
    limit: int = 1,
    image_id: str = None,
    object_class: str = None,
) -> List[Tuple[ImageRecord, List[DetectionRecord]]]:
    """
    (ImageRecord, [DetectionRecord, ...]) 튜플 리스트 반환.
    object_class 필터가 있으면 해당 클래스가 탐지된 이미지만 반환.
    """
    with get_sensor_session() as sess:
        q = sess.query(ImageRecord)

        if image_id:
            q = q.filter(ImageRecord.id == image_id)
        elif object_class:
            from sqlalchemy.orm import joinedload
            matching_image_ids = (
                sess.query(DetectionRecord.image_id)
                .filter(DetectionRecord.object_class.ilike(f"%{object_class}%"))
                .distinct()
                .subquery()
            )
            q = q.filter(ImageRecord.id.in_(matching_image_ids))

        q = q.order_by(ImageRecord.capture_time.desc()).limit(limit)
        images = q.all()

        result = []
        for img in images:
            det_q = sess.query(DetectionRecord).filter(
                DetectionRecord.image_id == img.id
            )
            if object_class:
                det_q = det_q.filter(
                    DetectionRecord.object_class.ilike(f"%{object_class}%")
                )
            dets = det_q.order_by(DetectionRecord.confidence.desc()).all()
            result.append((img, dets))

        # detach from session
        sess.expunge_all()
        return result


# ---------------------------------------------------------------------------
# 이미지 로드 (경로 해석)
# ---------------------------------------------------------------------------

def find_image_file(image_path: str) -> Optional[Path]:
    """
    DB에 저장된 image_path로 실제 파일을 찾는다.
    절대경로 → IMAGES_DIR 상대경로 → 파일명만으로 재시도 순서.
    """
    candidates = [
        Path(image_path),
        IMAGES_DIR / image_path,
        IMAGES_DIR / Path(image_path).name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# 메인 처리 루프
# ---------------------------------------------------------------------------

def process(
    limit: int = 1,
    image_id: str = None,
    object_class: str = None,
    draw_mask: bool = True,
    out_dir: Path = Path("detection_output"),
    show: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_image_records(limit=limit, image_id=image_id, object_class=object_class)

    if not records:
        print("  DB에 해당하는 탐지 결과가 없습니다.")
        return

    saved = []
    for img_rec, dets in records:
        print(f"\n[이미지]  {img_rec.id[:8]}  "
              f"캡처={img_rec.capture_time}  "
              f"탐지={len(dets)}건  "
              f"경로={img_rec.image_path}")

        # ── 원본 이미지 로드 ──────────────────────────────────────────
        # image_loader.iter_images() 와 동일한 max_dim 리사이즈를 적용해야
        # DB에 저장된 bbox 좌표계(탐지 시 리사이즈된 해상도)와 일치한다.
        MAX_DIM = 2048
        img_path = find_image_file(img_rec.image_path)
        if img_path:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                w, h = pil_img.size
                if max(w, h) > MAX_DIM:
                    scale = MAX_DIM / max(w, h)
                    pil_img = pil_img.resize(
                        (int(w * scale), int(h * scale)), Image.LANCZOS
                    )
            except Exception as e:
                print(f"  ⚠ 이미지 열기 실패 ({e}), 빈 캔버스 사용")
                pil_img = Image.new("RGB", (800, 600), (30, 30, 30))
        else:
            print(f"  ⚠ 파일 없음 → 빈 캔버스 사용 ({img_rec.image_path})")
            pil_img = Image.new("RGB", (800, 600), (30, 30, 30))

        # ── 탐지 결과 렌더링 ────────────────────────────────────────
        if dets:
            rendered = render_detections(pil_img, dets, draw_mask=draw_mask)
        else:
            print("  탐지 결과 없음 – 원본 이미지만 저장")
            rendered = pil_img

        # ── 탐지 정보 출력 ─────────────────────────────────────────
        for det in dets:
            w = int(det.bbox_x2 - det.bbox_x1)
            h = int(det.bbox_y2 - det.bbox_y1)
            lat_s = f"{det.lat:.4f}" if det.lat is not None else "N/A"
            lon_s = f"{det.lon:.4f}" if det.lon is not None else "N/A"
            print(f"    ├ {det.object_class:22} "
                  f"conf={det.confidence:.3f}  "
                  f"bbox=[{int(det.bbox_x1)},{int(det.bbox_y1)},{w}×{h}]  "
                  f"lat={lat_s}  lon={lon_s}")

        # ── 저장 ────────────────────────────────────────────────────
        ts = (img_rec.capture_time or datetime.utcnow()).strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{img_rec.id[:8]}_{len(dets)}dets.png"
        out_path = out_dir / fname
        rendered.save(out_path)
        print(f"  → 저장: {out_path}")
        saved.append(out_path)

        if show:
            try:
                rendered.show(title=fname)
            except Exception:
                pass

    print(f"\n총 {len(saved)}장 저장 완료 → {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 탐지 결과를 이미지로 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--limit",    type=int,  default=1,
                        help="처리할 이미지 수 (기본 1, 최신순)")
    parser.add_argument("--image-id", default=None,
                        help="특정 이미지 UUID")
    parser.add_argument("--class",    dest="obj_class", default=None,
                        help="클래스 이름 필터 (부분 일치)")
    parser.add_argument("--no-mask",  action="store_true",
                        help="마스크 오버레이 없이 bbox만 그리기")
    parser.add_argument("--out-dir",  default="detection_output",
                        help="PNG 저장 디렉터리 (기본 ./detection_output)")
    parser.add_argument("--show",     action="store_true",
                        help="저장 후 이미지 뷰어로 바로 열기")
    return parser.parse_args()


def main():
    args = parse_args()
    process(
        limit=args.limit,
        image_id=args.image_id,
        object_class=args.obj_class,
        draw_mask=not args.no_mask,
        out_dir=Path(args.out_dir),
        show=args.show,
    )


if __name__ == "__main__":
    main()
