"""
MSIS 위성 모의기 실행기 (Simulator Runner)

동작 단계:
  1. 위성 궤도 계산 → 현재 위경도 (한반도 최근접 위성 선택)
  2. data/images/sample/ 에서 PNG 2장 랜덤 선택
  3. data/images/metadata.json 업데이트
       - imgs[0] → 과거 프레임  (capture_time = now − 6h)
       - imgs[1] → 현재 프레임  (capture_time = now)
       - lat/lon  = 위성 현재 위치
  4. python main.py --metadata ... --report-output ... 실행
  5. 실행 결과 dict 반환

독립 실행:
  python simulator_runner.py

FastAPI 에서 호출:
  POST /api/simulator/step  →  run_step()
"""

import json
import logging
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.satellite.simulator import get_active_satellite, get_positions
from src.satellite.land_check import is_land
from src.config import IMAGE_MODE, CROP_AXIS, CROP_SIZE, CROP_OFFSET

logger = logging.getLogger(__name__)

# ── 경로 ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
SAMPLE_DIR    = BASE_DIR / "data" / "images" / "sample"
CROPS_DIR     = SAMPLE_DIR / ".crops"          # 크롭 임시 파일 저장 위치
METADATA_PATH = BASE_DIR / "data" / "images" / "metadata.json"
REPORT_OUTPUT = BASE_DIR / "data" / "reports" / "report.txt"

# IMAGE_MODE / CROP_AXIS / CROP_SIZE / CROP_OFFSET 은 src/config.py 에서 관리

PIPELINE_CMD = [
    sys.executable, str(BASE_DIR / "main.py"),
    "--metadata",      str(METADATA_PATH),
    "--report-output", str(REPORT_OUTPUT),
]


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

def _pick_images(n: int = 2) -> list:
    """
    SAMPLE_DIR 에서 tif/tiff/JPG/jpg 파일을 수집한 뒤,
    파일명을 '_' 로 split 했을 때 앞 2개 토큰이 동일한 파일끼리 쌍으로 묶어
    그 중 하나를 랜덤 선택해 반환한다.
    쌍이 없으면 전체 풀에서 랜덤 n 장으로 폴백.
    반환값은 metadata.json 에 쓰이는 상대 경로 'sample/<파일명>' 형식.
    """
    tifs  = sorted(SAMPLE_DIR.glob("*.tiff")) + sorted(SAMPLE_DIR.glob("*.tif"))
    jpgs  = sorted(SAMPLE_DIR.glob("*.JPG"))  + sorted(SAMPLE_DIR.glob("*.jpg"))
    pool  = tifs + jpgs

    # '_' split 앞 2토큰 기준으로 그룹핑
    groups: dict[str, list] = {}
    for p in pool:
        parts = p.stem.split("_")          # 확장자 제외 파일명 분리
        key   = "_".join(parts[:2]) if len(parts) >= 2 else p.stem
        groups.setdefault(key, []).append(p)

    # 파일이 n장 이상인 그룹만 추출
    valid = [files for files in groups.values() if len(files) >= n]

    if valid:
        chosen = random.choice(valid)      # 그룹 하나 랜덤 선택
        return [f"sample/{p.name}" for p in random.sample(chosen, n)]

    # 쌍을 찾지 못하면 전체 풀에서 랜덤 선택 (폴백)
    if len(pool) >= n:
        chosen = random.sample(pool, n)
        return [f"sample/{p.name}" for p in chosen]

    raise RuntimeError(
        f"data/images/sample/ 에 이미지가 {len(pool)}장뿐입니다 (최소 {n}장 필요). "
        f"tif/tiff: {len(tifs)}장, JPG/jpg: {len(jpgs)}장"
    )


def _pick_and_crop_image(
    axis: str = "vertical",
    crop_size: float = 0.7,
    crop_offset: float = 0.15,
) -> tuple:
    """
    sample/ 에서 이미지 1장을 랜덤 선택한 뒤 일부 겹치는 크롭 2장을 생성.

    crop_A (이전 프레임): 원점에서 crop_size 비율 크기
    crop_B (현재 프레임): crop_offset 만큼 이동한 시작점에서 같은 크기

    두 크롭의 겹침 영역 ≈ crop_size − crop_offset (기본: 0.70 − 0.15 = 55%)

    예시 (vertical, crop_size=0.7, crop_offset=0.15, W=1000):
      crop_A: x = [  0, 700]  (70% 너비)
      crop_B: x = [150, 850]  (70% 너비, 150px 이동)
      겹침:   x = [150, 700]  → 550px = 원본의 55%

    ┌──────────────────────────────┐ 원본 (W=1000)
    │ crop_A [0──────────700]      │
    │     crop_B    [150──────850] │
    │         겹침  [150──700]     │
    └──────────────────────────────┘

    Args:
        axis:        오프셋 방향 — "vertical" (X축 이동) | "horizontal" (Y축 이동)
        crop_size:   각 크롭 크기 (원본 대비 비율, 0.0~1.0)
        crop_offset: 두 크롭 시작점 간의 이동 비율 (원본 대비, 0.0~1.0)
                     crop_offset < (1 - crop_size) 조건을 만족해야 두 크롭 모두 원본 범위 내에 있음.

    Returns:
        (img_paths, past_time, current_time)
    """
    from PIL import Image as PILImage

    tifs = sorted(SAMPLE_DIR.glob("*.tiff")) + sorted(SAMPLE_DIR.glob("*.tif"))
    jpgs = sorted(SAMPLE_DIR.glob("*.JPG"))  + sorted(SAMPLE_DIR.glob("*.jpg"))
    pool = tifs + jpgs

    if not pool:
        raise RuntimeError(
            "data/images/sample/ 에 이미지가 없습니다. "
            "크롭 모드는 최소 1장이 필요합니다."
        )

    src = random.choice(pool)
    logger.info(
        f"[SimRunner/crop] 원본: {src.name}  axis={axis}  "
        f"crop_size={crop_size}  crop_offset={crop_offset}"
    )

    with PILImage.open(str(src)).convert("RGB") as img:
        w, h = img.size
        if axis == "horizontal":
            ch     = max(1, int(h * crop_size))
            offset = max(0, min(int(h * crop_offset), h - ch))
            crop_a = img.crop((0, 0,      w, ch))
            crop_b = img.crop((0, offset, w, offset + ch))
        else:  # vertical (기본)
            cw     = max(1, int(w * crop_size))
            offset = max(0, min(int(w * crop_offset), w - cw))
            crop_a = img.crop((0,      0, cw,      h))
            crop_b = img.crop((offset, 0, offset + cw, h))

    overlap_pct = max(0.0, crop_size - crop_offset) / crop_size * 100
    logger.info(f"[SimRunner/crop] 두 크롭 겹침 비율: {overlap_pct:.1f}%")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    now       = datetime.now(tz=timezone.utc)
    past_time = now - timedelta(hours=6)

    paths = []
    for crop, label, ct in [
        (crop_a, "A", past_time),
        (crop_b, "B", now),
    ]:
        fname = CROPS_DIR / f"{stem}_crop_{label}.png"
        crop.save(str(fname))
        # mtime 을 capture_time 으로 설정 → _extract_tiff_capture_time 폴백 시 올바른 순서 보장
        ts = ct.timestamp()
        os.utime(str(fname), (ts, ts))
        rel = fname.relative_to(SAMPLE_DIR.parent)  # "sample/.crops/…"
        paths.append(str(rel))

    logger.info(f"[SimRunner/crop] 크롭 저장: {paths}")
    # 각 크롭의 지리 범위 계산용 시작 비율 (A=0, B=offset)
    fracs = [0.0, crop_offset]
    return paths, past_time, now, fracs


def _pick_from_crops_dir() -> tuple:
    """
    sample/.crops/ 폴더에서 *_crop_A.png + *_crop_B.png 쌍을 찾아 하나를 선택.

    파일 수정시간(mtime)을 촬영 시각으로 사용한다.
    crop 모드로 생성된 파일은 생성 시 mtime이 촬영 시각으로 설정되어 있으므로
    올바른 과거/현재 순서가 자동으로 유지된다.

    Returns:
        (img_paths, past_time, current_time, crop_fracs)
          img_paths:  상대 경로 2개 리스트 — [과거, 현재] 순서
          crop_fracs: _build_metadata() 에 전달할 시작 비율 리스트
                      crop_A 가 과거면 [0.0, CROP_OFFSET],
                      crop_B 가 과거면 [CROP_OFFSET, 0.0]
    """
    if not CROPS_DIR.exists():
        raise RuntimeError(
            f".crops 폴더가 없습니다: {CROPS_DIR}\n"
            "IMAGE_MODE=crop 으로 먼저 크롭을 생성하거나 파일을 직접 배치해 주세요.\n"
            "  파일명 규칙: <이름>_crop_A.png (과거), <이름>_crop_B.png (현재)"
        )

    # *_crop_A.png 파일을 수집하고 대응하는 B 파일이 있는 쌍만 필터링
    pairs = []
    for a_file in sorted(CROPS_DIR.glob("*_crop_A.png")):
        stem = a_file.stem[:-7]  # "_crop_A" 제거
        b_file = CROPS_DIR / f"{stem}_crop_B.png"
        if b_file.exists():
            pairs.append((a_file, b_file))

    if not pairs:
        raise RuntimeError(
            f".crops 폴더에 *_crop_A.png + *_crop_B.png 쌍이 없습니다: {CROPS_DIR}\n"
            "  파일명 규칙: <이름>_crop_A.png (과거), <이름>_crop_B.png (현재)"
        )

    a_path, b_path = random.choice(pairs)

    # 파일 mtime → capture_time (crop 모드로 생성 시 os.utime 으로 설정됨)
    a_mtime = datetime.fromtimestamp(a_path.stat().st_mtime, tz=timezone.utc)
    b_mtime = datetime.fromtimestamp(b_path.stat().st_mtime, tz=timezone.utc)

    # mtime 기준으로 과거/현재 순서 결정
    # crop_A = frac 0.0 기준점, crop_B = CROP_OFFSET 기준점
    if a_mtime <= b_mtime:
        past_path, curr_path = a_path, b_path
        past_time, curr_time = a_mtime, b_mtime
        crop_fracs = [0.0, CROP_OFFSET]          # A=과거(0.0), B=현재(offset)
    else:
        past_path, curr_path = b_path, a_path
        past_time, curr_time = b_mtime, a_mtime
        crop_fracs = [CROP_OFFSET, 0.0]          # B=과거(offset), A=현재(0.0)

    paths = [
        str(past_path.relative_to(SAMPLE_DIR.parent)),
        str(curr_path.relative_to(SAMPLE_DIR.parent)),
    ]
    logger.info(
        "[SimRunner/crops_dir] 쌍 선택: %s (과거=%s, frac=%.2f) / %s (현재=%s, frac=%.2f)",
        past_path.name, past_time.strftime("%H:%M:%S"), crop_fracs[0],
        curr_path.name, curr_time.strftime("%H:%M:%S"), crop_fracs[1],
    )
    return paths, past_time, curr_time, crop_fracs


def _extract_tiff_capture_time(path: Path) -> datetime:
    """
    TIFF 파일에서 촬영 시각을 추출.
    우선순위:
      1. Tag 42112 (GDAL_METADATA) XML의 ACQUISITION_DATETIME
      2. Tag 306  (DateTime) "YYYY:MM:DD HH:MM:SS"
      3. 파일 수정시간 폴백
    """
    try:
        from PIL import Image as PILImage
        with PILImage.open(str(path)) as img:
            tag_info = img.tag_v2 if hasattr(img, "tag_v2") else {}

            # 1. Tag 42112 GDAL_METADATA
            gdal_xml = tag_info.get(42112)
            if gdal_xml:
                try:
                    root = ET.fromstring(gdal_xml)
                    for item in root.findall("Item"):
                        if item.get("name") == "ACQUISITION_DATETIME" and item.text:
                            return datetime.fromisoformat(item.text)
                except Exception:
                    pass

            # 2. Tag 306 DateTime "YYYY:MM:DD HH:MM:SS"
            dt_tag = tag_info.get(306)
            if dt_tag:
                try:
                    return datetime.strptime(dt_tag, "%Y:%m:%d %H:%M:%S").replace(
                        tzinfo=timezone.utc
                    )
                except Exception:
                    pass
    except Exception:
        pass

    # 3. 파일 수정시간 폴백
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


def _build_metadata(
    sat: dict,
    imgs: list,
    explicit_times: list = None,
    crop_axis: str = None,
    crop_size_ratio: float = None,
    crop_fracs: list = None,
) -> list:
    """
    위성 위치와 선택된 이미지 2장으로 metadata.json 항목 2개 생성.

    Args:
        sat:             활성 위성 정보 dict
        imgs:            이미지 상대경로 목록 (data/images/ 기준)
        explicit_times:  capture_time 명시 리스트 (크롭 모드에서 사용).
                         None 이면 TIFF 태그 / 파일 수정시간에서 자동 추출.
        crop_axis:       크롭 방향 ("vertical" | "horizontal"). 크롭 모드 전용.
        crop_size_ratio: 각 크롭 크기 비율 (0.0~1.0). 크롭 모드 전용.
        crop_fracs:      imgs 순서에 대응하는 각 크롭의 시작 비율 리스트.
                         예) [0.0, 0.15] → A는 0%, B는 15% 지점에서 시작.
    """
    lat, lon = sat["lat"], sat["lon"]
    r = 0.01   # 촬영 범위 반경 (° ≈ 1.1 km)

    base = dict(
        source_type    = "satellite",
        lat_center     = lat,
        lon_center     = lon,
        lat_min        = lat - r,
        lat_max        = lat + r,
        lon_min        = lon - r,
        lon_max        = lon + r,
        resolution_m   = 0.5,
        sensor_platform= sat["name"],
        region_name    = sat["id"],
    )

    # 크롭별 지리 범위 조정: img_rel → frac_start
    frac_map: dict[str, float] = {}
    if crop_axis and crop_size_ratio is not None and crop_fracs:
        for i, img_rel in enumerate(imgs):
            if i < len(crop_fracs):
                frac_map[img_rel] = crop_fracs[i]

    # capture_time 추출: 명시값 우선, 없으면 TIFF 태그 / 파일 수정시간 폴백
    timed = []
    for i, img_rel in enumerate(imgs):
        if explicit_times and i < len(explicit_times):
            ct = explicit_times[i]
        else:
            img_path = SAMPLE_DIR.parent / img_rel  # data/images/<img_rel>
            ct = _extract_tiff_capture_time(img_path)
        timed.append((ct, img_rel))

    timed.sort(key=lambda x: x[0])   # 오래된 것이 앞 (과거 프레임)

    entries = []
    for ct, img_rel in timed:
        entry = dict(base)
        entry["image_file"]   = img_rel
        entry["capture_time"] = ct.isoformat()

        # 크롭 모드: 각 크롭이 원본에서 차지하는 비율로 위경도 범위 재산정
        if img_rel in frac_map:
            frac_start = frac_map[img_rel]
            frac_end   = frac_start + crop_size_ratio
            span       = 2 * r  # 원본 전체 범위 (° 단위)

            if crop_axis == "vertical":
                # X축 이동 → 경도(lon) 범위 조정, 위도(lat) 불변
                entry["lon_min"]    = lon - r + frac_start * span
                entry["lon_max"]    = lon - r + frac_end   * span
                entry["lon_center"] = (entry["lon_min"] + entry["lon_max"]) / 2
            else:  # horizontal
                # Y축 이동 → 위도(lat) 범위 조정, 경도(lon) 불변
                entry["lat_min"]    = lat - r + frac_start * span
                entry["lat_max"]    = lat - r + frac_end   * span
                entry["lat_center"] = (entry["lat_min"] + entry["lat_max"]) / 2

        entries.append(entry)

    return entries


# ── 공개 API ───────────────────────────────────────────────────────────────

def run_step(
    image_mode: str = None,
) -> dict:
    """
    위성 모의기 1 스텝 실행.

    Args:
        image_mode:  "separate" (기본) | "crop"
                     None 이면 src/config.py IMAGE_MODE 값 사용.

    Returns:
        {
          success:     bool,
          elapsed_s:   float,
          image_mode:  str,          # 실제 사용된 모드
          satellites:  list[dict],
          active:      dict,
          images:      list[str],    # 선택/생성된 이미지 경로 (상대)
          stdout_tail: str,
          stderr_tail: str,
        }
    """
    mode  = image_mode or IMAGE_MODE
    mode = image_mode or IMAGE_MODE

    t0 = time.time()

    # ── 1. 위성 위치 계산 ──────────────────────────────────────────────────
    satellites = get_positions()
    active     = get_active_satellite()
    logger.info(
        f"[SimRunner] 활성 위성: {active['name']}  "
        f"lat={active['lat']:.4f}  lon={active['lon']:.4f}  alt={active['alt_km']}km"
    )

    # ── 1-b. 육지 여부 확인 — 바다 위면 파이프라인 건너뜀 ─────────────────
    lat, lon = active["lat"], active["lon"]
    if not is_land(lat, lon):
        skip_reason = f"바다 위 — lat={lat:.4f}, lon={lon:.4f}"
        logger.info("[SimRunner] %s → 이미지 수집·탐지·보고서 생성 건너뜀", skip_reason)
        return {
            "success":     True,
            "skipped":     True,
            "skip_reason": skip_reason,
            "elapsed_s":   round(time.time() - t0, 2),
            "image_mode":  mode,
            "satellites":  satellites,
            "active":      active,
            "images":      [],
            "stdout_tail": "",
            "stderr_tail": "",
        }
    logger.info("[SimRunner] 육지 확인 (lat=%.4f, lon=%.4f) → 파이프라인 진행", lat, lon)

    # ── 2. 이미지 선택 ────────────────────────────────────────────────────
    explicit_times = None
    crop_fracs = None
    if mode == "crop":
        imgs, past_time, curr_time, crop_fracs = _pick_and_crop_image(
            axis=CROP_AXIS,
            crop_size=CROP_SIZE,
            crop_offset=CROP_OFFSET,
        )
        explicit_times = [past_time, curr_time]
        logger.info(f"[SimRunner] 크롭 모드 — 이미지: {imgs}")
    elif mode == "crops_dir":
        imgs, past_time, curr_time, crop_fracs = _pick_from_crops_dir()
        explicit_times = [past_time, curr_time]
        logger.info(f"[SimRunner] .crops 폴더 모드 — 이미지: {imgs}")
    else:
        imgs = _pick_images(2)
        logger.info(f"[SimRunner] 분리 이미지 모드 — 이미지: {imgs}")

    # ── 3. metadata.json 업데이트 ─────────────────────────────────────────
    use_crop_geo = mode in ("crop", "crops_dir")
    meta = _build_metadata(
        active, imgs, explicit_times=explicit_times,
        crop_axis=CROP_AXIS if use_crop_geo else None,
        crop_size_ratio=CROP_SIZE if use_crop_geo else None,
        crop_fracs=crop_fracs,
    )
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[SimRunner] metadata.json 업데이트 완료")

    # ── 4. main.py 파이프라인 실행 ────────────────────────────────────────
    logger.info(f"[SimRunner] 파이프라인 실행: {' '.join(PIPELINE_CMD)}")
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        PIPELINE_CMD,
        capture_output=True,
        text=True,
        timeout=1800,   # 30분 (SR 8000×6000 + SAM3 + LLM 처리 시간 고려)
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0
    success = proc.returncode == 0

    if success:
        logger.info(f"[SimRunner] 파이프라인 완료 ({elapsed:.1f}s)")
    else:
        logger.error(f"[SimRunner] 파이프라인 오류 (rc={proc.returncode}):\n{proc.stderr[-1500:]}")

    return {
        "success":     success,
        "elapsed_s":   round(elapsed, 2),
        "image_mode":  mode,
        "satellites":  satellites,
        "active":      active,
        "images":      imgs,
        "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-500:]  if proc.stderr else "",
    }


def run_step_at(
    lat: float,
    lon: float,
    name: str = "임무 위성",
    target_description: str = "",
    image_mode: str = None,
) -> dict:
    """
    사용자 지정 위경도에서 파이프라인 1스텝 실행 (임무계획 전용).

    위성 궤도 계산 및 육지 체크를 건너뛰고,
    사용자가 지정한 lat/lon 을 촬영 좌표로 사용한다.

    Args:
        lat:        촬영 위도
        lon:        촬영 경도
        name:       위성/센서 이름 (metadata 기록용)
        image_mode: "separate" | "crop" (None 이면 config 기본값)

    Returns:
        run_step() 과 동일한 dict 구조
    """
    mode = image_mode or IMAGE_MODE
    t0   = time.time()

    logger.info(
        "[SimRunner/Mission] 임무 위치 실행: lat=%.4f lon=%.4f name=%s", lat, lon, name
    )

    # ── 이미지 선택 ────────────────────────────────────────────────────────
    explicit_times = None
    crop_fracs     = None
    if mode == "crop":
        imgs, past_time, curr_time, crop_fracs = _pick_and_crop_image(
            axis=CROP_AXIS, crop_size=CROP_SIZE, crop_offset=CROP_OFFSET,
        )
        explicit_times = [past_time, curr_time]
    elif mode == "crops_dir":
        imgs, past_time, curr_time, crop_fracs = _pick_from_crops_dir()
        explicit_times = [past_time, curr_time]
    else:
        imgs = _pick_images(2)

    logger.info("[SimRunner/Mission] 이미지: %s", imgs)

    # ── metadata.json 업데이트 ────────────────────────────────────────────
    sat_stub = {"lat": lat, "lon": lon, "name": name, "id": f"MISSION_{name}"}
    use_crop_geo = mode in ("crop", "crops_dir")
    meta = _build_metadata(
        sat_stub, imgs, explicit_times=explicit_times,
        crop_axis=CROP_AXIS if use_crop_geo else None,
        crop_size_ratio=CROP_SIZE if use_crop_geo else None,
        crop_fracs=crop_fracs,
    )
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[SimRunner/Mission] metadata.json 업데이트 완료")

    # ── 파이프라인 실행 ────────────────────────────────────────────────────
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = list(PIPELINE_CMD)
    if target_description:
        cmd += ["--target-description", target_description]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800,
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0
    success = proc.returncode == 0

    if success:
        logger.info("[SimRunner/Mission] 파이프라인 완료 (%.1fs)", elapsed)
    else:
        logger.error(
            "[SimRunner/Mission] 파이프라인 오류 (rc=%d):\n%s",
            proc.returncode, proc.stderr[-1500:],
        )

    return {
        "success":     success,
        "elapsed_s":   round(elapsed, 2),
        "image_mode":  mode,
        "satellites":  [],
        "active":      sat_stub,
        "images":      imgs,
        "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-500:]  if proc.stderr else "",
    }


def run_ingest_at(
    lat: float,
    lon: float,
    name: str = "임무 위성",
    image_mode: str = None,
) -> dict:
    """
    사용자 지정 위경도에서 이미지 적재만 수행 (단계별 워크플로우 Phase 1).

    탐지·페어링·보고서 생성은 하지 않는다.
    대신 stdout의 "INGEST_RESULT:<json>" 마커에서 session_id와 image_ids를 추출한다.

    Returns:
        {
          success:     bool,
          elapsed_s:   float,
          session_id:  str | None,
          image_ids:   list[str],
          images:      list[str],   # 이미지 파일 상대경로
          active:      dict,
        }
    """
    mode = image_mode or IMAGE_MODE
    t0   = time.time()

    logger.info(
        "[SimRunner/Ingest] 적재 전용 실행: lat=%.4f lon=%.4f name=%s", lat, lon, name
    )

    # ── 이미지 선택 ────────────────────────────────────────────────────────
    explicit_times = None
    crop_fracs     = None
    if mode == "crop":
        imgs, past_time, curr_time, crop_fracs = _pick_and_crop_image(
            axis=CROP_AXIS, crop_size=CROP_SIZE, crop_offset=CROP_OFFSET,
        )
        explicit_times = [past_time, curr_time]
    elif mode == "crops_dir":
        imgs, past_time, curr_time, crop_fracs = _pick_from_crops_dir()
        explicit_times = [past_time, curr_time]
    else:
        imgs = _pick_images(2)

    logger.info("[SimRunner/Ingest] 이미지: %s", imgs)

    # ── metadata.json 업데이트 ────────────────────────────────────────────
    sat_stub = {"lat": lat, "lon": lon, "name": name, "id": f"MISSION_{name}"}
    use_crop_geo = mode in ("crop", "crops_dir")
    meta = _build_metadata(
        sat_stub, imgs, explicit_times=explicit_times,
        crop_axis=CROP_AXIS if use_crop_geo else None,
        crop_size_ratio=CROP_SIZE if use_crop_geo else None,
        crop_fracs=crop_fracs,
    )
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[SimRunner/Ingest] metadata.json 업데이트 완료")

    # ── main.py --ingest-only 실행 ────────────────────────────────────────
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(BASE_DIR / "main.py"),
        "--metadata", str(METADATA_PATH),
        "--ingest-only",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0
    success = proc.returncode == 0

    # ── stdout에서 INGEST_RESULT 파싱 ─────────────────────────────────────
    session_id = None
    image_ids: list = []
    for line in (proc.stdout or "").splitlines():
        if line.startswith("INGEST_RESULT:"):
            try:
                parsed = json.loads(line[len("INGEST_RESULT:"):])
                session_id = parsed.get("session_id")
                image_ids  = parsed.get("image_ids", [])
            except Exception as exc:
                logger.warning("[SimRunner/Ingest] INGEST_RESULT 파싱 오류: %s", exc)
            break

    if success:
        logger.info(
            "[SimRunner/Ingest] 완료 (%.1fs) session=%s images=%d",
            elapsed, session_id, len(image_ids),
        )
    else:
        logger.error(
            "[SimRunner/Ingest] 오류 (rc=%d):\n%s",
            proc.returncode, proc.stderr[-1500:],
        )

    return {
        "success":     success,
        "elapsed_s":   round(elapsed, 2),
        "session_id":  session_id,
        "image_ids":   image_ids,
        "image_mode":  mode,
        "active":      sat_stub,
        "images":      imgs,
        "stdout_tail": proc.stdout[-1000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-500:]  if proc.stderr else "",
    }


# ── 독립 실행 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    result = run_step()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)
