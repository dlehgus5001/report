"""
위성 궤도 모의기 (Satellite Orbital Simulator)

CelesTrak / TLE 없이 케플러 원형궤도 공식으로 위성의 현재 위경도를 계산.

궤도 방정식 (원형궤도, 구면 근사):
  M  = ω·Δt + φ₀                              # 평균근점이각
  φ  = arcsin(sin(i)·sin(M))                  # 위위도 (sub-satellite latitude)
  λ  = RAAN + arctan2(cos(i)·sin(M), cos(M)) − ω_e·Δt  # 경도 (지구 자전 보정)

환경 변수:
  SIM_TIME_WARP  모의 가속 배율 (기본: 20x → 현실 1초 = 궤도 20초)
"""

import math
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

# ── 상수 ──────────────────────────────────────────────────────────────────
EARTH_ROT_DEG_PER_SEC = 360.0 / 86164.1   # 지구 자전 속도 (도/초, 항성일 기준)
TIME_WARP = float(os.getenv("SIM_TIME_WARP", "20"))  # 가속 배율

# 기준 에포크: 서버 시작 시점으로 동적 설정
# → RAAN + phase_deg 값이 서버 시작 즉시 각 위성을 목표 위치에 배치
_EPOCH = time.time()

# ── 모의 위성 정의 ─────────────────────────────────────────────────────────
# 한국 군사위성 계열 파라미터 (실제 태양동기궤도 기반)
#
# [중국·북한 집중 감시 전략]
#   _EPOCH = time.time() (서버 시작 시점)으로 설정하여,
#   phase_deg 값이 서버 시작 즉시 각 위성을 지정 목표 상공에 배치하도록 역산.
#
#   목표 위치 (서버 시작 시 초기 배치):
#     KMS-001 → 중국 동북 45°N 90°E  (RAAN 97.5°,  phase 45.5°)
#     KMS-002 → 중국 중부 35°N 109°E (RAAN 114.2°, phase 35.3°)
#     KMS-003 → 중국 동부 40°N 116°E (RAAN 123.3°, phase 40.5°)
#     KMS-004 → 북한 평양  39°N 126°E (RAAN 131.5°, phase 39.3°)
#     KMS-005 → 중국 남부 23°N 113°E (RAAN 116.2°, phase 23.2°)
SATELLITES: List[dict] = [
    {
        "id": "KMS-001", "name": "한국군위성-1", "country": "KOR",
        "inclination": 97.4,   # SSO (태양동기궤도)
        "period_s":    5640,   # ~94분 (고도 480km LEO)
        "raan_deg":    110.6,  # 전략E: 동아시아 중심, 36° 간격 분산
        "phase_deg":   37.4,   # 전략E: 72° 등간격 분산 → 독립적 커버리지
        "alt_km":      480,
        "category":    "military",
    },
    {
        "id": "KMS-002", "name": "한국군위성-2", "country": "KOR",
        "inclination": 97.4,
        "period_s":    5640,
        "raan_deg":    146.6,  # 전략E: +36°
        "phase_deg":   109.4,  # 전략E: +72°
        "alt_km":      480,
        "category":    "military",
    },
    {
        "id": "KMS-003", "name": "한국군위성-3", "country": "KOR",
        "inclination": 98.2,
        "period_s":    5760,
        "raan_deg":    182.6,  # 전략E: +72°
        "phase_deg":   181.4,  # 전략E: +144°
        "alt_km":      510,
        "category":    "military",
    },
    {
        "id": "KMS-004", "name": "한국군위성-4", "country": "KOR",
        "inclination": 97.0,
        "period_s":    5580,
        "raan_deg":    218.6,  # 전략E: +108°
        "phase_deg":   253.4,  # 전략E: +216°
        "alt_km":      460,
        "category":    "military",
    },
    {
        "id": "KMS-005", "name": "한국군위성-5", "country": "KOR",
        "inclination": 97.6,
        "period_s":    5700,
        "raan_deg":    254.6,  # 전략E: +144°
        "phase_deg":   325.4,  # 전략E: +288°
        "alt_km":      490,
        "category":    "military",
    },
]


# ── 핵심 계산 함수 ─────────────────────────────────────────────────────────

# 케플러 원형궤도 공식으로 위성 위경도 계산
def _compute_position(cfg: dict, t_real: float) -> dict:
    """
    실제 unix 시각 t_real 에서 위성의 위경도를 계산.
    TIME_WARP 배율을 적용해 궤도 시간으로 변환.

    Returns dict: id, name, country, category, lat, lon, alt_km, period_min, inclination
    """
    i   = math.radians(cfg["inclination"])
    T   = cfg["period_s"]
    w   = 2 * math.pi / T          # 각속도 (rad/s)
    phi = math.radians(cfg["phase_deg"])
    raan = math.radians(cfg["raan_deg"])

    # 궤도 경과 시간 (TIME_WARP 적용)
    dt = (t_real - _EPOCH) * TIME_WARP

    # 평균근점이각
    M = w * dt + phi

    # 위위도 φ = arcsin(sin(i)·sin(M))
    lat_rad = math.asin(math.sin(i) * math.sin(M))

    # 승교점에서의 경도 진행 + 지구 자전 빼기
    u = math.atan2(math.cos(i) * math.sin(M), math.cos(M))
    lon_deg = math.degrees(raan + u) - EARTH_ROT_DEG_PER_SEC * dt
    lon_deg = ((lon_deg + 180) % 360) - 180  # −180 ‥ 180 정규화

    return {
        "id":            cfg["id"],
        "name":          cfg["name"],
        "country":       cfg["country"],
        "category":      cfg["category"],
        "category_label": "군사위성",
        "period_min":    round(cfg["period_s"] / 60, 2),
        "inclination":   cfg["inclination"],
        "alt_km":        cfg["alt_km"],
        "lat":           round(math.degrees(lat_rad), 4),
        "lon":           round(lon_deg, 4),
    }


# 현재 시각 기준 모든 시뮬레이션 위성 위치 반환
def get_positions(t_real: Optional[float] = None) -> List[dict]:
    """현재 시각(또는 지정 unix 시각) 기준 모든 시뮬레이션 위성 위치 반환."""
    if t_real is None:
        t_real = time.time()
    return [_compute_position(cfg, t_real) for cfg in SATELLITES]


# 중국·북한 상공 감시 우선 위성을 선택하여 반환
def get_active_satellite(t_real: Optional[float] = None) -> dict:
    """
    중국·북한 상공 위성을 우선 선택하여 반환.
    metadata.json 의 촬영 좌표(lat_center/lon_center)로 사용됨.

    선택 우선순위:
      1순위: 중국·북한 감시 구역(위도 18–53°N, 경도 73–135°E) 내 위성 중
             북한 중심(40°N, 127°E)에 가장 가까운 위성
      2순위: 감시 구역 내 위성이 없을 때, 중국·북한 통합 중심(39°N, 112°E)에
             가장 가까운 위성 (궤도 이동 중 최단 시간 내 진입 위성)
    """
    if t_real is None:
        t_real = time.time()
    positions = get_positions(t_real)

    # 1순위: 중국·북한 감시 구역 내 위성
    CHINA_NK_LAT = (18.0, 53.0)
    CHINA_NK_LON = (73.0, 135.0)
    in_zone = [
        s for s in positions
        if CHINA_NK_LAT[0] <= s["lat"] <= CHINA_NK_LAT[1]
        and CHINA_NK_LON[0] <= s["lon"] <= CHINA_NK_LON[1]
    ]
    if in_zone:
        # 구역 내에서는 북한 중심에 가장 가까운 위성 선택 (고위협 우선)
        NK_CENTER = (40.0, 127.0)
        return min(
            in_zone,
            key=lambda s: (s["lat"] - NK_CENTER[0]) ** 2 + (s["lon"] - NK_CENTER[1]) ** 2,
        )

    # 2순위: 구역 내 위성 없음 → 중국·북한 통합 중심에 가장 가까운 위성
    FALLBACK_TARGET = (39.0, 112.0)
    return min(
        positions,
        key=lambda s: (s["lat"] - FALLBACK_TARGET[0]) ** 2 + (s["lon"] - FALLBACK_TARGET[1]) ** 2,
    )
