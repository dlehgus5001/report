"""
육지/바다 판별 유틸리티 (global_land_mask 기반)

global_land_mask 미설치 시 항상 육지(True)를 반환해 기존 동작을 유지한다.
설치: pip install global-land-mask
"""

import logging

logger = logging.getLogger(__name__)

try:
    from global_land_mask import globe as _globe
    _GLOBE_AVAILABLE = True
    logger.debug("[LandCheck] global_land_mask 로드 성공")
except ImportError:
    _GLOBE_AVAILABLE = False
    logger.warning(
        "[LandCheck] global_land_mask 미설치 — 항상 육지로 판별합니다. "
        "pip install global-land-mask 으로 설치하세요."
    )


# 위경도 좌표가 육지인지 판별하여 True/False 반환
def is_land(lat: float, lon: float) -> bool:
    """
    주어진 위경도 좌표가 육지인지 판별한다.

    Args:
        lat: 위도  (-90 ~ 90)
        lon: 경도  (-180 ~ 180)

    Returns:
        True  → 육지  (파이프라인 실행)
        False → 바다  (파이프라인 건너뜀)
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        logger.warning("[LandCheck] 잘못된 좌표 lat=%.4f lon=%.4f — 육지로 간주", lat, lon)
        return True

    if not _GLOBE_AVAILABLE:
        return True  # 폴백: 항상 육지로 간주

    return bool(_globe.is_land(lat, lon))
