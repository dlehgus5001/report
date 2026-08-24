"""
CelesTrak TLE fetcher + SGP4 propagator.

Fetches orbital elements for military/station satellites from CelesTrak and
propagates them to the current UTC time to obtain sub-satellite lat/lon/alt.

Adapted from Crucix open-source project (space.mjs) for Python/FastAPI backend.
Requires: sgp4 >= 2.22  (pip install sgp4)
"""

import json
import logging
import math
import urllib.request
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"
FETCH_TIMEOUT = 20  # seconds

# Intelligence alert thresholds (Crucix-compatible)
ALERTS = {
    "HIGH_LAUNCH_TEMPO": 50,
    "CHINA_LAUNCHES": 10,
    "RUSSIA_LAUNCHES": 5,
    "MILITARY_TOTAL": 500,
    "STARLINK_TOTAL": 6000,
}

CATEGORY_LABELS = {
    "military": "군사",
    "stations": "우주정거장",
    "starlink": "Starlink",
    "gps-ops": "GPS",
    "last-30-days": "최근발사",
}


# ---------------------------------------------------------------------------
# CelesTrak fetch
# ---------------------------------------------------------------------------

# CelesTrak에서 지정 그룹의 TLE JSON 데이터 가져오기
def _fetch_tle_json(group: str) -> list:
    """Fetch GP elements from CelesTrak as JSON list."""
    url = f"{CELESTRAK_BASE}?GROUP={group}&FORMAT=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MSIS/1.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(f"[TLETracker] CelesTrak fetch failed for '{group}': {exc}")
        return []


# ---------------------------------------------------------------------------
# SGP4 propagation helpers
# ---------------------------------------------------------------------------

# UTC 날짜를 율리우스 날짜(Julian Date)로 변환
def _jday(dt: datetime) -> float:
    """Convert UTC datetime to Julian Date."""
    y, m, d = dt.year, dt.month, dt.day
    h = dt.hour + dt.minute / 60.0 + (dt.second + dt.microsecond / 1e6) / 3600.0
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + h / 24.0 + B - 1524.5


# IAU 1982 기준 그리니치 평균 항성시(GMST) 라디안 계산
def _gmst_rad(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU 1982)."""
    J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    dt_utc = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    D = (dt_utc - J2000).total_seconds() / 86400.0
    gmst_deg = 280.46061837 + 360.98564736629 * D
    return math.radians(gmst_deg % 360.0)


# ECI 위치 벡터를 WGS84 지리 좌표로 변환
def _eci_to_geodetic(r: tuple, dt: datetime):
    """
    Convert ECI position vector r=(x,y,z) km to geodetic (lat_deg, lon_deg, alt_km).
    Uses WGS84 ellipsoid and Bowring iteration.
    """
    a = 6378.137          # WGS84 semi-major axis (km)
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f   # first eccentricity squared

    gmst = _gmst_rad(dt)
    x, y, z = r

    # ECI → ECEF longitude
    lon_rad = math.atan2(y, x) - gmst
    lon_rad = (lon_rad + math.pi) % (2 * math.pi) - math.pi  # normalise to [-π, π]

    # Bowring iteration for geodetic latitude
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(10):
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        lat_new = math.atan2(z + e2 * N * math.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    cos_lat = math.cos(lat)
    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) / abs(math.sin(lat)) - N * (1 - e2)

    return math.degrees(lat), math.degrees(lon_rad), alt


# SGP4로 TLE를 전파하여 위성 위경도·고도 반환
def _propagate_tle(tle1: str, tle2: str, dt: datetime) -> Optional[dict]:
    """
    Propagate TLE lines to dt using sgp4. Returns {lat, lon, alt_km} or None.
    Falls back gracefully if sgp4 is not installed.
    """
    try:
        from sgp4.api import Satrec, jday as sgp4_jday

        sat = Satrec.twoline2rv(tle1, tle2)
        jd = _jday(dt)
        jd_whole = int(jd)
        jd_frac = jd - jd_whole
        e, r, _ = sat.sgp4(jd_whole, jd_frac)
        if e != 0 or r is None:
            return None

        lat, lon, alt = _eci_to_geodetic(r, dt)
        return {"lat": round(lat, 4), "lon": round(lon, 4), "alt_km": round(alt, 1)}

    except ImportError:
        logger.debug("[TLETracker] sgp4 not installed – skipping SGP4 propagation.")
        return None
    except Exception as exc:
        logger.debug(f"[TLETracker] SGP4 propagation error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# CelesTrak TLE로 현재 위성 위치 목록 조회 및 반환
def get_satellite_positions(categories: List[str] = None) -> List[dict]:
    """
    Fetch TLE data from CelesTrak and propagate to current UTC positions.

    Returns a list of satellite dicts:
        name, norad_id, country, object_type, category,
        period_min, inclination, lat, lon, alt_km
    """
    if categories is None:
        categories = ["military", "stations"]

    now = datetime.now(timezone.utc)
    results = []

    for cat in categories:
        raw = _fetch_tle_json(cat)
        logger.info(f"[TLETracker] Fetched {len(raw)} entries from '{cat}'")

        for entry in raw:
            tle1 = entry.get("TLE_LINE1", "")
            tle2 = entry.get("TLE_LINE2", "")
            if not tle1 or not tle2:
                continue

            pos = _propagate_tle(tle1, tle2, now)
            if pos is None:
                continue

            results.append({
                "name": entry.get("OBJECT_NAME", "UNKNOWN").strip(),
                "norad_id": entry.get("NORAD_CAT_ID"),
                "country": entry.get("COUNTRY_CODE", ""),
                "object_type": entry.get("OBJECT_TYPE", ""),
                "period_min": entry.get("PERIOD"),
                "inclination": entry.get("INCLINATION"),
                "category": cat,
                "category_label": CATEGORY_LABELS.get(cat, cat),
                **pos,
            })

    logger.info(f"[TLETracker] Propagated {len(results)} satellite positions.")
    return results


# 우주 정보 집계 데이터(발사 현황·군사위성·ISS)를 반환
def get_space_intelligence() -> dict:
    """
    Returns aggregated space intelligence (Crucix space.mjs briefing equivalent).
    Includes recent launches, military counts, constellation sizes, and signals.
    """
    now = datetime.now(timezone.utc)
    signals = []

    # Recent launches
    recent = _fetch_tle_json("last-30-days")
    total_new = len(recent)
    launch_by_country: dict = {}
    for s in recent:
        cc = s.get("COUNTRY_CODE", "UNKNOWN")
        launch_by_country[cc] = launch_by_country.get(cc, 0) + 1

    if total_new > ALERTS["HIGH_LAUNCH_TEMPO"]:
        signals.append(f"HIGH LAUNCH TEMPO: {total_new}개 신규 우주 객체 (최근 30일)")
    if launch_by_country.get("PRC", 0) > ALERTS["CHINA_LAUNCHES"]:
        signals.append(f"CHINA SPACE ACTIVITY: 최근 30일 {launch_by_country['PRC']}회 발사")
    if launch_by_country.get("CIS", 0) > ALERTS["RUSSIA_LAUNCHES"]:
        signals.append(f"RUSSIA SPACE ACTIVITY: 최근 30일 {launch_by_country['CIS']}회 발사")

    # Military satellites
    military = _fetch_tle_json("military")
    mil_count = len(military)
    mil_by_country: dict = {}
    for s in military:
        cc = s.get("COUNTRY_CODE", "UNKNOWN")
        mil_by_country[cc] = mil_by_country.get(cc, 0) + 1
    if mil_count > ALERTS["MILITARY_TOTAL"]:
        signals.append(f"MILITARY CONSTELLATION: {mil_count}기 군사 위성 추적 중")

    # Space stations + ISS
    stations_raw = _fetch_tle_json("stations")
    iss = next(
        (s for s in stations_raw if s.get("NORAD_CAT_ID") == 25544),
        None,
    )
    iss_pos = None
    if iss:
        tle1, tle2 = iss.get("TLE_LINE1", ""), iss.get("TLE_LINE2", "")
        if tle1 and tle2:
            iss_pos = _propagate_tle(tle1, tle2, now)

    return {
        "timestamp": now.isoformat(),
        "totalNewObjects": total_new,
        "launchByCountry": launch_by_country,
        "militarySatellites": mil_count,
        "militaryByCountry": mil_by_country,
        "iss": {**(iss_pos or {}), "name": "ISS (ZARYA)"} if iss else None,
        "signals": signals,
    }
