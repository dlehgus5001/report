"""
MSIS Web API – FastAPI 백엔드 (위성 모의기 + 탐지 결과 대시보드)

엔드포인트:
  GET  /api/satellites          – 위성 모의기 현재 위치 (케플러 원형궤도 계산)
  POST /api/simulator/step      – 1스텝 실행 (metadata 갱신 → main.py 파이프라인)
  GET  /api/detections          – SAM3 탐지 결과 (DB 최신)
  GET  /api/image/latest        – 최신 위성영상 메타데이터 + base64
  GET  /api/report/latest       – 최신 판독 보고서 (DB)
  GET  /api/report/{id}         – 보고서 단건 조회
  GET  /api/reports             – 보고서 목록
  GET  /dashboard               – 대시보드 HTML

실행:
  uvicorn web_api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import io
import json
import logging
import queue
import threading
import time as _time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pycountry
import reverse_geocoder as _rg


# 위경도로 국가명 반환
def _get_country_name(lat: float, lon: float) -> Optional[str]:
    """위경도로 국가명(영문) 반환. 좌표 없으면 None."""
    try:
        cc = _rg.search((lat, lon))[0]["cc"]
        country = pycountry.countries.get(alpha_2=cc)
        return country.name if country else cc
    except Exception:
        return None

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import IMAGES_DIR, SR_TARGET_W, SR_TARGET_H
from src.database.pairing_db import (
    get_session_ids_near, get_session_location, get_pairings_by_session,
    update_pairings_detection_refs, get_pairings_near_time,
    update_pairing, delete_pairing,
)
from src.database.reports_db import (
    get_all_reports,
    get_latest_report_for_sessions,
    get_report_by_id,
    update_report_content,
)
from src.database.sensor_db import (
    get_latest_detections_near,
    get_latest_image_near,
    get_image_record_by_id,
    get_detection_by_id,
    get_detections_by_image,
    get_images_by_capture_time,
    get_images_by_session,
    get_all_images_with_count,
    replace_detections_for_image,
    delete_detection_by_id,
)
from src.satellite.simulator import get_positions
from src.utils.hwpx_writer import make_hwpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── uvicorn access 로그 필터 ───────────────────────────────────────────────
# 폴링성 엔드포인트(이미지 목록·썸네일·렌더 등)는 터미널 노이즈가 심하므로 숨긴다.
_MUTE_PATHS = (
    "/api/images",
    "/api/image/",   # /api/image/<id>/thumb, /api/image/<id>/rendered 등
    "/api/detections",
    "/api/events",
    "/api/satellites",
)

class _SilentAccessFilter(logging.Filter):
    # 폴링 경로 요청 로그를 필터링
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in _MUTE_PATHS)

for _uvicorn_logger_name in ("uvicorn.access",):
    _ul = logging.getLogger(_uvicorn_logger_name)
    _ul.addFilter(_SilentAccessFilter())

app = FastAPI(
    title="MSIS Satellite Dashboard API",
    description="위성 모의기 + SAM3 객체탐지 + 판독보고서 대시보드",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════
# Server-Sent Events (SSE) – 실시간 DB 업데이트 브로드캐스트
# ══════════════════════════════════════════════════════════════════════════

_sse_clients: list = []   # 연결된 클라이언트 Queue 목록
_sse_lock = threading.Lock()


# DB 갱신 이벤트를 SSE 클라이언트에 브로드캐스트
def _notify_db_updated(run_count: int = 0, success: bool = True, elapsed: float = 0.0,
                       changed: list | None = None):
    """모든 SSE 구독 클라이언트에게 DB 업데이트 이벤트를 브로드캐스트한다.

    changed: 갱신된 데이터 종류 목록. 예: ["detections", "reports"]
             None이면 전체("images", "detections", "reports")로 간주.
    """
    payload = json.dumps({
        "type":      "db_updated",
        "run_count": run_count,
        "success":   success,
        "elapsed":   elapsed,
        "changed":   changed or ["images", "detections", "reports"],
        "ts":        datetime.utcnow().isoformat(),
    })
    with _sse_lock:
        dead = []
        for q in list(_sse_clients):
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass


# 임의 SSE 페이로드를 모든 클라이언트에 브로드캐스트
def _broadcast_sse(payload: str):
    with _sse_lock:
        dead = []
        for q in list(_sse_clients):
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass


# 세션 ID → 분석 작업 상태 ("processing" | "done" | "error:<msg>")
_analyze_jobs: dict[str, str] = {}
_analyze_jobs_lock = threading.Lock()


# 분석을 백그라운드 스레드로 실행하고 완료 시 SSE로 알림
def _run_analyze_background(session_id: str, target_description: str):
    try:
        from pipeline import MavenPipeline
        pipeline = MavenPipeline()
        pipeline.run_pairing_and_report(
            session_id, target_description=target_description
        )
    except Exception as exc:
        logger.error("[API] background analyze error: %s", exc, exc_info=True)
        with _analyze_jobs_lock:
            _analyze_jobs[session_id] = f"error:{exc}"
        _broadcast_sse(json.dumps({
            "type":       "analyze_error",
            "session_id": session_id,
            "message":    str(exc),
        }))
        return

    # WP 완료 처리
    after_rpt_id = _get_latest_report_id()
    with _swp_lock:
        wp_info = _session_wp_map.pop(session_id, None)

    if wp_info:
        if len(wp_info) == 2:
            sat_id, wp_id = wp_info
            with _sat_lock:
                sm = _sat_missions.get(sat_id, {})
                wp = next((w for w in sm.get("waypoints", []) if w["id"] == wp_id), None)
                if wp:
                    wp["status"] = "complete"
                    if after_rpt_id:
                        wp["report_id"] = after_rpt_id
            _save_sat_missions()
        else:
            _, wp_id, mission_id = wp_info
            mission = _missions.get(mission_id)
            if mission:
                wp = next((w for w in mission["waypoints"] if w["id"] == wp_id), None)
                if wp:
                    wp["status"] = "complete"
                    if after_rpt_id:
                        wp["report_id"] = after_rpt_id
                non_done = [
                    w for w in mission["waypoints"]
                    if w["status"] not in ("complete", "ingested", "failed")
                ]
                if not non_done:
                    mission["status"]      = "complete"
                    mission["finished_at"] = datetime.utcnow().isoformat()
            _save_missions_to_file()

    with _analyze_jobs_lock:
        _analyze_jobs[session_id] = f"done:{after_rpt_id or ''}"

    _notify_db_updated(changed=["reports", "detections"])
    _broadcast_sse(json.dumps({
        "type":       "analyze_complete",
        "session_id": session_id,
        "report_id":  after_rpt_id,
    }))


# 파이프라인 단계 이벤트를 SSE 전송
def _notify_pipeline_stage(status: dict):
    """파이프라인 단계 변경 이벤트를 모든 SSE 클라이언트에 브로드캐스트한다."""
    payload = json.dumps({
        "type":  "pipeline_stage",
        "stage": status.get("stage", "idle"),
        "label": status.get("label", ""),
        "lat":   status.get("lat"),
        "lon":   status.get("lon"),
        "ts":    status.get("ts", ""),
    })
    with _sse_lock:
        dead = []
        for q in list(_sse_clients):
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass


# 이미지 적재 완료 이벤트 SSE 브로드캐스트
def _notify_image_ingested(
    session_id: str,
    image_ids: list,
    lat: float | None = None,
    lon: float | None = None,
    target_description: str = "",
    wp_name: str = "",
    wp_seq: int = 0,
):
    """이미지 적재 완료 이벤트 브로드캐스트 – 단계별 워크플로우 UI 트리거."""
    payload = json.dumps({
        "type":               "image_ingested",
        "session_id":         session_id,
        "image_ids":          image_ids,
        "lat":                lat,
        "lon":                lon,
        "target_description": target_description,
        "wp_name":            wp_name,
        "wp_seq":             wp_seq,
        "ts":                 datetime.utcnow().isoformat(),
    })
    with _sse_lock:
        dead = []
        for q in list(_sse_clients):
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass


# ── session_id → (sat_id, wp_id) 매핑: analyze 엔드포인트가 WP 완료 처리에 사용 ──
_session_wp_map: dict = {}  # {session_id: (sat_id, wp_id)}
_swp_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
# 임무계획 (Mission Planning) — JSON 파일 영속 저장
# ══════════════════════════════════════════════════════════════════════════

import uuid as _uuid

_MISSIONS_FILE = Path(__file__).parent / "data" / "missions.json"
_missions: dict = {}   # mission_id → mission dict
_mission_lock = threading.Lock()

# ── 위성별 임무계획 ────────────────────────────────────────────────────────
_SAT_MISSIONS_FILE = Path(__file__).parent / "data" / "sat_missions.json"
_sat_missions: dict = {}   # sat_id → {sat_id, sat_name, waypoints, assigned_at}
_sat_lock = threading.Lock()

# ── 캠페인 (전체 위성 일괄 실행) 상태 ────────────────────────────────────
_campaign: dict = {
    "status":       "idle",   # idle | running | complete | failed
    "started_at":   None,
    "finished_at":  None,
    "queue":        [],        # [{sat_id, sat_name, wp_id, wp_name}] 남은 작업
    "current":      None,      # {sat_id, sat_name, wp_name, lat, lon}
    "total":        0,
    "done":         0,
    "failed_count": 0,
}
_camp_lock = threading.Lock()


# 임무 파일 로드 및 고착 상태 초기화
def _load_missions_from_file():
    if _MISSIONS_FILE.exists():
        try:
            data = json.loads(_MISSIONS_FILE.read_text(encoding="utf-8"))
            _missions.update(data)
        except Exception:
            pass
    # 서버 재시작 시 running 상태로 고착된 임무/웨이포인트를 failed 로 초기화
    for m in _missions.values():
        if m.get("status") == "running":
            m["status"] = "failed"
        for wp in m.get("waypoints", []):
            if wp.get("status") == "running":
                wp["status"] = "failed"


# 임무 정보를 JSON 파일로 저장
def _save_missions_to_file():
    _MISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _mission_lock:
        _MISSIONS_FILE.write_text(
            json.dumps(_missions, ensure_ascii=False, indent=2), encoding="utf-8"
        )




_load_missions_from_file()


# 위성 임무 파일 로드 및 상태 초기화
def _load_sat_missions():
    if _SAT_MISSIONS_FILE.exists():
        try:
            data = json.loads(_SAT_MISSIONS_FILE.read_text(encoding="utf-8"))
            _sat_missions.update(data)
        except Exception:
            pass
    # 서버 재시작 시 running 고착 WP → failed 초기화
    for sm in _sat_missions.values():
        for wp in sm.get("waypoints", []):
            if wp.get("status") == "running":
                wp["status"] = "failed"


# 위성 임무 정보를 JSON 파일로 저장
def _save_sat_missions():
    _SAT_MISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _sat_lock:
        _SAT_MISSIONS_FILE.write_text(
            json.dumps(_sat_missions, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ── 서버 시작 시 상태 파일 초기화 ─────────────────────────────────────────
# 이전 실행에서 고착된 pipeline_status / sat_missions 를 항상 깨끗하게 시작한다.
# 서버 시작 시 상태 파일 초기화
def _reset_state_files() -> None:
    """pipeline_status.json → idle, sat_missions.json → {} 로 초기화."""
    try:
        from src.pipeline_status import clear_status as _ps_clear
        _ps_clear()
        logger.info("[Startup] pipeline_status.json 초기화 완료")
    except Exception as e:
        logger.warning("[Startup] pipeline_status 초기화 실패: %s", e)
    try:
        with _sat_lock:
            _sat_missions.clear()
        _SAT_MISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SAT_MISSIONS_FILE.write_text("{}", encoding="utf-8")
        logger.info("[Startup] sat_missions.json 초기화 완료")
    except Exception as e:
        logger.warning("[Startup] sat_missions 초기화 실패: %s", e)

_reset_state_files()

# 위성영상 정적 파일
_images_dir = Path(IMAGES_DIR)
if _images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(_images_dir)), name="images")

# 대시보드 정적 파일 (Leaflet CSS/JS, GeoJSON 등 – 폐쇄망 로컬 서빙)
# 경로를 /lib 으로 분리 – /static/images 와 prefix 충돌 방지
_dashboard_static = Path(__file__).parent / "dashboard" / "lib"
_dashboard_static.mkdir(parents=True, exist_ok=True)
app.mount("/lib", StaticFiles(directory=str(_dashboard_static)), name="dashboard_static")

_dashboard_path = Path(__file__).parent / "dashboard" / "index.html"


# ══════════════════════════════════════════════════════════════════════════
# 자동 시뮬레이션 백그라운드 스레드
# ══════════════════════════════════════════════════════════════════════════

_auto_state: dict = {
    "enabled":        False,  # 자동 실행 ON/OFF  (기본 OFF — 사용자가 수동으로 활성화)
    "running":        False,  # 파이프라인 실행 중 여부
    "last_run":       None,   # 마지막 완료 시각 (ISO str)
    "run_count":      0,      # 누적 실행 횟수
    "last_success":   None,   # 마지막 실행 성공 여부
    "last_elapsed":   None,   # 마지막 실행 소요 시간 (s)
    "last_images":    [],     # 마지막 선택 이미지
    "last_active":    None,   # 마지막 활성 위성
    "last_positions": {},     # sat_id → (lat, lon)  위치 변경 감지용
}
_active_mission_id: Optional[str] = None   # 현재 자동 시뮬레이터가 처리 중인 임무 ID
_auto_lock = threading.Lock()

POSITION_CHANGE_THRESHOLD_DEG = 0.003   # ~300 m – 이 이상 이동하면 실행
AUTO_SIM_POLL_SEC              = 10      # 위치 체크 간격 (초)


# 위성 위치 변경 여부 확인
def _any_satellite_moved(new_sats: list) -> bool:
    """이전 위치 대비 임계값 이상 이동한 위성이 있으면 True."""
    prev = _auto_state["last_positions"]
    if not prev:
        return True
    for s in new_sats:
        p = prev.get(s["id"])
        if p is None:
            return True
        if (abs(s["lat"] - p[0]) > POSITION_CHANGE_THRESHOLD_DEG or
                abs(s["lon"] - p[1]) > POSITION_CHANGE_THRESHOLD_DEG):
            return True
    return False


# DB에서 최신 보고서 ID 조회
def _get_latest_report_id() -> str | None:
    """DB에서 가장 최근 보고서 ID 반환. 조회 실패 시 None."""
    try:
        from src.database.reports_db import get_all_reports
        rpts = get_all_reports(limit=1)
        return rpts[0].id if rpts else None
    except Exception:
        return None


# 웨이포인트 실행 결과를 상태에 반영
def _finish_wp(wp: dict, result: dict, before_report_id: str | None = None):
    """웨이포인트 실행 결과 반영 (성공/실패 공통).

    파이프라인이 비정상 종료(returncode!=0)여도 DB에 새 보고서가 실제로
    생성된 경우(RAG 실패 후 LLM 폴백 등)는 complete 로 처리한다.

    Args:
        before_report_id: WP 실행 직전 DB 최신 보고서 ID.
                          실행 후 ID가 달라지면 새 보고서가 생성된 것.
    """
    after_report_id = _get_latest_report_id()
    new_report_created = (
        after_report_id is not None and after_report_id != before_report_id
    )

    if result["success"] or new_report_created:
        if after_report_id:
            wp["report_id"] = after_report_id
        wp["status"]    = "complete"
        wp["elapsed_s"] = result["elapsed_s"]
        wp.pop("error", None)
    else:
        wp["status"] = "failed"
        wp["error"]  = result.get("stderr_tail", "")[:300]


# 자동 시뮬레이션 백그라운드 루프 실행
def _auto_sim_worker():
    """
    자동 시뮬레이션 백그라운드 스레드.

    우선순위:
      1. 캠페인 실행 중 → 다음 위성 WP 실행
      2. 활성 임무(_active_mission_id)가 있으면 → 다음 pending 웨이포인트 실행
      3. 없으면 → 위성 궤도 위치에서 일반 시뮬레이션
    """
    global _active_mission_id
    logger.info("[AutoSim] 백그라운드 스레드 시작 (%.1fs 간격)", AUTO_SIM_POLL_SEC)
    _time.sleep(3)

    while True:
        try:
            # ── 캠페인 모드 (최우선) ──────────────────────────────────────
            with _camp_lock:
                camp_running = (_campaign["status"] == "running")

            if camp_running:
                if _auto_state["running"]:
                    _time.sleep(AUTO_SIM_POLL_SEC)
                    continue

                # 다음 WP 꺼내기
                with _camp_lock:
                    if not _campaign["queue"]:
                        _campaign["status"]      = "complete"
                        _campaign["finished_at"] = datetime.utcnow().isoformat()
                        logger.info("[Campaign] 캠페인 완료 (total=%d, failed=%d)",
                                    _campaign["total"], _campaign["failed_count"])
                        _time.sleep(AUTO_SIM_POLL_SEC)
                        continue
                    item = _campaign["queue"].pop(0)
                    _campaign["current"] = {
                        "sat_id":   item["sat_id"],
                        "sat_name": item["sat_name"],
                        "wp_name":  item["wp_name"],
                    }

                sat_id = item["sat_id"]
                wp_id  = item["wp_id"]

                # sat_missions 에서 WP 레퍼런스 찾기
                with _sat_lock:
                    sm     = _sat_missions.get(sat_id, {})
                    wp_ref = next((w for w in sm.get("waypoints", []) if w["id"] == wp_id), None)

                if not wp_ref:
                    logger.warning("[Campaign] WP 없음: sat=%s wp=%s", sat_id, wp_id)
                    with _camp_lock:
                        _campaign["done"]         += 1
                        _campaign["failed_count"] += 1
                        _campaign["current"]       = None
                    continue

                wp_ref["status"] = "running"
                _save_sat_missions()

                logger.info("[Campaign] WP 실행: %s / %s (lat=%.4f lon=%.4f)",
                            item["sat_name"], wp_ref["name"], wp_ref["lat"], wp_ref["lon"])

                with _auto_lock:
                    _auto_state["running"] = True
                try:
                    from simulator_runner import run_ingest_at
                    result = run_ingest_at(
                        lat=wp_ref["lat"],
                        lon=wp_ref["lon"],
                        name=item["sat_name"],
                    )
                    _auto_state["run_count"]   += 1
                    _auto_state["last_run"]     = datetime.utcnow().isoformat()
                    _auto_state["last_success"] = result["success"]
                    _auto_state["last_elapsed"] = result["elapsed_s"]
                    _auto_state["last_images"]  = result.get("images", [])
                    _auto_state["last_active"]  = {
                        "name": item["sat_name"],
                        "lat":  wp_ref["lat"],
                        "lon":  wp_ref["lon"],
                    }
                    if result["success"] and result.get("session_id"):
                        sess_id = result["session_id"]
                        wp_ref["status"]         = "ingested"
                        wp_ref["session_id"]     = sess_id
                        wp_ref["image_ids"]      = result.get("image_ids", [])
                        wp_ref["target_description"] = wp_ref.get("target_description", "")
                        with _swp_lock:
                            _session_wp_map[sess_id] = (sat_id, wp_id)
                        _notify_image_ingested(
                            sess_id,
                            result.get("image_ids", []),
                            lat=wp_ref["lat"],
                            lon=wp_ref["lon"],
                            target_description=wp_ref.get("target_description", ""),
                            wp_name=wp_ref.get("name", ""),
                            wp_seq=wp_ref.get("seq", 0) + 1,
                        )
                    else:
                        wp_ref["status"] = "failed"
                        with _camp_lock:
                            _campaign["failed_count"] += 1
                except Exception as exc:
                    wp_ref["status"] = "failed"
                    wp_ref["error"]  = str(exc)[:300]
                    logger.error("[Campaign] WP 오류: %s", exc)
                    with _camp_lock:
                        _campaign["failed_count"] += 1
                finally:
                    with _camp_lock:
                        _campaign["done"]    += 1
                        _campaign["current"]  = None
                    _save_sat_missions()
                    _notify_pipeline_stage({"stage": "idle", "label": "", "lat": None, "lon": None})
                    _notify_db_updated(
                        run_count=_auto_state["run_count"],
                        success=_auto_state.get("last_success", False),
                        elapsed=_auto_state.get("last_elapsed", 0),
                        changed=["images"],
                    )
                    with _auto_lock:
                        _auto_state["running"] = False
                continue   # 다음 WP로

            # auto 비활성 + 임무도 없으면 아무것도 하지 않음
            has_mission = bool(_active_mission_id)
            if not (_auto_state["enabled"] or has_mission):
                _time.sleep(AUTO_SIM_POLL_SEC)
                continue

            if _auto_state["running"]:
                _time.sleep(AUTO_SIM_POLL_SEC)
                continue

            # ── 임무 모드 ─────────────────────────────────────────────────
            if has_mission:
                mission = _missions.get(_active_mission_id)
                if not mission:
                    _active_mission_id = None
                    continue

                next_wp = next(
                    (wp for wp in mission["waypoints"] if wp["status"] == "pending"),
                    None,
                )

                if next_wp is None:
                    # 모든 웨이포인트 완료
                    mission["status"]      = "complete"
                    mission["finished_at"] = datetime.utcnow().isoformat()
                    _save_missions_to_file()
                    logger.info("[AutoSim/Mission] 임무 완료: %s", _active_mission_id)
                    _active_mission_id = None
                    continue

                # 웨이포인트 실행
                wp_seq = next_wp["seq"] + 1
                total  = len(mission["waypoints"])
                logger.info(
                    "[AutoSim/Mission] WP %d/%d — %s (lat=%.4f lon=%.4f)",
                    wp_seq, total, next_wp["name"], next_wp["lat"], next_wp["lon"],
                )
                next_wp["status"] = "running"
                _save_missions_to_file()

                with _auto_lock:
                    _auto_state["running"] = True
                try:
                    from simulator_runner import run_ingest_at
                    result = run_ingest_at(
                        lat=next_wp["lat"],
                        lon=next_wp["lon"],
                        name=next_wp.get("name", "임무 위성"),
                    )
                    _auto_state["run_count"]   += 1
                    _auto_state["last_run"]     = datetime.utcnow().isoformat()
                    _auto_state["last_success"] = result["success"]
                    _auto_state["last_elapsed"] = result["elapsed_s"]
                    _auto_state["last_images"]  = result.get("images", [])
                    _auto_state["last_active"]  = {
                        "name": next_wp.get("name", "임무 위성"),
                        "lat":  next_wp["lat"],
                        "lon":  next_wp["lon"],
                    }
                    if result["success"] and result.get("session_id"):
                        sess_id = result["session_id"]
                        next_wp["status"]     = "ingested"
                        next_wp["session_id"] = sess_id
                        next_wp["image_ids"]  = result.get("image_ids", [])
                        # 미션 ID + WP ID를 session 맵에 저장
                        with _swp_lock:
                            _session_wp_map[sess_id] = ("__mission__", next_wp["id"], _active_mission_id)
                        _notify_image_ingested(
                            sess_id,
                            result.get("image_ids", []),
                            lat=next_wp["lat"],
                            lon=next_wp["lon"],
                            target_description=next_wp.get("target_description", ""),
                            wp_name=next_wp.get("name", ""),
                            wp_seq=next_wp.get("seq", 0) + 1,
                        )
                    else:
                        next_wp["status"] = "failed"
                        next_wp["error"]  = result.get("stderr_tail", "")[:300]
                except Exception as exc:
                    next_wp["status"] = "failed"
                    next_wp["error"]  = str(exc)[:300]
                    logger.error("[AutoSim/Mission] WP 실행 오류: %s", exc)
                finally:
                    _save_missions_to_file()
                    _notify_pipeline_stage({"stage": "idle", "label": "", "lat": None, "lon": None})
                    _notify_db_updated(
                        run_count=_auto_state["run_count"],
                        success=_auto_state.get("last_success", False),
                        elapsed=_auto_state.get("last_elapsed", 0),
                        changed=["images"],
                    )
                    with _auto_lock:
                        _auto_state["running"] = False

            # ── 일반 궤도 모드 ─────────────────────────────────────────────
            else:
                sats = get_positions()
                if not _any_satellite_moved(sats):
                    _time.sleep(AUTO_SIM_POLL_SEC)
                    continue

                _auto_state["last_positions"] = {
                    s["id"]: (s["lat"], s["lon"]) for s in sats
                }
                with _auto_lock:
                    _auto_state["running"] = True

                logger.info("[AutoSim] 좌표 변경 감지 → 파이프라인 시작 (#%d)",
                            _auto_state["run_count"] + 1)
                try:
                    from simulator_runner import run_step
                    result = run_step()
                    if result.get("skipped"):
                        logger.info("[AutoSim] 건너뜀 — %s", result.get("skip_reason", ""))
                    else:
                        _auto_state["run_count"]   += 1
                        _auto_state["last_run"]     = datetime.utcnow().isoformat()
                        _auto_state["last_success"] = result["success"]
                        _auto_state["last_elapsed"] = result["elapsed_s"]
                        _auto_state["last_images"]  = result.get("images", [])
                        _auto_state["last_active"]  = result.get("active")
                        logger.info("[AutoSim] 완료 (#%d, %.1fs, success=%s)",
                                    _auto_state["run_count"],
                                    result["elapsed_s"], result["success"])
                        _notify_db_updated(
                            run_count=_auto_state["run_count"],
                            success=result["success"],
                            elapsed=result["elapsed_s"],
                        )
                except Exception as exc:
                    logger.error("[AutoSim] 실행 오류: %s", exc)
                    _notify_db_updated(run_count=_auto_state["run_count"], success=False)
                finally:
                    with _auto_lock:
                        _auto_state["running"] = False

        except Exception as exc:
            logger.error("[AutoSim] 루프 오류: %s", exc)

        _time.sleep(AUTO_SIM_POLL_SEC)


# ══════════════════════════════════════════════════════════════════════════
# DB 폴링 스레드 – 파이프라인 실행 중 탐지 결과 즉시 알림
# ══════════════════════════════════════════════════════════════════════════

_DB_POLL_INTERVAL = 2   # 폴링 간격 (초)

# DB 행 수 폴링 후 변화 시 SSE 알림
def _db_poll_worker():
    """탐지/이미지 DB 행 수를 주기적으로 확인해 변화 시 SSE 알림."""
    import sqlite3
    from src.config import SENSOR_DB_PATH, REPORTS_DB_PATH

    # DB 테이블별 행 수 조회
    def _row_counts():
        counts = {"images": 0, "detections": 0, "reports": 0}
        for db_path, queries in [
            (SENSOR_DB_PATH,  {"images": "SELECT COUNT(*) FROM image_records",
                               "detections": "SELECT COUNT(*) FROM detection_records"}),
            (REPORTS_DB_PATH, {"reports": "SELECT COUNT(*) FROM reports"}),
        ]:
            try:
                con = sqlite3.connect(db_path, timeout=2)
                for key, sql in queries.items():
                    try:
                        counts[key] = con.execute(sql).fetchone()[0]
                    except Exception:
                        pass
                con.close()
            except Exception:
                pass
        return counts

    prev = {"images": -1, "detections": -1, "reports": -1}
    _prev_stage_ts: str | None = None

    while True:
        _time.sleep(_DB_POLL_INTERVAL)
        try:
            cur = _row_counts()
            changed = [k for k in ("images", "detections", "reports") if cur[k] != prev[k]]
            if changed:
                # 파이프라인이 돌고 있을 때만 중간 알림 전송
                # (파이프라인 완료 후 _auto_sim_worker 가 이미 전체 알림을 보내므로
                #  running=False 일 때는 중복 전송하지 않음)
                if _auto_state.get("running"):
                    _notify_db_updated(
                        run_count=_auto_state.get("run_count", 0),
                        success=True,
                        elapsed=0.0,
                        changed=changed,
                    )
                prev = cur

            # 파이프라인 단계 상태 파일 폴링 → 변경 시 SSE 브로드캐스트
            try:
                from src.pipeline_status import read_status as _read_ps
                ps = _read_ps()
                cur_ts = ps.get("ts")
                if cur_ts != _prev_stage_ts:
                    _prev_stage_ts = cur_ts
                    _notify_pipeline_stage(ps)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("[DBPoll] 오류: %s", exc)


# 서버 시작 시 스레드 자동 실행
threading.Thread(target=_auto_sim_worker, daemon=True, name="AutoSimThread").start()
threading.Thread(target=_db_poll_worker,  daemon=True, name="DBPollThread").start()


# 이미지를 PNG base64 문자열로 변환
def _image_to_png_b64(image_path: Path, max_size: int = 512) -> str:
    """이미지 파일(TIFF 포함)을 PNG로 변환 후 base64 반환."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# bbox 좌표 공간 크기 계산 반환
def _det_space(file_w: int, file_h: int,
               det_width: int | None, det_height: int | None) -> tuple[int, int]:
    """bbox 좌표 기준 공간 크기 반환.
    ImageRecord.det_width/det_height 가 있으면 그대로 사용;
    없으면 (구버전 데이터) SR 설정값으로 역산.
    """
    if det_width and det_height:
        return det_width, det_height
    sr_scale = min(SR_TARGET_W / file_w, SR_TARGET_H / file_h)
    if sr_scale > 1.0:
        return int(file_w * sr_scale), int(file_h * sr_scale)
    return file_w, file_h


# 탐지 박스 오버레이 이미지를 base64로 반환
def _image_with_detections_b64(
    image_path: Path,
    detections: list,
    max_size: int = 480,
    det_width: int | None = None,
    det_height: int | None = None,
) -> str:
    """이미지에 탐지 결과 바운딩박스를 그린 뒤 base64 PNG 반환."""
    from PIL import Image, ImageDraw, ImageFont, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(image_path) as img:
        file_w, file_h = img.size
        det_w, det_h = _det_space(file_w, file_h, det_width, det_height)
        img.thumbnail((max_size, max_size))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        scale_x = img.width  / det_w
        scale_y = img.height / det_h
        draw = ImageDraw.Draw(img, "RGBA")
        for det in detections:
            x1 = det.bbox_x1 * scale_x
            y1 = det.bbox_y1 * scale_y
            x2 = det.bbox_x2 * scale_x
            y2 = det.bbox_y2 * scale_y
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 136, 220), width=2)
            label = f"{det.object_class} {det.confidence:.2f}"
            draw.rectangle([x1, y1 - 12, x1 + len(label) * 5.5, y1], fill=(0, 255, 136, 160))
            try:
                draw.text((x1 + 2, y1 - 12), label, fill=(0, 0, 0, 255))
            except Exception:
                pass
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════════════
# 위성 모의기
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/satellites")
# 현재 위성 위치 목록 반환
def api_satellites():
    """위성 모의기에서 계산된 현재 위성 위치 목록을 반환.

    캠페인/임무 실행 중인 위성은 궤도 좌표 대신 해당 임무 WP 좌표로 오버라이드:
      - running WP 가 있으면 → 그 WP 좌표
      - 캠페인이 running/complete 이고 완료 WP 가 있으면 → 마지막 완료 WP 좌표
    """
    sats = get_positions()

    camp_status = _campaign.get("status", "idle")
    with _sat_lock:
        for sat in sats:
            sm = _sat_missions.get(sat["id"])
            if not sm or not sm.get("waypoints"):
                continue
            wps = sm["waypoints"]

            # 실행 중인 WP
            running_wp = next((wp for wp in wps if wp["status"] == "running"), None)
            if running_wp:
                sat["lat"] = running_wp["lat"]
                sat["lon"] = running_wp["lon"]
                sat["at_mission"] = True
                continue

            # 캠페인이 진행/완료 중 → 마지막 처리된 WP 좌표 유지
            if camp_status in ("running", "complete"):
                done_wps = [wp for wp in wps if wp["status"] in ("complete", "failed")]
                if done_wps:
                    sat["lat"] = done_wps[-1]["lat"]
                    sat["lon"] = done_wps[-1]["lon"]
                    sat["at_mission"] = True

    return {"satellites": sats, "count": len(sats)}


@app.get("/api/simulator/status")
# 자동 시뮬레이션 상태 조회
def api_simulator_status():
    """자동 시뮬레이션 현재 상태 조회."""
    mission_info = None
    if _active_mission_id:
        m = _missions.get(_active_mission_id)
        if m:
            total     = len(m["waypoints"])
            completed = sum(1 for wp in m["waypoints"] if wp["status"] in ("complete", "ingested", "failed"))
            running_wp = next((wp for wp in m["waypoints"] if wp["status"] == "running"), None)
            mission_info = {
                "id":          m["id"],
                "name":        m["name"],
                "wp_total":    total,
                "wp_done":     completed,
                "current_wp":  running_wp["name"] if running_wp else None,
            }
    with _camp_lock:
        camp_info = {
            "status":       _campaign["status"],
            "total":        _campaign["total"],
            "done":         _campaign["done"],
            "failed_count": _campaign["failed_count"],
            "current":      _campaign["current"],
            "started_at":   _campaign["started_at"],
            "finished_at":  _campaign["finished_at"],
        }
    return {
        "auto_enabled":  _auto_state["enabled"],
        "running":       _auto_state["running"],
        "run_count":     _auto_state["run_count"],
        "last_run":      _auto_state["last_run"],
        "last_success":  _auto_state["last_success"],
        "last_elapsed":  _auto_state["last_elapsed"],
        "last_images":   _auto_state["last_images"],
        "last_active":   _auto_state["last_active"],
        "active_mission": mission_info,
        "campaign":      camp_info,
    }


@app.post("/api/simulator/auto/toggle")
# 자동 시뮬레이션 ON/OFF 토글
def api_simulator_auto_toggle():
    """자동 시뮬레이션 ON/OFF 토글."""
    _auto_state["enabled"] = not _auto_state["enabled"]
    logger.info("[AutoSim] 자동 실행 %s", "ON" if _auto_state["enabled"] else "OFF")
    return {"auto_enabled": _auto_state["enabled"]}


@app.post("/api/simulator/step")
# 위성 모의기 1스텝 실행
def api_simulator_step():
    """
    위성 모의기 1스텝 실행:
      1. 위성 궤도 계산 → 현재 위경도
      2. config.py IMAGE_MODE 에 따라 이미지 선택
         separate: sample/ 에서 서로 다른 이미지 2장
         crop:     sample/ 에서 1장 선택 후 크롭해 2장 생성
      3. metadata.json 갱신
      4. main.py 파이프라인 실행 (탐지 + 페어링 + 보고서 → DB 삽입)
    """
    try:
        from simulator_runner import run_step
        result = run_step()
    except Exception as exc:
        logger.error(f"[API] simulator step error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # 수동 실행 완료 → SSE 실시간 알림
    _notify_db_updated(
        run_count=_auto_state["run_count"],
        success=result["success"],
        elapsed=result["elapsed_s"],
    )

    return {
        "success":     result["success"],
        "skipped":     result.get("skipped", False),
        "skip_reason": result.get("skip_reason", ""),
        "elapsed_s":   result["elapsed_s"],
        "image_mode":  result.get("image_mode", body.image_mode),
        "active":      result["active"],
        "satellites":  result["satellites"],
        "images":      result["images"],
        "pipeline_ok": result["success"] and not result.get("skipped", False),
        "stderr_tail": result["stderr_tail"] if not result["success"] else "",
    }


# ══════════════════════════════════════════════════════════════════════════
# SAM3 탐지 결과
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/detections")
# 해당 지역 최신 탐지 결과 반환
def api_detections(
    lat:    float = Query(..., description="위도"),
    lon:    float = Query(..., description="경도"),
    radius: float = Query(default=0.1, description="검색 반경 (도)"),
):
    """DB에서 해당 지역 최신 SAM3 탐지 결과 반환."""
    records = get_latest_detections_near(lat, lon, radius_deg=radius)
    return {
        "detections": [
            {
                "id":            r.id,
                "object_class":  r.object_class,
                "confidence":    round(r.confidence, 3),
                "lat":           r.lat,
                "lon":           r.lon,
                "bbox":          {"x1": r.bbox_x1, "y1": r.bbox_y1,
                                  "x2": r.bbox_x2, "y2": r.bbox_y2},
                "source_type":   r.source_type,
                "detection_time": r.detection_time.isoformat() if r.detection_time else None,
                "image_id":      r.image_id,
            }
            for r in records
        ],
        "count": len(records),
    }


# ══════════════════════════════════════════════════════════════════════════
# 최신 위성영상
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/image/latest")
# 해당 지역 최신 위성영상 메타데이터 반환
def api_latest_image(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(default=0.1),
    embed:  bool  = Query(default=True, description="base64 이미지 데이터 포함 여부"),
):
    """해당 지역 최근 촬영 영상 메타데이터 + 선택적 base64 반환."""
    record = get_latest_image_near(lat, lon, radius_deg=radius)
    if record is None:
        raise HTTPException(status_code=404, detail="해당 지역 영상 없음.")

    image_path = Path(record.image_path)
    if not image_path.is_absolute():
        image_path = _images_dir / record.image_path

    image_b64 = None
    image_url  = None
    if image_path.exists():
        rel = image_path.relative_to(_images_dir) if image_path.is_relative_to(_images_dir) else image_path.name
        image_url = f"/static/images/{rel}"
        if embed:
            try:
                image_b64 = _image_to_png_b64(image_path)
            except Exception:
                image_b64 = base64.b64encode(image_path.read_bytes()).decode()

    return {
        "id":              record.id,
        "capture_time":    record.capture_time.isoformat() if record.capture_time else None,
        "source_type":     record.source_type,
        "sensor_platform": record.sensor_platform,
        "lat_center":      record.lat_center,
        "lon_center":      record.lon_center,
        "resolution_m":    record.resolution_m,
        "image_url":       image_url,
        "image_b64":       image_b64,
    }


# ══════════════════════════════════════════════════════════════════════════
# DB 전체 이미지/탐지 목록
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/images")
# 전체 위성영상 목록 및 탐지 건수 반환
def api_all_images(limit: int = Query(default=None)):
    """DB에 저장된 모든 위성영상 메타데이터 + 탐지 건수 목록 (capture_time DESC)."""
    rows = get_all_images_with_count(limit=limit)
    result = []
    for r, cnt in rows:
        result.append({
            "id":              r.id,
            "capture_time":    r.capture_time.isoformat() if r.capture_time else None,
            "source_type":     r.source_type,
            "sensor_platform": r.sensor_platform,
            "lat_center":      r.lat_center,
            "lon_center":      r.lon_center,
            "resolution_m":    r.resolution_m,
            "detection_count": cnt,
            "session_id":      r.session_id,
        })
    return {"images": result, "count": len(result)}


@app.get("/api/image/{image_id}/thumb")
# 이미지를 PNG base64 썸네일로 반환
def api_image_thumb(image_id: str):
    """특정 이미지를 PNG로 변환하여 base64 반환 (TIFF 포함)."""
    record = get_image_record_by_id(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="이미지 없음.")

    image_path = Path(record.image_path)
    if not image_path.is_absolute():
        image_path = _images_dir / record.image_path

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일 없음.")

    try:
        b64 = _image_to_png_b64(image_path, max_size=600)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"이미지 변환 실패: {exc}")

    return {
        "id":           record.id,
        "capture_time": record.capture_time.isoformat() if record.capture_time else None,
        "image_b64":    b64,
        "image_mime":   "image/png",
    }


@app.get("/api/detections/by-image/{image_id}")
# 특정 이미지의 탐지 결과 전체 반환 (합성 탐지 제외)
def api_detections_by_image(image_id: str):
    """특정 이미지에 속한 탐지 결과 전체 반환 (source_type='synthetic' 제외)."""
    records = [r for r in get_detections_by_image(image_id) if r.source_type != "synthetic"]
    return {
        "detections": [
            {
                "id":             r.id,
                "object_class":   r.object_class,
                "confidence":     round(r.confidence, 3),
                "lat":            r.lat,
                "lon":            r.lon,
                "bbox":           {"x1": r.bbox_x1, "y1": r.bbox_y1,
                                   "x2": r.bbox_x2, "y2": r.bbox_y2},
                "source_type":    r.source_type,
                "detection_time": r.detection_time.isoformat() if r.detection_time else None,
            }
            for r in records
        ],
        "count": len(records),
    }


@app.get("/api/images/by-session/{session_id}")
# 세션 ID로 이미지 목록 조회
def api_images_by_session(session_id: str):
    """동일 session_id를 가진 이미지 목록 반환 (탐지 에디터 세션 이동용)."""
    records = get_images_by_session(session_id)
    return {
        "images": [
            {
                "id":           r.id,
                "capture_time": r.capture_time.isoformat() if r.capture_time else None,
                "source_type":  r.source_type,
                "session_id":   r.session_id,
            }
            for r in records
        ],
        "count": len(records),
    }


# ══════════════════════════════════════════════════════════════════════════
# Pairing 수정 API
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/pairings/by-session/{session_id}")
# 세션 페어링 목록 반환
def api_pairings_by_session(session_id: str):
    """세션의 pairing 목록 반환 (pairing 에디터용)."""
    records = get_pairings_by_session(session_id)
    # 페어링 레코드를 딕셔너리로 변환
    def _fmt(r):
        return {
            "id":                    r.id,
            "status":                r.status,
            "lat_center":            r.lat_center,
            "lon_center":            r.lon_center,
            "current_detection_id":  r.current_detection_id,
            "current_object_class":  r.current_object_class,
            "current_confidence":    round(r.current_confidence, 3) if r.current_confidence is not None else None,
            "current_lat":           r.current_lat,
            "current_lon":           r.current_lon,
            "current_capture_time":  r.current_capture_time.isoformat() if r.current_capture_time else None,
            "current_bbox":          r.current_bbox,
            "past_detection_id":     r.past_detection_id,
            "past_object_class":     r.past_object_class,
            "past_confidence":       round(r.past_confidence, 3) if r.past_confidence is not None else None,
            "past_lat":              r.past_lat,
            "past_lon":              r.past_lon,
            "past_capture_time":     r.past_capture_time.isoformat() if r.past_capture_time else None,
            "past_bbox":             r.past_bbox,
            "session_id":            r.session_id,
        }
    return {"pairings": [_fmt(r) for r in records], "count": len(records)}


@app.put("/api/pairing/{pairing_id}")
# 페어링 단건 업데이트
async def api_update_pairing(pairing_id: str, request: Request):
    """pairing 단건 업데이트 (status / object_class / confidence 변경)."""
    body = await request.json()
    ok = update_pairing(pairing_id, body)
    if not ok:
        raise HTTPException(status_code=404, detail="pairing not found")
    return {"ok": True, "id": pairing_id}


@app.delete("/api/pairing/{pairing_id}")
# 페어링 단건 삭제
def api_delete_pairing(pairing_id: str):
    """pairing 단건 삭제."""
    ok = delete_pairing(pairing_id)
    if not ok:
        raise HTTPException(status_code=404, detail="pairing not found")
    return {"ok": True, "id": pairing_id}


# ══════════════════════════════════════════════════════════════════════════
# 판독 보고서
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/report/latest")
# 해당 지역 최신 판독 보고서 반환
def api_latest_report(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(default=2.0),
):
    """해당 지역 최신 판독 보고서 반환 (위치 매칭 → 전체 최신 순 폴백)."""
    session_ids = get_session_ids_near(lat, lon, radius_deg=radius)
    report = get_latest_report_for_sessions(session_ids)

    if report is None:
        all_r = get_all_reports(limit=1)
        report = all_r[0] if all_r else None

    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")

    return _report_dict(report)


@app.get("/api/report/{report_id}")
# 보고서 ID로 단건 조회
def api_report_by_id(report_id: str):
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")
    return _report_dict(report)


@app.get("/api/report/{report_id}/hwp")
# 보고서를 HWPX 파일로 다운로드
def api_report_hwp(report_id: str):
    """보고서를 HWPX 파일로 다운로드한다."""
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")
    text = report.report_content or ""
    hwpx_bytes = make_hwpx(text)
    ts = ""
    if report.report_time:
        try:
            ts = "_" + report.report_time.strftime("%Y%m%d_%H%M%S")
        except Exception:
            pass
    filename = f"MSIS_report{ts}.hwpx"
    return Response(
        content=hwpx_bytes,
        media_type="application/hwp+zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/report/{report_id}/images")
# 보고서 관련 현재·과거 위성사진 반환
def api_report_images(report_id: str):
    """보고서 생성에 활용된 이전/현재 위성사진 base64 반환.

    매핑 전략:
      현재 이미지 – ImageRecord.session_id == report.session_id (직접 매핑)
      과거 이미지 – PairingRecord.past_detection_id → DetectionRecord.image_id
                   (다른 세션에서 촬영된 과거 프레임)
      폴백        – session_id 없는 구 데이터는 pairing capture_time 경로 사용
    """
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")

    from datetime import datetime as _dt
    _MIN_DT = _dt.min

    # session_id 기반 pairing 조회 → 없으면 report_time 기반 폴백
    pairings = get_pairings_by_session(report.session_id) if report.session_id else []
    if not pairings and report.report_time:
        pairings = get_pairings_near_time(report.report_time)
    # 보고서 텍스트와 동일하게 가장 최근 pairing_time 배치만 사용
    if pairings:
        latest_pt = max(p.pairing_time for p in pairings)
        pairings = [p for p in pairings if p.pairing_time == latest_pt]

    # ── 현재 이미지: session_id 직접 매핑 ─────────────────────────────────────
    curr_imgs = get_images_by_session(report.session_id) if report.session_id else []
    if curr_imgs:
        # capture_time 기준 내림차순 → 가장 최신 프레임이 "현재"
        curr_imgs = sorted(curr_imgs, key=lambda x: x.capture_time or _MIN_DT)
        current_image_id      = curr_imgs[-1].id
        current_capture_time  = curr_imgs[-1].capture_time
    else:
        # ── 폴백: 구 데이터 – pairing current_detection_id 경로 ──────────────
        curr_seen: dict[str, object] = {}  # image_id → capture_time
        for p in pairings:
            if p.current_detection_id:
                det = get_detection_by_id(p.current_detection_id)
                if det and det.image_id and det.image_id not in curr_seen:
                    curr_seen[det.image_id] = p.current_capture_time
        if not curr_seen:
            for p in pairings:
                if p.current_capture_time:
                    for img in get_images_by_capture_time(p.current_capture_time):
                        if img.id not in curr_seen:
                            curr_seen[img.id] = p.current_capture_time
        if curr_seen:
            entries = sorted(curr_seen.items(), key=lambda x: x[1] or _MIN_DT)
            current_image_id, current_capture_time = entries[-1]
        else:
            current_image_id = current_capture_time = None

    # ── 과거 이미지: pairing past_detection_id → image_id (다른 세션) ─────────
    past_seen: dict[str, object] = {}  # image_id → capture_time
    for p in pairings:
        if p.past_detection_id:
            det = get_detection_by_id(p.past_detection_id)
            if det and det.image_id and det.image_id not in past_seen:
                past_seen[det.image_id] = p.past_capture_time
    # 폴백: capture_time 경로
    if not past_seen:
        for p in pairings:
            if p.past_capture_time:
                for img in get_images_by_capture_time(p.past_capture_time):
                    if img.id not in past_seen:
                        past_seen[img.id] = p.past_capture_time
    if past_seen:
        entries = sorted(past_seen.items(), key=lambda x: x[1] or _MIN_DT)
        past_image_id, past_capture_time = entries[-1]
    else:
        past_image_id = past_capture_time = None

    # ── _build_info: image_id → base64(탐지 박스 포함) ───────────────────────
    # 이미지 정보와 base64를 딕셔너리로 반환
    def _build_info(image_id: str, capture_time_fallback, with_detections: bool = False):
        rec = get_image_record_by_id(image_id)
        if rec is None:
            return None
        img_path = Path(rec.image_path)
        if not img_path.is_absolute():
            img_path = _images_dir / rec.image_path
        b64 = None
        if img_path.exists():
            try:
                if with_detections:
                    dets = get_detections_by_image(image_id)
                    b64 = _image_with_detections_b64(
                        img_path, dets, max_size=480,
                        det_width=rec.det_width, det_height=rec.det_height,
                    )
                else:
                    b64 = _image_to_png_b64(img_path, max_size=480)
            except Exception:
                b64 = None
        ct = rec.capture_time or capture_time_fallback
        return {
            "id":           rec.id,
            "capture_time": ct.isoformat() if ct else None,
            "image_b64":    b64,
            "session_id":   rec.session_id,
        }

    # 최후 수단: 동일 세션에 이미지가 2장 이상이면 가장 오래된 것을 과거 이미지로 사용
    if past_image_id is None and len(curr_imgs) >= 2:
        past_image_id    = curr_imgs[0].id
        past_capture_time = curr_imgs[0].capture_time

    curr_info = _build_info(current_image_id, current_capture_time, with_detections=True) \
                if current_image_id else None
    past_info = _build_info(past_image_id,    past_capture_time,    with_detections=True) \
                if past_image_id else None

    return {
        "current_images":   [curr_info] if curr_info else [],
        "past_images":      [past_info] if past_info else [],
        "current_image_id": current_image_id,
        "past_image_id":    past_image_id,
        "session_id":       report.session_id,
    }


@app.get("/api/report/{report_id}/pairings")
# 보고서 세션 페어링 목록 반환
def api_report_pairings(report_id: str):
    """보고서 세션의 페어링 목록 (bbox·클래스 포함) 반환 – 매칭 도시용.

    Returns:
        current_image_id: 현재 이미지 ID (raw 엔드포인트에서 원본 이미지 취득용)
        past_image_id:    과거 이미지 ID
        pairs:            페어링 목록 (status·label·bbox·class·conf)
    """
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")
    from datetime import datetime as _dt
    _MIN_DT = _dt.min

    pairings = get_pairings_by_session(report.session_id) if report.session_id else []
    if not pairings and report.report_time:
        pairings = get_pairings_near_time(report.report_time)
    if pairings:
        latest_pt = max(p.pairing_time for p in pairings)
        pairings = [p for p in pairings if p.pairing_time == latest_pt]

    if not pairings:
        return {"current_image_id": None, "past_image_id": None, "pairs": []}

    # ── 현재 이미지 ID (api_report_images 와 동일 로직) ─────────────────────
    curr_imgs = get_images_by_session(report.session_id) if report.session_id else []
    if curr_imgs:
        curr_imgs = sorted(curr_imgs, key=lambda x: x.capture_time or _MIN_DT)
        current_image_id = curr_imgs[-1].id
    else:
        current_image_id = None
        for p in pairings:
            if p.current_detection_id:
                det = get_detection_by_id(p.current_detection_id)
                if det and det.image_id:
                    current_image_id = det.image_id
                    break

    # ── 과거 이미지 ID ───────────────────────────────────────────────────────
    past_image_id = None
    for p in pairings:
        if p.past_detection_id:
            det = get_detection_by_id(p.past_detection_id)
            if det and det.image_id:
                past_image_id = det.image_id
                break

    # ── 페어링 목록 구성 ─────────────────────────────────────────────────────
    counters: dict[str, int] = {
        "matched": 0, "changed": 0, "moved": 0, "new": 0, "disappeared": 0,
        "past_not_included": 0, "current_not_included": 0,
    }
    pairs = []
    for p in pairings:
        st = p.status if p.status in counters else "new"
        counters[st] += 1
        cnt = counters[st]
        if st == "matched":
            label = str(cnt)
        elif st == "changed":
            label = f"C{cnt}"
        elif st == "moved":
            label = f"M{cnt}"
        elif st == "new":
            label = f"N{cnt}"
        elif st == "disappeared":
            label = f"D{cnt}"
        elif st == "past_not_included":
            label = f"PI{cnt}"
        elif st == "current_not_included":
            label = f"CI{cnt}"
        else:
            label = f"?{cnt}"

        pairs.append({
            "id":            p.id,
            "status":        st,
            "label":         label,
            "current_bbox":  p.current_bbox,
            "current_class": p.current_object_class,
            "current_conf":  round(p.current_confidence, 3) if p.current_confidence else None,
            "past_bbox":     p.past_bbox,
            "past_class":    p.past_object_class,
            "past_conf":     round(p.past_confidence, 3) if p.past_confidence else None,
        })

    return {
        "current_image_id": current_image_id,
        "past_image_id":    past_image_id,
        "pairs":            pairs,
    }


# ── 사용자 수정용 엔드포인트 ──────────────────────────────────────────────────

class BboxIn(BaseModel):
    x1: float; y1: float; x2: float; y2: float

class DetectionIn(BaseModel):
    object_class: str
    confidence: float
    bbox: BboxIn
    lat: float = 0.0
    lon: float = 0.0

class DetectionsUpdateBody(BaseModel):
    detections: list[DetectionIn]

class ReportContentBody(BaseModel):
    report_content: str


@app.get("/api/image/{image_id}/raw")
# 원본 이미지 및 출력 크기 반환
def api_image_raw(image_id: str, max_size: int = Query(default=1024, le=2048)):
    """원본 이미지(탐지 박스 없음) + 실제 출력 크기 반환."""
    from PIL import Image as PilImage
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")
    img_path = Path(rec.image_path)
    if not img_path.is_absolute():
        img_path = _images_dir / rec.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일 없음")
    from PIL import ImageFile as _PilIF
    _PilIF.LOAD_TRUNCATED_IMAGES = True
    with PilImage.open(img_path) as img:
        file_w, file_h = img.size
        # orig_width/orig_height = bbox 좌표 기준 공간.
        # ImageRecord.det_width/det_height 가 있으면 직접 사용 (정확),
        # 없으면 SR 설정값으로 역산 (구버전 데이터 폴백).
        orig_w, orig_h = _det_space(file_w, file_h, rec.det_width, rec.det_height)
        img.thumbnail((max_size, max_size))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        out_w, out_h = img.size
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "id":           image_id,
        "image_b64":    b64,
        "width":        out_w,
        "height":       out_h,
        "orig_width":   orig_w,
        "orig_height":  orig_h,
        "capture_time": rec.capture_time.isoformat() if rec.capture_time else None,
        "session_id":   rec.session_id,
        "source_type":  rec.source_type,
    }


@app.get("/api/image/{image_id}/rendered")
# 탐지 결과 오버레이 이미지 반환
def api_image_rendered(image_id: str, t: int = Query(default=0)):
    """현재 DB에 저장된 탐지 결과를 이미지에 그려서 반환 (캐시버스팅용 t 파라미터 지원)."""
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")
    img_path = Path(rec.image_path)
    if not img_path.is_absolute():
        img_path = _images_dir / rec.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일 없음")
    dets = [d for d in get_detections_by_image(image_id) if d.source_type != "synthetic"]
    b64 = _image_with_detections_b64(
        img_path, dets, max_size=600,
        det_width=rec.det_width, det_height=rec.det_height,
    )
    return {
        "id":           image_id,
        "image_b64":    b64,
        "capture_time": rec.capture_time.isoformat() if rec.capture_time else None,
    }


@app.put("/api/image/{image_id}/detections")
# 이미지 탐지 결과를 수정본으로 교체
def api_update_detections(image_id: str, body: DetectionsUpdateBody):
    """이미지의 탐지 결과를 사용자 수정본으로 교체."""
    from src.database.models import DetectionRecord as DetRec
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")
    new_dets = []
    for d in body.detections:
        new_dets.append(DetRec(
            image_id=image_id,
            detection_time=rec.capture_time,
            object_class=d.object_class,
            confidence=max(0.0, min(1.0, d.confidence)),
            bbox_x1=d.bbox.x1,
            bbox_y1=d.bbox.y1,
            bbox_x2=d.bbox.x2,
            bbox_y2=d.bbox.y2,
            lat=d.lat if d.lat != 0.0 else rec.lat_center,
            lon=d.lon if d.lon != 0.0 else rec.lon_center,
            source_type="human_edit",
            session_id=rec.session_id,   # 원본 이미지의 세션 ID 유지
        ))
    # 교체 전 구 detection_id 수집 → 교체 후 pairing 참조를 새 ID로 업데이트
    old_det_ids = {d.id for d in get_detections_by_image(image_id)}
    count = replace_detections_for_image(image_id, new_dets)
    new_det_list = get_detections_by_image(image_id)
    first_new_id = new_det_list[0].id if new_det_list else None
    update_pairings_detection_refs(old_det_ids, first_new_id)
    _notify_db_updated(changed=["detections", "images"])
    return {"updated": count, "image_id": image_id}


class _RegenBody(BaseModel):
    target_description: str = ""


@app.post("/api/image/{image_id}/regenerate-report")
# 수정된 탐지 기반 보고서 재생성
def api_regenerate_report(
    image_id: str,
    background_tasks: BackgroundTasks,
    body: _RegenBody = Body(default=_RegenBody()),
):
    """
    수정된 탐지 결과를 바탕으로 Temporal Pairing + 보고서 재생성.

    - SensorDB의 현재 탐지 결과(사용자 수정 포함)를 읽어 pairing 재실행
    - 동일 session_id의 기존 pairing_records / report_records를 교체
    - 새 보고서 내용과 report_id 반환
    - target_description: 사용자가 입력한 표적 설명 (선택, 보고서 생성에 반영)
    """
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")
    if not rec.session_id:
        raise HTTPException(status_code=400, detail="이미지에 session_id 없음")

    try:
        from pipeline import MavenPipeline
        pipeline = MavenPipeline()
        report_text = pipeline.rerun_from_detections(
            image_id, target_description=body.target_description
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"[API] regenerate-report error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # 새로 삽입된 보고서 레코드 조회
    from src.database.reports_db import get_reports_by_session
    reports = get_reports_by_session(rec.session_id)
    new_report = reports[0] if reports else None

    _notify_db_updated(run_count=_auto_state["run_count"], success=True, elapsed=0.0)

    return {
        "success":    True,
        "image_id":   image_id,
        "session_id": rec.session_id,
        "report_id":  new_report.id if new_report else None,
        "report_content": report_text,
    }


@app.post("/api/image/{image_id}/detect-sam3")
# SAM3 탐지 실행 및 결과 DB 저장
def api_detect_sam3(image_id: str):
    """
    단계별 워크플로우 Phase 2a (SAM3 경로):
    Sensor DB에 저장된 이미지에 SAM3 탐지를 실행하고 결과를 DB에 저장한다.
    """
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")

    try:
        from pipeline import MavenPipeline
        pipeline = MavenPipeline()
        orm_dets = pipeline.detect_sam3_for_image(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("[API] detect-sam3 error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    _notify_db_updated(changed=["detections", "images"])

    return {
        "success":         True,
        "image_id":        image_id,
        "session_id":      rec.session_id,
        "detection_count": len(orm_dets),
        "detections": [
            {
                "id":           d.id,
                "object_class": d.object_class,
                "confidence":   round(d.confidence, 3),
                "bbox":         {
                    "x1": d.bbox_x1, "y1": d.bbox_y1,
                    "x2": d.bbox_x2, "y2": d.bbox_y2,
                },
                "lat": d.lat,
                "lon": d.lon,
            }
            for d in orm_dets
        ],
    }


class _AnalyzeBody(BaseModel):
    target_description: str = ""


@app.post("/api/session/{session_id}/analyze")
# 세션 페어링 및 보고서 생성 — 즉시 202 반환 후 백그라운드 실행
def api_session_analyze(
    session_id: str,
    body: _AnalyzeBody = Body(default=_AnalyzeBody()),
):
    """
    단계별 워크플로우 Phase 2b:
    탐지 결과(SAM3 또는 수동)가 저장된 세션에 대해
    Temporal Pairing → Graph RAG 인덱싱 → 판독 보고서 생성을 백그라운드에서 실행한다.
    즉시 202 를 반환하고, 완료/오류 시 SSE 이벤트(analyze_complete/analyze_error)로 알린다.
    """
    with _analyze_jobs_lock:
        if _analyze_jobs.get(session_id) == "processing":
            raise HTTPException(status_code=409, detail="분석이 이미 진행 중입니다.")
        _analyze_jobs[session_id] = "processing"

    t = threading.Thread(
        target=_run_analyze_background,
        args=(session_id, body.target_description),
        daemon=True,
    )
    t.start()

    return JSONResponse(
        status_code=202,
        content={"status": "processing", "session_id": session_id},
    )


@app.delete("/api/detection/{detection_id}")
# 탐지 결과 단건 삭제 및 참조 정리
def api_delete_detection(detection_id: str):
    """탐지 결과 단건 삭제 → SensorDB에서 제거하고 pairing 참조도 정리."""
    det = get_detection_by_id(detection_id)
    if det is None:
        raise HTTPException(status_code=404, detail="탐지 결과 없음")
    delete_detection_by_id(detection_id)
    # pairing이 이 detection을 참조하고 있으면 null로 초기화
    update_pairings_detection_refs({detection_id}, None)
    _notify_db_updated(changed=["detections", "images"])
    return {"deleted": True, "detection_id": detection_id}


@app.patch("/api/report/{report_id}")
# 보고서 텍스트 수정
def api_update_report(report_id: str, body: ReportContentBody):
    """보고서 텍스트를 수정한다."""
    ok = update_report_content(report_id, body.report_content)
    if not ok:
        raise HTTPException(status_code=404, detail="보고서 없음")
    _notify_db_updated(changed=["reports"])
    return {"updated": True, "report_id": report_id}


@app.get("/api/reports")
# 전체 보고서 목록 반환
def api_all_reports(limit: int = Query(default=None)):
    reports = get_all_reports(limit=limit)
    items = []
    for r in reports:
        lat, lon = (None, None)
        if r.session_id:
            lat, lon = get_session_location(r.session_id)
        # session_id 없거나 pairing에 session_id 미설정 구 데이터 → 시간 기반 폴백
        if (lat is None or lon is None) and r.report_time:
            fb = get_pairings_near_time(r.report_time)
            if fb:
                lat = fb[0].lat_center
                lon = fb[0].lon_center
        country = _get_country_name(lat, lon) if lat is not None and lon is not None else None
        items.append({
            "id":           r.id,
            "report_time":  r.report_time.isoformat() if r.report_time else None,
            "saved_time":   r.saved_time.isoformat()  if r.saved_time  else None,
            "llm_model":    r.llm_model,
            "pairing_count":r.pairing_count,
            "session_id":   r.session_id,
            "lat_center":   lat,
            "lon_center":   lon,
            "country_name": country,
        })
    return {"reports": items, "count": len(items)}


# ══════════════════════════════════════════════════════════════════════════
# GraphRAG 지식 그래프 API
# ══════════════════════════════════════════════════════════════════════════

# GraphIndexer 싱글톤 인스턴스 반환
def _get_graph_indexer():
    """Return a module-level singleton GraphIndexer (lazy-init)."""
    from src.graph.graph_indexer import GraphIndexer
    if not hasattr(_get_graph_indexer, "_instance"):
        _get_graph_indexer._instance = GraphIndexer()
    return _get_graph_indexer._instance


@app.get("/api/graph/stats")
# 지식 그래프 통계 반환
def api_graph_stats():
    """지식 그래프 통계 (엔티티 수, 관계 수, 커뮤니티 수)."""
    try:
        stats = _get_graph_indexer().stats()
        return {"ok": True, **stats}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/graph/entities")
# 해당 지역 그래프 자산 엔티티 목록 반환
def api_graph_entities(
    lat:    float = Query(..., description="검색 중심 위도"),
    lon:    float = Query(..., description="검색 중심 경도"),
    radius: float = Query(default=0.1, description="검색 반경 (도)"),
):
    """
    해당 지역의 지식 그래프 자산(asset) 엔티티 목록 반환.
    각 엔티티는 과거 탐지 통계(new / persisted / moved / disappeared)를 포함한다.
    """
    try:
        indexer = _get_graph_indexer()
        assets = indexer.store.get_assets_near(lat, lon, radius)
        items = []
        for rec in assets:
            props = rec.properties or {}
            obs = rec.observation_count or 0
            avg_conf = (
                props.get("total_confidence", 0.0) / obs
                if obs > 0 else 0.0
            )
            items.append({
                "id":               rec.id,
                "object_class":     props.get("object_class", rec.name),
                "location_key":     props.get("location_key", ""),
                "lat":              props.get("lat"),
                "lon":              props.get("lon"),
                "new_count":        props.get("new_count", 0),
                "disappeared_count": props.get("disappeared_count", 0),
                "matched_count":    props.get("matched_count", 0),
                "moved_count":      props.get("moved_count", 0),
                "observation_count": obs,
                "avg_confidence":   round(avg_conf, 3),
                "first_seen":       rec.first_seen.isoformat() if rec.first_seen else None,
                "last_seen":        rec.last_seen.isoformat()  if rec.last_seen  else None,
                "sessions":         props.get("sessions", []),
            })
        items.sort(key=lambda x: -x["observation_count"])
        return {"entities": items, "count": len(items)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/graph/communities")
# 모든 그래프 커뮤니티 요약 반환
def api_graph_communities():
    """모든 커뮤니티(공출현 군집) 요약 반환."""
    try:
        communities = _get_graph_indexer().store.get_all_communities()
        items = []
        for c in communities:
            items.append({
                "id":             c.id,
                "community_index": c.community_index,
                "label":          c.label,
                "member_count":   len(c.member_ids or []),
                "member_summary": c.member_summary,
                "summary":        c.summary,
                "created_at":     c.created_at.isoformat() if c.created_at else None,
            })
        return {"communities": items, "count": len(items)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/graph/local-search")
# 지역 기반 그래프 로컬 검색 수행
def api_graph_local_search(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(default=0.05),
):
    """
    지역 기반 그래프 로컬 검색 – 해당 지역의 역사적 자산 활동 + 관련 커뮤니티 반환.
    보고서 생성 시 LLM 프롬프트에 주입되는 컨텍스트와 동일한 형태.
    """
    try:
        indexer = _get_graph_indexer()
        local = indexer.retriever.local_search(lat, lon, radius)
        global_comms = indexer.retriever.global_search(lat, lon, radius)
        return {
            "location": local["location"],
            "assets": local["assets"],
            "total_observations": local["total_observations"],
            "communities": global_comms,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/graph/reindex")
# 그래프 커뮤니티 감지 강제 재실행
def api_graph_reindex():
    """커뮤니티 감지를 강제로 재실행하고 결과 저장."""
    try:
        n = _get_graph_indexer().reindex_communities()
        return {"ok": True, "communities_detected": n}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════
# SSE 엔드포인트
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/events")
# SSE 스트림으로 실시간 이벤트 전송
async def api_events():
    """
    Server-Sent Events 스트림.
    DB(이미지/탐지/보고서)가 갱신될 때 'db_updated' 이벤트를 즉시 전송한다.
    25초마다 heartbeat 코멘트를 전송해 연결을 유지한다.
    """
    q: queue.Queue = queue.Queue()
    with _sse_lock:
        _sse_clients.append(q)

    # SSE 이벤트 비동기 생성기
    async def event_gen():
        try:
            yield "data: {\"type\":\"connected\"}\n\n"
            last_hb = asyncio.get_event_loop().time()
            while True:
                try:
                    msg = q.get_nowait()
                    yield f"data: {msg}\n\n"
                    last_hb = asyncio.get_event_loop().time()
                except queue.Empty:
                    await asyncio.sleep(0.3)
                    now = asyncio.get_event_loop().time()
                    if now - last_hb > 25:
                        yield ": heartbeat\n\n"
                        last_hb = now
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            with _sse_lock:
                try:
                    _sse_clients.remove(q)
                except ValueError:
                    pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# 임무계획 API
# ══════════════════════════════════════════════════════════════════════════

class _WaypointIn(BaseModel):
    name:               str = ""
    lat:                float
    lon:                float
    target_description: str = ""


class _MissionIn(BaseModel):
    name:      str
    waypoints: List[_WaypointIn]


@app.post("/api/missions")
# 웨이포인트 포함 임무 생성
def api_create_mission(body: _MissionIn):
    """임무 생성 (웨이포인트 목록 포함)."""
    mid = str(_uuid.uuid4())
    mission = {
        "id":          mid,
        "name":        body.name,
        "created_at":  datetime.utcnow().isoformat(),
        "status":      "pending",
        "started_at":  None,
        "finished_at": None,
        "waypoints": [
            {
                "id":                 str(_uuid.uuid4()),
                "seq":                i,
                "name":               wp.name or f"WP{i+1}",
                "lat":                wp.lat,
                "lon":                wp.lon,
                "target_description": wp.target_description,
                "status":             "pending",
                "report_id":          None,
                "elapsed_s":          None,
                "error":              None,
            }
            for i, wp in enumerate(body.waypoints)
        ],
    }
    with _mission_lock:
        _missions[mid] = mission
    _save_missions_to_file()
    return mission


@app.get("/api/missions")
# 임무 목록 최신 순 반환
def api_list_missions():
    """임무 목록 반환 (최신 순)."""
    with _mission_lock:
        lst = list(_missions.values())
    lst.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return lst


@app.get("/api/mission/{mission_id}")
# 임무 단건 상태 조회
def api_get_mission(mission_id: str):
    """임무 단건 조회 (상태 폴링용)."""
    with _mission_lock:
        m = _missions.get(mission_id)
    if not m:
        raise HTTPException(status_code=404, detail="임무 없음")
    return m


@app.post("/api/mission/{mission_id}/execute")
# 임무를 자동 시뮬레이터에 등록 실행
def api_execute_mission(mission_id: str):
    """임무를 자동 시뮬레이터에 등록하여 실행. 웨이포인트를 순서대로 처리."""
    global _active_mission_id
    with _mission_lock:
        m = _missions.get(mission_id)
    if not m:
        raise HTTPException(status_code=404, detail="임무 없음")
    if _active_mission_id:
        raise HTTPException(status_code=409,
                            detail=f"다른 임무 실행 중 ({_active_mission_id[:8]}). 완료 후 시도하세요.")
    # 상태 초기화
    m["status"]      = "running"
    m["started_at"]  = datetime.utcnow().isoformat()
    m["finished_at"] = None
    for wp in m["waypoints"]:
        wp["status"]    = "pending"
        wp["report_id"] = None
        wp["error"]     = None
        wp["elapsed_s"] = None
    _save_missions_to_file()
    _active_mission_id = mission_id
    logger.info("[Mission] 임무 등록: %s (%d WP)", m["name"], len(m["waypoints"]))
    return {"status": "started", "mission_id": mission_id}


@app.delete("/api/mission/{mission_id}")
# 임무 삭제
def api_delete_mission(mission_id: str):
    """임무 삭제."""
    global _active_mission_id
    with _mission_lock:
        if mission_id not in _missions:
            raise HTTPException(status_code=404, detail="임무 없음")
        del _missions[mission_id]
    if _active_mission_id == mission_id:
        _active_mission_id = None
    _save_missions_to_file()
    return {"status": "deleted"}


# ══════════════════════════════════════════════════════════════════════════
# 위성별 임무계획 API
# ══════════════════════════════════════════════════════════════════════════

class _SatWaypointIn(BaseModel):
    name:               str = ""
    lat:                float
    lon:                float
    target_description: str = ""


class _SatMissionIn(BaseModel):
    waypoints: List[_SatWaypointIn]


@app.get("/api/sat-missions")
# 위성별 임무계획 전체 반환
def api_get_sat_missions():
    """위성별 임무계획 전체 반환."""
    from src.satellite.simulator import SATELLITES as _SATS
    result = {}
    with _sat_lock:
        for cfg in _SATS:
            sid = cfg["id"]
            sm  = _sat_missions.get(sid)
            result[sid] = {
                "sat_id":      sid,
                "sat_name":    cfg["name"],
                "waypoints":   sm["waypoints"]   if sm else [],
                "assigned_at": sm.get("assigned_at") if sm else None,
            }
    return result


@app.put("/api/sat-mission/{sat_id}")
# 위성에 웨이포인트 목록 할당
def api_set_sat_mission(sat_id: str, body: _SatMissionIn):
    """위성에 웨이포인트 목록 할당 (기존 임무 덮어쓰기)."""
    from src.satellite.simulator import SATELLITES as _SATS
    sat_cfg = next((s for s in _SATS if s["id"] == sat_id), None)
    if not sat_cfg:
        raise HTTPException(status_code=404, detail="위성 없음")

    waypoints = [
        {
            "id":                 str(_uuid.uuid4()),
            "seq":                i,
            "name":               wp.name or f"WP{i+1}",
            "lat":                wp.lat,
            "lon":                wp.lon,
            "target_description": wp.target_description,
            "status":             "pending",
            "report_id":          None,
            "elapsed_s":          None,
            "error":              None,
        }
        for i, wp in enumerate(body.waypoints)
    ]

    with _sat_lock:
        _sat_missions[sat_id] = {
            "sat_id":      sat_id,
            "sat_name":    sat_cfg["name"],
            "assigned_at": datetime.utcnow().isoformat(),
            "waypoints":   waypoints,
        }
    _save_sat_missions()
    return _sat_missions[sat_id]


@app.delete("/api/sat-mission/{sat_id}")
# 위성 임무계획 초기화
def api_clear_sat_mission(sat_id: str):
    """위성 임무계획 초기화."""
    with _sat_lock:
        _sat_missions.pop(sat_id, None)
    _save_sat_missions()
    return {"status": "cleared"}


@app.post("/api/campaign/start")
# 모든 위성 임무계획 일괄 캠페인 시작
def api_campaign_start():
    """모든 위성 임무계획 일괄 실행 시작."""
    from src.satellite.simulator import SATELLITES as _SATS
    with _camp_lock:
        if _campaign["status"] == "running":
            raise HTTPException(status_code=409, detail="캠페인 이미 실행 중입니다.")

    queue: list = []
    with _sat_lock:
        for cfg in _SATS:
            sid = cfg["id"]
            sm  = _sat_missions.get(sid)
            if not sm or not sm.get("waypoints"):
                continue
            # 웨이포인트 상태 초기화
            for wp in sm["waypoints"]:
                wp["status"]    = "pending"
                wp["report_id"] = None
                wp["elapsed_s"] = None
                wp["error"]     = None
            # 큐에 추가 (위성별 WP 순서대로)
            for wp in sm["waypoints"]:
                queue.append({
                    "sat_id":   sid,
                    "sat_name": cfg["name"],
                    "wp_id":    wp["id"],
                    "wp_name":  wp["name"],
                })

    if not queue:
        raise HTTPException(
            status_code=400,
            detail="설정된 임무가 없습니다. 위성별 웨이포인트를 먼저 설정하세요.",
        )

    _save_sat_missions()

    with _camp_lock:
        _campaign["status"]       = "running"
        _campaign["started_at"]   = datetime.utcnow().isoformat()
        _campaign["finished_at"]  = None
        _campaign["queue"]        = queue
        _campaign["current"]      = None
        _campaign["total"]        = len(queue)
        _campaign["done"]         = 0
        _campaign["failed_count"] = 0

    logger.info("[Campaign] 시작: %d WP", len(queue))
    return {"status": "started", "total": len(queue)}


@app.delete("/api/campaign")
# 캠페인 취소 및 상태 초기화
def api_campaign_cancel():
    """캠페인 취소 / 상태 초기화."""
    with _camp_lock:
        _campaign["status"]       = "idle"
        _campaign["queue"]        = []
        _campaign["current"]      = None
        _campaign["total"]        = 0
        _campaign["done"]         = 0
        _campaign["failed_count"] = 0
        _campaign["finished_at"]  = None
    # 실행 중인 WP의 status는 failed로 유지 (이미 _auto_sim_worker 가 처리 중)
    return {"status": "cancelled"}


@app.get("/api/campaign/status")
# 캠페인 실행 상태 조회
def api_campaign_status():
    """캠페인 실행 상태 조회."""
    with _camp_lock:
        return dict(_campaign)


@app.get("/api/pipeline/status")
# 파이프라인 현재 단계 상태 조회
def api_pipeline_status():
    """현재 파이프라인 단계 상태 조회 (대시보드 초기 로드용)."""
    from src.pipeline_status import read_status as _read_ps
    return _read_ps()


@app.post("/api/system/reset")
# 시스템 전체 상태 초기화
def api_system_reset():
    """
    시스템 전체 초기화:
      - pipeline_status.json → idle
      - sat_missions.json    → {}  (메모리 포함)
      - 캠페인 상태          → idle
      - 파이프라인 실행 플래그 해제
    """
    global _active_mission_id

    # 1. pipeline_status 초기화
    try:
        from src.pipeline_status import clear_status as _ps_clear
        _ps_clear()
    except Exception as e:
        logger.warning("[Reset] pipeline_status 초기화 실패: %s", e)

    # 2. sat_missions 초기화 (파일 + 메모리)
    with _sat_lock:
        _sat_missions.clear()
    try:
        _SAT_MISSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SAT_MISSIONS_FILE.write_text("{}", encoding="utf-8")
    except Exception as e:
        logger.warning("[Reset] sat_missions 파일 초기화 실패: %s", e)

    # 3. 캠페인 상태 초기화
    with _camp_lock:
        _campaign.update({
            "status":       "idle",
            "started_at":   None,
            "finished_at":  None,
            "queue":        [],
            "current":      None,
            "total":        0,
            "done":         0,
            "failed_count": 0,
        })

    # 4. 파이프라인 실행 중 플래그 해제 + 활성 임무 해제
    with _auto_lock:
        _auto_state["running"] = False
    _active_mission_id = None

    # 5. SSE 로 pipeline_stage idle 브로드캐스트 (지도 마커 즉시 제거)
    try:
        from src.pipeline_status import read_status as _read_ps
        _notify_pipeline_stage(_read_ps())
    except Exception:
        pass

    logger.info("[Reset] 시스템 초기화 완료")
    return {"ok": True, "message": "시스템이 초기화되었습니다."}


# 보고서 레코드를 딕셔너리로 변환
def _report_dict(r) -> dict:
    return {
        "id":             r.id,
        "report_time":    r.report_time.isoformat() if r.report_time else None,
        "saved_time":     r.saved_time.isoformat()  if r.saved_time  else None,
        "llm_model":      r.llm_model,
        "llm_backend":    r.llm_backend,
        "pairing_count":  r.pairing_count,
        "session_id":     r.session_id,
        "file_path":      r.file_path,
        "report_content": r.report_content,
    }


# ══════════════════════════════════════════════════════════════════════════
# 대시보드 HTML
# ══════════════════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
# 대시보드 HTML 반환
def dashboard():
    if not _dashboard_path.exists():
        raise HTTPException(status_code=404, detail="dashboard/index.html 없음.")
    return HTMLResponse(content=_dashboard_path.read_text(encoding="utf-8"))


@app.get("/")
# API 루트 상태 메시지 반환
def root():
    return {"message": "MSIS API 실행 중. /dashboard 에서 지도 UI 확인."}
