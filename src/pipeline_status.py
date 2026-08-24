"""
파이프라인 단계 상태를 파일로 기록/조회하는 유틸리티.

pipeline.py (서브프로세스)가 data/pipeline_status.json 에 현재 단계를 기록하고,
web_api.py (부모 프로세스)가 해당 파일을 폴링하여 SSE 로 대시보드에 중계한다.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STATUS_FILE = Path(__file__).resolve().parent.parent / "data" / "pipeline_status.json"

# 단계 상수
STAGE_SR         = "sr"
STAGE_DETECTION  = "detection"
STAGE_PAIRING    = "pairing"
STAGE_REPORTING  = "reporting"
STAGE_IDLE       = "idle"

STAGE_LABELS = {
    STAGE_SR:        "판독보고서 생성중(Super resolution 진행중)",
    STAGE_DETECTION: "판독보고서 생성중(객체탐지 진행중)",
    STAGE_PAIRING:   "판독보고서 생성중(객체 Pairing 진행중)",
    STAGE_REPORTING: "판독보고서 생성중(LLM 판독보고서 생성 진행중)",
    STAGE_IDLE:      "",
}


# 현재 파이프라인 단계를 JSON 파일에 기록
def write_status(stage: str, lat: float = None, lon: float = None,
                 session_id: str = None) -> None:
    """현재 파이프라인 단계를 파일에 기록한다."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(
            json.dumps({
                "stage":      stage,
                "label":      STAGE_LABELS.get(stage, stage),
                "lat":        lat,
                "lon":        lon,
                "session_id": session_id,
                "ts":         datetime.utcnow().isoformat(),
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("[PipelineStatus] 상태 기록 실패: %s", exc)


# 파이프라인 상태를 idle로 초기화
def clear_status() -> None:
    """파이프라인 완료 후 idle 상태로 초기화한다."""
    write_status(STAGE_IDLE)


# 파일에서 마지막으로 기록된 파이프라인 상태 반환
def read_status() -> dict:
    """마지막으로 기록된 파이프라인 상태를 반환한다."""
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"stage": STAGE_IDLE, "label": "", "lat": None, "lon": None, "ts": None}
