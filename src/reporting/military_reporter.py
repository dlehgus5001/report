"""
Military Change Detection Report Generator using EXAONE4-32b.

Input:  Latest pairing records from the Pairing DB.
Output: Structured military intelligence report (Korean/English) covering:
         - Observation summary
         - Change analysis (new assets and disappeared assets only)
         - Threat assessment
         - Recommended actions
         NOTE: Stationary and repositioned/moved objects are excluded from the report.

LLM backend can be switched via config:
  LLM_BACKEND=vllm    → loads EXAONE4-32b via vLLM (default)
  LLM_BACKEND=ollama  → calls local Ollama server (model: exaone4:32b)
"""

import json
import logging
import textwrap
from datetime import datetime, timezone
from typing import List, Optional

from src.config import (
    DOCTRINE_ENABLED,
    DOCTRINE_MAX_CHARS,
    DOCTRINE_TOP_K,
    LLM_BACKEND,
    LLM_GPU_MEMORY_UTILIZATION,
    LLM_MAX_MODEL_LEN,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_TENSOR_PARALLEL_SIZE,
    LLM_TEMPERATURE,
    LLM_TRANSLATE_MAX_TOKENS,
    LLM_TRANSLATE_TO_KOREAN,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.database.models import PairingRecord
from src.database.reports_db import insert_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# LLM 시스템 프롬프트 생성 (교리 컨텍스트 포함 가능)
def _build_system_prompt(doctrine_context: str = "") -> str:
    base = (
        "당신은 군사 IMINT(영상정보) 분석관입니다. "
        "AI 기반 위성/드론 객체 탐지 데이터를 바탕으로 간결하고 공식적인 정보 보고서를 작성하세요. "
        "반드시 한국어로 작성하세요. "
        "표준 섹션 헤더를 사용하고 사실에 근거하여 작성하세요. "
        "활동 지표인 신규 출현 객체와 소실 객체를 중심으로 분석하되, "
        "정지·이동 객체의 종류(CLASS_SUMMARY)는 해당 지역 전력 구성 파악 및 위협 평가에 활용하세요. "
        "각 객체 클래스(TANK, APC, HELICOPTER, artillery, civilian building 등)의 군사적 의미를 반영하세요. "
        "정지 또는 위치 이동 객체를 변화 분석에서 개별 나열하지는 마세요. "
        "중요: 'CHANGED(시설변화)'는 같은 위경도에 있는 고정 시설물(건물·레이더 등)이 "
        "두 촬영 시점 간에 CLIP 영상 유사도 기준으로 구조적 변화가 감지된 객체를 의미합니다. "
        "시설 신축·증축·철거·피해 등 물리적 변화 가능성을 분석에 반영하세요. "
        "중요: 'DISAPPEARED(소실)'은 과거 영상에서 탐지되었으나 현재(최신) 영상에서 탐지되지 않은 객체를 의미합니다. "
        "객체가 완전히 소멸되거나 파괴되었음을 의미하지 않습니다. "
        "센서 시야 밖으로 이동했거나, 은폐되거나, 위치를 옮겼을 수 있습니다. "
        "소실 표현 시 반드시 '현재 영상에서 더 이상 관측되지 않음'으로 표현하세요. "
        "중요: 'PAST_NOT_INCLUDED(과거 미포함)'은 현재 탐지 객체가 과거 영상의 촬영 범위 밖에 위치함을 의미합니다. "
        "과거 센서가 해당 지역을 촬영하지 않아 객체의 이전 존재 여부를 확인할 수 없습니다. "
        "이러한 객체는 확인된 신규 활동이 아닌 추가 수집이 필요한 미검증 관측으로 취급하세요. "
        "중요: 'CURRENT_NOT_INCLUDED(현재 미포함)'은 과거 탐지 객체가 현재 영상의 촬영 범위 밖에 위치함을 의미합니다. "
        "현재 센서가 해당 지역을 촬영하지 않아 객체의 현재 존재 여부를 확인할 수 없습니다. "
        "이러한 객체는 확인된 소실이 아닌 촬영 공백으로 취급하세요."
    )
    if doctrine_context:
        base += (
            "\n\n다음은 참고용으로 제공된 군사 교리 문서 발췌문입니다. "
            "위협 평가, 교전 규칙, 권고 조치 수립 시 관련 내용을 참고하세요. "
            "교리 내용을 직접 인용하지 말고 분석에 자연스럽게 통합하세요.\n\n"
            + doctrine_context
        )
    return base


_MAX_DETAIL = 20   # max individual records shown per change category

# 차량류 클래스 – 보고서에 개별 항목 대신 건수(CLASS_SUMMARY)만 표시
_VEHICLE_CLASSES: frozenset = frozenset({
    "military tank",
    "armored personnel carrier",
    "military truck",
    "military jeep",
    "civilian vehicle",
})


# 페어링 레코드가 차량 클래스인지 판별
def _is_vehicle(p) -> bool:
    cls = (p.current_object_class or p.past_object_class or "").lower()
    return cls in _VEHICLE_CLASSES


# 객체 목록의 클래스별 건수를 요약 문자열로 반환
def _class_counts(objs) -> str:
    """Return 'TANK:3 APC:2 ...' summary string from a list of pairing records."""
    from collections import Counter
    counts = Counter(
        p.current_object_class if p.current_object_class else p.past_object_class
        for p in objs
    )
    return "  ".join(f"{cls}:{n}" for cls, n in counts.most_common())


# 페어링 레코드를 LLM 사용자 프롬프트 문자열로 직렬화
def _build_user_prompt(
    pairings: List[PairingRecord],
    historical_context: str = "",
    target_description: str = "",
) -> str:
    """Serialise pairing records into a compact prompt for the LLM."""

    # datetime을 ISO 8601 문자열로 포맷
    def fmt_dt(dt: Optional[datetime]) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else "N/A"

    new_objs          = [p for p in pairings if p.status == "new"]
    disappeared_objs  = [p for p in pairings if p.status == "disappeared"]
    past_not_inc      = [p for p in pairings if p.status == "past_not_included"]
    cur_not_inc       = [p for p in pairings if p.status == "current_not_included"]
    matched_objs      = [p for p in pairings if p.status == "matched"]
    changed_objs      = [p for p in pairings if p.status == "changed"]

    n_matched = len(matched_objs) + len(changed_objs)

    # 현재 프레임 탐지 건수 = new + matched + changed + past_not_included
    n_current_detections = len(new_objs) + n_matched + len(past_not_inc)

    lats = [p.lat_center for p in pairings]
    lons = [p.lon_center for p in pairings]
    lat_c = sum(lats) / len(lats) if lats else 0.0
    lon_c = sum(lons) / len(lons) if lons else 0.0

    past_times = [p.past_capture_time for p in pairings if p.past_capture_time]
    current_times = [p.current_capture_time for p in pairings if p.current_capture_time]
    time_past = min(past_times).strftime("%Y-%m-%dT%H:%M:%SZ") if past_times else "UNKNOWN"
    time_current = max(current_times).strftime("%Y-%m-%dT%H:%M:%SZ") if current_times else "UNKNOWN"

    # 섹션 내 객체를 차량/비차량으로 분류하여 출력
    def _append_section(section_objs, conf_key: str):
        """섹션 내 객체를 출력.
        차량류(_VEHICLE_CLASSES): CLASS_SUMMARY 건수만 표시 (개별 항목 생략).
        비차량류: 개별 항목 (좌표·신뢰도) 표시.
        """
        vehicles     = [p for p in section_objs if _is_vehicle(p)]
        non_vehicles = [p for p in section_objs if not _is_vehicle(p)]

        if vehicles:
            lines.append(f"  VEHICLE_COUNT: {_class_counts(vehicles)}"
                         f"  (총 {len(vehicles)}대 — 위치 상세 생략)")

        if non_vehicles:
            lines.append(f"  CLASS_SUMMARY: {_class_counts(non_vehicles)}")
            top = sorted(non_vehicles,
                         key=lambda p: (getattr(p, conf_key) or 0),
                         reverse=True)
            for p in top[:_MAX_DETAIL]:
                if conf_key == "current_confidence":
                    lines.append(
                        f"  {p.current_object_class} CONF={p.current_confidence:.2f}"
                        f" ({p.current_lat:.3f},{p.current_lon:.3f})"
                        f" DETECTED={fmt_dt(p.current_capture_time)}"
                    )
                else:
                    lines.append(
                        f"  {p.past_object_class} CONF={p.past_confidence:.2f}"
                        f" ({p.past_lat:.3f},{p.past_lon:.3f})"
                        f" LAST_SEEN={fmt_dt(p.past_capture_time)}"
                    )
            if len(non_vehicles) > _MAX_DETAIL:
                lines.append(
                    f"  ... +{len(non_vehicles) - _MAX_DETAIL} more:"
                    f" {_class_counts(non_vehicles[_MAX_DETAIL:])}"
                )

        if not vehicles and not non_vehicles:
            lines.append("  (none)")


    lines = []
    if target_description.strip():
        lines += [
            "=== 표적 정보 (임무계획 기재) ===",
            target_description.strip(),
            "",
        ]
    lines += [
        f"PAST_OBS: {time_past}  CURRENT_OBS: {time_current}  ROI: {lat_c:.3f},{lon_c:.3f}",
        f"CURRENT_FRAME_DETECTIONS: {n_current_detections}"
        f"  (NEW:{len(new_objs)}  STATIONARY:{len(matched_objs)}"
        f"  CHANGED:{len(changed_objs)}"
        f"  PAST_NOT_INCLUDED:{len(past_not_inc)})",
        f"PAST_ONLY: disappeared={len(disappeared_objs)}"
        f"  current_not_included={len(cur_not_inc)}",
        "NOTE: NEW = in overlap zone of both images, detected CURRENT but absent PAST.",
        "NOTE: CHANGED = fixed facility (building/radar) at same location; CLIP similarity below threshold"
        " — structural change detected between frames.",
        "NOTE: DISAPPEARED = in overlap zone of both images, detected PAST but absent CURRENT.",
        "NOTE: PAST_NOT_INCLUDED = current detection outside past image FOV"
        " — cannot confirm if new.",
        "NOTE: CURRENT_NOT_INCLUDED = past detection outside current image FOV"
        " — cannot confirm if gone.",
        "NOTE: Vehicle classes (tank/APC/truck/jeep/vehicle) are reported as counts only.",
        "NOTE: Report covers NEW, CHANGED, DISAPPEARED, PAST_NOT_INCLUDED, CURRENT_NOT_INCLUDED objects.",
    ]

    # --- STATIONARY objects (class breakdown only – for threat context) ---
    if matched_objs:
        lines.append(
            f"\n=== STATIONARY OBJECTS (overlap zone, present in BOTH frames — class breakdown for context) ==="
        )
        lines.append(f"  CLASS_SUMMARY: {_class_counts(matched_objs)}")

    # --- CHANGED objects (fixed facilities with structural change) ---
    if changed_objs:
        lines.append(
            f"\n=== CHANGED OBJECTS"
            f" (고정 시설물, 같은 위치·CLIP 유사도 임계값 미만 → 구조 변화 감지) ==="
        )
        _append_section(changed_objs, "current_confidence")

    # --- NEW objects ---
    lines.append(f"\n=== NEW OBJECTS (overlap zone, first observed at CURRENT_OBS: {time_current}) ===")
    _append_section(new_objs, "current_confidence")

    # --- DISAPPEARED objects ---
    lines.append(
        f"\n=== DISAPPEARED OBJECTS"
        f" (overlap zone, present at PAST_OBS: {time_past}, NOT observed at CURRENT_OBS: {time_current}) ==="
    )
    _append_section(disappeared_objs, "past_confidence")

    # --- PAST_NOT_INCLUDED objects (coverage gap – current detection, no past coverage) ---
    lines.append(
        f"\n=== PAST_NOT_INCLUDED OBJECTS"
        f" (detected CURRENT_OBS: {time_current}, outside past image FOV — unverifiable) ==="
    )
    _append_section(past_not_inc, "current_confidence")

    # --- CURRENT_NOT_INCLUDED objects (coverage gap – past detection, no current coverage) ---
    lines.append(
        f"\n=== CURRENT_NOT_INCLUDED OBJECTS"
        f" (detected PAST_OBS: {time_past}, outside current image FOV — unverifiable) ==="
    )
    _append_section(cur_not_inc, "past_confidence")

    # --- GraphRAG historical context (prepended before task) ---
    if historical_context:
        lines = [historical_context] + lines

    lines += [
        "\n=== 작성 지시 ===",
        "위 데이터를 바탕으로 군사 정보 보고서를 한국어로 작성하세요.",
        "  1) NEW(신규) 및 DISAPPEARED(소실) 객체: 두 영상의 공통 촬영 구역에서 확인된 변화.",
        "  2) CHANGED(시설변화) 객체: 같은 위경도에서 탐지된 고정 시설물(건물·레이더 등)이 두 촬영 시점 간"
        " 구조적 변화가 감지된 경우. 시설 신축·증축·철거·피해 가능성을 변화 분석에 포함하세요.",
        "  3) PAST_NOT_INCLUDED(과거 미포함) 객체: '현재 영상에서 탐지되었으나 과거 촬영 범위 밖에 위치"
        " — 신규 활동 여부 미확인'으로 표기.",
        "  4) CURRENT_NOT_INCLUDED(현재 미포함) 객체: '과거 영상에서 탐지되었으나 현재 촬영 범위 밖에 위치"
        " — 현재 상태 불명, 추가 수집 필요'로 표기.",
        "  5) STATIONARY(정지) 객체: 변화 분석 섹션에서 직접 나열하지 말고,"
        " 탐지된 객체 종류(CLASS_SUMMARY)를 바탕으로 해당 지역의 전력 구성 및 위협 수준 평가에 활용하세요.",
        "  6) 차량류(VEHICLE_COUNT): tank·APC·truck·jeep·vehicle은 개별 위치 없이 총 댓수와 종류만 보고서에 기재하세요.",
        "  7) GRAPHRAG 과거 컨텍스트가 제공된 경우, 위협 평가(6절) 및 정보공백(7절) 분석에 과거 패턴을 반영하세요.",
        "각 객체 종류(TANK, APC, HELICOPTER, civilian building 등)의 군사적 의미를 분석에 반영하세요.",
        "관측 시각은 오늘 날짜가 아닌 PAST_OBS 및 CURRENT_OBS 타임스탬프를 사용하세요.",
        "DISAPPEARED 객체는 '현재 영상에서 더 이상 관측되지 않음'으로 표현하세요.",
        "섹션 구성: 1.분류등급 2.핵심요약 3.상황 4.변화분석"
        " 5.촬영공백구역 6.위협평가 7.정보공백 8.권고조치 9.부록",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

class _VllmBackend:
    def __init__(self):
        self._llm = None

    # vLLM을 사용해 EXAONE 모델 로드
    def _load(self):
        import os
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        from vllm import LLM

        logger.info(f"[Reporter] Loading {LLM_MODEL_NAME} via vLLM (AWQ)...")
        self._llm = LLM(
            model=LLM_MODEL_NAME,
            dtype="float16",
            gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
            max_model_len=LLM_MAX_MODEL_LEN,
            tensor_parallel_size=LLM_TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
        )
        logger.info(f"[Reporter] {LLM_MODEL_NAME} loaded.")

    # vLLM으로 시스템/사용자 프롬프트에 대한 텍스트 생성
    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None) -> str:
        from vllm import SamplingParams

        if self._llm is None:
            self._load()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        sampling_params = SamplingParams(
            temperature=LLM_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else LLM_MAX_NEW_TOKENS,
        )
        outputs = self._llm.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text


class _FallbackBackend:
    """
    Rule-based report generator used when no LLM backend is available.
    Produces a structured summary directly from pairing records.
    """

    # LLM 없이 규칙 기반으로 폴백 보고서 생성
    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None) -> str:
        # Extract the data section from the user_prompt (everything before 작성 지시)
        data_section = user_prompt.split("=== 작성 지시 ===")[0].strip()
        return textwrap.dedent(f"""
            1. 분류등급: 비밀해제 // 공무상 사용 (자동 생성)

            2. 핵심요약
            LLM 백엔드(vLLM / Ollama)를 사용할 수 없어 규칙 기반 폴백 엔진으로 보고서가 생성되었습니다.
            정밀 분석을 위해 아래의 원시 탐지 페어링 데이터를 수동으로 검토하세요.

            3. 상황
            AI 기반 객체 탐지 및 시계열 페어링이 정상 완료되었습니다.
            객체별 상태 분류는 아래 변화 분석 섹션을 참고하세요.

            4. 변화분석
            {data_section}

            5. 촬영공백구역
            LLM 기반 촬영 공백 분석 미수행.

            6. 위협평가
            수동 검토 필요. LLM 기반 위협 평가를 사용할 수 없습니다.
            (탐지 객체 종류 참고: {data_section.split('CLASS_SUMMARY:')[1].split(chr(10))[0].strip() if 'CLASS_SUMMARY:' in data_section else '정보 없음'})

            7. 정보공백
            - LLM 위협 분석 미수행 (모델 미로드).
            - 전체 분석을 위해 transformers + EXAONE 모델을 설치하거나 Ollama를 설정하세요.

            8. 권고조치
            - 페어링 데이터베이스의 이동/신규/소실 객체를 검토하세요.
            - LLM 백엔드를 설정하고 재실행하여 전체 정보 평가를 수행하세요.

            9. 부록
            전체 객체 목록은 페어링 데이터베이스(data/db/object_pairings.db)를 참고하세요.
        """).strip()


class _OllamaBackend:
    """Calls a locally running Ollama server."""

    # Ollama 서버로 요청을 전송하여 텍스트 생성
    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None) -> str:
        import urllib.request

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": max_tokens if max_tokens is not None else LLM_MAX_NEW_TOKENS,
            },
        }).encode()

        url = f"{OLLAMA_BASE_URL}/api/chat"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        logger.info(f"[Reporter] Sending request to Ollama at {url} model={OLLAMA_MODEL}")
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode())
        return body["message"]["content"]


# 설정된 LLM_BACKEND에 맞는 백엔드 인스턴스 반환
def _get_backend():
    if LLM_BACKEND == "ollama":
        return _OllamaBackend()
    if LLM_BACKEND == "vllm":
        return _VllmBackend()
    logger.warning(f"[Reporter] 알 수 없는 LLM_BACKEND='{LLM_BACKEND}', vLLM으로 대체합니다.")
    return _VllmBackend()


class _SafeBackend:
    """
    Wraps any backend and catches load/generate errors,
    falling back to _FallbackBackend if the primary backend fails.
    """

    def __init__(self, primary):
        self._primary = primary
        self._fallback = _FallbackBackend()

    # 기본 백엔드 호출 실패 시 폴백 백엔드로 텍스트 생성
    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None) -> str:
        try:
            return self._primary.generate(system_prompt, user_prompt, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning(
                f"[Reporter] LLM backend failed ({exc}). "
                "Using rule-based fallback report."
            )
            return self._fallback.generate(system_prompt, user_prompt, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Korean translation helper
# ---------------------------------------------------------------------------

# 한국어 번역 전용 시스템 프롬프트 생성
def _build_translation_system_prompt() -> str:
    return (
        "당신은 군사 정보 문서를 전문으로 하는 한국어 번역가입니다. "
        "아래 보고서가 이미 한국어로 작성된 경우 그대로 출력하세요. "
        "영어로 작성된 경우 한국어로 번역하세요. "
        "규칙:\n"
        "1. 좌표, 타임스탬프, 신뢰도 점수, 객체 클래스명(TANK, APC 등), 고유명사는 번역하지 말고 원문 그대로 유지.\n"
        "2. 서술·분석·설명 텍스트는 충실하게 한국어로 번역.\n"
        "3. 내용을 추가·생략·재해석하지 마세요 — 번역만 수행.\n"
        "4. 번역된 텍스트만 출력하고, 서문이나 설명은 포함하지 마세요."
    )


# 기존 백엔드를 재사용하여 보고서를 한국어로 번역
def _translate_report_to_korean(backend, english_text: str) -> str:
    """
    이미 로드된 EXAONE 백엔드를 재사용해 보고서를 한국어로 번역한다.
    번역 실패 시 원문(영어)을 그대로 반환한다.
    """
    try:
        korean_text = backend.generate(
            _build_translation_system_prompt(),
            english_text,
            max_tokens=LLM_TRANSLATE_MAX_TOKENS,
        )
        logger.info("[Reporter] 한국어 번역 완료.")
        return korean_text
    except Exception as exc:
        logger.warning(f"[Reporter] 번역 실패 ({exc}) — 원문(영어)을 사용합니다.")
        return english_text


# ---------------------------------------------------------------------------
# Main reporter class
# ---------------------------------------------------------------------------

class MilitaryReporter:
    """Generates military change-detection reports using EXAONE4-32b."""

    def __init__(self):
        self._backend = _SafeBackend(_get_backend())

        # 교리 RAG 초기화 (DOCTRINE_ENABLED=true 일 때만 로드)
        self._retriever = None
        if DOCTRINE_ENABLED:
            try:
                from src.reporting.doctrine_retriever import DoctrineRetriever
                self._retriever = DoctrineRetriever()
                if self._retriever.is_ready:
                    logger.info("[Reporter] 교리 RAG 활성화됨.")
                else:
                    logger.warning("[Reporter] 교리 RAG 비활성화 — 벡터 DB를 먼저 구축하세요.")
                    self._retriever = None
            except Exception as exc:
                logger.warning(f"[Reporter] DoctrineRetriever 로드 실패 ({exc}) — RAG 없이 진행.")

    # 페어링 레코드로 군사 정보 보고서를 생성하고 DB에 저장
    def generate_report(
        self,
        pairings: List[PairingRecord],
        output_path: Optional[str] = None,
        session_id: Optional[str] = None,
        historical_context: str = "",
        target_description: str = "",
        n_real_detections: Optional[int] = None,
    ) -> str:
        """
        Generate a military intelligence report from the given pairing records,
        then persist it to the Reports DB (saved_time, file_path, full content).

        Args:
            pairings:           List of PairingRecord objects (latest session).
            output_path:        If provided, write the report text to this file path.
            session_id:         Pipeline session UUID (stored in the Reports DB row).
            historical_context: Optional GraphRAG context string injected into the
                                LLM prompt to enrich threat assessment and gap analysis.
            target_description: Optional mission target description prepended to the
                                LLM prompt.

        Returns:
            Report text as a string.
        """
        if not pairings:
            logger.warning("[Reporter] No pairing records provided – returning empty report.")
            return "NO DATA: No pairing records available for report generation."

        report_time = datetime.now(tz=timezone.utc)
        # --- 교리 RAG: 탐지 객체 클래스 기반으로 관련 교리 문서 검색 ---
        doctrine_context = ""
        if self._retriever is not None:
            object_classes = [
                p.current_object_class or p.past_object_class
                for p in pairings
                if p.status in ("new", "disappeared", "past_not_included", "current_not_included")
            ]
            lats = [p.lat_center for p in pairings]
            lons = [p.lon_center for p in pairings]
            region_lat = sum(lats) / len(lats) if lats else 0.0
            region_lon = sum(lons) / len(lons) if lons else 0.0

            doctrine_context = self._retriever.get_context(
                object_classes=object_classes,
                region_lat=region_lat,
                region_lon=region_lon,
                top_k=DOCTRINE_TOP_K,
                max_chars_per_chunk=DOCTRINE_MAX_CHARS,
            )
            if doctrine_context:
                logger.info("[Reporter] 교리 컨텍스트 삽입 완료.")

        system_prompt = _build_system_prompt(doctrine_context)
        user_prompt = _build_user_prompt(
            pairings,
            historical_context=historical_context,
            target_description=target_description,
        )

        logger.info(f"[Reporter] Generating military report for {len(pairings)} pairings...")
        report_text = self._backend.generate(system_prompt, user_prompt)

        # 한국어 번역 (LLM_TRANSLATE=false 환경변수로 비활성화 가능)
        if LLM_TRANSLATE_TO_KOREAN:
            logger.info("[Reporter] 보고서를 한국어로 번역 중...")
            report_text = _translate_report_to_korean(self._backend, report_text)

        # Derive observation period from pairing records (metadata.json capture times)
        past_times = [p.past_capture_time for p in pairings if p.past_capture_time]
        current_times = [p.current_capture_time for p in pairings if p.current_capture_time]
        obs_past = min(past_times).strftime("%Y-%m-%dT%H:%M:%SZ") if past_times else "UNKNOWN"
        obs_current = max(current_times).strftime("%Y-%m-%dT%H:%M:%SZ") if current_times else "UNKNOWN"

        # Prepend metadata header
        n_new_rep          = sum(1 for p in pairings if p.status == 'new')
        n_matched_rep      = sum(1 for p in pairings if p.status == 'matched')
        n_disappeared_rep  = sum(1 for p in pairings if p.status == 'disappeared')
        n_past_not_inc     = sum(1 for p in pairings if p.status == 'past_not_included')
        n_cur_not_inc      = sum(1 for p in pairings if p.status == 'current_not_included')
        # synthetic 탐지(pairing이 추가로 삽입한 추정 객체)를 제외한 실제 탐지 건수
        # n_real_detections 가 제공된 경우 그 값을 우선 사용
        n_current = n_real_detections if n_real_detections is not None \
            else (n_new_rep + n_matched_rep + n_past_not_inc)
        lang_note = "한국어 (EXAONE 번역)" if LLM_TRANSLATE_TO_KOREAN else "English"
        header = (
            f"{'='*72}\n"
            f"  MILITARY INTELLIGENCE REPORT\n"
            f"  Generated by: Multi-Source Intelligent System (MSIS)\n"
            f"  Model: {LLM_MODEL_NAME}\n"
            f"  Language: {lang_note}\n"
            f"  Past observation:    {obs_past}\n"
            f"  Current observation: {obs_current}\n"
            f"  Report generated:    {report_time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f"  Current frame detections: {n_current}"
            f"  (new={n_new_rep} / matched={n_matched_rep}"
            f" / past_not_included={n_past_not_inc})\n"
            f"  Disappeared (past only):  {n_disappeared_rep}\n"
            f"  Current not included:     {n_cur_not_inc}\n"
            f"  Total pairing records:    {len(pairings)}\n"
            f"{'='*72}\n\n"
        )
        full_report = header + report_text

        # Write to file if requested
        saved_file_path: Optional[str] = None
        if output_path:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_report)
            saved_file_path = str(Path(output_path).resolve())
            logger.info(f"[Reporter] Report written to {saved_file_path}")

        # Persist to Reports DB
        db_record = insert_report(
            report_time=report_time.replace(tzinfo=None),  # store as naive UTC
            report_content=full_report,
            llm_model=LLM_MODEL_NAME,
            llm_backend=LLM_BACKEND,
            pairing_count=len(pairings),
            session_id=session_id,
            file_path=saved_file_path,
        )
        logger.info(
            f"[Reporter] Report saved to DB  "
            f"id={db_record.id}  saved_time={db_record.saved_time}"
        )

        return full_report
