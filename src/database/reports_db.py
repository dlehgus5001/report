"""
Reports DB operations – insert and query generated military intelligence reports.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from .models import ReportRecord, create_report_engine
from src.config import REPORTS_DB_PATH

logger = logging.getLogger(__name__)

_engine = None


# 보고서 DB 엔진을 반환(싱글톤)
def get_engine():
    global _engine
    if _engine is None:
        _engine = create_report_engine(REPORTS_DB_PATH)
    return _engine


# 생성된 보고서를 DB에 저장하고 레코드 반환
def insert_report(
    report_time: datetime,
    report_content: str,
    llm_model: str,
    llm_backend: str,
    pairing_count: int,
    session_id: Optional[str] = None,
    file_path: Optional[str] = None,
) -> ReportRecord:
    """
    Save a generated report to the Reports DB.

    Args:
        report_time:    The timestamp embedded in the report header (LLM generation time).
        report_content: Full report text (including the header block).
        llm_model:      Model identifier used for generation (e.g. LGAI-EXAONE/EXAONE-4.0-32B-Instruct).
        llm_backend:    Backend used: "huggingface" | "ollama".
        pairing_count:  Number of pairing records that were analysed.
        session_id:     Pipeline session UUID (optional).
        file_path:      Absolute path to the saved .txt file on disk (optional).

    Returns:
        The inserted ReportRecord ORM object.
    """
    engine = get_engine()
    with Session(engine) as session:
        record = ReportRecord(
            report_time=report_time,
            report_content=report_content,
            llm_model=llm_model,
            llm_backend=llm_backend,
            pairing_count=pairing_count,
            session_id=session_id,
            file_path=file_path,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(
            f"[ReportsDB] Saved report id={record.id}  "
            f"saved_time={record.saved_time}  file_path={record.file_path}"
        )
        return record


# 저장된 모든 보고서를 최신순으로 반환
def get_all_reports(limit: int = None) -> List[ReportRecord]:
    """Return all reports ordered by saved_time descending."""
    engine = get_engine()
    with Session(engine) as session:
        q = (
            session.query(ReportRecord)
            .order_by(ReportRecord.saved_time.desc())
        )
        if limit is not None:
            q = q.limit(limit)
        records = q.all()
        for r in records:
            session.expunge(r)
        return records


# report_id로 보고서 레코드 조회
def get_report_by_id(report_id: str) -> Optional[ReportRecord]:
    engine = get_engine()
    with Session(engine) as session:
        record = session.get(ReportRecord, report_id)
        if record:
            session.expunge(record)
        return record


# 보고서 텍스트 내용을 수정
def update_report_content(report_id: str, new_content: str) -> bool:
    """보고서 텍스트를 수정한다. 성공 시 True, 보고서 없으면 False."""
    engine = get_engine()
    with Session(engine) as session:
        record = session.get(ReportRecord, report_id)
        if record is None:
            return False
        record.report_content = new_content
        session.commit()
        logger.info(f"[ReportsDB] Updated report_content id={report_id}")
        return True


# 세션의 모든 보고서 레코드 삭제
def delete_reports_by_session(session_id: str) -> int:
    """세션의 모든 report_records를 삭제한다. 보고서 재생성 전 호출."""
    engine = get_engine()
    with Session(engine) as session:
        deleted = (
            session.query(ReportRecord)
            .filter(ReportRecord.session_id == session_id)
            .delete(synchronize_session=False)
        )
        session.commit()
    logger.info(f"[ReportsDB] Deleted {deleted} reports for session {session_id}")
    return deleted


# 세션 ID에 해당하는 보고서 전체 반환
def get_reports_by_session(session_id: str) -> List[ReportRecord]:
    engine = get_engine()
    with Session(engine) as session:
        records = (
            session.query(ReportRecord)
            .filter(ReportRecord.session_id == session_id)
            .order_by(ReportRecord.saved_time.desc())
            .all()
        )
        for r in records:
            session.expunge(r)
        return records


# 주어진 세션 중 가장 최근 보고서 반환
def get_latest_report_for_sessions(session_ids: List[str]) -> Optional[ReportRecord]:
    """Return the most recent ReportRecord matching any of the given session_ids."""
    if not session_ids:
        return None
    engine = get_engine()
    with Session(engine) as session:
        record = (
            session.query(ReportRecord)
            .filter(ReportRecord.session_id.in_(session_ids))
            .order_by(ReportRecord.saved_time.desc())
            .first()
        )
        if record:
            session.expunge(record)
        return record
