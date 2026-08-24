"""
HWPX (HWP XML format) 파일 생성 유틸리티.

python-hwpx 라이브러리를 사용하여 한글과컴퓨터 HWPX 포맷으로 변환한다.
"""

import io
import logging

logger = logging.getLogger(__name__)


def make_hwpx(text: str) -> bytes:
    """텍스트를 HWPX 바이너리로 변환하여 반환한다.

    Args:
        text: 보고서 본문 텍스트 (개행 포함)

    Returns:
        HWPX ZIP 바이너리
    """
    from hwpx.document import HwpxDocument

    doc = HwpxDocument.new()
    lines = text.split("\n")
    for line in lines:
        doc.add_paragraph(line)

    buf = io.BytesIO()
    doc.save_to_stream(buf)
    return buf.getvalue()
