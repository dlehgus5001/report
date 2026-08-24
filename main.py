"""
MSIS Entry Point – Project Maven-inspired Multi-Source Intelligent System

Usage:
    python main.py [--metadata PATH] [--report-output PATH] [--generate-samples]
    python main.py --ingest-only   # 이미지 적재만 수행, session_id 출력

Examples:
    # Run with default sample data
    python main.py

    # Ingest images only (step-based workflow Phase 1)
    python main.py --ingest-only --metadata data/images/metadata.json
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from pipeline import MavenPipeline
from scripts.generate_sample_data import generate_sample_data

logger = logging.getLogger(__name__)


# 명령줄 인수 파싱 및 반환
def parse_args():
    parser = argparse.ArgumentParser(
        description="MSIS: Multi-Source Intelligent System (Project Maven pipeline)"
    )
    parser.add_argument(
        "--metadata",
        default="data/images/metadata.json",
        help="Path to the image metadata index JSON file",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Optional file path to save the generated military report",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate synthetic sample satellite/drone images for testing",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["huggingface", "ollama"],
        default=None,
        help="Override LLM backend (huggingface or ollama)",
    )
    parser.add_argument(
        "--target-description",
        default="",
        help="임무계획에서 입력한 표적 설명 (LLM 보고서 생성 시 컨텍스트로 활용)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="이미지 적재만 수행 (SAM3 탐지·페어링·보고서 생성 없음). "
             "stdout에 INGEST_RESULT:<json> 형식으로 session_id와 image_ids를 출력한다.",
    )
    return parser.parse_args()


# 파이프라인 진입점 실행
def main():
    args = parse_args()

    # Override LLM backend if specified
    if args.llm_backend:
        import os
        os.environ["LLM_BACKEND"] = args.llm_backend

    # Generate sample data if requested
    if args.generate_samples:
        logger.info("Generating synthetic sample data...")
        generate_sample_data(args.metadata)

    # Ensure metadata file exists
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("[]", encoding="utf-8")
        logger.info(
            f"[Main] Created empty metadata file at {meta_path}. "
            "Pipeline will auto-generate synthetic sample data."
        )

    pipeline = MavenPipeline()

    # ── 이미지 적재 전용 모드 (단계별 워크플로우 Phase 1) ───────────────────
    if args.ingest_only:
        session_id = str(uuid.uuid4())
        logger.info("[Main/IngestOnly] session_id=%s", session_id)
        image_ids = pipeline.ingest_only(args.metadata, session_id)
        result = {
            "session_id": session_id,
            "image_ids":  image_ids,
            "success":    True,
        }
        # 부모 프로세스(simulator_runner)가 파싱하는 마커 라인
        print("INGEST_RESULT:" + json.dumps(result, ensure_ascii=False))
        return 0

    # ── 전체 파이프라인 실행 ─────────────────────────────────────────────────
    report = pipeline.run(
        metadata_json=args.metadata,
        report_output_path=args.report_output,
        target_description=args.target_description,
    )

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72 + "\n")

    if args.report_output:
        print(f"Report saved to: {args.report_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
