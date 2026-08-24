"""
군사 교리 문서 RAG 검색기 (DoctrineRetriever)
==============================================

build_doctrine_vectordb.py 로 생성한 FAISS 인덱스를 로드하고,
주어진 쿼리에 대해 가장 관련성 높은 교리 문서 청크를 반환합니다.

사용 예:
    retriever = DoctrineRetriever()
    context = retriever.get_context(
        object_classes=["military tank", "artillery"],
        region_lat=37.5, region_lon=127.0,
        top_k=5,
    )
    # context: LLM 프롬프트에 삽입할 형식의 문자열
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DoctrineRetriever:
    """
    FAISS 기반 교리 문서 검색기.

    - build_doctrine_vectordb.py 로 생성한 doctrine.index / doctrine.meta.json 을 로드.
    - 탐지된 객체 클래스와 지역 정보를 자연어 쿼리로 변환해 유사 청크 검색.
    - 검색 결과를 LLM 프롬프트에 삽입 가능한 문자열로 포맷.
    - FAISS / sentence-transformers 미설치 시 경고 후 빈 컨텍스트 반환(파이프라인 중단 없음).
    """

    def __init__(self, db_dir: Optional[str] = None, embed_model: Optional[str] = None):
        """
        Args:
            db_dir:      벡터 DB 디렉터리. None 이면 config.DOCTRINE_DB_PATH 사용.
            embed_model: 임베딩 모델명. None 이면 build_info.json 의 모델명 자동 사용.
        """
        from src.config import DOCTRINE_DB_PATH, DOCTRINE_EMBED_MODEL

        self._db_dir     = Path(db_dir or DOCTRINE_DB_PATH)
        self._embed_model_override = embed_model or DOCTRINE_EMBED_MODEL or None

        self._index    = None   # faiss.Index
        self._chunks: list[dict] = []
        self._model    = None   # SentenceTransformer
        self._embed_model_name: str = ""
        self._ready    = False

        self._load()

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    # FAISS 인덱스와 청크 메타데이터를 디스크에서 로드
    def _load(self) -> None:
        """FAISS 인덱스와 메타데이터를 로드한다. 실패 시 경고만 출력하고 계속."""
        index_path = self._db_dir / "doctrine.index"
        meta_path  = self._db_dir / "doctrine.meta.json"
        info_path  = self._db_dir / "build_info.json"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(
                f"[DoctrineRAG] 벡터 DB 없음: {self._db_dir}\n"
                "  교리 RAG 비활성화 — build_doctrine_vectordb.py 를 먼저 실행하세요."
            )
            return

        try:
            import faiss
        except ImportError:
            logger.warning("[DoctrineRAG] faiss 미설치 — 교리 RAG 비활성화. pip install faiss-cpu")
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "[DoctrineRAG] sentence-transformers 미설치 — 교리 RAG 비활성화.\n"
                "  pip install sentence-transformers"
            )
            return

        # build_info 에서 모델명 읽기
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            model_name_from_build = info.get("embed_model", "")
        else:
            model_name_from_build = ""

        # 우선순위: 명시적 override > build_info > 기본값
        self._embed_model_name = (
            self._embed_model_override
            or model_name_from_build
            or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        logger.info(f"[DoctrineRAG] FAISS 인덱스 로드: {index_path}")
        self._index = faiss.read_index(str(index_path))

        logger.info(f"[DoctrineRAG] 메타데이터 로드: {meta_path}")
        self._chunks = json.loads(meta_path.read_text(encoding="utf-8"))

        logger.info(f"[DoctrineRAG] 임베딩 모델 로드: {self._embed_model_name}")
        self._model = SentenceTransformer(self._embed_model_name)

        self._ready = True
        logger.info(
            f"[DoctrineRAG] 준비 완료 — {self._index.ntotal}개 청크 "
            f"/ 모델: {self._embed_model_name}"
        )

    # ------------------------------------------------------------------
    # 쿼리 생성
    # ------------------------------------------------------------------

    # 탐지 클래스와 좌표를 교리 검색용 자연어 쿼리로 변환
    @staticmethod
    def _build_query(object_classes: list[str], region_lat: float, region_lon: float) -> str:
        """
        탐지 객체 클래스와 좌표를 자연어 쿼리로 변환.
        교리 문서에서 해당 무기 체계·전술·위협 평가 관련 내용을 검색하기 위한 쿼리.
        """
        # 고유 클래스 (빈도순)
        unique = list(dict.fromkeys(c for c in object_classes if c))
        class_str = ", ".join(unique[:10]) if unique else "military assets"

        return (
            f"Military doctrine tactics and threat assessment for: {class_str}. "
            f"Operational region coordinates: ({region_lat:.3f}, {region_lon:.3f}). "
            "Include rules of engagement, threat level indicators, "
            "force deployment doctrine, and recommended response actions."
        )

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------

    # 쿼리로 FAISS 인덱스에서 관련 교리 청크 검색
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        쿼리에 대해 상위 top_k 개의 관련 청크를 반환.

        Returns:
            [{"source": str, "page": int, "chunk_id": int, "text": str, "score": float}]
        """
        if not self._ready:
            return []

        import numpy as np

        emb = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        logger.info(
            f"[DoctrineRAG] 검색 완료 — top={k} "
            f"scores={[round(r['score'],3) for r in results]}"
        )
        return results

    # ------------------------------------------------------------------
    # 컨텍스트 포맷
    # ------------------------------------------------------------------

    # 검색 청크를 LLM 프롬프트 삽입용 문자열로 포맷
    @staticmethod
    def _format_context(passages: list[dict], max_chars_per_chunk: int = 600) -> str:
        """
        검색된 청크를 LLM 프롬프트에 삽입 가능한 문자열로 변환.
        각 청크는 max_chars_per_chunk 문자로 잘라서 프롬프트 길이를 제어.
        """
        if not passages:
            return ""

        lines = ["=== DOCTRINE REFERENCE ==="]
        for i, p in enumerate(passages, 1):
            source_info = f"[{i}] {p['source']}"
            if p.get("page"):
                source_info += f" (p.{p['page']})"
            text = p["text"]
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk].rsplit(" ", 1)[0] + "…"
            lines.append(f"{source_info}\n{text}")
        lines.append("=== END DOCTRINE REFERENCE ===")

        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    # 탐지 클래스·좌표 기반으로 관련 교리 컨텍스트 문자열 반환
    def get_context(
        self,
        object_classes: list[str],
        region_lat: float,
        region_lon: float,
        top_k: int = 5,
        max_chars_per_chunk: int = 600,
    ) -> str:
        """
        탐지된 객체 클래스 및 지역 좌표를 기반으로 관련 교리 문서를 검색하고
        LLM 프롬프트에 삽입 가능한 문자열로 반환합니다.

        검색 실패 / DB 없음 / 관련 내용 없음 시 빈 문자열 반환 (파이프라인 중단 없음).
        """
        if not self._ready:
            return ""

        try:
            query = self._build_query(object_classes, region_lat, region_lon)
            passages = self.retrieve(query, top_k=top_k)
            return self._format_context(passages, max_chars_per_chunk=max_chars_per_chunk)
        except Exception as exc:
            logger.warning(f"[DoctrineRAG] 검색 오류 ({exc}) — 컨텍스트 없이 진행.")
            return ""

    # 벡터 DB 로드 완료 여부를 반환하는 프로퍼티
    @property
    def is_ready(self) -> bool:
        """벡터 DB가 정상 로드되어 검색 가능한 상태이면 True."""
        return self._ready
