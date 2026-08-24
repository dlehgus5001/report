"""
군사 교리 문서 벡터 DB 구축 스크립트
=====================================

지정한 디렉터리의 교리 문서를 읽어 FAISS 벡터 인덱스를 생성합니다.
생성된 인덱스는 보고서 생성 시 RAG(Retrieval-Augmented Generation)에 사용됩니다.

지원 포맷:
  .txt  — 텍스트 파일
  .md   — 마크다운
  .pdf  — PDF (pypdf 필요: pip install pypdf)
  .docx — Word 문서 (python-docx 필요: pip install python-docx)

사용법:
  # 기본 경로 사용 (data/doctrine/docs → data/doctrine/vectordb)
  python build_doctrine_vectordb.py

  # 경로 직접 지정
  python build_doctrine_vectordb.py \\
      --docs-dir  /path/to/doctrine/docs \\
      --output-dir /path/to/vectordb \\
      --embed-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \\
      --chunk-size 512 \\
      --chunk-overlap 64

결과물:
  <output-dir>/
    doctrine.index   — FAISS 인덱스
    doctrine.meta.json — 청크 메타데이터 (source, page, text)
    build_info.json  — 빌드 정보 (모델명, 차원, 청크 수 등)
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 문서 로더
# ---------------------------------------------------------------------------

def _load_txt(path: Path) -> list[dict]:
    """TXT / MD 파일 로드 → [{"source": str, "page": 0, "text": str}]"""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"source": path.name, "page": 0, "text": text}]


def _load_pdf(path: Path) -> list[dict]:
    """PDF 로드 (pypdf). 페이지별로 분리."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf 미설치 — PDF 지원 불가. pip install pypdf")
        return []

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"source": path.name, "page": i + 1, "text": text})
    return pages


def _load_docx(path: Path) -> list[dict]:
    """DOCX 로드 (python-docx). 단락 전체를 하나의 텍스트로 합침."""
    try:
        from docx import Document
    except ImportError:
        logger.error("python-docx 미설치 — DOCX 지원 불가. pip install python-docx")
        return []

    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"source": path.name, "page": 0, "text": text}]


def load_documents(docs_dir: Path) -> list[dict]:
    """docs_dir 내 지원 파일을 모두 로드."""
    loaders = {
        ".txt":  _load_txt,
        ".md":   _load_txt,
        ".pdf":  _load_pdf,
        ".docx": _load_docx,
    }
    docs = []
    files = sorted(docs_dir.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        loader = loaders.get(f.suffix.lower())
        if loader is None:
            continue
        logger.info(f"  로드: {f.relative_to(docs_dir)}")
        loaded = loader(f)
        docs.extend(loaded)

    logger.info(f"총 {len(docs)}개 페이지/섹션 로드 완료.")
    return docs


# ---------------------------------------------------------------------------
# 청킹
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """마침표/개행 기준으로 문장 단위 분리."""
    # 한국어 문장 구분자: .  !  ?  \n\n
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(doc: dict, chunk_size: int, overlap: int) -> list[dict]:
    """
    단일 페이지/섹션 텍스트를 chunk_size 문자 단위로 분할.
    인접 청크 간 overlap 문자 중첩.
    """
    text = doc["text"]
    source = doc["source"]
    page = doc["page"]

    if len(text) <= chunk_size:
        return [{
            "source": source,
            "page": page,
            "chunk_id": 0,
            "text": text,
        }]

    # 문장 단위로 쌓아가다가 chunk_size 초과 시 분리
    sentences = _split_into_sentences(text)
    chunks = []
    buf = ""
    chunk_id = 0

    for sent in sentences:
        if len(buf) + len(sent) + 1 > chunk_size and buf:
            chunks.append({
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
                "text": buf.strip(),
            })
            # 오버랩: buf 끝에서 overlap 문자 유지
            buf = buf[-overlap:] if overlap > 0 else ""
            chunk_id += 1
        buf = (buf + " " + sent).strip()

    if buf.strip():
        chunks.append({
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "text": buf.strip(),
        })

    return chunks


def build_chunks(docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    logger.info(f"청크 총 {len(all_chunks)}개 생성 (chunk_size={chunk_size}, overlap={overlap})")
    return all_chunks


# ---------------------------------------------------------------------------
# 임베딩
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict], model_name: str) -> "np.ndarray":
    """sentence-transformers 모델로 청크 텍스트를 임베딩."""
    import numpy as np
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers 미설치.\n"
            "  pip install sentence-transformers"
        )
        sys.exit(1)

    logger.info(f"임베딩 모델 로드: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    logger.info(f"임베딩 중... (총 {len(texts)}개 청크)")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 정규화 → cosine = dot product
    )
    logger.info(f"임베딩 완료. shape={embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# FAISS 인덱스 구축 및 저장
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: "np.ndarray") -> "faiss.Index":
    import numpy as np
    try:
        import faiss
    except ImportError:
        logger.error(
            "faiss 미설치.\n"
            "  pip install faiss-cpu   # CPU 전용\n"
            "  pip install faiss-gpu   # GPU (CUDA)"
        )
        sys.exit(1)

    dim = embeddings.shape[1]
    logger.info(f"FAISS IndexFlatIP 생성 (dim={dim})")
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine (embeddings are L2-normalized)
    index.add(embeddings.astype("float32"))
    logger.info(f"인덱스에 {index.ntotal}개 벡터 추가 완료.")
    return index


def save_vectordb(
    index: "faiss.Index",
    chunks: list[dict],
    output_dir: Path,
    model_name: str,
    chunk_size: int,
    overlap: int,
) -> None:
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)

    # FAISS 인덱스
    index_path = output_dir / "doctrine.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 인덱스 저장: {index_path}")

    # 청크 메타데이터 (인덱스 순서와 동일)
    meta_path = output_dir / "doctrine.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"메타데이터 저장: {meta_path}")

    # 빌드 정보
    info = {
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "embed_model": model_name,
        "dimension": index.d,
        "total_chunks": index.ntotal,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "index_type": "IndexFlatIP",
    }
    info_path = output_dir / "build_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    logger.info(f"빌드 정보 저장: {info_path}")


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------

def parse_args():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="군사 교리 문서를 FAISS 벡터 DB로 변환합니다."
    )
    parser.add_argument(
        "--docs-dir",
        default=str(base / "data" / "doctrine" / "docs"),
        help="교리 문서 디렉터리 (기본: data/doctrine/docs)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(base / "data" / "doctrine" / "vectordb"),
        help="벡터 DB 저장 디렉터리 (기본: data/doctrine/vectordb)",
    )
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="sentence-transformers 임베딩 모델명 또는 로컬 경로",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="청크 최대 문자 수 (기본: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="인접 청크 간 중첩 문자 수 (기본: 64)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    docs_dir   = Path(args.docs_dir)
    output_dir = Path(args.output_dir)

    if not docs_dir.exists():
        logger.error(f"교리 문서 디렉터리가 존재하지 않습니다: {docs_dir}")
        logger.error("디렉터리를 생성하고 .txt/.pdf/.docx/.md 파일을 넣어주세요.")
        sys.exit(1)

    logger.info(f"=== 군사 교리 벡터 DB 구축 시작 ===")
    logger.info(f"  문서 디렉터리: {docs_dir}")
    logger.info(f"  출력 디렉터리: {output_dir}")
    logger.info(f"  임베딩 모델:   {args.embed_model}")
    logger.info(f"  청크 크기:     {args.chunk_size}자  (겹침: {args.chunk_overlap}자)")

    # 1. 문서 로드
    docs = load_documents(docs_dir)
    if not docs:
        logger.error("로드된 문서가 없습니다. 지원 포맷: .txt .md .pdf .docx")
        sys.exit(1)

    # 2. 청킹
    chunks = build_chunks(docs, args.chunk_size, args.chunk_overlap)

    # 3. 임베딩
    embeddings = embed_chunks(chunks, args.embed_model)

    # 4. FAISS 인덱스 구축
    index = build_faiss_index(embeddings)

    # 5. 저장
    save_vectordb(index, chunks, output_dir, args.embed_model, args.chunk_size, args.chunk_overlap)

    logger.info("=== 벡터 DB 구축 완료 ===")
    logger.info(f"  총 {index.ntotal}개 청크가 {output_dir}에 저장되었습니다.")
    logger.info("보고서 생성 시 자동으로 로드됩니다 (DOCTRINE_ENABLED=true 확인).")


if __name__ == "__main__":
    main()
