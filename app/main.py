from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import AnalysisResult
from app.services import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MAX_FILE_SIZE = 15 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Change Intelligence", version="0.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
pipeline = Pipeline()
results: dict[str, AnalysisResult] = {}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "pipeline": "demo"}


async def read_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "PNG, JPEG, WebP 이미지만 업로드할 수 있습니다.")
    data = await file.read(MAX_FILE_SIZE + 1)
    if not data:
        raise HTTPException(400, "빈 파일은 업로드할 수 없습니다.")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "이미지는 15MB 이하여야 합니다.")
    return data


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(before: UploadFile = File(...), after: UploadFile = File(...)) -> AnalysisResult:
    before_data, after_data = await read_image(before), await read_image(after)
    registration, detections, changes, report = pipeline.run(before_data, after_data)
    counts = Counter(change.type.value for change in changes)
    analysis_id = str(uuid4())
    result = AnalysisResult(
        analysis_id=analysis_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        inputs={"before": before.filename or "before", "after": after.filename or "after"},
        registration=registration,
        detections=detections,
        changes=changes,
        summary={kind: counts.get(kind, 0) for kind in ("new", "missing", "moved", "modified")},
        report=report,
    )
    results[analysis_id] = result
    return result


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str) -> AnalysisResult:
    if analysis_id not in results:
        raise HTTPException(404, "분석 결과를 찾을 수 없습니다.")
    return results[analysis_id]

