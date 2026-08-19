# 영상 변화탐지 판독 프로토타입

Before/After 영상을 업로드하고 **정합 → 객체 탐지 → 변화탐지 → 구조화 → LLM 판독** 파이프라인을 실행하는 웹 프로토타입입니다. 현재는 실제 모델을 차례로 연결하기 쉽도록 각 단계를 인터페이스로 분리했으며, 기본 구현은 결정론적인 데모 결과를 반환합니다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

브라우저에서 <http://localhost:8000>을 열어 두 이미지를 선택한 뒤 **분석 시작**을 누릅니다.

## 모델 연결 위치

- `app/services.py`의 `RegistrationModel`: 영상 정합 모델 연결
- `app/services.py`의 `DetectionModel`: 객체 탐지 모델 연결
- `app/services.py`의 `ChangeModel`: 변화 분류/영역 추출 모델 연결
- `app/services.py`의 `ReportModel`: LLM 또는 RAG 파이프라인 연결

각 클래스의 반환 스키마만 유지하면 UI와 API를 변경하지 않고 모델을 교체할 수 있습니다. 프로덕션에서는 작업 큐, 오브젝트 스토리지, 인증, 파일 바이러스 검사 및 모델별 타임아웃을 추가하세요.

## API

- `POST /api/analyze`: `before`, `after` 이미지 업로드 및 분석
- `GET /api/analyses/{analysis_id}`: 저장된 결과 조회
- `GET /api/health`: 상태 확인

