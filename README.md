# 영상 변화탐지 판독 프로토타입

Before/After 영상을 업로드하고 **정합 → 객체 탐지 → 변화탐지 → 구조화 → LLM 판독** 파이프라인을 실행하는 웹 프로토타입입니다. 현재는 실제 모델을 차례로 연결하기 쉽도록 각 단계를 인터페이스로 분리했으며, 기본 구현은 결정론적인 데모 결과를 반환합니다.

## 실행

### Streamlit 화면 실행 (권장)

의존성을 설치한 뒤 아래 명령을 실행합니다.

```bash
python -m streamlit run streamlit_app.py
```

브라우저에서 <http://localhost:8501>을 열면 꾸며진 대시보드에서 이미지 업로드,
Before/After 미리보기, 탐지 박스, 변화 목록, 판독 보고서 및 JSON 결과를 확인할 수 있습니다.
분석 결과 아래의 **탐지 박스 편집** 패널에서는 현재 박스의 좌표와 신뢰도를 확인하고,
정규화 좌표(0~1)를 입력해 박스를 추가하거나 목록에서 선택해 삭제할 수 있습니다.
수동 수정 결과는 주석 이미지와 구조화 JSON의 `detections`에 즉시 반영됩니다.
다른 사내 PC에서 접속할 때는 `--server.address 0.0.0.0` 옵션을 추가하세요.

```bash
python -m streamlit run streamlit_app.py --server.address 0.0.0.0
```

### FastAPI 화면 실행

### Debian / Ubuntu 사전 준비

`python -m venv` 실행 중 `ensurepip is not available` 오류가 발생하면 운영체제의
venv 패키지가 설치되지 않은 것입니다. 사용 중인 Python 버전에 맞는 패키지를
먼저 설치하세요.

```bash
python3 --version
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Ubuntu에서 특정 Python 버전(예: 3.10)을 사용하고 있다면 다음 패키지를 설치할
수도 있습니다.

```bash
sudo apt install -y python3.10-venv
```

실패하면서 일부 생성된 가상환경은 재사용하지 말고 삭제한 뒤 다시 만드세요.

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

회사 내부망의 다른 PC에서도 접속해야 한다면 마지막 명령의 host를
`0.0.0.0`으로 바꾸고 서버 방화벽에서 8000 포트를 허용하세요.

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 일반 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

브라우저에서 <http://localhost:8000>을 열어 두 이미지를 선택한 뒤 **분석 시작**을 누릅니다.

이미지를 선택하면 업로드 영역에서 즉시 미리볼 수 있습니다. 분석이 끝나면 결과 영역에도
Before/After 이미지가 나란히 표시되며, 데모 파이프라인이 반환한 객체 위치가 박스로
겹쳐 표시됩니다. 현재 박스는 실제 탐지 결과가 아니라 모델 연결 전 화면 흐름 확인용입니다.

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
