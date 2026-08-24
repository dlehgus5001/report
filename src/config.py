"""
Project Maven-inspired Multi-Source Intelligent System Configuration
Based on DoD Project Maven architecture for AI-driven aerial imagery analysis.
"""

import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_DIR = DATA_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# --- Database ---
SENSOR_DB_PATH = str(DB_DIR / "sensor_detections.db")
PAIRING_DB_PATH = str(DB_DIR / "object_pairings.db")
REPORTS_DB_PATH = str(DB_DIR / "reports.db")

# --- Super-Resolution ---
# SAM3 탐지 전 이미지를 업스케일할 목표 해상도 (픽셀).
# 비율을 유지하며 이 크기에 맞게 확대. Real-ESRGAN 미설치 시 PIL LANCZOS로 폴백.
SR_TARGET_W = int(os.getenv("SR_TARGET_W", "8000"))
SR_TARGET_H = int(os.getenv("SR_TARGET_H", "6000"))

# SR 백엔드 선택: "fsrcnn" (기본, OpenCV DNN) | "edsr" (한국 SNU) | "realesrgan" (중국 Xintao) | "lanczos" (폴백)
# 환경변수 SR_BACKEND 로 런타임 오버라이드 가능.
SR_BACKEND = os.getenv("SR_BACKEND", "fsrcnn")

# SR 모델 가중치 기본 다운로드 디렉터리 (프로젝트 루트/downloads/)
_SR_DOWNLOAD_DIR = BASE_DIR / "downloads"

# FSRCNN 로컬 가중치 경로 (OpenCV contrib dnn_superres 모듈 사용).
# 필요 패키지: opencv-contrib-python
# 다운로드:
#   mkdir -p downloads/fsrcnn
#   wget --no-check-certificate -P downloads/fsrcnn https://github.com/opencv/opencv_contrib/raw/master/modules/dnn_superres/models/FSRCNN_x4.pb
#   wget --no-check-certificate -P downloads/fsrcnn https://github.com/opencv/opencv_contrib/raw/master/modules/dnn_superres/models/FSRCNN_x2.pb
FSRCNN_X4_PATH = os.getenv(
    "FSRCNN_X4_PATH",
    str(_SR_DOWNLOAD_DIR / "fsrcnn" / "FSRCNN_x4.pb"),
)
FSRCNN_X2_PATH = os.getenv(
    "FSRCNN_X2_PATH",
    str(_SR_DOWNLOAD_DIR / "fsrcnn" / "FSRCNN_x2.pb"),
)

# EDSR 로컬 가중치 경로 (한국 SNU — 2017 NTIRE SR 1위, SR_BACKEND="edsr" 선택 시 사용).
# 다운로드:
#   mkdir -p downloads/edsr
#   wget --no-check-certificate -P downloads/edsr https://github.com/XPixelGroup/BasicSR/releases/download/V1.1/EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth
#   wget --no-check-certificate -P downloads/edsr https://github.com/XPixelGroup/BasicSR/releases/download/V1.1/EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth
EDSR_X4_PATH = os.getenv(
    "EDSR_X4_PATH",
    str(_SR_DOWNLOAD_DIR / "edsr" / "EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth"),
)
EDSR_X2_PATH = os.getenv(
    "EDSR_X2_PATH",
    str(_SR_DOWNLOAD_DIR / "edsr" / "EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth"),
)

# Real-ESRGAN 로컬 가중치 경로 (SR_BACKEND="realesrgan" 선택 시 사용).
# 다운로드:
#   mkdir -p downloads/realesrgan
#   wget --no-check-certificate -P downloads/realesrgan https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
#   wget --no-check-certificate -P downloads/realesrgan https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
REALESRGAN_X4_PATH = os.getenv(
    "REALESRGAN_X4_PATH",
    str(_SR_DOWNLOAD_DIR / "realesrgan" / "RealESRGAN_x4plus.pth"),
)
REALESRGAN_X2_PATH = os.getenv(
    "REALESRGAN_X2_PATH",
    str(_SR_DOWNLOAD_DIR / "realesrgan" / "RealESRGAN_x2plus.pth"),
)

# --- SAM3 Model (Segment Anything Model 3 by Meta AI) ---
# SAM3 performs text-prompted concept segmentation in a single forward pass,
# replacing the SAM3 + CLIP two-stage pipeline.
# Model card: https://huggingface.co/facebook/sam3
SAM3_MODEL_NAME = os.getenv("SAM3_MODEL_NAME", "facebook/sam3")
SAM3_DEVICE = os.getenv("SAM3_DEVICE", "cuda")  # "cuda" or "cpu"
# 로컬 가중치 파일 경로. 설정 시 SAM3_MODEL_NAME(HuggingFace ID) 대신 이 경로로 모델을 로드한다.
# 환경변수 SAM3_CHECKPOINT 로 런타임 오버라이드 가능.
SAM3_CHECKPOINT = os.getenv(
    "SAM3_CHECKPOINT",
    "/home/work/AILAB/LanguageModels/VLM/sam3/sam3.pt",
)
# SAM3 requires a fixed inference resolution of 1008×1008
SAM3_INFERENCE_SIZE = 1008

# --- Sliding Window (Tiled Detection) ---
# 이미지를 타일로 분할해 각 타일마다 SAM3를 실행 → 작은 객체 탐지 향상.
# TILE_ENABLED=false 로 비활성화 시 전체 이미지를 한 번에 처리 (기존 방식).
TILE_ENABLED    = os.getenv("TILE_ENABLED",    "true").lower() == "true"
# 타일 크기 (픽셀). SAM3 입력 해상도(1008)와 맞추는 것이 권장.
TILE_SIZE       = int(os.getenv("TILE_SIZE",   "1008"))
# 인접 타일 간 겹치는 픽셀 수.
# 값이 클수록 타일 경계에 걸친 객체가 최소 한 타일에 완전히 포함될 가능성이 높아짐.
# 군사 이미지에서 대형 건물/차량은 최대 400~500px 수준이므로 400으로 설정.
TILE_OVERLAP    = int(os.getenv("TILE_OVERLAP", "400"))
# 타일 NMS IoU 임계값 (타일 병합 시 중복 제거).
TILE_NMS_IOU    = float(os.getenv("TILE_NMS_IOU", "0.3"))
# IoMin 임계값: 교집합 / min(두 박스 면적).
# 멀티스케일에서 크기가 다른 두 박스가 같은 객체를 중복 탐지할 때
# IoU가 낮아 NMS를 통과하는 문제를 보완. IoU OR IoMin 중 하나라도 초과하면 억제.
TILE_NMS_IOMIN  = float(os.getenv("TILE_NMS_IOMIN", "0.5"))
# 인접 박스 병합 임계값 (픽셀).
# 0 = 비활성화. TILE_OVERLAP=400 + IoMin NMS로 타일 경계 분리는 이미 처리되므로
# TILE_MERGE_GAP을 켜두면 도심 밀집 지역에서 인접 건물들이 하나로 묶히는 부작용이 생긴다.
TILE_MERGE_GAP  = int(os.getenv("TILE_MERGE_GAP", "0"))
# 멀티스케일: 타일 탐지와 함께 전체 이미지 탐지도 병행 → 큰 객체 누락 방지.
# TILE_ENABLED=true 일 때만 적용. false 면 타일 탐지만 실행.
TILE_MULTISCALE = os.getenv("TILE_MULTISCALE", "true").lower() == "true"
# 중간 크기 타일 탐지 (TILE_MULTISCALE=true 일 때만 적용).
# TILE_SIZE의 2배 크기 타일로 중형 객체(차량, 건물 등) 탐지 향상.
TILE_MEDIUM_SCALE   = os.getenv("TILE_MEDIUM_SCALE",   "true").lower() == "true"
TILE_MEDIUM_SIZE    = int(os.getenv("TILE_MEDIUM_SIZE",    str(1008 * 2)))   # 기본 2016px
TILE_MEDIUM_OVERLAP = int(os.getenv("TILE_MEDIUM_OVERLAP", str(400  * 2)))   # 기본 800px

# --- Military Object Classes (aerial/satellite imagery) ---
MILITARY_OBJECT_CLASSES = [
    "military tank",
    "armored personnel carrier",
    "military truck",
    "military jeep",
    "fighter aircraft",
    "helicopter",
    "military ship",
    "civilian ship",
    "missile launcher",
    "artillery",
    "military building",
    "radar installation",
    "military personnel",
    "supply depot",
    "fuel storage",
    "command post",
    "civilian vehicle",
    "civilian building",
    "road",
    "runway",
    "unknown object",
]

# --- Coordinate Matching ---
# Radius (in degrees) to consider two detections from the "same region"
# ~1 degree ≈ 111 km; 0.001 degree ≈ 111 m
COORDINATE_MATCH_RADIUS_DEG = float(os.getenv("COORD_MATCH_RADIUS", "0.01"))

# Minimum lat/lon displacement (degrees) for static object same-location constraint.
# 0.001 deg ≈ 111 m at equator – used as geo proximity threshold for static classes.
MOVE_DISTANCE_THRESHOLD_DEG = float(os.getenv("MOVE_DISTANCE_THRESHOLD", "0.001"))

# Same-location threshold for physically fixed facilities (buildings, radar, etc.).
# Two detections of the same fixed object from different satellite frames
# can differ by 10-100 m due to image geo-registration errors and detection
# centroid variability (especially at 5-30 m/px resolution).
# Greedy nearest-first assignment in Step 0 prevents mismatching adjacent buildings
# even with a generous threshold.
# Default 0.001 deg ≈ 111 m (overridable via STATIC_EXACT_MATCH env var).
STATIC_EXACT_MATCH_DEG = float(os.getenv("STATIC_EXACT_MATCH", "0.001"))

# --- LLM (EXAONE-3.5-7.8B-Instruct-AWQ via vLLM) ---
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "/content/drive/MyDrive/multi-source-intelligent-system-claude-satellite-object-detection/models/EXAONE-3.5-7.8B-Instruct-AWQ",
)
LLM_DEVICE = os.getenv("LLM_DEVICE", "cuda")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_GPU_MEMORY_UTILIZATION = float(os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.85"))
LLM_TENSOR_PARALLEL_SIZE = int(os.getenv("LLM_TENSOR_PARALLEL_SIZE", "1"))
# vLLM max_model_len — EXAONE-3.5-7.8B supports 32 768 tokens.
# Default 8192 gives enough headroom for RAG context + generation.
# Set to 32768 to use the full context window (requires more VRAM).
LLM_MAX_MODEL_LEN = int(os.getenv("LLM_MAX_MODEL_LEN", "8192"))
# Ollama endpoint alternative
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone4:32b")
# Use "vllm" or "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "vllm")
# 보고서 번역 — LLM_TRANSLATE=false 로 비활성화 가능
LLM_TRANSLATE_TO_KOREAN = os.getenv("LLM_TRANSLATE", "true").lower() == "true"
# 번역 시 최대 출력 토큰 (보고서 길이를 여유 있게 커버)
LLM_TRANSLATE_MAX_TOKENS = int(os.getenv("LLM_TRANSLATE_MAX_TOKENS", "4096"))

# --- Detection Confidence ---
DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("DETECTION_CONFIDENCE", "0.3"))
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU", "0.3"))

# --- BBox Size Constraints ---
# 일반 객체(건물·차량·탱크 등)는 타일 면적의 15% 이하.
MAX_BBOX_AREA_RATIO = float(os.getenv("MAX_BBOX_AREA_RATIO", "0.15"))
# 대형 객체(선박·활주로·도로)는 타일 면적의 70%까지 허용.
MAX_BBOX_AREA_RATIO_LARGE = float(os.getenv("MAX_BBOX_AREA_RATIO_LARGE", "0.7"))
# 위 비율을 적용할 대형 클래스 목록 (소문자, 공백 포함).
LARGE_OBJECT_CLASSES: frozenset = frozenset({
    "military ship", "civilian ship", "runway", "road",
})
# SAM3 마스크 세그멘테이션 전용 신뢰도 임계값 (post_process_instance_segmentation)
SAM3_MASK_SCORE_THRESHOLD = float(os.getenv("SAM3_MASK_SCORE", "0.5"))

# --- Tracking Mode ---
# "sam3_tracker" : SAM3 video predictor tracks past objects into the current frame
#                  (requires GPU; accurate but compute-heavy)
# "similarity"   : matches current detections to past detections by geo-distance
#                  + class similarity; no video session needed (CPU-friendly)
TRACKING_MODE = os.getenv("TRACKING_MODE", "sam3_tracker")

# CLIP model used for visual embedding in similarity mode
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch16")
# Weight of CLIP cosine similarity vs geo proximity score (0.0 – 1.0)
# score = CLIP_WEIGHT * clip_sim + (1 - CLIP_WEIGHT) * geo_score
SIMILARITY_CLIP_WEIGHT = float(os.getenv("SIMILARITY_CLIP_WEIGHT", "0.7"))
# Weight of size similarity (bbox/mask area ratio) in the combined score (0.0 – 1.0)
# final_score = (1 - SIZE_WEIGHT) * clip_or_geo_score + SIZE_WEIGHT * size_sim
SIMILARITY_SIZE_WEIGHT = float(os.getenv("SIMILARITY_SIZE_WEIGHT", "0.2"))
# Minimum combined score for a current↔past pair to be accepted as a match.
# Pairs whose best score does not exceed this threshold are treated as
# "new" (current) or "disappeared" (past) instead of being matched.
# Range: -1.0 – 1.0 for pure CLIP cosine; 0.0 – 1.0 for combined score.
SIMILARITY_MATCH_THRESHOLD = float(os.getenv("SIMILARITY_MATCH_THRESHOLD", "0.5"))

# --- Simulator Image Mode ---
# SAM3 입력 이미지를 준비하는 방식.
#
#   IMAGE_MODE=separate   sample/ 에서 서로 다른 이미지 2장을 선택 (기본값)
#   IMAGE_MODE=crop       sample/ 에서 이미지 1장을 선택한 뒤 두 영역으로 크롭해 2장으로 활용
#                         두 크롭은 동일 지역을 포함하되 CROP_OFFSET 만큼 이동해 일부 겹침.
#   IMAGE_MODE=crops_dir  sample/.crops/ 폴더에 미리 배치된 *_crop_A.png + *_crop_B.png
#                         파일 쌍을 찾아 선택. 사전에 crop 모드로 생성하거나 직접 배치 가능.
#                         파일 수정시간(mtime)을 촬영 시각으로 사용하며, geo 범위 조정 없음.
#
# crop 모드 전용 파라미터:
#   CROP_AXIS    분할 축  — "vertical" (좌/우 오프셋, 기본) | "horizontal" (상/하 오프셋)
#   CROP_SIZE    각 크롭의 크기 (원본 이미지 대비 비율, 0.0~1.0, 기본 0.7)
#                예) 0.7 → 각 크롭이 원본의 70% 크기
#   CROP_OFFSET  두 크롭 시작점 간의 이동 비율 (원본 이미지 대비, 0.0~1.0, 기본 0.15)
#                예) 0.15 → crop_A 대비 crop_B를 15% 이동
#                겹침 영역 ≈ CROP_SIZE − CROP_OFFSET (= 기본 55%)
IMAGE_MODE   = os.getenv("IMAGE_MODE",   "separate")
CROP_AXIS    = os.getenv("CROP_AXIS",    "vertical")
CROP_SIZE    = float(os.getenv("CROP_SIZE",   "0.7"))
CROP_OFFSET  = float(os.getenv("CROP_OFFSET", "0.15"))

# --- Doctrine RAG ---
# 군사 교리 문서를 FAISS 벡터 DB로 색인해 보고서 생성 시 RAG로 참조합니다.
# build_doctrine_vectordb.py 로 벡터 DB를 먼저 구축해야 합니다.
#
#   DOCTRINE_ENABLED   true  이면 보고서 생성 시 교리 컨텍스트를 LLM 프롬프트에 삽입
#   DOCTRINE_DB_PATH   벡터 DB 디렉터리 (doctrine.index + doctrine.meta.json)
#   DOCTRINE_EMBED_MODEL  임베딩 모델명 (미설정 시 build_info.json 에서 자동 읽음)
#   DOCTRINE_TOP_K     검색할 상위 청크 수 (기본 5)
#   DOCTRINE_MAX_CHARS 청크당 최대 삽입 문자 수 (기본 600, 너무 크면 프롬프트 초과 주의)
DOCTRINE_ENABLED    = os.getenv("DOCTRINE_ENABLED", "false").lower() == "true"
DOCTRINE_DB_PATH    = os.getenv("DOCTRINE_DB_PATH",
                                str(DATA_DIR / "doctrine" / "vectordb"))
DOCTRINE_EMBED_MODEL = os.getenv("DOCTRINE_EMBED_MODEL", "")
DOCTRINE_TOP_K      = int(os.getenv("DOCTRINE_TOP_K", "5"))
DOCTRINE_MAX_CHARS  = int(os.getenv("DOCTRINE_MAX_CHARS", "600"))

# --- GraphRAG Knowledge Graph ---
GRAPH_DB_PATH = str(DB_DIR / "graph.db")
# How often to run community detection (every N pipeline runs; 1 = every run)
GRAPHRAG_COMMUNITY_INTERVAL = int(os.getenv("GRAPHRAG_COMMUNITY_INTERVAL", "1"))
# Radius (degrees) used when querying historical context from the graph
GRAPHRAG_CONTEXT_RADIUS_DEG = float(os.getenv("GRAPHRAG_CONTEXT_RADIUS", "0.05"))
# Set to "0" or "false" to disable injecting graph context into the LLM prompt.
GRAPHRAG_CONTEXT_ENABLED = os.getenv("GRAPHRAG_CONTEXT_ENABLED", "true").lower() not in ("0", "false", "no")

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
