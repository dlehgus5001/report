"""
Super-Resolution preprocessing for satellite/aerial imagery.

업스케일 우선순위 (SR_BACKEND 환경변수로 선택):
  1. FSRCNN     (opencv-contrib-python dnn_superres — 기본값)
  2. EDSR       (basicsr 패키지 — 한국 SNU, SR_BACKEND="edsr" 선택 시)
  3. Real-ESRGAN (basicsr + realesrgan 패키지 — SR_BACKEND="realesrgan" 선택 시)
  4. PIL LANCZOS (폴백 — 모델 사용 불가 시)

목표 해상도: config.SR_TARGET_W × SR_TARGET_H (기본 8000×6000).
이미지 비율을 유지하며 목표 크기 내에서 최대한 확대.
이미 목표 크기 이상이면 SR 없이 원본 반환.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    SR_TARGET_H, SR_TARGET_W,
    SR_BACKEND,
    FSRCNN_X4_PATH, FSRCNN_X2_PATH,
    EDSR_X4_PATH, EDSR_X2_PATH,
    REALESRGAN_X4_PATH, REALESRGAN_X2_PATH,
)

logger = logging.getLogger(__name__)


# 비율 유지하며 SR 목표 내 최대 출력 크기 계산
def _sr_output_size(w: int, h: int) -> tuple[int, int]:
    """비율 유지하며 SR_TARGET 내 최대 크기 계산. 이미 크면 원본 그대로."""
    scale = min(SR_TARGET_W / w, SR_TARGET_H / h)
    if scale <= 1.0:
        return w, h
    return int(w * scale), int(h * scale)


# 서브프로세스에서 FSRCNN으로 이미지 업스케일
def _upscale_fsrcnn(image_np: np.ndarray, scale: int) -> np.ndarray:
    """FSRCNN x{scale} 업스케일 (OpenCV dnn_superres, 로컬 .pb 가중치 파일 사용).

    cv2 네이티브 크래시(SIGSEGV/SIGABRT)가 메인 파이프라인 프로세스를 죽이는 것을
    방지하기 위해 별도 서브프로세스에서 실행한다.
    서브프로세스가 비정상 종료해도 RuntimeError 를 발생시켜 호출자가 LANCZOS 로 폴백.

    필요 패키지: opencv-contrib-python
    """
    import subprocess
    import sys
    import tempfile
    import os as _os

    model_path = FSRCNN_X4_PATH if scale == 4 else FSRCNN_X2_PATH

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"FSRCNN 가중치 파일을 찾을 수 없습니다: {model_path}\n"
            f"다운로드 명령:\n"
            f"  mkdir -p {Path(model_path).parent}\n"
            f"  wget --no-check-certificate -P {Path(model_path).parent} "
            f"https://github.com/opencv/opencv_contrib/raw/master/modules/"
            f"dnn_superres/models/FSRCNN_x{scale}.pb"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = _os.path.join(tmpdir, "in.npy")
        out_path = _os.path.join(tmpdir, "out.npy")
        np.save(in_path, image_np)

        # cv2 크래시를 서브프로세스로 격리
        script = (
            "import numpy as np, cv2; "
            f"a=np.load({repr(in_path)}); "
            f"sr=cv2.dnn_superres.DnnSuperResImpl_create(); "
            f"sr.readModel({repr(str(model_path))}); "
            f"sr.setModel('fsrcnn',{scale}); "
            "b=cv2.cvtColor(a,cv2.COLOR_RGB2BGR); "
            "c=sr.upsample(b); "
            "d=cv2.cvtColor(c,cv2.COLOR_BGR2RGB); "
            f"np.save({repr(out_path)},d)"
        )

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=300,   # 5분: 대형 이미지(8000×6000) FSRCNN 처리 시간 고려
        )

        if proc.returncode == 0 and Path(out_path).exists():
            result = np.load(out_path)
            import gc; gc.collect()
            return result

        stderr = proc.stderr.decode(errors="replace")
        raise RuntimeError(
            f"FSRCNN 서브프로세스 실패 (exitcode={proc.returncode}): "
            f"{stderr[:400] if stderr else '(stderr 없음)'}"
        )


# EDSR 모델로 이미지 업스케일 수행
def _upscale_edsr(image_np: np.ndarray, scale: int) -> np.ndarray:
    """EDSR x{scale} 업스케일 (basicsr, 한국 SNU 모델, 로컬 가중치 파일 사용).

    EDSR-L (Large) 설정: num_feat=256, num_block=32 — DIV2K 공식 가중치와 일치.
    """
    import torch
    from basicsr.archs.edsr_arch import EDSR

    model_path = EDSR_X4_PATH if scale == 4 else EDSR_X2_PATH

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"EDSR 가중치 파일을 찾을 수 없습니다: {model_path}\n"
            f"다운로드 명령:\n"
            f"  mkdir -p {Path(model_path).parent}\n"
            f"  wget --no-check-certificate -P {Path(model_path).parent} "
            f"https://github.com/XPixelGroup/BasicSR/releases/download/V1.1/"
            f"EDSR_Lx{scale}_f256b32_DIV2K_official-"
            f"{'76ee1c8f' if scale == 4 else 'be38e77d'}.pth"
        )

    model = EDSR(
        num_in_ch=3, num_out_ch=3,
        num_feat=256, num_block=32,
        upscale=scale, res_scale=0.1,
        img_range=255.0, rgb_mean=[0.4488, 0.4371, 0.4040],
    )
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("params_ema", state.get("params", state))
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    img_t = (
        torch.from_numpy(image_np.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        out_t = model(img_t)
    return (
        out_t.squeeze(0).permute(1, 2, 0)
        .clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    )


# Real-ESRGAN 모델로 이미지 업스케일 수행
def _upscale_realesrgan(image_np: np.ndarray, scale: int) -> np.ndarray:
    """Real-ESRGAN x{scale} 업스케일 (로컬 가중치 파일 사용)."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model_path = REALESRGAN_X4_PATH if scale == 4 else REALESRGAN_X2_PATH

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Real-ESRGAN 가중치 파일을 찾을 수 없습니다: {model_path}\n"
            f"다운로드 명령:\n"
            f"  mkdir -p {Path(model_path).parent}\n"
            f"  wget --no-check-certificate -P {Path(model_path).parent} "
            f"https://github.com/xinntao/Real-ESRGAN/releases/download/"
            f"{'v0.1.0' if scale == 4 else 'v0.2.1'}/"
            f"RealESRGAN_x{scale}plus.pth"
        )

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=scale,
    )
    upsampler = RealESRGANer(
        scale=scale, model_path=model_path, model=model,
        tile=512, tile_pad=10, pre_pad=0, half=True,
    )
    out, _ = upsampler.enhance(image_np, outscale=scale)
    return out


# SR 백엔드로 위성 이미지를 목표 해상도로 업스케일
def super_resolve(image_np: np.ndarray) -> np.ndarray:
    """
    위성/항공 이미지를 SR_TARGET(8000×6000) 기준으로 업스케일.

    SR_BACKEND 환경변수(config.SR_BACKEND)로 백엔드를 선택합니다.
      "fsrcnn"     → FSRCNN (opencv-contrib-python 필요) [기본값]
      "edsr"       → EDSR (basicsr 필요, 한국 SNU)
      "realesrgan" → Real-ESRGAN (basicsr + realesrgan 필요)
      "lanczos"    → PIL LANCZOS 강제 사용 (SR 모델 없이 빠른 폴백)

    어떤 백엔드도 실패하면 PIL LANCZOS 로 최종 폴백하며,
    PIL LANCZOS 마저 실패해도 원본 이미지를 반환해 파이프라인이 중단되지 않습니다.

    Args:
        image_np: H×W×3 uint8 RGB numpy array

    Returns:
        업스케일된 H×W×3 uint8 RGB numpy array (실패 시 원본 반환)
    """
    try:
        return _super_resolve_impl(image_np)
    except Exception as exc:
        logger.error(
            f"[SR] 예상치 못한 오류로 SR 전체 실패 ({exc}). 원본 이미지 반환. "
            "파이프라인은 계속 진행됩니다."
        )
        return image_np


# super_resolve의 실제 구현 로직 실행
def _super_resolve_impl(image_np: np.ndarray) -> np.ndarray:
    """super_resolve() 실제 구현 — 호출자가 try/except 로 감싼다."""
    h, w = image_np.shape[:2]
    target_w, target_h = _sr_output_size(w, h)

    logger.info(f"[SR] 시작: 입력={w}×{h}, 목표={target_w}×{target_h}, backend={SR_BACKEND!r}")

    if target_w == w and target_h == h:
        logger.info(f"[SR] 이미지({w}×{h}) 이미 목표 크기 이상, SR 건너뜀.")
        return image_np

    scale_needed = max(target_w / w, target_h / h)
    sr_scale = 4 if scale_needed > 2.0 else 2

    backend = SR_BACKEND.lower().strip()

    # SR 결과를 목표 크기로 미세 보정하여 반환
    def _finalize(sr_np: np.ndarray, label: str) -> np.ndarray:
        """SR 후 target 크기로 미세 보정 (소수점 오차)."""
        sr_h, sr_w = sr_np.shape[:2]
        if (sr_w, sr_h) != (target_w, target_h):
            pil = Image.fromarray(sr_np)
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            sr_np = np.array(pil, dtype=np.uint8)
        logger.info(f"[SR] {label} 완료: {sr_np.shape[1]}×{sr_np.shape[0]}")
        return sr_np

    # ── FSRCNN (기본값) ─────────────────────────────────────────────────────
    # cv2 는 서브프로세스(_upscale_fsrcnn) 안에서만 임포트되므로
    # 메인 프로세스는 cv2 크래시(SIGSEGV/SIGABRT)로부터 보호된다.
    if backend == "fsrcnn":
        _fsrcnn_path = FSRCNN_X4_PATH if sr_scale == 4 else FSRCNN_X2_PATH
        logger.info(f"[SR] FSRCNN 가중치 경로: {_fsrcnn_path} / 존재: {Path(_fsrcnn_path).is_file()}")
        if not Path(_fsrcnn_path).is_file():
            logger.info(
                f"[SR] FSRCNN 가중치 없음 → PIL LANCZOS 폴백. "
                "가중치 다운로드 후 FSRCNN 적용 가능."
            )
        else:
            try:
                logger.info(
                    f"[SR] FSRCNN x{sr_scale} 적용 (서브프로세스): "
                    f"{w}×{h} → ~{w * sr_scale}×{h * sr_scale}"
                )
                return _finalize(_upscale_fsrcnn(image_np, sr_scale), "FSRCNN")
            except Exception as exc:
                logger.warning(f"[SR] FSRCNN 실패 ({type(exc).__name__}: {exc}). PIL LANCZOS 폴백.")

    # ── EDSR ────────────────────────────────────────────────────────────────
    elif backend == "edsr":
        try:
            import basicsr  # noqa: F401
            logger.info(
                f"[SR] EDSR x{sr_scale} 적용 (한국 SNU): "
                f"{w}×{h} → ~{w * sr_scale}×{h * sr_scale}"
            )
            return _finalize(_upscale_edsr(image_np, sr_scale), "EDSR")
        except (ImportError, Exception) as exc:
            logger.warning(f"[SR] EDSR 사용 불가 ({type(exc).__name__}: {exc}). PIL LANCZOS 폴백.")

    # ── Real-ESRGAN ──────────────────────────────────────────────────────────
    elif backend == "realesrgan":
        try:
            import basicsr    # noqa: F401
            import realesrgan # noqa: F401
            logger.info(
                f"[SR] Real-ESRGAN x{sr_scale} 적용: "
                f"{w}×{h} → ~{w * sr_scale}×{h * sr_scale}"
            )
            return _finalize(_upscale_realesrgan(image_np, sr_scale), "Real-ESRGAN")
        except (ImportError, Exception) as exc:
            logger.warning(f"[SR] Real-ESRGAN 사용 불가 ({type(exc).__name__}: {exc}). PIL LANCZOS 폴백.")

    elif backend != "lanczos":
        logger.warning(f"[SR] 알 수 없는 SR_BACKEND='{SR_BACKEND}'. PIL LANCZOS 사용.")

    # ── PIL LANCZOS 폴백 ─────────────────────────────────────────────────────
    logger.info(f"[SR] PIL LANCZOS 실행: {w}×{h} → {target_w}×{target_h}")
    pil = Image.fromarray(image_np)
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    result = np.array(pil, dtype=np.uint8)
    logger.info(f"[SR] PIL LANCZOS 완료: {result.shape[1]}×{result.shape[0]}")
    return result
