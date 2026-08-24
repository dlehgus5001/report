"""
SAM3 파인튜닝 스크립트

데이터셋 구조 (두 가지 포맷 지원):
  1. COCO JSON 포맷:
       data/
         images/train/  *.jpg|*.png
         images/val/    *.jpg|*.png
         annotations/train.json   (COCO segmentation format)
         annotations/val.json

  2. 단순 폴더 포맷:
       data/
         images/train/  *.jpg|*.png
         images/val/    *.jpg|*.png
         masks/train/   *.png  (이진 마스크, 파일명 동일)
         masks/val/     *.png

Usage:
    python finetune_sam3.py --data-dir data/finetune --epochs 20
    python finetune_sam3.py --data-dir data/finetune --epochs 20 --freeze-encoder
    python finetune_sam3.py --data-dir data/finetune --epochs 20 --format coco
    python finetune_sam3.py --data-dir data/finetune --epochs 20 --resume checkpoints/last.pth
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 기본 설정 ──────────────────────────────────────────────────────────────
DEFAULT_SAM3_PATH   = Path(__file__).parent / "src/detection/sam3"
DEFAULT_MODEL_NAME  = "facebook/sam3"
DEFAULT_LR          = 1e-4
DEFAULT_BATCH_SIZE  = 2
DEFAULT_EPOCHS      = 20
DEFAULT_IMG_SIZE    = 1008   # SAM3 고정 입력 해상도
DEFAULT_CKPT_DIR    = "checkpoints"
DEFAULT_CLASSES = [
    "military tank",
    "armored personnel carrier",
    "military truck",
    "fighter aircraft",
    "helicopter",
    "military ship",
    "missile launcher",
    "artillery",
    "military building",
    "radar installation",
]


# ══════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred).flatten(1)
        target = target.flatten(1).float()
        inter  = (pred * target).sum(1)
        denom  = pred.sum(1) + target.sum(1)
        return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


class SegLoss(nn.Module):
    """BCE + Dice 결합 손실."""
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_w   = bce_weight
        self.dice    = DiceLoss()
        self.bce     = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        bce  = self.bce(logits, masks.float())
        dice = self.dice(logits, masks).mean()
        return self.bce_w * bce + (1.0 - self.bce_w) * dice


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class FolderDataset(Dataset):
    """이미지 폴더 + 이진 마스크 폴더 포맷."""

    def __init__(self, img_dir: str, mask_dir: str, img_size: int,
                 class_name: str = "object"):
        self.img_size   = img_size
        self.class_name = class_name
        img_dir  = Path(img_dir)
        mask_dir = Path(mask_dir)

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        self.samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                # jpg 마스크도 허용
                for ext in [".jpg", ".jpeg", ".tif"]:
                    alt = mask_dir / (img_path.stem + ext)
                    if alt.exists():
                        mask_path = alt
                        break
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
            else:
                logger.warning(f"마스크 없음, 건너뜀: {img_path.name}")

        logger.info(f"[Dataset] {len(self.samples)}개 샘플 ({img_dir})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img  = Image.open(img_path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST)

        img_t  = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy((np.array(mask) > 127).astype(np.uint8)).unsqueeze(0)

        return {"image": img_t, "mask": mask_t,
                "class": self.class_name, "path": str(img_path)}


class COCODataset(Dataset):
    """COCO segmentation JSON 포맷."""

    def __init__(self, img_dir: str, ann_file: str, img_size: int,
                 classes: list = None):
        self.img_dir  = Path(img_dir)
        self.img_size = img_size
        self.classes  = {c.lower() for c in (classes or DEFAULT_CLASSES)}

        with open(ann_file) as f:
            coco = json.load(f)

        # category id → name 매핑
        cat_map = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}
        # image id → file_name
        img_map = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        # image id → image size
        img_size_map = {img["id"]: (img["width"], img["height"])
                        for img in coco.get("images", [])}

        # 관심 클래스 annotation만 수집
        self.samples = []
        for ann in coco.get("annotations", []):
            cat_name = cat_map.get(ann.get("category_id"), "")
            if self.classes and cat_name not in self.classes:
                continue
            if not ann.get("segmentation"):
                continue
            img_id   = ann["image_id"]
            fname    = img_map.get(img_id, "")
            orig_wh  = img_size_map.get(img_id, (1, 1))
            self.samples.append({
                "img_path":  self.img_dir / fname,
                "seg":       ann["segmentation"],
                "orig_w":    orig_wh[0],
                "orig_h":    orig_wh[1],
                "class":     cat_name,
            })

        logger.info(f"[COCODataset] {len(self.samples)}개 annotation 로드 ({ann_file})")

    def _seg_to_mask(self, seg, orig_w, orig_h):
        """COCO segmentation(polygon or RLE) → 이진 numpy 마스크."""
        try:
            from pycocotools import mask as coco_mask
            if isinstance(seg, list):
                rle = coco_mask.frPyObjects(seg, orig_h, orig_w)
                m   = coco_mask.decode(coco_mask.merge(rle))
            else:
                m = coco_mask.decode(seg)
            return m.astype(np.uint8)
        except ImportError:
            # pycocotools 없으면 간단한 polygon fill
            from PIL import ImageDraw
            mask = Image.new("L", (orig_w, orig_h), 0)
            draw = ImageDraw.Draw(mask)
            if isinstance(seg, list):
                for poly in seg:
                    pts = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                    draw.polygon(pts, fill=255)
            return np.array(mask) > 127

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s        = self.samples[idx]
        img      = Image.open(s["img_path"]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        raw_mask = self._seg_to_mask(s["seg"], s["orig_w"], s["orig_h"])
        mask     = Image.fromarray(raw_mask * 255, "L").resize(
            (self.img_size, self.img_size), Image.NEAREST)

        img_t  = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy((np.array(mask) > 127).astype(np.uint8)).unsqueeze(0)

        return {"image": img_t, "mask": mask_t,
                "class": s["class"], "path": str(s["img_path"])}


# ══════════════════════════════════════════════════════════════════════════
# 지표
# ══════════════════════════════════════════════════════════════════════════

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union + 1e-6)


# ══════════════════════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, sam3_path: str, device: str, freeze_encoder: bool):
    """SAM3 이미지 모델 로드 및 파인튜닝 설정."""
    if sam3_path and Path(sam3_path).exists():
        if str(sam3_path) not in sys.path:
            sys.path.insert(0, str(sam3_path))

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    logger.info(f"SAM3 모델 로딩: {model_name} on {device}")
    model = build_sam3_image_model(model_name)
    model = model.to(device)

    if freeze_encoder:
        # 이미지 인코더(backbone) 파라미터 동결 → 마스크 디코더만 학습
        frozen, trainable = 0, 0
        for name, param in model.named_parameters():
            if any(k in name for k in ["image_encoder", "trunk", "neck", "backbone"]):
                param.requires_grad = False
                frozen += param.numel()
            else:
                param.requires_grad = True
                trainable += param.numel()
        logger.info(f"인코더 동결: {frozen:,}  /  학습 가능: {trainable:,} 파라미터")
    else:
        trainable = sum(p.numel() for p in model.parameters())
        logger.info(f"전체 학습: {trainable:,} 파라미터")

    processor = Sam3Processor(model)
    return model, processor


def forward_batch(model, processor, batch, device: str):
    """배치 단위 forward → 마스크 logits 반환."""
    images  = batch["image"].to(device)   # (B, 3, H, W)
    classes = batch["class"]              # list[str]

    all_logits = []
    for i in range(images.size(0)):
        pil_img = Image.fromarray(
            (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        state  = processor.set_image(pil_img)
        output = processor.set_text_prompt(state=state, prompt=classes[i])

        masks_out = output.get("masks", None)
        if masks_out is not None:
            if hasattr(masks_out, "cpu"):
                masks_out = masks_out.cpu()
            masks_out = torch.as_tensor(masks_out, dtype=torch.float32)
            # (N, H, W) → 가장 높은 score mask 선택
            scores = output.get("scores", None)
            if scores is not None:
                if hasattr(scores, "cpu"):
                    scores = scores.cpu()
                scores = torch.as_tensor(scores).flatten()
                best = scores.argmax().item()
                logit = masks_out[best:best+1]   # (1, H, W)
            else:
                logit = masks_out[:1]
        else:
            H, W = images.shape[2], images.shape[3]
            logit = torch.zeros(1, H, W)

        # SAM3 출력이 이미 sigmoid된 경우 logit 변환
        logit = logit.squeeze(0)   # (H, W)
        all_logits.append(logit)

    return torch.stack(all_logits, dim=0).unsqueeze(1).to(device)  # (B, 1, H, W)


def train_one_epoch(model, processor, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        gt_masks = batch["mask"].to(device).float()   # (B, 1, H, W)

        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            logits = forward_batch(model, processor, batch, device)   # (B, 1, H, W)
            loss   = criterion(logits, gt_masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if (step + 1) % 10 == 0:
            logger.info(f"  step {step+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, processor, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    for batch in loader:
        gt_masks = batch["mask"].to(device).float()
        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            logits = forward_batch(model, processor, batch, device)
            loss   = criterion(logits, gt_masks)

        total_loss += loss.item()
        pred_masks = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        gt_np      = gt_masks.cpu().numpy()
        for p, g in zip(pred_masks, gt_np):
            total_iou += compute_iou(p[0], g[0])

    n = max(len(loader), 1)
    return total_loss / n, total_iou / max(len(loader.dataset), 1)


def save_checkpoint(model, optimizer, epoch, val_loss, val_iou, path: str):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss":  val_loss,
        "val_iou":   val_iou,
    }, path)
    logger.info(f"  체크포인트 저장: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SAM3 파인튜닝")

    # 데이터
    parser.add_argument("--data-dir",   required=True,
                        help="데이터 루트 폴더 (images/train, masks/train 등 포함)")
    parser.add_argument("--format",     choices=["folder", "coco"], default="folder",
                        help="데이터셋 포맷 (folder: 이미지+마스크 폴더, coco: COCO JSON)")
    parser.add_argument("--classes",    nargs="+", default=DEFAULT_CLASSES,
                        help="학습할 클래스 텍스트 프롬프트 목록")
    parser.add_argument("--class-name", default="object",
                        help="folder 포맷 시 단일 클래스 이름")

    # 모델
    parser.add_argument("--model",         default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sam3-path",     default=str(DEFAULT_SAM3_PATH),
                        help="SAM3 라이브러리 경로")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="이미지 인코더 동결 (마스크 디코더만 학습)")
    parser.add_argument("--device",        default="cuda",
                        choices=["cuda", "cpu"])

    # 학습
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--img-size",   type=int,   default=DEFAULT_IMG_SIZE)
    parser.add_argument("--bce-weight", type=float, default=0.5,
                        help="BCE 손실 가중치 (나머지는 Dice)")
    parser.add_argument("--num-workers", type=int,  default=4)

    # 체크포인트
    parser.add_argument("--ckpt-dir",  default=DEFAULT_CKPT_DIR)
    parser.add_argument("--resume",    default=None,
                        help="재시작할 체크포인트 경로")
    parser.add_argument("--save-every", type=int, default=5,
                        help="N 에폭마다 체크포인트 저장")

    args = parser.parse_args()

    # ── 디바이스 확인 ─────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 사용 불가 → CPU로 전환")
        args.device = "cpu"
    logger.info(f"디바이스: {args.device}")

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if args.format == "coco":
        train_ds = COCODataset(
            img_dir  = str(data_dir / "images/train"),
            ann_file = str(data_dir / "annotations/train.json"),
            img_size = args.img_size,
            classes  = args.classes,
        )
        val_ds = COCODataset(
            img_dir  = str(data_dir / "images/val"),
            ann_file = str(data_dir / "annotations/val.json"),
            img_size = args.img_size,
            classes  = args.classes,
        )
    else:
        train_ds = FolderDataset(
            img_dir    = str(data_dir / "images/train"),
            mask_dir   = str(data_dir / "masks/train"),
            img_size   = args.img_size,
            class_name = args.class_name,
        )
        val_ds = FolderDataset(
            img_dir    = str(data_dir / "images/val"),
            mask_dir   = str(data_dir / "masks/val"),
            img_size   = args.img_size,
            class_name = args.class_name,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=(args.device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(args.device == "cuda"))

    # ── 모델 ──────────────────────────────────────────────────────────────
    model, processor = build_model(
        args.model, args.sam3_path, args.device, args.freeze_encoder
    )

    # ── 옵티마이저 / 스케줄러 / 손실 ──────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = SegLoss(bce_weight=args.bce_weight)
    scaler    = torch.cuda.amp.GradScaler(enabled=(args.device != "cpu"))

    # ── 재시작 ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_iou    = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt.get("val_iou", 0.0)
        logger.info(f"체크포인트 재개: epoch={start_epoch}, best_iou={best_iou:.4f}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"학습 시작: {args.epochs}에폭 | lr={args.lr} | bs={args.batch_size}")
    logger.info(f"train={len(train_ds)}개 | val={len(val_ds)}개")
    logger.info(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        logger.info(f"[Epoch {epoch}/{args.epochs}]")

        train_loss = train_one_epoch(
            model, processor, train_loader, optimizer, criterion, scaler, args.device
        )
        val_loss, val_iou = evaluate(
            model, processor, val_loader, criterion, args.device
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  val_IoU={val_iou:.4f}  lr={lr_now:.2e}  ({elapsed:.1f}s)"
        )

        # 최고 IoU 모델 저장
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch, val_loss, val_iou,
                            str(ckpt_dir / "best.pth"))
            logger.info(f"  ★ 최고 IoU 갱신: {best_iou:.4f}")

        # 주기적 저장
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, val_iou,
                            str(ckpt_dir / f"epoch_{epoch:03d}.pth"))

        # 마지막 체크포인트
        save_checkpoint(model, optimizer, epoch, val_loss, val_iou,
                        str(ckpt_dir / "last.pth"))

    logger.info(f"\n학습 완료. 최고 val IoU: {best_iou:.4f}")
    logger.info(f"최고 모델: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
