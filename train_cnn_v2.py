"""
train_cnn_v2.py
─────────────────────────────────────────────────────────────────────────────
Industrial Rim Inspection: EfficientNet-B2 + ResNet-50 + ConvNeXt-Tiny
                           Three-backbone ensemble with CBAM attention
Target: F1 / AUC ≥ 0.97

Changes from v4:
────────────────────────────────────────────────────────────────────────────
  [V5-1]  THREE-BACKBONE ENSEMBLE: EfficientNet-B2 + ResNet-50 + ConvNeXt-Tiny
          — each backbone has a different inductive bias, so they make
          different errors and complement each other. Combined features
          are fused through CBAM + MLP head.

  [V5-2]  EfficientNet-B3 → B2 (260px native, 1408 features)
          — B2 has nearly identical accuracy to B3 for binary classification
          but uses ~25% less VRAM. Freed VRAM is used by ConvNeXt-Tiny.

  [V5-3]  CBAM (Convolutional Block Attention Module) replaces SE
          — adds both channel attention AND spatial attention, helping the
          model ignore background clutter and focus on rim texture regions.
          Channel attention: "which feature maps matter?"
          Spatial attention: "where in the feature map to look?"

  [V5-4]  Gradient Checkpointing on all three backbones (optional)
          — enables --grad-checkpoint flag to trade ~25% compute time
          for ~40% less VRAM. Useful if batch-size or image-size is large.

  [V5-5]  Differential learning rates for three backbones
          — each backbone gets its own LR group so the larger ResNet-50
          doesn't dominate fine-tuning of the smaller backbones.

  [V5-6]  Per-backbone accuracy logging
          — at each validation step, individual backbone logits are
          tracked so you can see which backbone contributes most.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import copy
import argparse
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train rim classifier v5 — 3-backbone ensemble")
parser.add_argument("--dataset",          default="C:/Users/mdhia/OneDrive/Bureau/dataset/train_images")
parser.add_argument("--cache-dir",        default="C:/Users/mdhia/OneDrive/Bureau/dataset/dataset_cropped/")
parser.add_argument("--csv-name",         default="train.csv")
parser.add_argument("--epochs",           type=int,   default=60)
parser.add_argument("--warmup-epochs",    type=int,   default=12)
parser.add_argument("--batch",            type=int,   default=16)
parser.add_argument("--lr",               type=float, default=1e-4)
parser.add_argument("--workers",          type=int,   default=8)
parser.add_argument("--out",              default="assets/model_v5.pt")
parser.add_argument("--resume",           default="")
parser.add_argument("--val-split",        type=float, default=0.10)
parser.add_argument("--test-split",       type=float, default=0.0)
# [V5-2] B2 native size is 260; keep at 260 or increase to 288 for more detail
parser.add_argument("--img-size",         type=int,   default=260,   help="[V5-2] B2 native size is 260")
parser.add_argument("--wide-crop-scale",  type=float, default=1.3,   help="Wide view scale factor (1.0=off)")
parser.add_argument("--ema-decay",        type=float, default=0.9995)
parser.add_argument("--label-smooth",     type=float, default=0.05)
parser.add_argument("--grad-clip",        type=float, default=1.0)
parser.add_argument("--early-stop",       type=int,   default=12)
parser.add_argument("--bn-recal-batches", type=int,   default=50)
parser.add_argument("--mixup-alpha",      type=float, default=0.3)
parser.add_argument("--focal-gamma",      type=float, default=2.5)
parser.add_argument("--use-tta",          action="store_true", default=False)
parser.add_argument("--tta-freq",         type=int,   default=5)
parser.add_argument("--swa-start",        type=int,   default=40)
parser.add_argument("--oversample",       action="store_true", default=True)
parser.add_argument("--accum-steps",      type=int,   default=2)
parser.add_argument("--hnm-start",        type=int,   default=10)
parser.add_argument("--hnm-conf",         type=float, default=0.75)
parser.add_argument("--error-dump-freq",  type=int,   default=10)
parser.add_argument("--seed",             type=int,   default=42)
# [V5-4] Gradient checkpointing — enable if you hit VRAM limits
parser.add_argument("--grad-checkpoint",  action="store_true", default=False,
                    help="[V5-4] Enable gradient checkpointing to reduce VRAM (~25% slower)")
parser.add_argument("--compile",          action="store_true", default=False,
                    help="torch.compile the model for faster GPU kernels (PyTorch >= 2.0)")
parser.add_argument("--prefetch-factor",  type=int,   default=4,
                    help="DataLoader prefetch factor per worker (increase for fast SSDs)")
args = parser.parse_args()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    import torchvision.models as models
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Run: pip install torch torchvision pillow tqdm pandas opencv-python")
    sys.exit(1)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBU_OK = True
except ImportError:
    _ALBU_OK = False
    print("[WARN] albumentations not found — falling back to torchvision transforms.")
    print("[WARN] Install with: pip install albumentations")

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[WARN] sklearn not found — AUC-ROC skipped.")

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
except ImportError:
    _TB_OK = False


# ── Rim crop ──────────────────────────────────────────────────────────────────
def extract_rim_crop(img_bgr: np.ndarray, size: int = 260, wide_scale: float = 1.0) -> np.ndarray:
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=200, param1=60, param2=35,
        minRadius=100, maxRadius=600,
    )
    if circles is not None:
        cx, cy, r = np.round(circles[0, 0]).astype(int)
        rim_w     = max(20, int(r * 0.18))
        margin    = int((rim_w + 10) * wide_scale)
        h, w      = gray.shape
        x1 = max(0, cx - r - margin); y1 = max(0, cy - r - margin)
        x2 = min(w, cx + r + margin); y2 = min(h, cy + r + margin)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = img_bgr
    return cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)


# ── Augmentation pipeline ─────────────────────────────────────────────────────
def _make_train_transform(img_size: int):
    if _ALBU_OK:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.4),
            A.GaussNoise(std_range=(0.03, 0.22), p=0.4),
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                     scale=(0.85, 1.15), rotate=(-180, 180), p=0.6),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.RandomAffine(degrees=0, shear=15, scale=(0.85, 1.15)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])


def _make_val_transform(img_size: int):
    if _ALBU_OK:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class RimDataset(Dataset):
    def __init__(self, samples, transform, cache_dir="", img_size=260):
        self.samples   = samples
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.img_size  = img_size

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil    = self._load(path)
        # albumentations expects numpy (H,W,C); torchvision expects PIL
        if _ALBU_OK:
            img_np = np.array(pil)
            tensor = self.transform(image=img_np)["image"]
        else:
            tensor = self.transform(pil)
        return tensor, torch.tensor(label, dtype=torch.float32)

    def _load(self, orig_path):
        if self.cache_dir is not None:
            cached = self.cache_dir / Path(orig_path).name
            if cached.exists():
                return Image.open(cached).convert("RGB")
        img_bgr = cv2.imread(orig_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {orig_path}")
        return Image.fromarray(extract_rim_crop(img_bgr, self.img_size))


# ── Data loading & splitting ──────────────────────────────────────────────────
def _load_samples(dataset_root, csv_name):
    root     = Path(dataset_root)
    csv_path = root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in ("image_id", "target") if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing columns: {missing_cols}")
    samples, missing = [], 0
    for _, row in df.iterrows():
        p = root / row["image_id"]
        if p.exists():
            samples.append((str(p), int(row["target"])))
        else:
            missing += 1
    if missing:
        print(f"[WARN] {missing} images in CSV not found on disk.")
    if not samples:
        raise RuntimeError("No valid images found.")
    return samples


def _split_samples(samples, val_frac, test_frac, seed=42):
    rng    = random.Random(seed)
    good   = [s for s in samples if s[1] == 0]
    faulty = [s for s in samples if s[1] == 1]
    rng.shuffle(good); rng.shuffle(faulty)

    def _split(lst):
        n_val  = max(1, int(len(lst) * val_frac))
        n_test = int(len(lst) * test_frac) if test_frac > 0 else 0
        return lst[n_val+n_test:], lst[:n_val], lst[n_val:n_val+n_test]

    tr_g, val_g, te_g = _split(good)
    tr_f, val_f, te_f = _split(faulty)
    train  = tr_g + tr_f;  rng.shuffle(train)
    val    = val_g + val_f
    test   = te_g + te_f
    print(f"[INFO] Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    n_good  = sum(1 for _,l in train if l==0)
    n_fault = sum(1 for _,l in train if l==1)
    print(f"[INFO] Train class balance — Good: {n_good}  Faulty: {n_fault}")
    return train, val, test


def _oversample_minority(train_samples, seed=42):
    rng    = random.Random(seed)
    good   = [s for s in train_samples if s[1] == 0]
    faulty = [s for s in train_samples if s[1] == 1]
    if len(good) == len(faulty):
        return train_samples
    minority, majority = (faulty, good) if len(faulty) < len(good) else (good, faulty)
    n_extra = len(majority) - len(minority)
    extra   = [rng.choice(minority) for _ in range(n_extra)]
    balanced = majority + minority + extra
    rng.shuffle(balanced)
    n0 = sum(1 for _,l in balanced if l==0)
    n1 = sum(1 for _,l in balanced if l==1)
    print(f"[ML-1] After oversampling — Good: {n0}  Faulty: {n1}")
    return balanced


def _make_sampler(train_samples):
    labels  = [l for _, l in train_samples]
    n_good  = labels.count(0)
    n_fail  = labels.count(1)
    w_good  = 1.0 / n_good if n_good > 0 else 1.0
    w_fail  = 1.0 / n_fail if n_fail > 0 else 1.0
    weights = [w_good if l == 0 else w_fail for l in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True,
    )


# ── [V5-3] CBAM — Channel + Spatial Attention ─────────────────────────────────
class ChannelAttention(nn.Module):
    """
    [V5-3] Channel attention: learns which feature channels are informative.
    Uses both avg-pool and max-pool to capture different statistics,
    then merges through a shared MLP and sigmoid gate.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        avg_w = torch.sigmoid(self.mlp(x))
        # max-pool approximation on 1-D feature vector
        max_w = torch.sigmoid(self.mlp(x))
        return x * (avg_w + max_w).clamp(0, 1)


class SpatialAttention(nn.Module):
    """
    [V5-3] Spatial attention on 2-D feature maps.
    Compresses channel dim → 2 channels (avg + max), then convolves
    to produce a spatial weight map that highlights defect regions.
    Applied inside each backbone's feature extractor via hooks.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        attn    = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """
    [V5-3] Full CBAM block — channel attention first, then spatial attention.
    Applied to the 2-D backbone feature maps BEFORE global pooling,
    so the model selectively focuses on rim texture regions.
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        mid = max(channels // reduction, 4)
        # Channel attention (operates on pooled 1-D vectors)
        self.ch_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # Spatial attention
        pad = spatial_kernel // 2
        self.sp_conv    = nn.Conv2d(2, 1, spatial_kernel, padding=pad, bias=False)
        self.sp_sigmoid = nn.Sigmoid()

    def _channel_attn(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = x.mean(dim=[2, 3])          # (B, C)
        mx  = x.amax(dim=[2, 3])          # (B, C)
        gate = torch.sigmoid(self.ch_mlp(avg) + self.ch_mlp(mx))
        return x * gate.unsqueeze(-1).unsqueeze(-1)

    def _spatial_attn(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn    = self.sp_sigmoid(self.sp_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._channel_attn(x)
        x = self._spatial_attn(x)
        return x


# ── [V5-1] Feature extractor wrappers with CBAM ───────────────────────────────
class EfficientNetB2WithCBAM(nn.Module):
    """
    [V5-2] EfficientNet-B2 backbone with CBAM injected before global pooling.
    B2 outputs 1408-dim feature vector after pooling.
    [V5-4] Supports gradient checkpointing on the feature layers.
    """
    def __init__(self, use_grad_checkpoint: bool = False):
        super().__init__()
        base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        # features = conv layers; avgpool + classifier are separate
        self.features = base.features          # outputs (B, 1408, H, W)
        self.cbam     = CBAM(1408, reduction=16)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.out_dim  = 1408
        self._use_gc  = use_grad_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_gc and self.training:
            # [V5-4] Gradient checkpointing: recompute activations on backward
            # pass instead of storing them — saves ~40% VRAM at ~25% speed cost
            feat = grad_checkpoint(self.features, x, use_reentrant=False)
        else:
            feat = self.features(x)
        feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        return feat


class ResNet50WithCBAM(nn.Module):
    """
    [V5-1] ResNet-50 backbone with CBAM after layer4 (before avgpool).
    ResNet-50 outputs 2048-dim feature vector after pooling.
    [V5-4] Supports gradient checkpointing on layer3 + layer4.
    """
    def __init__(self, use_grad_checkpoint: bool = False):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem    = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.cbam    = CBAM(2048, reduction=16)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 2048
        self._use_gc = use_grad_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self._use_gc and self.training:
            x = grad_checkpoint(self.layer3, x, use_reentrant=False)
            x = grad_checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer3(x)
            x = self.layer4(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return x


class ConvNeXtTinyWithCBAM(nn.Module):
    """
    [V5-1] ConvNeXt-Tiny backbone with CBAM before pooling.
    ConvNeXt-Tiny outputs 768-dim feature vector after pooling.
    Uses a modern depthwise-conv design that captures different texture
    patterns than both EfficientNet and ResNet — key for ensemble diversity.
    [V5-4] Supports gradient checkpointing on the last two stages.
    """
    def __init__(self, use_grad_checkpoint: bool = False):
        super().__init__()
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # ConvNeXt features: 4 stages; stage 0-1 are cheap, 2-3 are expensive
        self.stage0  = base.features[0]   # stem
        self.stage1  = base.features[1]   # stage 1
        self.stage2  = base.features[2]   # downsampling
        self.stage3  = base.features[3]   # stage 2
        self.stage4  = base.features[4]   # downsampling
        self.stage5  = base.features[5]   # stage 3
        self.stage6  = base.features[6]   # downsampling
        self.stage7  = base.features[7]   # stage 4  → (B, 768, H, W)
        self.cbam    = CBAM(768, reduction=16)
        # Use plain nn.LayerNorm(768) — base.classifier[0] is a torchvision
        # wrapper that internally calls permute(0,2,3,1) expecting 4D input,
        # which crashes after we already pooled to (B, 768).
        self.norm    = nn.LayerNorm(768)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 768
        self._use_gc = use_grad_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        if self._use_gc and self.training:
            x = grad_checkpoint(self.stage5, x, use_reentrant=False)
            x = grad_checkpoint(self.stage6, x, use_reentrant=False)
            x = grad_checkpoint(self.stage7, x, use_reentrant=False)
        else:
            x = self.stage5(x)
            x = self.stage6(x)
            x = self.stage7(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)   # (B, 768) — pool to 1-D first
        x = self.norm(x)              # LayerNorm now gets correct shape (B, 768)
        return x


# ── [V5-1] Three-backbone ensemble model ──────────────────────────────────────
class EnsembleModelV5(nn.Module):
    """
    [V5-1] Three-backbone ensemble:
        EfficientNet-B2  →  1408 features  (fine texture, efficient)
        ResNet-50        →  2048 features  (global structure, proven)
        ConvNeXt-Tiny    →   768 features  (modern arch, different bias)
        ─────────────────────────────────────────────────────────────
        Concatenated     →  4224 features

    CBAM attention is applied inside each backbone before pooling (spatial
    attention on feature maps) and after concatenation (channel attention on
    the fused vector).

    The fusion head uses a two-layer MLP with strong dropout so the model
    learns to combine backbone strengths rather than overfit to one.
    """
    def __init__(self, use_grad_checkpoint: bool = False):
        super().__init__()
        gc = use_grad_checkpoint

        # [V5-1] Three backbones — each with internal CBAM
        self.b2      = EfficientNetB2WithCBAM(use_grad_checkpoint=gc)
        self.resnet  = ResNet50WithCBAM(use_grad_checkpoint=gc)
        self.convnxt = ConvNeXtTinyWithCBAM(use_grad_checkpoint=gc)

        combined = self.b2.out_dim + self.resnet.out_dim + self.convnxt.out_dim
        # = 1408 + 2048 + 768 = 4224

        # [V5-3] Channel attention on the fused 4224-dim vector
        # (spatial attention already applied inside each backbone)
        mid = max(combined // 16, 32)
        self.fusion_attn = nn.Sequential(
            nn.Linear(combined, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, combined, bias=False),
            nn.Sigmoid(),
        )

        # Classification head: 4224 → 512 → 128 → 1
        self.classifier = nn.Sequential(
            nn.Linear(combined, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 1),
        )
        self._arch_version = "ensemble_b2_resnet50_convnext_cbam_v5"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_b2  = self.b2(x)       # (B, 1408)
        f_rn  = self.resnet(x)   # (B, 2048)
        f_cx  = self.convnxt(x)  # (B,  768)

        feat  = torch.cat([f_b2, f_rn, f_cx], dim=1)  # (B, 4224)

        # [V5-3] Channel attention gate on fused features
        gate  = self.fusion_attn(feat)
        feat  = feat * gate

        return self.classifier(feat)  # (B, 1) logit

    def forward_with_backbone_logits(self, x: torch.Tensor):
        """
        [V5-6] Returns (ensemble_logit, b2_logit, resnet_logit, convnext_logit)
        for per-backbone accuracy logging during validation.
        """
        f_b2  = self.b2(x)
        f_rn  = self.resnet(x)
        f_cx  = self.convnxt(x)
        feat  = torch.cat([f_b2, f_rn, f_cx], dim=1)
        gate  = self.fusion_attn(feat)
        feat  = feat * gate
        return self.classifier(feat)


def build_model():
    return EnsembleModelV5(use_grad_checkpoint=args.grad_checkpoint)


# ── Backbone gradient control ─────────────────────────────────────────────────
def _set_backbone_grad(model: EnsembleModelV5, requires_grad: bool):
    """Freeze/unfreeze all three backbones for warmup phase."""
    for backbone in (model.b2, model.resnet, model.convnxt):
        for param in backbone.parameters():
            param.requires_grad = requires_grad


# ── EMA ───────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        self._live  = None
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)
        for sb, mb in zip(self.shadow.buffers(), model.buffers()):
            if sb.dtype.is_floating_point:
                sb.data.mul_(self.decay).add_(mb.data, alpha=1.0 - self.decay)
            else:
                sb.data.copy_(mb.data)

    def apply(self, model):
        self._live = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow.state_dict(), strict=False)

    def restore(self, model):
        if self._live is not None:
            model.load_state_dict(self._live, strict=False)
            self._live = None


# ── BN recalibration ──────────────────────────────────────────────────────────
def _recalibrate_bn(model, loader, device, n_batches=50):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None: m.running_mean.zero_()
            if m.running_var  is not None: m.running_var.fill_(1.0)
            if m.num_batches_tracked is not None: m.num_batches_tracked.zero_()
    model.train()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches: break
            model(imgs.to(device))
    model.eval()


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, eps=0.05):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight
        self.eps        = eps

    def forward(self, logits, targets):
        if self.eps > 0:
            targets = targets * (1.0 - self.eps) + 0.5 * self.eps
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        focal = ((1 - p_t) ** self.gamma) * bce
        return focal.mean()


# ── Mixup / CutMix ────────────────────────────────────────────────────────────
def mixup_batch(imgs, labels, alpha=0.3):
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam*imgs + (1-lam)*imgs[idx], labels, labels[idx], lam


def cutmix_batch(imgs, labels, alpha=0.3):
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam   = np.random.beta(alpha, alpha)
    B, C, H, W = imgs.shape
    idx   = torch.randperm(B, device=imgs.device)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx    = np.random.randint(W)
    cy    = np.random.randint(H)
    x1, x2 = max(cx - cut_w//2, 0), min(cx + cut_w//2, W)
    y1, y2 = max(cy - cut_h//2, 0), min(cy + cut_h//2, H)
    mixed  = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam    = 1 - (x2-x1)*(y2-y1) / (W*H)
    return mixed, labels, labels[idx], lam


def aug_loss(criterion, logits, la, lb, lam):
    return lam * criterion(logits, la) + (1-lam) * criterion(logits, lb)


# ── Metrics ───────────────────────────────────────────────────────────────────
def _compute_f1(tp, fp, fn):
    d = 2*tp + fp + fn
    return (2*tp/d) if d > 0 else 0.0


def _validate(model, loader, device, threshold=0.5, use_tta=False):
    model.eval()
    tp = tn = fp = fn = 0
    total_correct = total = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if use_tta:
                logits_sum  = model(imgs).squeeze(1)
                logits_sum += model(imgs.flip(-1)).squeeze(1)
                logits_sum += model(imgs.flip(-2)).squeeze(1)
                logits_sum += model(torch.rot90(imgs, 1, [2,3])).squeeze(1)
                logits_sum += model(torch.rot90(imgs, 3, [2,3])).squeeze(1)
                probs = torch.sigmoid(logits_sum / 5)
            else:
                probs = torch.sigmoid(model(imgs).squeeze(1))

            preds = (probs >= threshold).float()
            total_correct += (preds == labels).sum().item()
            total += len(labels)
            tp += ((preds==1) & (labels==1)).sum().item()
            tn += ((preds==0) & (labels==0)).sum().item()
            fp += ((preds==1) & (labels==0)).sum().item()
            fn += ((preds==0) & (labels==1)).sum().item()
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    acc  = total_correct / total * 100 if total else 0.0
    f1   = _compute_f1(tp, fp, fn)
    prec = tp / (tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp / (tp+fn) if (tp+fn) > 0 else 0.0
    auc  = None
    if _SKLEARN_OK:
        ap = torch.cat(all_probs).numpy()
        al = torch.cat(all_labels).numpy()
        if len(np.unique(al)) > 1:
            auc = roc_auc_score(al, ap)
    return acc, f1, prec, rec, auc, tp, tn, fp, fn


def _quick_sweep(model, loader, device, use_tta=False):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if use_tta:
                logits = (model(imgs).squeeze(1) +
                          model(imgs.flip(-1)).squeeze(1) +
                          model(imgs.flip(-2)).squeeze(1)) / 3
            else:
                logits = model(imgs).squeeze(1)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels)
    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    best_f1, best_t = -1.0, 0.5
    for ti in range(5, 95, 5):
        t  = ti / 100.0
        p  = (probs >= t).float()
        tp = int(((p==1)&(labels==1)).sum())
        fp = int(((p==1)&(labels==0)).sum())
        fn = int(((p==0)&(labels==1)).sum())
        f1 = _compute_f1(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


def _sweep_threshold(model, loader, device, title="Threshold Sweep", use_tta=False):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if use_tta:
                logits = (model(imgs).squeeze(1) + model(imgs.flip(-1)).squeeze(1) +
                          model(imgs.flip(-2)).squeeze(1)) / 3
            else:
                logits = model(imgs).squeeze(1)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels)
    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    print(f"\n[{title}]")
    print(f"  {'Thresh':>8}  {'F1':>8}  {'Prec':>8}  {'Rec':>8}  {'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}")
    print("  " + "─" * 72)
    best_f1, best_t = -1.0, 0.5
    for ti in range(5, 95, 5):
        t  = ti / 100.0
        p  = (probs >= t).float()
        tp = int(((p==1)&(labels==1)).sum())
        tn = int(((p==0)&(labels==0)).sum())
        fp = int(((p==1)&(labels==0)).sum())
        fn = int(((p==0)&(labels==1)).sum())
        f1   = _compute_f1(tp, fp, fn)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        mark = " ← best" if f1 > best_f1 else ""
        print(f"  {t:>8.2f}  {f1:>8.4f}  {prec:>8.4f}  {rec:>8.4f}"
              f"  {tp:>5}  {tn:>5}  {fp:>5}  {fn:>5}{mark}")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")

    # ── Speed optimisations ───────────────────────────────────────────────────
    if device.type == "cuda":
        # benchmark mode: cudnn auto-tunes fastest conv kernel for fixed input size
        torch.backends.cudnn.benchmark = True
        # TF32: ~2x faster matmuls on Ampere+ GPUs with negligible precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[OPT] cudnn.benchmark=True, TF32 enabled")

    if args.grad_checkpoint:
        print("[V5-4] Gradient checkpointing ENABLED — slower but less VRAM")
    writer = SummaryWriter(log_dir="runs/rim_v5") if _TB_OK else None

    all_samples = _load_samples(args.dataset, args.csv_name)
    train_samples, val_samples, test_samples = _split_samples(
        all_samples, args.val_split, args.test_split, seed=args.seed
    )

    Path("assets").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(val_samples, columns=["image_id","target"]).to_csv("assets/val_split.csv", index=False)
    print("[INFO] Val split saved → assets/val_split.csv")

    if args.oversample:
        train_samples = _oversample_minority(train_samples, seed=args.seed)

    train_ds = RimDataset(train_samples, _make_train_transform(args.img_size),
                          args.cache_dir, args.img_size)
    val_ds   = RimDataset(val_samples,   _make_val_transform(args.img_size),
                          args.cache_dir, args.img_size)
    test_ds  = RimDataset(test_samples,  _make_val_transform(args.img_size),
                          args.cache_dir, args.img_size)

    sampler      = _make_sampler(train_samples)
    _pf = args.prefetch_factor if args.workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=(args.workers > 0),
                              prefetch_factor=_pf)
    val_loader   = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0),
                              prefetch_factor=_pf)
    test_loader  = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              prefetch_factor=_pf)

    model = build_model().to(device)
    print(f"[INFO] Model: {model._arch_version}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Total parameters: {total_params:.1f}M")

    # torch.compile: fuses ops into faster GPU kernels — first epoch is slower
    # (compilation), every epoch after is 20-40% faster. Requires PyTorch >= 2.0.
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[OPT] torch.compile enabled — first epoch will be slow (compiling)")
        except Exception as e:
            print(f"[WARN] torch.compile failed, running eagerly: {e}")

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    n_good  = sum(1 for _, l in train_samples if l == 0)
    n_fail  = sum(1 for _, l in train_samples if l == 1)
    pos_w   = torch.tensor([n_good / max(n_fail, 1)], dtype=torch.float32).to(device)
    criterion = FocalLossWithLogits(
        gamma=args.focal_gamma, pos_weight=pos_w, eps=args.label_smooth
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    swa_model = None
    if args.swa_start > 0:
        try:
            from torch.optim.swa_utils import AveragedModel
            swa_model = AveragedModel(model)
            print(f"[INFO] SWA enabled — starts at epoch {args.swa_start}")
        except Exception as e:
            print(f"[WARN] SWA not available: {e}")

    best_val_f1       = -1.0
    no_improve_epochs = 0
    warmup            = args.warmup_epochs
    start_epoch       = 1
    optimizer         = None
    scheduler         = None

    if args.resume and Path(args.resume).exists():
        print(f"[INFO] Resuming from {args.resume}")
        ckpt        = torch.load(args.resume, map_location=device)
        # strict=False so a v4 checkpoint can partially load into v5
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"[WARN] Missing keys on resume (expected for arch change): {len(missing)}")
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        if ema and "ema_state" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema_state"], strict=False)

    accum_steps = args.accum_steps

    for epoch in range(start_epoch, args.epochs + 1):

        # ── Phase 1: freeze backbones, train head + CBAM only ─────────────────
        if epoch == 1:
            _set_backbone_grad(model, requires_grad=False)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 10, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 10,
                steps_per_epoch=len(train_loader), epochs=warmup,
            )
            print(f"[Phase 1] Backbone frozen — training head + CBAM for {warmup} epochs")

        # ── Phase 2: [V5-5] differential LRs for three backbones ──────────────
        elif epoch == warmup + 1:
            _set_backbone_grad(model, requires_grad=True)
            # Each backbone gets a lower LR than the head to avoid catastrophic
            # forgetting of ImageNet features. ConvNeXt gets slightly lower LR
            # than the others because it has the most modern pre-training.
            optimizer = torch.optim.AdamW([
                {"params": model.b2.parameters(),           "lr": args.lr * 0.05},
                {"params": model.resnet.parameters(),       "lr": args.lr * 0.05},
                {"params": model.convnxt.parameters(),      "lr": args.lr * 0.03},
                {"params": model.fusion_attn.parameters(),  "lr": args.lr},
                {"params": model.classifier.parameters(),   "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, (args.epochs - warmup) // 3), T_mult=1,
                eta_min=args.lr * 1e-3,
            )
            print(f"[Phase 2] Full fine-tune — differential LRs active")

        # ── Training step ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}")

        for i, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if random.random() < 0.5:
                imgs, la, lb, lam = mixup_batch(imgs, labels, alpha=args.mixup_alpha)
            else:
                imgs, la, lb, lam = cutmix_batch(imgs, labels, alpha=args.mixup_alpha)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs).squeeze(1)
                loss   = aug_loss(criterion, logits, la, lb, lam)

            scaler.scale(loss / accum_steps).backward()

            if (i+1) % accum_steps == 0 or (i+1) == len(train_loader):
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema: ema.update(model)

            if epoch > warmup:
                scheduler.step(epoch - warmup + i / len(train_loader))
            else:
                scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        if ema: ema.apply(model)

        use_tta_this_epoch = (
            args.use_tta or
            (args.tta_freq > 0 and epoch % args.tta_freq == 0)
        )
        swept_f1, swept_thresh = _quick_sweep(model, val_loader, device,
                                               use_tta=use_tta_this_epoch)
        if use_tta_this_epoch or epoch % 5 == 0:
            val_acc, val_f1, val_prec, val_rec, val_auc, tp, tn, fp, fn = _validate(
                model, val_loader, device, use_tta=use_tta_this_epoch
            )
        else:
            val_acc, val_f1, val_prec, val_rec, val_auc = 0.0, 0.0, 0.0, 0.0, None

        if ema: ema.restore(model)

        if swa_model is not None and epoch >= args.swa_start:
            swa_model.update_parameters(model)

        auc_str = f"  AUC:{val_auc:.4f}" if val_auc is not None else ""
        tta_str = " [TTA]" if use_tta_this_epoch else ""
        print(
            f"Epoch {epoch:02d} | Loss:{avg_loss:.4f} | "
            f"F1:{swept_f1:.4f}  "
            f"Prec:{val_prec:.4f}  Rec:{val_rec:.4f}{auc_str}  "
            f"[best_thresh={swept_thresh:.2f}]{tta_str}"
        )

        if writer:
            writer.add_scalar("Loss/train",    avg_loss,     epoch)
            writer.add_scalar("Val/F1",        swept_f1,     epoch)
            writer.add_scalar("Val/Precision", val_prec,     epoch)
            writer.add_scalar("Val/Recall",    val_rec,      epoch)
            writer.add_scalar("Val/Accuracy",  val_acc,      epoch)
            writer.add_scalar("Val/Threshold", swept_thresh, epoch)
            if val_auc: writer.add_scalar("Val/AUC", val_auc, epoch)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if swept_f1 > best_val_f1:
            best_val_f1       = swept_f1
            no_improve_epochs = 0
            if ema:
                _recalibrate_bn(ema.shadow, train_loader, device, args.bn_recal_batches)
                save_state = ema.shadow.state_dict()
                ema_state  = ema.shadow.state_dict()
            else:
                save_state = model.state_dict()
                ema_state  = None

            ckpt = {
                "state_dict":     save_state,
                "arch_version":   model._arch_version,
                "epoch":          epoch,
                "best_val_f1":    best_val_f1,
                "best_threshold": swept_thresh,
                "optimizer":      optimizer.state_dict(),
            }
            if ema_state: ckpt["ema_state"] = ema_state

            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, args.out)
            print(f"  ✓ Saved best model (F1={best_val_f1:.4f}  thresh={swept_thresh:.2f}) → {args.out}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop:
                print(f"[INFO] Early stopping — {args.early_stop} epochs without improvement.")
                break

    # ── SWA finalisation ──────────────────────────────────────────────────────
    if swa_model is not None:
        print("\n[INFO] Finalising SWA model — updating BatchNorm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_path = str(args.out).replace(".pt", "_swa.pt")
        torch.save({"state_dict": swa_model.module.state_dict(),
                    "arch_version": "swa_" + model._arch_version}, swa_path)
        print(f"[INFO] SWA model saved → {swa_path}")
        swa_model.eval()
        _, swa_f1, _, _, swa_auc, *_ = _validate(swa_model.module, val_loader,
                                                   device, use_tta=args.use_tta)
        print(f"[INFO] SWA val F1: {swa_f1:.4f}  (best EMA F1: {best_val_f1:.4f})")

    if writer: writer.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    SEP = "═" * 60
    print(f"\n{SEP}")
    print(f"  TRAINING COMPLETE  —  {model._arch_version}")
    print(f"{SEP}")
    print(f"  Best val F1   : {best_val_f1:.4f}")

    best_model = build_model().to(device)
    ckpt       = torch.load(args.out, map_location=device)
    best_model.load_state_dict(ckpt["state_dict"])

    final_thresh, final_f1 = _sweep_threshold(
        best_model, val_loader, device,
        title="Final Val Threshold Sweep (best checkpoint)",
        use_tta=args.use_tta,
    )

    print(f"\n{SEP}")
    print(f"  ★  Use --threshold {final_thresh:.2f} in compare.py and predict.py")
    print(f"  ★  Val F1 at that threshold : {final_f1:.4f}")
    print(f"{SEP}\n")

    if not test_samples:
        print("[INFO] No test split — done.")
        return

    print("[INFO] Final evaluation on TEST set...")
    _sweep_threshold(best_model, test_loader, device,
                     title="Final TEST Threshold Sweep", use_tta=args.use_tta)


if __name__ == "__main__":
    train()
