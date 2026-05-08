"""
predict.py
──────────────────────────────────────────────────────
Runs the trained EfficientNet-B0 on unlabelled test images
and writes a submission CSV: image_id, target
──────────────────────────────────────────────────────
Usage:
    python predict.py --model assets/model.pt
                      --test-dir C:/Users/mdhia/OneDrive/Bureau/dataset/test_images
                      --out submission.csv
"""

import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default="assets/model.pt")
parser.add_argument("--test-dir",  default="C:/Users/mdhia/OneDrive/Bureau/dataset/test_images")
parser.add_argument("--cache-dir", default="C:/Users/mdhia/OneDrive/Bureau/dataset/dataset_cropped_test/")
parser.add_argument("--out",       default="submission.csv")
parser.add_argument("--threshold", type=float, default=0.5,  help="Use the threshold printed at end of training")
parser.add_argument("--batch",     type=int,   default=64)
parser.add_argument("--workers",   type=int,   default=4)
parser.add_argument("--img-size",  type=int,   default=260)
args = parser.parse_args()

try:
    import torchvision.models as models
    import torch.nn as nn
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Run: pip install torch torchvision tqdm pandas")
    sys.exit(1)


# ── Same rim crop as train_cnn.py ─────────────────────────────────────────────
def extract_rim_crop(img_bgr: np.ndarray, size: int = 260, wide_scale: float = 1.3) -> np.ndarray:
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
        # Use the wide_scale here (1.3 matches your training/preprocess default)
        margin    = int((rim_w + 10) * wide_scale)
        h, w      = gray.shape
        x1 = max(0, cx - r - margin); y1 = max(0, cy - r - margin)
        x2 = min(w, cx + r + margin); y2 = min(h, cy + r + margin)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = img_bgr
    return cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)


# ── Dataset for unlabelled images ─────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, image_paths, transform, cache_dir="", img_size=260):
        self.paths     = image_paths
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.img_size  = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # Try cache first
        if self.cache_dir:
            cached = self.cache_dir / Path(path).name
            if cached.exists():
                img = Image.open(cached).convert("RGB")
                return self.transform(img), path
        # Fall back to live crop
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        crop = extract_rim_crop(img_bgr, self.img_size, wide_scale=1.3)
        return self.transform(Image.fromarray(crop)), path
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


# ── Main ──────────────────────────────────────────────────────────────────────
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device    : {device}")
    print(f"[INFO] Model     : {args.model}")
    print(f"[INFO] Threshold : {args.threshold}")

    # Collect all image paths
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted([str(p) for p in Path(args.test_dir).iterdir()
                    if p.suffix.lower() in exts])
    if not paths:
        raise RuntimeError(f"No images found in {args.test_dir}")
    print(f"[INFO] Found {len(paths)} test images")

    # Load model
    model      = build_model().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    # Transform (same as val transform in training)
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TestDataset(paths, transform, args.cache_dir, args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    image_ids, predictions = [], []

    with torch.no_grad():
        for imgs, img_paths in tqdm(loader, desc="Predicting"):
            logits = model(imgs.to(device)).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= args.threshold).astype(int)
            for path, pred in zip(img_paths, preds):
                image_ids.append(Path(path).name)   # just the filename
                predictions.append(int(pred))

    df = pd.DataFrame({"image_id": image_ids, "target": predictions})
    df.to_csv(args.out, index=False)

    n_good  = (df["target"] == 0).sum()
    n_fault = (df["target"] == 1).sum()
    print(f"\n[DONE] Predictions saved to: {args.out}")
    print(f"       Good (0): {n_good}  |  Faulty (1): {n_fault}")


if __name__ == "__main__":
    predict()