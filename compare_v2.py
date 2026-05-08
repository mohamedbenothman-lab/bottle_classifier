"""
compare.py

Evaluates the trained CNN on the validation split (same split used during
training) and prints a detailed metrics report.

Uses the same CSV + stratified split as train_cnn.py so the val set is
identical — giving you a reliable estimate of real-world performance.

Usage
  python compare.py
  python compare.py --threshold 0.40
  python compare.py --dataset path/to/train_images --csv-name train.csv --threshold 0.40

"""

import os
import sys
import argparse
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# CLI 
parser = argparse.ArgumentParser(description="Evaluate CNN on validation split")
parser.add_argument("--dataset",    default="C:/Users/mdhia/OneDrive/Bureau/dataset/train_images",
                    help="Directory with images and CSV")
parser.add_argument("--cache-dir",  default="C:/Users/mdhia/OneDrive/Bureau/dataset/dataset_cropped/",
                    help="Pre-cropped image cache")
parser.add_argument("--csv-name",   default="train.csv",        help="CSV filename")
parser.add_argument("--model",      default="assets/model.pt",  help="Trained model path")
parser.add_argument("--threshold",  type=float, default=0.40,
                    help="Sigmoid threshold — use the value printed at end of training")
parser.add_argument("--val-split",  type=float, default=0.2,    help="Must match training val-split")
parser.add_argument("--img-size",   type=int,   default=260,    help="Must match training img-size")
parser.add_argument("--batch",      type=int,   default=32)
parser.add_argument("--seed",       type=int,   default=42,     help="Must match training seed")
args = parser.parse_args()


# Rim crop (identical to train_cnn.py)
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


#  Dataset (identical to train_cnn.py)
class RimDataset(Dataset):
    def __init__(self, samples, transform, cache_dir="", img_size=260):
        self.samples   = samples
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.img_size  = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil    = self._load(path)
        tensor = self.transform(pil)
        return tensor, torch.tensor(label, dtype=torch.float32), path

    def _load(self, orig_path):
        if self.cache_dir is not None:
            cached = self.cache_dir / Path(orig_path).name
            if cached.exists():
                if not hasattr(self, '_cache_confirmed'):
                    print(f"[DEBUG] Using cache: {cached}")
                    self._cache_confirmed = True
                return Image.open(cached).convert("RGB")
        print(f"[DEBUG] Cache MISS — live Hough crop: {Path(orig_path).name}")
        img_bgr = cv2.imread(orig_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {orig_path}")
        return Image.fromarray(extract_rim_crop(img_bgr, self.img_size, wide_scale=1.3))
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


# Data helpers (identical split logic to train_cnn.py) 
def load_val_samples(csv_path="assets/val_split.csv"):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return [(row["image_id"], int(row["target"])) for _, row in df.iterrows()]


#  Metrics 
def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc      = (tp + tn) / len(y_true) if y_true else 0
    prec     = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1       = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return dict(acc=acc, prec=prec, recall=recall, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


#  Main 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device    : {device}")
    print(f"[INFO] Model     : {args.model}")
    print(f"[INFO] Threshold : {args.threshold}")
    print(f"[INFO] Val split : {args.val_split}")

    #  Load val split (same as training) 
    val_samples = load_val_samples("assets/val_split.csv")

    n_good = sum(1 for _, l in val_samples if l == 0)
    n_fail = sum(1 for _, l in val_samples if l == 1)
    print(f"[INFO] Val set   : {len(val_samples)} images  |  Good: {n_good}  |  Faulty: {n_fail}")

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = RimDataset(val_samples, transform, args.cache_dir, args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    # Load model 
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at {args.model} — run train_cnn.py first.")

    model      = build_model().to(device)
    checkpoint = torch.load(args.model, map_location=device)

    # Support both raw state_dict and wrapped checkpoint
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Checkpoint loaded successfully\n")

    # Run inference 
    all_probs, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Evaluating"):
            logits = model(imgs.to(device)).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().astype(int))
            all_paths.extend(paths)
    import numpy as np
    arr = np.array(all_probs)
    print(f'\nProb stats:')
    print(f'  Min    : {arr.min():.3f}')
    print(f'  Max    : {arr.max():.3f}')
    print(f'  Mean   : {arr.mean():.3f}')
    print(f'  Median : {np.median(arr):.3f}')
    print(f'  Below 0.10 : {(arr < 0.10).sum()}')
    print(f'  0.10 - 0.50: {((arr >= 0.10) & (arr < 0.50)).sum()}')
    print(f'  Above 0.50 : {(arr >= 0.50).sum()}')
    print()

    #  Threshold sweep 
    print("\n[Threshold Sweep]")
    print(f"  {'Threshold':>10}  {'F1':>8}  {'Acc':>7}  {'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}")
    print("  " + "─" * 62)

    best_f1, best_thresh = -1.0, args.threshold
    for t_int in range(5, 96, 5):
        t      = t_int / 100.0
        preds  = [1 if p >= t else 0 for p in all_probs]
        m      = compute_metrics(all_labels, preds)
        mark   = " ← best" if m["f1"] > best_f1 else ""
        print(f"  {t:>10.2f}  {m['f1']:>8.4f}  {m['acc']:>6.1%}  "
              f"{m['tp']:>5}  {m['tn']:>5}  {m['fp']:>5}  {m['fn']:>5}{mark}")
        if m["f1"] > best_f1:
            best_f1, best_thresh = m["f1"], t

    # Final report at chosen threshold
    final_preds = [1 if p >= args.threshold else 0 for p in all_probs]
    m = compute_metrics(all_labels, final_preds)

    SEP = "═" * 50
    print(f"\n{SEP}")
    print(f"  EVALUATION REPORT  (threshold = {args.threshold})")
    print(f"{SEP}")
    print(f"  Accuracy  : {m['acc']:>8.2%}")
    print(f"  Precision : {m['prec']:>8.2%}")
    print(f"  Recall    : {m['recall']:>8.2%}")
    print(f"  F1 Score  : {m['f1']:>8.4f}")
    print(f"{'─' * 50}")
    print(f"  TP (correct faulty)  : {m['tp']:>6}")
    print(f"  TN (correct good)    : {m['tn']:>6}")
    print(f"  FP (good → faulty)   : {m['fp']:>6}  ← false alarms")
    print(f"  FN (faulty → good)   : {m['fn']:>6}  ← missed defects")
    print(f"{SEP}")
    print(f"\n  Best threshold found : {best_thresh:.2f}  (F1 = {best_f1:.4f})")
    print(f"  → Use --threshold {best_thresh:.2f} in predict.py for best results")
    print(f"{SEP}\n")

    # ── Print misclassified images ────────────────────────────────────────────
    print("[Misclassified Images]")
    errors = [(p, l, pr, prob) for p, l, pr, prob in
              zip(all_paths, all_labels, final_preds, all_probs) if l != pr]
    if not errors:
        print("  None! Perfect score on val set.")
    else:
        print(f"  {'File':<40} {'True':>5}  {'Pred':>5}  {'Prob':>6}")
        print("  " + "─" * 62)
        for path, true, pred, prob in errors[:30]:  # show first 30
            fname = Path(path).name
            true_str = "FAIL" if true == 1 else "GOOD"
            pred_str = "FAIL" if pred == 1 else "GOOD"
            print(f"  {fname:<40} {true_str:>5}  {pred_str:>5}  {prob:.3f}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")


if __name__ == "__main__":
    main()