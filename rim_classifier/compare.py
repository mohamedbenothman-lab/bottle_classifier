"""
compare.py
──────────────────────────────────────────────────────────────────────────────
Evaluates the trained ensemble on the validation split saved by
train_cnn_v6.py (assets/val_split.csv) and prints a detailed metrics report.

Architecture matches train_cnn_v6.py exactly:
  - EfficientNet-B4  (upgraded from B2)
  - ResNet-50        (unchanged)
  - ConvNeXt-Small   (upgraded from Tiny)
  - CBAM on each backbone
  - Cross-branch MultiheadAttention before fusion
  - Fusion gate
  - Deeper MLP head
  - Dual output heads: binary_head + severity_head
  - Evaluation uses binary_head only

Usage:
    python compare.py
    python compare.py --threshold 0.42
    python compare.py --model assets/model_v6.pt --threshold 0.42
    python compare.py --tta
    python compare.py --calibrator assets/calibrator.pkl
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import pickle
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Evaluate EnsembleModelV6 on validation split")
parser.add_argument("--dataset",
    default="/kaggle/input/competitions/"
            "1st-krones-vision-ai-challenge/train_images",
    help="Directory that contains the original images")
parser.add_argument("--cache-dir",
    default="/kaggle/input/datasets/mohameddhiabenothman/"
            "cropped-dataset/dataset_cropped",
    help="Pre-cropped image cache directory")
parser.add_argument("--val-csv",
    default="/kaggle/working/assets/val_split.csv",
    help="Validation split CSV saved by train_cnn_v6.py")
parser.add_argument("--model",
    default="/kaggle/working/assets/model_v6.pt",
    help="Trained model checkpoint path")
parser.add_argument("--calibrator",
    default="",
    help="Optional Platt scaling calibrator (.pkl)")
parser.add_argument("--threshold",
    type=float, default=0.40,
    help="Sigmoid threshold")
parser.add_argument("--img-size",
    type=int, default=300,
    help="Must match --img-size used during training (default 300)")
parser.add_argument("--batch",
    type=int, default=24)
parser.add_argument("--tta",
    action="store_true", default=False,
    help="Enable 16-view test-time augmentation")
parser.add_argument("--grad-checkpoint",
    action="store_true", default=False)
parser.add_argument("--dropout-rate",
    type=float, default=0.35,
    help="Must match --dropout-rate used during training")
args = parser.parse_args()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class RimDataset(Dataset):
    """
    Loads labelled validation samples from the pre-cropped directory.
    Returns (tensor, label_float, path).
    """
    def __init__(self, samples, transform,
                 cache_dir="", img_size=300):
        self.samples   = samples
        self.transform = transform
        self.cache_dir = (Path(cache_dir)
                          if cache_dir else None)
        self.img_size  = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil    = self._load(path)
        tensor = self.transform(pil)
        return (tensor,
                torch.tensor(label, dtype=torch.float32),
                path)

    def _load(self, orig_path: str) -> Image.Image:
        # 1. Try pre-cropped cache first (fastest)
        if self.cache_dir is not None:
            cached = self.cache_dir / Path(orig_path).name
            if cached.exists():
                img = Image.open(cached).convert("RGB")
                if img.size != (self.img_size,
                                self.img_size):
                    img = img.resize(
                        (self.img_size, self.img_size),
                        Image.BILINEAR)
                return img

        # 2. Fall back: load original and resize
        if not Path(orig_path).exists():
            raise FileNotFoundError(
                f"Cannot read image: {orig_path}")
        img = Image.open(orig_path).convert("RGB")
        img = img.resize(
            (self.img_size, self.img_size),
            Image.BILINEAR)
        return img

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPONENTS  — must match train_cnn_v6.py exactly
# ─────────────────────────────────────────────────────────────────────────────
class AttentionPool2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv2d(
            channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = torch.softmax(
            self.score(x).view(B, 1, H * W), dim=2
        ).view(B, 1, H, W)
        return (x * w).sum(dim=(2, 3))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.ch_fc1  = nn.Linear(channels, mid,      bias=False)
        self.ch_fc2  = nn.Linear(mid,      channels, bias=False)
        self.sp_conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_c = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_c = F.adaptive_max_pool2d(x, 1).flatten(1)
        ch    = torch.sigmoid(
            self.ch_fc2(F.relu(self.ch_fc1(avg_c))) +
            self.ch_fc2(F.relu(self.ch_fc1(max_c)))
        ).unsqueeze(-1).unsqueeze(-1)
        x  = x * ch
        sp = torch.sigmoid(
            self.sp_conv(
                torch.cat(
                    [x.mean(1, keepdim=True),
                     x.max(1, keepdim=True).values],
                    dim=1)
            )
        )
        return x * sp


def _get_out_channels(module, img_size=64,
                       in_channels=3):
    device = next(module.parameters()).device
    with torch.no_grad():
        dummy = torch.zeros(
            1, in_channels, img_size, img_size,
            device=device)
        out = module(dummy)
    return out.shape[1]


class EnsembleModelV6(nn.Module):
    """
    Matches train_cnn_v6.py EnsembleModelV6 exactly:
      - EfficientNet-B4  stem[:6] + last[6:]
      - ResNet-50        stem (conv1…layer3) + layer4
      - ConvNeXt-Small   features[:6] + features[6:]
      - CBAM + AttentionPool2d per backbone
      - Linear projection to 320-d per backbone
      - Cross-branch MultiheadAttention (3 tokens × 320)
      - Fusion gate on concatenated 960-d vector
      - MLP head: 960→512→256→128
      - Dual output: binary_head + severity_head
    """
    _arch_version = "v6_ensemble_kaggle"

    def __init__(self, use_grad_checkpoint=False,
                 dropout_rate=0.35):
        super().__init__()
        self._use_grad_ckpt = use_grad_checkpoint

        # ── EfficientNet-B4 ───────────────────────────────────────────────
        eff          = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.b4_stem = eff.features[:6]
        self.b4_last = eff.features[6:]
        _b4_ch       = _get_out_channels(
            nn.Sequential(self.b4_stem, self.b4_last))
        self.b4_cbam = CBAM(_b4_ch)
        self.b4_pool = AttentionPool2d(_b4_ch)
        self.b4_proj = nn.Sequential(
            nn.Linear(_b4_ch, 320),
            nn.GELU(),
            nn.Dropout(dropout_rate))

        # ── ResNet-50 ─────────────────────────────────────────────────────
        rn           = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT)
        self.rn_stem = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool,
            rn.layer1, rn.layer2, rn.layer3)
        self.rn_last = rn.layer4
        _rn_ch       = _get_out_channels(
            nn.Sequential(self.rn_stem, self.rn_last))
        self.rn_cbam = CBAM(_rn_ch)
        self.rn_pool = AttentionPool2d(_rn_ch)
        self.rn_proj = nn.Sequential(
            nn.Linear(_rn_ch, 320),
            nn.GELU(),
            nn.Dropout(dropout_rate))

        # ── ConvNeXt-Small ────────────────────────────────────────────────
        cx           = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.DEFAULT)
        self.cx_stem = cx.features[:6]
        self.cx_last = cx.features[6:]
        _cx_ch       = _get_out_channels(
            nn.Sequential(self.cx_stem, self.cx_last))
        self.cx_cbam = CBAM(_cx_ch)
        self.cx_pool = AttentionPool2d(_cx_ch)
        self.cx_proj = nn.Sequential(
            nn.Linear(_cx_ch, 320),
            nn.GELU(),
            nn.Dropout(dropout_rate))

        # ModuleList aliases — must match training script
        self.b4      = nn.ModuleList([
            self.b4_stem, self.b4_last,
            self.b4_cbam, self.b4_pool, self.b4_proj])
        self.resnet  = nn.ModuleList([
            self.rn_stem, self.rn_last,
            self.rn_cbam, self.rn_pool, self.rn_proj])
        self.convnxt = nn.ModuleList([
            self.cx_stem, self.cx_last,
            self.cx_cbam, self.cx_pool, self.cx_proj])

        fused_dim        = 320 * 3          # 960
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid())

        # Cross-branch attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=320, num_heads=8,
            batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(320)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.3),
        )
        self.binary_head   = nn.Linear(128, 1)
        self.severity_head = nn.Linear(128, 1)

    def _fwd_backbone(self, stem, last,
                       cbam, pool, proj, x):
        feat = last(stem(x))
        return proj(pool(cbam(feat)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns binary logit (B,) — squeeze(1) applied."""
        f_b4 = self._fwd_backbone(
            self.b4_stem, self.b4_last,
            self.b4_cbam, self.b4_pool,
            self.b4_proj, x)
        f_rn = self._fwd_backbone(
            self.rn_stem, self.rn_last,
            self.rn_cbam, self.rn_pool,
            self.rn_proj, x)
        f_cx = self._fwd_backbone(
            self.cx_stem, self.cx_last,
            self.cx_cbam, self.cx_pool,
            self.cx_proj, x)

        # Cross-branch attention
        tokens = torch.stack(
            [f_b4, f_rn, f_cx], dim=1)   # (B, 3, 320)
        attn_out, _ = self.cross_attn(
            tokens, tokens, tokens)
        tokens = self.cross_norm(
            tokens + attn_out)

        f_b4 = tokens[:, 0]
        f_rn = tokens[:, 1]
        f_cx = tokens[:, 2]

        fused = torch.cat(
            [f_b4, f_rn, f_cx], dim=1)
        fused = fused * self.fusion_gate(fused)
        feat  = self.head(fused)
        return self.binary_head(feat).squeeze(1)


def build_model() -> EnsembleModelV6:
    return EnsembleModelV6(
        use_grad_checkpoint=args.grad_checkpoint,
        dropout_rate=args.dropout_rate)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD VALIDATION SAMPLES
# ─────────────────────────────────────────────────────────────────────────────
def load_val_samples(csv_path: str, dataset_root: str):
    """
    Read val_split.csv and resolve image paths.
    val_split.csv written by train_cnn_v6.py stores
    full absolute paths in image_id column.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {csv_path}\n"
            "Run train_cnn_v6.py first.")

    df       = pd.read_csv(csv_path)
    samples  = []
    missing  = 0

    for _, row in df.iterrows():
        img_id    = str(row["image_id"])
        # Priority 1: absolute path stored directly
        candidate = Path(img_id)
        if not candidate.exists():
            # Priority 2: relative to dataset_root
            candidate = Path(dataset_root) / img_id
        if not candidate.exists():
            # Priority 3: just the filename in dataset_root
            candidate = (Path(dataset_root)
                         / Path(img_id).name)
        if candidate.exists():
            samples.append(
                (str(candidate), int(row["target"])))
        else:
            missing += 1

    if missing:
        print(f"[WARN] {missing} image(s) in val CSV "
              "not found on disk — skipped.")
    if not samples:
        raise RuntimeError(
            "No valid validation samples found. "
            "Check --dataset and --val-csv.")
    return samples

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred)
             if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred)
             if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred)
             if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred)
             if t == 1 and p == 0)
    n      = len(y_true)
    acc    = (tp + tn) / n             if n            > 0 else 0.0
    prec   = tp / (tp + fp)            if (tp + fp)    > 0 else 0.0
    recall = tp / (tp + fn)            if (tp + fn)    > 0 else 0.0
    f1     = (2 * tp /
              (2 * tp + fp + fn)       if (2 * tp + fp + fn) > 0
              else 0.0)
    return dict(acc=acc, prec=prec, recall=recall,
                f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)

# ─────────────────────────────────────────────────────────────────────────────
# TTA  — 16-view (matches train_cnn_v6.py)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_batch(model: nn.Module,
                imgs:  torch.Tensor,
                use_tta: bool = False) -> torch.Tensor:
    """Return per-image probabilities (B,)."""
    if not use_tta:
        return torch.sigmoid(model(imgs))

    augmentations = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.flip(x, [2, 3]),
    ]
    preds = []
    for flip_fn in augmentations:
        for k in range(4):
            aug = torch.rot90(flip_fn(imgs), k, [2, 3])
            preds.append(torch.sigmoid(model(aug)))
    return torch.stack(preds).mean(0)   # 16 views

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═'*60}")
    print(f"  EnsembleModelV6 — Evaluation")
    print(f"{'═'*60}")
    print(f"  Device      : {device}")
    print(f"  Model       : {args.model}")
    print(f"  Val CSV     : {args.val_csv}")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Image size  : {args.img_size}")
    print(f"  TTA         : "
          f"{'16-view' if args.tta else 'disabled'}")
    if args.calibrator:
        print(f"  Calibrator  : {args.calibrator}")
    print(f"{'═'*60}\n")

    # ── Optional Platt calibrator ──────────────────────────────────────────
    calibrator = None
    if args.calibrator and Path(args.calibrator).exists():
        with open(args.calibrator, "rb") as f:
            calibrator = pickle.load(f)
        print("[INFO] Platt scaling calibrator loaded.")
    elif args.calibrator:
        print(f"[WARN] Calibrator not found at "
              f"'{args.calibrator}' — using raw sigmoid.")

    # ── Validation samples ─────────────────────────────────────────────────
    val_samples = load_val_samples(
        args.val_csv, args.dataset)
    n_good = sum(1 for _, l in val_samples if l == 0)
    n_fail = sum(1 for _, l in val_samples if l == 1)
    print(f"[INFO] Val set : {len(val_samples)} images"
          f"  |  Good: {n_good}  |  Faulty: {n_fail}\n")

    # ── DataLoader ─────────────────────────────────────────────────────────
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = RimDataset(
        val_samples, transform,
        args.cache_dir, args.img_size)
    loader  = DataLoader(
        dataset, batch_size=args.batch,
        shuffle=False, num_workers=0)

    # ── Load model ─────────────────────────────────────────────────────────
    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.model}\n"
            "Run train_cnn_v6.py first.")

    print("[INFO] Building model…")
    model = build_model().to(device)

    checkpoint = torch.load(
        args.model, map_location=device,
        weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(
        state, strict=False)
    if missing:
        print(f"[WARN] Missing keys  ({len(missing)}): "
              f"{missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}"
              f"{'…' if len(unexpected)>5 else ''}")
    if not missing and not unexpected:
        print("[INFO] State dict loaded — exact match ✓")

    model.eval()

    arch = checkpoint.get("arch_version", "unknown")
    ep   = checkpoint.get("epoch",        "?")
    f1   = checkpoint.get("best_val_f1",  float("nan"))
    bt   = checkpoint.get("best_threshold", args.threshold)

    print(f"[INFO] Checkpoint : epoch={ep}"
          f"  val_F1={f1:.4f}"
          f"  arch={arch}")
    if abs(bt - args.threshold) > 0.01:
        print(f"[HINT] Training best threshold = {bt:.2f}"
              f", you passed {args.threshold:.2f}."
              f"  Consider --threshold {bt:.2f}")

    # ── Inference ──────────────────────────────────────────────────────────
    all_probs, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(
                loader, desc="Evaluating"):
            imgs = imgs.to(device, non_blocking=True)

            # Raw probabilities
            probs = infer_batch(
                model, imgs,
                use_tta=args.tta).cpu().numpy()

            # Optional Platt calibration
            if calibrator is not None:
                probs = calibrator.predict_proba(
                    probs.reshape(-1, 1))[:, 1]

            all_probs.extend(probs.tolist())
            all_labels.extend(
                labels.numpy().astype(int).tolist())
            all_paths.extend(paths)

    # ── Probability distribution ───────────────────────────────────────────
    arr = np.array(all_probs)
    print(f"\n[Probability Distribution]")
    print(f"  Min    : {arr.min():.4f}")
    print(f"  Max    : {arr.max():.4f}")
    print(f"  Mean   : {arr.mean():.4f}")
    print(f"  Median : {np.median(arr):.4f}")
    print(f"  Std    : {arr.std():.4f}")
    print(f"  < 0.10 : {(arr < 0.10).sum():>5}")
    print(f"  0.10–0.50: "
          f"{((arr>=0.10)&(arr<0.50)).sum():>5}")
    print(f"  > 0.50 : {(arr >= 0.50).sum():>5}")

    # Per-class probability stats
    arr_good  = arr[np.array(all_labels) == 0]
    arr_fault = arr[np.array(all_labels) == 1]
    print(f"\n  Good  prob  mean={arr_good.mean():.4f}"
          f"  std={arr_good.std():.4f}"
          if len(arr_good) else "")
    print(f"  Fault prob  mean={arr_fault.mean():.4f}"
          f"  std={arr_fault.std():.4f}"
          if len(arr_fault) else "")

    # ── Threshold sweep ────────────────────────────────────────────────────
    # Find best at 0.01 resolution
    best_f1, best_thresh = -1.0, args.threshold
    for t_int in range(1, 99):
        t     = t_int / 100.0
        preds = [1 if p >= t else 0 for p in all_probs]
        m     = compute_metrics(all_labels, preds)
        if m["f1"] > best_f1:
            best_f1, best_thresh = m["f1"], t

    # Print table at 0.05 steps
    print(f"\n[Threshold Sweep]")
    print(f"  {'Thr':>6}  {'F1':>8}  {'Prec':>7}"
          f"  {'Rec':>7}  {'Acc':>7}"
          f"  {'TP':>5}  {'TN':>5}"
          f"  {'FP':>5}  {'FN':>5}")
    print("  " + "─" * 78)

    for t_int in range(5, 96, 5):
        t     = t_int / 100.0
        preds = [1 if p >= t else 0 for p in all_probs]
        m     = compute_metrics(all_labels, preds)
        mark  = (
            f"  ← best≈{best_thresh:.2f} F1={best_f1:.4f}"
            if abs(t - best_thresh) < 0.025 else "")
        print(
            f"  {t:>6.2f}  {m['f1']:>8.4f}"
            f"  {m['prec']:>7.4f}  {m['recall']:>7.4f}"
            f"  {m['acc']:>6.1%}"
            f"  {m['tp']:>5}  {m['tn']:>5}"
            f"  {m['fp']:>5}  {m['fn']:>5}{mark}")

    # ── AUC ───────────────────────────────────────────────────────────────
    try:
        from sklearn.metrics import (roc_auc_score,
                                      average_precision_score)
        auc_roc = roc_auc_score(all_labels, all_probs)
        auc_pr  = average_precision_score(
            all_labels, all_probs)
        print(f"\n  ROC-AUC : {auc_roc:.4f}")
        print(f"  PR-AUC  : {auc_pr:.4f}")
    except ImportError:
        pass

    # ── Final report at chosen threshold ──────────────────────────────────
    final_preds = [
        1 if p >= args.threshold else 0
        for p in all_probs]
    m = compute_metrics(all_labels, final_preds)

    SEP  = "═" * 58
    SEP2 = "─" * 58
    print(f"\n{SEP}")
    print(f"  EVALUATION REPORT"
          f"  (threshold = {args.threshold:.2f})")
    print(f"{SEP}")
    print(f"  Accuracy   : {m['acc']:>8.2%}")
    print(f"  Precision  : {m['prec']:>8.2%}")
    print(f"  Recall     : {m['recall']:>8.2%}")
    print(f"  F1 Score   : {m['f1']:>8.4f}")
    print(f"{SEP2}")
    print(f"  TP (correct faulty)  : {m['tp']:>6}")
    print(f"  TN (correct good)    : {m['tn']:>6}")
    print(f"  FP (good → faulty)   : {m['fp']:>6}"
          f"  ← false alarms")
    print(f"  FN (faulty → good)   : {m['fn']:>6}"
          f"  ← missed defects")
    print(f"{SEP}")
    print(f"\n  Best threshold : {best_thresh:.2f}"
          f"  (F1 = {best_f1:.4f})")
    print(f"  → Use --threshold {best_thresh:.2f}"
          f" in predict.py for best results")
    print(f"{SEP}\n")

    # ── Misclassified images ───────────────────────────────────────────────
    errors = [
        (p, l, pr, prob)
        for p, l, pr, prob in zip(
            all_paths, all_labels,
            final_preds, all_probs)
        if l != pr
    ]
    print(f"[Misclassified — {len(errors)} total]")
    if not errors:
        print("  None — perfect score on val set! 🎉")
    else:
        # Sort: worst (most confident wrong) first
        errors.sort(
            key=lambda x: abs(x[3] - 0.5),
            reverse=True)

        # Separate FP and FN
        fp_list = [(p, l, pr, prob)
                   for p, l, pr, prob in errors
                   if l == 0 and pr == 1]
        fn_list = [(p, l, pr, prob)
                   for p, l, pr, prob in errors
                   if l == 1 and pr == 0]

        print(f"\n  False Positives (good → faulty): "
              f"{len(fp_list)}")
        print(f"  {'File':<50} {'Prob':>6}")
        print("  " + "─" * 58)
        for path, *_, prob in fp_list[:15]:
            print(f"  {Path(path).name:<50} {prob:.4f}")
        if len(fp_list) > 15:
            print(f"  … and {len(fp_list)-15} more")

        print(f"\n  False Negatives (faulty → good): "
              f"{len(fn_list)}")
        print(f"  {'File':<50} {'Prob':>6}")
        print("  " + "─" * 58)
        for path, *_, prob in fn_list[:15]:
            print(f"  {Path(path).name:<50} {prob:.4f}")
        if len(fn_list) > 15:
            print(f"  … and {len(fn_list)-15} more")

    print()


if __name__ == "__main__":
    main()