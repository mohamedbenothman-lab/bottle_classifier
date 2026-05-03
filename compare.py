"""
compare.py
─────────────────────────────────────────────────────────────────────────────
Evaluates the trained CNN on the validation split (same split used during
training) and prints a detailed metrics report.

Uses the same CSV + stratified split as train_cnn.py so the val set is
identical — giving you a reliable estimate of real-world performance.

Usage
─────
  python compare.py
  python compare.py --threshold 0.40
  python compare.py --dataset path/to/train_images --csv-name train.csv --threshold 0.40
─────────────────────────────────────────────────────────────────────────────
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

# ── CLI ───────────────────────────────────────────────────────────────────────
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
parser.add_argument("--img-size",   type=int,   default=224,    help="Must match training img-size")
parser.add_argument("--batch",      type=int,   default=32)
parser.add_argument("--seed",       type=int,   default=42,     help="Must match training seed")
args = parser.parse_args()


# ── Rim crop (identical to train_cnn.py) ─────────────────────────────────────
def extract_rim_crop(img_bgr: np.ndarray, size: int = 260) -> np.ndarray:
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=200, param1=60, param2=35,
        minRadius=100, maxRadius=600,
    )
    if circles is not None:
        cx, cy, r = np.round(circles[0, 0]).astype(int)
        rim_w  = max(20, int(r * 0.18))
        margin = rim_w + 10
        h, w   = gray.shape
        x1 = max(0, cx - r - margin)
        y1 = max(0, cy - r - margin)
        x2 = min(w, cx + r + margin)
        y2 = min(h, cy + r + margin)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = img_bgr
    return cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)


# ── Dataset (identical to train_cnn.py) ──────────────────────────────────────
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
        return Image.fromarray(extract_rim_crop(img_bgr, self.img_size))


# ── Model (identical to train_cnn.py) ────────────────────────────────────────
def build_model():
    class EnsembleModel(nn.Module):
        def __init__(self):
            super(EnsembleModel, self).__init__()
            # Must match your training script exactly
            self.b2 = models.efficientnet_b2(weights=None)
            b2_out = self.b2.classifier[1].in_features  # 1408
            self.b2.classifier = nn.Identity()

            self.convnext = models.convnext_tiny(weights=None)
            cn_out = self.convnext.classifier[2].in_features  # 768
            self.convnext.classifier = nn.Identity()

            self.classifier = nn.Sequential(
                nn.Linear(b2_out + cn_out, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 1),
            )
            self._arch_version = "ensemble_b2_convnext_v1"

        def forward(self, x):
            f_b2 = self.b2(x)
            f_convnext = self.convnext(x)
            if f_convnext.dim() == 4:
                f_convnext = f_convnext.flatten(1)
            return self.classifier(torch.cat([f_b2, f_convnext], dim=1))

    return EnsembleModel()


# ── Data helpers (identical split logic to train_cnn.py) ─────────────────────
def load_val_samples(csv_path="assets/val_split.csv"):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return [(row["image_id"], int(row["target"])) for _, row in df.iterrows()]


# ── Metrics ───────────────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device    : {device}")
    print(f"[INFO] Model     : {args.model}")
    print(f"[INFO] Threshold : {args.threshold}")
    print(f"[INFO] Val split : {args.val_split}")

    # ── Load val split (same as training) ────────────────────────────────────
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

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at {args.model} — run train_cnn.py first.")

    model      = build_model().to(device)
    checkpoint = torch.load(args.model, map_location=device)

    # Support both raw state_dict and wrapped checkpoint
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Checkpoint loaded successfully\n")

    # ── Run inference ─────────────────────────────────────────────────────────
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

    # ── Threshold sweep ───────────────────────────────────────────────────────
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

    # ── Final report at chosen threshold ─────────────────────────────────────
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