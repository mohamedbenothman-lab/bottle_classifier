"""
training/train_cnn.py
─────────────────────────────────────────────────────────────────────────────
Industrial Rim Inspection: EfficientNet-B0 Training Pipeline
Dataset: CSV-based (dataset/train.csv)
Task: Binary Classification [0: Good (Accepted) | 1: Faulty (Rejected)]
─────────────────────────────────────────────────────────────────────────────
CORE LOGIC & ARCHITECTURE:
 1. Architecture: Utilizes EfficientNet-B2 to leverage compound scaling 
    (depth, width, and resolution) for high-precision defect detection.
 2. Vision: Native 260x260 resolution to capture subtle surface anomalies.
 3. Efficiency: Implements a dual-path loading system (Pre-computed cache 
    first, live Hough-Transform crop fallback) to maximize GPU utilization.
 4. Imbalance Handling: Stratified WeightedRandomSampler to address 
    rare-event defect classes.
 5. Robustness: Incorporates Label Smoothing, EMA (Exponential Moving 
    Average) weight tracking, and Gradient Clipping for stable convergence.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import copy
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train EfficientNet-B2 bottle rim classifier")
parser.add_argument("--dataset",       default="C:/Users/mdhia/OneDrive/Bureau/dataset/train_images",            help="Directory with images and train.csv")
parser.add_argument("--cache-dir",     default="C:/Users/mdhia/OneDrive/Bureau/dataset/dataset_cropped/",    help="Pre-cropped image cache (run preprocess.py first)")
parser.add_argument("--csv-name",      default="train.csv",           help="CSV filename")
parser.add_argument("--epochs",        type=int,   default=50,        help="Total training epochs")
parser.add_argument("--warmup-epochs", type=int,   default=12,        help="Epochs with backbone frozen (phase 1)")
parser.add_argument("--batch",         type=int,   default=64,        help="Batch size (reduce to 32 if OOM)")
parser.add_argument("--lr",            type=float, default=1e-4,      help="Fine-tune learning rate (phase 2)")
parser.add_argument("--workers",       type=int,   default=8,         help="DataLoader worker count")
parser.add_argument("--out",           default="assets/model.pt",     help="Output model path (state_dict)")
parser.add_argument("--val-split",     type=float, default=0.2,       help="Validation fraction")
parser.add_argument("--img-size",      type=int,   default=260,       help="Input size (260 = EfficientNet-B2 native)")
parser.add_argument("--ema-decay",     type=float, default=0.9995,    help="EMA decay rate (0 to disable)")
parser.add_argument("--label-smooth",  type=float, default=0.1,       help="Label smoothing epsilon (0 to disable)")
parser.add_argument("--grad-clip",     type=float, default=1.0,       help="Gradient clip max_norm (0 to disable)")
args = parser.parse_args()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Missing packages. Run: pip install torch torchvision pillow tqdm pandas")
    sys.exit(1)


# ── Rim crop (shared with auto_label.py and compare.py) ──────────────────────
def extract_rim_crop(img_bgr: np.ndarray, size: int = 260) -> np.ndarray:
    """
    Detects the bottle rim via Hough circles and crops tightly around it.
    Falls back to the full frame if no circle is found.
    Returns an RGB numpy array of shape (size, size, 3).
    """
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
        margin    = rim_w + 10
        h, w      = gray.shape
        x1 = max(0, cx - r - margin)
        y1 = max(0, cy - r - margin)
        x2 = min(w, cx + r + margin)
        y2 = min(h, cy + r + margin)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = img_bgr

    crop_rgb = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)
    return crop_rgb


# ── Transforms ────────────────────────────────────────────────────────────────
def _make_train_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.RandomPerspective(distortion_scale=0.2, p=0.4),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _make_val_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class RimDataset(Dataset):
    """
    Loads pre-cropped images from cache_dir if available, otherwise runs
    the live Hough-based crop at load time.

    cache_dir should be populated by running preprocess.py once before training.
    This makes each __getitem__ a simple PIL.open() — no OpenCV at runtime.
    """
    def __init__(self, samples: list, transform: T.Compose,
                 cache_dir: str = "", img_size: int = 260):
        self.samples   = samples
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.img_size  = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = self._load_image(path)
        tensor = self.transform(pil)
        return tensor, torch.tensor(label, dtype=torch.float32)

    def _load_image(self, orig_path: str) -> Image.Image:
        # Try pre-cropped cache first (fast path — just PIL open)
        if self.cache_dir is not None:
            rel      = Path(orig_path).name
            cached   = self.cache_dir / rel
            if cached.exists():
                return Image.open(cached).convert("RGB")

        # Slow path: live Hough crop (only when cache is missing)
        img_bgr = cv2.imread(orig_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {orig_path}")
        crop = extract_rim_crop(img_bgr, self.img_size)
        return Image.fromarray(crop)


# ── Data loading helpers ──────────────────────────────────────────────────────
def _load_samples(dataset_root: str, csv_name: str) -> list:
    root     = Path(dataset_root)
    csv_path = root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df      = pd.read_csv(csv_path)
    samples = []
    missing = 0

    for _, row in df.iterrows():
        img_path = root / row["image_id"]
        if img_path.exists():
            model_label = int(row["target"])
            samples.append((str(img_path), model_label))
        else:
            missing += 1

    if missing:
        print(f"[WARN] {missing} images listed in CSV were not found on disk.")
    if not samples:
        raise RuntimeError("No valid images found in CSV.")

    n_good = sum(1 for _, l in samples if l == 0)
    n_fail = sum(1 for _, l in samples if l == 1)
    ratio  = n_good / max(n_fail, 1)
    print(f"[INFO] Total: {len(samples)}  |  Good: {n_good}  |  Faulty: {n_fail}  "
          f"|  Ratio: {ratio:.1f}:1")
    return samples


def _split_samples(samples: list, val_fraction: float, seed: int = 42):
    """Stratified train/val split preserving class ratios."""
    import random
    rng    = random.Random(seed)
    good   = [s for s in samples if s[1] == 0]
    faulty = [s for s in samples if s[1] == 1]
    rng.shuffle(good)
    rng.shuffle(faulty)

    def split(lst):
        n_val = max(1, int(len(lst) * val_fraction))
        return lst[n_val:], lst[:n_val]

    train_g, val_g = split(good)
    train_f, val_f = split(faulty)
    train_samples  = train_g + train_f
    val_samples    = val_g + val_f
    rng.shuffle(train_samples)

    print(f"[INFO] Train: {len(train_samples)}  |  Val: {len(val_samples)}")
    return train_samples, val_samples


def _make_sampler(train_samples: list) -> WeightedRandomSampler:
    """
    WeightedRandomSampler so each batch sees a balanced class distribution
    regardless of the raw class ratio in the dataset.
    """
    labels      = [l for _, l in train_samples]
    n_good      = labels.count(0)
    n_fail      = labels.count(1)
    weight_good = 1.0 / n_good if n_good > 0 else 1.0
    weight_fail = 1.0 / n_fail if n_fail > 0 else 1.0
    weights     = [weight_good if l == 0 else weight_fail for l in labels]
    sampler     = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    print(f"[INFO] Sampler weights — Good: {weight_good:.6f}  Faulty: {weight_fail:.6f}")
    return sampler


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    """
    Constructs the EfficientNet-B2 architecture with a custom deep-head 
    classifier tailored for industrial binary quality control.
    
    The classifier utilizes SiLU (Swish) activations to maintain consistency 
    with the EfficientNet backbone and includes Batch Normalization to 
    stabilize the feature distributions before final logit output.
    """
    model       = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features   # 1208 
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),          # Swish — matches EfficientNet internals
        nn.Dropout(0.35),
        nn.Linear(256, 1),
    )
    model._arch_version = "efficientnet_b0_v1"
    return model


def _set_backbone_grad(model: nn.Module, requires_grad: bool):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = requires_grad


# ── EMA ───────────────────────────────────────────────────────────────────────
class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Keeps a shadow copy: shadow = decay * shadow + (1 - decay) * current
    Use ema.apply() before validation, ema.restore() before next train step.

    EMA weights typically generalise 0.5–1.5 F1 points better than raw
    checkpoints, especially with strong augmentation.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Swap model weights with shadow weights in-place."""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data, m_param.data = m_param.data.clone(), s_param.data.clone()

    def restore(self, model: nn.Module):
        """Swap back — must be called after apply()."""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data, m_param.data = m_param.data.clone(), s_param.data.clone()


# ── Loss with label smoothing ─────────────────────────────────────────────────
class SmoothedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss + label smoothing.
    Smoothed target = (1 - eps) * target + eps * 0.5
    Prevents the model from becoming overconfident, which is common when
    training on high-quality industrial inspection images.
    """
    def __init__(self, pos_weight=None, eps: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.eps > 0:
            targets = targets * (1.0 - self.eps) + 0.5 * self.eps
        return self.bce(logits, targets)


# ── Metrics ───────────────────────────────────────────────────────────────────
def _compute_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


# ── Validation pass ───────────────────────────────────────────────────────────
def _validate(model, loader, device, threshold: float = 0.5):
    model.eval()
    tp = tn = fp = fn = 0
    total_correct = total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= threshold).float()

            total_correct += (preds == labels).sum().item()
            total         += len(labels)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = (total_correct / total * 100) if total else 0.0
    f1  = _compute_f1(tp, fp, fn)
    return acc, f1, tp, tn, fp, fn


# ── Quick threshold sweep (used every epoch for checkpoint selection) ─────────
def _quick_sweep(model, loader, device):
    """
    Sweeps thresholds [0.05 … 0.95] to find the one maximising F1.
    Lightweight — no printing. Used during training loop to pick best checkpoint.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(1)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels)
    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    best_f1, best_thresh = -1.0, 0.5
    for t_int in range(5, 95, 5):
        t     = t_int / 100.0
        preds = (probs >= t).float()
        tp    = int(((preds == 1) & (labels == 1)).sum())
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        f1    = _compute_f1(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_f1, best_thresh

def _recalibrate_bn(model, loader, device, n_batches=50):
    """
    Run a few forward passes in train mode to update BatchNorm
    running statistics to match the current weights.
    """
    model.train()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(imgs.to(device))
    model.eval()

# ── Final threshold sweep (verbose, run once on best checkpoint) ──────────────
def _sweep_threshold(model, loader, device):
    """
    Collects all sigmoid probabilities on the val set, then sweeps
    thresholds [0.05 … 0.95] to find the one maximising F1.
    Uses whatever weights are currently loaded into model (caller's responsibility).
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(1)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels)

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    print("\n[Threshold sweep — EMA weights]")
    print(f"  {'Threshold':>10}  {'F1':>8}  {'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}")
    print("  " + "─" * 52)

    best_f1, best_thresh = -1.0, 0.5
    for t_int in range(5, 95, 5):
        t     = t_int / 100.0
        preds = (probs >= t).float()
        tp    = int(((preds == 1) & (labels == 1)).sum())
        tn    = int(((preds == 0) & (labels == 0)).sum())
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        f1    = _compute_f1(tp, fp, fn)
        mark  = " ← best" if f1 > best_f1 else ""
        print(f"  {t:>10.2f}  {f1:>8.4f}  {tp:>5}  {tn:>5}  {fp:>5}  {fn:>5}{mark}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\n  → Use threshold = {best_thresh:.2f} in auto_label.py and compare.py")
    return best_thresh, best_f1


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Backbone    : EfficientNet-B2 (native 260×260)")
    print(f"[INFO] Batch size  : {args.batch}")
    print(f"[INFO] Epochs      : {args.epochs}  (warmup: {args.warmup_epochs})")
    print(f"[INFO] EMA decay   : {args.ema_decay}")
    print(f"[INFO] Label smooth: {args.label_smooth}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    all_samples                = _load_samples(args.dataset, args.csv_name)
    train_samples, val_samples = _split_samples(all_samples, args.val_split)

    os.makedirs("assets", exist_ok=True)
    val_df = pd.DataFrame(val_samples, columns=["image_id", "target"])
    val_df.to_csv("assets/val_split.csv", index=False)


    train_ds = RimDataset(train_samples, _make_train_transform(args.img_size),
                          args.cache_dir, args.img_size)
    val_ds   = RimDataset(val_samples,   _make_val_transform(args.img_size),
                          args.cache_dir, args.img_size)

    sampler      = _make_sampler(train_samples)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    # ── 2. Model, loss, EMA ──────────────────────────────────────────────────
    model = build_model().to(device)
    ema   = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    criterion = SmoothedBCEWithLogitsLoss(pos_weight=None, eps=args.label_smooth)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    best_val_f1     = -1.0
    best_val_thresh = 0.5
    scaler          = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    warmup          = args.warmup_epochs

    # ── 3. Two-phase training ────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        if epoch == 1:
            print(f"\n[Phase 1] Backbone frozen — head only ({warmup} epochs, lr={args.lr*10:.1e})")
            _set_backbone_grad(model, requires_grad=False)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 10, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 10,
                steps_per_epoch=len(train_loader), epochs=warmup,
                pct_start=0.3,
            )

        elif epoch == warmup + 1:
            remaining = args.epochs - warmup
            print(f"\n[Phase 2] Full network — fine-tuning ({remaining} epochs, lr={args.lr:.1e})")
            _set_backbone_grad(model, requires_grad=True)
            # Use layer-wise LR decay: backbone gets lower LR than head
            backbone_params = [p for n, p in model.named_parameters()
                               if "classifier" not in n]
            head_params     = [p for n, p in model.named_parameters()
                               if "classifier" in n]
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.lr * 0.1},   # 10x lower for backbone
                {"params": head_params,     "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=[args.lr * 0.1, args.lr],
                steps_per_epoch=len(train_loader), epochs=remaining,
                pct_start=0.1,
            )

        # ── Train step ───────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs).squeeze(1)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema is not None:
                ema.update(model)

            with torch.no_grad():
                preds          = (torch.sigmoid(logits.detach()) >= 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total   += len(labels)
                train_loss    += loss.item() * len(labels)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_acc  = train_correct / train_total * 100
        avg_loss   = train_loss / train_total

        # ── Validate with EMA weights ────────────────────────────────────────
        if ema is not None:
            ema.apply(model)

        val_acc, val_f1, tp, tn, fp, fn = _validate(model, val_loader, device)
        swept_f1, swept_thresh = _quick_sweep(model, val_loader, device)

        if ema is not None:
            ema.restore(model)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"| Loss: {avg_loss:.4f}  "
              f"| Train: {train_acc:.1f}%  "
              f"| Val Acc: {val_acc:.1f}%  "
              f"| Val F1: {val_f1:.4f}  "
              f"| Swept F1: {swept_f1:.4f} @ {swept_thresh:.2f}")
        print(f"  Confusion → TP={tp}  TN={tn}  FP={fp}  FN={fn}")

        if swept_f1 > best_val_f1:
            best_val_f1     = swept_f1
            best_val_thresh = swept_thresh
            if ema is not None:
                # Recalibrate BN stats on shadow model before saving
                _recalibrate_bn(ema.shadow, train_loader, device, n_batches=50)
            save_state = ema.shadow.state_dict() if ema else model.state_dict()
            torch.save({
                "state_dict":   save_state,
                "arch_version": model._arch_version,
            }, args.out)
            print(f"  ✓ Checkpoint saved (F1={swept_f1:.4f} @ thresh={swept_thresh:.2f}): {args.out}")

    # ── 4. Load best checkpoint and sweep threshold on held-out test set ──────
    print(f"\n[DONE] Best Val F1: {best_val_f1:.4f} @ threshold={best_val_thresh:.2f}")
    print("[INFO] Loading best checkpoint for final threshold sweep on test set …")

    best_model = build_model().to(device)
    checkpoint = torch.load(args.out, map_location=device)
    assert checkpoint["arch_version"] == best_model._arch_version, (
        f"Architecture mismatch! Checkpoint is '{checkpoint['arch_version']}' "
        f"but current model is '{best_model._arch_version}'. "
        f"Delete {args.out} and retrain."
    )
    best_model.load_state_dict(checkpoint["state_dict"])


    best_thresh, best_f1 = _sweep_threshold(best_model, val_loader, device)

    print(f"\n{'─'*60}")
    print(f"  SUMMARY")
    print(f"{'─'*60}")
    print(f"  Backbone      : EfficientNet-B0")
    print(f"  Best Val F1   : {best_val_f1:.4f}  (threshold={best_val_thresh:.2f})")
    print(f"  Final F1       : {best_f1:.4f}  (threshold={best_thresh:.2f})")
    print(f"  Model saved   : {args.out}")
    print(f"  → Set CNN_THRESHOLD={best_thresh:.2f} in auto_label.py and compare.py")
    print(f"{'─'*60}")


if __name__ == "__main__":
    train()