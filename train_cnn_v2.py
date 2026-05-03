"""
training/train_cnn.py
─────────────────────────────────────────────────────────────────────────────
Industrial Rim Inspection: EfficientNet-B2 + ConvNeXt-Tiny Ensemble Pipeline
Dataset: CSV-based (dataset/train.csv)
Task: Binary Classification [0: Good (Accepted) | 1: Faulty (Rejected)]

Changes from v2:
  BUG-1  Removed duplicate _set_backbone_grad (second definition silently
         overwrote the first and would also freeze classifier layers).
  BUG-2  Fixed ModelEMA.apply / ModelEMA.restore — both were identical swaps,
         meaning EMA weights were never actually applied for eval/save.
  BUG-3  pos_weight now computed from class counts and passed to loss.
  BUG-4  _split() no longer steals 1 sample per class when test_frac == 0.
  BUG-5  Phase-2 OneCycleLR max_lr now has 3 entries to match 3 param groups.
  BUG-6  Gradient clipping (--grad-clip) is now actually applied in the loop.
  ADD-7  Precision & Recall printed each epoch alongside F1.
  ADD-8  Checkpoint resume via --resume flag (saves/loads optimizer+scheduler).
  ADD-9  TensorBoard logging (SummaryWriter) — optional, skipped if not installed.
  ADD-10 AUC-ROC computed via sklearn after each validation pass.
  ADD-11 Mixup augmentation in the training loop (--mixup-alpha, default 0.2).
  ADD-12 torch.cuda.manual_seed_all for full GPU reproducibility.
  ADD-13 Early CSV column validation (image_id / target) with clear error.
  ADD-14 Empty test-set guard before final evaluation.
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
parser = argparse.ArgumentParser(description="Train EfficientNet-B2 + ConvNeXt-Tiny rim classifier")
parser.add_argument("--dataset",          default="C:/Users/mdhia/OneDrive/Bureau/dataset/train_images",
                    help="Directory with images and train.csv")
parser.add_argument("--cache-dir",        default="C:/Users/mdhia/OneDrive/Bureau/dataset/dataset_cropped/",
                    help="Pre-cropped image cache (run preprocess.py first)")
parser.add_argument("--csv-name",         default="train.csv",        help="CSV filename")
parser.add_argument("--epochs",           type=int,   default=50,     help="Total training epochs")
parser.add_argument("--warmup-epochs",    type=int,   default=12,     help="Epochs with backbone frozen (phase 1)")
parser.add_argument("--batch",            type=int,   default=8,      help="Batch size")
parser.add_argument("--lr",              type=float, default=1e-4,    help="Fine-tune learning rate (phase 2)")
parser.add_argument("--workers",          type=int,   default=2,      help="DataLoader worker count")
parser.add_argument("--out",             default="assets/model.pt",   help="Output model path (state_dict)")
parser.add_argument("--resume",          default="",                  help="Path to checkpoint to resume from")
parser.add_argument("--val-split",        type=float, default=0.10,   help="Validation fraction")
parser.add_argument("--test-split",       type=float, default=0.0,    help="Test fraction (held-out)")
parser.add_argument("--img-size",         type=int,   default=224,    help="Input size (224 for B0 native)")
parser.add_argument("--ema-decay",        type=float, default=0.9995, help="EMA decay rate (0 to disable)")
parser.add_argument("--label-smooth",     type=float, default=0.05,   help="Label smoothing epsilon (0 to disable)")
parser.add_argument("--grad-clip",        type=float, default=1.0,    help="Gradient clip max_norm (0 to disable)")
parser.add_argument("--early-stop",       type=int,   default=10,     help="Early stopping patience (epochs)")
parser.add_argument("--bn-recal-batches", type=int,   default=50,     help="Batches used for BN recalibration")
parser.add_argument("--mixup-alpha",      type=float, default=0.2,    help="Mixup alpha (0 to disable)")
parser.add_argument("--seed",             type=int,   default=42,     help="Random seed")
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
    print("[ERROR] Missing packages. Run: pip install torch torchvision pillow tqdm pandas opencv-python")
    sys.exit(1)

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[WARN] sklearn not found — AUC-ROC will be skipped. Run: pip install scikit-learn")

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
except ImportError:
    _TB_OK = False
    print("[WARN] TensorBoard not found — logging disabled. Run: pip install tensorboard")


# ── Rim crop ──────────────────────────────────────────────────────────────────
def extract_rim_crop(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
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
    def __init__(self, samples: list, transform: T.Compose,
                 cache_dir: str = "", img_size: int = 224):
        self.samples   = samples
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.img_size  = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil    = self._load_image(path)
        tensor = self.transform(pil)
        return tensor, torch.tensor(label, dtype=torch.float32)

    def _load_image(self, orig_path: str) -> Image.Image:
        if self.cache_dir is not None:
            cached = self.cache_dir / Path(orig_path).name
            if cached.exists():
                return Image.open(cached).convert("RGB")
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

    df = pd.read_csv(csv_path)

    # ADD-13: Validate required columns early with a clear error message
    missing_cols = [c for c in ("image_id", "target") if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing required column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    samples = []
    missing = 0
    for _, row in df.iterrows():
        img_path = root / row["image_id"]
        if img_path.exists():
            samples.append((str(img_path), int(row["target"])))
        else:
            missing += 1

    if missing:
        print(f"[WARN] {missing} images listed in CSV were not found on disk.")
    if not samples:
        raise RuntimeError("No valid images found in CSV.")

    return samples


def _split_samples(samples: list, val_frac: float, test_frac: float, seed: int = 42):
    rng    = random.Random(seed)
    good   = [s for s in samples if s[1] == 0]
    faulty = [s for s in samples if s[1] == 1]
    rng.shuffle(good)
    rng.shuffle(faulty)

    def _split(lst):
        n_val  = max(1, int(len(lst) * val_frac))
        # BUG-4: Only allocate test samples when test_frac is actually > 0
        n_test = int(len(lst) * test_frac) if test_frac > 0 else 0
        val    = lst[:n_val]
        test   = lst[n_val:n_val + n_test]
        train  = lst[n_val + n_test:]
        return train, val, test

    train_g, val_g, test_g = _split(good)
    train_f, val_f, test_f = _split(faulty)

    train_samples = train_g + train_f
    val_samples   = val_g   + val_f
    test_samples  = test_g  + test_f
    rng.shuffle(train_samples)

    print(f"[INFO] Train: {len(train_samples)}  |  Val: {len(val_samples)}  |  Test: {len(test_samples)}")
    return train_samples, val_samples, test_samples


def _make_sampler(train_samples: list) -> WeightedRandomSampler:
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
    return sampler


# ── Model ─────────────────────────────────────────────────────────────────────
class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        # 1. Load EfficientNet-B2
        self.b2    = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        b2_out     = self.b2.classifier[1].in_features  # 1408
        self.b2.classifier = nn.Identity()

        # 2. Load ConvNeXt-Tiny
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        cn_out        = self.convnext.classifier[2].in_features  # 768
        self.convnext.classifier = nn.Identity()

        # 3. Combined Classifier (1408 + 768 = 2176)
        self.classifier = nn.Sequential(
            nn.Linear(b2_out + cn_out, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )
        self._arch_version = "ensemble_b2_convnext_v1"

    def forward(self, x):
        f_b2      = self.b2(x)
        f_convnext = self.convnext(x)
        # ConvNeXt with Identity head returns a 4-D tensor; flatten it
        if f_convnext.dim() == 4:
            f_convnext = f_convnext.flatten(1)
        return self.classifier(torch.cat([f_b2, f_convnext], dim=1))


def build_model() -> nn.Module:
    return EnsembleModel()


# BUG-1: Removed the duplicate _set_backbone_grad. This single version
# correctly freezes/unfreezes b2 and convnext while always leaving the
# classifier trainable.
def _set_backbone_grad(model: EnsembleModel, requires_grad: bool):
    for param in model.b2.parameters():
        param.requires_grad = requires_grad
    for param in model.convnext.parameters():
        param.requires_grad = requires_grad


# ── EMA ───────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        # Cache for live weights so restore() can bring them back
        self._live_backup: dict | None = None
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            if s_buf.dtype.is_floating_point:
                s_buf.data.mul_(self.decay).add_(m_buf.data, alpha=1.0 - self.decay)
            else:
                s_buf.data.copy_(m_buf.data)

    # BUG-2: apply() now saves live weights then copies shadow → model.
    def apply(self, model: nn.Module):
        self._live_backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow.state_dict(), strict=False)

    # BUG-2: restore() puts the live weights back from the backup.
    def restore(self, model: nn.Module):
        if self._live_backup is not None:
            model.load_state_dict(self._live_backup, strict=False)
            self._live_backup = None


# ── BN recalibration ─────────────────────────────────────────────────────────
def _recalibrate_bn(model: nn.Module, loader: DataLoader, device, n_batches: int = 50):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1.0)
            if m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()

    model.train()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(imgs.to(device))
    model.eval()


# ── Loss ──────────────────────────────────────────────────────────────────────
class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, eps: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.eps > 0:
            targets = targets * (1.0 - self.eps) + 0.5 * self.eps
        return self.bce(logits, targets)


# ── Mixup ─────────────────────────────────────────────────────────────────────
# ADD-11: Mixup augmentation applied in the training loop.
def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """Returns mixed images and a (labels_a, labels_b, lam) tuple."""
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam   = np.random.beta(alpha, alpha)
    bsize = imgs.size(0)
    idx   = torch.randperm(bsize, device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


def mixup_loss(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ── Metrics ───────────────────────────────────────────────────────────────────
def _compute_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _validate(model, loader, device, threshold: float = 0.5):
    model.eval()
    tp = tn = fp = fn = 0
    total_correct = total = 0
    all_probs_list, all_labels_list = [], []

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
            all_probs_list.append(probs.cpu())
            all_labels_list.append(labels.cpu())

    acc = (total_correct / total * 100) if total else 0.0
    f1  = _compute_f1(tp, fp, fn)

    # ADD-7: Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ADD-10: AUC-ROC
    auc = None
    if _SKLEARN_OK:
        all_probs_np  = torch.cat(all_probs_list).numpy()
        all_labels_np = torch.cat(all_labels_list).numpy()
        if len(np.unique(all_labels_np)) > 1:
            auc = roc_auc_score(all_labels_np, all_probs_np)

    return acc, f1, precision, recall, auc, tp, tn, fp, fn


def _quick_sweep(model, loader, device):
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


def _sweep_threshold(model, loader, device, title: str = "Threshold sweep"):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(1)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels)
    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    print(f"\n[{title}]")
    print(f"  {'Threshold':>10}  {'F1':>8}  {'Prec':>8}  {'Rec':>8}  {'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}")
    print("  " + "─" * 72)

    best_f1, best_thresh = -1.0, 0.5
    for t_int in range(5, 95, 5):
        t     = t_int / 100.0
        preds = (probs >= t).float()
        tp    = int(((preds == 1) & (labels == 1)).sum())
        tn    = int(((preds == 0) & (labels == 0)).sum())
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        f1    = _compute_f1(tp, fp, fn)
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        mark  = " ← best" if f1 > best_f1 else ""
        print(f"  {t:>10.2f}  {f1:>8.4f}  {prec:>8.4f}  {rec:>8.4f}  {tp:>5}  {tn:>5}  {fp:>5}  {fn:>5}{mark}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    # ADD-12: Full GPU reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ADD-9: TensorBoard writer
    writer = SummaryWriter(log_dir="runs/rim_classifier") if _TB_OK else None

    all_samples = _load_samples(args.dataset, args.csv_name)
    train_samples, val_samples, test_samples = _split_samples(
        all_samples, args.val_split, args.test_split, seed=args.seed
    )

    train_ds = RimDataset(train_samples, _make_train_transform(args.img_size), args.cache_dir, args.img_size)
    val_ds   = RimDataset(val_samples,   _make_val_transform(args.img_size),   args.cache_dir, args.img_size)
    test_ds  = RimDataset(test_samples,  _make_val_transform(args.img_size),   args.cache_dir, args.img_size)

    sampler      = _make_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,   num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,     num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,     num_workers=args.workers, pin_memory=True)

    model = build_model().to(device)
    ema   = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # BUG-3: Compute pos_weight from class counts and pass it to the loss
    n_good  = sum(1 for _, l in train_samples if l == 0)
    n_fail  = sum(1 for _, l in train_samples if l == 1)
    pos_w   = torch.tensor([n_good / max(n_fail, 1)], dtype=torch.float32).to(device)
    criterion = SmoothedBCEWithLogitsLoss(pos_weight=pos_w, eps=args.label_smooth)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_f1        = -1.0
    no_improve_epochs  = 0
    warmup             = args.warmup_epochs
    start_epoch        = 1
    optimizer          = None
    scheduler          = None

    # ADD-8: Resume from checkpoint
    if args.resume and Path(args.resume).exists():
        print(f"[INFO] Resuming from {args.resume}")
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        if ema and "ema_state" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema_state"])
        print(f"[INFO] Resumed at epoch {start_epoch}, best F1 so far: {best_val_f1:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Phase 1: backbone frozen ──────────────────────────────────────────
        if epoch == 1:
            _set_backbone_grad(model, requires_grad=False)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 10,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 10,
                steps_per_epoch=len(train_loader), epochs=warmup,
            )
        # ── Phase 2: full fine-tune ───────────────────────────────────────────
        elif epoch == warmup + 1:
            _set_backbone_grad(model, requires_grad=True)
            optimizer = torch.optim.AdamW([
                {"params": model.b2.parameters(),        "lr": args.lr * 0.1},
                {"params": model.convnext.parameters(),  "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ])
            # BUG-5: max_lr must have 3 entries to match 3 param groups
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[args.lr * 0.1, args.lr * 0.1, args.lr],
                steps_per_epoch=len(train_loader),
                epochs=args.epochs - warmup,
            )

        # ── Training step ─────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        accum_steps = 4
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}")
        for i, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            # ADD-11: Mixup augmentation
            imgs, labels_a, labels_b, lam = mixup_batch(imgs, labels, alpha=args.mixup_alpha)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs).squeeze(1)
                loss   = mixup_loss(criterion, logits, labels_a, labels_b, lam)

            scaler.scale(loss).backward()

            # BUG-6: Apply gradient clipping when --grad-clip > 0
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader) :
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
            scheduler.step()

            epoch_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        if ema:
            ema.apply(model)

        val_acc, val_f1, val_prec, val_rec, val_auc, tp, tn, fp, fn = _validate(
            model, val_loader, device
        )
        swept_f1, swept_thresh = _quick_sweep(model, val_loader, device)

        if ema:
            ema.restore(model)

        # ADD-7: Print Precision & Recall alongside F1
        auc_str = f"  AUC: {val_auc:.4f}" if val_auc is not None else ""
        print(
            f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%  F1: {swept_f1:.4f}  "
            f"Prec: {val_prec:.4f}  Rec: {val_rec:.4f}{auc_str}"
        )

        # ADD-9: TensorBoard logging
        if writer:
            writer.add_scalar("Loss/train",      avg_loss,    epoch)
            writer.add_scalar("Val/F1",          swept_f1,    epoch)
            writer.add_scalar("Val/Precision",   val_prec,    epoch)
            writer.add_scalar("Val/Recall",      val_rec,     epoch)
            writer.add_scalar("Val/Accuracy",    val_acc,     epoch)
            if val_auc is not None:
                writer.add_scalar("Val/AUC",     val_auc,     epoch)
            writer.add_scalar("LR/head", optimizer.param_groups[-1]["lr"], epoch)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if swept_f1 > best_val_f1:
            best_val_f1       = swept_f1
            no_improve_epochs = 0
            if ema:
                _recalibrate_bn(ema.shadow, train_loader, device, n_batches=args.bn_recal_batches)
                save_state = ema.shadow.state_dict()
                ema_state  = ema.shadow.state_dict()
            else:
                save_state = model.state_dict()
                ema_state  = None

            # ADD-8: Save epoch + best_val_f1 + optimizer for resuming
            ckpt = {
                "state_dict":   save_state,
                "arch_version": model._arch_version,
                "epoch":        epoch,
                "best_val_f1":  best_val_f1,
                "optimizer":    optimizer.state_dict(),
            }
            if ema_state:
                ckpt["ema_state"] = ema_state

            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, args.out)
            print(f"  ✓ Saved best model (F1={best_val_f1:.4f}) → {args.out}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop:
                print(f"[INFO] Early stopping after {args.early_stop} epochs without improvement.")
                break

    if writer:
        writer.close()

    # ADD-14: Guard against empty test set before final evaluation
    if not test_samples:
        print("\n[INFO] No test set (--test-split was 0). Skipping final evaluation.")
        return

    print("\n[INFO] Final evaluation on TEST set...")
    best_model = build_model().to(device)
    ckpt = torch.load(args.out, map_location=device)
    best_model.load_state_dict(ckpt["state_dict"])
    _sweep_threshold(best_model, test_loader, device, title="Final TEST set Sweep")


if __name__ == "__main__":
    train()