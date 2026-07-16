"""
predict.py
──────────────────────────────────────────────────────────────────────────────
Runs the trained ensemble (EfficientNet-B2 + ResNet-50 + ConvNeXt-Tiny)
on unlabelled test images and writes a submission CSV: image_id, target

Architecture matches train_cnn_v2.py exactly:
  - CBAM uses ch_fc1 / ch_fc2 (shared MLP, not ch_mlp Sequential)
  - ConvNeXtTinyWithCBAM uses nn.ModuleList(stages), not stage0…stage7
  - AttentionPool2d replaces AdaptiveAvgPool2d in every backbone
  - Dual output heads: binary_head + severity_head
  - Inference uses binary_head only

Usage:
    python predict.py
    python predict.py --threshold 0.42
    python predict.py --model assets/model_v5.pt \\
                      --test-dir path/to/test_images \\
                      --threshold 0.42 \\
                      --tta
    python predict.py --calibrator assets/calibrator.pkl --threshold 0.50
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Run: pip install torch torchvision tqdm pandas opencv-python")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Rim defect inference — v5 final")
parser.add_argument("--model",       default="assets/model_v5.pt",
                    help="Path to trained checkpoint (.pt)")
parser.add_argument("--calibrator",  default="",
                    help="Optional path to Platt scaling calibrator (.pkl). "
                         "When provided, calibrated probabilities are used instead "
                         "of raw sigmoid outputs.")
parser.add_argument("--test-dir",    default="dataset/test_images",
                    help="Folder containing unlabelled test images")
parser.add_argument("--cache-dir",   default="dataset/dataset_cropped_test/",
                    help="Pre-cropped image cache (same logic as training)")
parser.add_argument("--out",         default="submission.csv",
                    help="Output CSV filename")
parser.add_argument("--threshold",   type=float, default=0.5,
                    help="Probability threshold — use the value printed at end of training")
parser.add_argument("--batch",       type=int,   default=32)
parser.add_argument("--workers",     type=int,   default=4)
parser.add_argument("--img-size",    type=int,   default=260,
                    help="Must match the --img-size used during training")
parser.add_argument("--tta",         action="store_true", default=False,
                    help="Enable 6-view test-time augmentation (flips + 90°/270° rotations). "
                         "Adds ~6× inference time but reduces prediction variance.")
args = parser.parse_args()


# ── Rim crop (identical to train_cnn_v2.py) ───────────────────────────────────
def extract_rim_crop(
    img_bgr: np.ndarray, size: int = 260, wide_scale: float = 1.0
) -> np.ndarray:
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
        margin = int((rim_w + 10) * wide_scale)
        h, w   = gray.shape
        x1 = max(0, cx - r - margin); y1 = max(0, cy - r - margin)
        x2 = min(w, cx + r + margin); y2 = min(h, cy + r + margin)
        # FIX: Hough Circles Empty Crop — clamp can produce zero-area crop
        # if circle center is outside image. Check and fall back to full image.
        if x2 <= x1 or y2 <= y1:
            crop = img_bgr
        else:
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
        if self.cache_dir:
            cached = self.cache_dir / Path(path).name
            if cached.exists():
                img = Image.open(cached).convert("RGB")
                return self.transform(img), path
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        crop = extract_rim_crop(img_bgr, self.img_size, wide_scale=1.0)
        return self.transform(Image.fromarray(crop)), path


# ── AttentionPool2d (identical to train_cnn_v2.py) ────────────────────────────
class AttentionPool2d(nn.Module):
    """
    Learned spatial pooling — replaces GlobalAveragePooling in each backbone.
    A 1×1 conv scores each spatial position; softmax normalises the scores
    across H×W; the output is a weighted sum focused on informative regions.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        weights = torch.softmax(
            self.score(x).view(B, 1, H * W), dim=2
        ).view(B, 1, H, W)
        return (x * weights).sum(dim=(2, 3))   # (B, C)


# ── CBAM (identical to train_cnn_v2.py) ──────────────────────────────────────
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Channel attention uses a shared two-layer MLP (ch_fc1 / ch_fc2).
    Spatial attention uses a 7×7 conv (sp_conv).
    Both are applied sequentially and multiplicatively.
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.ch_fc1  = nn.Linear(channels, mid,      bias=False)
        self.ch_fc2  = nn.Linear(mid,      channels, bias=False)
        self.sp_conv = nn.Conv2d(
            2, 1, spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg  = x.mean(dim=(2, 3))
        mx   = x.amax(dim=(2, 3))
        gate = torch.sigmoid(
            self.ch_fc2(F.relu(self.ch_fc1(avg), inplace=True)) +
            self.ch_fc2(F.relu(self.ch_fc1(mx),  inplace=True))
        )
        x = x * gate[:, :, None, None]
        # Spatial attention
        sp_in = torch.stack([x.mean(dim=1), x.amax(dim=1)], dim=1)
        x     = x * torch.sigmoid(self.sp_conv(sp_in))
        return x


# ── Backbone wrappers (identical to train_cnn_v2.py) ─────────────────────────
class EfficientNetB2WithCBAM(nn.Module):
    def __init__(self, use_grad_checkpoint: bool = False):
        super().__init__()
        base          = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        self.features = base.features
        self.cbam     = CBAM(1408, reduction=16)
        self.pool     = AttentionPool2d(1408)
        self.out_dim  = 1408

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.cbam(feat)
        return self.pool(feat)   # (B, 1408)


class ResNet50WithCBAM(nn.Module):
    def __init__(self, use_grad_checkpoint: bool = False):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.cbam    = CBAM(2048, reduction=16)
        self.pool    = AttentionPool2d(2048)
        self.out_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        return self.pool(x)   # (B, 2048)


class ConvNeXtTinyWithCBAM(nn.Module):
    def __init__(self, use_grad_checkpoint: bool = False):
        super().__init__()
        base = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        # Use ModuleList — matches the training script exactly
        self.stages  = nn.ModuleList(list(base.features))
        self.cbam    = CBAM(768, reduction=16)
        self.norm    = nn.LayerNorm(768)
        self.pool    = AttentionPool2d(768)
        self.out_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self.cbam(x)
        x = self.pool(x)    # (B, 768)
        return self.norm(x)


# ── Ensemble model (identical to train_cnn_v2.py) ────────────────────────────
class EnsembleModelV5(nn.Module):
    """
    Three-backbone ensemble with fusion attention gate and dual output heads.
    At inference only binary_head is used; severity_head exists so the
    state_dict loads without errors.
    """
    _arch_version = "ensemble_b2_resnet50_convnext_cbam_v5_final"

    def __init__(self, use_grad_checkpoint: bool = False):
        super().__init__()
        self.b2      = EfficientNetB2WithCBAM()
        self.resnet  = ResNet50WithCBAM()
        self.convnxt = ConvNeXtTinyWithCBAM()

        combined = self.b2.out_dim + self.resnet.out_dim + self.convnxt.out_dim  # 4224
        mid      = max(combined // 16, 32)
        self.fusion_attn = nn.Sequential(
            nn.Linear(combined, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, combined, bias=False),
            nn.Sigmoid(),
        )

        # Shared trunk: 4224 → 512 → 128
        self.classifier = nn.Sequential(
            nn.Linear(combined, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.30),
        )

        # Dual output heads — both must exist to match the saved state_dict
        self.binary_head   = nn.Linear(128, 1)   # used at inference
        self.severity_head = nn.Linear(128, 1)   # training only, ignored here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the binary logit (scalar per image)."""
        f_b2 = self.b2(x)
        f_rn = self.resnet(x)
        f_cx = self.convnxt(x)
        feat = torch.cat([f_b2, f_rn, f_cx], dim=1)
        feat = feat * self.fusion_attn(feat)
        feat = self.classifier(feat)
        return self.binary_head(feat)   # (B, 1)


def build_model() -> EnsembleModelV5:
    return EnsembleModelV5()


# ── Inference helpers with optional TTA ───────────────────────────────────────
@torch.no_grad()
def batch_logits(
    model: nn.Module,
    imgs: torch.Tensor,
    use_tta: bool = False,
) -> torch.Tensor:
    """Return per-image binary logits (before sigmoid)."""
    if use_tta:
        return torch.stack([
            model(imgs).squeeze(1),
            model(imgs.flip(-1)).squeeze(1),
            model(imgs.flip(-2)).squeeze(1),
            model(torch.rot90(imgs, 1, [2, 3])).squeeze(1),
            model(torch.rot90(imgs, 2, [2, 3])).squeeze(1),
            model(torch.rot90(imgs, 3, [2, 3])).squeeze(1),
        ]).mean(0)
    return model(imgs).squeeze(1)


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    imgs: torch.Tensor,
    use_tta: bool = False,
) -> torch.Tensor:
    """Return per-image probabilities."""
    return torch.sigmoid(batch_logits(model, imgs, use_tta=use_tta))


def _resolve_calibrator_path(model_path: Path, calibrator_arg: str) -> str:
    """Pick SWA-specific calibrator when the checkpoint is SWA."""
    if not calibrator_arg:
        return calibrator_arg
    path = Path(calibrator_arg)
    is_swa_ckpt = "_swa" in model_path.stem.lower()
    if is_swa_ckpt:
        swa_cal = Path(str(path).replace(".pkl", "_swa.pkl"))
        if swa_cal.exists():
            return str(swa_cal)
        if path.exists():
            print(
                f"[WARN] SWA checkpoint but no {swa_cal.name}; "
                f"using {path.name}."
            )
    return calibrator_arg


# ── Main ──────────────────────────────────────────────────────────────────────
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device      : {device}")
    print(f"[INFO] Model       : {args.model}")
    print(f"[INFO] Threshold   : {args.threshold}")
    print(f"[INFO] TTA         : {'enabled (6 views)' if args.tta else 'disabled'}")

    # ── Collect image paths ────────────────────────────────────────────────
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted([
        str(p) for p in Path(args.test_dir).iterdir()
        if p.suffix.lower() in exts
    ])
    if not paths:
        print(f"[WARN] No images found in {args.test_dir}")
        print("[WARN] Writing empty submission CSV to avoid Kaggle submission error.")
        # FIX: Empty Submission File — always write a valid CSV header so Kaggle
        # doesn't fail during hidden initialization with an empty test folder.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["image_id", "target"]).to_csv(args.out, index=False)
        print(f"[DONE] Empty submission → {args.out}")
        return

    print(f"[INFO] Found {len(paths)} test image(s)")

    # ── Build and load model ───────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.model}\n"
            "Run train_cnn_v2.py first, or pass --model <path>."
        )

    # FIX: SWA Checkpoint Computed but Ignored — if the path is the standard
    # checkpoint but a _swa.pt exists nearby, offer to use it instead.
    swa_path = model_path.with_name(model_path.stem + "_swa.pt")
    if swa_path.exists() and "swa" not in str(model_path).lower():
        print(f"[HINT] SWA checkpoint found at {swa_path} — typically better "
              f"than {model_path.name}.")

    print("[INFO] Loading model weights...")
    model      = build_model().to(device)
    checkpoint = torch.load(str(model_path), map_location=device)
    state      = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    arch = checkpoint.get("arch_version", "unknown")
    ep   = checkpoint.get("epoch", "?")
    f1   = checkpoint.get("best_val_f1", float("nan"))
    bt   = checkpoint.get("best_threshold", args.threshold)
    print(f"[INFO] Checkpoint  : epoch {ep}  |  val F1: {f1:.4f}  |  "
          f"arch: {arch}")
    if bt != args.threshold:
        print(f"[HINT] Training recorded best threshold = {bt:.2f}. "
              f"You passed --threshold {args.threshold}. "
              f"Consider using --threshold {bt:.2f}.")

    # ── Load calibrator (after checkpoint — SWA path resolved first) ───────
    calibrator = None
    calibrator_path = _resolve_calibrator_path(model_path, args.calibrator)
    if calibrator_path:
        print(f"[INFO] Calibrator  : {calibrator_path}")
    if calibrator_path and Path(calibrator_path).exists():
        import pickle
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)
        print("[INFO] Platt scaling calibrator loaded — raw logits will be "
              "calibrated before thresholding.")
    elif calibrator_path:
        print(f"[WARN] Calibrator not found at '{calibrator_path}' — "
              "falling back to raw sigmoid probabilities.")

    # ── Transform (deterministic — same as val transform in training) ──────
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TestDataset(paths, transform, args.cache_dir, args.img_size)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch,
        shuffle     = False,
        num_workers = args.workers,
        pin_memory  = device.type == "cuda",
    )

    # ── Inference loop ─────────────────────────────────────────────────────
    image_ids, all_probs = [], []

    with torch.no_grad():
        for imgs, img_paths in tqdm(loader, desc="Predicting"):
            imgs = imgs.to(device, non_blocking=True)
            raw_logits = batch_logits(model, imgs, use_tta=args.tta).cpu().numpy()

            if calibrator is not None:
                probs = calibrator.predict_proba(
                    raw_logits.reshape(-1, 1)
                )[:, 1]
            else:
                probs = 1.0 / (1.0 + np.exp(-raw_logits))

            for path, prob in zip(img_paths, probs):
                image_ids.append(Path(path).name)
                all_probs.append(float(prob))

    # ── Apply threshold and write CSV ──────────────────────────────────────
    predictions = [1 if p >= args.threshold else 0 for p in all_probs]

    df = pd.DataFrame({
        "image_id": image_ids,
        "target":   predictions,
        "prob":     [round(p, 4) for p in all_probs],   # helpful for debugging
    })
    # Submission typically wants only image_id + target
    df[["image_id", "target"]].to_csv(args.out, index=False)

    n_good  = predictions.count(0)
    n_fault = predictions.count(1)
    print(f"\n[DONE] Predictions saved to: {args.out}")
    print(f"       Good (0): {n_good}  |  Faulty (1): {n_fault}  "
          f"|  Total: {len(predictions)}")
    print(f"       Prob  min={min(all_probs):.3f}  "
          f"max={max(all_probs):.3f}  "
          f"mean={sum(all_probs)/len(all_probs):.3f}")


if __name__ == "__main__":
    predict()