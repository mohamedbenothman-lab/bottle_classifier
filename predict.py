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
def extract_rim_crop(img_bgr, size=260):
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=200, param1=60, param2=35,
                               minRadius=100, maxRadius=600)
    if circles is not None:
        cx, cy, r = np.round(circles[0, 0]).astype(int)
        rim_w  = max(20, int(r * 0.18))
        margin = rim_w + 10
        h, w   = gray.shape
        x1, y1 = max(0, cx-r-margin), max(0, cy-r-margin)
        x2, y2 = min(w, cx+r+margin), min(h, cy+r+margin)
        crop   = img_bgr[y1:y2, x1:x2]
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
        crop = extract_rim_crop(img_bgr, self.img_size)
        return self.transform(Image.fromarray(crop)), path


# ── Model (must match train_cnn.py exactly) ───────────────────────────────────
# Replace the build_model() function in compare.py AND predict.py
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