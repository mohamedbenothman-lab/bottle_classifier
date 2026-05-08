import argparse
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="High-speed Parallel Rim Preprocessing")
parser.add_argument("--dataset",   default="C:/Users/mdhia/OneDrive/Bureau/dataset/train_images")
parser.add_argument("--csv-name",  default="train.csv")
parser.add_argument("--cache-dir", default="C:/Users/mdhia/OneDrive/Bureau/dataset/dataset_cropped/")
parser.add_argument("--img-size",  type=int, default=260) # Native B2 size
parser.add_argument("--workers",   type=int, default=None)
args = parser.parse_args()

# ── Processing Logic ──────────────────────────────────────────────────────────
def extract_rim_crop(img, size: int, wide_scale: float = 1.3):
    # Ensure grayscale for circle detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=200, param1=60, param2=35,
        minRadius=100, maxRadius=600,
    )

    if circles is not None:
        cx, cy, r = np.round(circles[0, 0]).astype(int)
        rim_w  = max(20, int(r * 0.18))
        # [V5-5] Matches wide_crop_scale for spatial context
        margin = int((rim_w + 10) * wide_scale)
        h, w   = gray.shape
        x1, y1 = max(0, cx - r - margin), max(0, cy - r - margin)
        x2, y2 = min(w, cx + r + margin), min(h, cy + r + margin)
        crop = gray[y1:y2, x1:x2]
        found = True
    else:
        crop = gray
        found = False

    # Resize to B2 native size (260)
    resized = cv2.resize(crop, (size, size))
    
    # Force 3-channel grayscale (R=G=B) so PIL/Torchvision doesn't flip colors
    final_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return final_rgb, found

def process_single_image(task_info):
    src_path, dest_path, img_size = task_info
    if dest_path.exists(): return "skipped"

    # IMREAD_UNCHANGED keeps it grayscale if it's 8-bit BW on disk
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None: return "failed"

    try:
        # Use 1.3 to match args.wide_crop_scale in train_cnn_v2.py
        final_rgb, found = extract_rim_crop(img, img_size, wide_scale=1.3)
        
        # Save (cv2.imwrite swaps R/B, but since R=G=B, it stays identical)
        cv2.imwrite(str(dest_path), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
        return "success" if found else "fallback"
    except Exception:
        return "failed"

# ── Main Entry ────────────────────────────────────────────────────────────────
def main():
    src = Path(args.dataset)
    dst = Path(args.cache_dir)
    csv_path = src / args.csv_name

    if not csv_path.exists():
        print(f"[ERROR] CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    dst.mkdir(parents=True, exist_ok=True)

    tasks = [(src / row["image_id"], dst / Path(row["image_id"]).name, args.img_size) for _, row in df.iterrows()]

    print(f"[INFO] Processing {len(tasks)} images...")
    stats = {"success": 0, "fallback": 0, "skipped": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(process_single_image, tasks), total=len(tasks)):
            stats[result] += 1

    print(f"\nPreprocessing Complete. Success: {stats['success']} | Fallback: {stats['fallback']}")

if __name__ == "__main__":
    main()