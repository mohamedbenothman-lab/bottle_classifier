import argparse
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="High-speed Parallel Rim Preprocessing")
parser.add_argument("--dataset",   default="C:/Users/mdhia/OneDrive/Bureau/dataset/train_images", help="Source images")
parser.add_argument("--csv-name",  default="train.csv",        help="CSV filename")
parser.add_argument("--cache-dir", default="C:/Users/mdhia/OneDrive/Bureau/dataset/dataset_cropped/",  help="Output directory")
parser.add_argument("--img-size",  type=int, default=260,       help="Crop size")
parser.add_argument("--workers",   type=int, default=None,      help="CPU cores to use (None = All)")
args = parser.parse_args()

# ── Processing Logic ──────────────────────────────────────────────────────────
def extract_rim_crop(img_bgr, size: int):
    """
    Detects the bottle rim via Hough circles and crops.
    Returns the cropped BGR image and a boolean indicating if a circle was found.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=200, param1=60, param2=35,
        minRadius=100, maxRadius=600,
    )

    found_circle = False
    if circles is not None:
        found_circle = True
        cx, cy, r = np.round(circles[0, 0]).astype(int)
        rim_w  = max(20, int(r * 0.18))
        margin = rim_w + 10
        h, w   = gray.shape
        x1, y1 = max(0, cx - r - margin), max(0, cy - r - margin)
        x2, y2 = min(w, cx + r + margin), min(h, cy + r + margin)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = img_bgr

    return cv2.resize(crop, (size, size)), found_circle

def process_single_image(task_info):
    """
    Worker function executed in parallel.
    """
    src_path, dest_path, img_size = task_info
    
    # Skip if already exists
    if dest_path.exists():
        return "skipped"

    img = cv2.imread(str(src_path))
    if img is None:
        return "failed"

    try:
        crop, found = extract_rim_crop(img, img_size)
        cv2.imwrite(str(dest_path), crop)
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

    # Prepare tasks: (Source, Destination, Size)
    tasks = []
    for _, row in df.iterrows():
        tasks.append((
            src / row["image_id"], 
            dst / Path(row["image_id"]).name, 
            args.img_size
        ))

    print(f"[INFO] Dataset: {len(tasks)} images")
    print(f"[INFO] Using ProcessPoolExecutor with {args.workers or 'all'} cores...")

    stats = {"success": 0, "fallback": 0, "skipped": 0, "failed": 0}

    # Parallel Execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # map() handles the distribution of tasks to your CPU cores
        for result in tqdm(executor.map(process_single_image, tasks), total=len(tasks)):
            stats[result] += 1

    # Summary
    processed = stats["success"] + stats["fallback"]
    print(f"\n{'─'*40}\n   PREPROCESSING COMPLETE\n{'─'*40}")
    print(f"  Total Images  : {len(tasks)}")
    print(f"  Newly Cropped : {stats['success']}")
    print(f"  Fallbacks     : {stats['fallback']} (Full frame used)")
    print(f"  Already Cached: {stats['skipped']}")
    print(f"  Failed        : {stats['failed']}")
    print(f"  Output Path   : {dst.resolve()}")
    
    if stats['fallback'] > (processed * 0.15):
        print("\n[!] WARNING: High fallback rate. Your Hough Circle parameters may need adjustment.")

if __name__ == "__main__":
    main()