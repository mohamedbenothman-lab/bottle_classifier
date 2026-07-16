import argparse
import cv2
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="High-speed Parallel Rim Preprocessing")
parser.add_argument("--dataset",
    default="C:/Users/mdhia/OneDrive/Bureau/dataset/train_images")
parser.add_argument("--csv",
    default="C:/Users/mdhia/OneDrive/Bureau/dataset/train_images/train.csv")
parser.add_argument("--cache-dir",
    default="C:/Users/mdhia/OneDrive/Bureau/dataset/dataset_cropped_v3/")
parser.add_argument("--img-size",  type=int, default=384)
parser.add_argument("--workers",   type=int, default=None)
parser.add_argument("--zip",       action="store_true", help="Zip the cropped images directory after completion")
parser.add_argument("--zip-path",  default=None, help="Custom path for the output zip file")

args = parser.parse_args()

# ── Processing Logic ──────────────────────────────────────────────────────────
def extract_rim_crop(img_bgr, size: int,
                     wide_scale: float = 1.3):
    """
    Images are grayscale on disk but loaded as BGR
    (all 3 channels are identical).
    - Circle detection on the single gray channel
    - Crop the BGR image
    - Save as BGR (identical R=G=B channels)
    - The model will receive a 3-channel tensor
      where R=G=B — this is fine, ImageNet
      normalization handles it correctly
    """
    # Since all channels are equal, just take one
    gray = img_bgr[:, :, 0]

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=200, param1=60, param2=35,
        minRadius=100, maxRadius=600,
    )

    if circles is not None:
        cx, cy, r = np.round(
            circles[0, 0]).astype(int)
        rim_w  = max(20, int(r * 0.18))
        margin = int((rim_w + 10) * wide_scale)
        h, w   = gray.shape
        x1 = max(0, cx - r - margin)
        y1 = max(0, cy - r - margin)
        x2 = min(w, cx + r + margin)
        y2 = min(h, cy + r + margin)
        # FIX: Guard against zero-area crop from out-of-bounds circle center
        if x2 <= x1 or y2 <= y1:
            crop  = img_bgr
            found = False
        else:
            # Crop the 3-channel image (R=G=B)
            crop  = img_bgr[y1:y2, x1:x2]
            found = True
    else:
        crop  = img_bgr
        found = False

    resized = cv2.resize(
        crop, (size, size),
        interpolation=cv2.INTER_LINEAR)

    return resized, found


def process_single_image(task_info):
    src_path, dest_path, img_size = task_info

    if dest_path.exists():
        return "skipped"

    # IMREAD_COLOR forces 3 channels even for grayscale
    # → model always gets (3, H, W) tensors
    img_bgr = cv2.imread(
        str(src_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return "failed"

    try:
        cropped, found = extract_rim_crop(
            img_bgr, img_size, wide_scale=1.3)
        cv2.imwrite(str(dest_path), cropped)
        return "success" if found else "fallback"
    except Exception as e:
        print(f"[ERR] {src_path.name}: {e}")
        return "failed"


# ── Main Entry ────────────────────────────────────────────────────────────────
def main():
    src      = Path(args.dataset)
    dst      = Path(args.cache_dir)
    csv_path = Path(args.csv)

    # ── Validate ───────────────────────────────────────────────────────────
    if not csv_path.exists():
        print(f"[ERROR] CSV not found      : {csv_path}")
        return
    if not src.exists():
        print(f"[ERROR] Dataset not found  : {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    print(f"\n{'─'*55}")
    print(f"  Rim Preprocessor V6  (Grayscale input)")
    print(f"{'─'*55}")
    print(f"  CSV        : {csv_path}")
    print(f"  Images     : {src}")
    print(f"  Output     : {dst}")
    print(f"  Size       : {args.img_size}×{args.img_size}")
    print(f"  Mode       : Grayscale → saved as 3-ch BGR")
    print(f"               (R=G=B, model sees 3-ch tensor)")
    print(f"  Workers    : "
          f"{'auto' if args.workers is None else args.workers}")
    print(f"  Total rows : {len(df)}")
    print(f"{'─'*55}\n")

    # ── Verify a sample image ──────────────────────────────────────────────
    # Peek at first image to confirm grayscale
    sample_id   = str(df.iloc[0]["image_id"])
    sample_path = src / sample_id
    if sample_path.exists():
        sample = cv2.imread(
            str(sample_path), cv2.IMREAD_UNCHANGED)
        if sample is not None:
            if len(sample.shape) == 2:
                print(f"[INFO] Confirmed: images are "
                      f"grayscale (single channel)")
            elif len(sample.shape) == 3:
                ch = sample.shape[2]
                # Check if all channels are identical
                if ch == 3:
                    diff = (
                        np.abs(sample[:,:,0].astype(int)
                               - sample[:,:,1].astype(int))
                        .max())
                    if diff == 0:
                        print(f"[INFO] Confirmed: images are "
                              f"grayscale stored as 3-ch "
                              f"(R=G=B, diff=0)")
                    else:
                        print(f"[WARN] Images appear to have "
                              f"real color (max ch diff={diff})"
                              f" — preprocessor still works")
            print(f"[INFO] Sample shape : {sample.shape}")
            print(f"[INFO] Sample dtype : {sample.dtype}\n")

    # ── Build task list ────────────────────────────────────────────────────
    tasks   = []
    missing = 0

    for _, row in df.iterrows():
        img_id   = str(row["image_id"])
        src_path = src / img_id

        if not src_path.exists():
            src_path = src / Path(img_id).name
        if not src_path.exists():
            missing += 1
            continue

        dest_path = dst / Path(img_id).name
        tasks.append(
            (src_path, dest_path, args.img_size))

    if missing:
        print(f"[WARN] {missing} image(s) not found "
              "in dataset dir — skipped.")
    if not tasks:
        print("[ERROR] No valid tasks. "
              "Check --dataset path.")
        return

    print(f"[INFO] Tasks to process : {len(tasks)}\n")

    # ── Process ────────────────────────────────────────────────────────────
    stats = {
        "success":  0,
        "fallback": 0,
        "skipped":  0,
        "failed":   0,
    }

    with ProcessPoolExecutor(
            max_workers=args.workers) as executor:
        for result in tqdm(
                executor.map(
                    process_single_image, tasks),
                total=len(tasks),
                desc="Preprocessing"):
            stats[result] += 1

    # ── Summary ────────────────────────────────────────────────────────────
    total_done = (stats["success"]
                  + stats["fallback"]
                  + stats["skipped"])
    print(f"\n{'─'*55}")
    print(f"  Preprocessing Complete")
    print(f"{'─'*55}")
    print(f"  Rim found   (success)  : {stats['success']}")
    print(f"  No rim      (fallback) : {stats['fallback']}")
    print(f"  Already done(skipped)  : {stats['skipped']}")
    print(f"  Failed                 : {stats['failed']}")
    print(f"  Total processed        : {total_done}")
    print(f"  Output dir             : {dst}")
    print(f"{'─'*55}\n")

    # ── Zipping ────────────────────────────────────────────────────────────
    if args.zip:
        zip_base = args.zip_path
        if not zip_base:
            zip_base = str(dst).rstrip("/\\")
        
        # Remove trailing .zip if present, make_archive appends it automatically
        if zip_base.lower().endswith(".zip"):
            zip_base = zip_base[:-4]
            
        print(f"Creating zip archive of: {dst}")
        try:
            archive_path = shutil.make_archive(zip_base, 'zip', str(dst))
            print(f"[INFO] Successfully created zip archive: {archive_path}\n")
        except Exception as e:
            print(f"[ERROR] Failed to create zip archive: {e}\n")


if __name__ == "__main__":
    main()