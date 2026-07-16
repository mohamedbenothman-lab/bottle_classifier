"""
Krones Vision AI Challenge — Bottle Sealing Surface Inspector
Entry point: opens camera (or video file), runs the inspection pipeline frame-by-frame.
"""

import cv2
import os
import argparse
from pathlib import Path
from core.bottle_inspector import BottleInspector
from utils.visualizer import Visualizer
from utils.smoother import Smoother

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ── CLI args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Bottle Sealing Surface Inspector")
parser.add_argument("--source", default=0,
                    help="Camera index (0) or path to video/image file")
parser.add_argument("--chip-threshold", type=int, default=200,
                    help="Min contour area (px²) to count as a chip (default: 200)")
parser.add_argument("--calibrate", action="store_true",
                    help="Pre-build baseline from --calibrate-dir (or --source if a folder)")
parser.add_argument("--calibrate-dir", default=None,
                    help="Folder of 'Good' images for --calibrate (default: --source if directory)")
parser.add_argument("--baseline", default="assets/baseline.npy",
                    help="Path to saved baseline profile (numpy array)")
args = parser.parse_args()

# ── Components ───────────────────────────────────────────────────────────────
inspector = BottleInspector(
    chip_threshold=args.chip_threshold,
    baseline_path=args.baseline,
)
visualizer = Visualizer(
    chip_threshold=args.chip_threshold,
    rim_width_ratio=inspector.rim_width_ratio,
)
smoother = Smoother(alpha=0.6)


def _calibrate_from_dir(directory: str) -> int:
    """Load all images from a folder into the baseline. Returns count added."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {directory}")

    count = 0
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in IMG_EXTS:
            continue
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"[WARN] Could not read {path}")
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        inspector.add_to_baseline(frame)
        count += 1
    return count


# ── Optional offline calibration ─────────────────────────────────────────────
if args.calibrate:
    cal_dir = args.calibrate_dir
    if cal_dir is None and isinstance(args.source, str) and not str(args.source).isdigit():
        if os.path.isdir(args.source):
            cal_dir = args.source
    if cal_dir:
        n = _calibrate_from_dir(cal_dir)
        print(f"[INFO] Baseline built from {n} image(s) in {cal_dir}.")
    else:
        print(
            "[INFO] Calibration mode: press C on good frames during the live loop "
            "(or pass --calibrate-dir <folder>)."
        )

# ── Source ───────────────────────────────────────────────────────────────────
source = int(args.source) if str(args.source).isdigit() else args.source
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source: {args.source}")

print("Press  Q  to quit |  C  to capture baseline frame |  R  to reset baseline")

while True:
    ret, frame = cap.read()
    if not ret:
        # Loop video files; stop on camera disconnect
        if isinstance(source, str):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        break

    # Detect circle once, then smooth before inspection so masks match overlay
    raw_circle = inspector.detect_circle(frame)

    display_circle = None
    if raw_circle is not None:
        cx, cy, r = smoother.update(raw_circle)
        display_circle = (int(cx), int(cy), int(r))
    else:
        smoother.reset()

    result = inspector.inspect(frame, circle=display_circle)

    display = visualizer.render(frame, result)
    cv2.imshow("Bottle Inspector — Krones Challenge", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        inspector.add_to_baseline(frame)
        print(f"[INFO] Frame added to baseline (n={inspector.baseline_frame_count}).")
    elif key == ord("r"):
        inspector.reset_baseline()
        smoother.reset()
        print("[INFO] Baseline reset.")

cap.release()
cv2.destroyAllWindows()
