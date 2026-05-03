"""
Krones Vision AI Challenge — Bottle Sealing Surface Inspector
Entry point: opens camera (or video file), runs the inspection pipeline frame-by-frame.
"""

import cv2
import argparse
from core.bottle_inspector import BottleInspector
from utils.visualizer import Visualizer
from utils.smoother import Smoother

# ── CLI args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Bottle Sealing Surface Inspector")
parser.add_argument("--source", default=0,
                    help="Camera index (0) or path to video/image file")
parser.add_argument("--chip-threshold", type=int, default=200,
                    help="Min contour area (px²) to count as a chip (default: 200)")
parser.add_argument("--calibrate", action="store_true",
                    help="Run calibration mode using 'Good' samples first")
parser.add_argument("--baseline", default="assets/baseline.npy",
                    help="Path to saved baseline profile (numpy array)")
args = parser.parse_args()

# ── Components ───────────────────────────────────────────────────────────────
inspector  = BottleInspector(chip_threshold=args.chip_threshold,
                              baseline_path=args.baseline)
visualizer = Visualizer()
smoother   = Smoother(alpha=0.6)          # smooths detected circle (cx, cy, r)

# ── Source ───────────────────────────────────────────────────────────────────
source = int(args.source) if str(args.source).isdigit() else args.source
cap    = cv2.VideoCapture(source)

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

    # ── Inspection ────────────────────────────────────────────────────────
    result = inspector.inspect(frame)

    # Smooth the rim circle to avoid jitter
    if result.circle is not None:
        cx, cy, r = result.circle
        cx, cy, r = smoother.update((cx, cy, r))
        result.circle = (int(cx), int(cy), int(r))

    # ── Visualisation ─────────────────────────────────────────────────────
    display = visualizer.render(frame, result)
    cv2.imshow("Bottle Inspector — Krones Challenge", display)

    # ── Key bindings ──────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        inspector.add_to_baseline(frame)
        print("[INFO] Frame added to baseline.")
    elif key == ord("r"):
        inspector.reset_baseline()
        print("[INFO] Baseline reset.")

cap.release()
cv2.destroyAllWindows()
