"""
evaluate.py
Offline batch evaluation — runs the inspector on a folder of labelled images
and prints a confusion matrix + per-image results.

Expected folder layout
─────────────────────
dataset/
  good/          ← "Good" label images
    img001.jpg
    ...
  faulty/        ← "Conditionally Faulty" label images
    img002.jpg
    ...

Usage
─────
  python evaluate.py --dataset dataset/ [--chip-threshold 200] [--save-vis results/]
"""

import cv2
import os
import argparse
from core.bottle_inspector import BottleInspector
from utils.visualizer import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--chip-threshold", type=int, default=200)
parser.add_argument("--baseline", default="assets/baseline.npy")
parser.add_argument("--save-vis", default=None,
                    help="If set, save annotated images to this directory.")
args = parser.parse_args()

inspector = BottleInspector(chip_threshold=args.chip_threshold,
                             baseline_path=args.baseline)
visualizer = Visualizer(
    chip_threshold=args.chip_threshold,
    rim_width_ratio=inspector.rim_width_ratio,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

results_log = []   # (filename, true_label, predicted, max_area)

for true_label, folder_name in [("PASS", "good"), ("FAIL", "faulty")]:
    folder = os.path.join(args.dataset, folder_name)
    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}, skipping.")
        continue

    for fname in sorted(os.listdir(folder)):
        if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
            continue
        path = os.path.join(folder, fname)
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"[WARN] Could not read {path}")
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        result = inspector.inspect(frame)
        predicted = result.status
        results_log.append((fname, true_label, predicted, result.max_chip_area))

        if args.save_vis:
            vis = visualizer.render(frame, result)
            os.makedirs(args.save_vis, exist_ok=True)
            out_name = f"{true_label}_{fname}"
            cv2.imwrite(os.path.join(args.save_vis, out_name), vis)

# ── Metrics ──────────────────────────────────────────────────────────────────
print(f"\n{'File':<30} {'True':>6} {'Pred':>6} {'MaxArea':>10}")
print("─" * 58)

tp = tn = fp = fn = no_bottle = 0
for fname, true_label, predicted, area in results_log:
    if predicted == "NO_BOTTLE":
        match = "~"
        no_bottle += 1
        # Treat undetected bottle as missed fault / false alarm on good
        if true_label == "FAIL":
            fn += 1
        elif true_label == "PASS":
            fp += 1
    else:
        match = "✓" if true_label == predicted else "✗"
        if true_label == "FAIL" and predicted == "FAIL":
            tp += 1
        elif true_label == "PASS" and predicted == "PASS":
            tn += 1
        elif true_label == "PASS" and predicted == "FAIL":
            fp += 1
        elif true_label == "FAIL" and predicted == "PASS":
            fn += 1

    print(f"{fname:<30} {true_label:>6} {predicted:>6} {area:>9.0f} px  {match}")

total = tp + tn + fp + fn
accuracy = (tp + tn) / total * 100 if total else 0
precision = tp / (tp + fp) * 100 if (tp + fp) else 0
recall = tp / (tp + fn) * 100 if (tp + fn) else 0

print(f"\n{'─'*58}")
print(f"  Total scored : {total}  (NO_BOTTLE: {no_bottle})")
print(f"  Accuracy     : {accuracy:.1f}%")
print(f"  Precision    : {precision:.1f}%  (of predicted FAIL, how many were truly FAIL)")
print(f"  Recall       : {recall:.1f}%   (of all true FAIL, how many were caught)")
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
