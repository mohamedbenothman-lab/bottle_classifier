"""
core/bottle_inspector.py
The main inspection pipeline:
  1. Locate bottle rim via Hough Circle Transform
  2. Mask to the annular ROI (rim only)
  3. Diff against a learned baseline (or simple adaptive threshold)
  4. Find contours → classify chips by area
  5. Return an InspectionResult dataclass
"""

import cv2
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ── Result dataclass ─────────────────────────────────────────────────────────
@dataclass
class Chip:
    contour: np.ndarray
    area: float
    bounding_box: Tuple[int, int, int, int]   # x, y, w, h


@dataclass
class InspectionResult:
    status: str                          # "PASS" | "FAIL" | "NO_BOTTLE"
    circle: Optional[Tuple[int, int, int]]  # (cx, cy, radius) in px
    chips: List[Chip] = field(default_factory=list)
    max_chip_area: float = 0.0
    rim_mask: Optional[np.ndarray] = None
    diff_map: Optional[np.ndarray] = None


# ── Inspector ────────────────────────────────────────────────────────────────
class BottleInspector:
    """
    Detects chips / residue on a glass bottle's sealing surface (top rim).

    Parameters
    ----------
    chip_threshold   : int   Minimum contour area (px²) to flag as a chip (default 200).
    rim_width_ratio  : float Fraction of the detected radius used as annulus width (default 0.15).
    baseline_path    : str   Path to a .npy file storing the mean baseline image.
    hough_dp         : float Inverse ratio of accumulator resolution (HoughCircles).
    hough_min_dist   : int   Min distance between detected circle centres.
    hough_param1     : int   Upper Canny threshold.
    hough_param2     : int   Accumulator threshold (lower → more detections).
    hough_min_radius : int   Min expected bottle radius in pixels.
    hough_max_radius : int   Max expected bottle radius in pixels.
    """

    def __init__(
        self,
        chip_threshold: int = 200,
        rim_width_ratio: float = 0.15,
        baseline_path: str = "assets/baseline.npy",
        hough_dp: float = 1.2,
        hough_min_dist: int = 100,
        hough_param1: int = 60,
        hough_param2: int = 35,
        hough_min_radius: int = 60,
        hough_max_radius: int = 300,
    ):
        self.chip_threshold   = chip_threshold
        self.rim_width_ratio  = rim_width_ratio
        self.baseline_path    = baseline_path

        # Hough parameters
        self.hough_dp         = hough_dp
        self.hough_min_dist   = hough_min_dist
        self.hough_param1     = hough_param1
        self.hough_param2     = hough_param2
        self.hough_min_radius = hough_min_radius
        self.hough_max_radius = hough_max_radius

        # Baseline (learned from "Good" samples)
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_count: int = 0
        self._load_baseline()

    # ── Public API ───────────────────────────────────────────────────────────
    def inspect(
        self,
        frame: np.ndarray,
        circle: Optional[Tuple[int, int, int]] = None,
    ) -> InspectionResult:
        """
        Run inspection on a frame.

        Parameters
        ----------
        circle : optional (cx, cy, r) from a temporal smoother; when set,
                 Hough detection is skipped so masks align with the display.
        """
        gray = self._to_gray(frame)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # 1. Detect bottle rim (or use provided circle)
        if circle is None:
            circle = self._find_circle(blurred)
        if circle is None:
            return InspectionResult(status="NO_BOTTLE", circle=None)

        cx, cy, r = circle
        rim_width = max(8, int(r * self.rim_width_ratio))

        # 2. Build annular ROI mask
        rim_mask = self._make_annular_mask(frame.shape[:2], cx, cy, r, rim_width)

        # 3. Anomaly map
        diff_map = self._anomaly_map(gray, rim_mask)

        # 4. Find contours of anomalies (only report chips at/above threshold)
        chips = self._find_chips(diff_map)

        # 5. Verdict
        max_area = max((c.area for c in chips), default=0.0)
        status = "FAIL" if max_area >= self.chip_threshold else "PASS"

        return InspectionResult(
            status=status,
            circle=circle,
            chips=chips,
            max_chip_area=max_area,
            rim_mask=rim_mask,
            diff_map=diff_map,
        )

    def add_to_baseline(self, frame: np.ndarray) -> None:
        """Add a 'Good' frame to the calibration set and recompute the mean."""
        gray = self._to_gray(frame).astype(np.float32)

        if self._baseline_mean is None:
            self._baseline_mean = gray.copy()
            self._baseline_count = 1
        elif self._baseline_mean.shape != gray.shape:
            print(
                f"[WARN] Baseline shape {self._baseline_mean.shape} != frame "
                f"{gray.shape}; resetting baseline from this frame."
            )
            self._baseline_mean = gray.copy()
            self._baseline_count = 1
        else:
            self._baseline_count += 1
            n = self._baseline_count
            self._baseline_mean = (
                self._baseline_mean * (n - 1) + gray
            ) / n

        os.makedirs(os.path.dirname(self.baseline_path) or ".", exist_ok=True)
        np.save(self.baseline_path, self._baseline_mean)

    def reset_baseline(self) -> None:
        self._baseline_mean = None
        self._baseline_count = 0
        if os.path.exists(self.baseline_path):
            os.remove(self.baseline_path)

    @property
    def baseline_frame_count(self) -> int:
        return self._baseline_count

    def detect_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Run Hough circle detection on a BGR or grayscale frame."""
        gray = self._to_gray(frame)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        return self._find_circle(blurred)

    # ── Private helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _load_baseline(self) -> None:
        if os.path.exists(self.baseline_path):
            self._baseline_mean = np.load(self.baseline_path).astype(np.float32)
            self._baseline_count = 1
            print(f"[INFO] Baseline loaded from {self.baseline_path}")

    def _find_circle(self, blurred_gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
        circles = cv2.HoughCircles(
            blurred_gray,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.hough_min_radius,
            maxRadius=self.hough_max_radius,
        )
        if circles is None:
            return None
        c = np.round(circles[0, 0]).astype(int)
        return (c[0], c[1], c[2])

    @staticmethod
    def _make_annular_mask(
        shape: Tuple[int, int],
        cx: int, cy: int, r: int, width: int
    ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, thickness=width)
        return mask

    def _baseline_for_shape(self, gray_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        if self._baseline_mean is None:
            return None
        if self._baseline_mean.shape == gray_shape:
            return self._baseline_mean
        print(
            f"[WARN] Baseline shape {self._baseline_mean.shape} != frame "
            f"{gray_shape}; resizing baseline to match."
        )
        h, w = gray_shape
        return cv2.resize(
            self._baseline_mean, (w, h), interpolation=cv2.INTER_LINEAR
        )

    def _anomaly_map(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return a binary map of anomalous pixels within the rim mask."""
        roi = cv2.bitwise_and(gray, gray, mask=mask)
        baseline = self._baseline_for_shape(gray.shape)

        if baseline is not None:
            diff = cv2.absdiff(
                roi.astype(np.float32), baseline.astype(np.float32)
            )
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            diff = cv2.bitwise_and(diff, diff, mask=mask)
            _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        else:
            # Fallback: adaptive threshold on the ROI
            binary = cv2.adaptiveThreshold(
                roi, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=11, C=4,
            )
            binary = cv2.bitwise_and(binary, binary, mask=mask)

        # Morphological clean-up: remove noise, join nearby blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        return binary

    def _find_chips(self, binary: np.ndarray) -> List[Chip]:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        chips = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            if area < self.chip_threshold:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            chips.append(Chip(contour=cnt, area=area, bounding_box=(x, y, w, h)))
        return chips
