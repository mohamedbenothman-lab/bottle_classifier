"""
utils/visualizer.py
Renders all HUD overlays on the inspection frame:
  - Annular rim ring (green = PASS, red = FAIL, yellow = NO_BOTTLE)
  - Red bounding boxes around chips that exceed the threshold
  - Status panel (PASS / FAIL) with max chip area
"""

import cv2
import numpy as np
from core.bottle_inspector import InspectionResult


COLOUR = {
    "PASS":      (0,   200,  50),    # green
    "FAIL":      (0,    40, 220),    # red  (BGR)
    "NO_BOTTLE": (0,   180, 220),    # amber
    "CHIP_BOX":  (0,    40, 220),    # red bounding boxes
    "TEXT_BG":   (20,   20,  20),
    "TEXT_FG":   (240, 240, 240),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
THICKNESS  = 2


class Visualizer:

    def __init__(self, chip_threshold: int = 200, rim_width_ratio: float = 0.15):
        self.chip_threshold = chip_threshold
        self.rim_width_ratio = rim_width_ratio

    def render(self, frame: np.ndarray, result: InspectionResult) -> np.ndarray:
        out = frame.copy()
        colour = COLOUR.get(result.status, COLOUR["NO_BOTTLE"])

        # 1. Rim ring
        if result.circle is not None:
            cx, cy, r = result.circle
            rim_w = max(8, int(r * self.rim_width_ratio))
            cv2.circle(out, (cx, cy), r, colour, thickness=rim_w)
            # thin centre-cross for reference
            cv2.drawMarker(out, (cx, cy), colour,
                           cv2.MARKER_CROSS, markerSize=20, thickness=1)

        # 2. Chip bounding boxes (all listed chips are at/above threshold)
        for chip in result.chips:
            x, y, w, h = chip.bounding_box
            box_colour = COLOUR["CHIP_BOX"]
            cv2.rectangle(out, (x, y), (x + w, y + h), box_colour, 2)
            label = f"{int(chip.area)} px"
            cv2.putText(out, label, (x, y - 6), FONT, 0.5,
                        box_colour, 1, cv2.LINE_AA)

        # 3. Status panel (top-left)
        self._draw_status_panel(out, result, colour)

        return out

    def _draw_status_panel(
        self,
        out: np.ndarray,
        result: InspectionResult,
        colour: tuple,
    ) -> None:
        lines = [
            f"Status  : {result.status}",
            f"Chips   : {len(result.chips)}",
            f"Max area: {result.max_chip_area:.0f} px",
            f"Threshold: {self.chip_threshold} px",
        ]
        pad    = 10
        lh     = 26                # line height
        width  = 260
        height = pad * 2 + lh * len(lines)

        # semi-transparent background
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (width, height),
                      COLOUR["TEXT_BG"], cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

        for i, line in enumerate(lines):
            y = pad + (i + 1) * lh - 4
            text_colour = colour if i == 0 else COLOUR["TEXT_FG"]
            cv2.putText(out, line, (pad, y), FONT, FONT_SCALE,
                        text_colour, THICKNESS if i == 0 else 1, cv2.LINE_AA)
