"""
utils/smoother.py
Exponential moving average smoother — keeps the detected rim circle stable
across frames, preventing jitter from HoughCircles frame-to-frame noise.
"""

from typing import Optional, Tuple
import numpy as np


class Smoother:
    """
    Parameters
    ----------
    alpha : float
        Smoothing factor in [0, 1].
        Higher  → follows new values faster (less smoothing).
        Lower   → more smoothing / lag.
    """

    def __init__(self, alpha: float = 0.5):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, value: Tuple) -> Tuple:
        v = np.array(value, dtype=np.float64)
        if self._state is None:
            self._state = v
        else:
            self._state = self.alpha * v + (1.0 - self.alpha) * self._state
        return tuple(self._state.tolist())

    def reset(self) -> None:
        self._state = None
