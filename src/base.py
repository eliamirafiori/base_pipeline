from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import cv2
import numpy as np


class KFSMethod(ABC):
    """Base interface compatible with methods.py and cascade.py."""

    def __init__(self, name: str, threshold: float = 0.0):
        self.name = name
        self.threshold = threshold
        self._ref_cache: Any = None
        self.last_motion_label: str = "N/A"
        self.last_stage_reached: int = 0

    @abstractmethod
    def extract_features(self, frame: np.ndarray) -> Any:
        pass

    @abstractmethod
    def compare_features(self, feat_ref: Any, feat_curr: Any) -> float:
        pass

    def is_Blur_or_Invalid(self, frame: np.ndarray) -> bool:
        return False

    def set_reference(self, frame: np.ndarray, feat_curr: Any = None) -> None:
        self._ref_cache = (
            feat_curr if feat_curr is not None else self.extract_features(frame)
        )

    def evaluate_frame(self, frame_curr: np.ndarray) -> Tuple[float, bool, Any]:
        """Returns (score, is_keyframe, feat_curr)."""
        feat_curr = self.extract_features(frame_curr)
        if self._ref_cache is None:
            self.set_reference(frame_curr, feat_curr)
            return 0.0, True, feat_curr

        if self.is_Blur_or_Invalid(frame_curr):
            return 0.0, False, feat_curr

        score = self.compare_features(self._ref_cache, feat_curr)
        is_kf = score >= self.threshold
        return score, is_kf, feat_curr


class UniformSamplingMethod(KFSMethod):
    """Reference Baseline (Sampling Method 1): picks every N-th frame."""

    def __init__(self, step: int = 30):
        super().__init__(f"Uniform Baseline (step={step})", threshold=float(step))
        self.step = step
        self.counter = 0

    def extract_features(self, frame: np.ndarray) -> int:
        self.counter += 1
        return self.counter

    def set_reference(self, frame: np.ndarray, feat_curr: Any = None) -> None:
        self.counter = 0
        self._ref_cache = 0

    def compare_features(self, feat_ref: Any, feat_curr: Any) -> float:
        return float(self.counter)
