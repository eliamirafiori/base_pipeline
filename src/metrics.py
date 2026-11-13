import re
from typing import Optional
import numpy as np


def parse_mc_letter(pred: str) -> str:
    match = re.search(r"\b([A-D])\b", pred.upper())
    return match.group(1) if match else pred.strip()[:1].upper()


def parse_number(pred: str) -> Optional[float]:
    match = re.search(r"[-+]?\d*\.?\d+", pred)
    return float(match.group(0)) if match else None


def score_vsi_prediction(pred: str, ground_truth: str, has_options: bool) -> float:
    """
    Scores a single VSI-Bench response:
      - Multiple choice -> 1.0 if option letter matches ground_truth, else 0.0
      - Numerical -> Official VSI Mean Relative Accuracy (MRA) across thresholds [0.5, 0.95]
    """
    gt_clean = ground_truth.strip()
    if has_options:
        pred_letter = parse_mc_letter(pred)
        gt_letter = parse_mc_letter(gt_clean)
        return 1.0 if pred_letter == gt_letter else 0.0

    pred_val = parse_number(pred)
    gt_val = parse_number(gt_clean)
    if pred_val is None or gt_val is None or gt_val == 0:
        return 1.0 if pred.strip() == gt_clean else 0.0

    rel_error = abs(pred_val - gt_val) / abs(gt_val)
    thresholds = np.linspace(0.50, 0.95, 10)
    return float(np.mean([1.0 if rel_error < (1.0 - t) else 0.0 for t in thresholds]))
