from abc import ABC, abstractmethod
import math
import time
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw


def format_ts(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}s"


class BaseSelector(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_indices(self, frames_bgr: List[np.ndarray]) -> List[int]:
        pass


class UniformSelector(BaseSelector):
    """Reference baseline: evenly spaced frames across the video."""

    def __init__(self, budget: int = 16):
        super().__init__(f"Uniform (K={budget})")
        self.budget = budget

    def select_indices(self, frames_bgr: List[np.ndarray]) -> List[int]:
        n = len(frames_bgr)
        if n <= self.budget:
            return list(range(n))
        return [int(round(i * (n - 1) / (self.budget - 1))) for i in range(self.budget)]


class CascadeMotionSelector(BaseSelector):
    """
    Lightweight 3-stage cascade (SAD static gate -> Farneback motion -> Laplacian blur gate).
    """

    def __init__(
        self,
        budget: int = 16,
        sad_thresh: float = 2.0,
        flow_thresh: float = 18.0,
        blur_thresh: float = 15.0,
    ):
        super().__init__(f"CascadeKFS (K={budget})")
        self.budget = budget
        self.sad_thresh = sad_thresh
        self.flow_thresh = flow_thresh
        self.blur_thresh = blur_thresh

    def select_indices(self, frames_bgr: List[np.ndarray]) -> List[int]:
        if not frames_bgr:
            return []

        selected = [0]
        prev_small = cv2.resize(
            cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY), (160, 120)
        )
        accum_flow = np.zeros((120, 160, 2), dtype=np.float32)

        for idx in range(1, len(frames_bgr)):
            gray = cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 120))

            # Stage 1: Fast pixel SAD gate
            if float(np.mean(cv2.absdiff(prev_small, small))) < self.sad_thresh:
                prev_small = small
                continue

            # Stage 2: Dense optical flow accumulation
            flow = cv2.calcOpticalFlowFarneback(
                prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            prev_small = small
            accum_flow += flow
            net_disp = float(np.mean(np.linalg.norm(accum_flow, axis=2)))

            if net_disp < self.flow_thresh:
                continue

            # Stage 3: Blur rejection gate
            if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.blur_thresh:
                continue

            selected.append(idx)
            accum_flow.fill(0.0)

        # Enforce fixed frame budget while strictly keeping chronological order
        if len(selected) > self.budget:
            keep = [
                int(round(i * (len(selected) - 1) / (self.budget - 1)))
                for i in range(self.budget)
            ]
            selected = [selected[i] for i in keep]

        return selected


def run_selector_on_video(
    video_path: str,
    selector: BaseSelector,
    sample_fps: float = 2.0,
) -> Tuple[List[Tuple[str, Image.Image]], float]:
    """Reads candidate stream at `sample_fps`, runs `selector`, returns stamped keyframes + latency."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / sample_fps)))

    frames_bgr: List[np.ndarray] = []
    timestamps: List[float] = []
    raw_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if raw_idx % stride == 0:
            frames_bgr.append(frame)
            timestamps.append(raw_idx / native_fps)
        raw_idx += 1
    cap.release()

    t0 = time.perf_counter()
    chosen_idxs = selector.select_indices(frames_bgr)
    elapsed = time.perf_counter() - t0

    keyframes: List[Tuple[str, Image.Image]] = []
    for idx in chosen_idxs:
        rgb = cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2RGB)
        keyframes.append((format_ts(timestamps[idx]), Image.fromarray(rgb)))

    return keyframes, elapsed


def build_temporal_montage(
    keyframes: List[Tuple[str, Image.Image]],
    tile_size: Tuple[int, int] = (224, 168),
) -> Image.Image:
    """
    Arranges chronological keyframes into a single labeled contact sheet
    so single-image VLMs can reason over temporal order and timestamps.
    """
    n = max(1, len(keyframes))
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    tw, th = tile_size

    canvas = Image.new("RGB", (cols * tw, rows * th), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for i, (ts, img) in enumerate(keyframes):
        r, c = divmod(i, cols)
        x0, y0 = c * tw, r * th
        resized = img.resize((tw, th), Image.Resampling.BILINEAR)
        canvas.paste(resized, (x0, y0))
        draw.rectangle([x0, y0, x0 + 95, y0 + 16], fill=(0, 0, 0))
        draw.text((x0 + 4, y0 + 2), f"#{i+1} {ts}", fill=(255, 255, 0))

    return canvas
