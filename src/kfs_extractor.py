from dataclasses import dataclass
import time
from typing import List, Optional, Tuple
import cv2
from PIL import Image

from base import KFSMethod


@dataclass
class KeyframeExtractionResult:
    method_name: str
    frames: List[Tuple[str, Image.Image]]  # Chronological (timestamp_str, RGB Image)
    frame_indices: List[int]
    raw_selected_count: int
    total_video_frames: int
    selection_time_sec: float


def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}s"


def extract_keyframes(
    video_path: str,
    method: KFSMethod,
    frame_budget: Optional[int] = 16,
    max_long_edge: int = 512,
    candidate_stride: int = 1,
) -> KeyframeExtractionResult:
    """
    Runs a KFS method over a video and returns chronologically ordered frames + timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    selected_raw: List[Tuple[int, str, Image.Image]] = []
    frame_idx = 0

    t0 = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % candidate_stride == 0:
            if frame_idx == 0:
                method.set_reference(frame_bgr)
                is_kf = True
                feat_curr = None
            else:
                _, is_kf, feat_curr = method.evaluate_frame(frame_bgr)
                if is_kf:
                    method.set_reference(frame_bgr, feat_curr)

            if is_kf:
                ts_str = format_timestamp(frame_idx / fps)
                # Resize keeping aspect ratio so VLM visual token cost per frame is fixed
                h, w = frame_bgr.shape[:2]
                scale = min(1.0, max_long_edge / max(h, w))
                if scale < 1.0:
                    frame_bgr = cv2.resize(
                        frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                    )
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                selected_raw.append((frame_idx, ts_str, Image.fromarray(rgb)))

        frame_idx += 1

    elapsed = time.perf_counter() - t0
    cap.release()

    raw_count = len(selected_raw)

    # If the method selected more frames than the allowed VLM budget, subsample evenly
    # while strictly preserving original chronological order.
    if frame_budget is not None and len(selected_raw) > frame_budget:
        idxs = [
            int(round(i * (len(selected_raw) - 1) / (frame_budget - 1)))
            for i in range(frame_budget)
        ]
        selected_raw = [selected_raw[i] for i in idxs]

    frames_with_ts = [(ts, img) for _, ts, img in selected_raw]
    indices = [idx for idx, _, _ in selected_raw]

    return KeyframeExtractionResult(
        method_name=method.name,
        frames=frames_with_ts,
        frame_indices=indices,
        raw_selected_count=raw_count,
        total_video_frames=frame_idx,
        selection_time_sec=round(elapsed, 4),
    )
