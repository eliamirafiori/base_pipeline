from collections import defaultdict
import math
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import zipfile

import cv2
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import BlipForQuestionAnswering, BlipProcessor

# Import your swappable KFS methods
from cascade import EgoMotionAndCascadeKFS
from methods import (
    ColorHistogramMethod,
    FarnebackOpticalFlowMethod,
    FourierMethod,
    LocalDescriptorMethod,
    LucasKanadeOpticalFlowMethod,
    PixelIntensityMethod,
)


# =====================================================================
# 1. LOAD VSI-BENCH DATASET
# =====================================================================
def load_vsi_bench(config: str = "full", split: str = "test") -> List[Dict[str, Any]]:
    """
    Loads VSI-Bench annotations using Hugging Face `load_dataset`.

    Args:
        config: 'full', 'debiased' (or 'unbiased'), or 'pruned'.
        split:  Dataset split ('test').
    """
    repo_id = "nyu-visionx/VSI-Bench"
    config_files = {
        "debiased": "test_debiased.parquet",
        "unbiased": "test_debiased.parquet",
        "pruned": "test_pruned.parquet",
        "full": "test.jsonl",
    }

    if config not in config_files:
        raise ValueError(
            f"Unknown config '{config}'. Choose from: {list(config_files.keys())}"
        )

    filename = config_files[config]
    local_file = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=filename
    )

    # Load via 'parquet' or 'json' builder so datasets==3.1.0 infers the schema natively
    builder = "parquet" if filename.endswith(".parquet") else "json"
    dataset = load_dataset(builder, data_files={split: local_file}, split=split)

    records: List[Dict[str, Any]] = []
    for row in dataset:
        options = row.get("options")
        records.append(
            {
                "id": int(row["id"]),
                "dataset": str(row["dataset"]),
                "scene_name": str(row["scene_name"]),
                "question_type": str(row.get("question_type", "unknown")),
                "question": str(row["question"]),
                "options": list(options) if options else None,
                "ground_truth": str(row["ground_truth"]).strip(),
            }
        )

    return records


# =====================================================================
# 2. KEYFRAME SELECTION (SWAPPABLE KFS METHOD)
# =====================================================================
def perform_kfs(
    video_path: str,
    kfs_method: Any,
    max_frames: Optional[int] = 16,
    frame_size: Tuple[int, int] = (384, 288),
) -> List[Dict[str, Any]]:
    """
    Extracts keyframes from a video using any swappable `kfs_method`.

    Returns:
        A chronologically ordered list of dictionaries:
        [{"frame": PIL.Image, "timestamp": float}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    # Fall back to 30 FPS if the video container lacks FPS metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sampled_video: List[Dict[str, Any]] = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video reached

        if frame_idx == 0:
            # Frame 0 always initializes the reference state and is kept as the first keyframe
            kfs_method.set_reference(frame_bgr)
            is_keyframe = True
        else:
            # Compare current frame against the last keyframe reference
            _, is_keyframe, current_features = kfs_method.evaluate_frame(frame_bgr)

            # When a scene change is detected, promote this frame to be the new reference
            if is_keyframe:
                kfs_method.set_reference(frame_bgr, current_features)

        if is_keyframe:
            # Downscale early to save RAM, convert OpenCV BGR -> PIL RGB, and store timestamp
            resized_bgr = cv2.resize(
                frame_bgr, frame_size, interpolation=cv2.INTER_AREA
            )
            frame_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            sampled_video.append(
                {
                    "frame": Image.fromarray(frame_rgb),
                    "timestamp": round(frame_idx / fps, 2),
                }
            )

        frame_idx += 1

    cap.release()

    # If the KFS method triggered more frames than `max_frames`, evenly subsample
    # across the selected keyframes while preserving strict chronological order
    if max_frames is not None and len(sampled_video) > max_frames:
        even_indices = np.linspace(0, len(sampled_video) - 1, max_frames, dtype=int)
        sampled_video = [sampled_video[i] for i in even_indices]

    return sampled_video


# =====================================================================
# 3. SMALL VLM LOADING & INFERENCE
# =====================================================================
def load_small_vlm(model_id: str = "Salesforce/blip-vqa-base") -> Dict[str, Any]:
    """
    Loads Salesforce/blip-vqa-base (~385M params) onto GPU in FP16 (~1.1 GB VRAM),
    making it fast and safe for a 6 GB GTX 980 Ti.
    """
    # Use CUDA on the GTX 980 Ti if available, otherwise fall back to Ryzen CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FP16 halves VRAM usage and speeds up inference on GPU; CPU requires FP32
    dtype = torch.float16 if device == "cuda" else torch.float32

    # BlipProcessor handles image resizing/normalization and text tokenization
    processor = BlipProcessor.from_pretrained(model_id)

    # Load model weights in the target precision and move them to the GPU
    model = BlipForQuestionAnswering.from_pretrained(model_id, torch_dtype=dtype).to(
        device
    )

    # Set to evaluation mode (disables dropout and gradient tracking overhead)
    model.eval()

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "dtype": dtype,
    }


def _pack_frames_with_timestamps(
    sampled_video: List[Dict[str, Any]],
    tile_size: Tuple[int, int] = (224, 168),
) -> Image.Image:
    """
    Arranges chronological {frame, timestamp} pairs into a single contact sheet
    with burned-in timestamps so a single-image VLM can reason across time.
    """
    num_frames = max(1, len(sampled_video))
    cols = math.ceil(math.sqrt(num_frames))
    rows = math.ceil(num_frames / cols)
    tile_w, tile_h = tile_size

    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid)

    for idx, item in enumerate(sampled_video):
        row, col = divmod(idx, cols)
        x0, y0 = col * tile_w, row * tile_h

        tile = item["frame"].resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        grid.paste(tile, (x0, y0))

        # Draw a small dark badge in the top-left corner of each tile with its timestamp
        draw.rectangle([x0, y0, x0 + 85, y0 + 16], fill=(0, 0, 0))
        draw.text((x0 + 4, y0 + 2), f"t={item['timestamp']}s", fill=(255, 255, 0))

    return grid


@torch.no_grad()
def ask_vlm(
    vlm_bundle: Dict[str, Any],
    sampled_video: List[Dict[str, Any]],
    question: str,
    options: Optional[List[str]] = None,
) -> str:
    """
    Passes the KFS contact sheet and VSI-Bench question to BLIP-VQA and returns its answer.
    """
    visual_input = _pack_frames_with_timestamps(sampled_video)

    if options:
        choices_str = " ".join(options)
        prompt = f"Question: {question} Options: {choices_str}. Answer:"
    else:
        prompt = f"Question: {question} Answer with a number:"

    processor = vlm_bundle["processor"]
    model = vlm_bundle["model"]
    device = vlm_bundle["device"]
    dtype = vlm_bundle["dtype"]

    # Prepare image & text tensors and cast floating-point image tensors to FP16 on GPU
    inputs = processor(images=visual_input, text=prompt, return_tensors="pt")
    inputs = {
        key: (
            val.to(device, dtype=dtype) if val.is_floating_point() else val.to(device)
        )
        for key, val in inputs.items()
    }

    output_ids = model.generate(**inputs, max_new_tokens=12)
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


# =====================================================================
# 4. ACCURACY EVALUATION AGAINST GROUND TRUTH
# =====================================================================
def _score_multiple_choice(
    prediction: str, ground_truth: str, options: List[str]
) -> float:
    """
    Scores multiple-choice questions (returns 1.0 if correct, 0.0 otherwise).
    Accepts either the option letter (e.g., 'B') or the option's text content.
    """
    # 1. Check if the model predicted a standalone option letter (A, B, C, or D)
    letter_match = re.search(r"\b([a-d])\b", prediction)
    if letter_match and letter_match.group(1) == ground_truth:
        return 1.0

    # 2. Otherwise, check if the model wrote out the text of the ground-truth option
    for option in options:
        opt_lower = option.lower().strip()
        is_gt_option = opt_lower.startswith(f"{ground_truth}.") or opt_lower.startswith(
            f"({ground_truth})"
        )
        if is_gt_option:
            # Strip the leading 'a.' or '(a)' prefix to compare the choice text itself
            option_body = re.sub(r"^\(?[a-d]\)?\.?\s*", "", opt_lower)
            if option_body and option_body in prediction:
                return 1.0

    return 1.0 if prediction == ground_truth else 0.0


def _score_numerical_mra(prediction: str, ground_truth: str) -> float:
    """
    Scores numerical questions using VSI-Bench's Mean Relative Accuracy (MRA).
    Measures how close the predicted number is to the ground truth across 10
    tolerance levels (from 50% relative error down to 5% relative error).
    """
    pred_match = re.search(r"[-+]?\d*\.?\d+", prediction)
    gt_match = re.search(r"[-+]?\d*\.?\d+", ground_truth)

    if not pred_match or not gt_match:
        return 1.0 if prediction == ground_truth else 0.0

    pred_val = float(pred_match.group(0))
    gt_val = float(gt_match.group(0))

    if gt_val == 0.0:
        return 1.0 if pred_val == 0.0 else 0.0

    relative_error = abs(pred_val - gt_val) / abs(gt_val)

    # Thresholds [0.50, 0.55, ..., 0.95] correspond to max allowed errors [50%, 45%, ..., 5%]
    thresholds = np.linspace(0.50, 0.95, 10)
    passed = [1.0 if relative_error < (1.0 - t) else 0.0 for t in thresholds]
    return float(np.mean(passed))


def evaluate_accuracy(
    prediction: str,
    ground_truth: str,
    options: Optional[List[str]] = None,
) -> float:
    """
    Compares the VLM's answer against the VSI-Bench ground truth.
    Returns a score between 0.0 and 1.0.
    """
    pred_clean = prediction.strip().lower()
    gt_clean = ground_truth.strip().lower()

    if options:
        return _score_multiple_choice(pred_clean, gt_clean, options)
    return _score_numerical_mra(pred_clean, gt_clean)


# =====================================================================
# 5. VIDEO STREAMING FROM HUGGING FACE ARCHIVES
# =====================================================================
def iter_vsi_videos(
    target_scenes: Optional[Set[str]] = None,
    cache_dir: str = "./vsi_videos",
    cleanup: bool = True,
) -> Generator[Tuple[str, str], None, None]:
    """
    Streams VSI-Bench videos one dataset archive at a time (`arkitscenes.zip`,
    `scannet.zip`, `scannetpp.zip`), yielding `(scene_name, local_mp4_path)`.

    If `target_scenes` is provided, only matching videos are extracted and
    remaining scenes are skipped once all targets have been yielded.
    """
    repo_id = "nyu-visionx/VSI-Bench"
    archives = ("arkitscenes.zip", "scannet.zip", "scannetpp.zip")
    remaining = set(target_scenes) if target_scenes is not None else None

    base_dir = Path(cache_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for zip_name in archives:
        # Stop downloading further archives if all target scenes were already found
        if remaining is not None and len(remaining) == 0:
            break

        print(f"\n[Storage] Downloading {zip_name}...")
        local_zip = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=zip_name,
            local_dir=str(base_dir / "zips"),
        )

        extract_dir = base_dir / Path(zip_name).stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(local_zip, "r") as zf:
                for member in sorted(zf.namelist()):
                    if not member.endswith(".mp4"):
                        continue

                    scene_name = Path(member).stem
                    if remaining is not None and scene_name not in remaining:
                        continue

                    extracted_path = zf.extract(member, path=extract_dir)
                    if remaining is not None:
                        remaining.remove(scene_name)

                    yield scene_name, extracted_path
        finally:
            # Always delete extracted MP4s and the zip archive after finishing this batch
            if cleanup:
                print(f"[Storage] Removing temporary files for {zip_name}...")
                shutil.rmtree(extract_dir, ignore_errors=True)
                Path(local_zip).unlink(missing_ok=True)


# =====================================================================
# 6. MAIN EVALUATION PIPELINE
# =====================================================================
def main(
    config: str = "full",
    max_videos: Optional[int] = 10,
    max_frames: Optional[int] = 16,
) -> None:
    """
    Runs all KFS methods across VSI-Bench videos (up to `max_videos`) and
    compares their downstream VLM accuracy.
    """
    # 1. Load VSI-Bench questions and group them by scene
    vsi_questions = load_vsi_bench(config=config, split="test")
    questions_by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in vsi_questions:
        questions_by_scene[q["scene_name"]].append(q)

    # Optionally limit the evaluation to the first N scenes
    all_scenes = sorted(questions_by_scene.keys())
    selected_scenes = set(all_scenes[:max_videos] if max_videos else all_scenes)
    total_q = sum(len(questions_by_scene[s]) for s in selected_scenes)

    print(
        f"Loaded VSI-Bench ({config}): {total_q} questions across "
        f"{len(selected_scenes)} videos."
    )

    # 2. Load the lightweight VLM once onto GPU
    vlm_bundle = load_small_vlm("Salesforce/blip-vqa-base")

    # 3. Define the swappable KFS methods to benchmark
    kfs_methods = [
        FourierMethod(threshold=0.28, blur_thresh=15.0),
        PixelIntensityMethod(threshold=25.0),
        ColorHistogramMethod(threshold=0.35, grid_size=(2, 2)),
        LocalDescriptorMethod(threshold=0.70, n_features=500),
        LucasKanadeOpticalFlowMethod(
            threshold=45.0, max_corners=250, min_survival_ratio=0.45
        ),
        FarnebackOpticalFlowMethod(threshold=25.0, grid_size=(160, 120)),
        EgoMotionAndCascadeKFS(
            flow_trigger_thresh=22.0,
            static_sad_thresh=1.8,
            blur_thresh=15.0,
            orb_thresh=0.65,
        ),
    ]

    # 4. Evaluate every scene across all KFS methods
    stats = defaultdict(lambda: {"acc": [], "frames": []})

    for vid_idx, (scene_name, video_path) in enumerate(
        iter_vsi_videos(target_scenes=selected_scenes, cleanup=True), start=1
    ):
        scene_questions = questions_by_scene[scene_name]
        print(
            f"\n[Video {vid_idx}/{len(selected_scenes)}] "
            f"Scene '{scene_name}' ({len(scene_questions)} questions)"
        )

        for method in kfs_methods:
            # Extract keyframes once per video (independent of the questions)
            sampled_video = perform_kfs(
                video_path=video_path,
                kfs_method=method,
                max_frames=max_frames,
            )

            # Reuse those exact keyframes across all questions for this scene
            for q in scene_questions:
                vlm_answer = ask_vlm(
                    vlm_bundle=vlm_bundle,
                    sampled_video=sampled_video,
                    question=q["question"],
                    options=q["options"],
                )
                acc = evaluate_accuracy(
                    prediction=vlm_answer,
                    ground_truth=q["ground_truth"],
                    options=q["options"],
                )
                stats[method.name]["acc"].append(acc)
                stats[method.name]["frames"].append(len(sampled_video))

    # 5. Print final comparison table
    print("\n" + "=" * 72)
    print(f"VSI-BENCH ({config.upper()}) SUMMARY — {len(selected_scenes)} VIDEOS")
    print("=" * 72)
    for method in kfs_methods:
        s = stats[method.name]
        mean_acc = float(np.mean(s["acc"])) * 100.0 if s["acc"] else 0.0
        mean_frames = float(np.mean(s["frames"])) if s["frames"] else 0.0
        print(
            f"Method: {method.name:<32} | "
            f"Avg Frames: {mean_frames:>4.1f} | "
            f"Overall Accuracy: {mean_acc:>5.2f}%"
        )


if __name__ == "__main__":
    # Set max_videos=None to run the entire dataset
    main(config="full", max_videos=10, max_frames=16)
