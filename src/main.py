from collections import defaultdict
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import zipfile

import cv2
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

# Import your swappable KFS classes (no __init__.py required)
from base import UniformSamplingMethod
from cascade import EgoMotionAndCascadeKFS
from methods import (
    ColorHistogramMethod,
    FarnebackOpticalFlowMethod,
    FourierMethod,
    LocalDescriptorMethod,
    LucasKanadeOpticalFlowMethod,
    PixelIntensityMethod,
    QuaternionFourierFusionMethod,
)


# =====================================================================
# 1. LOAD VSI-BENCH DATASET
# =====================================================================
def load_vsi_bench(
    config: str = "debiased", split: str = "test"
) -> List[Dict[str, Any]]:
    """
    Loads VSI-Bench annotations using Hugging Face `load_dataset`.

    Args:
        config: 'full', 'debiased' (or 'unbiased'), or 'pruned'.
        split:  Dataset split (VSI-Bench uses 'test').
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

    # Load via 'parquet' or 'json' builder so datasets==3.1.0 infers the schema cleanly
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
# 2. KEYFRAME SELECTION (SWAPPABLE KFS METHOD + INTRINSIC METRICS)
# =====================================================================
def perform_kfs(
    video_path: str,
    kfs_method: Any,
    max_frames: Optional[int] = None,
    frame_size: Tuple[int, int] = (384, 288),
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Runs any swappable `kfs_method` on `video_path` without seeing the question.

    Returns:
        sampled_video: Chronologically ordered list of dicts:
                       [{"frame": PIL.Image, "timestamp": float}, ...]
        metrics:       Intrinsic KFS quality stats (total_frames, selected_frames,
                       compression_ratio, avg_sharpness, latency_ms_per_frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    # Fall back to 30 FPS if the container lacks FPS metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sampled_video: List[Dict[str, Any]] = []
    sharpness_scores: List[float] = []
    frame_idx = 0

    t_start = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video reached

        if frame_idx == 0:
            # Frame 0 always initializes the reference state and becomes the first keyframe
            kfs_method.set_reference(frame_bgr)
            is_keyframe = True
        else:
            # Score current frame against the active reference keyframe
            _, is_keyframe, current_features = kfs_method.evaluate_frame(frame_bgr)

            # Promote current frame to be the new reference whenever a keyframe triggers
            if is_keyframe:
                kfs_method.set_reference(frame_bgr, current_features)

        if is_keyframe:
            # Measure Laplacian variance as a VLM-free sharpness indicator
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            sharpness_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            # Downscale early to save RAM, convert BGR -> RGB, and attach timestamp
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

    elapsed_sec = time.perf_counter() - t_start
    cap.release()

    # Optional hard cap: evenly subsample while strictly preserving chronological order
    if max_frames is not None and len(sampled_video) > max_frames:
        even_indices = np.linspace(0, len(sampled_video) - 1, max_frames, dtype=int)
        sampled_video = [sampled_video[i] for i in even_indices]
        sharpness_scores = [sharpness_scores[i] for i in even_indices]

    total_frames = max(1, frame_idx)
    selected_count = len(sampled_video)
    metrics = {
        "total_frames": float(total_frames),
        "selected_frames": float(selected_count),
        "compression_ratio": selected_count / total_frames,
        "avg_sharpness": float(np.mean(sharpness_scores)) if sharpness_scores else 0.0,
        "latency_ms_per_frame": (elapsed_sec * 1000.0) / total_frames,
    }

    return sampled_video, metrics


# =====================================================================
# 3. TWO-STAGE MODELS: CLIP QUESTION FILTER + SMALL VLM (BLIP-VQA)
# =====================================================================
def load_clip_filter(model_id: str = "openai/clip-vit-base-patch32") -> Dict[str, Any]:
    """
    Loads OpenAI CLIP ViT-B/32 (~350 MB VRAM in FP16) to retrieve the top-K
    question-relevant frames from the KFS keyframe pool before VLM inference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "dtype": dtype,
    }


@torch.no_grad()
def encode_keyframes_clip(
    clip_bundle: Dict[str, Any],
    sampled_video: List[Dict[str, Any]],
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Encodes all KFS-selected frames into L2-normalized CLIP embeddings in small
    mini-batches (`batch_size=32`) so VRAM stays flat even when `max_kfs_frames=None`
    produces hundreds of keyframes. Computed ONCE per video, not per question.
    """
    processor = clip_bundle["processor"]
    model = clip_bundle["model"]
    device = clip_bundle["device"]
    dtype = clip_bundle["dtype"]

    all_embeds: List[torch.Tensor] = []
    for start in range(0, len(sampled_video), batch_size):
        batch_images = [
            item["frame"] for item in sampled_video[start : start + batch_size]
        ]
        inputs = processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, dtype=dtype)

        img_embeds = model.get_image_features(pixel_values=pixel_values)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(img_embeds)

    return torch.cat(all_embeds, dim=0)


@torch.no_grad()
def select_top_k_for_question(
    clip_bundle: Dict[str, Any],
    sampled_video: List[Dict[str, Any]],
    image_embeds: torch.Tensor,
    question: str,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Picks the `top_k` most question-relevant frames using precomputed CLIP image
    embeddings, then re-sorts them chronologically.
    Keeping `top_k=4` ensures BLIP-VQA always sees a crisp 2x2 grid.
    """
    if len(sampled_video) <= top_k:
        return sampled_video

    processor = clip_bundle["processor"]
    model = clip_bundle["model"]
    device = clip_bundle["device"]

    text_inputs = processor(
        text=[question], return_tensors="pt", padding=True, truncation=True
    ).to(device)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Cosine similarity between the question and every pre-encoded keyframe
    scores = (image_embeds @ text_embeds.T).squeeze(-1)
    top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()

    # Re-sort chronologically so temporal order is preserved for the VLM
    return [sampled_video[i] for i in sorted(top_indices)]


def load_small_vlm(model_id: str = "Salesforce/blip-vqa-base") -> Dict[str, Any]:
    """
    Loads Salesforce/blip-vqa-base (~385M params) onto GPU in FP16 (~1.1 GB VRAM),
    making it lightweight and safe for a 6 GB GTX 980 Ti alongside CLIP.
    """
    # Use CUDA on the GTX 980 Ti if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FP16 halves VRAM usage and speeds up inference on GPU; CPU requires FP32
    dtype = torch.float16 if device == "cuda" else torch.float32

    # BlipProcessor handles image resizing/normalization and text tokenization
    processor = BlipProcessor.from_pretrained(model_id)

    # Load pretrained weights in target precision and transfer to GPU
    model = BlipForQuestionAnswering.from_pretrained(model_id, torch_dtype=dtype).to(
        device
    )

    # Set to evaluation mode (disables dropout and training overhead)
    model.eval()

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "dtype": dtype,
    }


def _pack_frames_with_timestamps(
    keyframes: List[Dict[str, Any]],
    tile_size: Tuple[int, int] = (224, 168),
) -> Image.Image:
    """
    Arranges chronological {frame, timestamp} pairs into a single contact sheet
    with burned-in timestamps so a single-image VLM can reason across time.
    """
    num_frames = max(1, len(keyframes))
    cols = math.ceil(math.sqrt(num_frames))
    rows = math.ceil(num_frames / cols)
    tile_w, tile_h = tile_size

    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid)

    for idx, item in enumerate(keyframes):
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
    keyframes: List[Dict[str, Any]],
    question: str,
    options: Optional[List[str]] = None,
) -> str:
    """
    Passes the chronological keyframe contact sheet and VSI-Bench question
    to BLIP-VQA and returns the model's predicted answer string.
    """
    visual_input = _pack_frames_with_timestamps(keyframes)

    if options:
        choices_str = " ".join(options)
        prompt = f"Question: {question} Options: {choices_str}. Answer:"
    else:
        prompt = f"Question: {question} Answer with a number:"

    processor = vlm_bundle["processor"]
    model = vlm_bundle["model"]
    device = vlm_bundle["device"]
    dtype = vlm_bundle["dtype"]

    # Prepare image & text tensors and cast floating-point tensors to FP16 on GPU
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
    remaining archives are skipped once all targets have been yielded.
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
# 6. MAIN TWO-STAGE EVALUATION PIPELINE
# =====================================================================
def main(
    config: str = "debiased",
    max_videos: Optional[int] = 10,
    max_kfs_frames: Optional[int] = None,
    vlm_top_k: int = 4,
    clip_batch_size: int = 32,
) -> None:
    """
    Evaluates all swappable KFS methods on VSI-Bench (up to `max_videos`) using:
      Stage 1: Question-agnostic KFS extraction + intrinsic quality metrics
      Stage 2: Mini-batched CLIP embedding + Top-K retrieval + BLIP-VQA scoring
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

    # 2. Load CLIP retriever (~350 MB VRAM) and BLIP-VQA (~1.1 GB VRAM) once on GPU
    clip_bundle = load_clip_filter("openai/clip-vit-base-patch32")
    vlm_bundle = load_small_vlm("Salesforce/blip-vqa-base")

    # 3. Define the swappable KFS methods to benchmark (including Uniform baseline)
    kfs_methods = [
        UniformSamplingMethod(step=30),
        FourierMethod(threshold=0.65, blur_thresh=15.0),
        PixelIntensityMethod(threshold=45.0),
        ColorHistogramMethod(threshold=0.35, grid_size=(2, 2)),
        LocalDescriptorMethod(threshold=0.70, n_features=500),
        LucasKanadeOpticalFlowMethod(
            threshold=85.0, max_corners=250, min_survival_ratio=0.45
        ),
        FarnebackOpticalFlowMethod(threshold=25.0, grid_size=(160, 120)),
        EgoMotionAndCascadeKFS(
            flow_trigger_thresh=25.0,
            static_sad_thresh=3.5,
            blur_thresh=15.0,
            orb_thresh=0.65,
        ),
        QuaternionFourierFusionMethod(threshold=0.36, blur_thresh=15.0, adaptive=True),
    ]

    # 4. Stream through VSI-Bench scenes and evaluate every KFS method
    stats = defaultdict(
        lambda: {
            "acc": [],
            "selected_frames": [],
            "compression_ratio": [],
            "avg_sharpness": [],
            "latency_ms": [],
        }
    )

    for vid_idx, (scene_name, video_path) in enumerate(
        iter_vsi_videos(target_scenes=selected_scenes, cleanup=True), start=1
    ):
        scene_questions = questions_by_scene[scene_name]
        print(
            f"\n[Video {vid_idx}/{len(selected_scenes)}] "
            f"Scene '{scene_name}' ({len(scene_questions)} questions)"
        )

        for method in kfs_methods:
            # Stage 1: Extract keyframes once per video (question-agnostic)
            sampled_video, kfs_metrics = perform_kfs(
                video_path=video_path,
                kfs_method=method,
                max_frames=max_kfs_frames,
            )

            stats[method.name]["selected_frames"].append(kfs_metrics["selected_frames"])
            stats[method.name]["compression_ratio"].append(
                kfs_metrics["compression_ratio"]
            )
            stats[method.name]["avg_sharpness"].append(kfs_metrics["avg_sharpness"])
            stats[method.name]["latency_ms"].append(kfs_metrics["latency_ms_per_frame"])

            # Pre-encode keyframes in mini-batches of 32 (prevents OOM when max_kfs_frames=None)
            image_embeds = encode_keyframes_clip(
                clip_bundle=clip_bundle,
                sampled_video=sampled_video,
                batch_size=clip_batch_size,
            )

            # Stage 2: Retrieve top-K frames per question via CLIP and query BLIP-VQA
            for q in scene_questions:
                top_frames = select_top_k_for_question(
                    clip_bundle=clip_bundle,
                    sampled_video=sampled_video,
                    image_embeds=image_embeds,
                    question=q["question"],
                    top_k=vlm_top_k,
                )
                vlm_answer = ask_vlm(
                    vlm_bundle=vlm_bundle,
                    keyframes=top_frames,
                    question=q["question"],
                    options=q["options"],
                )
                acc = evaluate_accuracy(
                    prediction=vlm_answer,
                    ground_truth=q["ground_truth"],
                    options=q["options"],
                )
                stats[method.name]["acc"].append(acc)

            # Free per-video CLIP embedding tensor before moving to the next KFS method
            del image_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 5. Print comprehensive comparison table (Intrinsic KFS + Downstream VLM)
    print("\n" + "=" * 108)
    print(
        f"VSI-BENCH ({config.upper()}) TWO-STAGE SUMMARY — "
        f"{len(selected_scenes)} VIDEOS (CLIP Top-{vlm_top_k} -> BLIP-VQA)"
    )
    print("=" * 108)
    print(
        f"{'KFS Method':<32} | {'Avg KFs':>7} | {'Keep %':>6} | "
        f"{'Sharpness':>9} | {'ms/frame':>8} | {'VLM Acc':>8}"
    )
    print("-" * 108)

    for method in kfs_methods:
        s = stats[method.name]
        avg_kfs = float(np.mean(s["selected_frames"])) if s["selected_frames"] else 0.0
        keep_pct = (
            float(np.mean(s["compression_ratio"])) * 100.0
            if s["compression_ratio"]
            else 0.0
        )
        sharp = float(np.mean(s["avg_sharpness"])) if s["avg_sharpness"] else 0.0
        lat_ms = float(np.mean(s["latency_ms"])) if s["latency_ms"] else 0.0
        mean_acc = float(np.mean(s["acc"])) * 100.0 if s["acc"] else 0.0

        print(
            f"{method.name:<32} | {avg_kfs:>7.1f} | {keep_pct:>5.1f}% | "
            f"{sharp:>9.1f} | {lat_ms:>8.2f} | {mean_acc:>7.2f}%"
        )


if __name__ == "__main__":
    # Safe to run with max_kfs_frames=None on GTX 980 Ti (6 GB VRAM)
    main(config="debiased", max_videos=10, max_kfs_frames=None, vlm_top_k=4)
