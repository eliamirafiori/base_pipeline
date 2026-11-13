import math
import re
import zipfile
import cv2
import numpy as np
import pandas as pd
import torch
import json
import shutil

from pathlib import Path
from typing import Any, Dict, Generator, List, Set, Optional, Tuple
from PIL import Image, ImageDraw
from collections import defaultdict
from huggingface_hub import HfFileSystem, hf_hub_download
from transformers import BlipForQuestionAnswering, BlipProcessor

# Swap or import your custom KFS classes here
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
def load_vsi_bench(config: str = "debiased") -> List[Dict[str, Any]]:
    """
    Loads VSI-Bench annotations from nyu-visionx/VSI-Bench.
    Supported configs:
      - 'debiased' (or 'unbiased') -> loads test_debiased.parquet
      - 'pruned'                    -> loads test_pruned.parquet
      - 'full'                      -> loads test.jsonl
    """
    repo_id = "nyu-visionx/VSI-Bench"
    config_map = {
        "debiased": "test_debiased.parquet",
        "unbiased": "test_debiased.parquet",
        "pruned": "test_pruned.parquet",
        "full": "test.jsonl",
    }

    if config not in config_map:
        raise ValueError(
            f"Unknown config '{config}'. Choose from: {list(config_map.keys())}"
        )

    filename = config_map[config]
    local_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=filename
    )

    if filename.endswith(".parquet"):
        df = pd.read_parquet(local_path)
        raw_rows = df.to_dict(orient="records")
    else:
        with open(local_path, "r", encoding="utf-8") as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]

    records = []
    for row in raw_rows:
        opts = row.get("options")
        if hasattr(opts, "tolist"):
            opts = opts.tolist()

        records.append(
            {
                "id": int(row["id"]),
                "dataset": str(row["dataset"]),
                "scene_name": str(row["scene_name"]),
                "question_type": str(row.get("question_type", "unknown")),
                "question": str(row["question"]),
                "options": opts if opts else None,
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
    resize_dim: Tuple[int, int] = (384, 288),
) -> List[Dict[str, Any]]:
    """
    Runs any swappable `kfs_method` on `video_path`.
    Returns a chronologically ordered list of dicts:
        [{"frame": PIL.Image, "timestamp": float}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sampled_video: List[Dict[str, Any]] = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Frame 0 always initializes the reference state in your KFSMethod classes
        if frame_idx == 0:
            kfs_method.set_reference(frame_bgr)
            is_keyframe = True
        else:
            _, is_keyframe, feat_curr = kfs_method.evaluate_frame(frame_bgr)
            if is_keyframe:
                kfs_method.set_reference(frame_bgr, feat_curr)

        if is_keyframe:
            timestamp_sec = round(frame_idx / fps, 2)
            resized_bgr = cv2.resize(
                frame_bgr, resize_dim, interpolation=cv2.INTER_AREA
            )
            frame_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            sampled_video.append(
                {
                    "frame": Image.fromarray(frame_rgb),
                    "timestamp": timestamp_sec,
                }
            )

        frame_idx += 1

    cap.release()

    # Enforce frame budget while preserving chronological order
    if max_frames is not None and len(sampled_video) > max_frames:
        keep_indices = [
            int(round(i * (len(sampled_video) - 1) / (max_frames - 1)))
            for i in range(max_frames)
        ]
        sampled_video = [sampled_video[i] for i in keep_indices]

    return sampled_video


# =====================================================================
# 3. SMALL VLM LOADING & INFERENCE
# =====================================================================
def load_small_vlm(model_id: str = "Salesforce/blip-vqa-base") -> Dict[str, Any]:
    """
    Loads a lightweight VLM compatible with torch==1.13.1 and GTX 980 Ti (6GB VRAM).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id, torch_dtype=dtype).to(
        device
    )
    model.eval()

    return {"model": model, "processor": processor, "device": device, "dtype": dtype}


def _pack_frames_with_timestamps(
    sampled_video: List[Dict[str, Any]], tile_size: Tuple[int, int] = (224, 168)
) -> Image.Image:
    """Stitches {frame, timestamp} items into a chronological contact sheet for the VLM."""
    n = max(1, len(sampled_video))
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    tw, th = tile_size

    grid = Image.new("RGB", (cols * tw, rows * th), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid)

    for idx, item in enumerate(sampled_video):
        r, c = divmod(idx, cols)
        x0, y0 = c * tw, r * th
        tile = item["frame"].resize((tw, th), Image.Resampling.BILINEAR)
        grid.paste(tile, (x0, y0))
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
    Takes the VLM bundle, the KFS output (`sampled_video`), and the VSI-Bench question,
    and returns the model's text answer.
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

    inputs = processor(images=visual_input, text=prompt, return_tensors="pt")
    inputs = {
        k: (v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device))
        for k, v in inputs.items()
    }

    output_ids = model.generate(**inputs, max_new_tokens=12)
    answer = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return answer


# =====================================================================
# 4. ACCURACY EVALUATION AGAINST GROUND TRUTH
# =====================================================================
def evaluate_accuracy(
    prediction: str,
    ground_truth: str,
    options: Optional[List[str]] = None,
) -> float:
    """
    Compares the VLM answer with the VSI-Bench ground truth.
    - Multiple-choice questions: returns 1.0 (correct) or 0.0 (incorrect).
    - Numerical questions: returns official VSI-Bench Mean Relative Accuracy (MRA).
    """
    pred_clean = prediction.strip().lower()
    gt_clean = ground_truth.strip().lower()

    # Multiple-choice evaluation
    if options:
        letter_match = re.search(r"\b([a-d])\b", pred_clean)
        if letter_match and letter_match.group(1) == gt_clean:
            return 1.0

        # Also check if the VLM output the text of the correct option directly
        for opt in options:
            opt_lower = opt.lower().strip()
            if opt_lower.startswith(f"{gt_clean}.") or opt_lower.startswith(
                f"({gt_clean})"
            ):
                opt_text = opt_lower.split(".", 1)[-1].strip()
                if opt_text and opt_text in pred_clean:
                    return 1.0

        return 1.0 if pred_clean == gt_clean else 0.0

    # Numerical evaluation (Mean Relative Accuracy across thresholds 0.50 -> 0.95)
    pred_num = re.search(r"[-+]?\d*\.?\d+", pred_clean)
    gt_num = re.search(r"[-+]?\d*\.?\d+", gt_clean)

    if not pred_num or not gt_num:
        return 1.0 if pred_clean == gt_clean else 0.0

    p_val = float(pred_num.group(0))
    g_val = float(gt_num.group(0))
    if g_val == 0:
        return 1.0 if p_val == 0 else 0.0

    rel_error = abs(p_val - g_val) / abs(g_val)
    thresholds = np.linspace(0.50, 0.95, 10)
    return float(np.mean([1.0 if rel_error < (1.0 - t) else 0.0 for t in thresholds]))


# =====================================================================
# HELPER: FETCH A SINGLE SCENE VIDEO FROM VSI-BENCH ZIP IF NEEDED
# =====================================================================
def get_vsi_video_path(
    dataset_name: str, scene_name: str, cache_dir: str = "./vsi_videos"
) -> str:
    """Extracts a single scene MP4 from the corresponding VSI-Bench zip archive on demand."""
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_mp4 = out_dir / f"{scene_name}.mp4"
    if target_mp4.exists():
        return str(target_mp4)

    zip_filename = f"{dataset_name}.zip"
    local_zip = hf_hub_download(
        repo_id="nyu-visionx/VSI-Bench",
        repo_type="dataset",
        filename=zip_filename,
        local_dir=str(out_dir),
    )
    with zipfile.ZipFile(local_zip, "r") as zf:
        for member in zf.namelist():
            if member.endswith(f"{scene_name}.mp4"):
                with zf.open(member) as src, open(target_mp4, "wb") as dst:
                    dst.write(src.read())
                return str(target_mp4)

    raise FileNotFoundError(f"Scene {scene_name}.mp4 not found inside {zip_filename}")


# =====================================================================
# MAIN EXECUTION EXAMPLE
# =====================================================================
def main():
    # 1. Load VSI-Bench questions
    vsi_questions = load_vsi_bench(split="test")
    print(f"Loaded {len(vsi_questions)} total VSI-Bench questions.")

    # 2. Load the small VLM once
    vlm_bundle = load_small_vlm("Salesforce/blip-vqa-base")

    # 3. Define the swappable KFS methods you want to compare
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

    # Pick the first scene in VSI-Bench (or point directly to a local .mp4)
    first_scene = vsi_questions[0]["scene_name"]
    first_dataset = vsi_questions[0]["dataset"]
    scene_questions = [q for q in vsi_questions if q["scene_name"] == first_scene]

    video_path = get_vsi_video_path(first_dataset, first_scene)
    print(f"\nEvaluating scene '{first_scene}' ({len(scene_questions)} questions)...")

    # 4. Run each swappable KFS method and compare VLM accuracy
    for method in kfs_methods:
        sampled_video = perform_kfs(
            video_path=video_path, kfs_method=method, max_frames=None
        )

        scores = []
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
            scores.append(acc)

        mean_acc = float(np.mean(scores)) * 100.0 if scores else 0.0
        print(
            f"Method: {method.name:<32} | "
            f"Sampled Frames: {len(sampled_video):>2} | "
            f"Scene Accuracy: {mean_acc:>5.1f}%"
        )


def iter_all_vsi_videos(
    cache_dir: str = "./vsi_videos",
    archives: Tuple[str, ...] = ("arkitscenes.zip", "scannet.zip", "scannetpp.zip"),
    cleanup: bool = True,
) -> Generator[Tuple[str, str], None, None]:
    """
    Downloads and extracts one VSI-Bench zip archive at a time, yields
    (scene_name, mp4_path) for every video inside, and deletes temporary files
    afterward (even if the caller breaks early).
    """
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)

    for zip_name in archives:
        print(f"\n[Disk Manager] Fetching {zip_name} from nyu-visionx/VSI-Bench...")
        local_zip = hf_hub_download(
            repo_id="nyu-visionx/VSI-Bench",
            repo_type="dataset",
            filename=zip_name,
            local_dir=str(base / "zips"),
        )

        extract_dir = base / Path(zip_name).stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(local_zip, "r") as zf:
                zf.extractall(extract_dir)

            for mp4_path in sorted(extract_dir.rglob("*.mp4")):
                yield mp4_path.stem, str(mp4_path)
        finally:
            if cleanup:
                print(f"[Disk Manager] Cleaning up {zip_name} to free disk space...")
                shutil.rmtree(extract_dir, ignore_errors=True)
                Path(local_zip).unlink(missing_ok=True)


def load_checkpoint(ckpt_path: Path) -> Tuple[Set[Tuple[str, str]], List[dict]]:
    """Loads previously saved results so interrupted runs resume cleanly."""
    done_pairs: Set[Tuple[str, str]] = set()
    records: List[dict] = []
    if ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records.append(rec)
                    done_pairs.add((rec["scene_name"], rec["method"]))
    return done_pairs, records


def main_long(max_videos: Optional[int] = 10):
    ckpt_path = Path("vsi_full_results.jsonl")

    # 1. Load all VSI-Bench QA pairs and group by scene_name
    vsi_questions = load_vsi_bench(config="full")
    questions_by_scene: Dict[str, List[dict]] = defaultdict(list)
    for q in vsi_questions:
        questions_by_scene[q["scene_name"]].append(q)

    print(
        f"Loaded {len(vsi_questions)} questions across "
        f"{len(questions_by_scene)} unique VSI-Bench scenes."
    )

    # 2. Resume from checkpoint if present
    done_pairs, all_records = load_checkpoint(ckpt_path)
    if done_pairs:
        print(
            f"Resuming: {len(done_pairs)} (scene, method) evaluations already completed."
        )

    # 3. Load the small VLM once on GPU
    vlm_bundle = load_small_vlm("Salesforce/blip-vqa-base")

    # 4. Swappable KFS methods to compare
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

    # 5. Stream through VSI-Bench up to `max_videos`
    processed_videos = 0
    target_total = max_videos if max_videos is not None else len(questions_by_scene)

    with ckpt_path.open("a", encoding="utf-8") as ckpt_f:
        for scene_name, video_path in iter_all_vsi_videos(cleanup=True):
            if max_videos is not None and processed_videos >= max_videos:
                break

            scene_questions = questions_by_scene.get(scene_name)
            if not scene_questions:
                continue

            processed_videos += 1
            print(
                f"\n[Video {processed_videos}/{target_total}] Scene '{scene_name}' ({len(scene_questions)} questions)"
            )

            for method in kfs_methods:
                if (scene_name, method.name) in done_pairs:
                    continue

                # Run KFS once per video (question-independent)
                sampled_video = perform_kfs(
                    video_path=video_path,
                    kfs_method=method,
                    max_frames=None,
                )

                # Reuse those exact keyframes across every question for this scene
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

                    rec = {
                        "scene_name": scene_name,
                        "question_id": q["id"],
                        "question_type": q["question_type"],
                        "method": method.name,
                        "num_frames": len(sampled_video),
                        "prediction": vlm_answer,
                        "ground_truth": q["ground_truth"],
                        "accuracy": acc,
                    }
                    ckpt_f.write(json.dumps(rec) + "\n")
                    all_records.append(rec)

                ckpt_f.flush()
                done_pairs.add((scene_name, method.name))

    # 6. Print overall accuracy across evaluated videos
    stats = defaultdict(lambda: {"acc": [], "frames": []})
    for r in all_records:
        stats[r["method"]]["acc"].append(r["accuracy"])
        stats[r["method"]]["frames"].append(r["num_frames"])

    print("\n" + "=" * 72)
    print(f"VSI-BENCH EVALUATION SUMMARY ({processed_videos} VIDEOS)")
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
    main_long(max_videos=10)  # Set to None to run all videos
