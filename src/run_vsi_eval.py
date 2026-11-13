from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, Set, Tuple
import numpy as np

from base import UniformSamplingMethod
from cascade import EgoMotionAndCascadeKFS
from kfs_extractor import extract_keyframes
from methods import (
    ColorHistogramMethod,
    FarnebackOpticalFlowMethod,
    FourierMethod,
    LocalDescriptorMethod,
    LucasKanadeOpticalFlowMethod,
    PixelIntensityMethod,
)
from metrics import score_vsi_prediction
from vlm_engine import SmallVLMEvaluator
from vsi_loader import iter_vsi_videos_low_disk, load_all_vsi_questions


def build_kfs_methods():
    return [
        UniformSamplingMethod(step=30),  # Reference Baseline (Sampling Method 1)
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


def load_checkpoint(ckpt_path: Path) -> Tuple[Set[Tuple[str, str]], list]:
    completed_pairs: Set[Tuple[str, str]] = set()
    records = []
    if ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records.append(rec)
                    completed_pairs.add((rec["scene_name"], rec["method"]))
    return completed_pairs, records


def main():
    FRAME_BUDGET = 16
    MAX_LONG_EDGE = 384  # Keeps memory & latency low on GTX 980 Ti
    CKPT_PATH = Path("vsi_checkpoint.jsonl")

    questions_by_scene = load_all_vsi_questions(split="test")
    total_scenes = len(questions_by_scene)
    print(
        f"Loaded {sum(len(v) for v in questions_by_scene.values())} questions across {total_scenes} VSI-Bench scenes."
    )

    completed_pairs, all_records = load_checkpoint(CKPT_PATH)
    if completed_pairs:
        print(
            f"Resuming from checkpoint: {len(completed_pairs)} (scene, method) runs already done."
        )

    vlm = SmallVLMEvaluator(model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    methods = build_kfs_methods()

    with CKPT_PATH.open("a", encoding="utf-8") as ckpt_file:
        for scene_idx, (scene_name, video_path) in enumerate(
            iter_vsi_videos_low_disk(
                work_dir="./assets/vsi_tmp", cleanup_after_zip=True
            ),
            start=1,
        ):
            questions = questions_by_scene.get(scene_name)
            if not questions:
                continue

            print(
                f"\n[{scene_idx}/{total_scenes}] Scene: {scene_name} ({len(questions)} questions)"
            )

            for method in methods:
                if (scene_name, method.name) in completed_pairs:
                    continue

                # 1. Run KFS once per video (question-independent)
                kf_res = extract_keyframes(
                    video_path=video_path,
                    method=method,
                    frame_budget=FRAME_BUDGET,
                    max_long_edge=MAX_LONG_EDGE,
                )

                # 2. Query VLM on all questions for this scene using the exact same selected frames
                for q in questions:
                    pred = vlm.answer_question(
                        frames_with_ts=kf_res.frames,
                        question=q["question"],
                        options=q["options"],
                    )
                    score = score_vsi_prediction(
                        pred=pred,
                        ground_truth=q["ground_truth"],
                        has_options=bool(q["options"]),
                    )

                    rec = {
                        "scene_name": scene_name,
                        "question_id": q["id"],
                        "question_type": q["question_type"],
                        "reasoning_category": q["reasoning_category"],
                        "method": method.name,
                        "num_frames": len(kf_res.frames),
                        "cost_sec": kf_res.selection_time_sec,
                        "prediction": pred,
                        "ground_truth": q["ground_truth"],
                        "score": score,
                    }
                    ckpt_file.write(json.dumps(rec) + "\n")
                    all_records.append(rec)

                ckpt_file.flush()
                completed_pairs.add((scene_name, method.name))

    # Build final summary table from all checkpointed records
    summary = defaultdict(
        lambda: {
            "num_frames": [],
            "cost_sec": [],
            "SPATIAL_LAYOUT": [],
            "ACTION_EVENT_STATE": [],
            "HYBRID": [],
            "ALL": [],
        }
    )
    for r in all_records:
        m = r["method"]
        summary[m]["num_frames"].append(r["num_frames"])
        summary[m]["cost_sec"].append(r["cost_sec"])
        summary[m][r["reasoning_category"]].append(r["score"])
        summary[m]["ALL"].append(r["score"])

    print("\n" + "=" * 95)
    print(
        f"{'Method':<34} | {'Frames':>6} | {'Cost(s)':>7} | {'Spatial':>8} | {'Temporal':>8} | {'Hybrid':>8} | {'Overall':>8}"
    )
    print("-" * 95)
    for m in methods:
        s = summary[m.name]
        avg_f = np.mean(s["num_frames"]) if s["num_frames"] else 0.0
        avg_c = np.mean(s["cost_sec"]) if s["cost_sec"] else 0.0
        sp = np.mean(s["SPATIAL_LAYOUT"]) * 100 if s["SPATIAL_LAYOUT"] else 0.0
        tm = np.mean(s["ACTION_EVENT_STATE"]) * 100 if s["ACTION_EVENT_STATE"] else 0.0
        hy = np.mean(s["HYBRID"]) * 100 if s["HYBRID"] else 0.0
        ov = np.mean(s["ALL"]) * 100 if s["ALL"] else 0.0
        print(
            f"{m.name:<34} | {avg_f:>6.1f} | {avg_c:>7.3f} | {sp:>7.1f}% | {tm:>7.1f}% | {hy:>7.1f}% | {ov:>7.1f}%"
        )


if __name__ == "__main__":
    main()
