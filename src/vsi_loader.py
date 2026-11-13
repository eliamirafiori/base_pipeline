from pathlib import Path
import re
import shutil
from typing import Any, Dict, Generator, List, Tuple
import zipfile
from datasets import load_dataset
from huggingface_hub import HfFileSystem, hf_hub_download

SPATIAL_CUES = {
    "left",
    "right",
    "behind",
    "in front of",
    "between",
    "next to",
    "above",
    "below",
    "under",
    "near",
    "closer",
    "farthest",
    "closest",
    "nearest",
    "distance",
    "meters",
    "feet",
    "dimensions",
    "room",
    "layout",
    "where is",
    "how many",
    "size",
    "length",
    "width",
    "height",
    "direction",
    "facing",
}

ACTION_CUES = {
    "move",
    "moved",
    "appear",
    "first",
    "last",
    "order",
    "sequence",
    "before",
    "after",
    "when",
    "then",
    "camera",
    "walk",
    "turn",
    "pan",
    "path",
    "route",
    "visited",
    "change",
    "changed",
}


def classify_reasoning_type(question: str, question_type: str) -> str:
    q_lower = question.lower()
    has_spatial = any(re.search(rf"\b{re.escape(w)}\b", q_lower) for w in SPATIAL_CUES)
    has_action = any(re.search(rf"\b{re.escape(w)}\b", q_lower) for w in ACTION_CUES)

    if "route" in question_type or "order" in question_type:
        return "HYBRID"
    if has_spatial and has_action:
        return "HYBRID"
    if has_action:
        return "ACTION_EVENT_STATE"
    return "SPATIAL_LAYOUT"


def load_all_vsi_questions(split: str = "test") -> Dict[str, List[Dict[str, Any]]]:
    """Downloads all VSI-Bench QA annotations (~2 MB) and groups them by scene_name."""
    ds = load_dataset("nyu-visionx/VSI-Bench", split=split)
    by_scene: Dict[str, List[Dict[str, Any]]] = {}

    for row in ds:
        scene = str(row["scene_name"])
        q_text = row["question"]
        q_type = row.get("question_type", "unknown")
        by_scene.setdefault(scene, []).append(
            {
                "id": int(row["id"]),
                "dataset": row["dataset"],
                "scene_name": scene,
                "question_type": q_type,
                "reasoning_category": classify_reasoning_type(q_text, q_type),
                "question": q_text,
                "options": row.get("options"),
                "ground_truth": str(row["ground_truth"]).strip(),
            }
        )
    return by_scene


def iter_vsi_videos_low_disk(
    work_dir: str = "./assets/vsi_tmp",
    cleanup_after_zip: bool = True,
) -> Generator[Tuple[str, str], None, None]:
    """
    Discovers the .zip archives in nyu-visionx/VSI-Bench on Hugging Face,
    downloads & extracts ONE archive at a time, yields (scene_name, mp4_path),
    and deletes the extracted files afterwards to keep disk usage minimal.
    """
    repo_id = "nyu-visionx/VSI-Bench"
    fs = HfFileSystem()
    zip_files = sorted(fs.glob(f"datasets/{repo_id}/**/*.zip"))

    if not zip_files:
        raise RuntimeError(f"No .zip archives found in {repo_id}.")

    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    for full_hf_path in zip_files:
        # Strip 'datasets/nyu-visionx/VSI-Bench/' prefix to get repo-relative filename
        rel_path = full_hf_path.split(f"{repo_id}/", 1)[1]
        print(f"\n[Storage Manager] Downloading archive: {rel_path} ...")
        local_zip = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=rel_path,
            local_dir=str(work_path / "zips"),
        )

        extract_dir = work_path / "extracted" / Path(rel_path).stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Storage Manager] Extracting {rel_path} ...")
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(extract_dir)

        for mp4 in sorted(extract_dir.rglob("*.mp4")):
            yield mp4.stem, str(mp4)

        if cleanup_after_zip:
            print(f"[Storage Manager] Cleaning up {rel_path} to free disk space...")
            shutil.rmtree(extract_dir, ignore_errors=True)
            Path(local_zip).unlink(missing_ok=True)
