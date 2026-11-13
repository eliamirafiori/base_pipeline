from pathlib import Path
import re
import shutil
from typing import Any, Dict, Generator, List, Tuple
import zipfile
from datasets import load_dataset
from huggingface_hub import hf_hub_download

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


def classify_question(question: str, question_type: str) -> str:
    q = question.lower()
    sp = any(re.search(rf"\b{re.escape(w)}\b", q) for w in SPATIAL_CUES)
    ac = any(re.search(rf"\b{re.escape(w)}\b", q) for w in ACTION_CUES)
    if "route" in question_type or "order" in question_type or (sp and ac):
        return "HYBRID"
    if ac:
        return "ACTION_EVENT_STATE"
    return "SPATIAL_LAYOUT"


def load_vsi_grouped_by_scene(split: str = "test") -> Dict[str, List[Dict[str, Any]]]:
    vsi_bench = load_dataset("nyu-visionx/VSI-Bench", "full")
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in vsi_bench[split]:
        scene = str(row["scene_name"])
        q_type = row.get("question_type", "unknown")
        grouped.setdefault(scene, []).append(
            {
                "id": int(row["id"]),
                "dataset": row["dataset"],
                "scene_name": scene,
                "question_type": q_type,
                "reasoning_category": classify_question(row["question"], q_type),
                "question": row["question"],
                "options": row.get("options"),
                "ground_truth": str(row["ground_truth"]).strip(),
            }
        )
    return grouped


def iter_vsi_videos_from_zips(
    work_dir: str = "./vsi_cache",
    archives: Tuple[str, ...] = ("arkitscenes.zip", "scannet.zip", "scannetpp.zip"),
) -> Generator[Tuple[str, str], None, None]:
    """Downloads and extracts one VSI-Bench zip archive at a time, then deletes it."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    for zip_name in archives:
        print(f"\nDownloading {zip_name} from nyu-visionx/VSI-Bench...")
        zip_path = hf_hub_download(
            repo_id="nyu-visionx/VSI-Bench",
            repo_type="dataset",
            filename=zip_name,
            local_dir=str(work / "zips"),
        )
        extract_dir = work / Path(zip_name).stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        for mp4 in sorted(extract_dir.rglob("*.mp4")):
            yield mp4.stem, str(mp4)

        shutil.rmtree(extract_dir, ignore_errors=True)
        Path(zip_path).unlink(missing_ok=True)
