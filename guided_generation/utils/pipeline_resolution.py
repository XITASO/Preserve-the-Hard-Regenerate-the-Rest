from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PIPELINE_METADATA_FILENAME = "pipeline_metadata.json"
KNOWN_DATASET_NAMES = ("cityscapes", "uavid", "pascal_voc", "cocostuff10k", "bdd100k", "ade20k")


def normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "auto":
        return None
    return stripped


def read_pipeline_metadata(root: str | Path | None) -> dict[str, Any]:
    if root is None:
        return {}

    metadata_path = Path(root) / PIPELINE_METADATA_FILENAME
    if not metadata_path.is_file():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def write_pipeline_metadata(root: str | Path, metadata: dict[str, Any]) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    existing_metadata = read_pipeline_metadata(root_path)
    merged_metadata = {
        key: value
        for key, value in {**existing_metadata, **metadata}.items()
        if value is not None
    }

    metadata_path = root_path / PIPELINE_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_metadata, handle, indent=2, sort_keys=True)


def _checkpoint_sort_key(path: Path) -> tuple[int, int, float]:
    match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt$", path.name)
    epoch = int(match.group(1)) if match else -1
    step = int(match.group(2)) if match else -1
    try:
        modified_time = path.stat().st_mtime
    except OSError:
        modified_time = -1.0
    return (step, epoch, modified_time)


def _prefer_non_archived_candidates(candidates: list[Path]) -> list[Path]:
    non_archived = [path for path in candidates if "old_ones" not in path.parts]
    return non_archived if non_archived else candidates


def _search_checkpoint_candidates(dataset_name: str) -> list[Path]:
    search_root = Path("training_logs")
    if not search_root.is_dir():
        return []

    patterns = [f"{dataset_name}/**/checkpoints/*.ckpt"]
    if dataset_name == "cityscapes":
        patterns.extend(
            [
                "pretrained*/**/checkpoints/*.ckpt",
                "cityscapes/**/checkpoints/*.ckpt",
            ]
        )

    candidates: list[Path] = []
    seen_paths: set[Path] = set()
    for pattern in patterns:
        for path in search_root.glob(pattern):
            if path not in seen_paths and path.is_file():
                seen_paths.add(path)
                candidates.append(path)

    candidates = _prefer_non_archived_candidates(candidates)
    return sorted(candidates, key=_checkpoint_sort_key, reverse=True)


def resolve_guidance_checkpoint(
    dataset_name: str,
    requested_path: str | None = None,
    metadata_root: str | Path | None = None,
) -> str | None:
    normalized_requested_path = normalize_optional_str(requested_path)
    if normalized_requested_path is not None:
        return normalized_requested_path

    metadata = read_pipeline_metadata(metadata_root)
    for key in ("guidance_checkpoint", "step1_checkpoint", "ckpt_path", "segmentation_checkpoint"):
        metadata_path = normalize_optional_str(metadata.get(key))
        if metadata_path is not None:
            return metadata_path

    candidates = _search_checkpoint_candidates(dataset_name)
    if candidates:
        return str(candidates[0])

    return None


def is_semseg_dataset_root(root: str | Path | None) -> bool:
    if root is None:
        return False

    root_path = Path(root)
    return (root_path / "images" / "train").is_dir() and (root_path / "annotations" / "train").is_dir()


def path_mentions_other_dataset(root: str | Path, dataset_name: str) -> bool:
    root_str = str(root)
    return any(other_name != dataset_name and other_name in root_str for other_name in KNOWN_DATASET_NAMES)


def _synthetic_root_sort_key(path: Path) -> float:
    image_dir = path / "images" / "train"
    annotation_dir = path / "annotations" / "train"
    mtimes = []
    for candidate in (image_dir, annotation_dir, path):
        try:
            mtimes.append(candidate.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else -1.0


def _search_synthetic_roots(dataset_name: str) -> list[Path]:
    dataset_root = Path("synthetic_data") / dataset_name
    if not dataset_root.is_dir():
        return []

    candidates = []
    seen_paths: set[Path] = set()
    for image_dir in dataset_root.glob("**/images/train"):
        candidate_root = image_dir.parent.parent
        if candidate_root in seen_paths or not is_semseg_dataset_root(candidate_root):
            continue
        seen_paths.add(candidate_root)
        candidates.append(candidate_root)

    return sorted(candidates, key=_synthetic_root_sort_key, reverse=True)


def resolve_synthetic_root(dataset_name: str, requested_root: str | None = None) -> str | None:
    normalized_requested_root = normalize_optional_str(requested_root)
    if normalized_requested_root is not None:
        requested_root_path = Path(normalized_requested_root)
        if is_semseg_dataset_root(requested_root_path) and not path_mentions_other_dataset(
            requested_root_path, dataset_name
        ):
            return normalized_requested_root

    candidates = _search_synthetic_roots(dataset_name)
    if candidates:
        return str(candidates[0])

    if normalized_requested_root is not None and not path_mentions_other_dataset(normalized_requested_root, dataset_name):
        return normalized_requested_root

    return None
