from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_COUNTS = {
    ("images/10k", "train"): 7000,
    ("images/10k", "val"): 1000,
    ("images/10k", "test"): 2000,
    ("labels/sem_seg/masks", "train"): 7000,
    ("labels/sem_seg/masks", "val"): 1000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize an unpacked BDD100K Kaggle/official download into the "
            "folder layout used by this repo."
        )
    )
    parser.add_argument(
        "--unpack-root",
        type=Path,
        default=Path("data/bdd100k/_unpack"),
        help="Directory where the BDD100K zips were unpacked.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/bdd100k"),
        help="Final dataset root consumed by Step 1/2/4.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Materialization mode for files in the final layout.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing final split folders before materializing them.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Warn instead of failing if a split has an unexpected number of files.",
    )
    parser.add_argument(
        "--mask-value-samples",
        type=int,
        default=32,
        help="Number of masks per split to inspect for valid values. Set 0 to skip.",
    )
    return parser.parse_args()


def find_source_dir(
    unpack_root: Path,
    rel_dir: Path,
    suffix: str,
) -> Path:
    parts = rel_dir.parts
    candidates: list[tuple[int, Path]] = []
    for candidate in unpack_root.rglob(parts[-1]):
        if not candidate.is_dir():
            continue
        if tuple(candidate.parts[-len(parts):]) != parts:
            continue
        count = sum(1 for path in candidate.iterdir() if path.is_file() and path.name.endswith(suffix))
        if count > 0:
            candidates.append((count, candidate))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find source directory ending in '{rel_dir}' with '*{suffix}' files under '{unpack_root}'."
        )

    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return candidates[0][1]


def materialize_dir(source: Path, target: Path, suffix: str, mode: str, replace: bool) -> int:
    if target.exists():
        existing = [path for path in target.iterdir() if path.is_file() and path.name.endswith(suffix)]
        if existing and not replace:
            return len(existing)
        if replace:
            shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    count = 0
    for source_file in sorted(path for path in source.iterdir() if path.is_file() and path.name.endswith(suffix)):
        target_file = target / source_file.name
        if target_file.exists() or target_file.is_symlink():
            target_file.unlink()
        if mode == "symlink":
            os.symlink(source_file.resolve(), target_file)
        else:
            shutil.copy2(source_file, target_file)
        count += 1
    return count


def check_count(label: str, count: int, expected: int, allow_mismatch: bool) -> None:
    if count == expected:
        print(f"OK: {label}: {count}")
        return
    message = f"Count mismatch for {label}: got {count}, expected {expected}"
    if allow_mismatch:
        print(f"WARNING: {message}")
        return
    raise RuntimeError(message)


def check_mask_values(mask_dir: Path, num_samples: int) -> None:
    if num_samples <= 0:
        return
    valid_values = set(range(19)) | {255}
    mask_paths = sorted(mask_dir.glob("*.png"))[:num_samples]
    bad_values: set[int] = set()
    for mask_path in mask_paths:
        values = np.unique(np.asarray(Image.open(mask_path)))
        bad_values.update(int(value) for value in values if int(value) not in valid_values)
    if bad_values:
        raise RuntimeError(
            f"Unexpected BDD100K mask values in '{mask_dir}': {sorted(bad_values)}. "
            "Expected train IDs 0..18 plus 255 ignore."
        )
    print(f"OK: mask values in {mask_dir} are within 0..18 plus 255")


def main() -> None:
    args = parse_args()
    unpack_root = args.unpack_root
    output_root = args.output_root

    if not unpack_root.is_dir():
        raise FileNotFoundError(f"--unpack-root does not exist: {unpack_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (Path("images/10k/train"), ".jpg"),
        (Path("images/10k/val"), ".jpg"),
        (Path("images/10k/test"), ".jpg"),
        (Path("labels/sem_seg/masks/train"), ".png"),
        (Path("labels/sem_seg/masks/val"), ".png"),
    ]

    for rel_dir, suffix in jobs:
        source = find_source_dir(unpack_root, rel_dir, suffix)
        target = output_root / rel_dir
        count = materialize_dir(source, target, suffix, args.mode, args.replace)
        expected = EXPECTED_COUNTS[(str(rel_dir.parent), rel_dir.name)]
        check_count(str(rel_dir), count, expected, args.allow_count_mismatch)
        print(f"{source} -> {target}")

    check_mask_values(output_root / "labels/sem_seg/masks/train", args.mask_value_samples)
    check_mask_values(output_root / "labels/sem_seg/masks/val", args.mask_value_samples)
    print(f"BDD100K semantic segmentation root is ready: {output_root}")


if __name__ == "__main__":
    main()
