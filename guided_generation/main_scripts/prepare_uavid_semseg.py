from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_configs import get_dataset_spec


SPLIT_DIRS = {
    "train": "uavid_train",
    "val": "uavid_val",
    "test": "uavid_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the official UAVID sequence layout into the flat train-ID "
            "layout used by this repository."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/uavid_official"),
        help="Root containing uavid_train/, uavid_val/, and optionally uavid_test/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/uavid"),
        help="Output root containing img_dir/<split> and ann_dir/<split>.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_DIRS),
        default=["train", "val"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files with the same names.",
    )
    return parser.parse_args()


def color_to_train_id(label: np.ndarray) -> np.ndarray:
    if label.ndim == 2:
        values = label.astype(np.uint8)
        if values.size and int(values.max()) > 7:
            raise ValueError("Single-channel UAVID labels must contain IDs 0..7.")
        return values
    if label.ndim != 3 or label.shape[2] < 3:
        raise ValueError(f"Expected an RGB or single-channel label, got {label.shape}.")

    palette = get_dataset_spec("uavid").palette
    output = np.zeros(label.shape[:2], dtype=np.uint8)
    matched = np.zeros(label.shape[:2], dtype=bool)
    rgb = label[:, :, :3]
    for train_id, color in palette.items():
        mask = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)
        output[mask] = train_id
        matched |= mask
    if not np.all(matched):
        unknown = np.unique(rgb[~matched].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown UAVID label colors: {unknown[:10].tolist()}")
    return output


def prepare_split(
    input_root: Path,
    output_root: Path,
    split: str,
    overwrite: bool,
) -> tuple[int, int]:
    source_root = input_root / SPLIT_DIRS[split]
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing official UAVID split: {source_root}")

    image_output = output_root / "img_dir" / split
    label_output = output_root / "ann_dir" / split
    image_output.mkdir(parents=True, exist_ok=True)
    if split != "test":
        label_output.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    for sequence in sorted(path for path in source_root.iterdir() if path.is_dir()):
        image_dir = sequence / "Images"
        label_dir = sequence / "Labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            stem = image_path.stem
            output_name = f"{sequence.name}_{stem}_img.png"
            output_path = image_output / output_name
            if overwrite or not output_path.exists():
                shutil.copy2(image_path, output_path)
            image_count += 1

            if split == "test":
                continue
            label_path = label_dir / image_path.name
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing UAVID label for {image_path}")
            label_name = f"{sequence.name}_{stem}_ann.png"
            label_output_path = label_output / label_name
            if overwrite or not label_output_path.exists():
                with Image.open(label_path) as label_image:
                    train_ids = color_to_train_id(np.asarray(label_image))
                Image.fromarray(train_ids, mode="L").save(label_output_path)
            label_count += 1

    if image_count == 0:
        raise RuntimeError(f"No UAVID PNG images found under {source_root}.")
    if split != "test" and image_count != label_count:
        raise RuntimeError(
            f"UAVID {split} image/label count mismatch: {image_count} vs {label_count}."
        )
    return image_count, label_count


def main() -> None:
    args = parse_args()
    for split in args.splits:
        image_count, label_count = prepare_split(
            args.input_root,
            args.output_root,
            split,
            args.overwrite,
        )
        print(f"{split}: {image_count} images, {label_count} labels")
    print(f"UAVID train-ID root is ready: {args.output_root}")


if __name__ == "__main__":
    main()
