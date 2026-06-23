#!/usr/bin/env python3
"""Create a scale-normalized UAVID root from larger crops.

The intended default turns each source frame into one deterministic random
2048x2048 crop and resizes it to 1024x1024. This keeps the original number of
images while making every 1024 image represent a larger UAVID field of view.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMG_SUFFIX = "_img.png"
ANN_SUFFIX = "_ann.png"
PRIORITY_SMALL_OBJECT_IDS = (3, 6, 7)  # static car, human, moving car


@dataclass(frozen=True)
class CropWindow:
    x: int
    y: int
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess UAVID by cropping larger source windows and resizing "
            "them to a smaller square training representation."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/uavid"),
        help="Existing UAVID root with img_dir/<split> and ann_dir/<split>.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("temp/uavid_2048to1024"),
        help=(
            "Output root to create in the same UAVID folder format."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to preprocess. Step 1/4 need train and val.",
    )
    parser.add_argument("--crop-size", type=int, default=2048, help="Source crop side length.")
    parser.add_argument("--output-size", type=int, default=1024, help="Resized output side length.")
    parser.add_argument(
        "--crop-policy",
        choices=["random-one", "grid"],
        default="random-one",
        help=(
            "How many crop windows to write per source image. 'random-one' keeps "
            "the output count equal to the input count; 'grid' writes coverage crops."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Seed for deterministic per-image random crops.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Crop stride in source pixels for --crop-policy grid. Defaults to --crop-size.",
    )
    parser.add_argument(
        "--axis-mode",
        choices=["center-if-small-remainder", "cover"],
        default="center-if-small-remainder",
        help=(
            "How to place crop windows along each axis. The default avoids near-duplicate "
            "edge crops when the source axis is only slightly larger than crop-size "
            "(for example UAVID's 2160px height with 2048px crops)."
        ),
    )
    parser.add_argument(
        "--small-remainder-threshold",
        type=float,
        default=0.25,
        help=(
            "For center-if-small-remainder, center a single crop on an axis when "
            "(axis_length - crop_size) / crop_size is below this value."
        ),
    )
    parser.add_argument(
        "--label-resize",
        choices=["nearest", "priority"],
        default="nearest",
        help=(
            "Label downsampling mode. 'nearest' is the standard semantic resize. "
            "'priority' preserves UAVID small-object ids inside each integer downsample block."
        ),
    )
    parser.add_argument(
        "--priority-label-ids",
        nargs="+",
        type=int,
        default=list(PRIORITY_SMALL_OBJECT_IDS),
        help="Train ids preserved first when --label-resize priority is used.",
    )
    parser.add_argument(
        "--skip-single-class",
        action="store_true",
        help="Skip output crops whose resized annotation contains only one class id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split source image limit for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output root before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating files.",
    )
    return parser.parse_args()


def _resample_filter(name: str) -> int:
    # Pillow 9+ exposes Image.Resampling; older builds keep constants on Image.
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


def axis_positions(
    length: int,
    crop_size: int,
    stride: int,
    axis_mode: str,
    small_remainder_threshold: float,
) -> list[int]:
    if length < crop_size:
        raise ValueError(f"Cannot take a {crop_size}px crop from an axis of length {length}.")
    if length == crop_size:
        return [0]

    remainder_fraction = (length - crop_size) / crop_size
    if axis_mode == "center-if-small-remainder" and remainder_fraction < small_remainder_threshold:
        return [(length - crop_size) // 2]

    positions = list(range(0, length - crop_size + 1, stride))
    edge = length - crop_size
    if positions[-1] != edge:
        positions.append(edge)
    return positions


def iter_windows(
    width: int,
    height: int,
    crop_size: int,
    stride: int,
    axis_mode: str,
    small_remainder_threshold: float,
) -> Iterable[CropWindow]:
    xs = axis_positions(width, crop_size, stride, axis_mode, small_remainder_threshold)
    ys = axis_positions(height, crop_size, stride, axis_mode, small_remainder_threshold)
    for y in ys:
        for x in xs:
            yield CropWindow(x=x, y=y, size=crop_size)


def random_window_for_image(
    *,
    width: int,
    height: int,
    crop_size: int,
    seed: int,
    split: str,
    image_name: str,
) -> CropWindow:
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Cannot take a {crop_size}px crop from image '{image_name}' with size {width}x{height}."
        )

    digest = hashlib.sha256(f"{seed}:{split}:{image_name}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    max_x = width - crop_size
    max_y = height - crop_size
    return CropWindow(
        x=rng.randint(0, max_x) if max_x > 0 else 0,
        y=rng.randint(0, max_y) if max_y > 0 else 0,
        size=crop_size,
    )


def crop_windows_for_image(
    *,
    width: int,
    height: int,
    crop_size: int,
    stride: int,
    crop_policy: str,
    axis_mode: str,
    small_remainder_threshold: float,
    seed: int,
    split: str,
    image_name: str,
) -> Iterable[CropWindow]:
    if crop_policy == "random-one":
        yield random_window_for_image(
            width=width,
            height=height,
            crop_size=crop_size,
            seed=seed,
            split=split,
            image_name=image_name,
        )
        return

    yield from iter_windows(
        width=width,
        height=height,
        crop_size=crop_size,
        stride=stride,
        axis_mode=axis_mode,
        small_remainder_threshold=small_remainder_threshold,
    )


def resize_annotation_nearest(ann_crop: Image.Image, output_size: int) -> Image.Image:
    return ann_crop.resize((output_size, output_size), resample=_resample_filter("NEAREST"))


def resize_annotation_priority(
    ann_crop: Image.Image,
    output_size: int,
    priority_label_ids: list[int],
) -> Image.Image:
    arr = np.asarray(ann_crop)
    crop_h, crop_w = arr.shape[:2]
    if crop_h != crop_w or crop_h % output_size != 0:
        return resize_annotation_nearest(ann_crop, output_size)

    factor = crop_h // output_size
    blocks = arr.reshape(output_size, factor, output_size, factor)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(output_size, output_size, factor * factor)

    # Default to nearest-equivalent top-left sampling, then override blocks that
    # contain small-object labels. This avoids dropping tiny UAVID humans/cars
    # during the 2048 -> 1024 shrink.
    resized = blocks[:, :, 0].copy()
    priority_ids = np.asarray(priority_label_ids, dtype=arr.dtype)
    if priority_ids.size == 0:
        return Image.fromarray(resized.astype(arr.dtype))

    counts = np.stack([(blocks == label_id).sum(axis=-1) for label_id in priority_ids], axis=0)
    has_priority = counts.sum(axis=0) > 0
    priority_choice = priority_ids[counts.argmax(axis=0)]
    resized[has_priority] = priority_choice[has_priority]
    return Image.fromarray(resized.astype(arr.dtype))


def resize_annotation(
    ann_crop: Image.Image,
    output_size: int,
    label_resize: str,
    priority_label_ids: list[int],
) -> Image.Image:
    if label_resize == "priority":
        return resize_annotation_priority(ann_crop, output_size, priority_label_ids)
    return resize_annotation_nearest(ann_crop, output_size)


def prepare_output_root(output_root: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    try:
        if output_root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output root already exists: {output_root}. Pass --overwrite to replace it."
                )
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EROFS):
            raise
        raise OSError(
            f"Cannot write output root '{output_root}'. Choose a writable path, "
            "for example '--output-root temp/uavid_2048to1024', or ask for write "
            "permission on the target parent directory."
        ) from exc


def validate_roots(input_root: Path, splits: list[str]) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    for split in splits:
        img_dir = input_root / "img_dir" / split
        ann_dir = input_root / "ann_dir" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image split directory: {img_dir}")
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"Missing annotation split directory: {ann_dir}")


def process_split(args: argparse.Namespace, split: str) -> dict[str, int]:
    input_img_dir = args.input_root / "img_dir" / split
    input_ann_dir = args.input_root / "ann_dir" / split
    output_img_dir = args.output_root / "img_dir" / split
    output_ann_dir = args.output_root / "ann_dir" / split

    if not args.dry_run:
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_ann_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_img_dir.glob(f"*{IMG_SUFFIX}"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    stats = {
        "source_images": 0,
        "written_crops": 0,
        "skipped_missing_annotations": 0,
        "skipped_single_class": 0,
    }

    stride = args.stride or args.crop_size
    for img_path in image_paths:
        ann_path = input_ann_dir / img_path.name.replace(IMG_SUFFIX, ANN_SUFFIX)
        if not ann_path.exists():
            stats["skipped_missing_annotations"] += 1
            continue

        with Image.open(img_path) as img_raw, Image.open(ann_path) as ann_raw:
            image = img_raw.convert("RGB")
            ann = ann_raw.copy()

        if image.size != ann.size:
            raise ValueError(f"Image/annotation size mismatch for {img_path.name}: {image.size} vs {ann.size}")

        width, height = image.size
        stats["source_images"] += 1
        for window in crop_windows_for_image(
            width=width,
            height=height,
            crop_size=args.crop_size,
            stride=stride,
            crop_policy=args.crop_policy,
            axis_mode=args.axis_mode,
            small_remainder_threshold=args.small_remainder_threshold,
            seed=args.seed,
            split=split,
            image_name=img_path.name,
        ):
            crop_box = (window.x, window.y, window.x + window.size, window.y + window.size)
            img_crop = image.crop(crop_box).resize(
                (args.output_size, args.output_size),
                resample=_resample_filter("LANCZOS"),
            )
            ann_crop = resize_annotation(
                ann.crop(crop_box),
                output_size=args.output_size,
                label_resize=args.label_resize,
                priority_label_ids=args.priority_label_ids,
            )

            if args.skip_single_class and len(np.unique(np.asarray(ann_crop))) <= 1:
                stats["skipped_single_class"] += 1
                continue

            base = img_path.name[: -len(IMG_SUFFIX)]
            crop_tag = f"x{window.x:04d}_y{window.y:04d}_s{args.crop_size}_to{args.output_size}"
            out_base = f"{base}_{crop_tag}"
            if not args.dry_run:
                img_crop.save(output_img_dir / f"{out_base}{IMG_SUFFIX}")
                ann_crop.save(output_ann_dir / f"{out_base}{ANN_SUFFIX}")
            stats["written_crops"] += 1

    return stats


def write_metadata(args: argparse.Namespace, split_stats: dict[str, dict[str, int]]) -> None:
    if args.dry_run:
        return
    metadata = {
        "source_dataset": "uavid",
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "crop_size": args.crop_size,
        "output_size": args.output_size,
        "crop_policy": args.crop_policy,
        "seed": args.seed,
        "stride": args.stride or args.crop_size,
        "axis_mode": args.axis_mode,
        "small_remainder_threshold": args.small_remainder_threshold,
        "label_resize": args.label_resize,
        "priority_label_ids": args.priority_label_ids,
        "splits": args.splits,
        "split_stats": split_stats,
        "format": {
            "img_dir": "img_dir/<split>/*_img.png",
            "ann_dir": "ann_dir/<split>/*_ann.png",
        },
    }
    with open(args.output_root / "preprocess_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if args.crop_size <= 0 or args.output_size <= 0:
        raise ValueError("--crop-size and --output-size must be positive.")
    if args.stride is not None and args.stride <= 0:
        raise ValueError("--stride must be positive when provided.")
    if not (0.0 <= args.small_remainder_threshold <= 1.0):
        raise ValueError("--small-remainder-threshold must be in [0, 1].")

    validate_roots(args.input_root, args.splits)
    prepare_output_root(args.output_root, overwrite=args.overwrite, dry_run=args.dry_run)

    split_stats: dict[str, dict[str, int]] = {}
    for split in args.splits:
        split_stats[split] = process_split(args, split)
        stats = split_stats[split]
        print(
            f"{split}: source_images={stats['source_images']} "
            f"written_crops={stats['written_crops']} "
            f"missing_annotations={stats['skipped_missing_annotations']} "
            f"single_class_skips={stats['skipped_single_class']}"
        )

    write_metadata(args, split_stats)
    if args.dry_run:
        print(f"Dry run complete. Would write to: {args.output_root}")
    else:
        print(f"Wrote preprocessed UAVID root: {args.output_root}")


if __name__ == "__main__":
    main()
