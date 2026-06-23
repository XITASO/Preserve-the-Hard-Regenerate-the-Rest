# guided_generation/main_scripts/step_2_select_samples.py
import argparse
from typing import Optional
import os

import albumentations as A
from guided_generation.guidance.dino_guide import DinoGuide
from guided_generation.datasets import build_guided_dataset, get_guided_dataset_spec
from guided_generation.sample_selection.sample_selector import SampleSelector
from guided_generation.utils.pipeline_resolution import write_pipeline_metadata
from vfm4ss.models.encoder_config import DEFAULT_ENCODER_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Select hard samples from a segmentation dataset using a segmentation guide.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cityscapes",
        choices=["cityscapes", "uavid", "pascal_voc", "cocostuff10k", "bdd100k", "ade20k"],
        help="Dataset to process (default: %(default)s)",
    )
    
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data/real_datasets/cityscapes", 
        help="Path to the root directory of the dataset (default: %(default)s)"
    )

    parser.add_argument(
        "--every_nth_sample", type=int, default=1, help="Use every Nth sample from the dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1, help="How many samples to load max from the dataset. -1 = load all."
    )
    parser.add_argument(
        "--subset_split",
        type=float,
        default=1.0,
        help="Deterministic train subset fraction, using the same fixed subset seed as Step 1.",
    )
    parser.add_argument(
        "--transforms_per_sample", type=int, default=1, help="How many random transformations to use of each image."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the segmentation guide checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of segmentation classes (default: dataset-specific).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Square image size used by the segmentation guide for scoring (default: dataset-specific).",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default=DEFAULT_ENCODER_NAME,
        help="timm encoder backbone name for the guidance segmenter (default: %(default)s)",
    )
    parser.add_argument(
        "--selector_type",
        type=str,
        default="highest_entropy_class_multi",
        choices=[
            "highest_entropy_class",
            "highest_entropy_class_multi",
            "lowest_entropy_class_multi",
            "random_class_multi",
            "random_square_region",
        ],
        help="Sample selector type (default: %(default)s)",
    )
    parser.add_argument(
        "--selector_seed",
        type=int,
        default=0,
        help="Deterministic seed for random selector variants (default: %(default)s)",
    )
    parser.add_argument(
        "--min_pixel",
        type=float,
        default=0.05,
        help="Minimum pixel fraction for selector_type_kwargs['min_pixel'] (default: %(default)s)",
    )
    parser.add_argument(
        "--min_obj_size",
        type=float,
        default=0,
        help="Minimum pixel fraction for selector_type_kwargs['min_obj_size'] (default: %(default)s)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Cache directory for selector (default: %(default)s)"
    )
    parser.add_argument("--clear_cache_on_init", type=bool, default=True, help="Clear selected cache folder on init")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to select. -1 = select all")
    parser.add_argument(
        "--square_cache_crops",
        action="store_true",
        help=(
            "Cache square image/annotation/mask crops instead of full-resolution samples. "
            "Useful for high-resolution datasets such as UAVID."
        ),
    )
    parser.add_argument(
        "--cache_crop_size",
        type=int,
        default=None,
        help=(
            "Side length for square cached crops when --square_cache_crops is active. "
            "Defaults to --image_size, so the existing 1024-crop setup is unchanged unless this is set."
        ),
    )
    
    # --- NEW FLAG ---
    parser.add_argument(
        "--save_heatmaps",
        action="store_true",
        help="If set, generates and saves entropy heatmaps to the cache (slows down processing)."
    )
    # ----------------

    # --- SIMPLE MODE ---
    parser.add_argument(
        "--simple_mode",
        action="store_true",
        help=(
            "Bypass entropy-based region selection. Build the inpaint mask "
            "directly from the semantic labels (background classes -> inpaint, "
            "foreground classes -> preserve). The seg model is not invoked."
        ),
    )
    parser.add_argument(
        "--simple_mode_erosion_kernel",
        type=int,
        default=0,
        help=(
            "Deprecated for step 2 mask writing. Simple mode now always saves "
            "object-only preserve masks; use step 3 --mask_erosion_kernel if "
            "you want generation-time erosion."
        ),
    )
    # -------------------

    return parser.parse_args()


def _build_square_cache_transform(crop_size: int) -> A.ReplayCompose:
    try:
        crop = A.RandomCrop(width=crop_size, height=crop_size, pad_if_needed=True)
        transforms = [crop]
    except TypeError:
        # Older Albumentations builds do not support RandomCrop(pad_if_needed).
        transforms = [
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size),
            A.RandomCrop(width=crop_size, height=crop_size),
        ]

    return A.ReplayCompose(
        [
            *transforms,
            A.HorizontalFlip(p=0.5),
            A.ToTensorV2(),
        ]
    )


def step_2_select_samples(
    dataset_name: str = "cityscapes",
    data_root: str = "data/real_datasets/cityscapes",
    every_nth_sample: int = 1,
    max_samples: int = -1,
    subset_split: float = 1.0,
    ckpt_path: str | None = None,
    num_classes: int | None = None,
    image_size: int | None = None,
    encoder_name: str = DEFAULT_ENCODER_NAME,
    selector_type: str = "highest_entropy_class_multi",
    selector_seed: int = 0,
    min_pixel: float = 0.05,
    min_obj_size: float = 0,
    cache_dir: Optional[str] = None,
    clear_cache_on_init: bool = True,
    num_samples: int = -1,
    transforms_per_sample: int = 1,
    square_cache_crops: bool = False,
    cache_crop_size: int | None = None,
    save_heatmaps: bool = False, # New Arg
    simple_mode: bool = False,
    simple_mode_erosion_kernel: int = 0,
) -> None:
    """Select samples from a segmentation dataset using a Dino guide and cache them to disk."""
    
    print(f"DEBUG: Initializing Dataset from root: {data_root}")
    dataset_spec = get_guided_dataset_spec(dataset_name)
    if image_size is None:
        image_size = dataset_spec.default_image_size
    if num_classes is None:
        num_classes = dataset_spec.num_classes
    if cache_crop_size is None:
        cache_crop_size = image_size
    if cache_crop_size <= 0:
        raise ValueError(f"--cache_crop_size must be positive, got {cache_crop_size}.")

    dataset = build_guided_dataset(
        dataset_name=dataset_name,
        root_dir=data_root, 
        split="train", 
        transform=A.ToTensorV2(), 
        every_nth_sample=every_nth_sample, 
        max_samples=max_samples,
        subset_split=subset_split,
    )
    
    print(f"DEBUG: Dataset initialized with {len(dataset)} samples.")

    guide_required = selector_type in {
        "highest_entropy_class",
        "highest_entropy_class_multi",
        "lowest_entropy_class_multi",
    }
    if simple_mode:
        # Simple mode skips entropy scoring entirely, so we do not need to
        # load the segmentation guide checkpoint.
        print("DEBUG: --simple_mode enabled -> skipping DinoGuide construction.")
        guide = None
    elif not guide_required:
        print(f"DEBUG: selector_type={selector_type} does not require DinoGuide construction.")
        guide = None
    else:
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No guidance checkpoint was provided for dataset '{dataset_name}'. "
                "Pass --ckpt_path or run through SLURM/run_pipeline.sh after step 1."
            )
        guide = DinoGuide(
            ckpt_path=ckpt_path,
            img_size=(image_size, image_size),
            num_classes=num_classes,
            encoder_name=encoder_name,
        ).to("cuda")

    # Entropy mode normally caches full-resolution samples and only resizes
    # temporary scoring copies. High-resolution datasets can opt into square
    # cached crops with a side length independent from the guide/scoring size.
    # For UAVID this lets us keep native-scale 2048 crops while still scoring
    # and generating through the 1024 model/bucket family.
    cache_square_crops = simple_mode or square_cache_crops
    if cache_square_crops:
        transform = _build_square_cache_transform(cache_crop_size)
    else:
        transform = A.ReplayCompose(
            [
                A.ToTensorV2(),
            ]
        )

    selector = SampleSelector(
        dataset=dataset,
        selector_type=selector_type,
        seg_model=guide.segmenter if guide is not None else None,
        clear_cache_on_init=clear_cache_on_init,
        selector_type_kwargs={
            "min_pixel": min_pixel,
            "min_obj_size": min_obj_size,
            "selector_seed": selector_seed,
        },
        cache_dir=cache_dir,
        transform=transform,
        num_workers=16,
        score_image_size=(image_size, image_size),
        transforms_per_sample=transforms_per_sample,
        save_heatmaps=save_heatmaps, # Pass to Selector
        simple_mode=simple_mode,
        simple_mode_erosion_kernel=simple_mode_erosion_kernel,
    )

    selected_samples = selector.select_samples(num_samples=num_samples)
    write_pipeline_metadata(
        selector.cache_dir,
        {
            "dataset_name": dataset_name,
            "data_root": data_root,
            "guidance_checkpoint": ckpt_path,
            "step1_checkpoint": ckpt_path,
            "encoder_name": encoder_name,
            "guide_image_size": [image_size, image_size],
            "num_classes": num_classes,
            "selector_type": "simple_class_based" if simple_mode else selector_type,
            "selector_seed": selector_seed,
            "simple_mode": simple_mode,
            "square_cache_crops": cache_square_crops,
            "cache_crop_size": cache_crop_size if cache_square_crops else None,
            "cached_image_size": [cache_crop_size, cache_crop_size] if cache_square_crops else None,
            "simple_mode_erosion_kernel": 0,
            "simple_mode_saved_mask": "foreground_objects_only" if simple_mode else None,
            "mask_suffix": dataset.ann_suffix,
            "cache_dir": selector.cache_dir,
            "subset_split": subset_split,
        },
    )
    return selected_samples


if __name__ == "__main__":
    args = parse_args()
    step_2_select_samples(**vars(args))
