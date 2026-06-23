#guided_generation/main_scripts/step_4_train_synthetic.py
import argparse
import json
import os
from pathlib import Path

from guided_generation.utils.pipeline_resolution import (
    normalize_optional_str,
    read_pipeline_metadata,
    resolve_guidance_checkpoint,
    resolve_synthetic_root,
)
from vfm4ss.segmentation_trainer import SemanticSegmentationTrainer
from vfm4ss.models.encoder_config import DEFAULT_ENCODER_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cityscapes",
        choices=["cityscapes", "uavid", "pascal_voc", "cocostuff10k", "bdd100k", "ade20k"],
        help="Dataset to train on (default: %(default)s)",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to the synthetic dataset root.",
    )
    parser.add_argument(
        "--real_root",
        type=str,
        default=None,
        help="Path to the real dataset root in standardized format.",
    )
    parser.add_argument(
        "--original_cs_root",
        type=str,
        default=None,
        help="Legacy alias for --real_root / Cityscapes root.",
    )
    parser.add_argument(
        "--logger_save_dir",
        type=str,
        default="training_logs/cityscapes_combined/temp",
        help="Directory to save training logs",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=5000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override optimizer learning rate. Defaults to 2e-5 for finetuning and 1e-4 from scratch.",
    )
    # UNIFIED SEED
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for model initialization, dataloader order, and training randomness. (default: %(default)s)"
    )
    parser.add_argument(
        "--real_split",
        type=float,
        default=1.0,
        help="Fraction of Real data to use [0.0 - 1.0]. (default: %(default)s)"
    )
    parser.add_argument(
        "--syn_split",
        type=float,
        default=1.0,
        help="Fraction of Synthetic data to use [0.0 - 1.0]. (default: %(default)s)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default=DEFAULT_ENCODER_NAME,
        help="timm encoder backbone name for the segmentation model (default: %(default)s)",
    )
    parser.add_argument(
        "--metric_output_file",
        type=str,
        default=None,
        help="If set, write final validation metrics as JSON to this file path.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def step_4_train_synthetic(
    dataset_name: str = "cityscapes",
    root: str = "",
    real_root: str | None = None,
    original_cs_root: str | None = None,
    logger_save_dir: str = "training_logs/cityscapes_combined/temp",
    train_steps: int = 5000,
    learning_rate: float | None = None,
    seed: int = 42,
    real_split: float = 1.0,
    syn_split: float = 1.0,
    early_stopping_patience: int = None,
    ckpt_path: str = None,
    encoder_name: str = DEFAULT_ENCODER_NAME,
    metric_output_file: str = None,
    batch_size: int = 4,
    num_workers: int = 16,
) -> None:
    resolved_real_root = real_root if real_root is not None else original_cs_root
    resolved_root = resolve_synthetic_root(dataset_name=dataset_name, requested_root=root)
    if resolved_root is None or not Path(resolved_root).exists():
        raise FileNotFoundError(
            f"Could not resolve a synthetic dataset root for dataset '{dataset_name}'. "
            f"Requested root was: '{root}'."
        )

    synthetic_metadata = read_pipeline_metadata(resolved_root)
    if encoder_name == DEFAULT_ENCODER_NAME:
        cached_encoder_name = normalize_optional_str(synthetic_metadata.get("encoder_name"))
        if cached_encoder_name is not None:
            encoder_name = cached_encoder_name

    resolved_ckpt_path = resolve_guidance_checkpoint(
        dataset_name=dataset_name,
        requested_path=ckpt_path,
        metadata_root=resolved_root,
    )

    segmentation_trainer = SemanticSegmentationTrainer(
        root=resolved_root,
        original_cs_root=resolved_real_root,
        real_root=resolved_real_root,
        dataset_type=f"{dataset_name}_combined",
        logger_save_dir=logger_save_dir,
        train_steps=train_steps,
        learning_rate=learning_rate,
        seed=seed,
        real_split=real_split,
        syn_split=syn_split,
        early_stopping_patience=early_stopping_patience,
        ckpt_path=resolved_ckpt_path,
        encoder_name=encoder_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    metrics = segmentation_trainer.train()

    if metric_output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(metric_output_file)), exist_ok=True)
        with open(metric_output_file, "w") as f:
            json.dump({"seed": seed, **metrics}, f, indent=2)

    miou = metrics.get("val_0_miou", float("nan"))
    final_miou = metrics.get("final_val_0_miou", float("nan"))
    best_checkpoint_path = metrics.get("best_checkpoint_path", "")
    print(
        "STEP4_RESULT "
        f"run_seed={seed} "
        f"val_0_miou={miou:.6f} "
        f"final_val_0_miou={final_miou:.6f} "
        f"best_checkpoint_path={best_checkpoint_path}"
    )


if __name__ == "__main__":
    args = parse_args()
    step_4_train_synthetic(**vars(args))
