#guided_generation/main_scripts/step_1_train_real.py
import argparse
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
        default="training_logs/cityscapes/temp",
        help="Directory to save training logs",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=5000,
        help="Number of training steps.",
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
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default=DEFAULT_ENCODER_NAME,
        help="timm encoder backbone name for the segmentation model (default: %(default)s)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def step_1_train_real(
    dataset_name: str = "cityscapes",
    real_root: str | None = None,
    original_cs_root: str | None = None,
    logger_save_dir: str = "training_logs/cityscapes/temp",
    train_steps: int = 5000,
    seed: int = 42,
    real_split: float = 1.0,
    early_stopping_patience: int = None, 
    encoder_name: str = DEFAULT_ENCODER_NAME,
    batch_size: int = 4,
    num_workers: int = 16,
) -> None:
    resolved_real_root = real_root if real_root is not None else original_cs_root
    segmentation_trainer = SemanticSegmentationTrainer(
        original_cs_root=resolved_real_root,
        real_root=resolved_real_root,
        dataset_type=dataset_name,
        logger_save_dir=logger_save_dir,
        train_steps=train_steps,
        seed=seed,
        real_split=real_split,
        early_stopping_patience=early_stopping_patience,  
        encoder_name=encoder_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    segmentation_trainer.train()


if __name__ == "__main__":
    args = parse_args()
    step_1_train_real(**vars(args))
