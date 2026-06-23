#vfm4ss/segmentation_trainer.py
import torch
import logging
import os
import csv
import json
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from dataset_configs import get_dataset_spec
from vfm4ss.datasets.cityscapes import CityscapesCombined, Cityscapes, CityscapesSyn
from vfm4ss.datasets.folder_semantic import (
    ADE20K,
    ADE20KCombined,
    ADE20KSyn,
    BDD100K,
    BDD100KCombined,
    BDD100KSyn,
    COCOStuff10k,
    COCOStuff10kCombined,
    COCOStuff10kSyn,
    PascalVOC,
    PascalVOCCombined,
    PascalVOCSyn,
    UAVID,
    UAVIDCombined,
    UAVIDSyn,
)
from vfm4ss.datasets.no_zip_dataset import FIXED_SUBSET_SEED
from vfm4ss.models.linear_decoder import LinearDecoder
from vfm4ss.models.encoder_config import (
    DEFAULT_ENCODER_NAME,
    resolve_checkpoint_encoder_candidates,
    resolve_linear_decoder_patch_size,
)
from vfm4ss.training.linear_semantic import LinearSemantic

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SemanticSegmentationTrainer:
    """
    Pytorch Lightning wrapper for running the segmentation training.
    """

    def __init__(
        self,
        seed: int = 42,
        root: str = "",
        original_cs_root: str = "",
        real_root: str | None = None,
        batch_size=4,
        num_workers=16,
        train_steps: int = 20000,
        learning_rate: float | None = None,
        dataset_type="cityscapes",
        logger_save_dir=None,
        ckpt_path: str = None,
        early_stopping_patience: int = None,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        # Split arguments
        real_split: float = 1.0,
        syn_split: float = 1.0,
        init_model: bool = True,
    ):
        # 1. Set Global Seed
        seed_everything(seed, workers=True) 

        # Cache params
        self.train_steps = train_steps
        self.input_ckpt_path = ckpt_path

        # Select the dataset
        dataset_map = {
            "cityscapes": Cityscapes,
            "cityscapes_combined": CityscapesCombined,
            "cityscapes_syn_only": CityscapesSyn,
            "uavid": UAVID,
            "uavid_combined": UAVIDCombined,
            "uavid_syn_only": UAVIDSyn,
            "pascal_voc": PascalVOC,
            "pascal_voc_combined": PascalVOCCombined,
            "pascal_voc_syn_only": PascalVOCSyn,
            "cocostuff10k": COCOStuff10k,
            "cocostuff10k_combined": COCOStuff10kCombined,
            "cocostuff10k_syn_only": COCOStuff10kSyn,
            "bdd100k": BDD100K,
            "bdd100k_combined": BDD100KCombined,
            "bdd100k_syn_only": BDD100KSyn,
            "ade20k": ADE20K,
            "ade20k_combined": ADE20KCombined,
            "ade20k_syn_only": ADE20KSyn,
        }
        dataset_type_key = dataset_type.lower()
        try:
            DatasetClass = dataset_map[dataset_type_key]
        except KeyError:
            raise KeyError(f"dataset_type must be in {dataset_map.keys()} but got {dataset_type}")

        resolved_real_root = real_root if real_root is not None else original_cs_root
        base_dataset_name = dataset_type_key
        for suffix in ("_combined", "_syn_only"):
            if base_dataset_name.endswith(suffix):
                base_dataset_name = base_dataset_name[: -len(suffix)]
                break
        dataset_spec = get_dataset_spec(base_dataset_name)
        fallback_img_size = (
            dataset_spec.default_image_size,
            dataset_spec.default_image_size,
        )
        encoder_patch_size = resolve_linear_decoder_patch_size(encoder_name)

        # Setup data module
        dataset_kwargs = dict(
            root=root,
            devices=[0],
            num_workers=num_workers,
            batch_size=batch_size,
            img_size=fallback_img_size,
            real_split=real_split,
            syn_split=syn_split,
        )
        if dataset_type_key.startswith("cityscapes"):
            dataset_kwargs["original_cs_root"] = resolved_real_root
        else:
            dataset_kwargs["real_root"] = resolved_real_root
        self.data_module = DatasetClass(**dataset_kwargs)
        self.data_module.setup(stage=None)

        # Setup loggers
        logger_save_dir = f"training_logs/{dataset_type}/temp" if not logger_save_dir else logger_save_dir
        self.csv_logger = CSVLogger(save_dir=logger_save_dir, name="csv_logs")
        trainer_loggers = [self.csv_logger]
        wandb_mode = os.environ.get("WANDB_MODE", "disabled").strip().lower()
        self.wandb_logger = None
        if wandb_mode not in {"disabled", "off", "false", "0"}:
            self.wandb_logger = WandbLogger(save_dir=logger_save_dir)
            trainer_loggers.insert(0, self.wandb_logger)

        is_finetuning = ckpt_path is not None
        default_lr = 2e-5 if is_finetuning else 1e-4
        finetune_lr = learning_rate if learning_rate is not None else default_lr

        self.model = None
        if init_model:
            network = LinearDecoder(
                encoder_name=encoder_name,
                num_classes=self.data_module.num_classes,
                img_size=self.data_module.img_size,
                patch_size=encoder_patch_size,
            )
            self.model = LinearSemantic(
                network=network,
                num_classes=self.data_module.num_classes,
                num_metrics=self.data_module.num_metrics,
                ignore_idx=self.data_module.ignore_idx,
                img_size=self.data_module.img_size,
                lr=finetune_lr,
            )

        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            last_error = None
            for candidate_encoder_name in resolve_checkpoint_encoder_candidates(encoder_name):
                candidate_patch_size = resolve_linear_decoder_patch_size(candidate_encoder_name)
                network = LinearDecoder(
                    encoder_name=candidate_encoder_name,
                    num_classes=self.data_module.num_classes,
                    img_size=self.data_module.img_size,
                    patch_size=candidate_patch_size,
                )
                candidate_model = LinearSemantic(
                    network=network,
                    num_classes=self.data_module.num_classes,
                    num_metrics=self.data_module.num_metrics,
                    ignore_idx=self.data_module.ignore_idx,
                    img_size=self.data_module.img_size,
                    lr=finetune_lr,
                )
                try:
                    candidate_model.load_state_dict(state_dict=state_dict)
                    self.model = candidate_model
                    if candidate_encoder_name != encoder_name:
                        encoder_name = candidate_encoder_name
                        logger.info(
                            f"Resolved checkpoint encoder mismatch: using '{encoder_name}' for '{ckpt_path}'."
                        )
                    break
                except RuntimeError as exc:
                    last_error = exc
            else:
                raise RuntimeError(
                    f"Failed to load checkpoint '{ckpt_path}' into encoder '{encoder_name}'. "
                    "Make sure step 4 uses the same --encoder_name that was used to create the checkpoint in step 1."
                ) from last_error
            logger.info(f"Loaded model weights from checkpoint: {ckpt_path}")
            logger.info(f"Fine-tuning with lr={finetune_lr} (encoder lr={finetune_lr * 0.1})")

        # --- Early Stopping Setup ---
        trainer_callbacks = [ModelSummary(max_depth=2)]
        trainer_callbacks.append(TQDMProgressBar(refresh_rate=10))
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_0_miou",
            mode="max",
            save_top_k=1,
            filename="best-miou-{epoch:03d}-{step:06d}-{val_0_miou:.4f}",
            auto_insert_metric_name=False,
        )
        trainer_callbacks.append(self.checkpoint_callback)
        if early_stopping_patience is not None and early_stopping_patience > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_0_miou",
                min_delta=0.00,
                patience=early_stopping_patience,
                verbose=False,
                mode="max",
                log_rank_zero_only=True,
            )
            trainer_callbacks.append(early_stop_callback)

        self.trainer = Trainer(
            max_steps=self.train_steps,
            val_check_interval=1000,
            accumulate_grad_batches=4,
            precision="16-mixed",
            log_every_n_steps=1,
            enable_model_summary=False,
            callbacks=trainer_callbacks,
            accelerator="gpu", 
            devices=self.data_module.devices,
            logger=trainer_loggers,
            check_val_every_n_epoch=None,
        )

        # --- DATASET STATISTICS ---
        len_real = len(self.data_module.real_train_dataset) if getattr(self.data_module, "real_train_dataset", None) is not None else 0
        len_syn = len(self.data_module.synthetic_train_dataset) if getattr(self.data_module, "synthetic_train_dataset", None) is not None else 0
        
        if dataset_type_key.endswith("_syn_only"):
            len_real = 0
            total_samples = len_syn
        elif dataset_type_key.endswith("_combined"):
            total_samples = len_real + len_syn
        else:
            len_syn = 0
            total_samples = len_real

        logger.info("-" * 40)
        logger.info(f"Dataset Statistics ({dataset_type}):")
        logger.info(f"  Encoder:     {encoder_name}")
        logger.info(f"  Img Size:    {self.data_module.img_size}")
        logger.info(f"  Master Seed: {seed}")
        logger.info(f"  Subset Seed: {FIXED_SUBSET_SEED}")
        logger.info(f"  Real Split:  {real_split} (Count: {len_real})")
        logger.info(f"  Syn Split:   {syn_split} (Count: {len_syn})")
        logger.info(f"  Total:       {total_samples}")

        # Logging to CSV/JSON
        try:
            version_dir = self.csv_logger.log_dir
            os.makedirs(version_dir, exist_ok=True)
            
            stats_path = os.path.join(version_dir, "dataset_stats.csv")
            with open(stats_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "dataset_type",
                    "seed",
                    "subset_seed",
                    "real_split",
                    "syn_split",
                    "real_count",
                    "syn_count",
                    "total_count",
                ])
                writer.writerow([
                    dataset_type_key,
                    seed,
                    FIXED_SUBSET_SEED,
                    real_split,
                    syn_split,
                    len_real,
                    len_syn,
                    total_samples,
                ])
            
            # File List logging (same as before)
            files_log = {"real": [], "synthetic": []}

            def extract_filenames(dataset):
                common_names = ['images', 'files', 'img_paths', 'paths', 'items', 'samples', 'data', 'imgs']
                for name in common_names:
                    if hasattr(dataset, name):
                        val = getattr(dataset, name)
                        if isinstance(val, list) and len(val) > 0:
                            if isinstance(val[0], tuple):
                                return [str(x[0]) for x in val]
                            return [str(x) for x in val]
                return []

            if getattr(self.data_module, "real_train_dataset", None) is not None:
                files_log["real"] = extract_filenames(self.data_module.real_train_dataset)

            if getattr(self.data_module, "synthetic_train_dataset", None) is not None:
                files_log["synthetic"] = extract_filenames(self.data_module.synthetic_train_dataset)

            json_path = os.path.join(version_dir, "training_files.json")
            with open(json_path, "w") as f:
                json.dump(files_log, f, indent=4)
            
        except Exception as e:
            logger.warning(f"Failed to write dataset statistics: {e}")
        logger.info("-" * 40)

    @staticmethod
    def _metrics_to_float(metrics: dict) -> dict[str, float]:
        converted = {}
        for key, value in metrics.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            converted[key] = float(value)
        return converted

    def train(self) -> dict:
        logger.info("Running initial validation before training starts.")
        initial_validation = self.trainer.validate(self.model, datamodule=self.data_module)
        initial_metrics = self._metrics_to_float(initial_validation[0]) if initial_validation else {}

        self.trainer.fit(self.model, datamodule=self.data_module)

        final_validation = self.trainer.validate(self.model, datamodule=self.data_module)
        final_metrics = self._metrics_to_float(final_validation[0]) if final_validation else {}

        best_metrics = dict(initial_metrics)
        best_checkpoint_path = self.input_ckpt_path or ""
        best_checkpoint_source = "initial"
        best_miou = best_metrics.get("val_0_miou", float("-inf"))

        candidate_checkpoint_path = self.checkpoint_callback.best_model_path
        if not candidate_checkpoint_path:
            fallback_checkpoint = (
                Path(self.csv_logger.log_dir) / "checkpoints" / "final.ckpt"
            )
            fallback_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self.trainer.save_checkpoint(str(fallback_checkpoint))
            candidate_checkpoint_path = str(fallback_checkpoint)
        if candidate_checkpoint_path:
            checkpoint_validation = self.trainer.validate(
                self.model,
                datamodule=self.data_module,
                ckpt_path=candidate_checkpoint_path,
            )
            checkpoint_metrics = (
                self._metrics_to_float(checkpoint_validation[0])
                if checkpoint_validation
                else {}
            )
            checkpoint_miou = checkpoint_metrics.get("val_0_miou", float("-inf"))
            if checkpoint_miou > best_miou:
                best_metrics = checkpoint_metrics
                best_miou = checkpoint_miou
                best_checkpoint_path = candidate_checkpoint_path
                best_checkpoint_source = "model_checkpoint"

        reported_metrics = dict(best_metrics)
        reported_metrics["best_val_0_miou"] = best_miou
        reported_metrics["best_checkpoint_score"] = best_miou
        reported_metrics["initial_val_0_miou"] = initial_metrics.get("val_0_miou", float("nan"))
        reported_metrics["final_val_0_miou"] = final_metrics.get("val_0_miou", float("nan"))
        reported_metrics["initial_val_loss"] = initial_metrics.get("val_loss", float("nan"))
        reported_metrics["final_val_loss"] = final_metrics.get("val_loss", float("nan"))
        reported_metrics["best_checkpoint_path"] = best_checkpoint_path
        reported_metrics["best_checkpoint_source"] = best_checkpoint_source

        logger.info(
            "Best validation checkpoint: "
            f"source={best_checkpoint_source}, "
            f"val_0_miou={best_miou:.6f}, "
            f"path={best_checkpoint_path or '<none>'}"
        )
        logger.info(
            "Validation mIoU summary: "
            f"initial={reported_metrics['initial_val_0_miou']:.6f}, "
            f"best={reported_metrics['best_val_0_miou']:.6f}, "
            f"final={reported_metrics['final_val_0_miou']:.6f}"
        )

        return reported_metrics

    def validate(self):
        self.trainer.validate(self.model, datamodule=self.data_module)


if __name__ == "__main__":
    trainer = SemanticSegmentationTrainer(
        original_cs_root="data/real_datasets/cityscapes",
        ckpt_path=None,
        early_stopping_patience=100
    )
    trainer.validate()
