#vfm4ss/training/linear_semantic.py
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import PolynomialLR

from vfm4ss.training.lightning_module import LightningModule


class LinearSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        num_metrics: int,
        num_classes: int,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
        )

        self.save_hyperparameters(ignore=["network"])

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)

        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        logits = self(imgs)
        logits = F.interpolate(logits, self.img_size, mode="bilinear")

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets = torch.stack(targets).long()

        loss_total = self.criterion(logits, targets)
        self.log("train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total
        
    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
        save_img=False,
    ):
        imgs, targets = batch

        crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        crop_logits = self(crops)
        crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")
        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes) # <--- This returns a LIST

        if is_notebook:
            return logits

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        # Validation images can have different spatial sizes (for example COCOStuff).
        # Compute the loss per image and average instead of stacking mismatched tensors.
        loss_val = torch.stack(
            [
                self.criterion(logit[None, ...], target[None, ...].long())
                for logit, target in zip(logits, targets)
            ]
        ).mean()
        self.log(f"{log_prefix}_loss", loss_val, sync_dist=True)

        self.update_metrics(logits, targets, dataloader_idx)

        if batch_idx == 0:
            name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
            plot = self.plot_semantic(
                imgs[0],
                targets[0],
                logits=logits[0],
            )
            # Only log images to W&B. CSVLogger also exposes an `experiment`
            # property, but its ExperimentWriter does not implement `.log()`.
            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log({name: [wandb.Image(plot)]})

    def test_step(self,
            batch,
            batch_idx=None,
            dataloader_idx=None,
            log_prefix=None,
            is_notebook=False
        ):
        self.validation_step(
            batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            log_prefix=log_prefix,
            is_notebook=is_notebook
        )

    def on_validation_epoch_end(self):
        # This function (in LightningModule) handles logging mIOU and Class-IoU
        self._on_eval_epoch_end_semantic("val")

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
