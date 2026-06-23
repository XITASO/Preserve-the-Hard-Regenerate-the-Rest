from typing import Tuple
import torch
import torch.nn.functional as F
from vfm4ss.models.linear_decoder import LinearDecoder
from vfm4ss.models.encoder_config import (
    DEFAULT_ENCODER_NAME,
    resolve_checkpoint_encoder_candidates,
    resolve_linear_decoder_patch_size,
)
from guided_generation.guidance.diffusion_guide import DiffusionGuide, shannon_entropy_loss


class DinoGuide(DiffusionGuide):

    def __init__(
        self,
        ckpt_path: str,
        img_size: Tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        patch_size: int | None = None,
    ):
        super().__init__()
        self.segmenter = self._init_segmenter(
            ckpt_path=ckpt_path,
            img_size=img_size,
            num_classes=num_classes,
            encoder_name=encoder_name,
            patch_size=patch_size,
        )

    def _init_segmenter(
        self,
        ckpt_path: str,
        img_size: Tuple[int, int],
        num_classes: int,
        encoder_name: str,
        patch_size: int | None,
    ):
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = {k.replace("network.", ""): v for k, v in state_dict["state_dict"].items()}

        last_error = None
        for candidate_encoder_name in resolve_checkpoint_encoder_candidates(encoder_name):
            resolved_patch_size = (
                patch_size
                if patch_size is not None
                else resolve_linear_decoder_patch_size(candidate_encoder_name)
            )
            model = LinearDecoder(
                encoder_name=candidate_encoder_name,
                num_classes=num_classes,
                img_size=img_size,
                patch_size=resolved_patch_size,
            )

            try:
                model.load_state_dict(state_dict=state_dict)
                if candidate_encoder_name != encoder_name:
                    print(
                        f"Resolved guidance checkpoint encoder mismatch: requested '{encoder_name}' "
                        f"but loaded checkpoint '{ckpt_path}' with '{candidate_encoder_name}'."
                    )
                model.encoder.set_grad_checkpointing(enable=True)
                return DinoModel(model, image_size=img_size)
            except RuntimeError as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to load guidance checkpoint '{ckpt_path}' into encoder '{encoder_name}' "
            f"or fallback encoders {resolve_checkpoint_encoder_candidates(encoder_name)}."
        ) from last_error

    def loss_function(self, logits: torch.Tensor, mask: torch.Tensor):
        return shannon_entropy_loss(logits, mask)

    def forward(self, image, mask, return_logits=False):
        target_size = self.segmenter.image_size
        if image.shape[-2:] != target_size:
            image = F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)

        if mask.dim() == 4:
            if mask.shape[1] == 1:
                mask = mask[:, 0]
            elif mask.shape[0] == 1:
                mask = mask[0]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"Expected mask with 2, 3, or 4 dimensions, got shape {tuple(mask.shape)}.")
        if mask.shape[-2:] != target_size:
            mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1).bool()

        logits = self.segmenter(image)
        if return_logits:
            return self.loss_function(logits, mask), logits
        return self.loss_function(logits, mask)
    

class DinoModel(torch.nn.Module):
    """Additional layer to resize the model output directly to the input size
    for more generality of the DiffusionGuide class."""
    def __init__(self, dino_model, image_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dino_model = dino_model
        self.image_size = image_size

    def forward(self, x):
        x = self.dino_model(x)
        return F.interpolate(x,self.image_size, mode="bilinear")
