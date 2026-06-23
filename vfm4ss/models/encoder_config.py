from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import timm
import torch
from timm.data import resolve_model_data_config


DEFAULT_ENCODER_NAME = "vit_small_patch14_dinov2"
DEFAULT_LEGACY_PATCH_SIZE = 16
COMMON_CHECKPOINT_ENCODER_FALLBACKS = (
    "vit_small_patch16_224.augreg_in1k",
    DEFAULT_ENCODER_NAME,
)


@dataclass(frozen=True)
class EncoderConfig:
    encoder_name: str
    input_size: tuple[int, int]
    patch_size: int
    is_vit_like: bool


def _normalize_patch_size(patch_size) -> int:
    if isinstance(patch_size, tuple):
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError(f"Expected square patch size, got {patch_size}.")
        patch_size = patch_size[0]
    return int(patch_size)


def _infer_cnn_stride(model, input_size: tuple[int, int]) -> int:
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size[0], input_size[1])
        feats = model.forward_features(dummy)
    if feats.dim() != 4:
        raise ValueError(
            f"Expected 4D feature map from CNN backbone, got shape {tuple(feats.shape)}."
        )
    h_stride = input_size[0] // feats.shape[-2]
    w_stride = input_size[1] // feats.shape[-1]
    if h_stride != w_stride:
        raise ValueError(
            f"Non-square stride ({h_stride}, {w_stride}) is not supported."
        )
    return int(h_stride)


@lru_cache(maxsize=None)
def get_timm_encoder_config(encoder_name: str) -> EncoderConfig:
    model = timm.create_model(encoder_name, pretrained=False, num_classes=0)
    data_config = resolve_model_data_config(model)
    input_size = data_config.get("input_size")
    if input_size is None or len(input_size) != 3:
        raise ValueError(
            f"Could not resolve a 3D input_size for encoder '{encoder_name}'. Got: {input_size}"
        )
    resolved_input_size = (int(input_size[-2]), int(input_size[-1]))

    has_patch_embed = hasattr(model, "patch_embed") and hasattr(
        getattr(model, "patch_embed", None), "patch_size"
    )

    if has_patch_embed:
        patch_size = _normalize_patch_size(model.patch_embed.patch_size)
        return EncoderConfig(
            encoder_name=encoder_name,
            input_size=resolved_input_size,
            patch_size=patch_size,
            is_vit_like=True,
        )

    stride = _infer_cnn_stride(model, resolved_input_size)
    return EncoderConfig(
        encoder_name=encoder_name,
        input_size=resolved_input_size,
        patch_size=stride,
        is_vit_like=False,
    )


def is_vit_like_encoder(encoder_name: str) -> bool:
    if encoder_name == DEFAULT_ENCODER_NAME:
        return True
    return get_timm_encoder_config(encoder_name).is_vit_like


def resolve_encoder_input_size(
    encoder_name: str,
    fallback_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if encoder_name == DEFAULT_ENCODER_NAME and fallback_size is not None:
        return fallback_size
    return get_timm_encoder_config(encoder_name).input_size


def resolve_linear_decoder_patch_size(encoder_name: str) -> int:
    if encoder_name == DEFAULT_ENCODER_NAME:
        return DEFAULT_LEGACY_PATCH_SIZE
    return get_timm_encoder_config(encoder_name).patch_size


def resolve_checkpoint_encoder_candidates(requested_encoder_name: str) -> list[str]:
    candidates = [requested_encoder_name, *COMMON_CHECKPOINT_ENCODER_FALLBACKS]
    ordered_unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in ordered_unique_candidates:
            ordered_unique_candidates.append(candidate)
    return ordered_unique_candidates
