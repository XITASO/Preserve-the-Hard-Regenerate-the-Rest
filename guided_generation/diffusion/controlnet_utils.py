from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_SENTINELS = {"default", "builtin", "built-in", "hf", "huggingface"}
CONTROLNET_ALIASES = {
    "hed": "edge",
}
DEFAULT_CONTROLNET_PATHS = {
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "edge": "alimama-creative/EcomXL_controlnet_softedge",
}
VALID_CONTROLNET_METHODS = {"seg", "depth", "edge"}


@dataclass(frozen=True)
class ResolvedControlNet:
    method: str
    path: str
    conditioning_scale: float


class DepthAnythingProcessor:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(
            self.device
        )

    def __call__(self, image: np.ndarray) -> Image.Image:
        image_pil = Image.fromarray(image)
        inputs = self.image_processor(images=image_pil, return_tensors="pt").to(self.device)
        autocast_context = torch.autocast(self.device.type) if self.device.type == "cuda" else nullcontext()
        with torch.no_grad(), autocast_context:
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.maximum(np.max(output), 1e-8)).astype("uint8")
        return Image.fromarray(formatted)


def normalize_controlnet_method(method: str) -> str:
    normalized = CONTROLNET_ALIASES.get(method.lower(), method.lower())
    if normalized not in VALID_CONTROLNET_METHODS:
        raise ValueError(
            f"Unknown control method '{method}'. Available methods: {', '.join(sorted(VALID_CONTROLNET_METHODS))}."
        )
    return normalized


def resolve_controlnets(
    control_methods: Optional[Sequence[str]] = None,
    controlnet_paths: Optional[Sequence[str]] = None,
    controlnet_conditioning_scales: Optional[Sequence[float]] = None,
    legacy_controlnet_path: Optional[str] = None,
    legacy_controlnet_conditioning_scale: float = 1.0,
) -> list[ResolvedControlNet]:
    has_new_controlnet_args = any(
        value is not None for value in (control_methods, controlnet_paths, controlnet_conditioning_scales)
    )
    if has_new_controlnet_args and legacy_controlnet_path is not None:
        raise ValueError(
            "Use either the legacy single-control arguments (`--controlnet_path`) or the new multi-control arguments "
            "(`--control_methods`, `--controlnet_paths`, `--controlnet_conditioning_scales`), not both."
        )

    if control_methods is None:
        if controlnet_paths is not None or controlnet_conditioning_scales is not None:
            raise ValueError(
                "`control_methods` must be provided when using `controlnet_paths` or "
                "`controlnet_conditioning_scales`."
            )
        if legacy_controlnet_path is None:
            return []
        return [
            ResolvedControlNet(
                method="seg",
                path=legacy_controlnet_path,
                conditioning_scale=float(legacy_controlnet_conditioning_scale),
            )
        ]

    normalized_methods = [normalize_controlnet_method(method) for method in control_methods]
    if len(normalized_methods) == 0:
        return []

    if len(set(normalized_methods)) != len(normalized_methods):
        raise ValueError(
            f"Duplicate control methods are not supported. Got: {', '.join(normalized_methods)}."
        )

    if controlnet_paths is not None and len(controlnet_paths) != len(normalized_methods):
        raise ValueError(
            "`controlnet_paths` must have the same length as `control_methods`."
        )

    if controlnet_conditioning_scales is not None and len(controlnet_conditioning_scales) != len(normalized_methods):
        raise ValueError(
            "`controlnet_conditioning_scales` must have the same length as `control_methods`."
        )

    resolved_scales = (
        [float(scale) for scale in controlnet_conditioning_scales]
        if controlnet_conditioning_scales is not None
        else [1.0] * len(normalized_methods)
    )

    resolved_controlnets: list[ResolvedControlNet] = []
    for idx, method in enumerate(normalized_methods):
        scale = resolved_scales[idx]
        if scale <= 0:
            continue

        raw_path = controlnet_paths[idx] if controlnet_paths is not None else None
        if raw_path is None:
            if method == "seg":
                raise ValueError(
                    "Method `seg` requires an explicit checkpoint path. "
                    "Provide it in `controlnet_paths` or use the legacy `--controlnet_path` argument."
                )
            resolved_path = DEFAULT_CONTROLNET_PATHS[method]
        else:
            normalized_path = raw_path.strip()
            if normalized_path.lower() in DEFAULT_SENTINELS:
                if method == "seg":
                    raise ValueError("Method `seg` has no built-in default checkpoint.")
                resolved_path = DEFAULT_CONTROLNET_PATHS[method]
            else:
                resolved_path = normalized_path

        resolved_controlnets.append(
            ResolvedControlNet(
                method=method,
                path=resolved_path,
                conditioning_scale=scale,
            )
        )

    return resolved_controlnets


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    image_np = np.clip(np.rint(image_np * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(image_np, "RGB")


@lru_cache(maxsize=1)
def get_depth_preprocessor() -> DepthAnythingProcessor:
    return DepthAnythingProcessor()


def get_edge_preprocessor():
    def _canny(image_np: np.ndarray) -> Image.Image:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if image_np.ndim == 3 else image_np
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    return _canny


def annotation_to_control_image(annotation: torch.Tensor, palette: torch.Tensor) -> Image.Image:
    palette = palette.to(annotation.device)
    rgb = palette[annotation.long()]
    rgb_np = rgb.cpu().numpy().astype(np.uint8)
    return Image.fromarray(rgb_np, "RGB")


def build_control_image(method: str, image: Image.Image, annotation: torch.Tensor, palette: torch.Tensor) -> Image.Image:
    normalized_method = normalize_controlnet_method(method)
    if normalized_method == "seg":
        return annotation_to_control_image(annotation, palette)

    image_np = np.array(image)
    if normalized_method == "depth":
        control_image = get_depth_preprocessor()(image_np)
    elif normalized_method == "edge":
        control_image = get_edge_preprocessor()(image_np)
    else:
        raise ValueError(f"Unsupported control method '{method}'.")

    if isinstance(control_image, np.ndarray):
        control_image = Image.fromarray(control_image)

    return control_image.resize(image.size, resample=Image.LANCZOS).convert("RGB")
