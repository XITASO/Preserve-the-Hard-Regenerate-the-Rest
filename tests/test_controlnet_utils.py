import numpy as np
import pytest
import torch
from PIL import Image

from guided_generation.diffusion.controlnet_utils import (
    ResolvedControlNet,
    build_control_image,
    resolve_controlnets,
)


def test_resolve_controlnets_legacy_single_seg():
    resolved = resolve_controlnets(
        legacy_controlnet_path="/tmp/seg-controlnet",
        legacy_controlnet_conditioning_scale=0.8,
    )

    assert resolved == [ResolvedControlNet(method="seg", path="/tmp/seg-controlnet", conditioning_scale=0.8)]


def test_resolve_controlnets_supports_defaults_and_aliases():
    resolved = resolve_controlnets(
        control_methods=["seg", "depth", "hed"],
        controlnet_paths=["/tmp/seg-controlnet", "default", "default"],
        controlnet_conditioning_scales=[1.0, 0.7, 0.5],
    )

    assert [item.method for item in resolved] == ["seg", "depth", "edge"]
    assert resolved[1].path == "diffusers/controlnet-depth-sdxl-1.0"
    assert resolved[2].path == "alimama-creative/EcomXL_controlnet_softedge"


def test_resolve_controlnets_rejects_seg_default():
    with pytest.raises(ValueError, match="no built-in default"):
        resolve_controlnets(
            control_methods=["seg"],
            controlnet_paths=["default"],
        )


def test_resolve_controlnets_requires_methods_for_new_lists():
    with pytest.raises(ValueError, match="control_methods"):
        resolve_controlnets(
            controlnet_paths=["default"],
        )


def test_build_control_image_seg_uses_annotation_palette():
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    annotation = torch.tensor([[0, 1], [1, 0]])
    palette = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60],
        ],
        dtype=torch.uint8,
    )

    control_image = build_control_image("seg", image, annotation, palette)

    assert np.array(control_image).tolist() == [
        [[10, 20, 30], [40, 50, 60]],
        [[40, 50, 60], [10, 20, 30]],
    ]


def test_build_control_image_edge_uses_preprocessor(monkeypatch):
    image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    annotation = torch.zeros((4, 4), dtype=torch.long)
    palette = torch.zeros((256, 3), dtype=torch.uint8)
    expected = np.full((4, 4, 3), 123, dtype=np.uint8)

    monkeypatch.setattr(
        "guided_generation.diffusion.controlnet_utils.get_edge_preprocessor",
        lambda: (lambda _: expected),
    )

    control_image = build_control_image("edge", image, annotation, palette)

    assert np.array(control_image).shape == (4, 4, 3)
    assert int(np.array(control_image)[0, 0, 0]) == 123
