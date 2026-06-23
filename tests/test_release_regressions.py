import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from vfm4ss.datasets.folder_semantic import FolderSemanticSegmentation
from vfm4ss.training.linear_semantic import LinearSemantic


class _TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=1)
        self.head = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


def _write_pair(root: Path, split: str, image_suffix: str) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations" / split).mkdir(parents=True, exist_ok=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, 8:] = 127
    label = np.zeros((16, 16), dtype=np.uint8)
    label[:, 8:] = 1
    Image.fromarray(image).save(root / "images" / split / f"sample{image_suffix}")
    Image.fromarray(label).save(root / "annotations" / split / "sample.png")


def test_network_is_not_serialized_as_a_hyperparameter() -> None:
    model = LinearSemantic(
        network=_TinyNetwork(),
        num_metrics=1,
        num_classes=2,
        ignore_idx=255,
        img_size=(16, 16),
    )
    assert "network" not in model.hparams


def test_synthetic_png_suffix_from_metadata_is_loaded(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    _write_pair(real_root, "train", ".jpg")
    _write_pair(real_root, "val", ".jpg")
    _write_pair(synthetic_root, "train", ".png")
    (synthetic_root / "pipeline_metadata.json").write_text(
        json.dumps({"synthetic_img_suffix": ".png"}),
        encoding="utf-8",
    )

    data_module = FolderSemanticSegmentation(
        dataset_name="pascal_voc",
        root=str(synthetic_root),
        real_root=str(real_root),
        train_source="combined",
        devices="auto",
        num_workers=0,
        batch_size=1,
        img_size=(16, 16),
        scale_range=(1.0, 1.0),
    ).setup()

    assert len(data_module.real_train_dataset) == 1
    assert len(data_module.synthetic_train_dataset) == 1
    assert len(data_module.merged_train_dataset) == 2
