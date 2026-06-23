from pathlib import Path

import numpy as np
from PIL import Image

from dataset_configs import get_dataset_spec
from guided_generation.main_scripts.prepare_uavid_semseg import (
    color_to_train_id,
    prepare_split,
)


def test_color_to_train_id() -> None:
    spec = get_dataset_spec("uavid")
    label = np.asarray(
        [[spec.palette[0], spec.palette[3]], [spec.palette[6], spec.palette[7]]],
        dtype=np.uint8,
    )
    assert color_to_train_id(label).tolist() == [[0, 3], [6, 7]]


def test_prepare_split(tmp_path: Path) -> None:
    input_root = tmp_path / "official"
    sequence = input_root / "uavid_train" / "seq1"
    (sequence / "Images").mkdir(parents=True)
    (sequence / "Labels").mkdir()

    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(
        sequence / "Images" / "000000.png"
    )
    Image.new("RGB", (4, 3), color=(128, 64, 128)).save(
        sequence / "Labels" / "000000.png"
    )

    output_root = tmp_path / "prepared"
    assert prepare_split(input_root, output_root, "train", False) == (1, 1)
    assert (output_root / "img_dir/train/seq1_000000_img.png").is_file()
    label_path = output_root / "ann_dir/train/seq1_000000_ann.png"
    assert label_path.is_file()
    assert np.unique(np.asarray(Image.open(label_path))).tolist() == [2]
