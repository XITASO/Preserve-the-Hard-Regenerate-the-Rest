from pathlib import Path

import numpy as np
from PIL import Image

from dataset_configs import ADE20K_RAW_CLASS_IDS, COCOSTUFF10K_RAW_CLASS_IDS, get_background_train_ids, get_dataset_spec
from guided_generation.datasets import build_guided_dataset


def test_pascal_voc_spec_matches_standardized_labels():
    spec = get_dataset_spec("pascal_voc")

    assert spec.num_classes == 21
    assert spec.ignore_idx == 255
    assert spec.raw_class_ids is None


def test_cocostuff10k_spec_uses_sparse_raw_ids():
    spec = get_dataset_spec("cocostuff10k")
    mapping = spec.base_dataset_id_remapping()

    assert spec.num_classes == 171
    assert len(COCOSTUFF10K_RAW_CLASS_IDS) == 171
    assert mapping[0] == 255
    assert mapping[1] == 0
    assert mapping[182] == 170
    assert mapping[12] == 255


def test_ade20k_spec_remaps_one_indexed_labels():
    spec = get_dataset_spec("ade20k")
    mapping = spec.base_dataset_id_remapping()

    assert spec.num_classes == 150
    assert len(ADE20K_RAW_CLASS_IDS) == 150
    assert mapping[0] == 255
    assert mapping[1] == 0
    assert mapping[150] == 149
    assert mapping[255] == 255


def test_background_train_ids_define_object_foreground():
    assert get_background_train_ids("pascal_voc") == {0, 255}
    assert 0 in get_background_train_ids("cityscapes")
    assert 11 not in get_background_train_ids("cityscapes")
    assert get_background_train_ids("uavid") == {0, 1, 2, 4, 5}
    assert min(get_background_train_ids("cocostuff10k")) == 80
    assert 79 not in get_background_train_ids("cocostuff10k")
    assert get_background_train_ids("ade20k") == {255}


def test_uavid_spec_matches_saved_layout():
    spec = get_dataset_spec("uavid")

    assert spec.num_classes == 8
    assert spec.ignore_idx == 0
    assert spec.default_image_size == 1024
    assert spec.img_dir == "img_dir"
    assert spec.ann_dir == "ann_dir"
    assert spec.img_suffix == "_img.png"
    assert spec.ann_suffix == "_ann.png"
    assert spec.img_stem_suffix == "_img"
    assert spec.ann_stem_suffix == "_ann"


def test_build_guided_dataset_supports_standardized_roots(tmp_path: Path):
    dataset = build_guided_dataset("pascal_voc", root_dir=str(tmp_path))

    assert dataset.dataset_name == "pascal_voc"
    assert dataset.ignore_idx == 255
    assert len(dataset) == 0


def test_pascal_voc_loader_preserves_paletted_label_indices(tmp_path: Path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "annotations" / "train").mkdir(parents=True)

    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(tmp_path / "images" / "train" / "sample.jpg")

    ann = Image.fromarray(np.array([[0, 1], [15, 255]], dtype=np.uint8), mode="P")
    palette = [0] * (256 * 3)
    palette[1 * 3 : 1 * 3 + 3] = [128, 0, 0]
    palette[15 * 3 : 15 * 3 + 3] = [192, 128, 128]
    palette[255 * 3 : 255 * 3 + 3] = [224, 224, 192]
    ann.putpalette(palette)
    ann.save(tmp_path / "annotations" / "train" / "sample.png")

    dataset = build_guided_dataset("pascal_voc", root_dir=str(tmp_path))

    loaded_ann = dataset[0]["ann"]
    assert set(loaded_ann.unique().tolist()) == {0, 1, 15, 255}


def test_build_guided_dataset_supports_uavid_root(tmp_path: Path):
    (tmp_path / "img_dir" / "train").mkdir(parents=True)
    (tmp_path / "ann_dir" / "train").mkdir(parents=True)

    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(tmp_path / "img_dir" / "train" / "seq1_000000_img.png")
    Image.fromarray(np.array([[0, 1], [3, 7]], dtype=np.uint8)).save(
        tmp_path / "ann_dir" / "train" / "seq1_000000_ann.png"
    )

    dataset = build_guided_dataset("uavid", root_dir=str(tmp_path))

    assert dataset.dataset_name == "uavid"
    assert dataset.ignore_idx == 0
    assert len(dataset) == 1
    assert set(dataset[0]["ann"].unique().tolist()) == {0, 1, 3, 7}
