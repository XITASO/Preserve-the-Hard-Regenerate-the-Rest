from __future__ import annotations

from dataclasses import dataclass


def build_pascal_style_palette(num_entries: int = 256) -> dict[int, list[int]]:
    """Classic bit-shift palette used by Pascal-style semantic masks."""
    palette: dict[int, list[int]] = {}
    for label in range(num_entries):
        value = label
        r = g = b = 0
        for bit in range(8):
            r |= ((value >> 0) & 1) << (7 - bit)
            g |= ((value >> 1) & 1) << (7 - bit)
            b |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette[label] = [r, g, b]
    return palette


CITYSCAPES_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    255: "clutter",
}

CITYSCAPES_PALETTE = {
    0: [128, 64, 128],
    1: [244, 35, 232],
    2: [70, 70, 70],
    3: [102, 102, 156],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180],
    11: [220, 20, 60],
    12: [255, 0, 0],
    13: [0, 0, 142],
    14: [0, 0, 70],
    15: [0, 60, 100],
    16: [0, 80, 100],
    17: [0, 0, 230],
    18: [119, 11, 32],
    255: [0, 0, 0],
}

UAVID_CLASSES = {
    0: "clutter",
    1: "building",
    2: "road",
    3: "static car",
    4: "tree",
    5: "vegetation",
    6: "human",
    7: "moving car",
}

UAVID_PALETTE = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [128, 64, 128],
    3: [192, 0, 192],
    4: [0, 128, 0],
    5: [128, 128, 0],
    6: [64, 64, 0],
    7: [64, 0, 128],
}

PASCAL_VOC_CLASSES = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "dining table",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "monitor",
    255: "ignore",
}


# Sparse raw ids observed in the standardized COCOStuff10k masks. We keep this
# mapping stable so cached subsets continue to use the same train ids.
COCOSTUFF10K_RAW_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
    159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
    173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
]

COCOSTUFF10K_CLASSES = {idx: f"class_{idx:03d}" for idx in range(len(COCOSTUFF10K_RAW_CLASS_IDS))}
COCOSTUFF10K_CLASSES[255] = "ignore"

ADE20K_RAW_CLASS_IDS = list(range(1, 151))
ADE20K_CLASSES = {idx: f"class_{idx:03d}" for idx in range(len(ADE20K_RAW_CLASS_IDS))}
ADE20K_CLASSES[255] = "ignore"


_COCOSTUFF_THING_RAW_ID_MAX = 91
BACKGROUND_TRAIN_IDS_BY_DATASET = {
    # Cityscapes 19-class train-id space.
    # Background: road, sidewalk, building, wall, fence, vegetation, terrain, sky.
    # Include 255 because several void/background-ish raw classes map there.
    "cityscapes": {0, 1, 2, 3, 4, 8, 9, 10, 255},
    # BDD100K semantic segmentation uses the same 19-class train-id space as
    # Cityscapes, with 255 for unknown/ignore pixels.
    "bdd100k": {0, 1, 2, 3, 4, 8, 9, 10, 255},
    # UAVID train-id space. We regenerate aerial background/context and preserve
    # the small object classes: static car, human, moving car.
    "uavid": {0, 1, 2, 4, 5},
    # Pascal VOC train-id space.
    "pascal_voc": {0, 255},
    # COCO-Stuff 10k train-id space: thing classes are raw ids <= 91,
    # stuff/background classes are raw ids > 91.
    "cocostuff10k": {
        train_id
        for train_id, raw_id in enumerate(COCOSTUFF10K_RAW_CLASS_IDS)
        if raw_id > _COCOSTUFF_THING_RAW_ID_MAX
    },
    # ADE20K has no paper-specific foreground/background split yet. Keep this
    # minimal so simple-mode imports remain defined without changing "ours".
    "ade20k": {255},
}


def get_background_train_ids(dataset_name: str) -> set[int]:
    normalized_name = dataset_name.lower()
    if normalized_name not in BACKGROUND_TRAIN_IDS_BY_DATASET:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available datasets: "
            f"{', '.join(sorted(BACKGROUND_TRAIN_IDS_BY_DATASET))}."
        )
    return set(BACKGROUND_TRAIN_IDS_BY_DATASET[normalized_name])


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    num_classes: int
    ignore_idx: int
    default_image_size: int
    base_prompt: str
    img_suffix: str
    ann_suffix: str
    classes: dict[int, str]
    palette: dict[int, list[int]]
    img_dir: str = "images"
    ann_dir: str = "annotations"
    img_stem_suffix: str = ""
    ann_stem_suffix: str = ""
    negative_prompt: str = ""
    raw_class_ids: list[int] | None = None

    def has_sparse_raw_ids(self) -> bool:
        return self.raw_class_ids is not None

    def raw_to_train_id_mapping(self) -> dict[int, int]:
        if self.raw_class_ids is None:
            return {raw_id: raw_id for raw_id in range(self.num_classes)}
        return {raw_id: train_id for train_id, raw_id in enumerate(self.raw_class_ids)}

    def base_dataset_id_remapping(self) -> dict[int, int] | None:
        if self.raw_class_ids is None:
            return None
        mapping = {raw_id: self.ignore_idx for raw_id in range(256)}
        mapping[0] = self.ignore_idx
        for train_id, raw_id in enumerate(self.raw_class_ids):
            mapping[raw_id] = train_id
        mapping[self.ignore_idx] = self.ignore_idx
        return mapping


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cityscapes": DatasetSpec(
        name="cityscapes",
        display_name="Cityscapes",
        num_classes=19,
        ignore_idx=255,
        default_image_size=1024,
        base_prompt=(
            "photorealistic, ultra-detailed, 4K high-resolution, sharp focus, high quality, "
            "in the style of Cityscapes dataset. "
        ),
        negative_prompt=(
            "blurry, low quality, deformed, melted structures, floating objects, "
            "cartoon, illustration, unrealistic shadows, out of perspective, wrong scale."
        ),
        img_suffix="_leftImg8bit.png",
        ann_suffix="_gtFine_labelIds.png",
        classes=CITYSCAPES_CLASSES,
        palette=CITYSCAPES_PALETTE,
    ),
    "bdd100k": DatasetSpec(
        name="bdd100k",
        display_name="BDD100K",
        num_classes=19,
        ignore_idx=255,
        default_image_size=1024,
        base_prompt=(
            "photorealistic dashcam image, diverse urban driving scene, high-resolution, "
            "realistic lighting and weather, sharp focus. "
        ),
        negative_prompt=(
            "blurry, low quality, deformed vehicles, melted structures, cartoon, "
            "illustration, unrealistic shadows, wrong scale, watermark, text."
        ),
        img_suffix=".jpg",
        ann_suffix=".png",
        classes=CITYSCAPES_CLASSES,
        palette=CITYSCAPES_PALETTE,
        img_dir="images/10k",
        ann_dir="labels/sem_seg/masks",
    ),
    "pascal_voc": DatasetSpec(
        name="pascal_voc",
        display_name="Pascal VOC 2012",
        num_classes=21,
        ignore_idx=255,
        default_image_size=512,
        base_prompt="photorealistic natural scene, high detail, realistic lighting. ",
        img_suffix=".jpg",
        ann_suffix=".png",
        classes=PASCAL_VOC_CLASSES,
        palette=build_pascal_style_palette(),
    ),
    "uavid": DatasetSpec(
        name="uavid",
        display_name="UAVID",
        num_classes=8,
        ignore_idx=0,
        default_image_size=1024,
        base_prompt=(
            "photorealistic aerial drone image, urban scene, high-resolution, "
            "realistic lighting, sharp overhead perspective. "
        ),
        negative_prompt=(
            "blurry, low quality, cartoon, illustration, distorted vehicles, "
            "warped roads, duplicated objects, unrealistic shadows, text, watermark."
        ),
        img_suffix="_img.png",
        ann_suffix="_ann.png",
        classes=UAVID_CLASSES,
        palette=UAVID_PALETTE,
        img_dir="img_dir",
        ann_dir="ann_dir",
        img_stem_suffix="_img",
        ann_stem_suffix="_ann",
    ),
    "cocostuff10k": DatasetSpec(
        name="cocostuff10k",
        display_name="COCOStuff10k",
        num_classes=len(COCOSTUFF10K_RAW_CLASS_IDS),
        ignore_idx=255,
        default_image_size=512,
        base_prompt="Generate a clean background.",
        negative_prompt="blurry, cartoon, anime, painting, drawing, illustration, low quality, deformed, artifacts, watermark, text, oversaturated",
        img_suffix=".jpg",
        ann_suffix=".png",
        classes=COCOSTUFF10K_CLASSES,
        palette=build_pascal_style_palette(),
        raw_class_ids=COCOSTUFF10K_RAW_CLASS_IDS,
    ),
    "ade20k": DatasetSpec(
        name="ade20k",
        display_name="ADE20K",
        num_classes=len(ADE20K_RAW_CLASS_IDS),
        ignore_idx=255,
        default_image_size=512,
        base_prompt=(
            "photorealistic indoor or outdoor scene, realistic objects and surfaces, "
            "high detail, natural lighting, sharp focus. "
        ),
        negative_prompt=(
            "blurry, low quality, cartoon, anime, painting, drawing, illustration, "
            "deformed objects, artifacts, watermark, text, oversaturated."
        ),
        img_suffix=".jpg",
        ann_suffix=".png",
        classes=ADE20K_CLASSES,
        palette=build_pascal_style_palette(),
        raw_class_ids=ADE20K_RAW_CLASS_IDS,
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    normalized_name = dataset_name.lower()
    if normalized_name not in DATASET_SPECS:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {', '.join(sorted(DATASET_SPECS))}."
        )
    return DATASET_SPECS[normalized_name]
