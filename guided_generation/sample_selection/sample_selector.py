# guided_generation/sample_selection/sample_selector.py
import os
import shutil
import hashlib
import math
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vision_F
from torch.utils.data import DataLoader
import tqdm
import heapq
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt # New import
from dataset_configs import get_background_train_ids
from guided_generation.datasets.base_dataset import BaseSegDataset
from guided_generation.guidance.diffusion_guide import calculate_shannon_entropy
import albumentations as A
from collections import defaultdict


# ---------------------------------------------------------------------------
# Simple-mode background class definitions (train-id space, per dataset).
#
# When --simple_mode is enabled, the rule is literal: every pixel whose train
# id is in the set below is INPAINT (background to be regenerated); every
# other pixel — including the dataset's ignore index, unless it is listed
# here explicitly — is PRESERVED. There is no thresholding (no min_pixel,
# no min_obj_size); those flags are entropy-mode only and never reach here.
#
# The mask file written to the cache follows the same "preserve mask"
# convention as the entropy-based path (white = keep), so the downstream
# inpainter handles both modes identically.
#
# Edit BACKGROUND_TRAIN_IDS_BY_DATASET in dataset_configs.py to tweak the split
# without touching the selector logic.
# ---------------------------------------------------------------------------

SIMPLE_MODE_BACKGROUND_CLASSES: Dict[str, set] = {
    # Cityscapes 19-class train-id space.
    # Background: road, sidewalk, building, wall, fence, vegetation, terrain, sky.
    # Foreground: pole, traffic light, traffic sign, person, rider, car, truck,
    #             bus, train, motorcycle, bicycle.
    # 255 is included because the standard 19-class mapping collapses the
    # background classes "guard rail / bridge / tunnel / polegroup" (and the
    # void/sensor-border/car-hood pixels) into the ignore index. This mirrors
    # the user's original Cityscapes background spec.
    "cityscapes": get_background_train_ids("cityscapes"),
    "bdd100k": get_background_train_ids("bdd100k"),
    "uavid": get_background_train_ids("uavid"),
    # Pascal VOC: index 0 (background) and index 255 (void/ignore) are treated
    # as background; the 20 object classes (1-20) are foreground.
    "pascal_voc": get_background_train_ids("pascal_voc"),
    # COCO-Stuff 10k: 91 stuff classes are background (train ids 80-170 in the
    # standardized ordering); the 80 thing classes (train ids 0-79) are
    # foreground. The ignore
    # index (255) is intentionally NOT in this set: per the rule "ALL BUT THE
    # BACKGROUND CLASSES get selected for the mask", any pixel whose train id
    # isn't a stuff class — including ignore — is preserved.
    "cocostuff10k": get_background_train_ids("cocostuff10k"),
    "ade20k": get_background_train_ids("ade20k"),
}


class SampleSelector:

    def __init__(
        self,
        dataset: BaseSegDataset,
        selector_type: str,
        seg_model: Optional[torch.nn.Module],
        selector_type_kwargs: Dict = {},
        device=None,
        cache_dir: str = None,
        clear_cache_on_init: bool = False,
        num_workers: int = 16,
        transform = None,
        score_image_size: tuple[int, int] | None = None,
        transforms_per_sample=1,
        save_heatmaps: bool = False, # New Arg
        simple_mode: bool = False,
        simple_mode_erosion_kernel: int = 0,
    ):
        """
        The sample selector selects the most critical samples from a dataset given a model.
        The selection can be based on different sample criteria e.g. prediction entropy or
        prediction accuracy. Additionally to the samples, a mask per sample corresponding to the
        selected fixed area of the image is selected.
        """
        self.dataset = dataset
        self.selector_type = selector_type
        self.seg_model = seg_model
        self.selector_type_kwargs = dict(selector_type_kwargs)
        self.selector_seed = int(self.selector_type_kwargs.pop("selector_seed", 0))
        self.num_workers = num_workers
        self.transform = transform
        self.score_image_size = score_image_size
        self.transforms_per_sample = transforms_per_sample
        self.save_heatmaps = save_heatmaps # Store flag
        self.simple_mode = simple_mode
        self.simple_mode_erosion_kernel = int(simple_mode_erosion_kernel)

        if self.simple_mode:
            if self.dataset.dataset_name not in SIMPLE_MODE_BACKGROUND_CLASSES:
                raise ValueError(
                    f"--simple_mode is not configured for dataset '{self.dataset.dataset_name}'. "
                    f"Supported datasets: {sorted(SIMPLE_MODE_BACKGROUND_CLASSES)}."
                )
            self.simple_mode_bg_classes = SIMPLE_MODE_BACKGROUND_CLASSES[self.dataset.dataset_name]
            print(
                f"[simple_mode] Dataset='{self.dataset.dataset_name}', "
                f"background train ids={sorted(self.simple_mode_bg_classes)}, "
                f"requested_erosion_kernel={self.simple_mode_erosion_kernel}."
            )
            if self.simple_mode_erosion_kernel > 0:
                print(
                    "[simple_mode] Note: step 2 now saves object-only preserve masks. "
                    "Use step 3 --mask_erosion_kernel for generation-time erosion."
                )

        assert isinstance(self.transform, A.ReplayCompose), "transform needs to be replayable!"
        self.cache_dir = (
            cache_dir
            if cache_dir
            else os.path.join(
                ".cached_images4gen", self.dataset.dataset_name, datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
            )
        )
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.req_cache_subdirs = ["images", "annotations", "masks"]

        # Add heatmaps directory only if flag is True
        if self.save_heatmaps:
            self.req_cache_subdirs.append("heatmaps")

        self.selector_function = self._init_mask_selector_function(selector_type=selector_type)
        self.selector_requires_logits = selector_type in {
            "highest_entropy_class",
            "highest_entropy_class_multi",
            "lowest_entropy_class_multi",
        }

        self._init_cache(clear_cache_on_init)

    def _rng_for_key(self, rng_key: str | None) -> np.random.Generator:
        key = "" if rng_key is None else str(rng_key)
        digest = hashlib.blake2s(
            f"{self.selector_seed}:{self.selector_type}:{key}".encode("utf-8"),
            digest_size=8,
        ).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False) % (2**32)
        return np.random.default_rng(seed)

    def _resize_for_scoring(
        self,
        image: torch.Tensor,
        ann: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.score_image_size is None:
            return image, ann

        image_for_scoring = F.interpolate(
            image.unsqueeze(0),
            size=self.score_image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        ann_for_scoring = F.interpolate(
            ann.unsqueeze(0).unsqueeze(0).float(),
            size=self.score_image_size,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
        return image_for_scoring, ann_for_scoring

    def _resize_mask_to_original_size(self, mask: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=original_size,
            mode="nearest",
        ).squeeze(0).squeeze(0).bool()

    def _resize_entropy_to_original_size(
        self,
        entropy_map: torch.Tensor,
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        return F.interpolate(
            entropy_map.unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    def _init_cache(self, clear_cache: bool):

        def _check_valid_cache_exists(folder_path, subdirs):
            """Check if all required subdirectories exist in the given folder."""
            if not os.path.exists(self.cache_dir):
                return False
            return all(os.path.isdir(os.path.join(folder_path, subdir)) for subdir in subdirs)

        # Check if the cache folder exists
        if _check_valid_cache_exists(self.cache_dir, self.req_cache_subdirs):
            # If clear_cache is True, clear the content of the folder
            if clear_cache:
                try:
                    for cache_path in [
                        os.path.join(self.cache_dir, subdir, "train") for subdir in self.req_cache_subdirs
                    ]:
                        if os.path.exists(cache_path):
                            shutil.rmtree(cache_path)
                    print(f"Cache folder '{self.cache_dir}' cleared.")
                except Exception as e:
                    print(f"Error clearing cache folder: {e}")

        else:
            if clear_cache:
                print(
                    f"Warning: cache folder '{self.cache_dir}' does not contain all required subdirectories nor does it not exist yet. Did not delete anything."
                )

        # Create the cache folder and subdirectories
        os.makedirs(self.cache_dir, exist_ok=True)
        for subdir in self.req_cache_subdirs:
            os.makedirs(os.path.join(self.cache_dir, subdir, "train"), exist_ok=True)
        print(f"Cache folder '{self.cache_dir}' was created.")

    def _build_class_based_mask(self, ann: torch.Tensor) -> torch.Tensor:
        """Build a *preserve* mask (1=preserve, 0=inpaint).

        Rule: preserve = ann is NOT in the dataset's declared background set.
        No thresholding, no class-frequency filter, no min_pixel — every pixel
        whose train id is not listed as background ends up preserved. The
        ignore index is treated like any other id: included in inpaint only if
        the dataset's bg list explicitly contains it (e.g. Cityscapes and
        Pascal VOC have 255 in their bg lists; COCO-Stuff does not).

        Cache convention (shared with the entropy-based path) is that the saved
        mask file marks the region to KEEP. The downstream inpainter inverts it
        to obtain the inpaint mask, so step 3 behavior is identical between
        modes — only the mask source differs.

        Erosion is intentionally not applied here. The cache mask should be a
        clean object-only preserve mask; generation-time erosion belongs in
        step 3, where it does not alter the saved step-2 mask.
        """
        if not self.simple_mode:
            raise RuntimeError("_build_class_based_mask is only valid in simple_mode.")

        inpaint_ids_t = torch.as_tensor(
            sorted(self.simple_mode_bg_classes), device=ann.device, dtype=ann.dtype
        )
        preserve_mask = ~torch.isin(ann, inpaint_ids_t)

        return preserve_mask

    def _init_mask_selector_function(self, selector_type):
        selector_types = {
            "highest_entropy_class": self._select_highest_entropy_class,
            "highest_entropy_class_multi": self._select_highest_entropy_class_multi,
            "lowest_entropy_class_multi": self._select_lowest_entropy_class_multi,
            "random_class_multi": self._select_random_class_multi,
            "random_square_region": self._select_random_square_region,
        }
        if selector_type in selector_types:
            return selector_types[selector_type]
        else:
            raise ValueError(
                f"Invalid selector_type '{selector_type}' provided. Available types are: [{', '.join(selector_types.keys())}]."
            )

    def _select_highest_entropy_class(self, logits: torch.Tensor, ann: torch.Tensor, min_pixel: float = 0, **kwargs):
        """Select the class with the highest mean entropy across the image."""
        assert 0<= min_pixel <=1, f"min_pixel {min_pixel} must be between 0 and 1"
        
        pixel_entropy = calculate_shannon_entropy(logits)
        
        mean_class_entropy = {}
        min_pixel = min_pixel * logits.shape[1] * logits.shape[2]
        for c in torch.unique(ann):
            if c == self.dataset.ignore_idx:
                continue
            if torch.sum(ann == c) > min_pixel:
                mean_class_entropy[c] = torch.mean(pixel_entropy[ann == c])
        if not mean_class_entropy:
            return None, None, pixel_entropy
        max_key = max(mean_class_entropy, key=mean_class_entropy.get)
        mask = ann == max_key
        
        # Return pixel_entropy as 3rd arg
        return mask, mean_class_entropy[max_key], pixel_entropy

    def _select_entropy_class_multi(
        self,
        logits: torch.Tensor,
        ann: torch.Tensor,
        min_obj_size: float = 0,
        min_pixel=0.05,
        reverse: bool = True,
        **kwargs,
    ):
        """Select classes by mean entropy until their union covers min_pixel."""
        assert 0<= min_pixel <=1, f"min_pixel {min_pixel} must be between 0 and 1"
        
        pixel_entropy = calculate_shannon_entropy(logits)
        
        mean_class_entropy = {}
        # Set to absolute pixel value if expressed as a percentage
        if min_obj_size < 1:
            min_obj_size = min_obj_size * logits.shape[1] * logits.shape[2]
        for c in torch.unique(ann):
            if c == self.dataset.ignore_idx:
                continue
            obj_size = torch.sum(ann == c)
            if obj_size > min_obj_size:
                mean_class_entropy[c] = [torch.mean(pixel_entropy[ann == c]), obj_size]
        if not mean_class_entropy:
            return None, None, pixel_entropy

        sorted_entropies = sorted(mean_class_entropy.items(), key=lambda x: x[1][0], reverse=reverse)
        selected_keys, cur_size = [], 0
        max_size = min_pixel * logits.shape[1] * logits.shape[2]
        for item in sorted_entropies:
            selected_keys.append(item[0])
            cur_size += item[1][1]
            if cur_size > max_size:
                break

        # Aggregate classes and recalculate the mean values
        mean_entropy, mask = 0, torch.zeros_like(ann)
        for key in selected_keys:
            mean_entropy += mean_class_entropy[key][0] * mean_class_entropy[key][1]
            mask = torch.logical_or(mask, ann == key)
        if cur_size == 0:
            return None, None, pixel_entropy
        mean_entropy = mean_entropy / cur_size
        score = mean_entropy if reverse else -mean_entropy
        
        # Return pixel_entropy as 3rd arg
        return mask, score, pixel_entropy

    def _select_highest_entropy_class_multi(
        self,
        logits: torch.Tensor,
        ann: torch.Tensor,
        min_obj_size: float = 0,
        min_pixel=0.05,
        **kwargs,
    ):
        """Select classes with the highest mean entropy until min_pixel coverage."""
        return self._select_entropy_class_multi(
            logits=logits,
            ann=ann,
            min_obj_size=min_obj_size,
            min_pixel=min_pixel,
            reverse=True,
            **kwargs,
        )

    def _select_lowest_entropy_class_multi(
        self,
        logits: torch.Tensor,
        ann: torch.Tensor,
        min_obj_size: float = 0,
        min_pixel=0.05,
        **kwargs,
    ):
        """Select classes with the lowest mean entropy until min_pixel coverage."""
        return self._select_entropy_class_multi(
            logits=logits,
            ann=ann,
            min_obj_size=min_obj_size,
            min_pixel=min_pixel,
            reverse=False,
            **kwargs,
        )

    def _valid_class_sizes(
        self,
        ann: torch.Tensor,
        min_obj_size: float,
    ) -> list[tuple[int, int]]:
        if min_obj_size < 1:
            min_obj_size = min_obj_size * ann.shape[0] * ann.shape[1]
        valid_classes: list[tuple[int, int]] = []
        for c in torch.unique(ann):
            class_id = int(c.item())
            if class_id == self.dataset.ignore_idx:
                continue
            obj_size = int(torch.sum(ann == c).item())
            if obj_size > min_obj_size:
                valid_classes.append((class_id, obj_size))
        return valid_classes

    def _select_random_class_multi(
        self,
        logits: torch.Tensor | None,
        ann: torch.Tensor,
        min_obj_size: float = 0,
        min_pixel=0.05,
        rng_key: str | None = None,
        **kwargs,
    ):
        """Randomly select semantic classes until their union covers min_pixel."""
        assert 0 <= min_pixel <= 1, f"min_pixel {min_pixel} must be between 0 and 1"
        valid_classes = self._valid_class_sizes(ann, min_obj_size=min_obj_size)
        if not valid_classes:
            return None, None, None

        rng = self._rng_for_key(rng_key)
        order = rng.permutation(len(valid_classes))
        max_size = min_pixel * ann.shape[0] * ann.shape[1]
        selected_keys: list[int] = []
        cur_size = 0
        for idx in order:
            class_id, obj_size = valid_classes[int(idx)]
            selected_keys.append(class_id)
            cur_size += obj_size
            if cur_size > max_size:
                break

        mask = torch.zeros_like(ann, dtype=torch.bool)
        for class_id in selected_keys:
            mask = torch.logical_or(mask, ann == class_id)
        if cur_size == 0 or mask.sum().item() == 0:
            return None, None, None
        return mask, float(rng.random()), None

    def _select_random_square_region(
        self,
        logits: torch.Tensor | None,
        ann: torch.Tensor,
        min_pixel=0.05,
        rng_key: str | None = None,
        **kwargs,
    ):
        """Select a random square preserve region with area approximately min_pixel."""
        assert 0 <= min_pixel <= 1, f"min_pixel {min_pixel} must be between 0 and 1"
        height, width = ann.shape[-2:]
        target_area = max(1, int(math.ceil(min_pixel * height * width)))
        side = max(1, min(int(math.ceil(math.sqrt(target_area))), height, width))
        rng = self._rng_for_key(rng_key)

        valid_mask = torch.ones_like(ann, dtype=torch.bool)
        if self.dataset.ignore_idx is not None:
            valid_mask = ann != self.dataset.ignore_idx
        if valid_mask.sum().item() == 0:
            return None, None, None

        for _ in range(20):
            top = int(rng.integers(0, height - side + 1))
            left = int(rng.integers(0, width - side + 1))
            mask = torch.zeros_like(ann, dtype=torch.bool)
            mask[top : top + side, left : left + side] = True
            if (mask & valid_mask).sum().item() > 0:
                return mask, float(rng.random()), None
        return None, None, None

    def _save_samples_to_cache(self, samples: List[Dict]):
        """
        Save the selected samples to the cache as images for flexible later use.
        """
        
        def save_sample(
            ann: torch.Tensor, img: torch.Tensor, mask: torch.Tensor, entropy_map: torch.Tensor, idx: int, img_name: str = ""
        ):
            # Convert the annotation indices back to the original ones if only using subset
            if isinstance(self.dataset.reverse_id_remapping, torch.Tensor):
                self.dataset.reverse_id_remapping = self.dataset.reverse_id_remapping.to(ann.device)
                ann = self.dataset.reverse_id_remapping[ann]

            # Convert tensors to appropriate image format
            img = vision_F.to_pil_image(img)
            ann = vision_F.to_pil_image(ann.cpu().byte())
            mask = vision_F.to_pil_image(255 * mask.cpu().byte())

            # Set the image and mask paths to the original dataset syntax
            mask_name = img_name.replace(self.dataset.img_suffix, self.dataset.ann_suffix)

            # Save each as a PNG file
            img.save(os.path.join(self.cache_dir, "images", "train", f"{idx:05}_{img_name}"))
            ann.save(os.path.join(self.cache_dir, "annotations", "train", f"{idx:05}_{mask_name}"))
            mask.save(os.path.join(self.cache_dir, "masks", "train", f"{idx:05}_{mask_name}"))

            # --- OPTIONAL: SAVE HEATMAP ---
            if self.save_heatmaps and entropy_map is not None:
                entropy_np = entropy_map.cpu().numpy()
                
                # Matplotlib setup
                plt.figure(figsize=(10, 10))
                plt.imshow(entropy_np, cmap='magma')
                plt.axis('off')
                
                heatmap_path = os.path.join(self.cache_dir, "heatmaps", "train", f"{idx:05}_{img_name}")
                plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
                plt.close() # Close figure to free memory

        def get_cache_size(cache_dir):
            max_images_count = 0
            for subdir in self.req_cache_subdirs:
                subdir_path = os.path.join(cache_dir, subdir)
                if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                    all_files = os.listdir(subdir_path)
                    max_images_count = max(max_images_count, len(all_files))
            return max_images_count
        
        def flush_batch(batch_data):
            if not batch_data:
                return

            # Save results
            for data in batch_data:
                save_sample(
                    ann=data["ann"],
                    img=data["img"],
                    mask=data["mask"],
                    entropy_map=data["entropy_map"], # Pass the map
                    idx=data["idx"],
                    img_name=data["img_name"]
                )

        if len(samples) == 0:
            return

        # Assign an index to each newly created image based on its score
        path2id_mapping = defaultdict(lambda: [])
        for idx, sample in enumerate(samples):
            path2id_mapping[sample[1]].append(idx)
        base_idx = get_cache_size(self.cache_dir)

        # Only iterate the *selected* dataset entries instead of the full dataset.
        # The previous full-dataset pass used to drive a VLM prompt generator, which
        # has been removed; now this loop only saves images/anns/masks, so loading
        # the non-selected samples is pure I/O waste.
        path_to_idx = {p: i for i, p in enumerate(self.dataset.img_paths)}
        try:
            selected_indices = [path_to_idx[p] for p in path2id_mapping]
        except KeyError as missing:
            raise RuntimeError(
                f"Selected sample path {missing} not found in dataset.img_paths; "
                "cannot build save subset."
            )
        save_subset = torch.utils.data.Subset(self.dataset, selected_indices)
        dataloader = DataLoader(dataset=save_subset, batch_size=1, shuffle=False, num_workers=self.num_workers)

        # Buffer to hold samples before writing them to disk
        batch_buffer = []
        skipped_invalid_crops = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Saving selected samples"):
                # All entries in the subset are pre-filtered to selected paths.
                image_path = batch["img_path"][0]

                org_image = batch["img"][0].permute(1, 2, 0).numpy()
                org_ann = batch["ann"][0].numpy()
                img_name = image_path.split(os.sep)[-1]

                # Retrive multiple indices if requested
                for id in path2id_mapping[image_path]:

                    # Reuse same transformation as before to avoid cashing results
                    replay_params = samples[id][2]
                    replayed = A.ReplayCompose.replay(replay_params, image=org_image, mask=org_ann)
                    image = replayed["image"]
                    ann = replayed["mask"]

                    if self.simple_mode:
                        # Build the *preserve* mask directly from the ground-truth
                        # labels at the original resolution (no seg model, no
                        # entropy). The inpainter inverts this saved mask to get
                        # the inpaint mask — i.e. background+ignore are regenerated
                        # and foreground objects are kept.
                        mask = self._build_class_based_mask(ann)
                        entropy_map = None
                        # Skip crops with nothing to regenerate (entire image is
                        # foreground) — they wouldn't add to a synthetic dataset.
                        if (~mask).sum().item() == 0:
                            skipped_invalid_crops += 1
                            continue
                    else:
                        score_image, score_ann = self._resize_for_scoring(image, ann)
                        score_ann = score_ann.to(self.device)
                        logits = None
                        if self.selector_requires_logits:
                            score_image = score_image.to(self.device)
                            # 1. Run Seg Model (Fast, per image)
                            logits = self.seg_model(score_image.unsqueeze(0))[0]

                        # Unpack 3 values: Mask, Score, optional Entropy Map
                        mask, score, entropy_map = self.selector_function(
                            logits=logits,
                            ann=score_ann,
                            rng_key=f"{image_path}:{repr(replay_params)}",
                            **self.selector_type_kwargs,
                        )
                        if score is None or mask is None:
                            skipped_invalid_crops += 1
                            continue
                        mask = self._resize_mask_to_original_size(mask, ann.shape[-2:])
                        if entropy_map is not None:
                            entropy_map = self._resize_entropy_to_original_size(entropy_map, ann.shape[-2:])
                    
                    # Add to buffer and flush periodically to limit peak memory use.
                    batch_buffer.append({
                        "ann": ann,
                        "img": image,
                        "mask": mask,
                        "entropy_map": entropy_map, # Store map in buffer
                        "idx": base_idx + id,
                        "img_name": img_name
                    })

                    if len(batch_buffer) >= self.num_workers:
                        flush_batch(batch_buffer)
                        batch_buffer = []

            if len(batch_buffer) > 0:
                flush_batch(batch_buffer)

        if skipped_invalid_crops > 0:
            print(f"Skipped {skipped_invalid_crops} cached crops without any valid non-ignore classes.")
        print(f"Saved {len(samples)} new samples to {self.cache_dir}")

    def select_samples(self, num_samples: int = 0) -> List:
        """
        Iterate over the full dataset and select the most interesting samples.
        """
        dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
        selected_samples = []
        skipped_invalid_crops = 0
        heapq.heapify(selected_samples)

        with torch.no_grad():
            for batch in tqdm(dataloader):
                image_id = batch["img_path"][0]
                image_org = batch["img"][0]
                ann_org = batch["ann"][0]

                for _ in range(self.transforms_per_sample):
                    # Run an extra transformation while tracking the parameters
                    augmented = self.transform(image=image_org.permute(1, 2, 0).numpy(), mask=ann_org.numpy())
                    image = augmented["image"]
                    ann = augmented["mask"]

                    if self.simple_mode:
                        # Skip any segmentation/entropy computation. Rank samples by the
                        # number of background (inpaint) pixels so images with more stuff
                        # to regenerate are preferred when num_samples is capped.
                        score_ann = ann
                        if self.score_image_size is not None:
                            _, score_ann = self._resize_for_scoring(image, ann)
                        preserve = self._build_class_based_mask(score_ann)
                        inpaint_pixels = float((~preserve).sum().item())
                        if inpaint_pixels <= 0:
                            skipped_invalid_crops += 1
                            continue
                        score = inpaint_pixels
                    else:
                        score_image, score_ann = self._resize_for_scoring(image, ann)
                        score_ann = score_ann.to(self.device)
                        logits = None
                        if self.selector_requires_logits:
                            score_image = score_image.to(self.device)
                            logits = self.seg_model(score_image.unsqueeze(0))[0]

                        # Unpack 3 values: We ignore the entropy map here (_) to save RAM
                        _, score, _ = self.selector_function(
                            logits=logits,
                            ann=score_ann,
                            rng_key=f"{image_id}:{repr(augmented['replay'])}",
                            **self.selector_type_kwargs,
                        )
                        if score is None:
                            skipped_invalid_crops += 1
                            continue

                    # Add to dict, but only keep at most num_samples
                    sample_item = [score, image_id, augmented["replay"]]
                    if len(selected_samples) < num_samples or num_samples <= 0:
                        heapq.heappush(selected_samples, sample_item)
                    else:
                        heapq.heappushpop(selected_samples, sample_item)

        # Sort samples from hardest to easiest
        selected_samples = sorted(selected_samples, reverse=True)
        if skipped_invalid_crops > 0:
            print(f"Skipped {skipped_invalid_crops} crops without any valid non-ignore classes.")
        self._save_samples_to_cache(selected_samples)

        return selected_samples
