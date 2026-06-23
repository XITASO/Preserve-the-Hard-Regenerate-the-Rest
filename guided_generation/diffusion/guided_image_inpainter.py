import inspect
import os
from pathlib import Path
import torch
import torch.nn.functional as torch_F
import numpy as np
import cv2
from PIL import Image
from torch.utils.data.dataloader import DataLoader

from dataset_configs import get_background_train_ids
from guided_generation.diffusion.pipeline_builder import build_pipeline
from guided_generation.diffusion.controlnet_utils import (
    build_control_image,
    resolve_controlnets,
    tensor_to_pil_image,
)
from guided_generation.diffusion.callbacks import make_entropy_guidance_callback, make_controlnet_entropy_guidance_callback
from guided_generation.guidance.dino_guide import DinoGuide
from vfm4ss.models.encoder_config import DEFAULT_ENCODER_NAME

from guided_generation.utils.seed import set_seed
from guided_generation.datasets.base_dataset import BaseSegDataset
from guided_generation.guidance.guidance_scheduler import GuidanceScheduler
from guided_generation.utils.pipeline_resolution import normalize_optional_str


class GuidedImageInpainter:
    POST_PROCESS_EROSION_RADIUS = 0
    POST_PROCESS_BLUR_SIGMA = 1.5

    # Canonical SDXL aspect-ratio buckets at the 1024 training family (height, width).
    # Running generation at any of these keeps the UNet close to its training
    # distribution; feeding e.g. 500x334 leads to heavy artifacts. For VRAM-constrained
    # runs (e.g. with classifier guidance enabled), these can be scaled down via
    # `generation_image_size` — quality drops below ~768 but the pixel count / backward
    # graph shrinks roughly quadratically.
    _BASE_SDXL_ASPECT_BUCKETS: tuple[tuple[int, int], ...] = (
        (1024, 1024),
        (1152, 896), (896, 1152),
        (1216, 832), (832, 1216),
        (1344, 768), (768, 1344),
        (1536, 640), (640, 1536),
    )

    def __init__(
        self,
        dataset: BaseSegDataset,
        seed: int = 0,
        model: str = "sdxl_inpainting",
        inpainter_model_id: str | None = None,
        output_folder: str = "data/synthetic_datasets/entropy_05",
        guidance_checkpoint = "",
        context_guidance_strength: float = 1.0,
        save_step_images_folder: str = None,
        base_prompt: str = "",
        negative_prompt: str | None = None,
        guidance_region: str = "not-selected",
        num_steps: int = 25,
        cfg_guidance_scale: float = 7.0,
        controlnet_path: str = None,
        controlnet_conditioning_scale: float = 1.0,
        control_methods: list[str] | None = None,
        controlnet_paths: list[str] | None = None,
        controlnet_conditioning_scales: list[float] | None = None,
        post_process: bool = True,
        classifier_guidance_schedule: bool = False,
        mask_erosion_kernel: int = 0,
        foreground_only_annotations: bool = False,
        ignore_generated_region: bool = True,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        guide_num_classes: int = 19,
        guide_img_size: tuple[int, int] = (1024, 1024),
        generation_image_size: int = 1024,
    ):
        # Seed / device / dtype
        set_seed(seed)
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Init params
        self.dataset = dataset
        self.dataloader = DataLoader(dataset=dataset, batch_size=1)
        self.base_prompt = base_prompt
        self.negative_prompt = negative_prompt
        self.num_steps = num_steps
        self.guidance_region = guidance_region
        self.output_folder = output_folder
        self.cfg_guidance_scale = cfg_guidance_scale
        self.controlnets = resolve_controlnets(
            control_methods=control_methods,
            controlnet_paths=controlnet_paths,
            controlnet_conditioning_scales=controlnet_conditioning_scales,
            legacy_controlnet_path=controlnet_path,
            legacy_controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        self.use_controlnet = len(self.controlnets) > 0
        self.post_process = post_process
        self.classifier_guidance_schedule = classifier_guidance_schedule
        self.mask_erosion_kernel = mask_erosion_kernel
        self.foreground_only_annotations = foreground_only_annotations
        self.ignore_generated_region = ignore_generated_region
        self.generation_image_size = generation_image_size
        self.sdxl_buckets = self._scaled_buckets(generation_image_size)
        self.output_img_dir = os.path.join(self.output_folder, dataset.img_dir, dataset.split)
        self.output_ann_dir = os.path.join(self.output_folder, dataset.ann_dir, dataset.split)
        self.guidance_checkpoint = normalize_optional_str(guidance_checkpoint)
        self.inpainter_model = model
        self.inpainter_model_id = normalize_optional_str(inpainter_model_id)

        # Auto-select model type if any ControlNet is configured.
        if self.use_controlnet and model == "sdxl_inpainting":
            model = "sdxl_controlnet_inpainting"
        if self.use_controlnet and model not in {"sdxl_controlnet_inpainting"}:
            raise ValueError("ControlNet generation is only supported for the SDXL ControlNet inpainter.")
        self.inpainter_model = model
        self.native_diffusers_inputs = model in {
            "flux_fill",
            "sdxl_diffusers_inpainting",
            "sd15_diffusers_inpainting",
        }
        if context_guidance_strength > 0 and self.native_diffusers_inputs:
            raise ValueError(
                "Segmentation classifier guidance is only supported for the custom SDXL inpainter. "
                "Set --context_guidance_strength 0.0 for native diffusers inpainters."
            )

        # Build pipeline
        resolved_controlnet_paths = [controlnet.path for controlnet in self.controlnets] if self.use_controlnet else None
        self.pipe = build_pipeline(
            model,
            self.dtype,
            self.device,
            controlnet_paths=resolved_controlnet_paths,
            model_id=self.inpainter_model_id,
        )
        self.guide = None
        schedule_type = "paper_early" if self.classifier_guidance_schedule else "constant"
        self.guidance_scheduler = GuidanceScheduler(
            base_eta=context_guidance_strength,
            schedule_type=schedule_type,
            diffusion_scheduler=self.pipe.scheduler,
        )

        # Optional callback
        self.callback_fn = None
        self.callback_inputs = []
        if context_guidance_strength > 0:
            if self.guidance_checkpoint is None:
                raise ValueError(
                    "Segmentation guidance is enabled but no guidance checkpoint could be resolved."
                )
            self.guide = DinoGuide(
                ckpt_path=self.guidance_checkpoint,
                img_size=guide_img_size,
                num_classes=guide_num_classes,
                encoder_name=encoder_name,
            ).to(self.device)
            callback_factory = (
                make_controlnet_entropy_guidance_callback if self.use_controlnet
                else make_entropy_guidance_callback
            )
            self.callback_fn, self.callback_inputs = callback_factory(
                guide=self.guide,
                guidance_scheduler=self.guidance_scheduler,
                guidance_region=self.guidance_region,
                save_step_images_folder=save_step_images_folder,
            )

        # Create output folders
        os.makedirs(self.output_img_dir, exist_ok=True)
        os.makedirs(self.output_ann_dir, exist_ok=True)

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    @classmethod
    def _scaled_buckets(cls, target_size: int) -> tuple[tuple[int, int], ...]:
        """Scale the base 1024-family buckets to a smaller target, rounded to multiples of 8."""
        if target_size == 1024:
            return cls._BASE_SDXL_ASPECT_BUCKETS
        scale = target_size / 1024
        scaled: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for bh, bw in cls._BASE_SDXL_ASPECT_BUCKETS:
            sbh = max(64, int(round(bh * scale / 8)) * 8)
            sbw = max(64, int(round(bw * scale / 8)) * 8)
            key = (sbh, sbw)
            if key in seen:
                continue
            seen.add(key)
            scaled.append(key)
        return tuple(scaled)

    def _select_sdxl_bucket(self, height: int, width: int) -> tuple[int, int]:
        target_aspect = height / width
        return min(
            self.sdxl_buckets,
            key=lambda bucket: abs((bucket[0] / bucket[1]) - target_aspect),
        )

    @staticmethod
    def _resize_inputs_to_bucket(
        image: torch.Tensor,
        annotation: torch.Tensor,
        preserve_mask: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.shape[-2] == target_height and image.shape[-1] == target_width:
            return image, annotation, preserve_mask
        resized_image = torch_F.interpolate(
            image.unsqueeze(0),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        resized_annotation = torch_F.interpolate(
            annotation.unsqueeze(0).unsqueeze(0).float(),
            size=(target_height, target_width),
            mode="nearest",
        ).squeeze(0).squeeze(0).to(annotation.dtype)
        resized_preserve_mask = torch_F.interpolate(
            preserve_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(target_height, target_width),
            mode="nearest",
        ).squeeze(0).squeeze(0).to(torch.bool)
        return resized_image, resized_annotation, resized_preserve_mask

    def _pad_generation_inputs(
        self,
        image: torch.Tensor,
        annotation: torch.Tensor,
        preserve_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        padded_height = self._round_up_to_multiple(height, 8)
        padded_width = self._round_up_to_multiple(width, 8)
        pad_height = padded_height - height
        pad_width = padded_width - width
        if pad_height == 0 and pad_width == 0:
            return image, annotation, preserve_mask

        image = torch_F.pad(image.unsqueeze(0), (0, pad_width, 0, pad_height), mode="replicate").squeeze(0)
        annotation = torch_F.pad(annotation, (0, pad_width, 0, pad_height), mode="constant", value=self.dataset.ignore_idx)
        preserve_mask = torch_F.pad(
            preserve_mask.float(),
            (0, pad_width, 0, pad_height),
            mode="constant",
            value=1.0,
        ).bool()
        return image, annotation, preserve_mask

    def _build_controlnet_inputs(
        self,
        init_image: torch.Tensor,
        annotation: torch.Tensor,
    ) -> tuple[Image.Image | list[Image.Image], float | list[float]]:
        image_pil = tensor_to_pil_image(init_image)
        control_images = [
            build_control_image(
                method=controlnet.method,
                image=image_pil,
                annotation=annotation,
                palette=self.dataset.palette,
            )
            for controlnet in self.controlnets
        ]
        conditioning_scales = [controlnet.conditioning_scale for controlnet in self.controlnets]

        if len(control_images) == 1:
            return control_images[0], conditioning_scales[0]
        return control_images, conditioning_scales

    @staticmethod
    def _mask_to_pil_image(mask: np.ndarray) -> Image.Image:
        mask_uint8 = np.clip(mask.astype(np.uint8) * 255, 0, 255)
        return Image.fromarray(mask_uint8, mode="L")

    def _prepare_pipeline_image_inputs(
        self,
        image: torch.Tensor,
        mask: np.ndarray,
    ) -> tuple[torch.Tensor | Image.Image, torch.Tensor | Image.Image]:
        if not self.native_diffusers_inputs:
            return image, torch.from_numpy(mask)
        return tensor_to_pil_image(image), self._mask_to_pil_image(mask)

    def _filter_pipeline_kwargs(self, pipe_kwargs: dict[str, object]) -> dict[str, object]:
        try:
            parameters = inspect.signature(self.pipe.__call__).parameters
        except (TypeError, ValueError):
            return {key: value for key, value in pipe_kwargs.items() if value is not None}
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return {key: value for key, value in pipe_kwargs.items() if value is not None}
        return {
            key: value
            for key, value in pipe_kwargs.items()
            if value is not None and key in parameters
        }

    def _build_foreground_annotation_mask(self, annotation: torch.Tensor) -> torch.Tensor:
        background_ids = get_background_train_ids(self.dataset.dataset_name)
        background_ids_t = torch.as_tensor(
            sorted(background_ids),
            device=annotation.device,
            dtype=annotation.dtype,
        )
        foreground_mask = ~torch.isin(annotation, background_ids_t)
        if self.dataset.ignore_idx is not None:
            foreground_mask &= annotation != self.dataset.ignore_idx
        return foreground_mask

    def _post_process_output(
        self,
        original_image: torch.Tensor,
        generated_image: Image.Image,
        preserve_mask: torch.Tensor,
    ) -> tuple[Image.Image, np.ndarray]:
        """Paste original foreground pixels back and feather the seam."""
        original_np = original_image.detach().cpu().permute(1, 2, 0).numpy()
        original_np = np.clip(np.rint(original_np * 255), 0, 255).astype(np.uint8)
        generated_np = np.asarray(generated_image, dtype=np.uint8)

        preserve_mask_np = preserve_mask.detach().cpu().numpy().astype(bool)
        inpaint_mask = (~preserve_mask_np).astype(np.uint8)

        kernel_size = 2 * self.POST_PROCESS_EROSION_RADIUS + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded_inpaint_mask = cv2.erode(inpaint_mask, kernel)

        alpha = eroded_inpaint_mask.astype(np.float32)
        alpha = cv2.GaussianBlur(
            alpha,
            ksize=(0, 0),
            sigmaX=self.POST_PROCESS_BLUR_SIGMA,
            sigmaY=self.POST_PROCESS_BLUR_SIGMA,
            borderType=cv2.BORDER_REPLICATE,
        )
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        composite = original_np.astype(np.float32) * (1.0 - alpha) + generated_np.astype(np.float32) * alpha
        composite = np.clip(np.rint(composite), 0, 255).astype(np.uint8)
        effective_generated_mask = alpha[..., 0] > 0.0

        return Image.fromarray(composite), effective_generated_mask

    def generate_samples(self, num_samples=-1):
        """Iterate through the given data and generate new samples."""
        for i, batch in enumerate(self.dataloader):
            if num_samples > 0 and i == num_samples:
                break

            # Load a sample from the dataset
            original_image = batch["img"][0]
            original_annotation = batch["ann"][0]
            original_preserve_mask = batch["mask"][0][0].type(torch.bool)
            image_name = batch["img_path"][0].split(os.sep)[-1]
            output_image_name = image_name
            if Path(image_name).suffix.lower() in {".jpg", ".jpeg"}:
                output_image_name = f"{Path(image_name).stem}.png"
            output_image_path = os.path.join(self.output_img_dir, output_image_name)
            output_annotation_path = os.path.join(
                self.output_ann_dir,
                image_name.replace(self.dataset.img_suffix, self.dataset.ann_suffix),
            )
            if os.path.exists(output_image_path) and os.path.exists(output_annotation_path):
                print(f"[resume] Skipping existing sample {i + 1}: {image_name}")
                continue

            _, original_height, original_width = original_image.shape

            # SDXL was trained on ~1024^2 aspect-ratio buckets; feeding raw COCO sizes
            # (~500x334) produces heavy artifacts. Scale to the nearest bucket for
            # generation, then downsample the result back to the native resolution.
            bucket_height, bucket_width = self._select_sdxl_bucket(original_height, original_width)
            init_image, annotation, preserve_mask = self._resize_inputs_to_bucket(
                original_image,
                original_annotation,
                original_preserve_mask,
                bucket_height,
                bucket_width,
            )
            init_image, annotation, preserve_mask = self._pad_generation_inputs(
                init_image,
                annotation,
                preserve_mask,
            )
            bucket_inpaint_mask_np = (~preserve_mask).numpy().astype(np.uint8)
            if self.mask_erosion_kernel > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (self.mask_erosion_kernel, self.mask_erosion_kernel)
                )
                bucket_inpaint_mask_np = cv2.erode(bucket_inpaint_mask_np, kernel)
            pipe_image, mask_image = self._prepare_pipeline_image_inputs(init_image, bucket_inpaint_mask_np)
            _, height, width = init_image.shape
            sample_prompt = batch.get("prompt", [""])[0] if isinstance(batch, dict) else ""
            prompt = self.base_prompt + sample_prompt
            print(prompt)

            # Build pipeline kwargs
            pipe_kwargs = dict(
                prompt=prompt,
                image=pipe_image,
                mask_image=mask_image,
                width=width,
                height=height,
                guidance_scale=self.cfg_guidance_scale,
                num_inference_steps=self.num_steps,
                generator=torch.Generator(device=self.device).manual_seed(self.seed + i),
                callback_on_step_end=self.callback_fn,
                callback_on_step_end_tensor_inputs=self.callback_inputs,
                negative_prompt=self.negative_prompt,
            )

            # Add ControlNet-specific args
            if self.use_controlnet:
                control_image, controlnet_conditioning_scale = self._build_controlnet_inputs(init_image, annotation)
                pipe_kwargs["control_image"] = control_image
                pipe_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

            # Run pipeline
            pipe_kwargs = self._filter_pipeline_kwargs(pipe_kwargs)
            out_image = self.pipe(**pipe_kwargs).images[0]

            # Strip safety padding, then resize the generated image back to the native size
            # so it aligns pixel-for-pixel with the original annotation and preserve mask.
            out_image = out_image.crop((0, 0, bucket_width, bucket_height))
            if (bucket_height, bucket_width) != (original_height, original_width):
                out_image = out_image.resize(
                    (original_width, original_height), resample=Image.LANCZOS
                )

            # Derive the native-res inpaint mask from the exact bucket mask the pipeline
            # consumed, so post-processing blends only what was actually regenerated.
            bucket_inpaint_mask_unpadded = bucket_inpaint_mask_np[:bucket_height, :bucket_width]
            if (bucket_height, bucket_width) != (original_height, original_width):
                native_inpaint_mask_np = cv2.resize(
                    bucket_inpaint_mask_unpadded,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                native_inpaint_mask_np = bucket_inpaint_mask_unpadded
            effective_generated_mask = native_inpaint_mask_np.astype(bool)

            if self.post_process:
                effective_preserve_mask = torch.from_numpy(~effective_generated_mask)
                out_image, effective_generated_mask = self._post_process_output(
                    original_image=original_image,
                    generated_image=out_image,
                    preserve_mask=effective_preserve_mask,
                )

            output_annotation = original_annotation.clone()
            if self.foreground_only_annotations:
                # Simple-mode masks are class-based background regeneration:
                # keep only original foreground labels and ignore all background,
                # regardless of erosion bands used for image generation.
                foreground_annotation_mask = self._build_foreground_annotation_mask(original_annotation)
                output_annotation[~foreground_annotation_mask] = self.dataset.ignore_idx
            elif self.ignore_generated_region:
                output_annotation[torch.from_numpy(effective_generated_mask)] = self.dataset.ignore_idx
            annotation_np = output_annotation.type(torch.uint8).detach().cpu().numpy()
            Image.fromarray(annotation_np).save(output_annotation_path)
            out_image.save(output_image_path)
