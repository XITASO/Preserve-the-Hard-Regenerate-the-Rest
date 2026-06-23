import argparse

from guided_generation.datasets import build_guided_dataset, get_guided_dataset_spec
from guided_generation.diffusion.guided_image_inpainter import GuidedImageInpainter
from guided_generation.diffusion.controlnet_utils import DEFAULT_SENTINELS
from guided_generation.utils.pipeline_resolution import (
    normalize_optional_str,
    read_pipeline_metadata,
    resolve_guidance_checkpoint,
    write_pipeline_metadata,
)
from vfm4ss.models.encoder_config import DEFAULT_ENCODER_NAME


def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate guided inpainted samples for a segmentation dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cityscapes",
        choices=["cityscapes", "uavid", "pascal_voc", "cocostuff10k", "bdd100k", "ade20k"],
        help="Dataset to process (default: %(default)s)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".cached_images4gen/cityscapes/multi_class_0.05",
        help="Root directory for the selected dataset/cache root (default: %(default)s)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1,
        help="Maximum number of samples to load from the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--context_guidance_strength",
        type=float,
        default=0.0,
        help="Context guidance strength for the inpainter (default: %(default)s)",
    )
    parser.add_argument(
        "--inpainter_model",
        type=str,
        default="sdxl_inpainting",
        choices=[
            "sdxl_inpainting",
            "sdxl_controlnet_inpainting",
            "sdxl_diffusers_inpainting",
            "sd15_diffusers_inpainting",
            "flux_fill",
        ],
        help="Inpainting backend used for synthetic generation (default: %(default)s).",
    )
    parser.add_argument(
        "--inpainter_model_id",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face model id/path overriding the selected inpainter backend "
            "(default: backend-specific)."
        ),
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/synthetic_datasets/cityscapes/temp",
        help="Folder to save generated images (default: %(default)s)",
    )
    parser.add_argument(
        "--guidance_checkpoint",
        type=str,
        default=None,
        help="Checkpoint used for guidance (default: %(default)s)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of diffusion steps for generation (default: %(default)s)",
    )
    parser.add_argument(
        "--cfg_guidance_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale passed to the inpainter (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--guide_num_classes",
        type=int,
        default=None,
        help="Number of classes for the guidance segmentation model (default: dataset-specific).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Square image size used by the guidance model (default: dataset-specific).",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default=DEFAULT_ENCODER_NAME,
        help="timm encoder backbone name for the guidance segmenter (default: %(default)s)",
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default=None,
        help="Optional prompt prefix prepended to cached prompts (default: dataset-specific).",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt for the inpainting pipeline (default: dataset-specific).",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        help=(
            "Legacy single-control shorthand for a segmentation ControlNet checkpoint. "
            "Use the new multi-control arguments for combining multiple controlnets."
        ),
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="Legacy single-control conditioning scale (default: %(default)s)",
    )
    parser.add_argument(
        "--control_methods",
        nargs="+",
        default=None,
        help=(
            "Ordered list of control methods to combine. Available methods: seg, depth, edge "
            "(hed is accepted as an alias for edge)."
        ),
    )
    parser.add_argument(
        "--controlnet_paths",
        nargs="+",
        default=None,
        help=(
            "Ordered list of ControlNet checkpoints or Hugging Face model ids aligned with `control_methods`. "
            f"Use one of {sorted(DEFAULT_SENTINELS)} to select the built-in default for depth or edge."
        ),
    )
    parser.add_argument(
        "--controlnet_conditioning_scales",
        nargs="+",
        type=float,
        default=None,
        help="Ordered list of conditioning scales aligned with `control_methods` (default: 1.0 per controlnet).",
    )
    parser.add_argument(
        "--post_process",
        type=parse_bool,
        default=True,
        help="Apply erosion, hard paste-back, and feathered blending to regenerated outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--label_noise_config",
        type=str,
        default=None,
        choices=["full", "ignore_only", "paste_only", "none"],
        help=(
            "Label-noise ablation mode. If set, overrides --post_process and controls whether "
            "generated pixels are ignored in the saved annotation: full=paste-back+ignore, "
            "ignore_only=no paste-back+ignore, paste_only=paste-back+no ignore, none=no paste-back+no ignore."
        ),
    )
    parser.add_argument(
        "--guidance_region",
        type=str,
        default="not-selected",
        choices=["selected", "not-selected", "full"],
        help="Region over which entropy loss is computed: 'not-selected'=foreground, 'selected'=background, 'full'=all (default: %(default)s)",
    )
    parser.add_argument(
        "--classifier_guidance_schedule",
        type=parse_bool,
        default=False,
        help=(
            "If true, apply paper-style early denoising guidance schedule for segmentation guidance. "
            "If false, guidance strength stays constant."
        ),
    )
    parser.add_argument(
        "--mask_erosion_kernel",
        type=int,
        default=0,
        help="Kernel size for eroding the inpainting mask before generation, shrinking the inpainted region to prevent objects from bleeding into the background (0 = disabled, paper uses 7).",
    )
    parser.add_argument(
        "--generation_image_size",
        type=int,
        default=1024,
        help=(
            "Target resolution of the SDXL bucket family used for generation. "
            "1024 (default) matches SDXL's training distribution for best quality. "
            "Lower values (e.g. 768) scale the buckets down proportionally to fit tighter VRAM budgets "
            "(useful when enabling classifier guidance), at the cost of some generation quality."
        ),
    )
    return parser.parse_args()


def step_3_generate_synthetic(
    dataset_name: str = "cityscapes",
    root_dir: str = ".cached_images4gen/cityscapes/multi_class_0.05",
    max_samples: int = 1,
    context_guidance_strength: float = 0.0,
    inpainter_model: str = "sdxl_inpainting",
    inpainter_model_id: str | None = None,
    output_folder: str = "data/synthetic_datasets/cityscapes/temp",
    guidance_checkpoint: str | None = None,
    num_steps: int = 20,
    cfg_guidance_scale: float = 7.0,
    seed: int = 42,
    guide_num_classes: int | None = None,
    image_size: int | None = None,
    encoder_name: str = DEFAULT_ENCODER_NAME,
    base_prompt: str | None = None,
    negative_prompt: str | None = None,
    controlnet_path: str = None,
    controlnet_conditioning_scale: float = 1.0,
    control_methods: list[str] | None = None,
    controlnet_paths: list[str] | None = None,
    controlnet_conditioning_scales: list[float] | None = None,
    post_process: bool = True,
    label_noise_config: str | None = None,
    guidance_region: str = "not-selected",
    classifier_guidance_schedule: bool = False,
    mask_erosion_kernel: int = 0,
    generation_image_size: int = 1024,
) -> None:
    """Generate guided inpainted samples.
    Args:
        dataset_name: (str) Dataset identifier.
        root_dir: (str) Root dir for the dataset/cache root.
        max_samples: (int) Max samples to load.
        context_guidance_strength: (float) Context guidance strength for inpainter.
        inpainter_model: (str) Inpainting backend.
        inpainter_model_id: (str | None) Optional backend model id override.
        output_folder: (str) Directory to save generated images.
        guidance_checkpoint: (str) Checkpoint path for guidance model.
        num_steps: (int) Number of diffusion steps to run.
        cfg_guidance_scale: (float) Classifier-free guidance scale.
        seed: (int) RNG seed for generation.
        guide_num_classes: (int | None) Number of classes for the guidance model.
        image_size: (int | None) Square image size used by the guidance model.
        base_prompt: (str | None) Prompt prefix prepended to cached prompts.
        controlnet_path: (str) Legacy single-control segmentation ControlNet checkpoint (optional).
        controlnet_conditioning_scale: (float) Legacy single-control conditioning scale.
        control_methods: (list[str] | None) Ordered list of control methods to combine.
        controlnet_paths: (list[str] | None) Ordered list of checkpoints/model ids aligned with control_methods.
        controlnet_conditioning_scales: (list[float] | None) Ordered list of conditioning scales aligned with control_methods.
        post_process: (bool) Whether to reinsert preserved pixels after generation.
        guidance_region: (str) Region for entropy loss: 'not-selected'=foreground, 'selected'=background, 'full'=all.
        classifier_guidance_schedule: (bool) Whether to apply paper-style early guidance scheduling.
    """
    dataset_spec = get_guided_dataset_spec(dataset_name)
    cache_metadata = read_pipeline_metadata(root_dir)
    foreground_only_annotations = bool(cache_metadata.get("simple_mode", False))
    ignore_generated_region = True
    if label_noise_config is not None:
        post_process = label_noise_config in {"full", "paste_only"}
        ignore_generated_region = label_noise_config in {"full", "ignore_only"}

    if guide_num_classes is None:
        guide_num_classes = dataset_spec.num_classes
    if image_size is None:
        image_size = dataset_spec.default_image_size
    if base_prompt is None:
        base_prompt = dataset_spec.base_prompt
    if negative_prompt is None:
        negative_prompt = dataset_spec.negative_prompt or None
    if encoder_name == DEFAULT_ENCODER_NAME:
        cached_encoder_name = normalize_optional_str(cache_metadata.get("encoder_name"))
        if cached_encoder_name is not None:
            encoder_name = cached_encoder_name
    guidance_checkpoint = resolve_guidance_checkpoint(
        dataset_name=dataset_name,
        requested_path=guidance_checkpoint,
        metadata_root=root_dir,
    )
    if context_guidance_strength > 0 and guidance_checkpoint is None:
        raise FileNotFoundError(
            f"Could not resolve a guidance checkpoint for dataset '{dataset_name}'. "
            "Pass --guidance_checkpoint explicitly or run step 2 with the updated code so metadata is written."
        )

    dataset = build_guided_dataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        img_dir="images",
        ann_dir="annotations",
        mask_suffix=dataset_spec.ann_suffix,
        load_masks=True,
        load_prompts=False,
        max_samples=max_samples,
    )

    image_generator = GuidedImageInpainter(
        seed=seed,
        dataset=dataset,
        model=inpainter_model,
        inpainter_model_id=inpainter_model_id,
        context_guidance_strength=context_guidance_strength,
        output_folder=output_folder,
        guidance_checkpoint=guidance_checkpoint,
        num_steps=num_steps,
        cfg_guidance_scale=cfg_guidance_scale,
        base_prompt=base_prompt,
        negative_prompt=negative_prompt,
        controlnet_path=controlnet_path,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_methods=control_methods,
        controlnet_paths=controlnet_paths,
        controlnet_conditioning_scales=controlnet_conditioning_scales,
        post_process=post_process,
        ignore_generated_region=ignore_generated_region,
        guidance_region=guidance_region,
        classifier_guidance_schedule=classifier_guidance_schedule,
        mask_erosion_kernel=mask_erosion_kernel,
        foreground_only_annotations=foreground_only_annotations,
        encoder_name=encoder_name,
        guide_num_classes=guide_num_classes,
        guide_img_size=(image_size, image_size),
        generation_image_size=generation_image_size,
    )
    image_generator.generate_samples()
    synthetic_img_suffix = dataset_spec.img_suffix
    if dataset_spec.img_suffix.lower().endswith((".jpg", ".jpeg")):
        synthetic_img_suffix = ".png"
    write_pipeline_metadata(
        output_folder,
        {
            "dataset_name": dataset_name,
            "source_cache_root": root_dir,
            "guidance_checkpoint": guidance_checkpoint,
            "step1_checkpoint": guidance_checkpoint,
            "inpainter_model": inpainter_model,
            "inpainter_model_id": inpainter_model_id,
            "cfg_guidance_scale": cfg_guidance_scale,
            "encoder_name": encoder_name,
            "guide_image_size": [image_size, image_size],
            "guide_num_classes": guide_num_classes,
            "control_methods": control_methods,
            "controlnet_paths": controlnet_paths,
            "controlnet_conditioning_scales": controlnet_conditioning_scales,
            "post_process": post_process,
            "label_noise_config": label_noise_config,
            "ignore_generated_region": ignore_generated_region,
            "guidance_region": guidance_region,
            "generation_image_size": generation_image_size,
            "foreground_only_annotations": foreground_only_annotations,
            "synthetic_img_suffix": synthetic_img_suffix,
        },
    )


if __name__ == "__main__":
    args = parse_args()
    step_3_generate_synthetic(**vars(args))
