import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL

# Custom sdxl allows for different callback behaviour
from guided_generation.diffusion.custom_diffusers.custom_sdxl import (
    StableDiffusionXLInpaintPipeline as CustomStableDiffusionXLInpaintPipeline,
)


def build_pipeline(
    model: str,
    dtype: torch.dtype,
    device: torch.device,
    controlnet_path: str = None,
    controlnet_paths: list[str] | None = None,
    model_id: str | None = None,
):
    if controlnet_paths is None and controlnet_path is not None:
        controlnet_paths = [controlnet_path]
    scheduler_model_name = None

    # Handle control net setup
    if model == "sd15_control":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", dtype=dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        )
        scheduler_model_name = pipe.config._name_or_path

    elif model == "sdxl_inpainting":
        pipe = CustomStableDiffusionXLInpaintPipeline.from_pretrained(
            model_id or "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        scheduler_model_name = pipe.config._name_or_path

    elif model == "sdxl_controlnet_inpainting":
        from guided_generation.diffusion.custom_diffusers.custom_sdxl_controlnet import (
            StableDiffusionXLControlNetInpaintPipeline,
        )

        if not controlnet_paths:
            raise ValueError("`controlnet_paths` must be provided when using `sdxl_controlnet_inpainting`.")

        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
        if len(controlnet_paths) == 1:
            controlnet = ControlNetModel.from_pretrained(controlnet_paths[0], torch_dtype=dtype)
        else:
            controlnet = [ControlNetModel.from_pretrained(path, torch_dtype=dtype) for path in controlnet_paths]
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            model_id or "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
        )
        scheduler_model_name = pipe.config._name_or_path

    elif model == "sdxl_diffusers_inpainting":
        from diffusers import StableDiffusionXLInpaintPipeline

        load_kwargs = {"torch_dtype": dtype}
        if dtype == torch.float16:
            load_kwargs["variant"] = "fp16"
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id or "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            **load_kwargs,
        )

    elif model == "sd15_diffusers_inpainting":
        from diffusers import StableDiffusionInpaintPipeline

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id or "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
        )

    elif model == "flux_fill":
        try:
            from diffusers import FluxFillPipeline
        except ImportError as exc:
            raise ImportError(
                "The `flux_fill` inpainter requires a diffusers version that provides "
                "`FluxFillPipeline`. Update the container's diffusers package before "
                "running FLUX.1-Fill-dev."
            ) from exc
        try:
            import sentencepiece  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The `flux_fill` inpainter requires `sentencepiece` for the FLUX T5 tokenizer. "
                "Rebuild the standard Docker image after the requirements.txt update, or install "
                "`sentencepiece` in the image before resubmitting the FLUX generation job."
            ) from exc

        flux_dtype = torch.bfloat16 if dtype == torch.float16 else dtype
        pipe = FluxFillPipeline.from_pretrained(
            model_id or "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=flux_dtype,
        )

    else:
        raise ValueError(f"Unknown model type: {model}")

    if scheduler_model_name is not None:
        pipe.scheduler = DDIMScheduler.from_pretrained(scheduler_model_name, subfolder="scheduler")

    if model == "flux_fill" and device.type == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe
