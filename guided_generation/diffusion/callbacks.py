from typing import Callable, List, Optional
import torch
from PIL import Image
import numpy as np
import os

from guided_generation.guidance.diffusion_guide import DiffusionGuide
from guided_generation.guidance.guidance_scheduler import GuidanceScheduler
from guided_generation.utils.images import latent2img


def select_mask4loss(guidance_region: str, mask: torch.Tensor):
    """Mask is a bool torch tensor, which describes the fixed region during the generation process."""
    if guidance_region == "selected":
        return mask
    if guidance_region == "not-selected":
        return torch.logical_not(mask)
    if guidance_region == "full":
        return torch.ones_like(mask)
    raise ValueError(f"Unknown value {guidance_region} for guidance_region.")


def make_entropy_guidance_callback(
    guide: DiffusionGuide,
    guidance_scheduler: GuidanceScheduler,
    guidance_region: str,
    save_step_images_folder: Optional[str] = False,
) -> tuple[Callable, List[str]]:
    """
    Returns (callback_fn, tensor_inputs) pair for diffusers callback_on_step_end.
    Sets global parameters and objects which are needed for guidance later.
    """

    def on_step_end(pipe, step_index: int, timestep: int, kwargs):

        def predict_noise(latent):
            latent_model_input = torch.cat([latent] * 2) if pipe.do_classifier_free_guidance else latent
            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

            # if num_channels_unet == 9:
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise_pred

        with torch.enable_grad():
            latent_t = kwargs["latents"].detach().requires_grad_(True)
            noise_pred = kwargs["noise_pred"].detach().requires_grad_(False)
            extra_step_kwargs = kwargs["extra_step_kwargs"]
            mask_image = kwargs["mask_image"].type(torch.bool).unsqueeze(0)
            prompt_embeds = kwargs["prompt_embeds"]
            added_cond_kwargs = kwargs["added_cond_kwargs"]
            mask = kwargs["mask"]
            masked_image_latents = kwargs["masked_image_latents"]

            _, latent_0_est = pipe.scheduler.step(
                noise_pred, timestep, latent_t, **extra_step_kwargs, return_dict=False
            )

            clean_img_est = latent2img(latent=latent_0_est, vae=pipe.vae)

            if save_step_images_folder:
                os.makedirs(save_step_images_folder, exist_ok=True)
                for i, img in enumerate(clean_img_est):
                    img_np = np.array(255 * img.detach().cpu().permute(1, 2, 0), dtype=np.uint8)
                    Image.fromarray(img_np).save(
                        os.path.join(save_step_images_folder, f"img{i}_step{step_index:03}.png")
                    )

            # Entropy loss and grad to latents
            loss_mask = select_mask4loss(guidance_region=guidance_region, mask=mask_image)
            loss = guide(clean_img_est, loss_mask)
            loss.backward()
            grad = latent_t.grad

        # Some gradient scaling is required to ensure not all values are 0 leading to NaNs.
        # High epsilon due to limited fp16 precision
        gnorm = grad.norm().clamp_min(1e-4)
        grad = grad * (1.0 / gnorm.clamp(max=5.0))

        # Guidance scheduling
        eta_t = guidance_scheduler.get_eta(timestep=timestep)

        with torch.no_grad():
            latent_t = latent_t - eta_t * grad  # gradient descent step
            updated_noise = predict_noise(latent=latent_t)
            delta_noise = torch.mean(torch.abs(updated_noise - noise_pred))
            print(f"Mean delta between noise predictions: {delta_noise.item()}")

            # Recalculate t+1 with updated latent_t
            latent_t_plus_one, _ = pipe.scheduler.step(
                updated_noise, timestep, latent_t, **extra_step_kwargs, return_dict=False
            )
            latent_t_plus_one = latent_t_plus_one.detach()

        kwargs["latents"] = latent_t_plus_one

        return kwargs

    return on_step_end, [
        "latents",
        "noise_pred",
        "extra_step_kwargs",
        "mask_image",
        "prompt_embeds",
        "added_cond_kwargs",
        "mask",
        "masked_image_latents"
    ]


def make_controlnet_entropy_guidance_callback(
    guide: DiffusionGuide,
    guidance_scheduler: GuidanceScheduler,
    guidance_region: str,
    save_step_images_folder: Optional[str] = False,
) -> tuple[Callable, List[str]]:
    """
    Returns (callback_fn, tensor_inputs) pair for the ControlNet pipeline's callback_on_step_end.
    Same gradient guidance as make_entropy_guidance_callback, but predict_noise calls
    ControlNet + UNet (4-channel UNet, no mask/masked_image concatenation).
    """

    def on_step_end(pipe, step_index: int, timestep: int, kwargs):

        def predict_noise(latent):
            latent_model_input = torch.cat([latent] * 2) if pipe.do_classifier_free_guidance else latent
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

            # ControlNet forward
            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_image,
                conditioning_scale=controlnet_conditioning_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

            # UNet forward with ControlNet residuals (4-channel, no mask concatenation)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise_pred

        with torch.enable_grad():
            latent_t = kwargs["latents"].detach().requires_grad_(True)
            noise_pred = kwargs["noise_pred"].detach().requires_grad_(False)
            extra_step_kwargs = kwargs["extra_step_kwargs"]
            mask_image = kwargs["mask_image"].type(torch.bool).unsqueeze(0)
            prompt_embeds = kwargs["prompt_embeds"]
            added_cond_kwargs = kwargs["added_cond_kwargs"]
            control_image = kwargs["control_image"]
            controlnet_conditioning_scale = kwargs["controlnet_conditioning_scale"]

            _, latent_0_est = pipe.scheduler.step(
                noise_pred, timestep, latent_t, **extra_step_kwargs, return_dict=False
            )

            clean_img_est = latent2img(latent=latent_0_est, vae=pipe.vae)

            if save_step_images_folder:
                os.makedirs(save_step_images_folder, exist_ok=True)
                for i, img in enumerate(clean_img_est):
                    img_np = np.array(255 * img.detach().cpu().permute(1, 2, 0), dtype=np.uint8)
                    Image.fromarray(img_np).save(
                        os.path.join(save_step_images_folder, f"img{i}_step{step_index:03}.png")
                    )

            # Entropy loss and grad to latents
            loss_mask = select_mask4loss(guidance_region=guidance_region, mask=mask_image)
            loss = guide(clean_img_est, loss_mask)
            loss.backward()
            grad = latent_t.grad

        # Some gradient scaling is required to ensure not all values are 0 leading to NaNs.
        # High epsilon due to limited fp16 precision
        gnorm = grad.norm().clamp_min(1e-4)
        grad = grad * (1.0 / gnorm.clamp(max=5.0))

        # Guidance scheduling
        eta_t = guidance_scheduler.get_eta(timestep=timestep)

        with torch.no_grad():
            latent_t = latent_t - eta_t * grad  # gradient descent step
            updated_noise = predict_noise(latent=latent_t)
            delta_noise = torch.mean(torch.abs(updated_noise - noise_pred))
            print(f"Mean delta between noise predictions: {delta_noise.item()}")

            # Recalculate t+1 with updated latent_t
            latent_t_plus_one, _ = pipe.scheduler.step(
                updated_noise, timestep, latent_t, **extra_step_kwargs, return_dict=False
            )
            latent_t_plus_one = latent_t_plus_one.detach()

        kwargs["latents"] = latent_t_plus_one

        return kwargs

    return on_step_end, [
        "latents",
        "noise_pred",
        "extra_step_kwargs",
        "mask_image",
        "prompt_embeds",
        "added_cond_kwargs",
        "mask",
        "masked_image_latents",
        "control_image",
        "controlnet_conditioning_scale",
    ]
