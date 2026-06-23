from PIL import Image
import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt


def _vae_decode(vae, latent):
    return vae.decode(latent).sample


def latent2img(latent, vae: nn.Module):
    """Decode a batched torch tensor latent to a full resolution image as a 3 channel torch tensor in [0;1]."""
    latent_0_est = 1 / vae.config.scaling_factor * latent
    if torch.is_grad_enabled():
        clean_img_est = checkpoint(_vae_decode, vae, latent_0_est, use_reentrant=False)
    else:
        clean_img_est = vae.decode(latent_0_est).sample
    return (clean_img_est / 2 + 0.5).clamp(0, 1)


def make_gif(image_folder, output_file, duration=500, scale_factor=1.0):
    # List all image files in the specified folder.
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]

    # Sort images by name to ensure correct order.
    images.sort()

    # Open images, resize them based on scale_factor, and append to frames list.
    frames = []
    for img in images:
        with Image.open(os.path.join(image_folder, img)) as image:
            if isinstance(scale_factor, float):
                if scale_factor != 1.0:
                    new_width = int(image.width * scale_factor)
                    new_height = int(image.height * scale_factor)
                    image = image.resize((new_width, new_height))
            elif isinstance(scale_factor, int):
                image = image.resize((scale_factor, scale_factor))
            else:
                # Assume tuple of w, h
                image = image.resize((scale_factor[0], scale_factor[1]))
            frames.append(image)

    # Save the frames as a GIF.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    frames[0].save(output_file, format="GIF", append_images=frames[1:], save_all=True, duration=duration, loop=0)

    print(f"GIF saved as {output_file}")


def view_seg_with_plt(seg_path):
    img = Image.open(seg_path).convert("L")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
