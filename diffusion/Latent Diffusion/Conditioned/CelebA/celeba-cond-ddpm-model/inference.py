# inference.py
import os
import json
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn.functional as F

# Import your model definitions and helper functions
from model import (
    UNet,
    VQVAE,
    LinearNoiseScheduler,
    get_tokenizer_and_model,
    get_text_representation,
    get_time_embedding,
)


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def sample_ddpm_inference(
    text_prompt, mask_image_path=None, guidance_scale=1.0, device=torch.device("cpu")
):
    config = load_config()

    diffusion_params = config["diffusion_params"]
    ldm_params = config["ldm_params"]
    autoencoder_params = config["autoencoder_params"]
    train_params = config["train_params"]
    dataset_params = config["dataset_params"]

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_params["num_timesteps"],
        beta_start=diffusion_params["beta_start"],
        beta_end=diffusion_params["beta_end"],
    )

    # Conditioning configuration
    condition_config = ldm_params.get("condition_config", {})
    condition_types = condition_config.get("condition_types", [])

    # Text conditioning
    text_model_type = condition_config["text_condition_config"]["text_embed_model"]
    text_tokenizer, text_model = get_tokenizer_and_model(text_model_type, device)
    empty_text_embed = get_text_representation([""], text_tokenizer, text_model, device)
    text_prompt_embed = get_text_representation(
        [text_prompt], text_tokenizer, text_model, device
    )

    # Image conditioning
    if "image" in condition_types:
        if mask_image_path is not None:
            mask_image = Image.open(mask_image_path).convert("RGB")
            mask_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (
                            ldm_params["condition_config"]["image_condition_config"][
                                "image_condition_h"
                            ],
                            ldm_params["condition_config"]["image_condition_config"][
                                "image_condition_w"
                            ],
                        )
                    ),
                    transforms.ToTensor(),
                ]
            )
            mask_tensor = mask_transform(mask_image).unsqueeze(0).to(device)
        else:
            ic = ldm_params["condition_config"]["image_condition_config"][
                "image_condition_input_channels"
            ]
            H = ldm_params["condition_config"]["image_condition_config"][
                "image_condition_h"
            ]
            W = ldm_params["condition_config"]["image_condition_config"][
                "image_condition_w"
            ]
            mask_tensor = torch.zeros((1, ic, H, W), device=device)
    else:
        mask_tensor = None

    # Build conditioning dictionaries
    uncond_input = {}
    cond_input = {}
    if "text" in condition_types:
        uncond_input["text"] = empty_text_embed
        cond_input["text"] = text_prompt_embed
    if "image" in condition_types:
        uncond_input["image"] = torch.zeros_like(mask_tensor)
        cond_input["image"] = mask_tensor

    # Instantiate and load UNet model
    unet = UNet(autoencoder_params["z_channels"], ldm_params).to(device)
    ldm_ckpt_path = os.path.join(
        train_params["task_name"], train_params["ldm_ckpt_name"]
    )
    if os.path.exists(ldm_ckpt_path):
        ckpt = torch.load(ldm_ckpt_path, map_location=device)
        unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    # Instantiate and load VQVAE autoencoder
    vae = VQVAE(dataset_params["image_channels"], autoencoder_params).to(device)
    vae_ckpt_path = os.path.join(
        train_params["task_name"], train_params["vqvae_autoencoder_ckpt_name"]
    )
    if os.path.exists(vae_ckpt_path):
        ckpt = torch.load(vae_ckpt_path, map_location=device)
        vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    # Determine latent space size (simplified calculation)
    latent_size = dataset_params["image_size"] // (
        2 ** sum(autoencoder_params["down_sample"])
    )
    batch = train_params["num_samples"]
    z_channels = autoencoder_params["z_channels"]

    # Sample initial latent noise
    xt = torch.randn((batch, z_channels, latent_size, latent_size), device=device)

    T = diffusion_params["num_timesteps"]
    for i in reversed(range(T)):
        t = torch.full((batch,), i, dtype=torch.long, device=device)
        noise_pred_cond = unet(xt, t, cond_input)
        if guidance_scale > 1:
            noise_pred_uncond = unet(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred = noise_pred_cond
        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t)

    with torch.no_grad():
        generated = vae.decode(xt)
    generated = torch.clamp(generated, -1, 1)
    generated = (generated + 1) / 2  # Scale to [0, 1]
    grid = make_grid(generated, nrow=1)
    pil_img = transforms.ToPILImage()(grid.cpu())
    return pil_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument(
        "--text", type=str, required=True, help="Text prompt for conditioning"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to mask image for conditioning (optional)",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_img = sample_ddpm_inference(args.text, args.mask, device=device)
    result_img.save("generated.png")
    print("Generated image saved as generated.png")
