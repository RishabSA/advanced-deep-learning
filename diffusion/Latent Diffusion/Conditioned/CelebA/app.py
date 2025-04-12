import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import gradio as gr
from model import (
    UNet,
    VQVAE,
    LinearNoiseScheduler,
    get_tokenizer_and_model,
    get_text_representation,
    dataset_params,
    diffusion_params,
    ldm_params,
    autoencoder_params,
    train_params,
)
from huggingface_hub import hf_hub_download
import spaces
import json


print("Gradio version:", gr.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Download config and checkpoint files from HF Hub
config_path = hf_hub_download(
    repo_id="RishabA/celeba-cond-ddpm", filename="config.json"
)
with open(config_path, "r") as f:
    config = json.load(f)

ldm_ckpt_path = hf_hub_download(
    repo_id="RishabA/celeba-cond-ddpm", filename="celebhq/ddpm_ckpt_class_cond.pth"
)
vae_ckpt_path = hf_hub_download(
    repo_id="RishabA/celeba-cond-ddpm", filename="celebhq/vqvae_autoencoder_ckpt.pth"
)

# Instantiate and load the models
unet = UNet(config["autoencoder_params"]["z_channels"], config["ldm_params"]).to(device)
vae = VQVAE(
    config["dataset_params"]["image_channels"], config["autoencoder_params"]
).to(device)

unet_state = torch.load(ldm_ckpt_path, map_location=device)
unet.load_state_dict(unet_state["model_state_dict"])
print(unet_state["epoch"])

vae_state = torch.load(vae_ckpt_path, map_location=device)
vae.load_state_dict(vae_state["model_state_dict"])

unet.eval()
vae.eval()

print("Model and checkpoints loaded successfully!")


@spaces.GPU
def sample_ddpm_inference(text_prompt):
    """
    Given a text prompt and (optionally) an image condition (as a PIL image),
    sample from the diffusion model and return a generated image (PIL image).
    """

    mask_image_pil = None
    guidance_scale = 2.0
    image_display_rate = 1

    # Create noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_params["num_timesteps"],
        beta_start=diffusion_params["beta_start"],
        beta_end=diffusion_params["beta_end"],
    )
    # Get conditioning config from ldm_params
    condition_config = ldm_params.get("condition_config", None)
    condition_types = (
        condition_config.get("condition_types", [])
        if condition_config is not None
        else []
    )

    # Load text tokenizer/model for conditioning
    text_model_type = condition_config["text_condition_config"]["text_embed_model"]
    text_tokenizer, text_model = get_tokenizer_and_model(text_model_type, device=device)

    # Get empty text representation for classifier-free guidance
    empty_text_embed = get_text_representation([""], text_tokenizer, text_model, device)

    # Get text representation of the input prompt
    text_prompt_embed = get_text_representation(
        [text_prompt], text_tokenizer, text_model, device
    )

    # Prepare image conditioning:
    # If the user uploaded a mask image (should be a PIL image), convert it; otherwise, use zeros.
    if "image" in condition_types:
        if mask_image_pil is not None:
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
            mask_tensor = (
                mask_transform(mask_image_pil).unsqueeze(0).to(device)
            )  # (1, channels, H, W)
        else:
            # Create a zero mask with the required number of channels (e.g. 18)
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

    # Build conditioning dictionaries for classifier-free guidance:
    # For unconditional, we use empty text and zero mask.
    uncond_input = {}
    cond_input = {}
    if "text" in condition_types:
        uncond_input["text"] = empty_text_embed
        cond_input["text"] = text_prompt_embed
    if "image" in condition_types:
        # Use zeros for unconditioning, and the provided mask for conditioning.
        uncond_input["image"] = torch.zeros_like(mask_tensor)
        cond_input["image"] = mask_tensor

    # Determine latent shape from VQVAE: (batch, z_channels, H_lat, W_lat)
    # For example, if image_size is 256 and there are 3 downsamplings, H_lat = 256 // 8 = 32.
    latent_size = dataset_params["image_size"] // (
        2 ** sum(autoencoder_params["down_sample"])
    )
    batch = train_params["num_samples"]
    z_channels = autoencoder_params["z_channels"]

    # Sample initial latent noise
    xt = torch.randn((batch, z_channels, latent_size, latent_size), device=device)

    # Sampling loop (reverse diffusion)
    T = diffusion_params["num_timesteps"]
    for i in reversed(range(T)):
        t = torch.full((batch,), i, dtype=torch.long, device=device)

        with torch.no_grad():
            # Get conditional noise prediction
            noise_pred_cond = unet(xt, t, cond_input)
            if guidance_scale > 1:
                noise_pred_uncond = unet(xt, t, uncond_input)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t)

            if i % image_display_rate == 0 or i == 0:
                # Decode current latent into image
                generated = vae.decode(xt)

                generated = torch.clamp(generated, -1, 1)
                generated = (generated + 1) / 2  # scale to [0,1]
                grid = make_grid(generated, nrow=1)
                pil_img = transforms.ToPILImage()(grid.cpu())

                yield pil_img


css_str = """
.title { 
    font-size: 48px; 
    text-align: center; 
    margin-top: 20px; 
}
.description { 
    font-size: 20px; 
    text-align: center; 
    margin-bottom: 40px; 
}
"""

with gr.Blocks(css=css_str) as demo:
    gr.Markdown("<div class='title'>Conditioned Latent Diffusion with CelebA</div>")
    gr.Markdown(
        "<div class='description'>Enter a text prompt and (optionally) upload a mask image for conditioning; the generated image will update as the reverse diffusion progresses.</div>"
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Text Prompt",
            lines=2,
            placeholder="E.g., 'He is a man with brown hair.'",
        )

    generate_button = gr.Button("Generate Image")
    output_image = gr.Image(label="Generated Image", type="pil")

    generate_button.click(
        fn=sample_ddpm_inference,
        inputs=[text_input],
        outputs=[output_image],
    )

if __name__ == "__main__":
    demo.launch(share=True)
