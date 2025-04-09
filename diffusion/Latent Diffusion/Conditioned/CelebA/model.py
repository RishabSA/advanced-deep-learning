import math
import os
import random
import glob
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import Adam
from torchvision.utils import make_grid
from PIL import Image
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    CLIPTokenizer,
    CLIPTextModel,
)

dataset_params = {
    "image_path": "data/CelebAMask-HQ",
    "image_channels": 3,
    "image_size": 256,
    "name": "celebhq",
}

diffusion_params = {
    "num_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
}

ldm_params = {
    "down_channels": [256, 384, 512, 768],
    "mid_channels": [768, 512],
    "down_sample": [True, True, True],
    "attn_down": [True, True, True],  # Attention in the DownBlock and UpBlock of VQ-VAE
    "time_emb_dim": 512,
    "norm_channels": 32,
    "num_heads": 16,
    "conv_out_channels": 128,
    "num_down_layers": 2,
    "num_mid_layers": 2,
    "num_up_layers": 2,
    "condition_config": {
        "condition_types": ["text", "image"],
        "text_condition_config": {
            "text_embed_model": "clip",
            "train_text_embed_model": False,
            "text_embed_dim": 512,  # Each token should map to text_embed_dim sized vector
            "cond_drop_prob": 0.1,  # Probability of dropping conditioning during training to allow the model to generate images without conditioning as well
        },
        "image_condition_config": {
            "image_condition_input_channels": 18,  # CelebA has 18 classes excluding background
            "image_condition_output_channels": 3,
            "image_condition_h": 512,  # Mask height
            "image_condition_w": 512,  # Mask width
            "cond_drop_prob": 0.1,  # Probability of dropping conditioning during training to allow the model to generate images without conditioning as well
        },
    },
}

autoencoder_params = {
    "z_channels": 4,
    "codebook_size": 8192,
    "down_channels": [64, 128, 256, 256],
    "mid_channels": [256, 256],
    "down_sample": [True, True, True],
    "attn_down": [
        False,
        False,
        False,
    ],  # No attention in the DownBlock and UpBlock of VQ-VAE
    "norm_channels": 32,
    "num_heads": 4,
    "num_down_layers": 2,
    "num_mid_layers": 2,
    "num_up_layers": 2,
}

train_params = {
    "seed": 1111,
    "task_name": "celebhq",  # Folder to save models and images to
    "ldm_batch_size": 16,
    "autoencoder_batch_size": 4,
    "disc_start": 15000,
    "disc_weight": 0.5,
    "codebook_weight": 1,
    "commitment_beta": 0.2,
    "perceptual_weight": 1,
    "kl_weight": 0.000005,
    "ldm_epochs": 100,
    "autoencoder_epochs": 20,
    "num_samples": 1,
    "num_grid_rows": 1,
    "ldm_lr": 0.000005,
    "autoencoder_lr": 0.00001,
    "autoencoder_acc_steps": 4,
    "autoencoder_img_save_steps": 64,
    "save_latents": True,
    "cf_guidance_scale": 1.0,
    "vqvae_latent_dir_name": "vqvae_latents",
    "ldm_ckpt_name": "ddpm_ckpt_class_cond.pth",
    "vqvae_autoencoder_ckpt_name": "vqvae_autoencoder_ckpt.pth",
}


def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        # original: (batch_size, c, h, w), t: tensor of timesteps (batch_size,)
        batch_size = original.shape[0]
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].view(
            batch_size, 1, 1, 1
        )
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(
            original.device
        )[t].view(batch_size, 1, 1, 1)
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        batch_size = xt.shape[0]
        alpha_cum_prod_t = self.alpha_cum_prod.to(xt.device)[t].view(
            batch_size, 1, 1, 1
        )
        sqrt_one_minus_alpha_cum_prod_t = self.sqrt_one_minus_alpha_cum_prod.to(
            xt.device
        )[t].view(batch_size, 1, 1, 1)
        x0 = (xt - sqrt_one_minus_alpha_cum_prod_t * noise_pred) / torch.sqrt(
            alpha_cum_prod_t
        )
        x0 = torch.clamp(x0, -1.0, 1.0)
        betas_t = self.betas.to(xt.device)[t].view(batch_size, 1, 1, 1)
        mean = (
            xt - betas_t / sqrt_one_minus_alpha_cum_prod_t * noise_pred
        ) / torch.sqrt(self.alphas.to(xt.device)[t].view(batch_size, 1, 1, 1))
        if t[0] == 0:
            return mean, x0
        else:
            prev_alpha_cum_prod = self.alpha_cum_prod.to(xt.device)[
                (t - 1).clamp(min=0)
            ].view(batch_size, 1, 1, 1)
            variance = (1 - prev_alpha_cum_prod) / (1 - alpha_cum_prod_t) * betas_t
            sigma = variance.sqrt()
            z = torch.randn_like(xt)
            return mean + sigma * z, x0


def get_tokenizer_and_model(model_type, device, eval_mode=True):
    assert model_type in (
        "bert",
        "clip",
    ), "Text model can only be one of 'clip' or 'bert'"
    if model_type == "bert":
        text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        text_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(
            device
        )
    else:
        text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16").to(
            device
        )
    if eval_mode:
        text_model.eval()
    return text_tokenizer, text_model


def get_text_representation(text, text_tokenizer, text_model, device, max_length=77):
    token_output = text_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        max_length=max_length,
    )
    tokens_tensor = torch.tensor(token_output["input_ids"]).to(device)
    mask_tensor = torch.tensor(token_output["attention_mask"]).to(device)
    text_embed = text_model(tokens_tensor, attention_mask=mask_tensor).last_hidden_state
    return text_embed


def get_time_embedding(time_steps, temb_dim):
    """
    Convert time steps tensor into an embedding using the sinusoidal time embedding formula
    time_steps: 1D tensor of length batch size
    temb_dim: Dimension of the embedding
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** (
        (
            torch.arange(
                start=0,
                end=temb_dim // 2,
                dtype=torch.float32,
                device=time_steps.device,
            )
            / (temb_dim // 2)
        )
    )

    t_emb = time_steps.unsqueeze(dim=-1).repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

    return t_emb  # (batch_size, temb_dim)


class DownBlock(nn.Module):
    """
    Down conv block with attention.
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample

    in_channels: Number of channels in the input feature map.
    out_channels: Number of channels produced by this block.
    t_emb_dim: Dimension of the time embedding. Only use for UNet for Diffusion. In an AutoEncoder, set it to None.
    down_sample: Whether to apply downsampling at the end.
    num_heads: Number of attention heads (used if attention is enabled).
    num_layers: How many sub-blocks to apply in sequence.
    attn: Whether to apply self-attention
    norm_channels: Number of groups for GroupNorm.
    cross_attn: Whether to apply cross-attention.
    context_dim: If performing cross-attention, provide a context_dim for extra conditioning context.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        down_sample,
        num_heads,
        num_layers,
        attn,
        norm_channels,
        cross_attn=False,
        context_dim=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),  # Normalizes over channels. For the first sub-block, the in_channels=in_channels, else out_channels
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=(in_channels if i == 0 else out_channels),
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # (batch_size, c, h, w) -> (batch_size, out_channels, h, w)
                )
                for i in range(num_layers)
            ]
        )

        # Only add the time embedding for diffusion and not AutoEncoder
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(
                            in_features=self.t_emb_dim, out_features=out_channels
                        ),  # (batch_size, t_emb_dim) -> (batch_size, out_channels)
                    )
                    for i in range(num_layers)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)
                )
                for i in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )  # (batch_size, in_channels, h, w) -> (batch_size, out_channels, h, w)
                for i in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for i in range(num_layers)]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=out_channels, num_heads=num_heads, batch_first=True
                    )
                    for i in range(num_layers)
                ]
            )

        # Cross attention for text conditioning
        if self.cross_attn:
            assert (
                context_dim is not None
            ), "Context Dimension must be passed for cross attention"

            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for i in range(num_layers)]
            )

            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=out_channels, num_heads=num_heads, batch_first=True
                    )
                    for i in range(num_layers)
                ]
            )

            self.context_proj = nn.ModuleList(
                [
                    nn.Linear(in_features=context_dim, out_features=out_channels)
                    for i in range(num_layers)
                ]
            )

        # Down sample by a factor of 2
        self.down_sample_conv = (
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            if self.down_sample
            else nn.Identity()
        )  # (batch_size, out_channels, h / 2, w / 2)

    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            # Resnet block of UNET
            resnet_input = out  # (batch_size, c, h, w)

            out = self.resnet_conv_first[i](out)  # (batch_size, out_channels, h, w)

            # Only add the time embedding for diffusion and not AutoEncoder
            if self.t_emb_dim is not None:
                # Add the embeddings for timesteps - (batch_size, t_emb_dim) -> (batch_size, out_channels, 1, 1)
                out = out + self.t_emb_layers[i](t_emb).unsqueeze(dim=-1).unsqueeze(
                    dim=-1
                )  # (batch_size, out_channels, h, w)

            out = self.resnet_conv_second[i](
                out
            )  # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)

            # Residual Connection
            out = out + self.residual_input_conv[i](
                resnet_input
            )  # (batch_size, out_channels, h, w)

            # Only do for Diffusion and not for AutoEncoder
            if self.attn:
                # Attention block of UNET
                batch_size, channels, h, w = (
                    out.shape
                )  # (batch_size, out_channels, h, w)

                in_attn = out.reshape(
                    batch_size, channels, h * w
                )  # (batch_size, out_channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)  # (batch_size, h * w, out_channels)

                # Self-Attention
                out_attn, attn_weights = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(
                    batch_size, channels, h, w
                )  # (batch_size, out_channels h, w)

                # Skip connection
                out = out + out_attn  # (batch_size, out_channels h, w)

            if self.cross_attn:
                assert (
                    context is not None
                ), "context cannot be None if cross attention layers are used"

                batch_size, channels, h, w = (
                    out.shape
                )  # (batch_size, out_channels, h, w)

                in_attn = out.reshape(
                    batch_size, channels, h * w
                )  # (batch_size, out_channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)  # (batch_size, h * w, out_channels)

                assert (
                    context.shape[0] == x.shape[0]
                    and context.shape[-1] == self.context_dim
                )  # Make sure the batch_size and context_dim match with the model's parameters
                context_proj = self.context_proj[i](
                    context
                )  # (batch_size, seq_len, context_dim) -> (batch_size, seq_len, out_channels)

                # Cross-Attention
                out_attn, attn_weights = self.cross_attentions[i](
                    in_attn, context_proj, context_proj
                )  # (batch_size, h * w, out_channels)
                out_attn = out_attn.transpose(1, 2).reshape(
                    batch_size, channels, h, w
                )  # (batch_size, out_channels, h, w)

                # Skip Connection
                out = out + out_attn  # (batch_size, out_channels, h, w)

        # Downsampling
        out = self.down_sample_conv(out)  # (batch_size, out_channels, h / 2, w / 2)
        return out


class MidBlock(nn.Module):
    """
    Mid conv block with attention.
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding

    in_channels: Number of channels in the input feature map.
    out_channels: Number of channels produced by this block.
    t_emb_dim: Dimension of the time embedding. Only use for UNet for Diffusion. In an AutoEncoder, set it to None.
    num_heads: Number of attention heads (used if attention is enabled).
    num_layers: How many sub-blocks to apply in sequence.
    norm_channels: Number of groups for GroupNorm.
    cross_attn: Whether to apply cross-attention.
    context_dim: If performing cross-attention, provide a context_dim for extra conditioning context.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        num_heads,
        num_layers,
        norm_channels,
        cross_attn=None,
        context_dim=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),  # Normalizes over channels. For the first sub-block, the in_channels=in_channels, else out_channels
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=(in_channels if i == 0 else out_channels),
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # (batch_size, c, h, w) -> (batch_size, out_channels, h, w)
                )
                for i in range(num_layers + 1)
            ]
        )

        # Only add the time embedding for diffusion and not AutoEncoder
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(
                            in_features=self.t_emb_dim, out_features=out_channels
                        ),  # (batch_size, t_emb_dim) -> (batch_size, out_channels)
                    )
                    for i in range(num_layers + 1)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)
                )
                for i in range(num_layers + 1)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )  # (batch_size, in_channels, h, w) -> (batch_size, out_channels, h, w)
                for i in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels) for i in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=out_channels, num_heads=num_heads, batch_first=True
                )
                for i in range(num_layers)
            ]
        )

        # Cross attention for text conditioning
        if self.cross_attn:
            assert (
                context_dim is not None
            ), "Context Dimension must be passed for cross attention"

            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for i in range(num_layers)]
            )

            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=out_channels, num_heads=num_heads, batch_first=True
                    )
                    for i in range(num_layers)
                ]
            )

            self.context_proj = nn.ModuleList(
                [
                    nn.Linear(in_features=context_dim, out_features=out_channels)
                    for i in range(num_layers)
                ]
            )

    def forward(self, x, t_emb=None, context=None):
        out = x

        # First ResNet block
        resnet_input = out  # (batch_size, c, h, w)
        out = self.resnet_conv_first[0](out)  # (batch_size, out_channels, h, w)

        # Only add the time embedding for diffusion and not AutoEncoder
        if self.t_emb_dim is not None:
            # Add the embeddings for timesteps - (batch_size, t_emb_dim) -> (batch_size, out_channels, 1, 1)
            out = out + self.t_emb_layers[0](t_emb).unsqueeze(dim=-1).unsqueeze(
                dim=-1
            )  # (batch_size, out_channels, h, w)

        out = self.resnet_conv_second[0](
            out
        )  # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)

        # Residual Connection
        out = out + self.residual_input_conv[0](
            resnet_input
        )  # (batch_size, out_channels, h, w)

        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w = out.shape  # (batch_size, out_channels, h, w)

            # Do for both Diffusion and AutoEncoder
            in_attn = out.reshape(
                batch_size, channels, h * w
            )  # (batch_size, out_channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)  # (batch_size, h * w, out_channels)

            # Self-Attention
            out_attn, attn_weights = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)

            # Skip connection
            out = out + out_attn  # (batch_size, out_channels h, w)

            if self.cross_attn:
                assert (
                    context is not None
                ), "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape

                in_attn = out.reshape(
                    batch_size, channels, h * w
                )  # (batch_size, out_channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)  # (batch_size, h * w, out_channels)

                assert (
                    context.shape[0] == x.shape[0]
                    and context.shape[-1] == self.context_dim
                )  # Make sure the batch_size and context_dim match with the model's parameters
                context_proj = self.context_proj[i](
                    context
                )  # (batch_size, seq_len, context_dim) -> (batch_size, seq_len, context_dim)

                # Cross-Attention
                out_attn, attn_weights = self.cross_attentions[i](
                    in_attn, context_proj, context_proj
                )
                out_attn = out_attn.transpose(1, 2).reshape(
                    batch_size, channels, h, w
                )  # (batch_size, out_channels, h, w)

                # Skip Connection
                out = out + out_attn  # (batch_size, out_channels h, w)

            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](
                out
            )  # (batch_size, out_channels h, w) -> (batch_size, out_channels h, w)

            # Only add the time embedding for diffusion and not AutoEncoder
            if self.t_emb_dim is not None:
                # Add the embeddings for timesteps - (batch_size, t_emb_dim) -> (batch_size, out_channels, 1, 1)
                out = out + self.t_emb_layers[i + 1](t_emb).unsqueeze(dim=-1).unsqueeze(
                    dim=-1
                )  # (batch_size, out_channels h, w)

            out = self.resnet_conv_second[i + 1](
                out
            )  # (batch_size, out_channels h, w) -> (batch_size, out_channels h, w)

            # Residual Connection
            out = out + self.residual_input_conv[i + 1](
                resnet_input
            )  # (batch_size, out_channels, h, w)

        return out


class UpBlock(nn.Module):
    """
    Up conv block with attention.
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block

    in_channels: Number of channels in the input feature map.
    out_channels: Number of channels produced by this block.
    t_emb_dim: Dimension of the time embedding. Only use for UNet for Diffusion. In an AutoEncoder, set it to None.
    up_sample: Whether to apply upsampling at the end.
    num_heads: Number of attention heads (used if attention is enabled).
    num_layers: How many sub-blocks to apply in sequence.
    attn: Whether to apply self-attention
    norm_channels: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        up_sample,
        num_heads,
        num_layers,
        attn,
        norm_channels,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn

        # Upsample by a factor of 2
        self.up_sample_conv = (
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            if self.up_sample
            else nn.Identity()
        )  # (batch_size, c, h * 2, w * 2)

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),  # Normalizes over channels. For the first sub-block, the in_channels=in_channels, else out_channels
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=(in_channels if i == 0 else out_channels),
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # (batch_size, c, h, w) -> (batch_size, out_channels, h, w)
                )
                for i in range(num_layers)
            ]
        )

        # Only add the time embedding for diffusion and not AutoEncoder
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(
                            in_features=self.t_emb_dim, out_features=out_channels
                        ),  # (batch_size, t_emb_dim) -> (batch_size, out_channels)
                    )
                    for i in range(num_layers)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)
                )
                for i in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )  # (batch_size, in_channels, h, w) -> (batch_size, out_channels, h, w)
                for i in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for i in range(num_layers)]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=out_channels, num_heads=num_heads, batch_first=True
                    )
                    for i in range(num_layers)
                ]
            )

    def forward(self, x, out_down=None, t_emb=None):
        # x shape: (batch_size, c, h, w)

        # Upsample
        x = self.up_sample_conv(
            x
        )  # (batch_size, c, h, w) -> (batch_size, c, h * 2, w * 2)

        # *Only do for diffusion
        # Concatenate with the output of respective DownBlock
        if out_down is not None:
            x = torch.cat(
                [x, out_down], dim=1
            )  # (batch_size, c, h * 2, w * 2) -> (batch_size, c * 2, h * 2, w * 2)

        out = x  # (batch_size, c, h * 2, w * 2)

        for i in range(self.num_layers):
            # Resnet block
            resnet_input = out
            out = self.resnet_conv_first[i](
                out
            )  # (batch_size, in_channels, h * 2, w * 2) -> (batch_size, out_channels, h * 2, w * 2)

            # Only add the time embedding for diffusion and not AutoEncoder
            if self.t_emb_dim is not None:
                # Add the embeddings for timesteps - (batch_size, t_emb_dim) -> (batch_size, out_channels, 1, 1)
                out = out + self.t_emb_layers[i](t_emb).unsqueeze(dim=-1).unsqueeze(
                    dim=-1
                )  # (batch_size, out_channels, h * 2, w * 2)

            out = self.resnet_conv_second[i](
                out
            )  # (batch_size, out_channels, h * 2, w * 2) -> (batch_size, out_channels, h * 2, w * 2)

            # Residual Connection
            out = out + self.residual_input_conv[i](
                resnet_input
            )  # (batch_size, out_channels, h * 2, w * 2)

            # Only do for Diffusion and not for AutoEncoder
            if self.attn:
                # Attention block of UNET
                batch_size, channels, h, w = out.shape

                in_attn = out.reshape(
                    batch_size, channels, h * w
                )  # (batch_size, out_channels, h * w * 4)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(
                    1, 2
                )  # (batch_size, h * w * 4, out_channels)

                # Self-Attention
                out_attn, attn_weights = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(
                    batch_size, channels, h, w
                )  # (batch_size, out_channels h * 2, w * 2)

                # Skip connection
                out = out + out_attn  # (batch_size, out_channels h * 2, w * 2)

        return out  # (batch_size, out_channels h * 2, w * 2)


class UpBlockUNet(nn.Module):
    """
    Up conv block with attention.
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block

    in_channels: Number of channels in the input feature map. (It is passed in multiplied by 2 for concatenation with DownBlock output)
    out_channels: Number of channels produced by this block.
    t_emb_dim: Dimension of the time embedding. Only use for UNet for Diffusion. In an AutoEncoder, set it to None.
    up_sample: Whether to apply upsampling at the end.
    num_heads: Number of attention heads (used if attention is enabled).
    num_layers: How many sub-blocks to apply in sequence.
    norm_channels: Number of groups for GroupNorm.
    cross_attn: Whether to apply cross-attention.
    context_dim: If performing cross-attention, provide a context_dim for extra conditioning context.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        up_sample,
        num_heads,
        num_layers,
        norm_channels,
        cross_attn=False,
        context_dim=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim

        self.up_sample_conv = (
            nn.ConvTranspose2d(
                in_channels=(in_channels // 2),
                out_channels=(in_channels // 2),
                kernel_size=4,
                stride=2,
                padding=1,
            )
            if self.up_sample
            else nn.Identity()
        )  # (batch_size, in_channels // 2, h * 2, w * 2)

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),  # Normalizes over channels. For the first sub-block, the in_channels=in_channels, else out_channels
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=(in_channels if i == 0 else out_channels),
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # (batch_size, in_channels, h * 2, w. * 2) -> (batch_size, out_channels, h * 2, w * 2) - Starts at in_channels and not in_channels // 2 because of concatenation
                )
                for i in range(num_layers)
            ]
        )

        # Only add the time embedding if needed for UNET in diffusion
        # Do not add the time embedding in the AutoEncoder
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(
                            in_features=self.t_emb_dim, out_features=out_channels
                        ),  # (batch_size, t_emb_dim) -> (batch_size, out_channels)
                    )
                    for i in range(num_layers)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # (batch_size, out_channels, h * 2, w * 2) -> (batch_size, out_channels, h * 2, w * 2)
                )
                for i in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for i in range(
                    num_layers
                )  # (batch_size, in_channels, h * 2, w * 2) -> (batch_size, out_channels, h * 2, w * 2)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels) for i in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=out_channels, num_heads=num_heads, batch_first=True
                )
                for i in range(num_layers)
            ]
        )

        # Cross attention for text conditioning
        if self.cross_attn:
            assert (
                context_dim is not None
            ), "Context Dimension must be passed for cross attention"

            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for i in range(num_layers)]
            )

            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=out_channels, num_heads=num_heads, batch_first=True
                    )
                    for i in range(num_layers)
                ]
            )

            self.context_proj = nn.ModuleList(
                [
                    nn.Linear(in_features=context_dim, out_features=out_channels)
                    for i in range(num_layers)
                ]
            )

    def forward(self, x, out_down=None, t_emb=None, context=None):
        # x shape: (batch_size, in_channels // 2, h, w)

        # Upsample
        x = self.up_sample_conv(
            x
        )  # (batch_size, in_channels // 2, h, w) -> (batch_size, in_channels // 2, h * 2, w * 2)

        # Concatenate with the output of respective DownBlock
        if out_down is not None:
            x = torch.cat(
                [x, out_down], dim=1
            )  # (batch_size, in_channels // 2, h * 2, w * 2) -> (batch_size, in_channels, h * 2, w * 2)

        out = x  # (batch_size, in_channels, h * 2, w * 2)
        for i in range(self.num_layers):
            # Resnet block
            resnet_input = out

            out = self.resnet_conv_first[i](
                out
            )  # (batch_size, in_channels, h * 2, w * 2) -> (batch_size, out_channels, h * 2, w * 2)

            if self.t_emb_dim is not None:
                # Add the embeddings for timesteps - (batch_size, t_emb_dim) -> (batch_size, out_channels, 1, 1)
                out = out + self.t_emb_layers[i](t_emb).unsqueeze(dim=-1).unsqueeze(
                    dim=-1
                )  # (batch_size, out_channels, h * 2, w * 2)

            out = self.resnet_conv_second[i](
                out
            )  # (batch_size, out_channels, h * 2, w * 2) -> (batch_size, out_channels, h * 2, w * 2)

            # Residual Connection
            out = out + self.residual_input_conv[i](
                resnet_input
            )  # (batch_size, out_channels, h * 2, w * 2)

            # Attention block of UNET
            batch_size, channels, h, w = (
                out.shape
            )  # (batch_size, out_channels, h * 2, w * 2)

            in_attn = out.reshape(
                batch_size, channels, h * w
            )  # (batch_size, out_channels, h * w * 4)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)  # (batch_size, h * w * 4, out_channels)

            # Self-Attention
            out_attn, attn_weights = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(
                batch_size, channels, h, w
            )  # (batch_size, out_channels h * 2, w * 2)

            # Skip connection
            out = out + out_attn  # (batch_size, out_channels h * 2, w * 2)

            if self.cross_attn:
                assert (
                    context is not None
                ), "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape

                in_attn = out.reshape(
                    batch_size, channels, h * w
                )  # (batch_size, out_channels, h * w * 4)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(
                    1, 2
                )  # (batch_size, h * w * 4, out_channels)

                assert (
                    len(context.shape) == 3
                ), "Context shape does not match batch_size, _, context_dim"

                assert (
                    context.shape[0] == x.shape[0]
                    and context.shape[-1] == self.context_dim
                ), "Context shape does not match batch_size, _, context_dim"  # Make sure the batch_size and context_dim match with the model's parameters
                context_proj = self.context_proj[i](
                    context
                )  # (batch_size, seq_len, context_dim) -> (batch_size, seq_len, context_dim)

                # Cross-Attention
                out_attn, attn_weights = self.cross_attentions[i](
                    in_attn, context_proj, context_proj
                )
                out_attn = out_attn.transpose(1, 2).reshape(
                    batch_size, channels, h, w
                )  # (batch_size, out_channels, h * 2, w * 2)

                # Skip Connection
                out = out + out_attn  # (batch_size, out_channels h * 2, w * 2)

        return out  # (batch_size, out_channels h * 2, w * 2)


class VQVAE(nn.Module):
    def __init__(self, image_channels, model_config):
        super().__init__()

        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.down_sample = model_config["down_sample"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]

        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config["attn_down"]

        # Latent Dimension
        self.z_channels = model_config[
            "z_channels"
        ]  # number of channels in the latent representation
        self.codebook_size = model_config[
            "codebook_size"
        ]  # number of discrete code vectors available
        self.norm_channels = model_config["norm_channels"]
        self.num_heads = model_config["num_heads"]

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Wherever we downsample in the encoder, use upsampling in the decoder at the corresponding location
        self.up_sample = list(reversed(self.down_sample))

        # Encoder
        self.encoder_conv_in = nn.Conv2d(
            in_channels=image_channels,
            out_channels=self.down_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )  # (batch_size, 3, h, w) -> (batch_size, c, h, w)

        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(
                DownBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    t_emb_dim=None,
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                    attn=self.attns[i],
                    norm_channels=self.norm_channels,
                )
            )

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i + 1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                )
            )

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])

        self.encoder_conv_out = nn.Conv2d(
            in_channels=self.down_channels[-1],
            out_channels=self.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # (batch_size, z_channels, h', w')

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(
            in_channels=self.z_channels,
            out_channels=self.z_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )  # (batch_size, z_channels, h', w')

        # Codebook Vectors
        self.embedding = nn.Embedding(
            self.codebook_size, self.z_channels
        )  # (codebook_size, z_channels)

        # Decoder

        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(
            in_channels=self.z_channels,
            out_channels=self.z_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )  # (batch_size, z_channels, h', w')

        self.decoder_conv_in = nn.Conv2d(
            in_channels=self.z_channels,
            out_channels=self.mid_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )  # (batch_size, c, h', w')

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i - 1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                )
            )

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(
                UpBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i - 1],
                    t_emb_dim=None,
                    up_sample=self.down_sample[i - 1],
                    num_heads=self.num_heads,
                    num_layers=self.num_up_layers,
                    attn=self.attns[i - 1],
                    norm_channels=self.norm_channels,
                )
            )

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])

        self.decoder_conv_out = nn.Conv2d(
            in_channels=self.down_channels[0],
            out_channels=image_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # (batch_size, c, h, w)

    def quantize(self, x):
        batch_size, c, h, w = x.shape  # (batch_size, z_channels, h, w)

        x = x.permute(
            0, 2, 3, 1
        )  # (batch_size, z_channels, h, w) -> (batch_size, h, w, z_channels)
        x = x.reshape(
            batch_size, -1, c
        )  # (batch_size, h, w, z_channels) -> (batch_size, h * w, z_channels)

        # Find the nearest codebook vector with distance between (batch_size, h * w, z_channels) and (batch_size, code_book_size, z_channels) -> (batch_size, h * w, code_book_size)
        dist = torch.cdist(
            x, self.embedding.weight.unsqueeze(dim=0).repeat((batch_size, 1, 1))
        )  # cdist calculates the batched p-norm distance

        # (batch_size, h * w) Get the index of the closet codebook vector
        min_encoding_indices = torch.argmin(dist, dim=-1)

        # Replace the encoder output with the nearest codebook
        quant_out = torch.index_select(
            self.embedding.weight, 0, min_encoding_indices.view(-1)
        )  # (batch_size, h * w, z_channels)

        x = x.reshape((-1, c))  # (batch_size * h * w, z_channels)

        # Commitment and Codebook Loss using mSE
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)

        quantize_losses = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        quant_out = quant_out.reshape(batch_size, h, w, c).permute(
            0, 3, 1, 2
        )  # (batch_size, z_channels, h, w)
        min_encoding_indices = min_encoding_indices.reshape(
            (-1, h, w)
        )  # (batch_size, h, w)

        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x):
        out = self.encoder_conv_in(x)  # (batch_size, self.down_channels[0], h, w)

        # (batch_size, self.down_channels[0], h, w) -> (batch_size, self.down_channels[-1], h', w')
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)

        # (batch_size, self.down_channels[-1], h', w') -> (batch_size, self.mid_channels[-1], h', w')
        for mid in self.encoder_mids:
            out = mid(out)

        out = self.encoder_norm_out(out)
        out = F.silu(out)

        out = self.encoder_conv_out(
            out
        )  # (batch_size, self.mid_channels[-1], h', w') -> (batch_size, self.z_channels, h', w')
        out = self.pre_quant_conv(
            out
        )  # (batch_size, self.z_channels, h', w') -> (batch_size, self.z_channels, h', w')

        out, quant_losses, min_encoding_indices = self.quantize(
            out
        )  # (batch_size, self.z_channels, h', w'), (codebook_loss, commitment_loss), (batch_size, h, w)
        return out, quant_losses

    def decode(self, z):
        out = z
        out = self.post_quant_conv(
            out
        )  # (batch_size, self.z_channels, h', w') -> (batch_size, self.z_channels, h', w')
        out = self.decoder_conv_in(
            out
        )  # (batch_size, self.z_channels, h', w') -> (batch_size, self.mid_channels[-1], h', w')

        # (batch_size, self.mid_channels[-1], h', w') -> (batch_size, self.down_channels[-1], h', w')
        for mid in self.decoder_mids:
            out = mid(out)

        # (batch_size, self.down_channels[-1], h', w') -> (batch_size, self.down_channels[0], h, w)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = F.silu(out)

        out = self.decoder_conv_out(
            out
        )  # (batch_size, self.down_channels[0], h, w) -> (batch_size, c, h, w)
        return out

    def forward(self, x):
        # x shape: (batch_size, c, h, w)

        z, quant_losses = self.encode(
            x
        )  # (batch_size, self.z_channels, h', w'), (codebook_loss, commitment_loss)
        out = self.decode(z)  # (batch_size, c, h, w)

        return out, z, quant_losses


def validate_image_conditional_input(cond_input, x):
    assert (
        "image" in cond_input
    ), "Model initialized with image conditioning but cond_input has no image information"
    assert (
        cond_input["image"].shape[0] == x.shape[0]
    ), "Batch size mismatch of image condition and input"
    assert (
        cond_input["image"].shape[2] % x.shape[2] == 0
    ), "Height/Width of image condition must be divisible by latent input"


def validate_class_conditional_input(cond_input, x, num_classes):
    assert (
        "class" in cond_input
    ), "Model initialized with class conditioning but cond_input has no class information"
    assert cond_input["class"].shape == (
        x.shape[0],
        num_classes,
    ), "Shape of class condition input must match (Batch Size, )"


def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value


class UNet(nn.Module):
    """
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """

    def __init__(self, image_channels, model_config):
        super().__init__()

        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.t_emb_dim = model_config["time_emb_dim"]
        self.down_sample = model_config["down_sample"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]
        self.attns = model_config["attn_down"]
        self.norm_channels = model_config["norm_channels"]
        self.num_heads = model_config["num_heads"]
        self.conv_out_channels = model_config["conv_out_channels"]

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Class, Mask, and Text Conditioning Config
        self.class_cond = False
        self.text_cond = False
        self.image_cond = False
        self.text_embed_dim = None
        self.condition_config = get_config_value(
            model_config, "condition_config", None
        )  # Get the dictionary containing conditional information

        if self.condition_config is not None:
            assert (
                "condition_types" in self.condition_config
            ), "Condition Type not provided in model config"
            condition_types = self.condition_config["condition_types"]

            # For class, text, and image, get necessary parameters
            if "class" in condition_types:
                self.class_cond = True
                self.num_classes = self.condition_config["class_condition_config"][
                    "num_classes"
                ]

            if "text" in condition_types:
                self.text_cond = True
                self.text_embed_dim = self.condition_config["text_condition_config"][
                    "text_embed_dim"
                ]

            if "image" in condition_types:
                self.image_cond = True
                self.image_cond_input_channels = self.condition_config[
                    "image_condition_config"
                ]["image_condition_input_channels"]
                self.image_cond_output_channels = self.condition_config[
                    "image_condition_config"
                ]["image_condition_output_channels"]

        if self.class_cond:
            # For class conditioning, do not add the class embedding information for unconditional generation
            self.class_emb = nn.Embedding(
                self.num_classes, self.t_emb_dim
            )  # (num_classes, t_emb_dim)

        if self.image_cond:
            # Map the mask image to a image_cond_output_channels channel image, and concat with input across the channel dimension
            self.cond_conv_in = nn.Conv2d(
                in_channels=self.image_cond_input_channels,
                out_channels=self.image_cond_output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

            self.conv_in_concat = nn.Conv2d(
                in_channels=(image_channels + self.image_cond_output_channels),
                out_channels=self.down_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_in = nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.down_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )  # (batch_size, image_channels, h, w) -> (batch_size, self.down_channels[0], h, w)

        self.cond = self.text_cond or self.image_cond or self.class_cond

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(in_features=self.t_emb_dim, out_features=self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(in_features=self.t_emb_dim, out_features=self.t_emb_dim),
        )  # (batch_size, t_emb_dim)

        self.up_sample = list(reversed(self.down_sample))

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            # Cross attention and Context Dim are only used for text conditioning
            self.downs.append(
                DownBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    t_emb_dim=self.t_emb_dim,
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                    attn=self.attns[i],
                    norm_channels=self.norm_channels,
                    cross_attn=self.text_cond,
                    context_dim=self.text_embed_dim,
                )
            )

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            # Cross attention and Context Dim are only used for text conditioning
            self.mids.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i + 1],
                    t_emb_dim=self.t_emb_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                    cross_attn=self.text_cond,
                    context_dim=self.text_embed_dim,
                )
            )

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            # Cross attention and Context Dim are only used for text conditioning
            self.ups.append(
                UpBlockUNet(
                    in_channels=(self.down_channels[i] * 2),
                    out_channels=(
                        self.down_channels[i - 1] if i != 0 else self.conv_out_channels
                    ),
                    t_emb_dim=self.t_emb_dim,
                    up_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_up_layers,
                    norm_channels=self.norm_channels,
                    cross_attn=self.text_cond,
                    context_dim=self.text_embed_dim,
                )
            )

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)

        self.conv_out = nn.Conv2d(
            in_channels=self.conv_out_channels,
            out_channels=image_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # (batch_size, conv_out_channels, h, w) -> (batch_size, image_channels, h, w)

    def forward(self, x, t, cond_input=None):
        # x shape: (batch_size, c, h, w)
        # cond_input is the conditioning vector
        # For class conditioning, it will be a one-hot vector of size # (batch_size, num_classes)

        if self.cond:
            assert (
                cond_input is not None
            ), "Model initialized with conditioning so cond_input cannot be None"

        if self.image_cond:
            # Mask Conditioning
            validate_image_conditional_input(cond_input, x)
            image_cond = cond_input["image"]
            image_cond = F.interpolate(image_cond, size=x.shape[-2:])
            image_cond = self.cond_conv_in(image_cond)
            assert image_cond.shape[-2:] == x.shape[-2:]

            x = torch.cat(
                [x, image_cond], dim=1
            )  # (batch_size, image_channels + image_cond_output_channels, h, w)
            out = self.conv_in_concat(x)  # (batch_size, down_channels[0], h, w)
        else:
            out = self.conv_in(x)  # (batch_size, down_channels[0], h, w)

        t_emb = get_time_embedding(
            torch.as_tensor(t).long(), self.t_emb_dim
        )  # (batch_size, t_emb_dim)
        t_emb = self.t_proj(t_emb)  # (batch_size, t_emb_dim)

        # Class Conditioning
        if self.class_cond:
            validate_class_conditional_input(cond_input, x, self.num_classes)

            # Take the matrix for class embedding vectors and matrix multiply it with the embedding matrix to get the class embedding for all images in a batch
            class_embed = torch.matmul(
                cond_input["class"].float(), self.class_emb.weight
            )  # (batch_size, t_emb_dim)
            t_emb += class_embed  # Add the class embedding to the time embedding

        context_hidden_states = None

        # Only use context hidden states in cross-attention for text conditioning
        if self.text_cond:
            assert (
                "text" in cond_input
            ), "Model initialized with text conditioning but cond_input has no text information"
            context_hidden_states = cond_input["text"]

        down_outs = []
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(
                out, t_emb, context_hidden_states
            )  # Use context_hidden_states for cross-attention
        # out = (batch_size, c4, h / 4, w / 4)

        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states)
        # out = (batch_size, c3, h / 4, w / 4)

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)
        # out = (batch_size, self.conv_out_channels, h, w)

        out = F.silu(self.norm_out(out))
        out = self.conv_out(
            out
        )  # (batch_size, self.conv_out_channels, h, w) -> (batch_size, image_channels, h, w)

        return out  # (batch_size, image_channels, h, w)


def sample_ddpm_inference(
    unet,
    vae,
    text_prompt,
    mask_image_pil=None,
    guidance_scale=1.0,
    device=torch.device("cpu"),
):
    """
    Given a text prompt and (optionally) an image condition (as a PIL image),
    sample from the diffusion model and return a generated image (PIL image).
    """
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

    # Load the diffusion UNet (and assume it has been pretrained and saved)
    # unet = UNet(
    #     image_channels=autoencoder_params["z_channels"], model_config=ldm_params
    # ).to(device)
    # ldm_checkpoint_path = os.path.join(
    #     train_params["task_name"], train_params["ldm_ckpt_name"]
    # )
    # if os.path.exists(ldm_checkpoint_path):
    #     checkpoint = torch.load(ldm_checkpoint_path, map_location=device)
    #     unet.load_state_dict(checkpoint["model_state_dict"])
    # unet.eval()

    # Load VQVAE (assume pretrained and saved)
    # vae = VQVAE(
    #     image_channels=dataset_params["image_channels"], model_config=autoencoder_params
    # ).to(device)
    # vae_checkpoint_path = os.path.join(
    #     train_params["task_name"], train_params["vqvae_autoencoder_ckpt_name"]
    # )
    # if os.path.exists(vae_checkpoint_path):
    #     checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    #     vae.load_state_dict(checkpoint["model_state_dict"])
    # vae.eval()

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

        with torch.no_grad():
            generated = vae.decode(xt)

        generated = torch.clamp(generated, -1, 1)
        generated = (generated + 1) / 2  # scale to [0,1]
        grid = make_grid(generated, nrow=1)
        pil_img = transforms.ToPILImage()(grid.cpu())

        if i % 10 == 0:
            yield pil_img
