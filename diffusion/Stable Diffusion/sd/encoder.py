import torch
from torch import nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # (batch_size, 3, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(
                in_channels=128, out_channels=128
            ),  # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(
                in_channels=128, out_channels=128
            ),  # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0
            ),  # (batch_size, 128, h, w) -> (batch_size, 128, h / 2, w / 2)
            VAE_ResidualBlock(
                in_channels=128, out_channels=256
            ),  # (batch_size, 128, h / 2, w / 2) -> (batch_size, 256, h / 2, w / 2)
            VAE_ResidualBlock(
                in_channels=256, out_channels=256
            ),  # (batch_size, 256, h / 2, w / 2) -> (batch_size, 256, h / 2, w / 2)
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0
            ),  # (batch_size, 256, h / 2, w / 2) -> (batch_size, 256, h / 4, w / 4)
            VAE_ResidualBlock(
                in_channels=256, out_channels=512
            ),  # (batch_size, 256, h / 4, w / 4) -> (batch_size, 512, h / 4, w / 4)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 4, w / 4)
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0
            ),  # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_AttentionBlock(
                channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            nn.GroupNorm(
                num_groups=32, num_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            nn.SiLU(),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            # Bottleneck
            nn.Conv2d(
                in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 8, h / 8, w / 8)
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0
            ),  # (batch_size, 8, h / 8, w / 8) -> (batch_size, 8, h / 8, w / 8)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 3, h, w)
        # noise (batch_size, out_channels, h / 8, w / 8)

        for layer in self:
            if getattr(layer, "stride", None) == (
                2,
                2,
            ):  # Padding at downsampling should be asymmetric
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = layer(x)

        # Chunk splits the tensor in half across the channel dimensions
        # (batch_size, 8, h / 8, w / 8) -> 2 tensors of shape (batch_size, 4, h / 8, w / 8) and (batch_size, 4, h / 8, w / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # z = N(0, 1) -> x = N(mean, variance)
        # Reparameterization Trick: x = mean + stdev * z
        x = mean + stdev * noise

        # scale the output by a constant
        x *= 0.18215

        return x
