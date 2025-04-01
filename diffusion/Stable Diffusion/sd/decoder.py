import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(n_heads=1, d_model=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, h, w)
        residual = x

        batch_size, c, h, w = x.shape

        # (batch_size, channels, h, w) -> (batch_size, channels, h * w)
        x = x.view(batch_size, c, h * w)

        # (batch_size, channels, h * w) -> (batch_size, h * w, channels)
        x = x.transpose(1, 2)

        x = self.attention(x)

        # (batch_size, h * w, channels) -> (batch_size, channels, h * w)
        x = x.transpose(1, 2)

        # (batch_size, channels, h * w) -> (batch_size, channels, h, w)
        x = x.view(batch_size, c, h, w)

        x += residual

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.residual_layer = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, in_channels, h, w)
        residual = x

        x = F.silu(self.group_norm_1(x))
        x = self.conv_1(x)

        x = F.silu(self.group_norm_2(x))
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(
                in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0
            ),  # (batch_size, 4, h / 8, w / 8) -> (batch_size, 4, h / 8, w / 8)
            nn.Conv2d(
                in_channels=4, out_channels=512, kernel_size=3, stride=1, padding=1
            ),  # (batch_size, 4, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_AttentionBlock(
                channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            nn.Upsample(
                scale_factor=2
            ),  # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 4, w / 4)
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),  # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 4, w / 4)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 4, w / 4)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 4, w / 4)
            VAE_ResidualBlock(
                in_channels=512, out_channels=512
            ),  # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 4, w / 4)
            nn.Upsample(
                scale_factor=2
            ),  # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 2, w / 2)
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),  # (batch_size, 512, h / 2, w / 2) -> (batch_size, 512, h / 2, w / 2)
            VAE_ResidualBlock(
                in_channels=512, out_channels=256
            ),  # (batch_size, 512, h / 2, w / 2) -> (batch_size, 256, h / 2, w / 2)
            VAE_ResidualBlock(
                in_channels=256, out_channels=256
            ),  # (batch_size, 256, h / 2, w / 2) -> (batch_size, 256, h / 2, w / 2)
            VAE_ResidualBlock(
                in_channels=256, out_channels=256
            ),  # (batch_size, 256, h / 2, w / 2) -> (batch_size, 256, h / 2, w / 2)
            nn.Upsample(
                scale_factor=2
            ),  # (batch_size, 256, h / 2, w / 2) -> (batch_size, 256, h, w)
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (batch_size, 256, h, w) -> (batch_size, 256, h, w)
            VAE_ResidualBlock(
                in_channels=256, out_channels=128
            ),  # (batch_size, 256, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(
                in_channels=128, out_channels=128
            ),  # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(
                in_channels=128, out_channels=128
            ),  # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1
            ),  # (batch_size, 128, h, w) -> (batch_size, 3, h, w)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch_size, 4, h / 8, w / 8)
        x /= 0.18215

        for layer in self:
            x = layer(x)

        # (batch_size, 3, h, w)
        return x
