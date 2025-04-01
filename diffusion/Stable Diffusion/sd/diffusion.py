import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=d_model, out_features=(4 * d_model))
        self.linear_2 = nn.Linear(in_features=(4 * d_model), out_features=(4 * d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x  # (batch_size, 1, 1280)


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.group_norm_feature = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.linear_time = nn.Linear(in_features=n_time, out_features=out_channels)

        self.group_norm_merged = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_merged = nn.Conv2d(
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

    def forward(self, feature, time):
        # feature shape: (batch_size, in_channels, h, w)
        # time shape: (1, 1280)
        residual = feature

        feature = F.silu(self.group_norm_feature(feature))
        feature = self.conv_feature(
            feature
        )  # (batch_size, in_channels, h, w) -> (batch_size, out_channels, h, w)

        time = (
            self.linear_time(F.silu(time)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        )  # (1, 1280) -> (1, out_channels, 1, 1)
        merged = feature + time  # (batch_size, out_channels, h, w)
        merged = F.silu(self.group_norm_merged(merged))
        merged = self.conv_merged(
            merged
        )  # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)

        return merged + self.residual_layer(
            residual
        )  # (batch_size, out_channels, h, w)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_context=768):
        super().__init__()
        channels = n_heads * d_model

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv_input = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(
            n_heads=n_heads, d_model=channels, in_proj_bias=False
        )

        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_heads=n_heads, d_model=channels, d_cross=d_context, in_proj_bias=False
        )

        self.layer_norm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(
            in_features=channels, out_features=(4 * channels * 2)
        )
        self.linear_geglu_2 = nn.Linear(
            in_features=(4 * channels), out_features=channels
        )

        self.conv_output = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, context):
        # x shape: (batch_size, features, h, w)
        # context shape: (batch_size, seq_len, d_model)
        residual_long = x

        x = self.group_norm(x)
        x = self.conv_input(x)

        batch_size, c, h, w = x.shape

        # (batch_size, features, h, w) -> (batch_size, features, h * w)
        x = x.view(batch_size, c, h * w)

        # (batch_size, features, h * w) -> (batch_size, h * w, features)
        x = x.transpose(1, 2)

        # Normalization + Self Attention with Skip Connection
        residual_short = x
        x = self.layer_norm_1(x)
        x = self.attention_1(x)
        x += residual_short

        # Normalization + Cross Attention with Skip Connection
        residual_short = x
        x = self.layer_norm_2(x)
        x = self.attention_2(x, context)
        x += residual_short

        # Normalization + FFN with GeGLU and Skip Connection
        residual_short = x
        x = self.layer_norm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residual_short

        # (batch_size, h * w, features) -> (batch_size, features, h * w)
        x = x.transpose(1, 2)

        # (batch_size, features, h * w) -> (batch_size, features, h, w)
        x = x.view(batch_size, c, h, w)

        return self.conv_output(x) + residual_long


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # x shape: (batch_size, features, h, w) -> (batch_size, features, h * 2, w * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                # (batch_size, 4, h / 8, w / 8)
                SwitchSequential(
                    nn.Conv2d(
                        in_channels=4,
                        out_channels=320,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                ),  # (batch_size, 4, h / 8, w / 8) -> (batch_size, 320, h / 8, w / 8)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=320, out_channels=320),
                    UNET_AttentionBlock(8, 40),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=320, out_channels=320),
                    UNET_AttentionBlock(8, 40),
                ),
                SwitchSequential(
                    nn.Conv2d(
                        in_channels=320,
                        out_channels=320,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),  # (batch_size, 320, h / 8, w / 8) -> (batch_size, 320, h / 16, w / 16)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=320, out_channels=640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=640, out_channels=640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    nn.Conv2d(
                        in_channels=640,
                        out_channels=640,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),  # (batch_size, 640, h / 16, w / 16) -> (batch_size, 640, h / 32, w / 32)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=640, out_channels=1280),
                    UNET_AttentionBlock(8, 160),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1280, out_channels=1280),
                    UNET_AttentionBlock(8, 160),
                ),
                SwitchSequential(
                    nn.Conv2d(
                        in_channels=1280,
                        out_channels=1280,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),  # (batch_size, 1280, h / 32, w / 32) -> (batch_size, 1280, h / 64, w / 64)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1280, out_channels=1280),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1280, out_channels=1280),
                ),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(in_channels=1280, out_channels=1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(in_channels=1280, out_channels=1280),
        )

        self.decoders = nn.ModuleList(
            [
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280)
                ),  # (batch_size, 2560, h / 64, w / 64) -> (batch_size, 1280, h / 64, w / 64)
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280),
                    Upsample(1280),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280),
                    UNET_AttentionBlock(8, 160),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=2560, out_channels=1280),
                    UNET_AttentionBlock(8, 160),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1920, out_channels=1280),
                    UNET_AttentionBlock(8, 160),
                    Upsample(1280),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1920, out_channels=640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=1280, out_channels=640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=960, out_channels=640),
                    UNET_AttentionBlock(8, 80),
                    Upsample(640),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=960, out_channels=320),
                    UNET_AttentionBlock(8, 40),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=640, out_channels=320),
                    UNET_AttentionBlock(8, 40),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(in_channels=640, out_channels=320),
                    UNET_AttentionBlock(8, 40),
                ),
            ]
        )

    def forward(self, x, context, time):
        # x shape: (batch_size, 4, h / 8, w / 8)
        # context shape: (batch_size, seq_len, d_model)
        # time shape: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # x shape: (batch_size, 320, h / 8, w / 8)
        x = F.silu(self.group_norm(x))
        x = self.conv(x)

        # (batch_size, 4, h / 8, w / 8)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent shape: (batch_size, 4, h / 8, w / 8)
        # context shape: (batch_size, seq_len, d_model)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch_size, 4, h / 8, w / 8) -> (batch_size, 320, h / 8, w / 8)
        output = self.unet(latent, context, time)

        # (batch_size, 320, h / 8, w / 8) -> (batch_size, 4, h / 8, w / 8)
        output = self.final(output)

        # (batch_size, 4, h / 8, w / 8)
        return output
