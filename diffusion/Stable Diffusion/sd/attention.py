import torch
from torch import nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_model: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(
            in_features=d_model, out_features=(3 * d_model), bias=in_proj_bias
        )
        self.out_proj = nn.Linear(
            in_features=d_model, out_features=d_model, bias=out_proj_bias
        )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x shape: (batch_size, seq_len=(h * w), d_model=channels)
        input_shape = x.shape
        batch_size, seq_len, d_model = input_shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model * 3) -> 3 tensors of shape (batch_size, seq_len, d_model)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_model // num_heads) -> (batch_size, num_heads, seq_len, d_model // num_heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch_size, num_heads, seq_len, seq_len)
        weight = torch.matmul(q, k.transpose(2, 3))

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_model // num_heads) -> (batch_size, num_heads, seq_len, d_model // num_heads)
        output = torch.matmul(weight, v)

        # (batch_size, num_heads, seq_len, d_model // num_heads) -> (batch_size, seq_len, num_heads, d_model // num_heads)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)
        output = self.out_proj(output)

        # (batch_size, seq_len, d_model)
        return output


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_cross: int,
        in_proj_bias=True,
        out_proj_bias=True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(
            in_features=d_model, out_features=d_model, bias=in_proj_bias
        )
        self.k_proj = nn.Linear(
            in_features=d_cross, out_features=d_model, bias=in_proj_bias
        )
        self.v_proj = nn.Linear(
            in_features=d_cross, out_features=d_model, bias=in_proj_bias
        )
        self.out_proj = nn.Linear(
            in_features=d_model, out_features=d_model, bias=out_proj_bias
        )

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x, y):
        # x (latent) shape: (batch_size, seq_len_q, d_model_q)
        # y (context) shape: (batch_size, seq_len_kv=77, d_model_kv=768)
        input_shape = x.shape
        batch_size, seq_len, d_model = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = torch.matmul(q, k.transpose(2, 3))
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_model // num_heads) -> (batch_size, num_heads, seq_len, d_model // num_heads)
        output = torch.matmul(weight, v)

        # (batch_size, num_heads, seq_len, d_model // num_heads) -> (batch_size, seq_len, num_heads, d_model // num_heads)
        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)
        output = self.out_proj(output)

        # (batch_size, seq_len_q, d_model_q)
        return output
