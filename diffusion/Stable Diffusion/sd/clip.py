import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.position_embedding = nn.Parameter(torch.zeros((n_tokens, d_model)))

    def forward(self, tokens):
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(n_heads=n_heads, d_model=d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.ffn_1 = nn.Linear(in_features=d_model, out_features=(4 * d_model))
        self.ffn_2 = nn.Linear(in_features=(4 * d_model), out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        residual = x

        # Self Attention
        x = self.layer_norm_1(x)
        x = self.attention(x, causal_mask=True)

        x += residual

        # Feed-forward Network
        residual = x
        x = self.layer_norm_2(x)
        x = self.ffn_1(x)
        x *= torch.sigmoid(1.702 * x)  # QuickGELU activation function

        x = self.ffn_2(x)
        x += residual
        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = CLIPEmbedding(vocab_size=49408, d_model=768, n_tokens=77)

        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])

        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layer_norm(state)
        return output  # (batch_size, seq_len, d_model)
