import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768):
        super().__init__()

        self.d_model = d_model

        self.lut = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )  # (vocab_size, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        return self.lut(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 768, dropout: float = 0.1, max_length: int = 128):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, d_model)  # (max_length, d_model)
        # Create position column
        k = torch.arange(0, max_length).unsqueeze(dim=1)  # (max_length, 1)

        # Use the log version of the function for positional encodings
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # (d_model / 2)

        # Use sine for the even indices and cosine for the odd indices
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)

        pe = pe.unsqueeze(dim=0)  # Add the batch dimension(1, max_length, d_model)

        # We use a buffer because the positional encoding is fixed and not a model paramter that we want to be updated during backpropagation.
        self.register_buffer(
            "pe", pe
        )  # Buffers are saved with the model state and are moved to the correct device

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x += self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_model // n_heads

        self.Wq = nn.Linear(in_features=d_model, out_features=d_model)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_model)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_model)
        self.Wo = nn.Linear(in_features=d_model, out_features=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        # input shape: (batch_size, seq_len, d_model)

        batch_size = key.size(0)

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_key).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, q_length, d_key)
        K = K.view(batch_size, -1, self.n_heads, self.d_key).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, k_length, d_key)
        V = V.view(batch_size, -1, self.n_heads, self.d_key).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, v_length, d_key)

        scaled_dot_product = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(
            self.d_key
        )  # (batch_size, n_heads, q_length, k_length)

        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(
                mask == 0, float("-inf")
            )

        attention_probs = torch.softmax(scaled_dot_product, dim=-1)

        A = torch.matmul(
            self.dropout(attention_probs), V
        )  # (batch_size, n_heads, q_length, d_key)

        A = A.permute(0, 2, 1, 3)  # (batch_size, q_length, n_heads, d_key)
        A = A.contiguous().view(
            batch_size, -1, self.n_heads * self.d_key
        )  # (batch_size, q_length, d_model)

        output = self.Wo(A)  # (batch_size, q_length, d_model)

        return output, attention_probs


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 768, dropout: float = 0.1):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=(d_model * 4)),
            nn.ReLU(),
            nn.Linear(in_features=(d_model * 4), out_features=d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # x shape: (batch_size, q_length, d_model)
        return self.ffn(x)  # (batch_size, q_length, d_model)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.attention_layer_norm = nn.LayerNorm(d_model)

        self.position_wise_ffn = PositionwiseFeedForward(
            d_model=d_model, dropout=dropout
        )
        self.ffn_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src: Tensor, src_mask: Tensor):
        _src, attention_probs = self.attention(
            query=src, key=src, value=src, mask=src_mask
        )
        src = self.attention_layer_norm(src + self.dropout(_src))

        _src = self.position_wise_ffn(src)
        src = self.ffn_layer_norm(src + self.dropout(_src))

        return src, attention_probs


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for layer in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src: Tensor, src_mask: Tensor):

        for layer in self.layers:
            src, attention_probs = layer(src, src_mask)

        self.attention_probs = attention_probs

        # src += torch.randn_like(src) * 0.001
        return src


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        src_embed: EmbeddingLayer,
        src_pad_idx: int,
        device,
        d_model: int = 768,
        num_labels: int = 5,
    ):
        super().__init__()

        self.encoder = encoder
        self.src_embed = src_embed
        self.device = device
        self.src_pad_idx = src_pad_idx

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(in_features=d_model, out_features=num_labels)

    def make_src_mask(self, src: Tensor):
        # Assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def forward(self, src: Tensor):
        src_mask = self.make_src_mask(src)  # (batch_size, 1, 1, src_seq_length)
        output = self.encoder(
            self.src_embed(src), src_mask
        )  # (batch_size, src_seq_length, d_model)
        output = output[
            :, 0, :
        ]  # Get the sos token vector representation (works sort of like a cls token in ViT) shape: (batch_size, 1, d_model)
        logits = self.classifier(self.dropout(output))

        return logits


def make_model(
    device,
    tokenizer,
    n_layers: int = 3,
    d_model: int = 768,
    num_labels: int = 5,
    n_heads: int = 8,
    dropout: float = 0.1,
    max_length: int = 128,
):
    encoder = Encoder(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout
    )

    src_embed = EmbeddingLayer(vocab_size=tokenizer.vocab_size, d_model=d_model)

    pos_enc = PositionalEncoding(
        d_model=d_model, dropout=dropout, max_length=max_length
    )

    model = Transformer(
        encoder=encoder,
        src_embed=nn.Sequential(src_embed, pos_enc),
        src_pad_idx=tokenizer.pad_token_id,
        device=device,
        d_model=d_model,
        num_labels=num_labels,
    )

    # Initialize parameters with Xaviar/Glorot
    # This maintains a consistent variance of activations throughout the network
    # Helps avoid issues like vanishing or exploding gradients.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def get_sentiment(text, model, tokenizer, device, max_length: int = 32):
    model.eval()

    encoded = model.src_embed[0].lut.weight.new_tensor([])
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    src_tensor = encoded["input_ids"].to(device)

    with torch.inference_mode():
        logits = model(src_tensor)  # shape: (batch_size, num_labels)

    pred_index = torch.argmax(logits, dim=1).item()

    sentiment_map = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive",
    }
    return sentiment_map.get(pred_index, "Unknown")
