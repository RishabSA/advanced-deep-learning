import torch
import torch.nn as nn
import math


class ExtractPatches(nn.Module):
    def __init__(self, patch_size: int = 16):
        super().__init__()

        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, c, h, w = x.shape

        # Unfold applies a slding window to generate patches
        # The transpose and reshape change the shape to (batch_size, num_patches, 3 * patch_size * patch_size), flattening the patches
        return (
            self.unfold(x)
            .transpose(1, 2)
            .reshape(batch_size, -1, c * self.patch_size * self.patch_size)
        )


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        """
        super().__init__()

        # Intead of precomputing fixed values, we will compute in the forward pass based off of the sinusodiual encoding formula
        self.d_model = d_model

    def forward(self, x):
        device = x.device
        half_dim = self.d_model // 2  # Use half for sin and half for cos
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]  # (batch_size, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_heads: number of self attention heads per sequence
        dropout: probability of dropout
        """
        super().__init__()
        assert (
            d_model % n_heads == 0
        )  # We want to make sure that the dimensions are split evenly among the attention heads.
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)  # Learnable weights for query
        self.Wk = nn.Linear(d_model, d_model)  # Learnable weights for key
        self.Wv = nn.Linear(d_model, d_model)  # Learnable weights for value
        self.Wo = nn.Linear(d_model, d_model)  # Learnable weights for output

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch_size, q_length, d_model)
        key: (batch_size, k_length, d_model)
        value: (batch_size, s_length, d_model)
        """
        batch_size = key.size(0)

        # Matrix multiplication for Q, K, and V tensors
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        # Split each tensor into heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_key).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, q_length, d_key)
        K = K.view(batch_size, -1, self.n_heads, self.d_key).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, k_length, d_key)
        V = V.view(batch_size, -1, self.n_heads, self.d_key).permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, v_length, d_key)

        # Scaled dot product
        # K^T becomees (batch_size, n_heads, d_key, k_length)
        scaled_dot_product = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(
            self.d_key
        )  # (batch_size, n_heads, q_length, k_length)

        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(
                mask == 0, -float("inf")
            )  # Filling it with 0 would result in 1 after the mask because e^0 = 1. Intead we fill it with an infinitley large negative number

        # Softmax function for attention probabilities
        attention_probs = torch.softmax(scaled_dot_product, dim=-1)

        # Multiply by V to get attention with respect to the values
        A = torch.matmul(self.dropout(attention_probs), V)

        # Reshape attention back to (batch_size, q_length, d_model)
        A = (
            A.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_key)
        )

        # Pass through the final linear layer
        output = self.Wo(A)

        return (
            output,
            attention_probs,
        )  # Output shape: (batch_size, q_length, d_model), Attention probs shape: (batch_size, n_heads, q_length, k_length)


# Position-Wise Feed Forward Network (FFN)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        dropout: probability of dropout
        """
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=(d_model * 4)),
            nn.ReLU(),
            nn.Linear(in_features=(d_model * 4), out_features=d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.ffn(x)


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_heads: number of self attention heads per sequence
        dropout: probability of dropout
        """
        super().__init__()

        # Multi-Head Self-Attention sublayer
        self.attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.attention_layer_norm = nn.LayerNorm(d_model)  # Layer normalization

        # Position-wise Feed-forward Network
        self.position_wise_ffn = PositionwiseFeedForward(
            d_model=d_model, dropout=dropout
        )
        self.ffn_layer_norm = nn.LayerNorm(d_model)  # Layer normalization

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        """
        src: embedded sequences (batch_size, seq_length, d_model)
        """
        # Multi-Head Attention

        _src, attention_probs = self.attention(
            src, src, src, None
        )  # Q, K, V, src_mask: we don't need a source mask because all images are the same dimension

        # Residual Addition and Layer Normalization
        src = self.attention_layer_norm(
            src + self.dropout(_src)
        )  # We do residual addition by adding back the src (the embeddings) to the output of Self-Attention

        # Position-wise Feed-forward Network
        _src = self.position_wise_ffn(src)

        # Residual Addition and Layer Normalization
        src = self.ffn_layer_norm(src + self.dropout(_src))

        return src, attention_probs


# The Encoder that takes in images and returns the encoding to be passed into the decoder
class Encoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int = 16,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_layers: number of encoder layers in the encoder block
        n_heads: number of self attention heads per sequence
        dropout: probability of dropout
        """
        super().__init__()

        self.patch_size = patch_size

        self.extract_patches = ExtractPatches(patch_size=patch_size)
        self.fc_in = nn.Linear(in_channels * patch_size * patch_size, d_model)

        seq_length = (image_size // patch_size) ** 2

        # Image src is going to use a learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, d_model).normal_(std=0.02)
        )

        # Create n_layers encoders
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for layer in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        """
        src: embedded sequences (batch_size, seq_length, d_model)
        """

        # Extract the patches and apply a linear layer
        batch_size = src.shape[0]
        src = self.fc_in(self.extract_patches(src))

        # Add the learned positional embedding
        src = src + self.pos_embedding

        # Pass the sequences through each encoder layer
        for layer in self.layers:
            src, attention_probs = layer(src)

        self.attention_probs = attention_probs

        return src


# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_heads: number of self attention heads per sequence
        dropout: probability of dropout
        """
        super().__init__()

        # Masked Multi-Head Self-Attention sublayer
        self.masked_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.masked_attention_layer_norm = nn.LayerNorm(d_model)  # Layer normalization

        # Multi-Head Self-Attention sublayer
        self.attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.attention_layer_norm = nn.LayerNorm(d_model)  # Layer normalization

        # Position-wise Feed-forward Network
        self.position_wise_ffn = PositionwiseFeedForward(
            d_model=d_model, dropout=dropout
        )
        self.ffn_layer_norm = nn.LayerNorm(d_model)  # Layer normalization

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg, src, trg_mask):
        """
        trg: embedded captions (batch_size, trg_seq_length, d_model)
        src: embedded images (batch_size, src_seq_length, d_model)
        trg_mask: mask for the captions preventing peeking at future tokens (batch_size, 1, trg_seq_length, trg_seq_length)
        """

        # Masked Multi-Head Attention

        # The target mask is used to prevent the model from seeing future tokens. This ensures that the prediction is made solely based on past and present tokens.
        _trg, masked_attention_probs = self.masked_attention(
            trg, trg, trg, trg_mask
        )  # Q, K, V, mask

        # Residual Addition and Layer Normalization
        trg = self.masked_attention_layer_norm(trg + self.dropout(_trg))

        # Multi-Head Attention - This time, we also pass in the output of the encoder layers as src.
        # This is important because this allows us to keep track of and learn relationships between the input and output tokens.
        _trg, attention_probs = self.attention(trg, src, src, None)  # Q, K, V, mask
        # Residual Addition and Layer Normalization
        trg = self.attention_layer_norm(trg + self.dropout(_trg))

        # Position-wise Feed-forward Network
        _trg = self.position_wise_ffn(trg)
        # Residual Addition and Layer Normalization
        trg = self.ffn_layer_norm(trg + self.dropout(_trg))

        return trg, attention_probs, masked_attention_probs


# The Decoder Module that takes the encoded images from the encoder and generates captions
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        vocab_size: size of the target vocabulary
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_layers: number of encoder layers in the encoder block
        n_heads: number of self attention heads per sequence
        dropout: probability of dropout
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.embedding.weight.data = 0.001 * self.embedding.weight.data

        # Initialize sinusoidal positional embeddings
        self.pos_emb = PositionalEncoding(d_model=d_model)

        # Create n_layers decoders
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for layer in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)

        # Output layer
        self.Wo = nn.Linear(in_features=d_model, out_features=vocab_size)

    def make_trg_mask(self, trg):
        seq_length = trg.shape[1]

        trg_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=trg.device)
        ).bool()

        return trg_mask.unsqueeze(0).unsqueeze(
            0
        )  # (batch_size=1, n_heads=1, seq_length, seq_length)

    def forward(self, trg, src):
        """
        trg: target sequences (batch_size, trg_seq_length, d_model)
        src: embedding images (batch_size, src_seq_length, d_model)
        """

        # Embed the target captions
        trg = self.embedding(trg)
        batch_size, l, h = trg.shape

        trg_index = torch.arange(l, device=trg.device)
        pos_emb = self.pos_emb(trg_index).reshape(1, l, h).expand(batch_size, l, h)
        # Add the fixed sinusodial positional embedding
        trg += pos_emb

        # Create a target mask for the target captions to prevent the model from peeking at future tokens
        trg_mask = self.make_trg_mask(
            trg
        )  # (batch_size, 1, trg_seq_length, trg_seq_length)

        # Pass the sequences through each decoder layer
        for layer in self.layers:
            trg, attention_probs, masked_attention_probs = layer(trg, src, trg_mask)

        self.attention_probs = attention_probs
        self.masked_attention_probs = masked_attention_probs  # (batch_size, n_heads, trg_seq_len, src_seq_len) trg_seq_len: length of the target caption \ src_seq_len: number of patches from the encoder

        # Final linear output layer
        return self.Wo(trg)


class CaptioningTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        vocab_size: int,
        device,
        patch_size: int = 16,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()

        self.device = device

        # Create an encoder and decoder with specified parameters
        self.encoder = Encoder(
            image_size=image_size,
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads
        )

    def forward(self, src, trg):
        # Encoder layers
        src = self.encoder(src)  # (batch_size, src_seq_length, d_model)

        # Decoder layers
        output = self.decoder(
            trg, src
        )  # Pass in both the target (for Masked Multi-Head Self-Attention) and source for (Cross-Attention)

        return output
