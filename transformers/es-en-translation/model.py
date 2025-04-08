import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import Counter


class Vocab:
    def __init__(self, stoi, itos, default_index):
        self.stoi = stoi  # mapping from token to index
        self.itos = itos  # list of tokens
        self.default_index = default_index  # default index for unknown words

    def __getitem__(self, token):
        # Return index of token
        return self.stoi.get(
            token, self.default_index
        )  # If not found return the default index

    def get_stoi(self):
        return self.stoi

    def lookup_tokens(self, indices):
        # Return the tokens at indices
        return [self.itos[i] for i in indices]

    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def __iter__(self):
        return iter(self.itos)

    def __repr__(self):
        return f"Vocab({len(self)} tokens)"


def build_vocab_from_iterator(token_iterator, min_freq, specials):
    counter = Counter()  # Use counter to get tokens and frequencies
    for tokens in token_iterator:
        counter.update(tokens)
    tokens = [
        token for token, freq in counter.items() if freq >= min_freq
    ]  # Keep tokens with frequency >= min_freq
    tokens = sorted(tokens)  # Sort alphabetically
    itos = list(specials) + tokens
    stoi = {token: idx for idx, token in enumerate(itos)}  # token-to-index
    return Vocab(stoi=stoi, itos=itos, default_index=stoi.get("<unk>", 0))


"""### Transformer Model"""


# Embedding Layer
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        vocab_size: size of the vocabulary
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        """
        super().__init__()

        self.d_model = d_model

        # Embedding look-up table (vocab_size, d_model)
        self.lut = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        # Multiply by the sqrt of the d_model as a scale factor
        return self.lut(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)


"""**Positional Encoding Equations**

$PE(k, 2i) = sin(\frac{k}{10000^{\frac{2i}{d_{model}}}})$

$PE(k, 2i + 1) = cos(\frac{k}{10000^{\frac{2i}{d_{model}}}})$
"""


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        dropout: probability of dropout
        max_length: max length of a sequence
        """
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
        # Add the positional encoding to the embeddings that are passed in
        x += self.pe[:, : x.size(1)]
        return self.dropout(x)


"""**Multi-Head Self-Attention Equations:**

$Q = X W_q$

$K = X W_k$

$V = X W_v$

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_{key}}})V$
"""


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

        self.Wq = nn.Linear(
            in_features=d_model, out_features=d_model
        )  # Learnable weights for query
        self.Wk = nn.Linear(
            in_features=d_model, out_features=d_model
        )  # Learnable weights for key
        self.Wv = nn.Linear(
            in_features=d_model, out_features=d_model
        )  # Learnable weights for value
        self.Wo = nn.Linear(
            in_features=d_model, out_features=d_model
        )  # Learnable weights for output

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
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
                mask == 0, float("-inf")
            )  # Filling it with 0 would result in 1 after the mask because e^0 = 1. Intead we fill it with an incredibly large negative number

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

        return output, attention_probs


# Position-Wise Feed Forward Network (FFN)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        d_ffn: dimensions of the feed-forward network
        dropout: probability of dropout
        """
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ffn),
            nn.ReLU(),
            nn.Linear(in_features=d_ffn, out_features=d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.ffn(x)


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_heads: number of self attention heads per sequence
        d_ffn: dimensions of the feed-forward network
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
            d_model=d_model, d_ffn=d_ffn, dropout=dropout
        )
        self.ffn_layer_norm = nn.LayerNorm(d_model)  # Layer normalization

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src: Tensor, src_mask: Tensor):
        """
        src: embedded sequences (batch_size, seq_length, d_model)
        src_mask: mask for the sequences (batch_size, 1, 1, seq_length)
        """
        # Multi-Head Attention

        # The source mask ensures the model ignores these padding positions by assigning them near-zero attention scores.
        _src, attention_probs = self.attention(src, src, src, src_mask)  # Q, K, V, mask

        # Residual Addition and Layer Normalization
        src = self.attention_layer_norm(
            src + self.dropout(_src)
        )  # We do residual addition by adding back the src (the embeddings) to the output of Self-Attention

        # Position-wise Feed-forward Network
        _src = self.position_wise_ffn(src)

        # Residual Addition and Layer Normalization
        src = self.ffn_layer_norm(src + self.dropout(_src))

        return src, attention_probs


# The Encoder
class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_layers: number of encoder layers in the encoder block
        n_heads: number of self attention heads per sequence
        d_ffn: dimensions of the feed-forward network
        dropout: probability of dropout
        """
        super().__init__()

        # Create n_layers encoders
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout
                )
                for layer in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src: Tensor, src_mask: Tensor):
        """
        src: embedded sequences (batch_size, seq_length, d_model)
        src_mask: mask for the sequences (batch_size, 1, 1, seq_length)
        """

        # Pass the sequences through each encoder layer
        for layer in self.layers:
            src, attention_probs = layer(src, src_mask)

        self.attention_probs = attention_probs

        src += torch.randn_like(src) * 0.001

        return src


# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        """
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_heads: number of self attention heads per sequence
        d_ffn: dimensions of the feed-forward network
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
            d_model=d_model, d_ffn=d_ffn, dropout=dropout
        )
        self.ffn_layer_norm = nn.LayerNorm(d_model)  # Layer normalization

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
        """
        trg: embedded sequences (batch_size, trg_seq_length, d_model)
        src: embedded sequences (batch_size, src_seq_length, d_model)
        trg_mask: mask for the sequences (batch_size, 1, trg_seq_length, trg_seq_length)
        src_mask: mask for the sequences (batch_size, 1, 1, src_seq_length)
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
        _trg, attention_probs = self.attention(trg, src, src, src_mask)  # Q, K, V, mask
        # Residual Addition and Layer Normalization
        trg = self.attention_layer_norm(trg + self.dropout(_trg))

        # Position-wise Feed-forward Network
        _trg = self.position_wise_ffn(trg)
        # Residual Addition and Layer Normalization
        trg = self.ffn_layer_norm(trg + self.dropout(_trg))

        return trg, attention_probs, masked_attention_probs


# The Decoder
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        """
        vocab_size: size of the target vocabulary
        d_model: dimensions of the embeddings (number of values in each embedding vector)
        n_layers: number of encoder layers in the encoder block
        n_heads: number of self attention heads per sequence
        d_ffn: dimensions of the feed-forward network
        dropout: probability of dropout
        """
        super().__init__()

        # Create n_layers decoders
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout
                )
                for layer in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)

        # Output layer
        self.Wo = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
        """
        trg: embedded sequences (batch_size, trg_seq_length, d_model)
        src: embedded sequences (batch_size, src_seq_length, d_model)
        trg_mask: mask for the sequences (batch_size, 1, trg_seq_length, trg_seq_length)
        src_mask: mask for the sequences (batch_size, 1, 1, src_seq_length)
        """

        # Pass the sequences through each decoder layer
        for layer in self.layers:
            trg, attention_probs, masked_attention_probs = layer(
                trg, src, trg_mask, src_mask
            )

        self.attention_probs = attention_probs
        self.masked_attention_probs = masked_attention_probs

        trg += torch.randn_like(trg) * 0.001

        return self.Wo(trg)


# The Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: EmbeddingLayer,
        trg_embed: EmbeddingLayer,
        src_pad_idx: int,
        trg_pad_idx: int,
        device,
    ):
        """
        encoder: encoder stack
        decoder: decoder stack
        src_embed: source embeddings
        trg_embd: target embeddings
        src_pad_idx: source padding index
        trg_pad_idx: target padding index
        device: device
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src: Tensor):
        # Assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg: Tensor):
        seq_length = trg.shape[1]

        # Assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions
        trg_mask = (
            (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        )  # (batch_size, 1, 1, seq_length)

        # Generate subsequent mask
        trg_sub_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=self.device)
        ).bool()  # (batch_size, 1, seq_length, seq_length)

        # Bottom triangle is True, top triangle is False
        trg_mask = trg_mask & trg_sub_mask

        return trg_mask

    def forward(self, src: Tensor, trg: Tensor):
        """
        trg: raw target sequences (batch_size, trg_seq_length)
        src: raw src sequences (batch_size, src_seq_length)
        """

        # Create source and target masks
        src_mask = self.make_src_mask(src)  # (batch_size, 1, 1, src_seq_length)

        # The lower triangle of the mask is filled with 1s
        trg_mask = self.make_trg_mask(
            trg
        )  # (batch_size, 1, trg_seq_length, trg_seq_length)

        # Encoder layers
        src = self.encoder(
            self.src_embed(src), src_mask
        )  # (batch_size, src_seq_length, d_model)

        # Decoder layers
        output = self.decoder(
            self.trg_embed(trg), src, trg_mask, src_mask
        )  # Pass in both the target (for Masked Multi-Head Self-Attention) and source for (Cross-Attention)

        return output


def make_model(
    device,
    src_vocab,
    trg_vocab,
    n_layers: int = 3,
    d_model: int = 512,
    d_ffn: int = 2048,
    n_heads: int = 8,
    dropout: float = 0.1,
    max_length: int = 5000,
):
    """
    src_vocab: source vocabulary
    trg_vocab: target vocabulary
    n_layers: number of encoder layers in the encoder block
    d_model: dimensions of the embeddings (number of values in each embedding vector)
    d_ffn: dimensions of the feed-forward network
    n_heads: number of self attention heads per sequence
    dropout: probability of dropout
    max_length: maximum sequence length for positional encodings
    """

    encoder = Encoder(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ffn=d_ffn,
        dropout=dropout,
    )

    decoder = Decoder(
        vocab_size=len(trg_vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ffn=d_ffn,
        dropout=dropout,
    )

    src_embed = EmbeddingLayer(vocab_size=len(src_vocab), d_model=d_model)
    trg_embed = EmbeddingLayer(vocab_size=len(trg_vocab), d_model=d_model)

    pos_enc = PositionalEncoding(
        d_model=d_model, dropout=dropout, max_length=max_length
    )

    model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=nn.Sequential(src_embed, pos_enc),
        trg_embed=nn.Sequential(trg_embed, pos_enc),
        src_pad_idx=src_vocab.get_stoi()["<pad>"],
        trg_pad_idx=trg_vocab.get_stoi()["<pad>"],
        device=device,
    )

    # Initialize parameters with Xaviar/Glorot
    # This maintains a consistent variance of activations throughout the network
    # Helps avoid issues like vanishing or exploding gradients.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def translate_sentence(
    sentence, model, vocab_src, vocab_trg, spacy_es, device, max_length=50
):
    model.eval()
    if isinstance(sentence, str):
        src = (
            ["<bos>"] + [token.text.lower() for token in spacy_es(sentence)] + ["<eos>"]
        )
    else:
        src = ["<bos>"] + sentence + ["<eos>"]
    src_indexes = [vocab_src[token] for token in src]
    src_tensor = torch.tensor(src_indexes).int().unsqueeze(0).to(device)
    trg_indexes = [vocab_trg.stoi["<bos>"]]
    for _ in range(max_length):
        trg_tensor = torch.tensor(trg_indexes).int().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(src_tensor, trg_tensor)
        pred_token = logits.argmax(dim=2)[:, -1].item()
        if pred_token == vocab_trg.stoi["<eos>"]:
            break
        trg_indexes.append(pred_token)
    trg_tokens = vocab_trg.lookup_tokens(trg_indexes)
    return " ".join(trg_tokens)
