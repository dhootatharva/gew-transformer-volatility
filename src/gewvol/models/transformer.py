"""Multi-market Transformer encoder used for the GEW-Transformer and the pure Transformer.

Input (B, T, M*F) holds F features for each of M markets per day. The model

1. adds a learned market embedding to each market's F features,
2. mixes markets at each time step with a single-head market attention (residual),
3. flattens time x market into a token sequence of length T*M (time-major),
4. scales features by fixed per-feature weights, tiles each token back to M*F
   dimensions and applies a sigmoid feature gate,
5. LayerNorm + linear projection to d_model, learned positional embedding,
   exponential temporal decay exp(linspace(-1, 0, T)),
6. runs N post-norm encoder layers that also return their attention maps,
7. mean-pools the M tokens of the last day, applies an MLP head and adds the mean
   (normalised) RV of the last day as a residual.

The output is in the (standardised) target space used during training.
"""

from __future__ import annotations

import torch
from torch import nn

# Fixed per-feature input weights from the thesis code.
FEATURE_WEIGHTS = {
    5: (0.5, 1.0, 2.0, 2.0, 1.5),  # GEW: r, RV, GARCH, EGARCH, EWMA
    2: (0.7, 1.3),                 # pure: r, RV
}


class EncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer that returns head-averaged attention weights."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x, attn_weights


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_markets: int = 3,
        d_model: int = 160,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_ff: int = 320,
        dropout: float = 0.05,
        head_size: int = 64,
        max_len: int = 500,
    ) -> None:
        super().__init__()
        if input_size % num_markets != 0:
            raise ValueError("input_size must be divisible by num_markets")
        self.num_markets = num_markets
        self.features_per_market = input_size // num_markets
        f = self.features_per_market

        self.market_embedding = nn.Embedding(num_markets, f)
        self.gate = nn.Sequential(
            nn.Linear(input_size, input_size), nn.ReLU(),
            nn.Linear(input_size, input_size), nn.Sigmoid(),
        )
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.market_attn = nn.MultiheadAttention(embed_dim=f, num_heads=1, batch_first=True)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )
        weights = FEATURE_WEIGHTS.get(f, (1.0,) * f)
        self.register_buffer("feature_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """x: (B, T, M*F) -> (prediction (B,), attention maps [layers x (B, T*M, T*M)])."""
        B, T, _ = x.shape
        M, F = self.num_markets, self.features_per_market
        x_original = x

        x_split = x.view(B, T, M, F)
        emb = self.market_embedding(torch.arange(M, device=x.device))
        x_split = x_split + emb.unsqueeze(0).unsqueeze(0)

        x_m = x_split.reshape(B * T, M, F)
        x_m, _ = self.market_attn(x_m, x_m, x_m)
        x_split = x_split + x_m.reshape(B, T, M, F)

        tokens = x_split.reshape(B, T * M, F) * self.feature_weights
        tokens = tokens.repeat(1, 1, M)                    # (B, T*M, M*F)
        tokens = tokens * self.gate(tokens)
        h = self.input_proj(self.input_norm(tokens))

        h = h + self.pos_embedding[:, : T * M, :]
        h = self.dropout(h)
        decay = torch.exp(torch.linspace(-1, 0, T, device=x.device)).repeat_interleave(M)
        h = h * decay.unsqueeze(0).unsqueeze(-1)

        attn_maps = []
        for layer in self.layers:
            h, attn = layer(h)
            attn_maps.append(attn)

        last = h.view(B, T, M, -1)[:, -1].mean(dim=1)
        rv_last = x_original[:, -1, 1::F].mean(dim=1)
        out = self.mlp(last).squeeze(-1) + rv_last
        return out, attn_maps
