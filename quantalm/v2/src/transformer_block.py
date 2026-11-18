"""Pre-LayerNorm decoder Transformer block."""

from __future__ import annotations

from torch import nn

from .attention import CausalSelfAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attention(self.norm1(hidden_states))
        return hidden_states + self.feed_forward(self.norm2(hidden_states))
