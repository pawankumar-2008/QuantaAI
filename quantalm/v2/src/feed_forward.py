"""Position-wise feed-forward network."""

from __future__ import annotations

from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feed_forward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or feed_forward_dim <= 0:
            raise ValueError("Embedding and feed-forward dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states):
        return self.network(hidden_states)
