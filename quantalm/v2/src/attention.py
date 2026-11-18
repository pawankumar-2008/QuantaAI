"""Multi-head causal self-attention for QuantaLM V2."""

from __future__ import annotations

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Scaled dot-product causal self-attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_heads <= 0:
            raise ValueError("embedding_dim and num_heads must be positive.")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout_probability = dropout
        self.qkv_projection = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, embedding]."
            )
        batch_size, sequence_length, embedding_dim = hidden_states.shape
        if sequence_length < 1:
            raise ValueError("sequence length must be at least one.")
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {embedding_dim}."
            )

        qkv = self.qkv_projection(hidden_states)
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            device=hidden_states.device,
            dtype=torch.bool,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )
        return self.residual_dropout(self.output_projection(attended))
