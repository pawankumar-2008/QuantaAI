"""Configurable decoder-only QuantaLM V2 Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .transformer_block import TransformerBlock
from .workspace_config import v2_config


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = v2_config.VOCAB_SIZE
    context_length: int = v2_config.CONTEXT_LENGTH
    embedding_dim: int = v2_config.EMBEDDING_DIM
    num_layers: int = v2_config.NUM_LAYERS
    num_heads: int = v2_config.NUM_HEADS
    feed_forward_dim: int = v2_config.FEED_FORWARD_DIM
    dropout: float = v2_config.DROPOUT
    architecture_version: str = "2.0"

    def __post_init__(self) -> None:
        positive_values = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
        }
        for name, value in positive_values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not self.architecture_version:
            raise ValueError("architecture_version must not be empty.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        expected = set(cls.__dataclass_fields__)
        provided = set(values)
        if provided != expected:
            raise ValueError(
                "Model configuration keys do not match. "
                f"Missing: {sorted(expected - provided)}; "
                f"unexpected: {sorted(provided - expected)}."
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuantaLMV2(nn.Module):
    """Decoder-only Transformer language model with tied token/output weights."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.embedding_dim,
        )
        self.position_embedding = nn.Embedding(
            self.config.context_length,
            self.config.embedding_dim,
        )
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=self.config.embedding_dim,
                    num_heads=self.config.num_heads,
                    feed_forward_dim=self.config.feed_forward_dim,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.embedding_dim)
        self.output_head = nn.Linear(
            self.config.embedding_dim,
            self.config.vocab_size,
            bias=False,
        )
        self.output_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs.")
        batch_size, sequence_length = input_ids.shape
        if batch_size < 1:
            raise ValueError("batch size must be at least one.")
        if sequence_length < 1:
            raise ValueError("sequence length must be at least one.")
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}."
            )
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError(
                f"Token IDs must be in [0, {self.config.vocab_size - 1}]."
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return self.output_head(hidden_states)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
