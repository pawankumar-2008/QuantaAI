import torch
from torch import nn

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CONTEXT_LENGTH,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    DROPOUT,
)


class TransformerBlock(nn.Module):
    """A decoder-only Transformer block."""

    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(EMBEDDING_DIM)

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(EMBEDDING_DIM)

        self.feed_forward = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM),
            nn.Dropout(DROPOUT),
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        sequence_length = x.size(1)

        # Prevent a position from attending to future tokens.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )

        x = x + self.dropout(attention_output)
        x = x + self.feed_forward(self.norm2(x))

        return x


class QuantaLM(nn.Module):
    """Small language model initialized from random weights."""

    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
        )

        self.position_embedding = nn.Embedding(
            CONTEXT_LENGTH,
            EMBEDDING_DIM,
        )

        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(NUM_LAYERS)]
        )

        self.final_norm = nn.LayerNorm(EMBEDDING_DIM)

        self.output_head = nn.Linear(
            EMBEDDING_DIM,
            vocab_size,
            bias=False,
        )

        # Tie input and output token embeddings.
        self.output_head.weight = self.token_embedding.weight

    def forward(self, token_ids, targets=None):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {CONTEXT_LENGTH}."
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.output_head(x)

        loss = None

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=None,
    ):
        self.eval()
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1.")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least 1.")

        prompt_length = token_ids.size(1)

        for _ in range(max_new_tokens):
            # Keep only the most recent context window.
            context = token_ids[:, -CONTEXT_LENGTH:]

            logits, _ = self(context)

            # Predict the next token from the last position.
            logits = logits[:, -1, :]

            temperature = max(temperature, 0.05)
            logits = logits / max(temperature, 0.05)

            generated_tokens = token_ids[:, prompt_length:]
            for batch_index, sequence in enumerate(generated_tokens):
                for token_id in torch.unique(sequence):
                    token_index = int(token_id.item())
                    token_logit = logits[batch_index, token_index]
                    logits[batch_index, token_index] = (
                        token_logit / repetition_penalty
                        if token_logit > 0
                        else token_logit * repetition_penalty
                    )

            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                    dim=-1,
                )
                cumulative_probabilities = torch.softmax(
                    sorted_logits,
                    dim=-1,
                ).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probabilities > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove,
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probabilities = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            token_ids = torch.cat(
                [token_ids, next_token],
                dim=1,
            )
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return token_ids


if __name__ == "__main__":
    torch.manual_seed(42)

    model = QuantaLM(vocab_size=378)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("QuantaLM model initialized successfully.")
    print(f"Parameters: {total_parameters:,}")
    print("Device: CPU")

    # Test a forward pass.
    sample = torch.tensor([[2, 5, 20, 46, 225, 18]])

    logits, _ = model(sample)

    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(logits.shape))
