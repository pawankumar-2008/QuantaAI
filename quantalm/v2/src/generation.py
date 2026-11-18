"""Autoregressive sampling utilities for QuantaLM V2."""

from __future__ import annotations

import math
import numbers

import torch
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    seed: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be an integer tensor of shape [batch, sequence].")
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have a non-empty batch and sequence.")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if (
        not isinstance(temperature, numbers.Real)
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be greater than zero.")
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer or None.")
    if (
        not isinstance(top_p, numbers.Real)
        or isinstance(top_p, bool)
        or not math.isfinite(float(top_p))
        or not 0.0 < top_p <= 1.0
    ):
        raise ValueError("top_p must be finite and in (0, 1].")
    if (
        not isinstance(repetition_penalty, numbers.Real)
        or isinstance(repetition_penalty, bool)
        or not math.isfinite(float(repetition_penalty))
        or repetition_penalty < 1.0
    ):
        raise ValueError("repetition_penalty must be finite and at least 1.")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool)
    ):
        raise TypeError("seed must be an integer or None.")

    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "context_length"):
        raise TypeError("model must expose config.context_length.")
    context_length = config.context_length
    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError("model context_length must be positive.")
    if top_k is not None and vocab_size is not None and top_k > vocab_size:
        raise ValueError(f"top_k cannot exceed model vocabulary size {vocab_size}.")
    if eos_token_id is not None and (
        not isinstance(eos_token_id, int)
        or isinstance(eos_token_id, bool)
        or eos_token_id < 0
        or (vocab_size is not None and eos_token_id >= vocab_size)
    ):
        raise ValueError("eos_token_id must be a valid vocabulary token ID.")

    device = input_ids.device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    generated = input_ids.to(dtype=torch.long)
    finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=device)
    model_was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = generated[:, -context_length:]
            logits = model(context)
            if logits.ndim != 3 or logits.shape[:2] != context.shape:
                raise ValueError("model must return [batch, sequence, vocabulary] logits.")
            next_logits = logits[:, -1, :] / temperature
            if repetition_penalty > 1.0:
                for batch_index, sequence in enumerate(generated):
                    repeated_ids = torch.unique(sequence)
                    repeated_logits = next_logits[batch_index, repeated_ids]
                    next_logits[batch_index, repeated_ids] = torch.where(
                        repeated_logits > 0,
                        repeated_logits / repetition_penalty,
                        repeated_logits * repetition_penalty,
                    )
            if top_k is not None:
                k = min(top_k, next_logits.shape[-1])
                threshold = torch.topk(next_logits, k, dim=-1).values[:, -1:]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold,
                    torch.finfo(next_logits.dtype).min,
                )
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits,
                    descending=True,
                    dim=-1,
                )
                sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
                cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probabilities > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(
                    sorted_indices_to_remove
                ).scatter(-1, sorted_indices, sorted_indices_to_remove)
                next_logits = next_logits.masked_fill(
                    indices_to_remove,
                    torch.finfo(next_logits.dtype).min,
                )
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
            if eos_token_id is not None:
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished[:, None], eos_fill, next_token)
                finished |= next_token.squeeze(1).eq(eos_token_id)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
    finally:
        model.train(model_was_training)
    return generated
