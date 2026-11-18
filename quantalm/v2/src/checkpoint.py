"""Safe V2-only checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .model import ModelConfig, QuantaLMV2
from .workspace_config import v2_config

ARCHITECTURE_VERSION = "2.0"
DEFAULT_CHECKPOINT_PATH = v2_config.CHECKPOINT_DIR / "quantalm_v2.pt"


def _validate_checkpoint_path(path: str | Path | None) -> Path:
    target = Path(path) if path is not None else DEFAULT_CHECKPOINT_PATH
    if not target.is_absolute():
        target = v2_config.CHECKPOINT_DIR / target
    target = target.resolve()
    checkpoint_root = v2_config.CHECKPOINT_DIR.resolve()
    if not target.is_relative_to(checkpoint_root):
        raise ValueError("Checkpoint files must be saved inside the V2 checkpoint directory.")
    if target.suffix not in {".pt", ".pth"}:
        raise ValueError("Checkpoint filename must end in .pt or .pth.")
    return target


def save_checkpoint(
    model: QuantaLMV2,
    path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    target = _validate_checkpoint_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists; pass overwrite=True to replace it: {target}"
        )

    payload = {
        "state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "vocab_size": model.config.vocab_size,
        "context_length": model.config.context_length,
        "architecture_version": ARCHITECTURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".quantalm-v2-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(payload, temporary_path)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Checkpoint appeared during save; refusing to replace it: {target}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def load_checkpoint(
    model: QuantaLMV2,
    path: str | Path | None = None,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    target = _validate_checkpoint_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"V2 checkpoint not found: {target}")

    payload = torch.load(
        target,
        map_location=map_location,
        weights_only=True,
    )
    required_keys = {
        "state_dict",
        "model_config",
        "vocab_size",
        "context_length",
        "architecture_version",
        "created_at",
    }
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ValueError("Checkpoint format is invalid or has missing/unexpected fields.")
    if payload["architecture_version"] != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Unsupported architecture version: {payload['architecture_version']}"
        )

    saved_config = ModelConfig.from_dict(payload["model_config"])
    if saved_config != model.config:
        raise ValueError(
            "Checkpoint model configuration is incompatible with the supplied model."
        )
    if payload["vocab_size"] != saved_config.vocab_size:
        raise ValueError("Checkpoint vocabulary metadata is inconsistent.")
    if payload["context_length"] != saved_config.context_length:
        raise ValueError("Checkpoint context metadata is inconsistent.")
    if not isinstance(payload["created_at"], str):
        raise ValueError("Checkpoint creation timestamp is invalid.")
    if not isinstance(payload["state_dict"], dict):
        raise ValueError("Checkpoint state_dict is invalid.")

    model.load_state_dict(payload["state_dict"], strict=strict)
    return {
        "model_config": saved_config.to_dict(),
        "vocab_size": payload["vocab_size"],
        "context_length": payload["context_length"],
        "architecture_version": payload["architecture_version"],
        "created_at": payload["created_at"],
    }
