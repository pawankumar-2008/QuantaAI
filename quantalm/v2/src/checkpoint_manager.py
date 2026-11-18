"""Atomic, safe, V2-only persistence for resumable training checkpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

try:
    from .workspace_config import v2_config
except ImportError:
    from workspace_config import v2_config


CHECKPOINT_ROOT = v2_config.CHECKPOINT_DIR.resolve()
CHECKPOINT_KINDS = {"latest", "best", "resume"}
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path | None = None) -> None:
        root = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINT_ROOT
        self.checkpoint_dir = root.resolve()
        if not self.checkpoint_dir.is_relative_to(CHECKPOINT_ROOT):
            raise ValueError("Training checkpoints must remain inside quantalm/v2/checkpoints/.")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, kind: str) -> Path:
        if kind not in CHECKPOINT_KINDS:
            raise ValueError(f"Checkpoint kind must be one of {sorted(CHECKPOINT_KINDS)}.")
        return self.checkpoint_dir / f"{kind}.pt"

    def save(self, kind: str, payload: dict[str, Any]) -> Path:
        target = self.path_for(kind)
        if "schema_version" in payload:
            raise ValueError("Checkpoint payload must not override schema_version.")
        envelope = {"schema_version": CHECKPOINT_SCHEMA_VERSION, **payload}
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{kind}-",
                suffix=".tmp",
                dir=self.checkpoint_dir,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            torch.save(envelope, temporary_path)
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return target

    def load(
        self,
        kind: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        target = self.path_for(kind)
        if not target.is_file():
            raise FileNotFoundError(f"Training checkpoint not found: {target}")
        payload = torch.load(target, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Training checkpoint has an unsupported or invalid schema.")
        required = {
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "epoch",
            "batch_in_epoch",
            "global_step",
            "training_config",
            "model_config",
            "best_validation_loss",
            "epochs_without_improvement",
            "random_seed",
            "tokenizer_hash",
            "dataset_manifest_hash",
            "torch_rng_state",
            "cuda_rng_state",
            "history",
            "partial_epoch",
            "validation_improved_this_epoch",
        }
        if set(payload) != required | {"schema_version"}:
            raise ValueError("Training checkpoint is missing or has unexpected fields.")
        return payload
