"""Conservative, explicit defaults for CPU-first QuantaLM V2 training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

try:
    from .workspace_config import v2_config
except ImportError:
    from workspace_config import v2_config


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 10
    gradient_clip_norm: float = 1.0
    random_seed: int = 20261106
    evaluation_interval: int = 50
    checkpoint_interval: int = 100
    early_stopping_patience: int = 3
    device: str = "cpu"
    num_workers: int = 0
    min_learning_rate_ratio: float = 0.1

    def __post_init__(self) -> None:
        integer_fields = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
            "random_seed": self.random_seed,
            "evaluation_interval": self.evaluation_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "early_stopping_patience": self.early_stopping_patience,
            "num_workers": self.num_workers,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative.")
        if self.evaluation_interval < 1 or self.checkpoint_interval < 1:
            raise ValueError("Evaluation and checkpoint intervals must be positive.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")
        if self.num_workers != 0:
            raise ValueError("CPU-first training currently requires num_workers=0.")
        numeric_fields = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "min_learning_rate_ratio": self.min_learning_rate_ratio,
        }
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in numeric_fields.values()
        ):
            raise ValueError("Training numeric settings must be finite numbers.")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if (
            not isinstance(self.gradient_clip_norm, (int, float))
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be positive.")
        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be in [0, 1].")
        if not isinstance(self.device, str) or (
            self.device != "cpu" and not self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'cpu' or a CUDA device.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_TRAINING_CONFIG = TrainingConfig()
TRAINING_METRICS_DIR = v2_config.DATA_DIR / "reports" / "training"
