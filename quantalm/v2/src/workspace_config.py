"""Load the authoritative configuration adjacent to the V2 source package."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.py"
_SPEC = importlib.util.spec_from_file_location(
    "_quantalm_v2_workspace_config",
    _CONFIG_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load QuantaLM V2 configuration: {_CONFIG_PATH}")

v2_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v2_config)
