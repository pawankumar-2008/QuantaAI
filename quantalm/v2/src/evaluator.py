"""Reproducible, V2-only evaluation and benchmarking utilities."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

_V2_ROOT = Path(__file__).resolve().parents[1]
if __package__:
    from .bpe_tokenizer import QuantaBPETokenizer
    from .generation import generate
    from .model import ModelConfig, QuantaLMV2
    from .workspace_config import v2_config
else:
    sys.path.insert(0, str(_V2_ROOT))
    from src.bpe_tokenizer import QuantaBPETokenizer
    from src.generation import generate
    from src.model import ModelConfig, QuantaLMV2
    from src.workspace_config import v2_config


V2_ROOT = _V2_ROOT
DATA_DIR = v2_config.DATA_DIR
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"
EVALUATION_REPORT_DIR = DATA_DIR / "reports" / "evaluation"
DATASET_MANIFEST_PATH = DATA_DIR / "manifests" / "dataset_manifest.json"
SPLITS = ("validation", "test")
EVALUATION_CATEGORIES = (
    "Computer Science",
    "Artificial Intelligence",
    "Mathematics",
    "Science",
    "General Knowledge",
    "Student Learning",
)
QUESTION_FIELDS = {
    "question_id",
    "category",
    "question",
    "reference_answer",
    "acceptable_key_points",
    "source_document_id",
    "source_document_hash",
    "source_text_reference",
    "difficulty",
    "human_verified",
}
HUMAN_SCORE_FIELDS = (
    "factual_accuracy_score",
    "relevance_score",
    "coherence_score",
    "instruction_following_score",
)
HUMAN_REVIEW_WEIGHTS = {
    "factual_accuracy_percentage": 0.35,
    "relevance_percentage": 0.25,
    "coherence_percentage": 0.15,
    "instruction_following_percentage": 0.15,
    "repetition_quality_percentage": 0.10,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Evaluation data file not found: {source_path}")
    records: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {source_path}:{line_number}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {source_path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record at {source_path}:{line_number} must be an object."
                )
            records.append(record)
    return records


def validate_evaluation_record(record: dict[str, Any]) -> None:
    missing = QUESTION_FIELDS - set(record)
    extra = set(record) - QUESTION_FIELDS
    if missing or extra:
        raise ValueError(
            f"Evaluation record fields mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(extra)}."
        )
    for field in (
        "question_id",
        "category",
        "question",
        "reference_answer",
        "source_document_id",
        "source_text_reference",
        "difficulty",
    ):
        value = record[field]
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string.")
    if not record["question_id"].strip():
        raise ValueError("question_id must not be empty.")
    if record["category"] not in EVALUATION_CATEGORIES:
        raise ValueError(f"Unsupported evaluation category: {record['category']!r}.")
    if not record["question"].strip():
        raise ValueError("question must not be empty.")
    if not record["source_document_id"].strip():
        raise ValueError("source_document_id must not be empty.")
    if not record["source_text_reference"].strip():
        raise ValueError("source_text_reference must not be empty.")
    if not record["difficulty"].strip():
        raise ValueError("difficulty must not be empty.")
    if not _is_sha256(record["source_document_hash"]):
        raise ValueError("source_document_hash must be a 64-character SHA-256 digest.")
    key_points = record["acceptable_key_points"]
    if not isinstance(key_points, list) or any(
        not isinstance(point, str) for point in key_points
    ):
        raise ValueError("acceptable_key_points must be a list of strings.")
    if not isinstance(record["human_verified"], bool):
        raise ValueError("human_verified must be a boolean.")
    if record["human_verified"] and not record["reference_answer"].strip():
        raise ValueError("A human-verified question must have a reference_answer.")


def load_evaluation_dataset(path: str | Path) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    seen_ids: set[str] = set()
    for record in records:
        validate_evaluation_record(record)
        question_id = record["question_id"]
        if question_id in seen_ids:
            raise ValueError(f"Duplicate evaluation question_id: {question_id}.")
        seen_ids.add(question_id)
    return records


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Answers must be strings.")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def lexical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE)


def reference_metrics(generated: str, reference: str) -> dict[str, float]:
    """Return per-answer exact match and multiset lexical precision/recall/F1."""
    generated_normalized = normalize_answer(generated)
    reference_normalized = normalize_answer(reference)
    generated_tokens = lexical_tokens(generated)
    reference_tokens = lexical_tokens(reference)
    generated_counts = Counter(generated_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum((generated_counts & reference_counts).values())
    precision = overlap / len(generated_tokens) if generated_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "exact_match": float(generated_normalized == reference_normalized),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def aggregate_reference_metrics(
    results: Sequence[dict[str, float]],
) -> dict[str, float | int | None]:
    if not results:
        return {
            "answer_count": 0,
            "exact_match_percentage": None,
            "token_precision_percentage": None,
            "token_recall_percentage": None,
            "token_f1_percentage": None,
        }
    keys = (
        ("exact_match", "exact_match_percentage"),
        ("token_precision", "token_precision_percentage"),
        ("token_recall", "token_recall_percentage"),
        ("token_f1", "token_f1_percentage"),
    )
    return {
        "answer_count": len(results),
        **{
            output_key: 100.0
            * statistics.fmean(result[input_key] for result in results)
            for input_key, output_key in keys
        },
    }


def calculate_perplexity(
    model: nn.Module,
    records: Iterable[dict[str, Any]],
    *,
    device: str | torch.device = "cpu",
    vocab_size: int = v2_config.VOCAB_SIZE,
    context_length: int = v2_config.CONTEXT_LENGTH,
    pad_token_id: int = v2_config.PAD_TOKEN_ID,
) -> dict[str, Any]:
    """Measure masked, token-weighted causal NLL on pre-shifted chunk targets."""
    target_device = torch.device(device)
    was_training = model.training
    model.eval()
    total_nll = 0.0
    evaluated_tokens = 0
    chunk_count = 0
    try:
        with torch.no_grad():
            for record_index, record in enumerate(records):
                input_ids = record.get("input_ids")
                target_ids = record.get("target_ids")
                loss_mask = record.get("loss_mask")
                if not all(isinstance(value, list) for value in (input_ids, target_ids, loss_mask)):
                    raise ValueError(f"Chunk {record_index} has invalid token arrays.")
                if not input_ids or len(input_ids) != len(target_ids) or len(input_ids) != len(loss_mask):
                    raise ValueError(f"Chunk {record_index} has inconsistent token lengths.")
                if len(input_ids) > context_length:
                    raise ValueError(f"Chunk {record_index} exceeds model context length.")
                for position, mask in enumerate(loss_mask):
                    if not isinstance(mask, (int, bool)) or mask not in (0, 1):
                        raise ValueError(f"Chunk {record_index} has an invalid loss mask.")
                    ids = (input_ids[position], target_ids[position])
                    if mask:
                        if any(
                            not isinstance(token, int)
                            or isinstance(token, bool)
                            or token < 0
                            or token >= vocab_size
                            for token in ids
                        ):
                            raise ValueError(
                                f"Chunk {record_index} contains invalid token IDs at valid positions."
                            )
                    elif ids != (pad_token_id, pad_token_id):
                        raise ValueError(
                            f"Chunk {record_index} has non-PAD masked positions."
                        )
                mask_tensor = torch.tensor(loss_mask, dtype=torch.bool, device=target_device)
                valid_count = int(mask_tensor.sum().item())
                if valid_count == 0:
                    raise ValueError(f"Chunk {record_index} has no valid targets.")
                model_input = torch.tensor(
                    [input_ids], dtype=torch.long, device=target_device
                )
                targets = torch.tensor(
                    target_ids, dtype=torch.long, device=target_device
                )
                logits = model(model_input)
                if (
                    logits.ndim != 3
                    or logits.shape[:2] != model_input.shape
                    or logits.shape[-1] != vocab_size
                ):
                    raise ValueError("Model logits must have shape [batch, sequence, vocabulary].")
                selected_logits = logits[0][mask_tensor]
                selected_targets = targets[mask_tensor]
                nll_sum = F.cross_entropy(
                    selected_logits,
                    selected_targets,
                    reduction="sum",
                )
                if not torch.isfinite(nll_sum):
                    raise FloatingPointError("Perplexity NLL is not finite.")
                total_nll += float(nll_sum.item())
                evaluated_tokens += valid_count
                chunk_count += 1
    finally:
        model.train(was_training)
    if evaluated_tokens == 0:
        raise ValueError("Perplexity evaluation contains no valid target tokens.")
    average_nll = total_nll / evaluated_tokens
    overflow = average_nll >= math.log(sys.float_info.max)
    perplexity = None if overflow else math.exp(average_nll)
    return {
        "available": True,
        "average_negative_log_likelihood": average_nll,
        "perplexity": perplexity,
        "perplexity_overflow": overflow,
        "evaluated_token_count": evaluated_tokens,
        "chunk_count": chunk_count,
        "target_alignment": "targets are already shifted; no second shift is applied",
        "padding_policy": "loss_mask positions only; masked padding is excluded",
        "overlap_policy": "overlapping validation chunk targets are counted per chunk",
    }


def generation_quality(
    generated_token_ids: Sequence[int],
    decoded_answer: str,
    *,
    eos_token_id: int = v2_config.EOS_TOKEN_ID,
    vocab_size: int = v2_config.VOCAB_SIZE,
    tokenizer: QuantaBPETokenizer | None = None,
) -> dict[str, Any]:
    all_ids = list(generated_token_ids)
    eos_terminated = bool(all_ids and all_ids[-1] == eos_token_id)
    answer_ids = all_ids[:-1] if eos_terminated else all_ids
    valid_ids = [
        token_id
        for token_id in answer_ids
        if isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and 0 <= token_id < vocab_size
    ]
    invalid_count = len(answer_ids) - len(valid_ids)
    token_counts = Counter(valid_ids)
    repeated_token_count = sum(count - 1 for count in token_counts.values())
    ngram_counts: Counter[tuple[int, ...]] = Counter()
    total_ngrams = 0
    for n in (2, 3, 4):
        ngrams = [
            tuple(valid_ids[index : index + n])
            for index in range(max(0, len(valid_ids) - n + 1))
        ]
        ngram_counts.update(ngrams)
        total_ngrams += len(ngrams)
    repeated_ngram_count = sum(count - 1 for count in ngram_counts.values() if count > 1)
    repeated_patterns = []
    for pattern, count in sorted(
        ngram_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if count <= 1:
            continue
        repeated_patterns.append(
            {
                "token_ids": list(pattern),
                "text": tokenizer.decode(list(pattern)) if tokenizer else None,
                "count": count,
            }
        )
    answer_length = len(valid_ids)
    return {
        "answer_length_tokens": answer_length,
        "repeated_token_ratio": (
            repeated_token_count / answer_length if answer_length else 0.0
        ),
        "repeated_ngram_ratio": (
            repeated_ngram_count / total_ngrams if total_ngrams else 0.0
        ),
        "empty_response": not decoded_answer.strip(),
        "eos_terminated": eos_terminated,
        "invalid_token_count": invalid_count,
        "invalid_token_rate": invalid_count / len(answer_ids) if answer_ids else 0.0,
        "repetition_patterns": repeated_patterns,
    }


def aggregate_generation_quality(
    samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "answer_length_tokens_mean": None,
            "repeated_token_ratio_mean": None,
            "repeated_ngram_ratio_mean": None,
            "empty_response_rate_percentage": None,
            "eos_termination_rate_percentage": None,
            "invalid_token_rate_percentage": None,
            "repetition_quality_percentage": None,
            "repetition_patterns": [],
        }
    nonempty = [sample for sample in samples if not sample["quality"]["empty_response"]]
    pattern_counts: Counter[str] = Counter()
    for sample in samples:
        for pattern in sample["quality"]["repetition_patterns"]:
            pattern_counts[pattern["text"] or str(pattern["token_ids"])] += (
                pattern["count"] - 1
            )
    repeated_ngram_ratio = statistics.fmean(
        sample["quality"]["repeated_ngram_ratio"] for sample in samples
    )
    return {
        "sample_count": len(samples),
        "answer_length_tokens_mean": statistics.fmean(
            sample["quality"]["answer_length_tokens"] for sample in samples
        ),
        "repeated_token_ratio_mean": statistics.fmean(
            sample["quality"]["repeated_token_ratio"] for sample in samples
        ),
        "repeated_ngram_ratio_mean": repeated_ngram_ratio,
        "empty_response_rate_percentage": 100.0
        * statistics.fmean(float(sample["quality"]["empty_response"]) for sample in samples),
        "eos_termination_rate_percentage": 100.0
        * statistics.fmean(float(sample["quality"]["eos_terminated"]) for sample in samples),
        "invalid_token_rate_percentage": 100.0
        * statistics.fmean(sample["quality"]["invalid_token_rate"] for sample in samples),
        "repetition_quality_percentage": (
            100.0 * (1.0 - repeated_ngram_ratio) if nonempty else None
        ),
        "repetition_patterns": [
            {"text": pattern, "repeated_occurrences": count}
            for pattern, count in pattern_counts.most_common(20)
        ],
    }


def validate_human_review(record: dict[str, Any]) -> None:
    required = {
        "question_id",
        *HUMAN_SCORE_FIELDS,
        "explanation",
        "reviewer_id",
        "reviewed",
    }
    if set(record) != required:
        raise ValueError(
            f"Human-review fields mismatch; missing={sorted(required - set(record))}, "
            f"unexpected={sorted(set(record) - required)}."
        )
    if not isinstance(record["question_id"], str) or not record["question_id"].strip():
        raise ValueError("question_id must be a non-empty string.")
    if not isinstance(record["reviewed"], bool):
        raise ValueError("reviewed must be a boolean.")
    if not record["reviewed"]:
        raise ValueError("Only completed human reviews can be validated.")
    for field in HUMAN_SCORE_FIELDS:
        score = record[field]
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            raise ValueError(f"{field} must be an integer score from 1 to 5.")
    if not isinstance(record["explanation"], str) or not record["explanation"].strip():
        raise ValueError("A completed review requires an explanation.")
    if not isinstance(record["reviewer_id"], str) or not record["reviewer_id"].strip():
        raise ValueError("A completed review requires a reviewer_id.")


def summarize_human_reviews(
    reviews: Iterable[dict[str, Any]],
    *,
    valid_question_ids: set[str] | None = None,
) -> dict[str, Any]:
    score_names = {
        "factual_accuracy_score": "factual_accuracy_percentage",
        "relevance_score": "relevance_percentage",
        "coherence_score": "coherence_percentage",
        "instruction_following_score": "instruction_following_percentage",
    }
    completed: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    incomplete_count = 0
    seen_reviewers: set[tuple[str, str]] = set()
    for index, record in enumerate(reviews):
        if not isinstance(record, dict):
            invalid.append({"record_index": str(index), "reason": "record is not an object"})
            continue
        if record.get("reviewed") is False:
            incomplete_count += 1
            continue
        try:
            validate_human_review(record)
            if valid_question_ids is not None and record["question_id"] not in valid_question_ids:
                raise ValueError("question_id does not exist in this evaluation dataset.")
            reviewer_key = (record["question_id"], record["reviewer_id"])
            if reviewer_key in seen_reviewers:
                raise ValueError("duplicate review by the same reviewer for this question.")
        except ValueError as error:
            invalid.append(
                {
                    "record_index": str(index),
                    "question_id": str(record.get("question_id", "")),
                    "reason": str(error),
                }
            )
            continue
        seen_reviewers.add(reviewer_key)
        completed.append(record)
    percentages: dict[str, float | None] = {}
    for score_field, output_field in score_names.items():
        percentages[output_field] = (
            25.0
            * statistics.fmean(record[score_field] - 1 for record in completed)
            if completed
            else None
        )
    return {
        "valid_review_count": len(completed),
        "incomplete_review_count": incomplete_count,
        "invalid_review_count": len(invalid),
        "invalid_reviews": invalid,
        "score_conversion": "(mean score - 1) / 4 * 100",
        "scores": percentages,
    }


def weighted_human_quality_score(
    human_summary: dict[str, Any],
    repetition_quality_percentage: float | None,
) -> dict[str, Any]:
    if human_summary.get("valid_review_count", 0) <= 0:
        return {
            "available": False,
            "overall_human_quality_percentage": None,
            "components": {},
            "effective_weights": {},
            "reason": "No valid completed human reviews are available.",
        }
    components = dict(human_summary["scores"])
    components["repetition_quality_percentage"] = repetition_quality_percentage
    available = {
        name: value
        for name, value in components.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }
    if not available:
        return {
            "available": False,
            "overall_human_quality_percentage": None,
            "components": {},
            "effective_weights": {},
            "reason": "No weighted quality components are available.",
        }
    weight_total = sum(HUMAN_REVIEW_WEIGHTS[name] for name in available)
    effective_weights = {
        name: HUMAN_REVIEW_WEIGHTS[name] / weight_total for name in available
    }
    score = sum(
        available[name] * effective_weights[name]
        for name in available
    )
    return {
        "available": True,
        "overall_human_quality_percentage": score,
        "components": available,
        "effective_weights": effective_weights,
        "reason": None,
    }


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int | None = 40
    random_seed: int = 20261106
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer.")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and greater than zero.")
        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None.")
        if (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
            or not math.isfinite(self.top_p)
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError("top_p must be finite and in (0, 1].")
        if (
            not isinstance(self.repetition_penalty, (int, float))
            or isinstance(self.repetition_penalty, bool)
            or not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty < 1.0
        ):
            raise ValueError("repetition_penalty must be finite and at least 1.")
        if (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer.")


def _memory_snapshot() -> tuple[int | None, str | None]:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if success:
            return int(counters.PeakWorkingSetSize), "Windows process peak working set"
        return None, f"GetProcessMemoryInfo failed with Win32 error {ctypes.get_last_error()}."
    try:
        import resource
    except ImportError:
        return None, "Peak process RSS is unavailable on this platform."
    usage = resource.getrusage(resource.RUSAGE_SELF)
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(usage.ru_maxrss * multiplier), "process peak RSS"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_information(device: str | torch.device) -> dict[str, Any]:
    selected = torch.device(device)
    info: dict[str, Any] = {
        "device": str(selected),
        "device_type": selected.type,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_execution_verified": False,
    }
    if selected.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(selected)
        info["gpu_execution_verified"] = True
    return info


def benchmark_generation(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 8,
    temperature: float = 0.8,
    top_k: int | None = 40,
    random_seed: int = 20261106,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    if not isinstance(warmup_runs, int) or isinstance(warmup_runs, bool) or warmup_runs < 0:
        raise ValueError("warmup_runs must be a non-negative integer.")
    if not isinstance(measured_runs, int) or isinstance(measured_runs, bool) or measured_runs < 1:
        raise ValueError("measured_runs must be a positive integer.")
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError("Benchmark prompt must have shape [1, sequence] and be non-empty.")
    model_device = next(model.parameters()).device
    prompt = prompt_ids.to(device=model_device, dtype=torch.long)
    if prompt.shape[1] > model.config.context_length:
        raise ValueError("Benchmark prompt exceeds model context length.")
    settings = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        random_seed=random_seed,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    if settings.top_k is not None and settings.top_k > model.config.vocab_size:
        raise ValueError("top_k cannot exceed model vocabulary size.")
    was_training = model.training
    model.eval()
    try:
        for warmup_index in range(warmup_runs):
            generate(
                model,
                prompt,
                settings.max_new_tokens,
                settings.temperature,
                settings.top_k,
                eos_token_id=v2_config.EOS_TOKEN_ID,
                seed=settings.random_seed + warmup_index,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
            )
        full_durations: list[float] = []
        first_token_durations: list[float] = []
        generated_counts: list[int] = []
        for run_index in range(measured_runs):
            _synchronize(model_device)
            started = time.perf_counter()
            one_token = generate(
                model,
                prompt,
                1,
                settings.temperature,
                settings.top_k,
                eos_token_id=v2_config.EOS_TOKEN_ID,
                seed=settings.random_seed + run_index,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
            )
            _synchronize(model_device)
            first_token_durations.append(time.perf_counter() - started)
            _synchronize(model_device)
            started = time.perf_counter()
            generated = generate(
                model,
                prompt,
                settings.max_new_tokens,
                settings.temperature,
                settings.top_k,
                eos_token_id=v2_config.EOS_TOKEN_ID,
                seed=settings.random_seed + run_index,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
            )
            _synchronize(model_device)
            full_durations.append(time.perf_counter() - started)
            generated_counts.append(int(generated.shape[1] - prompt.shape[1]))
            if one_token.shape[1] != prompt.shape[1] + 1:
                raise RuntimeError("One-token benchmark did not generate exactly one token.")
    finally:
        model.train(was_training)
    total_duration = sum(full_durations)
    total_tokens = sum(generated_counts)
    peak_memory, memory_method = _memory_snapshot()
    sorted_full = sorted(full_durations)
    p95_index = max(0, math.ceil(0.95 * len(sorted_full)) - 1)
    return {
        "prompt_source": "single_BOS_performance_probe; not an evaluation question or answer",
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "generation_parameters": asdict(settings),
        "first_token_latency_seconds_mean": statistics.fmean(first_token_durations),
        "total_generation_time_seconds_mean": statistics.fmean(full_durations),
        "total_generation_time_seconds_median": statistics.median(full_durations),
        "total_generation_time_seconds_p95": sorted_full[p95_index],
        "generated_token_count_mean": statistics.fmean(generated_counts),
        "generated_token_count_total": total_tokens,
        "tokens_per_second": total_tokens / total_duration if total_duration else None,
        "average_response_latency_seconds": statistics.fmean(full_durations),
        "peak_cpu_memory_bytes": peak_memory,
        "peak_cpu_memory_method": memory_method,
        "device_information": device_information(model_device),
    }


def _tokenizer_artifact_hashes(tokenizer_dir: Path) -> tuple[str, dict[str, str]]:
    names = ("bpe_vocab.json", "bpe_merges.json", "tokenizer_config.json")
    hashes = {name: sha256_file(tokenizer_dir / name) for name in names}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), hashes


def _model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_checkpoint_path(path: str | Path) -> Path:
    target = Path(path).resolve()
    if not target.is_relative_to(v2_config.CHECKPOINT_DIR.resolve()):
        raise ValueError("Evaluation checkpoints must remain inside quantalm/v2/checkpoints/.")
    if not target.is_file():
        raise FileNotFoundError(f"V2 evaluation checkpoint not found: {target}")
    return target


def _load_checkpoint_state(
    model: QuantaLMV2,
    path: Path,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("V2 checkpoint payload must be a dictionary.")
    saved_config_values = payload.get("model_config")
    if not isinstance(saved_config_values, dict):
        raise ValueError("V2 checkpoint is missing model_config.")
    saved_config = ModelConfig.from_dict(saved_config_values)
    if saved_config != model.config:
        raise ValueError("Checkpoint model configuration is incompatible with QuantaLM V2.")
    state_dict = payload.get("state_dict", payload.get("model_state"))
    if not isinstance(state_dict, dict):
        raise ValueError("V2 checkpoint is missing a model state dictionary.")
    if "model_state" in payload:
        tokenizer_hash, _ = _tokenizer_artifact_hashes(v2_config.TOKENIZER_DIR)
        if payload.get("tokenizer_hash") != tokenizer_hash:
            raise ValueError("V2 training checkpoint tokenizer hash is incompatible.")
    model.load_state_dict(state_dict, strict=True)
    training_step = payload.get("global_step")
    trained = (
        "model_state" in payload
        and isinstance(training_step, int)
        and not isinstance(training_step, bool)
        and training_step > 0
    )
    return {
        "checkpoint_kind": (
            "model_checkpoint" if "state_dict" in payload else "training_checkpoint"
        ),
        "created_at": payload.get("created_at"),
        "training_status": "trained" if trained else "unverified",
        "global_step": training_step,
    }


def _evaluation_source_documents(split: str) -> dict[str, dict[str, str]]:
    path = DATA_DIR / "splits" / split / "dataset.jsonl"
    records = _read_jsonl(path)
    documents: dict[str, dict[str, str]] = {}
    for record in records:
        document_id = record.get("document_id")
        source_hash = record.get("source_sha256")
        if not isinstance(document_id, str) or not _is_sha256(source_hash):
            raise ValueError(f"Invalid source document provenance in {path}.")
        if document_id in documents:
            raise ValueError(f"Duplicate source document ID in {path}: {document_id}.")
        documents[document_id] = {
            "source_sha256": source_hash,
            "source_path": str(record.get("source_path", "")),
        }
    return documents


def _load_chunk_records(split: str, *, allow_test_evaluation: bool) -> tuple[list[dict[str, Any]], str]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}.")
    if split == "test" and not allow_test_evaluation:
        raise PermissionError(
            "Held-out test evaluation is disabled by default; "
            "pass allow_test_evaluation=True for an approved final run."
        )
    with DATASET_MANIFEST_PATH.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    split_info = manifest.get("chunking", {}).get("splits", {}).get(split)
    if not isinstance(split_info, dict):
        raise ValueError(f"Dataset manifest has no {split} chunk metadata.")
    chunks_path = DATA_DIR / "chunks" / split / "chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")
    chunk_hash = sha256_file(chunks_path)
    if chunk_hash != split_info.get("chunks_sha256"):
        raise ValueError(f"{split} chunk file hash differs from the dataset manifest.")
    records = _read_jsonl(chunks_path)
    if len(records) != split_info.get("chunk_count"):
        raise ValueError(f"{split} chunk count differs from the dataset manifest.")
    source_documents = _evaluation_source_documents(split)
    observed_documents: set[str] = set()
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}
    tokenizer_hash = manifest.get("chunking", {}).get("tokenizer_sha256")
    for index, record in enumerate(records):
        document_id = record.get("document_id")
        if document_id not in source_documents:
            raise ValueError(f"{split} chunk {index} references a document outside its split.")
        if record.get("source_document_hash") != source_documents[document_id]["source_sha256"]:
            raise ValueError(f"{split} chunk {index} source hash does not match the source split.")
        if record.get("tokenizer_hash") != tokenizer_hash:
            raise ValueError(f"{split} chunk {index} tokenizer hash differs from the manifest.")
        input_ids = record.get("input_ids")
        target_ids = record.get("target_ids")
        loss_mask = record.get("loss_mask")
        if (
            not isinstance(input_ids, list)
            or not isinstance(target_ids, list)
            or not isinstance(loss_mask, list)
            or not input_ids
            or len(input_ids) != len(target_ids)
            or len(input_ids) != len(loss_mask)
            or len(input_ids) > v2_config.CONTEXT_LENGTH
        ):
            raise ValueError(f"{split} chunk {index} has incompatible sequence dimensions.")
        if input_ids[0] != v2_config.BOS_TOKEN_ID and record.get("is_first_chunk"):
            raise ValueError(f"{split} first chunk for {document_id} does not begin with BOS.")
        valid_positions = [
            position
            for position, mask in enumerate(loss_mask)
            if mask in (1, True)
        ]
        if not valid_positions:
            raise ValueError(f"{split} chunk {index} has no valid target positions.")
        if (
            record.get("is_final_chunk")
            and target_ids[valid_positions[-1]] != v2_config.EOS_TOKEN_ID
        ):
            raise ValueError(f"{split} final chunk for {document_id} does not target EOS.")
        observed_documents.add(document_id)
        chunks_by_document.setdefault(document_id, []).append(record)
    if len(observed_documents) != split_info.get("source_document_count"):
        raise ValueError(f"{split} chunk document count differs from the manifest.")
    for document_id, document_chunks in chunks_by_document.items():
        document_chunks.sort(key=lambda chunk: chunk.get("chunk_id", -1))
        if (
            not document_chunks[0].get("is_first_chunk")
            or not document_chunks[-1].get("is_final_chunk")
        ):
            raise ValueError(f"{split} chunk flags do not delimit {document_id}.")
    return records, chunk_hash


class Evaluator:
    """Load an untrained or trained V2 model and run explicitly selected evaluation."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        random_seed: int = 20261106,
        generation_config: GenerationConfig | None = None,
        model: QuantaLMV2 | None = None,
        tokenizer: QuantaBPETokenizer | None = None,
    ) -> None:
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("Evaluation device must be 'cpu' or a CUDA device.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
        self.device = torch.device(device)
        self.random_seed = random_seed
        self.generation_config = generation_config or GenerationConfig(random_seed=random_seed)
        model_started = time.perf_counter()
        self.checkpoint_path = (
            _validate_checkpoint_path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )
        checkpoint_payload = (
            torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if self.checkpoint_path is not None
            else None
        )
        checkpoint_model_config = None
        if checkpoint_payload is not None:
            values = checkpoint_payload.get("model_config")
            if not isinstance(values, dict):
                raise ValueError("V2 checkpoint is missing model_config.")
            checkpoint_model_config = ModelConfig.from_dict(values)
        if model is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(random_seed)
                model = QuantaLMV2(checkpoint_model_config)
        self.model = model
        if (
            self.model.config.vocab_size != v2_config.VOCAB_SIZE
            or self.model.config.context_length != v2_config.CONTEXT_LENGTH
        ):
            raise ValueError("Evaluation model vocabulary/context differs from V2 configuration.")
        self.checkpoint_metadata: dict[str, Any] | None = None
        self.checkpoint_sha256: str | None = None
        if self.checkpoint_path is not None:
            self.checkpoint_metadata = _load_checkpoint_state(
                self.model,
                self.checkpoint_path,
                checkpoint_payload,
            )
            self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.model_loading_time_seconds = time.perf_counter() - model_started
        self.tokenizer = tokenizer or QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)
        if self.tokenizer.vocab_size != v2_config.VOCAB_SIZE:
            raise ValueError("V2 tokenizer vocabulary differs from model configuration.")
        self.tokenizer_hash, self.tokenizer_artifact_hashes = _tokenizer_artifact_hashes(
            v2_config.TOKENIZER_DIR
        )
        self.model_state_sha256 = _model_state_hash(self.model)
        self.model_label = (
            "untrained_baseline"
            if self.checkpoint_path is None
            else (
                "trained_checkpoint"
                if self.checkpoint_metadata
                and self.checkpoint_metadata["training_status"] == "trained"
                else "checkpoint_training_status_unverified"
            )
        )

    def _load_questions(self, split: str) -> tuple[list[dict[str, Any]], Path]:
        dataset_path = EVALUATION_DATA_DIR / f"{split}.jsonl"
        questions = load_evaluation_dataset(dataset_path)
        source_documents = _evaluation_source_documents(split)
        for record in questions:
            source = source_documents.get(record["source_document_id"])
            if source is None:
                raise ValueError(
                    f"Question {record['question_id']} does not reference a document "
                    f"from the {split} split."
                )
            if record["source_document_hash"] != source["source_sha256"]:
                raise ValueError(
                    f"Question {record['question_id']} source hash is inconsistent."
                )
            source_path = source["source_path"]
            if record["source_text_reference"] not in (
                source_path,
                source_path + "#",
            ) and not record["source_text_reference"].startswith(source_path + "#"):
                raise ValueError(
                    f"Question {record['question_id']} source_text_reference does not "
                    "match its document source path."
                )
        return questions, dataset_path

    def _generate_answer(
        self,
        question: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        encoded = self.tokenizer.encode(question["question"])
        prompt = [v2_config.BOS_TOKEN_ID, *encoded]
        context_length = self.model.config.context_length
        prompt_truncated = len(prompt) > context_length
        if prompt_truncated:
            prompt = [v2_config.BOS_TOKEN_ID, *prompt[-(context_length - 1) :]]
        if any(token_id < 0 or token_id >= self.model.config.vocab_size for token_id in prompt):
            raise ValueError(f"Question {question['question_id']} encoded out-of-range token IDs.")
        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        question_seed = int.from_bytes(
            hashlib.sha256(question["question_id"].encode("utf-8")).digest()[:4],
            "big",
        )
        seed = (self.generation_config.random_seed + question_seed + index) % (2**31)
        generated = generate(
            self.model,
            input_ids,
            self.generation_config.max_new_tokens,
            self.generation_config.temperature,
            self.generation_config.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            seed=seed,
            top_p=self.generation_config.top_p,
            repetition_penalty=self.generation_config.repetition_penalty,
        )
        answer_ids = generated[0, len(prompt) :].detach().cpu().tolist()
        invalid_ids = [
            token_id
            for token_id in answer_ids
            if token_id < 0 or token_id >= self.model.config.vocab_size
        ]
        safe_ids = [
            token_id
            for token_id in answer_ids
            if 0 <= token_id < self.model.config.vocab_size
        ]
        answer = self.tokenizer.decode(safe_ids)
        quality = generation_quality(
            answer_ids,
            answer,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.model.config.vocab_size,
            tokenizer=self.tokenizer,
        )
        if invalid_ids:
            quality["invalid_token_count"] = len(invalid_ids)
            quality["invalid_token_rate"] = len(invalid_ids) / len(answer_ids)
        sample: dict[str, Any] = {
            "question_id": question["question_id"],
            "category": question["category"],
            "question": question["question"],
            "answer": answer,
            "human_verified_reference": question["human_verified"],
            "prompt_token_count": len(prompt),
            "prompt_truncated": prompt_truncated,
            "generated_token_ids": answer_ids,
            "quality": quality,
        }
        if question["human_verified"]:
            sample["reference_answer"] = question["reference_answer"]
            sample["reference_metrics"] = reference_metrics(
                answer,
                question["reference_answer"],
            )
        return sample

    def evaluate(
        self,
        split: str = "validation",
        *,
        allow_test_evaluation: bool = False,
        benchmark_warmup_runs: int = 1,
        benchmark_measured_runs: int = 3,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}.")
        if split == "test" and not allow_test_evaluation:
            raise PermissionError(
                "Held-out test evaluation is disabled by default. "
                "Pass allow_test_evaluation=True only for an explicitly approved final run."
            )
        questions, evaluation_path = self._load_questions(split)
        samples = [
            self._generate_answer(question, index)
            for index, question in enumerate(questions)
        ]
        verified_metrics = [
            sample["reference_metrics"]
            for sample in samples
            if sample.get("human_verified_reference")
        ]
        reference_summary = aggregate_reference_metrics(verified_metrics)
        generation_summary = aggregate_generation_quality(samples)

        chunk_records, chunk_hash = _load_chunk_records(
            split,
            allow_test_evaluation=allow_test_evaluation,
        )
        perplexity = calculate_perplexity(
            self.model,
            chunk_records,
            device=self.device,
        )
        reviews_path = EVALUATION_DATA_DIR / "human_reviews.jsonl"
        reviews = _read_jsonl(reviews_path) if reviews_path.is_file() else []
        human_summary = summarize_human_reviews(
            reviews,
            valid_question_ids={question["question_id"] for question in questions},
        )
        overall_score = weighted_human_quality_score(
            human_summary,
            generation_summary["repetition_quality_percentage"],
        )
        benchmark_prompt = torch.tensor(
            [[self.tokenizer.bos_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        performance = benchmark_generation(
            self.model,
            benchmark_prompt,
            max_new_tokens=min(8, self.generation_config.max_new_tokens),
            temperature=self.generation_config.temperature,
            top_k=self.generation_config.top_k,
            random_seed=self.generation_config.random_seed,
            top_p=self.generation_config.top_p,
            repetition_penalty=self.generation_config.repetition_penalty,
            warmup_runs=benchmark_warmup_runs,
            measured_runs=benchmark_measured_runs,
        )
        performance["model_loading_time_seconds"] = self.model_loading_time_seconds
        performance["tokenizer_hash"] = self.tokenizer_hash
        performance["checkpoint_sha256"] = self.checkpoint_sha256
        performance["model_state_sha256"] = self.model_state_sha256

        limitations = [
            "No human-verified evaluation questions are currently available; reference-based metrics are unavailable.",
            "The local corpus has 13 unique documents, with only one validation and one held-out test document.",
            "The dataset quality report marks source licenses as unknown and does not establish accuracy or reuse rights.",
            "General Knowledge has no source documents; category coverage is incomplete and imbalanced.",
            (
                "The untrained baseline is randomly initialized and its generations do not represent trained QuantaLM performance."
                if self.checkpoint_path is None
                else (
                    "The checkpoint has no recorded training step; training status is unverified."
                    if self.model_label == "checkpoint_training_status_unverified"
                    else "Results describe only the selected trained checkpoint and the small, imbalanced evaluation corpus."
                )
            ),
            "Lexical overlap and perplexity do not establish semantic correctness or factual accuracy.",
            "The performance probe uses a single BOS token and is not an answer-quality evaluation.",
        ]
        if not torch.cuda.is_available():
            limitations.append("CUDA was unavailable; GPU execution and GPU performance are unverified.")
        skipped = []
        if not verified_metrics:
            skipped.append(
                {
                    "evaluation": "reference_metrics",
                    "reason": "No human-verified reference answers are present.",
                }
            )
        if human_summary["valid_review_count"] == 0:
            skipped.append(
                {
                    "evaluation": "overall_human_quality_score",
                    "reason": "No valid completed human reviews are present.",
                }
            )
        valid_answers = sum(bool(sample["answer"].strip()) for sample in samples)
        dataset_hash = sha256_file(evaluation_path)
        result = {
            "schema_version": 1,
            "evaluation_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": split,
            "model_label": self.model_label,
            "checkpoint_path": (
                str(self.checkpoint_path.relative_to(V2_ROOT))
                if self.checkpoint_path is not None
                else None
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_state_sha256": self.model_state_sha256,
            "tokenizer_sha256": self.tokenizer_hash,
            "tokenizer_artifact_sha256": self.tokenizer_artifact_hashes,
            "evaluation_dataset_sha256": dataset_hash,
            "perplexity_chunk_dataset_sha256": chunk_hash,
            "model_configuration": self.model.config.to_dict(),
            "device": device_information(self.device),
            "generation_configuration": asdict(self.generation_config),
            "question_count": len(questions),
            "valid_answer_count": valid_answers,
            "human_verified_reference_count": len(verified_metrics),
            "reference_metrics": reference_summary,
            "perplexity": perplexity,
            "generation_quality": generation_summary,
            "human_evaluation": human_summary,
            "overall_human_quality_score": overall_score,
            "performance": performance,
            "failed_or_skipped_evaluations": skipped,
            "limitations": limitations,
            "generation_samples": samples,
        }
        self._write_reports(result, output_dir or EVALUATION_REPORT_DIR)
        return result

    @staticmethod
    def _write_reports(result: dict[str, Any], output_dir: str | Path) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "evaluation_date": result["evaluation_date"],
            "split": result["split"],
            "model_label": result["model_label"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "model_state_sha256": result["model_state_sha256"],
            "tokenizer_sha256": result["tokenizer_sha256"],
            "evaluation_dataset_sha256": result["evaluation_dataset_sha256"],
            "model_configuration": result["model_configuration"],
            "device": result["device"],
        }
        _atomic_write_json(directory / "evaluation_summary.json", {
            key: value
            for key, value in result.items()
            if key != "generation_samples"
        })
        _atomic_write_json(directory / "performance_report.json", {
            **result["performance"],
            "evaluation_metadata": metadata,
        })
        _atomic_write_json(directory / "human_review_summary.json", {
            **result["human_evaluation"],
            "evaluation_metadata": metadata,
        })
        samples_with_metadata = [
            {**sample, "evaluation_metadata": metadata}
            for sample in result["generation_samples"]
        ]
        _atomic_write_jsonl(
            directory / "generation_samples.jsonl",
            samples_with_metadata,
        )
        _write_metrics_csv(directory / "evaluation_metrics.csv", result, metadata)
        _write_markdown_report(directory / "evaluation_report.md", result)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False))
                target.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_metrics_csv(
    path: Path,
    result: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    values: dict[str, tuple[Any, str]] = {}
    for key, value in result["reference_metrics"].items():
        if key.endswith("_percentage"):
            values[key] = (value, "percent")
    values["perplexity"] = (result["perplexity"]["perplexity"], "nats exponentiated")
    values["evaluated_token_count"] = (
        result["perplexity"]["evaluated_token_count"],
        "tokens",
    )
    for key, value in result["generation_quality"].items():
        if key.endswith("_percentage") or key.endswith("_mean"):
            values[f"generation_{key}"] = (
                value,
                "percent" if key.endswith("_percentage") else "value",
            )
    for key, value in result["human_evaluation"]["scores"].items():
        values[key] = (value, "percent")
    values["overall_human_quality_percentage"] = (
        result["overall_human_quality_score"]["overall_human_quality_percentage"],
        "percent",
    )
    values["model_loading_time_seconds"] = (
        result["performance"]["model_loading_time_seconds"],
        "seconds",
    )
    values["first_token_latency_seconds_mean"] = (
        result["performance"]["first_token_latency_seconds_mean"],
        "seconds",
    )
    values["average_response_latency_seconds"] = (
        result["performance"]["average_response_latency_seconds"],
        "seconds",
    )
    values["tokens_per_second"] = (
        result["performance"]["tokens_per_second"],
        "tokens/second",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "metric",
                "value",
                "unit",
                "available",
                "split",
                "model_label",
                "checkpoint_sha256",
                "tokenizer_sha256",
                "evaluation_dataset_sha256",
                "device",
            )
        )
        for name, (value, unit) in sorted(values.items()):
            writer.writerow(
                (
                    name,
                    "" if value is None else value,
                    unit,
                    value is not None,
                    metadata["split"],
                    metadata["model_label"],
                    metadata["checkpoint_sha256"] or "",
                    metadata["tokenizer_sha256"],
                    metadata["evaluation_dataset_sha256"],
                    metadata["device"]["device"],
                )
            )


def _write_markdown_report(path: Path, result: dict[str, Any]) -> None:
    reference = result["reference_metrics"]
    quality = result["generation_quality"]
    human = result["human_evaluation"]
    performance = result["performance"]
    score = result["overall_human_quality_score"]
    def display(value: Any, suffix: str = "") -> str:
        return "Unavailable" if value is None else f"{value}{suffix}"

    lines = [
        "# QuantaLM V2 Evaluation Report",
        "",
        f"- **Evaluated:** {result['evaluation_date']}",
        f"- **Split:** {result['split']}",
        f"- **Model:** {result['model_label']}",
        f"- **Checkpoint SHA-256:** {result['checkpoint_sha256'] or 'None (untrained baseline)'}",
        f"- **Model-state SHA-256:** {result['model_state_sha256']}",
        f"- **Tokenizer SHA-256:** {result['tokenizer_sha256']}",
        f"- **Evaluation dataset SHA-256:** {result['evaluation_dataset_sha256']}",
        f"- **Device:** {result['device']['device']}",
        f"- **Questions:** {result['question_count']}; non-empty answers: {result['valid_answer_count']}",
        f"- **Model configuration:** `{json.dumps(result['model_configuration'], sort_keys=True)}`",
        "",
        "## Metrics",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Exact match | {display(reference['exact_match_percentage'], '%')} |",
        f"| Token precision (macro) | {display(reference['token_precision_percentage'], '%')} |",
        f"| Token recall (macro) | {display(reference['token_recall_percentage'], '%')} |",
        f"| Token F1 (macro) | {display(reference['token_f1_percentage'], '%')} |",
        f"| Perplexity | {display(result['perplexity']['perplexity'])} |",
        f"| Perplexity target tokens | {result['perplexity']['evaluated_token_count']} |",
        f"| Mean answer length | {display(quality['answer_length_tokens_mean'], ' tokens')} |",
        f"| Repeated n-gram ratio | {display(quality['repeated_ngram_ratio_mean'])} |",
        f"| Empty-response rate | {display(quality['empty_response_rate_percentage'], '%')} |",
        f"| EOS termination rate | {display(quality['eos_termination_rate_percentage'], '%')} |",
        f"| Human-reviewed questions | {human['valid_review_count']} |",
        f"| Overall human-quality score | {display(score['overall_human_quality_percentage'], '%')} |",
        "",
        "The human-quality score uses factual accuracy 35%, relevance 25%, coherence 15%, instruction following 15%, and repetition quality 10%. Weights are renormalized over available components, but the score remains unavailable when no valid human review exists.",
        "",
        "Token overlap and perplexity are lexical/modeling measures; neither establishes semantic correctness or factual accuracy.",
        "",
        "## Performance",
        "",
        f"- Model loading time: {performance['model_loading_time_seconds']:.6f} seconds.",
        f"- First-token latency (one-token generate call): {performance['first_token_latency_seconds_mean']:.6f} seconds.",
        f"- Average response latency: {performance['average_response_latency_seconds']:.6f} seconds.",
        f"- Generated tokens per second: {performance['tokens_per_second']:.6f}.",
        f"- Peak CPU memory: {display(performance['peak_cpu_memory_bytes'], ' bytes')} ({performance['peak_cpu_memory_method'] or 'unavailable'}).",
        f"- Warm-up runs: {performance['warmup_runs']}; measured runs: {performance['measured_runs']}.",
        "- Performance probe uses a single BOS token and is not an evaluation question or answer.",
        "",
        "## Limitations",
        "",
        *[f"- {limitation}" for limitation in result["limitations"]],
        "",
        "## Skipped evaluations",
        "",
        *(
            [
                f"- {item['evaluation']}: {item['reason']}"
                for item in result["failed_or_skipped_evaluations"]
            ]
            or ["- None."]
        ),
        "",
        "## Reproducibility",
        "",
        "Generation uses the configured random seed and a stable question-ID-derived seed. Exact-match normalization uses Unicode NFKC, case-folding, and whitespace normalization. Token precision/recall/F1 use multiset word-and-punctuation tokens and macro-average per answer.",
        "",
        "Held-out test evaluation is disabled unless explicitly enabled for a final evaluation run. No test split was evaluated during this development phase.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and benchmark QuantaLM V2.")
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--disable-top-k",
        action="store_true",
        help="Disable top-k filtering and sample from the full model vocabulary.",
    )
    parser.add_argument("--random-seed", type=int, default=20261106)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_REPORT_DIR)
    parser.add_argument(
        "--allow-test-evaluation",
        action="store_true",
        help="Explicitly enable held-out test evaluation for an approved final run only.",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test_evaluation:
        parser.error("Test evaluation requires --allow-test-evaluation.")
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        random_seed=args.random_seed,
        generation_config=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=None if args.disable_top_k else args.top_k,
            random_seed=args.random_seed,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
    )
    result = evaluator.evaluate(
        args.split,
        allow_test_evaluation=args.allow_test_evaluation,
        benchmark_warmup_runs=args.warmup_runs,
        benchmark_measured_runs=args.benchmark_runs,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "split": result["split"],
        "model_label": result["model_label"],
        "question_count": result["question_count"],
        "valid_answer_count": result["valid_answer_count"],
        "perplexity": result["perplexity"]["perplexity"],
        "report_directory": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
