"""Tokenize the fixed QuantaLM V2 document splits with the V2-only tokenizer."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bpe_tokenizer import QuantaBPETokenizer
from train_tokenizer import (
    MANIFEST_FILE,
    TRAIN_FILE,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)
from workspace_config import v2_config


V2_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = v2_config.DATA_DIR / "splits"
TOKENIZER_REPORT_FILE = v2_config.TOKENIZER_TRAINING_REPORT_FILE


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_tokenization() -> dict[str, Any]:
    if not TOKENIZER_REPORT_FILE.is_file():
        raise FileNotFoundError(
            "Train the V2 tokenizer first with `python src/train_tokenizer.py`."
        )
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {MANIFEST_FILE}")

    tokenizer = QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)
    if tokenizer.vocab_size != v2_config.VOCAB_SIZE:
        raise ValueError("Tokenizer and model vocabulary sizes do not match.")
    training_report = read_json(TOKENIZER_REPORT_FILE)
    manifest = read_json(MANIFEST_FILE)
    recorded_training_hash = training_report["training_source"]["split_sha256"]
    if recorded_training_hash != sha256_file(TRAIN_FILE):
        raise ValueError("Training split differs from the split used to fit the tokenizer.")
    training_ids = set(training_report["training_source"]["training_document_ids"])
    manifest_train_ids = {
        record["document_id"]
        for record in manifest["documents"]
        if record.get("split") == "train"
    }
    if training_ids != manifest_train_ids:
        raise ValueError("Tokenizer training document IDs do not match current training split.")
    for split_name in ("train", "validation", "test"):
        expected_hash = manifest["split_files"][split_name]["sha256"]
        actual_hash = sha256_file(SPLIT_DIR / split_name / "dataset.jsonl")
        if expected_hash != actual_hash:
            raise ValueError(f"{split_name} split file does not match the dataset manifest.")

    outputs: dict[str, list[dict[str, Any]]] = {}
    statistics: dict[str, Any] = {}
    all_document_splits: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        source_file = SPLIT_DIR / split_name / "dataset.jsonl"
        records = read_jsonl(source_file)
        if not records:
            raise ValueError(f"{split_name} split is empty.")
        token_records = []
        token_counts_by_category: Counter[str] = Counter()
        lengths: list[int] = []
        unknown_count = 0
        for record in records:
            document_id = record["document_id"]
            if document_id in all_document_splits:
                raise ValueError(
                    f"Document {document_id} appears in multiple dataset splits."
                )
            all_document_splits[document_id] = split_name
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty or invalid document in {split_name}: {document_id}")
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            if not token_ids or token_ids[0] != tokenizer.bos_token_id:
                raise AssertionError("BOS token was not added.")
            if token_ids[-1] != tokenizer.eos_token_id:
                raise AssertionError("EOS token was not added.")
            if tokenizer.decode(token_ids[1:-1]) != text:
                raise AssertionError(
                    f"Tokenization round trip failed for document {document_id}."
                )
            if any(token_id < 0 or token_id >= tokenizer.vocab_size for token_id in token_ids):
                raise AssertionError(f"Out-of-range token ID in document {document_id}.")
            doc_unknowns = token_ids.count(tokenizer.unk_token_id)
            unknown_count += doc_unknowns
            lengths.append(len(token_ids))
            token_counts_by_category[record["category"]] += len(token_ids)
            token_records.append(
                {
                    "document_id": document_id,
                    "split": split_name,
                    "category": record["category"],
                    "source_path": record["source_path"],
                    "source_sha256": record["source_sha256"],
                    "clean_text_sha256": sha256_bytes(text.encode("utf-8")),
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                    "unknown_token_count": doc_unknowns,
                    "bos_token_id": tokenizer.bos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
            )
        outputs[split_name] = token_records
        output_file = SPLIT_DIR / split_name / f"{split_name}_tokens.jsonl"
        write_jsonl(output_file, token_records)
        statistics[split_name] = {
            "source_document_count": len(records),
            "tokenized_document_count": len(token_records),
            "token_count_including_bos_eos": sum(lengths),
            "average_tokens_per_document": sum(lengths) / len(lengths),
            "maximum_sequence_length": max(lengths),
            "documents_exceeding_model_context_length": sum(
                length > v2_config.CONTEXT_LENGTH for length in lengths
            ),
            "model_context_length": v2_config.CONTEXT_LENGTH,
            "context_policy": "Full document encoded; no truncation. Chunking is required before model input.",
            "category_token_counts": dict(sorted(token_counts_by_category.items())),
            "unknown_token_count": unknown_count,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "output_path": output_file.relative_to(V2_ROOT).as_posix(),
            "output_sha256": sha256_file(output_file),
        }

    if training_ids != {
        record["document_id"] for record in outputs["train"]
    }:
        raise ValueError("Tokenized training documents differ from tokenizer training inputs.")
    if training_ids & {record["document_id"] for record in outputs["validation"]}:
        raise ValueError("Tokenizer training documents overlap validation documents.")
    if training_ids & {record["document_id"] for record in outputs["test"]}:
        raise ValueError("Tokenizer training documents overlap test documents.")

    tokenizer_directory = v2_config.TOKENIZER_DIR
    vocabulary_hash = sha256_file(v2_config.TOKENIZER_VOCAB_FILE)
    merges_hash = sha256_file(v2_config.TOKENIZER_MERGES_FILE)
    training_report_hash = sha256_file(TOKENIZER_REPORT_FILE)

    manifest["tokenizer"] = {
        "implementation": "src/bpe_tokenizer.py",
        "vocabulary_size": tokenizer.vocab_size,
        "special_tokens": tokenizer.special_token_ids,
        "eos_token_id": tokenizer.eos_token_id,
        "training_split_sha256": recorded_training_hash,
        "training_document_ids": sorted(training_ids),
        "training_report_sha256": training_report_hash,
        "vocabulary_sha256": vocabulary_hash,
        "merges_sha256": merges_hash,
        "fitted_on_validation": False,
        "fitted_on_test": False,
    }
    for split_name in ("train", "validation", "test"):
        token_path = SPLIT_DIR / split_name / f"{split_name}_tokens.jsonl"
        manifest["split_files"][split_name]["tokenized_path"] = (
            token_path.relative_to(V2_ROOT).as_posix()
        )
        manifest["split_files"][split_name]["tokenized_sha256"] = sha256_file(token_path)
    manifest["held_out_evaluation_warning"] = (
        "Tokenizer merges were fitted only on documents assigned to the training split. "
        "Validation and test documents were encoded after fitting and did not influence merges."
    )
    write_json(MANIFEST_FILE, manifest)
    for split_name in ("train", "validation", "test"):
        token_path = SPLIT_DIR / split_name / f"{split_name}_tokens.jsonl"
        tokenized_records = read_jsonl(token_path)
        expected = manifest["split_files"][split_name]
        if len(tokenized_records) != expected["document_count"]:
            raise AssertionError(f"Tokenized manifest count mismatch for {split_name}.")
        if sha256_file(token_path) != expected["tokenized_sha256"]:
            raise AssertionError(f"Tokenized manifest hash mismatch for {split_name}.")
    token_statistics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {
            "vocabulary_size": tokenizer.vocab_size,
            "special_tokens": tokenizer.special_token_ids,
            "vocabulary_sha256": vocabulary_hash,
            "merges_sha256": merges_hash,
            "training_report_sha256": training_report_hash,
        },
        "training_split_sha256": recorded_training_hash,
        "dataset_manifest_sha256": sha256_file(MANIFEST_FILE),
        "splits": statistics,
        "total_document_count": sum(
            item["tokenized_document_count"] for item in statistics.values()
        ),
        "total_token_count": sum(
            item["token_count_including_bos_eos"] for item in statistics.values()
        ),
        "total_unknown_token_count": sum(
            item["unknown_token_count"] for item in statistics.values()
        ),
    }
    reports_path = v2_config.DATA_DIR / "reports" / "token_statistics.json"
    write_json(reports_path, token_statistics)

    quality_file = v2_config.DATA_DIR / "reports" / "data_quality_report.json"
    quality = read_json(quality_file)
    quality["tokenizer_training_data_leakage"] = {
        "status": "resolved for this tokenizer artifact",
        "evidence": (
            "Tokenizer training report records the training split hash and document IDs; "
            "the fit script reads only train/split dataset JSONL."
        ),
        "validation_documents_read_for_fitting": 0,
        "test_documents_read_for_fitting": 0,
        "tokenizer_training_report_sha256": training_report_hash,
    }
    quality["tokenizer_vocabulary_matches_model"] = (
        tokenizer.vocab_size == v2_config.VOCAB_SIZE
    )
    quality["tokenizer_training_split_sha256"] = recorded_training_hash
    quality["tokenized_split_statistics"] = statistics
    quality["model_context_compatibility"] = {
        "configured_context_length": v2_config.CONTEXT_LENGTH,
        "documents_exceeding_context_length": sum(
            item["documents_exceeding_model_context_length"]
            for item in statistics.values()
        ),
        "documents_truncated": 0,
        "policy": "No truncation; split documents require a separate chunking policy before model training.",
    }
    quality["dataset_expansion"] = {
        "additional_approved_sources_added": 0,
        "reason": (
            "No approved external sources or licensing records were provided or found "
            "in the inspected local dataset."
        ),
        "category_gaps": [
            "General Knowledge has no documents.",
            "Mathematics, General Science, AI, and Student Learning each have one unique source document.",
            "No dedicated software engineering document or verified general-knowledge source was found.",
        ],
    }
    write_json(quality_file, quality)

    return {
        "tokenizer_vocabulary_size": tokenizer.vocab_size,
        "training_split_sha256": recorded_training_hash,
        "tokenizer_artifact_directory": tokenizer_directory.relative_to(V2_ROOT).as_posix(),
        "token_statistics_path": reports_path.relative_to(V2_ROOT).as_posix(),
        "split_statistics": statistics,
        "unknown_token_count": token_statistics["total_unknown_token_count"],
    }


if __name__ == "__main__":
    print(json.dumps(run_tokenization(), ensure_ascii=False, indent=2, sort_keys=True))
