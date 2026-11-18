"""Fit the independent V2 byte-BPE tokenizer on the training split only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bpe_tokenizer import QuantaBPETokenizer
from workspace_config import v2_config


V2_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = v2_config.DATA_DIR
TOKENIZER_DIR = v2_config.TOKENIZER_DIR
TRAIN_FILE = DATA_DIR / "splits" / "train" / "dataset.jsonl"
MANIFEST_FILE = DATA_DIR / "manifests" / "dataset_manifest.json"
EXPECTED_VOCAB_SIZE = v2_config.VOCAB_SIZE
TRAINING_SEED = 20261106


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_training_only_inputs(
    train_records: list[dict[str, Any]],
    manifest: dict[str, Any],
    train_file_hash: str,
) -> list[dict[str, Any]]:
    if not train_records:
        raise ValueError("The training split is empty; refusing to train tokenizer.")
    manifest_documents = manifest.get("documents")
    if not isinstance(manifest_documents, list):
        raise ValueError("Dataset manifest does not contain document records.")

    by_id = {document.get("document_id"): document for document in manifest_documents}
    train_ids = {record.get("document_id") for record in train_records}
    if None in train_ids or len(train_ids) != len(train_records):
        raise ValueError("Training split has missing or duplicate document IDs.")
    declared_train_ids = {
        document["document_id"]
        for document in manifest_documents
        if document.get("split") == "train"
    }
    if train_ids != declared_train_ids:
        raise ValueError("Training JSONL does not match manifest training assignments.")

    validation_ids = {
        document["document_id"]
        for document in manifest_documents
        if document.get("split") == "validation"
    }
    test_ids = {
        document["document_id"]
        for document in manifest_documents
        if document.get("split") == "test"
    }
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Dataset manifest contains overlapping document splits.")

    records_by_id = {record["document_id"]: record for record in train_records}
    provenance: list[dict[str, Any]] = []
    for document_id in sorted(train_ids):
        record = records_by_id[document_id]
        declared = by_id[document_id]
        if record.get("category") != declared.get("category"):
            raise ValueError(f"Category mismatch for training document {document_id}.")
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Training document {document_id} has no usable text.")
        clean_hash = sha256_bytes(text.encode("utf-8"))
        if clean_hash != declared.get("clean_text_sha256"):
            raise ValueError(f"Clean-text hash mismatch for training document {document_id}.")
        if record.get("source_sha256") != declared.get("source_sha256"):
            raise ValueError(f"Source hash mismatch for training document {document_id}.")
        provenance.append(
            {
                "document_id": document_id,
                "split": "train",
                "document_title": None,
                "title_status": "No document title metadata was supplied.",
                "category": record["category"],
                "source_path": record["source_path"],
                "source_sha256": record["source_sha256"],
                "clean_text_sha256": clean_hash,
                "classification_source": record.get("classification_source"),
                "classification_confidence": record.get("classification_confidence"),
                "license_status": record.get("license_status", "unknown"),
            }
        )

    actual_train_hash = sha256_file(TRAIN_FILE)
    if actual_train_hash != train_file_hash:
        raise AssertionError("Training split changed during tokenizer provenance check.")
    return provenance


def train_and_save(*, overwrite: bool = False) -> dict[str, Any]:
    if not TRAIN_FILE.is_file() or not MANIFEST_FILE.is_file():
        raise FileNotFoundError(
            "Run `python src/prepare_dataset.py` before training the V2 tokenizer."
        )
    existing_artifacts = [
        path
        for path in (
            TOKENIZER_DIR / "bpe_vocab.json",
            TOKENIZER_DIR / "bpe_merges.json",
            TOKENIZER_DIR / "tokenizer_config.json",
            TOKENIZER_DIR / "tokenizer_training_report.json",
        )
        if path.exists()
    ]
    if existing_artifacts and not overwrite:
        raise FileExistsError(
            "V2 tokenizer artifacts already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing_artifacts)
        )
    train_file_hash = sha256_file(TRAIN_FILE)
    train_records = read_jsonl(TRAIN_FILE)
    manifest = read_json(MANIFEST_FILE)
    provenance = verify_training_only_inputs(train_records, manifest, train_file_hash)
    training_provenance_hash = sha256_bytes(
        json.dumps(
            {
                "training_split_sha256": train_file_hash,
                "training_documents": provenance,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    tokenizer = QuantaBPETokenizer.train(
        documents=[record["text"] for record in train_records],
        vocab_size=EXPECTED_VOCAB_SIZE,
        seed=TRAINING_SEED,
    )
    if tokenizer.vocab_size != v2_config.VOCAB_SIZE:
        raise ValueError("Trained tokenizer vocabulary does not match V2 model config.")
    if tokenizer.eos_token_id != v2_config.EOS_TOKEN_ID:
        raise ValueError("Tokenizer EOS ID does not match V2 model config.")

    config_extra = {
        "training_split": "data/splits/train/dataset.jsonl",
        "training_split_sha256": train_file_hash,
        "training_provenance_sha256": training_provenance_hash,
        "training_random_seed": TRAINING_SEED,
        "training_document_ids": [item["document_id"] for item in provenance],
        "fitted_on_validation": False,
        "fitted_on_test": False,
    }
    tokenizer_paths = tokenizer.save(
        TOKENIZER_DIR,
        overwrite=overwrite,
        config_extra=config_extra,
    )
    report = {
        "report_type": "QuantaLM V2 independent tokenizer training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": (
            "Byte-level BPE; whitespace and non-whitespace pieces preserved; "
            "weighted pair-frequency merges; seeded tie-breaking."
        ),
        "training_random_seed": TRAINING_SEED,
        "seed_effect": "Used only to resolve equal-frequency BPE merge ties.",
        "vocabulary_size": tokenizer.vocab_size,
        "special_tokens": tokenizer.special_token_ids,
        "eos_token_id": tokenizer.eos_token_id,
        "training_source": {
            "split": "train",
            "split_path": "data/splits/train/dataset.jsonl",
            "split_sha256": train_file_hash,
            "manifest_path": "data/manifests/dataset_manifest.json",
            "training_provenance_sha256": training_provenance_hash,
            "training_document_count": len(train_records),
            "training_document_ids": [item["document_id"] for item in provenance],
            "training_documents": provenance,
            "validation_documents_read_for_fitting": 0,
            "test_documents_read_for_fitting": 0,
            "training_text_characters": sum(len(record["text"]) for record in train_records),
            "training_text_sha256": sha256_bytes(
                "\n".join(record["text"] for record in train_records).encode("utf-8")
            ),
        },
        "excluded_splits": [
            {"split": "validation", "used_for_fitting": False},
            {"split": "test", "used_for_fitting": False},
        ],
        "v1_tokenizer_modified": False,
        "artifacts": {
            "bpe_vocab_path": tokenizer_paths["bpe_vocab"].relative_to(V2_ROOT).as_posix(),
            "bpe_vocab_sha256": sha256_file(tokenizer_paths["bpe_vocab"]),
            "bpe_merges_path": tokenizer_paths["bpe_merges"].relative_to(V2_ROOT).as_posix(),
            "bpe_merges_sha256": sha256_file(tokenizer_paths["bpe_merges"]),
            "tokenizer_config_path": tokenizer_paths["tokenizer_config"].relative_to(V2_ROOT).as_posix(),
            "tokenizer_config_sha256": sha256_file(tokenizer_paths["tokenizer_config"]),
        },
        "model_compatibility": {
            "configured_vocabulary_size": v2_config.VOCAB_SIZE,
            "trained_vocabulary_size": tokenizer.vocab_size,
            "vocabulary_matches_model": tokenizer.vocab_size == v2_config.VOCAB_SIZE,
            "configured_eos_token_id": v2_config.EOS_TOKEN_ID,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "eos_matches_model_configuration": tokenizer.eos_token_id == v2_config.EOS_TOKEN_ID,
        },
    }
    report_path = TOKENIZER_DIR / "tokenizer_training_report.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"Tokenizer training report already exists; pass --overwrite to replace it: {report_path}"
        )
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing V2 tokenizer artifacts and training report.",
    )
    args = parser.parse_args()
    result = train_and_save(overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
