"""Audit V2 corpus provenance and stage explicitly licensed document ingestions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .bpe_tokenizer import QuantaBPETokenizer
    from .chunk_dataset import ChunkConfig, chunk_document
    from .workspace_config import v2_config
except ImportError:
    from bpe_tokenizer import QuantaBPETokenizer
    from chunk_dataset import ChunkConfig, chunk_document
    from workspace_config import v2_config


V2_ROOT = Path(__file__).resolve().parents[1]
INGESTION_RUNS_DIR = v2_config.DATA_DIR / "ingestion_runs"
SOURCE_INVENTORY_FILE = v2_config.DATA_DIR / "reports" / "source_inventory.json"
DATA_QUALITY_FILE = v2_config.DATA_DIR / "reports" / "data_quality_report.json"
DUPLICATE_REPORT_FILE = v2_config.DATA_DIR / "reports" / "duplicate_report.json"
DATASET_STATISTICS_FILE = v2_config.DATA_DIR / "reports" / "dataset_statistics.json"
DATASET_MANIFEST_FILE = v2_config.DATA_DIR / "manifests" / "dataset_manifest.json"
SOURCE_SNAPSHOT_DIR = v2_config.DATA_DIR / "raw" / "source_snapshot"
PROVENANCE_REPORT_FILE = (
    v2_config.DATA_DIR / "reports" / "dataset_provenance_report.json"
)

SUPPORTED_LICENSES = {
    "CC0-1.0",
    "CC-BY-4.0",
    "PSF-2.0",
}
CANONICAL_LICENSE_URLS = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "PSF-2.0": "https://docs.python.org/3/license.html",
}
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SUBSTANTIVE_LINE_MIN_LENGTH = 16
REPEATED_PASSAGE_MIN_LENGTH = 80
NEAR_DUPLICATE_THRESHOLD = 0.85
NEAR_DUPLICATE_SHINGLE_SIZE = 5
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ProvenanceError(ValueError):
    """Raised when an ingestion record lacks acceptable source authorization."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def _clean_text(text: str) -> tuple[str, dict[str, int]]:
    if not isinstance(text, str):
        raise TypeError("Document text must be a string.")
    if CONTROL_PATTERN.search(text):
        raise ValueError("Document contains unsupported control characters.")

    original = text
    text = text.removeprefix("\ufeff")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.rstrip() for line in text.split("\n")]
    compact: list[str] = []
    blank_count = 0
    collapsed_blank_lines = 0
    for line in lines:
        if line.strip():
            compact.append(line)
            blank_count = 0
        else:
            blank_count += 1
            if blank_count <= 2:
                compact.append("")
            else:
                collapsed_blank_lines += 1
    cleaned = "\n".join(compact).strip()
    if not cleaned:
        raise ValueError("Document is empty after whitespace normalization.")
    return cleaned, {
        "removed_utf8_bom": int(original.startswith("\ufeff")),
        "normalized_line_endings": int("\r" in original),
        "trimmed_trailing_whitespace_lines": sum(
            before != before.rstrip() for before in text.split("\n")
        ),
        "collapsed_excess_blank_lines": collapsed_blank_lines,
    }


def _valid_https_url(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceError(f"{field} is required.")
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise ProvenanceError(f"{field} must be an absolute HTTPS URL.")
    return value.strip()


def validate_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata attestation; license claims still require human review."""
    required_text = (
        "source_id",
        "source_name",
        "source_owner",
        "original_location",
        "source_version",
        "source_retrieved_at",
        "category",
        "license_id",
        "license_verified_by",
        "license_evidence",
        "license_verified_at",
        "text_path",
    )
    if not isinstance(record, dict):
        raise ProvenanceError("Each source manifest row must be a JSON object.")
    normalized = dict(record)
    for field in required_text:
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ProvenanceError(f"{field} is required.")
        normalized[field] = value.strip()

    if normalized["license_id"] not in SUPPORTED_LICENSES:
        raise ProvenanceError(
            f"License {normalized['license_id']!r} is not approved by the "
            "ingestion allowlist."
        )
    if normalized.get("license_verified") is not True:
        raise ProvenanceError("license_verified must be explicitly true.")
    if normalized.get("license_scope_reviewed") is not True:
        raise ProvenanceError("license_scope_reviewed must be explicitly true.")
    if normalized.get("third_party_content_reviewed") is not True:
        raise ProvenanceError("third_party_content_reviewed must be explicitly true.")
    normalized["source_url"] = _valid_https_url(
        normalized.get("source_url"), "source_url"
    )
    normalized["license_url"] = _valid_https_url(
        normalized.get("license_url"), "license_url"
    )
    expected_license_url = CANONICAL_LICENSE_URLS[normalized["license_id"]]
    if normalized["license_url"].rstrip("/") != expected_license_url.rstrip("/"):
        raise ProvenanceError(
            f"license_url does not match the canonical {normalized['license_id']} URL."
        )
    for field in ("license_verified_at", "source_retrieved_at"):
        try:
            datetime.fromisoformat(normalized[field].replace("Z", "+00:00"))
        except ValueError as error:
            raise ProvenanceError(
                f"{field} must be an ISO-8601 date or timestamp."
            ) from error
    return normalized


def _load_source_rows(
    manifest_path: Path,
    input_root: Path,
) -> list[dict[str, Any]]:
    root = input_root.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {manifest_path}:{line_number}: {error.msg}."
                ) from error
            metadata = validate_provenance(row)
            source_id = metadata["source_id"]
            if source_id in source_ids:
                raise ProvenanceError(f"Duplicate source_id {source_id!r}.")
            source_ids.add(source_id)

            relative_text_path = Path(metadata["text_path"])
            if relative_text_path.is_absolute():
                raise ProvenanceError("text_path must be relative to input_root.")
            if relative_text_path.suffix.lower() != ".txt":
                raise ProvenanceError("Only UTF-8 .txt documents are accepted.")
            text_path = (root / relative_text_path).resolve(strict=True)
            if not text_path.is_relative_to(root):
                raise ProvenanceError("text_path must remain inside input_root.")
            raw_bytes = text_path.read_bytes()
            try:
                raw_text = raw_bytes.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"Source {source_id!r} is not valid UTF-8."
                ) from error
            cleaned_text, cleaning = _clean_text(raw_text)
            row_record = {
                **metadata,
                "text_path": relative_text_path.as_posix(),
                "source_sha256": sha256_bytes(raw_bytes),
                "clean_text_sha256": sha256_bytes(cleaned_text.encode("utf-8")),
                "character_count": len(cleaned_text),
                "cleaning_actions": cleaning,
                "text": cleaned_text,
            }
            rows.append(row_record)
    if not rows:
        raise ValueError("Source manifest contains no documents.")
    return rows


def _tokenizer_identity(tokenizer: QuantaBPETokenizer) -> tuple[str, dict[str, str]]:
    artifact_paths = {
        "bpe_merges.json": v2_config.TOKENIZER_MERGES_FILE,
        "bpe_vocab.json": v2_config.TOKENIZER_VOCAB_FILE,
        "tokenizer_config.json": v2_config.TOKENIZER_CONFIG_FILE,
    }
    hashes = {name: sha256_file(path) for name, path in artifact_paths.items()}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(canonical), hashes


def _deduplicate_sources(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_by_hash: dict[str, dict[str, Any]] = {}
    documents: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["source_id"]):
        content_hash = row["clean_text_sha256"]
        canonical = canonical_by_hash.get(content_hash)
        if canonical is not None:
            duplicates.append(
                {
                    "duplicate_source_id": row["source_id"],
                    "canonical_source_id": canonical["source_id"],
                    "duplicate_type": "normalized_exact_text",
                    "clean_text_sha256": content_hash,
                }
            )
            continue
        canonical_by_hash[content_hash] = row
        documents.append(
            {
                **row,
                "document_id": content_hash[:16],
            }
        )
    return documents, duplicates


def _word_shingles(text: str) -> set[tuple[str, ...]]:
    words = WORD_PATTERN.findall(unicodedata.normalize("NFC", text).casefold())
    if len(words) < NEAR_DUPLICATE_SHINGLE_SIZE:
        return set()
    size = NEAR_DUPLICATE_SHINGLE_SIZE
    return {
        tuple(words[index : index + size])
        for index in range(len(words) - size + 1)
    }


def _leakage_groups(
    documents: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    parent = {document["document_id"]: document["document_id"] for document in documents}
    relations: dict[tuple[str, str], set[str]] = defaultdict(set)

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: str, second: str, reason: str) -> None:
        left, right = sorted((first, second))
        relations[(left, right)].add(reason)
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    line_owners: dict[str, list[str]] = defaultdict(list)
    passage_owners: dict[str, list[str]] = defaultdict(list)
    shingle_owners: dict[tuple[str, ...], list[str]] = defaultdict(list)
    shingles_by_document: dict[str, set[tuple[str, ...]]] = {}
    for document in documents:
        document_id = document["document_id"]
        text = document["text"]
        lines = {
            re.sub(r"\s+", " ", line).strip().casefold()
            for line in text.splitlines()
            if len(line.strip()) >= SUBSTANTIVE_LINE_MIN_LENGTH
        }
        passages = {
            re.sub(r"\s+", " ", passage).strip().casefold()
            for passage in re.split(r"\n\s*\n", text)
            if len(passage.strip()) >= REPEATED_PASSAGE_MIN_LENGTH
        }
        for line in lines:
            line_owners[line].append(document_id)
        for passage in passages:
            passage_owners[passage].append(document_id)
        shingles = _word_shingles(text)
        shingles_by_document[document_id] = shingles
        for shingle in shingles:
            shingle_owners[shingle].append(document_id)

    for owners in line_owners.values():
        for index, first in enumerate(sorted(set(owners))):
            for second in sorted(set(owners))[index + 1 :]:
                union(first, second, "shared_substantive_line")
    for owners in passage_owners.values():
        unique_owners = sorted(set(owners))
        for index, first in enumerate(unique_owners):
            for second in unique_owners[index + 1 :]:
                union(first, second, "shared_passage")

    candidate_pairs = {
        tuple(sorted((first, second)))
        for owners in shingle_owners.values()
        for index, first in enumerate(sorted(set(owners)))
        for second in sorted(set(owners))[index + 1 :]
    }
    for first, second in sorted(candidate_pairs):
        first_shingles = shingles_by_document[first]
        second_shingles = shingles_by_document[second]
        union_size = len(first_shingles | second_shingles)
        if not union_size:
            continue
        similarity = len(first_shingles & second_shingles) / union_size
        if similarity >= NEAR_DUPLICATE_THRESHOLD:
            union(first, second, "near_duplicate_5_word_shingle_jaccard")

    grouped: dict[str, list[str]] = defaultdict(list)
    for document_id in sorted(parent):
        grouped[find(document_id)].append(document_id)
    group_ids: dict[str, str] = {}
    for members in grouped.values():
        group_id = sha256_bytes("\n".join(sorted(members)).encode("utf-8"))[:16]
        group_ids.update({member: group_id for member in members})

    audit_relations = [
        {
            "document_ids": [first, second],
            "reasons": sorted(reasons),
        }
        for (first, second), reasons in sorted(relations.items())
    ]
    return group_ids, audit_relations


def _assign_splits(
    documents: list[dict[str, Any]],
    group_ids: dict[str, str],
    ratios: dict[str, float] = SPLIT_RATIOS,
    fixed_group_splits: dict[str, str] | None = None,
) -> dict[str, str]:
    if set(ratios) != {"train", "validation", "test"}:
        raise ValueError("Split ratios must define train, validation, and test.")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in ratios.values()):
        raise ValueError("Split ratios must all be positive.")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    groups: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        groups[group_ids[document["document_id"]]].append(document["document_id"])
    fixed_group_splits = fixed_group_splits or {}
    if any(split not in ratios for split in fixed_group_splits.values()):
        raise ValueError("A fixed leakage-group split is not recognized.")
    if not set(fixed_group_splits).issubset(groups):
        raise ValueError("A fixed split references a group with no incoming documents.")
    ordered_groups = sorted(
        (sorted(members) for members in groups.values()),
        key=lambda members: (
            -len(members),
            sha256_bytes("\n".join(members).encode("utf-8")),
        ),
    )

    split_names = ("train", "validation", "test")
    target_counts = {
        name: max(1.0, len(documents) * ratios[name])
        for name in split_names
    }
    counts = {name: 0 for name in split_names}
    group_split: dict[str, str] = {}
    for group_id, split in fixed_group_splits.items():
        counts[split] += len(groups[group_id])
        group_split.update({member: split for member in groups[group_id]})

    unassigned_groups = [
        members
        for members in ordered_groups
        if group_ids[members[0]] not in fixed_group_splits
    ]
    for members in unassigned_groups:
        empty_splits = [name for name in split_names if counts[name] == 0]
        if empty_splits:
            selected = empty_splits[0]
        else:
            candidates = []
            for name in split_names:
                projected = dict(counts)
                projected[name] += len(members)
                score = sum(
                    ((projected[split] - target_counts[split]) / target_counts[split]) ** 2
                    for split in split_names
                )
                candidates.append((score, split_names.index(name), name))
            selected = min(candidates)[2]
        counts[selected] += len(members)
        group_split.update({member: selected for member in members})
    if set(group_split.values()) != set(split_names):
        raise ValueError(
            "At least one independent incoming leakage group is required for each "
            "train, validation, and test split after existing-split matches."
        )
    return group_split


def _load_existing_split_anchors() -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen_document_ids: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = v2_config.DATA_DIR / "splits" / split / "dataset.jsonl"
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                document_id = record.get("document_id")
                text = record.get("text")
                if not isinstance(document_id, str) or not document_id:
                    raise ValueError(f"Existing split record has no document_id: {path}:{line_number}.")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Existing split record has no text: {path}:{line_number}.")
                previous_split = seen_document_ids.get(document_id)
                if previous_split is not None:
                    raise ValueError(
                        f"Existing document {document_id} occurs in both "
                        f"{previous_split} and {split}."
                    )
                seen_document_ids[document_id] = split
                anchors.append(
                    {
                        "document_id": document_id,
                        "text": text,
                        "category": record.get("category", "Uncategorized"),
                        "existing_split": split,
                        "clean_text_sha256": record.get(
                            "clean_text_sha256",
                            sha256_bytes(text.encode("utf-8")),
                        ),
                    }
                )
    return anchors


def _source_quality(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    stripped = [line.strip() for line in lines if line.strip()]
    counts = Counter(stripped)
    repeated_lines = sum(
        count - 1 for line, count in counts.items() if len(line) >= 40
    )
    return {
        "line_count": len(lines),
        "non_empty_line_count": len(stripped),
        "repeated_substantive_line_occurrences": repeated_lines,
        "replacement_character_count": text.count("\ufffd"),
        "long_line_count_over_300_chars": sum(len(line) > 300 for line in lines),
        "quality_review_required": True,
    }


def _write_ingestion_run(
    rows: list[dict[str, Any]],
    run_directory: Path,
    tokenizer: QuantaBPETokenizer,
    existing_anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    documents, exact_duplicates = _deduplicate_sources(rows)
    existing_hashes = {
        anchor["clean_text_sha256"]: anchor for anchor in existing_anchors
    }
    external_duplicates: list[dict[str, Any]] = []
    retained_documents = []
    for document in documents:
        anchor = existing_hashes.get(document["clean_text_sha256"])
        if anchor is None:
            retained_documents.append(document)
        else:
            external_duplicates.append(
                {
                    "source_id": document["source_id"],
                    "document_id": document["document_id"],
                    "existing_document_id": anchor["document_id"],
                    "existing_split": anchor["existing_split"],
                    "duplicate_type": "normalized_exact_match_to_existing_split",
                    "clean_text_sha256": document["clean_text_sha256"],
                }
            )
    documents = retained_documents
    if not documents:
        raise ValueError(
            "All approved input documents duplicate existing split content; "
            "nothing new can be staged."
        )

    all_documents = [*documents, *existing_anchors]
    all_group_ids, leakage_relations = _leakage_groups(all_documents)
    group_ids = {
        document["document_id"]: all_group_ids[document["document_id"]]
        for document in documents
    }
    existing_splits_by_group: dict[str, set[str]] = defaultdict(set)
    for anchor in existing_anchors:
        existing_splits_by_group[all_group_ids[anchor["document_id"]]].add(
            anchor["existing_split"]
        )
    if any(len(splits) > 1 for splits in existing_splits_by_group.values()):
        raise ValueError(
            "Existing documents in different splits share duplicate or substantial "
            "content; resolve the pre-existing leakage before staging."
        )
    fixed_splits = {
        group_id: next(iter(splits))
        for group_id, splits in existing_splits_by_group.items()
        if group_id in set(group_ids.values())
    }
    incoming_relations = [
        relation
        for relation in leakage_relations
        if any(
            document_id in group_ids
            for document_id in relation["document_ids"]
        )
    ]
    split_by_document = _assign_splits(
        documents,
        group_ids,
        fixed_group_splits=fixed_splits,
    )
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for document_id, split in split_by_document.items():
        split_by_group[group_ids[document_id]].add(split)
    if any(len(splits) != 1 for splits in split_by_group.values()):
        raise AssertionError("A leakage group was assigned across multiple splits.")
    tokenizer_hash, tokenizer_artifact_hashes = _tokenizer_identity(tokenizer)

    split_documents: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLIT_RATIOS
    }
    split_chunks: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLIT_RATIOS
    }
    document_quality: list[dict[str, Any]] = []

    for document in documents:
        document_id = document["document_id"]
        split = split_by_document[document_id]
        text = document["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if (
            not token_ids
            or token_ids[0] != v2_config.BOS_TOKEN_ID
            or token_ids[-1] != v2_config.EOS_TOKEN_ID
        ):
            raise ValueError(f"Tokenizer special-token output is invalid for {document_id}.")
        if any(token_id < 0 or token_id >= tokenizer.vocab_size for token_id in token_ids):
            raise ValueError(f"Tokenizer emitted an out-of-range ID for {document_id}.")
        if tokenizer.decode(token_ids[1:-1]) != text:
            raise ValueError(f"Tokenizer round trip failed for {document_id}.")

        source_record = {
            **document,
            "split": split,
            "leakage_group_id": group_ids[document_id],
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        split_documents[split].append(source_record)
        chunk_record = {
            "document_id": document_id,
            "category": document["category"],
            "source_sha256": document["source_sha256"],
            "clean_text_sha256": document["clean_text_sha256"],
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        chunks = chunk_document(chunk_record, tokenizer_hash, ChunkConfig())
        for chunk in chunks:
            chunk.update(
                {
                    "split": split,
                    "source_id": document["source_id"],
                    "source_name": document["source_name"],
                    "source_url": document["source_url"],
                    "license_id": document["license_id"],
                    "license_url": document["license_url"],
                    "license_evidence": document["license_evidence"],
                    "leakage_group_id": group_ids[document_id],
                }
            )
        split_chunks[split].extend(chunks)
        document_quality.append(
            {
                "document_id": document_id,
                "source_id": document["source_id"],
                "source_name": document["source_name"],
                "category": document["category"],
                "split": split,
                "source_sha256": document["source_sha256"],
                "clean_text_sha256": document["clean_text_sha256"],
                "token_count_including_bos_eos": len(token_ids),
                "chunk_count": len(chunks),
                "cleaning_actions": document["cleaning_actions"],
                **_source_quality(text),
            }
        )

    split_statistics: dict[str, Any] = {}
    output_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        docs = sorted(split_documents[split], key=lambda item: item["document_id"])
        chunks = sorted(
            split_chunks[split],
            key=lambda item: (item["document_id"], item["chunk_id"]),
        )
        document_output = run_directory / "documents" / split / "documents.jsonl"
        token_output = run_directory / "tokens" / split / "documents.jsonl"
        chunk_output = run_directory / "chunks" / split / "chunks.jsonl"
        _write_jsonl(document_output, docs)
        token_rows = [
            {
                key: value
                for key, value in document.items()
                if key != "text"
            }
            for document in docs
        ]
        _write_jsonl(token_output, token_rows)
        _write_jsonl(chunk_output, chunks)
        for path in (document_output, token_output, chunk_output):
            output_hashes[path.relative_to(run_directory).as_posix()] = sha256_file(path)

        category_counts = Counter(document["category"] for document in docs)
        total_tokens = sum(document["token_count"] for document in docs)
        chunk_target_positions = sum(
            chunk["valid_sequence_length"] for chunk in chunks
        )
        source_target_positions = sum(
            max(0, document["token_count"] - 1) for document in docs
        )
        split_statistics[split] = {
            "document_count": len(docs),
            "chunk_count": len(chunks),
            "token_count_including_bos_eos": total_tokens,
            "category_document_counts": dict(sorted(category_counts.items())),
            "chunk_target_positions_including_overlap": chunk_target_positions,
            "unique_target_positions": source_target_positions,
            "repeated_target_positions_from_overlap": (
                chunk_target_positions - source_target_positions
            ),
            "document_ids": [document["document_id"] for document in docs],
            "documents_sha256": output_hashes[
                document_output.relative_to(run_directory).as_posix()
            ],
            "tokenized_documents_sha256": output_hashes[
                token_output.relative_to(run_directory).as_posix()
            ],
            "chunks_sha256": output_hashes[
                chunk_output.relative_to(run_directory).as_posix()
            ],
        }

    source_provenance = [
        {
            key: row[key]
            for key in (
                "source_id",
                "source_name",
                "source_owner",
                "original_location",
                "source_url",
                "source_version",
                "source_retrieved_at",
                "category",
                "license_id",
                "license_url",
                "license_verified",
                "license_scope_reviewed",
                "third_party_content_reviewed",
                "license_verified_by",
                "license_verified_at",
                "license_evidence",
                "text_path",
                "source_sha256",
                "clean_text_sha256",
            )
        }
        for row in sorted(rows, key=lambda item: item["source_id"])
    ]
    provenance_report = {
        "schema_version": 1,
        "status": "staged_approved_documents",
        "license_allowlist": sorted(SUPPORTED_LICENSES),
        "license_claims_are_human_attestations_not_automatically_verified": True,
        "source_documents": source_provenance,
        "exact_duplicate_sources": exact_duplicates,
        "exact_duplicate_existing_sources": external_duplicates,
        "unique_documents_after_exact_deduplication": len(documents),
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_artifact_sha256": tokenizer_artifact_hashes,
    }
    quality_report = {
        "schema_version": 1,
        "status": "completed_with_human_review_required",
        "input_source_count": len(rows),
        "unique_document_count": len(documents),
        "exact_duplicate_source_count": (
            len(exact_duplicates) + len(external_duplicates)
        ),
        "near_duplicate_pair_count": sum(
            "near_duplicate_5_word_shingle_jaccard" in relation["reasons"]
            for relation in incoming_relations
        ),
        "shared_line_or_passage_pair_count": sum(
            any(reason.startswith("shared_") for reason in relation["reasons"])
            for relation in incoming_relations
        ),
        "leakage_groups": {
            document_id: group_id for document_id, group_id in sorted(group_ids.items())
        },
        "existing_split_matches": external_duplicates,
        "leakage_relations": incoming_relations,
        "split_assignment_by_document": dict(sorted(split_by_document.items())),
        "cross_split_leakage": False,
        "category_document_counts": dict(
            sorted(Counter(document["category"] for document in documents).items())
        ),
        "document_quality": sorted(
            document_quality, key=lambda item: item["document_id"]
        ),
        "split_statistics": split_statistics,
        "tokenizer_round_trip_validated": True,
        "all_chunk_target_lengths_within_context": all(
            len(chunk["target_ids"]) == v2_config.CONTEXT_LENGTH
            and len(chunk["input_ids"]) == v2_config.CONTEXT_LENGTH
            for chunks in split_chunks.values()
            for chunk in chunks
        ),
    }
    _write_json(run_directory / "provenance_report.json", provenance_report)
    _write_json(run_directory / "quality_report.json", quality_report)
    output_hashes["provenance_report.json"] = sha256_file(
        run_directory / "provenance_report.json"
    )
    output_hashes["quality_report.json"] = sha256_file(
        run_directory / "quality_report.json"
    )
    manifest = {
        "schema_version": 1,
        "run_status": "staged_not_added_to_existing_dataset",
        "existing_dataset_modified": False,
        "training_started": False,
        "input_source_count": len(rows),
        "unique_document_count": len(documents),
        "exact_duplicate_source_count": len(exact_duplicates),
        "split_ratios": SPLIT_RATIOS,
        "split_statistics": split_statistics,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_artifact_sha256": tokenizer_artifact_hashes,
        "source_manifest_sha256": None,
        "output_sha256": output_hashes,
    }
    _write_json(run_directory / "ingestion_manifest.json", manifest)
    return manifest


def run_ingestion(
    manifest_path: str | Path,
    input_root: str | Path,
    run_id: str,
    staging_root: str | Path = INGESTION_RUNS_DIR,
) -> dict[str, Any]:
    """Stage an ingestion under an isolated run directory; never edit V2 splits."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must use 1-80 letters, digits, underscores, or hyphens.")
    stage_root = Path(staging_root).resolve()
    ingestion_root = INGESTION_RUNS_DIR.resolve()
    if stage_root.is_relative_to(V2_ROOT.resolve()) and not stage_root.is_relative_to(
        ingestion_root
    ):
        raise ValueError("V2-local ingestion outputs must be under data/ingestion_runs.")
    protected_paths = (
        v2_config.DATA_DIR / "splits",
        v2_config.DATA_DIR / "chunks",
        v2_config.DATA_DIR / "manifests",
        v2_config.TOKENIZER_DIR,
        v2_config.CHECKPOINT_DIR,
    )
    for protected_path in protected_paths:
        protected = protected_path.resolve()
        if stage_root.is_relative_to(protected) or protected.is_relative_to(stage_root):
            raise ValueError(
                f"Ingestion staging may not overlap protected path: {protected_path}."
            )
    source_manifest = Path(manifest_path).resolve(strict=True)
    source_root = Path(input_root).resolve(strict=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    final_directory = stage_root / run_id
    if final_directory.exists():
        raise FileExistsError(f"Ingestion output already exists: {final_directory}")
    if not final_directory.resolve().is_relative_to(stage_root):
        raise ValueError("Ingestion output must remain inside the staging root.")

    rows = _load_source_rows(source_manifest, source_root)
    tokenizer = QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)
    if tokenizer.vocab_size != v2_config.VOCAB_SIZE:
        raise ValueError("Existing V2 tokenizer does not match the V2 vocabulary size.")

    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{run_id}-", dir=stage_root)
    )
    try:
        existing_anchors = _load_existing_split_anchors()
        manifest = _write_ingestion_run(
            rows,
            temporary_directory,
            tokenizer,
            existing_anchors,
        )
        manifest["source_manifest_sha256"] = sha256_file(source_manifest)
        _write_json(temporary_directory / "ingestion_manifest.json", manifest)
        if final_directory.exists():
            raise FileExistsError(f"Ingestion output already exists: {final_directory}")
        os.replace(temporary_directory, final_directory)
        return manifest
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)


def _existing_source_quality(text: str) -> list[str]:
    issues: list[str] = []
    if "\ufffd" in text:
        issues.append("replacement character present")
    if CONTROL_PATTERN.search(text):
        issues.append("unsupported control character present")
    if any(line.rstrip() != line for line in text.splitlines()):
        issues.append("trailing whitespace")
    if not text.endswith(("\n", "\r")):
        issues.append("no terminal newline")
    if text.startswith("\ufeff"):
        issues.append("UTF-8 BOM")
    repeated = sum(
        count - 1
        for line, count in Counter(
            line.strip() for line in text.splitlines() if len(line.strip()) >= 40
        ).items()
    )
    if repeated:
        issues.append(f"{repeated} repeated substantive line occurrences")
    if not text.strip():
        issues.append("empty document")
    return issues


def audit_existing_sources() -> dict[str, Any]:
    """Return a document-level audit using the V2 source snapshot and reports."""
    inventory = _read_json(SOURCE_INVENTORY_FILE)
    quality_report = _read_json(DATA_QUALITY_FILE)
    duplicate_report = _read_json(DUPLICATE_REPORT_FILE)
    statistics = _read_json(DATASET_STATISTICS_FILE)
    dataset_manifest = _read_json(DATASET_MANIFEST_FILE)
    tokenizer = QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)

    duplicate_sources = {
        duplicate
        for group in duplicate_report.get("exact_duplicate_groups", [])
        for duplicate in group.get("duplicate_sources", [])
    }
    split_by_source = {
        record.get("source_path"): record.get("split")
        for record in dataset_manifest.get("documents", [])
    }
    documents: list[dict[str, Any]] = []
    category_source_counts: Counter[str] = Counter()
    category_unique_counts: Counter[str] = Counter()
    raw_source_count = 0
    derived_source_count = 0
    for source in inventory.get("files", []):
        source_path = source["source_path"]
        if source_path.startswith("data/raw/"):
            snapshot_path = SOURCE_SNAPSHOT_DIR / "raw" / Path(
                source_path.removeprefix("data/raw/")
            )
            document_role = "raw_source"
            raw_source_count += 1
        elif source_path.startswith("data/processed/"):
            snapshot_path = (
                SOURCE_SNAPSHOT_DIR
                / "processed_reference"
                / Path(source_path).name
            )
            document_role = "derived_aggregate_not_independent_source"
            derived_source_count += 1
        else:
            raise ValueError(f"Unrecognized source inventory path: {source_path}")
        raw_bytes = snapshot_path.read_bytes()
        actual_hash = sha256_bytes(raw_bytes)
        if actual_hash != source.get("sha256"):
            raise ValueError(f"Source snapshot hash mismatch for {source_path}.")
        raw_text = raw_bytes.decode("utf-8-sig", errors="strict")
        token_count = len(tokenizer.encode(raw_text))
        category = source.get("category") or "Uncategorized"
        is_duplicate = source_path in duplicate_sources
        if document_role == "raw_source":
            category_source_counts[category] += 1
            if not is_duplicate:
                category_unique_counts[category] += 1
        quality_issues = _existing_source_quality(raw_text)
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            quality_issues.append("UTF-8 BOM")
        documents.append(
            {
                "source_name": source.get("filename") or Path(source_path).name,
                "original_location": source_path,
                "snapshot_location": snapshot_path.relative_to(V2_ROOT).as_posix(),
                "category": category,
                "document_format": source.get("format", "unknown"),
                "encoding": source.get("encoding", "UTF-8 (strictly decoded)"),
                "license_status": (
                    "not applicable; generated aggregate"
                    if document_role != "raw_source"
                    else "unknown; no source license or permission record"
                ),
                "provenance": source.get(
                    "available_source_information",
                    "Repository path only; author and acquisition details unavailable.",
                ),
                "document_role": document_role,
                "split_status": (
                    "derived; excluded from split"
                    if document_role != "raw_source"
                    else (
                        "excluded exact duplicate"
                        if is_duplicate
                        else split_by_source.get(source_path, "not found in retained splits")
                    )
                ),
                "source_sha256": actual_hash,
                "character_count": len(raw_text),
                "file_size_bytes": len(raw_bytes),
                "approximate_token_count_v2_bpe_without_special_tokens": token_count,
                "quality_issues": quality_issues,
            }
        )

    formatting_issue_counts = Counter(
        issue
        for document in documents
        for issue in document["quality_issues"]
    )
    return {
        "schema_version": 1,
        "generated_by": "src/dataset_ingestion.py audit",
        "status": quality_report.get("status", "unknown"),
        "scope": "Read-only audit of all source_inventory entries and their V2 snapshots.",
        "raw_source_document_count": raw_source_count,
        "derived_aggregate_count": derived_source_count,
        "retained_unique_document_count": quality_report.get(
            "retained_unique_document_count"
        ),
        "license_status": quality_report.get(
            "provenance_and_licensing", {}
        ).get("license_status", "unknown"),
        "training_use_status": "not approved pending source-license verification",
        "license_warning": (
            "Repository presence and copied source files do not establish training rights."
        ),
        "subject_distribution_raw": dict(sorted(category_source_counts.items())),
        "subject_distribution_unique_raw": dict(
            sorted(category_unique_counts.items())
        ),
        "subject_distribution_current_splits": statistics.get(
            "category_distribution", {}
        ),
        "subject_token_counts_current_splits": {
            category: sum(
                split_info.get("category_token_counts", {}).get(category, 0)
                for split_info in quality_report.get(
                    "tokenized_split_statistics", {}
                ).values()
            )
            for category in sorted(
                {
                    category
                    for split_info in quality_report.get(
                        "tokenized_split_statistics", {}
                    ).values()
                    for category in split_info.get("category_token_counts", {})
                }
            )
        },
        "subject_token_count_method": (
            "Existing V2 BPE token counts by subject, including BOS and EOS."
        ),
        "source_quality_issue_counts": dict(sorted(formatting_issue_counts.items())),
        "duplicate_audit": {
            "exact_duplicate_documents_removed": duplicate_report.get(
                "exact_duplicate_documents_removed"
            ),
            "normalized_duplicate_documents_removed": duplicate_report.get(
                "normalized_duplicate_documents_removed"
            ),
            "near_duplicate_documents_removed": duplicate_report.get(
                "near_duplicate_detection", {}
            ).get("near_duplicate_documents_removed"),
            "shared_substantive_line_count": duplicate_report.get(
                "repeated_lines", {}
            ).get("shared_substantive_line_count"),
            "shared_passage_count": duplicate_report.get(
                "repeated_passages", {}
            ).get("shared_passage_count"),
        },
        "split_contamination": {
            **quality_report.get("split_leakage_validation", {}),
            "train_validation_test_document_counts": {
                split: statistics.get("split_statistics", {})
                .get(split, {})
                .get("document_count", 0)
                for split in ("train", "validation", "test")
            },
            "assessment": (
                "Recorded lexical and duplicate checks pass; semantic paraphrase "
                "contamination is not ruled out."
            ),
        },
        "quality_findings": {
            "empty_documents": quality_report.get("empty_document_count"),
            "corrupted_documents": quality_report.get("corrupted_document_count"),
            "invalid_encoding_documents": quality_report.get("invalid_encoding_count"),
            "documents_exceeding_context_length": quality_report.get(
                "model_context_compatibility", {}
            ).get("documents_exceeding_context_length"),
            "documents_truncated": quality_report.get(
                "model_context_compatibility", {}
            ).get("documents_truncated"),
            "human_factuality_review": "not established by automated quality checks",
            "known_limitations": quality_report.get("known_limitations", []),
        },
        "source_documents": sorted(
            documents, key=lambda item: item["original_location"]
        ),
        "preservation": {
            "existing_splits_modified": False,
            "existing_chunks_modified": False,
            "tokenizer_modified": False,
            "training_started": False,
        },
    }


def write_existing_provenance_report(
    output_path: str | Path = PROVENANCE_REPORT_FILE,
) -> dict[str, Any]:
    report = audit_existing_sources()
    _write_json(Path(output_path), report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="Write the existing-source audit.")
    audit_parser.add_argument(
        "--output",
        type=Path,
        default=PROVENANCE_REPORT_FILE,
    )
    ingest_parser = commands.add_parser(
        "ingest",
        help="Stage approved UTF-8 text sources without changing current splits.",
    )
    ingest_parser.add_argument("--manifest", type=Path, required=True)
    ingest_parser.add_argument("--input-root", type=Path, required=True)
    ingest_parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()

    if arguments.command == "audit":
        report = write_existing_provenance_report(arguments.output)
        print(
            json.dumps(
                {
                    "report": str(arguments.output.resolve()),
                    "raw_source_document_count": report["raw_source_document_count"],
                    "license_status": report["license_status"],
                },
                sort_keys=True,
            )
        )
    else:
        manifest = run_ingestion(
            arguments.manifest,
            arguments.input_root,
            arguments.run_id,
        )
        print(
            json.dumps(
                {
                    "run_id": arguments.run_id,
                    "output_directory": str(
                        (INGESTION_RUNS_DIR / arguments.run_id).resolve()
                    ),
                    "unique_document_count": manifest["unique_document_count"],
                    "run_status": manifest["run_status"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    _main()
