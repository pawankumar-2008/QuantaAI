"""Offline-only dataset registration, cleaning, deduplication, and release tools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_V2_ROOT = Path(__file__).resolve().parents[1]
if __package__:
    from . import dataset_ingestion as ingestion
    from .bpe_tokenizer import QuantaBPETokenizer
    from .workspace_config import v2_config
else:
    sys.path.insert(0, str(_V2_ROOT))
    import src.dataset_ingestion as ingestion
    from src.bpe_tokenizer import QuantaBPETokenizer
    from src.workspace_config import v2_config

DATA_DIR = v2_config.DATA_DIR
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
REGISTRY_DIR = METADATA_DIR / "registrations"
IMPORT_DIR = PROCESSED_DIR / "imports"
CLEANED_DIR = PROCESSED_DIR / "cleaned"
DEDUPLICATED_DIR = PROCESSED_DIR / "deduplicated"
RELEASES_DIR = DATA_DIR / "releases"
RELEASE_REPORTS_DIR = DATA_DIR / "reports" / "releases"

CATEGORIES = (
    "general_english",
    "science",
    "mathematics",
    "computer_science",
    "programming",
    "general_knowledge",
    "education",
)
CATEGORY_TOKEN_TARGETS = {
    "general_english": 500_000,
    "science": 500_000,
    "mathematics": 300_000,
    "computer_science": 500_000,
    "programming": 500_000,
    "general_knowledge": 500_000,
    "education": 500_000,
}
SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}
SUPPORTED_FORMATS = {"txt", "csv", "json", "jsonl"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
CANONICAL_LICENSES = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "CC-BY-SA-2.5": "https://creativecommons.org/licenses/by-sa/2.5/",
    "PSF-2.0": "https://docs.python.org/3/license.html",
    "0BSD": "https://opensource.org/license/0bsd",
}
SHARE_ALIKE_LICENSES = {"CC-BY-SA-2.5", "CC-BY-SA-3.0", "CC-BY-SA-4.0"}
DOCUMENT_FIELDS = (
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
    "attribution_requirements",
    "language",
)
ENGLISH_SIGNAL_WORDS = frozenset(
    "a about after all also an and are as at be because been before being between "
    "both but by can could did do does during each few for from had has have he her "
    "here hers him his how i if in into is it its may more most must my no not of "
    "on one only or other our out over same she should since so some such than that "
    "the their them then there these they this those through to under until up us "
    "was we were what when where which while who will with would you your".split()
)


class SourceMetadataError(ValueError):
    """Raised when registered source metadata is incomplete or unsupported."""


class _VisibleTextParser(HTMLParser):
    BLOCK_TAGS = {
        "article", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li",
        "p", "section", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden_depth += 1
        elif not self.hidden_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden_depth:
            self.hidden_depth -= 1
        elif not self.hidden_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


def _sha256_file(path: Path) -> str:
    return ingestion.sha256_file(path)


def _validate_https(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceMetadataError(f"{field} is required.")
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise SourceMetadataError(f"{field} must be an absolute HTTPS URL.")
    return value.strip()


def _validate_iso_date(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceMetadataError(f"{field} is required.")
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            date.fromisoformat(value.strip())
        except ValueError as error:
            raise SourceMetadataError(f"{field} must be an ISO-8601 date/time.") from error
    return value.strip()


def validate_document_metadata(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SourceMetadataError("Document metadata must be a JSON object.")
    normalized = dict(record)
    for field in DOCUMENT_FIELDS:
        if field in {
            "license_verified",
            "license_scope_reviewed",
            "third_party_content_reviewed",
        }:
            continue
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SourceMetadataError(f"Document field {field} is required.")
        normalized[field] = value.strip()
    if normalized["category"] not in CATEGORIES:
        raise SourceMetadataError(f"Unsupported category: {normalized['category']!r}.")
    if normalized["language"].casefold() not in {"en", "eng", "english"}:
        raise SourceMetadataError("Only English documents can enter this release workflow.")
    license_id = normalized["license_id"]
    if license_id not in CANONICAL_LICENSES:
        raise SourceMetadataError(f"License {license_id!r} is not approved by the release policy.")
    normalized["source_url"] = _validate_https(normalized["source_url"], "source_url")
    normalized["license_url"] = _validate_https(normalized["license_url"], "license_url")
    if normalized["license_url"].rstrip("/") != CANONICAL_LICENSES[license_id].rstrip("/"):
        raise SourceMetadataError(f"license_url is not canonical for {license_id}.")
    for flag in ("license_verified", "license_scope_reviewed", "third_party_content_reviewed"):
        if normalized[flag] is not True:
            raise SourceMetadataError(f"{flag} must be explicitly true per document.")
    if license_id in SHARE_ALIKE_LICENSES:
        if normalized.get("share_alike_reviewed") is not True:
            raise SourceMetadataError(
                "Share-alike content requires explicit share_alike_reviewed=true."
            )
        if not str(normalized.get("share_alike_review_notes", "")).strip():
            raise SourceMetadataError("Share-alike review notes are required.")
    normalized["source_retrieved_at"] = _validate_iso_date(
        normalized["source_retrieved_at"], "source_retrieved_at"
    )
    normalized["license_verified_at"] = _validate_iso_date(
        normalized["license_verified_at"], "license_verified_at"
    )
    return normalized


def _safe_dataset_id(value: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError("dataset_id must use 1-80 letters, digits, dots, underscores, or hyphens.")
    return value


def _registration_path(dataset_id: str, data_dir: str | Path = DATA_DIR) -> Path:
    return Path(data_dir) / "metadata" / "registrations" / f"{_safe_dataset_id(dataset_id)}.json"


def _load_registration(dataset_id: str, data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    _safe_dataset_id(dataset_id)
    path = _registration_path(dataset_id, data_dir)
    registration = _read_json(path)
    if registration.get("dataset_id") != dataset_id:
        raise SourceMetadataError(f"Registration ID does not match {path}.")
    data_root = Path(data_dir).resolve()
    raw_root = (data_root / "raw").resolve()
    source_path = (data_root / registration["registered_path"]).resolve()
    if not source_path.is_relative_to(raw_root):
        raise SourceMetadataError("Registered source path must remain inside data/raw/.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Registered source file is missing: {source_path}")
    if _sha256_file(source_path) != registration["file_sha256"]:
        raise ValueError(f"Registered source checksum changed: {source_path}")
    registration["_source_path"] = source_path
    return registration


def register_dataset(
    source_file: str | Path,
    metadata: dict[str, Any],
    *,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Copy one local source into raw/ and register its immutable file checksum."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")
    dataset_id = _safe_dataset_id(metadata.get("dataset_id"))
    for field in ("dataset_name", "original_source", "download_location"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise SourceMetadataError(f"{field} is required.")
    source_url = _validate_https(metadata.get("source_url"), "source_url")
    download_date = _validate_iso_date(metadata.get("download_date"), "download_date")
    category = metadata.get("category")
    if category not in CATEGORIES:
        raise SourceMetadataError(f"Unsupported category: {category!r}.")

    source = Path(source_file).resolve(strict=True)
    if not source.is_file() or source.suffix.lower().lstrip(".") not in SUPPORTED_FORMATS:
        raise ValueError(f"Source must be a .txt, .csv, .json, or .jsonl file: {source}")
    file_format = source.suffix.lower().lstrip(".")
    registration = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_name": metadata["dataset_name"].strip(),
        "original_source": metadata["original_source"].strip(),
        "source_url": source_url,
        "download_location": metadata["download_location"].strip(),
        "download_date": download_date,
        "category": category,
        "format": file_format,
        "text_field": metadata.get("text_field", "text"),
        "records_key": metadata.get("records_key"),
        "record_fields": metadata.get("record_fields"),
        "document_metadata": metadata.get("document_metadata"),
        "strip_html": metadata.get("strip_html", False),
        "minimum_characters": metadata.get("minimum_characters", 80),
    }
    if not isinstance(registration["minimum_characters"], int) or registration["minimum_characters"] < 1:
        raise SourceMetadataError("minimum_characters must be a positive integer.")
    if not isinstance(registration["strip_html"], bool):
        raise SourceMetadataError("strip_html must be a boolean.")
    if file_format == "txt":
        document_metadata = validate_document_metadata(registration["document_metadata"])
        if document_metadata["category"] != category:
            raise SourceMetadataError("TXT document category must match its registered category.")
        registration["document_metadata"] = document_metadata
    else:
        fields = registration["record_fields"]
        required_fields = {"text", *DOCUMENT_FIELDS}
        if not isinstance(fields, dict) or not required_fields.issubset(fields):
            raise SourceMetadataError(
                "Structured imports require record_fields mappings for text and every "
                "document-level provenance/license field."
            )
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise SourceMetadataError("record_fields values must be non-empty field paths.")
        if file_format == "csv" and any("." in value for value in fields.values()):
            raise SourceMetadataError("CSV record_fields must be flat column names.")

    root = Path(data_dir).resolve()
    target_dir = root / "raw" / category / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists():
        raise FileExistsError(f"Registered source file already exists: {target}")
    shutil.copy2(source, target)
    registration.update(
        {
            "registered_path": target.relative_to(root).as_posix(),
            "file_name": target.name,
            "file_size_bytes": target.stat().st_size,
            "file_sha256": _sha256_file(target),
        }
    )
    _write_json(_registration_path(dataset_id, root), registration)
    return {key: value for key, value in registration.items() if not key.startswith("_")}


def _get_field(record: dict[str, Any], field_path: str) -> Any:
    value: Any = record
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise SourceMetadataError(f"Missing record field {field_path!r}.")
        value = value[part]
    return value


def _read_source_records(registration: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    path = registration["_source_path"]
    file_format = registration["format"]
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"Registered source is not valid UTF-8: {path}") from error
    if file_format == "txt":
        return [(text, registration["document_metadata"])]
    if file_format == "csv":
        rows = list(csv.DictReader(text.splitlines()))
    elif file_format == "jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from error
    else:
        payload = json.loads(text)
        if isinstance(payload, dict) and registration.get("records_key"):
            payload = _get_field(payload, registration["records_key"])
        rows = payload if isinstance(payload, list) else [payload]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No records found in {path}.")

    fields = registration["record_fields"]
    output: list[tuple[str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SourceMetadataError(f"Record {index} in {path} must be a JSON object.")
        raw_text = _get_field(row, fields["text"])
        if not isinstance(raw_text, str):
            raise SourceMetadataError(f"Text field in record {index} must be a string.")
        metadata = {field: _get_field(row, fields[field]) for field in DOCUMENT_FIELDS}
        for flag in ("license_verified", "license_scope_reviewed", "third_party_content_reviewed", "share_alike_reviewed"):
            if flag not in metadata and fields.get(flag):
                metadata[flag] = _get_field(row, fields[flag])
            value = metadata.get(flag)
            if isinstance(value, str):
                metadata[flag] = value.strip().casefold() == "true"
        notes_field = fields.get("share_alike_review_notes")
        metadata["share_alike_review_notes"] = (
            _get_field(row, notes_field) if notes_field else ""
        )
        metadata = validate_document_metadata(metadata)
        if metadata["category"] != registration["category"]:
            raise SourceMetadataError(
                f"Record {metadata['source_id']} category differs from registered category."
            )
        if metadata["source_id"] in seen_ids:
            raise SourceMetadataError(f"Duplicate source_id {metadata['source_id']!r}.")
        seen_ids.add(metadata["source_id"])
        output.append((raw_text, metadata))
    return output


def validate_dataset(dataset_id: str, *, data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    registration = _load_registration(dataset_id, data_dir)
    records = _read_source_records(registration)
    return {
        "dataset_id": dataset_id,
        "format": registration["format"],
        "file_sha256": registration["file_sha256"],
        "record_count": len(records),
        "license_distribution": dict(sorted(Counter(meta["license_id"] for _, meta in records).items())),
        "document_metadata_validated_per_record": True,
        "network_access_performed": False,
    }


def import_dataset(dataset_id: str, *, data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    _safe_dataset_id(dataset_id)
    registration = _load_registration(dataset_id, data_dir)
    records = _read_source_records(registration)
    output_dir = Path(data_dir) / "processed" / "imports" / dataset_id
    if output_dir.exists():
        raise FileExistsError(f"Import output already exists: {output_dir}")
    documents = []
    for raw_text, metadata in records:
        documents.append(
            {
                **metadata,
                "dataset_id": dataset_id,
                "dataset_name": registration["dataset_name"],
                "download_date": registration["download_date"],
                "download_location": registration["download_location"],
                "registered_path": registration["registered_path"],
                "registered_file_sha256": registration["file_sha256"],
                "raw_text": raw_text,
            }
        )
    _write_jsonl(output_dir / "documents.jsonl", documents)
    report = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "imported_document_count": len(documents),
        "source_file_sha256": registration["file_sha256"],
        "source_format": registration["format"],
        "output_sha256": _sha256_file(output_dir / "documents.jsonl"),
        "network_access_performed": False,
    }
    _write_json(output_dir / "import_report.json", report)
    return report


def _strip_html(raw_text: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(raw_text)
    parser.close()
    return html.unescape("".join(parser.parts))


def _english_signal_ratio(text: str) -> tuple[int, float]:
    words = ingestion.WORD_PATTERN.findall(text.casefold())
    if not words:
        return 0, 0.0
    signals = sum(word in ENGLISH_SIGNAL_WORDS for word in words)
    return len(words), signals / len(words)


def _check_english(text: str, declared_language: str) -> tuple[bool, dict[str, Any]]:
    if declared_language.casefold() not in {"en", "eng", "english"}:
        return False, {"declared_language": declared_language, "status": "rejected_non_english"}
    word_count, signal_ratio = _english_signal_ratio(text)
    if word_count >= 25 and signal_ratio < 0.02:
        return False, {
            "declared_language": declared_language,
            "english_signal_word_ratio": signal_ratio,
            "status": "rejected_language_mismatch_or_low_confidence",
        }
    return True, {
        "declared_language": declared_language,
        "english_signal_word_ratio": signal_ratio,
        "status": "declared_english_heuristic_passed" if word_count >= 25 else "declared_english_short_document",
    }


def clean_dataset(
    dataset_id: str,
    *,
    data_dir: str | Path = DATA_DIR,
    minimum_characters: int | None = None,
) -> dict[str, Any]:
    _safe_dataset_id(dataset_id)
    registration = _load_registration(dataset_id, data_dir)
    import_path = Path(data_dir) / "processed" / "imports" / dataset_id / "documents.jsonl"
    if not import_path.is_file():
        raise FileNotFoundError(f"Import dataset before cleaning: {import_path}")
    output_dir = Path(data_dir) / "processed" / "cleaned" / dataset_id
    if output_dir.exists():
        raise FileExistsError(f"Cleaned output already exists: {output_dir}")
    minimum = minimum_characters or registration["minimum_characters"]
    records = [json.loads(line) for line in import_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted = []
    removed: list[dict[str, str]] = []
    action_counts: Counter[str] = Counter()
    for record in records:
        raw_text = record.pop("raw_text")
        if registration["strip_html"]:
            raw_text = _strip_html(raw_text)
            action_counts["html_cleaned"] += 1
        try:
            cleaned, actions = ingestion._clean_text(raw_text)
        except (TypeError, ValueError) as error:
            removed.append({"source_id": record["source_id"], "reason": str(error)})
            continue
        if "\ufffd" in cleaned:
            removed.append({"source_id": record["source_id"], "reason": "replacement character indicates corrupted text"})
            continue
        if len(cleaned) < minimum:
            removed.append({"source_id": record["source_id"], "reason": f"below minimum length {minimum}"})
            continue
        language_ok, language_result = _check_english(cleaned, record["language"])
        if not language_ok:
            removed.append({"source_id": record["source_id"], "reason": language_result["status"]})
            continue
        action_counts.update({key: value for key, value in actions.items() if isinstance(value, int)})
        accepted.append(
            {
                **record,
                "text": cleaned,
                "clean_text_sha256": ingestion.sha256_bytes(cleaned.encode("utf-8")),
                "character_count": len(cleaned),
                "word_count": len(ingestion.WORD_PATTERN.findall(cleaned)),
                "language_check": language_result,
                "cleaning_actions": actions,
            }
        )
    _write_jsonl(output_dir / "documents.jsonl", accepted)
    report = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "input_document_count": len(records),
        "retained_document_count": len(accepted),
        "removed_document_count": len(removed),
        "minimum_characters": minimum,
        "cleaning_actions": dict(sorted(action_counts.items())),
        "removed_documents": removed,
        "output_sha256": _sha256_file(output_dir / "documents.jsonl"),
        "processing_config": {
            "unicode_normalization": "NFC",
            "html_cleanup": registration["strip_html"],
            "language_filter": "declared English plus common-English-word heuristic v1",
        },
    }
    _write_json(output_dir / "cleaning_report.json", report)
    return report


def deduplicate_datasets(
    dataset_ids: list[str],
    dedupe_id: str,
    *,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    if not dataset_ids:
        raise ValueError("At least one dataset_id is required.")
    _safe_dataset_id(dedupe_id)
    docs = []
    imported_document_count = 0
    cleaning_removed_count = 0
    for dataset_id in sorted(set(dataset_ids)):
        _safe_dataset_id(dataset_id)
        path = Path(data_dir) / "processed" / "cleaned" / dataset_id / "documents.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Cleaned dataset is missing: {path}")
        docs.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        cleaning_report_path = path.parent / "cleaning_report.json"
        if cleaning_report_path.is_file():
            cleaning_report = _read_json(cleaning_report_path)
            imported_document_count += cleaning_report["input_document_count"]
            cleaning_removed_count += cleaning_report["removed_document_count"]
    if not docs:
        raise ValueError("Selected datasets contain no cleaned documents.")
    source_ids = [record["source_id"] for record in docs]
    if len(source_ids) != len(set(source_ids)):
        raise SourceMetadataError("source_id values must be unique across selected datasets.")
    unique, duplicates = ingestion._deduplicate_sources(docs)
    group_ids, relations = ingestion._leakage_groups(unique)
    for document in unique:
        document["leakage_group_id"] = group_ids[document["document_id"]]
    output_dir = Path(data_dir) / "processed" / "deduplicated" / dedupe_id
    if output_dir.exists():
        raise FileExistsError(f"Deduplicated output already exists: {output_dir}")
    _write_jsonl(output_dir / "documents.jsonl", unique)
    report = {
        "schema_version": 1,
        "dedupe_id": dedupe_id,
        "input_dataset_ids": sorted(set(dataset_ids)),
        "input_document_count": len(docs),
        "imported_document_count": imported_document_count or len(docs) + cleaning_removed_count,
        "cleaning_removed_document_count": cleaning_removed_count,
        "retained_unique_document_count": len(unique),
        "exact_duplicate_documents_removed": len(duplicates),
        "near_duplicate_or_shared_content_relations": relations,
        "near_duplicate_document_count": len(
            {
                document_id
                for relation in relations
                if any(reason.startswith("near_duplicate") for reason in relation["reasons"])
                for document_id in relation["document_ids"]
            }
        ),
        "leakage_group_count": len(set(group_ids.values())),
        "duplicates": duplicates,
        "output_sha256": _sha256_file(output_dir / "documents.jsonl"),
        "processing_config": {
            "exact_duplicates": "NFC/line-ending/trailing-space cleaned text SHA-256",
            "near_duplicates": "5-word-shingle Jaccard >= 0.85; retain and group in one split",
        },
    }
    _write_json(output_dir / "deduplication_report.json", report)
    return report


def _tokenizer_hash() -> tuple[str, dict[str, str]]:
    tokenizer_dir = v2_config.TOKENIZER_DIR
    files = ("bpe_vocab.json", "bpe_merges.json", "tokenizer_config.json")
    hashes = {name: _sha256_file(tokenizer_dir / name) for name in files}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ingestion.sha256_bytes(canonical), hashes


def _document_statistics(documents: list[dict[str, Any]], tokenizer: QuantaBPETokenizer) -> dict[str, Any]:
    categories = {category: [] for category in CATEGORIES}
    licenses: Counter[str] = Counter()
    for document in documents:
        categories[document["category"]].append(document)
        licenses[document["license_id"]] += 1
    category_statistics = {}
    for category, items in categories.items():
        token_count = sum(len(tokenizer.encode(item["text"], add_special_tokens=True)) for item in items)
        category_statistics[category] = {
            "document_count": len(items),
            "character_count": sum(item["character_count"] for item in items),
            "word_count": sum(item["word_count"] for item in items),
            "token_count_including_bos_eos": token_count,
            "target_tokens": CATEGORY_TOKEN_TARGETS[category],
            "target_shortfall_tokens": max(0, CATEGORY_TOKEN_TARGETS[category] - token_count),
            "sufficient_for_target": token_count >= CATEGORY_TOKEN_TARGETS[category],
        }
    return {
        "document_count": len(documents),
        "character_count": sum(item["character_count"] for item in documents),
        "word_count": sum(item["word_count"] for item in documents),
        "token_count_including_bos_eos": sum(
            len(tokenizer.encode(item["text"], add_special_tokens=True)) for item in documents
        ),
        "documents_per_category": {key: value["document_count"] for key, value in category_statistics.items()},
        "tokens_per_category": {
            key: value["token_count_including_bos_eos"] for key, value in category_statistics.items()
        },
        "category_statistics": category_statistics,
        "license_distribution_documents": dict(sorted(licenses.items())),
        "insufficient_categories": [
            category for category, details in category_statistics.items()
            if not details["sufficient_for_target"]
        ],
    }


def _build_release_directory(
    dedupe_id: str,
    version: str,
    release_dir: Path,
    *,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    _safe_dataset_id(version)
    _safe_dataset_id(dedupe_id)
    source_dir = Path(data_dir) / "processed" / "deduplicated" / dedupe_id
    source_file = source_dir / "documents.jsonl"
    if not source_file.is_file():
        raise FileNotFoundError(f"Deduplicated documents are missing: {source_file}")
    documents = [json.loads(line) for line in source_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({document["leakage_group_id"] for document in documents}) < 3:
        raise ValueError("At least three independent leakage groups are required for a release.")
    tokenizer = QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)
    token_hash, token_files = _tokenizer_hash()
    group_ids = {document["document_id"]: document["leakage_group_id"] for document in documents}
    assignments = ingestion._assign_splits(documents, group_ids, ratios=SPLIT_RATIOS)

    split_records = {split: [] for split in SPLIT_RATIOS}
    for document in sorted(documents, key=lambda item: item["document_id"]):
        split = assignments[document["document_id"]]
        token_count = len(tokenizer.encode(document["text"], add_special_tokens=True))
        split_records[split].append(
            {
                **document,
                "split": split,
                "token_count_including_bos_eos": token_count,
            }
        )

    release_dir.mkdir(parents=True, exist_ok=False)
    split_hashes = {}
    split_statistics = {}
    for split, records in split_records.items():
        output = release_dir / split / "documents.jsonl"
        _write_jsonl(output, records)
        split_hashes[f"{split}/documents.jsonl"] = _sha256_file(output)
        split_statistics[split] = _document_statistics(records, tokenizer)
    all_statistics = _document_statistics(documents, tokenizer)
    dedupe_report = _read_json(source_dir / "deduplication_report.json")
    registration_hashes = {}
    for dataset_id in dedupe_report["input_dataset_ids"]:
        registration = _load_registration(dataset_id, data_dir)
        cleaned_report = Path(data_dir) / "processed" / "cleaned" / dataset_id / "cleaning_report.json"
        registration_hashes[dataset_id] = {
            "registration_sha256": _sha256_file(_registration_path(dataset_id, data_dir)),
            "source_file_sha256": registration["file_sha256"],
            "registered_path": registration["registered_path"],
            "format": registration["format"],
            "category": registration["category"],
            "download_date": registration["download_date"],
            "minimum_characters": registration["minimum_characters"],
            "strip_html": registration["strip_html"],
            "cleaning_report_sha256": _sha256_file(cleaned_report),
        }
    processor_hashes = {
        path.name: _sha256_file(path)
        for path in (
            _V2_ROOT / "src" / "dataset_release.py",
            _V2_ROOT / "src" / "dataset_ingestion.py",
        )
    }
    manifest = {
        "schema_version": 1,
        "dataset_version": version,
        "release_status": "immutable_local_release_requires_rights_and_quality_review",
        "training_started": False,
        "source_dedupe_id": dedupe_id,
        "input_registrations": registration_hashes,
        "deduplicated_documents_sha256": _sha256_file(source_file),
        "deduplication_report_sha256": _sha256_file(
            source_dir / "deduplication_report.json"
        ),
        "document_count": len(documents),
        "split_ratios": SPLIT_RATIOS,
        "split_statistics": split_statistics,
        "overall_statistics": all_statistics,
        "duplicate_summary": {
            "exact_duplicate_documents_removed": dedupe_report["exact_duplicate_documents_removed"],
            "cleaning_removed_document_count": dedupe_report.get("cleaning_removed_document_count", 0),
            "all_removed_document_count": (
                dedupe_report["exact_duplicate_documents_removed"]
                + dedupe_report.get("cleaning_removed_document_count", 0)
            ),
            "near_duplicate_or_shared_content_relations": len(dedupe_report["near_duplicate_or_shared_content_relations"]),
        },
        "tokenizer": {
            "identity_sha256": token_hash,
            "artifact_sha256": token_files,
            "vocabulary_size": tokenizer.vocab_size,
            "tokenizer_artifacts_modified": False,
        },
        "processing_config": {
            "unicode_normalization": "NFC",
            "minimum_characters": "registration-specific, default 80",
            "language_filter": "declared English and common-English-word heuristic v1",
            "split_ratios": SPLIT_RATIOS,
            "leakage_policy": "near duplicates/shared substantive content remain in one split",
            "category_token_targets": CATEGORY_TOKEN_TARGETS,
            "processor_sha256": processor_hashes,
        },
        "file_sha256": split_hashes,
    }
    _write_json(release_dir / "release_manifest.json", manifest)
    manifest["manifest_sha256"] = _sha256_file(release_dir / "release_manifest.json")
    return manifest


def create_release(
    dedupe_id: str,
    version: str,
    *,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    _safe_dataset_id(version)
    releases_root = Path(data_dir) / "releases"
    releases_root.mkdir(parents=True, exist_ok=True)
    final_directory = releases_root / version
    if final_directory.exists():
        raise FileExistsError(f"Dataset release already exists: {final_directory}")
    report_path = Path(data_dir) / "reports" / "releases" / version / "dataset_report.json"
    if report_path.exists():
        raise FileExistsError(f"Dataset release report already exists: {report_path}")
    with tempfile.TemporaryDirectory(prefix=f".{version}-", dir=releases_root) as temporary:
        staging_directory = Path(temporary) / "release"
        manifest = _build_release_directory(
            dedupe_id,
            version,
            staging_directory,
            data_dir=data_dir,
        )
        if final_directory.exists():
            raise FileExistsError(f"Dataset release already exists: {final_directory}")
        os.replace(staging_directory, final_directory)
    _write_json(report_path, manifest)
    return manifest


def inspect_datasets(*, data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    root = Path(data_dir)
    registrations = []
    registry = root / "metadata" / "registrations"
    if registry.is_dir():
        for path in sorted(registry.glob("*.json")):
            record = _read_json(path)
            registrations.append(
                {
                    "dataset_id": record.get("dataset_id"),
                    "dataset_name": record.get("dataset_name"),
                    "category": record.get("category"),
                    "format": record.get("format"),
                    "file_sha256": record.get("file_sha256"),
                }
            )
    raw_files = [path for path in (root / "raw").rglob("*") if path.is_file()]
    return {
        "registered_datasets": registrations,
        "registered_dataset_count": len(registrations),
        "raw_file_count": len(raw_files),
        "category_targets_tokens": CATEGORY_TOKEN_TARGETS,
        "license_policy": {
            "allowed": sorted(CANONICAL_LICENSES),
            "share_alike_requires_separate_review": True,
            "non_commercial_and_unknown_licenses_allowed": False,
        },
        "current_checkpoint_bound_splits_modified": False,
    }


def get_release_statistics(version: str, *, data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    manifest_path = Path(data_dir) / "releases" / _safe_dataset_id(version) / "release_manifest.json"
    return _read_json(manifest_path)


def get_candidate_statistics(dedupe_id: str, *, data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    root = Path(data_dir) / "processed" / "deduplicated" / _safe_dataset_id(dedupe_id)
    document_path = root / "documents.jsonl"
    report_path = root / "deduplication_report.json"
    if not document_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"Deduplicated candidate not found: {root}")
    documents = [
        json.loads(line)
        for line in document_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)
    deduplication = _read_json(report_path)
    return {
        "dedupe_id": dedupe_id,
        "overall_statistics": _document_statistics(documents, tokenizer),
        "exact_duplicate_documents_removed": deduplication[
            "exact_duplicate_documents_removed"
        ],
        "cleaning_removed_document_count": deduplication.get(
            "cleaning_removed_document_count", 0
        ),
        "near_duplicate_or_shared_content_relations": len(
            deduplication["near_duplicate_or_shared_content_relations"]
        ),
        "deduplicated_documents_sha256": _sha256_file(document_path),
    }


def _load_metadata_file(path: Path) -> dict[str, Any]:
    return _read_json(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("inspect", help="Inspect registered datasets and category targets.")

    register = commands.add_parser("register", help="Copy and register a local source file; no network access is used.")
    register.add_argument("--source-file", type=Path, required=True)
    register.add_argument("--metadata", type=Path, required=True, help="JSON registration metadata; license fields are per document.")

    for command_name, help_text in (
        ("validate", "Validate checksums, content records, language declarations, and per-document license evidence."),
        ("import", "Parse a registered TXT, CSV, JSON, or JSONL source into immutable import records."),
        ("clean", "Normalize, HTML-clean when configured, language-filter, and remove invalid/short documents."),
    ):
        command = commands.add_parser(command_name, help=help_text)
        command.add_argument("--dataset-id", required=True)
        if command_name == "clean":
            command.add_argument("--minimum-characters", type=int)

    dedupe = commands.add_parser("dedupe", help="Remove exact duplicates and group near duplicates for split protection.")
    dedupe.add_argument("--dataset-ids", nargs="+", required=True)
    dedupe.add_argument("--dedupe-id", required=True)

    stats = commands.add_parser("stats", help="Print statistics for a candidate or immutable release.")
    stats_source = stats.add_mutually_exclusive_group(required=True)
    stats_source.add_argument("--dedupe-id", help="Report statistics before creating a release.")
    stats_source.add_argument("--version", help="Report statistics for an immutable release.")

    release = commands.add_parser("release", help="Create a new immutable 90/5/5 local dataset release.")
    release.add_argument("--dedupe-id", required=True)
    release.add_argument("--version", required=True)

    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_datasets()
    elif args.command == "register":
        result = register_dataset(args.source_file, _load_metadata_file(args.metadata))
    elif args.command == "validate":
        result = validate_dataset(args.dataset_id)
    elif args.command == "import":
        result = import_dataset(args.dataset_id)
    elif args.command == "clean":
        result = clean_dataset(args.dataset_id, minimum_characters=args.minimum_characters)
    elif args.command == "dedupe":
        result = deduplicate_datasets(args.dataset_ids, args.dedupe_id)
    elif args.command == "stats":
        result = (
            get_candidate_statistics(args.dedupe_id)
            if args.dedupe_id
            else get_release_statistics(args.version)
        )
    else:
        result = create_release(args.dedupe_id, args.version)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
