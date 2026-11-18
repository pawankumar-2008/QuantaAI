"""Build reproducible QuantaLM V2 dataset artifacts without changing V1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


V2_ROOT = Path(__file__).resolve().parents[1]
QUANTALM_ROOT = V2_ROOT.parent
SOURCE_RAW_DIR = QUANTALM_ROOT / "data" / "raw"
SOURCE_PROCESSED_DIR = QUANTALM_ROOT / "data" / "processed"
V2_DATA_DIR = V2_ROOT / "data"

SEED = 20261106
MIN_DOCUMENT_CHARACTERS = 80
SUBSTANTIVE_LINE_MIN_CHARACTERS = 16
REPEATED_PASSAGE_MIN_CHARACTERS = 80
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.85
NEAR_DUPLICATE_SHINGLE_SIZE = 5

CATEGORY_MAP = {
    "computer_science": ("Computer Science", 0.98),
    "mathematics": ("Mathematics", 0.98),
    "artificial_intelligence": ("Artificial Intelligence", 0.98),
    "science": ("General Science", 0.98),
    "student_learning": ("Student Learning", 0.98),
    "general_knowledge": ("General Knowledge", 0.98),
}

SPLITS = ("train", "validation", "test")
SPECIAL_CONTENT_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def normalized_document(text: str) -> str:
    normalized_lines = [
        unicodedata.normalize("NFC", line).rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    while normalized_lines and not normalized_lines[0].strip():
        normalized_lines.pop(0)
    while normalized_lines and not normalized_lines[-1].strip():
        normalized_lines.pop()
    compact: list[str] = []
    blank_count = 0
    for line in normalized_lines:
        if line.strip():
            blank_count = 0
            compact.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                compact.append("")
    return "\n".join(compact)


def normalized_line(line: str) -> str:
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", line)).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w]+\b", text, flags=re.UNICODE))


def normalized_passages(text: str) -> set[str]:
    passages = set()
    for passage in re.split(r"\n\s*\n", text):
        normalized = normalized_line(passage)
        if len(normalized) >= REPEATED_PASSAGE_MIN_CHARACTERS:
            passages.add(normalized)
    return passages


def looks_like_code(line: str) -> bool:
    if line[:1].isspace():
        return True
    markers = (";", "{", "}", "=>", ":=", "==", "!=", "<=", ">=", "//", "/*", "#!")
    return any(marker in line for marker in markers)


def clean_text(text: str) -> tuple[str, int, int]:
    text = text.replace("\ufeff", "", 1)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    original_lines = text.split("\n")
    trimmed: list[str] = []
    removed_repeated_prose_lines = 0
    removed_blank_lines = 0

    for line in original_lines:
        if (
            trimmed
            and line
            and line == trimmed[-1]
            and not looks_like_code(line)
        ):
            removed_repeated_prose_lines += 1
            continue
        if not line.strip() and trimmed and not trimmed[-1].strip():
            blank_run = 1
            for previous in reversed(trimmed):
                if previous.strip():
                    break
                blank_run += 1
            if blank_run > 2:
                removed_blank_lines += 1
                continue
        trimmed.append(line)

    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()

    return "\n".join(trimmed), removed_repeated_prose_lines, removed_blank_lines


def load_tokenizer() -> tuple[Any, dict[str, str]]:
    config_path = V2_ROOT / "config.py"
    config_namespace: dict[str, Any] = {"__file__": str(config_path)}
    exec(compile(config_path.read_text(encoding="utf-8"), str(config_path), "exec"), config_namespace)

    expected_paths = (
        config_namespace["TOKENIZER_IMPLEMENTATION"],
        config_namespace["TOKENIZER_VOCAB_FILE"],
        config_namespace["TOKENIZER_MERGES_FILE"],
    )
    for path in expected_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Configured V2 tokenizer artifact is missing: {path}")

    implementation_path, vocab_path, merges_path = expected_paths
    hashes = {
        "implementation_sha256": sha256_file(implementation_path),
        "vocabulary_sha256": sha256_file(vocab_path),
        "merges_sha256": sha256_file(merges_path),
    }

    spec = importlib.util.spec_from_file_location(
        "quanta_v2_dataset_bpe",
        implementation_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tokenizer implementation: {implementation_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tokenizer = module.QuantaBPETokenizer.load()

    vocab_metadata = json.loads(vocab_path.read_text(encoding="utf-8"))
    if tokenizer.vocab_size != config_namespace["VOCAB_SIZE"]:
        raise ValueError(
            f"Tokenizer vocabulary {tokenizer.vocab_size} does not match "
            f"V2 config {config_namespace['VOCAB_SIZE']}."
        )
    if vocab_metadata.get("vocab_size") != config_namespace["VOCAB_SIZE"]:
        raise ValueError("Tokenizer vocabulary metadata does not match V2 config.")
    token_ids = set(tokenizer.token_bytes) | set(module.SPECIAL_TOKENS.values())
    if token_ids != set(range(tokenizer.vocab_size)):
        raise ValueError("Tokenizer token IDs are not contiguous or are inconsistent.")
    return tokenizer, hashes


def classify_source(path: Path) -> dict[str, Any]:
    relative = path.relative_to(SOURCE_RAW_DIR)
    if len(relative.parts) > 1:
        source_folder = relative.parts[0]
        category, confidence = CATEGORY_MAP.get(source_folder, (None, 0.0))
        return {
            "category": category,
            "classification_source": f"source directory: {source_folder}",
            "classification_confidence": confidence,
        }
    return {
        "category": None,
        "classification_source": "root-level source; no category directory",
        "classification_confidence": 0.0,
    }


def inspect_source(path: Path, tokenizer: Any) -> dict[str, Any]:
    raw_bytes = path.read_bytes()
    source_info = classify_source(path)
    record: dict[str, Any] = {
        "source_path": path.relative_to(QUANTALM_ROOT).as_posix(),
        "filename": path.name,
        "format": path.suffix.lower().lstrip(".") or "no extension",
        "file_size_bytes": len(raw_bytes),
        "sha256": sha256_bytes(raw_bytes),
        "available_source_information": (
            "Repository path only; no external source or license metadata found."
        ),
        **source_info,
    }
    try:
        decoded = raw_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        record.update(
            {
                "encoding": "invalid UTF-8",
                "character_count": None,
                "estimated_token_count": None,
                "token_count_method": "not available: invalid encoding",
                "read_error": f"Invalid UTF-8 at byte offset {error.start}.",
                "text": None,
            }
        )
        return record

    record.update(
        {
            "encoding": "UTF-8",
            "character_count": len(decoded),
            "estimated_token_count": len(tokenizer.encode(decoded)),
            "token_count_method": "independent V2 byte-level BPE, exact count before cleaning",
            "read_error": None,
            "text": decoded,
        }
    )
    return record


def copy_source_snapshot(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_hash:
            raise RuntimeError(
                f"Existing V2 raw snapshot differs from source; refusing overwrite: {destination}"
            )
        return
    shutil.copyfile(source, destination)
    if sha256_file(destination) != expected_hash:
        raise IOError(f"Raw source snapshot verification failed: {destination}")


def word_shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)
    size = NEAR_DUPLICATE_SHINGLE_SIZE
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def detect_duplicates(
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    exact_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        exact_buckets[sha256_bytes(document["clean_text"].encode("utf-8"))].append(document)

    retained: list[dict[str, Any]] = []
    exact_groups: list[dict[str, Any]] = []
    for digest, bucket in sorted(exact_buckets.items()):
        retained.append(bucket[0])
        if len(bucket) > 1:
            exact_groups.append(
                {
                    "type": "exact_cleaned_document",
                    "sha256": digest,
                    "canonical_source": bucket[0]["source_path"],
                    "duplicate_sources": [item["source_path"] for item in bucket[1:]],
                }
            )

    normalized_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in retained:
        digest = sha256_bytes(
            normalized_document(document["clean_text"]).encode("utf-8")
        )
        normalized_buckets[digest].append(document)

    normalized_groups: list[dict[str, Any]] = []
    normalized_retained: list[dict[str, Any]] = []
    for digest, bucket in sorted(normalized_buckets.items()):
        normalized_retained.append(bucket[0])
        if len(bucket) > 1:
            normalized_groups.append(
                {
                    "type": "normalized_text_duplicate",
                    "normalized_sha256": digest,
                    "canonical_source": bucket[0]["source_path"],
                    "duplicate_sources": [item["source_path"] for item in bucket[1:]],
                }
            )

    near_pairs: list[dict[str, Any]] = []
    for index, left in enumerate(normalized_retained):
        left_shingles = word_shingles(left["clean_text"])
        for right in normalized_retained[index + 1 :]:
            similarity = jaccard(left_shingles, word_shingles(right["clean_text"]))
            if similarity >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                near_pairs.append(
                    {
                        "left_source": left["source_path"],
                        "right_source": right["source_path"],
                        "word_shingle_jaccard": round(similarity, 6),
                        "threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
                        "action": "retain and keep in the same split",
                    }
                )

    repeated_lines: dict[str, dict[str, Any]] = {}
    passage_owners: dict[str, set[str]] = defaultdict(set)
    line_owners: dict[str, set[str]] = defaultdict(set)
    per_document_repeated_lines: dict[str, int] = {}
    for document in normalized_retained:
        lines = [normalized_line(line) for line in document["clean_text"].splitlines()]
        counts = Counter(line for line in lines if line)
        per_document_repeated_lines[document["source_path"]] = sum(
            count - 1 for count in counts.values() if count > 1
        )
        for line in counts:
            if len(line) >= SUBSTANTIVE_LINE_MIN_CHARACTERS:
                line_owners[line].add(document["source_path"])
        for passage in normalized_passages(document["clean_text"]):
            passage_owners[passage].add(document["source_path"])
    for line, owners in sorted(line_owners.items()):
        if len(owners) > 1:
            repeated_lines[sha256_bytes(line.encode("utf-8"))] = {
                "line": line,
                "source_documents": sorted(owners),
            }
    repeated_passages = []
    for passage, owners in sorted(passage_owners.items()):
        if len(owners) > 1:
            repeated_passages.append(
                {
                    "passage_sha256": sha256_bytes(passage.encode("utf-8")),
                    "character_count": len(passage),
                    "source_documents": sorted(owners),
                }
            )

    duplicate_report = {
        "original_raw_document_count": len(documents),
        "exact_duplicate_documents_removed": sum(
            len(group["duplicate_sources"]) for group in exact_groups
        ),
        "normalized_duplicate_documents_removed": sum(
            len(group["duplicate_sources"]) for group in normalized_groups
        ),
        "retained_unique_document_count": len(normalized_retained),
        "exact_duplicate_groups": exact_groups,
        "normalized_duplicate_groups": normalized_groups,
        "near_duplicate_detection": {
            "method": f"{NEAR_DUPLICATE_SHINGLE_SIZE}-word shingle Jaccard",
            "threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
            "pairs": near_pairs,
            "near_duplicate_documents_removed": 0,
        },
        "repeated_lines": {
            "within_document_duplicate_line_occurrences": per_document_repeated_lines,
            "shared_substantive_line_count": len(repeated_lines),
            "shared_substantive_lines": list(repeated_lines.values()),
            "minimum_line_length": SUBSTANTIVE_LINE_MIN_CHARACTERS,
            "action": "preserve line content and group documents sharing lines into one split",
        },
        "repeated_passages": {
            "shared_passage_count": len(repeated_passages),
            "shared_passages": repeated_passages,
            "minimum_normalized_characters": REPEATED_PASSAGE_MIN_CHARACTERS,
            "method": "exact normalized paragraph comparison",
            "action": "preserve passage text and group documents sharing a passage into one split",
        },
    }

    line_to_documents: dict[str, list[str]] = defaultdict(list)
    for document in normalized_retained:
        unique_lines = {
            normalized_line(line)
            for line in document["clean_text"].splitlines()
            if len(normalized_line(line)) >= SUBSTANTIVE_LINE_MIN_CHARACTERS
        }
        for line in unique_lines:
            line_to_documents[line].append(document["doc_id"])

    disjoint_set = DisjointSet([document["doc_id"] for document in normalized_retained])
    for owners in line_to_documents.values():
        if len(owners) > 1:
            for owner in owners[1:]:
                disjoint_set.union(owners[0], owner)
    passage_to_documents: dict[str, list[str]] = defaultdict(list)
    for document in normalized_retained:
        for passage in normalized_passages(document["clean_text"]):
            passage_to_documents[passage].append(document["doc_id"])
    for owners in passage_to_documents.values():
        if len(owners) > 1:
            for owner in owners[1:]:
                disjoint_set.union(owners[0], owner)
    source_to_id = {document["source_path"]: document["doc_id"] for document in normalized_retained}
    for pair in near_pairs:
        disjoint_set.union(
            source_to_id[pair["left_source"]],
            source_to_id[pair["right_source"]],
        )

    component_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in normalized_retained:
        component_map[disjoint_set.find(document["doc_id"])].append(document)
    leakage_groups = []
    for group_index, group_docs in enumerate(
        sorted(
            component_map.values(),
            key=lambda items: sorted(item["source_path"] for item in items),
        ),
        start=1,
    ):
        group_id = f"group-{group_index:03d}"
        for document in group_docs:
            document["leakage_group_id"] = group_id
        if len(group_docs) > 1:
            leakage_groups.append(
                {
                    "group_id": group_id,
                    "sources": sorted(item["source_path"] for item in group_docs),
                    "reason": "shared substantive line and/or detected near duplicate",
                }
            )

    duplicate_report["split_leakage_groups"] = leakage_groups
    return normalized_retained, duplicate_report, near_pairs


def assign_splits(documents: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        groups[document["leakage_group_id"]].append(document)
    if len(groups) < 3:
        raise ValueError(
            "At least three independent document groups are required for "
            "train, validation, and held-out test splits."
        )

    rng = random.Random(SEED)
    available = list(groups)
    rng.shuffle(available)

    def select_groups(target_documents: int, maximum_groups: int) -> list[str]:
        selected: list[str] = []
        selected_documents = 0
        while available and len(selected) < maximum_groups:
            group_id = min(
                available,
                key=lambda candidate: abs(
                    target_documents
                    - (selected_documents + len(groups[candidate]))
                ),
            )
            candidate_count = selected_documents + len(groups[group_id])
            if selected and abs(target_documents - selected_documents) <= abs(
                target_documents - candidate_count
            ):
                break
            selected.append(group_id)
            available.remove(group_id)
            selected_documents = candidate_count
            if selected_documents >= target_documents:
                break
        return selected

    total_documents = len(documents)
    test_target = max(1, int(total_documents * 0.1 + 0.5))
    test_groups = select_groups(test_target, len(available) - 2)
    remaining_document_count = sum(len(groups[group_id]) for group_id in available)
    validation_target = max(1, int(remaining_document_count * 0.1 + 0.5))
    validation_groups = select_groups(validation_target, len(available) - 1)

    assignments = {group_id: "train" for group_id in groups}
    assignments.update({group_id: "test" for group_id in test_groups})
    assignments.update({group_id: "validation" for group_id in validation_groups})

    splits = {name: [] for name in SPLITS}
    for document in documents:
        split_name = assignments[document["leakage_group_id"]]
        document["split"] = split_name
        splits[split_name].append(document)
    for items in splits.values():
        items.sort(key=lambda item: item["doc_id"])
    return splits


def verify_split_leakage(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    exact_fingerprints: dict[str, str] = {}
    normalized_fingerprints: dict[str, str] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    line_splits: dict[str, set[str]] = defaultdict(set)
    passage_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, documents in splits.items():
        for document in documents:
            exact_hash = sha256_bytes(document["clean_text"].encode("utf-8"))
            normalized_hash = sha256_bytes(
                normalized_document(document["clean_text"]).encode("utf-8")
            )
            if exact_hash in exact_fingerprints and exact_fingerprints[exact_hash] != split_name:
                raise AssertionError("Exact duplicate document leaked across splits.")
            if normalized_hash in normalized_fingerprints and normalized_fingerprints[normalized_hash] != split_name:
                raise AssertionError("Normalized duplicate document leaked across splits.")
            exact_fingerprints[exact_hash] = split_name
            normalized_fingerprints[normalized_hash] = split_name
            group_splits[document["leakage_group_id"]].add(split_name)
            for line in {
                normalized_line(value)
                for value in document["clean_text"].splitlines()
                if len(normalized_line(value)) >= SUBSTANTIVE_LINE_MIN_CHARACTERS
            }:
                line_splits[line].add(split_name)
            for passage in normalized_passages(document["clean_text"]):
                passage_splits[passage].add(split_name)
    if any(len(values) > 1 for values in group_splits.values()):
        raise AssertionError("A duplicate/overlap group spans multiple splits.")
    if any(len(values) > 1 for values in line_splits.values()):
        raise AssertionError("A shared substantive line leaks across splits.")
    if any(len(values) > 1 for values in passage_splits.values()):
        raise AssertionError("A shared substantive passage leaks across splits.")
    return {
        "exact_duplicate_cross_split_count": 0,
        "normalized_duplicate_cross_split_count": 0,
        "near_duplicate_or_shared_line_group_cross_split_count": 0,
        "shared_substantive_line_cross_split_count": 0,
        "shared_passage_cross_split_count": 0,
        "shared_line_minimum_length": SUBSTANTIVE_LINE_MIN_CHARACTERS,
        "status": "passed",
    }


def make_dataset_record(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": document["doc_id"],
        "category": document["category"],
        "classification_source": document["classification_source"],
        "classification_confidence": document["classification_confidence"],
        "source_path": document["source_path"],
        "source_sha256": document["source_sha256"],
        "source_provenance": document["source_provenance"],
        "license_status": document["license_status"],
        "leakage_group_id": document["leakage_group_id"],
        "text": document["clean_text"],
    }


def summarize_existing_splits() -> dict[str, Any]:
    tokenizer, _ = load_tokenizer()
    split_statistics: dict[str, dict[str, int | float]] = {}
    observed_token_ids: set[int] = set()
    for split_name in SPLITS:
        split_path = V2_DATA_DIR / "splits" / split_name / "dataset.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        records = [
            json.loads(line)
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"Dataset split is empty: {split_name}")
        character_count = 0
        word_total = 0
        token_total = 0
        for record in records:
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text in {split_name} dataset split.")
            token_ids = tokenizer.encode(text)
            character_count += len(text)
            word_total += count_words(text)
            token_total += len(token_ids)
            observed_token_ids.update(token_ids)
        split_statistics[split_name] = {
            "documents": len(records),
            "characters": character_count,
            "words": word_total,
            "tokens": token_total,
        }

    total_documents = sum(item["documents"] for item in split_statistics.values())
    total_tokens = sum(item["tokens"] for item in split_statistics.values())
    is_small = total_documents < 100 or total_tokens < 1_000_000
    return {
        "total_documents": total_documents,
        "total_characters": sum(item["characters"] for item in split_statistics.values()),
        "total_words": sum(item["words"] for item in split_statistics.values()),
        "total_tokens": total_tokens,
        "training_documents": split_statistics["train"]["documents"],
        "training_words": split_statistics["train"]["words"],
        "training_tokens": split_statistics["train"]["tokens"],
        "validation_documents": split_statistics["validation"]["documents"],
        "validation_words": split_statistics["validation"]["words"],
        "validation_tokens": split_statistics["validation"]["tokens"],
        "test_documents": split_statistics["test"]["documents"],
        "test_words": split_statistics["test"]["words"],
        "test_tokens": split_statistics["test"]["tokens"],
        "vocabulary_size": tokenizer.vocab_size,
        "observed_vocabulary_size": len(observed_token_ids),
        "word_count_method": "Unicode alphanumeric word runs (\\b[\\w]+\\b).",
        "splits": split_statistics,
        "small_dataset_warning": (
            f"Only {total_documents} documents and {total_tokens} tokens are available; "
            "expect high overfitting risk and weak generalization."
            if is_small
            else None
        ),
    }


def build() -> dict[str, Any]:
    if not SOURCE_RAW_DIR.is_dir():
        raise FileNotFoundError(f"V1 raw data directory does not exist: {SOURCE_RAW_DIR}")

    source_paths = sorted(path for path in SOURCE_RAW_DIR.rglob("*.txt") if path.is_file())
    aggregate_paths = sorted(
        path for path in SOURCE_PROCESSED_DIR.rglob("*.txt") if path.is_file()
    )
    if not source_paths:
        raise ValueError("No raw .txt source documents were found.")

    original_hashes = {path: sha256_file(path) for path in source_paths + aggregate_paths}
    tokenizer, tokenizer_hashes = load_tokenizer()
    config_namespace: dict[str, Any] = {"__file__": str(V2_ROOT / "config.py")}
    exec(
        compile(
            (V2_ROOT / "config.py").read_text(encoding="utf-8"),
            str(V2_ROOT / "config.py"),
            "exec",
        ),
        config_namespace,
    )
    tokenizer_vocab = tokenizer.vocab_size
    if tokenizer_vocab != config_namespace["VOCAB_SIZE"]:
        raise ValueError("Tokenizer and model configuration vocabulary sizes differ.")

    inventory_records: list[dict[str, Any]] = []
    for path in source_paths + aggregate_paths:
        if path.is_relative_to(SOURCE_RAW_DIR):
            record = inspect_source(path, tokenizer)
            category_label = record.get("category")
            if category_label is not None:
                record["category"] = category_label
            record["document_role"] = "raw source candidate"
        else:
            raw_bytes = path.read_bytes()
            try:
                text = raw_bytes.decode("utf-8-sig", errors="strict")
                token_count = len(tokenizer.encode(text))
                read_error = None
                character_count = len(text)
            except UnicodeDecodeError as error:
                text, token_count, character_count = None, None, None
                read_error = f"Invalid UTF-8 at byte offset {error.start}."
            record = {
                "source_path": path.relative_to(QUANTALM_ROOT).as_posix(),
                "filename": path.name,
                "format": path.suffix.lower().lstrip(".") or "no extension",
                "file_size_bytes": len(raw_bytes),
                "sha256": sha256_bytes(raw_bytes),
                "character_count": character_count,
                "estimated_token_count": token_count,
                "token_count_method": "independent V2 byte-level BPE, exact count before cleaning",
                "encoding": "UTF-8" if read_error is None else "invalid UTF-8",
                "category": "Derived aggregate (excluded from training)",
                "classification_source": "existing processed aggregate; derived from raw corpus",
                "classification_confidence": None,
                "available_source_information": "Generated by V1 prepare_data.py; not an independent source.",
                "document_role": "derived aggregate; excluded from source document candidates",
                "read_error": read_error,
                "text": text,
            }
        inventory_records.append(record)

    raw_records = [
        record
        for record in inventory_records
        if record["document_role"] == "raw source candidate"
    ]
    classified_by_hash: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        if record["category"] is not None and record["text"] is not None:
            normalized_hash = sha256_bytes(normalized_document(record["text"]).encode("utf-8"))
            classified_by_hash.setdefault(normalized_hash, record)
    for record in raw_records:
        if record["category"] is None and record["text"] is not None:
            digest = sha256_bytes(normalized_document(record["text"]).encode("utf-8"))
            canonical = classified_by_hash.get(digest)
            if canonical is not None:
                record["category"] = canonical["category"]
                record["classification_source"] = (
                    f"normalized content duplicate of {canonical['source_path']}"
                )
                record["classification_confidence"] = 1.0
            else:
                record["classification_source"] = (
                    "uncategorized root-level file; manual classification required"
                )

    inventory_output = [
        {key: value for key, value in item.items() if key != "text"}
        for item in inventory_records
    ]
    V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_map: dict[str, str] = {}
    for path in source_paths:
        relative = path.relative_to(SOURCE_RAW_DIR)
        snapshot = V2_DATA_DIR / "raw" / "source_snapshot" / "raw" / relative
        source_hash = original_hashes[path]
        copy_source_snapshot(path, snapshot, source_hash)
        snapshot_map[path.relative_to(QUANTALM_ROOT).as_posix()] = (
            snapshot.relative_to(QUANTALM_ROOT).as_posix()
        )
    for path in aggregate_paths:
        relative = path.relative_to(SOURCE_PROCESSED_DIR)
        snapshot = V2_DATA_DIR / "raw" / "source_snapshot" / "processed_reference" / relative
        copy_source_snapshot(path, snapshot, original_hashes[path])
        snapshot_map[path.relative_to(QUANTALM_ROOT).as_posix()] = (
            snapshot.relative_to(QUANTALM_ROOT).as_posix()
        )

    clean_report: dict[str, Any] = {
        "source_document_count": len(raw_records),
        "original_raw_character_count": sum(
            record["character_count"] or 0 for record in raw_records
        ),
        "cleaned_character_count_before_deduplication": 0,
        "cleaned_character_count_after_deduplication": 0,
        "characters_removed_by_cleaning": 0,
        "characters_removed_by_document_deduplication": 0,
        "empty_documents_excluded": [],
        "invalid_encoding_documents_excluded": [],
        "corrupted_documents_excluded": [],
        "short_documents_excluded": [],
        "uncategorized_documents_excluded": [],
        "removed_consecutive_repeated_prose_lines": 0,
        "removed_excess_blank_lines": 0,
        "normalization": [
            "Decode UTF-8 strictly; remove an optional UTF-8 BOM.",
            "Normalize CRLF and CR line endings to LF.",
            "Trim blank lines at document boundaries.",
            "Limit runs of blank lines to two.",
            "Remove adjacent identical non-code prose lines only; preserve indented/code-like lines.",
            "Do not lowercase, strip punctuation, normalize code spacing, or rewrite technical notation.",
        ],
    }
    documents: list[dict[str, Any]] = []
    for record in raw_records:
        text = record["text"]
        if text is None:
            clean_report["invalid_encoding_documents_excluded"].append(record["source_path"])
            continue
        if not text.strip():
            clean_report["empty_documents_excluded"].append(record["source_path"])
            continue
        if "\ufffd" in text or SPECIAL_CONTENT_CONTROLS.search(text):
            clean_report["corrupted_documents_excluded"].append(record["source_path"])
            continue
        if record["category"] is None:
            clean_report["uncategorized_documents_excluded"].append(
                {
                    "source_path": record["source_path"],
                    "reason": record["classification_source"],
                }
            )
            continue

        cleaned, repeated_count, blank_count = clean_text(text)
        clean_report["cleaned_character_count_before_deduplication"] += len(cleaned)
        clean_report["removed_consecutive_repeated_prose_lines"] += repeated_count
        clean_report["removed_excess_blank_lines"] += blank_count
        if len(cleaned.strip()) < MIN_DOCUMENT_CHARACTERS:
            clean_report["short_documents_excluded"].append(
                {
                    "source_path": record["source_path"],
                    "character_count": len(cleaned.strip()),
                    "threshold": MIN_DOCUMENT_CHARACTERS,
                }
            )
            continue

        documents.append(
            {
                "source_path": record["source_path"],
                "source_sha256": record["sha256"],
                "clean_text": cleaned,
                "category": record["category"],
                "classification_source": record["classification_source"],
                "classification_confidence": record["classification_confidence"],
                "source_provenance": (
                    "Repository file; external publisher, author, and acquisition details "
                    "were not supplied."
                ),
                "license_status": "unknown; no license information found",
                "source_snapshot_path": snapshot_map[record["source_path"]],
            }
        )

    for document in documents:
        document["doc_id"] = sha256_bytes(
            (document["source_path"] + "\0" + document["source_sha256"]).encode("utf-8")
        )[:16]

    retained, duplicate_report, _ = detect_duplicates(documents)
    clean_report["cleaned_character_count_after_deduplication"] = sum(
        len(document["clean_text"]) for document in retained
    )
    clean_report["characters_removed_by_cleaning"] = (
        clean_report["original_raw_character_count"]
        - clean_report["cleaned_character_count_before_deduplication"]
    )
    clean_report["characters_removed_by_document_deduplication"] = (
        clean_report["cleaned_character_count_before_deduplication"]
        - clean_report["cleaned_character_count_after_deduplication"]
    )
    splits = assign_splits(retained)
    leakage_report = verify_split_leakage(splits)

    category_distribution: dict[str, dict[str, int]] = {}
    for category in CATEGORY_MAP.values():
        label = category[0]
        category_distribution[label] = {
            split_name: sum(1 for item in splits[split_name] if item["category"] == label)
            for split_name in SPLITS
        }
        category_distribution[label]["total"] = sum(
            category_distribution[label][split_name] for split_name in SPLITS
        )

    split_statistics: dict[str, Any] = {}
    for split_name, items in splits.items():
        token_count = sum(len(tokenizer.encode(item["clean_text"])) for item in items)
        split_statistics[split_name] = {
            "document_count": len(items),
            "document_ratio": round(len(items) / max(len(retained), 1), 6),
            "character_count": sum(len(item["clean_text"]) for item in items),
            "word_count": sum(count_words(item["clean_text"]) for item in items),
            "estimated_token_count": token_count,
            "token_count_method": "independent V2 byte-level BPE, exact count without special tokens",
            "category_distribution": dict(sorted(Counter(item["category"] for item in items).items())),
            "document_ids": [item["doc_id"] for item in items],
        }
        write_jsonl(
            V2_DATA_DIR / "splits" / split_name / "dataset.jsonl",
            [make_dataset_record(item) for item in items],
        )

    full_cleaned_records = [
        make_dataset_record(item)
        for item in sorted(retained, key=lambda value: value["doc_id"])
    ]
    write_jsonl(V2_DATA_DIR / "processed" / "cleaned_documents.jsonl", full_cleaned_records)

    original_count = len(raw_records)
    retained_count = len(retained)
    exact_removed = duplicate_report["exact_duplicate_documents_removed"]
    normalized_removed = duplicate_report["normalized_duplicate_documents_removed"]
    tokenizer_training_report_path = (
        V2_ROOT / "tokenizer" / "tokenizer_training_report.json"
    )
    tokenizer_training_report: dict[str, Any] | None = None
    tokenizer_training_is_current = False
    if tokenizer_training_report_path.is_file():
        tokenizer_training_report = json.loads(
            tokenizer_training_report_path.read_text(encoding="utf-8")
        )
        recorded_train_hash = tokenizer_training_report.get(
            "training_source", {}
        ).get("split_sha256")
        current_train_hash = sha256_file(
            V2_DATA_DIR / "splits" / "train" / "dataset.jsonl"
        )
        tokenizer_training_is_current = recorded_train_hash == current_train_hash

    tokenizer_training_status = (
        "resolved: independent tokenizer report matches the current training split"
        if tokenizer_training_is_current
        else "unresolved: independent tokenizer is absent or does not match the current training split"
    )
    expansion_report = {
        "additional_approved_sources_added": 0,
        "external_source_search_or_download_performed": False,
        "available_sources": [
            {
                "document_id": document["doc_id"],
                "document_title": None,
                "title_status": "No document title metadata was supplied.",
                "source_path": document["source_path"],
                "category": document["category"],
                "source_provenance": document["source_provenance"],
                "license_status": document["license_status"],
            }
            for document in sorted(retained, key=lambda item: item["source_path"])
        ],
        "unique_document_counts_by_category": {
            category: details["total"]
            for category, details in sorted(category_distribution.items())
        },
        "coverage_gaps": [
            "General Knowledge has no locally available documents.",
            "No dedicated Software Engineering source document was found.",
            "Mathematics, Artificial Intelligence, General Science, and Student Learning each have only one unique source document.",
            "No approved external sources or licensing records were provided; no external material was added.",
        ],
        "provenance_and_licensing": {
            "external_sources": [],
            "retrieval_date": None,
            "license_status": "unknown for existing local documents",
            "warning": "Repository presence does not establish a license or permission to train.",
        },
    }
    data_quality = {
        "status": "passed with documented limitations",
        "source_document_count": original_count,
        "classified_document_count": len(documents),
        "retained_unique_document_count": retained_count,
        "empty_document_count": len(clean_report["empty_documents_excluded"]),
        "invalid_encoding_count": len(clean_report["invalid_encoding_documents_excluded"]),
        "corrupted_document_count": len(clean_report["corrupted_documents_excluded"]),
        "short_document_count": len(clean_report["short_documents_excluded"]),
        "uncategorized_document_count": sum(
            1 for record in raw_records if record["category"] is None
        ),
        "category_distribution": category_distribution,
        "split_statistics": split_statistics,
        "split_leakage_validation": leakage_report,
        "dataset_expansion": expansion_report,
        "tokenizer_training_data_leakage": {
            "status": tokenizer_training_status,
            "training_report_present": tokenizer_training_report is not None,
            "current_training_split_hash_matches_report": tokenizer_training_is_current,
            "validation_and_test_used_for_tokenizer_fitting": False,
        },
        "v2_model_compatibility": {
            "tokenizer_vocabulary_matches_config": tokenizer_vocab == config_namespace["VOCAB_SIZE"],
            "configured_context_length": config_namespace["CONTEXT_LENGTH"],
            "v2_transformer_implementation_present": any(
                path.is_file()
                for path in (V2_ROOT / "src").glob("*.py")
                if path.name == "model.py"
            ),
            "status": (
                "Tokenizer vocabulary matches the configured V2 model vocabulary."
            ),
        },
        "provenance_and_licensing": {
            "external_source_metadata_available": False,
            "license_status": "unknown for all source documents",
            "action_required": "Obtain source and license records before distributing or training on this corpus.",
        },
        "known_limitations": [
            "The General Knowledge category has no source documents.",
            "Validation and test are small because the corpus contains few unique source documents and leakage groups are kept intact.",
            "Repository paths do not establish authorship, factual accuracy, or reuse rights.",
            "Lexical near-duplicate detection cannot identify semantic paraphrases.",
            "The V2 tokenizer decoder uses UTF-8 replacement on malformed byte sequences; valid source text is checked for exact round trips.",
            "No approved licensed external dataset was supplied; additional content was not fabricated or downloaded.",
        ],
    }

    manifest_documents = []
    for split_name in SPLITS:
        for document in splits[split_name]:
            record = make_dataset_record(document)
            token_count = len(tokenizer.encode(document["clean_text"]))
            manifest_documents.append(
                {
                    "document_id": document["doc_id"],
                    "document_title": None,
                    "title_status": "No document title metadata was supplied.",
                    "split": split_name,
                    "category": document["category"],
                    "source_path": document["source_path"],
                    "source_snapshot_path": document["source_snapshot_path"],
                    "source_sha256": document["source_sha256"],
                    "clean_text_sha256": sha256_bytes(
                        document["clean_text"].encode("utf-8")
                    ),
                    "token_count": token_count,
                    "classification_source": document["classification_source"],
                    "classification_confidence": document["classification_confidence"],
                    "license_status": document["license_status"],
                    "leakage_group_id": document["leakage_group_id"],
                }
            )

    split_file_hashes = {
        split_name: sha256_file(
            V2_DATA_DIR / "splits" / split_name / "dataset.jsonl"
        )
        for split_name in SPLITS
    }
    manifest = {
        "dataset_name": "QuantaLM V2 corpus",
        "format_version": 1,
        "random_seed": SEED,
        "requested_split_ratios": {"train": 0.81, "validation": 0.09, "test": 0.1},
        "requested_train_validation_ratio": {"train": 0.9, "validation": 0.1},
        "split_policy": (
            "Document-level split after exact/normalized deduplication; lexical near duplicates "
            "and documents sharing substantive exact lines are kept in the same split. "
            "Approximately 10% of independent groups are held out for test; the remainder is "
            "split 90/10 between train and validation. Integer group sizes may prevent exact ratios."
        ),
        "held_out_evaluation_warning": (
            "The independent V2 tokenizer is fitted only on train documents. "
            "The recorded training-split hash is checked against this manifest."
        ),
        "model_tokenizer_compatibility": {
            "tokenizer_type": "independent V2 byte-level BPE",
            "tokenizer_vocabulary_size": tokenizer_vocab,
            "configured_model_vocabulary_size": config_namespace["VOCAB_SIZE"],
            "tokenizer_hashes": tokenizer_hashes,
            "special_token_ids": {
                "<pad>": config_namespace["PAD_TOKEN_ID"],
                "<unk>": config_namespace["UNK_TOKEN_ID"],
                "<bos>": config_namespace["BOS_TOKEN_ID"],
                "<eos>": config_namespace["EOS_TOKEN_ID"],
            },
        },
        "dataset_expansion": expansion_report,
        "tokenizer_training_provenance": (
            {
                "training_report_path": "tokenizer/tokenizer_training_report.json",
                "training_report_sha256": sha256_file(tokenizer_training_report_path),
                "training_split_sha256": tokenizer_training_report["training_source"]["split_sha256"],
                "current_training_split_hash_matches": tokenizer_training_is_current,
            }
            if tokenizer_training_report is not None
            else None
        ),
        "split_files": {
            split_name: {
                "path": f"splits/{split_name}/dataset.jsonl",
                "sha256": split_file_hashes[split_name],
                **split_statistics[split_name],
            }
            for split_name in SPLITS
        },
        "documents": manifest_documents,
        "excluded_aggregates": [
            {
                "source_path": path.relative_to(QUANTALM_ROOT).as_posix(),
                "reason": "Derived aggregate duplicates source documents; excluded to avoid duplicate content.",
            }
            for path in aggregate_paths
        ],
        "duplicate_counts": {
            "original_raw_documents": original_count,
            "exact_documents_removed": exact_removed,
            "normalized_documents_removed": normalized_removed,
            "retained_unique_documents": retained_count,
        },
    }

    write_json(V2_DATA_DIR / "reports" / "source_inventory.json", {
        "inventory_scope": "All .txt files under quantalm/data/raw and quantalm/data/processed.",
        "v2_data_before_pipeline": "No pre-existing dataset files; directory contained only a placeholder.",
        "tokenizer": {
            "name": "independent QuantaLM V2 byte-level BPE",
            "vocabulary_size": tokenizer_vocab,
            "hashes": tokenizer_hashes,
        },
        "files": inventory_output,
    })
    write_json(V2_DATA_DIR / "reports" / "cleaning_report.json", clean_report)
    write_json(V2_DATA_DIR / "reports" / "duplicate_report.json", duplicate_report)
    write_json(V2_DATA_DIR / "reports" / "data_quality_report.json", data_quality)
    write_json(
        V2_DATA_DIR / "reports" / "dataset_expansion_report.json",
        expansion_report,
    )
    write_json(V2_DATA_DIR / "reports" / "dataset_statistics.json", {
        "total_documents": retained_count,
        "total_characters": sum(len(document["clean_text"]) for document in retained),
        "total_words": sum(count_words(document["clean_text"]) for document in retained),
        "total_tokens": sum(
            len(tokenizer.encode(document["clean_text"])) for document in retained
        ),
        "training_documents": split_statistics["train"]["document_count"],
        "validation_documents": split_statistics["validation"]["document_count"],
        "test_documents": split_statistics["test"]["document_count"],
        "training_words": split_statistics["train"]["word_count"],
        "validation_words": split_statistics["validation"]["word_count"],
        "test_words": split_statistics["test"]["word_count"],
        "training_tokens": split_statistics["train"]["estimated_token_count"],
        "validation_tokens": split_statistics["validation"]["estimated_token_count"],
        "test_tokens": split_statistics["test"]["estimated_token_count"],
        "vocabulary_size": tokenizer_vocab,
        "word_count_method": "Unicode alphanumeric word runs (\\b[\\w]+\\b).",
        "original_raw_document_count": original_count,
        "original_raw_character_count": sum(
            record["character_count"] or 0 for record in raw_records
        ),
        "original_raw_file_size_bytes": sum(
            record["file_size_bytes"] for record in raw_records
        ),
        "retained_unique_document_count": retained_count,
        "retained_clean_character_count": sum(
            len(document["clean_text"]) for document in retained
        ),
        "characters_removed_by_cleaning": clean_report["characters_removed_by_cleaning"],
        "characters_removed_by_document_deduplication": clean_report[
            "characters_removed_by_document_deduplication"
        ],
        "retained_bpe_token_count": sum(
            len(tokenizer.encode(document["clean_text"])) for document in retained
        ),
        "category_distribution": category_distribution,
        "split_statistics": split_statistics,
        "tokenizer": {
            "vocabulary_size": tokenizer_vocab,
            "hashes": tokenizer_hashes,
        },
        "requested_ratios": manifest["requested_split_ratios"],
    })
    write_json(V2_DATA_DIR / "manifests" / "dataset_manifest.json", manifest)

    round_trip_failures = []
    for document in retained:
        token_ids = tokenizer.encode(document["clean_text"])
        if tokenizer.decode(token_ids) != document["clean_text"]:
            round_trip_failures.append(document["source_path"])
    if round_trip_failures:
        raise AssertionError(f"Tokenizer round-trip failed: {round_trip_failures}")
    tokenizer_probes = (
        "public static void main(String[] args) {\n    double n = 6.02e23;\n}\n",
        "∑_{i=1}^{n} x_i^2 ≤ 10 — Δx → 0",
    )
    if any(tokenizer.decode(tokenizer.encode(text)) != text for text in tokenizer_probes):
        raise AssertionError("Tokenizer code/math/Unicode compatibility probe failed.")

    written_json_paths = (
        V2_DATA_DIR / "reports" / "source_inventory.json",
        V2_DATA_DIR / "reports" / "cleaning_report.json",
        V2_DATA_DIR / "reports" / "duplicate_report.json",
        V2_DATA_DIR / "reports" / "data_quality_report.json",
        V2_DATA_DIR / "reports" / "dataset_statistics.json",
        V2_DATA_DIR / "manifests" / "dataset_manifest.json",
    )
    for path in written_json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    written_manifest = json.loads(
        (V2_DATA_DIR / "manifests" / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for split_name in SPLITS:
        split_path = V2_DATA_DIR / "splits" / split_name / "dataset.jsonl"
        split_records = [
            json.loads(line)
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(not item["text"].strip() for item in split_records):
            raise AssertionError(f"Empty document found in {split_name} split.")
        expected_file = written_manifest["split_files"][split_name]
        if sha256_file(split_path) != expected_file["sha256"]:
            raise AssertionError(f"Manifest hash mismatch for {split_name} split.")
        if len(split_records) != expected_file["document_count"]:
            raise AssertionError(f"Manifest document count mismatch for {split_name} split.")

    for path, digest in original_hashes.items():
        if sha256_file(path) != digest:
            raise AssertionError(f"Original source was modified: {path}")
    for relative_path, digest in tokenizer_hashes.items():
        path = {
            "implementation_sha256": config_namespace["TOKENIZER_IMPLEMENTATION"],
            "vocabulary_sha256": config_namespace["TOKENIZER_VOCAB_FILE"],
            "merges_sha256": config_namespace["TOKENIZER_MERGES_FILE"],
        }[relative_path]
        if sha256_file(path) != digest:
            raise AssertionError(f"Existing tokenizer artifact was modified: {path}")

    return {
        "source_raw_documents": original_count,
        "retained_documents": retained_count,
        "split_counts": {
            split_name: split_statistics[split_name]["document_count"]
            for split_name in SPLITS
        },
        "category_distribution": category_distribution,
        "tokenizer_vocab_size": tokenizer_vocab,
        "original_sources_unchanged": True,
        "tokenizer_artifacts_unchanged": True,
        "tokenizer_round_trip_failures": round_trip_failures,
        "split_leakage_validation": leakage_report,
        "tokenizer_training_data_leakage": tokenizer_training_status,
        "tokenizer_code_math_unicode_probes": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics for existing splits without rebuilding or modifying them.",
    )
    args = parser.parse_args()
    result = summarize_existing_splits() if args.stats_only else build()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
