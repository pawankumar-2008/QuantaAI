"""Create deterministic, document-isolated, context-sized V2 LM examples."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .workspace_config import v2_config
except ImportError:
    from workspace_config import v2_config


V2_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "validation", "test")
CHUNK_OUTPUT_ROOT = v2_config.DATA_DIR / "chunks"
MANIFEST_FILE = v2_config.DATA_DIR / "manifests" / "dataset_manifest.json"
MODEL_CONFIG_FILE = V2_ROOT / "model_config.json"
CONTEXT_LENGTH = v2_config.CONTEXT_LENGTH
CHUNK_OVERLAP = 32
CHUNK_STRIDE = 225
MIN_FINAL_CHUNK_TOKENS = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


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


@dataclass(frozen=True)
class ChunkConfig:
    context_length: int = CONTEXT_LENGTH
    chunk_overlap: int = CHUNK_OVERLAP
    chunk_stride: int = CHUNK_STRIDE
    min_final_chunk_tokens: int = MIN_FINAL_CHUNK_TOKENS
    vocab_size: int = v2_config.VOCAB_SIZE
    pad_token_id: int = v2_config.PAD_TOKEN_ID
    bos_token_id: int = v2_config.BOS_TOKEN_ID
    eos_token_id: int = v2_config.EOS_TOKEN_ID

    def __post_init__(self) -> None:
        for name in (
            "context_length",
            "chunk_overlap",
            "chunk_stride",
            "min_final_chunk_tokens",
            "vocab_size",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        window_size = self.context_length + 1
        if self.context_length < 1:
            raise ValueError("context_length must be positive.")
        if not 0 <= self.chunk_overlap < window_size:
            raise ValueError("chunk_overlap must be in [0, context_length].")
        if self.chunk_stride != window_size - self.chunk_overlap:
            raise ValueError(
                "chunk_stride must equal context_length + 1 - chunk_overlap."
            )
        if not 2 <= self.min_final_chunk_tokens <= window_size:
            raise ValueError(
                "min_final_chunk_tokens must be between 2 and context_length + 1."
            )
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        special_ids = (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in special_ids):
            raise ValueError("Special token IDs must be within the configured vocabulary.")
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("PAD, BOS, and EOS token IDs must be distinct.")


DEFAULT_CHUNK_CONFIG = ChunkConfig()


def _chunk_ranges(
    token_count: int,
    config: ChunkConfig,
) -> list[tuple[int, int]]:
    window_size = config.context_length + 1
    starts = [0]
    while starts[-1] + window_size < token_count:
        next_start = starts[-1] + config.chunk_stride
        if token_count - next_start < config.min_final_chunk_tokens:
            next_start = token_count - config.min_final_chunk_tokens
        if next_start <= starts[-1]:
            raise ValueError("Chunk configuration cannot advance through the document.")
        starts.append(next_start)
    return [(start, min(start + window_size, token_count)) for start in starts]


def _validate_document(record: dict[str, Any], config: ChunkConfig) -> list[int]:
    document_id = record.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("Each tokenized document must have a non-empty document_id.")
    token_ids = record.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"Document {document_id} has an empty or invalid token sequence.")
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise ValueError(f"Document {document_id} contains a non-integer token ID.")
    if any(token_id < 0 or token_id >= config.vocab_size for token_id in token_ids):
        raise ValueError(f"Document {document_id} contains an out-of-range token ID.")
    if len(token_ids) < 2:
        raise ValueError(f"Document {document_id} must contain BOS and EOS.")
    if token_ids[0] != config.bos_token_id:
        raise ValueError(f"Document {document_id} does not begin with BOS.")
    if token_ids[-1] != config.eos_token_id:
        raise ValueError(f"Document {document_id} does not end with EOS.")
    if (
        record.get("bos_token_id") != config.bos_token_id
        or record.get("eos_token_id") != config.eos_token_id
    ):
        raise ValueError(f"Document {document_id} special-token metadata is inconsistent.")
    if config.pad_token_id in token_ids:
        raise ValueError(f"Document {document_id} contains PAD in its source sequence.")
    if config.bos_token_id in token_ids[1:] or config.eos_token_id in token_ids[:-1]:
        raise ValueError(f"Document {document_id} contains an unexpected BOS or EOS.")
    if record.get("token_count") != len(token_ids):
        raise ValueError(f"Document {document_id} token_count does not match token_ids.")
    for hash_field in ("source_sha256", "clean_text_sha256"):
        value = record.get(hash_field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Document {document_id} has an invalid {hash_field}.")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(
                f"Document {document_id} has an invalid {hash_field}."
            ) from error
    if not isinstance(record.get("category"), str) or not record["category"]:
        raise ValueError(f"Document {document_id} has no category.")
    return token_ids


def chunk_document(
    record: dict[str, Any],
    tokenizer_hash: str,
    config: ChunkConfig = DEFAULT_CHUNK_CONFIG,
) -> list[dict[str, Any]]:
    """Convert one BOS/EOS-delimited document into padded next-token examples."""
    if len(tokenizer_hash) != 64:
        raise ValueError("tokenizer_hash must be a SHA-256 hex digest.")
    try:
        int(tokenizer_hash, 16)
    except ValueError as error:
        raise ValueError("tokenizer_hash must be a SHA-256 hex digest.") from error

    token_ids = _validate_document(record, config)
    document_id = record["document_id"]
    ranges = _chunk_ranges(len(token_ids), config)
    chunks: list[dict[str, Any]] = []
    target_positions: Counter[int] = Counter()
    represented_positions: set[int] = set()

    for chunk_id, (start, end) in enumerate(ranges):
        token_window = token_ids[start:end]
        input_ids = token_window[:-1]
        target_ids = token_window[1:]
        if not input_ids or len(input_ids) != len(target_ids):
            raise ValueError(f"Document {document_id} produced an invalid chunk.")
        if len(input_ids) > config.context_length:
            raise ValueError(f"Document {document_id} produced an oversized chunk.")

        valid_length = len(input_ids)
        padded_input_ids = input_ids + [config.pad_token_id] * (
            config.context_length - valid_length
        )
        padded_target_ids = target_ids + [config.pad_token_id] * (
            config.context_length - valid_length
        )
        attention_mask = [1] * valid_length + [0] * (
            config.context_length - valid_length
        )
        loss_mask = [1] * valid_length + [0] * (
            config.context_length - valid_length
        )

        represented_positions.update(range(start, end))
        target_positions.update(range(start + 1, end))
        chunks.append(
            {
                "input_ids": padded_input_ids,
                "target_ids": padded_target_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "category": record["category"],
                "source_document_hash": record["source_sha256"],
                "clean_text_sha256": record["clean_text_sha256"],
                "source_sha256": record["source_sha256"],
                "tokenizer_hash": tokenizer_hash,
                "source_token_start": start,
                "source_token_end": end,
                "valid_sequence_length": valid_length,
                "is_first_chunk": chunk_id == 0,
                "is_final_chunk": chunk_id == len(ranges) - 1,
            }
        )

    expected_positions = set(range(len(token_ids)))
    expected_targets = set(range(1, len(token_ids)))
    if represented_positions != expected_positions:
        raise AssertionError(f"Token coverage failed for document {document_id}.")
    if set(target_positions) != expected_targets:
        raise AssertionError(f"Target coverage failed for document {document_id}.")
    if chunks[0]["input_ids"][0] != config.bos_token_id:
        raise AssertionError(f"BOS was not preserved for document {document_id}.")
    if chunks[-1]["target_ids"][chunks[-1]["valid_sequence_length"] - 1] != config.eos_token_id:
        raise AssertionError(f"EOS target was not preserved for document {document_id}.")

    return chunks


def _validate_configuration() -> None:
    model_config = read_json(MODEL_CONFIG_FILE)
    tokenizer_config = read_json(v2_config.TOKENIZER_CONFIG_FILE)
    expected_special_tokens = {
        "<pad>": v2_config.PAD_TOKEN_ID,
        "<unk>": v2_config.UNK_TOKEN_ID,
        "<bos>": v2_config.BOS_TOKEN_ID,
        "<eos>": v2_config.EOS_TOKEN_ID,
    }
    configured_special_tokens = model_config["tokenizer"]["special_tokens"]
    tokenizer_special_tokens = tokenizer_config["special_tokens"]
    if model_config["architecture"]["context_length"] != v2_config.CONTEXT_LENGTH:
        raise ValueError("model_config.json and config.py context lengths differ.")
    if model_config["architecture"]["vocabulary_size"] != v2_config.VOCAB_SIZE:
        raise ValueError("model_config.json and config.py vocabulary sizes differ.")
    if model_config["tokenizer"]["vocabulary_size"] != v2_config.VOCAB_SIZE:
        raise ValueError("Tokenizer and model vocabulary sizes differ.")
    if tokenizer_config["vocab_size"] != v2_config.VOCAB_SIZE:
        raise ValueError("tokenizer_config.json has the wrong vocabulary size.")
    if configured_special_tokens != expected_special_tokens:
        raise ValueError("model_config.json special token IDs differ from config.py.")
    if tokenizer_special_tokens != expected_special_tokens:
        raise ValueError("tokenizer_config.json special token IDs differ from config.py.")
    if (
        tokenizer_config["pad_token_id"] != v2_config.PAD_TOKEN_ID
        or tokenizer_config["bos_token_id"] != v2_config.BOS_TOKEN_ID
        or tokenizer_config["eos_token_id"] != v2_config.EOS_TOKEN_ID
    ):
        raise ValueError("Tokenizer special token IDs are inconsistent.")


def _tokenizer_identity(
    manifest: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    paths = {
        "bpe_vocab.json": v2_config.TOKENIZER_VOCAB_FILE,
        "bpe_merges.json": v2_config.TOKENIZER_MERGES_FILE,
        "tokenizer_config.json": v2_config.TOKENIZER_CONFIG_FILE,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    recorded_tokenizer = manifest.get("tokenizer", {})
    expected_hashes = {
        "bpe_vocab.json": recorded_tokenizer.get("vocabulary_sha256"),
        "bpe_merges.json": recorded_tokenizer.get("merges_sha256"),
    }
    for name, expected_hash in expected_hashes.items():
        if expected_hash != hashes[name]:
            raise ValueError(f"V2 tokenizer artifact changed: {name}.")
    training_report = read_json(v2_config.TOKENIZER_TRAINING_REPORT_FILE)
    report_hashes = training_report["artifacts"]
    for name, report_key in (
        ("bpe_vocab.json", "bpe_vocab_sha256"),
        ("bpe_merges.json", "bpe_merges_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
    ):
        if report_hashes[report_key] != hashes[name]:
            raise ValueError(f"Tokenizer training report hash mismatch for {name}.")
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest(), hashes


def _statistics_for_split(
    chunks: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    config: ChunkConfig,
) -> dict[str, Any]:
    lengths = [chunk["valid_sequence_length"] for chunk in chunks]
    input_padding = sum(config.context_length - length for length in lengths)
    output_padding = input_padding
    target_occurrences: Counter[tuple[str, int]] = Counter()
    represented_tokens: set[tuple[str, int]] = set()
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        document_id = chunk["document_id"]
        chunks_by_document.setdefault(document_id, []).append(chunk)
        represented_tokens.update(
            (document_id, position)
            for position in range(chunk["source_token_start"], chunk["source_token_end"])
        )
        target_occurrences.update(
            (document_id, position)
            for position in range(chunk["source_token_start"] + 1, chunk["source_token_end"])
        )
    repeated_targets = sum(count - 1 for count in target_occurrences.values())
    source_token_count = sum(record["token_count"] for record in source_records)
    category_chunk_counts = Counter(chunk["category"] for chunk in chunks)
    overlap_sizes = [
        max(
            0,
            previous["source_token_end"] - current["source_token_start"],
        )
        for document_chunks in chunks_by_document.values()
        for previous, current in zip(document_chunks, document_chunks[1:])
    ]
    return {
        "original_document_count": len(source_records),
        "original_token_count_including_bos_eos": source_token_count,
        "generated_chunk_count": len(chunks),
        "average_chunk_length": sum(lengths) / len(lengths),
        "maximum_chunk_length": max(lengths),
        "minimum_chunk_length": min(lengths),
        "maximum_sequence_length": max(lengths),
        "context_length": config.context_length,
        "input_padding_token_count": input_padding,
        "target_padding_token_count": output_padding,
        "padding_token_count": input_padding,
        "total_padding_positions_in_input_and_target": input_padding + output_padding,
        "overlap_token_count": repeated_targets,
        "overlap_input_token_count": sum(overlap_sizes),
        "configured_overlap_tokens": config.chunk_overlap,
        "configured_stride_tokens": config.chunk_stride,
        "average_realized_window_overlap_tokens": (
            sum(overlap_sizes) / len(overlap_sizes) if overlap_sizes else 0.0
        ),
        "total_represented_tokens": len(represented_tokens),
        "total_target_tokens_represented": sum(
            record["token_count"] - 1 for record in source_records
        ),
        "target_token_positions": sum(target_occurrences.values()),
        "documents_successfully_chunked": len(chunks_by_document),
        "category_chunk_counts": dict(sorted(category_chunk_counts.items())),
    }


def _chunk_quality(
    split_chunks: dict[str, list[dict[str, Any]]],
    split_sources: dict[str, list[dict[str, Any]]],
    statistics: dict[str, dict[str, Any]],
    config: ChunkConfig,
) -> dict[str, Any]:
    source_split_by_id: dict[str, str] = {}
    for split_name, records in split_sources.items():
        for record in records:
            document_id = record["document_id"]
            if document_id in source_split_by_id:
                raise ValueError(f"Document {document_id} appears in multiple splits.")
            source_split_by_id[document_id] = split_name

    checks = {
        "all_documents_chunked": True,
        "all_source_token_positions_covered": True,
        "all_next_token_targets_covered_including_eos": True,
        "no_document_mixing": True,
        "input_target_shift_is_correct": True,
        "all_token_ids_in_vocabulary": True,
        "all_sequences_within_context": True,
        "padding_is_masked_from_loss": True,
        "bos_at_document_start": True,
        "eos_is_final_document_target": True,
        "no_eos_at_artificial_boundaries": True,
        "source_and_tokenizer_hashes_present": True,
    }

    for split_name, chunks in split_chunks.items():
        source_by_id = {r["document_id"]: r for r in split_sources[split_name]}
        chunks_by_id: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            document_id = chunk["document_id"]
            if source_split_by_id.get(document_id) != split_name:
                checks["no_document_mixing"] = False
            chunks_by_id.setdefault(document_id, []).append(chunk)
            for field in ("input_ids", "target_ids", "attention_mask", "loss_mask"):
                if len(chunk[field]) != config.context_length:
                    checks["all_sequences_within_context"] = False
            if chunk["valid_sequence_length"] > config.context_length:
                checks["all_sequences_within_context"] = False
            if any(
                token_id < 0 or token_id >= config.vocab_size
                for token_id in chunk["input_ids"] + chunk["target_ids"]
            ):
                checks["all_token_ids_in_vocabulary"] = False
            valid_length = chunk["valid_sequence_length"]
            start = chunk["source_token_start"]
            end = chunk["source_token_end"]
            source_tokens = source_by_id[document_id]["token_ids"]
            expected_inputs = source_tokens[start : end - 1]
            expected_targets = source_tokens[start + 1 : end]
            if (
                chunk["input_ids"][:valid_length] != expected_inputs
                or chunk["target_ids"][:valid_length] != expected_targets
            ):
                checks["input_target_shift_is_correct"] = False
            if (
                chunk["input_ids"][valid_length:] != [config.pad_token_id] * (config.context_length - valid_length)
                or chunk["target_ids"][valid_length:] != [config.pad_token_id] * (config.context_length - valid_length)
                or chunk["attention_mask"] != [1] * valid_length + [0] * (config.context_length - valid_length)
                or chunk["loss_mask"] != [1] * valid_length + [0] * (config.context_length - valid_length)
            ):
                checks["padding_is_masked_from_loss"] = False
            if (
                chunk["source_document_hash"] != source_by_id[document_id]["source_sha256"]
                or chunk["clean_text_sha256"] != source_by_id[document_id]["clean_text_sha256"]
                or len(chunk["tokenizer_hash"]) != 64
            ):
                checks["source_and_tokenizer_hashes_present"] = False

        for document_id, document_chunks in chunks_by_id.items():
            document_chunks.sort(key=lambda item: item["chunk_id"])
            source_tokens = source_by_id[document_id]["token_ids"]
            expected_positions = set(range(len(source_tokens)))
            represented_positions: set[int] = set()
            target_positions: set[int] = set()
            for chunk in document_chunks:
                start = chunk["source_token_start"]
                end = chunk["source_token_end"]
                represented_positions.update(range(start, end))
                target_positions.update(range(start + 1, end))
                is_final = chunk is document_chunks[-1]
                valid_length = chunk["valid_sequence_length"]
                if is_final and chunk["target_ids"][valid_length - 1] != config.eos_token_id:
                    checks["eos_is_final_document_target"] = False
                if not is_final and config.eos_token_id in chunk["target_ids"][:valid_length]:
                    checks["no_eos_at_artificial_boundaries"] = False
            if represented_positions != expected_positions:
                checks["all_source_token_positions_covered"] = False
            if target_positions != set(range(1, len(source_tokens))):
                checks["all_next_token_targets_covered_including_eos"] = False
            if document_chunks[0]["input_ids"][0] != config.bos_token_id:
                checks["bos_at_document_start"] = False

    for split_name, source_records in split_sources.items():
        if statistics[split_name]["documents_successfully_chunked"] != len(source_records):
            checks["all_documents_chunked"] = False

    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "overlap_note": (
            "Overlapping windows intentionally repeat some next-token targets. "
            "The overlap count measures repeated target positions, not token-ID "
            "value coincidences."
        ),
        "padding_note": (
            "input_ids and target_ids are right-padded to the model context. "
            "attention_mask and loss_mask are zero at padding positions; training "
            "must use loss_mask when reducing token loss."
        ),
        "model_compatibility": {
            "input_and_target_tensor_shape": [None, config.context_length],
            "model_logits_shape": [None, config.context_length, config.vocab_size],
            "model_forward_accepts_input_ids": True,
            "model_forward_consumes_attention_mask": False,
            "right_padding_is_causally_inert_for_valid_prefix": True,
            "loss_mask_must_be_applied_by_training_loop": True,
        },
        "split_statistics": statistics,
    }


def run_chunking() -> dict[str, Any]:
    _validate_configuration()
    config = DEFAULT_CHUNK_CONFIG
    manifest = read_json(MANIFEST_FILE)
    tokenizer_hash, tokenizer_artifact_hashes = _tokenizer_identity(manifest)
    manifest_documents = {
        document["document_id"]: document for document in manifest["documents"]
    }

    sources_by_split: dict[str, list[dict[str, Any]]] = {}
    chunks_by_split: dict[str, list[dict[str, Any]]] = {}
    statistics: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    all_document_ids: set[str] = set()

    for split_name in SPLIT_NAMES:
        split_info = manifest["split_files"][split_name]
        dataset_file = v2_config.DATA_DIR / split_info["path"]
        tokenized_file = V2_ROOT / split_info["tokenized_path"]
        dataset_hash = sha256_file(dataset_file)
        tokenized_hash = sha256_file(tokenized_file)
        if dataset_hash != split_info["sha256"]:
            raise ValueError(f"{split_name} source dataset hash differs from manifest.")
        if tokenized_hash != split_info["tokenized_sha256"]:
            raise ValueError(f"{split_name} tokenized source hash differs from manifest.")

        dataset_records = read_jsonl(dataset_file)
        token_records = read_jsonl(tokenized_file)
        if not dataset_records or len(dataset_records) != len(token_records):
            raise ValueError(f"{split_name} has missing or mismatched document records.")
        dataset_by_id = {record["document_id"]: record for record in dataset_records}
        if len(dataset_by_id) != len(dataset_records):
            raise ValueError(f"{split_name} contains duplicate document IDs.")
        split_chunks: list[dict[str, Any]] = []
        seen_split_ids: set[str] = set()
        for record in token_records:
            document_id = record["document_id"]
            if document_id in all_document_ids:
                raise ValueError(f"Document {document_id} appears in multiple splits.")
            all_document_ids.add(document_id)
            seen_split_ids.add(document_id)
            source_record = dataset_by_id.get(document_id)
            if source_record is None:
                raise ValueError(f"No source dataset record for document {document_id}.")
            manifest_record = manifest_documents.get(document_id)
            if manifest_record is None or manifest_record["split"] != split_name:
                raise ValueError(f"Document {document_id} is absent from the split manifest.")
            for field in ("category", "source_sha256"):
                if record[field] != source_record[field]:
                    raise ValueError(
                        f"{field} differs between tokenized and source record "
                        f"for document {document_id}."
                    )
                if record[field] != manifest_record[field]:
                    raise ValueError(
                        f"{field} differs from the dataset manifest for document "
                        f"{document_id}."
                    )
            if record["clean_text_sha256"] != manifest_record["clean_text_sha256"]:
                raise ValueError(
                    f"Clean-text hash differs from the dataset manifest for "
                    f"document {document_id}."
                )
            if record["split"] != split_name:
                raise ValueError(f"Document {document_id} has the wrong split label.")
            split_chunks.extend(chunk_document(record, tokenizer_hash, config))
        if seen_split_ids != set(dataset_by_id):
            raise ValueError(f"{split_name} tokenized and source document IDs differ.")
        expected_document_ids = set(split_info["document_ids"])
        if seen_split_ids != expected_document_ids:
            raise ValueError(f"{split_name} document IDs differ from the manifest.")
        sources_by_split[split_name] = token_records
        chunks_by_split[split_name] = split_chunks
        statistics[split_name] = _statistics_for_split(
            split_chunks,
            token_records,
            config,
        )
        source_hashes[split_name] = {
            "document_dataset_sha256": dataset_hash,
            "tokenized_dataset_sha256": tokenized_hash,
        }

    quality = _chunk_quality(chunks_by_split, sources_by_split, statistics, config)
    if not quality["all_checks_passed"]:
        raise AssertionError("Chunk quality validation failed.")

    split_manifest: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        output_file = CHUNK_OUTPUT_ROOT / split_name / "chunks.jsonl"
        write_jsonl(output_file, chunks_by_split[split_name])
        split_manifest[split_name] = {
            "chunk_count": len(chunks_by_split[split_name]),
            "source_document_count": len(sources_by_split[split_name]),
            "maximum_sequence_length": statistics[split_name]["maximum_sequence_length"],
            "chunks_path": output_file.relative_to(V2_ROOT).as_posix(),
            "chunks_sha256": sha256_file(output_file),
            "source_dataset_sha256": source_hashes[split_name][
                "document_dataset_sha256"
            ],
            "source_tokenized_dataset_sha256": source_hashes[split_name][
                "tokenized_dataset_sha256"
            ],
        }

    chunk_manifest = {
        "algorithm": "document-ordered shifted next-token windows",
        "context_length": config.context_length,
        "chunk_window_token_count": config.context_length + 1,
        "chunk_overlap": config.chunk_overlap,
        "chunk_stride": config.chunk_stride,
        "minimum_final_chunk_tokens": config.min_final_chunk_tokens,
        "tail_policy": (
            "If a regular-stride final window would be shorter than the minimum, "
            "move its start backward to retain a larger final window. This may "
            "increase final-window overlap; no source or target token is dropped."
        ),
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_artifact_sha256": tokenizer_artifact_hashes,
        "splits": split_manifest,
    }
    manifest["chunking"] = chunk_manifest
    write_json(MANIFEST_FILE, manifest)

    statistics_report = {
        "context_length": config.context_length,
        "chunk_window_token_count": config.context_length + 1,
        "chunk_overlap": config.chunk_overlap,
        "chunk_stride": config.chunk_stride,
        "minimum_final_chunk_tokens": config.min_final_chunk_tokens,
        "tokenizer_sha256": tokenizer_hash,
        "splits": statistics,
    }
    write_json(v2_config.DATA_DIR / "reports" / "chunk_statistics.json", statistics_report)
    write_json(v2_config.DATA_DIR / "reports" / "chunk_quality_report.json", quality)

    return {
        "manifest": chunk_manifest,
        "statistics": statistics,
        "quality": quality,
    }


def main() -> None:
    result = run_chunking()
    for split_name in SPLIT_NAMES:
        split_stats = result["statistics"][split_name]
        print(
            f"{split_name}: {split_stats['original_document_count']} documents, "
            f"{split_stats['generated_chunk_count']} chunks, "
            f"{split_stats['total_represented_tokens']} tokens covered"
        )
    print("Chunk quality:", result["quality"]["status"])


if __name__ == "__main__":
    main()
