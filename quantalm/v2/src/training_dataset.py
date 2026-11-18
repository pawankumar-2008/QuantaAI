"""Validated train/validation loader for pre-chunked QuantaLM V2 examples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from .training_config import DEFAULT_TRAINING_CONFIG, TrainingConfig
    from .workspace_config import v2_config
except ImportError:
    from training_config import DEFAULT_TRAINING_CONFIG, TrainingConfig
    from workspace_config import v2_config


V2_ROOT = Path(__file__).resolve().parents[1]
SPLITS: tuple[str, ...] = ("train", "validation")
MANIFEST_FILE = v2_config.DATA_DIR / "manifests" / "dataset_manifest.json"
CHUNK_LENGTH_FIELDS = ("input_ids", "target_ids", "attention_mask", "loss_mask")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        records = []
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Blank chunk record at {path}:{line_number}.")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid chunk record at {path}:{line_number}.")
            records.append(record)
    return records


class ChunkDataset(Dataset[dict[str, Any]]):
    """Loads a single train or validation split; test is intentionally excluded."""

    def __init__(
        self,
        split: Literal["train", "validation"],
        data_dir: str | Path | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError("Training loaders accept only 'train' or 'validation'.")
        root = Path(data_dir) if data_dir is not None else v2_config.DATA_DIR
        self.split = split
        self.context_length = v2_config.CONTEXT_LENGTH
        self.vocab_size = v2_config.VOCAB_SIZE
        self.pad_token_id = v2_config.PAD_TOKEN_ID
        self.manifest_path = root / "manifests" / "dataset_manifest.json"
        self.manifest = read_json(self.manifest_path)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.manifest_hash = self.manifest_sha256
        compatibility = self.manifest.get("model_tokenizer_compatibility", {})
        if compatibility.get("configured_model_vocabulary_size") != self.vocab_size:
            raise ValueError("Dataset manifest vocabulary differs from the V2 model.")
        if compatibility.get("tokenizer_vocabulary_size") != self.vocab_size:
            raise ValueError("Dataset manifest tokenizer vocabulary differs from the model.")
        chunking = self.manifest.get("chunking")
        if not isinstance(chunking, dict) or chunking.get("context_length") != self.context_length:
            raise ValueError("Dataset manifest chunk context differs from the V2 model.")
        self.tokenizer_hash = self.manifest["chunking"]["tokenizer_sha256"]
        tokenizer_artifacts = {
            "bpe_vocab.json": v2_config.TOKENIZER_VOCAB_FILE,
            "bpe_merges.json": v2_config.TOKENIZER_MERGES_FILE,
            "tokenizer_config.json": v2_config.TOKENIZER_CONFIG_FILE,
        }
        artifact_hashes = {
            name: sha256_file(path) for name, path in tokenizer_artifacts.items()
        }
        canonical_hash_data = json.dumps(
            artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(canonical_hash_data).hexdigest() != self.tokenizer_hash:
            raise ValueError("V2 tokenizer files differ from the chunk manifest.")
        tokenizer_config = read_json(v2_config.TOKENIZER_CONFIG_FILE)
        expected_tokens = {
            "<pad>": v2_config.PAD_TOKEN_ID,
            "<unk>": v2_config.UNK_TOKEN_ID,
            "<bos>": v2_config.BOS_TOKEN_ID,
            "<eos>": v2_config.EOS_TOKEN_ID,
        }
        if tokenizer_config.get("vocab_size") != self.vocab_size:
            raise ValueError("V2 tokenizer vocabulary size differs from config.py.")
        if tokenizer_config.get("special_tokens") != expected_tokens:
            raise ValueError("V2 tokenizer special token IDs differ from config.py.")
        split_info = self.manifest["chunking"]["splits"][split]
        self.path = root / "chunks" / split / "chunks.jsonl"
        if self.path.relative_to(root.parent).as_posix() != split_info["chunks_path"]:
            raise ValueError(f"{split} chunk path differs from the dataset manifest.")
        if sha256_file(self.path) != split_info["chunks_sha256"]:
            raise ValueError(f"{split} chunk file hash differs from the manifest.")
        if split_info["source_document_count"] <= 0:
            raise ValueError(f"{split} has no source documents.")
        source_split_info = self.manifest["split_files"][split]
        source_dataset_path = root / source_split_info["path"]
        tokenized_path = root.parent / source_split_info["tokenized_path"]
        if sha256_file(source_dataset_path) != split_info["source_dataset_sha256"]:
            raise ValueError(f"{split} document dataset differs from the chunk manifest.")
        if (
            sha256_file(tokenized_path)
            != split_info["source_tokenized_dataset_sha256"]
        ):
            raise ValueError(f"{split} tokenized dataset differs from the chunk manifest.")
        self.records = _read_jsonl(self.path)
        if len(self.records) != split_info["chunk_count"]:
            raise ValueError(f"{split} chunk count differs from the manifest.")
        self.document_ids: set[str] = set()
        self._validate_records(split_info)

    def _validate_records(self, split_info: dict[str, Any]) -> None:
        documents: set[str] = set()
        chunk_ids: dict[str, list[int]] = {}
        for record in self.records:
            document_id = record.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("Chunk record has an invalid document ID.")
            if record.get("split") is not None and record["split"] != self.split:
                raise ValueError(f"Chunk {document_id} is labeled with the wrong split.")
            if record.get("tokenizer_hash") != self.tokenizer_hash:
                raise ValueError(f"Chunk {document_id} tokenizer hash mismatch.")
            if (
                not isinstance(record.get("source_document_hash"), str)
                or not isinstance(record.get("clean_text_sha256"), str)
                or len(record["source_document_hash"]) != 64
                or len(record["clean_text_sha256"]) != 64
            ):
                raise ValueError(f"Chunk {document_id} source provenance hashes are invalid.")
            values: dict[str, torch.Tensor] = {}
            for field in CHUNK_LENGTH_FIELDS:
                raw_values = record.get(field)
                if (
                    not isinstance(raw_values, list)
                    or len(raw_values) != self.context_length
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in raw_values)
                ):
                    raise ValueError(
                        f"Chunk {document_id} field {field} must have "
                        f"{self.context_length} integer values."
                    )
                values[field] = torch.tensor(raw_values, dtype=torch.long)
            input_ids = values["input_ids"]
            target_ids = values["target_ids"]
            attention_mask = values["attention_mask"]
            loss_mask = values["loss_mask"]
            if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
                raise ValueError(f"Chunk {document_id} has out-of-range input IDs.")
            if torch.any(target_ids < 0) or torch.any(target_ids >= self.vocab_size):
                raise ValueError(f"Chunk {document_id} has out-of-range target IDs.")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError(f"Chunk {document_id} has an invalid attention mask.")
            if not torch.all((loss_mask == 0) | (loss_mask == 1)):
                raise ValueError(f"Chunk {document_id} has an invalid loss mask.")
            if not torch.equal(loss_mask, attention_mask):
                raise ValueError(f"Chunk {document_id} loss and attention masks differ.")
            valid = int(loss_mask.sum().item())
            if valid != record.get("valid_sequence_length") or valid < 1:
                raise ValueError(f"Chunk {document_id} valid length is inconsistent.")
            if (
                torch.any(input_ids[valid:] != self.pad_token_id)
                or torch.any(target_ids[valid:] != self.pad_token_id)
                or torch.any(attention_mask[valid:] != 0)
                or torch.any(loss_mask[valid:] != 0)
            ):
                raise ValueError(f"Chunk {document_id} padding or mask is inconsistent.")
            if (
                torch.any(input_ids[:valid] == self.pad_token_id)
                or torch.any(target_ids[:valid] == self.pad_token_id)
            ):
                raise ValueError(f"Chunk {document_id} has PAD in a valid position.")
            chunk_index = record.get("chunk_id")
            if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
                raise ValueError(f"Chunk {document_id} has an invalid chunk_id.")
            chunk_ids.setdefault(document_id, []).append(chunk_index)
            documents.add(document_id)
        if len(documents) != split_info["source_document_count"]:
            raise ValueError(f"{self.split} source-document count differs from the manifest.")
        for document_id, indices in chunk_ids.items():
            if indices != list(range(len(indices))):
                raise ValueError(f"Chunks for document {document_id} are not ordered.")
        self.document_ids = documents

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
            "target_ids": torch.tensor(record["target_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(record["attention_mask"], dtype=torch.bool),
            "loss_mask": torch.tensor(record["loss_mask"], dtype=torch.bool),
            "document_id": record["document_id"],
            "chunk_id": record["chunk_id"],
            "category": record["category"],
            "valid_sequence_length": record["valid_sequence_length"],
        }


class DeterministicEpochSampler(Sampler[int]):
    """Recreate an epoch permutation and resume it from a sample offset."""

    def __init__(self, data_source: Dataset[Any], seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        if epoch < 0 or start_index < 0 or start_index > len(self.data_source):
            raise ValueError("Sampler epoch or start_index is invalid.")
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.data_source), generator=generator).tolist()
        return iter(permutation[self.start_index :])

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


def build_training_loaders(
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    data_dir: str | Path | None = None,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    train_dataset = ChunkDataset("train", data_dir)
    validation_dataset = ChunkDataset("validation", data_dir)
    sampler = DeterministicEpochSampler(train_dataset, config.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, validation_loader
