"""Independent byte-level BPE tokenizer for QuantaLM V2."""

from __future__ import annotations

import json
import hashlib
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


V2_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = V2_ROOT / "tokenizer"
VOCAB_FILE = TOKENIZER_DIR / "bpe_vocab.json"
MERGES_FILE = TOKENIZER_DIR / "bpe_merges.json"
CONFIG_FILE = TOKENIZER_DIR / "tokenizer_config.json"

SPECIAL_TOKENS = {
    "<pad>": 256,
    "<unk>": 257,
    "<bos>": 258,
    "<eos>": 259,
}
PAD_ID = SPECIAL_TOKENS["<pad>"]
UNK_ID = SPECIAL_TOKENS["<unk>"]
BOS_ID = SPECIAL_TOKENS["<bos>"]
EOS_ID = SPECIAL_TOKENS["<eos>"]
BYTE_TOKEN_COUNT = 256
MERGE_START_ID = 260
DEFAULT_VOCAB_SIZE = 2048


def split_into_pieces(text: str) -> list[str]:
    """Keep whitespace runs and non-whitespace runs intact."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return re.findall(r"\s+|[^\s]+", text)


def _apply_merge(
    sequence: list[int],
    left: int,
    right: int,
    merged_id: int,
) -> list[int]:
    output: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == left
            and sequence[index + 1] == right
        ):
            output.append(merged_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output


class QuantaBPETokenizer:
    """UTF-8 byte BPE with standalone, V2-only artifact paths."""

    def __init__(
        self,
        merges: list[list[int]],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ) -> None:
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise TypeError("vocab_size must be an integer.")
        if vocab_size < MERGE_START_ID:
            raise ValueError(f"vocab_size must be at least {MERGE_START_ID}.")
        if not isinstance(merges, list):
            raise TypeError("merges must be a list.")

        self.merges = merges
        self.vocab_size = vocab_size
        self.merge_map: dict[tuple[int, int], int] = {}
        self.merge_order: dict[tuple[int, int], int] = {}
        self.token_bytes = {index: bytes([index]) for index in range(BYTE_TOKEN_COUNT)}

        for rank, merge in enumerate(merges):
            if (
                not isinstance(merge, list)
                or len(merge) != 3
                or any(not isinstance(value, int) or isinstance(value, bool) for value in merge)
            ):
                raise ValueError(f"Invalid merge record at rank {rank}.")
            left, right, merged_id = merge
            pair = (left, right)
            expected_id = MERGE_START_ID + rank
            if merged_id != expected_id:
                raise ValueError(
                    f"Merge rank {rank} must create token ID {expected_id}."
                )
            if pair in self.merge_map:
                raise ValueError(f"Duplicate merge pair: {pair}.")
            if left not in self.token_bytes or right not in self.token_bytes:
                raise ValueError(f"Merge {rank} references a token not created earlier.")
            if merged_id >= vocab_size:
                raise ValueError(f"Merge token ID {merged_id} is outside the vocabulary.")
            self.merge_map[pair] = merged_id
            self.merge_order[pair] = rank
            self.token_bytes[merged_id] = (
                self.token_bytes[left] + self.token_bytes[right]
            )

        if MERGE_START_ID + len(merges) != vocab_size:
            raise ValueError(
                f"Expected exactly {vocab_size - MERGE_START_ID} merge rules, "
                f"got {len(merges)}."
            )
        self.special_token_ids = dict(SPECIAL_TOKENS)
        self.pad_token_id = PAD_ID
        self.unk_token_id = UNK_ID
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID

    @classmethod
    def train(
        cls,
        documents: list[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        seed: int = 20261106,
    ) -> "QuantaBPETokenizer":
        if vocab_size != DEFAULT_VOCAB_SIZE:
            raise ValueError(
                f"V2 vocabulary size is fixed at {DEFAULT_VOCAB_SIZE}; "
                "change requires an explicit model compatibility decision."
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer.")
        if not documents or any(not isinstance(text, str) for text in documents):
            raise ValueError("Training requires at least one text document.")
        if not any(text.strip() for text in documents):
            raise ValueError("Training documents contain no non-whitespace text.")

        piece_counts: Counter[str] = Counter()
        for text in documents:
            piece_counts.update(split_into_pieces(text))
        sequences = [
            (list(piece.encode("utf-8")), frequency)
            for piece, frequency in sorted(piece_counts.items())
        ]

        rng = random.Random(seed)
        merges: list[list[int]] = []
        desired_merge_count = vocab_size - MERGE_START_ID

        for _ in range(desired_merge_count):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence, frequency in sequences:
                for index in range(len(sequence) - 1):
                    pair_counts[(sequence[index], sequence[index + 1])] += frequency
            if not pair_counts:
                break

            highest_frequency = max(pair_counts.values())
            tied_pairs = sorted(
                pair for pair, frequency in pair_counts.items()
                if frequency == highest_frequency
            )
            left, right = rng.choice(tied_pairs)
            merged_id = MERGE_START_ID + len(merges)
            sequences = [
                (_apply_merge(sequence, left, right, merged_id), frequency)
                for sequence, frequency in sequences
            ]
            merges.append([left, right, merged_id])

        if len(merges) != desired_merge_count:
            achieved_vocab = MERGE_START_ID + len(merges)
            raise ValueError(
                f"Training data supports only {achieved_vocab} distinct tokens, "
                f"not the required {vocab_size}; no padding tokens were invented."
            )
        return cls(merges=merges, vocab_size=vocab_size)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        token_ids: list[int] = []
        for piece in split_into_pieces(text):
            sequence = list(piece.encode("utf-8"))
            while len(sequence) >= 2:
                best_pair: tuple[int, int] | None = None
                best_rank: int | None = None
                for index in range(len(sequence) - 1):
                    pair = (sequence[index], sequence[index + 1])
                    rank = self.merge_order.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_pair = pair
                        best_rank = rank
                if best_pair is None:
                    break
                sequence = _apply_merge(
                    sequence,
                    best_pair[0],
                    best_pair[1],
                    self.merge_map[best_pair],
                )
            token_ids.extend(sequence)
        if add_special_tokens:
            return [BOS_ID, *token_ids, EOS_ID]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        output = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token IDs must be integers.")
            if token_id in self.token_bytes:
                output.extend(self.token_bytes[token_id])
            elif token_id in SPECIAL_TOKENS.values():
                continue
            else:
                raise ValueError(f"Invalid token ID: {token_id}.")
        return output.decode("utf-8", errors="replace")

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def load(
        cls,
        tokenizer_dir: str | Path | None = None,
    ) -> "QuantaBPETokenizer":
        directory = Path(tokenizer_dir) if tokenizer_dir is not None else TOKENIZER_DIR
        vocab_path = directory / "bpe_vocab.json"
        merges_path = directory / "bpe_merges.json"
        config_path = directory / "tokenizer_config.json"
        for path in (vocab_path, merges_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"V2 tokenizer artifact not found: {path}")

        vocab = cls._read_json(vocab_path)
        merges = cls._read_json(merges_path)
        config = cls._read_json(config_path)
        if vocab.get("model") != "QuantaLM-BPE":
            raise ValueError("Invalid V2 BPE vocabulary metadata.")
        if vocab.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("V2 BPE vocabulary special-token metadata mismatch.")
        if config.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("V2 tokenizer special token configuration mismatch.")
        expected_special_ids = {
            "pad_token_id": PAD_ID,
            "unk_token_id": UNK_ID,
            "bos_token_id": BOS_ID,
            "eos_token_id": EOS_ID,
        }
        if any(config.get(name) != expected for name, expected in expected_special_ids.items()):
            raise ValueError("V2 tokenizer special token IDs are inconsistent.")
        if vocab.get("vocab_size") != config.get("vocab_size"):
            raise ValueError("V2 tokenizer vocabulary metadata mismatch.")
        if vocab.get("byte_tokens") != BYTE_TOKEN_COUNT:
            raise ValueError("V2 tokenizer byte vocabulary metadata mismatch.")
        if vocab.get("merge_start_id") != MERGE_START_ID:
            raise ValueError("V2 tokenizer merge-start metadata mismatch.")
        tokenizer = cls(merges=merges, vocab_size=config["vocab_size"])
        if tokenizer.vocab_size != DEFAULT_VOCAB_SIZE:
            raise ValueError("Loaded V2 tokenizer does not have vocabulary size 2048.")
        if vocab.get("learned_merges") != len(merges):
            raise ValueError("V2 BPE merge count does not match vocabulary metadata.")
        training_report_path = directory / "tokenizer_training_report.json"
        if training_report_path.is_file():
            training_report = cls._read_json(training_report_path)
            expected_hashes = training_report.get("artifacts", {})
            if expected_hashes.get("bpe_vocab_sha256") != hashlib.sha256(
                vocab_path.read_bytes()
            ).hexdigest():
                raise ValueError("V2 BPE vocabulary hash does not match its training report.")
            if expected_hashes.get("bpe_merges_sha256") != hashlib.sha256(
                merges_path.read_bytes()
            ).hexdigest():
                raise ValueError("V2 BPE merges hash does not match its training report.")
            if expected_hashes.get("tokenizer_config_sha256") != hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest():
                raise ValueError("V2 tokenizer config hash does not match its training report.")
        return tokenizer

    def vocab_metadata(self) -> dict[str, Any]:
        return {
            "model": "QuantaLM-BPE",
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "byte_tokens": BYTE_TOKEN_COUNT,
            "merge_start_id": MERGE_START_ID,
            "learned_merges": len(self.merges),
        }

    def config_metadata(self) -> dict[str, Any]:
        return {
            "model": "QuantaLM-BPE",
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "pad_token_id": PAD_ID,
            "unk_token_id": UNK_ID,
            "bos_token_id": BOS_ID,
            "eos_token_id": EOS_ID,
            "byte_token_count": BYTE_TOKEN_COUNT,
            "merge_start_id": MERGE_START_ID,
            "encoding": "UTF-8 bytes with whitespace-preserving BPE",
            "decoding": "UTF-8 with replacement for malformed token byte streams",
        }

    def save(
        self,
        tokenizer_dir: str | Path | None = None,
        *,
        overwrite: bool = False,
        config_extra: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        directory = Path(tokenizer_dir) if tokenizer_dir is not None else TOKENIZER_DIR
        directory = directory.resolve()
        if not directory.is_relative_to(TOKENIZER_DIR.resolve()):
            raise ValueError("V2 tokenizer artifacts must remain inside v2/tokenizer/.")
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "bpe_vocab": directory / "bpe_vocab.json",
            "bpe_merges": directory / "bpe_merges.json",
            "tokenizer_config": directory / "tokenizer_config.json",
        }
        existing_paths = [path for path in paths.values() if path.exists()]
        if existing_paths and not overwrite:
            raise FileExistsError(
                "V2 tokenizer artifacts already exist; pass overwrite=True to replace them: "
                + ", ".join(str(path) for path in existing_paths)
            )
        config = self.config_metadata()
        if config_extra:
            config.update(config_extra)
        payloads = {
            "bpe_vocab": self.vocab_metadata(),
            "bpe_merges": self.merges,
            "tokenizer_config": config,
        }
        for key, path in paths.items():
            encoded = (
                json.dumps(payloads[key], ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{path.stem}-",
                    suffix=".tmp",
                    dir=directory,
                    delete=False,
                ) as temporary:
                    temporary.write(encoded)
                    temporary_path = Path(temporary.name)
                if path.exists() and not overwrite:
                    raise FileExistsError(
                        f"V2 tokenizer artifact appeared during save: {path}"
                    )
                os.replace(temporary_path, path)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        return paths
