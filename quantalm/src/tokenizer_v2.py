import json
import re
import time
from collections import Counter
from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_training.txt"
)

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"

VOCAB_FILE = TOKENIZER_DIR / "bpe_vocab.json"
MERGES_FILE = TOKENIZER_DIR / "bpe_merges.json"

TARGET_VOCAB_SIZE = 2048

# Byte IDs: 0-255
# Special IDs: 256-259
# Learned merge IDs: 260 onwards

SPECIAL_TOKENS = {
    "<pad>": 256,
    "<unk>": 257,
    "<bos>": 258,
    "<eos>": 259,
}

PAD_ID = 256
UNK_ID = 257
BOS_ID = 258
EOS_ID = 259

MERGE_START_ID = 260


# ============================================================
# TEXT PROCESSING
# ============================================================

def split_into_pieces(text):
    """
    Split text into whitespace and non-whitespace pieces.

    This preserves spaces, punctuation and line breaks.
    """
    return re.findall(r"\s+|[^\s]+", text)


def apply_merge(sequence, left, right, new_id):
    """
    Merge every occurrence of an adjacent token pair.
    """
    result = []

    i = 0

    while i < len(sequence):

        if (
            i + 1 < len(sequence)
            and sequence[i] == left
            and sequence[i + 1] == right
        ):

            result.append(new_id)
            i += 2

        else:

            result.append(sequence[i])
            i += 1

    return result


# ============================================================
# QUANTALM BPE TOKENIZER
# ============================================================

class QuantaBPETokenizer:

    def __init__(self, merges):

        self.merges = merges

        self.merge_map = {}
        self.merge_order = {}

        # ----------------------------------------------------
        # Build merge lookup tables
        # ----------------------------------------------------

        for rank, merge in enumerate(merges):

            left, right, new_id = merge

            pair = (left, right)

            if pair in self.merge_map:
                raise ValueError(
                    f"Duplicate BPE merge pair: {pair}"
                )

            if new_id != MERGE_START_ID + rank:
                raise ValueError(
                    "Invalid BPE token ID sequence."
                )

            self.merge_map[pair] = new_id
            self.merge_order[pair] = rank

        # ----------------------------------------------------
        # Build token-to-byte mapping
        # ----------------------------------------------------

        self.token_bytes = {
            i: bytes([i])
            for i in range(256)
        }

        for left, right, new_id in merges:

            if left not in self.token_bytes:
                raise ValueError(
                    f"Invalid left token in merge: {left}"
                )

            if right not in self.token_bytes:
                raise ValueError(
                    f"Invalid right token in merge: {right}"
                )

            if new_id in self.token_bytes:
                raise ValueError(
                    f"Duplicate token ID: {new_id}"
                )

            self.token_bytes[new_id] = (
                self.token_bytes[left]
                + self.token_bytes[right]
            )

        self.vocab_size = (
            MERGE_START_ID + len(merges)
        )

        if self.vocab_size > TARGET_VOCAB_SIZE:
            raise ValueError(
                "Vocabulary exceeds configured limit."
            )

    # ========================================================
    # TRAIN BPE
    # ========================================================

    @classmethod
    def train(
        cls,
        text,
        vocab_size=TARGET_VOCAB_SIZE
    ):

        if not text.strip():

            raise ValueError(
                "Training dataset is empty."
            )

        if vocab_size < MERGE_START_ID:

            raise ValueError(
                "Vocabulary size is too small."
            )

        max_merges = vocab_size - MERGE_START_ID

        pieces = Counter(
            split_into_pieces(text)
        )

        sequences = [

            (
                list(piece.encode("utf-8")),
                frequency
            )

            for piece, frequency in pieces.items()

        ]

        merges = []

        print("\nStarting QuantaLM BPE training...")
        print("Unique text pieces:", len(sequences))
        print("Maximum vocabulary:", vocab_size)

        start_time = time.time()

        # ----------------------------------------------------
        # Learn merge rules
        # ----------------------------------------------------

        for step in range(max_merges):

            pair_counts = Counter()

            for sequence, frequency in sequences:

                for i in range(len(sequence) - 1):

                    pair = (
                        sequence[i],
                        sequence[i + 1]
                    )

                    pair_counts[pair] += frequency

            if not pair_counts:
                break

            best_pair, frequency = (
                pair_counts.most_common(1)[0]
            )

            new_id = MERGE_START_ID + len(merges)

            updated_sequences = []

            for sequence, weight in sequences:

                updated = apply_merge(
                    sequence,
                    best_pair[0],
                    best_pair[1],
                    new_id
                )

                updated_sequences.append(
                    (updated, weight)
                )

            sequences = updated_sequences

            merges.append([
                best_pair[0],
                best_pair[1],
                new_id
            ])

            if (step + 1) % 200 == 0:

                print(
                    f"Merges learned: {step + 1}, "
                    f"pair frequency: {frequency}"
                )

        elapsed = time.time() - start_time

        tokenizer = cls(merges)

        print("\nBPE training completed.")
        print("Vocabulary size:", tokenizer.vocab_size)
        print("Learned merges:", len(merges))
        print("Training time:", round(elapsed, 2), "seconds")

        return tokenizer

    # ========================================================
    # OPTIMIZED ENCODING
    # ========================================================

    def encode(
        self,
        text,
        add_special_tokens=False
    ):

        token_ids = []

        for piece in split_into_pieces(text):

            sequence = list(
                piece.encode("utf-8")
            )

            # Repeatedly apply the highest-priority
            # available BPE merge.

            while len(sequence) >= 2:

                best_pair = None
                best_rank = float("inf")

                for i in range(len(sequence) - 1):

                    pair = (
                        sequence[i],
                        sequence[i + 1]
                    )

                    rank = self.merge_order.get(pair)

                    if (
                        rank is not None
                        and rank < best_rank
                    ):

                        best_rank = rank
                        best_pair = pair

                if best_pair is None:
                    break

                new_id = self.merge_map[best_pair]

                sequence = apply_merge(
                    sequence,
                    best_pair[0],
                    best_pair[1],
                    new_id
                )

            token_ids.extend(sequence)

        if add_special_tokens:

            token_ids = (
                [BOS_ID]
                + token_ids
                + [EOS_ID]
            )

        return token_ids

    # ========================================================
    # DECODING
    # ========================================================

    def decode(self, token_ids):

        result = bytearray()

        for token_id in token_ids:

            token_id = int(token_id)

            # Regular byte or merged token
            if token_id in self.token_bytes:

                result.extend(
                    self.token_bytes[token_id]
                )

            # Special tokens are skipped
            elif token_id in SPECIAL_TOKENS.values():

                continue

            else:

                raise ValueError(
                    f"Invalid token ID: {token_id}"
                )

        return result.decode(
            "utf-8",
            errors="replace"
        )

    # ========================================================
    # SAVE TOKENIZER
    # ========================================================

    def save(self):

        TOKENIZER_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        vocab_data = {

            "model": "QuantaLM-BPE",

            "vocab_size": self.vocab_size,

            "special_tokens": SPECIAL_TOKENS,

            "byte_tokens": 256,

            "merge_start_id": MERGE_START_ID,

            "learned_merges": len(self.merges)

        }

        with open(
            VOCAB_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                vocab_data,
                file,
                indent=4
            )

        with open(
            MERGES_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.merges,
                file,
                indent=4
            )

        print("\nTokenizer saved successfully.")

        print("Vocabulary:", VOCAB_FILE)
        print("Merges:", MERGES_FILE)

    # ========================================================
    # LOAD TOKENIZER
    # ========================================================

    @classmethod
    def load(cls):

        with open(
            VOCAB_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            vocab_data = json.load(file)

        with open(
            MERGES_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            merges = json.load(file)

        if vocab_data["model"] != "QuantaLM-BPE":

            raise ValueError(
                "Invalid tokenizer metadata."
            )

        if vocab_data["special_tokens"] != SPECIAL_TOKENS:

            raise ValueError(
                "Special token configuration mismatch."
            )

        tokenizer = cls(merges)

        if tokenizer.vocab_size != vocab_data["vocab_size"]:

            raise ValueError(
                "Vocabulary size mismatch."
            )

        return tokenizer


# ============================================================
# TOKENIZER TESTING
# ============================================================

def test_tokenizer(tokenizer):

    samples = [

        "QuantaLM is our custom language model.",

        "A database stores organized information.",

        "Artificial intelligence and machine learning.",

        "Byte-level tokenization handles unfamiliar words.",

        "भारत एक विविधताओं वाला देश है।",

        "public static void main(String[] args) {}",

        "E = mc^2; x = y + z",

        "Computer networks connect multiple devices."

    ]

    print("\nROUND-TRIP TESTS")

    for index, sample in enumerate(samples, start=1):

        encoded = tokenizer.encode(sample)

        decoded = tokenizer.decode(encoded)

        passed = sample == decoded

        print(f"\nTest {index}")
        print("Original:", sample)
        print("Token count:", len(encoded))
        print("Decoded:", decoded)
        print("Result:", "PASS" if passed else "FAIL")

        if not passed:

            raise RuntimeError(
                f"Round-trip test {index} failed."
            )

    print("\nAll round-trip tests passed!")


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset(tokenizer):

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    token_ids = tokenizer.encode(text)

    character_count = len(text)
    token_count = len(token_ids)

    if token_count == 0:

        raise ValueError(
            "Dataset produced zero tokens."
        )

    print("\nDATASET TOKENIZATION REPORT")

    print("Characters:", character_count)
    print("Tokens:", token_count)

    print(
        "Average characters/token:",
        round(character_count / token_count, 2)
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    print(
        "Compression ratio:",
        round(character_count / token_count, 2),
        "characters per token"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Training dataset not found: {DATA_FILE}"
        )

    text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    print("Dataset loaded successfully.")
    print("Characters:", len(text))

    # Train the tokenizer
    tokenizer = QuantaBPETokenizer.train(
        text,
        vocab_size=TARGET_VOCAB_SIZE
    )

    # Save the tokenizer
    tokenizer.save()

    # Reload from disk
    loaded_tokenizer = QuantaBPETokenizer.load()

    print("\nTokenizer save/load verification passed.")

    # Run tests
    test_tokenizer(loaded_tokenizer)

    # Analyze full dataset
    analyze_dataset(loaded_tokenizer)

    print("\nQUANTALM TOKENIZER V2 READY FOR FURTHER TESTING.")


if __name__ == "__main__":

    main()