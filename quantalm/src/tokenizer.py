import json
import re
from collections import Counter
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "training.txt"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
VOCAB_FILE = TOKENIZER_DIR / "vocab.json"

VOCAB_SIZE = 2048

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def split_text(text):
    """Split text into words, numbers, and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def train_tokenizer(text, vocab_size=VOCAB_SIZE):
    """Build a vocabulary from the supplied training corpus."""
    tokens = split_text(text)
    frequencies = Counter(tokens)

    available_slots = max(0, vocab_size - len(SPECIAL_TOKENS))

    # Frequent tokens are retained first.
    vocabulary_tokens = [
        token
        for token, _ in frequencies.most_common(available_slots)
    ]

    vocabulary = {
        token: index
        for index, token in enumerate(SPECIAL_TOKENS)
    }

    for token in vocabulary_tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


class QuantaTokenizer:
    def __init__(self, vocabulary):
        self.vocab = vocabulary
        self.id_to_token = {
            index: token for token, index in vocabulary.items()
        }

        self.unk_id = self.vocab["<unk>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]

    def encode(self, text, add_special_tokens=True):
        tokens = split_text(text)
        ids = [
            self.vocab.get(token, self.unk_id)
            for token in tokens
        ]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        return ids

    def decode(self, token_ids):
        tokens = [
            self.id_to_token.get(int(token_id), "<unk>")
            for token_id in token_ids
        ]

        # Remove boundary and padding markers.
        tokens = [
            token for token in tokens
            if token not in SPECIAL_TOKENS
        ]

        text = " ".join(tokens)

        # Clean up spacing around common punctuation.
        text = re.sub(r"\s+([.,!?;:%)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)

        return text

    @property
    def vocab_size(self):
        return len(self.vocab)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_FILE}"
        )

    text = DATA_FILE.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("The training dataset is empty.")

    vocabulary = train_tokenizer(text)

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    VOCAB_FILE.write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tokenizer = QuantaTokenizer(vocabulary)

    print("QuantaLM tokenizer trained successfully.")
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Training tokens:", len(split_text(text)))
    print("Vocabulary saved to:", VOCAB_FILE)

    sample = "A database stores organized information."
    encoded = tokenizer.encode(sample)

    print("\nSample text:", sample)
    print("Token IDs:", encoded)
    print("Decoded text:", tokenizer.decode(encoded))


if __name__ == "__main__":
    main()