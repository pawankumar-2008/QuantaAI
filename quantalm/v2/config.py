"""Configuration for the independent QuantaLM V2 workspace."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
V1_ROOT = PROJECT_ROOT.parent

SOURCE_DIR = PROJECT_ROOT / "src"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
LOG_DIR = PROJECT_ROOT / "logs"

# Independent V2 tokenizer; V1 artifacts are reference-only and never written.
TOKENIZER_IMPLEMENTATION = SOURCE_DIR / "bpe_tokenizer.py"
TOKENIZER_VOCAB_FILE = TOKENIZER_DIR / "bpe_vocab.json"
TOKENIZER_MERGES_FILE = TOKENIZER_DIR / "bpe_merges.json"
TOKENIZER_CONFIG_FILE = TOKENIZER_DIR / "tokenizer_config.json"
TOKENIZER_TRAINING_REPORT_FILE = TOKENIZER_DIR / "tokenizer_training_report.json"
REFERENCE_V1_TOKENIZER_IMPLEMENTATION = V1_ROOT / "src" / "tokenizer_v2.py"

PAD_TOKEN_ID = 256
UNK_TOKEN_ID = 257
BOS_TOKEN_ID = 258
EOS_TOKEN_ID = 259

MODEL_NAME = "QuantaLM-V2"
VOCAB_SIZE = 2048
CONTEXT_LENGTH = 256
EMBEDDING_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
FEED_FORWARD_DIM = 512
DROPOUT = 0.1

DEVICE = "cpu"
DTYPE = "float32"

# Training remains a separate manual action and requires --confirm-training.
TRAINING_ENABLED = True
