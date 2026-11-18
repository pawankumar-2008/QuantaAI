from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
SOURCE_DIR = PROJECT_ROOT / "src"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Initial small-model configuration
MODEL_NAME = "QuantaLM"

VOCAB_SIZE = 2048
CONTEXT_LENGTH = 128

EMBEDDING_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
TRAIN_STEPS = 1000

# CPU training
DEVICE = "cpu"

# Dataset files
TRAINING_DATA_FILE = RAW_DATA_DIR / "training.txt"
