import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAINING_DATA_FILE,
    CHECKPOINT_DIR,
    CONTEXT_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    TRAIN_STEPS,
    DEVICE,
)

from src.model import QuantaLM
from src.tokenizer import QuantaTokenizer


def main():
    torch.manual_seed(42)

    # Load tokenizer vocabulary
    vocab_file = PROJECT_ROOT / "tokenizer" / "vocab.json"

    import json
    vocabulary = json.loads(vocab_file.read_text(encoding="utf-8"))
    tokenizer = QuantaTokenizer(vocabulary)

    # Load and tokenize training data
    text = TRAINING_DATA_FILE.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)

    data = torch.tensor(token_ids, dtype=torch.long)

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Total tokens:", len(data))

    if len(data) <= CONTEXT_LENGTH:
        raise ValueError("Training data is too small for the context length.")

    # Prepare training batches
    def get_batch():
        max_start = len(data) - CONTEXT_LENGTH - 1

        starts = torch.randint(
            0, max_start + 1, (BATCH_SIZE,)
        )

        x = torch.stack([
            data[i:i + CONTEXT_LENGTH]
            for i in starts
        ])

        y = torch.stack([
            data[i + 1:i + CONTEXT_LENGTH + 1]
            for i in starts
        ])

        return x.to(DEVICE), y.to(DEVICE)

    # Initialize model
    model = QuantaLM(vocab_size=tokenizer.vocab_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    print("\nTraining QuantaLM...")

    for step in range(TRAIN_STEPS):
        x, y = get_batch()

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

        if (step + 1) % 100 == 0:
            print(
                f"Step {step + 1}/{TRAIN_STEPS} | "
                f"Loss: {loss.item():.4f}"
            )

    # Save trained model
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / "quantalm.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_LENGTH,
    }, checkpoint_path)

    print("\nTraining completed!")
    print("Checkpoint saved:", checkpoint_path)


if __name__ == "__main__":
    main()