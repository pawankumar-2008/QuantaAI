import sys
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import QuantaLM
from src.tokenizer import QuantaTokenizer
from config import CONTEXT_LENGTH, DEVICE


def load_model():
    vocab_file = PROJECT_ROOT / "tokenizer" / "vocab.json"
    checkpoint_file = PROJECT_ROOT / "checkpoints" / "quantalm.pth"

    vocabulary = json.loads(vocab_file.read_text(encoding="utf-8"))
    tokenizer = QuantaTokenizer(vocabulary)

    checkpoint = torch.load(
        checkpoint_file,
        map_location=DEVICE,
        weights_only=True
    )

    model = QuantaLM(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if not token_ids:
        print("Please enter a valid prompt.")
        return

    input_tensor = torch.tensor(
        [token_ids], dtype=torch.long, device=DEVICE
    )

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=40,
            temperature=0.7
        )

    generated_ids = output[0].tolist()
    new_tokens = generated_ids[len(token_ids):]

    print("\nQuantaLM:", tokenizer.decode(new_tokens))


def main():
    model, tokenizer = load_model()

    print("\n================================")
    print("       QuantaLM Terminal")
    print("================================")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() == "exit":
            print("QuantaLM session ended.")
            break

        if prompt:
            generate_response(model, tokenizer, prompt)


if __name__ == "__main__":
    main()