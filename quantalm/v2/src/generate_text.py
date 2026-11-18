"""Generate text from an explicitly selected QuantaLM V2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_V2_ROOT = Path(__file__).resolve().parents[1]
if __package__:
    from .evaluator import Evaluator, GenerationConfig
    from .generation import generate
    from .workspace_config import v2_config
else:
    sys.path.insert(0, str(_V2_ROOT))
    from src.evaluator import Evaluator, GenerationConfig
    from src.generation import generate
    from src.workspace_config import v2_config


def generate_sample(
    evaluator: Evaluator,
    prompt_text: str,
    config: GenerationConfig,
) -> dict[str, object]:
    if not prompt_text.strip():
        raise ValueError("prompt must contain non-whitespace text.")
    prompt_ids = [
        evaluator.tokenizer.bos_token_id,
        *evaluator.tokenizer.encode(prompt_text),
    ]
    if len(prompt_ids) > evaluator.model.config.context_length:
        prompt_ids = [
            evaluator.tokenizer.bos_token_id,
            *prompt_ids[-(evaluator.model.config.context_length - 1) :],
        ]
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=evaluator.device,
    )
    generated = generate(
        evaluator.model,
        input_ids,
        config.max_new_tokens,
        config.temperature,
        config.top_k,
        eos_token_id=evaluator.tokenizer.eos_token_id,
        seed=config.random_seed,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )
    answer_ids = generated[0, len(prompt_ids) :].detach().cpu().tolist()
    return {
        "model_label": evaluator.model_label,
        "checkpoint": (
            str(evaluator.checkpoint_path)
            if evaluator.checkpoint_path is not None
            else None
        ),
        "prompt": prompt_text,
        "generated_text": evaluator.tokenizer.decode(answer_ids),
        "generated_token_count": len(answer_ids),
        "generation_config": config.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=v2_config.CHECKPOINT_DIR / "best.pt",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--disable-top-k", action="store_true")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=None if args.disable_top_k else args.top_k,
            random_seed=args.seed,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        evaluator = Evaluator(
            checkpoint_path=args.checkpoint,
            device=args.device,
            random_seed=args.seed,
            generation_config=config,
        )
        result = generate_sample(evaluator, args.prompt, config)
    except (FileNotFoundError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
