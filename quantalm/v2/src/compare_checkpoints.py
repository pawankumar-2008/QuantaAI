"""Compare two V2 checkpoints on the same validation data and diagnostic prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_V2_ROOT = Path(__file__).resolve().parents[1]
if __package__:
    from .evaluator import (
        EVALUATION_REPORT_DIR,
        Evaluator,
        GenerationConfig,
        generation_quality,
        sha256_file,
    )
    from .generate_text import generate_sample
    from .workspace_config import v2_config
else:
    sys.path.insert(0, str(_V2_ROOT))
    from src.evaluator import (
        EVALUATION_REPORT_DIR,
        Evaluator,
        GenerationConfig,
        generation_quality,
        sha256_file,
    )
    from src.generate_text import generate_sample
    from src.workspace_config import v2_config


DEFAULT_COMPARISON_DIR = EVALUATION_REPORT_DIR.parent / "checkpoint_comparisons"


def summarize_comparison(
    previous: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    for key in ("split", "evaluation_dataset_sha256", "perplexity_chunk_dataset_sha256", "tokenizer_sha256"):
        if previous.get(key) != candidate.get(key):
            raise ValueError(f"Checkpoint evaluations have incompatible {key} values.")

    previous_perplexity = previous["perplexity"].get("perplexity")
    candidate_perplexity = candidate["perplexity"].get("perplexity")
    delta = (
        candidate_perplexity - previous_perplexity
        if previous_perplexity is not None and candidate_perplexity is not None
        else None
    )
    relative_delta = (
        100.0 * delta / previous_perplexity
        if delta is not None and previous_perplexity
        else None
    )
    return {
        "split": previous["split"],
        "evaluation_dataset_sha256": previous["evaluation_dataset_sha256"],
        "perplexity_chunk_dataset_sha256": previous["perplexity_chunk_dataset_sha256"],
        "tokenizer_sha256": previous["tokenizer_sha256"],
        "previous": {
            "model_label": previous["model_label"],
            "checkpoint_sha256": previous["checkpoint_sha256"],
            "perplexity": previous_perplexity,
            "average_negative_log_likelihood": previous["perplexity"][
                "average_negative_log_likelihood"
            ],
        },
        "candidate": {
            "model_label": candidate["model_label"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "perplexity": candidate_perplexity,
            "average_negative_log_likelihood": candidate["perplexity"][
                "average_negative_log_likelihood"
            ],
        },
        "candidate_minus_previous_perplexity": delta,
        "relative_perplexity_change_percentage": relative_delta,
        "interpretation": (
            "Lower validation perplexity indicates lower next-token NLL on this "
            "specific split; it does not establish factual accuracy or generalization."
        ),
    }


def compare_checkpoints(
    previous_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    *,
    prompt: str,
    output_dir: str | Path = DEFAULT_COMPARISON_DIR,
    device: str = "cpu",
    config: GenerationConfig | None = None,
) -> dict[str, Any]:
    if not prompt.strip():
        raise ValueError("A diagnostic prompt is required.")
    generation_config = config or GenerationConfig()
    previous_path = Path(previous_checkpoint)
    candidate_path = Path(candidate_checkpoint)
    previous_hash = sha256_file(previous_path)
    candidate_hash = sha256_file(candidate_path)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    comparison_id = hashlib.sha256(
        f"{previous_hash}:{candidate_hash}:{prompt_hash}".encode("ascii")
    ).hexdigest()[:12]
    destination_root = Path(output_dir).resolve()
    reports_root = (v2_config.DATA_DIR / "reports").resolve()
    if not destination_root.is_relative_to(reports_root):
        raise ValueError("Comparison reports must remain under quantalm/v2/data/reports/.")
    destination = destination_root / f"checkpoint-comparison-{comparison_id}"
    if destination.exists():
        raise FileExistsError(f"Comparison report already exists: {destination}")

    previous_evaluator = Evaluator(
        checkpoint_path=previous_path,
        device=device,
        generation_config=generation_config,
    )
    candidate_evaluator = Evaluator(
        checkpoint_path=candidate_path,
        device=device,
        generation_config=generation_config,
    )
    if previous_evaluator.tokenizer_hash != candidate_evaluator.tokenizer_hash:
        raise ValueError("Checkpoints use different tokenizer artifacts and cannot be compared.")

    destination.mkdir(parents=True)
    previous_result = previous_evaluator.evaluate(
        "validation",
        benchmark_warmup_runs=0,
        benchmark_measured_runs=1,
        output_dir=destination / "previous-evaluation",
    )
    candidate_result = candidate_evaluator.evaluate(
        "validation",
        benchmark_warmup_runs=0,
        benchmark_measured_runs=1,
        output_dir=destination / "candidate-evaluation",
    )
    comparison = summarize_comparison(previous_result, candidate_result)

    previous_sample = generate_sample(previous_evaluator, prompt, generation_config)
    candidate_sample = generate_sample(candidate_evaluator, prompt, generation_config)
    for evaluator, sample in (
        (previous_evaluator, previous_sample),
        (candidate_evaluator, candidate_sample),
    ):
        token_ids = evaluator.tokenizer.encode(sample["generated_text"])
        sample["quality"] = generation_quality(
            token_ids,
            sample["generated_text"],
            tokenizer=evaluator.tokenizer,
        )

    report = {
        **comparison,
        "comparison_id": comparison_id,
        "prompt": prompt,
        "generation_config": generation_config.__dict__,
        "previous_generation": previous_sample,
        "candidate_generation": candidate_sample,
        "limitations": [
            "Validation contains one source document in the current corpus.",
            "There are no human-verified reference answers or human ratings.",
            "Perplexity and repetition diagnostics do not measure factual accuracy.",
            "The corpus is small and source licenses are unknown.",
        ],
    }
    (destination / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    delta_text = "Unavailable" if report["candidate_minus_previous_perplexity"] is None else (
        f"{report['candidate_minus_previous_perplexity']:.6f}"
    )
    markdown = "\n".join(
        [
            "# QuantaLM V2 Checkpoint Comparison",
            "",
            f"- Comparison ID: `{comparison_id}`",
            f"- Split: `{report['split']}`",
            f"- Previous checkpoint SHA-256: `{previous_hash}`",
            f"- Candidate checkpoint SHA-256: `{candidate_hash}`",
            f"- Tokenizer SHA-256: `{report['tokenizer_sha256']}`",
            f"- Previous validation perplexity: {report['previous']['perplexity']}",
            f"- Candidate validation perplexity: {report['candidate']['perplexity']}",
            f"- Candidate minus previous perplexity: {delta_text}",
            "",
            "## Diagnostic generations",
            "",
            "### Previous checkpoint",
            "",
            report["previous_generation"]["generated_text"] or "(empty output)",
            "",
            "### Candidate checkpoint",
            "",
            report["candidate_generation"]["generated_text"] or "(empty output)",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )
    (destination / "comparison.md").write_text(markdown, encoding="utf-8", newline="\n")
    report["report_directory"] = str(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    try:
        config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            random_seed=args.seed,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        result = compare_checkpoints(
            args.previous,
            args.candidate,
            prompt=args.prompt,
            output_dir=args.output_dir,
            device=args.device,
            config=config,
        )
    except (FileExistsError, FileNotFoundError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
