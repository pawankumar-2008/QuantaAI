# QuantaLM V2 Evaluation Report

- **Evaluated:** 2026-11-05T22:08:52+00:00
- **Split:** validation
- **Model:** untrained_baseline
- **Checkpoint SHA-256:** None (untrained baseline)
- **Model-state SHA-256:** 52a55379783f610b55e56064cd8b5e05bbe8042fd259ebaa1ad42a829b8d0943
- **Tokenizer SHA-256:** 91b1a25afa1bc56ce8c6456a1daa7f65ebfe3df177ee1ba66b3a966ddf2c63a0
- **Evaluation dataset SHA-256:** e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- **Device:** cpu
- **Questions:** 0; non-empty answers: 0
- **Model configuration:** `{"architecture_version": "2.0", "context_length": 256, "dropout": 0.1, "embedding_dim": 128, "feed_forward_dim": 512, "num_heads": 4, "num_layers": 4, "vocab_size": 2048}`

## Metrics

| Metric | Result |
| --- | ---: |
| Exact match | Unavailable |
| Token precision (macro) | Unavailable |
| Token recall (macro) | Unavailable |
| Token F1 (macro) | Unavailable |
| Perplexity | 2.0193483000083173e+38 |
| Perplexity target tokens | 9610 |
| Mean answer length | Unavailable |
| Repeated n-gram ratio | Unavailable |
| Empty-response rate | Unavailable |
| EOS termination rate | Unavailable |
| Human-reviewed questions | 0 |
| Overall human-quality score | Unavailable |

The human-quality score uses factual accuracy 35%, relevance 25%, coherence 15%, instruction following 15%, and repetition quality 10%. Weights are renormalized over available components, but the score remains unavailable when no valid human review exists.

Token overlap and perplexity are lexical/modeling measures; neither establishes semantic correctness or factual accuracy.

## Performance

- Model loading time: 0.031991 seconds.
- First-token latency (one-token generate call): 0.009101 seconds.
- Average response latency: 0.101356 seconds.
- Generated tokens per second: 78.929895.
- Peak CPU memory: 245424128 bytes (Windows process peak working set).
- Warm-up runs: 1; measured runs: 3.
- Performance probe uses a single BOS token and is not an evaluation question or answer.

## Limitations

- No human-verified evaluation questions are currently available; reference-based metrics are unavailable.
- The local corpus has 13 unique documents, with only one validation and one held-out test document.
- The dataset quality report marks source licenses as unknown and does not establish accuracy or reuse rights.
- General Knowledge has no source documents; category coverage is incomplete and imbalanced.
- The untrained baseline is randomly initialized and its generations do not represent trained QuantaLM performance.
- Lexical overlap and perplexity do not establish semantic correctness or factual accuracy.
- The performance probe uses a single BOS token and is not an answer-quality evaluation.
- CUDA was unavailable; GPU execution and GPU performance are unverified.

## Skipped evaluations

- reference_metrics: No human-verified reference answers are present.
- overall_human_quality_score: No valid completed human reviews are present.

## Reproducibility

Generation uses the configured random seed and a stable question-ID-derived seed. Exact-match normalization uses Unicode NFKC, case-folding, and whitespace normalization. Token precision/recall/F1 use multiset word-and-punctuation tokens and macro-average per answer.

Held-out test evaluation is disabled unless explicitly enabled for a final evaluation run. No test split was evaluated during this development phase.
