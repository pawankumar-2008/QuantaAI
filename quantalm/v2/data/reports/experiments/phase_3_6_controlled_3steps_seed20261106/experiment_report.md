# Controlled Training Experiment: phase_3_6_controlled_3steps_seed20261106

- **Status:** controlled_experiment_complete
- **Scope:** Bounded three-step CPU training experiment on the existing V2 training split; not a full-corpus run.
- **Model label:** three_step_experimental_checkpoint_not_trained_model_claim
- **Steps:** 3
- **Device:** cpu
- **Training time:** 0.850 seconds
- **Total run time:** 4.002 seconds
- **Process peak working set:** 383221760 bytes (Windows process peak working set)
- **Experimental best checkpoint:** `checkpoints\experiments\phase_3_6_controlled_3steps_seed20261106\best.pt`
- **Checkpoint SHA-256:** `a98ec4c854b8861e4a47ea38d7b68eb1bbc23be377f8401880b71ba4025e4d01`

## Dataset audit

- Training documents/chunks: 11 / 285
- Validation documents/chunks: 1 / 38
- Corpus deduplication: 2 exact duplicates removed; no normalized or near-duplicate documents removed; 78 shared substantive lines and 8 shared passages were leakage-grouped.
- Validation data was used only for loss monitoring and checkpoint selection, never for gradient updates.
- Held-out test split was not evaluated.
- Split leakage audit: passed; source licensing remains unknown.
- No new or external data was used.

## Loss and perplexity

- Baseline validation perplexity: 2.01935e+38
- Experimental checkpoint validation perplexity: 1.69935e+37
- Perplexity delta: -1.84941e+38
- Baseline validation NLL: 88.201008
- Experimental validation NLL: 85.725892

| Step | Training loss | Validation loss | Learning rate | Gradient norm | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 81.853325 | 87.388800 | 0.0001 | 23.574890 | 0.291 |
| 2 | 83.151161 | 86.569401 | 0.0001 | 24.969391 | 0.225 |
| 3 | 84.164520 | 85.725892 | 0.0001 | 30.914776 | 0.335 |

## Generation probe (not a correctness test)

Prompt: Explain the relationship between an algorithm and a program.

Untrained baseline output:

```text
................................
```

Experimental checkpoint output:

```text
................................
```

- Repeated-token ratio: 0.9688 → 0.9688
- Repeated n-gram ratio: 0.9667 → 0.9667
- EOS terminated: False → False
- Outputs are diagnostic samples only; no human quality or factuality score is available.

## Limitations

- All source-document license statuses are unknown; this experiment adds no external data and does not establish reuse authorization.
- The 11-document training corpus and one-document validation set are too small and imbalanced for meaningful generalization claims.
- A three-step run is only an optimization-pipeline experiment, not evidence that the model is capable or intelligent.
- There are no human-verified evaluation questions; generation comparison is a fixed diagnostic prompt, not correctness evaluation.
- Validation perplexity reuses one document and overlapping chunks; it is monitoring only and can be overfit.
- Peak process working set is process-wide and not an isolated measurement of model-only memory.
- No held-out test evaluation, full-corpus training, or GPU execution was performed.

No full-corpus training or test evaluation was run. Obtain data-license clarification and human-reviewed validation examples before designing a larger experiment.
