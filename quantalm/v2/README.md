# QuantaLM V2

QuantaLM V2 is an independent, custom PyTorch decoder-only Transformer. Its
architecture implementation does not load pretrained weights or use a
Hugging Face model. Training is not started or enabled by the model code.
V1 source, tokenizer, and checkpoints remain separate.

## Architecture

The defaults in `config.py` are:

| Setting | Value |
| --- | ---: |
| Vocabulary | 2,048 |
| Context length | 256 tokens |
| Embedding dimension | 128 |
| Transformer blocks | 4 |
| Attention heads | 4 (32 dimensions per head) |
| Feed-forward dimension | 512 |
| Activation / normalization | GELU / pre-LayerNorm |
| Dropout | 0.1 |
| Position encoding | Learned absolute positions |
| Tied token/output embeddings | Yes |
| Unique trainable parameters | 1,088,256 |

Each block applies causal scaled dot-product multi-head self-attention and a
two-layer GELU feed-forward network, each with a residual connection and
pre-LayerNorm. The model applies learned token and position embeddings, a final
LayerNorm, and a vocabulary projection tied to the token embedding weights.
Attention masks all positions after each query token.

`ModelConfig` exposes all architecture dimensions and dropout as validated
parameters. The model accepts `input_ids` shaped `[batch_size, sequence_length]`
and returns logits shaped
`[batch_size, sequence_length, vocabulary_size]`. The maximum sequence length
is the configured context length.

## Forward pass

Run from `quantalm/v2/`:

```python
import torch
from src.model import QuantaLMV2

model = QuantaLMV2()
input_ids = torch.randint(0, model.config.vocab_size, (2, 32))
logits = model(input_ids)
print(logits.shape)  # torch.Size([2, 32, 2048])
```

The model runs on CPU by default. To use a CUDA GPU when available, move both
the model and input tensor to the same CUDA device.

## Text generation

The independent V2 byte-level BPE tokenizer is implemented in
`src/bpe_tokenizer.py`. It stores its 2,048-token vocabulary, merges, and
configuration under `tokenizer/`; IDs 256–259 are pad, unknown, BOS, and EOS.
The EOS ID is 259. The V1 tokenizer implementation and artifacts are not used
or modified.

For a fresh V2 tokenizer artifact set, run these commands from `quantalm/v2/` to
fit only on training documents, encode all splits, and create context-sized
chunks:

```powershell
python src/train_tokenizer.py
python src/tokenize_dataset.py
python src/chunk_dataset.py
```

The tokenizer trainer verifies manifest membership and clean-text hashes and
records the training split hash, document provenance, seed, and artifact
hashes. It never reads validation or test text for fitting. Existing tokenizer
artifacts are not overwritten unless `--overwrite` is explicitly supplied. This
workspace already has fitted tokenizer artifacts and dependent experiment
checkpoints; reuse them rather than running this tokenizer-training command.
No new dataset was added because no approved source or license information was
provided. The local corpus has no General Knowledge documents and is heavily
imbalanced. See `data/reports/dataset_expansion_report.json`.

## Dataset provenance and approved-source ingestion

The document-level read-only provenance audit is stored in
`data/reports/dataset_provenance_report.json`. Rebuild it from the V2 source
snapshot with:

```powershell
python src\dataset_ingestion.py audit
```

The approved-source ingestion tool accepts UTF-8 `.txt` documents referenced
by a JSONL manifest. Every manifest row must include source identity and
location, owner, version and retrieval date, category, source URL, license
identifier and URL, explicit license verification, explicit license-scope and
third-party-content review, reviewer, verification date, and evidence. The default
allowlist is `CC0-1.0`, `CC-BY-4.0`, and `PSF-2.0`; this is a metadata gate, not
an automatic legal verification. Unknown, non-commercial, or otherwise
unapproved licenses are rejected.

Each JSONL object must contain `source_id`, `source_name`, `source_owner`,
`original_location`, `source_url`, `source_version`, `source_retrieved_at`,
`category`, `license_id`, `license_url`, `license_verified: true`,
`license_scope_reviewed: true`, `third_party_content_reviewed: true`,
`license_verified_by`, `license_verified_at`, `license_evidence`, and a
`text_path` relative to `--input-root`. Use the canonical license URLs checked
by the tool: Creative Commons CC0 1.0, Creative Commons CC BY 4.0, or the
official Python PSF-2.0 license page. The reviewer must verify the exact
document/version and all embedded third-party material; a claimed boolean by
itself is not proof of permission.

Example after independently verifying and preparing a manifest:

```powershell
python src\dataset_ingestion.py ingest `
  --manifest D:\approved-material\manifest.jsonl `
  --input-root D:\approved-material `
  --run-id review-batch-001
```

The pipeline writes documents, existing-V2-tokenizer outputs, document-aware
chunks, provenance, quality, and leakage reports under
`data/ingestion_runs/<run-id>/`. It refuses to overwrite a run, removes exact
normalized duplicate copies only after listing them in its provenance report,
keeps near-duplicate/shared-content groups in a single split, and does not
alter the current `data/splits/`, `data/chunks/`, tokenizer, checkpoints, or
training state. A staged run is not automatically approved or added to the
training dataset. See `data/reports/dataset_expansion_plan.json`; no external
source was downloaded or ingested for this phase.

Token JSONL files keep whole documents; they are not silently truncated. The
current documents exceed the 256-token model context. Create context-sized
document-aware examples before any future model training:

```powershell
python src/chunk_dataset.py
```

The chunker uses 257-token source windows so shifted input and target arrays
can each contain 256 positions. Its configured window overlap is 32 tokens
with a 225-token stride. It moves a short final window backward when needed,
which can increase the final overlap. Every token remains represented, BOS
stays at the document start, EOS remains the final target, and documents are
never combined. Overlap intentionally repeats some next-token targets.
Right-padding is marked out of loss by `loss_mask`; any future training loop
must apply that mask. The current model `forward` consumes `input_ids` only;
the chunk attention mask is metadata, and causal attention makes right-padding
after a valid prefix inert. Each chunk records the raw source-file SHA-256 as
`source_document_hash`, the cleaned-text hash separately, and a hash of the
tokenizer artifacts.

Chunk outputs and their deterministic statistics are written to
`data/chunks/{train,validation,test}/chunks.jsonl` and `data/reports/`.

```python
import torch

from src.bpe_tokenizer import EOS_ID, QuantaBPETokenizer
from src.generation import generate
from src.model import QuantaLMV2

tokenizer = QuantaBPETokenizer.load()
model = QuantaLMV2().eval()
prompt_ids = torch.tensor(
    [tokenizer.encode("Explain a binary search.", add_special_tokens=True)],
    dtype=torch.long,
)
result = generate(
    model,
    prompt_ids,
    max_new_tokens=40,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.15,
    eos_token_id=EOS_ID,
    seed=123,
)
print(tokenizer.decode(result[0].tolist()))
```

Generation truncates model input to the most recent context window and does not
use a KV cache. A provided seed makes sampling reproducible on the same device
and runtime.

## Checkpoints

`src.checkpoint.save_checkpoint()` writes only within `v2/checkpoints/`, stores
the state dict, full model configuration, vocabulary and context sizes,
architecture version, and UTC creation timestamp. Existing files are not
overwritten unless `overwrite=True` is explicitly passed. Loading uses
`torch.load(..., weights_only=True)` and rejects configuration or architecture
version mismatches.

```python
from src.checkpoint import load_checkpoint, save_checkpoint
from src.model import QuantaLMV2

model = QuantaLMV2()
path = save_checkpoint(model)  # v2/checkpoints/quantalm_v2.pt
metadata = load_checkpoint(QuantaLMV2(model.config), path)
```

Do not copy or load V1 checkpoints into V2. The root `best.pt`, `latest.pt`,
and `resume.pt` files are created only by an explicitly confirmed training run;
existing controlled-experiment checkpoints under `checkpoints/experiments/`
are preserved separately.

## Training pipeline (not yet run)

The CPU-first training implementation consumes only the existing train and
validation chunk files. The test split is not exposed by the training loader
or evaluated by the trainer. The loss is token-level cross entropy over the
already aligned targets, normalized by valid `loss_mask` positions; inputs and
targets are not shifted again. AdamW, warmup/cosine learning-rate scheduling,
gradient clipping, best-validation checkpoint selection, and epoch-based early
stopping are implemented without changing the model or tokenizer.

Defaults are centralized in `src/training_config.py` (batch size 1, learning
rate 0.0001, weight decay 0.01, 10 epochs, seed 20261106, warmup 10 steps,
evaluation every 50 steps, checkpoint every 100 steps, and patience 3).
Training checkpoints are stored under `checkpoints/`; epoch metrics are written
to `data/reports/training/`. Resuming validates model, training, tokenizer, and
dataset-manifest compatibility. The small, imbalanced corpus presents a
substantial overfitting risk; decreasing training loss alone is not evidence
of generalization.

Training is a manual operation and requires `--confirm-training`. It is not
started by dataset preparation, import, backend startup, or chat requests. CPU
is the default; command-line overrides are checked against the prepared context
and a 2 GiB estimated-memory ceiling. The current corpus is only 13 unique
documents and 76,734 tokens, so training should be treated as a pipeline
experiment with substantial overfitting risk. The source licenses are unknown.

Run all commands from `D:\QuantaAI\quantalm\v2` after activating the project
environment:

```powershell
python src\prepare_dataset.py --stats-only
$runId = "run-20261108-001"
python src\trainer.py --confirm-training --run-id $runId `
  --device cpu --batch-size 1 --epochs 10
```

Resume from the last saved optimizer/scheduler/RNG state with:

```powershell
python src\trainer.py --confirm-training --resume resume `
  --checkpoint-dir "checkpoints\$runId" --device cpu
```

Training also accepts `--learning-rate`, `--weight-decay`, `--warmup-steps`,
`--gradient-clip-norm`, `--evaluation-interval`, `--checkpoint-interval`,
`--early-stopping-patience`, `--seed`, `--embedding-dim`, `--num-layers`,
`--num-heads`, `--feed-forward-dim`, and `--dropout`. `--context-length` must
match the chunk dataset; rebuild chunks before changing it. The training engine
has only been checked with unit tests and tiny synthetic or three-step
experiments; no full-corpus training has been performed.

Generate a sample from the selected best checkpoint:

```powershell
python -m src.generate_text --checkpoint "checkpoints\$runId\best.pt" `
  --prompt "A useful way to reason about an unfamiliar problem is" `
  --max-new-tokens 40 `
  --temperature 0.8 --top-k 40 --top-p 0.9 --repetition-penalty 1.15
```

The current tokenizer artifacts are already trained and hash-bound to existing
checkpoints. Do not use `python src\train_tokenizer.py --overwrite` for this
model; a tokenizer change requires a separate compatible model/checkpoint
version. The tokenization pipeline rejects hash mismatches.

## Evaluation and benchmarking

The evaluation framework is implemented in `src/evaluator.py`. It supports
the current untrained model as a clearly labelled baseline and safe V2 model
or training checkpoints. Evaluation questions live under
`data/evaluation/{validation,test}.jsonl`; both are currently empty because
the corpus has no human-verified reference questions and source licensing is
unknown. Existing examples under the legacy `evaluation/` directory are not
treated as ground truth. Review the schema and rubric in
`data/evaluation/README.md` before adding source-grounded questions.

Run validation evaluation from `quantalm/v2/`:

```powershell
python src/evaluator.py --split validation --device cpu
```

Evaluate a trained checkpoint with the configurable sampler:

```powershell
python src\evaluator.py --split validation --checkpoint "checkpoints\$runId\best.pt" `
  --device cpu --max-new-tokens 64 --temperature 0.8 --top-k 40 `
  --top-p 0.9 --repetition-penalty 1.15
```

Compare a previous checkpoint with the candidate checkpoint on the same
validation chunks and diagnostic prompt:

```powershell
python -m src.compare_checkpoints `
  --previous "checkpoints\previous-run\best.pt" `
  --candidate "checkpoints\$runId\best.pt" `
  --prompt "A useful way to reason about an unfamiliar problem is"
```

The command saves side-by-side evaluations, diagnostic generations, validation
perplexity deltas, and a Markdown/JSON report under
`data/reports/checkpoint_comparisons/`. It refuses to compare different
tokenizers or evaluation datasets and never reads the held-out test split.

The command writes JSON, CSV, JSONL, and Markdown reports under
`data/reports/evaluation/`. It measures token-weighted perplexity on the
existing validation chunks, while exact match and lexical precision/recall/F1
are unavailable until human-verified references exist. Lexical overlap and
perplexity do not establish factual accuracy. Generation uses the V2 tokenizer
and generation function with an explicit seed. A separate single-BOS
performance probe is reported only as a throughput/latency measurement, not
as an answer-quality sample.

Held-out test evaluation is disabled by default. Only after approving a final
evaluation run, enable it explicitly:

```powershell
python src/evaluator.py --split test --allow-test-evaluation --device cpu
```

Do not use the test split during development. The corpus contains only one
validation and one held-out test document, has severe category imbalance, and
has unknown source licenses. No full-corpus-trained checkpoint currently
exists, so the untrained baseline and existing three-step controlled
experiment cannot be described as production-model performance. CUDA is
reported as unverified unless it is available and actually selected for a run.

## Phase 3.6 controlled experiment

Source licenses are unknown in the existing corpus. No external documents are
added; the bounded experiment is explicitly limited to this existing local
corpus, and does not establish that its contents are licensed for distribution
or reuse. The reproducible run configuration is
`experiments/phase_3_6_controlled_cpu.json`. It caps training at three CPU
optimizer steps, evaluates validation loss after every step, records
train/validation separation, uses no test examples, and saves checkpoints into
a run-specific directory without overwriting existing checkpoints.

To run the bounded experiment explicitly from `quantalm/v2/`:

```powershell
python -m src.controlled_experiment --confirm-controlled-experiment
```

The runner refuses to exceed 10 steps and refuses to reuse an existing
experiment ID. It compares the seeded random initialization with the best
validation checkpoint using validation perplexity and a fixed diagnostic
generation prompt. Generated text is not a correctness test. Reports and
per-step metrics are written under
`data/reports/experiments/<experiment_id>/`, with checkpoint artifacts under
`checkpoints/experiments/<experiment_id>/`. This is a pipeline experiment, not
full-corpus training or evidence of intelligence or generalization. The small
corpus, one-document validation set, and unresolved licenses preclude such
claims.

## Dataset pipeline and statistics

The reusable preparation pipeline scans multiple local `.txt` files under
`quantalm/data/raw/`, cleans text, removes exact/normalized duplicates, groups
near-duplicates to avoid leakage, and writes V2-only splits/reports. Its target
is a 90/10 train-validation ratio with an additional approximately 10% held-out
test partition; whole-document leakage groups and small sample counts can make
the realized document ratios approximate. The current artifacts contain 13
unique documents: 11 train, 1 validation, and 1 test. The stats-only command
does not modify data:

```powershell
python src\prepare_dataset.py --stats-only
python src\prepare_dataset.py
```

The second command regenerates splits and reports. If the training split
changes, tokenizer hashes make old checkpoints incompatible; preserve them and
establish a separate tokenizer/model version before fitting a new vocabulary.
The current tokenizer has 2,048 IDs (1,806 observed in this corpus), and its
special-token IDs and artifact hashes are validated during loading.

## Dataset acquisition and releases

The separate offline acquisition tool stores registered local sources below
`data/raw/<category>/<dataset_id>/`, metadata under `data/metadata/`, working
artifacts below `data/processed/`, and immutable releases under
`data/releases/<version>/`. It never downloads data. Registration copies the
source file and records its SHA-256; it refuses to replace an existing dataset
ID. TXT metadata is attested per document. CSV/JSON/JSONL records must map each
document's own source, license, attribution, verification, category, language,
and text fields. Unknown, non-commercial, and no-derivatives licenses are
rejected. CC BY-SA records require a separate documented share-alike review.

The official source review is recorded in
`data/metadata/source_candidates.json`. Project Gutenberg requires title- and
jurisdiction-specific rights review and prohibits automated bulk access to its
main site; Wikimedia and Stack Exchange require item/revision attribution and
may impose share-alike obligations; the inspected OpenStax College Algebra
title is CC BY-NC-SA and is not accepted by the default policy; Python
documentation is PSF-2.0, with separately licensed examples and embedded
third-party material. No external source was downloaded or registered for this
change. These notes are not legal advice.

From `D:\QuantaAI\quantalm\v2`, inspect and register a locally acquired file.
First create a registration JSON based on
`data/metadata/registration.example.json`, replacing the sample source and
item-level fields. For TXT, put the item's full provenance/license attestation
in `document_metadata`; for structured formats the `record_fields` map points
to the corresponding per-record values.

```powershell
python -m src.dataset_release inspect
python -m src.dataset_release register `
  --source-file D:\approved-data\documents.jsonl `
  --metadata data\metadata\my-dataset.json
python -m src.dataset_release validate --dataset-id my-dataset-v1
python -m src.dataset_release import --dataset-id my-dataset-v1
python -m src.dataset_release clean --dataset-id my-dataset-v1 --minimum-characters 80
```

Combine approved cleaned datasets, inspect duplicate and category statistics,
then create a versioned 90/5/5 release:

```powershell
python -m src.dataset_release dedupe `
  --dataset-ids my-dataset-v1 another-approved-set `
  --dedupe-id candidate-2026-11
python -m src.dataset_release stats --dedupe-id candidate-2026-11
python -m src.dataset_release release `
  --dedupe-id candidate-2026-11 --version v1.0.0
python -m src.dataset_release stats --version v1.0.0
```

The release report contains actual document/character/word/token counts by
category, license distribution, cleaning and duplicate removals, leakage-aware
split statistics, source/registration checksums, split checksums, and the fixed
V2 tokenizer identity used for counts. Category figures are measured, never
filled to reach collection targets. A release version is immutable; choose a
new version for changed inputs. Existing `data/splits/`, tokenizer artifacts,
and checkpoints are not modified. Dataset curation does not train a tokenizer
or model.

## FastAPI Offline integration

The backend defaults to the existing V1 QuantaLM and retains its API contract.
To select a trained V2 checkpoint, set these variables before starting FastAPI.
The selected checkpoint must contain a completed optimizer step and a matching
tokenizer hash:

```powershell
$env:QUANTAAI_QUANTALM_VERSION = "v2"
$env:QUANTAAI_V2_CHECKPOINT = "D:\QuantaAI\quantalm\v2\checkpoints\run-20261108-001\best.pt"
python -m uvicorn backend.main:app --reload
```

Online provider order is unchanged. This option affects local QuantaLM
inference only; it does not start training or make network requests.

## Tests

From `quantalm/v2/`, run:

```powershell
python -m unittest discover -s tests -v
```

Tests cover tensor shapes, context validation, causality, gradients, CPU
execution, generation controls and EOS stopping, checkpoint safety and
compatibility, tokenizer fitting provenance, fresh-process tokenizer loading,
Unicode/code/math/whitespace round trips, token-ID/EOS consistency, chunk
coverage and padding masks, train/validation loader isolation, loss masking,
finite gradients, deterministic initialization, optimizer/scheduler updates,
safe checkpoint resume, tiny synthetic CPU training mechanics, evaluation
metrics, masked perplexity, human-review scoring, seeded generation,
performance benchmarking, report creation, and the held-out test safeguard.
The dataset release tests also exercise TXT/CSV/JSON/JSONL ingestion,
per-document license validation, HTML/Unicode/language cleaning, exact and
near-duplicate handling, category statistics, 90/5/5 splitting, checksums,
reproducibility, and protection of existing splits/tokenizer/checkpoints.
No full-corpus training or test-split evaluation is performed by the test
suite. CUDA execution is tested only when CUDA is available.
