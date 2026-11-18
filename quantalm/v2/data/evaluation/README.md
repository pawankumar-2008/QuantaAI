# QuantaLM V2 Evaluation Data

`validation.jsonl` and `test.jsonl` contain no questions yet. The local corpus
has no human-verified question/reference pairs, and its source licenses and
authorship are unknown. Existing prompts in `quantalm/v2/evaluation/` are not
treated as ground truth and are not consumed by the evaluator. No question,
reference answer, or label has been fabricated.

The JSONL question schema requires exactly these fields:

```json
{
  "question_id": "review-assigned-id",
  "category": "Computer Science",
  "question": "A source-grounded question awaiting review",
  "reference_answer": "An answer supported by the cited source text",
  "acceptable_key_points": [],
  "source_document_id": "source document ID",
  "source_document_hash": "64-character source SHA-256",
  "source_text_reference": "relative source path or precise local reference",
  "difficulty": "introductory, intermediate, advanced, or another documented level",
  "human_verified": false
}
```

Only use material from the corresponding document-level split: validation
questions must cite validation documents, while test questions must cite held
out test documents and are reserved for an explicitly approved final run.
Reference answers and key points must be grounded in the cited source. Keep
`human_verified` false until a human has reviewed the question and reference.
The evaluator checks source IDs and hashes against the selected split.

Supported categories are Computer Science, Artificial Intelligence,
Mathematics, Science, General Knowledge, and Student Learning. General
Knowledge has no source documents in the inspected corpus. The corpus contains
only one validation and one test document, so category coverage and all
statistical conclusions are severely limited.

Copy `human_review_template.jsonl` to `human_reviews.jsonl` and create one
review record per reviewer and question. Leave `reviewed` false and scores
empty until a person completes the review. The scale is:

| Score | Meaning |
| ---: | --- |
| 1 | Incorrect, irrelevant, incoherent, or does not follow the instruction |
| 2 | Major issues; only a small portion is acceptable |
| 3 | Partially acceptable, with notable omissions or weaknesses |
| 4 | Mostly correct and useful, with minor issues |
| 5 | Fully correct, relevant, coherent, and instruction-following |

Each of factual accuracy, relevance, coherence, and instruction following must
be scored separately. A completed review requires all four integer scores,
an explanation, reviewer ID, and `reviewed: true`. Reported percentages map
score 1 to 0% and score 5 to 100%. The weighted score is unavailable until at
least one complete valid human review exists. Lexical overlap and perplexity
are not substitutes for factual review.
