# ArabicMMLU answer-key task v1

This is the first executable Najd text task. It checks whether a model picks the right option for an Arabic question. It is an **experimental task score**, not yet a certified leaderboard score.

| Field | Frozen value or rule |
|---|---|
| Dataset | [`najdresearch/najd-benchmark`](https://huggingface.co/datasets/najdresearch/najd-benchmark/tree/cb30c1c9e46c62f691380c3269885cdb8f22f52b/datasets/najd-benchmark/2026.09.14) `2026.09.14`, revision `cb30c1c9e46c62f691380c3269885cdb8f22f52b` |
| Selection | Certified `arabicmmlu` source rows only; 104 cases; sorted ID digest `f65459ada9049628c8c676a8c7d51a2b16736cdab67743a80a64a05b0ea19b41` |
| Input | Original question, followed by every option in A–E order |
| Output | One uppercase Latin key, or JSON containing only `answerKey` |
| Score | Exact key match, 0 or 1; invalid, missing, or out-of-range output scores zero |
| Aggregate | Mean across all 104 cases; failures stay in the denominator |
| Inference | One response per case, temperature 0; record model ID, max output tokens, and other settings |
| Abstention | Invalid or missing answer scores zero and is reported separately |

The case answers are A: 33, B: 29, C: 23, D: 18, E: 1. Always predicting A scores **33/104 = 31.73%** on this subset. A per-case uniform guess has an expected score of **29.39%** because cases have two to five options. These are sanity baselines, not useful model comparisons.

The scoring package validates the frozen case IDs and includes the options in the prompt. The current Arena development runtime does not include those options for this source, so its earlier ArabicMMLU outputs must not be compared with this task. Arena's current general-choice parser also accepts a first-line answer followed by explanation or a JSON object with extra fields; this version deliberately requires the stated output contract. These differences must be visible in any migration report.

To score responses locally, save the pinned release's `cases.jsonl` and a response JSONL with `case_id` and `output` per line, then run:

```bash
uv run najd-benchmark --cases /path/to/cases.jsonl --responses /path/to/responses.jsonl
```

The CLI checks the entire certified case file's SHA-256 before scoring. Missing responses score zero and reduce coverage. Duplicate or unknown IDs fail the run. The report is a task score, not an Arena publication decision.

## Evidence and remaining gate

Unit tests cover correct keys, wrong keys, malformed answers, extra text, out-of-range options, and prompt rendering. The scorer accepted the reference answer on all 104 pinned cases. A synthetic response comparison with Arena matched all 104 exact-answer and wrong-answer scores, and diverged on all 104 examples when the correct key was followed by extra text, lowercase, or JSON with extra fields. This is an intended protocol change, so results need a new task version.

The published cases have `review_status: not_reviewed`. Before this task can count toward a certified Najd Text score, an independent Arabic reviewer should inspect a preregistered sample for label correctness, option clarity, and source leakage. Record every reviewed case ID, verdict, and correction separately from the frozen release. If label errors exceed the prespecified tolerance, revise the dataset and task version instead of silently editing the score. This public subset cannot serve as a hidden holdout.
