# Najd Benchmark

Versioned task definitions, scoring code, baselines and reproducible reports for Arabic and Saudi AI evaluation. This package consumes datasets from Hugging Face. Arena owns managed jobs and public presentation.

## What works today

| Capability | Status |
|---|---|
| ArabicMMLU answer-key task | Implemented: validates and scores the frozen 104-case subset |
| Score saved responses with the CLI | Implemented; exact dataset hash required |
| Local/remote OpenAI-compatible model execution | Planned shared runner; currently implemented in Arena's legacy runtime |
| Arabic customer-support version comparison | Next task pack; not implemented |
| General multi-task scorecards and reports | Contract defined; remaining implementations planned |

## Run the scorer

```sh
uv sync --extra dev
uv run pytest
uv run najd-benchmark --cases /path/to/cases.jsonl --responses /path/to/responses.jsonl
```

Download `datasets/najd-benchmark/2026.09.27/cases.jsonl` from [this immutable dataset revision](https://huggingface.co/datasets/najdresearch/najd-benchmark/tree/3fa471f6c8ed2ebc37a8b60d40c88170d682b569). Each response line contains `case_id` and `output`. See the [task guide](docs/tasks/arabic-mmlu-key-v1.md) for formats, baselines and metrics.

Current case-file SHA-256: `149e6815a29434a462e9d1f8f3ccbb21a142c79848947ad20c19b9639c72c081`.
This revision removes review annotations without changing case content. The 104-case selection and scorer are unchanged. Missing/invalid outputs score zero; unknown or duplicate response IDs fail validation. Annotations do not control eligibility.

## Repository boundaries

| Repository | Owns |
|---|---|
| [datasets](https://github.com/najdresearch/datasets) | Source collection, cleaning, normalization, generation and pinned data releases |
| benchmark | Task definitions, shared execution/scoring package and local reports |
| [najd-arena](https://github.com/najdresearch/najd-arena) | Authentication, quotas, orchestration, evidence, publication and web results |

Local CLI reports are for development and cannot be uploaded as Arena-publishable results. Public Arena reporting requires a website-managed run, evidence checks and the applicable publication approvals. Moving the remaining execution logic into this package is planned; no duplicate scoring implementation should be added to Arena.

## Documentation

- [Evaluation contract v2](docs/evaluation-contract-v2.md): current responsibilities, result records, metric interpretation and milestones.
- [ArabicMMLU task](docs/tasks/arabic-mmlu-key-v1.md): executable contract and how to interpret its score.
- [Historical platform contract](docs/platform-contract.md): original text protocol; ownership/publication rules superseded by v2.
- [Contributing](CONTRIBUTING.md): fixtures, versioning and evidence required for changes.

## Next milestone

A reviewed task definition, baseline and deterministic scorer for the internal Arabic customer-support pilot, followed by an OpenAI-compatible runner and parity tests with Arena's managed worker. Preserve prompts, thinking settings and harnesses as separate configurations.

## Unified dataset release

The current pin contains all 6,089 cases without audit-status or review-status fields. The ArabicMMLU task still selects the same 104 case IDs and uses the same scoring contract. This package does not yet implement a full-suite endpoint runner. Previously generated reports retain their original dataset revisions.

## File-based tasks and answer corrections

The package provides a bounded virtual filesystem for six public fixture tasks and a
strict scorer for four corrected Absher items. See [the protocol and scoring guide](docs/fixture-tasks.md).
Arena uses the same implementation for local runs and hosted workers; tool transcripts
and output files are retained as evidence. This harness condition is distinct from raw
single-turn inference and Pi.

## Shared contracts (Step 2)

[Contracts v1](docs/shared-contracts-v1.md) define dataset manifests, task packs and
private result bundles. `najd-contract` validates them and runs the small Arabic support
routing example against local or remote OpenAI-compatible endpoints. Arena has an
admin-only preview for these bundles; CLI results cannot become public Arena results.
