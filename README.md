# Najd Benchmark

[Evaluation contract v2](docs/evaluation-contract-v2.md) · [Contributing](CONTRIBUTING.md)

Source repository for task definitions, scoring contracts, baselines, and reproducible reports. The benchmark consumes versioned, approved datasets from `najdresearch/datasets`; the Arena application handles submissions, managed worker orchestration, review, and presentation. Shared endpoint execution belongs in the benchmark package under the v2 contract.

The [platform contract](docs/platform-contract.md) records the first text score, repository boundaries, submission policy, and release gates. Classification, tool decisions, computer use, embeddings, reranking, OCR, STT, and TTS remain separate scorecards with their own scoring methods.

The first executable slice is [ArabicMMLU answer-key task v1](docs/tasks/arabic-mmlu-key-v1.md). Arena integration and scorer validation determine leaderboard readiness; the dataset's `review_status` field is informational for now.
