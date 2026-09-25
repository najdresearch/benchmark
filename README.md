# Najd Benchmark

Source repository for task definitions, scoring contracts, baselines, and reproducible reports. The benchmark consumes versioned, approved datasets from `najdresearch/datasets`; the Arena application handles submissions, execution, review, and presentation.

The [platform contract](docs/platform-contract.md) records the first text score, repository boundaries, submission policy, and release gates. Classification, tool decisions, computer use, embeddings, reranking, OCR, STT, and TTS remain separate scorecards with their own scoring methods.

The first executable slice is [ArabicMMLU answer-key task v1](docs/tasks/arabic-mmlu-key-v1.md). Arena integration and scorer validation determine leaderboard readiness; the dataset's `review_status` field is informational for now.
