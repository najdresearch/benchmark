# Natural-development quality and load diagnostic

This experiment compares decisions on the same 216 public draft cases, then checks how each endpoint behaves with more simultaneous clients. It is not an enterprise readiness certification or an Arena leaderboard result.

## Frozen inputs and scoring

Dataset: https://huggingface.co/datasets/najdresearch/system-one/tree/84bf0a30ced090e2b552beafa5f0f3c9a3665c0c/natural-development . Cases SHA-256: `48ce8ead1d739af3749425c3b2fd4be3e4a9bfa298b099729c4dc17e61b83694`.

There are 72 scenario families, each expressed in English, MSA and Saudi Arabic. These are development examples with draft labels and fictional policies. All systems receive state and typed questions; gold answers and metadata are excluded. Strict typed scoring, including invalid outputs, uses `decision-dev-v0.1`. Models may select abstention/clarification only when available in the task's output contract. No tuning or model changes occur between concurrency lanes.

| Setting | Self-hosted lane | Hosted lane |
|---|---|---|
| Systems | Pinned GLiNER2.5-Decide, Laya English, Laya multilingual | Jev 1.13 through OpenRouter Decisions API |
| Hardware/client | One A100 80GB pod, loopback client | Same Mac HTTP client for all hosted lanes; provider hardware unknown |
| Precision | FP32, Laya autocast disabled | Provider controlled |
| Concurrency | 1, then 4, then 16 | 1, then 4, then 16 |
| Requests per lane | 216 cases × 3 repeats = 648 | 216 cases × 3 repeats = 648 |
| Warmup | Three per model before lanes | One before each lane |
| Retries | None | None |
| Request timeout | 20 seconds | 20 seconds |
| Scheduling budget per lane | 180 seconds | 600 seconds |
| Error stop | Stop scheduling after >20% errors once 20 requests complete | Same |
| Queue/batching | Serialized prediction lock, backlog 128, no dynamic batching | Provider controlled |
| Startup | Process-to-ready with cached weights, max 90 seconds | First request only; provider cold start unknown |

Pending requests drain after scheduling stops. Missing requests remain in the quality denominator. Timing covers full response receipt and parsing, not time to first token. Quantiles include successful requests only, with errors reported separately. Throughput is successful completions divided by lane wall time, including client scheduling overhead. Closed-loop load with new HTTP connections is client- and adapter-dependent, not maximum backend capacity.

Three repeats are not three independent datasets. Report 216 unique cases and 72 related families; do not treat 648 samples as independent quality evidence. Observed p99 has only about six upper-tail observations per lane and is exploratory. Fixed order, cache effects and provider queueing are uncontrolled. One run per load setting cannot establish an SLA or statistical ranking. Public data may be contaminated; held-out evaluation and independent label review remain required before publication claims.

## Commands

Run clients on the laptop; run GPU worker only on rented hardware.

```sh
PYTHONPATH=src python scripts/run_hosted_decisions.py \
  --pack dataset/natural-development --output results/jev-c1 \
  --revision 84bf0a30ced090e2b552beafa5f0f3c9a3665c0c \
  --concurrency 1 --repeats 3

PYTHONPATH=src USE_TF=0 python scripts/remote_decision_load.py \
  --pack dataset/natural-development --output results \
  --revision 84bf0a30ced090e2b552beafa5f0f3c9a3665c0c
```

The hosted runner reads `OPENROUTER_API_KEY` from the environment or an explicitly supplied credential file. Never place keys in reports or command arguments. Use fresh output paths. See `remote-cpu-gpu-smoke.md` for dependency installation and same-host CPU/GPU comparisons. Provisioning is external: arm a cleanup watchdog, download evidence, verify its hash and delete the pod.

## Label review found during this run

B04 says an invoice matches, while policy requires a matching purchase order. All three language variants require review. Keep the frozen headline score unchanged. A separately labeled post-hoc sensitivity analysis may exclude the entire B04 family for every model, never only its failing variants. The review is tracked in the datasets repository at `reviews/system-one-b04-ambiguity.json`. A corrected dataset must get a new revision and rerun all systems.

## Evidence

Local ignored artifacts: `build/jev-natural-216/` and `build/gpu-load-216/`. The report contains model revisions, raw response paths, completion counts, language slices, errors, latency percentiles, throughput and startup scope. Raw outputs remain available locally; no private or employer inputs are included.
