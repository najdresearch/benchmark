# Natural-development evaluation — 2026-09-27

Development diagnostic only. Same 216 cases (72 aligned English/MSA/Saudi families), frozen dataset revision `84bf0a30ced090e2b552beafa5f0f3c9a3665c0c`. No independent label review yet; not an Arena leaderboard result.

## Quality on 216 unique cases

| Model | Correct | English /72 | MSA /72 | Saudi /72 | Unsafe /168 |
|---|---:|---:|---:|---:|---:|
| Jev 1.13 (hosted) | 212/216 (98.1%) | 72 | 71 | 69 | 0 |
| GLiNER2.5-Decide | 101/216 (46.8%) | 39 | 33 | 29 | 25 |
| Laya English | 79/216 (36.6%) | 37 | 19 | 23 | 36 |
| Laya multilingual | 94/216 (43.5%) | 32 | 30 | 32 | 23 |

Self-hosted quality above uses the earlier same-host CPU run; FP32 GPU answers matched all 216 cases exactly. The new GPU load run also preserves those answers. Hosted Jev resolves to `typesafe/jev-1.13-20260917`.

## Jev error review

| Case | Expected → actual | Assessment |
|---|---|---|
| N0104-ar-SA | clarify → maintenance | “الطلب تعطل مرة ثانية، وين أرسله؟” does not identify the service. The model guessed maintenance. |
| N0201-ar-SA | info → create | “متى تفتحون؟ ما عندي مشكلة بالخدمة.” asks opening hours and explicitly denies a service issue. The model chose complaint creation. |
| B04-ar-MSA | true → false | Ambiguous source wording: matching invoice does not explicitly establish a matching purchase order. |
| B04-ar-SA | true → false | Same ambiguity. English B04 is also flagged even though it matched gold. |

The original scores remain unchanged. The table below is a **post-hoc sensitivity check**, excluding all three B04 variants for every model; it is not a replacement headline. Proposed wording fixes are recorded in the datasets repository review file.

| Model | Correct excluding B04 |
|---|---:|
| Jev 1.13 (hosted) | 211/213 |
| GLiNER2.5-Decide | 100/213 |
| Laya English | 77/213 |
| Laya multilingual | 92/213 |

## Load tests

Each row has 648 completed requests: three repeats of the same 216 cases. Quantiles are full-response milliseconds. GPU latency includes loopback HTTP and the serialized inference queue. Hosted latency includes internet/provider time from the Mac. These are separate latency scopes.

| Model | Clients | Correct /648 | p50 ms | p95 ms | Observed p99 ms | Requests/s | Errors | Cases with repeat disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev 1.13 (hosted) | 1 | 635 | 523.9 | 619.2 | 818.2 | 1.86 | 0 | 1/216 |
| Jev 1.13 (hosted) | 4 | 635 | 523.4 | 677.6 | 1278.1 | 7.23 | 0 | 1/216 |
| Jev 1.13 (hosted) | 16 | 636 | 524.7 | 1088.4 | 1308.1 | 26.89 | 0 | 0/216 |
| GLiNER2.5-Decide | 1 | 303 | 30.3 | 38.1 | 49.3 | 19.26 | 0 | 0/216 |
| GLiNER2.5-Decide | 4 | 303 | 91.0 | 111.1 | 116.7 | 31.68 | 0 | 0/216 |
| GLiNER2.5-Decide | 16 | 303 | 45.6 | 184.7 | 292.9 | 22.62 | 0 | 0/216 |
| Laya English | 1 | 237 | 23.9 | 31.1 | 36.9 | 21.75 | 0 | 0/216 |
| Laya English | 4 | 237 | 28.9 | 58.5 | 85.2 | 33.56 | 0 | 0/216 |
| Laya English | 16 | 237 | 45.0 | 191.0 | 321.4 | 33.87 | 0 | 0/216 |
| Laya multilingual | 1 | 282 | 13.5 | 15.8 | 26.6 | 26.66 | 0 | 0/216 |
| Laya multilingual | 4 | 282 | 14.3 | 30.0 | 47.6 | 32.29 | 0 | 0/216 |
| Laya multilingual | 16 | 282 | 16.0 | 66.9 | 139.5 | 31.16 | 0 | 0/216 |

## Interpretation and limits

- This is a short closed-loop load diagnostic, with one run per setting, no retries and no dynamic batching. The self-hosted adapter serializes prediction; concurrency measures queueing and client/server overhead, not an optimized GPU server. p99 is exploratory with only about six tail observations per row.
- Repeats and language variants are correlated. No confidence interval or significance claim is made from treating 648 requests as independent quality samples. Independent scenario-family sampling, label review and a held-out test are still needed.
- All self-hosted models use one A100-SXM4-80GB allocation with FP32 and four Torch threads. The pod has 24 allocated vCPUs and 117 GB RAM. Earlier CPU/GPU results used another A100 host, so do not attribute differences between runs solely to load.
- Fixed request order and repeated inputs may interact with provider caching. Provider hardware, precision, queueing and cold starts are unknown. API cost is not inferred from advertised prices.
- Startup below measures process-to-ready with downloaded weights. It excludes pod provisioning and downloads; it is not uncached cold start.

| Model | Cached readiness seconds |
|---|---:|
| GLiNER2.5-Decide | 9.04 |
| Laya English | 5.61 |
| Laya multilingual | 6.81 |

## Evidence and cleanup

- GPU archive: `build/gpu-load-216/evidence.tgz`, SHA-256 `bd2b89d0a5a4268ff8b4ad883211588c3cc84d5a9ae3459fb5ff54b8fa0758f6`. Nine lanes, 5,832 responses.
- Hosted evidence: `build/jev-natural-216/{quality,load-c1,load-c4,load-c16}/` with responses, warmups and scored reports. 216 quality requests plus 1,944 load requests.
- GPU logs include runtime precision, model pins and installed packages. The rented pod was deleted and its absence verified; see `build/gpu-load-216/termination.json`.
- Reproduction and stop rules: [decision-load-protocol.md](decision-load-protocol.md).
- Dataset: https://huggingface.co/datasets/najdresearch/system-one .
- The dataset review is in `najdresearch/datasets/reviews/system-one-b04-ambiguity.json`. No frozen labels were changed.
