# Same-host CPU and GPU smoke test

Run the same pinned models against the same 24 development cases on a single rented machine. This is a compatibility and latency diagnostic, not the full publication benchmark.

| Setting | Value |
|---|---|
| Models | GLiNER2.5-Decide, Laya English, Laya multilingual |
| Device order | All CPU configurations, then all CUDA configurations |
| API | Loopback Chat Completions decision adapter |
| CPU threads | 4 |
| Concurrency | 1 |
| Warmup | 3 requests per configuration, saved separately |
| Measured requests | 24, one attempt per case |
| Readiness deadline | 90 seconds |
| Request deadline | 15 seconds |
| Request scheduling budget | 120 seconds per configuration |
| Precision | Backend defaults; no explicit quantization or dtype conversion |
| Publication status | Ineligible: development sample, independent review pending |

## Run

Place the exact `cases.jsonl` selection in the working directory, along with `src/` and `scripts/remote_cpu_gpu_smoke.py`. Install pinned encoder dependencies in an isolated environment with CUDA-enabled PyTorch 2.8.0. GLiNER also needs `peft==0.21.0` and `accelerate==1.15.0`.

```sh
PYTHONPATH=src USE_TF=0 python scripts/remote_cpu_gpu_smoke.py \
  --pack dataset/natural-development --output results \
  --dataset-revision 84bf0a30ced090e2b552beafa5f0f3c9a3665c0c
```

The script downloads pinned public model snapshots before measurement and creates `results/environment.json`, plus raw responses, startup logs, warmups, and scorer reports for each configuration. Existing result directories cause failure rather than overwrite. Device selection is `--device cpu` or `--device cuda` in `decision_server` for GLiNER and Laya.

Provisioning and termination are external to this worker. Arm a cleanup watchdog before running, retrieve and verify artifacts, then delete only the Pod created for this run. A local watchdog depends on the laptop staying awake and connected; it is not a provider spending cap.

## Interpretation

Complete-response timings include loopback HTTP, preprocessing, inference and serialization. They exclude the laptop-to-server network. Hosted API latency is a separate lane. Startup uses already-cached weights and is not a full uncached cold start. The Mac comparison has one warmup rather than three and a different OS/CPU; report this limitation.

The CPU lane uses the GPU Pod's host allocation. Its rental rate does not estimate CPU-only deployment cost. Do not extrapolate p99, loaded throughput, or general quality from 24 cases. Compare answer differences directly across devices and keep setup failures separate from incorrect model decisions.

Evidence: `build/remote-cpu-gpu/` (local, ignored). Provider documentation: https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods .

## Published natural-development experiment

The public draft contains 5,184 cases in 11 packs; the next registered execution scope is its 216-case natural-development pack. Fast Decisions remains outside the public release pending contact-pattern review. This run expands coverage beyond the original 24-case smoke and does not imply all 5,184 rows have been evaluated.

```sh
python scripts/fetch_system_one.py \
  --revision 84bf0a30ced090e2b552beafa5f0f3c9a3665c0c --output dataset
PYTHONPATH=src USE_TF=0 python scripts/remote_cpu_gpu_smoke.py \
  --pack dataset/natural-development --output results \
  --dataset-revision 84bf0a30ced090e2b552beafa5f0f3c9a3665c0c --max-seconds 600
```

The fetcher requires an immutable commit SHA and verifies every file in the release lock. The worker verifies the pack hash and accepts development/public-reference lanes only. Reserved/validation packs cannot silently enter the development runner. The remote cleanup watchdog for this larger run is 40 minutes. Dependencies must be installed before timing begins, including pinned PEFT and Accelerate.

### Precision control

Laya 0.3.20 enables GPU autocast by default, while CPU AMP is opt-in. Therefore `--precision native` compares backend configurations, not hardware alone. Use `--precision fp32` to disable Laya autocast and compare FP32 CPU with FP32 CUDA. Keep native GPU results as a separate optimized configuration. GLiNER remains at its loaded default precision. The follow-up GPU FP32 run uses the same machine and cases.
