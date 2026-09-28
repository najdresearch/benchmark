# Sev capability audit of System One v0.1

The previous Sev run labeled 81 rejected HTTP requests as `transport_error`. The
client caught `HTTPError` under `URLError`, which hid the distinction between an
HTTP rejection and a network failure. The response status and body were not
saved, so their exact historical HTTP codes cannot be recovered.

An encoder-only audit with the locally pinned Sev Preview tokenizer and 384-token
configuration predicts all 81 rejected case IDs as outside the adapter's
capability. None of the historical rejected IDs is predicted supported. This
requires no model inference and does not alter the frozen dataset or gold labels.

| Reason | Cases in 5,184-case release | Where |
|---|---:|---|
| Supported by pinned Sev adapter | 2,418 | Seven packs or supported subsets |
| Type or fewer than two choices | 771 | Score tasks and three single-option cases |
| Question and choices exceed context | 1,844 | ArBanking77, MASSIVE, Arabic Function Calling |
| State would be truncated | 151 | Paired Tool Use and one Absher case |

The prior filtered run selected 4,413 cases before token preflight. It received
2,499 responses, of which 2,418 were valid. The other 81 were reported as
`transport_error` by the old client; the encoder-only audit maps 60 of those
to oversized question tails and 21 to state truncation. Missing cases after
the error circuit must not be treated as model mistakes.

The corrected path exposes a read-only loopback capability endpoint, filters
unsupported cases before inference, records a per-case capability audit, and
distinguishes HTTP 422/4xx/5xx from network errors. The old 2,418 valid
responses remain historical evidence and are not reused as the rerun.

## Verified GPU rerun

On September 28, 2026, the corrected code (`59677b5`) ran on a newly rented
A100-SXM4-80GB with the same frozen public dataset, Sev model snapshot, and
pinned runtime. All **2,418/2,418 supported cases** returned valid responses;
**1,012/2,418** matched the frozen labels. All eleven pack reports were
produced, including explicit zero-eligible reports for the four packs outside
Sev's context. There were no request errors or error circuits. These are
development results for Sev's supported subset, not an all-task ranking.

| Pack | Correct / supported | Valid / supported |
|---|---:|---:|
| Controlled development | 152/360 | 360/360 |
| Controlled validation | 235/510 | 510/510 |
| Controlled reserved evaluation | 343/810 | 810/810 |
| Natural development | 58/168 | 168/168 |
| Absher diagnostic | 125/379 | 379/379 |
| Agent diagnostic | 36/41 | 41/41 |
| AraTrust diagnostic | 63/150 | 150/150 |

The natural-development language slices were English 23/56, MSA 18/56, and
Saudi Arabic 17/56. Complete-response latency at concurrency 1 was p50
39.8 ms, p95 56.9 ms, and p99 64.8 ms on that 168-case subset. This rental was
a new host with a different CPU/RAM allocation from the prior native roster;
do not merge its latency into same-machine comparisons.

All **39** remote result/log files matched local SHA-256 hashes before rental
deletion. The local evidence is in `build/sev-rerun-2026-09-28/`, including
`results/sev-capability-filtered/`, `remote-hash-verification.json`,
`rental-deleted.json`, and the runtime package/hardware manifest. After deletion,
RunPod reported no running pods. The account balance changed from $44.9631 to
$44.5259 across the run, roughly $0.44 including any non-pod charges.

Reproduce the no-inference audit with `scripts/audit_sev_capability.py` using
the pinned public dataset and model snapshot. Local output is saved at
`build/sev-capability-audit.json`; raw response and checkpoint files remain
under `build/overnight-checkpoints/`.

Runtime basis: [Sev Preview source](https://github.com/3ssiri/sev-arabic-preview/blob/bdc797da20fdcb899d7f076f33e87b0f79bec666/sev_preview/runtime.py),
[serializer](https://github.com/3ssiri/sev-arabic-preview/blob/bdc797da20fdcb899d7f076f33e87b0f79bec666/sev_preview/_jevlite/td_data.py),
[model card](https://huggingface.co/3ssiri/Sev-Arabic-Preview-v0.1).
