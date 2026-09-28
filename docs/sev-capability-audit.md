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

The corrected path now exposes a read-only loopback capability endpoint, filters
unsupported cases before inference, records a per-case capability audit, and
distinguishes HTTP 422/4xx/5xx from network errors. It still needs a real Sev
inference rerun before reporting a new quality result. Do not compare the old
2,418 valid responses as if they came from a completed 2,418-case run.

Reproduce the no-inference audit with `scripts/audit_sev_capability.py` using
the pinned public dataset and model snapshot. Local output is saved at
`build/sev-capability-audit.json`; raw response and checkpoint files remain
under `build/overnight-checkpoints/`.

Runtime basis: [Sev Preview source](https://github.com/3ssiri/sev-arabic-preview/blob/bdc797da20fdcb899d7f076f33e87b0f79bec666/sev_preview/runtime.py),
[serializer](https://github.com/3ssiri/sev-arabic-preview/blob/bdc797da20fdcb899d7f076f33e87b0f79bec666/sev_preview/_jevlite/td_data.py),
[model card](https://huggingface.co/3ssiri/Sev-Arabic-Preview-v0.1).
