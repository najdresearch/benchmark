# Public fixture tasks

Six cases require access to supplied files. `fixture-tools-v1` runs them through a bounded,
in-memory filesystem using ordinary OpenAI-compatible chat completions. This is an
explicit harness condition, not raw single-turn inference or the Pi coding harness.

| Contract | Rule |
|---|---|
| Inputs | Case prompt and immutable, checksum-verified public fixture files |
| Operations | One JSON `read`, `write`, or `finish` action per response |
| Budget | At most 12 responses, 64 KB source files, 64 KB generated files |
| Isolation | No shell, network, host filesystem or source-file mutation |
| Evidence | Ordered tool transcript, created files, final response and summed usage |
| Scoring | Missing required artifacts or turn-limit exhaustion score zero; then judge correctness, bindings, grounding and format using source files |
| Stop rule | Missing/corrupt fixtures or invalid judge output fail execution; never invent a grade |
| Split | Existing public test cases; no claim of sealed evaluation |

The baseline is an endpoint invoked under the same fixed protocol and budget. Report the
endpoint/model version, returned model identifier, dataset revision, harness version,
judge version, token usage and elapsed time. Six cases are too few for a stable model
ranking; disclose the count and per-case failures, and do not tune on these public cases.
No abstention credit: missing answers/artifacts and exhausted model turns fail the case.
Infrastructure/judge failures remain errors rather than model failures.

Use `run_fixture(prompt, files, invoke)` with an async callback returning `content`,
`model`, and `usage`. `artifact_grade(expected, result)` enforces execution requirements
before a semantic judge. Gold answers must never be supplied to the callback.

Four corrected Absher cases use `absher-corrected-key-v1`: only the documented Arabic
option key is accepted (surrounding whitespace and trailing period/parenthesis allowed).
No semantic judge, partial credit, or substring matching is used. Two prompts were corrected;
historical responses must not be rescored as though they answered the new prompts.

Data and correction evidence: [Najd Benchmark](https://huggingface.co/datasets/najdresearch/najd-benchmark).
