# DiffusionGemma Arabic verification

Arabic text must reach the model as readable Arabic. Escapes used only to transport JSON are harmless after decoding; literal escape sequences inside the model's text change its input.

## What was checked

| Check | Evidence | Result |
|---|---|---|
| Earlier experiment | `exp/decision-model-benchmark/src/run_structured.py`, `--unicode-state` | The earlier workaround serialized state with `ensure_ascii=False` and sent it as a string |
| Current adapter | `systemone_payload` in `decision_server.py` | Uses sorted-key readable Unicode state strings, including nested state |
| Regression test | `test_diffusion_wrapper_receives_readable_arabic_after_json_transport` | Passes, including Arabic digits and instructions |
| Actual tokenizer | Pinned wrapper's `jev_state`, `system_text`, `chat_prompt_ids`, and tokenizer decode | Original state appears unchanged in all three decoded prompts |
| Live inference | N0101-en, N0101-ar-MSA, N0101-ar-SA | Three valid `licensing` answers through the common Chat Completions adapter |
| Full quality pass | Frozen public 5,184-case suite | Running; the three checks are not a benchmark score |

## Startup and runtime issues

The original launch script waited four seconds after starting the structured wrapper. The wrapper loads its tokenizer before accepting HTTP requests, so a fixed sleep is not a readiness check. Initial warmups failed with transport errors. The replacement waits for `/health`; a direct structured request and all three common-adapter checks succeeded after readiness.

The failed attempt remains in the saved artifacts. Its transport errors do not establish a model-quality failure. The original logs did not retain the HTTP error body, so they cannot prove every failed warmup had the same cause.

A separate Qwen/SGLang environment had mismatched FlashInfer packages, then mismatched Torch/TorchVision packages. Repairs are isolated from the active DiffusionGemma runtime and require a separate preflight and successful inference before scoring.

## LocalJev compatibility is a separate limitation

Two direct HTTP probes confirmed why unmodified LocalJev cannot use this diffusion backend through ordinary Chat Completions:

| Probe | Runtime response |
|---|---|
| LocalJev-style temperature and seed | HTTP 400: these sampling parameters are unsupported for diffusion models |
| Remove those controls, keep JSON-schema response format | HTTP 400: structured outputs are unsupported for diffusion language models |

The structured diffusion wrapper uses a different read mechanism and is successfully evaluating the dataset. This does not make ordinary constrained Chat Completions compatible. Do not silently remove constraints and present that as the same LocalJev configuration. Keep LocalJev + DiffusionGemma marked unsupported on this pinned runtime; LocalJev + Qwen remains a separate candidate.

Raw server responses are retained in `large-results/localjev-diffusion-compatibility.json` and `large-results/localjev-diffusion-schema-compatibility.json` in the local checkpoints.

## Frozen configuration

| Item | Value |
|---|---|
| Model | `google/diffusiongemma-26B-A4B-it` |
| Model revision | `f7f5b7f5fa82ffc52addd066915886d497f5517b` |
| vLLM and structured wrapper | `e9757321527ca1ecd514c07c1418dd2c53da3d19` |
| Public dataset revision | `84bf0a30ced090e2b552beafa5f0f3c9a3665c0c` |
| Structured sampling | One read, seed 42, no thinking |
| Hardware | A100-SXM4-80GB, BF16 |
| Scope | Text decisions, draft labels, no Arena publication |

Local evidence is saved under `build/overnight-checkpoints/large/large-results/dg-unicode-verification/`: `responses.json` and `tokenizer-audit.json`. Full-suite artifacts are under the sibling `dg-full/` directory. These ignored build artifacts contain the source requests and model responses; this document is not a substitute for them.

## Sources

- [Pinned structured wrapper](https://github.com/vllm-project/vllm/blob/e9757321527ca1ecd514c07c1418dd2c53da3d19/examples/features/structured_diffusion/structured_server.py)
- [DiffusionGemma model](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [Najd public dataset](https://huggingface.co/datasets/najdresearch/system-one/tree/84bf0a30ced090e2b552beafa5f0f3c9a3665c0c)

## Full-suite coverage audit

The completed run has valid responses for all **3,337 supported cases**. The pinned wrapper requires 2–26 alternatives. ArBanking77 (77), MASSIVE (60), and Arabic function calling (28) exceed that limit; three agent-diagnostic cases have only one option. These **1,847 unsupported cases** remain visible in the original reports as transport failures or unattempted cases after circuit breaks. A separate post-run capability audit documents the source-based eligibility rule; it does not rewrite the failed attempts or claim all 5,184 cases were evaluated. Future runs apply the capability filter before inference.

Concurrency 1/4/16 completed 648 requests each with zero errors. Complete-response p50/p95/p99 were 70/76/95 ms, 133/183/246 ms, and 279/732/760 ms respectively. These are loopback serialized-adapter measurements on the A100 host; repeat agreement is not perfect, so a fixed seed must not be described as bitwise deterministic.

The large-model rental was deleted after all 66 remote artifact files matched their local SHA-256 hashes. Qwen/SGLang did not reach readiness: an inherited FlashInfer JIT-cache package still conflicted with the isolated FlashInfer version. Preserve that as an environment failure, not a model score.
