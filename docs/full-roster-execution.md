# Full roster execution

Extend the frozen 216-case diagnostic to every previously discussed decision-model configuration. Keep unsupported tasks visible and separate API errors from wrong answers. Heavy inference runs on rented Linux hosts, never the laptop.

## Registered arms

| System | Base / runtime | Coverage and execution |
|---|---|---|
| Jev 1.13 | Hosted `typesafe/jev-1.13-20260917` | 216 cases; quality and concurrency 1/4/16 completed previously |
| Span-01 | Hosted `respan/span-01-20260925` | 48 Boolean cases; compare every model on this same subset |
| GLiNER2.5-Decide | Pinned 0.3B native model | CPU and GPU, full pack |
| GLiNER2.5-multi-Decide | Pinned multilingual checkpoint | CPU and GPU, full pack |
| Laya English / multilingual | Pinned native models | CPU and GPU, full pack |
| JevK5 | Pinned JevK5 4B weights and runtime | CPU and GPU, BF16, CUDA graphs off |
| SemIf | Pinned Qwen3.5-4B and direct-options-v1 | CPU and GPU, BF16, native last-token logits |
| Julia 1 | Pinned native model | CPU and GPU, strict encoding, full pack |
| Sev Arabic Preview | Pinned weights and mmBERT base | Choice-only subset; do not mislabel unsupported Boolean/score tasks as errors |
| Needle 3 | Cactus native engine 3.0.1, pinned weights | Schema-based decision extraction; CPU engine, no equivalent CUDA inference API |
| Ordinary Qwen | Qwen3.8-27B | JSON decisions, thinking off, max 256 output tokens |
| LocalJev + Qwen | Same Qwen3.8-27B | Upstream LocalJev prompt/probability generation; no repair retries |
| TypeLLM direct + Qwen | Same Qwen3.8-27B | Typed candidate-token readout through SGLang |
| TypeLLM thinking-64 + Qwen | Same Qwen3.8-27B | Separate arm, 64 thinking tokens per field |
| DiffusionGemma-as-Jev | DiffusionGemma 26B-A4B, pinned structured-read vLLM runtime | Direct structured logits, not ordinary generation |
| LocalJev + DiffusionGemma | Same DiffusionGemma base | Generated probabilities, a distinct configuration |

Jev + LFM was a tool-argument extraction pipeline. This pack tests decisions and contains no argument-extraction gold, so adding LFM after the decision cannot be evaluated here. Keep that pipeline in the tool-routing/extraction experiment. Do not count it as a missing standalone decision model or invent a score for its extra work. TypeLLM permutation averaging remains an optional separate inference-budget study, not the default direct arm.

## Controls and stop rules

- Dataset revision and case hash are unchanged from `decision-load-protocol.md`. No gold or provenance enters model requests.
- CPU then GPU sequentially on the native-model rental. A separate A100 rental hosts the incompatible large-model runtimes; record its CPU/RAM allocation and hardware identity separately. Do not claim all rows ran on one physical host.
- Four Torch threads. Encoders use FP32; JevK5/SemIf use BF16 on both devices. Large serving models use BF16, with runtime settings recorded.
- Common loopback Chat Completions adapter serializes prediction. Concurrency 1/4/16 measures queueing plus inference. It does not establish optimized batching capacity.
- One quality pass on CPU; three repeats per GPU load lane. Supported subset counts remain explicit. Repeats are correlated, not additional independent quality cases.
- Startup deadline 180 seconds for adapters; large serving backends receive a separate bounded readiness window. Per-request timeout 60 seconds after warmup; warmup uses 20 seconds. Stop scheduling after 600 seconds per native lane or >20% errors after 20 completions. A failed warmup stops that configuration.
- Each rental has a 90-minute local deletion watchdog and a $2.50/hour launch-rate guard. Download and hash-check evidence before deleting each pod. A laptop watchdog is not a provider spending cap.
- Fix setup issues in a new attempt directory. Never overwrite failed attempts or treat startup failure as zero model accuracy.

## Publication gate

Development labels remain independently unreviewed. B04 ambiguity is tracked without editing frozen gold. Publish only a clearly labeled diagnostic report until bilingual review, adapter audit and held-out evaluation are complete.

Sources: [LocalJev](https://github.com/githubnext/localjev), [TypeLLM](https://github.com/TypeLLM/TypeLLM), [structured DiffusionGemma readout](https://github.com/vllm-project/vllm/tree/e9757321527ca1ecd514c07c1418dd2c53da3d19/examples/features/structured_diffusion), [Needle](https://huggingface.co/Cactus-Compute/needle3), [Qwen base](https://huggingface.co/Qwen/Qwen3.8-27B).

## Post-run concurrency correction

The earlier Qwen, LocalJev, and DiffusionGemma load results passed through a
single Python prediction lock. Their concurrency-4/16 latency chiefly measures
that adapter queue. The adapter now allows concurrent upstream HTTP calls for
those three systems and records `adapter_serialized` in response and report
metadata. Native model and TypeLLM adapters keep the lock because their loaded
in-process clients have not been verified thread-safe. This is a protocol
change: the earlier latency rows remain historical and cannot be mixed with new
concurrent-upstream measurements. Real concurrent GPU results are still pending.
