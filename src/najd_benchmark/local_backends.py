"""Pinned native adapters for the laptop compatibility lane."""

import json

PINS = {
    "laya_english": ("convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"),
    "laya_multi": (
        "convaiinnovations/laya-multilingual",
        "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    ),
    "jevk5": ("alibiserikbay/JevK5", "c4f7fdb3aeab5582336406e78d3bef11bf98833d"),
    "semif": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
}


def load(system, device=None, precision="native"):
    import torch
    from huggingface_hub import snapshot_download

    from .decision_server import canonical, typed_answers

    model_id, revision = PINS[system]
    local = snapshot_download(
        model_id,
        revision=revision,
        local_files_only=True,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
    )
    if system.startswith("laya"):
        import laya
        from laya.common import render_options

        model = laya.load(local, device=device or "cpu", fast=False)
        if precision == "fp32":
            model.amp_enabled = False
            model.dtype = torch.float32
        if device and model.device.type != device:
            raise RuntimeError("Requested device unavailable; refusing silent fallback")
        print(
            json.dumps(
                {
                    "device": str(model.device),
                    "weight_dtype": str(next(model.model.parameters()).dtype),
                    "autocast": model.amp_enabled,
                    "autocast_dtype": str(model.dtype),
                }
            ),
            flush=True,
        )

        def predict(state, questions):
            text = canonical(state)
            for q in questions.values():
                internal = model._to_internal(q)
                lengths = [
                    len(model.tok(" " + x, add_special_tokens=False)["input_ids"])
                    for x in render_options(internal)
                ]
                head = len(
                    model.tok(
                        internal["t"] + " question: " + str(internal["ins"]),
                        add_special_tokens=False,
                    )["input_ids"]
                )
                state_len = len(model.tok(text, add_special_tokens=False)["input_ids"])
                if (
                    max(lengths) > 48
                    or head + sum(n + 1 for n in lengths) > model.cfg.get("head_max_len", 192)
                    or state_len + head + sum(n + 1 for n in lengths) + 4
                    > model.cfg.get("max_len", 512)
                ):
                    raise ValueError("Input would be truncated")
            with torch.inference_mode():
                raw = model.predict(text, questions)
            return typed_answers(questions, raw), raw
    elif system == "jevk5":
        from jevk5 import JevK5

        model = JevK5(
            local,
            device=device or "mps",
            dtype=torch.float32 if precision == "fp32" else torch.bfloat16,
            graphs=False,
        )

        def predict(state, questions):
            raw = {"answers": {k: model.decide(canonical(state), q) for k, q in questions.items()}}
            if model.device == "mps":
                torch.mps.synchronize()
            elif str(model.device).startswith("cuda"):
                torch.cuda.synchronize()
            return typed_answers(questions, raw), raw
    else:
        from jevk5.prompt import answer, decision_options

        if device in ("cpu", "cuda"):
            import transformers
            from semif_phase1.direct import score

            config = transformers.AutoConfig.from_pretrained(local).get_text_config()
            tok = transformers.AutoTokenizer.from_pretrained(local)
            model = transformers.Qwen3_5ForCausalLM.from_pretrained(
                local,
                config=config,
                dtype=torch.float32 if precision == "fp32" else torch.bfloat16,
                device_map={"": device},
            ).eval()
            metadata = {
                "source": model_id,
                "revision": revision,
                "device": device,
                "dtype": str(next(model.parameters()).dtype),
            }
        else:
            from semif_phase1.mlx_backend import load_model, score

            model, tok, metadata = load_model(model_id, revision)

        def predict(state, questions):
            answers = {}
            evidence = {}
            for k, q in questions.items():
                result = score(
                    model,
                    tok,
                    {
                        "id": k,
                        "state": canonical(state),
                        "question": q["instructions"],
                        "options": [{"id": i, "description": d} for i, d in decision_options(q)],
                    },
                    metadata,
                )
                answers[k] = answer(
                    q,
                    dict(zip(result["option_ids"], result["probabilities"], strict=True)),
                    result["input_tokens"],
                )
                evidence[k] = result
            raw = {"answers": answers, "evidence": evidence}
            # MLX scalar/list conversion in score materializes device work before return.
            return typed_answers(questions, raw), json.loads(json.dumps(raw))

    return predict
