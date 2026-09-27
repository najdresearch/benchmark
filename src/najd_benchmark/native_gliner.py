"""Offline native GLiNER development adapter. Uses cached, pinned weights only."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from .decisions import load_pack, score, valid

MODEL = "fastino/GLiNER2.5-Decide"
REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
MODELS = {
    "gliner": (MODEL, REVISION),
    "gliner_multi": ("fastino/GLiNER2.5-multi-Decide", "6bc1d43d201b0691e733626389af8c57eea3ea68"),
}


def tasks_for(questions):
    tasks = {}
    for key, question in questions.items():
        kind = question["type"]
        if kind == "choice":
            labels = question["criteria"]
        elif kind == "score":
            labels = {str(i): text for i, text in enumerate(question["criteria"])}
        elif kind == "noul":
            labels = {"false": "No", "true": "Yes"}
        else:
            raise ValueError("Unsupported decision type")
        tasks[key] = {"labels": labels, "prompt": question["instructions"], "multi_label": False}
    return tasks


def normalize(questions, raw):
    if set(raw) != set(questions):
        raise ValueError("Unexpected question keys")
    output = {}
    for key, question in questions.items():
        label = raw[key]["label"] if isinstance(raw[key], dict) else raw[key]
        labels = tasks_for({key: question})[key]["labels"]
        if not isinstance(label, str) or label not in labels:
            raise ValueError("Unknown label")
        output[key] = (
            int(label)
            if question["type"] == "score"
            else label == "true"
            if question["type"] == "noul"
            else label
        )
    return output


def predict(model, state, questions):
    # Only evidence and task definitions cross the adapter boundary, never gold labels.
    text = json.dumps(state, ensure_ascii=False, sort_keys=True)
    tasks = json.loads(json.dumps(tasks_for(questions), ensure_ascii=False, sort_keys=True))
    raw = model.classify_text(text, tasks, include_confidence=True)
    return (
        normalize(questions, raw),
        raw,
        hashlib.sha256(
            json.dumps({"text": text, "tasks": tasks}, ensure_ascii=False).encode()
        ).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", choices=MODELS, default="gliner")
    args = parser.parse_args()
    model_id, revision = MODELS[args.system]
    manifest, cases = load_pack(args.pack)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    import torch
    from gliner2 import AutoExtractor
    from huggingface_hub import snapshot_download

    torch.set_num_threads(4)
    torch.manual_seed(42)
    start = time.perf_counter()
    local = snapshot_download(model_id, revision=revision, local_files_only=True)
    model = AutoExtractor.from_pretrained(local).to("cpu").eval()
    setup_seconds = time.perf_counter() - start
    rows = []
    failures = 0
    with (args.output / "responses.jsonl").open("x") as sink:
        for case in cases:
            row = {"case_id": case["id"], "status": "circuit_open", "output": None}
            if failures < 3:
                start = time.perf_counter()
                try:
                    with torch.inference_mode():
                        output, raw, digest = predict(model, case["state"], case["questions"])
                    row.update(
                        output=output,
                        raw=raw,
                        request_sha256=digest,
                        status="ok" if valid(case, output) else "invalid",
                    )
                except Exception as exc:
                    row.update(status="adapter_error", error_type=type(exc).__name__)
                row["elapsed_ms"] = (time.perf_counter() - start) * 1000
                failures = 0 if row["status"] == "ok" else failures + 1
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            rows.append(row)
    report = score(cases, rows)
    report.update(
        task_pack=manifest["task_pack"],
        cases_sha256=manifest["cases_sha256"],
        configuration={
            "adapter": "native-gliner-dev-v2",
            "model": model_id,
            "revision": revision,
            "device": "cpu",
            "threads": 4,
            "seed": 42,
            "setup_seconds": setup_seconds,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("gliner2", "torch", "transformers", "huggingface-hub")
            },
            "attempts_per_case": 1,
            "offline": True,
            "serialization": "sorted-object-keys-readable-unicode-v1",
        },
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("expected", "correct", "valid", "statuses")}))


if __name__ == "__main__":
    main()
