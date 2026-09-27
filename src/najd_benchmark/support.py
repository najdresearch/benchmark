"""Small classification runner and scorer exercising the shared v1 contracts."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from datetime import UTC, datetime

import httpx

from .contracts import digest, seal_bundle


def metrics(records: list[dict], labels: list[str]) -> dict:
    n = len(records)
    if not n:
        raise ValueError("Cannot score an empty run")
    correct = sum(
        r["status"] == "ok" and r["predicted_label"] == r["expected_label"] for r in records
    )
    invalid = sum(r["status"] == "invalid_output" for r in records)
    errors = sum(r["status"] == "provider_error" for r in records)
    per_label, confusion = [], []
    for label in labels:
        support = sum(r["expected_label"] == label for r in records)
        tp = sum(r["expected_label"] == label and r["predicted_label"] == label for r in records)
        predicted = sum(r["predicted_label"] == label for r in records)
        precision, recall = tp / predicted if predicted else 0, tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_label.append(
            {"label": label, "support": support, "precision": precision, "recall": recall, "f1": f1}
        )
        for output in [*labels, None]:
            count = sum(
                r["expected_label"] == label and r["predicted_label"] == output for r in records
            )
            if count:
                confusion.append({"expected": label, "predicted": output, "count": count})
    p, z = correct / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    times = sorted(r["latency_ms"] for r in records)
    return {
        "case_count": n,
        "correct": correct,
        "invalid_outputs": invalid,
        "provider_errors": errors,
        "accuracy": p,
        "grading_coverage": (n - errors) / n,
        "macro_f1": sum(r["f1"] for r in per_label) / len(labels),
        "per_label": per_label,
        "confusion_matrix": confusion,
        "accuracy_ci95": {
            "low": max(0, center - radius),
            "high": min(1, center + radius),
            "method": "wilson",
        },
        "latency_ms": {
            "p50": times[math.ceil(n * 0.5) - 1],
            "p95": times[math.ceil(n * 0.95) - 1],
            "count": n,
            "method": "nearest-rank",
        },
        "tokens": {
            "prompt": sum(r["usage"]["prompt_tokens"] or 0 for r in records),
            "completion": sum(r["usage"]["completion_tokens"] or 0 for r in records),
            "cases_with_missing_usage": sum(
                any(v is None for v in r["usage"].values()) for r in records
            ),
        },
        "cost_usd": None,
    }


def parse_label(output: str, labels: list[str]) -> str | None:
    try:
        value = json.loads(output)
        if isinstance(value, dict) and set(value) == {"label"} and value["label"] in labels:
            return value["label"]
    except (ValueError, TypeError):
        pass
    return None


def run(
    manifest: dict,
    pack: dict,
    cases: list[dict],
    *,
    endpoint: str,
    model: str,
    api_key: str | None = None,
    endpoint_kind: str = "remote",
    transport: httpx.BaseTransport | None = None,
    benchmark_revision: str = "unknown",
) -> dict:
    if (
        pack["task_type"],
        pack["output_format"],
        pack["scorer"]["id"],
        pack["scorer"]["version"],
        pack["harness"]["id"],
        pack["harness"]["version"],
    ) != ("classification", "json_label", "exact-label", "1", "single-turn", "1"):
        raise ValueError("Unsupported task contract; no fallback scorer")
    from importlib.resources import files
    from pathlib import Path

    package_root = Path(str(files("najd_benchmark")))
    source_hash = digest(
        {
            str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package_root.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json"}
        }
    )
    started = datetime.now(UTC).isoformat()
    records, models = [], set()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(
        timeout=pack["budget"]["timeout_seconds"],
        headers=headers,
        transport=transport,
        follow_redirects=False,
    ) as client:
        for case in cases:
            before, output, predicted, status, error = time.monotonic(), "", None, "ok", None
            usage = {"prompt_tokens": None, "completion_tokens": None}
            try:
                response = client.post(
                    endpoint.rstrip("/") + "/chat/completions",
                    json={
                        "model": model,
                        "temperature": 0,
                        "max_tokens": pack["budget"]["max_tokens"],
                        "messages": [
                            {"role": "system", "content": pack["system_prompt"]},
                            {"role": "user", "content": case["prompt"]},
                        ],
                    },
                )
                response.raise_for_status()
                data = response.json()
                output = data["choices"][0]["message"]["content"]
                if not isinstance(output, str):
                    raise ValueError("Response content must be text")
                if isinstance(data.get("model"), str) and data["model"]:
                    models.add(data["model"])
                for key in usage:
                    count = (data.get("usage") or {}).get(key)
                    usage[key] = count if type(count) is int and count >= 0 else None
                predicted = parse_label(output, pack["labels"])
                status = "ok" if predicted is not None else "invalid_output"
            except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
                status, error, output, predicted = "provider_error", type(exc).__name__, "", None
            records.append(
                {
                    "case_id": case["id"],
                    "status": status,
                    "output": output,
                    "expected_label": case["label"],
                    "predicted_label": predicted,
                    "latency_ms": round((time.monotonic() - before) * 1000, 3),
                    "usage": usage,
                    "error_class": error,
                    "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "artifacts": [],
                }
            )
    bundle = {
        "schema_version": "1.0.0",
        "kind": "result_bundle",
        "dataset_manifest": manifest,
        "task_pack": pack,
        "run": {
            "id": str(uuid.uuid4()),
            "origin": "local_cli",
            "organization_id": None,
            "visibility": "private",
            "benchmark_version": "0.2.0",
            "benchmark_revision": benchmark_revision,
            "benchmark_source_sha256": source_hash,
            "model_requested": model,
            "models_returned": sorted(models),
            "endpoint_kind": endpoint_kind,
            "temperature": 0,
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
        },
        "records": records,
        "metrics": metrics(records, pack["labels"]),
    }
    return seal_bundle(bundle)
