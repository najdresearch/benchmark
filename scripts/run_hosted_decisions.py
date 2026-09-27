"""Bounded OpenRouter Jev quality/load run. Secrets never enter saved artifacts."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from najd_benchmark.decision_server import canonical, typed_answers
from najd_benchmark.decisions import load_pack, output_schema, request_payload, score, valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument(
        "--model",
        choices=["typesafe/jev-1.13", "respan/span-01", "qwen/qwen3.8-27b"],
        default="typesafe/jev-1.13",
    )
    p.add_argument("--concurrency", type=int, choices=[1, 4, 16], default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-seconds", type=int, default=600)
    p.add_argument("--credential-file", type=Path)
    a = p.parse_args()
    if not 1 <= a.repeats <= 3 or not 1 <= a.max_seconds <= 900:
        p.error("Bounded to 1–3 repeats and 1–900 seconds")
    manifest, originals = load_pack(a.pack)
    if len(originals) > 216:
        p.error("This runner is bounded to 216 unique cases")
    total_cases = len(originals)
    if a.model == "respan/span-01":
        originals = [
            c for c in originals if all(q["type"] == "noul" for q in c["questions"].values())
        ]
    if not originals:
        p.error("No supported cases")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and a.credential_file:
        match = re.search(
            r"""(?m)^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*["']?([^\s"'\n#]+)""",
            a.credential_file.read_text(),
        )
        key = match.group(1) if match else None
    if not key:
        p.error("Configure OPENROUTER_API_KEY; never paste it into the command")
    a.output.mkdir(parents=True, exist_ok=False)
    model = a.model

    def call(c):
        questions = c["questions"]
        state = c["state"]
        if model == "respan/span-01":
            state = canonical(state)
            questions = {
                k: {**q, "criteria": q.get("criteria", {"true": "Yes", "false": "No"})}
                for k, q in questions.items()
            }
        body = canonical({"model": model, "state": state, "questions": questions}).encode()
        url = "https://openrouter.ai/api/alpha/decisions"
        if model.startswith("qwen/"):
            payload = request_payload(c, model)
            payload.update(
                seed=42, reasoning={"enabled": False}, provider={"require_parameters": True}
            )
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "strict": True,
                    "schema": output_schema(c["questions"]),
                },
            }
            body = canonical(payload).encode()
            url = "https://openrouter.ai/api/v1/chat/completions"
        req = urllib.request.Request(
            url,
            body,
            {"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        row = {
            "case_id": c["id"],
            "attempt": 1,
            "output": None,
            "source_case_id": c.get("source_case_id", c["id"]),
            "request_sha256": hashlib.sha256(body).hexdigest(),
        }
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("Oversized response")
            payload = json.loads(raw)
            row.update(raw=payload, response_sha256=hashlib.sha256(raw).hexdigest())
            if model.startswith("qwen/"):
                choice = payload["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Incomplete output")
                output = json.loads(choice["message"]["content"])
            else:
                output = typed_answers(c["questions"], payload)
            row.update(
                output=output,
                raw=payload,
                response_sha256=hashlib.sha256(raw).hexdigest(),
            )
            row["status"] = "ok" if valid(c, row["output"]) else "invalid"
        except urllib.error.HTTPError as exc:
            row.update(status="transport_error", http_status=exc.code)
        except (TimeoutError, urllib.error.URLError):
            row["status"] = "transport_error"
        except (ValueError, KeyError, TypeError):
            row["status"] = "invalid"
        row["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return row

    warm = [call(originals[0])]
    (a.output / "warmup.json").write_text(json.dumps(warm, ensure_ascii=False, indent=2))
    if warm[0]["status"] != "ok":
        raise SystemExit("Warmup failed; saved status explains the failure")
    cases = [
        {
            **c,
            "source_case_id": c["id"],
            "id": c["id"] if a.repeats == 1 else c["id"] + f"__repeat{r}",
        }
        for r in range(a.repeats)
        for c in originals
    ]
    rows = []
    started_at = datetime.now(UTC).isoformat()
    start = time.monotonic()
    # Fixed number of clients; one outstanding request per client, no retries.
    iterator = iter(cases)
    with (
        concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool,
        (a.output / "responses.jsonl").open("w") as file,
    ):
        pending = {
            pool.submit(call, next(iterator)): None for _ in range(min(a.concurrency, len(cases)))
        }
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                pending.pop(future)
                row = future.result()
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                if len(rows) % 72 == 0:
                    print(f"{len(rows)}/{len(cases)} completed", flush=True)
                errors = sum(r["status"] != "ok" for r in rows)
                if time.monotonic() - start < a.max_seconds and not (
                    len(rows) >= 20 and errors / len(rows) > 0.2
                ):
                    case = next(iterator, None)
                    if case is not None:
                        pending[pool.submit(call, case)] = None
    duration = time.monotonic() - start
    report = score(cases, rows)
    lat = sorted(r["elapsed_ms"] for r in rows if r["status"] == "ok")
    report.update(
        model=model,
        dataset_revision=a.revision,
        cases_sha256=manifest["cases_sha256"],
        unique_cases=len(originals),
        full_pack_cases=total_cases,
        unsupported_cases=total_cases - len(originals),
        repeats=a.repeats,
        concurrency=a.concurrency,
        started_at=started_at,
        duration_seconds=duration,
        successful_requests_per_second=sum(r["status"] == "ok" for r in rows) / duration,
        completed_requests_per_second=len(rows) / duration,
        observed_p99_ms=lat[max(0, math.ceil(0.99 * len(lat)) - 1)] if lat else None,
        scope=(
            "Hosted network-inclusive, closed-loop clients, no retries; "
            "repeated cases are not independent quality samples"
        ),
        client="Same Mac client for all hosted lanes; provider cache/queueing uncontrolled",
        resolved_models=sorted({r["raw"].get("model", "unknown") for r in rows if "raw" in r}),
    )
    (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["details", "slices"]}, ensure_ascii=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
