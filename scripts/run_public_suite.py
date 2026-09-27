"""Evaluate a frozen, redistribution-cleared public suite without changing prompts or gold."""

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
from pathlib import Path

from najd_benchmark.decision_cli import run
from najd_benchmark.decision_server import canonical, typed_answers
from najd_benchmark.decisions import load_pack, output_schema, request_payload, score, valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8793/v1")
    p.add_argument("--hosted", action="store_true")
    p.add_argument("--credential-file", type=Path)
    p.add_argument("--types", nargs="+", default=["choice", "noul", "score"])
    p.add_argument("--max-options", type=int, default=0)
    p.add_argument("--concurrency", type=int, choices=[1, 4, 16], default=1)
    p.add_argument("--pack-seconds", type=int, default=3600)
    p.add_argument("--total-seconds", type=int, default=21600)
    a = p.parse_args()
    # This exact public draft was cleared and published before this evaluation.
    if a.revision != "84bf0a30ced090e2b552beafa5f0f3c9a3665c0c":
        p.error("Register a new immutable release explicitly before evaluating it")
    suite = json.loads((a.dataset / "suite.json").read_text())
    if len(suite["pack_paths"]) != 11:
        p.error("Expected the 11-pack public release, not an unpublished local candidate")
    key = os.environ.get("OPENROUTER_API_KEY")
    if a.hosted and not key and a.credential_file:
        match = re.search(
            r"""(?m)^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*["']?([^\s"'\n#]+)""",
            a.credential_file.read_text(),
        )
        key = match.group(1) if match else None
    if a.hosted and (
        not key or a.model not in ["typesafe/jev-1.13", "respan/span-01", "qwen/qwen3.8-27b"]
    ):
        p.error("Supported hosted model and configured key required")
    a.output.mkdir(parents=True, exist_ok=False)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k != "credential_file"
    }
    config.update(
        publication_eligible=False,
        scope="Frozen public draft quality; no prompt tuning on evaluation splits",
    )
    (a.output / "configuration.json").write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    cost = 0.0

    def call(c):
        if not a.hosted:
            return next(run([c], a.base_url, a.model, timeout=60, max_seconds=60))
        row = {"case_id": c["id"], "attempt": 1, "output": None}
        begin = time.perf_counter()
        questions = c["questions"]
        state = c["state"]
        if a.model == "respan/span-01":
            state = canonical(state)
            questions = {
                k: {**q, "criteria": q.get("criteria", {"true": "Yes", "false": "No"})}
                for k, q in questions.items()
            }
        body = {"model": a.model, "state": state, "questions": questions}
        url = "https://openrouter.ai/api/alpha/decisions"
        if a.model.startswith("qwen/"):
            body = request_payload(c, a.model)
            body.update(
                seed=42, reasoning={"enabled": False}, provider={"require_parameters": True}
            )
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "strict": True,
                    "schema": output_schema(questions),
                },
            }
            url = "https://openrouter.ai/api/v1/chat/completions"
        encoded = canonical(body).encode()
        row["request_sha256"] = hashlib.sha256(encoded).hexdigest()
        req = urllib.request.Request(
            url, encoded, {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("Oversized response")
            payload = json.loads(raw)
            row.update(raw=payload, response_sha256=hashlib.sha256(raw).hexdigest())
            if a.model.startswith("qwen/"):
                choice = payload["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Incomplete generation")
                output = json.loads(choice["message"]["content"])
            else:
                output = typed_answers(questions, payload)
            row.update(output=output, status="ok" if valid(c, output) else "invalid")
        except urllib.error.HTTPError as exc:
            row.update(status="transport_error", http_status=exc.code)
        except (TimeoutError, urllib.error.URLError):
            row["status"] = "transport_error"
        except (ValueError, KeyError, TypeError, IndexError):
            row["status"] = "invalid"
        row["elapsed_ms"] = round((time.perf_counter() - begin) * 1000, 3)
        return row

    summaries = []
    for pack in suite["pack_paths"]:
        manifest, all_cases = load_pack(a.dataset / pack, allow_public_evaluation=True)
        if manifest.get("redistribution_cleared") is not True:
            raise ValueError("Uncleared pack")
        cases = [
            c
            for c in all_cases
            if all(
                q["type"] in a.types
                and (
                    not a.max_options or q["type"] == "noul" or len(q["criteria"]) <= a.max_options
                )
                for q in c["questions"].values()
            )
        ]
        dest = a.output / pack
        dest.mkdir(parents=True)
        rows = []
        begin = time.monotonic()
        iterator = iter(cases)
        stop = "unsupported" if not cases else None
        with (
            concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool,
            (dest / "responses.jsonl").open("w") as sink,
        ):
            can_start = time.monotonic() - started < a.total_seconds and cost < 3
            pending = (
                {pool.submit(call, next(iterator)) for _ in range(min(a.concurrency, len(cases)))}
                if can_start
                else set()
            )
            if not can_start:
                stop = "total_budget"
            errors = 0
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    row = future.result()
                    rows.append(row)
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    errors += row["status"] != "ok"
                    cost += float((row.get("raw", {}).get("usage") or {}).get("cost") or 0)
                    if len(rows) >= 20 and errors / len(rows) > 0.2:
                        stop = "error_circuit"
                    if time.monotonic() - begin >= a.pack_seconds:
                        stop = "pack_budget"
                    if time.monotonic() - started >= a.total_seconds or cost >= 3:
                        stop = "total_budget"
                    if not stop:
                        c = next(iterator, None)
                        if c is not None:
                            pending.add(pool.submit(call, c))
        report = (
            score(cases, rows)
            if cases
            else {"expected": 0, "received": 0, "correct": 0, "valid": 0}
        )
        report.update(
            pack=pack,
            full_pack_cases=len(all_cases),
            unsupported_cases=len(all_cases) - len(cases),
            dataset_revision=a.revision,
            cases_sha256=manifest["cases_sha256"],
            duration_seconds=time.monotonic() - begin,
            stop_reason=stop,
            model=a.model,
            observed_api_cost_usd=cost,
            publication_eligible=False,
        )
        lat = sorted(r["elapsed_ms"] for r in rows if r["status"] == "ok")
        report["observed_p99_ms"] = lat[math.ceil(0.99 * len(lat)) - 1] if lat else None
        (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        summary = {k: v for k, v in report.items() if k not in ("details", "slices")}
        summaries.append(summary)
        (a.output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
        print(
            a.model,
            pack,
            report["correct"],
            report["valid"],
            report["received"],
            "/",
            len(cases),
            stop,
            flush=True,
        )
    (a.output / "DONE").write_text(
        "Every pack was attempted; inspect stops and coverage before claiming completion.\n"
    )


if __name__ == "__main__":
    main()
