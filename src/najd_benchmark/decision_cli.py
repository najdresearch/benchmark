"""Local development slice: bounded loopback endpoint execution and saved-response scoring."""

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .decisions import first_option, load_pack, request_payload, score, valid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def endpoint_url(base_url):
    url = urllib.parse.urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "::1")
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("This development runner accepts only literal loopback HTTP endpoints")
    return base_url.rstrip("/") + "/chat/completions"


def run(cases, base_url, model, timeout=10.0, max_seconds=120.0):
    url = endpoint_url(base_url)
    if any(not math.isfinite(v) or v <= 0 for v in (timeout, max_seconds)):
        raise ValueError("Timeout and run budget must be positive finite seconds")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + max_seconds
    for case in cases:
        remaining = deadline - time.monotonic()
        start = time.perf_counter()
        row = {
            "case_id": case["id"],
            "attempt": 1,
            "output": None,
            "status": "budget_exhausted",
            "elapsed_ms": None,
        }
        if remaining <= 0:
            yield row
            continue
        body = json.dumps(request_payload(case, model), ensure_ascii=False).encode()
        row["request_sha256"] = hashlib.sha256(body).hexdigest()
        request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=min(timeout, remaining)) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("Response exceeds 1 MiB")
            row["response_sha256"] = hashlib.sha256(raw).hexdigest()
            payload = json.loads(raw)
            choice = payload["choices"][0]
            row["content"] = choice["message"]["content"]
            row["usage"] = payload.get("usage")
            row["adapter_evidence"] = payload.get("najd")
            row["output"] = json.loads(row["content"])
            row["status"] = (
                "ok"
                if choice.get("finish_reason") == "stop" and valid(case, row["output"])
                else "invalid"
            )
        except (TimeoutError, urllib.error.URLError) as exc:
            row["status"] = (
                "timeout"
                if isinstance(exc, TimeoutError)
                or isinstance(getattr(exc, "reason", None), TimeoutError)
                else "transport_error"
            )
        except (ValueError, KeyError, IndexError, TypeError):
            row["status"] = "invalid"
        row["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "score", "run"])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="local-development")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--max-cases", type=int, default=48)
    args = parser.parse_args()
    manifest, cases = load_pack(args.pack)
    if args.mode == "run" and (not args.base_url or len(cases) > args.max_cases):
        parser.error("Run needs --base-url and a case cap covering the pack")
    if args.mode == "score" and not args.responses:
        parser.error("Score needs --responses")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "baseline":
        rows = [{"case_id": c["id"], "status": "ok", "output": first_option(c)} for c in cases]
    elif args.mode == "score":
        rows = [json.loads(line) for line in args.responses.read_text().splitlines() if line]
    else:
        rows = run(cases, args.base_url, args.model, args.timeout, args.max_seconds)
    saved = []
    with (args.output / "responses.jsonl").open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            saved.append(row)
    report = score(cases, saved)
    report["task_pack"] = manifest["task_pack"]
    report["cases_sha256"] = manifest["cases_sha256"]
    report["configuration"] = {
        "mode": args.mode,
        "model": args.model,
        "endpoint_sha256": hashlib.sha256((args.base_url or "").encode()).hexdigest(),
        "timeout_seconds": args.timeout,
        "max_seconds": args.max_seconds,
        "max_cases": args.max_cases,
        "attempts_per_case": 1,
    }
    report["result_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()
    ).hexdigest()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ["expected", "correct", "statuses", "publication_eligible"]}
        )
    )


if __name__ == "__main__":
    main()
