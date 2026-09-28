"""Run a pinned CPU/GPU roster sequentially on a rented Linux machine."""

import argparse
import concurrent.futures
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from najd_benchmark.decision_cli import run
from najd_benchmark.decision_server import CONCURRENT_UPSTREAM_SYSTEMS
from najd_benchmark.decisions import load_pack, score

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--pack", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--revision", required=True)
p.add_argument("--plan", type=Path, required=True)
a = p.parse_args()
if platform.system() != "Linux":
    p.error("Remote Linux runner only; heavy laptop inference is disabled")
manifest, originals = load_pack(a.pack)
assert len(originals) == 216
all_cases = originals
plan = json.loads(a.plan.read_text())
root = a.output
root.mkdir(parents=True, exist_ok=False)
hardware = {
    "dataset_revision": a.revision,
    "cases_sha256": manifest["cases_sha256"],
    "platform": platform.platform(),
    "cpu": subprocess.check_output(["lscpu"], text=True),
    "gpu": subprocess.check_output(["nvidia-smi"], text=True),
    "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    "plan": plan,
    "precision": "Per configuration in plan",
    "scope": (
        "Loopback HTTP includes adapter and upstream time; "
        "inspect each system's adapter_serialized flag"
    ),
}
(root / "environment.json").write_text(json.dumps(hardware, indent=2))
for spec in plan:
    name = spec["system"]
    originals = [
        c
        for c in all_cases
        if all(
            q["type"] in spec.get("types", ["choice", "noul", "score"])
            for q in c["questions"].values()
        )
    ]
    out = root / spec["id"]
    out.mkdir()
    (out / "packages.txt").write_text(
        subprocess.check_output([spec["python"], "-m", "pip", "freeze"], text=True)
    )
    start = time.monotonic()
    with (out / "server.log").open("w") as log:
        env = {
            **os.environ,
            "PYTHONPATH": "src",
            "HF_HUB_OFFLINE": "1",
            "USE_TF": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
        }
        server = subprocess.Popen(
            [
                spec["python"],
                "-m",
                "najd_benchmark.decision_server",
                "--system",
                name,
                "--device",
                spec["device"],
                "--precision",
                spec["precision"],
                "--port",
                "8793",
            ]
            + (["--model-path", spec["model_path"]] if spec.get("model_path") else [])
            + (["--upstream", spec["upstream"]] if spec.get("upstream") else []),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while time.monotonic() - start < 180 and server.poll() is None:
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:8793/v1/models", timeout=1
                    ) as response:
                        json.load(response)
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Startup failed")
            ready = time.monotonic() - start
            warm = list(
                run(
                    [originals[0]] * 3,
                    "http://127.0.0.1:8793/v1",
                    name,
                    timeout=20,
                    max_seconds=60,
                )
            )
            (out / "warmup.json").write_text(json.dumps(warm, ensure_ascii=False))
            if any(r["status"] != "ok" for r in warm):
                raise RuntimeError("Warmup failed")
            for concurrency in spec.get("concurrencies", [1, 4, 16]):
                lane = out / ("c" + str(concurrency))
                lane.mkdir()
                cases = [
                    {**c, "id": c["id"] + f"__repeat{r}", "source_case_id": c["id"]}
                    for r in range(spec.get("repeats", 3))
                    for c in originals
                ]

                def call(c, system=name):
                    row = next(
                        run([c], "http://127.0.0.1:8793/v1", system, timeout=60, max_seconds=60)
                    )
                    row["source_case_id"] = c["source_case_id"]
                    return row

                rows = []
                started = time.monotonic()
                iterator = iter(cases)
                with (
                    concurrent.futures.ThreadPoolExecutor(concurrency) as pool,
                    (lane / "responses.jsonl").open("w") as file,
                ):
                    pending = {pool.submit(call, next(iterator)): None for _ in range(concurrency)}
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
                            errors = sum(r["status"] != "ok" for r in rows)
                            if time.monotonic() - started < spec.get("max_seconds", 600) and not (
                                len(rows) >= 20 and errors / len(rows) > 0.2
                            ):
                                case = next(iterator, None)
                                if case is not None:
                                    pending[pool.submit(call, case)] = None
                duration = time.monotonic() - started
                report = score(cases, rows)
                lat = sorted(r["elapsed_ms"] for r in rows if r["status"] == "ok")
                report.update(
                    system=name,
                    concurrency=concurrency,
                    unique_cases=len(originals),
                    full_pack_cases=len(all_cases),
                    unsupported_cases=len(all_cases) - len(originals),
                    repeats=spec.get("repeats", 3),
                    precision=spec["precision"],
                    device=spec["device"],
                    adapter_serialized=name not in CONCURRENT_UPSTREAM_SYSTEMS,
                    ready_seconds=ready,
                    duration_seconds=duration,
                    successful_requests_per_second=sum(r["status"] == "ok" for r in rows)
                    / duration,
                    observed_p99_ms=lat[max(0, math.ceil(0.99 * len(lat)) - 1)] if lat else None,
                )
                (lane / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(
                    json.dumps(
                        {k: v for k, v in report.items() if k not in ["details", "slices"]},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if spec.get("full_suite"):
                command = [
                    sys.executable,
                    "scripts/run_public_suite.py",
                    "--dataset",
                    spec["full_suite"],
                    "--revision",
                    a.revision,
                    "--output",
                    str(out / "full-quality"),
                    "--model",
                    name,
                    "--total-seconds",
                    "3600",
                    "--pack-seconds",
                    "900",
                    "--types",
                    *spec.get("types", ["choice", "noul", "score"]),
                ]
                if spec.get("max_options"):
                    command += ["--max-options", str(spec["max_options"])]
                subprocess.run(command, env=env, check=True)
        except Exception as exc:
            (out / "failure.json").write_text(
                json.dumps({"error_type": type(exc).__name__, "message": str(exc)[:200]})
            )
            print(spec["id"], type(exc).__name__, flush=True)
        finally:
            server.kill()
            server.wait()
