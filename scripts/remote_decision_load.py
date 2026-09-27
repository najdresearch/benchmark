"""Measure serialized FP32 GPU decision endpoints with 1, 4 and 16 closed-loop clients."""

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

from huggingface_hub import snapshot_download

from najd_benchmark.decision_cli import run
from najd_benchmark.decisions import load_pack, score
from najd_benchmark.local_backends import PINS
from najd_benchmark.native_gliner import MODELS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--pack", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--revision", required=True)
a = p.parse_args()
manifest, originals = load_pack(a.pack)
assert len(originals) == 216
root = a.output
root.mkdir(parents=True, exist_ok=False)
systems = ["gliner", "laya_english", "laya_multi"]
for name in systems:
    model, rev = (MODELS | PINS)[name]
    snapshot_download(
        model,
        revision=rev,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
hardware = {
    "dataset_revision": a.revision,
    "cases_sha256": manifest["cases_sha256"],
    "platform": platform.platform(),
    "cpu": subprocess.check_output(["lscpu"], text=True),
    "gpu": subprocess.check_output(["nvidia-smi"], text=True),
    "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    "model_pins": {n: (MODELS | PINS)[n] for n in systems},
    "precision": "FP32, Laya autocast disabled",
    "scope": "Serialized adapter, no batching; loopback HTTP includes lock queue time",
}
(root / "environment.json").write_text(json.dumps(hardware, indent=2))
for name in systems:
    out = root / name
    out.mkdir()
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
                sys.executable,
                "-m",
                "najd_benchmark.decision_server",
                "--system",
                name,
                "--device",
                "cuda",
                "--precision",
                "fp32",
                "--port",
                "8793",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while time.monotonic() - start < 90 and server.poll() is None:
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
            for concurrency in [1, 4, 16]:
                lane = out / ("c" + str(concurrency))
                lane.mkdir()
                cases = [
                    {**c, "id": c["id"] + f"__repeat{r}", "source_case_id": c["id"]}
                    for r in range(3)
                    for c in originals
                ]

                def call(c, system=name):
                    row = next(
                        run([c], "http://127.0.0.1:8793/v1", system, timeout=20, max_seconds=20)
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
                            if time.monotonic() - started < 180 and not (
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
                    unique_cases=216,
                    repeats=3,
                    precision="fp32",
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
        finally:
            server.kill()
            server.wait()
