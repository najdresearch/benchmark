"""Sequential same-host CPU/CUDA smoke; run inside the rented machine."""

import argparse
import hashlib
import json
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pack", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dataset-revision", required=True)
parser.add_argument("--max-seconds", type=int, default=600)
parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
parser.add_argument(
    "--systems",
    nargs="+",
    choices=["gliner", "laya_english", "laya_multi"],
    default=["gliner", "laya_english", "laya_multi"],
)
parser.add_argument("--precision", choices=["native", "fp32"], default="native")
args = parser.parse_args()
manifest, cases = load_pack(args.pack)
cases_bytes = (args.pack / "cases.jsonl").read_bytes()
root = args.output
root.mkdir(parents=True, exist_ok=False)
systems = args.systems
for name in systems:
    mid, rev = (MODELS | PINS)[name]
    snapshot_download(
        mid, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
    )
hardware = {
    "dataset_revision": args.dataset_revision,
    "pack": manifest["task_pack"],
    "platform": platform.platform(),
    "cpu": subprocess.check_output(["lscpu"], text=True),
    "gpu": subprocess.check_output(["nvidia-smi"], text=True),
    "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
    "model_pins": {n: (MODELS | PINS)[n] for n in systems},
}
(root / "environment.json").write_text(json.dumps(hardware, indent=2))
for device in args.devices:
    for name in systems:
        out = root / (name + "-" + device)
        out.mkdir(exist_ok=False)
        start = time.monotonic()
        rows = []
        with (out / "server.log").open("w") as log:
            env = {
                **os.environ,
                "PYTHONPATH": "src",
                "HF_HUB_OFFLINE": "1",
                "USE_TF": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4",
            }
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "najd_benchmark.decision_server",
                    "--system",
                    name,
                    "--device",
                    device,
                    "--precision",
                    args.precision,
                    "--port",
                    "8793",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            try:
                while time.monotonic() - start < 90 and p.poll() is None:
                    try:
                        with urllib.request.urlopen(
                            "http://127.0.0.1:8793/v1/models", timeout=1
                        ) as r:
                            json.load(r)
                        break
                    except Exception:
                        time.sleep(0.2)
                else:
                    raise RuntimeError("Startup failure")
                ready = time.monotonic() - start
                warm = list(
                    run(
                        [cases[0]] * 3, "http://127.0.0.1:8793/v1", name, timeout=15, max_seconds=45
                    )
                )
                (out / "warmup.json").write_text(json.dumps(warm, ensure_ascii=False))
                if any(r["status"] != "ok" for r in warm):
                    raise RuntimeError("Warmup failure")
                rows = list(
                    run(
                        cases,
                        "http://127.0.0.1:8793/v1",
                        name,
                        timeout=15,
                        max_seconds=args.max_seconds,
                    )
                )
                report = score(cases, rows)
                report.update(
                    system=name,
                    device=device,
                    threads=4,
                    precision=args.precision,
                    concurrency=1,
                    warmup_requests=3,
                    ready_seconds=ready,
                )
            except Exception as e:
                report = {"system": name, "device": device, "failure": str(e)}
            finally:
                p.kill()
                p.wait()
        (out / "responses.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ["details", "slices"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
