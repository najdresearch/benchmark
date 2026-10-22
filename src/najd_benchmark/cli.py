"""Score frozen task responses from local, versioned release files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .arabic_mmlu import CASES_SHA256, score_responses


def _read_jsonl(path: Path) -> list[dict]:
    # Split on "\n" only: str.splitlines() also breaks on U+2028/U+2029/U+0085, which are legal
    # unescaped characters inside JSON strings (json.dumps(..., ensure_ascii=False) writes them).
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines if line.strip(" \t\r")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASES_SHA256:
        parser.error("cases file does not match the pinned release")
    report = score_responses(_read_jsonl(args.cases), _read_jsonl(args.responses))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
