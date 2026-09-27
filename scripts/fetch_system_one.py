"""Fetch an immutable public System One release and verify every declared file."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def verify(root):
    lock = json.loads((root / "release-lock.json").read_text())
    for name, expected in lock["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Release hash mismatch: " + name)
    return lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="najdresearch/system-one")
    p.add_argument("--revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", a.revision):
        p.error("Use an immutable 40-character commit SHA")
    from huggingface_hub import snapshot_download

    snapshot_download(a.repo, repo_type="dataset", revision=a.revision, local_dir=a.output)
    lock = verify(a.output)
    print(
        json.dumps(
            {
                "repo": a.repo,
                "revision": a.revision,
                "cases": lock["cases"],
                "files_verified": len(lock["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
