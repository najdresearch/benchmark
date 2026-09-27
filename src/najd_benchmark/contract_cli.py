"""Validate and exercise versioned dataset, task, and result contracts."""

import argparse
import json
import os
from pathlib import Path

from .contracts import load_pack, validate_bundle
from .support import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-pack")
    validate.add_argument("directory", type=Path)
    bundle = commands.add_parser("validate-bundle")
    bundle.add_argument("path", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("--pack", required=True, type=Path)
    execute.add_argument("--endpoint", required=True)
    execute.add_argument("--model", required=True)
    execute.add_argument("--api-key-env")
    execute.add_argument("--endpoint-kind", choices=["local", "remote"], default="remote")
    execute.add_argument("--benchmark-revision", default="unknown")
    execute.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "validate-pack":
        load_pack(args.directory)
        print("Task pack and dataset verified")
    elif args.command == "validate-bundle":
        validate_bundle(json.loads(args.path.read_text()))
        print("Bundle structure, hashes, case accounting and metrics verified; origin is untrusted")
    else:
        if args.output.exists():
            parser.error("Output already exists; use a new result path")
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if args.api_key_env and not api_key:
            parser.error("Configured API key environment variable is missing")
        manifest, pack, cases = load_pack(args.pack)
        result = run(
            manifest,
            pack,
            cases,
            endpoint=args.endpoint,
            model=args.model,
            api_key=api_key,
            endpoint_kind=args.endpoint_kind,
            benchmark_revision=args.benchmark_revision,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"Saved private development bundle: {args.output}")
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
