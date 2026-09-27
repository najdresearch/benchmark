"""Version 1 schemas, content hashes, and cross-record integrity checks."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import rfc8785
from jsonschema import Draft7Validator, FormatChecker


def digest(value: object) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def schema(name: str) -> dict:
    return json.loads(
        files("najd_benchmark").joinpath("schemas", name + ".schema.json").read_text()
    )


def validate(name: str, value: dict) -> None:
    Draft7Validator(schema(name), format_checker=FormatChecker()).validate(value)


def validate_pack(manifest: dict, pack: dict, cases: list[dict], raw: bytes) -> None:
    validate("dataset-manifest", manifest)
    validate("task-pack", pack)
    for case in cases:
        validate("case", case)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)) or ids != manifest["case_ids"]:
        raise ValueError("Case IDs or ordering differ from manifest")
    if len(cases) != manifest["cases"]["count"]:
        raise ValueError("Case count differs from manifest")
    if hashlib.sha256(raw).hexdigest() != manifest["cases"]["sha256"]:
        raise ValueError("Dataset bytes differ from manifest")
    reference = pack["dataset"]
    if reference != {
        "id": manifest["id"],
        "version": manifest["version"],
        "cases_sha256": manifest["cases"]["sha256"],
    }:
        raise ValueError("Task pack references a different dataset")
    if any(c["label"] not in pack["labels"] for c in cases):
        raise ValueError("Unknown reference label")


def load_pack(directory: Path) -> tuple[dict, dict, list[dict]]:
    manifest = json.loads((directory / "dataset-manifest.json").read_text())
    validate("dataset-manifest", manifest)  # validates safe flat filename before reading
    pack = json.loads((directory / "task-pack.json").read_text())
    raw = (directory / manifest["cases"]["path"]).read_bytes()
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    validate_pack(manifest, pack, cases, raw)
    return manifest, pack, cases


def seal_bundle(bundle: dict) -> dict:
    bundle["dataset_manifest_sha256"] = digest(bundle["dataset_manifest"])
    bundle["task_pack_sha256"] = digest(bundle["task_pack"])
    bundle["bundle_sha256"] = digest({k: v for k, v in bundle.items() if k != "bundle_sha256"})
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict) -> None:
    from .support import metrics, parse_label

    validate("result-bundle", bundle)
    for field in ("dataset_manifest", "task_pack"):
        if digest(bundle[field]) != bundle[field + "_sha256"]:
            raise ValueError(f"{field} digest mismatch")
    unsigned = {k: v for k, v in bundle.items() if k != "bundle_sha256"}
    if digest(unsigned) != bundle["bundle_sha256"]:
        raise ValueError("Bundle digest mismatch")
    manifest, pack, records = bundle["dataset_manifest"], bundle["task_pack"], bundle["records"]
    ids = [r["case_id"] for r in records]
    if ids != manifest["case_ids"] or len(set(ids)) != len(ids):
        raise ValueError("Result IDs differ from assigned cases")
    if len(ids) != manifest["cases"]["count"]:
        raise ValueError("Result count differs from assigned cases")
    if pack["dataset"] != {
        "id": manifest["id"],
        "version": manifest["version"],
        "cases_sha256": manifest["cases"]["sha256"],
    }:
        raise ValueError("Task dataset identity mismatch")
    for record in records:
        if hashlib.sha256(record["output"].encode()).hexdigest() != record["response_sha256"]:
            raise ValueError("Response digest mismatch")
        if record["expected_label"] not in pack["labels"]:
            raise ValueError("Unknown reference label")
        predicted = record["predicted_label"]
        if record["status"] != "provider_error":
            parsed = parse_label(record["output"], pack["labels"])
            if predicted != parsed or (record["status"] == "ok") != (parsed is not None):
                raise ValueError("Parsed response differs from recorded prediction")
            if record["error_class"] is not None:
                raise ValueError("Successful provider response cannot carry an error")
        elif record["error_class"] is None or record["output"]:
            raise ValueError("Provider error must retain its class, not response content")
        if record["status"] == "ok" and predicted not in pack["labels"]:
            raise ValueError("Valid output must have a known label")
        if record["status"] != "ok" and predicted is not None:
            raise ValueError("Invalid/error output cannot have a prediction")
    if bundle["metrics"] != metrics(records, pack["labels"]):
        raise ValueError("Metrics differ from recorded outcomes")
