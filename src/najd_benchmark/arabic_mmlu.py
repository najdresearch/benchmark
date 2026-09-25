"""ArabicMMLU subset from the certified Najd text release.

This task accepts one Latin answer key or a JSON object with one answerKey field.
It scores response format as part of the task, so explanations and guessed option
text are not silently accepted as correct answers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

SOURCE_ID = "arabicmmlu"
TASK_VERSION = "arabic-mmlu-key-v1"
DATASET_REVISION = "cb30c1c9e46c62f691380c3269885cdb8f22f52b"
CASE_COUNT = 104
CASE_IDS_SHA256 = "f65459ada9049628c8c676a8c7d51a2b16736cdab67743a80a64a05b0ea19b41"
CASES_SHA256 = "b8b52ded0f6731b7a4764a90476edfe643f46d6a286096bab02b7ed189af94f6"
_KEY = re.compile(r"[A-E]")
SYSTEM_INSTRUCTION = (
    "Answer the Arabic multiple-choice question. Return only the Latin letter "
    "of the best answer, with no explanation."
)


def validate_cases(cases: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Select and validate the exact frozen task subset from the pinned release."""
    selected = [case for case in cases if case["provenance"]["sourceId"] == SOURCE_ID]
    ids = [str(case["id"]) for case in selected]
    if len(ids) != CASE_COUNT or len(set(ids)) != CASE_COUNT:
        raise ValueError("ArabicMMLU case count or IDs changed")
    id_hash = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    if id_hash != CASE_IDS_SHA256:
        raise ValueError("ArabicMMLU case IDs differ from the frozen task")
    for case in selected:
        if case.get("audit_status") != "certified":
            raise ValueError(f"non-certified task case: {case['id']}")
        expected = case["expected"]
        options = expected.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 5:
            raise ValueError(f"invalid options: {case['id']}")
        key = expected.get("answerKey")
        if not isinstance(key, str) or key not in "ABCDE"[: len(options)]:
            raise ValueError(f"invalid answer key: {case['id']}")
    return selected


def _parse_key(output: str, option_count: int) -> str | None:
    text = output.strip()
    if _KEY.fullmatch(text):
        key = text
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or set(parsed) != {"answerKey"}:
            return None
        key = parsed["answerKey"]
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            return None
    return key if key in "ABCDE"[:option_count] else None


def build_messages(case: Mapping[str, Any]) -> list[dict[str, str]]:
    """Render the question and every option in a fixed order."""
    if case["provenance"]["sourceId"] != SOURCE_ID:
        raise ValueError("case is outside this task")
    options = case["expected"]["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= 5:
        raise ValueError("case has invalid options")
    user = case["prompt"].strip() + "\n\n" + "\n".join(
        f"{key}. {option}" for key, option in zip("ABCDE", options, strict=False)
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user},
    ]


def grade(case: Mapping[str, Any], output: str) -> dict[str, Any]:
    """Grade one answer; do not accept answer leakage in explanatory text."""
    if case["provenance"]["sourceId"] != SOURCE_ID:
        raise ValueError("case is outside this task")
    expected = case["expected"]
    options = expected["options"]
    key = _parse_key(output, len(options))
    return {
        "method": TASK_VERSION,
        "score": float(key == expected["answerKey"]),
        "passed": key == expected["answerKey"],
        "parsed_key": key,
        "valid_output": key is not None,
    }


def score_responses(
    cases: Iterable[Mapping[str, Any]], responses: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score a complete or partial response set with missing cases in the denominator."""
    selected = validate_cases(cases)
    by_id: dict[str, str] = {}
    for response in responses:
        case_id = response.get("case_id")
        output = response.get("output")
        if not isinstance(case_id, str) or case_id in by_id:
            raise ValueError("response has a missing or duplicate case ID")
        if not isinstance(output, str):
            raise ValueError(f"response output is not text: {case_id}")
        by_id[case_id] = output
    selected_ids = {str(case["id"]) for case in selected}
    if extra := set(by_id) - selected_ids:
        raise ValueError(f"response includes an unknown case ID: {sorted(extra)[0]}")
    passed = valid = 0
    for case in selected:
        result = grade(case, by_id.get(str(case["id"]), ""))
        passed += result["passed"]
        valid += result["valid_output"]
    return {
        "task_version": TASK_VERSION,
        "dataset_revision": DATASET_REVISION,
        "case_ids_sha256": CASE_IDS_SHA256,
        "cases": CASE_COUNT,
        "submitted": len(by_id),
        "correct": passed,
        "valid_outputs": valid,
        "accuracy": passed / CASE_COUNT,
        "coverage": len(by_id) / CASE_COUNT,
        "invalid_or_missing": CASE_COUNT - valid,
    }
