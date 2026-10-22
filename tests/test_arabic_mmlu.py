import copy
import hashlib

import pytest

from najd_benchmark.arabic_mmlu import build_messages, grade, validate_cases

CASE = {
    "id": "example", "provenance": {"sourceId": "arabicmmlu"},
    "expected": {"answerKey": "B", "options": ["one", "two", "three"]},
}


@pytest.mark.parametrize("output", ["B", ' {"answerKey": "B"} '])
def test_exact_key_passes(output):
    assert grade(CASE, output) == {
        "method": "arabic-mmlu-key-v1", "score": 1.0, "passed": True,
        "parsed_key": "B", "valid_output": True,
    }


@pytest.mark.parametrize("output", ["A", "C"])
def test_wrong_valid_key_fails(output):
    result = grade(CASE, output)
    assert result["score"] == 0.0 and result["valid_output"]


@pytest.mark.parametrize(
    "output",
    ["", "B because ...", "A or B", "D", "b", '{"answerKey":"B","reason":"x"}',
     '{"answerKey":2}', "The answer is B"],
)
def test_ambiguous_or_invalid_output_fails(output):
    result = grade(CASE, output)
    assert result["score"] == 0.0 and not result["valid_output"]


def test_other_source_rejected():
    case = copy.deepcopy(CASE)
    case["provenance"]["sourceId"] = "other"
    with pytest.raises(ValueError, match="outside this task"):
        grade(case, "B")


def test_frozen_subset_rejects_missing_rows():
    with pytest.raises(ValueError, match="count or IDs changed"):
        validate_cases([CASE])


def test_prompt_includes_ordered_options():
    case = copy.deepcopy(CASE)
    case["prompt"] = "Question?"
    assert build_messages(case)[1]["content"] == "Question?\n\nA. one\nB. two\nC. three"


def test_cases_without_review_metadata_are_eligible(monkeypatch):
    from najd_benchmark import arabic_mmlu

    monkeypatch.setattr(arabic_mmlu, "CASE_COUNT", 1)
    monkeypatch.setattr(
        arabic_mmlu, "CASE_IDS_SHA256", hashlib.sha256(b"example\n").hexdigest()
    )
    value = copy.deepcopy(CASE)
    value.pop("audit_status", None)
    value.pop("review_status", None)
    assert validate_cases([value]) == [value]


@pytest.mark.parametrize(
    "output",
    ["1" * 5000, "[" * 100000, '{"answerKey":' * 100000],
    ids=["digit-limit", "array-nesting", "object-nesting"],
)
def test_hostile_json_output_is_invalid_not_a_crash(output):
    result = grade(CASE, output)
    assert result["score"] == 0.0 and not result["valid_output"]


def test_hostile_output_does_not_abort_scoring(monkeypatch):
    from najd_benchmark import arabic_mmlu

    monkeypatch.setattr(arabic_mmlu, "CASE_COUNT", 2)
    monkeypatch.setattr(
        arabic_mmlu, "CASE_IDS_SHA256", hashlib.sha256(b"example\nexample-2\n").hexdigest()
    )
    other = copy.deepcopy(CASE)
    other["id"] = "example-2"
    report = arabic_mmlu.score_responses(
        [CASE, other],
        [
            {"case_id": "example", "output": "[" * 100000},
            {"case_id": "example-2", "output": "B"},
        ],
    )
    assert (report["correct"], report["valid_outputs"], report["invalid_or_missing"]) == (1, 1, 1)
