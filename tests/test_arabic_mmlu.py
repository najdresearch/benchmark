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


def test_review_status_does_not_control_eligibility(monkeypatch):
    from najd_benchmark import arabic_mmlu

    monkeypatch.setattr(arabic_mmlu, "CASE_COUNT", 1)
    monkeypatch.setattr(
        arabic_mmlu, "CASE_IDS_SHA256", hashlib.sha256(b"example\n").hexdigest()
    )
    value = copy.deepcopy(CASE)
    value["audit_status"] = "certified"
    value["review_status"] = "not_reviewed"
    assert validate_cases([value]) == [value]
