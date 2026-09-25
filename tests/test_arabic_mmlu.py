import copy

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
