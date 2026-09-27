from najd_benchmark.typellm_adapter import questions_for


def test_rubric_semantics_and_ordinal_order_are_retained():
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Route it",
            "criteria": {"billing": "مدفوعات", "technical": "Bugs"},
        },
        "urgency": {"type": "score", "instructions": "Urgency", "criteria": ["low", "high"]},
        "eligible": {"type": "noul", "instructions": "Eligible?"},
    }
    result = questions_for(questions)
    assert result["route"]["enum"] == ["billing", "technical"]
    assert "مدفوعات" in result["route"]["instructions"]
    assert result["urgency"]["enum"] == [0, 1]
    assert '["low", "high"]' in result["urgency"]["instructions"]
    assert result["eligible"]["type"] == "boolean"
