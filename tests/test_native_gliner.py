import pytest

from najd_benchmark.native_gliner import normalize, predict, tasks_for

QUESTIONS = {
    "gate": {"type": "noul", "instructions": "Is it allowed?"},
    "urgency": {"type": "score", "instructions": "Urgency", "criteria": ["low", "high"]},
}


def test_native_types_and_invalid_labels():
    assert normalize(QUESTIONS, {"gate": "false", "urgency": "1"}) == {
        "gate": False,
        "urgency": 1,
    }
    for raw in ({"gate": True, "urgency": "1"}, {"gate": "false", "urgency": "01"}, {}):
        with pytest.raises(ValueError):
            normalize(QUESTIONS, raw)


def test_native_boundary_preserves_unicode_and_criteria():
    class Model:
        def classify_text(self, text, tasks, include_confidence):
            assert "مرحبا" in text and "expected" not in text
            assert tasks == tasks_for(QUESTIONS)
            assert include_confidence
            return {"gate": {"label": "false", "confidence": 0.8}, "urgency": "0"}

    output, raw, digest = predict(Model(), {"query": "مرحبا"}, QUESTIONS)
    assert output == {"gate": False, "urgency": 0}
    assert len(digest) == 64


def test_serialization_is_independent_of_object_insertion_order():
    calls = []

    class Model:
        def classify_text(self, text, tasks, include_confidence):
            calls.append((text, list(tasks)))
            return {"gate": "false", "urgency": "0"}

    a = predict(Model(), {"z": "مرحبا", "a": 1}, QUESTIONS)
    b = predict(Model(), {"a": 1, "z": "مرحبا"}, dict(reversed(list(QUESTIONS.items()))))
    assert calls[0] == calls[1]
    assert a == b
