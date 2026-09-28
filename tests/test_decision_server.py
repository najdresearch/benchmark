import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from najd_benchmark.decision_server import UnsupportedDecision, handler_for, sev_capability
from najd_benchmark.decisions import request_payload


def test_http_contract_preserves_arabic_and_rejects_custom_instructions():
    def predict(state, questions):
        assert state == {"text": "مرحبا"}
        assert questions["x"]["type"] == "noul"
        return {"x": False}, {"probability": 0.1}

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for("test", predict))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    case = {
        "state": {"text": "مرحبا"},
        "questions": {"x": {"type": "noul", "instructions": "Yes?"}},
    }
    payload = request_payload(case, "test")

    def call():
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req)

    try:
        with call() as response:
            result = json.load(response)
        assert json.loads(result["choices"][0]["message"]["content"]) == {"x": False}
        payload["messages"][0]["content"] = "Different instructions"
        with pytest.raises(urllib.error.HTTPError) as error:
            call()
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_invalid_probability_rejected():
    from najd_benchmark.decision_server import typed_answers

    for value in [float("nan"), float("inf"), -0.1, 1.1]:
        with pytest.raises(ValueError):
            typed_answers({"x": {"type": "noul"}}, {"answers": {"x": {"noul": value}}})


def test_diffusion_wrapper_receives_readable_arabic_after_json_transport():
    from najd_benchmark.decision_server import systemone_payload

    state = {"query": "أبي أجدّد الإقامة، رقم الطلب ١٢٣", "policy": "لا ترسل الطلب"}
    questions = {"action": {"type": "choice", "criteria": {"انتظار": "انتظر الموافقة"}}}
    body = systemone_payload(state, questions, diffusion=True)
    # HTTP JSON escaping is harmless once decoded; the model-facing state must be text.
    received = json.loads(json.dumps(body))
    wrapper_text = received["state"]
    assert isinstance(wrapper_text, str)
    assert "أبي أجدّد الإقامة" in wrapper_text
    assert "\\u062" not in wrapper_text
    assert json.loads(wrapper_text) == state
    assert received["questions"] == questions
    assert (received["samples"], received["seed"]) == (1, 42)


def test_sev_capability_checks_context_before_inference():
    class Model:
        max_questions = 8

        def __init__(self, truncated=False, error=None):
            self.truncated = truncated
            self.error = error
            self.seen = None

        def _encode(self, state, questions):
            self.seen = state
            if self.error:
                raise ValueError(self.error)
            return {}, self.truncated

    questions = {
        "action": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}
    }
    def validate(questions, limit):
        return None
    model = Model()
    assert sev_capability(model, {"text": "أهلا"}, questions, {"choice"}, validate) == {
        "supported": True,
        "reason": None,
    }
    assert "أهلا" in model.seen
    assert (
        sev_capability(Model(truncated=True), {}, questions, {"choice"}, validate)["reason"]
        == "state_truncated"
    )
    assert (
        sev_capability(
            Model(error="max_len=384 too small for 600 question tokens"),
            {},
            questions,
            {"choice"},
            validate,
        )["reason"]
        == "question_tokens_exceed_context"
    )
    assert (
        sev_capability(model, {}, {"x": {"type": "score"}}, {"choice"}, validate)["reason"]
        == "unsupported_question_type"
    )


def test_capability_endpoint_does_not_run_model():
    def predict(*args):
        raise AssertionError("Inference should not run")

    def capability(state, questions):
        return {"supported": False, "reason": "state_truncated"}

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for("sev_choice_noul", predict, capability)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    case = {
        "state": {"text": "مرحبا"},
        "questions": {"x": {"type": "noul", "instructions": "Yes?"}},
    }
    payload = request_payload(case, "sev_choice_noul")

    def call(path):
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/{path}",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req)

    try:
        with call("decision-capability") as response:
            assert json.load(response) == {"supported": False, "reason": "state_truncated"}
        with pytest.raises(urllib.error.HTTPError) as error:
            call("chat/completions")
        assert error.value.code == 500
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unsupported_decision_uses_http_422():
    def predict(*args):
        raise UnsupportedDecision("state_truncated")

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for("sev_choice_noul", predict))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    case = {"state": {}, "questions": {"x": {"type": "noul", "instructions": "Yes?"}}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        json.dumps(request_payload(case, "sev_choice_noul")).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req)
        assert error.value.code == 422
        assert json.load(error.value)["error"]["code"] == "state_truncated"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
