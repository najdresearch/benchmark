import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from najd_benchmark.decision_server import handler_for
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
