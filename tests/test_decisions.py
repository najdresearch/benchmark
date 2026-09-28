import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from najd_benchmark.decision_cli import endpoint_url, run
from najd_benchmark.decisions import request_payload, score, valid


@pytest.fixture
def case():
    return {
        "id": "P01-ar",
        "family_id": "P01",
        "task": "policy",
        "language": "ar",
        "state": {"request": "هل أقدر أسترجع المبلغ؟"},
        "questions": {"eligible": {"type": "noul", "instructions": "Allowed?"}},
        "expected": {"eligible": False},
        "unsafe": [],
        "split": "development",
    }


def test_strict_boolean_and_all_case_denominator(case):
    assert not valid(case, {"eligible": 0})
    assert not valid(case, {"eligible": "false"})
    assert not valid(case, {"eligible": False, "extra": True})
    assert score([case], [])["statuses"] == {"missing": 1}
    report = score(
        [case], [{"case_id": case["id"], "status": "timeout", "output": case["expected"]}]
    )
    assert report["correct"] == 0
    assert report["expected"] == 1
    assert not report["publication_eligible"]


def test_unknown_duplicate_and_bad_time_rejected(case):
    r = {"case_id": case["id"], "status": "ok", "output": case["expected"]}
    for rows in ([r, r], [{**r, "case_id": "foreign"}], [{**r, "elapsed_ms": float("nan")}]):
        with pytest.raises(ValueError):
            score([case], rows)


def test_request_never_contains_gold_or_escaped_arabic(case):
    payload = request_payload(case, "fake")
    user = payload["messages"][1]["content"]
    assert "\\u" not in user
    assert set(json.loads(user)) == {"state", "questions"}
    assert json.loads(user)["state"] == case["state"]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://localhost/v1",
        "http://user:secret@127.0.0.1/v1",
        "http://127.0.0.1/v1?key=secret",
    ],
)
def test_only_literal_loopback(url):
    with pytest.raises(ValueError):
        endpoint_url(url)


def test_local_endpoint_success_invalid_timeout_and_budget(case):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            mode = request["model"]
            if mode == "slow":
                time.sleep(0.15)
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "http://example.com")
                self.end_headers()
                return
            if mode in ("unsupported", "server_error"):
                status = 422 if mode == "unsupported" else 500
                data = json.dumps({"error": {"code": "state_truncated"}}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            content = '{"eligible": false}' if mode != "invalid" else '{"eligible": "false"}'
            data = json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        ok = list(run([case], url, "ok"))
        assert score([case], ok)["correct"] == 1
        assert ok[0]["elapsed_ms"] > 0
        assert len(ok[0]["request_sha256"]) == 64
        assert list(run([case], url, "invalid"))[0]["status"] == "invalid"
        assert list(run([case], url, "redirect"))[0]["status"] == "redirect_refused"
        unsupported = list(run([case], url, "unsupported"))[0]
        assert (unsupported["status"], unsupported["http_status"], unsupported["error_code"]) == (
            "unsupported",
            422,
            "state_truncated",
        )
        assert list(run([case], url, "server_error"))[0]["status"] == "server_error"
        second = copy.deepcopy(case)
        second["id"] = "P02-ar"
        rows = list(run([case, second], url, "slow", timeout=0.02, max_seconds=0.01))
        assert [r["status"] for r in rows] == ["timeout", "budget_exhausted"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_registers_are_separate_and_unannotated_safety_is_visible(case):
    msa = {**case, "id": "a", "register": "ar-MSA", "unsafe_annotation_status": "not_annotated"}
    saudi = {**case, "id": "b", "register": "ar-SA", "unsafe_annotation_status": "not_annotated"}
    result = score([msa, saudi], [{"case_id": "a", "status": "ok", "output": case["expected"]}])
    assert result["register_slices"] == {
        "ar-MSA": {"correct": 1, "total": 1},
        "ar-SA": {"correct": 0, "total": 1},
    }
    assert result["unsafe_annotation_cases"] == 0


def test_boolean_unsafe_annotations_do_not_match_integer(case):
    case = {
        **case,
        "unsafe_values": {"eligible": [True]},
        "unsafe_annotation_status": "draft_annotated",
    }

    def result(value):
        return score(
            [case], [{"case_id": case["id"], "status": "ok", "output": {"eligible": value}}]
        )

    assert result(True)["unsafe_choices"] == 1
    assert result(1)["unsafe_choices"] == 0
    assert result(False)["unsafe_choices"] == 0


def test_public_evaluation_requires_explicit_opt_in_and_clearance(tmp_path, case):
    import hashlib

    from najd_benchmark.decisions import load_pack

    case["split"] = "reserved_evaluation"
    data = (json.dumps(case) + "\n").encode()
    (tmp_path / "cases.jsonl").write_bytes(data)
    manifest = {"cases": 1, "cases_sha256": hashlib.sha256(data).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_pack(tmp_path, allow_public_evaluation=True)
    manifest["redistribution_cleared"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_pack(tmp_path)
    assert load_pack(tmp_path, allow_public_evaluation=True)[1] == [case]
