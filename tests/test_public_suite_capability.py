import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from najd_benchmark.decision_capability import decision_capability


def test_capability_audit_sends_only_model_input():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/decision-capability"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            user = json.loads(payload["messages"][1]["content"])
            assert set(user) == {"state", "questions"}
            assert "expected" not in user
            body = json.dumps({"supported": False, "reason": "state_truncated"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    case = {
        "state": {"text": "أهلا"},
        "questions": {"x": {"type": "noul", "instructions": "Yes?"}},
        "expected": {"x": False},
    }
    try:
        assert decision_capability(case, f"http://127.0.0.1:{server.server_port}/v1", "sev") == {
            "supported": False,
            "reason": "state_truncated",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_capability_audit_fails_closed_when_server_unavailable():
    case = {"state": {}, "questions": {"x": {"type": "noul", "instructions": "Yes?"}}}
    with pytest.raises(RuntimeError, match="Capability audit failed"):
        decision_capability(case, "http://127.0.0.1:1/v1", "sev")
