"""Read-only capability checks against an already loaded loopback decision model."""

import json
import urllib.error
import urllib.request

from .decision_server import canonical
from .decisions import request_payload


def decision_capability(case, base_url, model):
    """Ask whether a case fits without sending gold labels or invoking inference."""
    body = canonical(request_payload(case, model)).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/decision-capability",
        body,
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Capability audit failed; refusing to score a partial subset") from exc
    if type(result.get("supported")) is not bool or (
        not result["supported"] and not isinstance(result.get("reason"), str)
    ):
        raise RuntimeError("Invalid model capability response")
    return result
