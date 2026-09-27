"""Loopback Chat Completions subset for typed decisions; not a general chat server."""

import argparse
import json
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .decisions import valid
from .local_backends import PINS
from .native_gliner import MODELS, normalize


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def typed_answers(questions, raw):
    answers = raw["answers"]
    output = {}
    for key, q in questions.items():
        a = answers[key]
        if q["type"] == "choice":
            output[key] = a["choice"]
        elif q["type"] == "noul":
            probability = float(a["noul"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid Boolean probability")
            output[key] = probability >= 0.5
        else:
            output[key] = max(
                range(len(q["criteria"])), key=lambda i: float(a["probabilities"][str(i)])
            )
    return output


def handler_for(model_id, predict):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": model_id, "object": "model"}]})
            else:
                self.reply(404, {"error": {"message": "Unknown path"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": {"message": "Unknown path"}})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 1048576:
                    raise ValueError("Invalid request size")
                req = json.loads(self.rfile.read(n))
                if req["model"] != model_id or req.get("stream", False):
                    raise ValueError("Unknown model or unsupported streaming")
                if req.get("temperature", 0) != 0:
                    raise ValueError("Only deterministic mode supported")
                messages = req["messages"]
                if (
                    len(messages) != 2
                    or messages[0]["role"] != "system"
                    or messages[1]["role"] != "user"
                ):
                    raise ValueError("Expected benchmark system message and typed user payload")
                from .decisions import request_payload

                expected_system = request_payload({"state": {}, "questions": {}}, model_id)[
                    "messages"
                ][0]
                if messages[0] != expected_system:
                    raise ValueError("Unsupported system instructions")
                data = json.loads(messages[1]["content"])
                if set(data) != {"state", "questions"} or not data["questions"]:
                    raise ValueError("Expected state and questions only")
                # Validate task schema before allowing model work.
                from .decisions import first_option

                if not valid(data, first_option(data)):
                    raise ValueError("Invalid question schema")
                started = time.perf_counter()
                with lock:
                    output, raw = predict(
                        json.loads(canonical(data["state"])),
                        json.loads(canonical(data["questions"])),
                    )
                if not valid(data, output):
                    raise ValueError("Model output failed validation")
                self.reply(
                    200,
                    {
                        "id": "chatcmpl-" + uuid.uuid4().hex,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": canonical(output)},
                            }
                        ],
                        "najd": {
                            "raw": raw,
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                            "serialization": "sorted-object-keys-readable-unicode-v1",
                        },
                    },
                )
            except (ValueError, KeyError, TypeError, IndexError):
                self.reply(400, {"error": {"message": "Invalid or unsupported decision request"}})
            except Exception as exc:
                self.reply(500, {"error": {"message": type(exc).__name__}})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=[
            *MODELS,
            *PINS,
            "julia",
            "sev",
            "jev_api",
            "span_api",
            "typellm_direct",
            "typellm_thinking64",
        ],
        required=True,
    )
    parser.add_argument("--model-path")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.system in ("typellm_direct", "typellm_thinking64"):
        from dataclasses import asdict

        from typellm import TypeLLMClient

        from .typellm_adapter import questions_for

        if not args.model_path:
            parser.error("TypeLLM needs --model-path for the pinned upstream model")
        thinking = args.system == "typellm_thinking64"
        client = TypeLLMClient(
            "http://127.0.0.1:8770",
            model=args.model_path,
            tokenizer=args.model_path,
            timeout=20,
            seed=42,
            thinking_budget=64 if thinking else None,
        )

        def predict(state, questions):
            mapped = questions_for(questions)
            for question in mapped.values():
                question["thinking"] = thinking
            output = client.generate(context=canonical(state), questions=mapped)
            return output, {
                "usage": asdict(client.last_usage) if client.last_usage else None,
                "thinking": client.last_thinking,
                "prompts": client.last_prompts,
            }
    elif args.system in ("jev_api", "span_api"):
        import os
        import urllib.request

        model_id = "typesafe/jev-1.13" if args.system == "jev_api" else "respan/span-01"
        key = os.environ["OPENROUTER_API_KEY"]

        def predict(state, questions):
            if args.system == "span_api" and any(q["type"] != "noul" for q in questions.values()):
                raise ValueError("Span-01 endpoint only supports Boolean questions")
            if args.system == "span_api":
                questions = {
                    k: {**q, "criteria": q.get("criteria", {"true": "Yes", "false": "No"})}
                    for k, q in questions.items()
                }
            payload = {
                "model": model_id,
                "state": canonical(state) if args.system == "span_api" else state,
                "questions": questions,
            }
            request = urllib.request.Request(
                "https://openrouter.ai/api/alpha/decisions",
                canonical(payload).encode(),
                {"Content-Type": "application/json", "Authorization": "Bearer " + key},
            )
            with urllib.request.urlopen(request, timeout=12) as response:
                raw = json.load(response)
            return typed_answers(questions, raw), raw
    elif args.system in PINS:
        from .local_backends import load

        predict = load(args.system, device=args.device)
    elif args.system == "julia":
        from julia import load_model

        model = load_model(
            args.model_path, device="cpu", strict_encoding=True, max_length=8192, head_length=512
        )

        def predict(state, questions):
            with torch.inference_mode():
                raw = model.predict(state=canonical(state), questions=questions)
            return typed_answers(questions, raw), raw
    elif args.system == "sev":
        from sev_preview import SevPreview

        model = SevPreview(args.model_path, device="cpu")

        def predict(state, questions):
            if any(q["type"] != "choice" for q in questions.values()):
                raise ValueError("Sev preview supports only choice in this benchmark")
            raw = model.decide(canonical(state), questions)
            if raw.get("meta", {}).get("state_truncated"):
                raise ValueError("State would be truncated")
            return typed_answers(questions, raw), raw
    else:
        from gliner2 import AutoExtractor
        from huggingface_hub import snapshot_download

        from .native_gliner import tasks_for

        mid, rev = MODELS[args.system]
        model = (
            AutoExtractor.from_pretrained(
                snapshot_download(mid, revision=rev, local_files_only=True)
            )
            .to(args.device or "cpu")
            .eval()
        )

        def predict(state, questions):
            with torch.inference_mode():
                raw = model.classify_text(
                    canonical(state), tasks_for(questions), include_confidence=True
                )
            return normalize(questions, raw), raw

    print("READY", args.system, flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(args.system, predict)).serve_forever()


if __name__ == "__main__":
    main()
