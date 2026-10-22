import hashlib
import json
import sys

import pytest

from najd_benchmark import cli, decision_cli

LINE_SEPARATORS = ["\u2028", "\u2029", "\x85"]


@pytest.mark.parametrize("separator", LINE_SEPARATORS)
def test_scorer_cli_keeps_unicode_line_separators_inside_a_response(tmp_path, separator):
    rows = [
        {"case_id": "a", "output": f"B{separator}because"},
        {"case_id": "b", "output": "C"},
    ]
    path = tmp_path / "responses.jsonl"
    # ensure_ascii=False writes the separator raw, which is valid JSON.
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    assert separator in path.read_text(encoding="utf-8")
    assert cli._read_jsonl(path) == rows


def test_scorer_cli_tolerates_blank_lines_and_crlf(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_bytes(b'{"case_id": "a", "output": "B"}\r\n\r\n\n{"case_id": "b", "output": "C"}')
    assert [r["case_id"] for r in cli._read_jsonl(path)] == ["a", "b"]


@pytest.mark.parametrize("separator", LINE_SEPARATORS)
def test_decision_score_mode_keeps_unicode_line_separators(tmp_path, monkeypatch, separator):
    case = {
        "id": "P01-ar", "family_id": "P01", "task": "policy", "language": "ar",
        "state": {"request": "x"},
        "questions": {"eligible": {"type": "noul", "instructions": "Allowed?"}},
        "expected": {"eligible": False}, "unsafe": [], "split": "development",
    }
    pack = tmp_path / "pack"
    pack.mkdir()
    raw = (json.dumps(case) + "\n").encode()
    (pack / "cases.jsonl").write_bytes(raw)
    (pack / "manifest.json").write_text(json.dumps(
        {"cases": 1, "cases_sha256": hashlib.sha256(raw).hexdigest(), "task_pack": "t"}))
    responses = tmp_path / "responses.jsonl"
    row = {"case_id": "P01-ar", "status": "ok", "output": {"eligible": False},
           "content": f"line one{separator}line two"}
    responses.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["najd-decision", "score", "--pack", str(pack),
                                      "--output", str(out), "--responses", str(responses)])
    decision_cli.main()
    assert json.loads((out / "report.json").read_text())["correct"] == 1
    assert json.loads((out / "responses.jsonl").read_text(encoding="utf-8")) == row
