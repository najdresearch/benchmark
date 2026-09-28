"""Audit Sev's pinned input limits with its tokenizer; never load model weights."""

import argparse
import json
from collections import Counter
from pathlib import Path

from sev_preview.runtime import state_to_text, td_data, validate_questions
from transformers import AutoTokenizer

from najd_benchmark.decision_server import sev_capability
from najd_benchmark.decisions import load_pack


class EncoderOnly:
    def __init__(self, model_dir):
        config = json.loads((model_dir / "sev_preview_config.json").read_text())
        self.max_questions = config["max_questions"]
        self.max_input_tokens = config["max_input_tokens"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir / "tokenizer", local_files_only=True
        )
        _, self.qid, self.lid = td_data.add_markers(self.tokenizer)

    def _encode(self, state, questions):
        row = {"state": state, "questions": questions}
        enc = td_data.encode(
            row, self.tokenizer, self.max_input_tokens, self.qid, self.lid, with_gold=False
        )
        state_tokens = len(self.tokenizer.encode(state_to_text(state), add_special_tokens=False))
        full = td_data.encode(
            row,
            self.tokenizer,
            len(enc["input_ids"]) + state_tokens + 34,
            self.qid,
            self.lid,
            with_gold=False,
        )
        return enc, len(full["input_ids"]) > len(enc["input_ids"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = EncoderOnly(args.model_dir)
    suite = json.loads((args.dataset / "suite.json").read_text())
    summary = {}
    rows = []
    for pack in suite["pack_paths"]:
        _, cases = load_pack(args.dataset / pack, allow_public_evaluation=True)
        counts = Counter()
        for case in cases:
            questions = case["questions"]
            if any(
                q["type"] not in ("choice", "noul")
                or (q["type"] == "choice" and len(q["criteria"]) < 2)
                for q in questions.values()
            ):
                reason = "type_or_min_options_filter"
            else:
                result = sev_capability(
                    model, case["state"], questions, {"choice", "noul"}, validate_questions
                )
                reason = "supported" if result["supported"] else result["reason"]
            counts[reason] += 1
            rows.append({"case_id": case["id"], "pack": pack, "reason": reason})
        summary[pack] = dict(counts)
        print(pack, dict(counts), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"model_dir": str(args.model_dir), "summary": summary, "cases": rows},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
