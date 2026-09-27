"""Explicit typed-question mapping for pinned TypeLLM runtime."""

import json


def questions_for(questions):
    result = {}
    for key, q in questions.items():
        instructions = q["instructions"]
        if q["type"] == "choice":
            if not 1 < len(q["criteria"]) <= 24:
                raise ValueError("Unsupported enum cardinality")
            field = {"type": "string", "enum": list(q["criteria"])}
            instructions += "\nCandidate meanings: " + json.dumps(q["criteria"], ensure_ascii=False)
        elif q["type"] == "score":
            if not 1 < len(q["criteria"]) <= 24:
                raise ValueError("Unsupported enum cardinality")
            field = {"type": "integer", "enum": list(range(len(q["criteria"])))}
            instructions += "\nZero-based level meanings: " + json.dumps(
                q["criteria"], ensure_ascii=False
            )
        elif q["type"] == "noul":
            field = {"type": "boolean"}
            if q.get("criteria"):
                instructions += "\nBoolean meanings: " + json.dumps(
                    q["criteria"], ensure_ascii=False
                )
        else:
            raise ValueError("Unsupported question type")
        field["instructions"] = instructions
        result[key] = field
    return result
