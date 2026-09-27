"""Strict option-key scoring for four documented source-based corrections."""
VERSION = '2026.09.27.1'
METHOD = 'absher-corrected-key-v1'


def grade(expected: dict, output: str) -> dict:
    key = output.strip().rstrip('.)')
    score = float(key == expected['answer'])
    return {'method': METHOD, 'score': score, 'passed': score == 1}
