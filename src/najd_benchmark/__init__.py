"""Najd's versioned evaluation contracts."""

from .arabic_mmlu import (
    CASE_COUNT,
    CASE_IDS_SHA256,
    CASES_SHA256,
    DATASET_REVISION,
    SOURCE_ID,
    TASK_VERSION,
    build_messages,
    grade,
    score_responses,
    validate_cases,
)

__all__ = [
    "CASE_COUNT", "CASE_IDS_SHA256", "CASES_SHA256", "DATASET_REVISION", "SOURCE_ID",
    "TASK_VERSION", "build_messages", "grade", "score_responses", "validate_cases",
]
