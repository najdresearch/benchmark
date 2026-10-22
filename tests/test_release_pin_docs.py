"""User-facing instructions must name the release that release.py pins."""

import re
from pathlib import Path

from najd_benchmark import release

ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
TASK_GUIDE = (ROOT / "docs/tasks/arabic-mmlu-key-v1.md").read_text(encoding="utf-8")


def test_readme_names_the_pinned_release():
    for value in (
        release.DATASET_VERSION,
        release.DATASET_REVISION,
        release.CASES_PATH,
        release.CASES_SHA256,
    ):
        assert value in README


def test_readme_does_not_point_at_another_case_file():
    paths = re.findall(r"datasets/najd-benchmark/([^/\s`]+)/cases\.jsonl", README)
    assert set(paths) == {release.DATASET_VERSION}


def test_task_guide_names_the_pinned_release():
    for value in (release.DATASET_VERSION, release.DATASET_REVISION, release.CASES_SHA256):
        assert value in TASK_GUIDE


def test_readme_local_links_resolve():
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", README):
        if "://" not in target:
            assert (ROOT / target).exists(), target
