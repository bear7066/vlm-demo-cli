"""The HuggingFace cache scan behind the page's model suggestions."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_demo.models import cached_model_ids, hub_cache_dir


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return cache


def test_cache_directories_become_repo_ids(hub: Path):
    for name in (
        "models--google--gemma-4-e4b-it",
        "models--bert-base-uncased",
        "datasets--someone--a-dataset",  # not a model
        ".locks",
    ):
        (hub / name).mkdir()
    (hub / "version.txt").write_text("1")  # a file, not a repo

    assert cached_model_ids() == ["bert-base-uncased", "google/gemma-4-e4b-it"]


def test_a_missing_cache_is_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "nowhere"))
    assert cached_model_ids() == []


def test_hf_home_is_honoured_when_hf_hub_cache_is_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert hub_cache_dir() == tmp_path / "hub"
