from __future__ import annotations

import pytest

from a5c_engram.embed.base import FakeEmbedder
from a5c_engram.llm.fake import FakeLLM
from a5c_engram.profile import Profile
from a5c_engram.storage.sqlite import SqliteStorage


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    """Force the fake embedder for every test so CI never triggers a
    fastembed model download. Tests that need a real embedder construct
    one explicitly."""
    monkeypatch.setenv("A5C_ENGRAM_EMBEDDER", "fake")


@pytest.fixture
def storage(tmp_path):
    s = SqliteStorage(path=tmp_path / "engram.db")
    s.init()
    return s


@pytest.fixture
def profile(storage):
    return Profile(
        "test",
        storage=storage,
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
    )
