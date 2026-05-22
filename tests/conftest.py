from __future__ import annotations

import pytest

from a5c_engram.embed.base import FakeEmbedder
from a5c_engram.llm.fake import FakeLLM
from a5c_engram.profile import Profile
from a5c_engram.storage.sqlite import SqliteStorage


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
