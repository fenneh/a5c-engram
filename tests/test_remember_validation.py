"""Direct-write input validation on Profile.remember()."""

from __future__ import annotations

import pytest

from a5c_engram.profile import MAX_DIRECT_CONTENT_CHARS
from a5c_engram.schema import MemoryType


def test_empty_content_rejected(profile):
    with pytest.raises(ValueError, match="non-empty"):
        profile.remember("", type=MemoryType.FACT)


def test_whitespace_only_content_rejected(profile):
    with pytest.raises(ValueError, match="non-empty"):
        profile.remember("   \n\t  ", type=MemoryType.FACT)


def test_oversized_content_rejected(profile):
    big = "x" * (MAX_DIRECT_CONTENT_CHARS + 1)
    with pytest.raises(ValueError, match="too long"):
        profile.remember(big, type=MemoryType.FACT)


def test_at_cap_accepted(profile):
    content = "x" * MAX_DIRECT_CONTENT_CHARS
    m = profile.remember(content, type=MemoryType.FACT)
    assert m.content == content


def test_unknown_type_rejected(profile):
    with pytest.raises(ValueError, match="unknown memory type"):
        profile.remember("hello", type="banana")
    # Error names the valid types so callers can fix their input.
    with pytest.raises(ValueError, match="fact"):
        profile.remember("hello", type="banana")
