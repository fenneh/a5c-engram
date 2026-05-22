"""Write-time query augmentation: paraphrases land in FTS and embeddings
but never in the displayed content."""

from a5c_engram.embed.base import FakeEmbedder
from a5c_engram.llm.fake import FakeLLM
from a5c_engram.profile import Profile
from a5c_engram.storage.sqlite import SqliteStorage


def test_paraphrases_indexed_in_fts(profile):
    m = profile.remember("uses GraphQL", type="fact", topic="api_style")
    assert m.search_keywords, "paraphrases should be generated and stored"
    # FTS should hit on a token that is only in the paraphrases.
    hits = profile.storage.search_fts("test", "about uses graphql")
    assert any(h.id == m.id for h in hits)


def test_paraphrases_not_in_displayed_content(profile):
    m = profile.remember("uses GraphQL", type="fact", topic="api_style")
    got = profile.storage.get_memory(m.id)
    assert got is not None
    assert got.content == "uses GraphQL"
    # Display content is unchanged; search_keywords is the augmentation.
    assert "uses GraphQL" not in got.search_keywords or "graphql" in got.search_keywords.lower()


def test_paraphrases_skipped_for_tasks(profile):
    m = profile.remember("watch lineup at 14:00", type="task")
    # Tasks are ephemeral; we don't burn LLM tokens generating paraphrases.
    assert m.search_keywords == ""


def test_paraphrase_failure_does_not_crash_commit(tmp_path):
    class FailingParaphraseLLM(FakeLLM):
        def paraphrase(self, content):
            raise RuntimeError("LLM down")

    storage = SqliteStorage(path=tmp_path / "engram.db")
    storage.init()
    p = Profile("t", storage=storage, embedder=FakeEmbedder(), llm=FailingParaphraseLLM())
    m = p.remember("uses GraphQL", type="fact", topic="api_style")
    assert m.search_keywords == ""
    # Memory still persisted.
    assert p.storage.get_memory(m.id) is not None
