"""Vector + HyDE channels: assert that semantically similar content ranks
higher when the embedder knows it. Uses MockEmbedder so we control the
geometry deterministically."""

import pytest

from a5c_engram.llm.fake import FakeLLM
from a5c_engram.profile import Profile
from a5c_engram.storage.sqlite import SqliteStorage
from tests.helpers import MockEmbedder


def _profile_with(embedder, llm, tmp_path):
    storage = SqliteStorage(path=tmp_path / "engram.db", dim=embedder.dim)
    storage.init()
    return Profile("test", storage=storage, embedder=embedder, llm=llm)


@pytest.fixture
def vec_profile(tmp_path):
    """Profile with a controllable embedder mapping. We pin three distinct
    'cluster' vectors:
      - graphql/api family at ~(1, 0, 0)
      - cooking family at  ~(0, 1, 0)
      - weather family at  ~(0, 0, 1)
    """
    api = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cooking = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    weather = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    emb = MockEmbedder({
        "We use GraphQL for the API.": api,
        "Roast the chicken at 200C.": cooking,
        "It will rain tomorrow.": weather,
        "What API do we use?": api,
        "What is for dinner?": cooking,
    })
    return _profile_with(emb, FakeLLM(), tmp_path)


def test_vector_channel_returns_nearest_first(vec_profile):
    vec_profile.remember("We use GraphQL for the API.", type="fact")
    vec_profile.remember("Roast the chicken at 200C.", type="fact")
    vec_profile.remember("It will rain tomorrow.", type="event")
    result = vec_profile.recall("What API do we use?", use_hyde=False)
    vec_hits = result.by_channel["vector"]
    assert vec_hits, "vector channel should return hits"
    assert "GraphQL" in vec_hits[0].memory.content


def test_vector_channel_does_not_return_unrelated(vec_profile):
    vec_profile.remember("We use GraphQL for the API.", type="fact")
    vec_profile.remember("Roast the chicken at 200C.", type="fact")
    result = vec_profile.recall("What is for dinner?", use_hyde=False)
    vec_hits = result.by_channel["vector"]
    assert "chicken" in vec_hits[0].memory.content


def test_hyde_uses_hallucinated_answer(tmp_path):
    """HyDE embeds the LLM's hypothetical answer, not the query. If the
    answer's embedding clusters with a particular memory, HyDE should rank
    that memory."""
    api = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cooking = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class FixedAnswerLLM(FakeLLM):
        def hyde(self, query):
            # Deliberately hallucinate a cooking answer regardless of query.
            return "cooking topic"

    emb = MockEmbedder({
        "GraphQL fact": api,
        "Roast chicken at 200C.": cooking,
        # The hypothetical answer text routes to the cooking cluster.
        "cooking topic": cooking,
        # The raw query routes to the API cluster, so HyDE and vector
        # should pull in *different* memories.
        "abstract question": api,
    })
    storage = SqliteStorage(path=tmp_path / "engram.db", dim=emb.dim)
    storage.init()
    p = Profile("t", storage=storage, embedder=emb, llm=FixedAnswerLLM())
    p.remember("GraphQL fact", type="fact")
    p.remember("Roast chicken at 200C.", type="fact")

    result = p.recall("abstract question")
    vec_top = result.by_channel["vector"][0].memory.content
    hyde_top = result.by_channel["hyde"][0].memory.content
    assert vec_top == "GraphQL fact"
    assert hyde_top == "Roast chicken at 200C."


def test_task_memories_are_not_in_vector_index(vec_profile):
    """Tasks are excluded from the vector index (per design)."""
    vec_profile.remember("watch Arsenal lineup at 14:00", type="task")
    qvec = vec_profile.embedder.embed("Arsenal")
    hits = vec_profile.storage.search_vector("test", qvec, k=10)
    assert all(h.type.value != "task" for h in hits)
