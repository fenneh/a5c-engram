from a5c_engram.extract.llm import CHUNK_CHARS, llm_extract
from a5c_engram.llm.base import ExtractionCandidate
from a5c_engram.llm.fake import FakeLLM
from tests.helpers import CountingLLM, MalformedLLM, ScriptedLLM


def test_empty_input_no_calls():
    llm = CountingLLM(FakeLLM())
    out = llm_extract("", llm)
    assert out == []
    assert llm.extract_calls == []


def test_whitespace_input_no_calls():
    llm = CountingLLM(FakeLLM())
    assert llm_extract("   \n\n  ", llm) == []
    assert llm.extract_calls == []


def test_short_text_single_call():
    llm = CountingLLM(FakeLLM())
    llm_extract("project uses GraphQL", llm)
    # short input → only one full-pass call, no detail windows.
    assert len(llm.extract_calls) == 1


def test_long_text_triggers_detail_pass():
    llm = CountingLLM(FakeLLM())
    long_text = "uses GraphQL " * 800  # ~10k chars
    llm_extract(long_text, llm)
    # Long text triggers both passes — at least one detail window beyond
    # the single full chunk.
    assert len(llm.extract_calls) > 1


def test_very_long_text_chunks():
    llm = CountingLLM(FakeLLM())
    huge = "x" * (CHUNK_CHARS * 3 + 100)
    llm_extract(huge, llm)
    # > 3 full chunks for the full pass, plus detail windows.
    assert len(llm.extract_calls) >= 4


def test_dedup_collapses_repeats():
    # Same candidate emitted from every pass; the dedup must collapse it.
    cand = ExtractionCandidate(type="fact", content="X uses Y.", fact_key="x")
    scripted = ScriptedLLM([[cand], [cand], [cand], [cand]])
    out = llm_extract("uses GraphQL " * 800, scripted)
    assert len(out) == 1
    assert out[0].fact_key == "x"


def test_dedup_normalises_whitespace_and_case():
    a = ExtractionCandidate(type="fact", content="The   project uses GraphQL.")
    b = ExtractionCandidate(type="fact", content="the project USES graphql.")
    scripted = ScriptedLLM([[a, b]])
    out = llm_extract("text", scripted)
    assert len(out) == 1


def test_dedup_keeps_different_types():
    f = ExtractionCandidate(type="fact", content="same content")
    e = ExtractionCandidate(type="event", content="same content")
    scripted = ScriptedLLM([[f, e]])
    out = llm_extract("text", scripted)
    assert len(out) == 2
    types = {c.type for c in out}
    assert types == {"fact", "event"}


def test_malformed_llm_returns_garbage():
    # An LLM that emits wrong-type candidates should not crash the pipeline.
    # llm_extract itself doesn't verify; the verifier filters later. Here we
    # just confirm we don't blow up.
    bad = MalformedLLM(mode="wrong_type")
    out = llm_extract("any text", bad)
    assert out == [ExtractionCandidate(type="banana", content="weird")]
