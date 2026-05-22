from a5c_engram.extract.verify import verify_candidate
from a5c_engram.llm.base import ExtractionCandidate


def test_empty_rejected():
    ok, _ = verify_candidate(ExtractionCandidate(type="fact", content=""), "anything")
    assert not ok


def test_too_long_rejected():
    cand = ExtractionCandidate(type="fact", content="x " * 400)
    ok, _ = verify_candidate(cand, "x")
    assert not ok


def test_unknown_type_rejected():
    cand = ExtractionCandidate(type="weird", content="some content here")
    ok, _ = verify_candidate(cand, "some content here")
    assert not ok


def test_no_support_rejected():
    cand = ExtractionCandidate(type="fact", content="Project uses GraphQL")
    ok, _ = verify_candidate(cand, "we mostly talk about kittens")
    assert not ok


def test_fact_key_must_be_snake_case():
    cand = ExtractionCandidate(
        type="fact", content="uses GraphQL", fact_key="API Style", evidence="GraphQL"
    )
    ok, _ = verify_candidate(cand, "the project uses GraphQL for the API")
    assert not ok


def test_happy_path():
    cand = ExtractionCandidate(
        type="fact",
        content="The project uses GraphQL.",
        fact_key="api_style",
        evidence="we use GraphQL everywhere",
    )
    ok, reason = verify_candidate(cand, "we use GraphQL everywhere it makes sense")
    assert ok, reason
