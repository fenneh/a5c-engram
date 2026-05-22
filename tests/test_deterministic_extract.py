from a5c_engram.extract.deterministic import deterministic_extract, has_relative_date


def test_self_reference_name():
    out = deterministic_extract("Hi, my name is Alice Walker, nice to meet you.")
    facts = [c for c in out if c.type == "fact"]
    assert any(c.fact_key == "user_name" for c in facts)


def test_iso_date_emits_event():
    out = deterministic_extract("Deployment is scheduled for 2026-05-23.")
    assert any(c.type == "event" for c in out)


def test_numeric_fact_factkey():
    out = deterministic_extract("threshold: 42, retries=3, latency=120.5")
    facts = {c.fact_key for c in out if c.type == "fact"}
    assert "threshold" in facts
    assert "retries" in facts
    assert "latency" in facts


def test_instruction_verb():
    out = deterministic_extract("Always run tests before pushing.")
    assert any(c.type == "instruction" for c in out)


def test_task_verb():
    out = deterministic_extract("Remind me to review PR 42 tomorrow.")
    assert any(c.type == "task" for c in out)


def test_relative_date_helper():
    assert has_relative_date("see you tomorrow")
    assert not has_relative_date("on the 2026-05-23 deployment")
