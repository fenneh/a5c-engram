from a5c_engram.schema import Memory, MemoryType


def test_add_and_get(storage):
    m = Memory.new(profile="p", type=MemoryType.FACT, content="hello world")
    storage.add_memory(m)
    got = storage.get_memory(m.id)
    assert got is not None and got.content == "hello world"


def test_fts_search(storage):
    storage.add_memory(Memory.new(profile="p", type=MemoryType.FACT, content="uses GraphQL"))
    storage.add_memory(Memory.new(profile="p", type=MemoryType.FACT, content="uses REST"))
    hits = storage.search_fts("p", "GraphQL")
    assert len(hits) == 1
    assert "GraphQL" in hits[0].content


def test_supersession_chain(storage):
    a = Memory.new(profile="p", type=MemoryType.FACT, content="uses GraphQL", fact_key="api")
    storage.supersede("p", "api", a)
    b = Memory.new(profile="p", type=MemoryType.FACT, content="uses REST", fact_key="api")
    storage.supersede("p", "api", b)
    c = Memory.new(profile="p", type=MemoryType.FACT, content="uses gRPC", fact_key="api")
    storage.supersede("p", "api", c)

    latest = storage.latest_for_factkey("p", "api")
    assert latest is not None and latest.content == "uses gRPC"
    assert latest.version == 3

    chain = storage.supersession_chain(latest.id)
    assert [m.content for m in chain] == ["uses GraphQL", "uses REST", "uses gRPC"]


def test_factkey_supersession_hides_old_from_fts(storage):
    a = Memory.new(profile="p", type=MemoryType.FACT, content="uses GraphQL", fact_key="api")
    storage.supersede("p", "api", a)
    b = Memory.new(profile="p", type=MemoryType.FACT, content="uses gRPC", fact_key="api")
    storage.supersede("p", "api", b)
    hits = storage.search_fts("p", "GraphQL")
    # superseded memory must not surface from FTS — that's the point.
    assert all(h.content != "uses GraphQL" for h in hits)
