import citedguard.ingest.__main__ as ingest_main
from conftest import FakeRedis
from citedguard.locks import INGESTION_LOCK


def _patch_pipeline(monkeypatch, calls):
    monkeypatch.setattr(
        ingest_main, "fetch_all_parts",
        lambda raw_dir, parts=None, force=False: calls.append("fetch") or ("2026-08-31", {160: "p"}),
    )
    monkeypatch.setattr(ingest_main, "parse_all", lambda paths, issue_date: calls.append("parse") or ["s"])
    monkeypatch.setattr(ingest_main, "chunk_all", lambda sections: calls.append("chunk") or ["c"])
    monkeypatch.setattr(ingest_main, "store_chunks", lambda *a, **kw: calls.append("store"))


def _patch_redis(monkeypatch, redis):
    monkeypatch.setattr(ingest_main, "try_lock", _lock_with(redis))


def _lock_with(redis):
    from contextlib import contextmanager

    from citedguard.locks import try_lock as real_try_lock

    @contextmanager
    def _try_lock(name, **kwargs):
        with real_try_lock(name, redis_client=redis) as acquired:
            yield acquired

    return _try_lock


def test_run_executes_the_pipeline_when_the_lock_is_free(monkeypatch):
    calls = []
    _patch_pipeline(monkeypatch, calls)
    _patch_redis(monkeypatch, FakeRedis())

    assert ingest_main.run() is True
    assert calls == ["fetch", "parse", "chunk", "store"]


def test_run_skips_entirely_when_another_ingestion_holds_the_lock(monkeypatch):
    """The point of the lock: no fetching, no embedding, no writes."""
    calls = []
    _patch_pipeline(monkeypatch, calls)
    _patch_redis(monkeypatch, FakeRedis(held=[INGESTION_LOCK]))

    assert ingest_main.run() is False
    assert calls == []


def test_run_proceeds_when_redis_is_unavailable(monkeypatch):
    calls = []
    _patch_pipeline(monkeypatch, calls)
    _patch_redis(monkeypatch, FakeRedis(unreachable=True))

    assert ingest_main.run() is True
    assert calls == ["fetch", "parse", "chunk", "store"]


def test_lock_is_freed_after_a_run_so_the_next_one_can_proceed(monkeypatch):
    calls = []
    redis = FakeRedis()
    _patch_pipeline(monkeypatch, calls)
    _patch_redis(monkeypatch, redis)

    assert ingest_main.run() is True
    assert redis.store == {}
    assert ingest_main.run() is True
