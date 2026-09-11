from conftest import FakeRedis
from locks import INGESTION_LOCK, try_lock


def test_lock_is_acquired_when_free_and_released_on_exit():
    redis = FakeRedis()

    with try_lock("some:lock", redis_client=redis) as acquired:
        assert acquired is True
        assert "some:lock" in redis.store  # held for the duration of the block

    assert redis.store == {}  # released on the way out


def test_lock_is_not_acquired_when_another_process_holds_it():
    redis = FakeRedis(held=["some:lock"])

    with try_lock("some:lock", redis_client=redis) as acquired:
        assert acquired is False


def test_a_contended_lock_is_not_released_by_the_process_that_failed_to_take_it():
    """Releasing a lock we never acquired would free the holder's lock."""
    redis = FakeRedis(held=["some:lock"])
    holder = redis.store["some:lock"]

    with try_lock("some:lock", redis_client=redis):
        pass

    assert redis.store["some:lock"] is holder


def test_lock_is_released_even_when_the_body_raises():
    redis = FakeRedis()

    try:
        with try_lock("some:lock", redis_client=redis):
            raise RuntimeError("ingestion blew up")
    except RuntimeError:
        pass

    assert redis.store == {}


def test_unreachable_redis_proceeds_without_a_lock():
    """Redis is optional infrastructure -- it must not block a plain CLI run."""
    redis = FakeRedis(unreachable=True)

    with try_lock("some:lock", redis_client=redis) as acquired:
        assert acquired is True


def test_a_lock_that_expired_mid_run_does_not_raise_on_release():
    """The TTL can elapse during a long ingestion; that must not crash the run."""
    redis = FakeRedis(expire_on_release=True)

    with try_lock("some:lock", redis_client=redis) as acquired:
        assert acquired is True
    # exits cleanly rather than propagating LockNotOwnedError


def test_two_sequential_runs_can_each_take_the_lock():
    redis = FakeRedis()

    with try_lock(INGESTION_LOCK, redis_client=redis) as first:
        assert first is True
    with try_lock(INGESTION_LOCK, redis_client=redis) as second:
        assert second is True
