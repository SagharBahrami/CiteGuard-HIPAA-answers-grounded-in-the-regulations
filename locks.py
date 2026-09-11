"""Cross-process advisory locking on Redis.

Ingestion mutates two stores that both assume a single writer: Chroma's local
persistent client, and the SQLite history archive. Nothing previously stopped
two runs from overlapping -- docker-compose runs four worker replicas, and a
scheduled update check can fire while a manual `python -m ingest` is still
going. Overlapping runs can interleave archive-then-upsert pairs and race the
delete of chunks the other run just wrote.

The semantics are deliberately "skip if already running" rather than "wait
your turn": a second ingestion started while one is in flight has nothing new
to do, and queueing behind the lock only to re-embed the whole corpus costs
real money for no change.

Redis is optional infrastructure here -- `python -m ingest` works without it,
only the RQ queue needs it -- so an unreachable Redis logs a warning and
proceeds unlocked instead of blocking ingestion outright. That degradation is
safe: without Redis the workers can't run at all, leaving a manual CLI run as
the only caller, with nothing to race against.
"""

import logging
from contextlib import contextmanager

from redis import Redis
from redis.exceptions import LockNotOwnedError, RedisError

from config import settings

logger = logging.getLogger(__name__)

INGESTION_LOCK = "citedguard:lock:ingestion"

# Long enough to cover a full re-fetch and re-embed of the corpus, short
# enough that a worker killed mid-run doesn't wedge ingestion indefinitely.
INGESTION_LOCK_TTL = 30 * 60


@contextmanager
def try_lock(name: str, ttl: int = INGESTION_LOCK_TTL, redis_client: Redis | None = None):
    """Yield True if this process took `name`, False if someone else holds it.

    Never blocks. Yields True without a lock if Redis can't be reached.
    """
    client = redis_client or Redis.from_url(settings.redis_url)
    lock = client.lock(name, timeout=ttl)
    try:
        acquired = lock.acquire(blocking=False)
    except RedisError as exc:
        logger.warning("Redis unavailable (%s) -- proceeding without the %r lock", exc, name)
        yield True
        return

    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock.release()
            except LockNotOwnedError:
                # The TTL elapsed mid-run, so someone else may hold it now.
                # Releasing anyway would free *their* lock.
                logger.warning("Lock %r expired before release (TTL %ss exceeded)", name, ttl)
