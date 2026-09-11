"""Redis/RQ job queue: background processing for ingestion and question
answering, instead of blocking the caller (a Streamlit request, an MCP tool
call, or the CLI) until an OpenAI round-trip finishes.

A separate worker process (`rq worker citedguard --url <redis_url>`, or the
`worker` service in docker-compose.yml) actually executes enqueued jobs --
this module only defines the queue and what gets put on it.
"""

from redis import Redis
from rq import Queue
from rq.job import Job

from citedguard.config import settings

QUEUE_NAME = "citedguard"

_redis = Redis.from_url(settings.redis_url)
queue = Queue(QUEUE_NAME, connection=_redis)


def enqueue_question(query: str, top_k: int = 3) -> Job:
    from citedguard.qa import answer_question

    return queue.enqueue(answer_question, query, top_k=top_k, job_timeout="5m")


def enqueue_ingestion(force: bool = False) -> Job:
    from citedguard.ingest.__main__ import run

    return queue.enqueue(run, force=force, job_timeout="30m")


def enqueue_update_check() -> Job:
    """Check eCFR for a newer issue date and re-ingest only if one exists.

    Schedule this on a recurring trigger (cron, RQ-scheduler) to pick up
    regulatory changes automatically -- see ingest/update_check.py.
    """
    from citedguard.ingest.update_check import check_and_maybe_ingest

    return queue.enqueue(check_and_maybe_ingest, job_timeout="30m")


def get_job(job_id: str) -> Job:
    return Job.fetch(job_id, connection=_redis)
