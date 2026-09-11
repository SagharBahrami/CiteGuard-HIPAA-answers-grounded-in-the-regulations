import citedguard.jobs as jobs
from citedguard.ingest.__main__ import run as ingest_run
from citedguard.qa import answer_question


def test_enqueue_question_submits_answer_question_to_the_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs.queue, "enqueue", lambda fn, *a, **kw: calls.append((fn, a, kw)) or "fake-job")

    result = jobs.enqueue_question("What are the technical safeguards?", top_k=3)

    assert result == "fake-job"
    fn, args, kwargs = calls[0]
    assert fn is answer_question
    assert args == ("What are the technical safeguards?",)
    assert kwargs["top_k"] == 3
    assert kwargs["job_timeout"] == "5m"


def test_enqueue_ingestion_submits_run_to_the_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs.queue, "enqueue", lambda fn, *a, **kw: calls.append((fn, a, kw)) or "fake-job")

    result = jobs.enqueue_ingestion(force=True)

    assert result == "fake-job"
    fn, args, kwargs = calls[0]
    assert fn is ingest_run
    assert kwargs["force"] is True
    assert kwargs["job_timeout"] == "30m"


def test_get_job_fetches_by_id_on_the_same_connection(monkeypatch):
    seen = {}

    def fake_fetch(job_id, connection):
        seen["job_id"] = job_id
        seen["connection"] = connection
        return f"job:{job_id}"

    monkeypatch.setattr(jobs.Job, "fetch", fake_fetch)

    result = jobs.get_job("abc123")

    assert result == "job:abc123"
    assert seen["job_id"] == "abc123"
    assert seen["connection"] is jobs._redis
