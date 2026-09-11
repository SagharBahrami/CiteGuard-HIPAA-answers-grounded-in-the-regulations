import ingest.update_check as update_check
from ingest.update_check import check_and_maybe_ingest


def _patch(monkeypatch, latest, ingested, runs, run_result=True):
    monkeypatch.setattr(update_check, "get_current_issue_date", lambda title: latest)
    monkeypatch.setattr(update_check, "current_issue_date", lambda persist_dir, collection: ingested)
    monkeypatch.setattr(update_check, "run", lambda force=False: runs.append(force) or run_result)


def test_no_ingestion_when_the_issue_date_is_unchanged(monkeypatch):
    runs = []
    _patch(monkeypatch, latest="2026-07-02", ingested="2026-07-02", runs=runs)

    assert check_and_maybe_ingest() is False
    assert runs == []


def test_reingests_when_ecfr_publishes_a_newer_issue_date(monkeypatch):
    runs = []
    _patch(monkeypatch, latest="2026-09-01", ingested="2026-07-02", runs=runs)

    assert check_and_maybe_ingest() is True
    assert runs == [False]


def test_reingests_when_nothing_has_been_ingested_yet(monkeypatch):
    runs = []
    _patch(monkeypatch, latest="2026-07-02", ingested=None, runs=runs)

    assert check_and_maybe_ingest() is True
    assert runs == [False]


def test_reports_no_ingestion_when_the_run_was_skipped_for_a_held_lock(monkeypatch):
    """A newer issue date existed, but another ingestion was already in flight --
    this check must not claim it updated anything."""
    runs = []
    _patch(monkeypatch, latest="2026-09-01", ingested="2026-07-02", runs=runs, run_result=False)

    assert check_and_maybe_ingest() is False
    assert runs == [False]
