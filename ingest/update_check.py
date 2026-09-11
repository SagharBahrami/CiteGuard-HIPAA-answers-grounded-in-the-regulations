"""Detect when eCFR has published a newer issue date than what's ingested,
and re-ingest only when it has.

This is what makes version tracking automatic rather than something someone
has to remember to run by hand: schedule check_and_maybe_ingest (via
jobs.enqueue_update_check, on a cron/RQ-scheduler trigger) and updates show
up in Chroma -- with the superseded versions preserved in history.py --
without manual intervention.
"""

import logging
from pathlib import Path

from config import settings
from ingest.__main__ import run
from ingest.embed_store import current_issue_date
from ingest.fetch import TITLE, get_current_issue_date

logger = logging.getLogger(__name__)


def check_and_maybe_ingest() -> bool:
    """Re-ingest if eCFR's latest issue date differs from what's currently stored.

    Returns True if a new issue date was found and ingestion actually ran --
    False if nothing changed, or if another ingestion was already in flight
    and this run was skipped (see locks.py).
    """
    latest = get_current_issue_date(TITLE)
    ingested = current_issue_date(Path(settings.chroma_dir), settings.chroma_collection)

    if ingested == latest:
        logger.info("Already up to date with eCFR issue date %s", latest)
        return False

    logger.info("New eCFR issue date %s (had %s) -- re-ingesting", latest, ingested)
    return run(force=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    updated = check_and_maybe_ingest()
    print("Updated" if updated else "Already current")
