"""Fetch and cache raw HIPAA regulation text from the official eCFR API.

eCFR blocks HTML scraping of its rendered pages (CAPTCHA wall) and directs
programmatic access to its versioner API instead, so this pulls structured
XML per CFR part rather than scraping www.ecfr.gov directly.
"""

import logging
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ecfr.gov/api/versioner/v1"
HEADERS = {"User-Agent": "citedguard-project (learning project; contact via GitHub)"}
TITLE = 45
PARTS = [160, 162, 164]  # Subchapter C: Administrative Data Standards & Related Requirements


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_current_issue_date(title: int) -> str:
    """Return the most recent issue date eCFR has for a title, e.g. '2026-07-02'."""
    resp = requests.get(f"{BASE_URL}/titles.json", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    titles = resp.json()["titles"]
    match = next(t for t in titles if t["number"] == title)
    return match["latest_issue_date"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_part_xml(title: int, part: int, issue_date: str) -> bytes:
    """Download the full XML text of one CFR part as of a given issue date."""
    url = f"{BASE_URL}/full/{issue_date}/title-{title}.xml"
    resp = requests.get(url, headers=HEADERS, params={"part": part}, timeout=60)
    resp.raise_for_status()
    return resp.content


def fetch_all_parts(raw_dir: Path, parts: list[int] = PARTS, force: bool = False) -> dict[int, Path]:
    """Fetch each part's XML into raw_dir, skipping parts already cached on disk.

    Returns a mapping of part number -> path to its cached XML file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    issue_date = get_current_issue_date(TITLE)
    logger.info("Using eCFR title %s issue date %s", TITLE, issue_date)

    paths: dict[int, Path] = {}
    for part in parts:
        dest = raw_dir / f"title-{TITLE}-part-{part}.xml"
        if dest.exists() and not force:
            logger.info("Part %s already cached at %s, skipping fetch", part, dest)
        else:
            logger.info("Fetching part %s as of %s", part, issue_date)
            content = fetch_part_xml(TITLE, part, issue_date)
            dest.write_bytes(content)
        paths[part] = dest
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_all_parts(Path("data/raw"))
    for part, path in result.items():
        print(f"Part {part}: {path} ({path.stat().st_size:,} bytes)")
