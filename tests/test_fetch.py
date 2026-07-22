import types

import ingest.fetch as fetch
from ingest.fetch import TITLE, fetch_all_parts, get_current_issue_date


def test_get_current_issue_date_finds_matching_title(monkeypatch):
    def fake_get(url, headers, timeout):
        return types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"titles": [
                {"number": 45, "latest_issue_date": "2026-01-01"},
                {"number": 46, "latest_issue_date": "2026-02-02"},
            ]},
        )

    monkeypatch.setattr(fetch.requests, "get", fake_get)

    assert get_current_issue_date(45) == "2026-01-01"


def test_fetch_all_parts_skips_files_already_cached(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cached = raw_dir / f"title-{TITLE}-part-160.xml"
    cached.write_bytes(b"cached content")

    fetched_parts = []
    monkeypatch.setattr(fetch, "get_current_issue_date", lambda title: "2026-01-01")
    monkeypatch.setattr(
        fetch, "fetch_part_xml",
        lambda title, part, issue_date: fetched_parts.append(part) or b"fresh content",
    )

    fetch_all_parts(raw_dir, parts=[160, 162], force=False)

    assert fetched_parts == [162]
    assert cached.read_bytes() == b"cached content"


def test_fetch_all_parts_force_refetches_even_cached_files(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cached = raw_dir / f"title-{TITLE}-part-160.xml"
    cached.write_bytes(b"stale content")

    monkeypatch.setattr(fetch, "get_current_issue_date", lambda title: "2026-01-01")
    monkeypatch.setattr(fetch, "fetch_part_xml", lambda title, part, issue_date: b"new content")

    fetch_all_parts(raw_dir, parts=[160], force=True)

    assert cached.read_bytes() == b"new content"


def test_fetch_all_parts_returns_path_mapping_for_every_part(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(fetch, "get_current_issue_date", lambda title: "2026-01-01")
    monkeypatch.setattr(fetch, "fetch_part_xml", lambda title, part, issue_date: b"content")

    paths = fetch_all_parts(raw_dir, parts=[160, 162, 164], force=False)

    assert set(paths.keys()) == {160, 162, 164}
    assert all(p.exists() for p in paths.values())
