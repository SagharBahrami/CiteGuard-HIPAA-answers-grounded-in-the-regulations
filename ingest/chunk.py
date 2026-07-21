"""Split parsed Sections into retrieval-sized Chunks.

A section becomes a single chunk if it's already small. If it's longer than
MAX_CHARS, its paragraphs are packed greedily into successive chunks -- each
chunk fills up to MAX_CHARS before starting a new one -- so a split only ever
falls on a paragraph boundary, never mid-sentence. This is simpler than
reconstructing the section's (a)/(1)/(i) legal outline (see parse.py), which
isn't reliably recoverable from the flat XML paragraphs anyway.

When a section does split, the last paragraph of each chunk is repeated as
the first paragraph of the next one, so a chunk retrieved on its own isn't
missing the requirement/standard it's an implementation detail of.
"""

from dataclasses import dataclass

from ingest.parse import Section

MAX_CHARS = 1500


@dataclass
class Chunk:
    citation: str
    heading: str
    part: int
    subpart: str | None
    text: str
    chunk_index: int
    total_chunks: int


def chunk_section(section: Section, max_chars: int = MAX_CHARS) -> list[Chunk]:
    if not section.paragraphs:
        return []

    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for para in section.paragraphs:
        added_len = len(para) + (1 if current else 0)  # +1 for the joining newline
        if current and current_len + added_len > max_chars:
            groups.append(current)
            carried = current[-1]  # repeat as lead-in context for the next chunk
            current = [carried, para]
            current_len = len(carried) + 1 + len(para)
        else:
            current.append(para)
            current_len += added_len
    if current:
        groups.append(current)

    total = len(groups)
    return [
        Chunk(
            citation=section.citation,
            heading=section.heading,
            part=section.part,
            subpart=section.subpart,
            text="\n".join(group),
            chunk_index=i + 1,
            total_chunks=total,
        )
        for i, group in enumerate(groups)
    ]


def chunk_all(sections: list[Section], max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(chunk_section(section, max_chars))
    return chunks


if __name__ == "__main__":
    from pathlib import Path

    from ingest.fetch import PARTS, TITLE
    from ingest.parse import parse_all

    paths = {p: Path(f"data/raw/title-{TITLE}-part-{p}.xml") for p in PARTS}
    all_sections = parse_all(paths)
    all_chunks = chunk_all(all_sections)

    print(f"{len(all_sections)} sections -> {len(all_chunks)} chunks")
    split_sections = [s for s in all_sections if len(chunk_section(s)) > 1]
    print(f"{len(split_sections)} sections were split into multiple chunks")

    sample = next(c for c in all_chunks if c.citation == "45 CFR 164.312")
    print(f"\nSample chunk 1/{sample.total_chunks} of {sample.citation}:")
    print(sample.text[:300], "...")
