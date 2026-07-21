"""Parse cached eCFR part XML into per-section records.

Each regulation section (e.g. "45 CFR 164.312") becomes one Section, with its
paragraphs kept as a flat, ordered list of strings. The XML has no nested
structure for the (a)/(1)/(i) legal outline within a section -- it's all flat
<P> tags with the outline markers as leading text -- so we don't try to
reconstruct that hierarchy here; chunk.py handles splitting long sections.
"""

from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree


@dataclass
class Section:
    part: int
    subpart: str | None
    citation: str
    heading: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_part_xml(xml_path: Path) -> list[Section]:
    """Extract all SECTION elements from one cached part XML file, in order."""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    sections: list[Section] = []
    current_subpart: str | None = None

    for elem in root.iter():
        tag_type = elem.get("TYPE")
        if tag_type == "SUBPART":
            head = elem.find("HEAD")
            current_subpart = _clean(head.text) if head is not None else None
        elif tag_type == "SECTION":
            head = elem.find("HEAD")
            heading = _clean(head.text) if head is not None else ""
            paragraphs = [
                _clean("".join(p.itertext()))
                for p in elem.findall("P")
            ]
            paragraphs = [p for p in paragraphs if p]
            sections.append(
                Section(
                    part=int(elem.get("N").split(".")[0]),
                    subpart=current_subpart,
                    citation=f"45 CFR {elem.get('N')}",
                    heading=heading,
                    paragraphs=paragraphs,
                )
            )
    return sections


def parse_all(raw_paths: dict[int, Path]) -> list[Section]:
    """Parse every cached part XML file, returning sections in part order."""
    sections: list[Section] = []
    for part in sorted(raw_paths):
        sections.extend(parse_part_xml(raw_paths[part]))
    return sections


if __name__ == "__main__":
    from ingest.fetch import PARTS, TITLE

    paths = {p: Path(f"data/raw/title-{TITLE}-part-{p}.xml") for p in PARTS}
    all_sections = parse_all(paths)
    print(f"Parsed {len(all_sections)} sections total")
    for s in all_sections[:3]:
        print(f"- {s.citation} [{s.subpart}] {s.heading!r} ({len(s.paragraphs)} paragraphs)")
