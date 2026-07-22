from ingest.chunk import chunk_all, chunk_section
from ingest.parse import Section


def _section(paragraphs, citation="45 CFR 164.100"):
    return Section(part=164, subpart="Test Subpart", citation=citation, heading="Test Heading", paragraphs=paragraphs)


def test_empty_section_produces_no_chunks():
    assert chunk_section(_section([])) == []


def test_short_section_stays_as_one_chunk():
    section = _section(["Paragraph one.", "Paragraph two."])

    chunks = chunk_section(section, max_chars=1500)

    assert len(chunks) == 1
    assert chunks[0].text == "Paragraph one.\nParagraph two."
    assert chunks[0].chunk_index == 1
    assert chunks[0].total_chunks == 1
    assert chunks[0].citation == section.citation
    assert chunks[0].heading == section.heading


def test_long_section_splits_only_at_paragraph_boundaries():
    paragraphs = [f"Paragraph {i} " + "x" * 50 for i in range(10)]
    section = _section(paragraphs)

    chunks = chunk_section(section, max_chars=200)

    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.text.split("\n"):
            assert line in paragraphs, "chunk contains a line not matching a whole original paragraph"


def test_split_chunks_repeat_last_paragraph_as_lead_in():
    paragraphs = [f"Paragraph {i} " + "x" * 50 for i in range(10)]
    section = _section(paragraphs)

    chunks = chunk_section(section, max_chars=200)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.text.split("\n")[-1] == nxt.text.split("\n")[0]


def test_total_chunks_is_consistent_across_all_chunks_of_a_section():
    paragraphs = [f"Paragraph {i} " + "x" * 50 for i in range(10)]
    chunks = chunk_section(_section(paragraphs), max_chars=200)

    assert len(chunks) > 1
    assert {c.total_chunks for c in chunks} == {len(chunks)}
    assert [c.chunk_index for c in chunks] == list(range(1, len(chunks) + 1))


def test_chunk_all_flattens_multiple_sections_in_order():
    sections = [_section(["a"], citation="45 CFR 1"), _section(["b"], citation="45 CFR 2")]

    chunks = chunk_all(sections)

    assert [c.citation for c in chunks] == ["45 CFR 1", "45 CFR 2"]
