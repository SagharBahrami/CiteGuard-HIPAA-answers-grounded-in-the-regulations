from ingest.parse import parse_part_xml

XML = """<?xml version="1.0"?>
<DIV5 N="164">
  <DIV6 TYPE="SUBPART" N="C">
    <HEAD>Subpart C - Security Standards</HEAD>
  </DIV6>
  <DIV8 TYPE="SECTION" N="164.312">
    <HEAD>Technical safeguards.</HEAD>
    <P>(a)   Standard:    Access   control.</P>
    <P>(b) Standard: Audit controls.</P>
  </DIV8>
  <DIV6 TYPE="SUBPART" N="D">
    <HEAD>Subpart D - Notification</HEAD>
  </DIV6>
  <DIV8 TYPE="SECTION" N="164.400">
    <HEAD>Applicability.</HEAD>
    <P>This subpart applies to breaches.</P>
  </DIV8>
</DIV5>
"""


def _write(tmp_path, xml=XML):
    path = tmp_path / "part.xml"
    path.write_text(xml)
    return path


def test_extracts_sections_in_document_order(tmp_path):
    sections = parse_part_xml(_write(tmp_path))

    assert [s.citation for s in sections] == ["45 CFR 164.312", "45 CFR 164.400"]
    assert [s.heading for s in sections] == ["Technical safeguards.", "Applicability."]
    assert all(s.part == 164 for s in sections)


def test_sections_track_the_most_recent_subpart(tmp_path):
    sections = parse_part_xml(_write(tmp_path))

    assert sections[0].subpart == "Subpart C - Security Standards"
    assert sections[1].subpart == "Subpart D - Notification"


def test_paragraph_whitespace_is_collapsed(tmp_path):
    sections = parse_part_xml(_write(tmp_path))

    assert sections[0].paragraphs[0] == "(a) Standard: Access control."
    assert sections[0].paragraphs[1] == "(b) Standard: Audit controls."


def test_section_text_property_joins_paragraphs_with_newlines(tmp_path):
    sections = parse_part_xml(_write(tmp_path))

    assert sections[0].text == "(a) Standard: Access control.\n(b) Standard: Audit controls."


def test_section_with_no_head_gets_empty_heading(tmp_path):
    xml = """<?xml version="1.0"?>
    <ROOT>
      <ELEM TYPE="SECTION" N="164.500">
        <P>Some text.</P>
      </ELEM>
    </ROOT>
    """
    sections = parse_part_xml(_write(tmp_path, xml))

    assert sections[0].heading == ""
    assert sections[0].subpart is None
