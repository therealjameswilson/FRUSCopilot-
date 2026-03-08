import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (
    MANUAL_VOLUME_OPTIONS,
    app,
    doc_matches_volume,
    extract_target_volume_names,
    get_volume_options,
    parse_max_results,
    parse_volume,
)


def test_parse_max_results_bounds():
    assert parse_max_results("500") == 200
    assert parse_max_results("0") == 1
    assert parse_max_results("bad") == 25


def test_parse_volume_extracts_documents(tmp_path: Path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        """<TEI><text><body>
        <div type='document'><head>Doc A</head><p>Hello world</p></div>
        <div type='appendix'><head>Skip</head></div>
        </body></text></TEI>"""
    )
    docs = parse_volume(xml)
    assert len(docs) == 1
    assert docs[0]["title"] == "Doc A"
    assert "Hello world" in docs[0]["text"]


def test_healthz_route():
    with app.test_client() as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}


def test_extract_target_volume_names_filters_statuses():
    html = """
    <table>
      <tr><th>Volume</th><th>Status</th></tr>
      <tr><td><a href='/historicaldocuments/frus1969-76v11'>Vol XI</a></td><td>Being Researched</td></tr>
      <tr><td><a href='/historicaldocuments/frus1969-76v12'>Vol XII</a></td><td>Planner</td></tr>
      <tr><td><a href='/historicaldocuments/frus1969-76v13'>Vol XIII</a></td><td>Published</td></tr>
    </table>
    """
    assert extract_target_volume_names(html) == ["Vol XI", "Vol XII"]


def test_doc_matches_volume_by_source_name():
    doc = {"source": "frus1969-76v11.xml", "title": "Doc", "text": "Body"}
    assert doc_matches_volume(doc, "FRUS 1969-76, Volume XI")
    assert not doc_matches_volume(doc, "FRUS 1969-76, Volume XII")




def test_doc_matches_volume_accepts_all_filter_value():
    doc = {"source": "frus1969-76v11.xml", "title": "Doc", "text": "Body"}
    assert doc_matches_volume(doc, "All")
def test_doc_matches_volume_does_not_match_empty_source_or_title():
    doc = {"source": "", "title": "", "text": "Body"}
    assert not doc_matches_volume(doc, "FRUS 1969-76, Volume XII")


def test_get_volume_options_includes_manual_entries(monkeypatch):
    html = """
    <table>
      <tr><td><a href='/historicaldocuments/frus1969-76v11'>Vol XI</a></td><td>Being Researched</td></tr>
    </table>
    """

    monkeypatch.setattr("app.fetch_status_page_html", lambda: html)

    options, error = get_volume_options()

    assert error is None
    assert "Vol XI" in options
    for manual in MANUAL_VOLUME_OPTIONS:
        assert manual in options


def test_get_volume_options_falls_back_to_manual_entries(monkeypatch):
    def raise_error():
        raise TimeoutError("timed out")

    monkeypatch.setattr("app.fetch_status_page_html", raise_error)

    options, error = get_volume_options()

    assert options == MANUAL_VOLUME_OPTIONS
    assert "Could not load volume options" in error
