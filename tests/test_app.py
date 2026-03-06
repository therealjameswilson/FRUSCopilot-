import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app, parse_max_results, parse_volume


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