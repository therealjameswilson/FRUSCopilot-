from pathlib import Path

from agents.frus_loader import infer_volume_start_year, is_supported_volume_slug, load_documents


def test_infer_volume_start_year_handles_slug_formats():
    assert infer_volume_start_year("frus1961-63v01") == 1961
    assert infer_volume_start_year("frus1997v01") == 1997
    assert infer_volume_start_year("volume1961") is None


def test_is_supported_volume_slug_filters_before_1961():
    assert is_supported_volume_slug("frus1961-63v01")
    assert not is_supported_volume_slug("frus1958-60v01")


def test_load_documents_skips_pre_1961_volumes(tmp_path: Path):
    volumes = tmp_path / "volumes"
    repo_root = tmp_path

    supported = volumes / "frus1961-63v01" / "d1.md"
    unsupported = volumes / "frus1958-60v01" / "d1.md"

    supported.parent.mkdir(parents=True)
    unsupported.parent.mkdir(parents=True)

    supported.write_text("# Supported\nJanuary 1, 1961\nText", encoding="utf-8")
    unsupported.write_text("# Unsupported\nJanuary 1, 1960\nText", encoding="utf-8")

    docs = list(load_documents(volumes, repo_root))

    assert len(docs) == 1
    assert docs[0].volume_slug == "frus1961-63v01"
