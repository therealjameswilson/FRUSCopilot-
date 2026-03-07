import agents.volume_suggester as volume_suggester


def test_build_thematic_queries_adds_arms_control_expansions():
    queries = volume_suggester.build_thematic_queries(
        topic="Nunn-Lugar",
        selected_volume="Being Researched — 1993–2000, Volume XX, Arms Control and Nonproliferation",
    )

    assert queries[0] == ("primary", "Nunn-Lugar")
    assert any(theme == "arms_control_nonproliferation" for theme, _ in queries)
    assert len(queries) > 1


def test_retrieve_thematic_documents_merges_duplicates_and_tags_themes(monkeypatch):
    payloads = {
        "Nunn-Lugar": [
            {
                "chunk_id": "c1",
                "title": "Doc 1",
                "volume_slug": "frus1969-76v34",
                "document_number": "12",
                "history_state_url": "https://history.state.gov/historicaldocuments/frus1969-76v34/d12",
                "score": 0.5,
            }
        ],
    }

    def fake_search(query: str, top_k: int = 20, filters=None):
        if "SALT negotiations strategic arms limitation talks ABM Treaty" in query:
            return [
                {
                    "chunk_id": "c1",
                    "title": "Doc 1",
                    "volume_slug": "frus1969-76v34",
                    "document_number": "12",
                    "history_state_url": "https://history.state.gov/historicaldocuments/frus1969-76v34/d12",
                    "score": 0.9,
                },
                {
                    "chunk_id": "c2",
                    "title": "Doc 2",
                    "volume_slug": "frus1969-76v33",
                    "document_number": "77",
                    "history_state_url": "https://history.state.gov/historicaldocuments/frus1969-76v33/d77",
                    "score": 0.8,
                },
            ]
        return payloads.get(query, [])

    monkeypatch.setattr(volume_suggester, "search", fake_search)

    out = volume_suggester.retrieve_thematic_documents(
        topic="Nunn-Lugar",
        selected_volume="Being Researched — 1993–2000, Volume XX, Arms Control and Nonproliferation",
        top_k=5,
    )

    assert [item["chunk_id"] for item in out] == ["c1", "c2"]
    assert out[0]["score"] == 0.9
    assert "primary" in out[0]["matched_themes"]
    assert "arms_control_nonproliferation" in out[0]["matched_themes"]
