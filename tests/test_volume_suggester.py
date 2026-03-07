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


def test_suggest_documents_accepts_selected_volume_kwarg(monkeypatch):
    captured = {}

    def fake_search(query: str, top_k: int = 20, filters=None):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["filters"] = filters
        return [{"chunk_id": "c1", "score": 1.0}]

    monkeypatch.setattr(volume_suggester, "search", fake_search)

    out = volume_suggester.suggest_documents(
        topic="arms control",
        top_k=10,
        volume_slug="frus1969-76v34",
        selected_volume="Being Researched — 1993–2000, Volume XX",
    )

    assert out == [{"chunk_id": "c1", "score": 1.0}]
    assert captured == {
        "query": "arms control",
        "top_k": 10,
        "filters": {"volume_slug": "frus1969-76v34"},
    }


def test_retrieve_thematic_documents_falls_back_to_history_state(monkeypatch):
    monkeypatch.setattr(volume_suggester, "search", lambda query, top_k=20, filters=None: [])

    captured = {}

    def fake_fallback(query: str, top_k: int = 20):
        captured["query"] = query
        captured["top_k"] = top_k
        return [{"chunk_id": "history-state|frus1969-76v34|1", "score": 0.7}]

    monkeypatch.setattr(volume_suggester, "search_history_state_documents", fake_fallback)

    out = volume_suggester.retrieve_thematic_documents(topic="Nunn-Lugar", top_k=7)

    assert out == [{"chunk_id": "history-state|frus1969-76v34|1", "score": 0.7}]
    assert captured == {"query": "Nunn-Lugar", "top_k": 7}


def test_search_history_state_documents_parses_and_deduplicates_results(monkeypatch):
    html = """
    <html>
      <body>
        <a href="/historicaldocuments/frus1969-76v34/d12">Nunn-Lugar planning memo</a>
        <a href="https://history.state.gov/historicaldocuments/frus1969-76v34/d12">Nunn-Lugar planning memo</a>
        <a href="/historicaldocuments/frus1969-76v33/d77">Strategic arms negotiation summary</a>
      </body>
    </html>
    """

    monkeypatch.setattr(volume_suggester, "_fetch_url", lambda _: html)

    out = volume_suggester.search_history_state_documents(query="Nunn-Lugar", top_k=10)

    assert len(out) == 2
    assert out[0]["history_state_url"] == "https://history.state.gov/historicaldocuments/frus1969-76v34/d12"
    assert out[0]["matched_themes"] == ["history_state_fallback"]
    assert out[0]["source_path"] == "history.state.gov search"
