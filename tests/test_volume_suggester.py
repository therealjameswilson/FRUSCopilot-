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

    def fake_search(query: str, top_k: int = 20, filters=None, strategy="hybrid"):
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

    def fake_search(query: str, top_k: int = 20, filters=None, strategy="hybrid"):
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


def test_suggest_declassified_sources_returns_fallback_on_api_failure(monkeypatch):
    class BrokenResponses:
        def create(self, **kwargs):
            raise TimeoutError("request timed out")

    class BrokenClient:
        responses = BrokenResponses()

    monkeypatch.setattr(volume_suggester, "_get_client", lambda: BrokenClient())

    output = volume_suggester.suggest_declassified_sources(topic="NSC system")

    assert "timed out or failed" in output
    assert "TimeoutError" in output


def test_suggest_classified_archives_returns_fallback_on_api_failure(monkeypatch):
    class BrokenResponses:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    class BrokenClient:
        responses = BrokenResponses()

    monkeypatch.setattr(volume_suggester, "_get_client", lambda: BrokenClient())

    output = volume_suggester.suggest_classified_archives(topic="NSC system")

    assert "timed out or failed" in output
    assert "RuntimeError" in output



def test_infer_search_plan_returns_heuristic_without_api_key(monkeypatch):
    monkeypatch.setattr(volume_suggester, "OPENAI_API_KEY", None)

    plan = volume_suggester.infer_search_plan(
        topic="Scowcroft system",
        selected_volume="Being Researched — 1989–1992, Volume II, Organization and Management of Foreign Policy",
    )

    assert plan["normalized_topic"] == "Scowcroft system"
    assert any("Brent Scowcroft" in q for q in plan["query_rewrites"])


def test_retrieve_compiler_assist_documents_uses_inference_fallback(monkeypatch):
    calls = []

    def fake_search(query: str, top_k: int = 20, filters=None, strategy="hybrid"):
        calls.append(query)
        if query == "Scowcroft system":
            return [
                {
                    "chunk_id": "c1",
                    "title": "Policy Note",
                    "text": "A note on NSC process and staffing.",
                    "volume_slug": "frus1989-92v01",
                    "document_number": "3",
                    "history_state_url": "https://history.state.gov/historicaldocuments/frus1989-92v01/d3",
                    "score": 0.4,
                    "source_type": "frus_github",
                }
            ]
        if "Brent Scowcroft" in query:
            return [
                {
                    "chunk_id": "c2",
                    "title": "Memorandum for Scowcroft",
                    "text": "Memorandum on White House foreign policy process.",
                    "volume_slug": "frus1989-92v01",
                    "document_number": "11",
                    "history_state_url": "https://history.state.gov/historicaldocuments/frus1989-92v01/d11",
                    "score": 0.7,
                    "source_type": "frus_github",
                }
            ]
        return []

    monkeypatch.setattr(volume_suggester, "search", fake_search)
    monkeypatch.setattr(volume_suggester, "OPENAI_API_KEY", None)

    payload = volume_suggester.retrieve_compiler_assist_documents(
        topic="Scowcroft system",
        selected_volume="Being Researched — 1989–1992, Volume II, Organization and Management of Foreign Policy",
        top_k=5,
        min_exact_results=5,
    )

    assert payload["used_inference"] is True
    assert payload["results"]
    assert payload["brief"]["top_documents"]
    assert "Brent Scowcroft" in " ".join(calls)
