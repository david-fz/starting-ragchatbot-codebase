"""API endpoint tests for `backend/app.py`'s `/api/*` routes.

These tests run against `test_app` (conftest.py), an inline FastAPI app that
mirrors app.py's route bodies against a mocked `RAGSystem` (`mock_rag_system`),
since importing `app.py` directly would try to mount `../frontend` (absent in
the test environment) at import time.
"""
import pytest

pytestmark = pytest.mark.api


class TestQueryEndpoint:
    def test_query_with_existing_session_returns_answer_and_sources(self, client, mock_rag_system):
        response = client.post(
            "/api/query", json={"query": "What are vector databases?", "session_id": "abc-123"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Vector databases store embeddings for similarity search."
        assert body["sources"] == [
            {"text": "Intro to RAG - Lesson 2", "link": "https://example.com/intro-to-rag/lesson-2"}
        ]
        assert body["session_id"] == "abc-123"
        mock_rag_system.query.assert_called_once_with("What are vector databases?", "abc-123")

    def test_query_without_session_id_creates_one(self, client, mock_rag_system):
        response = client.post("/api/query", json={"query": "What are vector databases?"})

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "test-session-id"
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What are vector databases?", "test-session-id")

    def test_query_with_no_sources_returns_empty_list(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("General knowledge answer.", [])

        response = client.post("/api/query", json={"query": "What is 2+2?"})

        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_query_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={"session_id": "abc-123"})

        assert response.status_code == 422

    def test_query_propagates_rag_system_error_as_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("Gemini API unavailable")

        response = client.post("/api/query", json={"query": "What are vector databases?"})

        assert response.status_code == 500
        assert response.json()["detail"] == "Gemini API unavailable"


class TestCoursesEndpoint:
    def test_get_course_stats_returns_analytics(self, client, mock_rag_system):
        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {
            "total_courses": 2,
            "course_titles": ["Intro to RAG", "Advanced RAG"],
        }
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_course_stats_with_no_courses(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {"total_courses": 0, "course_titles": []}

    def test_get_course_stats_propagates_error_as_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("ChromaDB unavailable")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert response.json()["detail"] == "ChromaDB unavailable"


class TestClearSessionEndpoint:
    def test_clear_session_succeeds(self, client, mock_rag_system):
        response = client.post("/api/session/clear", json={"session_id": "abc-123"})

        assert response.status_code == 200
        assert response.json() == {"success": True}
        mock_rag_system.session_manager.clear_session.assert_called_once_with("abc-123")

    def test_clear_session_missing_session_id_returns_422(self, client):
        response = client.post("/api/session/clear", json={})

        assert response.status_code == 422

    def test_clear_session_propagates_error_as_500(self, client, mock_rag_system):
        mock_rag_system.session_manager.clear_session.side_effect = RuntimeError("boom")

        response = client.post("/api/session/clear", json={"session_id": "abc-123"})

        assert response.status_code == 500
        assert response.json()["detail"] == "boom"


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")

        assert response.status_code == 200
