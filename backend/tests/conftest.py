"""Shared fixtures and test doubles for the backend test suite.

Notes on approach:
- The only real I/O boundaries in this codebase are ChromaDB/sentence-transformers
  (behind `VectorStore`) and the network call to Gemini (behind `AIGenerator.client`).
  Tests replace exactly those two things and use the real classes for everything else
  (`ToolManager`, `CourseSearchTool`, `CourseOutlineTool`, `RAGSystem`, `SessionManager`)
  so the actual wiring between components is what's under test.
- `FakeGeminiClient` deliberately enforces the same constraint the real Gemini API
  enforces on `types.Content.role` ("user" or "model" only). This lets tests reproduce
  the production "query failed" bug deterministically and without any network access.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# Ensure `backend/` is importable even if pytest's rootdir-based pythonpath
# resolution doesn't kick in (belt-and-suspenders alongside pyproject.toml's
# [tool.pytest.ini_options] pythonpath setting).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_store import VectorStore, SearchResults
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from config import config as real_config
from rag_system import RAGSystem


class FakeGeminiClient:
    """Test double for `google.genai.Client` that never touches the network.

    Mimics the one constraint that matters for this bug: the real Gemini API only
    accepts `Content.role` of "user" or "model". Any other role raises, just like a
    real 400 INVALID_ARGUMENT response would.
    """

    class _Models:
        def __init__(self, outer: "FakeGeminiClient"):
            self._outer = outer

        def generate_content(self, *, model, contents, config):
            for content in contents:
                role = getattr(content, "role", None)
                if role not in ("user", "model", None):
                    raise ValueError(
                        f"Invalid Content.role {role!r}; Gemini only accepts 'user' or 'model'"
                    )
            # Snapshot the list now: callers may keep mutating (appending to)
            # the same `contents` list across rounds, and callers should be
            # able to inspect each call's `contents` as they were *at that
            # call*, not as they end up after later rounds mutate the list.
            self._outer.calls.append({"model": model, "contents": list(contents), "config": config})
            index = len(self._outer.calls) - 1
            if index >= len(self._outer.responses):
                raise AssertionError(
                    f"FakeGeminiClient received more generate_content calls "
                    f"({index + 1}) than scripted responses ({len(self._outer.responses)})"
                )
            return self._outer.responses[index]

    def __init__(self, responses: List[Any]):
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []
        self.models = FakeGeminiClient._Models(self)


def make_search_results(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    distances: Optional[List[float]] = None,
) -> SearchResults:
    """Build a real `SearchResults` without going through ChromaDB."""
    return SearchResults(
        documents=documents,
        metadata=metadatas,
        distances=distances or [0.1] * len(documents),
    )


@pytest.fixture
def fake_gemini_client_factory():
    """Fixture wrapper around `FakeGeminiClient`, for the same reason as
    `search_results_factory` above."""
    return FakeGeminiClient


@pytest.fixture
def search_results_factory():
    """Fixture wrapper around `make_search_results`, so test files don't need a
    same-package `import conftest` (which is awkward once `backend/tests` is a
    package with `__init__.py`)."""
    return make_search_results


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """A `VectorStore` double preconfigured with one course/lesson worth of data."""
    store = MagicMock(spec=VectorStore)
    store.search.return_value = make_search_results(
        documents=["Vector databases store embeddings for similarity search."],
        metadatas=[{"course_title": "Intro to RAG", "lesson_number": 2}],
    )
    store.get_lesson_link.return_value = "https://example.com/intro-to-rag/lesson-2"
    store.get_course_link.return_value = "https://example.com/intro-to-rag"
    store.get_course_outline.return_value = {
        "title": "Intro to RAG",
        "course_link": "https://example.com/intro-to-rag",
        "instructor": "Jane Doe",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "What is RAG?", "lesson_link": "https://example.com/intro-to-rag/lesson-1"},
            {"lesson_number": 2, "lesson_title": "Vector databases", "lesson_link": "https://example.com/intro-to-rag/lesson-2"},
        ],
    }
    return store


@pytest.fixture
def tool_manager_with_search_tool(mock_vector_store: MagicMock) -> ToolManager:
    """A real `ToolManager` with a real `CourseSearchTool` + `CourseOutlineTool` on a mock store."""
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))
    manager.register_tool(CourseOutlineTool(mock_vector_store))
    return manager


@pytest.fixture
def rag_system(monkeypatch: pytest.MonkeyPatch, mock_vector_store: MagicMock) -> RAGSystem:
    """A real `RAGSystem` (real ToolManager/CourseSearchTool/CourseOutlineTool/AIGenerator/
    SessionManager) with only `VectorStore` replaced by `mock_vector_store` — the one real
    I/O dependency (ChromaDB + sentence-transformers) that would otherwise run here.

    `ai_generator.client` is still the real `genai.Client`; tests that exercise the
    Gemini call path should replace it with a `fake_gemini_client_factory(...)` instance
    (see test_ai_generator.py/test_rag_system.py), or mock `ai_generator.generate_response`
    directly when they only care about RAGSystem's own orchestration logic.
    """
    monkeypatch.setattr("rag_system.VectorStore", lambda *args, **kwargs: mock_vector_store)
    return RAGSystem(real_config)
