"""Tests for how RAGSystem.query() handles content-related questions
(backend/rag_system.py).

The `rag_system` fixture (conftest.py) builds a real RAGSystem with only VectorStore
swapped for a mock (the one real I/O dependency) — ToolManager, CourseSearchTool,
CourseOutlineTool, AIGenerator, and SessionManager are all the real classes, so these
tests exercise the actual orchestration wiring.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_rag_system_handles_content_question_end_to_end(
    rag_system, fake_gemini_client_factory
):
    """Regression test for the reported bug: a content question should complete and
    return an answer plus sources. As of ai_generator.py:151, this currently raises
    (the same failure the app surfaces as HTTP 500 / "Query failed"), because
    RAGSystem.query() has no try/except around ai_generator.generate_response and lets
    it propagate untouched.
    """
    initial_response = SimpleNamespace(
        function_calls=[
            SimpleNamespace(
                name="search_course_content", args={"query": "vector databases"}
            )
        ],
        candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))],
    )
    final_response = SimpleNamespace(
        function_calls=None,
        text="Vector databases store embeddings for fast similarity search.",
    )
    rag_system.ai_generator.client = fake_gemini_client_factory(
        responses=[initial_response, final_response]
    )

    answer, sources = rag_system.query("What are vector databases?", session_id=None)

    assert answer == "Vector databases store embeddings for fast similarity search."
    assert sources == [
        {
            "text": "Intro to RAG - Lesson 2",
            "link": "https://example.com/intro-to-rag/lesson-2",
        }
    ]


def test_query_passes_prompt_and_tools_to_ai_generator(rag_system):
    rag_system.ai_generator = MagicMock()
    rag_system.ai_generator.generate_response.return_value = "some answer"

    rag_system.query("What are vector databases?", session_id=None)

    _, kwargs = rag_system.ai_generator.generate_response.call_args
    assert "What are vector databases?" in kwargs["query"]
    assert kwargs["tool_manager"] is rag_system.tool_manager
    tool_names = {t["name"] for t in kwargs["tools"]}
    assert tool_names == {"search_course_content", "get_course_outline"}


def test_query_returns_and_resets_sources(rag_system):
    # Drive a real tool execution (via the real ToolManager/CourseSearchTool on
    # the mocked VectorStore) so sources land in tool_manager the same way a
    # real Gemini tool call would, rather than poking a tool's internals.
    rag_system.tool_manager.execute_tool(
        "search_course_content", query="vector databases"
    )
    rag_system.ai_generator = MagicMock()
    rag_system.ai_generator.generate_response.return_value = "some answer"

    _, sources = rag_system.query("What are vector databases?", session_id=None)

    assert sources == [
        {
            "text": "Intro to RAG - Lesson 2",
            "link": "https://example.com/intro-to-rag/lesson-2",
        }
    ]
    assert rag_system.tool_manager.get_last_sources() == []


def test_query_updates_session_history(rag_system):
    rag_system.ai_generator = MagicMock()
    rag_system.ai_generator.generate_response.return_value = (
        "RAG combines retrieval and generation."
    )
    session_id = rag_system.session_manager.create_session()

    rag_system.query("What is RAG?", session_id=session_id)

    history = rag_system.session_manager.get_conversation_history(session_id)
    assert "What is RAG?" in history
    assert "RAG combines retrieval and generation." in history


def test_query_does_not_swallow_ai_generator_exceptions(rag_system):
    rag_system.ai_generator = MagicMock()
    rag_system.ai_generator.generate_response.side_effect = ValueError(
        "Invalid Content.role 'tool'; Gemini only accepts 'user' or 'model'"
    )

    with pytest.raises(ValueError):
        rag_system.query("What are vector databases?", session_id=None)
