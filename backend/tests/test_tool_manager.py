"""Tests for ToolManager (backend/search_tools.py), focused on source tracking
across multiple execute_tool() calls — needed now that a single query can
involve up to 2 sequential tool calls (see AIGenerator.MAX_TOOL_ROUNDS).

Before this fix, ToolManager.get_last_sources() returned only the first
registered tool's last_sources, so calling two different tools (or the same
tool twice) within one query would silently drop earlier sources from the UI.
"""

from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool


def test_get_last_sources_accumulates_across_different_tools(mock_vector_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))
    manager.register_tool(CourseOutlineTool(mock_vector_store))

    manager.execute_tool("get_course_outline", course_name="Intro to RAG")
    manager.execute_tool("search_course_content", query="vector databases")

    sources = manager.get_last_sources()
    assert sources == [
        {"text": "Intro to RAG", "link": "https://example.com/intro-to-rag"},
        {
            "text": "Intro to RAG - Lesson 2",
            "link": "https://example.com/intro-to-rag/lesson-2",
        },
    ]


def test_get_last_sources_accumulates_across_repeated_calls_to_same_tool(
    mock_vector_store, search_results_factory
):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))

    mock_vector_store.search.return_value = search_results_factory(
        documents=["First result"],
        metadatas=[{"course_title": "Course A", "lesson_number": 1}],
    )
    manager.execute_tool("search_course_content", query="first search")

    mock_vector_store.search.return_value = search_results_factory(
        documents=["Second result"],
        metadatas=[{"course_title": "Course B", "lesson_number": 3}],
    )
    manager.execute_tool("search_course_content", query="second search")

    sources = manager.get_last_sources()
    assert sources == [
        {
            "text": "Course A - Lesson 1",
            "link": "https://example.com/intro-to-rag/lesson-2",
        },
        {
            "text": "Course B - Lesson 3",
            "link": "https://example.com/intro-to-rag/lesson-2",
        },
    ]


def test_reset_sources_clears_accumulated_sources(mock_vector_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))

    manager.execute_tool("search_course_content", query="vector databases")
    assert manager.get_last_sources() != []

    manager.reset_sources()

    assert manager.get_last_sources() == []


def test_execute_tool_with_unknown_name_does_not_affect_accumulated_sources(
    mock_vector_store,
):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))

    manager.execute_tool("search_course_content", query="vector databases")
    sources_after_first_call = manager.get_last_sources()

    result = manager.execute_tool("nonexistent_tool")

    assert result == "Tool 'nonexistent_tool' not found"
    assert manager.get_last_sources() == sources_after_first_call
