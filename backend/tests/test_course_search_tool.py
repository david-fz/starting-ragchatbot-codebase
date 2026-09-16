"""Unit tests for CourseSearchTool.execute() (backend/search_tools.py).

VectorStore is mocked throughout — these tests are about CourseSearchTool's own
logic (parameter passthrough, error/empty-result handling, formatting, and source
tracking), not ChromaDB behavior.
"""

from search_tools import CourseSearchTool
from vector_store import SearchResults


def test_execute_with_results_and_no_filters_formats_and_queries_correctly(
    mock_vector_store,
):
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="what are vector databases")

    mock_vector_store.search.assert_called_once_with(
        query="what are vector databases", course_name=None, lesson_number=None
    )
    assert result == (
        "[Intro to RAG - Lesson 2]\n"
        "Vector databases store embeddings for similarity search."
    )


def test_execute_passes_course_name_and_lesson_number_through(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="embeddings", course_name="Intro to RAG", lesson_number=2)

    mock_vector_store.search.assert_called_once_with(
        query="embeddings", course_name="Intro to RAG", lesson_number=2
    )


def test_execute_returns_store_error_verbatim(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults.empty(
        "No course found matching 'Nonexistent Course'"
    )
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="anything", course_name="Nonexistent Course")

    assert result == "No course found matching 'Nonexistent Course'"


def test_execute_empty_results_no_filters(mock_vector_store, search_results_factory):
    mock_vector_store.search.return_value = search_results_factory(
        documents=[], metadatas=[]
    )
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="anything")

    assert result == "No relevant content found."


def test_execute_empty_results_with_course_name_only(
    mock_vector_store, search_results_factory
):
    mock_vector_store.search.return_value = search_results_factory(
        documents=[], metadatas=[]
    )
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="anything", course_name="Intro to RAG")

    assert result == "No relevant content found in course 'Intro to RAG'."


def test_execute_empty_results_with_lesson_number_only(
    mock_vector_store, search_results_factory
):
    mock_vector_store.search.return_value = search_results_factory(
        documents=[], metadatas=[]
    )
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="anything", lesson_number=3)

    assert result == "No relevant content found in lesson 3."


def test_execute_empty_results_with_course_name_and_lesson_number(
    mock_vector_store, search_results_factory
):
    mock_vector_store.search.return_value = search_results_factory(
        documents=[], metadatas=[]
    )
    tool = CourseSearchTool(mock_vector_store)

    result = tool.execute(query="anything", course_name="Intro to RAG", lesson_number=3)

    assert result == "No relevant content found in course 'Intro to RAG' in lesson 3."


def test_execute_dedupes_sources_from_same_lesson_and_resolves_lesson_link(
    mock_vector_store, search_results_factory
):
    mock_vector_store.search.return_value = search_results_factory(
        documents=["First chunk about embeddings.", "Second chunk, same lesson."],
        metadatas=[
            {"course_title": "Intro to RAG", "lesson_number": 2},
            {"course_title": "Intro to RAG", "lesson_number": 2},
        ],
    )
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="embeddings")

    assert tool.last_sources == [
        {
            "text": "Intro to RAG - Lesson 2",
            "link": "https://example.com/intro-to-rag/lesson-2",
        }
    ]
    # Link lookup happens per chunk, before dedup, so it's called once per chunk
    # (2 chunks here) even though only one deduped source ends up in last_sources.
    assert mock_vector_store.get_lesson_link.call_args_list == [
        (("Intro to RAG", 2),),
        (("Intro to RAG", 2),),
    ]
    mock_vector_store.get_course_link.assert_not_called()


def test_execute_course_level_chunk_resolves_course_link_not_lesson_link(
    mock_vector_store, search_results_factory
):
    mock_vector_store.search.return_value = search_results_factory(
        documents=["Course-level overview text."],
        metadatas=[{"course_title": "Intro to RAG", "lesson_number": None}],
    )
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="overview")

    assert tool.last_sources == [
        {"text": "Intro to RAG", "link": "https://example.com/intro-to-rag"}
    ]
    mock_vector_store.get_course_link.assert_called_once_with("Intro to RAG")
    mock_vector_store.get_lesson_link.assert_not_called()
