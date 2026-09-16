"""Tests for AIGenerator (backend/ai_generator.py): does it invoke CourseSearchTool
(via ToolManager) correctly, and does it complete the sequential (up to
MAX_TOOL_ROUNDS) tool-use flow?

`AIGenerator.client` is swapped for `FakeGeminiClient` (see conftest.py) so no network
call is made. FakeGeminiClient enforces the same `Content.role in {"user", "model"}`
constraint the real Gemini API enforces, and records every call so tests can assert on
external behavior: how many API calls were made, what was sent in each one, which
tools were executed with which args, and what text was ultimately returned.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, CourseOutlineTool


def _real_tool_defs(mock_vector_store):
    """Real, schema-valid tool definitions (as search_tools.py produces them),
    needed whenever a test wants `tools` non-empty so _convert_tools actually
    runs (it indexes ["description"]/["input_schema"], unlike a bare {"name": ...})."""
    return [
        CourseSearchTool(mock_vector_store).get_tool_definition(),
        CourseOutlineTool(mock_vector_store).get_tool_definition(),
    ]


def _tool_call_response(*calls):
    """Build a fake Gemini response that requests the given (name, args) tool calls."""
    return SimpleNamespace(
        function_calls=[SimpleNamespace(name=name, args=args) for name, args in calls],
        candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))],
    )


def _text_response(text):
    """Build a fake Gemini response with no tool calls — a final answer."""
    return SimpleNamespace(function_calls=None, text=text)


def test_generate_response_without_tool_call_returns_text_directly(
    fake_gemini_client_factory, mock_vector_store
):
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    response = SimpleNamespace(
        function_calls=None, text="Paris is the capital of France."
    )
    ai_gen.client = fake_gemini_client_factory(responses=[response])
    tool_manager = MagicMock()

    result = ai_gen.generate_response(
        query="What is the capital of France?",
        tools=[CourseSearchTool(mock_vector_store).get_tool_definition()],
        tool_manager=tool_manager,
    )

    assert result == "Paris is the capital of France."
    tool_manager.execute_tool.assert_not_called()
    assert len(ai_gen.client.calls) == 1


def test_convert_tools_builds_matching_function_declarations(mock_vector_store):
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    tool_def = CourseSearchTool(mock_vector_store).get_tool_definition()

    tools = ai_gen._convert_tools([tool_def])

    assert len(tools) == 1
    declarations = tools[0].function_declarations
    assert len(declarations) == 1
    assert declarations[0].name == "search_course_content"
    assert declarations[0].description == tool_def["description"]
    assert declarations[0].parameters_json_schema == tool_def["input_schema"]


def test_generate_response_completes_tool_call_round_trip(
    fake_gemini_client_factory, mock_vector_store
):
    """A single tool-calling round still works exactly as before: 1 tool call
    round + 1 follow-up call = 2 total API calls."""
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    ai_gen.client = fake_gemini_client_factory(
        responses=[
            _tool_call_response(
                ("search_course_content", {"query": "vector databases"})
            ),
            _text_response("Here is what I found about vector databases."),
        ]
    )

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = (
        "[Intro to RAG - Lesson 2]\nVector databases store embeddings."
    )

    result = ai_gen.generate_response(
        query="What are vector databases?",
        tools=[CourseSearchTool(mock_vector_store).get_tool_definition()],
        tool_manager=tool_manager,
    )

    assert result == "Here is what I found about vector databases."
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="vector databases"
    )
    assert len(ai_gen.client.calls) == 2


def test_generate_response_executes_all_tool_calls_in_a_single_round(
    fake_gemini_client_factory,
):
    """Multiple tool calls requested within one round are all executed and
    batched into that round's single function-response turn."""
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    ai_gen.client = fake_gemini_client_factory(
        responses=[
            _tool_call_response(
                ("search_course_content", {"query": "a"}),
                ("get_course_outline", {"course_name": "Intro to RAG"}),
            ),
            _text_response("Combined answer."),
        ]
    )

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["search result", "outline result"]

    result = ai_gen.generate_response(query="q", tools=[], tool_manager=tool_manager)

    assert result == "Combined answer."
    assert tool_manager.execute_tool.call_count == 2
    tool_manager.execute_tool.assert_any_call("search_course_content", query="a")
    tool_manager.execute_tool.assert_any_call(
        "get_course_outline", course_name="Intro to RAG"
    )
    assert len(ai_gen.client.calls) == 2


def test_generate_response_two_rounds_uses_second_search_result(
    fake_gemini_client_factory,
    mock_vector_store,
):
    """The sequential-tool-calling happy path: round 1 gets a course outline,
    round 2 (informed by round 1's result) searches content, round 3 answers."""
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    ai_gen.client = fake_gemini_client_factory(
        responses=[
            _tool_call_response(
                ("get_course_outline", {"course_name": "Intro to RAG"})
            ),
            _tool_call_response(
                ("search_course_content", {"query": "vector databases"})
            ),
            _text_response(
                "Course Y also covers vector databases, like lesson 4 of Intro to RAG."
            ),
        ]
    )

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = [
        "Course Title: Intro to RAG\nLessons: ... Lesson 4: Vector databases ...",
        "[Course Y - Lesson 1]\nVector databases explained.",
    ]

    result = ai_gen.generate_response(
        query="Find a course that covers the same topic as lesson 4 of Intro to RAG",
        tools=_real_tool_defs(mock_vector_store),
        tool_manager=tool_manager,
    )

    assert (
        result
        == "Course Y also covers vector databases, like lesson 4 of Intro to RAG."
    )
    assert tool_manager.execute_tool.call_count == 2
    tool_manager.execute_tool.assert_any_call(
        "get_course_outline", course_name="Intro to RAG"
    )
    tool_manager.execute_tool.assert_any_call(
        "search_course_content", query="vector databases"
    )
    assert len(ai_gen.client.calls) == 3


def test_generate_response_stops_early_when_second_round_has_no_tool_calls(
    fake_gemini_client_factory,
):
    """If Gemini is satisfied after round 1, the loop must not force a 3rd
    (wasted) API call — it should return round 2's text immediately."""
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    ai_gen.client = fake_gemini_client_factory(
        responses=[
            _tool_call_response(("search_course_content", {"query": "embeddings"})),
            _text_response("Embeddings are vector representations of data."),
        ]
    )

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "Embeddings info."

    result = ai_gen.generate_response(
        query="What are embeddings?", tools=[], tool_manager=tool_manager
    )

    assert result == "Embeddings are vector representations of data."
    tool_manager.execute_tool.assert_called_once()
    assert len(ai_gen.client.calls) == 2


def test_generate_response_caps_at_max_rounds_and_forces_final_answer(
    fake_gemini_client_factory,
    mock_vector_store,
):
    """If Gemini still wants a tool after MAX_TOOL_ROUNDS rounds, the round-2
    tool call is still executed (its result isn't discarded), and a final
    call is made with tools stripped so a 3rd tool request can't happen."""
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    ai_gen.client = fake_gemini_client_factory(
        responses=[
            _tool_call_response(("search_course_content", {"query": "a"})),
            _tool_call_response(("search_course_content", {"query": "b"})),
            _text_response("Best answer given what was found."),
        ]
    )

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["result a", "result b"]

    result = ai_gen.generate_response(
        query="q",
        tools=_real_tool_defs(mock_vector_store),
        tool_manager=tool_manager,
    )

    assert result == "Best answer given what was found."
    assert tool_manager.execute_tool.call_count == 2
    tool_manager.execute_tool.assert_any_call("search_course_content", query="a")
    tool_manager.execute_tool.assert_any_call("search_course_content", query="b")
    assert len(ai_gen.client.calls) == 3
    final_config = ai_gen.client.calls[2]["config"]
    assert not final_config.tools


def test_generate_response_recovers_from_tool_execution_exception(
    fake_gemini_client_factory,
    mock_vector_store,
):
    """A tool call that raises must not crash the request — the error is fed
    back to Gemini as a function response, and the flow completes normally."""
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    ai_gen.client = fake_gemini_client_factory(
        responses=[
            _tool_call_response(
                ("search_course_content", {"query": "vector databases"})
            ),
            _text_response("I wasn't able to retrieve that information."),
        ]
    )

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("boom")

    result = ai_gen.generate_response(
        query="What are vector databases?",
        tools=_real_tool_defs(mock_vector_store),
        tool_manager=tool_manager,
    )

    assert result == "I wasn't able to retrieve that information."
    assert len(ai_gen.client.calls) == 2


def test_generate_response_tool_error_string_flows_through_normally(
    fake_gemini_client_factory,
    mock_vector_store,
):
    """A tool returning its own error string (not raising) needs no special
    handling — it's just another function response."""
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    ai_gen.client = fake_gemini_client_factory(
        responses=[
            _tool_call_response(
                ("search_course_content", {"query": "nonexistent topic"})
            ),
            _text_response("No relevant content was found for that topic."),
        ]
    )

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "No relevant content found."

    result = ai_gen.generate_response(
        query="Tell me about nonexistent topic",
        tools=_real_tool_defs(mock_vector_store),
        tool_manager=tool_manager,
    )

    assert result == "No relevant content was found for that topic."
    tool_manager.execute_tool.assert_called_once()
    assert len(ai_gen.client.calls) == 2


def test_generate_response_preserves_history_across_rounds(
    fake_gemini_client_factory, mock_vector_store
):
    """Each successive API call's `contents` must grow (not reset), proving
    conversation context — including earlier rounds' tool results — carries
    forward across rounds."""
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    ai_gen.client = fake_gemini_client_factory(
        responses=[
            _tool_call_response(
                ("get_course_outline", {"course_name": "Intro to RAG"})
            ),
            _tool_call_response(
                ("search_course_content", {"query": "vector databases"})
            ),
            _text_response("Final answer."),
        ]
    )

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["outline result", "search result"]

    ai_gen.generate_response(
        query="Find a related course",
        tools=_real_tool_defs(mock_vector_store),
        tool_manager=tool_manager,
    )

    calls = ai_gen.client.calls
    assert len(calls) == 3
    assert len(calls[1]["contents"]) > len(calls[0]["contents"])
    assert len(calls[2]["contents"]) > len(calls[1]["contents"])


def test_generate_response_includes_conversation_history_in_system_instruction(
    fake_gemini_client_factory,
):
    ai_gen = AIGenerator(api_key="test-key", model="test-model")
    response = SimpleNamespace(function_calls=None, text="Sure, following up on that.")
    ai_gen.client = fake_gemini_client_factory(responses=[response])

    ai_gen.generate_response(
        query="And what about lesson 3?",
        conversation_history=(
            "User: What is RAG?\nAssistant: It's retrieval-augmented generation."
        ),
        tools=None,
        tool_manager=None,
    )

    sent_config = ai_gen.client.calls[0]["config"]
    assert "Previous conversation:" in sent_config.system_instruction
    assert "retrieval-augmented generation" in sent_config.system_instruction
