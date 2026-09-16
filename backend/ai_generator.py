from google import genai
from google.genai import types
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Google's Gemini API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: use for questions about specific course content or detailed educational material within lessons
- **get_course_outline**: use for questions about a course's structure/syllabus — returns the course title, course link, and the full list of lessons (number and title for every lesson)
- **Up to 2 sequential tool calls per query**: call a tool, review its result, and call a second tool (or the same tool again with refined arguments) only if the first result doesn't fully answer the question — e.g. comparing two courses/lessons, a multi-part question, or when the first result (like a course outline) reveals what to search for next (like a specific lesson title)
- Do not make a second call if the first result already answers the question
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content first, then answer
- **Course outline/structure questions**: Use get_course_outline first, then answer — always include the course title, course link, and every lesson's number and title from the result
- **Multi-part or comparison questions**: make a first tool call, use its result to inform a second tool call if needed, then synthesize both into one answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Maximum number of sequential tool-calling rounds per user query. Each
    # round is one full generate_content call in which Gemini can reason
    # about every previous round's tool results before deciding whether to
    # call another tool or answer directly. Bumping this later is a one-line
    # change; the loop in _run_tool_loop doesn't need to change shape.
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Pre-build base generation config params
        self.base_config = {"temperature": 0, "max_output_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare initial conversation content
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=query)])
        ]

        # Get response from Gemini
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._build_config(system_content, tools),
        )

        # Handle tool execution if needed
        if response.function_calls and tool_manager:
            return self._run_tool_loop(
                response, contents, system_content, tools, tool_manager
            )

        # Return direct response
        return response.text

    def _build_config(
        self, system_content: str, tools: Optional[List[Dict[str, Any]]] = None
    ) -> types.GenerateContentConfig:
        """
        Build a GenerateContentConfig, attaching tools only when provided.

        Args:
            system_content: The system instruction for this call
            tools: Anthropic-shaped tool definitions to attach, or None/empty to omit tools

        Returns:
            A GenerateContentConfig ready to pass to generate_content
        """
        config_kwargs = {**self.base_config, "system_instruction": system_content}
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)
        return types.GenerateContentConfig(**config_kwargs)

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[types.Tool]:
        """
        Convert Anthropic-shaped tool definitions (name/description/input_schema,
        as produced by search_tools.py) into Gemini FunctionDeclaration tools.

        Args:
            tools: List of tool definitions with name, description, input_schema

        Returns:
            List containing a single Tool with all function declarations
        """
        declarations = [
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters_json_schema=tool["input_schema"],
            )
            for tool in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def _execute_tool_calls(self, function_calls, tool_manager) -> List[types.Part]:
        """
        Execute a round's tool calls and build the function-response parts to
        send back to Gemini. A tool call that raises is not allowed to crash
        the request: its error is turned into a plain-text function response,
        exactly like a tool that returns its own error string, so Gemini can
        see what happened and decide how to respond.

        Args:
            function_calls: The function call requests from a Gemini response
            tool_manager: Manager to execute tools

        Returns:
            List of Parts (one function response per call) to send back to Gemini
        """
        response_parts = []
        for call in function_calls:
            try:
                tool_result = tool_manager.execute_tool(call.name, **call.args)
            except Exception as e:
                tool_result = f"Tool '{call.name}' failed: {e}"

            response_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": tool_result}
                )
            )
        return response_parts

    def _run_tool_loop(
        self,
        response,
        contents: List[types.Content],
        system_content: str,
        tools: Optional[List],
        tool_manager,
    ) -> str:
        """
        Drive up to MAX_TOOL_ROUNDS sequential rounds of tool calling, giving
        Gemini a chance to reason about each round's results before deciding
        whether to call another tool or answer directly.

        Terminates when: (a) MAX_TOOL_ROUNDS rounds have been completed, (b) a
        response has no tool calls (Gemini is ready to answer), or execution
        of a tool call fails (handled gracefully in _execute_tool_calls, which
        feeds the error back to Gemini instead of raising).

        Args:
            response: The first response that requested tool call(s)
            contents: The conversation contents sent so far
            system_content: The system instruction used for prior calls
            tools: The tool definitions to keep offering on intermediate rounds
            tool_manager: Manager to execute tools

        Returns:
            Final response text after the tool-calling rounds complete
        """
        for round_num in range(1, self.MAX_TOOL_ROUNDS + 1):
            # Termination (b): Gemini didn't ask for a tool this round — it's
            # ready to answer.
            if not response.function_calls:
                return response.text

            # Add the model's function-call turn, then execute the calls and
            # add their results as a single "user" turn (Gemini's Content.role
            # only accepts "user"/"model" — there's no OpenAI-style "tool" role).
            contents.append(response.candidates[0].content)
            response_parts = self._execute_tool_calls(
                response.function_calls, tool_manager
            )
            if response_parts:
                contents.append(types.Content(role="user", parts=response_parts))

            # Termination (a): round budget exhausted. Don't loop back for
            # another tools-enabled call — fall through to the forced,
            # tools-stripped final call below.
            if round_num == self.MAX_TOOL_ROUNDS:
                break

            # Tools stay attached: Gemini may use this round's results to
            # decide on another tool call, or may already be ready to answer.
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._build_config(system_content, tools),
            )

        # Final call without tools, to force a plain-text answer once the
        # round budget is spent (or if we ever fell through some other way).
        final_response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._build_config(system_content, tools=None),
        )
        return final_response.text
