from google import genai
from google.genai import types
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Google's Gemini API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
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

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Pre-build base generation config params
        self.base_config = {
            "temperature": 0,
            "max_output_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
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
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=query)])]

        # Prepare API call config efficiently
        config_kwargs = {
            **self.base_config,
            "system_instruction": system_content
        }

        # Add tools if available
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        # Get response from Gemini
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs)
        )

        # Handle tool execution if needed
        if response.function_calls and tool_manager:
            return self._handle_tool_execution(response, contents, system_content, tool_manager)

        # Return direct response
        return response.text

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
                parameters_json_schema=tool["input_schema"]
            )
            for tool in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def _handle_tool_execution(self, initial_response, contents: List[types.Content],
                               system_content: str, tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing function call requests
            contents: The conversation contents sent so far
            system_content: The system instruction used for the initial call
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing contents and add the model's function-call turn
        contents = contents + [initial_response.candidates[0].content]

        # Execute all tool calls and collect results
        response_parts = []
        for call in initial_response.function_calls:
            tool_result = tool_manager.execute_tool(
                call.name,
                **call.args
            )

            response_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": tool_result}
                )
            )

        # Add tool results as a single content entry
        if response_parts:
            contents.append(types.Content(role="tool", parts=response_parts))

        # Final call without tools, to force a plain-text answer
        final_response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                **self.base_config,
                system_instruction=system_content
            )
        )
        return final_response.text
