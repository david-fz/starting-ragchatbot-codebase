from typing import Dict, Any, Optional, Protocol, List
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return a JSON-schema tool definition (name/description/input_schema) for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources: List[Dict[str, Optional[str]]] = []  # Track sources (text + optional link) from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return a JSON-schema tool definition (name/description/input_schema) for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        seen_sources = set()  # De-dupe by (text, link) across chunks

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Track source for the UI
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"

            # Resolve a link: lesson-specific link when we have a lesson
            # number, otherwise fall back to the course link. Both store
            # methods return None on any lookup failure.
            if lesson_num is not None:
                link = self.store.get_lesson_link(course_title, lesson_num)
            else:
                link = self.store.get_course_link(course_title)

            # Only add each distinct (text, link) source once, even if
            # multiple chunks come from the same course/lesson.
            source_key = (source, link)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({"text": source, "link": link})

            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)

class CourseOutlineTool(Tool):
    """Tool for retrieving a course's outline: title, link, and full lesson list"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources: List[Dict[str, Optional[str]]] = []  # Track source (text + optional link) from last lookup

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return a JSON-schema tool definition (name/description/input_schema) for this tool"""
        return {
            "name": "get_course_outline",
            "description": (
                "Get a course's outline: its title, course link, and complete list "
                "of lessons (lesson number and title for each). Use this for "
                "questions about a course's structure, syllabus, or which lessons "
                "it contains — not for questions about specific lesson content."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the outline lookup.

        Args:
            course_name: Course title (partial matches work)

        Returns:
            Formatted outline text, or a not-found message
        """
        outline = self.store.get_course_outline(course_name)

        if outline is None:
            return f"No course found matching '{course_name}'."

        return self._format_outline(outline)

    def _format_outline(self, outline: Dict[str, Any]) -> str:
        """Format course outline into a readable string for the LLM"""
        title = outline.get("title", "Unknown course")
        course_link = outline.get("course_link")
        lessons = outline.get("lessons", [])

        lines = [
            f"Course Title: {title}",
            f"Course Link: {course_link if course_link else 'N/A'}",
            f"Lessons ({len(lessons)}):"
        ]
        for lesson in sorted(lessons, key=lambda l: l.get("lesson_number", 0)):
            lines.append(f"  Lesson {lesson.get('lesson_number')}: {lesson.get('lesson_title')}")

        # Track a source for UI citation consistency with CourseSearchTool
        self.last_sources = [{"text": title, "link": course_link}]

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for LLM tool/function calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []