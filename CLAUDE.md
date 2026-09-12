# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Always use `uv` for dependency management and for running any Python file — never call `pip` or `python`/`python3` directly in this repo** (e.g. `uv run python backend/some_script.py`, not `python backend/some_script.py`).

```bash
# Setup (first time)
cp .env.example .env        # then set ANTHROPIC_API_KEY inside
uv sync

# Run the app (backend serves the frontend too)
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

- App: http://localhost:8000  |  API docs (Swagger): http://localhost:8000/docs
- No test suite, linter, or formatter is configured in this repo.
- Dependencies are managed with `uv` (see `pyproject.toml`); there is no `requirements.txt`. Add dependencies with `uv add <package>`, run any script with `uv run <script>`, never `pip install`.
- Python 3.13+ required (`.python-version`).

## Architecture

This is a single-process RAG (Retrieval-Augmented Generation) app: **FastAPI serves both the REST API and the static frontend** from one server (`backend/app.py` mounts `../frontend` as static files at `/`). There is no separate frontend build/deploy.

### Request flow (`POST /api/query`)

`app.py` → `RAGSystem.query()` orchestrates everything:

1. `SessionManager` fetches prior conversation history (in-memory only, keyed by `session_id`).
2. `AIGenerator` calls Claude with the system prompt, history, and tool definitions (`tool_choice: auto`).
3. Claude decides whether to call the `search_course_content` tool (only for course-specific questions — general knowledge is answered directly per the system prompt in `ai_generator.py`).
4. If invoked, `CourseSearchTool.execute()` (`search_tools.py`) calls `VectorStore.search()`, which first resolves a fuzzy course name via semantic search against the `course_catalog` collection, then queries the `course_content` collection (ChromaDB) filtered by course/lesson.
5. Tool results are sent back to Claude in a **second** API call (no further tool loop — one search per query, enforced by the system prompt) to produce the final answer.
6. `RAGSystem` pulls tracked sources off the tool (`ToolManager.get_last_sources()`/`reset_sources()`), updates session history, and returns `(answer, sources)` to the API layer.

### Component responsibilities (`backend/`)

- `rag_system.py` — orchestrator/composition root; wires all components together. Start here to trace any end-to-end behavior.
- `vector_store.py` — the only place that talks to ChromaDB. Maintains two collections: `course_catalog` (one doc per course, used for fuzzy course-name resolution) and `course_content` (chunked text, used for actual retrieval). Embeddings use `sentence-transformers` (`all-MiniLM-L6-v2`).
- `document_processor.py` — parses course documents into `Course`/`Lesson`/`CourseChunk` (see expected file format below) and chunks lesson text sentence-aware with overlap (`CHUNK_SIZE`/`CHUNK_OVERLAP` from `config.py`). The first chunk of each lesson gets a `"Lesson N content: ..."` prefix for retrieval context.
- `search_tools.py` — implements Anthropic's tool-use pattern: `Tool` ABC, `CourseSearchTool` (schema + `execute`), and `ToolManager` (registry, dispatch, source tracking). Add new retrieval capabilities here as new `Tool` subclasses registered in `rag_system.py`.
- `ai_generator.py` — the only place that calls the Anthropic API. Owns the system prompt and the two-call tool-execution flow (`_handle_tool_execution`). No streaming.
- `session_manager.py` — in-memory conversation history only; nothing is persisted, history is lost on restart, and is capped at `MAX_HISTORY` exchanges.
- `models.py` — shared Pydantic models (`Course`, `Lesson`, `CourseChunk`) used across document processing, vector storage, and the API layer.
- `config.py` — single dataclass config (Claude model, chunk size/overlap, `MAX_RESULTS`, `CHROMA_PATH`, `MAX_HISTORY`), loaded from `.env` via `python-dotenv`.
- `app.py` — FastAPI routes (`/api/query`, `/api/courses`) and the startup hook that auto-ingests `../docs` into ChromaDB on launch (skips courses whose title already exists, so restarts don't duplicate data).

### Document ingestion format

Course files in `docs/` (`.txt`, `.pdf`, `.docx`) are expected in this exact structure (see `document_processor.py`):

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```

Course title is the unique ID used both as the ChromaDB document ID in `course_catalog` and as the `course_title` filter field in `course_content`.

### Persistence

- Vector data: `backend/chroma_db/` (gitignored, persists across restarts).
- Conversation history: in-memory dict, lost on restart.
- Ingestion is additive/idempotent by course title, not a full resync — deleting a file from `docs/` does not remove it from ChromaDB; use `VectorStore.clear_all_data()` for a full rebuild.
