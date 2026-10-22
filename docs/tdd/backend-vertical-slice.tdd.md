# TDD Evidence: Backend Vertical Slice (Plan Phases 0–1)

**Source plan:** [`Plan.md`](../../Plan.md), Phase 0 (steps 1–5) and Phase 1 (steps 6–10)
**Branch:** `feat/backend-vertical-slice`
**Runner:** `uv` + `pytest` (Python 3.13.17), from `apps/api`

## User journeys (from the plan's requirements and acceptance criteria)

1. As a writer, I submit a topic prompt and get back one article with cited sources.
2. As a writer, if my prompt is ambiguous, I'm asked one clarifying question, and the run continues after I answer.
3. As a writer, I can watch each agent start and finish live while the run is in progress.
4. As an operator, a broken or hostile web page, or a failing third-party service, does not crash the run or reach internal network addresses.

## RED → GREEN checkpoints

| Cycle | Scope | RED commit (evidence) | GREEN commit (evidence) |
|---|---|---|---|
| A | Research tools + knowledge base | `8ad805e`: `pytest` → 8 collection errors, `ModuleNotFoundError: cwas.tools` / `cwas.kb` | `a5790e2`: 68 passed, coverage 97.48%, ruff + mypy strict clean |
| B | LLM wrapper, 5 agents, LangGraph orchestrator | `a65fb32`: 7 collection errors, `ModuleNotFoundError: cwas.llm` | `c92a00b`: 103 passed, coverage 98.5%, ruff + mypy strict clean |
| C | Run manager, HTTP + SSE API, wiring | `adeda6e`: 2 collection errors, `ModuleNotFoundError: cwas.runs` | `498fe5e`: 120 passed, coverage 98.03%, ruff + mypy strict clean |

| D | Free local LLM (Ollama) + DuckDuckGo search | `c5d9881`: 2 collection errors (`OllamaLLM`, `DuckDuckGoSearch` missing); `tests/test_wiring.py` → 6 failed, 1 passed | `a833974`: 137 passed, coverage 98%, ruff + mypy strict clean |

Every RED in cycles A–C was a compile-time (import) failure caused by the missing implementation, not by broken test setup. In cycle D the wiring tests failed at runtime for the intended reason: Ollama settings and wiring didn't exist yet.

One addition after GREEN: in cycle C, `test_app_lifespan_wires_a_real_run_manager` was added to cover existing app-startup code. It was not written test-first.

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | URLs with non-http schemes, private, loopback or cloud-metadata addresses, or names that resolve to private IPs are refused | `tests/tools/test_url_safety.py` | unit | PASS |
| 2 | Scraper honours robots.txt (4xx = allow, 5xx = deny), never follows a redirect to an unsafe host, extracts the main article, caches pages | `tests/tools/test_scrape.py` | unit (HTTP mocked) | PASS |
| 3 | Tavily and Voyage clients map responses, batch requests, and raise typed errors on HTTP/network failure | `tests/tools/test_search.py`, `test_embed.py` | unit (HTTP mocked) | PASS |
| 4 | Grammar fixes never touch citations, links, code or capitalised proper nouns; offsets stay valid | `tests/tools/test_grammar.py` | unit | PASS |
| 5 | Chunks respect the size limit, overlap, and never drop text | `tests/kb/test_chunker.py` | unit | PASS |
| 6 | Vector search ranks by cosine, dedupes, isolates runs, filters source vs competitor | `tests/kb/test_store.py`, `test_knowledge_base.py` | unit | PASS |
| 7 | Opus and Sonnet calls enable server-side refusal fallbacks; refusals and truncation raise | `tests/test_llm.py` | unit | PASS |
| 8 | User text and scraped pages reach the model only inside untrusted-data tags | `test_prompt_enhancer.py`, `test_writer.py` | unit | PASS |
| 9 | The writer can't cite a source that doesn't exist: invented `[n]` markers are stripped | `tests/agents/test_writer.py` | unit | PASS |
| 10 | Research tags ranking pages as competitors, dedupes URLs, enforces limits, turns a failed query into a warning, and fails clearly when nothing is collected | `tests/agents/test_researcher.py` | unit | PASS |
| 11 | Full graph: 5 agents in order, a cited article with a Sources section, interrupt then resume, `agent_failed` event on error | `tests/graph/test_graph.py` | integration | PASS |
| 12 | API: 202/404/409/422 contracts, SSE replay ends in `done`/`error`, clarification round-trip, per-run cleanup | `tests/api/test_api.py` | integration | PASS |
| 13 | Production wiring fails fast without `TAVILY_API_KEY` and builds without network calls | `tests/test_wiring.py` | integration | PASS |

## Coverage

`uv run pytest --cov` → **120 passed, total coverage 98.03%** (the gate is 80%). The only file below 85% is `main.py` (2 lines, the ASGI entrypoint), and it was exercised by the manual smoke test below.

## Manual smoke test (real server, placeholder keys)

`uvicorn cwas.main:app`: `/health` → `{"status":"ok"}`; empty prompt → 422; unknown run → 404; OpenAPI lists all 5 routes. Without `TAVILY_API_KEY`, startup fails with `ConfigError: TAVILY_API_KEY is required for web research`.

## Known gaps (not verified)

- **No live end-to-end run.** No Anthropic or Tavily keys were available, so the real model and search calls haven't been exercised. Every external call is covered only through mocks and fakes.
- **Docker images not built.** The Docker daemon wasn't running; only `docker compose config` was validated.
- **The CI workflow hasn't run yet.** It runs once the branch is pushed.
