# TDD Evidence: Backend Vertical Slice (Plan Phases 0–1)

**Source plan:** [`Plan.md`](../../Plan.md), Phase 0 (steps 1–5) and Phase 1 (steps 6–10); cycles E–G follow the `/plan` for Gemini + Groq fallback and an AI-likeness score below 10; cycles H–M are Phase A (groundwork) of the text-only 12-agent plan in `Plan.md`
**Branch:** `feat/backend-vertical-slice`
**Runner:** `uv` + `pytest` (Python 3.13.17), from `apps/api`. Cycles E–F ran `.venv/Scripts/python.exe -m pytest` directly because `uv` was not on PATH.

## User journeys (from the plan's requirements and acceptance criteria)

1. As a writer, I submit a topic prompt and get back one article with cited sources.
2. As a writer, if my prompt is ambiguous, I'm asked one clarifying question, and the run continues after I answer.
3. As a writer, I can watch each agent start and finish live while the run is in progress.
4. As an operator, a broken or hostile web page, or a failing third-party service, does not crash the run or reach internal network addresses.
5. As an operator, I set `GEMINI_API_KEY` for fast results, and when Gemini fails or rate-limits, the same call finishes on Groq without the run failing.
6. As a writer, the article I get back reads as human-written: its AI-likeness score is below 10, or I'm told it isn't, and its facts and citations are unchanged.

## RED → GREEN checkpoints

| Cycle | Scope | RED commit (evidence) | GREEN commit (evidence) |
|---|---|---|---|
| A | Research tools + knowledge base | `8ad805e`: `pytest` → 8 collection errors, `ModuleNotFoundError: cwas.tools` / `cwas.kb` | `a5790e2`: 68 passed, coverage 97.48%, ruff + mypy strict clean |
| B | LLM wrapper, 5 agents, LangGraph orchestrator | `a65fb32`: 7 collection errors, `ModuleNotFoundError: cwas.llm` | `c92a00b`: 103 passed, coverage 98.5%, ruff + mypy strict clean |
| C | Run manager, HTTP + SSE API, wiring | `adeda6e`: 2 collection errors, `ModuleNotFoundError: cwas.runs` | `498fe5e`: 120 passed, coverage 98.03%, ruff + mypy strict clean |

| D | Free local LLM (Ollama) + DuckDuckGo search | `c5d9881`: 2 collection errors (`OllamaLLM`, `DuckDuckGoSearch` missing); `tests/test_wiring.py` → 6 failed, 1 passed | `a833974`: 137 passed, coverage 98%, ruff + mypy strict clean |
| E | Gemini primary + Groq fallback LLM chain | `dfb2d8a`: 4 collection errors (`ModuleNotFoundError`: `cwas.gemini_llm`, `cwas.groq_llm`, `cwas.fallback_llm`) | `aecea25`: 187 passed, coverage 98.17%, ruff + `mypy src` clean |
| F | AI-likeness score + Humanizer agent | `765fdba`: 3 collection errors (`cwas.tools.ai_score`, `cwas.agents.humanizer`, `AIScore`); `tests/graph tests/api` → 2 failed (no `humanizer` in agent order; `KeyError: 'ai_score'`) | `cf35c2f`: 213 passed, coverage 98.08%, ruff + `mypy src` clean |
| G | Live-run regression: em dashes kept the score above 10 | `77f6aac`: collection error (`ImportError: replace_em_dashes`) | `edc0116`: 218 passed, coverage 98.10%, ruff + `mypy src` clean |
| H | Scrape keeps headings; competitor meta description + JSON-LD types; numeric fact extraction | `87febdf`: collection error (`cwas.kb.facts`); 4 failed (headings flattened, no `description`, `KeyError: 'facts'`) | `69069b1`: 226 passed, coverage 98.20% |
| I | Gemini embeddings + embedder selection | `d04fb60`: 2 collection errors (`ImportError: GeminiEmbedder`) | `a8892f3`: 235 passed, coverage 98.36% |
| J | Per-run, per-agent token usage | `78e98c0`: 14 collection errors (`cwas.usage`, `Usage`) | `a200915`: 244 passed, coverage 98.43% |
| K | Client-side rate limits + shared rewrite guards | `030b2de`: 3 collection errors; 13 failed (`FallbackLLM` has no `sleep`; humanizer accepts added numbers and dropped keyword) | `91050b9`: 260 passed, coverage 98.45% |
| L | Never wait out a long rate limit | `6bc13f8`: 1 failed (slept 120 s) | `be1ca8d`: 261 passed, coverage 98.46% |
| M | Live-run findings: weak facts, silent missing competitors | `a2c3f49`: 2 failed | `72e1c6f`: 263 passed, coverage 98.46% |
| N | Default main Gemini model → `gemini-3.6-flash` (3.8 Flash measured ~117 s/call) | `d17f69d`: 1 failed | `19f73b2`: 263 passed, 98.46% |
| O | Gemini embedding free-tier 429 (100 texts/min, delay only in `RetryInfo`) | `ea886b2`: 6 failed | `c49bb4c`: 270 passed, 98.26% |
| P | 503 overload as rate limit; evenly paced 3-sentence paragraphs targeted | `90db7dd`: 2 failed | `3968844`: 272 passed, 98.06% |
| Q | gpt-oss `【n†source】` citations from the Groq fallback | `e3a0a7e`: 3 failed | `6ef7770`: 275 passed, 98.06% |
| R | Headings glued to paragraphs; article-wide rhythm targets; Groq TPM 413 as rate limit | `2b3b689`: 3 failed | `92e3205`: 278 passed, 98.04% |
| S | Contractions added without an LLM call (unless formal) | `41a53a6`: collection error | `aca49ce`: 287 passed, 98.26% |
| T | Outline instead of the full article when most paragraphs are rewritten | `a95acb9`: 1 failed | `a71df3f`: 288 passed, 98.26% |
| U | Gemini Flash-Lite as the first fallback before Groq | `5c52761`: 3 failed | `c1ccead`: 291 passed, 98.27% |

Every RED in cycles A–C was a compile-time (import) failure caused by the missing implementation, not by broken test setup. In cycle D the wiring tests failed at runtime for the intended reason: Ollama settings and wiring didn't exist yet.

Process notes for cycles H–M: the GREEN commit for H first said 227 passed because the doubled `-q` hid pytest's summary line; it was amended to the real 226 before anything else was committed, and later counts come from `pytest -o addopts=""`. Cycle L closed a gap found while writing K (it was not in the plan). All ruff and `mypy src` checks were clean at every GREEN.

Two process notes for cycles E–F. The RED commit `dfb2d8a` saved `tests/test_wiring.py` with CRLF line endings; `a74d594` restored LF, with no content change. The RED commit `765fdba` missed one existing assertion of the agent order (`test_api.py::test_stream_replays_and_ends_with_done`); it failed under GREEN as expected and was updated in `cf35c2f`.

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
| 14 | Gemini: JSON-schema output with one validation retry; the key goes in the `x-goog-api-key` header, never the URL or errors; thought parts dropped; `SAFETY`/`RECITATION`/blocked prompt → refusal; `MAX_TOKENS` → truncation error; 429 carries `Retry-After` | `tests/test_gemini_llm.py` | unit (HTTP mocked) | PASS |
| 15 | Groq: `json_schema` output; server-side `json_validate_failed` is retried like a validation error; `reasoning_effort` only for gpt-oss models; 413/429/timeout → typed `LLMError` without leaking the key | `tests/test_groq_llm.py` | unit (HTTP mocked) | PASS |
| 16 | Fallback: Groq is untouched while Gemini answers; any `LLMError` (including refusals and 429) hands the call to Groq with a warning; a rate-limited provider sits out `Retry-After` (default 30 s) but is still tried last; non-LLM bugs are not swallowed; if all fail, the error names each provider | `tests/test_fallback_llm.py` | unit | PASS |
| 17 | `LLM_PROVIDER=gemini` requires `GEMINI_API_KEY`, builds Gemini→Groq with `GROQ_API_KEY` and Gemini alone (with a warning) without it, never logs keys, and starts with no network calls | `tests/test_wiring.py` | integration | PASS |
| 18 | The AI-likeness score is deterministic: the machine-typical sample scores 80, the human sample 0; headings, code, links and citations don't change it; each signal (stock phrases, rhythm, transitions, em dashes, lists, contractions) contributes; the score is capped at 100 | `tests/tools/test_ai_score.py` | unit | PASS |
| 19 | Humanizer: no LLM call when already below 10; rewrites only flagged prose paragraphs (headings and other blocks unchanged); rejects rewrites that drop or invent citations; discards rounds that don't lower the score; stops after `MAX_HUMANIZE_ROUNDS` with a warning; an LLM failure keeps the draft and warns | `tests/agents/test_humanizer.py` | unit | PASS |
| 20 | Pipeline runs writer → humanizer → grammar; grammar re-scores the final text; `GET /runs/{id}` returns `ai_score` | `tests/graph/test_graph.py`, `tests/agents/test_grammar_agent.py`, `tests/api/test_api.py` | integration | PASS |
| 21 | Em dashes in prose are replaced in code before any LLM round and after each rewrite (ranges → en dash, clause dashes → commas); headings keep theirs | `tests/agents/test_humanizer.py` | unit | PASS |
| 22 | Scraped pages are Markdown with headings, lists and tables, and carry the meta description and every JSON-LD `@type` (including `@graph` and list types; invalid blocks ignored) | `tests/tools/test_scrape.py` | unit (HTTP mocked) | PASS |
| 23 | Facts are verbatim source sentences with a non-year number, 6–60 words, not list introductions, deduped across pages, capped per page, with normalised numbers and ids | `tests/kb/test_facts.py`, `tests/agents/test_kb_builder.py` | unit | PASS |
| 24 | Gemini embeddings: task prefixes, 768 dims, key in header, batches of 100, typed errors without the key; selection is Voyage → Gemini → hashing | `tests/tools/test_embed.py`, `tests/test_wiring.py` | unit + integration | PASS |
| 25 | Every backend records tokens per call; usage is attributed to the running agent, follows child tasks, appears in `agent_finished` events and in `GET /runs/{id}` | `tests/test_usage.py`, backend tests, `tests/graph/test_graph.py`, `tests/api/test_api.py` | unit + integration | PASS |
| 26 | Rate limits: calls under the per-minute limit pass, short waits sleep, long waits raise a rate-limit error for the fallback, at most N calls in flight; when all providers are rate-limited the chain waits once (≤ 60 s) for the soonest | `tests/test_throttled_llm.py`, `tests/test_fallback_llm.py` | unit | PASS |
| 27 | Rewrite guards: citations unchanged, no numbers outside the original or allowed facts, keyword count not reduced; the Humanizer enforces all three and explains rejections | `tests/tools/test_guards.py`, `tests/agents/test_humanizer.py` | unit | PASS |
| 28 | Research warns when no competitor page could be collected | `tests/agents/test_researcher.py` | unit | PASS |

## Coverage

`uv run pytest --cov` → **120 passed, total coverage 98.03%** (the gate is 80%). The only file below 85% is `main.py` (2 lines, the ASGI entrypoint), and it was exercised by the manual smoke test below.

After cycle G: `pytest --cov` → **218 passed, total coverage 98.10%**. New modules: `gemini_llm.py` 100%, `groq_llm.py` 99%, `fallback_llm.py` 100%, `humanizer.py` 100%, `ai_score.py` 96%.

After Phase A (cycle M): `pytest --cov` → **263 passed, total coverage 98.46%**.

After cycle U: `pytest --cov` → **291 passed, total coverage 98.27%**.

## Manual smoke test (real server, placeholder keys)

`uvicorn cwas.main:app`: `/health` → `{"status":"ok"}`; empty prompt → 422; unknown run → 404; OpenAPI lists all 5 routes. Without `TAVILY_API_KEY`, startup fails with `ConfigError: TAVILY_API_KEY is required for web research`.

## Live end-to-end runs (local Ollama, no API keys)

Setup: `qwen3-vl:8b-instruct` on an RTX 4060 (8 GB), DuckDuckGo search, hashing embedder, no LanguageTool. Prompt: *"Write a beginner-friendly blog post about how home solar panels work and whether they are worth it in India."*

| Run | Result | Time |
|---|---|---|
| 1 (`a833974`) | Completed, but **0 citations**: the model wrote `[n1]` (the prompt said "[n]"), which the parser dropped. The brief also had **0 search queries** (the field was optional). | 2m52s (writer 127s) |
| 2 (`9d1022b`, after the RED `6f545df` → GREEN fix) | Completed: 6 search queries, 1,934 words, citations `[1]`–`[3]` with a Sources section. One DuckDuckGo query returned no results and was recorded as a warning. | 3m34s (writer 170s) |

Both defects were turned into regression tests before they were fixed. Remaining quality notes: the model cited only 3 of the pages it fetched, and some sentences open with "[1] says…". The Fact-Checker and Rephraser agents (Phase 3) are planned to handle this.

### Cycles E–G: AI-likeness gate (local Ollama, no Gemini/Groq keys on this machine)

Same prompt and setup as above, run through `RunManager` with `.env` defaults.

| Run | Result | Time |
|---|---|---|
| 3 (`cf35c2f`) | Completed, 1,602 words, but **AI-likeness 15.3, above the target**. Almost all of it was `em_dashes` 10.0: 34 em dashes, mostly one per paragraph, so only 8 paragraphs were flagged, and the 8B model kept its dashes through 2 rewrite rounds. A warning reported the miss. Replaying the later fix offline on this article: 15.3 → 5.3. | 4m52s (humanizer 96s) |
| 4 (`edc0116`, after the RED `77f6aac` → GREEN fix) | Completed, 1,332 words, **AI-likeness 5.8** (only `lists` 5.8). Replacing dashes in code was enough, so no LLM rewrite was needed. Citations [1][2] match the 2 listed sources; the 2 em dashes left are in headings. 4 of 6 DuckDuckGo queries timed out and became warnings. | 2m27s (humanizer 14 ms) |

### Phase A live check (local Ollama, DuckDuckGo, hashing embedder; `69069b1`…`be1ca8d`)

| Check | Result |
|---|---|
| Full run | Completed in 2m54s, AI-likeness **4.1**, usage 2 calls / 2,943 input / 2,365 output tokens, per-agent tokens in the timeline (prompt enhancer 480, writer 4,828) |
| Competitors | **0 collected in that run.** DuckDuckGo returned no results for 3 of the queries, and nothing warned about it. Fixed in cycle M (a warning now appears). |
| Scraper on real competitor pages (direct check, keyword "home solar panels India") | 3 of 5 results collected with headings (12–15 per long page), meta descriptions and JSON-LD types such as `Article`, `BreadcrumbList`, `Organization`; 2 skipped because their robots.txt disallows crawling |
| Facts | 15 extracted from the one source page. Two were weak (a year-only sentence and a list introduction ending in a colon); fixed in cycle M |

Found for Phase B: competitor headings keep bold markers (`## **Which Are…**`), so the Strategist must strip inline formatting when reading outlines.

### Cycles N–U: first live testing with Gemini and Groq keys (2026-10-24)

Provider checks (each API alone): Gemini text and JSON on both tiers, Groq text and JSON on both tiers, and Gemini embeddings (768 dims, normalised, the relevant document ranks first) all passed.

| Run | Chain actually used | Result | AI-likeness |
|---|---|---|---|
| 1 | Gemini 503 → Groq for the brief | **Failed** in kb_builder: Gemini embeddings 429 (free tier 100 texts/min) → cycle O | – |
| 2 | Gemini, Groq for one humanizer round | Completed in 75 s, citations [1]–[5]; humanizer round 2 lost to Gemini 503 + Groq 429 → cycle P | 13.0 |
| 3 | Gemini cooling → **Groq writer** | Completed in 68 s, but **no citations**: gpt-oss writes `【2†source】` → cycle Q | 7.8 |
| 4 | Writer forced onto Groq (invalid Gemini model) | Failed: Groq 413 on the writer (8,000 tokens/min) | – |
| 5 | Gemini | Completed in 89 s, citations [1]–[3]; text under glued headings was never scored or rewritten → cycle R | 23.4 |
| 6 / 7 | Groq writer (forced) / Gemini | Both completed, citations [1]–[4] (gpt-oss fix works); humanizer starved by Groq 413 → cycles S, T | 13.1 / 13.0 |
| 8 / 9 | Gemini / forced Groq | 8 completed (citations [1]–[3]); 9 failed (Groq 413 on the writer). Root cause found: **`gemini-3.6-flash` free tier allows 20 requests/day**, used up → cycle U | 21.0 / – |
| 10 | Flash quota spent → **Flash-Lite** writer and humanizer | Completed in 133 s, 674 words, citations [1]–[6], embeddings survived their 429 by waiting | 17.8 |

What the live runs confirm: the fallback chain, embedding retry, gpt-oss citation handling, heading-preserving research (2–3 competitors per run with outlines and JSON-LD types), fact extraction (50–120 facts) and token accounting all work against the real APIs. What they don't confirm: **the AI-likeness target.** Only 1 of 7 completed hosted runs finished below 10 (7.8, and that run had lost its citations). The others ended at 13–23, mostly on `uniform_rhythm` and `no_contractions`, because the humanizer's rewrite call was starved by quotas or made by Flash-Lite, which ignores the contraction and rhythm instructions.

## Known gaps (not verified)

- **AI-likeness below 10 is not reliable on free-tier quotas.** See the table above. Options: a paid Gemini tier (no 20 requests/day cap), more rewrite rounds, or calibrating the local score against an external detector before tuning further.
- **Free-tier quotas measured on this account:** `gemini-3.6-flash` 20 requests/day; `gemini-embedding-2` 100 texts/min; Groq `gpt-oss-120b` and `gpt-oss-20b` 8,000 tokens/min and 1,000 requests/day (Groq's Llama models return 404 for this account). A writer call (~10,000 requested tokens) can never fit Groq's per-minute limit, so Groq can only take smaller calls.

- **The client-side rate limiter's limits are guesses** (10 Gemini / 30 Groq requests per minute); the real limits above are per day or per token, so they should be tuned to the account.
- **The AI-likeness score is a local proxy, not GPTZero, Originality.ai or Turnitin.** Its weights haven't been checked against any external detector, so "below 10" here doesn't guarantee a low score there. Plugging in an external detector is still open.
- **Groq free-tier token limits** may reject the writer's prompt (HTTP 413). Those failures surface as `LLMError`; the prompt isn't trimmed and retried.

- **No live run on the Anthropic or Tavily backends.** No keys were available; those code paths are covered only by mocks.
- **Docker images not built.** The Docker daemon wasn't running; only `docker compose config` was validated.
- **The CI workflow hasn't run yet.** It runs once the branch is pushed.
