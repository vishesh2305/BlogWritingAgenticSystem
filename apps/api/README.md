# CWAS API: backend for the content-writing multi-agent system

FastAPI plus LangGraph. This is the **Phase 1 vertical slice** from [`Plan.md`](../../Plan.md): a prompt goes through six agents and comes back as a cited, humanized, grammar-checked article, with every step streamed live.

```
prompt ─▶ 1 Prompt Enhancer ─(unclear?)─▶ ask user ─┐
              ▲─────────────────────────────────────┘
          ─▶ 2 Web Research ─▶ 3 Knowledge Base ─▶ 5 Writer ─▶ 7 Humanizer ─▶ 10 Grammar ─▶ article
```

## Run it (free: no API keys)

By default every agent runs on a **local Ollama model** and web research uses **DuckDuckGo**, so you need no paid keys.

```bash
ollama pull qwen3-vl:8b-instruct  # once; the default model (~6 GB, fits an 8 GB GPU)
cd apps/api
python -m pip install uv          # once
uv sync                           # installs Python 3.13 deps into .venv
cp .env.example .env              # defaults are fine
uv run uvicorn cwas.main:app --port 8000
```

The server checks at startup that Ollama is running and the model is pulled, and stops with a clear message if not. To use another local model, set `OLLAMA_MODEL` to any name from `ollama list`.

| Setting | Free default | Paid upgrade |
|---|---|---|
| LLM | `LLM_PROVIDER=ollama` | `LLM_PROVIDER=gemini` + `GEMINI_API_KEY` (+ `GROQ_API_KEY` fallback), or `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` |
| Search | DuckDuckGo (rate-limited) | `TAVILY_API_KEY` |
| Embeddings | local hashing embedder | `GEMINI_API_KEY` (`gemini-embedding-2`) or `VOYAGE_API_KEY` |

A local 8B model is slower and writes less polished prose than Claude. Expect a few minutes per article on an RTX 4060.

### Fast hosted models: Gemini with Groq fallback

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
GROQ_API_KEY=...     # optional fallback: https://console.groq.com/keys
```

Every call goes to Gemini first (`gemini-3.6-flash`, and `gemini-3.5-flash-lite` for the light tier; `gemini-3.8-flash` measured ~117 s per call against ~2 s, so it is not the default). If Gemini rate-limits, errors, times out, blocks the request or returns bad JSON, the same call is retried on Groq (`openai/gpt-oss-120b` / `gpt-oss-20b`). A rate-limited provider sits out its `Retry-After` window, then rejoins. Without `GROQ_API_KEY`, Gemini runs alone and the server logs a warning. Override models with `GEMINI_MODEL_OPUS|SONNET|HAIKU` and `GROQ_MODEL_OPUS|SONNET|HAIKU`.

Both providers are rate-limited on the client side (`GEMINI_REQUESTS_PER_MINUTE`, default 10; `GROQ_REQUESTS_PER_MINUTE`, default 30; at most `LLM_MAX_CONCURRENCY` calls in flight). Set these to your API tier. A call that would wait more than `LLM_MAX_WAIT` seconds (default 5) for a slot goes to the other provider instead. If both are rate-limited, the call waits once for the sooner one, but never more than 60 seconds, so an exhausted daily quota fails fast. With `GEMINI_API_KEY` set, the knowledge base also uses Gemini embeddings instead of the offline hashing embedder.

### AI-likeness score and the Humanizer

The Humanizer (Agent 7) scores the draft from 0 to 100 for machine-typical style: stock phrases ("delve", "in today's fast-paced world"…), even sentence rhythm, formal transition openers, em dashes, bullet-heavy layout and missing contractions. It rewrites only the flagged paragraphs, rejects any rewrite that changes a citation, adds a number or drops the primary keyword, and repeats until the score is below `AI_SCORE_THRESHOLD` (default 10) or `MAX_HUMANIZE_ROUNDS` (default 2) is used up. If the target is still missed, the article is delivered with a warning. The final score is in `GET /runs/{id}` as `ai_score`.

This score is a local stand-in that tracks what detectors look for. It is **not** GPTZero, Originality.ai or Turnitin, and those tools disagree with each other, so a low score here doesn't guarantee a low score there.

To turn on grammar auto-correction, start LanguageTool:

```bash
docker compose -f infra/docker-compose.yml up -d languagetool   # from the repo root
```

You can also run the whole stack: `docker compose -f infra/docker-compose.yml up --build`.

## Use it

```bash
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
     -d '{"prompt": "Write a beginner guide to home solar panels in the UK"}'
# → {"run_id": "…"}

curl -N localhost:8000/runs/<run_id>/stream     # live agent timeline (SSE)
curl localhost:8000/runs/<run_id>               # status, brief, article, warnings
curl -X POST localhost:8000/runs/<run_id>/answer -H 'content-type: application/json' \
     -d '{"answer": "First-time buyers"}'      # only when status = waiting_for_input
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| POST | `/runs` | Start a run → `202 {run_id}` |
| GET | `/runs/{id}` | `status` (`running` / `waiting_for_input` / `completed` / `failed`), `question`, `brief`, `article`, `ai_score`, `usage` (LLM calls and tokens), `warnings`, `error` |
| GET | `/runs/{id}/stream` | SSE events: `run_started`, `agent_started`, `agent_finished`, `agent_failed`, `question`, `answer_received`, `done`, `error`. A late subscriber gets every earlier event replayed. |
| POST | `/runs/{id}/answer` | Answer the clarifying question → `202`, or `409` if the run isn't waiting |

## Develop

```bash
uv run pytest --cov          # 263 tests, coverage gate 80%
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src              # strict
```

Tests never touch the network. HTTP is mocked with `respx`, and the agents run against the fakes in `tests/fakes.py`.

## Layout

```
src/cwas/
  agents/      prompt_enhancer, researcher, kb_builder, writer, humanizer, grammar, prompts, deps
  graph/       state (RunState), builder (orchestrator), events, finalize
  tools/       search (Tavily / DuckDuckGo), scrape (robots + SSRF guard), embed (Voyage/Gemini/hashing), grammar, ai_score, guards (rewrite safety), url_safety
  kb/          chunker, store (vector store), knowledge_base, facts (numeric source sentences)
  api/app.py   FastAPI routes      runs.py  run lifecycle + event fan-out
  llm.py       LLM interface + Anthropic backend      ollama_llm.py  local Ollama backend
  gemini_llm.py, groq_llm.py  hosted backends      fallback_llm.py  provider chain (Gemini → Groq)
  throttled_llm.py  client-side rate limits      usage.py  per-run, per-agent token meter
  wiring.py    production dependency wiring      config.py  settings
```

## Limits of this slice (planned for later phases)

| Here now | Planned replacement |
|---|---|
| In-memory vector store, page cache, checkpointer and run registry: runs are lost on restart | Postgres + pgvector, Redis, Postgres checkpointer |
| Runs execute as asyncio tasks inside the API process | ARQ worker queue |
| httpx + trafilatura scraping (no JavaScript rendering) | Crawl4AI fallback for JavaScript-heavy pages |
| Competitors = Tavily results for the primary keyword | DataForSEO SERP top-N (Phase 4) |
| One draft variant; `POST /runs` takes JSON text only | N variants + Benchmark Judge (Phase 5); file uploads (Phase 6) |
| No cost or budget tracking | Langfuse tracing + per-run budget (Phase 7) |
