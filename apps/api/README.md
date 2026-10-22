# CWAS API: backend for the content-writing multi-agent system

FastAPI plus LangGraph. This is the **Phase 1 vertical slice** from [`Plan.md`](../../Plan.md): a prompt goes through five agents and comes back as a cited, grammar-checked article, with every step streamed live.

```
prompt ─▶ 1 Prompt Enhancer ─(unclear?)─▶ ask user ─┐
              ▲─────────────────────────────────────┘
          ─▶ 2 Web Research ─▶ 3 Knowledge Base ─▶ 5 Writer ─▶ 10 Grammar ─▶ article
```

## Run it

```bash
cd apps/api
python -m pip install uv          # once
uv sync                           # installs Python 3.13 deps into .venv
cp .env.example .env              # add ANTHROPIC_API_KEY and TAVILY_API_KEY
uv run uvicorn cwas.main:app --reload --port 8000
```

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
| GET | `/runs/{id}` | `status` (`running` / `waiting_for_input` / `completed` / `failed`), `question`, `brief`, `article`, `warnings`, `error` |
| GET | `/runs/{id}/stream` | SSE events: `run_started`, `agent_started`, `agent_finished`, `agent_failed`, `question`, `answer_received`, `done`, `error`. A late subscriber gets every earlier event replayed. |
| POST | `/runs/{id}/answer` | Answer the clarifying question → `202`, or `409` if the run isn't waiting |

## Develop

```bash
uv run pytest --cov          # 120 tests, coverage gate 80%
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src              # strict
```

Tests never touch the network. HTTP is mocked with `respx`, and the agents run against the fakes in `tests/fakes.py`.

## Layout

```
src/cwas/
  agents/      prompt_enhancer, researcher, kb_builder, writer, grammar, prompts, deps
  graph/       state (RunState), builder (orchestrator), events, finalize
  tools/       search (Tavily), scrape (robots + SSRF guard), embed (Voyage/hashing), grammar, url_safety
  kb/          chunker, store (vector store), knowledge_base
  api/app.py   FastAPI routes      runs.py  run lifecycle + event fan-out
  llm.py       Anthropic wrapper (model tiers, effort, refusal fallbacks)
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
