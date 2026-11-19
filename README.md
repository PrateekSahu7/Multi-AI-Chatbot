# Multi-AI Agent Chat Platform

Send one prompt to several AI agents and compare their answers side by side.
Built from the SE lab SRS: FastAPI + SQLite + WebSocket, with mock agents (no key needed) and OpenAI / Gemini adapters.

## Run (Windows PowerShell; on Mac/Linux use `source .venv/bin/activate`)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # optional: add OPENAI_API_KEY / GEMINI_API_KEY
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000. Add real API keys under **API keys** in the sidebar (or use `.env`). Mock Alpha and Mock Beta are selected by default; Mock Flaky always fails so you can demo failure isolation and Retry.

## Test

```powershell
pytest
```

## Where the SRS shows up

| SRS item | Where |
|---|---|
| FR-02..FR-10 single prompt, parallel dispatch, per-agent status, failure isolation | `handle_prompt` / `run_agent` in `backend/main.py` |
| FR-11..FR-13, FR-19 sessions, history, follow-ups, delete | `/api/sessions`, `build_history` |
| FR-14, FR-15, FR-20 agent config, provider adapters | `/api/agents`, `ADAPTERS` |
| FR-16, BR-07 keys stay server-side | `.env` + `KEY_VARS`; `agent_view` only exposes `configured` |
| Section 7 error handling and retry | `ProviderError` kinds, retry button |
| AC-01..AC-10 | `tests/test_api.py` |

Add a real agent with the "Add agent" form (for example provider `openai`, model `gpt-4o-mini`) or edit the seed list in `init_db`.

## Prototype limits

No login (SRS 8.1 allows this for the lab demo), no token streaming, plain-text answers. See SRS 10.3 for future scope.

## Adding any AI agent

Use **Add agent** in the sidebar. Pick OpenAI, Claude, Gemini or **Other (OpenAI-compatible)**, then give a name, the model id and the API key.
For "Other", also give the base URL. Examples: `https://api.groq.com/openai/v1`, `https://openrouter.ai/api/v1`, `https://api.deepseek.com/v1`, `http://localhost:11434/v1` (Ollama, no key needed).
Each agent keeps its own key on the server (`local_keys.json`); keys are never sent back to the browser.

## Deploying on Render (free)

Build: `pip install -r requirements.txt` | Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
Set `GROQ_API_KEY`, `GEMINI_API_KEY` (and optionally `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) under Environment. Free instances lose chat history when they spin down.

## Visitors enter their own API key

The **Your API keys** box saves keys in the visitor's browser (localStorage) and sends them with each prompt; the server never stores them.
On a public deployment set `REQUIRE_USER_KEYS=true` so server-side keys are never used for visitors.
