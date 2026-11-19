"""Multi-AI Agent Chat Platform - backend.

Layers (see SRS section 2.1 / component diagram):
  Presentation  -> ../static/index.html
  Application   -> REST routes + WebSocket orchestrator (this file)
  Adapters      -> call_mock / call_openai / call_gemini
  Persistence   -> SQLite (agents, sessions, messages)
"""
import asyncio
import json
import logging
import os
import random
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
log = logging.getLogger("multi-ai")


def _load_env() -> None:
    """Tiny .env reader so no extra dependency is needed."""
    f = BASE / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

DB_PATH = os.getenv("DB_PATH", str(BASE / "chat.db"))
MAX_PROMPT = int(os.getenv("MAX_PROMPT_CHARS", "4000"))
TIMEOUT = float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "30"))
KEY_VARS = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY"}  # credential *references*
# On a public site set REQUIRE_USER_KEYS=true so visitors must bring their own keys
# and nobody can spend the owner's quota.
REQUIRE_USER_KEYS = os.getenv("REQUIRE_USER_KEYS", "").lower() in ("1", "true", "yes")
KEYS_FILE = Path(os.getenv("KEYS_FILE", str(BASE / "local_keys.json")))  # keys typed in the browser


def _load_saved_keys() -> dict:
    try:
        data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if (k in KEY_VARS or k.startswith("agent:")) and isinstance(v, str) and v}
    except (OSError, ValueError):
        return {}


SAVED_KEYS = _load_saved_keys()


def _save_keys() -> None:
    KEYS_FILE.write_text(json.dumps(SAVED_KEYS), encoding="utf-8")


def provider_key(provider: str) -> str | None:
    """Key typed in the browser wins; otherwise fall back to the .env variable."""
    return SAVED_KEYS.get(provider) or os.getenv(KEY_VARS[provider])


def agent_key(agent: dict) -> str | None:
    """An agent's own key first, then the shared key for its provider."""
    if agent.get("_key"):  # key the visitor sent with this prompt; never stored
        return agent["_key"]
    if REQUIRE_USER_KEYS:
        return None
    own = SAVED_KEYS.get(f"agent:{agent['id']}")
    if own:
        return own
    return provider_key(agent["provider"]) if agent["provider"] in KEY_VARS else None


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def q(sql, args=(), one=False, commit=False):
    """Run one SQL statement. Returns rows (list/dict) or lastrowid when commit=True."""
    with closing(sqlite3.connect(DB_PATH)) as c:
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        cur = c.execute(sql, args)
        if commit:
            c.commit()
            return cur.lastrowid
        rows = [dict(r) for r in cur.fetchall()]
    if one:
        return rows[0] if rows else None
    return rows


def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS agents(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, model TEXT NOT NULL, provider TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1, base_url TEXT);
            CREATE TABLE IF NOT EXISTS sessions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                parent_id INTEGER, agent_id INTEGER,
                role TEXT NOT NULL, content TEXT NOT NULL,
                status TEXT NOT NULL, error_kind TEXT, latency_ms INTEGER,
                created_at TEXT NOT NULL);
            """
        )
        if c.execute("SELECT COUNT(*) FROM agents").fetchone()[0] == 0:
            c.executemany(
                "INSERT INTO agents(name, model, provider) VALUES (?,?,?)",
                [
                    ("Mock Alpha", "mock-alpha", "mock"),
                    ("Mock Beta", "mock-beta", "mock"),
                    ("Mock Flaky", "mock-flaky", "mock"),
                    ("GPT", "gpt-4o-mini", "openai"),
                    ("Gemini", "gemini-2.5-flash", "gemini"),
                    ("Claude", "claude-sonnet-5-5", "anthropic"),
                ],
            )
        cols = [r[1] for r in c.execute("PRAGMA table_info(agents)")]
        if "base_url" not in cols:  # database created by an older version
            c.execute("ALTER TABLE agents ADD COLUMN base_url TEXT")
        if not c.execute("SELECT 1 FROM agents WHERE provider='groq'").fetchone():
            c.executemany(
                "INSERT INTO agents(name, model, provider) VALUES (?,?,'groq')",
                [("Groq GPT-OSS 20B", "openai/gpt-oss-20b"), ("Groq GPT-OSS 120B", "openai/gpt-oss-120b"),
                 ("Groq Qwen3", "qwen/qwen3.8-27b")],
            )
        if not c.execute("SELECT 1 FROM agents WHERE provider='anthropic'").fetchone():
            c.execute("INSERT INTO agents(name, model, provider) VALUES ('Claude','claude-sonnet-5-5','anthropic')")
        c.commit()


init_db()


# --------------------------------------------------------------------------
# Provider adapters: common request -> provider API -> common result
# --------------------------------------------------------------------------
class ProviderError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


async def _post(url: str, headers: dict, body: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        raise ProviderError("timeout", "The provider took too long to answer.")
    except httpx.HTTPError:
        raise ProviderError("network", "Could not reach the provider. Check the connection.")
    if r.status_code in (401, 403):
        raise ProviderError("authentication", "The provider rejected the API key.")
    if r.status_code == 429:
        raise ProviderError("rate_limit", "Rate limit or quota reached. Retry later.")
    if r.status_code >= 400:
        detail = ""
        try:  # most providers explain the problem in error.message
            err = r.json().get("error", {})
            detail = err.get("message", "") if isinstance(err, dict) else str(err)
        except (ValueError, AttributeError):
            pass
        detail = " ".join(detail.split())[:200]
        raise ProviderError("provider", f"The provider returned HTTP {r.status_code}." + (f" {detail}" if detail else ""))
    try:
        return r.json()
    except ValueError:
        raise ProviderError("provider", "The provider sent an unreadable response.")


def _key(agent: dict) -> str:
    key = agent_key(agent)
    if not key:
        raise ProviderError("authentication", f"No API key for {agent['name']}. Add one in the sidebar.")
    return key


async def _openai_style(url: str, agent: dict, history: list, key: str | None) -> str:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = await _post(url, headers, {"model": agent["model"], "messages": history})
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("provider", "Unexpected response format from the provider.")


async def call_openai(agent: dict, history: list) -> str:
    return await _openai_style("https://api.openai.com/v1/chat/completions", agent, history, _key(agent))


async def call_groq(agent: dict, history: list) -> str:
    return await _openai_style("https://api.groq.com/openai/v1/chat/completions", agent, history, _key(agent))


async def call_custom(agent: dict, history: list) -> str:
    """Any service that copies OpenAI's chat format (Groq, OpenRouter, DeepSeek, Ollama...)."""
    base = (agent.get("base_url") or "").rstrip("/")
    if not base:
        raise ProviderError("provider", "This agent has no base URL.")
    return await _openai_style(base + "/chat/completions", agent, history, agent_key(agent))


async def call_anthropic(agent: dict, history: list) -> str:
    data = await _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": _key(agent), "anthropic-version": "2023-06-01"},
        {"model": agent["model"], "max_tokens": 1024, "messages": history},
    )
    try:
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
    except (KeyError, TypeError, AttributeError):
        raise ProviderError("provider", "Unexpected response format from the provider.")
    if not text:
        raise ProviderError("provider", "The provider returned an empty answer.")
    return text


async def call_gemini(agent: dict, history: list) -> str:
    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        for m in history
    ]
    data = await _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{agent['model']}:generateContent",
        {"x-goog-api-key": _key(agent)},
        {"contents": contents},
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("provider", "Unexpected response format from the provider.")
    if not text:
        raise ProviderError("provider", "The provider returned an empty answer.")
    return text


async def call_mock(agent: dict, history: list) -> str:
    scale = float(os.getenv("MOCK_DELAY_SCALE", "1"))
    model = agent["model"]
    if model.endswith("flaky"):
        await asyncio.sleep(0.8 * scale)
        raise ProviderError("provider", "Simulated provider outage (mock agent).")
    base = {"mock-alpha": 1.0, "mock-beta": 2.2}.get(model, 1.5)
    await asyncio.sleep(base * scale * random.uniform(0.8, 1.2))
    last = history[-1]["content"]
    short = last if len(last) <= 120 else last[:117] + "..."
    turn = sum(1 for m in history if m["role"] == "user")
    if model.endswith("alpha"):
        return f'Short answer to "{short}": this is a mock reply from {agent["name"]} (turn {turn}).'
    return (
        f'Detailed mock reply from {agent["name"]} (turn {turn}):\n'
        f'- You asked: "{short}"\n'
        "- A real agent would reason about this here.\n"
        "- Replace this adapter with a real provider by choosing OpenAI or Gemini."
    )


ADAPTERS = {"mock": call_mock, "openai": call_openai, "gemini": call_gemini,
            "anthropic": call_anthropic, "groq": call_groq, "custom": call_custom}


def build_history(session_id: int, agent_id: int) -> list:
    """This agent's view of the session: all user prompts + its own good answers."""
    rows = q("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,))
    out = []
    for m in rows:
        if m["role"] == "user" or (m["agent_id"] == agent_id and m["status"] == "complete"):
            role = "user" if m["role"] == "user" else "assistant"
            if out and out[-1]["role"] == role:  # providers dislike repeated roles
                out[-1]["content"] += "\n\n" + m["content"]
            else:
                out.append({"role": role, "content": m["content"]})
    return out


# --------------------------------------------------------------------------
# REST API
# --------------------------------------------------------------------------
app = FastAPI(title="Multi-AI Agent Chat Platform")


class AgentIn(BaseModel):
    name: str
    model: str
    provider: str
    enabled: bool = True
    base_url: str | None = None   # only for provider "custom"
    api_key: str | None = None    # stored on the server, never returned


def agent_view(a: dict) -> dict:
    """Never exposes keys - only whether the server has one (FR-16, BR-07)."""
    if a["provider"] == "mock":
        configured = True
    elif a["provider"] == "custom":
        configured = bool(a.get("base_url"))
    else:
        configured = bool(agent_key(a))
    return {
        "id": a["id"], "name": a["name"], "model": a["model"], "provider": a["provider"],
        "base_url": a.get("base_url"), "enabled": bool(a["enabled"]), "configured": configured,
    }


@app.get("/api/config")
def config():
    return {"require_user_keys": REQUIRE_USER_KEYS}


@app.get("/api/agents")
def list_agents():
    return [agent_view(a) for a in q("SELECT * FROM agents WHERE enabled=1 ORDER BY id")]


@app.post("/api/agents", status_code=201)
def create_agent(body: AgentIn, request: Request):
    _local_only(request)
    name, model, provider = body.name.strip(), body.model.strip(), body.provider.strip().lower()
    if not name or not model:
        raise HTTPException(400, "Agent name and model are required.")
    if provider not in ADAPTERS:
        raise HTTPException(400, f"Provider must be one of: {', '.join(ADAPTERS)}.")
    base_url = None
    if provider == "custom":
        base_url = (body.base_url or "").strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")) or " " in base_url:
            raise HTTPException(400, "Custom agents need a base URL starting with http:// or https://.")
    key = (body.api_key or "").strip()
    if key and (len(key) > 500 or any(c.isspace() for c in key)):
        raise HTTPException(400, "That does not look like an API key.")
    aid = q("INSERT INTO agents(name, model, provider, enabled, base_url) VALUES (?,?,?,?,?)",
            (name, model, provider, int(body.enabled), base_url), commit=True)
    if key:
        SAVED_KEYS[f"agent:{aid}"] = key
        _save_keys()
    return agent_view(q("SELECT * FROM agents WHERE id=?", (aid,), one=True))


@app.delete("/api/agents/{aid}", status_code=204)
def remove_agent(aid: int, request: Request):
    """Hides the agent and deletes its key; old answers stay in session history."""
    _local_only(request)
    if not q("SELECT id FROM agents WHERE id=?", (aid,), one=True):
        raise HTTPException(404, "Agent not found.")
    q("UPDATE agents SET enabled=0 WHERE id=?", (aid,), commit=True)
    if SAVED_KEYS.pop(f"agent:{aid}", None):
        _save_keys()


class KeyIn(BaseModel):
    provider: str
    key: str


def _local_only(request: Request) -> None:
    """Keys can only be changed from the machine running the server."""
    if (request.client.host if request.client else "") not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "API keys can only be changed from the server machine.")


@app.get("/api/keys")
def key_status():
    """Which providers have a key. Never returns the key itself."""
    return {p: bool(provider_key(p)) for p in KEY_VARS}


@app.post("/api/keys")
def save_key(body: KeyIn, request: Request):
    _local_only(request)
    provider, key = body.provider.strip().lower(), body.key.strip()
    if provider not in KEY_VARS:
        raise HTTPException(400, f"Provider must be one of: {', '.join(KEY_VARS)}.")
    if not key or len(key) > 500 or any(c.isspace() for c in key):
        raise HTTPException(400, "That does not look like an API key.")
    SAVED_KEYS[provider] = key
    _save_keys()
    return {"provider": provider, "configured": True}


@app.delete("/api/keys/{provider}", status_code=204)
def remove_key(provider: str, request: Request):
    _local_only(request)
    if provider not in KEY_VARS:
        raise HTTPException(400, "Unknown provider.")
    SAVED_KEYS.pop(provider, None)
    _save_keys()


class SessionIn(BaseModel):
    title: str | None = None


@app.post("/api/sessions", status_code=201)
def create_session(body: SessionIn | None = None):
    title = ((body.title if body else None) or "New session").strip()[:80] or "New session"
    t = now()
    sid = q("INSERT INTO sessions(title, created_at, updated_at) VALUES (?,?,?)", (title, t, t), commit=True)
    return q("SELECT * FROM sessions WHERE id=?", (sid,), one=True)


@app.get("/api/sessions")
def list_sessions():
    return q("SELECT * FROM sessions ORDER BY updated_at DESC, id DESC")


@app.get("/api/sessions/{sid}")
def get_session(sid: int):
    s = q("SELECT * FROM sessions WHERE id=?", (sid,), one=True)
    if not s:
        raise HTTPException(404, "Session not found.")
    s["messages"] = q("SELECT * FROM messages WHERE session_id=? ORDER BY id", (sid,))
    return s


@app.delete("/api/sessions/{sid}", status_code=204)
def delete_session(sid: int):
    if not q("SELECT id FROM sessions WHERE id=?", (sid,), one=True):
        raise HTTPException(404, "Session not found.")
    q("DELETE FROM sessions WHERE id=?", (sid,), commit=True)


# --------------------------------------------------------------------------
# WebSocket: validate -> store prompt once -> parallel dispatch -> live events
# --------------------------------------------------------------------------
def check_prompt(prompt) -> tuple[str | None, str]:
    text = prompt.strip() if isinstance(prompt, str) else ""
    if not text:
        return "Prompt must not be empty.", ""
    if len(text) > MAX_PROMPT:
        return f"Prompt is too long (limit {MAX_PROMPT} characters).", ""
    return None, text


def check_agents(agent_ids) -> tuple[str | None, list]:
    if not isinstance(agent_ids, list) or not agent_ids:
        return "Select at least one agent.", []
    agents = []
    for aid in dict.fromkeys(agent_ids):
        a = q("SELECT * FROM agents WHERE id=? AND enabled=1", (aid,), one=True) if isinstance(aid, int) else None
        if not a:
            return f"Agent {aid} is not available.", []
        agents.append(a)
    return None, agents


def clean_keys(raw) -> dict:
    """Keys sent by the browser with a prompt: used for this request only, never stored or logged."""
    out = {}
    if isinstance(raw, dict):
        for provider, key in raw.items():
            if provider in KEY_VARS and isinstance(key, str):
                key = key.strip()
                if key and len(key) <= 500 and not any(c.isspace() for c in key):
                    out[provider] = key
    return out


async def run_agent(send, sid: int, parent: int, agent: dict, keys: dict) -> None:
    await send({"type": "status", "parent_id": parent, "agent_id": agent["id"], "status": "processing"})
    start = time.perf_counter()
    status, kind = "complete", None
    try:
        content = await ADAPTERS[agent["provider"]](
            {**agent, "_key": keys.get(agent["provider"])}, build_history(sid, agent["id"]))
    except ProviderError as e:
        status, kind, content = "error", e.kind, e.message
    except Exception:  # never leak stack traces to the browser (section 7)
        log.exception("Unexpected failure in agent %s", agent["id"])
        status, kind, content = "error", "unexpected", "Something went wrong. Please retry."
    latency = int((time.perf_counter() - start) * 1000)
    try:
        mid = q(
            "INSERT INTO messages(session_id,parent_id,agent_id,role,content,status,error_kind,latency_ms,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, parent, agent["id"], "assistant", content, status, kind, latency, now()), commit=True)
    except sqlite3.Error:
        log.exception("Could not store result")
        mid, status, kind, content = None, "error", "storage", "The answer could not be saved (session may have been deleted)."
    await send({"type": "result", "parent_id": parent, "agent_id": agent["id"], "message_id": mid,
                "status": status, "error_kind": kind, "content": content, "latency_ms": latency})


async def handle_prompt(send, sid: int, data: dict) -> None:
    err, agents = check_agents(data.get("agent_ids"))
    if err:
        return await send({"type": "error", "message": err})

    if data.get("retry"):  # re-ask only the chosen agents about the latest prompt
        last = q("SELECT id FROM messages WHERE session_id=? AND role='user' ORDER BY id DESC LIMIT 1", (sid,), one=True)
        if not last:
            return await send({"type": "error", "message": "Nothing to retry yet."})
        parent = last["id"]
        for a in agents:
            q("DELETE FROM messages WHERE parent_id=? AND agent_id=? AND status!='complete'",
              (parent, a["id"]), commit=True)
    else:
        err, text = check_prompt(data.get("prompt"))
        if err:
            return await send({"type": "error", "message": err})
        t = now()
        try:
            parent = q("INSERT INTO messages(session_id,role,content,status,created_at) VALUES (?,?,?,?,?)",
                       (sid, "user", text, "complete", t), commit=True)
        except sqlite3.Error:
            return await send({"type": "error", "message": "This session no longer exists."})
        title = text[:40] + ("..." if len(text) > 40 else "")
        q("UPDATE sessions SET updated_at=?, title=CASE WHEN title='New session' THEN ? ELSE title END WHERE id=?",
          (t, title, sid), commit=True)
        session = q("SELECT title FROM sessions WHERE id=?", (sid,), one=True)
        await send({"type": "prompt_saved", "title": session["title"],
                    "message": {"id": parent, "role": "user", "content": text, "created_at": t}})

    keys = clean_keys(data.get("keys"))
    await asyncio.gather(*(run_agent(send, sid, parent, a, keys) for a in agents))


@app.websocket("/ws/sessions/{sid}")
async def session_socket(ws: WebSocket, sid: int):
    if not q("SELECT id FROM sessions WHERE id=?", (sid,), one=True):
        await ws.accept()  # accept first so the browser sees close code 4404, not a bare 403
        await ws.close(code=4404)
        return
    await ws.accept()
    lock = asyncio.Lock()

    async def send(obj: dict) -> None:
        try:
            async with lock:
                await ws.send_json(obj)
        except Exception:  # client left; results are still stored for history
            pass

    tasks: set = set()
    try:
        while True:
            data = await ws.receive_json()
            if not isinstance(data, dict) or data.get("type") != "prompt":
                await send({"type": "error", "message": "Unsupported message."})
                continue
            task = asyncio.create_task(handle_prompt(send, sid, data))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
