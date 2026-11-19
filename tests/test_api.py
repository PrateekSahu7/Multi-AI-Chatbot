"""Maps to SRS acceptance criteria AC-01 .. AC-10 (mock agents, no network needed)."""
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def agent_ids(*models):
    agents = {a["model"]: a["id"] for a in client.get("/api/agents").json()}
    return [agents[m] for m in models]


def new_session():
    return client.post("/api/sessions").json()["id"]


def ask(ws, prompt, ids, expected_results):
    ws.send_json({"type": "prompt", "prompt": prompt, "agent_ids": ids})
    events, results = [], []
    while len(results) < expected_results:
        ev = ws.receive_json()
        events.append(ev)
        if ev["type"] == "result":
            results.append(ev)
    return events, results


def test_three_agents_get_one_prompt():  # AC-01, AC-02, AC-03, AC-05
    sid = new_session()
    ids = agent_ids("mock-alpha", "mock-beta", "mock-flaky")
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        events, results = ask(ws, "Explain cloud computing", ids, 3)
    assert sum(e["type"] == "prompt_saved" for e in events) == 1  # BR-04
    assert {r["agent_id"] for r in results} == set(ids)
    assert {e["agent_id"] for e in events if e["type"] == "status"} == set(ids)


def test_failure_is_isolated():  # AC-04
    sid = new_session()
    ids = agent_ids("mock-alpha", "mock-flaky")
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        _, results = ask(ws, "hello", ids, 2)
    by_id = {r["agent_id"]: r for r in results}
    assert by_id[ids[0]]["status"] == "complete"
    assert by_id[ids[1]]["status"] == "error"
    assert len(client.get(f"/api/sessions/{sid}").json()["messages"]) == 3


def test_validation():  # AC-09
    sid = new_session()
    ids = agent_ids("mock-alpha")
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        for payload in (
            {"prompt": "   ", "agent_ids": ids},
            {"prompt": "hi", "agent_ids": []},
            {"prompt": "hi", "agent_ids": [9999]},
        ):
            ws.send_json({"type": "prompt", **payload})
            assert ws.receive_json()["type"] == "error"
    assert client.get(f"/api/sessions/{sid}").json()["messages"] == []


def test_retry_only_failed_agent():  # UC-09
    sid = new_session()
    ids = agent_ids("mock-alpha", "mock-flaky")
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ask(ws, "hello", ids, 2)
        ws.send_json({"type": "prompt", "retry": True, "agent_ids": [ids[1]]})
        events = [ws.receive_json(), ws.receive_json()]
    assert events[0]["type"] == "status" and events[0]["agent_id"] == ids[1]
    assert events[1]["type"] == "result"
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert len([m for m in msgs if m["role"] == "user"]) == 1


def test_history_and_delete():  # AC-06, FR-19
    sid = new_session()
    ids = agent_ids("mock-alpha")
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ask(ws, "first", ids, 1)
        ask(ws, "second", ids, 1)
    session = client.get(f"/api/sessions/{sid}").json()
    assert session["title"] == "first"
    assert len(session["messages"]) == 4
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_add_agent_becomes_selectable():  # AC-07
    r = client.post("/api/agents", json={"name": "Extra", "model": "mock-extra", "provider": "mock"})
    assert r.status_code == 201
    assert any(a["name"] == "Extra" for a in client.get("/api/agents").json())
    bad = client.post("/api/agents", json={"name": "X", "model": "m", "provider": "nope"})
    assert bad.status_code == 400


def test_missing_key_reports_config_error():
    sid = new_session()
    ids = agent_ids("gpt-4o-mini")
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        _, results = ask(ws, "hi", ids, 1)
    assert results[0]["status"] == "error" and results[0]["error_kind"] == "authentication"


def test_no_credentials_in_api_output(monkeypatch):  # AC-08
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-test-value")
    assert "sk-secret-test-value" not in client.get("/api/agents").text


def test_key_entered_in_browser_is_saved_but_never_returned():
    secret = "sk-browser-entered-123"
    assert client.get("/api/keys").json()["openai"] is False
    assert client.post("/api/keys", json={"provider": "openai", "key": secret}).status_code == 200
    assert client.get("/api/keys").json()["openai"] is True
    for path in ("/api/keys", "/api/agents"):
        assert secret not in client.get(path).text
    gpt = [a for a in client.get("/api/agents").json() if a["provider"] == "openai"][0]
    assert gpt["configured"] is True
    assert client.delete("/api/keys/openai").status_code == 204
    assert client.get("/api/keys").json()["openai"] is False


def test_bad_key_input_rejected():
    assert client.post("/api/keys", json={"provider": "nope", "key": "abc"}).status_code == 400
    assert client.post("/api/keys", json={"provider": "openai", "key": "has space"}).status_code == 400


def test_custom_agent_uses_base_url_and_own_key(monkeypatch):
    bad = client.post("/api/agents", json={"name": "C", "model": "m", "provider": "custom"})
    assert bad.status_code == 400  # needs a base URL
    r = client.post("/api/agents", json={"name": "Groq Llama", "model": "llama-3.3", "provider": "custom",
                                         "base_url": "https://api.groq.com/openai/v1/", "api_key": "gsk-own-key-1"})
    assert r.status_code == 201
    agent = r.json()
    assert agent["configured"] is True and "gsk-own-key-1" not in client.get("/api/agents").text

    seen = {}

    async def fake_post(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {"choices": [{"message": {"content": "hello from groq"}}]}

    monkeypatch.setattr(main, "_post", fake_post)
    sid = new_session()
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        _, results = ask(ws, "hi", [agent["id"]], 1)
    assert results[0]["content"] == "hello from groq"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer gsk-own-key-1"


def test_anthropic_adapter_and_remove_agent(monkeypatch):
    agent = client.post("/api/agents", json={"name": "Claude 2", "model": "claude-x", "provider": "anthropic",
                                             "api_key": "sk-ant-test"}).json()

    async def fake_post(url, headers, body):
        assert url == "https://api.anthropic.com/v1/messages" and headers["x-api-key"] == "sk-ant-test"
        return {"content": [{"type": "text", "text": "hi from claude"}]}

    monkeypatch.setattr(main, "_post", fake_post)
    sid = new_session()
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        _, results = ask(ws, "hi", [agent["id"]], 1)
    assert results[0]["content"] == "hi from claude"
    assert client.delete(f"/api/agents/{agent['id']}").status_code == 204
    assert all(a["id"] != agent["id"] for a in client.get("/api/agents").json())
