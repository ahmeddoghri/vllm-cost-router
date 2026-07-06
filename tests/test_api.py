from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_completion_endpoint_routes_and_returns_result():
    resp = client.post("/v1/completions", json={"prompt": "What are your business hours?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body and "model" in body and "cost_usd" in body


def test_batch_completion_endpoint():
    resp = client.post("/v1/batch/completions", json={
        "requests": [
            {"prompt": "What are your business hours?"},
            {"prompt": "Explain the trade-offs between REST and gRPC in detail."},
        ]
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2


def test_metrics_endpoint_reflects_traffic():
    client.post("/v1/completions", json={"prompt": "hello metrics test unique prompt"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] >= 1


def test_invalid_max_tokens_rejected():
    resp = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 0})
    assert resp.status_code == 422


def test_readyz():
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_empty_prompt_rejected():
    resp = client.post("/v1/completions", json={"prompt": ""})
    assert resp.status_code == 422


def test_oversized_prompt_rejected():
    from app.config import settings
    resp = client.post("/v1/completions", json={"prompt": "x" * (settings.max_prompt_chars + 1)})
    assert resp.status_code == 422


def test_empty_batch_rejected():
    resp = client.post("/v1/batch/completions", json={"requests": []})
    assert resp.status_code == 422


def test_request_id_header_present():
    resp = client.get("/healthz")
    assert resp.headers.get("x-request-id")


def test_api_key_enforced_when_configured(monkeypatch):
    # Rebuild the app with an API_KEY set so the auth dependency activates.
    monkeypatch.setenv("API_KEY", "s3cret")
    import importlib

    import app.config
    import app.main
    import app.security
    importlib.reload(app.config)
    importlib.reload(app.security)
    reloaded = importlib.reload(app.main)
    guarded = TestClient(reloaded.app)
    assert guarded.post("/v1/completions", json={"prompt": "hi"}).status_code == 401
    ok = guarded.post("/v1/completions", json={"prompt": "hi"},
                      headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    # Restore the default (open) app for any later tests.
    monkeypatch.delenv("API_KEY", raising=False)
    importlib.reload(app.config)
    importlib.reload(app.security)
    importlib.reload(app.main)
