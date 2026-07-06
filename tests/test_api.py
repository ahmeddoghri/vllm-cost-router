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
