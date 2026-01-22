from fastapi.testclient import TestClient
from webapp.api import app


def test_query_endpoint():
    client = TestClient(app)
    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 200
    assert "answer" in resp.json()
