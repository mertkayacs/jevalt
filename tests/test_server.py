from fastapi.testclient import TestClient

from jevalt.engine import DecisionEngine
from jevalt.server import create_app

from .test_engine import REQUEST, StubBackend


def client(**kw):
    return TestClient(create_app(DecisionEngine(StubBackend(), model="JevAlt-test"), **kw))


def test_systemone_roundtrip():
    r = client().post("/v1/systemone", json={**REQUEST, "model": "jev-latest"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "JevAlt-test" and set(body["answers"]) == {"team", "anger", "urgent"}


def test_validation_error_is_422():
    assert client().post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
    bad = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}}
    assert client().post("/v1/systemone", json=bad).status_code == 422


def test_bearer_key_enforced():
    c = client(api_key="s3cret")
    assert c.post("/v1/systemone", json=REQUEST).status_code == 401
    assert c.post("/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_private_network_preflight():
    r = client().options("/v1/systemone", headers={"Origin": "https://jevoss.mertkayacs.com", "Access-Control-Request-Method": "POST", "Access-Control-Request-Private-Network": "true"})
    assert r.headers.get("access-control-allow-private-network") == "true"
