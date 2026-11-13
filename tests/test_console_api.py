"""The console API (console/server.py) on a real SimSession, no server process."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from console.server import create_app
from console.session import SimSession


@pytest.fixture(scope="module")
def client():
    session = SimSession(preset="plain", seed=5, users_per_tick=3, settings={"deploy_delay": 5, "epochs": 1},
                         log_metrics=True, run_id="console-api")
    yield TestClient(create_app(session))
    session.close()


def test_ui_and_state(client):
    page = client.get("/")
    assert page.status_code == 200 and "SimPipeline Console" in page.text
    assert "/api/snapshot" in page.text         # the UI reads the API (one combined poll)

    state = client.get("/api/state").json()
    assert state["tick"] == 0 and state["model_version"] == 0
    assert 0 < state["reference"]["random_ctr"] < state["reference"]["oracle_ctr"]


def test_step_retrain_rollback_flow(client):
    assert client.post("/api/step", json={"ticks": 20}).json()["tick"] == 20

    rows = client.get("/api/metrics", params={"since": 10}).json()["rows"]
    assert [r["tick"] for r in rows] == list(range(11, 21))
    assert all(r["shown"] == 3 and r["active_users"] == 3 for r in rows)

    queued = client.post("/api/retrain").json()
    assert queued["action"]["status"] == "pending"
    client.post("/api/step", json={"ticks": 7})        # deploys at tick 25
    models = client.get("/api/models").json()
    assert models["serving_version"] == 1 and models["rollback_target"] == 0

    client.post("/api/rollback")
    client.post("/api/step", json={"ticks": 1})
    assert client.get("/api/state").json()["model_version"] == 0

    kinds = [e["kind"] for e in client.get("/api/events").json()["events"]]
    assert "train" in kinds and kinds.count("deploy") >= 3


def test_settings_and_errors(client):
    assert client.post("/api/settings", json={"slate_size": 999}).status_code == 400
    assert client.post("/api/speed", json={"speed": 7}).status_code == 400

    client.post("/api/settings", json={"slate_size": 8, "spark_every": 5})
    client.post("/api/step", json={"ticks": 6})
    pipe = client.get("/api/pipeline").json()
    assert pipe["settings"]["slate_size"] == 8 and pipe["server"]["slate_size"] == 8
    assert pipe["spark"]["last_run_tick"] is not None

    client.post("/api/play")
    assert client.post("/api/step", json={"ticks": 1}).status_code == 409
    client.post("/api/pause")
    assert client.get("/api/incidents").json() == {"incidents": []}


def test_snapshot_bundles_every_read(client):
    snap = client.get("/api/snapshot", params={"since": 0, "events_since": 0}).json()
    assert set(snap) == {"state", "metrics", "events", "incidents", "models", "pipeline", "faults"}
    assert snap["state"]["tick"] == client.get("/api/state").json()["tick"]
    assert snap["metrics"]["rows"] == client.get("/api/metrics", params={"since": 0}).json()["rows"]
    assert snap["events"] == client.get("/api/events").json()["events"]
