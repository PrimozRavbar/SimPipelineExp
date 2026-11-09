"""Metrics log (runs/<run_id>/metrics.jsonl) and the HTML dashboard."""

import json
import random
from types import SimpleNamespace

import pytest

from dashboard import render_dashboard, render_fragment
from ml_system.kafka import Kafka
from ml_system.metrics import CTRTracker, MetricsLogger, read_metrics, reference_ctrs
from simulation.events import RecommendationClickedEvent, RecommendationShownEvent
from world.external_data_loader import load_movielens
from world.relevance import relevance, relevance_matrix
from world.user_simulator import UserSimulator


class FakeSim:
    def __init__(self):
        self.kafka = Kafka()
        self.clock = SimpleNamespace(now=0)

    def get_service(self, name):
        return self.kafka


def logged_run(run_dir):
    """Three ticks: 2 users/tick, one click per tick, a retrain after tick 2."""
    sim = FakeSim()
    tracker = CTRTracker()
    tracker.start(sim)
    logger = MetricsLogger(run_dir, tracker, run_info={
        "run_id": "unit", "seed": 3, "config": {"users_per_tick": 2},
        "reference": {"random_ctr": 0.17, "oracle_ctr": 0.69},
    })
    logger.start(sim)

    rec = 0
    for tick in (1, 2, 3):
        sim.clock.now = tick
        for user_id in (10, 20):
            rec += 1
            sim.kafka.receive(RecommendationShownEvent(
                timestamp=tick, user_id=user_id, recommendation_id=rec, movie_ids=[1, 2]))
        sim.kafka.receive(RecommendationClickedEvent(
            timestamp=tick, user_id=10, recommendation_id=rec - 1, movie_id=1))
        tracker.process(sim)
        logger.process(sim)
        if tick == 2:
            logger.log_retrain(tick, training_examples=42)

    logger.log_event("evaluation", 3, k=20, recall_at_k=0.08, random_recall_at_k=0.012)
    return logger


def test_metrics_log_contents(tmp_path):
    logger = logged_run(tmp_path / "run")
    records = read_metrics(tmp_path / "run")

    assert [r["type"] for r in records] == [
        "run_start", "tick", "tick", "retrain", "deploy", "tick", "evaluation"]
    assert records[0]["seed"] == 3 and records[0]["model_version"] == 0
    assert records[0]["reference"] == {"random_ctr": 0.17, "oracle_ctr": 0.69}

    ticks = [r for r in records if r["type"] == "tick"]
    assert [(r["tick"], r["shown"], r["clicks"], r["ctr"], r["active_users"]) for r in ticks] == [
        (1, 2, 1, 0.5, 2), (2, 2, 1, 0.5, 2), (3, 2, 1, 0.5, 2)]
    assert [r["model_version"] for r in ticks] == [0, 0, 1]
    assert records[3]["training_examples"] == 42 and records[4]["model_version"] == 1
    assert logger.model_version == 1

    # Every line is standalone JSON.
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert all(json.loads(line) for line in lines)


def test_dashboard_renders_from_log_only(tmp_path):
    logged_run(tmp_path / "run")

    path = render_dashboard(tmp_path / "run", window=1)

    assert path == tmp_path / "run" / "dashboard.html"
    page = path.read_text()
    assert page.startswith("<!doctype html>")
    assert "<script" not in page and "http" not in page   # self-contained
    for text in ("unit", "Recall@20 vs random", "8.0%", "Active users / tick",
                 "Retrains", "oracle 69%", "random 17%", "model v1 deployed",
                 "Event log", "<svg"):
        assert text in page, text
    assert render_fragment(tmp_path / "run", window=1).startswith('<div class="simdash">')


def test_relevance_matrix_matches_relevance():
    data = load_movielens()
    users = list(data["user_states"].values())[:40]
    movies = list(data["movie_objects"].values())[:200]

    matrix = relevance_matrix(users, movies)

    for i, user in enumerate(users):
        for j in range(0, 200, 7):
            assert matrix[i, j] == pytest.approx(relevance(user, movies[j]))


def test_reference_ctrs_match_claude_md():
    data = load_movielens()
    world = SimpleNamespace(
        user_objects=data["user_objects"], user_states=data["user_states"],
        movie_objects=data["movie_objects"],
    )
    world.user_simulator = UserSimulator(world, rng=random.Random(0))

    ref = reference_ctrs(world, seed=1)

    assert 0.64 < ref["oracle_ctr"] < 0.72
    assert 0.14 < ref["random_ctr"] < 0.20
    assert ref == reference_ctrs(world, seed=1)


torch = pytest.importorskip("torch")


def test_full_run_logs_retrain_and_renders(tmp_path):
    from ml_system.build_recommendation_system import build_recommendation_system
    from ml_system.retrain import retrain

    data = load_movielens()

    def build(run_id):
        return build_recommendation_system(
            data["movie_objects"], data["user_objects"],
            data["user_states"], data["movie_states"],
            seed=5, users_per_tick=3, runs_dir=tmp_path, run_id=run_id,
        )

    system = build("a")
    system["sim"].run(30, verbose=False)
    system["offline_pipeline"].trainer.epochs = 1
    retrain(system)
    system["sim"].run(5, verbose=False)

    records = read_metrics(tmp_path / "a")
    ticks = [r for r in records if r["type"] == "tick"]
    assert len(ticks) == 35 and all(r["active_users"] == 3 for r in ticks)
    assert [r["type"] for r in records if r["type"] in ("retrain", "deploy")] == ["retrain", "deploy"]
    assert ticks[29]["model_version"] == 0 and ticks[30]["model_version"] == 1
    assert render_dashboard(system["run_dir"]).exists()

    # Same seed, same per-tick log.
    again = build("b")
    again["sim"].run(30, verbose=False)
    strip = lambda rs: [r for r in rs if r["type"] == "tick"]
    assert strip(read_metrics(tmp_path / "b")) == ticks[:30]
