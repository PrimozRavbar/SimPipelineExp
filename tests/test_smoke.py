"""End-to-end smoke test: the whole loop builds, runs, retrains and serves.

Run with:  pytest -q tests/test_smoke.py
Takes about a minute on a laptop CPU (most of it is HNSW index builds).
"""

import pytest

torch = pytest.importorskip("torch")

from world.external_data_loader import load_movielens
from ml_system.build_recommendation_system import build_recommendation_system
from ml_system.retrain import retrain
from simulation.events import RetrievalRequest


def build(seed=7):
    data = load_movielens()

    return build_recommendation_system(
        movie_objects=data["movie_objects"],
        user_objects=data["user_objects"],
        user_states=data["user_states"],
        movie_states=data["movie_states"],
        seed=seed,
    )


def event_signature(events):
    return [
        (
            type(event).__name__,
            event.timestamp,
            event.user_id,
            getattr(event, "movie_id", None),
            tuple(getattr(event, "movie_ids", ()) or ()),
        )
        for event in events
    ]


def test_end_to_end_loop_runs():
    system = build()

    system["sim"].run(200, verbose=False)

    events = system["datalake"].events
    assert len(events) >= 200, "every tick should log at least one shown event"

    system["spark"].run_offline_feature_job(
        events,
        system["offline_pipeline"].feature_pipeline,
    )

    system["offline_pipeline"].trainer.epochs = 1
    retrain(system)

    recommendations = system["retriever"].retrieve(
        RetrievalRequest(user_id=1, k=20)
    )

    assert len(recommendations) == 20
    assert len(set(recommendations)) == 20


def test_same_seed_gives_same_simulation():
    first = build(seed=7)
    first["sim"].run(100, verbose=False)

    second = build(seed=7)
    second["sim"].run(100, verbose=False)

    assert event_signature(first["datalake"].events) == event_signature(
        second["datalake"].events
    )
