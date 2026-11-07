"""Held-out RetrievalEvaluator: builds, trains on the past, scores the future."""

import pytest

torch = pytest.importorskip("torch")

from world.external_data_loader import load_movielens
from ml_system.build_recommendation_system import build_recommendation_system
from ml_system.evaluation import RetrievalEvaluator


@pytest.fixture(scope="module")
def simulated():
    data = load_movielens()

    system = build_recommendation_system(
        movie_objects=data["movie_objects"],
        user_objects=data["user_objects"],
        user_states=data["user_states"],
        movie_states=data["movie_states"],
        seed=7,
    )
    system["sim"].run(300, verbose=False)

    return data, list(system["datalake"].events)


def evaluate(data, events, seed):
    evaluator = RetrievalEvaluator(
        data["movie_objects"],
        num_users=len(data["user_objects"]),
        seed=seed,
        epochs=1,
    )
    return evaluator.evaluate(events)


def test_evaluator_runs_on_held_out_split(simulated):
    """Regression: it used to crash on SimpleUserTower() without num_users."""
    data, events = simulated

    result = evaluate(data, events, seed=0)

    assert result["train_events"] + result["validation_events"] == len(events)
    assert result["evaluated"] > 0
    assert 0.0 <= result["recall_at_k"] <= 1.0
    assert len(result["validation_population"]) == result["evaluated"]


def test_evaluator_same_seed_same_result(simulated):
    data, events = simulated

    first = evaluate(data, events, seed=0)
    second = evaluate(data, events, seed=0)

    assert first == second


def test_evaluator_rejects_empty_training_split(simulated):
    data, _ = simulated

    with pytest.raises(ValueError):
        evaluate(data, [], seed=0)
