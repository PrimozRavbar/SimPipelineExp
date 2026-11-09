"""Realism switches: relevance-driven outcomes, several users per tick,
and the popularity baseline without the held-out leak."""

import random
from types import SimpleNamespace

import pytest

from world.external_data_loader import load_movielens
from world.relevance import relevance
from world.user_simulator import UserSimulator
from ml_system.evaluation import evaluate_popularity
from simulation.events import RatingEvent, WatchEvent, RecommendationClickedEvent


@pytest.fixture(scope="module")
def data():
    return load_movielens()


def outcomes(data, outcome_model, seed=5, n=3000):
    """Force a click on one movie per user and collect watch/rating."""
    world = SimpleNamespace(
        user_states=data["user_states"], movie_objects=data["movie_objects"]
    )
    simulator = UserSimulator(world, rng=random.Random(seed), outcome_model=outcome_model)
    sim = SimpleNamespace(clock=SimpleNamespace(now=0))
    rng = random.Random(seed)
    users = list(data["user_objects"].values())
    movies = list(data["movie_objects"])
    rows = []

    for _ in range(n):
        user = rng.choice(users)
        movie_id = rng.choice(movies)
        rel = simulator.relevance_of(user, movie_id)
        watched = simulator.should_watch(rel)
        watch = simulator.watch(sim, user, movie_id, rel)
        rating = simulator.rate(sim, user, movie_id, rel)
        rows.append((rel, watched, watch.completion, rating.rating))

    return rows


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def test_relevance_outcomes_rise_with_relevance(data):
    rows = outcomes(data, "relevance")
    high = [r for r in rows if r[0] >= 0.65]
    low = [r for r in rows if r[0] < 0.3]

    assert mean(r[3] for r in high) > mean(r[3] for r in low) + 1.5   # stars
    assert mean(r[2] for r in high) > mean(r[2] for r in low) + 0.3   # completion
    assert mean(r[1] for r in high) > mean(r[1] for r in low)         # watch rate
    assert all(1 <= r[3] <= 5 for r in rows)
    assert all(0 < r[2] <= 1 for r in rows)


def test_random_outcomes_ignore_relevance(data):
    rows = outcomes(data, "random")
    high = [r for r in rows if r[0] >= 0.65]
    low = [r for r in rows if r[0] < 0.3]

    assert abs(mean(r[3] for r in high) - mean(r[3] for r in low)) < 0.4


def test_outcomes_are_seeded(data):
    assert outcomes(data, "relevance", seed=9, n=200) == outcomes(data, "relevance", seed=9, n=200)
    assert outcomes(data, "relevance", seed=9, n=200) != outcomes(data, "relevance", seed=10, n=200)

    with pytest.raises(ValueError):
        UserSimulator(SimpleNamespace(), outcome_model="nope")


def test_popularity_baseline_does_not_pad_with_held_out_movies():
    train = [
        RecommendationClickedEvent(timestamp=t, user_id=1, recommendation_id=t, movie_id=m)
        for t, m in enumerate([5, 5, 6])
    ]
    held_out_movie = 999
    events = train + [WatchEvent(timestamp=9, user_id=1, movie_id=held_out_movie,
                                 watch_time=1.0, completion=1.0)]
    population = [{"user_id": 1, "movie_id": held_out_movie}]

    # Split so that the three clicks are training events.
    result = evaluate_popularity(events, population, split_ratio=0.75, k=20)
    assert result["hits"] == 0

    padded = evaluate_popularity(
        events, population, split_ratio=0.75, k=3,
        movie_objects={1: None, 5: None, 6: None, held_out_movie: None},
    )
    assert padded["hits"] == 0   # padded with movie 1, not the answer


# ---------------------------------------------------------------------
# Several users per tick (needs the full system)
# ---------------------------------------------------------------------

torch = pytest.importorskip("torch")

from ml_system.build_recommendation_system import build_recommendation_system  # noqa: E402
from simulation.events import RecommendationShownEvent  # noqa: E402


def run_multi_user(data, seed, ticks=15, users_per_tick=5):
    system = build_recommendation_system(
        data["movie_objects"], data["user_objects"],
        data["user_states"], data["movie_states"],
        seed=seed, users_per_tick=users_per_tick,
    )
    system["sim"].run(ticks, verbose=False)
    return system


def test_several_users_per_tick(data):
    system = run_multi_user(data, seed=4)
    shown = [e for e in system["datalake"].events if isinstance(e, RecommendationShownEvent)]

    per_tick = {}
    for event in shown:
        per_tick.setdefault(event.timestamp, []).append(event.user_id)

    assert len(per_tick) == 15
    assert all(len(users) == 5 == len(set(users)) for users in per_tick.values())

    # Slate ids stay unique, so clicks join the right slate in training.
    assert len({e.recommendation_id for e in shown}) == len(shown)

    rows = system["ctr_tracker"].per_tick()
    assert all(row["shown"] == 5 for row in rows)

    signature = [(type(e).__name__, e.timestamp, e.user_id, getattr(e, "movie_id", None))
                 for e in system["datalake"].events]
    again = run_multi_user(data, seed=4)
    assert signature == [(type(e).__name__, e.timestamp, e.user_id, getattr(e, "movie_id", None))
                         for e in again["datalake"].events]
