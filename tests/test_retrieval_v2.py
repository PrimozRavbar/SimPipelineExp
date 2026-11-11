"""Retrieval v2: logQ correction, random negatives, graded positives,
recent-watch history in the user tower, multi-task ranker, stage report."""

import copy
import random

import pytest

torch = pytest.importorskip("torch")

from ml_system.training import TwoTowerTrainer
from ml_system.training_dataset import TrainingDatasetGenerator, grade
from ml_system.two_tower import SimpleItemTower, SimpleUserTower
from simulation.events import RecommendationClickedEvent, RecommendationShownEvent, WatchEvent, RatingEvent
from tests.test_batched_training import synthetic_data


def test_grade():
    assert grade({}) == 0.3
    assert grade({"completion": 0.5}) == 1.0
    assert grade({"completion": 1.0, "rating": 5}) == 2.0


class Store:
    def get_user_features_as_of(self, user_id, t, order):
        return {"genre_preferences": {"Drama": 1.0}}

    def get_item_features_as_of(self, movie_id, t, order):
        return {"movie_id": movie_id, "genres": ["Drama"]}


def test_dataset_grades_clicks_and_attaches_history():
    gen = TrainingDatasetGenerator(Store(), negative_samples=2, rng=random.Random(0), graded=True, history=True)
    events = [
        RecommendationShownEvent(timestamp=1, user_id=1, recommendation_id=1, movie_ids=[5, 6, 7]),
        RecommendationClickedEvent(timestamp=1, user_id=1, recommendation_id=1, movie_id=6),
        WatchEvent(timestamp=1, user_id=1, movie_id=6, watch_time=96, completion=0.8),
        RatingEvent(timestamp=1, user_id=1, movie_id=6, rating=5),
        RecommendationShownEvent(timestamp=2, user_id=1, recommendation_id=2, movie_ids=[8, 9]),
        RecommendationClickedEvent(timestamp=2, user_id=1, recommendation_id=2, movie_id=9),
    ]
    for order, e in enumerate(events):
        gen.add_event(e, order)
    data = sorted(gen.build(), key=lambda e: e["positive_item_features"]["movie_id"])
    assert [e["weight"] for e in data] == [pytest.approx(0.5 + 0.8 + 0.5), 0.3]
    assert data[0]["user_features"]["recent_watches"] == []        # history before the first click
    assert data[1]["user_features"]["recent_watches"] == [6]       # the watch came before click 2


def trained(**options):
    torch.manual_seed(0)
    trainer = TwoTowerTrainer(SimpleUserTower(num_users=30), SimpleItemTower(num_movies=50),
                              epochs=2, batch_size=16, **options)
    data = synthetic_data()
    trainer.item_pool = [e["positive_item_features"] for e in data]
    for i, e in enumerate(data):
        e["weight"] = 0.3 + (i % 3) * 0.5
    trainer.train(data)
    return trainer


def test_options_change_training_and_are_deterministic():
    base = trained()
    for options in ({"logq_correction": True}, {"random_negatives": 3}, {"graded": True}):
        a, b = trained(**options), trained(**options)
        assert a.last_losses == b.last_losses, options
        assert a.last_losses != base.last_losses, options


def test_options_need_batched():
    with pytest.raises(ValueError):
        trained(batched=False, logq_correction=True)


def test_history_tower():
    tower = SimpleUserTower(num_users=5, use_history=True, num_movies=10)
    features = {"user_id": 1, "genre_preferences": {"Drama": 1.0}}
    a = tower.embed(features)
    b = tower.embed({**features, "recent_watches": [3, 4]})
    assert len(a) == len(b) and a != b
    with pytest.raises(ValueError):
        SimpleUserTower(num_users=5, use_history=True)
    # Off by default: no extra parameters, same input size as before.
    assert not hasattr(SimpleUserTower(num_users=5), "history_embedding")


def test_multitask_ranker_learns_completion():
    from ml_system.offline_feature_store import OfflineFeatureStore
    from ml_system.ranking import MLPRanker
    from ml_system.redis import Redis

    store = OfflineFeatureStore()
    for m in range(1, 7):
        store.write_item_features(m, {"movie_id": m, "genres": ["Comedy"] if m % 2 else ["Drama"],
                                      "popularity": 1, "average_rating": 3}, timestamp=0, event_order=-1)
    store.write_user_features(1, {"genre_preferences": {"Comedy": 1.0, "Drama": 1.0}}, timestamp=0, event_order=-1)
    events, rec = [], 0
    for t in range(1, 300):
        rec += 1
        slate = [1, 2] if t % 2 else [2, 1]
        clicked = slate[1]           # second slot: the first one is a skipped negative
        events += [RecommendationShownEvent(timestamp=t, user_id=1, recommendation_id=rec, movie_ids=slate),
                   RecommendationClickedEvent(timestamp=t, user_id=1, recommendation_id=rec, movie_id=clicked),
                   WatchEvent(timestamp=t, user_id=1, movie_id=clicked, watch_time=1,
                              completion=0.9 if clicked % 2 else 0.1)]
    genres = {m: (["Comedy"] if m % 2 else ["Drama"]) for m in range(1, 7)}
    rows = MLPRanker.training_rows_from(events, store, genres.get, "v3", with_completion=True)
    assert len(rows) == 4 and any(c is not None for c in rows[3])
    ranker = MLPRanker(lambda u: store.get_user_features(u), Redis(), store, genres.get, seed=1,
                       feature_set="v3", multitask=True)
    ranker.fit(*rows, epochs=40)
    # Both genres get clicked equally; Comedy is watched far more fully.
    order = ranker.rank(type("U", (), {"user_id": 1})(), [2, 4, 6, 1, 3, 5])
    assert set(order[:3]) == {1, 3, 5}


def test_stage_report_is_read_only_and_consistent():
    from console.session import SimSession

    s = SimSession(seed=3, users_per_tick=4, log_metrics=False,
                   settings={"deploy_delay": 10, "epochs": 1, "ranker": True, "profile_candidates": 20,
                             "onboarding": True, "exploration_pct": 10})
    s.step(40)
    s.submit("retrain")
    s.step(15)
    rng_state = s.explore_rng.getstate()
    r = s.insight()
    assert s.explore_rng.getstate() == rng_state
    assert r["users"] > 0 and r["oracle"] >= r["candidates"] - 1e-9
    assert r["candidates"] >= max(r["two_tower"], r["extra"]) - 1e-9
    assert r["retrieval_loss"] == pytest.approx(r["oracle"] - r["candidates"])
    s.close()
