"""Movie signals and co-watching (ml_system/item_signals.py), the v4
ranker features and the co-watch candidate source."""

import pytest

from ml_system.candidate_sources import CoWatchRetrieval
from ml_system.item_signals import COWATCH_MEMORY, ItemSignals
from ml_system.ranking import feature_names, features_v4
from ml_system.redis import Redis
from simulation.events import (RatingEvent, RecommendationClickedEvent, RecommendationShownEvent,
                               RetrievalRequest, WatchEvent)


def watch(user, movie, completion=0.8):
    return WatchEvent(timestamp=1, user_id=user, movie_id=movie, watch_time=1, completion=completion)


def test_statistics_are_smoothed():
    s = ItemSignals()
    s.observe(RecommendationShownEvent(timestamp=1, user_id=1, recommendation_id=1, movie_ids=[1, 2]))
    s.observe(RecommendationClickedEvent(timestamp=1, user_id=1, recommendation_id=1, movie_id=1))
    s.observe(watch(1, 1, 1.0))
    s.observe(RatingEvent(timestamp=1, user_id=1, movie_id=1, rating=5))
    ctr1, comp1, rating1, imp1, coec1 = s.stats(1)
    ctr2, _, _, _, coec2 = s.stats(2)
    assert coec1 > 1 > coec2                   # clicked more / less than expected in its slot
    assert 1 > ctr1 > ctr2 > 0                # both smoothed towards the global 50%
    assert comp1 == pytest.approx(1.0) and rating1 == pytest.approx(1.0)
    assert s.stats(99)[3] == 0.0               # never shown


def test_cowatch_counts_and_neighbours():
    s = ItemSignals()
    for user in (1, 2, 3):
        s.observe(watch(user, 10))
        s.observe(watch(user, 20))
    s.observe(watch(4, 10))
    s.observe(watch(4, 30))
    assert s.similarity(10, 20) == pytest.approx(3 / (4 * 3) ** 0.5)
    assert s.neighbours([10], 2) == [20, 30]
    assert s.cowatch_score(20, [10, 30]) == pytest.approx(s.similarity(10, 20))


def test_cowatch_memory_is_bounded():
    s = ItemSignals()
    for m in range(1, COWATCH_MEMORY + 5):
        s.observe(watch(1, m))
    assert len(s.user_watches[1]) == COWATCH_MEMORY
    assert 1 not in s.cowatch.get(COWATCH_MEMORY + 4, {})     # too far back to count


def test_v4_features_and_cowatch_source():
    s = ItemSignals()
    for user in (1, 2):
        s.observe(watch(user, 10))
        s.observe(watch(user, 20))
    row = features_v4({"genre_preferences": {"Drama": 1.0}}, [10], [], {"genres": ["Drama"], "popularity": 1},
                      20, lambda m: ["Drama"], signals=s)
    named = dict(zip(feature_names("v4"), row))
    assert len(row) == len(feature_names("v4"))
    assert named["cowatch_score"] == pytest.approx(1.0)
    redis = Redis()
    redis.set("item_signals", "all", s)
    redis.set("recent_watches", 7, [10])
    assert CoWatchRetrieval(redis).retrieve(RetrievalRequest(user_id=7, k=5)) == [20]
    assert CoWatchRetrieval(redis).retrieve(RetrievalRequest(user_id=8, k=5)) == []


torch = pytest.importorskip("torch")


def test_flink_signals_match_a_replay_of_the_log():
    from ml_system.build_recommendation_system import build_recommendation_system
    from world.external_data_loader import load_movielens

    data = load_movielens()
    system = build_recommendation_system(data["movie_objects"], data["user_objects"], data["user_states"],
                                         data["movie_states"], seed=2, users_per_tick=5, log_metrics=False)
    system["sim"].get_service("Flink").track_item_signals = True
    system["sim"].run(80, verbose=False)
    online = system["redis"].get("item_signals", "all")
    replay = ItemSignals()
    for e in system["datalake"].events:
        replay.observe(e)
    assert online.impressions == replay.impressions and online.cowatch == replay.cowatch
    shown = sorted(online.impressions)[:20]
    assert len(shown) == 20
    for m in shown:
        assert online.stats(m) == pytest.approx(replay.stats(m))
