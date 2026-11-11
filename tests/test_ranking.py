"""Ranking stage (ml_system/ranking.py): training rows from logged slates,
position as a feature, serving order, the watched filter."""

import pytest

from ml_system.offline_feature_store import OfflineFeatureStore
from ml_system.ranking import LogisticRanker, WatchedFilter, genre_match
from ml_system.redis import Redis
from simulation.events import RecommendationClickedEvent, RecommendationShownEvent


class U:
    def __init__(self, user_id):
        self.user_id = user_id


def store_with_items():
    store = OfflineFeatureStore()
    for m in range(1, 11):
        genres = ["Comedy"] if m % 2 else ["Drama"]
        feats = {"movie_id": m, "genres": genres, "popularity": 1, "average_rating": 3}
        store.write_item_features(m, feats, timestamp=0, event_order=-1)
    store.write_user_features(1, {"genre_preferences": {"Comedy": 1.0}}, timestamp=0, event_order=-1)
    return store


def test_genre_match():
    assert genre_match({"Comedy": 3, "Drama": 1}, ["Comedy"]) == pytest.approx(0.75)
    assert genre_match({}, ["Comedy"]) == 0.0


def test_rows_follow_the_cascade():
    store = store_with_items()
    events = [
        RecommendationShownEvent(timestamp=1, user_id=1, recommendation_id=1, movie_ids=[2, 4, 3, 6]),
        RecommendationClickedEvent(timestamp=1, user_id=1, recommendation_id=1, movie_id=3),
        RecommendationShownEvent(timestamp=2, user_id=1, recommendation_id=2, movie_ids=[5, 7]),
    ]
    rows, labels = LogisticRanker.training_rows_from(events, store)
    # Clicked slate: positions 0..2 (stop at the click); unclicked slate: both.
    assert labels == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert [row[4:].index(1.0) for row in rows] == [0, 1, 2, 0, 1]


def test_ranker_learns_genre_match_and_ignores_position_at_serving():
    store = store_with_items()
    events, rec = [], 0
    for t in range(1, 200):
        rec += 1
        slate = [2, 1, 4, 3] if t % 2 else [1, 2, 3, 4]       # Comedy = odd ids
        events.append(RecommendationShownEvent(timestamp=t, user_id=1, recommendation_id=rec, movie_ids=slate))
        clicked = next(m for m in slate if m % 2)              # always the first Comedy
        events.append(RecommendationClickedEvent(timestamp=t, user_id=1, recommendation_id=rec, movie_id=clicked))

    ranker = LogisticRanker(lambda user_id: store.get_user_features(user_id), store)
    ranker.fit(*LogisticRanker.training_rows_from(events, store))
    assert ranker.content_weights()["genre_match"] > 0
    order = ranker.rank(U(1), [2, 4, 6, 1, 3, 5])
    assert set(order[:3]) == {1, 3, 5}


def test_fit_needs_both_labels():
    ranker = LogisticRanker(lambda u: {}, OfflineFeatureStore())
    with pytest.raises(ValueError):
        ranker.fit([[0.0] * 14], [0.0])


def test_watched_filter():
    redis = Redis()
    redis.store[("watched_movies", 1)] = {2, 3}
    assert WatchedFilter(redis).rerank(U(1), [1, 2, 3, 4]) == [1, 4]
    assert WatchedFilter(redis).rerank(U(2), [1, 2]) == [1, 2]


# ---------------------------------------------------------------------
# MLP ranker (v2 features)
# ---------------------------------------------------------------------

from ml_system.ranking import FEATURES_V2, MLPRanker, RECENT_SLATES, RECENT_WATCHES, RecentHistory, features_v2
from simulation.events import WatchEvent

GENRES = {m: (["Comedy"] if m % 2 else ["Drama"]) for m in range(1, 11)}


def test_recent_history_is_bounded():
    h = RecentHistory()
    for m in range(10):
        h.add_watch(1, m)
        h.add_slate(1, [m, m + 1])
    watches, slates = h.get(1)
    assert watches == list(range(10 - RECENT_WATCHES, 10))
    assert len(slates) == RECENT_SLATES and slates[-1] == (9, 10)


def test_features_v2():
    item = {"genres": ["Comedy"], "popularity": 3, "average_rating": 4}
    row = features_v2({"genre_preferences": {"Comedy": 1.0}}, [1, 3], [(5, 7), (5,)], item, 5, GENRES.get)
    assert len(row) == len(FEATURES_V2)
    named = dict(zip(FEATURES_V2, row))
    assert named["recent_genre_match"] == pytest.approx(1.0)
    assert named["watched_recently"] == 0.0 and named["shown_recently"] == 2.0
    assert named["coldstart_x_popularity"] == 0.0


def test_mlp_rows_use_history_from_before_each_slate():
    store = store_with_items()
    events = [
        RecommendationShownEvent(timestamp=1, user_id=1, recommendation_id=1, movie_ids=[2, 3]),
        RecommendationClickedEvent(timestamp=1, user_id=1, recommendation_id=1, movie_id=3),
        WatchEvent(timestamp=1, user_id=1, movie_id=3, watch_time=60, completion=0.5),
        RecommendationShownEvent(timestamp=2, user_id=1, recommendation_id=2, movie_ids=[3, 4]),
    ]
    rows, positions, labels = MLPRanker.training_rows_from(events, store, GENRES.get)
    named = [dict(zip(FEATURES_V2, r)) for r in rows]
    # First slate: no history yet. Second slate: movie 3 was watched and shown.
    assert named[0]["shown_recently"] == 0 and named[1]["watched_recently"] == 0
    assert named[2]["watched_recently"] == 1.0 and named[2]["shown_recently"] == 1.0
    assert positions == [0, 1, 0, 1] and labels == [0.0, 1.0, 0.0, 0.0]


def test_mlp_ranker_learns_and_is_deterministic():
    store = store_with_items()
    events, rec = [], 0
    for t in range(1, 300):
        rec += 1
        slate = [2, 1, 4, 3] if t % 2 else [1, 2, 3, 4]
        events.append(RecommendationShownEvent(timestamp=t, user_id=1, recommendation_id=rec, movie_ids=slate))
        clicked = next(m for m in slate if m % 2)
        events.append(RecommendationClickedEvent(timestamp=t, user_id=1, recommendation_id=rec, movie_id=clicked))

    def fit():
        from ml_system.redis import Redis
        ranker = MLPRanker(lambda u: store.get_user_features(u), Redis(), store, GENRES.get, seed=5)
        return ranker.fit(*MLPRanker.training_rows_from(events, store, GENRES.get), epochs=30)

    a, b = fit(), fit()
    assert a.losses == b.losses
    order = a.rank(U(1), [2, 4, 6, 1, 3, 5])
    assert set(order[:3]) == {1, 3, 5}


def test_flink_publishes_recent_history():
    torch = pytest.importorskip("torch")
    from ml_system.build_recommendation_system import build_recommendation_system
    from world.external_data_loader import load_movielens

    data = load_movielens()
    system = build_recommendation_system(data["movie_objects"], data["user_objects"], data["user_states"],
                                         data["movie_states"], seed=2, users_per_tick=5, log_metrics=False)
    system["sim"].get_service("Flink").track_recent = True
    system["sim"].run(40, verbose=False)
    redis = system["redis"]
    slates = [k for k in redis.store if k[0] == "recent_slates"]
    assert slates and all(len(redis.store[k]) <= RECENT_SLATES for k in slates)
    assert any(k[0] == "recent_watches" for k in redis.store)
