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
