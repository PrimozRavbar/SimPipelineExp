"""Personalisation features: shared genre profiles (watches / rich),
onboarding, profile-based candidates, v3 ranker features."""

import random

import pytest

from ml_system.candidate_sources import ProfileRetrieval, merge_candidates
from ml_system.features import ONBOARDING_WEIGHT, RICH_DECAY, update_genre_counts
from ml_system.offline_feature_store import OfflineFeatureStore
from ml_system.ranking import FEATURES_V2, feature_names, features_v3
from simulation.events import RetrievalRequest


def test_watches_mode_matches_the_original_counting():
    counts = {}
    assert update_genre_counts(counts, "watch", ["Drama", "War"], "watches", 0.3)
    assert counts == {"Drama": 1, "War": 1}
    assert not update_genre_counts(counts, "click", ["Comedy"], "watches")
    assert not update_genre_counts(counts, "rating", ["Comedy"], "watches", 5)


def test_rich_mode_weights_and_decays():
    counts = {"Drama": 1.0}
    update_genre_counts(counts, "watch", ["Comedy"], "rich", 0.5)
    assert counts == pytest.approx({"Drama": RICH_DECAY, "Comedy": 0.5})
    update_genre_counts(counts, "rating", ["Comedy"], "rich", 1)      # 1 star: -1, floored at 0
    assert counts["Comedy"] == 0.0
    update_genre_counts(counts, "onboarding", ["Horror"], "rich")
    assert counts["Horror"] == ONBOARDING_WEIGHT


def test_merge_candidates_round_robin_without_duplicates():
    assert merge_candidates([1, 2, 3, 4], [[9, 2, 8]], 5) == [1, 9, 2, 3, 8]
    assert merge_candidates([1, 2], [[]], 5) == [1, 2]


class Movie:
    def __init__(self, genres):
        self.genres = genres


def test_profile_retrieval_ranks_by_taste_then_popularity():
    movies = {1: Movie(["Drama"]), 2: Movie(["Comedy"]), 3: Movie(["Drama"]), 4: Movie(["Drama", "Comedy"])}
    store = OfflineFeatureStore()
    for m, pop in {1: 1, 2: 50, 3: 9, 4: 0}.items():
        store.write_item_features(m, {"popularity": pop})
    profiles = {7: {"genre_preferences": {"Drama": 1.0}}, 8: {}}
    source = ProfileRetrieval(lambda u: profiles.get(u, {}), store, movies)
    assert source.retrieve(RetrievalRequest(user_id=7, k=3)) == [3, 1, 4]   # Drama, popular first
    assert source.retrieve(RetrievalRequest(user_id=8, k=3)) == []          # no profile, no candidates


def test_v3_features():
    row = features_v3({"genre_preferences": {"Drama": 3, "Comedy": 1}}, [], [],
                      {"genres": ["Drama"], "popularity": 2, "average_rating": 4}, 5, lambda m: ["Drama"])
    names = feature_names("v3")
    assert len(row) == len(names) and names[:len(FEATURES_V2)] == list(FEATURES_V2)
    named = dict(zip(names, row))
    assert named["genre_Drama"] == pytest.approx(0.75) and named["genre_Comedy"] == 0.0
    assert named["profile_cosine"] == pytest.approx(0.75 / (0.75 ** 2 + 0.25 ** 2) ** 0.5)


torch = pytest.importorskip("torch")

from ml_system.build_recommendation_system import build_recommendation_system  # noqa: E402
from world.external_data_loader import load_movielens  # noqa: E402


def build(**kwargs):
    data = load_movielens()
    return build_recommendation_system(data["movie_objects"], data["user_objects"], data["user_states"],
                                       data["movie_states"], seed=4, users_per_tick=5, log_metrics=False, **kwargs)


def test_onboarding_publishes_declared_genres_online_and_offline():
    system = build(onboarding={"genres": 3, "noise": 0.0})
    flink = system["sim"].get_service("Flink")
    flink.track_onboarding = True
    system["sim"].run(5, verbose=False)
    events = [e for e in system["datalake"].events if type(e).__name__ == "OnboardingEvent"]
    assert len(events) == len({e.user_id for e in events}) == len(system["world"].onboarded) > 0
    # With no noise the declared genres are the user's true top genres.
    from world.relevance import genre_share
    e = events[0]
    share = genre_share(system["world"].user_states[e.user_id])
    assert set(e.genres) <= set(share) and len(e.genres) == 3
    # Online profile (Redis) and offline profile (Spark) both see them.
    assert set(system["redis"].get("genre_preferences", e.user_id)) >= set(e.genres)
    system["spark"].run_offline_feature_job(system["datalake"].events, system["offline_pipeline"].feature_pipeline)
    assert set(system["feature_store"].get_user_features(e.user_id)["genre_preferences"]) >= set(e.genres)


def test_onboarding_is_seeded_and_off_by_default():
    a, b = build(onboarding={"genres": 3, "noise": 0.5}), build(onboarding={"genres": 3, "noise": 0.5})
    for s in (a, b):
        s["sim"].run(5, verbose=False)
    picks = lambda s: [(e.user_id, tuple(e.genres)) for e in s["datalake"].events if type(e).__name__ == "OnboardingEvent"]
    assert picks(a) == picks(b)
    off = build()
    off["sim"].run(5, verbose=False)
    assert not picks(off)


def test_rich_profiles_online_match_offline():
    system = build()
    system["sim"].get_service("Flink").profile_mode = "rich"
    system["offline_pipeline"].feature_pipeline.profile_mode = "rich"
    system["sim"].run(150, verbose=False)
    system["spark"].run_offline_feature_job(system["datalake"].events, system["offline_pipeline"].feature_pipeline)
    redis, store = system["redis"], system["feature_store"]
    users = [u for (f, u) in redis.store if f == "genre_preferences"]
    assert users
    for u in users:
        online = redis.get("genre_preferences", u)
        offline = store.get_user_features(u)["genre_preferences"]
        assert online == pytest.approx(offline)
