"""Fixes that let retraining lift online CTR (see CLAUDE.md, "Why CTR
was flat"): a correct HNSW graph, serving features that match training,
and tower input switches."""

import math
import random

import pytest

from ml_system.ann import HNSWIndex, cosine_distance
from ml_system.kafka import Kafka
from ml_system.offline_feature_store import OfflineFeatureStore
from ml_system.redis import Redis


def clustered_embeddings(n=500, dim=16, clusters=12, seed=1):
    rng = random.Random(seed)
    centers = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(clusters)]
    embeddings = {}

    for item_id in range(1, n + 1):
        center = centers[rng.randrange(clusters)]
        vector = [x + rng.gauss(0, 0.35) for x in center]
        norm = math.sqrt(sum(x * x for x in vector))
        embeddings[item_id] = [x / norm for x in vector]

    return embeddings, rng


def test_hnsw_finds_true_nearest_neighbours_on_clustered_data():
    # Regression: picking only the M nearest links built isolated cliques
    # and found ~50% of the true top-10 on clustered embeddings.
    embeddings, rng = clustered_embeddings()
    index = HNSWIndex(seed=0)
    index.build(embeddings)

    recall = 0.0
    queries = 50

    for _ in range(queries):
        query = [rng.gauss(0, 1) for _ in range(16)]
        exact = sorted(embeddings, key=lambda m: cosine_distance(query, embeddings[m]))[:10]
        recall += len(set(exact) & set(index.search(query, 10))) / 10

    assert recall / queries >= 0.9


def test_hnsw_is_deterministic_for_a_seed():
    embeddings, _ = clustered_embeddings(n=200)
    query = embeddings[7]

    first = HNSWIndex(seed=3)
    first.build(embeddings)
    second = HNSWIndex(seed=3)
    second.build(embeddings)

    assert first.search(query, 20) == second.search(query, 20)


# ---------------------------------------------------------------------
# Serving features match training features
# ---------------------------------------------------------------------

torch = pytest.importorskip("torch")

from ml_system.flink import Flink  # noqa: E402
from ml_system.two_tower import (  # noqa: E402
    TwoTowerRetrieval,
    SimpleUserTower,
    SimpleItemTower,
    USER_FEATURE_DEFAULTS,
)


def make_retriever(feature_store=None, redis=None):
    return TwoTowerRetrieval(
        user_tower=None,
        item_tower=None,
        ann_index=None,
        redis=redis or Redis(),
        feature_store=feature_store or OfflineFeatureStore(),
    )


def test_cold_start_user_gets_the_same_defaults_as_training():
    # Training sees an empty genre profile and zero counts for a user with
    # no history; serving used to send a uniform profile over a hard-coded
    # genre list and a 3.5 average rating.
    retriever = make_retriever()

    features = retriever.user_features(5)

    assert features == {"user_id": 5, **USER_FEATURE_DEFAULTS}
    assert features["genre_preferences"] == {}
    # Serving must not write invented values back into the stores.
    assert retriever.feature_store.user_features == {}


def test_redis_wins_over_offline_store_per_feature():
    store = OfflineFeatureStore()
    store.write_user_features(5, {"genre_preferences": {"Drama": 1.0}, "watch_count": 4})
    redis = Redis()
    redis.set("watch_count", 5, 9)

    features = make_retriever(store, redis).user_features(5)

    assert features["genre_preferences"] == {"Drama": 1.0}
    assert features["watch_count"] == 9
    assert features["average_rating"] == 0


def test_flink_publishes_interactions_like_the_offline_pipeline():
    from types import SimpleNamespace
    from simulation.events import RecommendationClickedEvent

    redis = Redis()
    kafka = Kafka()
    services = {"Kafka": kafka, "Redis": redis}
    sim = SimpleNamespace(get_service=services.__getitem__)

    flink = Flink()
    flink.start(sim)
    kafka.receive(RecommendationClickedEvent(timestamp=1, user_id=3, recommendation_id=1, movie_id=10))
    kafka.receive(RecommendationClickedEvent(timestamp=2, user_id=3, recommendation_id=2, movie_id=11))
    flink.process(sim)

    assert redis.get("interactions", 3) == 2
    assert redis.get("activity_level", 3) == 2


def test_tower_switches_drop_inputs():
    torch.manual_seed(0)
    tower = SimpleUserTower(num_users=10, use_id_embedding=False, use_activity_features=False)
    base = {"user_id": 1, "genre_preferences": {"Drama": 1.0}, **{
        k: v for k, v in USER_FEATURE_DEFAULTS.items() if k != "genre_preferences"
    }}
    busy = dict(base, user_id=2, watch_count=50, interactions=80, average_rating=4.5)

    assert tower.embed(base) == tower.embed(busy)

    item = SimpleItemTower(num_movies=10, use_id_embedding=False, use_rating_feature=False)
    movie = {"movie_id": 1, "genres": ["Drama"], "average_rating": 1, "popularity": 3}

    assert item.embed(movie) == item.embed(dict(movie, movie_id=2, average_rating=5))
    assert item.embed(movie) != item.embed(dict(movie, popularity=300))

    # Defaults keep every input, so old checkpoints behave as before.
    full = SimpleUserTower(num_users=10)
    assert full.use_id_embedding and full.use_activity_features


def test_build_uses_configured_towers_for_training_and_explores_at_bootstrap():
    from world.external_data_loader import load_movielens
    from ml_system.build_recommendation_system import build_recommendation_system

    data = load_movielens()
    system = build_recommendation_system(
        data["movie_objects"], data["user_objects"],
        data["user_states"], data["movie_states"], seed=3,
    )

    trainer = system["offline_pipeline"].trainer
    assert not trainer.user_tower.use_activity_features
    assert not trainer.user_tower.use_id_embedding
    assert not trainer.item_tower.use_rating_feature

    # The untrained bootstrap model keeps ID embeddings so different users
    # see different slates (exploration for the first retrain).
    retriever = system["retriever"]
    assert retriever.user_tower.use_id_embedding
    slates = {
        tuple(system["model_server"].recommend(user))
        for user in list(data["user_objects"].values())[:20]
    }
    assert len(slates) > 1
