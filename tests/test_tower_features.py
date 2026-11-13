"""Two-tower inputs for the rich world: engagement (skips), profile
strength, demographics, a movie table shared with the watch history."""

import pytest

from ml_system.demographics import AGE_BRACKETS, demographic_vector
from ml_system.features import engagement_features, new_engagement_state, update_engagement
from world.entities import User


def test_engagement_counts_skips_and_strength():
    state = new_engagement_state()
    update_engagement(state, "shown", [["Drama"], ["Drama", "War"], ["Comedy"]])
    update_engagement(state, "click", ["Comedy"])
    rates, strength = engagement_features(state)
    assert rates["Comedy"] > rates["Drama"] > 0          # shown Drama twice, never clicked
    assert 0 < strength < 1
    assert engagement_features(new_engagement_state()) == ({}, 0.0)


def test_demographic_vector():
    occupations = ["artist", "student"]
    v = demographic_vector(User(user_id=1, age=30, gender="F", occupation="student", zip_code="0"), occupations)
    assert len(v) == len(AGE_BRACKETS) + 1 + 1 + 2
    assert v[2] == 1.0 and sum(v[:len(AGE_BRACKETS) + 1]) == 1.0      # 25-34 bracket
    assert v[len(AGE_BRACKETS) + 1] == 1.0 and v[-1] == 1.0


torch = pytest.importorskip("torch")

from ml_system.demographics import demographics_table  # noqa: E402
from ml_system.training import TwoTowerTrainer  # noqa: E402
from ml_system.two_tower import SimpleItemTower, SimpleUserTower  # noqa: E402


def test_tower_options_and_shared_table():
    users = {1: User(1, 30, "F", "student", "0"), 2: User(2, 60, "M", "artist", "0")}
    item = SimpleItemTower(num_movies=10, use_id_embedding=True, id_init_std=0.01)
    assert item.movie_embedding.weight.std() < 0.05
    user = SimpleUserTower(num_users=2, use_history=True, num_movies=10, use_engagement=True,
                           demographics=demographics_table(users), history_table=item.movie_embedding)
    assert user.history_embedding is item.movie_embedding
    features = {"user_id": 1, "genre_preferences": {"Drama": 1.0}, "recent_watches": [3],
                "genre_engagement": {"Drama": 0.2}, "profile_strength": 0.5}
    assert len(user.embed(features)) == 32
    assert user.embed(features) != user.embed({**features, "user_id": 2})      # demographics differ
    trainer = TwoTowerTrainer(user, item)
    ids = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))                                         # shared table once


def test_online_and_offline_engagement_agree():
    from ml_system.build_recommendation_system import build_recommendation_system
    from world.external_data_loader import load_movielens

    data = load_movielens()
    system = build_recommendation_system(data["movie_objects"], data["user_objects"], data["user_states"],
                                         data["movie_states"], seed=2, users_per_tick=5, log_metrics=False)
    system["sim"].get_service("Flink").track_engagement = True
    pipeline = system["offline_pipeline"].feature_pipeline
    pipeline.engagement = True
    system["sim"].run(60, verbose=False)
    system["spark"].run_offline_feature_job(system["datalake"].events, pipeline)
    redis, store = system["redis"], system["feature_store"]
    users = [u for (f, u) in redis.store if f == "genre_engagement"]
    assert users
    for u in users:
        offline = store.get_user_features(u)
        assert redis.get("genre_engagement", u) == pytest.approx(offline["genre_engagement"])
        assert redis.get("profile_strength", u) == pytest.approx(offline["profile_strength"])


def test_export_round_trip_with_new_inputs(tmp_path):
    from ml_system.ann import HNSWIndex
    from ml_system.model_registry import export_bundle, load_bundle

    users = {1: User(1, 30, "F", "student", "0")}
    item = SimpleItemTower(num_movies=5, use_id_embedding=True, id_init_std=0.01)
    user = SimpleUserTower(num_users=1, use_history=True, num_movies=5, use_engagement=True,
                           demographics=demographics_table(users), history_table=item.movie_embedding)
    index = HNSWIndex(seed=1)
    index.build({m: item.embed({"movie_id": m, "genres": ["Drama"], "popularity": m}) for m in range(1, 6)})
    export_bundle(tmp_path / "b", user, item, index)
    bundle = load_bundle(tmp_path / "b", index_seed=1)
    features = {"user_id": 1, "genre_preferences": {"Drama": 1.0}, "recent_watches": [2]}
    assert bundle["user_tower"].embed(features) == pytest.approx(user.eval().embed(features))
    assert bundle["user_tower"].history_embedding is bundle["item_tower"].movie_embedding


def test_v5_tower_score_feature():
    from ml_system.offline_feature_store import OfflineFeatureStore
    from ml_system.ranking import MLPRanker, feature_names
    from ml_system.redis import Redis
    from simulation.events import RecommendationClickedEvent, RecommendationShownEvent

    torch.manual_seed(0)
    item = SimpleItemTower(num_movies=6)
    user = SimpleUserTower(num_users=2)
    store = OfflineFeatureStore()
    genres = {m: (["Comedy"] if m % 2 else ["Drama"]) for m in range(1, 7)}
    for m in range(1, 7):
        store.write_item_features(m, {"movie_id": m, "genres": genres[m], "popularity": 1},
                                  timestamp=0, event_order=-1)
    store.write_user_features(1, {"genre_preferences": {"Comedy": 1.0}}, timestamp=0, event_order=-1)
    events = []
    for t in range(1, 60):
        events += [RecommendationShownEvent(timestamp=t, user_id=1, recommendation_id=t, movie_ids=[2, 1]),
                   RecommendationClickedEvent(timestamp=t, user_id=1, recommendation_id=t, movie_id=1)]
    inputs = []
    rows, positions, labels = MLPRanker.training_rows_from(events, store, genres.get, "v5", inputs=inputs)
    assert len(rows[0]) == len(feature_names("v5")) and len(inputs) == len(rows)
    embeddings = {m: item.embed(store.get_item_features(m)) for m in range(1, 7)}
    ranker = MLPRanker(lambda u: {**store.get_user_features(u), "user_id": u}, Redis(), store, genres.get,
                       feature_set="v5", seed=1)
    ranker.set_towers(user, embeddings)
    ranker.fill_tower_scores(rows, inputs)
    expected = float(torch.tensor(user.embed({"genre_preferences": {"Comedy": 1.0}, "user_id": 1,
                                              "recent_watches": []})) @ torch.tensor(embeddings[2]))
    assert rows[0][-1] == pytest.approx(expected, abs=1e-5)
    ranker.fit(rows, positions, labels, epochs=5)
    assert sorted(ranker.rank(type("U", (), {"user_id": 1})(), [1, 2, 3, 4])) == [1, 2, 3, 4]


def test_v5_ranker_trains_on_scores_logged_at_serving():
    """The fix for the v5 leak: training rows use the retrieval score that
    was logged when the slate was served, by the model serving then."""
    from console.session import SimSession

    s = SimSession(preset="plain", seed=3, users_per_tick=4, log_metrics=False, settings={
        "deploy_delay": 10, "epochs": 1, "spark_every": 0, "ranker": True, "ranker_mlp": True,
        "ranker_signals": True, "ranker_tower_score": True})
    s.step(40)
    assert len(s.score_log) == 40 * 4 * 10          # every shown movie of every slate
    s.submit("retrain")
    s.step(12)
    info = s.versions[1].info["ranker_info"]
    assert info["feature_set"] == "v5" and info["logged_share"] == 1.0
    assert all(isinstance(v, float) for v in s.score_log.values())
    s.close()
