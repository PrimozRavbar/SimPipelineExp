"""Infrastructure added for the realism work: point-in-time lookups,
feature-history reset, Kafka lag and bounded polls, Flink pause/capacity,
Redis read transforms. None of it changes default behaviour (see
test_defaults_unchanged.py)."""

from ml_system.kafka import Kafka
from ml_system.offline_feature_store import OfflineFeatureStore
from ml_system.redis import Redis


def test_as_of_matches_a_linear_scan():
    store = OfflineFeatureStore()
    for t in range(1, 30):
        for order in range(3):
            store.write_user_features(7, {"v": (t, order)}, timestamp=t, event_order=order)

    for t in range(0, 32):
        for order in (-1, 0, 1, 2, 5):
            expected = None
            for key, feats in store.user_feature_history[7]:
                if key <= (t, order):
                    expected = feats
            assert store.get_user_features_as_of(7, t, order) == expected

    assert store.get_user_features_as_of(99, 5, 0) is None


def test_out_of_order_history_falls_back_to_a_scan():
    store = OfflineFeatureStore()
    store.write_item_features(1, {"v": "late"}, timestamp=10, event_order=0)
    store.write_item_features(1, {"v": "early"}, timestamp=2, event_order=0)
    assert store.get_item_features_as_of(1, 5, 0) == {"v": "early"}
    assert store.get_item_features_as_of(1, 11, 0) == {"v": "early"}   # last written wins, as before


def test_clear_history_keeps_latest_features():
    store = OfflineFeatureStore()
    store.write_user_features(1, {"a": 1}, timestamp=1, event_order=0)
    store.clear_history()
    assert store.get_user_features(1) == {"a": 1}
    assert store.get_user_features_as_of(1, 5, 0) is None


def test_kafka_bounded_poll_and_lag():
    kafka = Kafka()
    kafka.subscribe("c", "A")
    for i in range(5):
        kafka.topics.setdefault("A", []).append(i)
    assert kafka.lag("c") == 5
    assert kafka.poll("c", "A", max_records=2) == [0, 1]
    assert kafka.lag("c") == 3
    assert kafka.poll("c", "A") == [2, 3, 4]
    assert kafka.lag("c") == 0


def test_redis_read_transform():
    redis = Redis()
    redis.set("x", 1, 10)
    redis.read_transforms["x"] = lambda v: -v
    assert redis.get("x", 1) == -10
    assert redis.store[("x", 1)] == 10
