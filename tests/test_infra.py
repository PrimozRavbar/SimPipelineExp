"""Operational behaviour of the infrastructure components (Kafka, Redis,
Flink, Feast, KServe). Defaults are pass-throughs; see
tests/test_defaults_unchanged.py for the byte-identical check."""

from types import SimpleNamespace

import pytest

from ml_system.feast import Feast
from ml_system.flink import Flink
from ml_system.kafka import Kafka
from ml_system.kserve import KServe
from ml_system.model_server import ModelServer, RetrievalService
from ml_system.offline_feature_store import OfflineFeatureStore
from ml_system.redis import Redis
from simulation.events import RecommendationClickedEvent


def sim_at(tick, **services):
    clock = SimpleNamespace(now=tick)
    return SimpleNamespace(clock=clock, get_service=lambda name: services[name])


def click(tick, user):
    return RecommendationClickedEvent(tick, user, tick * 100 + user, 1)


# -- Kafka -------------------------------------------------------------------------

def test_kafka_partitions_and_a_broker_outage_without_replication():
    kafka = Kafka(partitions=3, brokers=3, replication_factor=1)
    kafka.subscribe("c", "RecommendationClickedEvent")
    for user in (1, 2, 3, 4):                      # partitions 1, 2, 0, 1
        kafka.receive(click(1, user))
    kafka.broker_down(2)                           # leads partition 2 only
    got = kafka.poll("c", "RecommendationClickedEvent")
    assert [e.user_id for e in got] == [1]         # stops at user 2 (partition 2): order is kept
    assert kafka.lag("c") == 3 and kafka.unavailable_partitions() == [2]
    kafka.broker_up(2)
    assert [e.user_id for e in kafka.poll("c", "RecommendationClickedEvent")] == [2, 3, 4]
    d = kafka.describe()
    assert d["topics"]["RecommendationClickedEvent"]["per_partition"] == [1, 2, 1]


def test_kafka_replication_survives_a_broker_outage():
    kafka = Kafka(partitions=3, brokers=3, replication_factor=2)
    kafka.subscribe("c", "RecommendationClickedEvent")
    for user in (1, 2, 3):
        kafka.receive(click(1, user))
    kafka.broker_down(2)
    assert len(kafka.poll("c", "RecommendationClickedEvent")) == 3


def test_kafka_retention_deletes_old_messages_and_slow_consumers_lose_them():
    kafka = Kafka(retention_ticks=5)
    kafka.subscribe("slow", "RecommendationClickedEvent")
    kafka.subscribe("fast", "RecommendationClickedEvent")
    for tick in range(1, 11):
        kafka.receive(click(tick, 1))
        kafka.poll("fast", "RecommendationClickedEvent")
        kafka.process(sim_at(tick))
    assert kafka.log_start["RecommendationClickedEvent"] == 4      # ticks 1-4 deleted at tick 10
    got = kafka.poll("slow", "RecommendationClickedEvent")
    assert [e.timestamp for e in got] == list(range(5, 11))
    assert kafka.lost[("slow", "RecommendationClickedEvent")] == 4
    assert kafka.lag("fast") == 0 and kafka.lag("slow") == 0
    kafka.subscribe("late", "RecommendationClickedEvent", from_tick=8)
    assert [e.timestamp for e in kafka.poll("late", "RecommendationClickedEvent")] == [8, 9, 10]


# -- Redis -------------------------------------------------------------------------

def test_redis_lru_eviction_keeps_recently_used_and_pinned_keys():
    redis = Redis(max_keys=3)
    redis.set("item_signals", "all", object())                # pinned
    redis.set("genre_preferences", 1, {"Drama": 1})
    redis.set("genre_preferences", 2, {"Comedy": 1})
    redis.get("genre_preferences", 1)                          # 1 is now more recent than 2
    redis.set("genre_preferences", 3, {"War": 1})
    assert redis.get("genre_preferences", 2) is None and redis.get("genre_preferences", 1)
    assert redis.get("item_signals", "all") is not None
    assert redis.stats["evictions"] == 1


def test_redis_noeviction_refuses_writes_and_ttl_expires_values():
    redis = Redis(max_keys=1, eviction="noeviction")
    redis.set("a", 1, 1)
    redis.set("a", 2, 2)
    assert redis.get("a", 2) is None and redis.stats["write_errors"] == 1

    redis = Redis(ttl_ticks={"genre_preferences": 10})
    redis.sim = sim_at(0)
    redis.set("genre_preferences", 1, {"Drama": 1})
    redis.sim.clock.now = 10
    assert redis.get("genre_preferences", 1) == {"Drama": 1}
    redis.sim.clock.now = 11
    assert redis.get("genre_preferences", 1) is None and redis.stats["expirations"] == 1


# -- Flink -------------------------------------------------------------------------

def test_flink_capacity_is_parallelism_times_slot_capacity_and_rescale_restarts():
    flink = Flink()
    assert flink.capacity() is None
    flink.slot_capacity = 5
    flink.parallelism = 2
    assert flink.capacity() == 10
    flink.max_events_per_tick = 4
    assert flink.capacity() == 4
    flink.rescale(4, now=10)
    assert flink.parallelism == 4 and flink.restoring_until == 13
    flink.lag_history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert flink.backpressured()


# -- Feast -------------------------------------------------------------------------

def test_feast_materialized_mode_serves_the_last_materialized_copy():
    store = OfflineFeatureStore()
    store.item_features[7] = {"popularity": 1}
    feast = Feast(Redis(), store)
    assert feast.get_item_features(7) == {"popularity": 1}        # direct: live offline store
    feast.mode = "materialized"
    feast.materialize_every = 10
    assert feast.get_item_features(7) is None                      # nothing materialized yet
    feast.process(sim_at(5))
    store.item_features[7] = {"popularity": 9}                     # Spark updates offline
    assert feast.get_item_features(7) == {"popularity": 1}         # serving still sees the copy
    feast.process(sim_at(15))
    assert feast.get_item_features(7) == {"popularity": 9}
    assert feast.materializations == 2 and feast.misses["item"] == 1
    assert feast.get_item_features_as_of == store.get_item_features_as_of   # training path untouched


# -- KServe ------------------------------------------------------------------------

class Fixed(RetrievalService):
    def __init__(self, ids):
        super().__init__()
        self.ids = ids

    def retrieve(self, request):
        return list(self.ids)


def test_kserve_overflow_goes_to_the_fallback_and_the_autoscaler_adds_replicas():
    server = ModelServer(Fixed([1, 2, 3]), num_candidates=3, slate_size=3)
    server.overflow_fallback = Fixed([9, 8, 7])
    kserve = KServe(server, min_replicas=1, max_replicas=4, replica_capacity=2,
                    target_utilization=1.0, cold_start_ticks=2)
    server.admission = kserve.admit
    user = SimpleNamespace(user_id=1)
    slates = [server.recommend(user) for _ in range(5)]
    assert slates[:2] == [[1, 2, 3]] * 2 and slates[2:] == [[9, 8, 7]] * 3
    kserve.process(sim_at(1))
    assert kserve.last["overflow"] == 3 and kserve.ready == 1 and len(kserve.starting) == 2
    kserve.process(sim_at(3))                                      # cold start done
    assert kserve.ready == 3
    assert [server.recommend(user) for _ in range(5)][-1] == [1, 2, 3]
