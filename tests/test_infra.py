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


# -- console: operating the components, money, swapping the ANN index ---------------

def _session(**settings):
    from console.session import SimSession
    return SimSession(preset="plain", seed=3, users_per_tick=5, log_metrics=False,
                      settings={"deploy_delay": 10, "epochs": 1, "spark_every": 0, **settings})


def test_console_swaps_the_ann_index_and_charges_the_migration():
    pytest.importorskip("torch")
    from ml_system.ann_indexes import ExactIndex, IVFIndex

    s = _session()
    s.step(30)
    cash = s.economy.cash
    s.submit("swap", component="ann_index", to="ivf")
    s.step(1)
    assert isinstance(s.retriever.ann_index, IVFIndex) and s.index_kind == "ivf"
    assert s.economy.ledger[-1][1] == 500 and s.economy.cash < cash - 500 + 50
    s.submit("retrain")
    s.step(12)
    assert isinstance(s.versions[1].index, IVFIndex)              # new models use the new type
    s.submit("swap", component="ann_index", to="exact")
    s.step(1)
    assert isinstance(s.retriever.ann_index, ExactIndex)
    with pytest.raises(Exception):
        s.submit("swap", component="kafka", to="pulsar")
    s.submit("swap", component="ann_index", to="exact")          # already in use: rejected at the boundary
    s.step(1)
    assert any("already" in e["text"] for e in s.events())
    s.close()


def test_console_infra_settings_reach_the_components_and_cost_money():
    pytest.importorskip("torch")
    s = _session()
    s.step(5)
    base = s.economy.last_tick["cost"]
    s.submit("settings", kafka_brokers=5, kafka_replication=2, flink_parallelism=3, redis_max_keys=500,
             redis_eviction="volatile-ttl", redis_ttl_ticks=50, feast_mode="materialized",
             feast_materialize_every=10, kserve_max_replicas=3, kserve_replica_capacity=2)
    s.step(20)
    assert s.kafka.brokers == 5 and s.kafka.replication_factor == 2
    assert s.flink.parallelism == 3 and s.system["redis"].max_keys == 500
    assert len(s.system["redis"].store) <= 500 + 2               # + pinned shared objects
    assert s.feast.mode == "materialized" and s.feast.materializations >= 2
    assert s.kserve.max_replicas == 3 and s.kserve.ready >= 2    # scaled up for 5 requests a tick
    assert s.economy.last_tick["cost"] > base
    assert any("Flink rescale to 3" in r for _, _, r in s.economy.ledger)
    with pytest.raises(Exception):
        s.submit("settings", kafka_replication=9)               # more copies than brokers
    with pytest.raises(Exception):
        s.submit("settings", redis_eviction="random")
    s.close()


def test_console_broker_drill_stalls_unreplicated_partitions_and_the_repair_ends_it():
    pytest.importorskip("torch")
    s = _session()
    s.step(5)
    s.submit("fault", kind="kafka_broker_down", duration=200)
    s.step(5)
    assert s.kafka.unavailable_partitions() == [0]
    assert any(i["rule"] == "kafka_partitions" for i in s.incidents_view())
    assert s.kafka.lag("Flink") > 0
    s.submit("restart_kafka_broker")
    s.step(2)
    assert s.kafka.unavailable_partitions() == [] and s.kafka.lag("Flink") == 0
    s.close()


def test_revenue_counts_published_clicks_even_when_metrics_stall():
    # Regression: revenue was read from the CTR tracker, a Kafka consumer,
    # so a broker outage that stalled it also stopped all revenue.
    pytest.importorskip("torch")
    s = _session()
    s.submit("fault", kind="kafka_broker_down", duration=200)
    s.step(40)
    clicks = s.kafka.published.get("RecommendationClickedEvent", 0)
    assert clicks > 0 and s.economy.revenue == clicks * 1.0
    s.close()
