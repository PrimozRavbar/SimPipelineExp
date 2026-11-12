"""Fault injection (ml_system/faults.py) and the console's health rules and
repair actions (console/session.py, console/health.py)."""

import pytest

pytest.importorskip("torch")

from console.session import SimSession
from ml_system.build_recommendation_system import build_recommendation_system
from ml_system.faults import FaultInjector, RemappedIndex, rotate_genres
from world.external_data_loader import load_movielens


def build(seed=4):
    data = load_movielens()
    system = build_recommendation_system(
        data["movie_objects"], data["user_objects"], data["user_states"], data["movie_states"],
        seed=seed, users_per_tick=3, log_metrics=False)
    injector = FaultInjector(system)
    system["sim"].register(injector)
    return system, injector


def test_faults_start_on_schedule_and_undo_cleanly():
    system, faults = build()
    sim, flink, redis = system["sim"], system["sim"].get_service("Flink"), system["redis"]
    original_index = system["retriever"].ann_index

    faults.schedule("flink_outage", start=5, duration=10)
    faults.schedule("feature_corruption", start=5, duration=10)
    faults.schedule("index_id_mismatch", start=5, duration=10)
    faults.schedule("model_server_outage", start=5, duration=10)
    faults.schedule("traffic_spike", start=5, duration=10)

    sim.run(3, verbose=False)
    assert not flink.paused and not faults.active()
    sim.run(1, verbose=False)                      # end of tick 4: switched on for tick 5
    assert flink.paused and "genre_preferences" in redis.read_transforms
    assert isinstance(system["retriever"].ann_index, RemappedIndex)
    assert system["model_server"].retriever_down and system["world"].users_per_tick == 9

    sim.run(10, verbose=False)                     # ends after tick 14
    assert not faults.active()
    assert not flink.paused and not redis.read_transforms
    assert system["retriever"].ann_index is original_index
    assert not system["model_server"].retriever_down and system["world"].users_per_tick == 3
    assert system["model_server"].fallback_requests == 10 * 9


def test_flink_outage_builds_lag_and_catches_up():
    system, faults = build()
    kafka = system["sim"].get_service("Kafka")
    faults.schedule("flink_outage", start=1, duration=None)
    system["sim"].run(60, verbose=False)
    lag = kafka.lag("Flink")
    assert lag > 0
    faults.fix("restart_flink", now=60)
    system["sim"].run(1, verbose=False)
    assert kafka.lag("Flink") == 0


def test_flink_capacity_limits_processing():
    system, _ = build()
    flink = system["sim"].get_service("Flink")
    flink.max_events_per_tick = 1
    system["sim"].run(80, verbose=False)
    assert flink.processed_last_tick <= 1
    assert system["sim"].get_service("Kafka").lag("Flink") > 0


def test_rotate_genres_moves_weight():
    out = rotate_genres({"Action": 0.7, "Drama": 0.3}, shift=1)
    assert "Action" not in out and abs(sum(out.values()) - 1.0) < 1e-9


def chaos_run(seed):
    s = SimSession(preset="plain", seed=seed, users_per_tick=2, log_metrics=False, settings={"chaos_every": 40})
    s.step(200)
    s.close()
    return [(f.kind, f.start, f.status) for f in s.faults.faults]


def test_chaos_schedule_is_seeded():
    first, second = chaos_run(9), chaos_run(9)
    assert first == second and len(first) >= 3


def test_index_fault_opens_incident_and_rebuild_fixes_it():
    s = SimSession(preset="plain", seed=4, users_per_tick=3, log_metrics=False, settings={"spark_every": 0})
    s.step(20)
    s.submit("fault", kind="index_id_mismatch", duration=None)
    s.step(10)                                     # self-test runs at tick 25
    incidents = s.incidents_view()
    assert any(i["rule"] == "index_health" and i["status"] == "Open" for i in incidents)
    assert s.faults_view()["faults"][0]["drill"] is True

    s.submit("rebuild_index")
    s.step(25)                                     # next self-test at tick 50
    assert s.retriever.ann_index is s.versions[0].index
    assert not s.faults.active()
    assert all(i["status"] != "Open" for i in s.incidents_view() if i["rule"] == "index_health")
    s.close()


def test_outage_opens_fallback_incident_and_restart_resolves_it():
    s = SimSession(preset="plain", seed=4, users_per_tick=3, log_metrics=False, settings={"spark_every": 0})
    s.step(5)
    s.submit("fault", kind="model_server_outage", duration=None)
    s.step(11)                                     # the share is over the last 10 ticks
    assert s.health["fallback_share"] == 1.0
    assert any(i["rule"] == "fallback" and i["status"] == "Open" for i in s.incidents_view())
    s.submit("restart_model_server")
    s.step(12)
    assert s.health["fallback_share"] == 0.0
    assert all(i["status"] == "Resolved" for i in s.incidents_view() if i["rule"] == "fallback")
    s.close()


def test_healthy_retrained_indexes_never_fail_the_self_test():
    # Regression: the old self-test compared against exact search and
    # scored healthy trained indexes as low as ~75% (alert at 70%), so a
    # plain deploy could open "ANN index self-test failing".
    from console import health
    from ml_system.faults import RemappedIndex

    s = SimSession(preset="plain", seed=42, users_per_tick=5, log_metrics=False, world="dynamic",
                   settings={"deploy_delay": 10, "epochs": 1, "ranker": True})
    movies = s.system["world"].movie_objects
    for _ in range(3):
        s.step(150)
        s.submit("retrain")
        s.step(40)
        assert s.health["index_self_test"] == 1.0
        golden = s._golden()
        ids = sorted(movies)
        broken = RemappedIndex(s.retriever.ann_index, {m: ids[(i + 7) % len(ids)] for i, m in enumerate(ids)})
        assert health.index_self_test(broken, golden) < 0.2
    assert s.serving_version == 3
    assert not any(i["rule"] == "index_health" for i in s.incidents_view())
    s.close()
