"""SimSession (console/session.py): determinism, actions at tick boundaries,
background retrain with a fixed deploy tick, rollback, the CTR-drop alert."""

import time

import pytest

torch = pytest.importorskip("torch")

from console.session import ActionError, SimSession, ctr_drop_alert


FAST = {"deploy_delay": 10, "epochs": 1, "spark_every": 0}


def make(seed=3, **settings):
    return SimSession(seed=seed, users_per_tick=3, settings={**FAST, **settings}, log_metrics=False)


def signature(session):
    return [
        (type(e).__name__, e.timestamp, e.user_id, getattr(e, "movie_id", None),
         tuple(getattr(e, "movie_ids", ()) or ()))
        for e in session.system["datalake"].events
    ]


def scripted(seed=3):
    """Same actions at the same ticks: retrain at 30, roll back at 45."""
    s = make(seed)
    s.step(30)
    s.submit("retrain")
    s.step(15)                       # deploy lands at tick 40
    s.submit("rollback")
    s.step(5)
    s.close()
    return s


def test_same_seed_and_actions_give_same_run():
    first, second = scripted(), scripted()

    assert signature(first) == signature(second)
    assert first.deploy_history == second.deploy_history == [(0, 0), (40, 1), (45, 0)]
    # Latency is wall-clock time, so compare everything else.
    strip = lambda rows: [{k: v for k, v in r.items() if not k.startswith("latency")} for r in rows]
    assert strip(first.metrics()["rows"]) == strip(second.metrics()["rows"])

    # Same trained weights.
    w1 = first.versions[1].user_tower.state_dict()
    w2 = second.versions[1].user_tower.state_dict()
    assert all(torch.equal(w1[k], w2[k]) for k in w1)


def test_actions_apply_only_at_the_next_tick_boundary():
    s = make()
    s.step(20)
    record = s.submit("settings", slate_size=5)

    assert record["status"] == "pending"
    assert s.model_server.slate_size == 10          # not applied yet

    s.step(1)
    assert record["status"] == "applied" and record["tick"] == 20
    assert s.model_server.slate_size == 5
    shown = [e for e in s.system["datalake"].events if type(e).__name__ == "RecommendationShownEvent"]
    assert all(len(e.movie_ids) == 5 for e in shown if e.timestamp == 21)
    s.close()


def test_retrain_serves_old_model_until_the_fixed_deploy_tick():
    s = make()
    s.step(30)
    s.submit("retrain")
    s.step(1)
    assert s.retrain is not None and s.retrain["deploy_tick"] == 40

    s.step(9)                                        # ticks 32..40 still on v0
    assert s.serving_version == 0
    s.step(1)                                        # boundary at 40 deploys v1
    assert s.serving_version == 1 and s.index_version == 1
    assert s.tick_versions[40] == 0 and s.tick_versions[41] == 1

    models = s.models()
    assert models["rollback_target"] == 0
    assert models["versions"][0]["training_examples"] > 0

    # The retrain ran the Spark feature job as its first stage.
    assert s.spark_runs == 1 and s.last_spark_tick == 30

    # A second retrain while one is in flight is rejected, not queued.
    s.submit("retrain")
    s.step(1)
    second = s.submit("retrain")
    s.step(1)
    assert second["status"] == "rejected"
    s.close()


def test_rollback_restores_previous_towers_and_index():
    s = make()
    s.step(30)
    s.submit("retrain")
    s.step(12)
    bootstrap = s.versions[0]
    s.submit("rollback")
    s.step(1)
    assert s.serving_version == 0
    assert s.retriever.user_tower is bootstrap.user_tower
    assert s.retriever.ann_index is bootstrap.index
    s.close()


def test_scheduled_spark_and_settings_validation():
    s = make(spark_every=10)
    s.step(21)
    assert s.spark_runs == 2 and s.last_spark_tick == 20

    with pytest.raises(ActionError):
        s.submit("settings", slate_size=500)         # > candidates
    with pytest.raises(ActionError):
        s.submit("settings", nope=1)
    with pytest.raises(ActionError):
        s.submit("launch_rockets")
    s.close()


def test_ctr_drop_rule():
    assert ctr_drop_alert(0.30, 0.20, 20)
    assert not ctr_drop_alert(0.30, 0.25, 20)
    assert not ctr_drop_alert(None, 0.1, 20)
    assert not ctr_drop_alert(0.0, 0.0, 20)


def test_alert_opens_an_incident_after_a_bad_deploy():
    s = make(alert_window=5, alert_ticks=50)
    s.step(10)
    # Fake a deploy watch with a high baseline: the next full window of
    # real traffic is far below it.
    s._watch = {"tick": s.tick, "version": 0, "previous": 0, "baseline": 0.99, "until": s.tick + 50}
    s.step(6)
    incidents = s.incidents_view()
    assert len(incidents) == 1 and incidents[0]["status"] == "Open"
    assert incidents[0]["runbook"]
    s.close()


def test_background_clock_runs_and_pauses():
    s = make()
    s.start()
    s.set_speed(100)
    s.play()
    deadline = time.monotonic() + 30
    while s.tick < 5 and time.monotonic() < deadline:
        time.sleep(0.05)
    s.pause()
    time.sleep(0.2)
    paused_at = s.tick
    time.sleep(0.3)
    assert paused_at >= 5 and s.tick == paused_at
    s.close()


def test_ranker_setting_trains_and_serves_a_ranker():
    s = make(ranker=True, filter_watched=True)
    s.step(30)
    s.submit("retrain")
    s.step(12)
    assert s.serving_version == 1
    assert s.model_server.ranker is s.versions[1].ranker is not None
    assert s.models()["versions"][0]["ranker_info"]["training_rows"] > 0
    assert s.model_server.reranker is not None
    s.submit("settings", ranker=False)
    s.step(1)
    assert s.model_server.ranker is None
    s.close()


def test_slow_spark_job_delays_training_and_failures_abort_it():
    s = make(spark_throughput=50)
    s.step(30)
    s.submit("retrain")
    s.step(1)
    job = s.spark_job
    assert job is not None and s.retrain["stage"] == "features"
    events = len(job["events"])
    assert job["done_tick"] == 30 + -(-events // 50)
    s.step(job["done_tick"] - s.tick + 1)
    assert s.spark_job is None and s.retrain["stage"] == "training"
    assert s.last_spark_tick == job["done_tick"]
    s.step(s.retrain["deploy_tick"] - s.tick + 1)
    assert s.serving_version == 1
    s.close()

    s = make(spark_throughput=50)
    s.step(30)
    s.submit("fault", kind="spark_failure", duration=200)
    s.submit("retrain")
    s.step(40)
    assert s.retrain is None and s.serving_version == 0 and s.last_spark_tick is None
    assert any("FAILED" in e["text"] for e in s.events())
    s.close()
