"""Spark in the console: jobs on the simulated cluster, the data-lake sensor,
Spark faults and their incidents, viral movies in the World."""

import pytest

torch = pytest.importorskip("torch")

from console.session import SimSession


FAST = {"deploy_delay": 10, "epochs": 1, "spark_every": 0}


def make(seed=3, world="static", **settings):
    return SimSession(preset="plain", seed=seed, users_per_tick=3, world=world,
                      settings={**FAST, **settings}, log_metrics=False)


def test_a_healthy_job_is_ready_at_once_and_billed():
    s = make(spark_every=20)
    s.step(21)
    assert s.spark_runs == 1 and s.last_spark_tick == 20 and s.spark_job is None
    view = s.pipeline()["spark"]
    job = view["last_job"]
    assert job["status"] == "succeeded" and job["ticks"] == 0 and job["reason"] == "first run"
    assert [st["name"] for st in job["stages"]] == ["scan lake", "user features", "item features"]
    assert s.economy.last_tick["breakdown"]["Spark"] > 0       # billed in the tick it finished
    s.step(1)
    assert s.economy.last_tick["breakdown"]["Spark"] == 0
    s.step(19)
    assert s.system["spark"].history[-1]["reason"] == "incremental"
    s.close()


def test_stuck_lake_makes_the_job_wait_then_fail_and_restart_fixes_it():
    s = make(spark_sensor_timeout=5)
    s.step(10)
    s.submit("fault", kind="lake_ingestion_lag", duration=None)
    s.step(4)
    assert s.system["datalake"].paused and s.kafka.lag("DataLake") > 0
    s.submit("spark")
    s.step(1)
    assert s.spark_job["status"] == "waiting"
    s.step(6)
    assert s.spark_job is None and s.spark_runs == 0
    assert s.last_spark_error["reason"] == "sensor_timeout"
    rules = {i["rule"] for i in s.incidents_view() if i["status"] == "Open"}
    assert {"spark_job", "lake_ingestion"} <= rules

    s.submit("restart_lake_sink")
    s.step(1)
    assert not s.system["datalake"].paused and s.kafka.lag("DataLake") == 0
    s.submit("spark")
    s.step(2)
    assert s.spark_runs == 1 and s.last_spark_error is None
    s.step(2)
    rules = {i["rule"] for i in s.incidents_view() if i["status"] == "Open"}
    assert "spark_job" not in rules and "lake_ingestion" not in rules
    s.close()


def test_without_the_sensor_a_job_runs_on_incomplete_data():
    s = make(spark_wait_for_data=False)
    s.step(10)
    s.submit("fault", kind="lake_ingestion_lag", duration=None)
    s.step(5)
    s.submit("spark")
    s.step(1)
    assert s.spark_runs == 1                                   # "succeeded", silently stale
    assert s.system["spark"].last_run.batch.size < len(s.system["datalake"].events) + s.kafka.lag("DataLake")
    s.close()


def test_retrain_waits_for_the_lake_and_trains_on_complete_data():
    s = make()
    s.step(20)
    s.submit("fault", kind="lake_ingestion_lag", duration=4)
    s.step(1)
    s.submit("retrain")
    s.step(1)
    assert s.retrain["stage"] == "features" and s.spark_job["status"] == "waiting"
    s.step(5)
    assert s.retrain["stage"] == "training"
    run = s.system["spark"].last_run
    assert len(run.batch.events) == len(s.system["datalake"].events) or run.batch.events[-1].timestamp >= 24
    s.close()


def test_fragile_cluster_fails_under_spot_loss_and_retrains_abort():
    s = make(spark_executors=1, spark_executor_cores=1)
    s.step(20)
    s.submit("fault", kind="spark_executor_loss", duration=None)
    s.step(1)
    s.system["spark"].conditions.executor_loss_mean_s = 0.3      # a spot storm: executors last ~0.3 s
    s.submit("retrain")
    s.step(3)
    assert s.retrain is None and s.serving_version == 0
    assert s.last_spark_error is not None
    assert any(i["rule"] == "spark_job" for i in s.incidents_view())
    s.close()


def test_slow_node_drill_and_replacement():
    s = make(spark_speculation=False, spark_executors=1, spark_executor_cores=1)
    s.step(10)
    s.submit("fault", kind="spark_slow_node", duration=None)
    s.step(1)
    s.submit("spark")
    s.step(1)
    slow = s.system["spark"].last_run.report
    assert slow.executors[0]["slow"] == 10
    s.submit("replace_spark_node")
    s.step(1)
    s.submit("spark")
    s.step(1)
    fast = s.system["spark"].last_run.report
    assert fast.executors[0]["slow"] == 1
    assert not s.faults.active("spark_slow_node")
    s.close()


def test_viral_movie_is_world_side_and_creates_a_hot_key():
    s = make(world="dynamic")
    s.step(5)
    s.submit("fault", kind="viral_movie", duration=60)
    s.step(61)
    fault = s.faults.faults[0]
    movie = fault.params["movie_id"]
    watches = [e for e in s.system["datalake"].events
               if type(e).__name__ == "WatchEvent" and e.movie_id == movie and e.timestamp > 6]
    assert len(watches) > 10                                   # people heard about it and watched
    assert s.dynamics.viral is None and s.simulator.buzz == {}   # over after the fault
    s.close()


def test_scheduled_viral_movies_are_off_by_default_and_seeded():
    s = make()
    assert s.settings["viral_every"] == 0
    s.close()
    runs = []
    for _ in range(2):
        s = make(viral_every=20, viral_duration=10)
        s.step(45)
        runs.append(list(s.dynamics.viral_log))
        s.close()
    assert runs[0] == runs[1] and len(runs[0]) == 2
