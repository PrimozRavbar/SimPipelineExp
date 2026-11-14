"""The retrain workflow: a simulated GPU training job, data and model
validation, workflow steps, and the faults that break them."""

import math
import random

import pytest

from ml_system import training_job as T
from ml_system import validation as V


# -- the training job ----------------------------------------------------------

def job(examples=10000, cluster=None, conditions=None, rng=None, epochs=5, negatives=5):
    return T.simulate_training(examples, epochs, negatives, cluster or T.TrainingCluster(), conditions, rng,
                               items=1682, extra_rows=50000)


def test_default_job_succeeds_and_scales():
    base = job()
    assert base.succeeded and base.preemptions == 0
    assert [p[0] for p in base.phases] == ["start-up", "load data", "train", "index + rankers", "save"]
    assert job(examples=40000).duration_s > base.duration_s
    assert job(cluster=T.TrainingCluster(gpus=4)).compute_s < base.compute_s / 3
    assert job(cluster=T.TrainingCluster(batch_size=256)).compute_s < base.compute_s / 2


def test_gpu_out_of_memory_is_deterministic():
    big = T.TrainingCluster(batch_size=4096, gpu_memory_gb=16)
    report = job(cluster=big)
    assert not report.succeeded and report.reason == "gpu_oom" and "out of memory" in report.error
    assert report.duration_s < 300                      # fails right after start-up
    assert job(cluster=T.TrainingCluster(batch_size=4096, gpu_memory_gb=192)).succeeded


def test_preemption_resumes_from_checkpoints():
    preempt = T.TrainingConditions(preemption_mean_s=900)
    with_ckpt = job(examples=60000, conditions=preempt, rng=random.Random("p"),
                    cluster=T.TrainingCluster(max_restarts=50))
    again = job(examples=60000, conditions=preempt, rng=random.Random("p"),
                cluster=T.TrainingCluster(max_restarts=50))
    assert with_ckpt.summary() == again.summary()       # deterministic
    assert with_ckpt.succeeded and with_ckpt.preemptions > 0
    assert with_ckpt.duration_s > job(examples=60000).duration_s
    no_ckpt = job(examples=60000, conditions=preempt, rng=random.Random("p"),
                  cluster=T.TrainingCluster(checkpoint_minutes=0, max_restarts=50))
    assert no_ckpt.duration_s > with_ckpt.duration_s
    fragile = job(examples=60000, conditions=preempt, rng=random.Random("p"),
                  cluster=T.TrainingCluster(checkpoint_minutes=0, max_restarts=1))
    assert not fragile.succeeded and fragile.reason == "preempted"


# -- validation --------------------------------------------------------------

def stats(examples=1000, click_rate=0.3, empty=5.0, through=100):
    return {"examples": examples, "click_rate": click_rate, "new_clicks": 300, "empty_profile_pct": empty,
            "through_tick": through}


def test_data_validation():
    assert V.passed(V.validate_data(stats()))                       # first run: no reference
    ref = stats()
    assert V.passed(V.validate_data(stats(examples=1500, click_rate=0.45, empty=8.0), ref))
    failures = {c["name"] for c in V.validate_data(stats(examples=400), ref) if not c["ok"]}
    assert failures == {"examples vs last run"}
    failures = {c["name"] for c in V.validate_data(stats(empty=40.0), ref) if not c["ok"]}
    assert failures == {"empty profiles in new data (%)"}
    failures = {c["name"] for c in V.validate_data(stats(click_rate=0.1), ref) if not c["ok"]}
    assert failures == {"click rate"}


def test_model_validation():
    good = {m: [0.1 * (m % 7), 0.2, -0.1 * (m % 3)] for m in range(50)}
    assert V.passed(V.validate_model([3.0, 2.5, 2.2], good))
    names = lambda checks: {c["name"] for c in checks if not c["ok"]}
    assert names(V.validate_model([3.0, float("nan")], good)) == {"loss is a number"}
    assert names(V.validate_model([2.0, 2.4, 3.1], good)) == {"loss did not diverge"}
    assert names(V.validate_model([3.0, 2.0], {m: [0.5, 0.5, 0.5] for m in range(50)})) == {"embeddings not collapsed"}
    bad = dict(good)
    bad[3] = [math.inf, 0.0, 0.0]
    assert names(V.validate_model([3.0, 2.0], bad)) == {"embeddings are finite"}


# -- the workflow in the console ------------------------------------------------

torch = pytest.importorskip("torch")
from console.session import SimSession  # noqa: E402

FAST = {"deploy_delay": 10, "epochs": 1, "spark_every": 0}


def make(preset="plain", seed=3, users=3, **settings):
    return SimSession(preset=preset, seed=seed, users_per_tick=users, settings={**FAST, **settings},
                      log_metrics=False)


def test_a_retrain_runs_every_step_and_deploys_on_time():
    s = make()
    s.step(30)
    s.submit("retrain")
    s.step(1)
    assert s.retrain["deploy_tick"] == 40                  # training is quick: the release window decides
    s.step(14)
    run = s.workflow_runs[-1]
    assert run["status"] == "succeeded" and s.serving_version == 1
    assert [st["name"] for st in run["steps"]] == ["features", "validate data", "train", "validate model", "deploy"]
    assert all(st["status"] == "succeeded" for st in run["steps"])
    assert run["training"]["status"] == "succeeded" and run["train_ticks"] == 0
    view = s.pipeline()["workflow"]
    assert view["last"]["version"] == 1 and view["history"][-1]["status"] == "succeeded"
    s.close()


def test_a_slow_training_job_deploys_late():
    s = make(train_gpu_memory_gb=192, train_batch_size=1, epochs=3)
    s.system["training_conditions"].queue_wait_s = 6 * 3600   # GPUs scarce: 24 ticks of waiting
    s.step(30)
    s.submit("retrain")
    s.step(1)
    assert s.retrain["deploy_tick"] >= 30 + 24
    s.close()


def test_gpu_out_of_memory_fails_the_workflow_and_keeps_the_model():
    s = make(train_batch_size=4096)
    s.step(30)
    s.submit("retrain")
    s.step(15)
    run = s.workflow_runs[-1]
    assert run["status"] == "failed" and run["failed_step"] == "train"
    assert "out of memory" in run["training"]["error"]
    assert s.serving_version == 0 and s.retrain is None
    assert any(i["rule"] == "workflow" and i["status"] == "Open" for i in s.incidents_view())
    assert s.economy.running > 0
    s.submit("settings", train_batch_size=32)
    s.submit("retrain")
    s.step(15)
    assert s.workflow_runs[-1]["status"] == "succeeded" and s.serving_version == 2
    s.step(1)
    assert all(i["status"] != "Open" for i in s.incidents_view() if i["rule"] == "workflow")
    s.close()


def test_a_broken_feature_job_is_caught_by_data_validation_and_fixed_by_a_backfill():
    s = make(preset="best", users=10)
    s.step(60)
    s.submit("retrain")
    s.step(12)
    assert s.workflow_runs[-1]["status"] == "succeeded"
    s.submit("fault", kind="feature_nulls", duration=40)
    s.step(60)                                           # the fault ends; its output stays
    s.submit("retrain")
    s.step(12)
    run = s.workflow_runs[-1]
    assert run["status"] == "failed" and run["failed_step"] == "validate data"
    assert not next(c for c in run["data_checks"]["checks"] if c["name"].startswith("empty profiles"))["ok"]
    assert s.serving_version == 1
    s.submit("backfill_features")
    s.step(2)
    assert s.system["spark"].history[-1]["reason"] == "backfill: requested"
    s.submit("retrain")
    s.step(12)
    assert s.workflow_runs[-1]["status"] == "succeeded"
    s.close()


def test_without_data_validation_the_broken_data_is_trained_on():
    s = make(preset="best", users=10, validate_data=False)
    s.step(60)
    s.submit("retrain")
    s.step(12)
    s.submit("fault", kind="feature_nulls", duration=None)
    s.step(60)
    s.submit("retrain")
    s.step(12)
    run = s.workflow_runs[-1]
    assert run["status"] == "succeeded"
    assert next(st for st in run["steps"] if st["name"] == "validate data")["status"] == "skipped"
    s.close()


def test_scheduler_down_silently_skips_scheduled_runs():
    s = make(spark_every=20, retrain_every=40)
    s.step(5)
    s.submit("fault", kind="scheduler_down", duration=None)
    s.step(50)
    assert s.spark_runs == 0 and len(s.workflow_runs) == 0
    assert {k for _, k in s.missed_runs} == {"spark", "retrain"}
    assert not any("skipped" in e["text"] for e in s.events())     # silent
    assert any(i["rule"] == "schedule_missed" and i["status"] == "Open" for i in s.incidents_view())
    s.submit("restart_scheduler")
    s.step(2)
    assert all(i["status"] != "Open" for i in s.incidents_view() if i["rule"] == "schedule_missed")
    s.step(10)
    assert s.spark_runs >= 1
    s.close()


def test_gpu_preemption_makes_training_slower_but_checkpoints_save_it():
    s = make(preset="plain", users=10, epochs=5)
    s.step(150)
    s.submit("fault", kind="gpu_preemption", duration=None)
    s.system["training_conditions"].preemption_mean_s = 60
    s.submit("retrain")
    s.step(1)
    report = s.retrain["train_report"] if s.retrain else s.workflow_runs[-1]["training"]
    summary = report.summary() if hasattr(report, "summary") else report
    assert summary["preemptions"] > 0
    s.close()
