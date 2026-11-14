"""The retrain workflow in the console: steps, deploy timing, and the faults
that break it (GPU out of memory, a broken feature job, the scheduler down,
GPU preemption)."""

import pytest

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
    assert s.workflow_runs[-1]["status"] == "succeeded" and s.serving_version == 1   # the failed run registered nothing
    s.step(1)
    assert all(i["status"] != "Open" for i in s.incidents_view() if i["rule"] == "workflow")
    s.close()


def test_a_broken_feature_job_is_caught_by_data_validation_and_fixed_by_a_backfill():
    s = make(preset="best", users=10, spark_every=20)    # scheduled jobs run while the job is broken
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
    s = make(preset="plain", users=10, epochs=5, train_batch_size=1)    # a long job: batch 1 wastes the GPU
    s.step(150)
    s.submit("fault", kind="gpu_preemption", duration=None)
    s.step(1)
    s.system["training_conditions"].preemption_mean_s = 60        # a preemption storm
    s.submit("retrain")
    s.step(1)
    report = s.retrain["train_report"].summary() if s.retrain else s.workflow_runs[-1]["training"]
    assert report["preemptions"] > 0
    assert any("preempted" in text for _, text in report["log"])
    s.close()


def test_a_bad_learning_rate_is_caught_by_model_validation():
    # Measured: lr 0.1 ends at loss ~3.6 vs ~2.7 for the default 0.001; the
    # check compares the new and the serving model on the same new examples.
    s = make(preset="best", users=10, epochs=5)
    s.step(150)
    s.submit("retrain")
    s.step(12)
    s.submit("retrain")
    s.step(12)
    assert [r["status"] for r in s.workflow_runs] == ["succeeded", "succeeded"]
    s.submit("settings", learning_rate="0.1")
    s.submit("retrain")
    s.step(12)
    run = s.workflow_runs[-1]
    assert run["status"] == "failed" and run["failed_step"] == "validate model"
    assert s.serving_version == 2 and s.versions[3].info.get("rejected")
    s.close()
