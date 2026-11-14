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


def test_model_validation_compares_with_the_last_good_model():
    good = {m: [0.1 * (m % 7), 0.2, -0.1 * (m % 3)] for m in range(50)}
    assert V.passed(V.validate_model([3.0, 2.7], good, reference_loss=2.67))
    # Measured: learning rate 0.1 instead of 0.001 ends at ~3.63 instead of ~2.67.
    failed = {c["name"] for c in V.validate_model([3.66, 3.63], good, reference_loss=2.67) if not c["ok"]}
    assert failed == {"loss vs last good model"}
