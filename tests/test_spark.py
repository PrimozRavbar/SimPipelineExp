"""Spark: the partitioned, incremental feature job on a simulated cluster.

The features must never depend on how the job runs (full or incremental,
any partitioning, any cluster, retries), a failed job must publish nothing,
and the simulated schedule must model the usual failure modes."""

import random

import pytest

from ml_system.datalake import DataLake
from ml_system.offline_feature_store import OfflineFeatureStore
from ml_system.offline_pipeline import OfflineFeaturePipeline
from ml_system.spark import Spark
from ml_system.spark_sim import cluster as C
from ml_system.spark_sim.cluster import ClusterConfig, Conditions
from ml_system.spark_sim.engine import StageSpec, TaskSpec, plan_partitions, simulate, ticks_for
from world.external_data_loader import load_movielens

from spark_helpers import store_digest, synthetic_events

# Digests of the feature store after process(events[:500]); process(events)
# with the implementation before this change (one sequential loop).
LEGACY = {
    ("watches", False): "1443acc7bb2461d4d9db0333f9a3f5dead1b592003821db556f373dc96b57007",
    ("rich", True): "1abe09800ad0dd1371a990b8c83444afe181889561df81141ed004e9ac7c27cf",
}


@pytest.fixture(scope="module")
def movies():
    return load_movielens()["movie_objects"]


@pytest.fixture(scope="module")
def events(movies):
    return synthetic_events(movies)


def pipeline(movies, mode="rich", engagement=True):
    store = OfflineFeatureStore()
    p = OfflineFeaturePipeline(movies, store)
    p.profile_mode, p.engagement = mode, engagement
    return p, store


@pytest.mark.parametrize("mode,engagement", sorted(LEGACY))
def test_full_recompute_matches_the_previous_implementation(movies, events, mode, engagement):
    p, store = pipeline(movies, mode, engagement)
    p.process(events[:500])
    p.process(events)
    assert store_digest(store) == LEGACY[(mode, engagement)]


@pytest.mark.parametrize("mode,engagement", sorted(LEGACY))
def test_incremental_runs_give_the_same_features(movies, events, mode, engagement):
    p, store = pipeline(movies, mode, engagement)
    reasons = [p.process_incremental(events[:cut]).reason for cut in (500, 501, 900, len(events))]
    assert reasons == ["first run", "incremental", "incremental", "incremental"]
    assert store_digest(store) == LEGACY[(mode, engagement)]


@pytest.mark.parametrize("config,conditions", [
    (ClusterConfig(), None),
    (ClusterConfig(aqe=False, shuffle_partitions=7), None),
    (ClusterConfig(executors=1, cores=1, memory_gb=1, shuffle_partitions=1), None),
    (ClusterConfig(), Conditions(executor_loss_mean_s=20)),
    (ClusterConfig(speculation=True), Conditions(slow_executor_factor=10)),
])
def test_spark_jobs_give_the_same_features_on_any_cluster(movies, events, config, conditions):
    p, store = pipeline(movies, "rich", True)
    spark = Spark(seed=3, config=config)
    if conditions:
        spark.conditions = conditions
    spark.run_offline_feature_job(events[:500], p)
    run = spark.run_offline_feature_job(events, p)
    assert run.report.succeeded and run.batch.reason == "incremental"
    assert store_digest(store) == LEGACY[("rich", True)]


def test_profile_change_triggers_a_backfill(movies, events):
    p, store = pipeline(movies, "watches", False)
    spark = Spark()
    spark.run_offline_feature_job(events[:500], p)
    p.profile_mode, p.engagement = "rich", True
    run = spark.run_offline_feature_job(events, p)
    assert run.batch.full and run.batch.reason.startswith("backfill")
    reference, ref_store = pipeline(movies, "rich", True)
    reference.process(events)
    assert store_digest(store) == store_digest(ref_store)


def test_a_failed_job_publishes_nothing(movies, events):
    p, store = pipeline(movies)
    spark = Spark()
    spark.run_offline_feature_job(events[:500], p)
    before = store_digest(store)
    state = (p.processed, p.last_event)
    spark.failing = True
    with pytest.raises(RuntimeError):
        spark.run_offline_feature_job(events, p)
    assert store_digest(store) == before and (p.processed, p.last_event) == state
    # An out-of-memory job fails too, and the next healthy run catches up.
    spark.failing = False
    spark.config = ClusterConfig(executors=1, cores=8, memory_gb=1, aqe=False, shuffle_partitions=1)
    spark.incremental = False
    run = spark.plan_offline_feature_job(events * 20, p)
    assert not run.report.succeeded and run.report.reason == "oom"
    assert store_digest(store) == before
    spark.config, spark.incremental = ClusterConfig(), True
    spark.run_offline_feature_job(events, p)
    assert store_digest(store) == LEGACY[("rich", True)]


# -- the engine ---------------------------------------------------------------

def reduce_job(rows_per_task, config, conditions=None, rng=None):
    stages = [StageSpec(0, "scan", "scan", [TaskSpec(100, files=1) for _ in range(4)]),
              StageSpec(1, "agg", "reduce", [TaskSpec(r) for r in rows_per_task], parents=(0,))]
    return simulate(stages, config, conditions, rng)


def test_a_healthy_default_job_finishes_inside_a_tick():
    report = reduce_job([200] * 50, ClusterConfig())
    assert report.succeeded and ticks_for(report.duration_s) == 0
    assert ticks_for(C.TICK_SECONDS - 1) == 0 and ticks_for(C.TICK_SECONDS) == 1


def test_more_executors_finish_faster():
    small = reduce_job([500] * 64, ClusterConfig(executors=1, cores=1))
    big = reduce_job([500] * 64, ClusterConfig(executors=8, cores=4))
    assert small.duration_s > 5 * big.duration_s


def test_memory_spill_and_out_of_memory():
    config = ClusterConfig(executors=2, cores=4, memory_gb=2)       # ~307 MB, ~1229 rows per task
    spill = reduce_job([2000, 100], config)
    assert spill.succeeded and spill.stages[1]["spill_mb"] > 0
    oom = reduce_job([6000, 100], config)
    assert not oom.succeeded and oom.reason == "oom"
    assert oom.stages[1]["failed_attempts"] == config.max_task_failures
    assert any("out of memory" in text for _, text in oom.log)


def test_aqe_splits_a_skewed_partition_but_never_a_key():
    rows = {k: 5 for k in range(1, 400)}
    rows[7] = 3000                                                  # a hot movie
    rows.update({k: 400 for k in range(1000, 1010)})               # a heavy partition of many keys
    config = ClusterConfig(shuffle_partitions=10)
    plain, _ = plan_partitions(rows, ClusterConfig(shuffle_partitions=10, aqe=False))
    tasks, notes = plan_partitions(rows, config)
    assert len(plain) == 10
    assert sum(t.rows for t in tasks) == sum(rows.values())
    assert sorted(k for t in tasks for k in t.keys) == sorted(rows)
    assert any(t.split for t in tasks) and any("skewed partition" in n for n in notes)
    assert max(t.rows for t in tasks) >= 3000                       # the hot key stays whole
    assert max(t.rows for t in tasks) < max(t.rows for t in plain)


def test_speculation_beats_a_slow_node():
    rows = [300] * 40
    slow = Conditions(slow_executor_factor=10)
    without = reduce_job(rows, ClusterConfig(speculation=False), slow)
    with_spec = reduce_job(rows, ClusterConfig(speculation=True), slow)
    assert with_spec.stages[1]["speculative"] > 0
    assert with_spec.stages[1]["max_s"] < without.stages[1]["max_s"]


def test_lost_executors_are_retried_and_recomputed_deterministically():
    rows = [400] * 60
    loss = Conditions(executor_loss_mean_s=15)
    a = reduce_job(rows, ClusterConfig(), loss, random.Random("x"))
    b = reduce_job(rows, ClusterConfig(), loss, random.Random("x"))
    assert a.summary() == b.summary()
    assert a.succeeded
    assert sum(s["failed_attempts"] for s in a.stages) > 0
    assert any("lost" in text for _, text in a.log)
    healthy = reduce_job(rows, ClusterConfig())
    assert a.duration_s > healthy.duration_s
    # Preemption is not the tasks' fault: it does not count towards
    # spark.task.maxFailures ...
    strict = reduce_job(rows, ClusterConfig(max_task_failures=1), loss, random.Random("x"))
    assert strict.succeeded
    # ... but an application that keeps losing executors gives up
    # (spark.executor.maxNumFailures = max(2 x executors, 3)).
    fragile = reduce_job(rows * 3, ClusterConfig(executors=1), Conditions(executor_loss_mean_s=10),
                         random.Random("x"))
    assert not fragile.succeeded and fragile.reason == "executors_lost"
    assert "executor failures" in fragile.error


def test_cluster_down_and_busy():
    down = reduce_job([10], ClusterConfig(), Conditions(cluster_down=True))
    assert not down.succeeded and down.reason == "cluster_down"
    busy = reduce_job([300] * 32, ClusterConfig(), Conditions(queue_wait_s=1800, granted_share=0.5))
    assert busy.succeeded and busy.queue_s == 1800 and len(busy.executors) == 2
    assert ticks_for(busy.duration_s) >= 2


# -- the data lake ------------------------------------------------------------

def test_lake_files_and_compaction():
    lake = DataLake()
    for tick in range(10):
        for topic in ("a", "b"):
            first = len(lake.events)
            lake.events.extend(range(5))
            lake.files.append([first, len(lake.events), tick, topic, False])
    assert len(lake.file_rows(0)) == 20 and sum(lake.file_rows(0)) == 100
    assert sum(lake.file_rows(50)) == 50
    before, after = lake.compact(before_tick=8)
    assert before == 20 and after == 1 + 4
    assert sum(lake.file_rows(0)) == 100
    events = list(lake.events)
    assert lake.events == events                                     # only metadata changed


def test_a_job_over_a_snapshot_of_the_lake_reads_its_files(movies, events):
    # Regression: the console passes a copy of the lake's list; the scan
    # stage used to see it as one file (one task).
    lake = DataLake()
    for i in range(0, len(events), 25):
        first = len(lake.events)
        lake.events.extend(events[i:i + 25])
        lake.files.append([first, len(lake.events), i // 25, "t", False])
    p, _ = pipeline(movies)
    run = Spark().plan_offline_feature_job(list(lake.events), p, lake=lake)
    scan = run.report.stages[0]
    assert scan["tasks"] > 1 and scan["rows"] == len(events)
