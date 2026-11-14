"""Spark: the offline feature job, on a simulated cluster.

A job has three stages, as the real job would:

  stage 0  scan        read the lake files (incremental: only the files
                       since the last run) and shuffle every event to its
                       user's and its movie's partition
  stage 1  user features   one task per user partition
  stage 2  item features   one task per movie partition
  commit   the driver writes a new feature version and swaps it in at once

`plan_offline_feature_job` decides what to compute (incremental or a full
backfill), partitions the work (hash partitioning, AQE) and simulates the
schedule on the cluster (ml_system/spark_sim/engine.py): how long it takes
and whether it fails. `run_offline_feature_job` then does the real work,
task by task in a fixed order, and commits. A failed job commits nothing,
so the feature store keeps the last good version. Results are the same
for any cluster, partitioning or schedule (tests/test_spark.py).

Randomness: only lost executors draw random numbers, from an RNG seeded by
the build seed and the job number ("spark-<seed>-<job>"), so switching a
fault on never shifts any other draw.
"""

import random

from simulation.service import Service

from ml_system.spark_sim import cluster as C
from ml_system.spark_sim.engine import StageSpec, plan_partitions, plan_scan, simulate, ticks_for


class SparkJobRun:
    """One planned job: its batch, partitioning and simulated report."""

    def __init__(self, job_id, batch, user_tasks, item_tasks, report, config, submitted_tick=None):
        self.job_id = job_id
        self.batch = batch
        self.user_tasks = user_tasks
        self.item_tasks = item_tasks
        self.report = report
        self.config = config
        self.submitted_tick = submitted_tick
        self.ran = False

    @property
    def ticks(self):
        return ticks_for(self.report.duration_s)

    def summary(self, timeline=False):
        report = self.report.summary(timeline=timeline)
        report["failure"] = report.pop("reason")         # oom, executors_lost, ... (None if it succeeded)
        return {**report, "job_id": self.job_id, "mode": "full" if self.batch.full else "incremental",
                "reason": self.batch.reason, "events": self.batch.size, "lake_events": len(self.batch.events),
                "submitted_tick": self.submitted_tick, "ticks": self.ticks,
                "cost": round(self.report.executor_seconds / 3600.0 * self.config.price_per_hour()
                              / max(1, self.config.executors), 2),
                "config": self.config.summary()}


class Spark(Service):

    def __init__(self, seed=0, config=None):
        super().__init__()
        self.sim = None
        self.seed = seed
        self.config = config or C.ClusterConfig()
        self.conditions = C.Conditions()
        self.incremental = True
        self.next_job_id = 1
        self.history = []             # SparkJobRun summaries, newest last
        self.max_history = 20
        self.last_run = None

        # Fault injection: while set, no executor can start (jobs fail).
        self.failing = False

    def start(self, sim):
        super().start(sim)
        self.sim = sim

    # -- planning --------------------------------------------------------

    def plan_offline_feature_job(self, events, feature_pipeline, lake=None, submitted_tick=None):
        """Plan and simulate one feature job over `events` (the lake)."""
        batch = feature_pipeline.plan_batch(events, incremental=self.incremental)
        rows = batch.events_with_order()
        user_rows, item_rows = {}, {}
        for _, event in rows:
            user_rows[event.user_id] = user_rows.get(event.user_id, 0) + 1
            if hasattr(event, "movie_id"):
                item_rows[event.movie_id] = item_rows.get(event.movie_id, 0) + 1

        if lake is not None and getattr(lake, "files", None) is not None and self._reads_lake(events, lake):
            file_rows = lake.file_rows(batch.start, len(events))
        else:
            file_rows = [batch.size] if batch.size else []
        scan = plan_scan(file_rows)
        user_tasks, user_notes = plan_partitions(user_rows, self.config)
        item_tasks, item_notes = plan_partitions(item_rows, self.config)
        stages = [
            StageSpec(0, "scan lake", "scan", scan),
            StageSpec(1, "user features", "reduce", user_tasks, parents=(0,), notes=user_notes),
            StageSpec(2, "item features", "reduce", item_tasks, parents=(0,), notes=item_notes),
        ]
        job_id = self.next_job_id
        self.next_job_id += 1
        conditions = self.conditions
        if self.failing:
            conditions = C.Conditions(**{**conditions.__dict__, "cluster_down": True})
        rng = random.Random("spark-%s-%d" % (self.seed, job_id)) if conditions.executor_loss_mean_s else None
        report = simulate(stages, self.config, conditions, rng, listing_files=len(file_rows),
                          commit_rows=len(user_rows) + len(item_rows))
        report.notes = ["%s over %d events (%d lake files)" % (batch.reason, batch.size, len(file_rows))]
        return SparkJobRun(job_id, batch, user_tasks, item_tasks, report, self.config, submitted_tick)

    @staticmethod
    def _reads_lake(events, lake):
        """True if `events` is (a snapshot of) the lake's log, so its files
        describe how the job reads it."""
        n = len(events)
        return n <= len(lake.events) and (n == 0 or events[n - 1] is lake.events[n - 1])

    # -- running ---------------------------------------------------------

    def run_offline_feature_job(self, events, feature_pipeline, run=None, lake=None):
        """Run a planned job (or plan one now) and commit its features.
        Raises RuntimeError if the job fails; nothing is committed then."""
        if run is None:
            run = self.plan_offline_feature_job(events, feature_pipeline, lake=lake)
        run.ran = True
        if self.failing and run.report.succeeded:
            # The cluster went down while the job was running.
            run.report.status = "failed"
            run.report.reason = "cluster_down"
            run.report.error = "SparkException: lost all executors (cluster went down during the job)"
        self._record(run)
        if not run.report.succeeded:
            raise RuntimeError("Spark job failed: %s" % run.report.error)

        batch = run.batch
        rows = batch.events_with_order()
        user_results, item_results = {}, {}
        # Real work, one task at a time in task order (any order gives the
        # same features: each entity lives in exactly one task).
        user_of = {}
        for index, task in enumerate(run.user_tasks):
            for key in task.keys:
                user_of[key] = index
        item_of = {}
        for index, task in enumerate(run.item_tasks):
            for key in task.keys:
                item_of[key] = index
        user_parts = [[] for _ in run.user_tasks]
        item_parts = [[] for _ in run.item_tasks]
        for row in rows:
            event = row[1]
            user_parts[user_of[event.user_id]].append(row)
            if hasattr(event, "movie_id"):
                item_parts[item_of[event.movie_id]].append(row)
        for part in user_parts:
            if part:
                user_results.update(feature_pipeline.user_task(batch, part))
        for part in item_parts:
            if part:
                item_results.update(feature_pipeline.item_task(batch, part))
        feature_pipeline.commit(batch, user_results, item_results)
        return run

    def _record(self, run):
        self.history.append(run.summary(timeline=False))
        self.last_run = run
        del self.history[:-self.max_history]

    def summary(self):
        return {"config": self.config.summary(), "incremental": self.incremental,
                "conditions": dict(self.conditions.__dict__), "failing": self.failing,
                "jobs": len(self.history) and self.history[-1]["job_id"]}
