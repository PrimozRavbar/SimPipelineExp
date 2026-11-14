"""A simulated Spark engine: DAG and task scheduling in simulated time.

`plan_partitions` turns per-key row counts into reduce tasks the way Spark
does: hash partitioning into `shuffle_partitions`, and with adaptive query
execution (AQE) coalescing small partitions and splitting skewed ones (a
single key is never split).

`simulate` runs a job's stages on a cluster as a discrete-event
simulation and returns a JobReport: how long each phase and task took,
which tasks failed and were retried, which executors were lost, how much
data spilled. Nothing here touches the data: the caller runs the real work
afterwards (see ml_system/spark.py), so the schedule can never change a
result, only when it is ready and whether the job succeeds.

What it models:
- Executors x cores task slots; start-up, queue wait for resources.
- Scan stage (map side, writes shuffle files) -> reduce stages (users,
  items) that start when every map output is available.
- Memory: a task holds rows x ROW_MB; beyond its share of the executor's
  memory it spills to disk (slower); beyond OVERHEAD_HEADROOM x that, the
  container is killed (OOM) and the executor is lost with it.
- Task retries (max_task_failures) and stage re-attempts: a lost executor
  takes its map outputs with it, so reduce stages hit FetchFailed and the
  lost map tasks run again (up to MAX_STAGE_ATTEMPTS).
- Stragglers (a slow node) and speculative execution.
- Spot executors reclaimed at random (an exponential lifetime) and
  replaced after a start-up delay.
"""

import heapq
import math
import statistics

from ml_system.spark_sim import cluster as C

SPECULATION_INTERVAL_S = 1.0


class TaskSpec:
    """One task's work: rows, files to open (scan), keys (reduce)."""

    def __init__(self, rows, files=0, keys=None, max_key_rows=0, split=False):
        self.rows = rows
        self.files = files
        self.keys = keys if keys is not None else []
        self.max_key_rows = max_key_rows
        self.split = split            # produced by an AQE skew split


class StageSpec:

    def __init__(self, stage_id, name, kind, tasks, parents=(), notes=None):
        self.stage_id = stage_id
        self.name = name
        self.kind = kind              # "scan" or "reduce"
        self.tasks = tasks
        self.parents = tuple(parents)
        self.notes = notes or []


def plan_scan(file_rows):
    """Pack lake files (row counts, in order) into scan tasks of about
    SCAN_PARTITION_MB."""
    tasks, rows, files = [], 0, 0
    limit = C.SCAN_PARTITION_MB / C.ROW_MB
    for n in file_rows:
        if files and rows + n > limit:
            tasks.append(TaskSpec(rows, files))
            rows, files = 0, 0
        rows += n
        files += 1
    if files:
        tasks.append(TaskSpec(rows, files))
    return tasks


def plan_partitions(key_rows, config):
    """Reduce tasks for {key: rows}. Returns (tasks, notes)."""
    count = max(1, int(config.shuffle_partitions))
    parts = [[] for _ in range(count)]
    for key in sorted(key_rows):
        parts[key % count].append(key)
    sizes = [sum(key_rows[k] for k in keys) for keys in parts]
    notes = []
    if not config.aqe:
        return [TaskSpec(size, keys=keys, max_key_rows=max((key_rows[k] for k in keys), default=0))
                for keys, size in zip(parts, sizes)], notes

    advisory = C.ADVISORY_MB / C.ROW_MB
    nonempty = sorted(s for s in sizes if s)
    median = statistics.median(nonempty) if nonempty else 0
    skew_limit = max(C.SKEW_FACTOR * median, C.SKEW_MIN_MB / C.ROW_MB)
    tasks, run, run_rows = [], [], 0

    def flush():
        nonlocal run, run_rows
        if run:
            tasks.append(TaskSpec(run_rows, keys=run, max_key_rows=max(key_rows[k] for k in run)))
        run, run_rows = [], 0

    for index, (keys, size) in enumerate(zip(parts, sizes)):
        if not size:
            continue
        if size > skew_limit:
            flush()
            # Split by key into chunks of about the advisory size.
            chunk, chunk_rows, pieces = [], 0, 0
            for k in sorted(keys, key=lambda k: -key_rows[k]):
                if chunk and chunk_rows + key_rows[k] > advisory:
                    tasks.append(TaskSpec(chunk_rows, keys=chunk, max_key_rows=max(key_rows[x] for x in chunk),
                                          split=True))
                    pieces += 1
                    chunk, chunk_rows = [], 0
                chunk.append(k)
                chunk_rows += key_rows[k]
            if chunk:
                tasks.append(TaskSpec(chunk_rows, keys=chunk, max_key_rows=max(key_rows[x] for x in chunk),
                                      split=True))
                pieces += 1
            notes.append("AQE: skewed partition %d (%d rows, %.0fx the median) split into %d tasks"
                         % (index, size, size / median if median else 0, pieces))
            continue
        if run and run_rows + size > advisory:
            flush()
        run.extend(keys)
        run_rows += size
    flush()
    notes.append("AQE: %d shuffle partitions coalesced into %d tasks" % (count, len(tasks)))
    return tasks, notes


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class _Executor:

    def __init__(self, executor_id, cores, ready_at, slow=1.0):
        self.id = executor_id
        self.cores = cores
        self.ready_at = ready_at
        self.slow = slow
        self.alive = True
        self.lost_at = None
        self.lost_reason = None
        self.running = {}             # attempts, in launch order (a set would iterate by address)


class _Attempt:

    def __init__(self, task, executor, start, end, outcome, speculative=False):
        self.task = task
        self.executor = executor
        self.start = start
        self.end = end
        self.outcome = outcome        # "success" or "oom" (planned); "killed"/"lost" set later
        self.speculative = speculative
        self.done = False


class _Task:

    def __init__(self, stage, index, spec):
        self.stage = stage
        self.index = index
        self.spec = spec
        self.state = "pending"        # pending -> running -> done
        self.failures = 0
        self.attempts = []
        self.output_on = None         # scan tasks: executor holding the map output
        self.duration = None


class _Stage:

    def __init__(self, spec):
        self.spec = spec
        self.tasks = [_Task(self, i, t) for i, t in enumerate(spec.tasks)]
        self.attempt = 1
        self.start = None
        self.end = None
        self.spill_mb = 0.0

    def done(self):
        return all(t.state == "done" for t in self.tasks)


class JobReport:

    def __init__(self):
        self.status = "running"
        self.error = None
        self.reason = None            # short failure class: oom, executors_lost, fetch_failed, cluster_down
        self.duration_s = 0.0
        self.queue_s = 0.0
        self.startup_s = 0.0
        self.listing_s = 0.0
        self.commit_s = 0.0
        self.stages = []
        self.executors = []
        self.timeline = []
        self.log = []
        self.executor_seconds = 0.0
        self.notes = []

    @property
    def succeeded(self):
        return self.status == "succeeded"

    def summary(self, timeline=True):
        out = {k: getattr(self, k) for k in ("status", "error", "reason", "duration_s", "queue_s",
                                             "startup_s", "listing_s", "commit_s", "stages",
                                             "executors", "log", "executor_seconds", "notes")}
        if timeline:
            out["timeline"] = self.timeline
        return out


def task_seconds(stage_kind, spec, config, slow=1.0):
    """(seconds, spill MB, oom?) for one attempt of a task."""
    if stage_kind == "scan":
        return (C.TASK_OVERHEAD_S + spec.files * C.FILE_OPEN_S + spec.rows * C.SCAN_ROW_S) * slow, 0.0, False
    memory = config.task_memory_mb()
    need = spec.rows * C.ROW_MB
    spill = max(0.0, need - memory)
    oom = need > memory * C.OVERHEAD_HEADROOM
    seconds = C.TASK_OVERHEAD_S + spec.rows * C.ROW_S
    if spill:
        seconds += (spill / C.ROW_MB) * C.ROW_S * C.SPILL_PENALTY
    return seconds * slow, spill, oom


def simulate(stages, config, conditions=None, rng=None, listing_files=0, commit_rows=0, max_timeline=1000):
    """Run `stages` (StageSpecs; parents must come first) on a cluster.
    rng: a random.Random, used only when executors can be lost."""
    cond = conditions or C.Conditions()
    report = JobReport()
    t = 0.0

    def log(time, text):
        report.log.append([round(time, 1), text])

    # Queue for resources, then start the driver and the executors.
    report.queue_s = cond.queue_wait_s
    if cond.queue_wait_s:
        log(0, "Waiting for cluster resources (%.0f s): pods pending" % cond.queue_wait_s)
    t = cond.queue_wait_s
    if cond.cluster_down:
        report.status = "failed"
        report.reason = "cluster_down"
        report.error = "SparkException: no executors could be started (cluster unavailable)"
        report.startup_s = C.DRIVER_STARTUP_S + C.EXECUTOR_STARTUP_S
        report.duration_s = t + report.startup_s
        log(report.duration_s, "Job aborted: " + report.error)
        return report

    granted = max(1, int(round(config.executors * cond.granted_share)))
    if granted < config.executors:
        log(t, "Only %d of %d executors granted (cluster busy)" % (granted, config.executors))
    ready = t + C.EXECUTOR_STARTUP_S
    report.listing_s = listing_files * C.LIST_S_PER_FILE
    driver_ready = t + C.DRIVER_STARTUP_S + report.listing_s
    report.startup_s = max(ready, driver_ready) - t
    begin = max(ready, driver_ready)

    executors = []
    events = []                       # heap of (time, seq, kind, payload)
    seq = [0]

    def push(time, kind, payload):
        seq[0] += 1
        heapq.heappush(events, (time, seq[0], kind, payload))

    def add_executor(ready_at):
        executor = _Executor(len(executors) + 1, config.cores, ready_at,
                             slow=cond.slow_executor_factor if not executors else 1.0)
        executors.append(executor)
        if cond.executor_loss_mean_s and rng is not None:
            push(ready_at + rng.expovariate(1.0 / cond.executor_loss_mean_s), "lost", (executor, "spot instance reclaimed"))
        push(ready_at, "ready", executor)
        return executor

    for _ in range(granted):
        add_executor(ready)
    if cond.slow_executor_factor > 1:
        log(ready, "Executor 1 is %.0fx slower than the others (degraded node)" % cond.slow_executor_factor)

    st = [_Stage(spec) for spec in stages]
    by_id = {s.spec.stage_id: s for s in st}
    failed = [None]
    now = [begin]
    executor_failures = [0]
    max_executor_failures = max(2 * config.executors, 3)
    check_pending = [False]

    def runnable(stage):
        return all(by_id[p].done() for p in stage.spec.parents)

    def free_slots(executor):
        if not executor.alive or executor.ready_at > now[0]:
            return 0
        return executor.cores - len(executor.running)

    def launch(task, executor, speculative=False):
        seconds, spill, oom = task_seconds(task.stage.spec.kind, task.spec, config, executor.slow)
        start = now[0]
        if task.stage.start is None:
            task.stage.start = start
        if oom:
            end = start + seconds * 0.3
            outcome = "oom"
        else:
            end = start + seconds
            outcome = "success"
            task.stage.spill_mb += spill
        attempt = _Attempt(task, executor, start, end, outcome, speculative)
        task.attempts.append(attempt)
        task.state = "running"
        executor.running[attempt] = None
        push(end, "end", attempt)

    def schedule():
        pending = [task for stage in st if runnable(stage) for task in stage.tasks if task.state == "pending"]
        for executor in executors:
            while free_slots(executor) and pending:
                launch(pending.pop(0), executor)
        if not config.speculation:
            return
        for stage in st:
            done = [task for task in stage.tasks if task.state == "done"]
            if not stage.tasks or len(done) < C.SPECULATION_QUANTILE * len(stage.tasks):
                continue
            median = statistics.median(task.duration for task in done)
            for task in stage.tasks:
                live = [a for a in task.attempts if not a.done]
                if task.state != "running" or len(live) != 1:
                    continue
                attempt = live[0]
                if now[0] - attempt.start <= C.SPECULATION_MULTIPLIER * median:
                    continue
                for executor in executors:
                    if executor is not attempt.executor and free_slots(executor):
                        log(now[0], "Speculative copy of %s task %d (running %.0f s, median %.1f s)"
                            % (stage.spec.name, task.index, now[0] - attempt.start, median))
                        launch(task, executor, speculative=True)
                        break

    def finish_attempt(attempt, outcome):
        attempt.done = True
        attempt.outcome = outcome
        attempt.executor.running.pop(attempt, None)
        if len(report.timeline) < max_timeline:
            report.timeline.append([attempt.executor.id, attempt.task.stage.spec.stage_id, attempt.task.index,
                                    round(attempt.start, 2), round(now[0], 2), outcome,
                                    1 if attempt.speculative else 0])

    def fail_task(task, why):
        task.failures += 1
        if task.failures >= config.max_task_failures:
            failed[0] = ("Task %d in stage %d (%s) failed %d times, most recent failure: %s"
                         % (task.index, task.stage.spec.stage_id, task.stage.spec.name, task.failures, why))
            return
        if not any(not a.done for a in task.attempts):
            task.state = "pending"

    def lose_executor(executor, reason, caused_by_app=False):
        if not executor.alive:
            return
        executor.alive = False
        executor.lost_at = now[0]
        executor.lost_reason = reason
        executor_failures[0] += 1
        log(now[0], "Executor %d lost: %s" % (executor.id, reason))
        for attempt in list(executor.running):
            finish_attempt(attempt, "lost")
            task = attempt.task
            if task.state == "done":
                continue
            if caused_by_app:
                fail_task(task, "ExecutorLostFailure (executor %d exited caused by one of the running "
                                "tasks: %s)" % (executor.id, reason))
            elif not any(not a.done for a in task.attempts):
                # Not the task's fault (preemption): retried without
                # counting towards spark.task.maxFailures.
                task.state = "pending"
        if failed[0] is None and executor_failures[0] >= max_executor_failures:
            failed[0] = ("Max number of executor failures (%d) reached (spark.executor.maxNumFailures)"
                         % max_executor_failures)
            report.reason = "oom" if caused_by_app else "executors_lost"
        # Map outputs on it are gone: reduce stages that still need them
        # fail to fetch, and the lost map tasks run again.
        lost_maps = [task for stage in st if stage.spec.kind == "scan" for task in stage.tasks
                     if task.state == "done" and task.output_on is executor]
        needed = any(not stage.done() for stage in st if stage.spec.kind == "reduce")
        if lost_maps and needed:
            for stage in st:
                if stage.spec.kind == "reduce" and not stage.done():
                    running = [a for task in stage.tasks for a in task.attempts if not a.done]
                    for attempt in running:
                        finish_attempt(attempt, "killed")
                    for task in stage.tasks:
                        if task.state == "running":
                            task.state = "pending"
                    if running:
                        stage.attempt += 1
                        log(now[0], "FetchFailed: stage %d (%s) re-attempt %d"
                            % (stage.spec.stage_id, stage.spec.name, stage.attempt))
                        if stage.attempt > C.MAX_STAGE_ATTEMPTS:
                            failed[0] = ("Stage %d (%s) failed %d times (FetchFailed: shuffle output lost "
                                         "with its executors)" % (stage.spec.stage_id, stage.spec.name,
                                                                  C.MAX_STAGE_ATTEMPTS))
            scan = lost_maps[0].stage
            scan.attempt += 1
            for task in lost_maps:
                task.state = "pending"
                task.output_on = None
            log(now[0], "Resubmitting %d lost map tasks of stage %d (%s)"
                % (len(lost_maps), scan.spec.stage_id, scan.spec.name))
        if failed[0] is None:
            replacement = add_executor(now[0] + C.EXECUTOR_STARTUP_S)
            log(now[0], "Requested replacement executor %d (ready in %d s)" % (replacement.id, C.EXECUTOR_STARTUP_S))

    push(begin, "schedule", None)
    while events and failed[0] is None:
        time, _, kind, payload = heapq.heappop(events)
        now[0] = max(now[0], time)
        if now[0] - begin > C.JOB_TIMEOUT_S:
            failed[0] = "job did not finish within %d hours" % (C.JOB_TIMEOUT_S // 3600)
            report.reason = "timeout"
            break
        if all(stage.done() for stage in st):
            break
        if kind == "end":
            attempt = payload
            if attempt.done:
                continue
            task = attempt.task
            if attempt.outcome == "success":
                finish_attempt(attempt, "success")
                if task.state != "done":
                    task.state = "done"
                    task.duration = attempt.end - attempt.start
                    task.output_on = attempt.executor
                    if attempt.speculative:
                        log(now[0], "Speculative copy of %s task %d won" % (task.stage.spec.name, task.index))
                for other in task.attempts:
                    if not other.done:
                        finish_attempt(other, "killed")
                if task.stage.done():
                    task.stage.end = now[0]
            else:
                finish_attempt(attempt, "oom")
                fail_task(task, "ExecutorLostFailure (container killed: exceeding memory limits, "
                                "%.0f MB needed, %.0f MB per task)" % (task.spec.rows * C.ROW_MB,
                                                                     config.task_memory_mb()))
                if failed[0] is None:
                    log(now[0], "Task %d of %s ran out of memory (%d rows)" % (task.index, task.stage.spec.name,
                                                                               task.spec.rows))
                    lose_executor(attempt.executor, "container killed (out of memory)", caused_by_app=True)
                else:
                    report.reason = "oom"
        elif kind == "lost":
            executor, reason = payload
            if executor.alive and executor.ready_at <= time:
                lose_executor(executor, reason)
                if failed[0] is not None and report.reason is None:
                    report.reason = "executors_lost"
            elif executor.alive:
                push(executor.ready_at + 1, "lost", payload)
        if kind == "check":
            check_pending[0] = False
        if failed[0] is None:
            schedule()
            # Speculation looks for stragglers on a timer, not only when
            # tasks end (spark.speculation.interval).
            if config.speculation and not check_pending[0] and any(e.running for e in executors):
                check_pending[0] = True
                push(now[0] + SPECULATION_INTERVAL_S, "check", None)

    end = now[0]
    report.stages = []
    for stage in st:
        durations = [task.duration for task in stage.tasks if task.duration is not None]
        attempts = [a for task in stage.tasks for a in task.attempts]
        report.stages.append({
            "id": stage.spec.stage_id, "name": stage.spec.name, "tasks": len(stage.tasks),
            "rows": sum(task.spec.rows for task in stage.tasks),
            "done": sum(1 for task in stage.tasks if task.state == "done"),
            "attempts": len(attempts),
            "failed_attempts": sum(1 for a in attempts if a.outcome in ("oom", "lost")),
            "speculative": sum(1 for a in attempts if a.speculative),
            "stage_attempts": stage.attempt,
            "median_s": round(statistics.median(durations), 2) if durations else None,
            "max_s": round(max(durations), 2) if durations else None,
            "skew": round(max(durations) / statistics.median(durations), 1) if durations and statistics.median(durations) else None,
            "max_task_rows": max((task.spec.rows for task in stage.tasks), default=0),
            "split_tasks": sum(1 for task in stage.tasks if task.spec.split),
            "spill_mb": round(stage.spill_mb),
            "start": round(stage.start, 1) if stage.start is not None else None,
            "end": round(stage.end, 1) if stage.end is not None else None,
            "notes": stage.spec.notes,
        })
    for executor in executors:
        stop = executor.lost_at if executor.lost_at is not None else end
        report.executor_seconds += max(0.0, stop - min(executor.ready_at - C.EXECUTOR_STARTUP_S, stop))
        report.executors.append({"id": executor.id, "cores": executor.cores, "ready_at": round(executor.ready_at, 1),
                                 "lost_at": None if executor.lost_at is None else round(executor.lost_at, 1),
                                 "reason": executor.lost_reason, "slow": executor.slow})
    if failed[0] is not None:
        report.status = "failed"
        report.error = "SparkException: Job aborted due to stage failure: " + failed[0]
        if report.reason is None:
            report.reason = "fetch_failed" if "FetchFailed" in failed[0] else "task_failures"
        report.duration_s = end
        log(end, "Job aborted: " + failed[0])
        return report
    if not all(stage.done() for stage in st):
        report.status = "failed"
        report.reason = "executors_lost"
        report.error = "SparkException: job could not finish (no executors left)"
        report.duration_s = end
        log(end, "Job aborted: " + report.error)
        return report
    report.commit_s = C.COMMIT_S + commit_rows * C.COMMIT_ROW_S
    report.duration_s = end + report.commit_s
    report.executor_seconds += len([e for e in executors if e.alive]) * report.commit_s
    report.status = "succeeded"
    log(report.duration_s, "Job succeeded: new feature version committed")
    return report


def ticks_for(seconds):
    """Whole ticks a job of `seconds` takes beyond the tick it starts in: a
    job that finishes inside its first tick is ready at that boundary (the
    same as an instant job)."""
    return int(math.floor(seconds / C.TICK_SECONDS))
