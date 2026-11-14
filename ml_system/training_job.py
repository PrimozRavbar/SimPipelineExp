"""The training job on a simulated GPU cluster: how long it takes and how it
fails. The real training runs in the console's background thread; this
model only decides the simulated time and the outcome, so (like Spark)
the cluster never changes the model, only when it is ready and whether
the run succeeds.

Scale and time follow ml_system/spark_sim/cluster.py: one simulated
example stands for ~1000 production examples, and one tick is ~15 minutes.

Phases of one attempt:
  start-up   GPU pods scheduled, image pulled, data loader warm
  load       read the training set (examples x LOAD_S)
  train      epochs x examples x (1 + negatives) x EXAMPLE_S, divided by the
             GPU throughput (batch-size efficiency x GPUs x scaling)
  extras     index build, ranker / preference model / CF fits
  save       write the model and its index to the registry

Failures:
- GPU out of memory: model + batch activations + the in-batch logits
  (batch^2) exceed the GPU memory. Deterministic: retries fail the same way.
- Spot preemption: GPUs reclaimed at random (exponential lifetime). The
  job restarts after RESTART_S and resumes from its last checkpoint (or
  from scratch without checkpoints); it fails after max_restarts.
"""

import math
import random
from dataclasses import dataclass, asdict

# Start-up and fixed costs (seconds)
STARTUP_S = 120              # GPU pods scheduled, image pulled
RESTART_S = 180              # a preempted job: new node, restart, reload
LOAD_S = 0.004               # per example: read and decode the training set
SAVE_S = 30                  # model + index to the registry
CHECKPOINT_S = 10            # writing one checkpoint

# Compute
EXAMPLE_S = 0.004            # per example and candidate (positive or negative) at full efficiency
FULL_BATCH = 256             # batches this large keep the GPU busy; smaller ones waste it
MIN_EFFICIENCY = 0.15
SCALING = 0.9                # throughput kept per doubling of GPUs (communication)
INDEX_S_PER_ITEM = 0.02      # embed and index one movie
RANKER_S_PER_ROW = 0.0004    # ranker / preference model / CF fits, per training row

# Memory (GB)
MODEL_GB = 1.5               # weights, optimizer state, framework
EXAMPLE_GB = 0.004           # activations per example and candidate
LOGITS_GB_PER_1024_SQ = 1.0  # in-batch softmax: batch x batch logits and gradients

# Price (money, console economy)
PRICE_PER_GPU_HOUR = 3.0


@dataclass
class TrainingCluster:
    gpus: int = 1
    gpu_memory_gb: int = 16
    batch_size: int = 32
    checkpoint_minutes: int = 10     # 0 = no checkpoints
    max_restarts: int = 3

    def summary(self):
        return asdict(self)


@dataclass
class TrainingConditions:
    preemption_mean_s: float = 0     # GPUs reclaimed after ~this long (0 = never)
    queue_wait_s: float = 0          # waiting for GPUs


def memory_gb(batch_size, negatives):
    candidates = 1 + negatives
    return (MODEL_GB + batch_size * candidates * EXAMPLE_GB
            + (batch_size / 1024.0) ** 2 * LOGITS_GB_PER_1024_SQ)


def efficiency(batch_size):
    return max(MIN_EFFICIENCY, min(1.0, math.sqrt(batch_size / FULL_BATCH)))


class TrainingReport:

    def __init__(self):
        self.status = "succeeded"
        self.reason = None           # gpu_oom, preempted
        self.error = None
        self.duration_s = 0.0
        self.phases = []             # [name, start, end]
        self.log = []                # [time, text]
        self.preemptions = 0
        self.memory_gb = 0.0
        self.gpu_seconds = 0.0
        self.compute_s = 0.0

    @property
    def succeeded(self):
        return self.status == "succeeded"

    def summary(self):
        return {k: getattr(self, k) for k in ("status", "reason", "error", "duration_s", "phases", "log",
                                              "preemptions", "memory_gb", "gpu_seconds", "compute_s")}


def simulate_training(examples, epochs, negatives, cluster, conditions=None, rng=None,
                      items=0, extra_rows=0):
    """Simulated time and outcome of one training run. rng (random.Random)
    is used only while GPUs can be preempted."""
    cond = conditions or TrainingConditions()
    r = TrainingReport()
    t = 0.0

    def log(text):
        r.log.append([round(t, 1), text])

    if cond.queue_wait_s:
        log("Waiting for GPUs (%.0f min)" % (cond.queue_wait_s / 60))
        r.phases.append(["queue", 0.0, cond.queue_wait_s])
        t = cond.queue_wait_s
    r.memory_gb = round(memory_gb(cluster.batch_size, negatives), 2)
    r.phases.append(["start-up", t, t + STARTUP_S])
    t += STARTUP_S
    if r.memory_gb > cluster.gpu_memory_gb:
        r.status, r.reason = "failed", "gpu_oom"
        r.error = ("CUDA out of memory: batch %d needs %.1f GB, the GPU has %d GB"
                   % (cluster.batch_size, r.memory_gb, cluster.gpu_memory_gb))
        t += 20
        log(r.error)
        r.duration_s = t
        r.gpu_seconds = t * cluster.gpus
        return r

    load = examples * LOAD_S
    r.phases.append(["load data", t, t + load])
    t += load
    gpus = max(1, cluster.gpus)
    throughput = efficiency(cluster.batch_size) * gpus * SCALING ** math.log2(gpus)
    compute = epochs * examples * (1 + negatives) * EXAMPLE_S / throughput
    r.compute_s = compute
    checkpoint_every = cluster.checkpoint_minutes * 60
    checkpoints = int(compute // checkpoint_every) if checkpoint_every else 0
    work = compute + checkpoints * CHECKPOINT_S

    # Training with preemptions: progress since the last checkpoint is lost.
    train_start, done, saved = t, 0.0, 0.0
    while done < work:
        left = work - done
        life = rng.expovariate(1.0 / cond.preemption_mean_s) if (cond.preemption_mean_s and rng) else None
        if life is None or life >= left:
            t += left
            done = work
            break
        t += life
        done += life
        r.preemptions += 1
        if checkpoint_every:
            step = checkpoint_every + CHECKPOINT_S
            saved = (done // step) * step
        else:
            saved = 0.0
        log("GPUs preempted at %.0f%% of training; resuming from %s"
            % (100 * done / work, "the last checkpoint (%.0f%%)" % (100 * saved / work) if saved else "scratch"))
        done = saved
        if r.preemptions > cluster.max_restarts:
            r.status, r.reason = "failed", "preempted"
            r.error = "training preempted %d times (max restarts %d)" % (r.preemptions, cluster.max_restarts)
            log(r.error)
            r.phases.append(["train", train_start, t])
            r.duration_s = t
            r.gpu_seconds = t * gpus
            return r
        t += RESTART_S
    r.phases.append(["train", train_start, t])
    extras = items * INDEX_S_PER_ITEM + extra_rows * RANKER_S_PER_ROW
    r.phases.append(["index + rankers", t, t + extras])
    t += extras
    r.phases.append(["save", t, t + SAVE_S])
    t += SAVE_S
    log("Training finished: %d examples x %d epochs in %.0f min of compute%s"
        % (examples, epochs, compute / 60, ", %d preemptions" % r.preemptions if r.preemptions else ""))
    r.duration_s = t
    r.gpu_seconds = t * gpus
    return r


def rng_for(seed, version):
    return random.Random("training-%s-%d" % (seed, version))
