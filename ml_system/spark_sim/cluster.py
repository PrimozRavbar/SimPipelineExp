"""Spark cluster configuration and the cost model of the simulated engine.

Time is simulated, not measured: a job's duration comes from the work its
tasks do (rows, files), the cluster (executors x cores, memory) and what
goes wrong (lost executors, slow nodes, skew), never from wall-clock time.
The real feature computation runs afterwards, partition by partition, in a
fixed order, so results never depend on the simulated schedule.

Scale. One simulated event stands for about SCALE production rows (a game
with 20 users a tick stands for a large service), so the per-row costs
below are per simulated event: realistic per production row.

Ticks. One tick stands for TICK_SECONDS (15 minutes) of real time: the
default Spark schedule (every 100 ticks) is then about once a day, and a
healthy feature job finishes inside one tick.
"""

from dataclasses import dataclass, asdict

TICK_SECONDS = 900           # one tick ~ 15 minutes
SCALE = 1000                 # production rows per simulated event (documentation)

# Start-up
DRIVER_STARTUP_S = 15        # driver pod, SparkContext, plan
EXECUTOR_STARTUP_S = 60      # executor pods scheduled and registered

# Reading the data lake
LIST_S_PER_FILE = 0.005      # driver lists the files to read
FILE_OPEN_S = 0.05           # per file a scan task opens (small files hurt)
SCAN_ROW_S = 0.002           # per row read and shuffled
SCAN_PARTITION_MB = 128      # spark.sql.files.maxPartitionBytes

# Aggregation tasks
ROW_S = 0.01                 # per row aggregated (features + point-in-time snapshot)
TASK_OVERHEAD_S = 0.3        # scheduling, deserialisation, shuffle fetch set-up
ROW_MB = 0.25                # memory per row a task holds
MEMORY_FRACTION = 0.6        # spark.memory.fraction: execution + storage share of the heap
SPILL_PENALTY = 1.5          # extra time per unit of data spilled to disk (x row time)
OVERHEAD_HEADROOM = 4.0      # a task can spill up to this multiple of its memory; beyond,
                             # buffers and off-heap overhead exceed the container limit

# Adaptive query execution (AQE)
ADVISORY_MB = 64             # spark.sql.adaptive.advisoryPartitionSizeInBytes
SKEW_FACTOR = 5              # skewedPartitionFactor (x median)
SKEW_MIN_MB = 256            # skewedPartitionThresholdInBytes

# Speculative execution
SPECULATION_QUANTILE = 0.75  # share of a stage's tasks done before speculating
SPECULATION_MULTIPLIER = 1.5 # a task this many times slower than the median gets a copy

# Failures
MAX_STAGE_ATTEMPTS = 4       # spark.stage.maxConsecutiveAttempts
JOB_TIMEOUT_S = 24 * 3600     # the scheduler gives up on a job after a day

# Output
COMMIT_S = 10                # driver: write the new feature version, swap the pointer
COMMIT_ROW_S = 0.0005

# Price (money, console economy): per executor-hour.
PRICE_PER_CORE_HOUR = 0.5
PRICE_PER_GB_HOUR = 0.1


@dataclass
class ClusterConfig:
    executors: int = 4
    cores: int = 4                    # per executor (task slots)
    memory_gb: int = 8                # per executor
    shuffle_partitions: int = 200     # spark.sql.shuffle.partitions
    aqe: bool = True                  # coalesce small partitions, split skewed ones
    speculation: bool = True          # re-launch straggling tasks elsewhere
    max_task_failures: int = 4        # spark.task.maxFailures

    def task_memory_mb(self):
        return self.memory_gb * 1024 * MEMORY_FRACTION / max(1, self.cores)

    def price_per_hour(self):
        return self.executors * (self.cores * PRICE_PER_CORE_HOUR + self.memory_gb * PRICE_PER_GB_HOUR)

    def summary(self):
        return asdict(self)


@dataclass
class Conditions:
    """What the environment does to a job (faults; defaults = healthy)."""
    cluster_down: bool = False        # no executor can start
    executor_loss_mean_s: float = 0   # executors are reclaimed after ~this long (0 = never)
    slow_executor_factor: float = 1   # the first executor runs this many times slower
    queue_wait_s: float = 0           # time before the cluster grants resources
    granted_share: float = 1.0        # share of the requested executors granted
