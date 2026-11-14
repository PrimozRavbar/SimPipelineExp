"""Fault injection: real failures for the player to detect and fix.

FaultInjector is a Service registered last, so a fault scheduled to start
at tick S is switched on at the end of tick S-1 and affects ticks S,
S+1, ... It uses no randomness itself; a random schedule ("chaos") is
drawn by the caller from a seeded RNG. Every fault can end on its own
(`duration`) or be fixed with `fix(kind)` (what a player's repair action
calls).

Kinds and what they break:

- flink_outage         Flink stops; events pile up in Kafka (consumer lag)
                       and online features go stale.
- feature_corruption   Serving reads user genre preferences under the
                       wrong genre names (a bad feature-transform deploy).
- index_id_mismatch    The ANN index returns the wrong movie ids (index rows
                       mapped to movie ids with an offset).
- model_server_outage  Two-tower retrieval is down; the model server serves
                       its popularity fallback.
- traffic_spike        Several times more users per tick.
- spark_failure        The Spark cluster is down: feature jobs fail (offline
                       features go stale, retrains that need them abort).
- spark_executor_loss  Spot executors are reclaimed mid-job (each lives ~10
                       minutes): tasks are retried and lost shuffle output
                       recomputed (slower jobs); a job fails once
                       max(2 x executors, 3) executors have been lost.
- spark_slow_node      One executor runs 10x slower (a degraded node):
                       stragglers, unless speculation re-runs them.
- spark_cluster_busy   The shared cluster is busy: jobs wait ~30 minutes
                       for resources and get half the executors.
- lake_ingestion_lag   The data lake sink is stuck: events wait in Kafka,
                       batch jobs see incomplete input.
- viral_movie          A movie goes viral in the World: everyone clicks and
                       watches it (hot key in every job, a shift in what
                       users want).
- gpu_preemption       Training GPUs are spot instances being reclaimed:
                       jobs restart from their last checkpoint (or from
                       scratch) and take longer, or fail.
- feature_nulls        A bad deploy of the Spark feature job writes empty
                       genre profiles for most users. The corrupted
                       snapshots stay in the store after the fault ends,
                       until a full backfill rewrites them.
- scheduler_down       The workflow scheduler (Airflow) is down: scheduled
                       Spark runs and retrains silently do not start.
"""

from ml_system.ann import ANNIndex
from ml_system.fallback import PopularityRetrieval
from simulation.service import Service
from world.external_data_loader import GENRE_NAMES

FAULT_KINDS = {
    "flink_outage": "Flink stream job down",
    "feature_corruption": "Online user features corrupted",
    "index_id_mismatch": "ANN index returns wrong movie ids",
    "model_server_outage": "Two-tower retrieval down (serving fallback)",
    "traffic_spike": "Traffic spike",
    "spark_failure": "Spark feature jobs failing",
    "kafka_broker_down": "Kafka broker 0 down",
    "spark_executor_loss": "Spark executors being reclaimed (spot)",
    "spark_slow_node": "Degraded Spark node (slow executor)",
    "spark_cluster_busy": "Spark cluster busy (resources pending)",
    "lake_ingestion_lag": "Data lake sink stuck (ingestion lag)",
    "viral_movie": "A movie went viral",
    "gpu_preemption": "Training GPUs being preempted (spot)",
    "feature_nulls": "Feature job writing empty profiles",
    "scheduler_down": "Workflow scheduler down",
}

# Which repair action ends which fault (see FaultInjector.fix).
FIXES = {
    "restart_flink": ("flink_outage", "feature_corruption"),
    "rebuild_index": ("index_id_mismatch",),
    "restart_model_server": ("model_server_outage",),
    "restart_kafka_broker": ("kafka_broker_down",),
    "replace_spark_node": ("spark_slow_node",),
    "restart_lake_sink": ("lake_ingestion_lag",),
    "restart_scheduler": ("scheduler_down",),
    "backfill_features": ("feature_nulls",),
}


class RemappedIndex(ANNIndex):
    """Wraps an index and maps every returned id through `mapping`: the
    neighbours are right, the labels are wrong."""

    def __init__(self, inner, mapping):
        self.inner = inner
        self.mapping = mapping

    def build(self, item_embeddings):
        self.inner.build(item_embeddings)

    def search(self, query_embedding, k):
        return [self.mapping.get(i, i) for i in self.inner.search(query_embedding, k)]


def rotate_genres(preferences, shift=5):
    """Move each genre's weight to the genre `shift` places later."""
    names = [g for g in GENRE_NAMES if g != "unknown"]
    index = {g: i for i, g in enumerate(names)}
    out = {}
    for genre, value in preferences.items():
        if genre in index:
            genre = names[(index[genre] + shift) % len(names)]
        out[genre] = out.get(genre, 0) + value
    return out


class Fault:

    def __init__(self, fault_id, kind, start, duration=None, **params):
        if kind not in FAULT_KINDS:
            raise ValueError("unknown fault kind %r" % kind)
        self.id = fault_id
        self.kind = kind
        self.start = start
        self.duration = duration
        self.params = params
        self.status = "scheduled"     # scheduled -> active -> ended
        self.ended_tick = None
        self.ended_by = None          # "timeout", a fix name, or "superseded"
        self._undo = None

    def summary(self):
        return {
            "id": self.id, "kind": self.kind, "title": FAULT_KINDS[self.kind],
            "start": self.start, "duration": self.duration, "status": self.status,
            "ended_tick": self.ended_tick, "ended_by": self.ended_by,
        }


class FaultInjector(Service):

    def __init__(self, system):
        super().__init__()
        self.system = system
        self.faults = []

    # -- scheduling ------------------------------------------------------

    def schedule(self, kind, start, duration=None, **params):
        fault = Fault("F-%d" % (len(self.faults) + 1), kind, start, duration, **params)
        self.faults.append(fault)
        return fault

    def active(self, kind=None):
        return [f for f in self.faults if f.status == "active" and (kind is None or f.kind == kind)]

    def process(self, sim):
        self.apply_due(sim.clock.now)

    def apply_due(self, now):
        """End faults that are over and start those due for tick now+1.
        Runs at the end of every tick; call it directly to make a fault
        scheduled at a tick boundary take effect from the next tick."""
        upcoming = now + 1
        for fault in self.faults:
            if fault.status == "active" and fault.duration is not None \
                    and upcoming >= fault.start + fault.duration:
                self._end(fault, now, "timeout")
        for fault in self.faults:
            if fault.status == "scheduled" and upcoming >= fault.start:
                self._start(fault)

    def fix(self, action, now):
        """Apply a repair action; returns the faults it ended."""
        kinds = FIXES.get(action, ())
        ended = [f for f in self.active() if f.kind in kinds]
        for fault in ended:
            self._end(fault, now, action)
        return ended

    # -- effects ---------------------------------------------------------

    def _start(self, fault):
        if self.active(fault.kind):
            # Same fault already running: this one is folded into it.
            fault.status = "ended"
            fault.ended_by = "superseded"
            fault.ended_tick = fault.start
            return
        fault.status = "active"
        fault._undo = getattr(self, "_start_" + fault.kind)(fault)
        self._log("fault_start", fault)

    def _end(self, fault, now, by):
        if fault._undo is not None:
            fault._undo()
        fault.status = "ended"
        fault.ended_tick = now
        fault.ended_by = by
        self._log("fault_end", fault)

    def _log(self, kind, fault):
        logger = self.system.get("metrics_logger")
        if logger is not None:
            logger.log_event(kind, self.system["sim"].clock.now, fault_id=fault.id, kind=fault.kind,
                             ended_by=fault.ended_by)

    def _start_flink_outage(self, fault):
        flink = self.system["sim"].get_service("Flink")
        flink.paused = True

        def undo():
            flink.paused = False
        return undo

    def _start_feature_corruption(self, fault):
        redis = self.system["redis"]
        shift = fault.params.get("shift", 5)
        redis.read_transforms["genre_preferences"] = lambda v: rotate_genres(v, shift)

        def undo():
            redis.read_transforms.pop("genre_preferences", None)
        return undo

    def _start_index_id_mismatch(self, fault):
        retriever = self.system["retriever"]
        inner = retriever.ann_index
        ids = sorted(self.system["world"].movie_objects)
        offset = fault.params.get("offset", 7)
        mapping = {m: ids[(i + offset) % len(ids)] for i, m in enumerate(ids)}
        broken = RemappedIndex(inner, mapping)
        retriever.ann_index = broken

        def undo():
            # A deploy may already have replaced the index; only unwrap ours.
            if retriever.ann_index is broken:
                retriever.ann_index = inner
        return undo

    def _start_model_server_outage(self, fault):
        server = self.system["model_server"]
        if server.fallback is None:
            server.fallback = PopularityRetrieval(self.system["redis"], self.system["world"].movie_objects)
        server.retriever_down = True

        def undo():
            server.retriever_down = False
        return undo

    def _start_spark_failure(self, fault):
        spark = self.system["spark"]
        spark.failing = True

        def undo():
            spark.failing = False
        return undo

    def _spark_conditions(self, **values):
        conditions = self.system["spark"].conditions
        previous = {k: getattr(conditions, k) for k in values}
        for k, v in values.items():
            setattr(conditions, k, v)

        def undo():
            for k, v in previous.items():
                setattr(conditions, k, v)
        return undo

    def _start_spark_executor_loss(self, fault):
        return self._spark_conditions(executor_loss_mean_s=fault.params.get("mean_seconds", 600))

    def _start_spark_slow_node(self, fault):
        return self._spark_conditions(slow_executor_factor=fault.params.get("factor", 10))

    def _start_spark_cluster_busy(self, fault):
        return self._spark_conditions(queue_wait_s=fault.params.get("wait_seconds", 1800),
                                      granted_share=fault.params.get("granted_share", 0.5))

    def _start_lake_ingestion_lag(self, fault):
        lake = self.system["datalake"]
        lake.paused = True

        def undo():
            lake.paused = False
        return undo

    def _start_viral_movie(self, fault):
        dynamics = self.system.get("world_dynamics")
        if dynamics is None:
            raise ValueError("viral_movie needs a world with WorldDynamics")
        now = self.system["sim"].clock.now
        movie_id = dynamics.start_viral(now, movie_id=fault.params.get("movie_id"), duration=0)
        fault.params["movie_id"] = movie_id

        def undo():
            if dynamics.viral is not None and dynamics.viral["movie_id"] == movie_id:
                dynamics.end_viral()
        return undo

    def _start_gpu_preemption(self, fault):
        conditions = self.system["training_conditions"]
        previous = conditions.preemption_mean_s
        conditions.preemption_mean_s = fault.params.get("mean_seconds", 1200)

        def undo():
            conditions.preemption_mean_s = previous
        return undo

    def _start_feature_nulls(self, fault):
        pipeline = self.system["offline_pipeline"].feature_pipeline
        pipeline.null_profile_share = fault.params.get("share", 0.6)

        def undo():
            # The job is fixed; what it already wrote stays until a backfill.
            pipeline.null_profile_share = 0.0
        return undo

    def _start_scheduler_down(self, fault):
        scheduler = self.system["scheduler"]
        scheduler["up"] = False

        def undo():
            scheduler["up"] = True
        return undo

    def _start_kafka_broker_down(self, fault):
        """Broker 0 stops: partitions with no other live replica stall
        their consumers (Flink, the CTR tracker) until it is back."""
        kafka = self.system["sim"].get_service("Kafka")
        broker = fault.params.get("broker", 0)
        kafka.broker_down(broker)

        def undo():
            kafka.broker_up(broker)
        return undo

    def _start_traffic_spike(self, fault):
        world = self.system["world"]
        before = world.users_per_tick
        world.users_per_tick = max(before + 1, int(before * fault.params.get("factor", 3)))

        def undo():
            world.users_per_tick = before
        return undo
