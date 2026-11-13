"""Online metrics computed from the live event stream, and the per-run
metrics log (runs/<run_id>/metrics.jsonl) the dashboard reads."""

import json
import random
import time
from pathlib import Path

from simulation.service import Service


SHOWN_TOPIC = "RecommendationShownEvent"
CLICKED_TOPIC = "RecommendationClickedEvent"


class CTRTracker(Service):
    """Click-through rate per simulation tick.

    Consumes shown/clicked events from Kafka like any other downstream
    service and buckets them by the tick they happened in (the event
    timestamp), so consumer lag never shifts a click into the wrong tick.
    CTR = clicked slates / shown slates. A slate is clicked at most once
    under the cascade click model.

    It uses no randomness, so registering it does not change a seeded run.
    """

    def __init__(self):
        super().__init__()

        # tick -> [shown, clicks]
        self.counts = {}

        # tick -> set of user ids shown a slate in that tick
        self.users = {}

    def start(self, sim):
        super().start(sim)

        kafka = sim.get_service("Kafka")
        kafka.subscribe(self.name, SHOWN_TOPIC)
        kafka.subscribe(self.name, CLICKED_TOPIC)

    def process(self, sim):
        kafka = sim.get_service("Kafka")

        for event in kafka.poll(self.name, SHOWN_TOPIC):
            self.record(event.timestamp, shown=1)
            self.users.setdefault(event.timestamp, set()).add(event.user_id)

        for event in kafka.poll(self.name, CLICKED_TOPIC):
            self.record(event.timestamp, clicks=1)

    def record(self, tick, shown=0, clicks=0):
        bucket = self.counts.setdefault(tick, [0, 0])
        bucket[0] += shown
        bucket[1] += clicks

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def ticks(self):
        """Every tick from the first to the last one seen, gaps included."""
        if not self.counts:
            return []
        return list(range(min(self.counts), max(self.counts) + 1))

    def per_tick(self):
        """List of dicts: tick, shown, clicks, ctr (None if nothing shown)."""
        rows = []

        for tick in self.ticks():
            shown, clicks = self.counts.get(tick, (0, 0))
            rows.append({
                "tick": tick,
                "shown": shown,
                "clicks": clicks,
                "ctr": clicks / shown if shown else None,
                "active_users": len(self.users.get(tick, ())),
            })

        return rows

    def ctr_between(self, start=None, end=None):
        """CTR over ticks in [start, end] (inclusive, None = open)."""
        shown = clicks = 0

        for tick, (tick_shown, tick_clicks) in self.counts.items():
            if start is not None and tick < start:
                continue
            if end is not None and tick > end:
                continue
            shown += tick_shown
            clicks += tick_clicks

        return clicks / shown if shown else None

    def overall_ctr(self):
        return self.ctr_between()

    def rolling_ctr(self, window=100):
        """Trailing-window CTR per tick: list of (tick, ctr or None).

        With one active user per tick, raw per-tick CTR is 0 or 1, so a
        window of ~100 ticks is what you want to plot.
        """
        if window < 1:
            raise ValueError("window must be >= 1")

        ticks = self.ticks()
        series = []
        shown_sum = clicks_sum = 0

        for i, tick in enumerate(ticks):
            shown, clicks = self.counts.get(tick, (0, 0))
            shown_sum += shown
            clicks_sum += clicks

            if i >= window:
                old_shown, old_clicks = self.counts.get(ticks[i - window], (0, 0))
                shown_sum -= old_shown
                clicks_sum -= old_clicks

            series.append((tick, clicks_sum / shown_sum if shown_sum else None))

        return series


# ---------------------------------------------------------------------
# Reference CTRs (World-side ground truth, for charts and scoring)
# ---------------------------------------------------------------------

def reference_ctrs(world, slate_size=10, seed=0, random_slates_per_user=3):
    """Expected CTR, averaged over all users, of two reference slates:
    "oracle" (the slate_size most relevant movies, best first) and
    "random" (uniform random slates, seeded). Exact under the World's
    cascade click model; uses the ground truth, so it is for evaluation
    only, never for serving.
    """
    import numpy as np

    from world.relevance import relevance_matrix

    simulator = world.user_simulator
    user_ids = list(world.user_objects)
    movie_ids = list(world.movie_objects)

    rel = relevance_matrix(
        [world.user_states[user_id] for user_id in user_ids],
        [world.movie_objects[movie_id] for movie_id in movie_ids],
        taste=getattr(world, "taste", None),
    )

    rng = random.Random(seed)
    oracle = random_total = 0.0

    for row, user_id in enumerate(user_ids):
        # Most relevant first; ties broken by movie order.
        best = np.argsort(-rel[row], kind="stable")[:slate_size]
        oracle += simulator.expected_click_rate(user_id, [movie_ids[j] for j in best])

        for _ in range(random_slates_per_user):
            slate = rng.sample(movie_ids, slate_size)
            random_total += simulator.expected_click_rate(user_id, slate) / random_slates_per_user

    return {
        "oracle_ctr": oracle / len(user_ids),
        "random_ctr": random_total / len(user_ids),
    }


# ---------------------------------------------------------------------
# Metrics log
# ---------------------------------------------------------------------

METRICS_FILE = "metrics.jsonl"


class MetricsLogger(Service):
    """Appends one JSON object per line to <run_dir>/metrics.jsonl.

    Record types (all have "type"; tick-based ones have "tick"):

    - run_start   run_id, seed, created_at (wall clock), config, reference
                  {random_ctr, oracle_ctr}, model_version (0 = bootstrap)
    - tick        tick, shown, clicks, ctr (null if nothing shown),
                  active_users, model_version
    - retrain     tick, training_examples, model_version (the new one)
    - deploy      tick, model_version
    - evaluation  tick, plus whatever is passed (e.g. k, recall_at_k,
                  random_recall_at_k, popularity_recall_at_k)

    It reads per-tick counts from the CTRTracker (register it after the
    tracker so a tick's events are counted before the row is written). A
    line is written every tick, so the file is always up to date. Uses no
    randomness, so logging does not change a seeded run.
    """

    def __init__(self, run_dir, ctr_tracker, run_info=None):
        super().__init__()

        self.run_dir = Path(run_dir)
        self.path = self.run_dir / METRICS_FILE
        self.ctr_tracker = ctr_tracker
        self.run_info = dict(run_info or {})
        self.model_version = 0

    def start(self, sim):
        super().start(sim)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        # A run directory holds one run: start a fresh log.
        self.path.write_text("")

        self.write({
            "type": "run_start",
            "tick": sim.clock.now,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_version": self.model_version,
            **self.run_info,
        })

    def process(self, sim):
        tick = sim.clock.now
        shown, clicks = self.ctr_tracker.counts.get(tick, (0, 0))

        self.write({
            "type": "tick",
            "tick": tick,
            "shown": shown,
            "clicks": clicks,
            "ctr": clicks / shown if shown else None,
            "active_users": len(self.ctr_tracker.users.get(tick, ())),
            "model_version": self.model_version,
        })

    def log_event(self, event_type, tick, **fields):
        self.write({"type": event_type, "tick": tick, **fields})

    def log_retrain(self, tick, training_examples=None):
        """Record a retrain and the deploy of the resulting model."""
        self.model_version += 1
        self.log_event(
            "retrain", tick,
            training_examples=training_examples,
            model_version=self.model_version,
        )
        self.log_event("deploy", tick, model_version=self.model_version)

    def write(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def read_metrics(run_dir):
    """All records of <run_dir>/metrics.jsonl, in order."""
    path = Path(run_dir) / METRICS_FILE

    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
