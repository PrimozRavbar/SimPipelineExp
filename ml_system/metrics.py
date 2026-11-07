"""Online metrics computed from the live event stream."""

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

    def start(self, sim):
        super().start(sim)

        kafka = sim.get_service("Kafka")
        kafka.subscribe(self.name, SHOWN_TOPIC)
        kafka.subscribe(self.name, CLICKED_TOPIC)

    def process(self, sim):
        kafka = sim.get_service("Kafka")

        for event in kafka.poll(self.name, SHOWN_TOPIC):
            self.record(event.timestamp, shown=1)

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
