"""DataLake: the raw event log every offline job reads.

Each tick the lake sink consumes its Kafka topics and writes one file per
topic with that tick's events (like a streaming sink writing Parquet files
partitioned by time). `files` records them as [first, end, tick, topic,
compacted] over the append-only `events` list, so Spark can cost a scan by
the files it opens (many small files are slow to list and open) and a
compaction job can merge them. The events themselves never change.

`paused` (fault injection: the lake sink is stuck) stops ingestion: the
events wait in Kafka (consumer lag) and `ingested_tick` stops moving, so a
batch job can check whether its input is complete before it runs.
"""

from bisect import bisect_right

from simulation.service import Service

TOPICS = [
    "RecommendationShownEvent",
    "RecommendationClickedEvent",
    "WatchEvent",
    "RatingEvent",
    "OnboardingEvent",
    "BrowseEvent",
    "InterestEvent",
]

# Target rows per compacted file (~128 MB at the Spark cost model's scale).
COMPACT_FILE_ROWS = 512


class DataLake(Service):

    def __init__(self):
        super().__init__()
        self.events = []
        self.files = []              # [first, end, tick, topic, compacted]
        self.paused = False
        self.ingested_tick = -1      # data up to this tick has landed
        self.compactions = 0

    def receive(self, event):
        self.events.append(event)

    def start(self, sim):
        super().start(sim)

        kafka = sim.get_service("Kafka")

        for topic in TOPICS:
            kafka.subscribe("DataLake", topic)

    def process(self, sim):

        if self.paused:
            return

        kafka = sim.get_service("Kafka")

        for topic in TOPICS:
            events = kafka.poll(
                "DataLake",
                topic
            )

            if events:
                first = len(self.events)
                self.events.extend(events)
                self.files.append([first, len(self.events), sim.clock.now, topic, False])

        self.ingested_tick = sim.clock.now

    # -- maintenance and reads ------------------------------------------

    def compact(self, before_tick):
        """Merge the small files of ticks < before_tick into files of about
        COMPACT_FILE_ROWS rows (Delta OPTIMIZE / Iceberg rewrite). Only
        file metadata changes. Returns (files before, files after)."""
        before = len(self.files)
        out, run = [], None
        for f in self.files:
            small = f[2] < before_tick and f[1] - f[0] < COMPACT_FILE_ROWS
            if small and run is not None and run[1] == f[0] and run[1] - run[0] + f[1] - f[0] <= COMPACT_FILE_ROWS:
                run[1] = f[1]
                run[2] = f[2]
                continue
            if run is not None:
                out.append(run)
                run = None
            if small:
                run = [f[0], f[1], f[2], "compacted", True]
            else:
                out.append(f)
        if run is not None:
            out.append(run)
        self.files = out
        self.compactions += 1
        return before, len(self.files)

    def file_rows(self, start, end=None):
        """Row counts of the files a job reading events[start:end] opens
        (a file partly in range is read whole)."""
        end = len(self.events) if end is None else end
        firsts = [f[0] for f in self.files]
        i = max(0, bisect_right(firsts, start) - 1)
        rows = []
        for f in self.files[i:]:
            if f[0] >= end:
                break
            if f[1] > start:
                rows.append(f[1] - f[0])
        covered = sum(rows)
        if not rows and end > start:
            rows = [end - start]     # events written without files (receive)
        elif covered < end - start:
            rows.append(end - start - covered)
        return rows

    def summary(self):
        return {"events": len(self.events), "files": len(self.files),
                "small_files": sum(1 for f in self.files if f[1] - f[0] < COMPACT_FILE_ROWS),
                "ingested_tick": self.ingested_tick, "paused": self.paused, "compactions": self.compactions}
