"""Kafka: topics with partitions, replication, retention and consumer offsets.

The defaults behave exactly like the original single-log Kafka: every
message is kept, every partition is available, and a consumer's poll()
returns the topic's messages in publish order. The operational knobs, all
off or neutral by default:

- partitions: messages are keyed by user id (partition = user_id %
  partitions). Order is per partition in real Kafka; here the log keeps the
  global publish order and records each message's partition, so per-
  partition counts and lag can be reported without changing what
  consumers read.
- brokers, replication_factor: partition p is led by broker p % brokers and
  replicated on the next replication_factor - 1 brokers. A partition with
  no live replica is unavailable: a consumer's poll stops at the first
  message of an unavailable partition (it keeps the topic's order, so it
  waits), and lag builds up until the broker is back.
- retention_ticks: messages older than this are deleted at the end of each
  tick (0 = keep forever). A consumer whose offset falls behind the log
  start loses those messages (counted in `lost`) and resumes at the start.

Offsets are logical (they never shrink when old messages are deleted).
"""

from simulation.service import Service


class Kafka(Service):

    def __init__(self, partitions=3, replication_factor=1, brokers=3, retention_ticks=0):
        super().__init__()
        self.topics = {}             # topic -> retained messages, publish order
        self.offsets = {}            # (consumer, topic) -> logical offset
        self.subscribers = {}
        self.partitions = partitions
        self.replication_factor = replication_factor
        self.brokers = brokers
        self.retention_ticks = retention_ticks
        self.down_brokers = set()
        self.log_start = {}          # topic -> logical offset of topics[topic][0]
        self.message_partitions = {} # topic -> partition of each retained message
        self.published = {}          # topic -> messages ever published
        self.deleted = {}            # topic -> messages deleted by retention
        self.lost = {}               # (consumer, topic) -> messages it never read (retention)
        self.partition_counts = {}   # topic -> {partition: retained messages}
        self.blocked_polls = 0       # polls that stopped at an unavailable partition

    # -- producing -----------------------------------------------------------

    def partition_of(self, event):
        key = getattr(event, "user_id", 0) or 0
        return key % max(1, self.partitions)

    def receive(self, event):

        topic = type(event).__name__

        if topic not in self.topics:
            self.topics[topic] = []
            self.message_partitions[topic] = []
            self.log_start[topic] = 0

        self.topics[topic].append(event)
        partition = self.partition_of(event)
        self.message_partitions[topic].append(partition)
        counts = self.partition_counts.setdefault(topic, {})
        counts[partition] = counts.get(partition, 0) + 1
        self.published[topic] = self.published.get(topic, 0) + 1

    # -- consuming -----------------------------------------------------------

    def subscribe(self, consumer_name, topic, from_end=False, from_tick=None):
        """from_end=True starts at the newest message (a consumer that only
        wants what is published from now on); from_tick=T starts at the
        first message with timestamp >= T (messages are in time order)."""
        self.subscribers.setdefault(topic, []).append(consumer_name)
        messages = self.topics.get(topic, [])
        start = self.log_start.get(topic, 0)
        if from_tick is not None:
            index = next((i for i, e in enumerate(messages) if e.timestamp >= from_tick), len(messages))
        elif from_end:
            index = len(messages)
        else:
            index = 0
        self.offsets[(consumer_name, topic)] = start + index

    def _index(self, consumer_name, topic):
        """Position of the consumer's offset in the retained log (moving it
        to the log start if retention deleted what it had not read)."""
        start = self.log_start.get(topic, 0)
        offset = self.offsets[(consumer_name, topic)]
        if offset < start:
            key = (consumer_name, topic)
            self.lost[key] = self.lost.get(key, 0) + start - offset
            self.offsets[key] = offset = start
        return offset - start

    def poll(self, consumer_name, topic, max_records=None):
        """Messages after the consumer's offset (at most `max_records`),
        and advance the offset past them."""

        index = self._index(consumer_name, topic)
        end = None if max_records is None else index + max_records
        messages = self.topics.get(topic, [])[index:end]

        if self.down_brokers and messages:
            parts = self.message_partitions[topic][index:index + len(messages)]
            for i, p in enumerate(parts):
                if not self.partition_available(p):
                    messages = messages[:i]
                    self.blocked_polls += 1
                    break

        self.offsets[(consumer_name, topic)] += len(messages)

        return messages

    def lag(self, consumer_name, topics=None):
        """Messages published but not yet polled by this consumer, summed
        over its subscribed topics (or the given ones)."""
        total = 0
        for (consumer, topic), offset in self.offsets.items():
            if consumer != consumer_name:
                continue
            if topics is not None and topic not in topics:
                continue
            end = self.log_start.get(topic, 0) + len(self.topics.get(topic, []))
            total += end - max(offset, self.log_start.get(topic, 0))
        return total

    # -- brokers and partitions ----------------------------------------------

    def replicas(self, partition):
        n = max(1, self.brokers)
        return [(partition + i) % n for i in range(min(self.replication_factor, n))]

    def partition_available(self, partition):
        return any(b not in self.down_brokers for b in self.replicas(partition))

    def unavailable_partitions(self):
        return [p for p in range(self.partitions) if not self.partition_available(p)]

    def broker_down(self, broker):
        self.down_brokers.add(broker % max(1, self.brokers))

    def broker_up(self, broker=None):
        if broker is None:
            self.down_brokers.clear()
        else:
            self.down_brokers.discard(broker)

    # -- retention -----------------------------------------------------------

    def process(self, sim):
        if not self.retention_ticks:
            return
        cutoff = sim.clock.now - self.retention_ticks
        for topic, messages in self.topics.items():
            drop = 0
            while drop < len(messages) and messages[drop].timestamp < cutoff:
                drop += 1
            if drop:
                counts = self.partition_counts.get(topic, {})
                for p in self.message_partitions[topic][:drop]:
                    counts[p] -= 1
                del messages[:drop]
                del self.message_partitions[topic][:drop]
                self.log_start[topic] = self.log_start.get(topic, 0) + drop
                self.deleted[topic] = self.deleted.get(topic, 0) + drop

    # -- reporting -----------------------------------------------------------

    def describe(self):
        topics = {}
        for topic, messages in self.topics.items():
            counts = self.partition_counts.get(topic, {})
            per_partition = [counts.get(p, 0) for p in range(max(self.partitions, max(counts, default=-1) + 1))]
            topics[topic] = {"retained": len(messages), "published": self.published.get(topic, 0),
                             "deleted": self.deleted.get(topic, 0), "per_partition": per_partition}
        consumers = {}
        for (consumer, topic), offset in self.offsets.items():
            end = self.log_start.get(topic, 0) + len(self.topics.get(topic, []))
            c = consumers.setdefault(consumer, {"lag": 0, "lost": 0})
            c["lag"] += end - max(offset, self.log_start.get(topic, 0))
            c["lost"] += self.lost.get((consumer, topic), 0)
        return {"partitions": self.partitions, "replication_factor": self.replication_factor,
                "brokers": self.brokers, "down_brokers": sorted(self.down_brokers),
                "unavailable_partitions": self.unavailable_partitions(),
                "retention_ticks": self.retention_ticks, "blocked_polls": self.blocked_polls,
                "topics": topics, "consumers": consumers}
