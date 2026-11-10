from simulation.service import Service

class Kafka(Service):

    def __init__(self):
        super().__init__()
        self.topics = {}
        self.offsets = {}
        self.subscribers = {}

    def receive(self, event):

        topic = type(event).__name__

        if topic not in self.topics:
            self.topics[topic] = []

        self.topics[topic].append(event)

    def subscribe(self, consumer_name, topic):
        self.subscribers.setdefault(topic, []).append(consumer_name)
        self.offsets[(consumer_name, topic)] = 0


    def poll(self, consumer_name, topic, max_records=None):
        """Messages after the consumer's offset (at most `max_records`),
        and advance the offset past them."""

        offset = self.offsets[(consumer_name, topic)]

        end = None if max_records is None else offset + max_records
        messages = self.topics.get(topic, [])[offset:end]

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
            total += len(self.topics.get(topic, [])) - offset
        return total

