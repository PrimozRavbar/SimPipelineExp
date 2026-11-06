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


    def poll(self, consumer_name, topic):

        offset = self.offsets[(consumer_name, topic)]

        messages = self.topics.get(topic, [])[offset:]

        self.offsets[(consumer_name, topic)] += len(messages)

        return messages

