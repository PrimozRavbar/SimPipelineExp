from simulation.service import Service

class DataLake(Service):

    def __init__(self):
        super().__init__()
        self.events = []

    def receive(self, event):
        self.events.append(event)

    def start(self, sim):
        super().start(sim)

        kafka = sim.get_service("Kafka")

        kafka.subscribe(
            "DataLake",
            "RecommendationShownEvent"
        )

        kafka.subscribe(
            "DataLake",
            "RecommendationClickedEvent"
        )

        kafka.subscribe(
            "DataLake",
            "WatchEvent"
        )

        kafka.subscribe(
            "DataLake",
            "RatingEvent"
        )

    def process(self, sim):

        kafka = sim.get_service("Kafka")

        for topic in [
            "RecommendationShownEvent",
            "RecommendationClickedEvent",
            "WatchEvent",
            "RatingEvent"
        ]:
            events = kafka.poll(
                "DataLake",
                topic
            )

            for event in events:
                self.events.append(event)

