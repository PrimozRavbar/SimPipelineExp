from simulation.service import Service

class Redis(Service):

    def __init__(self):
        super().__init__()
        self.store = {}

    def receive(self, feature):

        if "user_id" in feature:
            key = (
                feature["feature"],
                feature["user_id"]
            )

        elif "movie_id" in feature:
            key = (
                feature["feature"],
                feature["movie_id"]
            )

        self.store[key] = feature["value"]

    def get(self, feature, entity_id):
        return self.store.get(
            (feature, entity_id)
        )

    def set(self, feature, entity_id, value):
        self.store[(feature, entity_id)] = value

