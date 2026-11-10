from simulation.service import Service

class Redis(Service):

    def __init__(self):
        super().__init__()
        self.store = {}

        # feature name -> fn(value) applied on read. Empty in normal
        # operation; fault injection uses it to corrupt what serving sees.
        self.read_transforms = {}

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
        value = self.store.get(
            (feature, entity_id)
        )
        transform = self.read_transforms.get(feature)
        if transform is not None and value is not None:
            value = transform(value)
        return value

    def set(self, feature, entity_id, value):
        self.store[(feature, entity_id)] = value

