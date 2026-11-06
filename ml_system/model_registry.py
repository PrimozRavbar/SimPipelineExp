class ModelRegistry:

    def __init__(self):
        self.artifacts = {}

    def save(self, name, artifact):
        self.artifacts[name] = artifact

    def load(self, name):
        return self.artifacts[name]


class OfflineArtifacts:

    def __init__(
        self,
        user_tower,
        item_tower,
        hnsw_index
    ):
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.hnsw_index = hnsw_index
