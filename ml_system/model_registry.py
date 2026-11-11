"""Model registry: versioned model bundles that can be saved and loaded.

A *bundle* is everything serving needs from one training run: user tower,
item tower, the ANN index's item embeddings, the optional ranker, and
metadata (training stats, stage history). Keeping them together is what
prevents a user tower from being served against another model's index.

export_bundle / load_bundle write and read a bundle directory:

    towers.pt       state dicts + constructor config of both towers
    index.json      {movie_id: embedding} the index was built from
    ranker.json     ranker parameters (if any)
    metadata.json   free-form metadata
"""

import json
from pathlib import Path


class ModelRegistry:
    """In-memory name -> artifact store (kept for existing callers)."""

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


def tower_config(tower):
    """Constructor arguments that rebuild a SimpleUserTower/SimpleItemTower."""
    name = type(tower).__name__
    out_dim = tower.network[-1].out_features
    if name == "SimpleUserTower":
        return {"class": name, "num_users": tower.user_embedding.num_embeddings - 1,
                "embedding_dim": out_dim, "use_id_embedding": tower.use_id_embedding,
                "use_activity_features": tower.use_activity_features,
                "use_history": getattr(tower, "use_history", False),
                "num_movies": getattr(tower, "num_movies", None)}
    if name == "SimpleItemTower":
        return {"class": name, "num_movies": tower.movie_embedding.num_embeddings - 1,
                "embedding_dim": out_dim, "use_id_embedding": tower.use_id_embedding,
                "use_rating_feature": tower.use_rating_feature,
                "use_popularity_feature": tower.use_popularity_feature}
    raise ValueError("cannot export tower type %s" % name)


def index_embeddings(index):
    """{movie_id: embedding} stored in an HNSW index (unwrapping wrappers)."""
    base = index
    while hasattr(base, "inner"):
        base = base.inner
    return {item_id: list(node.embedding) for item_id, node in base.nodes.items()}


def export_bundle(directory, user_tower, item_tower, index, ranker=None, metadata=None):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({
        "user_tower": {"config": tower_config(user_tower), "state": user_tower.state_dict()},
        "item_tower": {"config": tower_config(item_tower), "state": item_tower.state_dict()},
    }, directory / "towers.pt")
    (directory / "index.json").write_text(json.dumps(
        {str(k): v for k, v in sorted(index_embeddings(index).items())}))
    if ranker is not None and ranker.weights is not None:
        (directory / "ranker.json").write_text(json.dumps({
            "weights": ranker.weights, "bias": ranker.bias, "mean": ranker.mean, "scale": ranker.scale,
            "position_weights": getattr(ranker, "position_weights", None),
            "training_rows": ranker.training_rows, "positive_rate": ranker.positive_rate,
        }, indent=1))
    (directory / "metadata.json").write_text(json.dumps(metadata or {}, indent=1, sort_keys=True, default=str))
    return directory


def load_bundle(directory, index_seed=0):
    """Rebuild towers (eval mode), a fresh HNSW index over the stored
    embeddings, and return them with ranker parameters and metadata."""
    import torch

    from ml_system.ann import HNSWIndex
    from ml_system.two_tower import SimpleItemTower, SimpleUserTower

    directory = Path(directory)
    towers = torch.load(directory / "towers.pt", weights_only=False)
    built = {}
    for key, cls in (("user_tower", SimpleUserTower), ("item_tower", SimpleItemTower)):
        config = dict(towers[key]["config"])
        config.pop("class")
        tower = cls(**config)
        tower.load_state_dict(towers[key]["state"])
        built[key] = tower.eval()

    embeddings = {int(k): v for k, v in json.loads((directory / "index.json").read_text()).items()}
    index = HNSWIndex(seed=index_seed)
    index.build(embeddings)

    ranker_path = directory / "ranker.json"
    return {
        **built,
        "index": index,
        "ranker": json.loads(ranker_path.read_text()) if ranker_path.exists() else None,
        "metadata": json.loads((directory / "metadata.json").read_text()),
    }
