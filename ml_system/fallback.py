"""Fallback retrieval: what the model server serves when the two-tower path
is down. Most-watched movies (Flink's Redis `item_popularity`), padded with
the catalog in id order. Deterministic, no model, no personalisation."""

from ml_system.model_server import RetrievalService


class PopularityRetrieval(RetrievalService):

    def __init__(self, redis, movie_ids):
        super().__init__()
        self.redis = redis
        self.movie_ids = sorted(movie_ids)

    def retrieve(self, request):
        popularity = {
            movie_id: value
            for (feature, movie_id), value in self.redis.store.items()
            if feature == "item_popularity"
        }
        ranked = sorted(popularity, key=lambda m: (-popularity[m], m))
        seen = set(ranked)
        ranked += [m for m in self.movie_ids if m not in seen]
        return ranked[:request.k]
