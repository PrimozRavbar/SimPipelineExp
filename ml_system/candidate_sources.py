"""Extra candidate sources merged with two-tower retrieval.

Real systems retrieve from several sources (embeddings, co-occurrence,
content, popularity...) and let one ranker order the union. Here:

ProfileRetrieval: content-based retrieval from the user's learned genre
profile (the same online features the towers see). Score = sum of the
user's genre shares over the movie's genres / sqrt(#genres) (the
`genre_match` of ml_system.ranking), ties broken by popularity (offline
feature store), then by id. Users without a profile get nothing from it.

merge_candidates: round-robin merge of the primary list with extra
sources, without duplicates, cut to k.
"""

import numpy as np

from ml_system.model_server import RetrievalService
from world.external_data_loader import GENRE_NAMES


class ProfileRetrieval(RetrievalService):

    def __init__(self, user_features_fn, feature_store, movie_objects):
        super().__init__()
        self.user_features_fn = user_features_fn
        self.feature_store = feature_store
        self.movie_ids = sorted(movie_objects)
        index = {g: i for i, g in enumerate(GENRE_NAMES)}
        matrix = np.zeros((len(self.movie_ids), len(GENRE_NAMES)))
        for row, movie_id in enumerate(self.movie_ids):
            genres = [g for g in set(movie_objects[movie_id].genres) if g in index]
            for genre in genres:
                matrix[row, index[genre]] = 1.0 / np.sqrt(len(genres))
        self.matrix = matrix
        self.genre_index = index

    def retrieve(self, request):
        prefs = (self.user_features_fn(request.user_id) or {}).get("genre_preferences") or {}
        total = sum(max(v, 0.0) for v in prefs.values())
        if total <= 0:
            return []
        vector = np.zeros(len(GENRE_NAMES))
        for genre, value in prefs.items():
            if genre in self.genre_index:
                vector[self.genre_index[genre]] = max(value, 0.0) / total
        scores = self.matrix @ vector
        popularity = np.array([float((self.feature_store.get_item_features(m) or {}).get("popularity", 0) or 0)
                               for m in self.movie_ids])
        ids = np.array(self.movie_ids)
        order = np.lexsort((ids, -popularity, -scores))[:request.k]
        return [int(ids[i]) for i in order if scores[i] > 0]


def merge_candidates(primary, extras, k):
    """Round-robin merge: primary[0], extra1[0], extra2[0], primary[1], ...
    without duplicates, cut to k."""
    lists = [list(primary)] + [list(extra) for extra in extras]
    merged, seen = [], set()
    position = 0
    while len(merged) < k and any(position < len(lst) for lst in lists):
        for lst in lists:
            if position < len(lst) and lst[position] not in seen:
                seen.add(lst[position])
                merged.append(lst[position])
                if len(merged) == k:
                    break
        position += 1
    return merged
