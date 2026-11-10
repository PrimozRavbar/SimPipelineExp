"""Health checks the console runs on the live system, the way an on-call
dashboard would. They read only ML-system state (never World ground
truth) and use no randomness.

- index_self_test: ANN recall against exact (brute-force) search for a
  sample of movies. A wrong id mapping fails it.
- feature_skew: for recently served users, how far the online (Redis,
  what serving reads) genre preferences are from the offline ones (feature
  store, last Spark run). Total-variation distance, 0..1.
- coverage: share of the catalog shown in a window.
- genre_psi: population stability index between two genre distributions
  (the console compares recent clicks with the first clicks after the
  current model's deploy, so the model's own effect on what is clicked
  doesn't count as drift).
"""

import math

from world.external_data_loader import GENRE_NAMES


def index_self_test(index, movie_ids, sample=40, k=5, tolerance=0.02):
    """ANN recall check against exact search: for a sample of movies, query
    the serving index with the movie's stored embedding and count it as a
    pass when the best of the top k hits is (within `tolerance` cosine) as
    close as the exact nearest neighbour. Healthy HNSW scores ~0.9-1.0; a
    broken id mapping returns movies whose vectors don't match the query.
    Returns the pass rate, or None if the index has no stored embeddings."""
    base = index
    while hasattr(base, "inner"):
        base = base.inner
    nodes = getattr(base, "nodes", None)
    if not nodes:
        return None
    ids = [m for m in sorted(movie_ids) if m in nodes]
    if not ids:
        return None
    import numpy as np

    matrix = np.array([nodes[m].unit for m in ids])
    position = {m: i for i, m in enumerate(ids)}
    step = max(1, len(ids) // sample)
    probe = ids[::step][:sample]

    hits = 0
    for movie_id in probe:
        row = position[movie_id]
        scores = matrix @ matrix[row]
        scores[row] = -np.inf
        exact = scores.max()
        found = [m for m in index.search(nodes[movie_id].embedding, k + 1)
                 if m in position and m != movie_id][:k]
        if found and max(scores[position[m]] for m in found) >= exact - tolerance:
            hits += 1
    return hits / len(probe)


def _normalise(preferences):
    total = sum(max(v, 0.0) for v in (preferences or {}).values())
    if total <= 0:
        return {}
    return {g: max(v, 0.0) / total for g, v in preferences.items()}


def feature_skew(redis, feature_store, user_ids):
    """Mean total-variation distance over users that have both an online
    and an offline genre profile; None if there are none."""
    distances = []
    for user_id in user_ids:
        online = _normalise(redis.get("genre_preferences", user_id))
        offline = _normalise((feature_store.get_user_features(user_id) or {}).get("genre_preferences"))
        if not online or not offline:
            continue
        genres = set(online) | set(offline)
        distances.append(0.5 * sum(abs(online.get(g, 0) - offline.get(g, 0)) for g in genres))
    return sum(distances) / len(distances) if distances else None


def genre_distribution(movie_ids, movie_objects, smoothing=0.5):
    """Genre shares of the given movies, with additive smoothing so small
    samples don't blow up the PSI."""
    counts = {g: smoothing for g in GENRE_NAMES}
    for movie_id in movie_ids:
        for genre in movie_objects[movie_id].genres:
            counts[genre] = counts.get(genre, smoothing) + 1.0
    total = sum(counts.values())
    return {g: c / total for g, c in counts.items()}


def genre_psi(expected, actual):
    """Population stability index; > 0.2 is the usual 'shifted' line."""
    if not expected or not actual:
        return None
    return sum((actual[g] - expected[g]) * math.log(actual[g] / expected[g]) for g in expected)
