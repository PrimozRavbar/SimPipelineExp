"""Health checks the console runs on the live system, the way an on-call
dashboard would. They read only ML-system state (never World ground
truth) and use no randomness.

- golden_queries / index_self_test: probe queries recorded when an index
  is built ("golden queries"); the serving index must still return the
  same ids. A wrong id mapping fails it; a healthy index scores 100%.
  (An earlier version compared against exact search, which gave false
  alarms: trained item embeddings are nearly collapsed, so a healthy HNSW
  index matched exact search for only ~75-95% of probes.)
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


def golden_queries(index, movie_ids, sample=40, k=5):
    """Probe queries with their expected answers, recorded from a freshly
    built (trusted) index: a sample of movies' own stored embeddings and
    the top k ids the index returns for each. Returns [] if the index
    exposes no stored embeddings."""
    base = index
    while hasattr(base, "inner"):
        base = base.inner
    nodes = getattr(base, "nodes", None)
    if not nodes:
        return []
    ids = [m for m in sorted(movie_ids) if m in nodes]
    step = max(1, len(ids) // sample)
    return [(nodes[m].embedding, index.search(nodes[m].embedding, k)) for m in ids[::step][:sample]]


def index_self_test(index, golden):
    """Share of golden queries for which the serving index still returns
    exactly the recorded ids. HNSW search is deterministic, so a healthy
    index scores 1.0; a broken id mapping scores ~0. None without probes."""
    if not golden:
        return None
    k = len(golden[0][1])
    return sum(index.search(query, k) == expected for query, expected in golden) / len(golden)


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
