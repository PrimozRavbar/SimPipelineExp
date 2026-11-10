"""Ranking and re-ranking stages after retrieval.

Retrieval (two-tower + ANN) finds ~100 candidates cheaply; a ranker then
re-orders them with features the towers don't use, and a re-ranker applies
rules (here: drop movies the user already watched). Both plug into
ModelServer(ranker=..., reranker=...); both are off by default.

LogisticRanker
--------------
Logistic regression on ML-side features only (never World ground truth):

- genre_match     user's learned genre preferences (online, as serving sees
                  them) against the movie's genres
- has_profile     the user has any genre preferences yet
- log_popularity  log(1 + watches) from the offline feature store
- item_rating     the movie's average rating (0 if unrated)
- position_k      one-hot slate position (training only)

Training rows come from logged slates: a click is a positive; under the
cascade the movies shown above it were examined and skipped, so they are
negatives; slates without a click give negatives at every position. Slate
position is a feature in training and fixed at serving ("position as a
feature"): the model learns how much of a click is due to being shown
first, instead of crediting the features of whatever was shown first.
Features for each row are looked up as of the event (point in time).

Fitting is deterministic full-batch gradient descent in numpy.
"""

import math
from abc import ABC, abstractmethod

from simulation.events import RecommendationClickedEvent, RecommendationShownEvent


class Ranker(ABC):

    @abstractmethod
    def rank(self, user, candidates):
        """Return the candidates re-ordered, best first."""


class Reranker(ABC):

    @abstractmethod
    def rerank(self, user, candidates):
        """Return the candidates filtered / re-ordered by rules."""


CONTENT_FEATURES = ("genre_match", "has_profile", "log_popularity", "item_rating")
POSITIONS = 10


def genre_match(preferences, genres):
    """sum of the user's genre shares over the movie's genres / sqrt(#genres)."""
    if not preferences or not genres:
        return 0.0
    total = sum(max(v, 0.0) for v in preferences.values())
    if total <= 0:
        return 0.0
    return sum(max(preferences.get(g, 0.0), 0.0) for g in genres) / total / math.sqrt(len(genres))


def content_features(user_features, item_features):
    preferences = (user_features or {}).get("genre_preferences") or {}
    item = item_features or {}
    return [
        genre_match(preferences, item.get("genres") or ()),
        1.0 if preferences else 0.0,
        math.log1p(max(float(item.get("popularity", 0) or 0), 0.0)),
        float(item.get("average_rating", 0) or 0) / 5.0,
    ]


class LogisticRanker(Ranker):

    def __init__(self, user_features_fn, feature_store, weights=None, bias=0.0, mean=None, scale=None):
        # user_features_fn(user_id) -> dict: the same online features
        # retrieval uses (Redis, else offline, else defaults).
        self.user_features_fn = user_features_fn
        self.feature_store = feature_store
        self.weights = weights
        self.bias = bias
        self.mean = mean
        self.scale = scale
        self.training_rows = 0
        self.positive_rate = None

    # -- serving ---------------------------------------------------------

    def score(self, user_features, movie_id):
        x = content_features(user_features, self.feature_store.get_item_features(movie_id))
        z = self.bias
        for value, w, m, s in zip(x, self.weights, self.mean, self.scale):
            z += w * (value - m) / s
        return z

    def rank(self, user, candidates):
        if self.weights is None:
            return list(candidates)
        user_features = self.user_features_fn(user.user_id)
        scored = [(-self.score(user_features, m), i, m) for i, m in enumerate(candidates)]
        scored.sort()
        return [m for _, _, m in scored]

    def content_weights(self):
        if self.weights is None:
            return {}
        return {name: round(w, 3) for name, w in zip(CONTENT_FEATURES, self.weights)}

    # -- training --------------------------------------------------------

    @staticmethod
    def training_rows_from(events, feature_store):
        """(content features + position one-hot, label) rows from logged
        slates, features as of each slate's event (point in time)."""
        slates = {}
        order = {}
        clicks = {}
        for event_order, event in enumerate(events):
            if isinstance(event, RecommendationShownEvent):
                slates[event.recommendation_id] = event
                order[event.recommendation_id] = event_order
            elif isinstance(event, RecommendationClickedEvent):
                clicks[event.recommendation_id] = event.movie_id

        rows, labels = [], []
        for rec_id, shown in slates.items():
            event_order = order[rec_id]
            user = feature_store.get_user_features_as_of(shown.user_id, shown.timestamp, event_order) or {}
            clicked = clicks.get(rec_id)
            movie_ids = list(shown.movie_ids)
            last = movie_ids.index(clicked) if clicked in movie_ids else len(movie_ids) - 1
            for position, movie_id in enumerate(movie_ids[:last + 1]):
                item = feature_store.get_item_features_as_of(movie_id, shown.timestamp, event_order)
                one_hot = [0.0] * POSITIONS
                one_hot[min(position, POSITIONS - 1)] = 1.0
                rows.append(content_features(user, item) + one_hot)
                labels.append(1.0 if movie_id == clicked else 0.0)
        return rows, labels

    def fit(self, rows, labels, iterations=300, learning_rate=0.5, l2=1e-3):
        import numpy as np

        if not rows or not any(labels) or all(labels):
            raise ValueError("need both clicked and skipped examples to fit the ranker")
        x = np.array(rows, dtype=float)
        y = np.array(labels, dtype=float)
        n_content = len(CONTENT_FEATURES)

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        mean[n_content:] = 0.0        # keep position one-hots as they are
        scale[n_content:] = 1.0
        xs = (x - mean) / scale

        w = np.zeros(x.shape[1])
        b = math.log(y.mean() / (1 - y.mean()))
        for _ in range(iterations):
            p = 1.0 / (1.0 + np.exp(-(xs @ w + b)))
            error = p - y
            w -= learning_rate * (xs.T @ error / len(y) + l2 * w)
            b -= learning_rate * error.mean()

        self.weights = [float(v) for v in w[:n_content]]
        self.position_weights = [float(v) for v in w[n_content:]]
        self.bias = float(b)
        self.mean = [float(v) for v in mean[:n_content]]
        self.scale = [float(v) for v in scale[:n_content]]
        self.training_rows = len(y)
        self.positive_rate = float(y.mean())
        return self


class WatchedFilter(Reranker):
    """Drop movies the user already watched (Flink's Redis "watched_movies",
    so Flink must run with track_watched=True)."""

    def __init__(self, redis):
        self.redis = redis

    def rerank(self, user, candidates):
        watched = self.redis.get("watched_movies", user.user_id) or ()
        if not watched:
            return list(candidates)
        return [m for m in candidates if m not in watched]
