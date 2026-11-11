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


# ---------------------------------------------------------------------
# Richer ranker: cross and recent-history features, small MLP
# ---------------------------------------------------------------------

RECENT_WATCHES = 5      # last N watched movies per user
RECENT_SLATES = 3       # last N shown slates per user

FEATURES_V2 = (
    "genre_match", "has_profile", "log_popularity", "item_rating",
    "match_x_popularity",        # cross: taste match x popularity
    "coldstart_x_popularity",    # cross: no profile yet x popularity
    "recent_genre_match",        # match with the genres of the last watches
    "watched_recently",          # movie among the last watches
    "shown_recently",            # times shown in the last slates (fatigue)
)


class RecentHistory:
    """Per-user recent watches and shown slates. The same class is updated
    by Flink online (published to Redis) and replayed over the event log
    for training, so both sides use one definition."""

    def __init__(self):
        self.watches = {}    # user_id -> list of movie ids, newest last
        self.slates = {}     # user_id -> list of tuples of movie ids, newest last

    def observe(self, event):
        name = type(event).__name__
        if name == "WatchEvent":
            self.add_watch(event.user_id, event.movie_id)
        elif name == "RecommendationShownEvent":
            self.add_slate(event.user_id, event.movie_ids)

    def add_watch(self, user_id, movie_id):
        watches = self.watches.setdefault(user_id, [])
        watches.append(movie_id)
        del watches[:-RECENT_WATCHES]
        return list(watches)

    def add_slate(self, user_id, movie_ids):
        slates = self.slates.setdefault(user_id, [])
        slates.append(tuple(movie_ids))
        del slates[:-RECENT_SLATES]
        return list(slates)

    def get(self, user_id):
        return list(self.watches.get(user_id, ())), list(self.slates.get(user_id, ()))


def features_v2(user_features, recent_watches, recent_slates, item_features, movie_id, genres_of):
    """`genres_of(movie_id)` gives a movie's catalog genres."""
    base = content_features(user_features, item_features)
    match, has_profile, log_pop, _ = base
    recent = {}
    for watched in recent_watches:
        for genre in genres_of(watched):
            recent[genre] = recent.get(genre, 0.0) + 1.0
    genres = (item_features or {}).get("genres") or genres_of(movie_id)
    return base + [
        match * log_pop,
        (1.0 - has_profile) * log_pop,
        genre_match(recent, genres),
        1.0 if movie_id in recent_watches else 0.0,
        float(sum(movie_id in slate for slate in recent_slates)),
    ]


class MLPRanker(Ranker):
    """score = MLP(features) + position_bias[position] in training; the
    position term (a separate 'shallow tower') is dropped at serving, so
    the MLP learns relevance without the credit that comes from being shown
    at the top. Trained with binary cross-entropy on cascade rows, seeded,
    CPU only."""

    def __init__(self, user_features_fn, redis, feature_store, genres_of, hidden=16, seed=0):
        self.user_features_fn = user_features_fn
        self.redis = redis
        self.feature_store = feature_store
        self.genres_of = genres_of
        self.hidden = hidden
        self.seed = seed
        self.model = None
        self.mean = None
        self.scale = None
        self.training_rows = 0
        self.positive_rate = None
        self.losses = []

    # -- serving ---------------------------------------------------------

    def rank(self, user, candidates):
        if self.model is None or not candidates:
            return list(candidates)
        import torch

        user_features = self.user_features_fn(user.user_id)
        watches = self.redis.get("recent_watches", user.user_id) or []
        slates = self.redis.get("recent_slates", user.user_id) or []
        rows = [features_v2(user_features, watches, slates, self.feature_store.get_item_features(m),
                            m, self.genres_of) for m in candidates]
        with torch.no_grad():
            x = (torch.tensor(rows, dtype=torch.float32) - self.mean) / self.scale
            scores = self.model(x).squeeze(1).tolist()
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [candidates[i] for i in order]

    # -- training --------------------------------------------------------

    @staticmethod
    def training_rows_from(events, feature_store, genres_of):
        """Cascade rows (see LogisticRanker.training_rows_from) with v2
        features; recent history is replayed in event order, as of just
        before each slate. Returns (rows, positions, labels)."""
        clicks = {e.recommendation_id: e.movie_id for e in events
                  if isinstance(e, RecommendationClickedEvent)}
        history = RecentHistory()
        rows, positions, labels = [], [], []
        for event_order, event in enumerate(events):
            if isinstance(event, RecommendationShownEvent):
                watches, slates = history.get(event.user_id)
                user = feature_store.get_user_features_as_of(event.user_id, event.timestamp, event_order) or {}
                clicked = clicks.get(event.recommendation_id)
                movie_ids = list(event.movie_ids)
                last = movie_ids.index(clicked) if clicked in movie_ids else len(movie_ids) - 1
                for position, movie_id in enumerate(movie_ids[:last + 1]):
                    item = feature_store.get_item_features_as_of(movie_id, event.timestamp, event_order)
                    rows.append(features_v2(user, watches, slates, item, movie_id, genres_of))
                    positions.append(min(position, POSITIONS - 1))
                    labels.append(1.0 if movie_id == clicked else 0.0)
            history.observe(event)
        return rows, positions, labels

    def fit(self, rows, positions, labels, epochs=8, batch_size=512, learning_rate=0.01):
        import torch
        import torch.nn as nn

        if not rows or not any(labels) or all(labels):
            raise ValueError("need both clicked and skipped examples to fit the ranker")

        x = torch.tensor(rows, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        pos = torch.tensor(positions, dtype=torch.long)
        self.mean = x.mean(dim=0)
        self.scale = x.std(dim=0)
        self.scale[self.scale == 0] = 1.0
        xs = (x - self.mean) / self.scale

        # Seeded init and shuffling without touching the global torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            model = nn.Sequential(nn.Linear(x.shape[1], self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1))
            position_bias = nn.Embedding(POSITIONS, 1)
            nn.init.zeros_(position_bias.weight)
            generator = torch.Generator().manual_seed(self.seed)

        optimizer = torch.optim.Adam(list(model.parameters()) + list(position_bias.parameters()),
                                     lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        self.losses = []
        for _ in range(epochs):
            order = torch.randperm(len(y), generator=generator)
            total, batches = 0.0, 0
            for start in range(0, len(y), batch_size):
                idx = order[start:start + batch_size]
                logits = model(xs[idx]).squeeze(1) + position_bias(pos[idx]).squeeze(1)
                loss = loss_fn(logits, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
            self.losses.append(total / batches)

        self.model = model.eval()
        self.position_bias = [float(v) for v in position_bias.weight.detach().squeeze(1)]
        self.training_rows = len(y)
        self.positive_rate = float(y.mean())
        return self

    def summary(self):
        return {"model": "mlp", "features": list(FEATURES_V2), "hidden": self.hidden,
                "training_rows": self.training_rows,
                "click_rate": None if self.positive_rate is None else round(self.positive_rate, 4),
                "final_loss": round(self.losses[-1], 4) if self.losses else None,
                "position_bias": [round(v, 3) for v in getattr(self, "position_bias", [])]}
