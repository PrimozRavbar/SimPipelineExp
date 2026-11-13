"""Feast: the feature store in front of Redis (online) and the offline store.

Serving reads features through Feast; training keeps reading the offline
store point in time (Feast's "historical retrieval"). A registry of feature
views says where each feature comes from:

- stream views, pushed by Flink into the online store (Redis) as events
  arrive: the user profile and counts, recent history, movie signals, the
  preference model;
- batch views, computed by Spark into the offline store: user features and
  item features (genres, popularity, average rating).

Modes for batch views:
- "direct" (default, the original behaviour): serving reads the offline
  store at request time, so batch features are always as fresh as the last
  Spark run.
- "materialized": serving reads only the online store, which holds a copy
  of the batch views refreshed by `materialize()` every
  `materialize_every` ticks (as `feast materialize` does). Between runs
  serving sees older values (and none for brand-new entities); a missing
  run (`materialize_every` = 0) leaves the online copy where it was.

Feast exposes the read interface of the offline store that serving uses
(get_user_features, get_item_features) and forwards everything else, so it
can be passed wherever serving took the feature store.
"""

import copy

from simulation.service import Service

STREAM_VIEWS = {
    "user_profile": ("user", ["genre_preferences", "activity_level", "interactions", "average_rating"]),
    "user_history": ("user", ["recent_watches", "recent_slates", "watched_movies"]),
    "movie_signals": ("movie", ["item_signals"]),
    "preference_model": ("user", ["preference_model"]),
}
BATCH_VIEWS = {
    "user_batch": ("user", "user_features"),
    "item_batch": ("movie", "item_features"),
}
MODES = ("direct", "materialized")


class Feast(Service):

    def __init__(self, redis, offline_store, mode="direct", materialize_every=100):
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode must be one of %s" % (MODES,))
        self.redis = redis
        self.offline = offline_store
        self.mode = mode
        self.materialize_every = materialize_every
        self.online_batch = {"user_features": {}, "item_features": {}}
        self.last_materialized = None
        self.materializations = 0
        self.reads = {"user": 0, "item": 0}
        self.misses = {"user": 0, "item": 0}

    # -- the read interface serving uses -----------------------------------------

    def get_user_features(self, user_id):
        self.reads["user"] += 1
        if self.mode == "direct":
            return self.offline.get_user_features(user_id)
        value = self.online_batch["user_features"].get(user_id)
        if value is None:
            self.misses["user"] += 1
        return value

    def get_item_features(self, movie_id):
        self.reads["item"] += 1
        if self.mode == "direct":
            return self.offline.get_item_features(movie_id)
        value = self.online_batch["item_features"].get(movie_id)
        if value is None:
            self.misses["item"] += 1
        return value

    def __getattr__(self, name):
        # Anything else (point-in-time lookups, raw tables): the offline store.
        offline = self.__dict__.get("offline")
        if offline is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(offline, name)

    # -- materialization -----------------------------------------------------------

    def materialize(self, now):
        """Copy the batch views from the offline store to the online store."""
        self.online_batch = {
            "user_features": {u: copy.copy(f) for u, f in self.offline.user_features.items()},
            "item_features": {m: copy.copy(f) for m, f in self.offline.item_features.items()},
        }
        self.last_materialized = now
        self.materializations += 1

    def process(self, sim):
        if self.mode != "materialized" or not self.materialize_every:
            return
        now = sim.clock.now
        if self.last_materialized is None or now - self.last_materialized >= self.materialize_every:
            self.materialize(now)

    # -- reporting -------------------------------------------------------------------

    def describe(self, now=None):
        views = [{"name": n, "entity": e, "source": "stream (Flink -> Redis)", "features": f}
                 for n, (e, f) in STREAM_VIEWS.items()]
        views += [{"name": n, "entity": e, "source": "batch (Spark -> offline store)", "table": t}
                  for n, (e, t) in BATCH_VIEWS.items()]
        age = None if self.last_materialized is None or now is None else now - self.last_materialized
        return {"mode": self.mode, "materialize_every": self.materialize_every,
                "last_materialized": self.last_materialized, "materialized_age": age,
                "materializations": self.materializations, "reads": dict(self.reads),
                "misses": dict(self.misses), "views": views}
