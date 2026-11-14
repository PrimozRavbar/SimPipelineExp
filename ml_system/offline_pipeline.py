from copy import deepcopy

from simulation.events import (
    RatingEvent,
    WatchEvent,
    RecommendationClickedEvent,
)

from simulation.events import OnboardingEvent
from ml_system.features import update_genre_counts, update_engagement, engagement_features, new_engagement_state


class OfflinePipeline:

    def __init__(
        self,
        feature_pipeline,
        dataset_generator,
        trainer,
        embedding_generator
    ):
        self.feature_pipeline = feature_pipeline
        self.dataset_generator = dataset_generator
        self.trainer = trainer
        self.embedding_generator = embedding_generator


    def run(self, events):

        self.feature_pipeline.process(events)

        for event_order, event in enumerate(events):
            self.dataset_generator.add_event(
                event,
                event_order
            )

        training_data = (
            self.dataset_generator.build()
        )

        user_tower, item_tower = self.trainer.train(
            training_data
        )

        item_embeddings = (
            self.embedding_generator.generate(
                item_tower
            )
        )

        return {
            "user_tower": user_tower,
            "item_tower": item_tower,
            "item_embeddings": item_embeddings,
            "training_examples": len(training_data)
        }


class OfflineFeaturePipeline:
    """Offline (batch) user and item features over the event log, with
    point-in-time history for training.

    The computation is split per entity: a user's features depend only on
    that user's events, a movie's only on that movie's events (movie genres
    are static catalog data). So it can run in partitions (Spark hashes
    users and movies to shuffle partitions, see ml_system/spark.py) and
    incrementally (only the events since the last run, carrying each
    entity's running state forward). Every way of running it writes the
    same features and the same history as `process(events)`, the full
    recompute; tests/test_spark.py checks this.

    - `process(events)`: full recompute in one go (the original API).
    - `plan_batch(events, incremental)`: what a job has to compute: all
      events (a full run, or a backfill when the profile settings changed)
      or only the new ones.
    - `user_task` / `item_task`: the work of one partition. They read the
      carried state but never change it or the feature store.
    - `commit(batch, user_results, item_results)`: publishes a finished job
      at once (nothing is written before every task has succeeded).
    """

    def __init__(self, movie_objects, feature_store):

        self.movie_objects = movie_objects
        self.feature_store = feature_store

        # Carried state of the last committed run: entity -> running state
        # (raw genre counts, rating sums...), how many events it covers and
        # the last of them (to check the next log extends the same one).
        self.user_state = {}
        self.item_state = {}
        self.processed = 0
        self.last_event = None
        self.state_config = None


    # How user genre profiles are built ("watches" = the original; "rich":
    # see ml_system.features.update_genre_counts). Flink uses the same.
    profile_mode = "watches"

    # engagement: also compute per-genre click-through ("genre_engagement")
    # and "profile_strength" (ml_system.features); off by default.
    engagement = False

    # Fault injection (feature_nulls): a bad deploy of the job writes empty
    # genre profiles for this share of users (picked by user id). Only what
    # it writes is wrong (history snapshots and latest features); the
    # carried state stays right, so a full backfill repairs the store.
    null_profile_share = 0.0

    def _broken(self, user_id):
        return self.null_profile_share > 0 and (user_id * 2654435761) % 1000 < self.null_profile_share * 1000

    def config(self):
        """Settings the features depend on: a change needs a backfill."""
        return (self.profile_mode, bool(self.engagement))

    # -- the whole job in one go --------------------------------------

    def process(self, events):
        batch = self.plan_batch(events, incremental=False)
        user_results = self.user_task(batch, batch.events_with_order())
        item_results = self.item_task(batch, batch.item_events_with_order())
        self.commit(batch, user_results, item_results)

    def process_incremental(self, events):
        """Only the events since the last run (or everything, when the log
        or the settings changed). Same result as process(events)."""
        batch = self.plan_batch(events, incremental=True)
        user_results = self.user_task(batch, batch.events_with_order())
        item_results = self.item_task(batch, batch.item_events_with_order())
        self.commit(batch, user_results, item_results)
        return batch

    # -- planning -----------------------------------------------------

    def plan_batch(self, events, incremental=True):
        events = list(events)
        reason = None
        if not incremental:
            reason = "full recompute"
        elif self.state_config is None:
            reason = "first run"
        elif self.state_config != self.config():
            reason = "backfill: profile settings changed"
        elif len(events) < self.processed or (
                self.processed and events[self.processed - 1] is not self.last_event):
            reason = "backfill: the event log changed"
        if reason is not None:
            return FeatureBatch(events, 0, True, reason, self.config())
        return FeatureBatch(events, self.processed, False, "incremental", self.config())

    # -- partition tasks ----------------------------------------------

    def _carried(self, states, batch, entity_id):
        if batch.full:
            return None
        state = states.get(entity_id)
        return deepcopy(state) if state is not None else None

    def _new_user_state(self):
        features = {
            "genre_preferences": {},
            "average_rating": 0,
            "activity_level": 0,
            "interactions": 0,
            "watch_count": 0,
            "user_click_counts": 0
        }
        return {"features": features, "rating": [0, 0]}

    def user_task(self, batch, rows):
        """rows: (event_order, event) of the users of one partition, in
        log order. Returns {user_id: (state, history entries)}."""
        out = {}
        profile_mode = batch.config[0]
        engagement = batch.config[1]
        movies = self.movie_objects
        for event_order, event in rows:
            user_id = event.user_id
            entry = out.get(user_id)
            if entry is None:
                state = self._carried(self.user_state, batch, user_id) or self._new_user_state()
                entry = out[user_id] = (state, [])
            state, history = entry
            user = state["features"]

            if engagement and "_engagement" not in user:
                user["_engagement"] = new_engagement_state()
                user["genre_engagement"], user["profile_strength"] = {}, 0.0

            # Snapshot BEFORE processing the current event.
            user_snapshot = deepcopy({k: v for k, v in user.items() if k != "_engagement"})
            total = sum(user_snapshot["genre_preferences"].values())
            if total > 0:
                user_snapshot["genre_preferences"] = {
                    genre: count / total
                    for genre, count in user_snapshot["genre_preferences"].items()
                }
            history.append(((event.timestamp, event_order), user_snapshot))

            name = type(event).__name__
            if isinstance(event, OnboardingEvent):
                update_genre_counts(user["genre_preferences"], "onboarding", event.genres)

            if name == "BrowseEvent":
                update_genre_counts(user["genre_preferences"], "browse", [event.genre], profile_mode)

            if name == "InterestEvent" and event.kind == "save":
                update_genre_counts(user["genre_preferences"], "save",
                                    movies[event.movie_id].genres, profile_mode)

            if engagement:
                kind = {"RecommendationShownEvent": "shown", "RecommendationClickedEvent": "click",
                        "WatchEvent": "watch", "OnboardingEvent": "onboarding"}.get(name)
                if kind == "shown":
                    update_engagement(user["_engagement"], kind,
                                      [movies[m].genres for m in event.movie_ids])
                elif kind in ("click", "watch"):
                    update_engagement(user["_engagement"], kind, movies[event.movie_id].genres)
                elif kind == "onboarding":
                    update_engagement(user["_engagement"], kind, event.genres)
                if kind:
                    user["genre_engagement"], user["profile_strength"] = engagement_features(user["_engagement"])

            if hasattr(event, "movie_id"):
                movie = movies[event.movie_id]
                user["interactions"] += 1
                user["activity_level"] = user["interactions"]

                if isinstance(event, WatchEvent):
                    user["watch_count"] += 1
                    if profile_mode == "watches":
                        for genre in movie.genres:
                            user["genre_preferences"][genre] = (
                                user["genre_preferences"].get(genre, 0) + 1
                            )
                    else:
                        update_genre_counts(user["genre_preferences"], "watch", movie.genres,
                                            profile_mode, event.completion)

                if profile_mode != "watches":
                    if isinstance(event, RecommendationClickedEvent):
                        update_genre_counts(user["genre_preferences"], "click", movie.genres,
                                            profile_mode)
                    elif isinstance(event, RatingEvent):
                        update_genre_counts(user["genre_preferences"], "rating", movie.genres,
                                            profile_mode, event.rating)

                if isinstance(event, RecommendationClickedEvent):
                    user["user_click_counts"] += 1

                if isinstance(event, RatingEvent):
                    rating = state["rating"]
                    rating[0] = rating[0] + event.rating
                    rating[1] = rating[1] + 1
                    user["average_rating"] = rating[0] / rating[1]
        return out

    def _new_item_state(self, movie_id):
        movie = self.movie_objects[movie_id]
        return {"features": {"movie_id": movie_id, "genres": movie.genres, "average_rating": 0,
                             "popularity": 0, "interactions": 0},
                "rating": [0, 0]}

    def item_task(self, batch, rows):
        """rows: (event_order, event) with a movie_id, for the movies of
        one partition, in log order. Returns {movie_id: (state, history)}."""
        out = {}
        for event_order, event in rows:
            movie_id = event.movie_id
            entry = out.get(movie_id)
            if entry is None:
                state = self._carried(self.item_state, batch, movie_id) or self._new_item_state(movie_id)
                entry = out[movie_id] = (state, [])
            state, history = entry
            item = state["features"]

            # Snapshot item BEFORE processing the current event.
            history.append(((event.timestamp, event_order), deepcopy(item)))

            item["interactions"] += 1
            if isinstance(event, WatchEvent):
                item["popularity"] += 1
            if isinstance(event, RatingEvent):
                rating = state["rating"]
                rating[0] = rating[0] + event.rating
                rating[1] = rating[1] + 1
                item["average_rating"] = rating[0] / rating[1]
        return out

    # -- publishing ---------------------------------------------------

    def commit(self, batch, user_results, item_results):
        """Publish a finished job: history snapshots, latest features and
        the carried state, all at once. `*_results` are task outputs
        ({entity: (state, history)}, partitions merged in any order)."""
        store = self.feature_store
        if batch.full:
            # History is recomputed from the full event log; start clean so
            # repeated runs don't stack duplicate snapshots.
            store.clear_history()
            self.user_state, self.item_state = {}, {}
            for movie_id in self.movie_objects:
                state = self._new_item_state(movie_id)
                self.item_state[movie_id] = state
                store.write_item_features(movie_id, deepcopy(state["features"]),
                                          timestamp=0, event_order=-1)

        # New entities enter the store in order of their first event (as a
        # single pass over the log would add them).
        def first_order(result):
            return result[1][1][0][0][1]
        for results, states, kind in ((user_results, self.user_state, "user"),
                                      (item_results, self.item_state, "item")):
            ordered = sorted(results.items(), key=first_order)
            for entity_id, (state, history) in ordered:
                if kind == "user" and self._broken(entity_id):
                    for _, snapshot in history:
                        snapshot["genre_preferences"] = {}
                store.append_history(kind, entity_id, history)
                states[entity_id] = state
                if kind == "user":
                    latest = self.latest_user_features(state)
                    if self._broken(entity_id):
                        latest["genre_preferences"] = {}
                    store.write_user_features(entity_id, latest)
                else:
                    store.write_item_features(entity_id, deepcopy(state["features"]))
        if batch.full:
            for movie_id, state in self.item_state.items():
                if movie_id not in item_results:
                    store.write_item_features(movie_id, deepcopy(state["features"]))

        self.processed = len(batch.events)
        self.last_event = batch.events[-1] if batch.events else None
        self.state_config = batch.config

    @staticmethod
    def latest_user_features(state):
        features = deepcopy({k: v for k, v in state["features"].items() if k != "_engagement"})
        total = sum(features["genre_preferences"].values())
        if total > 0:
            features["genre_preferences"] = {
                genre: count / total for genre, count in features["genre_preferences"].items()
            }
        return features


class FeatureBatch:
    """The events one feature job computes: events[start:] (start 0 for a
    full run), with their global event order (the log position)."""

    def __init__(self, events, start, full, reason, config):
        self.events = events
        self.start = start
        self.full = full
        self.reason = reason
        self.config = config

    @property
    def size(self):
        return len(self.events) - self.start

    def events_with_order(self):
        return [(i, self.events[i]) for i in range(self.start, len(self.events))]

    def item_events_with_order(self):
        return [(i, e) for i, e in self.events_with_order() if hasattr(e, "movie_id")]
