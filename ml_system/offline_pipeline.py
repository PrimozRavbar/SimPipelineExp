from copy import deepcopy

from simulation.events import (
    RatingEvent,
    WatchEvent,
    RecommendationClickedEvent,
)

from simulation.events import OnboardingEvent
from ml_system.features import update_genre_counts, update_engagement, engagement_features, new_engagement_state
from ml_system.features import (
    RatingFeatureComputer,
    ItemRatingFeatureComputer,
)


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

    def __init__(self, movie_objects, feature_store):

        self.movie_objects = movie_objects
        self.feature_store = feature_store

        self.rating_features = RatingFeatureComputer()
        self.item_rating_features = ItemRatingFeatureComputer()


    # How user genre profiles are built ("watches" = the original; "rich":
    # see ml_system.features.update_genre_counts). Flink uses the same.
    profile_mode = "watches"

    # engagement: also compute per-genre click-through ("genre_engagement")
    # and "profile_strength" (ml_system.features); off by default.
    engagement = False

    def process(self, events):

        # History is recomputed from the full event log below; start clean
        # so repeated runs (Spark schedule, retrains) don't stack duplicate
        # snapshots (memory, and slower point-in-time lookups).
        self.feature_store.clear_history()

        self.rating_features = RatingFeatureComputer()
        self.item_rating_features = ItemRatingFeatureComputer()

        user_features = {}
        item_features = {}

        # Initialize a time-zero baseline for every movie.
        for movie_id, movie in self.movie_objects.items():
            item_features[movie_id] = {
                "movie_id": movie_id,
                "genres": movie.genres,
                "average_rating": 0,
                "popularity": 0,
                "interactions": 0
            }

            self.feature_store.write_item_features(
                movie_id,
                deepcopy(item_features[movie_id]),
                timestamp=0,
                event_order=-1
            )


        for event_order, event in enumerate(events):

            user_id = event.user_id

            if user_id not in user_features:

                user_features[user_id] = {
                    "genre_preferences": {},
                    "average_rating": 0,
                    "activity_level": 0,
                    "interactions": 0,
                    "watch_count": 0,
                    "user_click_counts": 0
                }


            user = user_features[user_id]

            if self.engagement and "_engagement" not in user:
                user["_engagement"] = new_engagement_state()
                user["genre_engagement"], user["profile_strength"] = {}, 0.0


            # Snapshot BEFORE processing the current event.
            user_snapshot = deepcopy({k: v for k, v in user.items() if k != "_engagement"})

            total = sum(
                user_snapshot["genre_preferences"].values()
            )

            if total > 0:
                user_snapshot["genre_preferences"] = {
                    genre: count / total
                    for genre, count
                    in user_snapshot["genre_preferences"].items()
                }

            self.feature_store.write_user_features(
                user_id,
                user_snapshot,
                timestamp=event.timestamp,
                event_order=event_order
            )


            if isinstance(event, OnboardingEvent):
                update_genre_counts(user["genre_preferences"], "onboarding", event.genres)

            if type(event).__name__ == "BrowseEvent":
                update_genre_counts(user["genre_preferences"], "browse", [event.genre], self.profile_mode)

            if self.engagement:
                kind = {"RecommendationShownEvent": "shown", "RecommendationClickedEvent": "click",
                        "WatchEvent": "watch", "OnboardingEvent": "onboarding"}.get(type(event).__name__)
                if kind == "shown":
                    update_engagement(user["_engagement"], kind,
                                      [self.movie_objects[m].genres for m in event.movie_ids])
                elif kind in ("click", "watch"):
                    update_engagement(user["_engagement"], kind, self.movie_objects[event.movie_id].genres)
                elif kind == "onboarding":
                    update_engagement(user["_engagement"], kind, event.genres)
                if kind:
                    user["genre_engagement"], user["profile_strength"] = engagement_features(user["_engagement"])

            if hasattr(event, "movie_id"):

                movie_id = event.movie_id
                movie = self.movie_objects[movie_id]

                if movie_id not in item_features:

                    item_features[movie_id] = {
                        "movie_id": movie_id,
                        "genres": movie.genres,
                        "average_rating": 0,
                        "popularity": 0,
                        "interactions": 0
                    }


                item = item_features[movie_id]


                # Snapshot item BEFORE processing the current event.
                self.feature_store.write_item_features(
                    movie_id,
                    deepcopy(item),
                    timestamp=event.timestamp,
                    event_order=event_order
                )


                # Process the current event.
                item["interactions"] += 1

                user["interactions"] += 1
                user["activity_level"] = user["interactions"]


                if isinstance(event, WatchEvent):

                    user["watch_count"] += 1

                    item["popularity"] += 1

                    if self.profile_mode == "watches":
                        for genre in movie.genres:

                            user["genre_preferences"][genre] = (
                                user["genre_preferences"].get(
                                    genre,
                                    0
                                ) + 1
                            )
                    else:
                        update_genre_counts(user["genre_preferences"], "watch", movie.genres,
                                            self.profile_mode, event.completion)

                if self.profile_mode != "watches":
                    if isinstance(event, RecommendationClickedEvent):
                        update_genre_counts(user["genre_preferences"], "click", movie.genres,
                                            self.profile_mode)
                    elif isinstance(event, RatingEvent):
                        update_genre_counts(user["genre_preferences"], "rating", movie.genres,
                                            self.profile_mode, event.rating)


                if isinstance(event, RecommendationClickedEvent):

                    user["user_click_counts"] += 1


                if isinstance(event, RatingEvent):

                    user["average_rating"] = (
                        self.rating_features.update(
                            event.user_id,
                            event.rating
                        )
                    )

                    item["average_rating"] = (
                        self.item_rating_features.update(
                            event.movie_id,
                            event.rating
                        )
                    )


        # Preserve latest feature API.
        for user_id, features in user_features.items():

            features.pop("_engagement", None)

            total = sum(
                features["genre_preferences"].values()
            )

            if total > 0:

                features["genre_preferences"] = {
                    genre: count / total
                    for genre, count
                    in features["genre_preferences"].items()
                }

            self.feature_store.write_user_features(
                user_id,
                features
            )


        for movie_id, features in item_features.items():

            self.feature_store.write_item_features(
                movie_id,
                features
            )
