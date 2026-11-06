from copy import deepcopy

from simulation.events import (
    RatingEvent,
    WatchEvent,
    RecommendationClickedEvent,
)

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
            "item_embeddings": item_embeddings
        }


class OfflineFeaturePipeline:

    def __init__(self, movie_objects, feature_store):

        self.movie_objects = movie_objects
        self.feature_store = feature_store

        self.rating_features = RatingFeatureComputer()
        self.item_rating_features = ItemRatingFeatureComputer()


    def process(self, events):

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


            # Snapshot BEFORE processing the current event.
            user_snapshot = deepcopy(user)

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

                    for genre in movie.genres:

                        user["genre_preferences"][genre] = (
                            user["genre_preferences"].get(
                                genre,
                                0
                            ) + 1
                        )


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
