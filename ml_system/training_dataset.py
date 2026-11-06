import random

from simulation.events import (
    RecommendationShownEvent,
    RecommendationClickedEvent,
)


class TrainingDatasetGenerator:

    def __init__(
        self,
        feature_store,
        negative_samples=5
    ):
        self.feature_store = feature_store
        self.interactions = []
        self.shown_recommendations = {}
        self.negative_samples = negative_samples


    def add_event(self, event, event_order):

        if isinstance(event, RecommendationShownEvent):

            self.shown_recommendations[
                event.recommendation_id
            ] = event

        elif isinstance(event, RecommendationClickedEvent):

            shown = self.shown_recommendations.get(
                event.recommendation_id
            )

            if shown is None:
                return

            candidates = [
                movie_id
                for movie_id in shown.movie_ids
                if movie_id != event.movie_id
            ]

            sampled_negatives = random.sample(
                candidates,
                min(
                    self.negative_samples,
                    len(candidates)
                )
            )

            self.interactions.append({
                "user_id": event.user_id,
                "positive_movie_id": event.movie_id,
                "negative_movie_ids": sampled_negatives,
                "timestamp": event.timestamp,
                "event_order": event_order
            })


    def build(self):

        random.shuffle(self.interactions)

        dataset = []

        for interaction in self.interactions:

            user_features = (
                self.feature_store.get_user_features_as_of(
                    interaction["user_id"],
                    interaction["timestamp"],
                    interaction["event_order"]
                )
            )

            if user_features is None:
                continue

            user_features = dict(user_features)
            user_features["user_id"] = interaction["user_id"]

            positive_features = (
                self.feature_store.get_item_features_as_of(
                    interaction["positive_movie_id"],
                    interaction["timestamp"],
                    interaction["event_order"]
                )
            )

            negative_features = []

            for movie_id in interaction["negative_movie_ids"]:

                features = (
                    self.feature_store.get_item_features_as_of(
                        movie_id,
                        interaction["timestamp"],
                        interaction["event_order"]
                    )
                )

                if features is not None:
                    negative_features.append(features)

            if positive_features is None or not negative_features:
                continue

            dataset.append({
                "user_features": user_features,
                "positive_item_features": positive_features,
                "negative_item_features": negative_features
            })

        self.interactions = []
        self.shown_recommendations = {}

        return dataset
