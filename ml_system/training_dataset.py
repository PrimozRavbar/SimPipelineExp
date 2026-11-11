import random

from simulation.events import (
    RecommendationShownEvent,
    RecommendationClickedEvent,
)


class TrainingDatasetGenerator:

    def __init__(
        self,
        feature_store,
        negative_samples=5,
        rng=None,
        graded=False,
        history=False
    ):
        self.feature_store = feature_store
        self.rng = rng if rng is not None else random.Random(0)
        self.interactions = []
        self.shown_recommendations = {}
        self.negative_samples = negative_samples

        # graded: set each example's "weight" from what followed the click
        # (see grade()). history: attach the user's recent watches as of the
        # click (user_features["recent_watches"], the same definition Flink
        # serves). Both off by default.
        self.graded = graded
        self.history = history
        self._last_click = {}        # (user_id, movie_id) -> interaction
        self._recent = None


    def add_event(self, event, event_order):

        if self.history and self._recent is None:
            from ml_system.ranking import RecentHistory
            self._recent = RecentHistory()

        name = type(event).__name__
        if name in ("WatchEvent", "RatingEvent"):
            interaction = self._last_click.get((event.user_id, event.movie_id))
            if interaction is not None:
                if name == "WatchEvent":
                    interaction["completion"] = event.completion
                else:
                    interaction["rating"] = event.rating
            if self._recent is not None:
                self._recent.observe(event)
            return

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

            # Negatives come only from the slate the user was shown (and
            # skipped), never from retrieved-but-unshown candidates.
            candidates = [
                movie_id
                for movie_id in shown.movie_ids
                if movie_id != event.movie_id
            ]

            sampled_negatives = self.rng.sample(
                candidates,
                min(
                    self.negative_samples,
                    len(candidates)
                )
            )

            interaction = {
                "user_id": event.user_id,
                "positive_movie_id": event.movie_id,
                "negative_movie_ids": sampled_negatives,
                "timestamp": event.timestamp,
                "event_order": event_order
            }
            if self._recent is not None:
                interaction["recent_watches"] = self._recent.get(event.user_id)[0]
            self.interactions.append(interaction)
            self._last_click[(event.user_id, event.movie_id)] = interaction


    def build(self):

        self.rng.shuffle(self.interactions)

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
            if self.history:
                user_features["recent_watches"] = interaction.get("recent_watches", [])

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

            example = {
                "user_features": user_features,
                "positive_item_features": positive_features,
                "negative_item_features": negative_features
            }
            if self.graded:
                example["weight"] = grade(interaction)
            dataset.append(example)

        self.interactions = []
        self.shown_recommendations = {}
        self._last_click = {}
        self._recent = None

        return dataset


def grade(interaction):
    """Weight of a click as a positive: a click that was not watched is
    probably a noise click (0.3); a watched one counts 0.5 + completion,
    plus 0.25 per star above 3 if rated."""
    completion = interaction.get("completion")
    if completion is None:
        return 0.3
    weight = 0.5 + completion
    rating = interaction.get("rating")
    if rating is not None:
        weight += 0.25 * max(0.0, rating - 3)
    return weight
