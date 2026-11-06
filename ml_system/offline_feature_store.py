from copy import deepcopy
from simulation.service import Service


class OfflineFeatureStore(Service):

    def __init__(self):
        super().__init__()

        self.name = "OfflineFeatureStore"

        self.user_features = {}
        self.item_features = {}

        self.user_feature_history = {}
        self.item_feature_history = {}


    def write_user_features(
        self,
        user_id,
        features,
        timestamp=None,
        event_order=None
    ):

        self.user_features[user_id] = features

        if timestamp is not None:
            key = (timestamp, event_order)

            self.user_feature_history.setdefault(
                user_id,
                []
            ).append(
                (key, deepcopy(features))
            )


    def write_item_features(
        self,
        movie_id,
        features,
        timestamp=None,
        event_order=None
    ):

        self.item_features[movie_id] = features

        if timestamp is not None:
            key = (timestamp, event_order)

            self.item_feature_history.setdefault(
                movie_id,
                []
            ).append(
                (key, deepcopy(features))
            )


    def get_user_features(self, user_id):

        return self.user_features.get(user_id)


    def get_item_features(self, movie_id):

        return self.item_features.get(movie_id)


    def get_user_features_as_of(
        self,
        user_id,
        timestamp,
        event_order=None
    ):

        history = self.user_feature_history.get(
            user_id,
            []
        )

        target = (timestamp, event_order)

        candidates = [
            features
            for snapshot_key, features in history
            if snapshot_key <= target
        ]

        if not candidates:
            return None

        return candidates[-1]


    def get_item_features_as_of(
        self,
        movie_id,
        timestamp,
        event_order=None
    ):

        history = self.item_feature_history.get(
            movie_id,
            []
        )

        target = (timestamp, event_order)

        candidates = [
            features
            for snapshot_key, features in history
            if snapshot_key <= target
        ]

        if not candidates:
            return None

        return candidates[-1]
