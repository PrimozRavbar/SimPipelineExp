from bisect import bisect_right
from copy import deepcopy

from simulation.service import Service


class OfflineFeatureStore(Service):
    """Latest features per user/item, plus point-in-time history.

    History entries are (key, features) with key = (timestamp, event_order).
    `get_*_as_of(id, t, order)` returns the newest snapshot with key <= (t,
    order), so training sees each feature as it was when an event happened.
    Lookups use binary search while a history is in key order (the offline
    pipeline writes it that way) and fall back to a scan otherwise.
    """

    def __init__(self):
        super().__init__()

        self.name = "OfflineFeatureStore"

        self.user_features = {}
        self.item_features = {}

        self.user_feature_history = {}
        self.item_feature_history = {}

        # Parallel key lists for binary search, and ids whose history was
        # written out of key order (looked up by scanning).
        self._user_keys = {}
        self._item_keys = {}
        self._unsorted = set()


    def clear_history(self):
        """Drop all point-in-time history (the latest features stay). The
        offline pipeline calls this before recomputing history from the
        full event log, so repeated runs don't stack duplicate snapshots."""
        self.user_feature_history = {}
        self.item_feature_history = {}
        self._user_keys = {}
        self._item_keys = {}
        self._unsorted = set()


    def _append(self, history, keys, kind, entity_id, key, features):
        entity_keys = keys.setdefault(entity_id, [])
        if entity_keys and key < entity_keys[-1]:
            self._unsorted.add((kind, entity_id))
        entity_keys.append(key)
        history.setdefault(entity_id, []).append((key, deepcopy(features)))


    def write_user_features(
        self,
        user_id,
        features,
        timestamp=None,
        event_order=None
    ):

        self.user_features[user_id] = features

        if timestamp is not None:
            self._append(self.user_feature_history, self._user_keys, "user",
                         user_id, (timestamp, event_order), features)


    def write_item_features(
        self,
        movie_id,
        features,
        timestamp=None,
        event_order=None
    ):

        self.item_features[movie_id] = features

        if timestamp is not None:
            self._append(self.item_feature_history, self._item_keys, "item",
                         movie_id, (timestamp, event_order), features)


    def get_user_features(self, user_id):

        return self.user_features.get(user_id)


    def get_item_features(self, movie_id):

        return self.item_features.get(movie_id)


    def _as_of(self, history, keys, kind, entity_id, timestamp, event_order):
        entries = history.get(entity_id)
        if not entries:
            return None

        target = (timestamp, event_order)

        if (kind, entity_id) in self._unsorted:
            candidates = [
                features
                for snapshot_key, features in entries
                if snapshot_key <= target
            ]
            return candidates[-1] if candidates else None

        position = bisect_right(keys[entity_id], target)
        return entries[position - 1][1] if position else None


    def get_user_features_as_of(
        self,
        user_id,
        timestamp,
        event_order=None
    ):
        return self._as_of(self.user_feature_history, self._user_keys, "user",
                           user_id, timestamp, event_order)


    def get_item_features_as_of(
        self,
        movie_id,
        timestamp,
        event_order=None
    ):
        return self._as_of(self.item_feature_history, self._item_keys, "item",
                           movie_id, timestamp, event_order)
