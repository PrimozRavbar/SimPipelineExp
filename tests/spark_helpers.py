"""Shared helpers for the Spark tests: a seeded synthetic event log and a
digest of everything the offline feature store holds."""

import hashlib
import random

from simulation.events import (BrowseEvent, InterestEvent, OnboardingEvent, RatingEvent,
                               RecommendationClickedEvent, RecommendationShownEvent, WatchEvent)
from world.external_data_loader import GENRE_NAMES


def synthetic_events(movie_ids, n_ticks=60, users=40, seed="spark-test"):
    """Every event kind, with a hot movie (skew) and hot users."""
    rng = random.Random(seed)
    movie_ids = sorted(movie_ids)
    hot = movie_ids[10]
    genres = [g for g in GENRE_NAMES if g != "unknown"]
    events, rec = [], 0
    for t in range(1, n_ticks + 1):
        for _ in range(8):
            u = rng.randint(1, users) if rng.random() < 0.8 else rng.randint(1, 3)
            kind = rng.random()
            if kind < 0.05:
                events.append(OnboardingEvent(timestamp=t, user_id=u, genres=rng.sample(genres, 3)))
                continue
            if kind < 0.15:
                events.append(BrowseEvent(timestamp=t, user_id=u, genre=rng.choice(genres)))
            rec += 1
            slate = rng.sample(movie_ids[:200], 5)
            events.append(RecommendationShownEvent(timestamp=t, user_id=u, recommendation_id=rec, movie_ids=slate))
            m = hot if rng.random() < 0.3 else rng.choice(slate)
            events.append(InterestEvent(timestamp=t, user_id=u, movie_id=rng.choice(slate),
                                        kind=rng.choice(["save", "dismiss"])))
            if rng.random() < 0.5:
                events.append(RecommendationClickedEvent(timestamp=t, user_id=u, recommendation_id=rec, movie_id=m))
                if rng.random() < 0.8:
                    c = rng.random()
                    events.append(WatchEvent(timestamp=t, user_id=u, movie_id=m, watch_time=120 * c, completion=c))
                    if rng.random() < 0.3:
                        events.append(RatingEvent(timestamp=t, user_id=u, movie_id=m, rating=rng.randint(1, 5)))
    return events


def store_digest(store):
    return hashlib.sha256(repr((
        list(store.user_features.items()), list(store.item_features.items()),
        list(store.user_feature_history.items()), list(store.item_feature_history.items()),
        list(store._user_keys.items()), list(store._item_keys.items()), sorted(store._unsorted),
    )).encode()).hexdigest()
