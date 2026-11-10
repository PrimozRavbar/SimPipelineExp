"""Guard: everything added for realism is opt-in. A default build must
produce exactly the events it produced before (fingerprint taken on main
before the change, seed 11, 5 users per tick, 150 ticks)."""

import hashlib

import pytest

pytest.importorskip("torch")

from ml_system.build_recommendation_system import build_recommendation_system
from world.external_data_loader import load_movielens

GOLDEN_EVENTS = 922
GOLDEN_SHA256 = "c57286da6f5682e637fa25cf95af5cce0ec2b3265de699a05e264e391026429a"


def fingerprint(events):
    h = hashlib.sha256()
    for e in events:
        h.update(repr((type(e).__name__, e.timestamp, e.user_id, getattr(e, "movie_id", None),
                       tuple(getattr(e, "movie_ids", ()) or ()), getattr(e, "rating", None),
                       round(getattr(e, "completion", 0) or 0, 9))).encode())
    return h.hexdigest()


def test_default_build_matches_main():
    data = load_movielens()
    system = build_recommendation_system(
        data["movie_objects"], data["user_objects"], data["user_states"], data["movie_states"],
        seed=11, users_per_tick=5, log_metrics=False,
    )
    system["sim"].run(150, verbose=False)
    events = system["datalake"].events
    assert len(events) == GOLDEN_EVENTS
    assert fingerprint(events) == GOLDEN_SHA256
