"""CTR-per-tick tracking."""

from types import SimpleNamespace

import pytest

from ml_system.kafka import Kafka
from ml_system.metrics import CTRTracker
from simulation.events import RecommendationClickedEvent, RecommendationShownEvent


def tracker_fed_with(events):
    kafka = Kafka()
    sim = SimpleNamespace(get_service=lambda name: kafka)

    tracker = CTRTracker()
    tracker.start(sim)

    for event in events:
        kafka.receive(event)

    tracker.process(sim)
    return tracker


def shown(tick, rec_id):
    return RecommendationShownEvent(
        timestamp=tick, user_id=1, recommendation_id=rec_id, movie_ids=[1, 2]
    )


def clicked(tick, rec_id):
    return RecommendationClickedEvent(
        timestamp=tick, user_id=1, recommendation_id=rec_id, movie_id=1
    )


def test_per_tick_counts_and_gaps():
    tracker = tracker_fed_with([
        shown(1, 1), clicked(1, 1),
        shown(2, 2),
        shown(4, 4), shown(4, 5), clicked(4, 4),
    ])

    rows = tracker.per_tick()

    assert [row["tick"] for row in rows] == [1, 2, 3, 4]
    assert [row["ctr"] for row in rows] == [1.0, 0.0, None, 0.5]
    assert tracker.overall_ctr() == pytest.approx(2 / 4)
    assert tracker.ctr_between(2, 3) == 0.0
    assert tracker.ctr_between(5, None) is None


def test_rolling_ctr_window():
    tracker = tracker_fed_with([
        shown(1, 1), clicked(1, 1),
        shown(2, 2),
        shown(3, 3),
        shown(4, 4), clicked(4, 4),
    ])

    assert tracker.rolling_ctr(window=2) == [
        (1, 1.0), (2, 0.5), (3, 0.0), (4, 0.5)
    ]

    with pytest.raises(ValueError):
        tracker.rolling_ctr(window=0)


def test_tracker_matches_datalake_in_simulation():
    pytest.importorskip("torch")
    from world.external_data_loader import load_movielens
    from ml_system.build_recommendation_system import build_recommendation_system

    data = load_movielens()
    system = build_recommendation_system(
        movie_objects=data["movie_objects"],
        user_objects=data["user_objects"],
        user_states=data["user_states"],
        movie_states=data["movie_states"],
        seed=7,
    )
    system["sim"].run(100, verbose=False)

    events = system["datalake"].events
    shown_count = sum(isinstance(e, RecommendationShownEvent) for e in events)
    click_count = sum(isinstance(e, RecommendationClickedEvent) for e in events)

    tracker = system["ctr_tracker"]
    rows = tracker.per_tick()

    assert sum(row["shown"] for row in rows) == shown_count
    assert sum(row["clicks"] for row in rows) == click_count
    assert tracker.overall_ctr() == pytest.approx(click_count / shown_count)
