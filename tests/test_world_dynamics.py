"""World dynamics (world/dynamics.py, UserSimulator fatigue/rewatch)."""

import pytest

pytest.importorskip("torch")

from ml_system.build_recommendation_system import build_recommendation_system
from world.dynamics import DYNAMIC_WORLD
from world.external_data_loader import load_movielens


def build(world_dynamics, seed=3):
    data = load_movielens()
    return build_recommendation_system(
        data["movie_objects"], data["user_objects"], data["user_states"], data["movie_states"],
        seed=seed, users_per_tick=5, log_metrics=False, world_dynamics=world_dynamics,
    )


def signature(system):
    return [(type(e).__name__, e.timestamp, e.user_id, getattr(e, "movie_id", None))
            for e in system["datalake"].events]


def test_dynamic_world_is_deterministic_and_differs_from_static():
    a, b = build(DYNAMIC_WORLD), build(DYNAMIC_WORLD)
    a["sim"].run(220, verbose=False)
    b["sim"].run(220, verbose=False)
    assert signature(a) == signature(b)

    static = build(None)
    static["sim"].run(220, verbose=False)
    assert signature(static) != signature(a)


def test_users_join_over_time_and_only_active_users_are_served():
    system = build({"initial_user_share": 0.5, "users_joining_per_tick": 2})
    world, dynamics = system["world"], system["world_dynamics"]
    start = set(world.active_user_ids)
    assert len(start) == round(0.5 * len(world.user_objects))

    system["sim"].run(10, verbose=False)
    assert len(world.active_user_ids) == len(start) + 20
    joined = {user for _, user in dynamics.joined_log}
    served = {e.user_id for e in system["datalake"].events}
    assert served <= start | joined

    system["sim"].run(400, verbose=False)
    assert world.active_user_ids is None and not dynamics.waiting


def test_taste_drift_moves_preferences_towards_one_genre():
    system = build({"taste_drift_every": 5, "taste_drift_share": 0.1, "taste_drift_strength": 0.8})
    system["sim"].run(5, verbose=False)
    log = system["world_dynamics"].drift_log
    assert len(log) == round(0.1 * len(system["world"].user_objects))
    _, user_id, genre = log[0]
    prefs = system["world"].user_states[user_id].genre_preferences
    assert prefs[genre] / sum(prefs.values()) >= 0.8


def test_fatigue_and_rewatch_lower_click_probability():
    system = build({"fatigue": 0.2, "rewatch": 0.0})
    simulator = system["world"].user_simulator
    assert simulator.exposure_factor(1, 50) == 1.0
    simulator.impressions[(1, 50)] = 2
    assert simulator.exposure_factor(1, 50) == pytest.approx(0.64)
    simulator.watched.add((1, 50))
    assert simulator.exposure_factor(1, 50) == 0.0


def test_fatigue_wears_off_with_recovery():
    system = build({"fatigue": 0.2, "fatigue_recovery": 100})
    simulator = system["world"].user_simulator
    simulator.impressions[(1, 50)] = 2
    simulator.impression_ticks[(1, 50)] = 10
    simulator.now = 10
    assert simulator.exposure_factor(1, 50) == pytest.approx(0.64)
    simulator.now = 110                      # one half-life later: 1 impression's worth
    assert simulator.exposure_factor(1, 50) == pytest.approx(0.8)
    simulator.now = 10_000
    assert simulator.exposure_factor(1, 50) == pytest.approx(1.0, abs=1e-6)

    # In a run, a repeat impression adds to the decayed count.
    system["sim"].run(30, verbose=False)
    assert all(t <= 30 for t in simulator.impression_ticks.values())
    assert any(c != int(c) for c in simulator.impressions.values())


def test_unknown_dynamics_key_is_rejected():
    with pytest.raises(ValueError):
        build({"taste_drift": 1})
