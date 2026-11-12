"""PreferenceModel (ml_system/online_model.py): learns from the event log,
online and in batch; Flink's live copy matches a replay of the data lake;
the console fits it at retrains and ranks with it."""

import pytest

torch = pytest.importorskip("torch")

from ml_system.online_model import PreferenceModel, PreferenceRanker
from simulation.events import (OnboardingEvent, RecommendationClickedEvent, RecommendationShownEvent,
                               WatchEvent)
from world.entities import Movie, User


def catalog():
    movies = {i: Movie(i, "m%d" % i, "01-Jan-1995", ["Comedy"] if i % 2 else ["Drama"]) for i in range(1, 21)}
    users = {u: User(u, 30, "M", "engineer", "00000") for u in range(1, 4)}
    return movies, users


def game_events():
    """User 1 always clicks movie 3 (a comedy) when shown at position 2."""
    events, rid = [], 0
    for tick in range(1, 40):
        rid += 1
        slate = [2, 4, 3, 6, 8]
        events.append(RecommendationShownEvent(tick, 1, rid, slate))
        events.append(RecommendationClickedEvent(tick, 1, rid, 3))
        events.append(WatchEvent(tick, 1, 3, 100.0, 0.9))
        rid += 1
        events.append(RecommendationShownEvent(tick, 2, rid, [5, 7, 9]))
    return [OnboardingEvent(0, 1, ["Comedy"])] + events


def replay(model, events):
    for e in events:
        model.observe(e)
    model.flush()
    return model


def test_learns_clicked_over_skipped_online_and_in_batch():
    movies, users = catalog()
    online = replay(PreferenceModel(movies, users, seed=1), game_events())
    s = online.scores(1, [3, 2, 4])
    assert s[0] > s[1] and s[0] > s[2]              # clicked beats skipped above it
    batch = replay(PreferenceModel(movies, users, seed=1, online_lr=0.0, epochs=100), game_events()).fit()
    s = batch.scores(1, [3, 2, 4])
    assert s[0] > s[1] and s[0] > s[2]
    assert batch.params["pos"][0] == pytest.approx(batch.params["pos"][0])
    assert batch.top(1, 1) == [3]


def test_same_events_same_model_and_install_replays_new_rows():
    movies, users = catalog()
    events = game_events()
    a = replay(PreferenceModel(movies, users, seed=1, epochs=50), events[:40])
    b = replay(PreferenceModel(movies, users, seed=1, epochs=50), events[:40])
    fa, fb = a.snapshot().fit(), b.snapshot().fit()
    assert all((fa.params[k] == fb.params[k]).all() for k in fa.params)

    # Install a fit made on a snapshot, after more events arrived: the same
    # as installing first and then learning the new events online.
    late = replay(PreferenceModel(movies, users, seed=1, epochs=50), events[:40])
    fitted = late.snapshot().fit()
    for e in events[40:]:
        late.observe(e)
    late.flush()
    late.install(fitted)

    early = replay(PreferenceModel(movies, users, seed=1, epochs=50), events[:40])
    early.install(early.snapshot().fit())
    for e in events[40:]:
        early.observe(e)
    early.flush()
    assert all(abs(late.params[k] - early.params[k]).max() < 1e-9 for k in ("b", "W"))


def test_ranker_falls_back_until_first_fit():
    movies, users = catalog()
    model = PreferenceModel(movies, users)

    class Reverse:
        def rank(self, user, candidates):
            return list(reversed(candidates))

    ranker = PreferenceRanker(lambda: model, fallback=Reverse())
    assert ranker.rank(users[1], [1, 2, 3]) == [3, 2, 1]
    replay(model, game_events()).fit(epochs=50)
    assert ranker.rank(users[1], [2, 3, 4])[0] == 3


def test_console_flink_copy_matches_a_replay_of_the_data_lake():
    from console.session import SimSession

    s = SimSession(preset="plain", seed=3, users_per_tick=3, log_metrics=False,
                   settings={"deploy_delay": 10, "epochs": 1, "spark_every": 0, "preference_model": True,
                             "onboarding": True})
    s.step(30)
    s.submit("retrain")
    s.step(12)
    live = s.flink.preferences
    assert s.serving_version == 1 and s.versions[1].preferences.fits == 1
    assert isinstance(s.model_server.ranker, PreferenceRanker)
    assert s.system["redis"].get("preference_model", "all") is live

    # Flink saw every event (slates, clicks, watches, onboarding) in order.
    world = s.system["world"]
    replayed = PreferenceModel(world.movie_objects, world.user_objects, seed=s.seed, online_lr=0.0)
    ticks = {}
    for e in s.system["datalake"].events:
        ticks.setdefault(e.timestamp, []).append(e)
    for tick in sorted(ticks):
        replayed.observe_tick(ticks[tick])
        replayed.flush(tick + 1)
    assert replayed.rows == live.rows and replayed.grades == live.grades
    assert replayed.onboarding.keys() == live.onboarding.keys()
    s.close()
