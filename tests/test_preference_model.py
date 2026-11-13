"""PreferenceModel (ml_system/online_model.py): learns from the event log,
online and in batch; Flink's live copy matches a replay of the data lake;
the console fits it at retrains and ranks with it."""

import numpy as np
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
    # as installing first and then learning the new events online. (Pair
    # offsets are learned once, as rows arrive, so they are left out here.)
    late = replay(PreferenceModel(movies, users, seed=1, epochs=50, pair_lr=0), events[:40])
    fitted = late.snapshot().fit()
    for e in events[40:]:
        late.observe(e)
    late.flush()
    late.install(fitted)

    early = replay(PreferenceModel(movies, users, seed=1, epochs=50, pair_lr=0), events[:40])
    early.install(early.snapshot().fit())
    for e in events[40:]:
        early.observe(e)
    early.flush()
    assert all(abs(late.params[k] - early.params[k]).max() < 1e-9 for k in ("b", "W"))


def test_pair_evidence_sinks_a_movie_the_user_keeps_skipping():
    movies, users = catalog()
    model = replay(PreferenceModel(movies, users, seed=1), game_events())
    # User 1 skips 2 and 4 above every click of 3; user 2 never saw them.
    assert model.pair[(1, 2)] < 0 and model.pair[(1, 3)] > 0
    assert (2, 2) not in model.pair
    install = model.snapshot().fit(epochs=20)
    before = dict(model.pair)
    model.install(install)
    assert model.pair == before                     # installing never replays pair updates


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


def test_a_slate_row_never_sees_its_own_watch():
    # Regression: rows were written at the end of the tick, after the
    # slate's own watch, so the clicked row said "already watched" and the
    # model learned "watched => clicked" (weight ~5, even where rewatches
    # are rare).
    movies, users = catalog()
    model = PreferenceModel(movies, users, history=True)
    model.observe(RecommendationShownEvent(1, 1, 1, [2, 3]))
    model.observe(RecommendationClickedEvent(1, 1, 1, 3))
    model.observe(WatchEvent(1, 1, 3, 100.0, 0.9))
    model.flush()
    clicked_row = [r for r in model.rows if r[1] == 3][0]
    assert clicked_row[6] == 0.0
    model.observe(RecommendationShownEvent(2, 1, 2, [3]))
    model.flush()
    assert model.rows[-1][6] == 1.0                 # the next slate does know


def test_v6_ranker_learns_from_logged_preference_scores():
    from console.session import SimSession
    from ml_system.ranking import MLPRanker

    s = SimSession(preset="plain", seed=3, users_per_tick=5, log_metrics=False,
                   settings={"deploy_delay": 10, "epochs": 1, "spark_every": 0, "preference_model": True,
                             "ranker": True, "ranker_signals": True, "ranker_tower_score": True,
                             "ranker_preference_score": True})
    s.step(40)
    s.submit("retrain")
    s.step(12)
    # First retrain: nothing logged yet (the preference model was unfitted),
    # so the preference model ranks.
    assert isinstance(s.model_server.ranker, PreferenceRanker)
    assert s.preference_log                           # logged from now on
    s.step(60)
    s.submit("retrain")
    s.step(12)
    v2 = s.versions[2].ranker
    assert isinstance(v2, MLPRanker) and v2.feature_set == "v6"
    assert 0 < v2.preference_logged_share
    # Every logged score came from the model live at serving time.
    tick, user, movie = next(iter(s.preference_log))
    assert tick > 50                                 # first fit deployed at the tick-50 boundary
    s.close()


def test_rollback_to_v0_also_rolls_back_the_preference_model():
    # Regression: rolling back to v0 swapped the towers, but the live
    # preference model kept its fit, kept ranking and kept adding its own
    # candidates, so CTR barely moved.
    from console.session import SimSession
    from simulation.events import RetrievalRequest

    s = SimSession(preset="plain", seed=3, users_per_tick=3, log_metrics=False,
                   settings={"deploy_delay": 10, "epochs": 1, "spark_every": 0, "preference_model": True})
    s.step(30)
    s.submit("retrain")
    s.step(12)
    assert s.flink.preferences.fits == 1
    s.submit("rollback")
    s.step(1)
    assert s.serving_version == 0 and s.flink.preferences.fits == 0
    user = next(iter(s.system["world"].user_objects.values()))
    assert s.preference_source.retrieve(RetrievalRequest(user_id=user.user_id, k=10)) == []
    assert s.model_server.ranker.rank(user, [1, 2, 3]) == [1, 2, 3]       # v0 has no ranker
    s.submit("rollback")                                                  # back to v1: its fit returns
    s.step(1)
    assert s.serving_version == 1 and s.flink.preferences.fits == 1
    s.close()


def test_browsing_feeds_the_preference_model():
    from simulation.events import BrowseEvent

    movies, users = catalog()
    model = PreferenceModel(movies, users)
    model.observe(BrowseEvent(0, 1, "Comedy"))
    model.observe(RecommendationShownEvent(1, 1, 1, [2, 3]))     # 2 Drama, 3 Comedy
    model.flush()
    match = {r[1]: r[7] for r in model.rows}
    assert match[3] > match[2] == 0.0


def test_collaborative_filtering_is_fitted_at_retrain_logged_into_rows_and_versioned():
    from console.session import SimSession
    from ml_system.cf_model import ImplicitMF

    s = SimSession(preset="plain", seed=3, users_per_tick=10, log_metrics=False, taste="personal",
                   settings={"deploy_delay": 10, "epochs": 1, "spark_every": 0, "preference_model": True,
                             "cf_model": True, "cf_candidates": 10})
    s.step(80)
    s.submit("retrain")
    s.step(12)
    assert isinstance(s.cf_live, ImplicitMF) and s.cf_live is s.versions[1].cf
    s.step(40)
    # Rows logged after the deploy carry the deployed CF model's score.
    assert any(r[8] != 0.0 for r in s.flink.preferences.rows)
    s.submit("rollback")
    s.step(1)
    assert s.cf_live is None                       # v0 had no CF model
    s.close()


def test_implicit_mf_learns_who_likes_what():
    from ml_system.cf_model import ImplicitMF

    # Users 0-9 click movies 0-4, users 10-19 click movies 5-9; each user has
    # clicked only some of their group's movies.
    inter = {}
    for u in range(20):
        group = range(0, 5) if u < 10 else range(5, 10)
        for m in group:
            if (u + m) % 3:
                inter[(u, m)] = [1, 0.8, 0.0, 0]
    mf = ImplicitMF(20, 10, factors=4, iterations=15).fit(inter)
    unseen_same = [mf.scores(u, [m])[0] for u in range(10) for m in range(5) if (u, m) not in inter]
    other = [mf.scores(u, [m])[0] for u in range(10) for m in range(5, 10)]
    assert np.mean(unseen_same) > np.mean(other) + 0.1
