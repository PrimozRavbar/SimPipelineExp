"""Rich world (world/taste.py): hidden tastes and movie quality from the
real ratings, mixed into the single definition of relevance."""

import numpy as np
import pytest

from world.external_data_loader import load_movielens
from world.relevance import (calibrate_threshold, is_relevant, relevance, relevance_matrix,
                             relevance_threshold, RELEVANCE_THRESHOLD)
from world.taste import GENRE_WEIGHT, fit_taste_model, ratings_from_user_states


@pytest.fixture(scope="module")
def world_data():
    data = load_movielens()
    taste = fit_taste_model(ratings_from_user_states(data["user_states"]))
    users = [data["user_states"][u] for u in sorted(data["user_states"])]
    movies = [data["movie_objects"][m] for m in sorted(data["movie_objects"])]
    calibrate_threshold(taste, users, movies)
    return data, taste, users, movies


def test_fit_is_deterministic_and_fits_the_ratings(world_data):
    data, taste, _, _ = world_data
    again = fit_taste_model(ratings_from_user_states(load_movielens()["user_states"]))
    assert again is taste or np.allclose(again.movie_factors, taste.movie_factors)
    assert taste.fit_rmse < 0.8
    assert all(0.0 <= taste.score(u, m) <= 1.0 for u in (1, 50, 900) for m in (1, 100, 1600))


def test_scalar_and_matrix_relevance_agree(world_data):
    _, taste, users, movies = world_data
    some_users, some_movies = users[::97], movies[::151]
    matrix = relevance_matrix(some_users, some_movies, taste=taste)
    for i, u in enumerate(some_users):
        for j, m in enumerate(some_movies):
            assert matrix[i, j] == pytest.approx(relevance(u, m, taste=taste))
            genre = relevance(u, m)
            assert matrix[i, j] == pytest.approx(GENRE_WEIGHT * genre + (1 - GENRE_WEIGHT) * taste.score(u.user_id, m.movie_id))


def test_threshold_gives_the_median_user_ten_percent(world_data):
    _, taste, users, movies = world_data
    rel = relevance_matrix(users, movies, taste=taste)
    share = (rel >= relevance_threshold(taste)).mean(axis=1)
    assert np.median(share) == pytest.approx(0.10, abs=0.01)
    assert relevance_threshold(None) == RELEVANCE_THRESHOLD
    assert is_relevant(users[0], movies[0], taste=taste) == (rel[0, 0] >= taste.threshold)


def test_genres_alone_cannot_find_the_best_movies(world_data):
    """The point of the rich world: the best slate by genre alone falls
    clearly short of the true best slate."""
    _, taste, users, movies = world_data
    rich = relevance_matrix(users, movies, taste=taste)
    genre = relevance_matrix(users, movies)
    by_genre = np.take_along_axis(rich, np.argsort(-genre, axis=1, kind="stable")[:, :10], axis=1).mean()
    best = np.sort(rich, axis=1)[:, ::-1][:, :10].mean()
    assert best - by_genre > 0.05


torch = pytest.importorskip("torch")

from ml_system.build_recommendation_system import build_recommendation_system  # noqa: E402


def test_rich_build_runs_and_is_reproducible():
    def run():
        data = load_movielens()
        system = build_recommendation_system(data["movie_objects"], data["user_objects"], data["user_states"],
                                             data["movie_states"], seed=6, users_per_tick=4, log_metrics=False,
                                             taste="rich")
        system["sim"].run(30, verbose=False)
        return system, [(type(e).__name__, e.timestamp, e.user_id, getattr(e, "movie_id", None))
                        for e in system["datalake"].events]

    (a, ea), (_, eb) = run(), run()
    assert ea == eb
    assert a["world"].taste is not None
    with pytest.raises(ValueError):
        data = load_movielens()
        build_recommendation_system(data["movie_objects"], data["user_objects"], data["user_states"],
                                    data["movie_states"], log_metrics=False, taste="spicy")


def test_personal_world_scales_hidden_tastes_and_leaves_rich_unchanged():
    from ml_system.build_recommendation_system import build_recommendation_system
    from world.external_data_loader import load_movielens
    from world.taste import PERSONAL_HIDDEN_SCALE

    d = load_movielens()
    rich = build_recommendation_system(d["movie_objects"], d["user_objects"], d["user_states"], d["movie_states"],
                                       seed=1, log_metrics=False, taste="rich")["world"].taste
    d = load_movielens()
    personal = build_recommendation_system(d["movie_objects"], d["user_objects"], d["user_states"], d["movie_states"],
                                           seed=1, log_metrics=False, taste="personal")["world"].taste
    assert rich.hidden_scale == 1.0 and personal.hidden_scale == PERSONAL_HIDDEN_SCALE
    assert abs(rich.threshold - 0.660) < 0.002            # unchanged by the new world
    assert personal.threshold > rich.threshold
    u, m = 1, 50
    base = rich.mu + rich.quality(m)
    hidden = float(rich.user_factors[rich.user_index[u]] @ rich.movie_factors[rich.movie_index[m]])
    assert abs(personal.score(u, m) - min(1, max(0, (base + 3 * hidden - 1) / 4))) < 1e-9
