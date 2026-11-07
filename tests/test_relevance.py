"""Ground-truth relevance and the relevance-driven click model. No PyTorch needed."""

import random
from types import SimpleNamespace

import pytest

from world.entities import Movie
from world.external_data_loader import load_movielens
from world.relevance import (
    RELEVANCE_THRESHOLD,
    genre_share,
    is_relevant,
    relevance,
    relevance_above_threshold,
)
from world.user_simulator import UserSimulator


@pytest.fixture(scope="module")
def data():
    return load_movielens()


def user_state_with(preferences):
    return SimpleNamespace(genre_preferences=preferences)


def movie_with(genres, movie_id=1):
    return Movie(movie_id=movie_id, title="t", release_date="", genres=genres)


def test_relevance_bounds_and_exact_match():
    drama_fan = user_state_with({"Drama": 40})

    assert relevance(drama_fan, movie_with(["Drama"])) == pytest.approx(1.0)
    assert relevance(drama_fan, movie_with(["Horror"])) == 0.0
    assert relevance(drama_fan, movie_with([])) == 0.0
    assert relevance(user_state_with({}), movie_with(["Drama"])) == 0.0

    assert is_relevant(drama_fan, movie_with(["Drama"]))
    assert not is_relevant(drama_fan, movie_with(["Horror"]))
    assert relevance_above_threshold(drama_fan, movie_with(["Horror"])) == 0.0
    assert relevance_above_threshold(drama_fan, movie_with(["Drama"])) == pytest.approx(1.0)


def test_relevance_in_unit_interval_on_real_data(data):
    user_states = data["user_states"]
    movies = list(data["movie_objects"].values())

    for user_id in list(user_states)[:50]:
        share = genre_share(user_states[user_id])
        for movie in movies:
            assert 0.0 <= relevance(user_states[user_id], movie, share=share) <= 1.0


def test_relevant_fraction_is_selective(data):
    """Regression: with the old genre-overlap rule almost every movie counted."""
    user_states = data["user_states"]
    movies = list(data["movie_objects"].values())

    fractions = []
    for user_id in user_states:
        share = genre_share(user_states[user_id])
        relevant = sum(
            is_relevant(user_states[user_id], movie, share=share)
            for movie in movies
        )
        fractions.append(relevant / len(movies))

    mean_fraction = sum(fractions) / len(fractions)
    assert 0.05 < mean_fraction < 0.35, (RELEVANCE_THRESHOLD, mean_fraction)


def click_through_rate(data, slate_for_user, seed=1):
    world = SimpleNamespace(
        user_states=data["user_states"],
        movie_objects=data["movie_objects"],
    )
    simulator = UserSimulator(world, rng=random.Random(seed))
    users = list(data["user_objects"].values())

    clicks = sum(
        simulator.choose_click(user, slate_for_user(user)) is not None
        for user in users
    )
    return clicks / len(users)


def ranked_slates(data, best_first):
    user_states = data["user_states"]
    movies = data["movie_objects"]

    def slate(user):
        state = user_states[user.user_id]
        share = genre_share(state)
        ranked = sorted(
            movies,
            key=lambda movie_id: relevance(state, movies[movie_id], share=share),
            reverse=best_first,
        )
        return ranked[:100]

    return slate


def test_clicks_depend_on_relevance(data):
    """Regression: clicks used to be ~60% whatever the model showed."""
    movie_ids = list(data["movie_objects"])
    rng = random.Random(2)

    oracle = click_through_rate(data, ranked_slates(data, best_first=True))
    random_slate = click_through_rate(data, lambda user: rng.sample(movie_ids, 100))
    worst = click_through_rate(data, ranked_slates(data, best_first=False))

    assert oracle > 2 * random_slate, (oracle, random_slate)
    assert random_slate > worst, (random_slate, worst)
    assert worst < 0.15, worst


def test_rank_order_matters(data):
    """The same 100 movies earn more clicks with the relevant ones on top."""
    best_first = ranked_slates(data, best_first=True)

    top = click_through_rate(data, best_first)
    reversed_slate = click_through_rate(data, lambda user: best_first(user)[::-1])

    assert top > reversed_slate, (top, reversed_slate)
