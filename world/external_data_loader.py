from pathlib import Path
from collections import defaultdict

import pandas as pd

from world.entities import (
    User,
    Movie,
    Rating,
    UserState,
    MovieState,
)


# The MovieLens 100k folder that ships with the repo.
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "ml-100k"


GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def load_movielens(base_path=None):
    """Load MovieLens 100k. Defaults to the ml-100k folder in the repo,
    so it works from any working directory (local, Colab, cloud)."""
    base_path = Path(base_path) if base_path is not None else DEFAULT_DATA_PATH

    ratings = pd.read_csv(
        base_path / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    movies = pd.read_csv(
        base_path / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
    )

    users = pd.read_csv(
        base_path / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )

    user_objects = {
        row.user_id: User(
            row.user_id,
            row.age,
            row.gender,
            row.occupation,
            row.zip_code,
        )
        for row in users.itertuples(index=False)
    }

    movie_objects = {}

    for row in movies.itertuples(index=False):
        genres = [
            genre
            for genre, flag in zip(GENRE_NAMES, row[5:])
            if flag == 1
        ]

        movie_objects[row[0]] = Movie(
            movie_id=row[0],
            title=row[1],
            release_date=row[2],
            genres=genres,
        )

    rating_objects = [
        Rating(*row)
        for row in ratings.itertuples(index=False, name=None)
    ]

    user_states = {
        user_id: UserState(user)
        for user_id, user in user_objects.items()
    }

    initialize_user_states(
        user_states,
        movie_objects,
        rating_objects,
    )

    movie_states = {
        movie_id: MovieState(movie)
        for movie_id, movie in movie_objects.items()
    }

    initialize_movie_states(
        user_states,
        movie_states,
        rating_objects,
    )

    return {
        "user_objects": user_objects,
        "movie_objects": movie_objects,
        "rating_objects": rating_objects,
        "user_states": user_states,
        "movie_states": movie_states,
    }


def initialize_user_states(
    user_states,
    movie_objects,
    rating_objects,
):
    for rating in rating_objects:
        state = user_states[rating.user_id]

        state.watched_movies.add(rating.movie_id)
        state.ratings[rating.movie_id] = rating.rating

        movie = movie_objects[rating.movie_id]

        for genre in movie.genres:
            state.genre_preferences[genre] = (
                state.genre_preferences.get(genre, 0) + rating.rating
            )

    for state in user_states.values():
        if state.ratings:
            state.average_rating = (
                sum(state.ratings.values())
                / len(state.ratings)
            )

        state.activity_level = len(state.ratings)


def initialize_movie_states(
    user_states,
    movie_states,
    rating_objects,
):
    movie_rating_sum = defaultdict(float)

    for rating in rating_objects:
        user_states[rating.user_id].watched_movies.add(
            rating.movie_id
        )

        movie_states[rating.movie_id].num_ratings += 1
        movie_rating_sum[rating.movie_id] += rating.rating

    for movie_id, state in movie_states.items():
        if state.num_ratings > 0:
            state.average_rating = (
                movie_rating_sum[movie_id]
                / state.num_ratings
            )
