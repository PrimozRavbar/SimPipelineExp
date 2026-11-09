"""Ground-truth relevance of a movie for a simulated user.

This is the single definition of "relevant" (see CLAUDE.md, "Relevance").
It reads World-side user state, so the ML system must never call it to make
recommendations; it is for the simulated users and for evaluation only.
Pure Python and deterministic: no RNG, no torch.
"""

import math

# Median MovieLens user: ~10% of the catalog is relevant (mean ~17%, skewed
# up by users whose taste matches a big genre such as Drama).
RELEVANCE_THRESHOLD = 0.65


def genre_share(user_state):
    """User's genre preferences normalised to sum to 1 (empty dict if none)."""
    preferences = {
        genre: max(float(value), 0.0)
        for genre, value in user_state.genre_preferences.items()
    }
    total = sum(preferences.values())

    if total <= 0:
        return {}

    return {genre: value / total for genre, value in preferences.items()}


def relevance(user_state, movie, share=None):
    """Cosine similarity in [0, 1] between the user's genre-share vector and
    the movie's 0/1 genre vector. Pass a precomputed `share` to save work
    when scoring many movies for the same user."""
    share = genre_share(user_state) if share is None else share
    genres = set(movie.genres)

    if not share or not genres:
        return 0.0

    user_norm = math.sqrt(sum(value * value for value in share.values()))
    dot = sum(share.get(genre, 0.0) for genre in genres)

    return min(1.0, dot / (user_norm * math.sqrt(len(genres))))


def is_relevant(user_state, movie, threshold=RELEVANCE_THRESHOLD, share=None):
    return relevance(user_state, movie, share=share) >= threshold


def relevance_above_threshold(user_state, movie, threshold=RELEVANCE_THRESHOLD, share=None):
    """How far above the threshold the movie is, rescaled to [0, 1].
    0 for every non-relevant movie, 1 for a perfect genre match."""
    rel = relevance(user_state, movie, share=share)
    return max(0.0, (rel - threshold) / (1.0 - threshold))


def relevance_matrix(user_states, movies):
    """rel(u, m) for many users and movies at once, as a numpy array of
    shape (len(user_states), len(movies)). Same definition as relevance()
    (tests check they agree); used to find oracle slates for reference
    metrics without 1.6M Python calls.

    `user_states` and `movies` are sequences (rows/columns in that order).
    """
    import numpy as np

    from world.external_data_loader import GENRE_NAMES

    index = {genre: i for i, genre in enumerate(GENRE_NAMES)}

    users = np.zeros((len(user_states), len(GENRE_NAMES)))
    for row, state in enumerate(user_states):
        for genre, value in genre_share(state).items():
            if genre in index:
                users[row, index[genre]] = value

    items = np.zeros((len(movies), len(GENRE_NAMES)))
    for row, movie in enumerate(movies):
        for genre in set(movie.genres):
            if genre in index:
                items[row, index[genre]] = 1.0

    user_norm = np.linalg.norm(users, axis=1, keepdims=True)
    item_norm = np.linalg.norm(items, axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (users / user_norm) @ (items / item_norm).T

    return np.clip(np.nan_to_num(rel), 0.0, 1.0)
