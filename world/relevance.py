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


def relevance(user_state, movie, share=None, taste=None):
    """Cosine similarity in [0, 1] between the user's genre-share vector and
    the movie's 0/1 genre vector. Pass a precomputed `share` to save work
    when scoring many movies for the same user.

    With a `taste` model (world.taste.TasteModel, a "rich" world) the
    relevance is taste.genre_weight * that cosine + (1 - genre_weight) *
    the hidden-taste score of (user, movie)."""
    if taste is not None:
        genre = relevance(user_state, movie, share=share)
        return (taste.genre_weight * genre
                + (1 - taste.genre_weight) * taste.score(user_state.user_id, movie.movie_id))
    share = genre_share(user_state) if share is None else share
    genres = set(movie.genres)

    if not share or not genres:
        return 0.0

    user_norm = math.sqrt(sum(value * value for value in share.values()))
    dot = sum(share.get(genre, 0.0) for genre in genres)

    return min(1.0, dot / (user_norm * math.sqrt(len(genres))))


def relevance_threshold(taste=None):
    """The "relevant" threshold: RELEVANCE_THRESHOLD in the genre world,
    the calibrated one in a rich world."""
    return RELEVANCE_THRESHOLD if taste is None else taste.threshold


def is_relevant(user_state, movie, threshold=None, share=None, taste=None):
    if threshold is None:
        threshold = relevance_threshold(taste)
    return relevance(user_state, movie, share=share, taste=taste) >= threshold


def relevance_above_threshold(user_state, movie, threshold=None, share=None, taste=None):
    """How far above the threshold the movie is, rescaled to [0, 1].
    0 for every non-relevant movie, 1 for a perfect match."""
    if threshold is None:
        threshold = relevance_threshold(taste)
    rel = relevance(user_state, movie, share=share, taste=taste)
    return max(0.0, (rel - threshold) / (1.0 - threshold))


def relevance_matrix(user_states, movies, taste=None):
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

    rel = np.clip(np.nan_to_num(rel), 0.0, 1.0)

    if taste is not None:
        scores = taste.score_matrix([s.user_id for s in user_states], [m.movie_id for m in movies])
        rel = taste.genre_weight * rel + (1 - taste.genre_weight) * scores

    return rel


def calibrate_threshold(taste, user_states, movies, target_share=None):
    """Set taste.threshold so the median user finds `target_share` of the
    catalog relevant (default world.taste.TARGET_RELEVANT_SHARE)."""
    import numpy as np

    from world.taste import TARGET_RELEVANT_SHARE

    target = TARGET_RELEVANT_SHARE if target_share is None else target_share
    rel = relevance_matrix(list(user_states), list(movies), taste=taste)
    per_user = np.quantile(rel, 1.0 - target, axis=1)
    taste.threshold = float(np.median(per_user))
    return taste.threshold
