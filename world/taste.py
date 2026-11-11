"""Hidden tastes beyond genres, learned from the real MovieLens ratings.

World-side only: the ML system never sees any of this; it can only learn
it from in-game behaviour. Used by world.relevance when the World is built
with taste="rich" (the default world stays genre-only).

The model is a biased matrix factorisation of the 100k MovieLens ratings,

    predicted rating = mu + b_movie + p_user . q_movie

fitted by alternating least squares with fixed settings (seeded init), so
the true tastes are the same in every game whatever the game seed. The user
bias is left out of relevance on purpose: generous raters are not "more
interested" in every movie. b_movie is the movie's quality (how much better
or worse it is liked than average, shrunk towards 0 for movies with few
ratings); p . q are hidden taste factors (audiences, styles) that genres
don't capture.

Relevance in a rich world (world.relevance) is

    rel = GENRE_WEIGHT * genre cosine + (1 - GENRE_WEIGHT) * taste score

with taste score = clip((mu + b_movie + p . q - 1) / 4, 0, 1), and a
relevance threshold calibrated so that the median user finds
TARGET_RELEVANT_SHARE of the catalog relevant (as in the genre world).
"""

import numpy as np

GENRE_WEIGHT = 0.4
TARGET_RELEVANT_SHARE = 0.10

FACTORS = 16
FACTOR_REG = 8.0
BIAS_REG = 10.0
ITERATIONS = 15
SEED = 0

_CACHE = {}


class TasteModel:

    def __init__(self, mu, user_ids, movie_ids, user_factors, movie_factors, movie_bias,
                 genre_weight=GENRE_WEIGHT):
        self.mu = float(mu)
        self.user_index = {u: i for i, u in enumerate(user_ids)}
        self.movie_index = {m: i for i, m in enumerate(movie_ids)}
        self.user_factors = user_factors
        self.movie_factors = movie_factors
        self.movie_bias = movie_bias
        self.genre_weight = genre_weight
        self.threshold = None          # set by calibrate()

    def score(self, user_id, movie_id):
        """Taste score in [0, 1] (0 for unknown users or movies)."""
        u = self.user_index.get(user_id)
        m = self.movie_index.get(movie_id)
        if u is None or m is None:
            return 0.0
        rating = self.mu + self.movie_bias[m] + float(self.user_factors[u] @ self.movie_factors[m])
        return min(1.0, max(0.0, (rating - 1.0) / 4.0))

    def score_matrix(self, user_ids, movie_ids):
        rows = np.zeros((len(user_ids), len(movie_ids)))
        u_idx = [self.user_index.get(u) for u in user_ids]
        m_idx = [self.movie_index.get(m) for m in movie_ids]
        known_u = [i for i, u in enumerate(u_idx) if u is not None]
        known_m = [j for j, m in enumerate(m_idx) if m is not None]
        if known_u and known_m:
            P = self.user_factors[[u_idx[i] for i in known_u]]
            Q = self.movie_factors[[m_idx[j] for j in known_m]]
            b = self.movie_bias[[m_idx[j] for j in known_m]]
            ratings = self.mu + b[None, :] + P @ Q.T
            rows[np.ix_(known_u, known_m)] = np.clip((ratings - 1.0) / 4.0, 0.0, 1.0)
        return rows

    def quality(self, movie_id):
        m = self.movie_index.get(movie_id)
        return 0.0 if m is None else float(self.movie_bias[m])


def fit_taste_model(ratings, factors=FACTORS, factor_reg=FACTOR_REG, bias_reg=BIAS_REG,
                    iterations=ITERATIONS, seed=SEED):
    """ratings: iterable of (user_id, movie_id, rating). Deterministic."""
    triples = sorted((int(u), int(m), float(r)) for u, m, r in ratings)
    key = (len(triples), hash(tuple(triples[:50])), hash(tuple(triples[-50:])),
           factors, factor_reg, bias_reg, iterations, seed)
    if key in _CACHE:
        return _CACHE[key]

    user_ids = sorted({u for u, _, _ in triples})
    movie_ids = sorted({m for _, m, _ in triples})
    ui = {u: i for i, u in enumerate(user_ids)}
    mi = {m: i for i, m in enumerate(movie_ids)}
    U = np.array([ui[u] for u, _, _ in triples])
    M = np.array([mi[m] for _, m, _ in triples])
    R = np.array([r for _, _, r in triples])

    mu = R.mean()
    # Biases: regularised means, movie first (quality), then user.
    movie_bias = np.zeros(len(movie_ids))
    np.add.at(movie_bias, M, R - mu)
    movie_counts = np.bincount(M, minlength=len(movie_ids))
    movie_bias = movie_bias / (movie_counts + bias_reg)
    user_bias = np.zeros(len(user_ids))
    np.add.at(user_bias, U, R - mu - movie_bias[M])
    user_bias = user_bias / (np.bincount(U, minlength=len(user_ids)) + bias_reg)
    residual = R - mu - movie_bias[M] - user_bias[U]

    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, (len(user_ids), factors))
    Q = rng.normal(0, 0.1, (len(movie_ids), factors))
    by_user = [np.where(U == i)[0] for i in range(len(user_ids))]
    by_movie = [np.where(M == j)[0] for j in range(len(movie_ids))]
    eye = np.eye(factors)
    for _ in range(iterations):
        for i, idx in enumerate(by_user):
            Qi = Q[M[idx]]
            P[i] = np.linalg.solve(Qi.T @ Qi + factor_reg * eye, Qi.T @ residual[idx])
        for j, idx in enumerate(by_movie):
            Pj = P[U[idx]]
            Q[j] = np.linalg.solve(Pj.T @ Pj + factor_reg * eye, Pj.T @ residual[idx])

    model = TasteModel(mu, user_ids, movie_ids, P, Q, movie_bias)
    model.fit_rmse = float(np.sqrt(np.mean((residual - np.sum(P[U] * Q[M], axis=1)) ** 2)))
    _CACHE[key] = model
    return model


def ratings_from_user_states(user_states):
    """(user, movie, rating) from the World's pre-game ratings."""
    return [(u, m, r) for u, state in user_states.items() for m, r in state.ratings.items()]
