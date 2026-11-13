"""Collaborative filtering from implicit feedback (weighted ALS).

Learns a few hidden taste numbers per user and per movie from everyone's
clicks, watches, ratings and skips together, so a user's score for a movie
reflects "people whose clicks look like yours liked this", beyond genres.

Model (Hu, Koren & Volinsky 2008, implicit feedback): for every user u and
movie m, a preference p_um (1 = liked, 0 = not) with a confidence c_um:

- clicked:           p = 1, c = 1 + alpha * (1 + completion + max(0, rating - 3) / 2)
- shown and skipped: p = 0, c = 1 + beta * times skipped (examined, not chosen)
- never shown:       p = 0, c = 1 (weak: the user may simply not know it)

minimise  sum c_um (p_um - x_u . y_m)^2 + lam * (n_u ||x_u||^2 + n_m ||y_m||^2)

by alternating least squares (each step is exact; uses the Y^T Y trick so
unseen pairs cost nothing). n_u, n_m are the number of observed pairs, so a
user with little history is pulled to zero (ALS-WR): their factors grow
only as evidence comes in. Deterministic (seeded initialisation).
"""

import numpy as np


class ImplicitMF:

    def __init__(self, num_users, num_movies, factors=16, alpha=10.0, beta=1.0, lam=0.05,
                 iterations=10, seed=0):
        self.k, self.alpha, self.beta, self.lam = factors, alpha, beta, lam
        self.iterations, self.seed = iterations, seed
        self.num_users, self.num_movies = num_users, num_movies
        self.X = np.zeros((num_users, factors))
        self.Y = np.zeros((num_movies, factors))
        self.fitted = False
        self.observed_users = 0

    @staticmethod
    def interactions(events):
        """(user, movie) -> [clicks, completion, best rating, skips] from an
        event log, under the cascade (skips = movies shown above a click, or
        a whole unclicked slate)."""
        out = {}
        slates, clicked = {}, {}
        for e in events:
            name = type(e).__name__
            if name == "RecommendationShownEvent":
                slates[e.recommendation_id] = (e.user_id, list(e.movie_ids))
            elif name == "RecommendationClickedEvent":
                clicked[e.recommendation_id] = e.movie_id
                rec = out.setdefault((e.user_id, e.movie_id), [0, 0.0, 0.0, 0])
                rec[0] += 1
            elif name == "WatchEvent":
                rec = out.setdefault((e.user_id, e.movie_id), [0, 0.0, 0.0, 0])
                rec[1] = max(rec[1], e.completion)
            elif name == "RatingEvent":
                rec = out.setdefault((e.user_id, e.movie_id), [0, 0.0, 0.0, 0])
                rec[2] = max(rec[2], e.rating)
        for rid, (u, movies) in slates.items():
            c = clicked.get(rid)
            upto = movies.index(c) if c in movies else len(movies)
            for m in movies[:upto]:
                rec = out.setdefault((u, m), [0, 0.0, 0.0, 0])
                rec[3] += 1
        return out

    def fit(self, interactions):
        rows = {}
        for (u, m), (clicks, completion, rating, skips) in interactions.items():
            if u >= self.num_users or m >= self.num_movies:
                continue
            if clicks:
                p, c = 1.0, 1.0 + self.alpha * (1.0 + completion + max(0.0, rating - 3.0) / 2.0)
            else:
                p, c = 0.0, 1.0 + self.beta * skips
            rows.setdefault(("u", u), []).append((m, p, c))
            rows.setdefault(("m", m), []).append((u, p, c))
        rng = np.random.default_rng(self.seed)
        X = rng.normal(0, 0.01, (self.num_users, self.k))
        Y = rng.normal(0, 0.01, (self.num_movies, self.k))
        eye = np.eye(self.k)

        def solve(A, B, kind, count):
            BtB = B.T @ B
            out = np.zeros_like(A)
            for i in range(count):
                obs = rows.get((kind, i))
                if not obs:
                    continue
                idx = np.array([o[0] for o in obs])
                p = np.array([o[1] for o in obs])
                c = np.array([o[2] for o in obs])
                Bo = B[idx]
                M = BtB + (Bo.T * (c - 1.0)) @ Bo + self.lam * len(obs) * eye
                out[i] = np.linalg.solve(M, Bo.T @ (c * p))
            return out

        for _ in range(self.iterations):
            X = solve(X, Y, "u", self.num_users)
            Y = solve(Y, X, "m", self.num_movies)
        self.X, self.Y = X, Y
        self.fitted = True
        self.observed_users = sum(1 for key in rows if key[0] == "u")
        return self

    def scores(self, user_id, movies):
        if not self.fitted or user_id >= self.num_users:
            return np.zeros(len(movies))
        return self.Y[np.asarray(movies, dtype=int)] @ self.X[user_id]


# -- serving -----------------------------------------------------------------

from ml_system.model_server import RetrievalService  # noqa: E402


class CFRetrieval(RetrievalService):
    """Candidates: the movies whose hidden-taste vectors best match the
    user's ("people with clicks like yours liked these"), from the whole
    catalog, skipping movies the user already watched. `model_fn()` returns
    the deployed ImplicitMF (None: no candidates)."""

    def __init__(self, model_fn, movie_ids, is_watched=None):
        super().__init__()
        self.model_fn = model_fn
        self.movie_ids = np.array(sorted(movie_ids))
        self.is_watched = is_watched          # (user, movie) -> bool

    def retrieve(self, request):
        model = self.model_fn()
        if model is None or not model.fitted or request.user_id >= model.num_users:
            return []
        if not np.any(model.X[request.user_id]):
            return []                      # no evidence about this user yet
        scores = model.scores(request.user_id, self.movie_ids)
        out = []
        for i in np.lexsort((self.movie_ids, -scores)):
            m = int(self.movie_ids[i])
            if self.is_watched is None or not self.is_watched(request.user_id, m):
                out.append(m)
                if len(out) == request.k:
                    break
        return out
