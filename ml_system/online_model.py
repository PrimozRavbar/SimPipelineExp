"""An online model of user taste that learns from everything the game logs.

The batch models (two-tower, ranker) learn from slices of the log and only
at retrains. This model is updated event by event, like ItemSignals: Flink
keeps one live (Flink.track_online_model, published to Redis as
("online_model", "all")) and anything that needs it point in time can
replay the log through another instance. It uses no World ground truth.

What it uses, all from the event log or the account/catalog:
- every shown slate: the clicked movie is a positive; movies shown above
  it were (possibly) examined and skipped; an unclicked slate is a weak
  negative for all its movies. A learned per-position term in the click
  logit absorbs position bias (the "shallow tower" idea), so the movie and
  user parameters learn relevance, not slate position;
- watch completion and ratings after a click: graded evidence of how
  relevant the clicked movie was (a second head predicts a 0..1 grade);
- a click that was not followed by a watch: weaker evidence;
- onboarding genres (a learned prior on the user's genre weights) and
  demographics (age bracket, gender, occupation: learned shared genre and
  factor offsets, so a new user is not a blank);
- the catalog: movie genres and release year;
- per (user, movie) history: times already shown, already watched (repeat
  impressions and rewatches are learned, not hard-coded).

Score of movie m for user u (the relevance logit):

    s(u, m) = b_m + y(year_m) + g_m . (w0 + D^T d_u + a * o_u + w_u)
              + q_m . (E^T d_u + p_u)
              + h1 * log(1 + shown(u, m)) + h2 * watched(u, m)

with g_m the movie's normalised genre vector, o_u the onboarding genres,
d_u the demographic one-hot. Click logit = c + s + pos[i]. Grade head:
grade = sigmoid(c2 + k * s). Trained by Adagrad SGD, one step per event,
in log order, so the same events always give the same model (factor
initialisation uses its own seeded generator).
"""

import math

import numpy as np

from ml_system.demographics import AGE_BRACKETS

NUM_POSITIONS = 32


def _year(release_date):
    try:
        return int(str(release_date)[-4:])
    except ValueError:
        return None


class OnlineTasteModel:

    def __init__(self, movie_objects, user_objects=None, genre_names=None, factors=16, seed=0,
                 learning_rate=0.05, reg=0.01, factor_reg=0.02, grade_weight=1.0,
                 negative_weight=1.0, unclicked_weight=0.5, uncertainty=0.0):
        from world.external_data_loader import GENRE_NAMES

        self.genre_names = [g for g in (genre_names or GENRE_NAMES) if g != "unknown"]
        G = len(self.genre_names)
        self.movie_ids = np.array(sorted(movie_objects))
        size = int(self.movie_ids.max()) + 1
        self.genres = np.zeros((size, G))
        self.year_bucket = np.zeros(size, dtype=int)
        gi = {g: i for i, g in enumerate(self.genre_names)}
        for m, movie in movie_objects.items():
            for g in movie.genres:
                if g in gi:
                    self.genres[m, gi[g]] = 1.0
            year = _year(getattr(movie, "release_date", ""))
            self.year_bucket[m] = 0 if year is None else 1 + min(max((year - 1920) // 10, 0), 8)
        norms = np.linalg.norm(self.genres, axis=1, keepdims=True)
        self.genres = np.divide(self.genres, norms, out=np.zeros_like(self.genres), where=norms > 0)

        # Demographics: user id -> list of active one-hot columns.
        self.demo = {}
        occupations = sorted({u.occupation for u in (user_objects or {}).values()})
        self.num_demo = len(AGE_BRACKETS) + 1 + 2 + len(occupations)
        for uid, user in (user_objects or {}).items():
            age = sum(user.age >= b for b in AGE_BRACKETS)
            gender = len(AGE_BRACKETS) + 1 + (1 if user.gender == "F" else 0)
            occ = len(AGE_BRACKETS) + 3 + occupations.index(user.occupation)
            self.demo[uid] = [age, gender, occ]

        rng = np.random.default_rng(seed)
        self.k = factors
        self.lr, self.reg, self.factor_reg = learning_rate, reg, factor_reg
        self.grade_weight = grade_weight
        self.negative_weight = negative_weight
        self.unclicked_weight = unclicked_weight
        self.uncertainty = uncertainty

        # Parameters (and Adagrad accumulators, same shapes, prefixed g_).
        self.params = {
            "c": np.array([-3.0]), "c2": np.array([0.0]), "kappa": np.array([0.5]),
            "pos": np.zeros(NUM_POSITIONS),
            "b": np.zeros(size), "year": np.zeros(10),
            "w0": np.zeros(G), "D": np.zeros((self.num_demo, G)), "alpha": np.array([0.0]),
            "E": np.zeros((self.num_demo, factors)),
            "q": rng.normal(0, 0.05, (size, factors)),
            "h": np.zeros(2),
        }
        self.user_w = {}              # user -> genre deviation
        self.user_p = {}              # user -> factor deviation
        self.acc = {k: np.full_like(v, 1e-6, dtype=float) for k, v in self.params.items()}
        self.acc_w, self.acc_p = {}, {}
        self.onboarding = {}          # user -> 0/1 genre vector
        self.shown = {}               # (user, movie) -> impressions
        self.watched = set()          # (user, movie)
        self.impressions = np.zeros(size)
        self.pending = {}             # recommendation id -> [tick, user, movies, clicked index]
        self.clicked = {}             # (user, movie) -> latest grade waiting for watch/rating
        self.updates = 0
        self.last_tick = None

    # -- representation ----------------------------------------------------

    def _user(self, user_id):
        if user_id not in self.user_w:
            self.user_w[user_id] = np.zeros(len(self.genre_names))
            self.user_p[user_id] = np.zeros(self.k)
            self.acc_w[user_id] = np.full(len(self.genre_names), 1e-6)
            self.acc_p[user_id] = np.full(self.k, 1e-6)
        P = self.params
        d = self.demo.get(user_id, [])
        taste = P["w0"] + self.user_w[user_id] + P["alpha"][0] * self.onboarding.get(user_id, 0.0)
        factor = self.user_p[user_id].copy()
        for j in d:
            taste = taste + P["D"][j]
            factor = factor + P["E"][j]
        return taste, factor, d

    def _history(self, user_id, movies):
        shown = np.array([math.log1p(self.shown.get((user_id, m), 0)) for m in movies])
        watched = np.array([1.0 if (user_id, m) in self.watched else 0.0 for m in movies])
        return shown, watched

    def scores(self, user_id, movies=None):
        """Relevance logits s(u, m) for the given movies (default: catalog)."""
        movies = self.movie_ids if movies is None else np.asarray(movies, dtype=int)
        taste, factor, _ = self._user(user_id)
        P = self.params
        s = P["b"][movies] + P["year"][self.year_bucket[movies]] + self.genres[movies] @ taste \
            + P["q"][movies] @ factor
        if self.shown or self.watched:
            shown, watched = self._history(user_id, movies)
            s = s + P["h"][0] * shown + P["h"][1] * watched
        return s

    def click_probability(self, user_id, movies, position=0):
        P = self.params
        z = P["c"][0] + P["pos"][min(position, NUM_POSITIONS - 1)] + self.scores(user_id, movies)
        return 1.0 / (1.0 + np.exp(-z))

    def top(self, user_id, k, exclude=(), rng=None):
        """The k best movies for a user. With `uncertainty` > 0 and an rng,
        movie biases are sampled (Thompson-style exploration of rarely
        shown movies)."""
        s = self.scores(user_id)
        if rng is not None and self.uncertainty > 0:
            noise = np.array([rng.gauss(0.0, 1.0) for _ in range(len(self.movie_ids))])
            s = s + self.uncertainty * noise / np.sqrt(1.0 + self.impressions[self.movie_ids])
        order = np.lexsort((self.movie_ids, -s))
        excluded = set(exclude)
        out = []
        for i in order:
            m = int(self.movie_ids[i])
            if m not in excluded:
                out.append(m)
                if len(out) == k:
                    break
        return out

    # -- learning ------------------------------------------------------------

    def _step(self, name, grad, index=None):
        acc, param = self.acc[name], self.params[name]
        if index is None:
            acc += grad * grad
            param -= self.lr * grad / np.sqrt(acc)
        else:
            acc[index] += grad * grad
            param[index] -= self.lr * grad / np.sqrt(acc[index])

    def _user_step(self, user_id, gw, gp):
        self.acc_w[user_id] += gw * gw
        self.user_w[user_id] -= self.lr * gw / np.sqrt(self.acc_w[user_id])
        self.acc_p[user_id] += gp * gp
        self.user_p[user_id] -= self.lr * gp / np.sqrt(self.acc_p[user_id])

    def _update(self, user_id, movies, positions, targets, weights, head):
        """One SGD step on a group of (movie, target) for one user.
        head "click": logistic loss on the click logit; "grade": logistic
        loss on sigmoid(c2 + kappa * s) against a 0..1 grade."""
        movies = np.asarray(movies, dtype=int)
        targets = np.asarray(targets, dtype=float)
        weights = np.asarray(weights, dtype=float)
        P = self.params
        taste, factor, demo = self._user(user_id)
        s = self.scores(user_id, movies)
        if head == "click":
            pos = np.minimum(np.asarray(positions), NUM_POSITIONS - 1)
            z = P["c"][0] + P["pos"][pos] + s
            err = weights * (1.0 / (1.0 + np.exp(-z)) - targets)        # dL/dz
            ds = err
            self._step("c", np.array([err.sum()]))
            g_pos = np.zeros(NUM_POSITIONS)
            np.add.at(g_pos, pos, err)
            g_pos[0] = 0.0                                               # position 0 is the reference
            self._step("pos", g_pos)
        else:
            kappa = P["kappa"][0]
            z = P["c2"][0] + kappa * s
            err = weights * (1.0 / (1.0 + np.exp(-z)) - targets)
            ds = err * kappa
            self._step("c2", np.array([err.sum()]))
            self._step("kappa", np.array([(err * s).sum()]))

        g = self.genres[movies]
        q = P["q"][movies]
        # Movie parameters (a slate never repeats a movie: indexed updates).
        self._step("b", ds + self.reg * P["b"][movies], movies)
        gy = np.zeros(10)
        np.add.at(gy, self.year_bucket[movies], ds)
        self._step("year", gy)
        self._step("q", ds[:, None] * factor[None, :] + self.factor_reg * q, movies)
        # User-side parameters.
        g_taste = ds @ g
        g_factor = ds @ q
        self._step("w0", g_taste)
        self._step("alpha", np.array([g_taste @ np.asarray(self.onboarding.get(user_id, np.zeros(len(g_taste))))]))
        if demo:
            gD = np.zeros_like(P["D"])
            gE = np.zeros_like(P["E"])
            for j in demo:
                gD[j] = g_taste + self.reg * P["D"][j]
                gE[j] = g_factor + self.reg * P["E"][j]
            self._step("D", gD)
            self._step("E", gE)
        self._user_step(user_id, g_taste + self.reg * self.user_w[user_id],
                        g_factor + self.factor_reg * self.user_p[user_id])
        if self.shown or self.watched:
            shown, watched = self._history(user_id, movies)
            self._step("h", np.array([ds @ shown, ds @ watched]))
        self.updates += 1

    def _finish_slate(self, user_id, movies, clicked):
        if clicked is None:
            idx = list(range(len(movies)))
            targets = [0.0] * len(idx)
            weights = [self.unclicked_weight] * len(idx)
        else:
            idx = list(range(clicked + 1))
            targets = [0.0] * clicked + [1.0]
            weights = [self.negative_weight] * clicked + [1.0]
        self._update(user_id, [movies[i] for i in idx], idx, targets, weights, "click")
        for m in movies:
            self.shown[(user_id, m)] = self.shown.get((user_id, m), 0) + 1
            self.impressions[m] += 1

    def flush(self, before_tick=None):
        """Learn from every pending slate shown before `before_tick` (all if None)."""
        for rid in list(self.pending):
            tick, user_id, movies, clicked = self.pending[rid]
            if before_tick is None or tick < before_tick:
                del self.pending[rid]
                self._finish_slate(user_id, movies, clicked)

    def observe(self, event):
        if self.last_tick is not None and event.timestamp > self.last_tick:
            self.flush(event.timestamp)
        self.last_tick = event.timestamp
        name = type(event).__name__
        if name == "RecommendationShownEvent":
            self.pending[event.recommendation_id] = [event.timestamp, event.user_id, list(event.movie_ids), None]
        elif name == "RecommendationClickedEvent":
            slate = self.pending.get(event.recommendation_id)
            if slate is not None and event.movie_id in slate[2]:
                slate[3] = slate[2].index(event.movie_id)
            # A click is a weak grade until the watch (or its absence) is known.
            self.clicked[(event.user_id, event.movie_id)] = event.timestamp
        elif name == "WatchEvent":
            self.clicked.pop((event.user_id, event.movie_id), None)
            self.watched.add((event.user_id, event.movie_id))
            self._update(event.user_id, [event.movie_id], None,
                         [min(1.0, max(0.0, (event.completion - 0.1) / 0.85))], [self.grade_weight], "grade")
        elif name == "RatingEvent":
            self._update(event.user_id, [event.movie_id], None,
                         [min(1.0, max(0.0, (event.rating - 1.0) / 4.0))], [self.grade_weight], "grade")
        elif name == "OnboardingEvent":
            vec = np.zeros(len(self.genre_names))
            for g in event.genres:
                if g in self.genre_names:
                    vec[self.genre_names.index(g)] = 1.0
            n = np.linalg.norm(vec)
            self.onboarding[event.user_id] = vec / n if n else vec
        # Clicks with no watch by the next tick: a low grade.
        for (u, m), tick in list(self.clicked.items()):
            if tick < event.timestamp:
                del self.clicked[(u, m)]
                self._update(u, [m], None, [0.3], [0.5 * self.grade_weight], "grade")

    def summary(self):
        P = self.params
        return {"updates": self.updates, "users": len(self.user_w),
                "position_bias": [round(float(x), 2) for x in P["pos"][:10]],
                "onboarding_weight": round(float(P["alpha"][0]), 3),
                "history_weights": [round(float(x), 3) for x in P["h"]]}


# -- serving -----------------------------------------------------------------

from ml_system.model_server import RetrievalService  # noqa: E402
from ml_system.ranking import Ranker  # noqa: E402


class OnlineRetrieval(RetrievalService):
    """Candidates: the online model's best movies from the whole catalog.
    `model_fn()` returns the live model (Redis ("online_model", "all"))."""

    def __init__(self, model_fn, rng=None):
        super().__init__()
        self.model_fn = model_fn
        self.rng = rng

    def retrieve(self, request):
        model = self.model_fn()
        if model is None:
            return []
        return model.top(request.user_id, request.k, rng=self.rng)


class OnlineRanker(Ranker):
    """Orders candidates by the online model's relevance score (ties by
    the incoming order)."""

    def __init__(self, model_fn):
        self.model_fn = model_fn

    def rank(self, user, candidates):
        model = self.model_fn()
        if model is None or not candidates:
            return candidates
        scores = model.scores(user.user_id, list(candidates))
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [candidates[i] for i in order]
