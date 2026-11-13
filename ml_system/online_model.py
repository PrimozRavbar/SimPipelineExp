"""A preference model that learns from everything the game logs.

The two-tower model and the ranker each learn from a slice of the log, and
only at retrains. PreferenceModel uses every logged signal, is refitted in
batch at every retrain and is updated online, event by event, in between
(Flink.track_preferences keeps the live instance, published to Redis as
("preference_model", "all")). It uses no World ground truth.

Signals, all from the event log or the account/catalog:
- every shown slate: the clicked movie is a positive, movies shown above
  it were (possibly) examined and skipped (negatives), an unclicked slate
  is a weaker negative for all its movies (`unclicked_weight`). A learned
  per-position term in the click logit absorbs position bias, so the movie
  and user parameters learn relevance rather than slate position;
- watch completion and ratings: graded evidence of how relevant a clicked
  movie was (a second head on the same score); a click with no watch is a
  low grade;
- onboarding genres (a learned prior weight on the declared genres) and
  demographics (age bracket, gender, occupation: shared genre offsets);
- the catalog: genres and release decade;
- per (user, movie) evidence: a small offset r(u, m) per pair, moved by
  every click (up) and examined skip (down) of that pair, by how surprising
  it was, so a movie the user keeps skipping sinks for that user only
  (fatigue, dislikes). Pairs never shown stay at 0: being shown is not
  itself a penalty (a "times shown" feature was: it pushed every seen movie
  below unknown ones and cost 16 points);
- `history`: "already watched" as a feature (rewatches are learned). Only
  slate rows carry it: a rating arrives after its watch, so a grade row
  with it set would teach "watched means liked" (a leak).

Relevance score of movie m for user u:

    s(u, m) = b_m + y[decade_m] + g_m . (w0 + sum_j D[demo_j] + a * o_u + w_u)
              + q_m . p_u + r(u, m) + h * watched(u, m)

Click logit = c + pos[i] + s; grade = sigmoid(c2 + kappa * s).

Learning:
- fit(): full-batch Adam on every logged row from scratch (seeded,
  deterministic), with L2 regularisation `reg`;
- online: one SGD step per new row, starting from the last fit
  (`online_lr`); install(fitted) swaps in a model fitted on a snapshot and
  replays the rows logged since that snapshot, so no evidence is lost.
"""

import copy
import math

import numpy as np

from ml_system.demographics import AGE_BRACKETS
from ml_system.model_server import RetrievalService
from ml_system.ranking import Ranker

POSITIONS = 32
DECADES = 10


def _decade(release_date):
    try:
        year = int(str(release_date)[-4:])
    except ValueError:
        return 0
    return 1 + min(max((year - 1920) // 10, 0), DECADES - 2)


class PreferenceModel:

    ORDER = {"OnboardingEvent": 0, "RecommendationShownEvent": 1, "RecommendationClickedEvent": 2,
             "WatchEvent": 3, "RatingEvent": 4}

    def __init__(self, movie_objects, user_objects=None, factors=0, reg=0.01, seed=0,
                 online_lr=0.05, unclicked_weight=0.5, grade_weight=1.0, epochs=300,
                 history=False, demographics=False, pair_lr=0.3, pair_reg=0.05, shown_feature=False,
                 half_life=0, optimism=0.0):
        from world.external_data_loader import GENRE_NAMES

        self.genre_names = [g for g in GENRE_NAMES if g != "unknown"]
        self.movie_ids = np.array(sorted(movie_objects))
        self.num_movies = int(self.movie_ids.max()) + 1
        nG = len(self.genre_names)
        gi = {g: i for i, g in enumerate(self.genre_names)}
        self.genres = np.zeros((self.num_movies, nG))
        self.decade = np.zeros(self.num_movies, dtype=int)
        for m, movie in movie_objects.items():
            for g in movie.genres:
                if g in gi:
                    self.genres[m, gi[g]] = 1.0
            self.decade[m] = _decade(getattr(movie, "release_date", ""))
        norms = np.linalg.norm(self.genres, axis=1, keepdims=True)
        self.genres = np.divide(self.genres, norms, out=np.zeros_like(self.genres), where=norms > 0)

        occupations = sorted({u.occupation for u in (user_objects or {}).values()})
        self.num_demo = len(AGE_BRACKETS) + 1 + 2 + len(occupations)
        self.demo = {}
        for uid, user in ((user_objects or {}).items() if demographics else ()):
            self.demo[uid] = (sum(user.age >= b for b in AGE_BRACKETS),
                              len(AGE_BRACKETS) + 1 + (1 if user.gender == "F" else 0),
                              len(AGE_BRACKETS) + 3 + occupations.index(user.occupation))
        self.num_users = (max(user_objects) + 1) if user_objects else 1

        self.factors, self.reg, self.seed = factors, reg, seed
        self.online_lr = online_lr
        self.unclicked_weight, self.grade_weight = unclicked_weight, grade_weight
        self.epochs = epochs
        self.use_history, self.use_demographics = history, demographics
        self.pair_lr, self.pair_reg = pair_lr, pair_reg
        self.shown_feature = shown_feature
        self.half_life = half_life      # fit(): a row's weight halves every half_life ticks (0 = all equal)
        # optimism: when choosing (scores(), not training), a movie gets a
        # bonus optimism / sqrt(1 + impressions / 10): rarely shown movies,
        # whose quality is still uncertain, get tried (UCB-style
        # exploration, deterministic); the bonus fades as evidence comes in.
        self.optimism = optimism
        self.pair = {}            # (user, movie) -> offset r(u, m), learned online only
        self.params = self._initial_params(self.num_users)

        # The model's own log of the evidence (rows are what fit() trains on).
        self.rows = []            # (user, movie, position, target, weight, shown, watched)
        self.grades = []          # (user, movie, target, weight, shown, watched)
        self.sequence = []        # (is_grade, index): the order rows arrived in
        self.row_ticks = {False: [], True: []}   # tick each row was logged at
        self.onboarding = {}      # user -> unit vector over genres
        self.shown = {}           # (user, movie) -> impressions so far
        self.watched = set()
        self.impressions = np.zeros(self.num_movies)
        self.pending = {}         # recommendation id -> [tick, user, movies, clicked index, history]
        self.open_clicks = {}     # (user, movie) -> tick, until its watch (or not) is known
        self.last_tick = None
        self.fitted_rows = 0      # length of `sequence` when last fitted
        self.fits = 0
        self.fit_loss = None
        self.online_steps = 0

    def _initial_params(self, num_users):
        nG, k = len(self.genre_names), self.factors
        rng = np.random.default_rng(self.seed)
        return {
            "c": np.array([-3.0]), "pos": np.zeros(POSITIONS), "c2": np.array([0.0]), "kappa": np.array([1.0]),
            "b": np.zeros(self.num_movies), "y": np.zeros(DECADES),
            "w0": np.zeros(nG), "D": np.zeros((self.num_demo, nG)), "a": np.array([0.0]),
            "W": np.zeros((num_users, nG)), "h": np.zeros(2),
            "q": rng.normal(0, 0.01, (self.num_movies, k)), "p": rng.normal(0, 0.01, (num_users, k)),
        }

    def _grow_users(self, user_id):
        if user_id >= self.num_users:
            extra = user_id + 1 - self.num_users
            self.params["W"] = np.vstack([self.params["W"], np.zeros((extra, len(self.genre_names)))])
            self.params["p"] = np.vstack([self.params["p"], np.zeros((extra, self.factors))])
            self.num_users = user_id + 1

    # -- scoring -------------------------------------------------------------

    def _taste(self, user_id):
        P = self.params
        self._grow_users(user_id)
        taste = P["w0"] + P["W"][user_id] + P["a"][0] * self.onboarding.get(user_id, 0.0)
        for j in self.demo.get(user_id, ()):
            taste = taste + P["D"][j]
        return taste

    def scores(self, user_id, movies=None):
        """Relevance scores s(u, m) (default: every movie in the catalog)."""
        movies = self.movie_ids if movies is None else np.asarray(movies, dtype=int)
        P = self.params
        s = P["b"][movies] + P["y"][self.decade[movies]] + self.genres[movies] @ self._taste(user_id)
        if self.factors:
            s = s + P["q"][movies] @ P["p"][user_id]
        if P["h"][0]:
            s = s + P["h"][0] * np.array([min(self.shown.get((user_id, int(m)), 0), 30) for m in movies])
        if P["h"][1]:
            s = s + P["h"][1] * np.array([1.0 if (user_id, int(m)) in self.watched else 0.0 for m in movies])
        if self.pair:
            s = s + np.array([self.pair.get((user_id, int(m)), 0.0) for m in movies])
        if self.optimism:
            s = s + self.optimism / np.sqrt(1.0 + self.impressions[movies] / 10.0)
        return s

    def top(self, user_id, k):
        s = self.scores(user_id)
        order = np.lexsort((self.movie_ids, -s))[:k]
        return [int(m) for m in self.movie_ids[order]]

    # -- logging evidence ----------------------------------------------------

    def _history(self, user_id, movie_id):
        if not self.use_history:
            return (0.0, 0.0)
        shown = min(self.shown.get((user_id, movie_id), 0), 30) if self.shown_feature else 0.0
        return (shown, 1.0 if (user_id, movie_id) in self.watched else 0.0)

    def _add_row(self, row, grade=False):
        target = self.grades if grade else self.rows
        target.append(row)
        self.row_ticks[grade].append(self.last_tick or 0)
        self.sequence.append((grade, len(target) - 1))
        if not grade and self.pair_lr:
            self._pair_step(row)
        if self.online_lr:
            self._online_step(row, grade)

    def _finish_slate(self, user_id, movies, clicked, history):
        last = len(movies) - 1 if clicked is None else clicked
        for i in range(last + 1):
            if clicked is None:
                target, weight = 0.0, self.unclicked_weight
            else:
                target, weight = (1.0 if i == clicked else 0.0), 1.0
            self._add_row((user_id, movies[i], i, target, weight) + history[i])
        for m in movies:
            self.shown[(user_id, m)] = self.shown.get((user_id, m), 0) + 1
            self.impressions[m] += 1

    def flush(self, before_tick=None):
        """Learn from pending slates shown before `before_tick` (all if None)."""
        for rid in sorted(self.pending):
            tick, user_id, movies, clicked, history = self.pending[rid]
            if before_tick is None or tick < before_tick:
                del self.pending[rid]
                self._finish_slate(user_id, movies, clicked, history)
        for (u, m), tick in sorted(self.open_clicks.items()):
            if before_tick is None or tick < before_tick:
                del self.open_clicks[(u, m)]          # clicked, not watched
                self._add_row((u, m, 0.3, 0.5 * self.grade_weight) + (0.0, 0.0), grade=True)

    def observe_tick(self, events):
        """Apply a batch of events in canonical order (Flink and replays
        must see the same order)."""
        for event in sorted(events, key=lambda e: (e.timestamp, self.ORDER.get(type(e).__name__, 5))):
            self.observe(event)

    def observe(self, event):
        if self.last_tick is not None and event.timestamp > self.last_tick:
            self.flush(event.timestamp)
        self.last_tick = event.timestamp
        name = type(event).__name__
        if name == "RecommendationShownEvent":
            # History as of the moment the slate is shown (its own click and
            # watch arrive later in the tick and must not leak into its rows).
            history = [self._history(event.user_id, m) for m in event.movie_ids]
            self.pending[event.recommendation_id] = [event.timestamp, event.user_id, list(event.movie_ids), None, history]
        elif name == "RecommendationClickedEvent":
            slate = self.pending.get(event.recommendation_id)
            if slate is not None and event.movie_id in slate[2]:
                slate[3] = slate[2].index(event.movie_id)
            self.open_clicks[(event.user_id, event.movie_id)] = event.timestamp
        elif name == "WatchEvent":
            u, m = event.user_id, event.movie_id
            self.open_clicks.pop((u, m), None)
            grade = min(1.0, max(0.0, (event.completion - 0.1) / 0.85))
            self._add_row((u, m, grade, self.grade_weight) + (0.0, 0.0), grade=True)
            self.watched.add((u, m))
        elif name == "RatingEvent":
            u, m = event.user_id, event.movie_id
            self._add_row((u, m, (event.rating - 1.0) / 4.0, self.grade_weight) + (0.0, 0.0), grade=True)
        elif name == "OnboardingEvent":
            vec = np.zeros(len(self.genre_names))
            for g in event.genres:
                if g in self.genre_names:
                    vec[self.genre_names.index(g)] = 1.0
            n = np.linalg.norm(vec)
            self.onboarding[event.user_id] = vec / n if n else vec

    # -- online learning -----------------------------------------------------

    def _row_score(self, u, m, shown, watched):
        P = self.params
        s = (P["b"][m] + P["y"][self.decade[m]] + self.genres[m] @ self._taste(u)
             + P["h"][0] * shown + P["h"][1] * watched + self.pair.get((u, m), 0.0))
        if self.factors:
            s += P["q"][m] @ P["p"][u]
        return s

    def _pair_step(self, row):
        """Per-(user, movie) evidence, updated once per slate row as it
        arrives (never replayed: install() keeps the pairs)."""
        u, m, position, target, weight, shown, watched = row
        P = self.params
        z = P["c"][0] + P["pos"][min(position, POSITIONS - 1)] + self._row_score(u, m, shown, watched)
        err = weight * (1.0 / (1.0 + math.exp(-z)) - target)
        r = self.pair.get((u, m), 0.0)
        self.pair[(u, m)] = r - self.pair_lr * (err + self.pair_reg * r)

    def _online_step(self, row, grade):
        """One SGD step on a single row. Only the user's and the movie's own
        parameters move online; shared ones change at the next fit."""
        P, lr = self.params, self.online_lr
        if grade:
            u, m, target, weight, shown, watched = row
            z = P["c2"][0] + P["kappa"][0] * self._row_score(u, m, shown, watched)
            ds = weight * (1.0 / (1.0 + math.exp(-z)) - target) * P["kappa"][0]
        else:
            u, m, position, target, weight, shown, watched = row
            z = P["c"][0] + P["pos"][min(position, POSITIONS - 1)] + self._row_score(u, m, shown, watched)
            ds = weight * (1.0 / (1.0 + math.exp(-z)) - target)
        P["b"][m] -= lr * (ds + self.reg * P["b"][m])
        P["W"][u] -= lr * (ds * self.genres[m] + self.reg * P["W"][u])
        if self.factors:
            pu = P["p"][u].copy()
            P["p"][u] -= lr * (ds * P["q"][m] + self.reg * pu)
            P["q"][m] -= lr * (ds * pu + self.reg * P["q"][m])
        self.online_steps += 1

    # -- batch fit -----------------------------------------------------------

    def snapshot(self):
        """A copy to fit in the background (the live one keeps learning)."""
        return copy.deepcopy(self)

    def fit(self, epochs=None, learning_rate=0.05):
        """Refit every parameter on all logged rows, from scratch (seeded)."""
        import torch

        epochs = self.epochs if epochs is None else epochs
        if not self.rows:
            return self
        for u in list(self.onboarding) + [int(r[0]) for r in self.rows[-1:]]:
            self._grow_users(u)
        for row in self.rows:
            if row[0] >= self.num_users:
                self._grow_users(row[0])
        nG, nU = len(self.genre_names), self.num_users
        rows = np.array(self.rows, dtype=float)
        grades = np.array(self.grades, dtype=float).reshape(-1, 6)
        onboarding = np.zeros((nU, nG))
        for u, vec in self.onboarding.items():
            onboarding[u] = vec
        demo = np.zeros((nU, self.num_demo))
        for u, cols in self.demo.items():
            if u < nU:
                demo[u, list(cols)] = 1.0

        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            T = lambda a, dtype=torch.float32: torch.tensor(np.asarray(a), dtype=dtype)
            gen, dec, onb, dem = T(self.genres), T(self.decade, torch.long), T(onboarding), T(demo)
            P = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True)
                 for k, v in self._initial_params(nU).items()}

            def score(u, m, shown, watched):
                taste = P["w0"] + P["W"][u] + P["a"] * onb[u] + dem[u] @ P["D"]
                s = (P["b"][m] + P["y"][dec[m]] + (gen[m] * taste).sum(-1)
                     + P["h"][0] * shown + P["h"][1] * watched)
                if self.factors:
                    s = s + (P["p"][u] * P["q"][m]).sum(-1)
                return s

            ru, rm = T(rows[:, 0], torch.long), T(rows[:, 1], torch.long)
            rp = T(np.minimum(rows[:, 2], POSITIONS - 1), torch.long)
            ry, rw, rs, rwa = T(rows[:, 3]), T(rows[:, 4]), T(rows[:, 5]), T(rows[:, 6])
            gu, gm = T(grades[:, 0], torch.long), T(grades[:, 1], torch.long)
            gy, gw, gs, gwa = T(grades[:, 2]), T(grades[:, 3]), T(grades[:, 4]), T(grades[:, 5])
            bce = torch.nn.functional.binary_cross_entropy_with_logits
            optimizer = torch.optim.Adam(P.values(), lr=learning_rate)
            if self.half_life:
                now = float(self.last_tick or 0)
                decay = lambda ticks: T(0.5 ** ((now - np.asarray(ticks, dtype=float)) / self.half_life))
                rw = rw * decay(self.row_ticks[False])
                gw = gw * decay(self.row_ticks[True])
            total = rw.sum()
            for _ in range(epochs):
                optimizer.zero_grad()
                z = P["c"] + P["pos"][rp] + score(ru, rm, rs, rwa)
                loss = (bce(z, ry, reduction="none") * rw).sum() / total
                if len(grades):
                    zg = P["c2"] + P["kappa"] * score(gu, gm, gs, gwa)
                    loss = loss + (bce(zg, gy, reduction="none") * gw).sum() / total
                loss = loss + self.reg * (P["W"].pow(2).sum() + P["b"].pow(2).sum() + P["D"].pow(2).sum()
                                          + P["p"].pow(2).sum() + P["q"].pow(2).sum()) / 100.0
                loss.backward()
                optimizer.step()
            self.params = {k: v.detach().numpy().astype(float) for k, v in P.items()}
        self.fit_loss = float(loss.item())
        self.fitted_rows = len(self.sequence)
        self.fits += 1
        return self

    def unfit(self):
        """Back to the never-fitted state (a rollback to a version that had
        no fitted preference model, e.g. the bootstrap v0). The evidence log
        is kept, so a later install() or fit() still has every row."""
        self.params = self._initial_params(self.num_users)
        self.fits, self.fit_loss, self.fitted_rows = 0, None, 0

    def install(self, fitted):
        """Swap in the parameters of a model fitted on a snapshot of this one,
        then replay the rows logged since that snapshot as online steps."""
        self.params = copy.deepcopy(fitted.params)
        self.num_users = len(self.params["W"])
        self.fits, self.fit_loss = fitted.fits, fitted.fit_loss
        if self.online_lr:
            for grade, i in self.sequence[fitted.fitted_rows:]:
                self._online_step((self.grades if grade else self.rows)[i], grade)
        self.fitted_rows = fitted.fitted_rows

    def summary(self):
        P = self.params
        return {"rows": len(self.rows), "grades": len(self.grades), "fits": self.fits,
                "online_steps": self.online_steps, "fit_loss": self.fit_loss,
                "position_bias": [round(float(x), 2) for x in P["pos"][:10]],
                "onboarding_weight": round(float(P["a"][0]), 3),
                "history_weights": [round(float(x), 3) for x in P["h"]]}


# -- serving -----------------------------------------------------------------

class PreferenceRetrieval(RetrievalService):
    """Candidates: the preference model's best movies in the whole catalog.
    `model_fn()` returns the live model (None or unfitted: no candidates)."""

    def __init__(self, model_fn):
        super().__init__()
        self.model_fn = model_fn

    def retrieve(self, request):
        model = self.model_fn()
        return [] if model is None or not model.fits else model.top(request.user_id, request.k)


class PreferenceRanker(Ranker):
    """Orders candidates by the preference model's score (ties: incoming
    order). Until the model has been fitted once, `fallback` ranks."""

    def __init__(self, model_fn, fallback=None):
        self.model_fn = model_fn
        self.fallback = fallback

    def rank(self, user, candidates):
        model = self.model_fn()
        if model is None or not model.fits or not candidates:
            return self.fallback.rank(user, candidates) if self.fallback is not None else candidates
        scores = model.scores(user.user_id, list(candidates))
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [candidates[i] for i in order]
