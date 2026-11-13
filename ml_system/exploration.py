"""Exploration: show users a few movies the model would not pick, on purpose.

Real systems keep learning about things they rarely show (a user's taste
for a genre they have hardly seen, a drifted taste, a new movie) by giving
a small, controlled share of the slate to exploration, and they log the
probability of each pick so training can correct for it. SlotExplorer does
that, as the last re-ranker:

- Budget: each slate, with probability `rate` (or `boosted_rate` for a user
  the drift detector flags), up to `slots` slots are given to exploration
  picks, at random positions from `first_position` down (lower positions are
  seen less, so they cost fewer clicks).
- Its own candidates: unlike ExplorationReranker (which only reshuffles the
  retrieved candidates), picks come from the whole catalog: a genre is
  chosen, then a well-rated movie of that genre the user has rarely been
  shown (one of the `pool` best by the model's learned quality, at random).
- Which genre:
    "uniform"  - weighted towards genres this user has seen least
                 (1 / sqrt(1 + impressions of the genre)),
    "thompson" - for each genre draw from N(learned taste, uncertainty)
                 with uncertainty shrinking as the user is shown the genre,
                 and explore the best draw (explores where the model is
                 unsure *and* might be wrong in the user's favour).
- Targeting: the preference model's drift detector
  (PreferenceModel.drifting(user)) flags users whose recent click rate fell
  well below their own long-run rate, or whose recent browsing disagrees
  with their learned top genres; they get `boosted_rate`.
- Propensity logging: every shown movie's probability of being where it is
  because of this policy goes to `propensities[(tick, user, movie)]`
  (explored picks: the chance of that exploration; model picks: the chance
  the slot was not explored). The preference model weights its training
  rows by a capped 1 / propensity (inverse propensity scoring).

Uses its own seeded RNG ("explorer-<seed>"), drawn in request order, so
runs stay reproducible.
"""

import math

import numpy as np

from ml_system.ranking import Reranker

POLICIES = ("uniform", "thompson")


class SlotExplorer(Reranker):

    def __init__(self, model_fn, slate_size_fn, rng, rate=0.1, boosted_rate=0.5, slots=1,
                 policy="thompson", first_position=3, pool=5, sim_fn=None):
        if policy not in POLICIES:
            raise ValueError("policy must be one of %s" % (POLICIES,))
        self.model_fn = model_fn
        self.slate_size_fn = slate_size_fn
        self.rng = rng
        self.rate, self.boosted_rate = rate, boosted_rate
        self.slots, self.policy = slots, policy
        self.first_position, self.pool = first_position, pool
        self.sim_fn = sim_fn
        self.propensities = {}      # (tick, user, movie) -> probability
        self.stats = {"slates": 0, "explored_slates": 0, "explored_slots": 0, "boosted_slates": 0}
        self.explored = {}          # (tick, user, movie) -> genre, for the console's view

    # -- choosing ------------------------------------------------------------------

    def _genre_weights(self, model, user_id):
        seen = model.user_genre_shown.get(user_id)
        G = len(model.genre_names)
        seen = np.zeros(G) if seen is None else seen
        if self.policy == "uniform":
            w = 1.0 / np.sqrt(1.0 + seen)
            return w / w.sum(), None
        # Thompson: one draw per genre from N(taste, sd), sd shrinking with exposure.
        taste = model._taste(user_id)
        sd = 0.5 / np.sqrt(1.0 + seen / 5.0)
        draws = np.array([self.rng.gauss(float(taste[g]), float(sd[g])) for g in range(G)])
        best = int(np.argmax(draws))
        # Propensity of the chosen genre is not available in closed form;
        # approximate it by the softmax of taste / sd (higher for genres the
        # policy would usually pick).
        z = taste / np.maximum(sd, 1e-6)
        p = np.exp(z - z.max())
        return p / p.sum(), best

    def _movie_for(self, model, user_id, genre, exclude):
        movies = model.movie_ids
        in_genre = movies[model.genres[movies, genre] > 0]
        if not len(in_genre):
            return None, 0.0
        quality = model.params["b"][in_genre]
        shown = np.array([model.shown.get((user_id, int(m)), 0) for m in in_genre])
        watched = np.array([(user_id, int(m)) in model.watched for m in in_genre])
        ok = (shown <= 1) & ~watched & np.array([int(m) not in exclude for m in in_genre])
        if not ok.any():
            return None, 0.0
        cand = in_genre[ok]
        order = np.lexsort((cand, -quality[ok]))[:self.pool]
        pick = int(cand[order[self.rng.randrange(len(order))]])
        return pick, 1.0 / len(order)

    # -- re-ranking ----------------------------------------------------------------

    def rerank(self, user, candidates):
        model = self.model_fn()
        candidates = list(candidates)
        size = self.slate_size_fn()
        self.stats["slates"] += 1
        tick = self.sim_fn().clock.now if self.sim_fn else 0
        if model is None or not model.fits or self.slots <= 0:
            return candidates
        boosted = model.drifting(user.user_id)
        rate = self.boosted_rate if boosted else self.rate
        if boosted:
            self.stats["boosted_slates"] += 1
        head = candidates[:size]
        if self.rng.random() >= rate:
            for m in head:
                self.propensities[(tick, user.user_id, m)] = 1.0 - rate
            return candidates
        self.stats["explored_slates"] += 1
        exclude = set(head)
        picks = []
        for _ in range(min(self.slots, max(0, size - self.first_position))):
            weights, best = self._genre_weights(model, user.user_id)
            genre = best if best is not None else self._sample(weights)
            movie, p_movie = self._movie_for(model, user.user_id, genre, exclude)
            if movie is None:
                continue
            exclude.add(movie)
            picks.append((movie, rate * float(weights[genre]) * p_movie, model.genre_names[genre]))
        out = [m for m in head]
        for movie, propensity, genre in picks:
            position = self.rng.randrange(self.first_position, max(self.first_position + 1, size))
            out.insert(min(position, len(out)), movie)
            self.propensities[(tick, user.user_id, movie)] = max(propensity, 1e-4)
            self.explored[(tick, user.user_id, movie)] = genre
            self.stats["explored_slots"] += 1
        out = out[:size]
        for m in out:
            self.propensities.setdefault((tick, user.user_id, m), 1.0 - rate)
        rest = [m for m in candidates if m not in set(out)]
        return out + rest

    def _sample(self, weights):
        pick = self.rng.random()
        for g, w in enumerate(weights):
            pick -= w
            if pick <= 0:
                return g
        return len(weights) - 1

    def summary(self):
        s = self.stats
        return {**s, "policy": self.policy, "rate": self.rate, "boosted_rate": self.boosted_rate,
                "slots": self.slots,
                "explored_share": s["explored_slots"] / (s["slates"] * max(1, self.slate_size_fn()))
                if s["slates"] else 0.0}
