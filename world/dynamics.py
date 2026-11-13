"""A world that changes: taste drift and users joining over time.

WorldDynamics is a Service registered after the World, so changes made in
tick T take effect from tick T+1. It has its own seeded RNG (derived from
the build seed, not drawn from the shared seed stream), so switching it on
or off never shifts any other component's randomness.

Click-side effects (fatigue, no rewatching) live in UserSimulator; see
DYNAMIC_WORLD for a preset that turns everything on.
"""

import random

from simulation.service import Service
from world.external_data_loader import GENRE_NAMES

# Genres a user's taste can drift towards ("unknown" is not a taste).
DRIFT_GENRES = [genre for genre in GENRE_NAMES if genre != "unknown"]

# Keys accepted by build_recommendation_system(world_dynamics=...).
DYNAMICS_KEYS = (
    "taste_drift_every", "taste_drift_share", "taste_drift_strength",
    "initial_user_share", "users_joining_per_tick", "taste_drift_mode", "taste_drift_steps",
)
DRIFT_MODES = ("abrupt", "gradual")
EXPOSURE_KEYS = ("fatigue", "rewatch", "fatigue_recovery")

# A preset with everything on, for the console's dynamic world.
DYNAMIC_WORLD = {
    "taste_drift_every": 200,      # every 200 ticks...
    "taste_drift_share": 0.05,     # ...5% of users...
    "taste_drift_strength": 0.6,   # ...move 60% of their taste to one genre
    "taste_drift_mode": "gradual", # ...a related genre, in small steps (see WorldDynamics)
    "taste_drift_steps": 4,
    "initial_user_share": 0.7,     # 30% of users join later (cold start)
    "users_joining_per_tick": 0.3,
    "fatigue": 0.15,               # each repeat impression: clicks x0.85
    "fatigue_recovery": 1000,      # ...wearing off slowly: halves every 1000 ticks
    "rewatch": 0.1,                # watched movies are rarely clicked again
}


class WorldDynamics(Service):

    def __init__(
        self,
        world,
        seed=0,
        taste_drift_every=0,
        taste_drift_share=0.05,
        taste_drift_strength=0.5,
        initial_user_share=1.0,
        users_joining_per_tick=0.0,
        taste_drift_mode="abrupt",
        taste_drift_steps=4,
    ):
        super().__init__()

        if not 0.0 < initial_user_share <= 1.0:
            raise ValueError("initial_user_share must be in (0, 1]")
        if taste_drift_mode not in DRIFT_MODES:
            raise ValueError("taste_drift_mode must be one of %s" % (DRIFT_MODES,))

        self.world = world
        self.rng = random.Random("world-dynamics-%s" % seed)

        self.taste_drift_every = int(taste_drift_every)
        self.taste_drift_share = float(taste_drift_share)
        self.taste_drift_strength = float(taste_drift_strength)
        self.users_joining_per_tick = float(users_joining_per_tick)
        # Drift modes:
        # - "abrupt" (the original): every taste_drift_every ticks a share of
        #   users moves taste_drift_strength of its genre taste to one random
        #   genre at once.
        # - "gradual": the same share of users starts a drift journey every
        #   taste_drift_every ticks, towards a genre related to what they
        #   already like (genres that appear together in the catalog), and
        #   moves there in taste_drift_steps equal steps spread over the next
        #   taste_drift_every ticks; the total shift is the same.
        self.taste_drift_mode = taste_drift_mode
        self.taste_drift_steps = max(1, int(taste_drift_steps))
        self.journeys = {}           # user -> [target genre, steps left]
        self._relatedness = None

        self.drift_log = []          # (tick, user_id, genre)
        self.joined_log = []         # (tick, user_id)

        # Users who have not joined yet, in a seeded random order.
        user_ids = sorted(world.user_objects)
        self.rng.shuffle(user_ids)
        initial = max(1, round(initial_user_share * len(user_ids)))
        self.waiting = user_ids[initial:]
        self._join_credit = 0.0
        if self.waiting:
            world.active_user_ids = sorted(user_ids[:initial])

    def process(self, sim):
        now = sim.clock.now

        if self.taste_drift_every and self.taste_drift_mode == "abrupt" \
                and now % self.taste_drift_every == 0:
            self.drift_tastes(now)
        elif self.taste_drift_every and self.taste_drift_mode == "gradual":
            step = max(1, self.taste_drift_every // self.taste_drift_steps)
            if now % self.taste_drift_every == 0:
                self.start_journeys(now)
            if now % step == 0:
                self.advance_journeys()

        if self.waiting and self.users_joining_per_tick > 0:
            self._join_credit += self.users_joining_per_tick
            while self._join_credit >= 1 and self.waiting:
                self._join_credit -= 1
                user_id = self.waiting.pop(0)
                self.world.active_user_ids.append(user_id)
                self.joined_log.append((now, user_id))
            if not self.waiting:
                self.world.active_user_ids = None   # everyone is in

    def drift_tastes(self, now):
        """Move a share of users' taste towards one random genre."""
        pool = sorted(self.world.active_user_ids or self.world.user_objects)
        count = int(round(self.taste_drift_share * len(pool)))
        for user_id in self.rng.sample(pool, min(count, len(pool))):
            genre = self.rng.choice(DRIFT_GENRES)
            state = self.world.user_states[user_id]
            preferences = dict(state.genre_preferences)
            total = sum(max(v, 0.0) for v in preferences.values()) or 1.0
            s = self.taste_drift_strength
            preferences = {g: (1 - s) * v for g, v in preferences.items()}
            preferences[genre] = preferences.get(genre, 0.0) + s * total
            state.genre_preferences = preferences
            self.drift_log.append((now, user_id, genre))

    # -- gradual drift ---------------------------------------------------------

    def relatedness(self):
        """Genre x genre: how often two genres appear on the same movie
        (row-normalised, no self-links). Fixed by the catalog."""
        if self._relatedness is None:
            index = {g: i for i, g in enumerate(DRIFT_GENRES)}
            n = len(DRIFT_GENRES)
            counts = [[0.0] * n for _ in range(n)]
            for movie in self.world.movie_objects.values():
                genres = [index[g] for g in set(movie.genres) if g in index]
                for a in genres:
                    for b in genres:
                        if a != b:
                            counts[a][b] += 1.0
            for row in counts:
                total = sum(row)
                if total:
                    for j in range(n):
                        row[j] /= total
            self._relatedness = counts
        return self._relatedness

    def related_target(self, preferences):
        """A genre related to the user's current taste, not one they already
        mostly like: weights = taste share x relatedness, x (1 - own share)."""
        total = sum(max(v, 0.0) for v in preferences.values()) or 1.0
        share = [max(preferences.get(g, 0.0), 0.0) / total for g in DRIFT_GENRES]
        R = self.relatedness()
        n = len(DRIFT_GENRES)
        weights = [sum(share[a] * R[a][b] for a in range(n)) * (1.0 - share[b]) for b in range(n)]
        if sum(weights) <= 0:
            return self.rng.choice(DRIFT_GENRES)
        pick = self.rng.random() * sum(weights)
        for genre, w in zip(DRIFT_GENRES, weights):
            pick -= w
            if pick <= 0:
                return genre
        return DRIFT_GENRES[-1]

    def start_journeys(self, now):
        pool = sorted(u for u in (self.world.active_user_ids or self.world.user_objects)
                      if u not in self.journeys)
        count = int(round(self.taste_drift_share * len(pool)))
        for user_id in self.rng.sample(pool, min(count, len(pool))):
            genre = self.related_target(self.world.user_states[user_id].genre_preferences)
            self.journeys[user_id] = [genre, self.taste_drift_steps]
            self.drift_log.append((now, user_id, genre))

    def advance_journeys(self):
        # Equal steps that add up to taste_drift_strength.
        step = 1.0 - (1.0 - self.taste_drift_strength) ** (1.0 / self.taste_drift_steps)
        for user_id in sorted(self.journeys):
            genre, left = self.journeys[user_id]
            state = self.world.user_states[user_id]
            preferences = dict(state.genre_preferences)
            total = sum(max(v, 0.0) for v in preferences.values()) or 1.0
            preferences = {g: (1 - step) * v for g, v in preferences.items()}
            preferences[genre] = preferences.get(genre, 0.0) + step * total
            state.genre_preferences = preferences
            if left <= 1:
                del self.journeys[user_id]
            else:
                self.journeys[user_id][1] = left - 1

    def summary(self):
        return {
            "active_users": len(self.world.active_user_ids)
            if self.world.active_user_ids is not None else len(self.world.user_objects),
            "waiting_users": len(self.waiting),
            "drifted_users": len(self.drift_log),
            "drifting_now": len(self.journeys),
            "drift_mode": self.taste_drift_mode,
            "last_drift_tick": self.drift_log[-1][0] if self.drift_log else None,
        }
