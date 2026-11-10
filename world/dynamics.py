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
    "initial_user_share", "users_joining_per_tick",
)
EXPOSURE_KEYS = ("fatigue", "rewatch")

# A preset with everything on, for the console's dynamic world.
DYNAMIC_WORLD = {
    "taste_drift_every": 200,      # every 200 ticks...
    "taste_drift_share": 0.05,     # ...5% of users...
    "taste_drift_strength": 0.6,   # ...move 60% of their taste to one genre
    "initial_user_share": 0.7,     # 30% of users join later (cold start)
    "users_joining_per_tick": 0.3,
    "fatigue": 0.15,               # each repeat impression: clicks x0.85
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
    ):
        super().__init__()

        if not 0.0 < initial_user_share <= 1.0:
            raise ValueError("initial_user_share must be in (0, 1]")

        self.world = world
        self.rng = random.Random("world-dynamics-%s" % seed)

        self.taste_drift_every = int(taste_drift_every)
        self.taste_drift_share = float(taste_drift_share)
        self.taste_drift_strength = float(taste_drift_strength)
        self.users_joining_per_tick = float(users_joining_per_tick)

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

        if self.taste_drift_every and now % self.taste_drift_every == 0:
            self.drift_tastes(now)

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

    def summary(self):
        return {
            "active_users": len(self.world.active_user_ids)
            if self.world.active_user_ids is not None else len(self.world.user_objects),
            "waiting_users": len(self.waiting),
            "drifted_users": len(self.drift_log),
            "last_drift_tick": self.drift_log[-1][0] if self.drift_log else None,
        }
