"""Traffic splitting and experiment statistics for canary releases and
A/B tests.

Users are assigned to an arm by a stable hash of (user id, salt): the same
user always sees the same arm for one experiment, and assignment uses no
RNG. A slate (shown event) is the unit of analysis; CTR per arm is compared
with a two-proportion z-test.
"""

import math


MASK = 0xFFFFFFFF


def bucket(user_id, salt):
    """0..99, stable for (user_id, salt); a new salt reshuffles everyone
    (32-bit integer mixing, no RNG)."""
    x = (user_id * 0x9E3779B1 + salt * 0x85EBCA77 + 0x165667B1) & MASK
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK
    x ^= x >> 16
    return x * 100 >> 32


def in_treatment(user_id, salt, share_pct):
    return bucket(user_id, salt) < share_pct


def two_proportion_test(clicks_a, shown_a, clicks_b, shown_b):
    """B minus A: (difference, 95% CI low, high, z). None if an arm is empty."""
    if not shown_a or not shown_b:
        return None
    pa, pb = clicks_a / shown_a, clicks_b / shown_b
    diff = pb - pa
    pooled = (clicks_a + clicks_b) / (shown_a + shown_b)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / shown_a + 1 / shown_b)) or 1e-12
    se = math.sqrt(pa * (1 - pa) / shown_a + pb * (1 - pb) / shown_b)
    return {"diff": diff, "low": diff - 1.96 * se, "high": diff + 1.96 * se, "z": diff / se_pooled}


class Experiment:
    """Counts shown/clicked slates per arm from the event stream."""

    def __init__(self, control, treatment, share_pct, start_tick, end_tick, salt):
        self.control = control
        self.treatment = treatment
        self.share_pct = share_pct
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.salt = salt
        self.counts = {"control": [0, 0], "treatment": [0, 0]}   # [shown, clicks]
        self._arm_of_slate = {}

    def arm(self, user_id):
        return "treatment" if in_treatment(user_id, self.salt, self.share_pct) else "control"

    def observe(self, event):
        name = type(event).__name__
        if name == "RecommendationShownEvent":
            arm = self.arm(event.user_id)
            self._arm_of_slate[event.recommendation_id] = arm
            self.counts[arm][0] += 1
        elif name == "RecommendationClickedEvent":
            arm = self._arm_of_slate.get(event.recommendation_id)
            if arm is not None:
                self.counts[arm][1] += 1

    def stats(self):
        (sc, cc), (st, ct) = self.counts["control"], self.counts["treatment"]
        return {
            "control": {"version": self.control, "shown": sc, "clicks": cc, "ctr": cc / sc if sc else None},
            "treatment": {"version": self.treatment, "shown": st, "clicks": ct, "ctr": ct / st if st else None},
            "test": two_proportion_test(cc, sc, ct, st),
        }
