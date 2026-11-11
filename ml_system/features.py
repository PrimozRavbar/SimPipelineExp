class RatingFeatureComputer:

    def __init__(self):
        self.rating_sum = {}
        self.rating_count = {}

    def update(self, entity_id, rating):

        self.rating_sum[entity_id] = (
            self.rating_sum.get(entity_id, 0) + rating
        )

        self.rating_count[entity_id] = (
            self.rating_count.get(entity_id, 0) + 1
        )

        return (
            self.rating_sum[entity_id] /
            self.rating_count[entity_id]
        )

class ItemRatingFeatureComputer:

    def __init__(self):
        self.rating_sum = {}
        self.rating_count = {}

    def update(self, movie_id, rating):

        self.rating_sum[movie_id] = (
            self.rating_sum.get(movie_id, 0) + rating
        )

        self.rating_count[movie_id] = (
            self.rating_count.get(movie_id, 0) + 1
        )

        return (
            self.rating_sum[movie_id] /
            self.rating_count[movie_id]
        )

# ---------------------------------------------------------------------
# User genre profile: one definition for Flink (online) and the offline
# pipeline (training), so both build the same profile from the same events.
# ---------------------------------------------------------------------

PROFILE_MODES = ("watches", "rich")
ONBOARDING_WEIGHT = 2.0      # each declared genre counts like two watches
RICH_DECAY = 0.97            # rich mode: older signal fades with each new one
RICH_CLICK_WEIGHT = 0.5


def update_genre_counts(counts, kind, genres, mode="watches", value=None):
    """Add one event's signal to a user's raw genre counts (in place).

    kind: "onboarding" (declared favourite genres), "click", "watch"
    (value = completion) or "rating" (value = 1..5).

    - "watches" mode (the original): +1 per genre of each watched movie;
      clicks and ratings are ignored.
    - "rich" mode: existing counts decay by RICH_DECAY, then a click adds
      0.5, a watch adds its completion (0.05..1) and a rating adds
      (rating - 3) / 2 (negative for 1-2 stars, floored at 0).
    - Onboarding adds ONBOARDING_WEIGHT per declared genre in both modes.

    Returns True if the counts changed.
    """
    if kind == "onboarding":
        weight = ONBOARDING_WEIGHT
    elif mode == "watches":
        if kind != "watch":
            return False
        weight = 1
    elif mode == "rich":
        if kind == "click":
            weight = RICH_CLICK_WEIGHT
        elif kind == "watch":
            weight = float(value)
        elif kind == "rating":
            weight = (float(value) - 3.0) / 2.0
        else:
            return False
        for genre in counts:
            counts[genre] *= RICH_DECAY
    else:
        raise ValueError("profile mode must be one of %s" % (PROFILE_MODES,))

    for genre in genres:
        counts[genre] = max(0.0, counts.get(genre, 0) + weight)
    return True


# ---------------------------------------------------------------------
# Engagement features (skips): per genre, how often the user clicked when
# shown a movie of that genre, and how much evidence their profile rests on.
# One definition for Flink (online) and the offline pipeline (training).
# ---------------------------------------------------------------------

ENGAGEMENT_PRIOR = 3.0       # impressions of prior per genre
ENGAGEMENT_PRIOR_RATE = 0.05


def update_engagement(state, kind, genres_list):
    """state: {"impressions": {g: n}, "clicks": {g: n}, "signals": n}.
    kind "shown": genres_list = list of genre lists (one per slate movie);
    kind "click" / "watch" / "onboarding": genres_list = one genre list."""
    if kind == "shown":
        for genres in genres_list:
            for g in genres:
                state["impressions"][g] = state["impressions"].get(g, 0) + 1
        return
    if kind == "click":
        for g in genres_list:
            state["clicks"][g] = state["clicks"].get(g, 0) + 1
    if kind in ("click", "watch", "onboarding"):
        state["signals"] = state.get("signals", 0) + 1


def new_engagement_state():
    return {"impressions": {}, "clicks": {}, "signals": 0}


def engagement_features(state):
    """(genre -> smoothed click-through, profile strength in [0, 1))."""
    rates = {}
    for g, shown in state["impressions"].items():
        rates[g] = (state["clicks"].get(g, 0) + ENGAGEMENT_PRIOR * ENGAGEMENT_PRIOR_RATE) / (shown + ENGAGEMENT_PRIOR)
    import math
    return rates, math.tanh(math.log1p(state.get("signals", 0)) / 3.0)
